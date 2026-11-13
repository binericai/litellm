use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Activation, Callback, CallbackError, Field, FieldSet, Hook, HookInput, HookOutput, Manifest,
    Patch, RequestState, Selection, Subscription, Verdict,
};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Map, Value};

use crate::{
    capacity::{Engine, Frame, PreCall, Request, Settlement},
    error::CapacityError,
    model::{KeyFacts, STEERED_FROM_HEADER, UsageTokens},
    pyfmt,
    refusal::reraise,
    seam::{CapacityHost, V3Limiter},
    state::{Anchor, SharedStash, Stashes},
};

pub const NAME: &str = "dynamic_rate_limiter_v3";

#[derive(Debug, Default, Deserialize)]
struct PreCallFacts {
    call_id: Option<String>,
    #[serde(default)]
    key: KeyFacts,
}

#[derive(Debug, Default, Deserialize)]
struct KeyOnly {
    #[serde(default)]
    key: KeyFacts,
}

#[derive(Debug, Default, Deserialize)]
struct CallOnly {
    call_id: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct SettlementFacts {
    #[serde(default)]
    valid: bool,
    priority: Option<Value>,
    team_id: Option<String>,
    team_alias: Option<String>,
}

const SOCKET_CALL_TYPE: &str = "aresponses_websocket";

#[derive(Debug, Default, Deserialize)]
struct AttemptFacts {
    call_id: Option<String>,
    call_type: Option<String>,
    socket_call_id: Option<String>,
    #[serde(default)]
    failed: bool,
    #[serde(default)]
    model_group_known: bool,
    model_group: Option<String>,
    usage: Option<UsageTokens>,
    #[serde(default)]
    settlement: SettlementFacts,
}

pub struct ModelCapacity {
    manifest: Manifest,
    engine: Engine,
    stashes: Arc<Stashes>,
}

fn failed(error: CapacityError) -> CallbackError {
    CallbackError::Failed(error.to_string())
}

fn raise(error: CapacityError) -> HookOutput {
    match error.raised() {
        Some(raised) => Ok(Some(Verdict::Abort(reraise(raised.token, &raised.message)))),
        None => Err(failed(error)),
    }
}

fn facts<T: DeserializeOwned + Default>(input: &HookInput) -> Result<T, CallbackError> {
    match input.fields.raw(Field::DeprecatedKwargs)? {
        None | Some(Value::Null) => Ok(T::default()),
        Some(value) => {
            T::deserialize(value).map_err(|error| CallbackError::Failed(error.to_string()))
        }
    }
}

impl ModelCapacity {
    pub fn new(limiter: Arc<dyn V3Limiter>, host: Arc<dyn CapacityHost>) -> Self {
        let kwargs = FieldSet::of(&[Field::DeprecatedKwargs]);
        Self {
            manifest: Manifest {
                selection: Selection {
                    activation: Activation::Required,
                    ..Selection::named(NAME)
                },
                subscriptions: vec![
                    Subscription {
                        reads: FieldSet::of(&[Field::Model, Field::DeprecatedKwargs]),
                        writes: FieldSet::of(&[Field::Model]),
                        may_abort: true,
                        ..Subscription::on(Hook::PreCall)
                    },
                    Subscription {
                        reads: FieldSet::of(&[
                            Field::DeprecatedKwargs,
                            Field::DeprecatedResponseObj,
                        ]),
                        writes: FieldSet::of(&[Field::Response]),
                        ..Subscription::on(Hook::PostCallSuccess)
                    },
                    Subscription {
                        reads: kwargs,
                        writes: FieldSet::of(&[Field::ResponseHeaders]),
                        ..Subscription::on(Hook::ResponseHeaders)
                    },
                    Subscription {
                        reads: kwargs,
                        awaited: true,
                        ..Subscription::on(Hook::AttemptEnded)
                    },
                ],
                has_request_state: true,
            },
            engine: Engine { limiter, host },
            stashes: Arc::default(),
        }
    }

    fn stash_for_call(&self, input: &HookInput, call_id: Option<&str>) -> Option<SharedStash> {
        input
            .state::<Anchor>()
            .and_then(|anchor| anchor.for_call(call_id))
            .or_else(|| self.stashes.find(call_id?))
    }

    fn settled_stash(&self, input: &HookInput, facts: &AttemptFacts) -> Option<SharedStash> {
        let call_id = facts.call_id.as_deref();
        if let Some(stash) = self.stash_for_call(input, call_id) {
            return Some(stash);
        }
        let (owner, connection) = input.state::<Anchor>()?.current()?;
        let socket_turn = facts.call_type.as_deref() == Some(SOCKET_CALL_TYPE)
            || (facts.socket_call_id.is_some() && facts.socket_call_id == owner);
        socket_turn.then_some(connection)
    }

    async fn pre_call(&self, input: &HookInput) -> HookOutput {
        let Some(Value::String(model)) = input.fields.raw(Field::Model)?.cloned() else {
            return Ok(None);
        };
        let facts: PreCallFacts = facts(input)?;
        let call_id = facts.call_id.as_deref();
        let stash = match input.state::<Anchor>() {
            Some(anchor) => anchor.claim(call_id),
            None => self.stashes.find_or_claim(call_id),
        };
        let group = match self.engine.host.group(&model).await {
            Ok(Some(group)) => group,
            Ok(None) => return Ok(None),
            Err(error) => return raise(error),
        };
        let settings = match self.engine.host.settings().await {
            Ok(settings) => settings,
            Err(error) => return raise(error),
        };
        let call_type = input.call.as_ref().map(|call| call.kind.call_type.clone());
        let request = Request {
            key: &facts.key,
            call_type: call_type.as_deref(),
            settings: &settings,
            stash: &stash,
        };
        match self.engine.pre_call(&request, &model, &group).await {
            PreCall::Allow => Ok(None),
            PreCall::Refuse(refusal) => Ok(Some(Verdict::Abort(refusal.into_abort()))),
            PreCall::Raise(error) => raise(error),
            PreCall::Steer(steered) => {
                stash.lock().steered_from = Some(model);
                if let Err(error) = self.engine.host.steer(&steered.access_groups).await {
                    return raise(error);
                }
                Ok(Some(Verdict::Patch(
                    Patch::default().set(Field::Model, Value::String(steered.model)),
                )))
            }
        }
    }

    async fn post_call_success(&self, input: &HookInput) -> HookOutput {
        let Ok(facts) = facts::<KeyOnly>(input) else {
            return Ok(None);
        };
        if self.engine.limiter.post_call_success().await.is_err() {
            return Ok(None);
        }
        let headers = Engine::post_call_headers(&facts.key);
        let _ = self.engine.host.add_response_headers(&headers).await;
        Ok(None)
    }

    fn response_headers(&self, input: &HookInput) -> HookOutput {
        let facts: CallOnly = facts(input)?;
        let Some(stash) = self.stash_for_call(input, facts.call_id.as_deref()) else {
            return Ok(None);
        };
        let Some(steered_from) = stash.lock().steered_from.clone() else {
            return Ok(None);
        };
        if !pyfmt::is_latin1(&steered_from) {
            return Ok(None);
        }
        let mut headers = Map::new();
        headers.insert(STEERED_FROM_HEADER.into(), Value::String(steered_from));
        Ok(Some(Verdict::Patch(
            Patch::default().merge(Field::ResponseHeaders, headers),
        )))
    }

    async fn attempt_ended(&self, input: &HookInput) -> HookOutput {
        let facts: AttemptFacts = facts(input)?;
        let stash = self.settled_stash(input, &facts);
        if facts.failed {
            return self.failure(facts, stash).await;
        }
        let _ = self.success(facts, stash).await;
        Ok(None)
    }

    async fn success(
        &self,
        facts: AttemptFacts,
        stash: Option<SharedStash>,
    ) -> Result<(), CapacityError> {
        if !facts.settlement.valid || !facts.model_group_known {
            return Ok(());
        }
        let settings = self.engine.host.settings().await?;
        let total = UsageTokens::settled(facts.usage.as_ref(), settings.rate_limit_type);
        let settlement = Settlement {
            model_group: facts.model_group.as_deref(),
            team_id: facts.settlement.team_id.as_deref(),
            team_alias: facts.settlement.team_alias.as_deref(),
        };
        let priority = facts.settlement.priority.as_ref().and_then(Value::as_str);
        self.engine
            .settle_success(&settings, stash.as_ref(), &settlement, priority, total)
            .await
    }

    async fn failure(&self, facts: AttemptFacts, stash: Option<SharedStash>) -> HookOutput {
        let Some(stash) = stash else {
            return Ok(None);
        };
        if !facts.model_group_known {
            return Ok(None);
        }
        let Ok(settings) = self.engine.host.settings().await else {
            return Ok(None);
        };
        let used = UsageTokens::settled(facts.usage.as_ref(), settings.rate_limit_type);
        let Ok(hold) = self
            .engine
            .settle_failure(&settings, &stash, facts.model_group.as_deref(), used)
            .await
        else {
            return Ok(None);
        };
        if !facts.settlement.valid {
            return Ok(None);
        }
        let settlement = Settlement {
            model_group: facts.model_group.as_deref(),
            team_id: facts.settlement.team_id.as_deref(),
            team_alias: facts.settlement.team_alias.as_deref(),
        };
        self.engine
            .record_team_share(
                &settings,
                Some(&stash),
                facts.model_group.as_deref(),
                &settlement,
                hold.as_deref(),
            )
            .await;
        Ok(None)
    }

    pub async fn fallback_target_allowed(
        &self,
        call_id: Option<&str>,
        key: &KeyFacts,
        model: &str,
        commit: bool,
    ) -> Result<bool, CapacityError> {
        let stash = self.stashes.find_or_claim(call_id);
        let settings = self.engine.host.settings().await?;
        let request = Request {
            key,
            call_type: None,
            settings: &settings,
            stash: &stash,
        };
        self.engine
            .fallback_target_allowed(&request, model, commit)
            .await
    }

    pub async fn enforce_frame(
        &self,
        connection_call_id: Option<&str>,
        key: &KeyFacts,
        model: &str,
    ) -> Result<Frame, CapacityError> {
        let connection = connection_call_id.and_then(|call_id| self.stashes.find(call_id));
        let stash = connection.clone().unwrap_or_default();
        let settings = self.engine.host.settings().await?;
        let request = Request {
            key,
            call_type: None,
            settings: &settings,
            stash: &stash,
        };
        self.engine
            .enforce_frame(&request, model, connection.as_ref())
            .await
    }
}

impl Callback for ModelCapacity {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn request_state(&self) -> Option<RequestState> {
        Some(Arc::new(Anchor::new(Arc::clone(&self.stashes))))
    }

    fn handle<'a>(&'a self, input: &'a HookInput) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            match input.hook {
                Hook::PreCall => self.pre_call(input).await,
                Hook::PostCallSuccess => self.post_call_success(input).await,
                Hook::ResponseHeaders => self.response_headers(input),
                Hook::AttemptEnded => self.attempt_ended(input).await,
                _ => Ok(None),
            }
        })
    }
}
