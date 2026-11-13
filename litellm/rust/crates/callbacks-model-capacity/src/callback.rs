use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Activation, Callback, CallbackError, Field, FieldSet, Hook, HookInput, HookOutput, Manifest,
    Patch, RequestState, Selection, Subscription, Verdict,
};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Map, Value};

use crate::{
    capacity::{Engine, PreCall, Request, Settlement},
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

#[derive(Debug, Default, Deserialize)]
struct AttemptFacts {
    call_id: Option<String>,
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

    fn stash(&self, call_id: Option<&str>) -> Option<SharedStash> {
        self.stashes.find(call_id?)
    }

    async fn pre_call(&self, input: &HookInput) -> HookOutput {
        let Some(Value::String(model)) = input.fields.raw(Field::Model)?.cloned() else {
            return Ok(None);
        };
        let facts: PreCallFacts = facts(input)?;
        let call_id = facts
            .call_id
            .clone()
            .or_else(|| input.call.as_ref().map(|call| call.ids.call_id.clone()))
            .unwrap_or_default();
        let stash = match input.state::<Anchor>() {
            Some(anchor) => anchor.hold(&call_id),
            None => self.stashes.claim(&call_id, false),
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
        let Some(stash) = self.stash(facts.call_id.as_deref()) else {
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
        if facts.failed {
            return self.failure(facts).await;
        }
        let _ = self.success(facts).await;
        Ok(None)
    }

    async fn success(&self, facts: AttemptFacts) -> Result<(), CapacityError> {
        if !facts.settlement.valid || !facts.model_group_known {
            return Ok(());
        }
        let settings = self.engine.host.settings().await?;
        let total = UsageTokens::settled(facts.usage.as_ref(), settings.rate_limit_type);
        let stash = self.stash(facts.call_id.as_deref());
        let settlement = Settlement {
            model_group: facts.model_group.as_deref(),
            team_id: facts.settlement.team_id.as_deref(),
            team_alias: facts.settlement.team_alias.as_deref(),
        };
        let priority = facts.settlement.priority.filter(Value::is_string);
        self.engine
            .settle_success(
                &settings,
                stash.as_ref(),
                &settlement,
                priority.as_ref(),
                total,
            )
            .await
    }

    async fn failure(&self, facts: AttemptFacts) -> HookOutput {
        let Some(stash) = self.stash(facts.call_id.as_deref()) else {
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
        let stash = self.stashes.claim(call_id.unwrap_or_default(), false);
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
