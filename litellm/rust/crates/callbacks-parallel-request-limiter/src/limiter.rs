use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Abort, Activation, Callback, CallbackError, Field, FieldSet, Hook, HookInput, HookOutput,
    Manifest, Patch, RequestState, Selection, Subscription, Verdict,
};
use litellm_redis_store::{RequestRedis, rate_limits::Increment};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

use crate::{
    descriptor::{Descriptor, PTU_TEAM},
    engine::{Ctx, Engine, Op, Policy, SlotAcquisition, Unverifiable},
    refusal,
    state::{LimiterState, RateLimitedModel, Stash},
    status::header_map,
};

pub const NAME: &str = "parallel_request_limiter_v3";

const PER_MODEL_KEYS: [&str; 7] = [
    "model_per_key",
    "model_per_team",
    "model_per_end_user",
    "model_per_organization",
    "model_per_project",
    "model_per_project_itpm",
    "model_per_project_otpm",
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyFacts {
    #[serde(default)]
    pub fail_closed: bool,
    #[serde(default)]
    pub force_grouping: bool,
}

impl From<PolicyFacts> for Policy {
    fn from(facts: PolicyFacts) -> Self {
        Self {
            fail_closed: facts.fail_closed,
            force_grouping: facts.force_grouping,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreCallFacts {
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub descriptors: Vec<Descriptor>,
    #[serde(default)]
    pub requested_model: Option<String>,
    #[serde(default)]
    pub estimated_tokens: Option<i64>,
    #[serde(default)]
    pub reserved_model: Option<RateLimitedModel>,
    #[serde(default)]
    pub policy: PolicyFacts,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallIdFacts {
    #[serde(default)]
    pub call_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostCallFailureFacts {
    #[serde(default)]
    pub tpm_actual: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetFacts {
    pub user_api_key_hash: Option<String>,
    pub user_api_key_user_id: Option<String>,
    pub user_api_key_team_id: Option<String>,
    pub user_api_key_org_id: Option<String>,
    pub user_api_key_project_id: Option<String>,
    pub end_user_id: Option<String>,
    pub agent_id: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    pub group: String,
    pub key_owns: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PtuFacts {
    pub value: String,
    pub actual_tokens: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFacts {
    pub candidates: Vec<Candidate>,
    #[serde(default)]
    pub ptu: Option<PtuFacts>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuccessFacts {
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub exempt: bool,
    #[serde(default)]
    pub total_tokens: i64,
    #[serde(default)]
    pub targets: TargetFacts,
    #[serde(default)]
    pub served_model: Option<RateLimitedModel>,
    #[serde(default)]
    pub models: BTreeMap<String, ModelFacts>,
    #[serde(default)]
    pub team_label: String,
    #[serde(default)]
    pub policy: PolicyFacts,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureFacts {
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub tpm_actual: i64,
}

pub struct ParallelRequestLimiter {
    manifest: Manifest,
    engine: Engine,
}

fn decode<T: DeserializeOwned + Default>(input: &HookInput) -> Result<T, CallbackError> {
    match input.fields.raw(Field::DeprecatedKwargs)? {
        Some(value) => serde_json::from_value(value.clone()).map_err(|error| {
            CallbackError::Failed(format!("the {:?} facts do not decode: {error}", input.hook))
        }),
        None => Ok(T::default()),
    }
}

fn truthy(value: &Option<String>) -> Option<&str> {
    value.as_deref().filter(|value| !value.is_empty())
}

pub fn manifest() -> Manifest {
    let kwargs = FieldSet::of(&[Field::DeprecatedKwargs]);
    Manifest {
        selection: Selection {
            activation: Activation::Required,
            ..Selection::named(NAME)
        },
        subscriptions: vec![
            Subscription {
                reads: kwargs,
                may_abort: true,
                ..Subscription::on(Hook::PreCall)
            },
            Subscription {
                reads: kwargs,
                awaited: true,
                ..Subscription::on(Hook::PostCallSuccess)
            },
            Subscription {
                reads: kwargs,
                awaited: true,
                ..Subscription::on(Hook::PostCallFailure)
            },
            Subscription {
                reads: FieldSet::of(&[Field::DeprecatedKwargs, Field::StandardLoggingPayload]),
                writes: FieldSet::of(&[Field::StandardLoggingPayload]),
                ..Subscription::on(Hook::LogPrepare)
            },
        ],
        has_request_state: true,
    }
}

impl ParallelRequestLimiter {
    pub fn new(engine: Engine) -> Self {
        Self {
            manifest: manifest(),
            engine,
        }
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    async fn release_stashed(&self, state: &LimiterState, in_logging_callback: bool, ctx: Ctx<'_>) {
        let _guard = state.release.lock().await;
        let Some(acquisition) = state.stash().parallel_slot.clone() else {
            return;
        };
        let deferred = in_logging_callback && self.engine.defer_release(&acquisition, ctx).await;
        if !deferred {
            self.engine.release_slots(&acquisition).await;
        }
        state.stash().parallel_slot = None;
    }

    pub async fn pre_call(
        &self,
        state: &LimiterState,
        facts: PreCallFacts,
        request: Option<&RequestRedis>,
    ) -> Option<Abort> {
        let ctx = Ctx {
            policy: facts.policy.into(),
            request,
        };
        match self.admit(state, &facts, ctx).await {
            Ok(abort) => abort,
            Err(Unverifiable) => Some(refusal::unverifiable()),
        }
    }

    async fn admit(
        &self,
        state: &LimiterState,
        facts: &PreCallFacts,
        ctx: Ctx<'_>,
    ) -> Result<Option<Abort>, Unverifiable> {
        let descriptors = &facts.descriptors;
        let settings = self.engine.settings();
        {
            let mut stash = state.stash();
            stash.owned = true;
            if let Some(call_id) = &facts.call_id {
                stash.owner_call_id = Some(call_id.clone());
            }
            stash.tpm_limited_tags = descriptors
                .iter()
                .filter(|descriptor| {
                    descriptor.key == "tag"
                        && descriptor
                            .rate_limit
                            .as_ref()
                            .is_some_and(|limits| limits.tokens_per_unit.is_some())
                })
                .map(|descriptor| descriptor.value.clone())
                .collect();
        }
        if descriptors.is_empty() {
            return Ok(None);
        }
        let gauges: Vec<_> = descriptors
            .iter()
            .filter(|descriptor| descriptor.limits().max_parallel_requests.is_some())
            .map(Descriptor::scope)
            .collect();
        let slot_id = (!gauges.is_empty()).then(|| uuid::Uuid::new_v4().simple().to_string());
        let first_pass: Vec<Descriptor> = if settings.tpm_reservation_enabled {
            descriptors.clone()
        } else {
            descriptors
                .iter()
                .filter(|descriptor| !descriptor.is_project_io())
                .cloned()
                .collect()
        };
        let response = self
            .engine
            .should_rate_limit(
                &first_pass,
                false,
                settings.tpm_reservation_enabled,
                slot_id.as_deref(),
                false,
                ctx,
            )
            .await?;
        if response.is_over() {
            return Ok(self.refuse(&response, facts));
        }
        {
            let mut stash = state.stash();
            stash.first_pass_admitted = true;
            stash.rate_limit_response = Some(response);
            if let Some(slot_id) = slot_id {
                stash.parallel_slot = Some(SlotAcquisition { slot_id, gauges });
            }
        }
        let has_tpm_limits = descriptors.iter().any(|descriptor| {
            !descriptor.is_project_io() && descriptor.limits().tokens_per_unit.is_some()
        });
        if !(has_tpm_limits && settings.tpm_reservation_enabled) {
            return Ok(None);
        }
        let estimated = facts.estimated_tokens.unwrap_or(0).max(1);
        let tpm_response = self
            .engine
            .reserve_tpm_tokens(descriptors, estimated, ctx)
            .await?;
        if tpm_response.is_over() {
            self.release_stashed(state, false, ctx).await;
            return Ok(self.refuse(&tpm_response, facts));
        }
        let mut stash = state.stash();
        stash.discard_released_tpm_reservation();
        stash.reserved_tokens = estimated;
        stash.reserved_model = facts.reserved_model.clone();
        stash.reserved_scopes = descriptors
            .iter()
            .filter(|descriptor| {
                !descriptor.is_project_io() && descriptor.limits().tokens_per_unit.is_some()
            })
            .map(|descriptor| (descriptor.key.clone(), descriptor.value.clone()))
            .collect();
        if let Some(stored) = stash.rate_limit_response.as_mut() {
            stored.statuses.extend(tpm_response.statuses);
        }
        Ok(None)
    }

    fn refuse(&self, response: &crate::status::Response, facts: &PreCallFacts) -> Option<Abort> {
        refusal::rate_limited(
            response,
            &facts.descriptors,
            facts.requested_model.as_deref(),
            self.engine.now(),
            self.engine.settings().window_size,
        )
    }

    pub async fn post_call_success(&self, state: &LimiterState, facts: CallIdFacts) {
        let applies = {
            let stash = state.stash();
            stash.owned && stash.for_call(facts.call_id.as_deref())
        };
        if applies {
            self.release_stashed(
                state,
                false,
                Ctx {
                    policy: Policy::default(),
                    request: None,
                },
            )
            .await;
        }
    }

    pub async fn release_on_disconnect(&self, state: &LimiterState) {
        if state.stash().owned {
            self.release_stashed(
                state,
                false,
                Ctx {
                    policy: Policy::default(),
                    request: None,
                },
            )
            .await;
        }
    }

    pub async fn post_call_failure(&self, state: &LimiterState, facts: PostCallFailureFacts) {
        if !state.stash().owned {
            return;
        }
        self.release_stashed(
            state,
            false,
            Ctx {
                policy: Policy::default(),
                request: None,
            },
        )
        .await;
        let operations = {
            let mut stash = state.stash();
            let reserved = if stash.reservation_released {
                0
            } else {
                stash.reserved_tokens
            };
            if reserved <= 0 {
                return;
            }
            let operations = self.failure_operations(&stash, facts.tpm_actual, reserved);
            stash.reservation_released = true;
            operations
        };
        if !operations.is_empty() {
            self.engine.increment_many(&operations).await;
        }
    }

    fn failure_operations(&self, stash: &Stash, actual: i64, reserved: i64) -> Vec<Increment> {
        let targets: Vec<(String, String)> = stash
            .reserved_scopes
            .iter()
            .filter(|scope| scope.0 != PTU_TEAM)
            .cloned()
            .collect();
        self.tpm_operations(&targets, &stash.reserved_scopes, actual, reserved)
    }

    fn tpm_operations(
        &self,
        targets: &[(String, String)],
        reserved_scopes: &BTreeSet<(String, String)>,
        actual: i64,
        reserved: i64,
    ) -> Vec<Increment> {
        targets
            .iter()
            .filter_map(|(key, value)| {
                let target = (key.clone(), value.clone());
                let amount = if reserved_scopes.contains(&target) {
                    actual - reserved
                } else {
                    actual
                };
                (amount != 0).then(|| {
                    self.engine.tokens_increment(
                        &litellm_redis_store::rate_limits::Scope {
                            descriptor: key.clone(),
                            value: value.clone(),
                        },
                        amount,
                    )
                })
            })
            .collect()
    }

    pub async fn settle_failure(
        &self,
        state: &LimiterState,
        facts: FailureFacts,
        request: Option<&RequestRedis>,
    ) {
        let ctx = Ctx {
            policy: Policy::default(),
            request,
        };
        {
            let stash = state.stash();
            if !stash.owned || !stash.for_call(facts.call_id.as_deref()) {
                return;
            }
        }
        self.release_stashed(state, true, ctx).await;
        let operations = {
            let mut stash = state.stash();
            let reserved = if stash.reservation_released {
                0
            } else {
                stash.reserved_tokens
            };
            if reserved <= 0 {
                Vec::new()
            } else {
                let operations = self.failure_operations(&stash, facts.tpm_actual, reserved);
                stash.reservation_released = true;
                operations
            }
        };
        if !operations.is_empty() {
            self.engine.increment_many_post_call(&operations, ctx).await;
        }
    }

    pub async fn settle_success(
        &self,
        state: &LimiterState,
        facts: SuccessFacts,
        request: Option<&RequestRedis>,
    ) {
        let ctx = Ctx {
            policy: facts.policy.into(),
            request,
        };
        {
            let stash = state.stash();
            if !stash.owned || !stash.for_call(facts.call_id.as_deref()) {
                return;
            }
        }
        self.release_stashed(state, true, ctx).await;
        let operations = {
            let mut stash = state.stash();
            let operations = if facts.exempt {
                Vec::new()
            } else {
                self.success_operations(&stash, &facts)
            };
            stash.reserved_tokens = 0;
            operations
        };
        self.engine.apply_usage(&operations, ctx).await;
    }

    fn success_operations(&self, stash: &Stash, facts: &SuccessFacts) -> Vec<Op> {
        let reserved_tokens = if stash.reservation_released {
            0
        } else {
            stash.reserved_tokens
        };
        let reserved_model = stash.reserved_model.as_ref();
        let served = facts.served_model.as_ref();
        let fell_back = match (reserved_model, served) {
            (Some(reserved), Some(served)) => {
                served.requested != reserved.requested
                    && stash.fallback_models.contains(&served.requested)
            }
            _ => false,
        };
        let reconcile = if fell_back || reserved_model.is_none() {
            served
        } else {
            reserved_model
        };
        let reconcile_facts = reconcile.and_then(|model| facts.models.get(&model.requested));
        let candidates: Vec<Candidate> = reconcile_facts
            .map(|facts| facts.candidates.clone())
            .unwrap_or_default();
        let groups: Vec<String> = candidates
            .iter()
            .map(|candidate| candidate.group.clone())
            .collect();
        let mut targets = collect_targets(
            &facts.targets,
            reconcile.map(|model| model.group.as_str()),
            &groups,
            &stash.tpm_limited_tags,
        );
        for scope in &stash.reserved_scopes {
            if (scope.0 == "agent" || scope.0 == "agent_session") && !targets.contains(scope) {
                targets.push(scope.clone());
            }
        }
        let key_owned: BTreeSet<(String, String)> = candidates
            .iter()
            .filter(|model| {
                candidates
                    .iter()
                    .filter(|candidate| candidate.group == model.group)
                    .all(|candidate| candidate.key_owns)
            })
            .map(|model| {
                (
                    "model_per_team".to_owned(),
                    format!("{}:{}", facts.team_label, model.group),
                )
            })
            .collect();
        let charged: Vec<(String, String)> = targets
            .into_iter()
            .filter(|target| !key_owned.contains(target))
            .collect();
        let abandoned: Vec<(String, String)> = if fell_back {
            stash
                .reserved_scopes
                .iter()
                .filter(|scope| {
                    PER_MODEL_KEYS.contains(&scope.0.as_str()) && !charged.contains(scope)
                })
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        let mut operations: Vec<Op> = self
            .tpm_operations(
                &charged,
                &stash.reserved_scopes,
                facts.total_tokens,
                reserved_tokens,
            )
            .into_iter()
            .chain(self.tpm_operations(&abandoned, &stash.reserved_scopes, 0, reserved_tokens))
            .map(Op::Plain)
            .collect();
        if let Some(ptu) = reconcile_facts.and_then(|facts| facts.ptu.as_ref()) {
            operations.extend(
                self.tpm_operations(
                    &[(PTU_TEAM.to_owned(), ptu.value.clone())],
                    &stash.reserved_scopes,
                    ptu.actual_tokens,
                    0,
                )
                .into_iter()
                .map(Op::Plain),
            );
        }
        operations
    }

    pub async fn fallback_target_allowed(
        &self,
        state: Option<&LimiterState>,
        descriptors: Vec<Descriptor>,
        model: &str,
        policy: PolicyFacts,
        request: Option<&RequestRedis>,
    ) -> Result<bool, Box<Abort>> {
        let ctx = Ctx {
            policy: policy.into(),
            request,
        };
        let allowed = descriptors.is_empty()
            || !self
                .engine
                .should_rate_limit(&descriptors, false, false, None, false, ctx)
                .await
                .map_err(|Unverifiable| Box::new(refusal::unverifiable()))?
                .is_over();
        if allowed && let Some(state) = state {
            state.stash().fallback_models.insert(model.to_owned());
        }
        Ok(allowed)
    }

    fn log_prepare(&self, state: &LimiterState, input: &HookInput) -> HookOutput {
        let facts: CallIdFacts = decode(input)?;
        let statuses = {
            let stash = state.stash();
            if !stash.owned || !stash.for_call(facts.call_id.as_deref()) {
                return Ok(None);
            }
            stash
                .rate_limit_response
                .as_ref()
                .map(|response| response.statuses.clone())
                .unwrap_or_default()
        };
        if statuses.is_empty() {
            return Ok(None);
        }
        let Some(Value::Object(payload)) =
            input.fields.raw(Field::StandardLoggingPayload)?.cloned()
        else {
            return Ok(None);
        };
        let mut hidden = match payload.get("hidden_params") {
            Some(Value::Object(hidden)) => hidden.clone(),
            _ => Map::new(),
        };
        let mut headers = match hidden.get("additional_headers") {
            Some(Value::Object(headers)) => headers.clone(),
            _ => Map::new(),
        };
        headers.extend(header_map(&statuses));
        hidden.insert("additional_headers".into(), Value::Object(headers));
        Ok(Some(Verdict::Patch(Patch::default().merge(
            Field::StandardLoggingPayload,
            Map::from_iter([("hidden_params".to_owned(), Value::Object(hidden))]),
        ))))
    }
}

fn collect_targets(
    facts: &TargetFacts,
    model_group: Option<&str>,
    model_groups: &[String],
    tags: &BTreeSet<String>,
) -> Vec<(String, String)> {
    let key = truthy(&facts.user_api_key_hash);
    let user = truthy(&facts.user_api_key_user_id);
    let team = truthy(&facts.user_api_key_team_id);
    let organization = truthy(&facts.user_api_key_org_id);
    let project = truthy(&facts.user_api_key_project_id);
    let end_user = truthy(&facts.end_user_id);
    let agent = truthy(&facts.agent_id);
    let session = truthy(&facts.session_id);
    let mut targets: Vec<(String, String)> = Vec::new();
    let mut push = |key: &str, value: String| targets.push((key.to_owned(), value));
    if let Some(key) = key {
        push("api_key", key.to_owned());
    }
    if let Some(user) = user {
        push("user", user.to_owned());
    }
    if let Some(team) = team {
        push("team", team.to_owned());
    }
    if let (Some(team), Some(user)) = (team, user) {
        push("team_member", format!("{team}:{user}"));
    }
    if let Some(end_user) = end_user {
        push("end_user", end_user.to_owned());
    }
    if let Some(organization) = organization {
        push("organization", organization.to_owned());
    }
    let owners = [
        ("model_per_key", key),
        ("model_per_team", team),
        ("model_per_end_user", end_user),
        ("model_per_organization", organization),
        ("model_per_project", project),
    ];
    if let Some(model_group) = model_group {
        let mut seen: Vec<&str> = Vec::new();
        let groups: Vec<&str> = if model_groups.is_empty() {
            vec![model_group]
        } else {
            model_groups.iter().map(String::as_str).collect()
        };
        for group in groups {
            if seen.contains(&group) {
                continue;
            }
            seen.push(group);
            for (scope, owner) in owners {
                if let Some(owner) = owner {
                    push(scope, format!("{owner}:{group}"));
                }
            }
        }
    }
    if let Some(agent) = agent {
        push("agent", agent.to_owned());
        if let Some(session) = session {
            push("agent_session", format!("{agent}:{session}"));
        }
    }
    for tag in tags {
        push("tag", tag.clone());
    }
    targets
}

impl Callback for ParallelRequestLimiter {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn request_state(&self) -> Option<RequestState> {
        Some(Arc::new(LimiterState::default()))
    }

    fn handle<'a>(&'a self, input: &'a HookInput) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            let Some(state) = input.state::<LimiterState>() else {
                return Ok(None);
            };
            match input.hook {
                Hook::PreCall => {
                    let facts: PreCallFacts = decode(input)?;
                    let request = state.request();
                    Ok(self
                        .pre_call(state, facts, request.as_ref())
                        .await
                        .map(Verdict::Abort))
                }
                Hook::PostCallSuccess => {
                    self.post_call_success(state, decode(input)?).await;
                    Ok(None)
                }
                Hook::PostCallFailure => {
                    self.post_call_failure(state, decode(input)?).await;
                    Ok(None)
                }
                Hook::LogPrepare => self.log_prepare(state, input),
                _ => Ok(None),
            }
        })
    }
}
