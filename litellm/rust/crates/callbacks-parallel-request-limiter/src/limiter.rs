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
    descriptor::{Descriptor, PROJECT_ITPM, PROJECT_OTPM, PTU_TEAM},
    engine::{Ctx, Engine, Op, Policy, Settlement, SlotAcquisition, Unverifiable},
    ptu::{Ceiling, Usage, settlement_tokens},
    refusal,
    state::{IoReservation, LimiterState, RateLimitedModel, Stash},
    status::{Code, Response, Status, WindowIdentity, header_map},
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

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
    #[serde(default)]
    pub ptu_ceiling: Option<Ceiling>,
    #[serde(default)]
    pub ptu_estimate: Option<Estimate>,
    #[serde(default)]
    pub io_estimate: Option<Estimate>,
    #[serde(default)]
    pub reuses_request_slot: bool,
    #[serde(default)]
    pub prior_response: Option<PriorResponse>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PriorResponse {
    pub overall_code: Code,
    pub statuses: Vec<Status>,
}

impl From<PriorResponse> for Response {
    fn from(prior: PriorResponse) -> Self {
        Self {
            overall: prior.overall_code,
            statuses: prior.statuses,
            windows: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Estimate {
    pub input_tokens: i64,
    pub output_tokens: i64,
    #[serde(default)]
    pub explicit_output_cap: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IoUsage {
    #[serde(default)]
    pub response: Option<Usage>,
    #[serde(default)]
    pub combined: Option<Usage>,
    #[serde(default)]
    pub aggregate_total: i64,
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
    #[serde(default)]
    pub partial_usage: Option<Usage>,
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
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub io_usage: IoUsage,
    #[serde(default)]
    pub keep_slot: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureFacts {
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub tpm_actual: i64,
    #[serde(default)]
    pub partial_usage: Option<Usage>,
}

fn scopes_keyed(descriptors: &[Descriptor], key: &str) -> BTreeSet<(String, String)> {
    descriptors
        .iter()
        .filter(|descriptor| descriptor.key == key)
        .map(|descriptor| (descriptor.key.clone(), descriptor.value.clone()))
        .collect()
}

fn windows_keyed(windows: &[WindowIdentity], key: &str) -> Vec<WindowIdentity> {
    windows
        .iter()
        .filter(|identity| identity.counter.scope.descriptor == key)
        .cloned()
        .collect()
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
            stash.first_pass_admitted = false;
            stash.io_reached = false;
            stash.ptu_ceiling = facts.ptu_ceiling.clone();
            stash.rate_limit_response = facts.prior_response.clone().map(Response::from);
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
        let reuse = facts.reuses_request_slot;
        let gauges: Vec<_> = if reuse {
            Vec::new()
        } else {
            descriptors
                .iter()
                .filter(|descriptor| descriptor.limits().max_parallel_requests.is_some())
                .map(Descriptor::scope)
                .collect()
        };
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
                reuse,
                ctx,
            )
            .await?;
        if response.is_over() {
            return Ok(self.refuse(&response, facts));
        }
        {
            let mut stash = state.stash();
            stash.first_pass_admitted = true;
            if !reuse {
                stash.rate_limit_response = Some(response);
            }
            if let Some(slot_id) = slot_id {
                stash.parallel_slot = Some(SlotAcquisition { slot_id, gauges });
            }
        }
        let has_tpm_limits = descriptors.iter().any(|descriptor| {
            !descriptor.is_project_io() && descriptor.limits().tokens_per_unit.is_some()
        });
        let mut reservation_amount = 0;
        if has_tpm_limits && settings.tpm_reservation_enabled {
            let estimated = facts.estimated_tokens.unwrap_or(0).max(1);
            let ptu_estimated = match (&facts.ptu_ceiling, facts.ptu_estimate) {
                (Some(ceiling), Some(estimate)) => {
                    ceiling.estimate(estimate.input_tokens, estimate.output_tokens)
                }
                _ => estimated,
            };
            if let Some(ceiling) = &facts.ptu_ceiling
                && ptu_estimated > ceiling.tpm_limit
            {
                self.release_unless_reused(state, facts, ctx).await;
                return Ok(Some(refusal::ptu_share_exceeded(ceiling, ptu_estimated)));
            }
            let tpm_response = self
                .engine
                .reserve_tpm_tokens(descriptors, estimated, &[(PTU_TEAM, ptu_estimated)], ctx)
                .await?;
            if tpm_response.is_over() {
                self.release_unless_reused(state, facts, ctx).await;
                return Ok(self.refuse(&tpm_response, facts));
            }
            let mut stash = state.stash();
            stash.discard_released_tpm_reservation();
            stash.reserved_tokens = estimated;
            stash.ptu_reserved_tokens = ptu_estimated;
            stash.ptu_windows = windows_keyed(tpm_response.windows(), PTU_TEAM);
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
            reservation_amount = estimated;
        }
        self.reserve_io(state, facts, reservation_amount, ctx).await
    }

    async fn reserve_io(
        &self,
        state: &LimiterState,
        facts: &PreCallFacts,
        reservation_amount: i64,
        ctx: Ctx<'_>,
    ) -> Result<Option<Abort>, Unverifiable> {
        let descriptors: Vec<Descriptor> = facts
            .descriptors
            .iter()
            .filter(|descriptor| descriptor.is_project_io())
            .cloned()
            .collect();
        if descriptors.is_empty() {
            return Ok(None);
        }
        state.stash().io_reached = true;
        let estimate = facts.io_estimate.unwrap_or_default();
        let input_tokens = estimate.input_tokens.max(1);
        let output_tokens = if estimate.explicit_output_cap {
            estimate.output_tokens
        } else {
            estimate.output_tokens.max(1)
        };
        let (response, itpm_reserved, otpm_reserved) = self
            .engine
            .reserve_io_tokens(&descriptors, input_tokens, output_tokens, ctx)
            .await?;
        if response.is_over() {
            if reservation_amount > 0 {
                let (scopes, ptu_scopes, ptu_tokens, ptu_windows) = {
                    let stash = state.stash();
                    let (ptu, other): (Vec<_>, Vec<_>) = stash
                        .reserved_scopes
                        .iter()
                        .cloned()
                        .partition(|scope| scope.0 == PTU_TEAM);
                    (
                        other,
                        ptu,
                        stash.ptu_reserved_tokens,
                        stash.ptu_windows.clone(),
                    )
                };
                self.engine
                    .refund_reserved(&scopes, reservation_amount, &[], ctx)
                    .await;
                self.engine
                    .refund_reserved(&ptu_scopes, ptu_tokens, &ptu_windows, ctx)
                    .await;
                state.stash().reservation_released = true;
            }
            self.release_unless_reused(state, facts, ctx).await;
            return Ok(self.refuse(&response, facts));
        }
        let mut stash = state.stash();
        stash.discard_released_io_reservation();
        if itpm_reserved > 0 {
            stash.itpm = IoReservation {
                tokens: itpm_reserved,
                scopes: scopes_keyed(&descriptors, PROJECT_ITPM),
                windows: windows_keyed(response.windows(), PROJECT_ITPM),
            };
        }
        if otpm_reserved > 0 {
            stash.otpm = IoReservation {
                tokens: otpm_reserved,
                scopes: scopes_keyed(&descriptors, PROJECT_OTPM),
                windows: windows_keyed(response.windows(), PROJECT_OTPM),
            };
        }
        match stash.rate_limit_response.as_mut() {
            Some(stored) => stored.statuses.extend(response.statuses),
            None if !response.statuses.is_empty() => stash.rate_limit_response = Some(response),
            None => {}
        }
        Ok(None)
    }

    async fn release_unless_reused(
        &self,
        state: &LimiterState,
        facts: &PreCallFacts,
        ctx: Ctx<'_>,
    ) {
        if !facts.reuses_request_slot {
            self.release_stashed(state, false, ctx).await;
        }
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
        let ctx = Ctx {
            policy: Policy::default(),
            request: None,
        };
        self.release_stashed(state, false, ctx).await;
        let settlements = {
            let mut stash = state.stash();
            let (reserved, itpm, otpm) = Self::open_reservations(&stash);
            if reserved <= 0 && itpm <= 0 && otpm <= 0 {
                return;
            }
            let settlements = self.failure_settlements(
                &stash,
                facts.tpm_actual,
                facts.partial_usage.as_ref(),
                (reserved, itpm, otpm),
            );
            stash.mark_reservation_released();
            settlements
        };
        for settlement in settlements {
            self.engine.settle(settlement, ctx, false).await;
        }
    }

    fn open_reservations(stash: &Stash) -> (i64, i64, i64) {
        let reserved = if stash.reservation_released {
            0
        } else {
            stash.reserved_tokens
        };
        let (itpm, otpm) = if stash.io_reservation_released {
            (0, 0)
        } else {
            (stash.itpm.tokens, stash.otpm.tokens)
        };
        (reserved, itpm, otpm)
    }

    fn failure_settlements(
        &self,
        stash: &Stash,
        tpm_actual: i64,
        partial_usage: Option<&Usage>,
        (reserved, itpm, otpm): (i64, i64, i64),
    ) -> Vec<Settlement> {
        let combined = if reserved > 0 {
            let targets: Vec<(String, String)> = stash
                .reserved_scopes
                .iter()
                .filter(|scope| scope.0 != PTU_TEAM)
                .cloned()
                .collect();
            self.engine
                .reservation_ops(&targets, &stash.reserved_scopes, tpm_actual, reserved)
        } else {
            Vec::new()
        };
        let ptu = if reserved > 0 {
            self.ptu_failure_ops(stash, tpm_actual, partial_usage)
        } else {
            Vec::new()
        };
        vec![
            Settlement::Plain(combined),
            self.io_failure_settlement(
                &stash.itpm,
                partial_usage.map_or(0, Usage::billable_input_tokens),
                itpm,
            ),
            self.io_failure_settlement(
                &stash.otpm,
                partial_usage.map_or(0, |usage| usage.completion_tokens),
                otpm,
            ),
            Settlement::Project(ptu),
        ]
    }

    fn io_failure_settlement(
        &self,
        reservation: &IoReservation,
        actual: i64,
        reserved: i64,
    ) -> Settlement {
        let targets: Vec<(String, String)> = reservation.scopes.iter().cloned().collect();
        if reserved <= 0 {
            return Settlement::Project(Vec::new());
        }
        if reservation.windows.is_empty() {
            return Settlement::Plain(self.engine.reservation_ops(
                &targets,
                &reservation.scopes,
                actual,
                reserved,
            ));
        }
        Settlement::Project(self.engine.project_ops(
            &targets,
            &reservation.scopes,
            actual,
            reserved,
            &reservation.windows,
        ))
    }

    fn ptu_failure_ops(&self, stash: &Stash, tpm_actual: i64, usage: Option<&Usage>) -> Vec<Op> {
        let targets: Vec<(String, String)> = stash
            .reserved_scopes
            .iter()
            .filter(|scope| scope.0 == PTU_TEAM)
            .cloned()
            .collect();
        if targets.is_empty() {
            return Vec::new();
        }
        self.engine.project_ops(
            &targets,
            &stash.reserved_scopes,
            settlement_tokens(stash.ptu_ceiling.as_ref(), usage, tpm_actual),
            stash.ptu_reserved_tokens,
            &stash.ptu_windows,
        )
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
        let settlements = {
            let mut stash = state.stash();
            let open = Self::open_reservations(&stash);
            let settlements = self.failure_settlements(
                &stash,
                facts.tpm_actual,
                facts.partial_usage.as_ref(),
                open,
            );
            if open.0 > 0 || open.1 > 0 || open.2 > 0 {
                stash.mark_reservation_released();
            }
            settlements
        };
        for settlement in settlements {
            self.engine.settle(settlement, ctx, true).await;
        }
    }

    fn tpm_ops(
        &self,
        targets: &[(String, String)],
        reserved_scopes: &BTreeSet<(String, String)>,
        actual: i64,
        reserved: i64,
    ) -> Vec<Increment> {
        self.engine
            .reservation_ops(targets, reserved_scopes, actual, reserved)
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
        if !facts.keep_slot {
            self.release_stashed(state, true, ctx).await;
        }
        let (operations, io) = {
            let mut stash = state.stash();
            let operations = if facts.exempt {
                Vec::new()
            } else {
                self.success_operations(&stash, &facts)
            };
            let io = self.io_success_settlement(&stash, &facts.io_usage);
            stash.clear_reserved_tokens();
            (operations, io)
        };
        self.engine.apply_usage(&operations, ctx).await;
        match io {
            Some(Settlement::Plain(operations)) if !operations.is_empty() => {
                self.engine
                    .increment_with_ttl_preservation(&operations, ctx)
                    .await;
            }
            Some(Settlement::Project(operations)) if !operations.is_empty() => {
                self.engine.reservation_aware(&operations).await;
            }
            _ => {}
        }
    }

    fn io_success_settlement(&self, stash: &Stash, usage: &IoUsage) -> Option<Settlement> {
        let (itpm, otpm) = (stash.itpm.tokens, stash.otpm.tokens);
        if itpm <= 0 && otpm <= 0 {
            return None;
        }
        let released = stash.io_reservation_released;
        if usage.response.is_none()
            && usage.combined.is_none()
            && usage.aggregate_total <= 0
            && !released
        {
            return None;
        }
        let (billable, completion) = match usage.response.or(usage.combined) {
            Some(resolved) => (resolved.billable_input_tokens(), resolved.completion_tokens),
            None if usage.aggregate_total > 0 => (usage.aggregate_total, usage.aggregate_total),
            None => (itpm, otpm),
        };
        let empty = BTreeSet::new();
        let reserved_scopes = |reservation: &IoReservation| {
            if released {
                empty.clone()
            } else {
                reservation.scopes.clone()
            }
        };
        let targets = |reservation: &IoReservation| -> Vec<(String, String)> {
            reservation.scopes.iter().cloned().collect()
        };
        if released || (stash.itpm.windows.is_empty() && stash.otpm.windows.is_empty()) {
            let reserved = |amount: i64| if released { 0 } else { amount };
            let mut operations = self.engine.reservation_ops(
                &targets(&stash.itpm),
                &reserved_scopes(&stash.itpm),
                billable,
                reserved(itpm),
            );
            operations.extend(self.engine.reservation_ops(
                &targets(&stash.otpm),
                &reserved_scopes(&stash.otpm),
                completion,
                reserved(otpm),
            ));
            return Some(Settlement::Plain(operations));
        }
        let mut operations = Vec::new();
        for (reservation, actual, reserved) in [
            (&stash.itpm, billable, itpm),
            (&stash.otpm, completion, otpm),
        ] {
            if reserved > 0 {
                operations.extend(self.engine.project_ops(
                    &targets(reservation),
                    &reserved_scopes(reservation),
                    actual,
                    reserved,
                    &reservation.windows,
                ));
            }
        }
        Some(Settlement::Project(operations))
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
            .tpm_ops(
                &charged,
                &stash.reserved_scopes,
                facts.total_tokens,
                reserved_tokens,
            )
            .into_iter()
            .chain(self.tpm_ops(&abandoned, &stash.reserved_scopes, 0, reserved_tokens))
            .map(Op::Plain)
            .collect();
        let reserved_ptu: Vec<(String, String)> = stash
            .reserved_scopes
            .iter()
            .filter(|scope| scope.0 == PTU_TEAM)
            .cloned()
            .collect();
        if !reserved_ptu.is_empty() {
            operations.extend(self.engine.project_ops(
                &reserved_ptu,
                &stash.reserved_scopes,
                settlement_tokens(
                    stash.ptu_ceiling.as_ref(),
                    facts.usage.as_ref(),
                    facts.total_tokens,
                ),
                if stash.reservation_released {
                    0
                } else {
                    stash.ptu_reserved_tokens
                },
                &stash.ptu_windows,
            ));
            return operations;
        }
        let team = truthy(&facts.targets.user_api_key_team_id);
        let ptu = match (&stash.ptu_ceiling, team, reconcile) {
            (Some(ceiling), Some(team), Some(_)) => Some((
                format!("{team}:{}", ceiling.model_group),
                settlement_tokens(Some(ceiling), facts.usage.as_ref(), facts.total_tokens),
            )),
            (None, Some(_), Some(_)) => reconcile_facts
                .and_then(|facts| facts.ptu.as_ref())
                .map(|ptu| (ptu.value.clone(), ptu.actual_tokens)),
            _ => None,
        };
        if let Some((value, actual)) = ptu {
            operations.extend(
                self.engine
                    .reservation_ops(
                        &[(PTU_TEAM.to_owned(), value)],
                        &stash.reserved_scopes,
                        actual,
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
