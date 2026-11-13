use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use crate::{
    error::CapacityError,
    model::{
        BorrowingLimits, Counter, CounterKind, Descriptor, GroupInfo, Increment, KeyFacts,
        LimiterResponse, MODEL_SATURATION_CHECK, PRIORITY_CLASS_SHARE_AT_SATURATION,
        PRIORITY_MODEL, RateLimit, ReservationOp, Settings, TEAM_MODEL_LIMIT_AT_SATURATION,
        TeamShare, Window,
    },
    priority::{
        allocation, counter_value, normalized_weights, pool_member, priority_class, team_member,
        team_share,
    },
    pyfmt,
    refusal::{Context, Refusal},
    seam::{CapacityHost, Estimate, Throttle, V3Limiter},
    state::{Hold, SharedStash},
};

pub struct Engine {
    pub limiter: Arc<dyn V3Limiter>,
    pub host: Arc<dyn CapacityHost>,
}

pub enum AdmitError {
    Refused(Refusal),
    Failed(CapacityError),
}

impl From<CapacityError> for AdmitError {
    fn from(error: CapacityError) -> Self {
        Self::Failed(error)
    }
}

pub struct Steered {
    pub model: String,
    pub group: GroupInfo,
    pub access_groups: Vec<String>,
}

pub enum Frame {
    Allow,
    Refuse(Refusal),
    Unverifiable(CapacityError),
}

const SOCKET_TURN_UNCOUNTED: [&str; 3] = [
    "model_per_project_itpm",
    "model_per_project_otpm",
    "model_per_team_ptu",
];

fn socket_turn_tpm_windows(descriptors: &[Descriptor]) -> Vec<Descriptor> {
    descriptors
        .iter()
        .filter(|descriptor| !SOCKET_TURN_UNCOUNTED.contains(&descriptor.key.as_str()))
        .filter_map(|descriptor| {
            let tokens = descriptor.tokens_per_unit()?;
            Some(Descriptor {
                key: descriptor.key.clone(),
                value: descriptor.value.clone(),
                rate_limit: Some(RateLimit {
                    requests_per_unit: None,
                    tokens_per_unit: Some(tokens),
                    window_size: descriptor
                        .rate_limit
                        .as_ref()
                        .and_then(|limit| limit.window_size),
                }),
            })
        })
        .collect()
}

pub enum PreCall {
    Allow,
    Steer(Steered),
    Refuse(Refusal),
    Raise(CapacityError),
}

pub struct BorrowingShare {
    pub limits: BorrowingLimits,
    pub share: Descriptor,
    pub ceiling: Option<Descriptor>,
}

impl BorrowingShare {
    pub fn enforced(&self, saturated: bool) -> Option<&Descriptor> {
        if saturated {
            Some(&self.share)
        } else {
            self.ceiling.as_ref()
        }
    }
}

pub struct Request<'a> {
    pub key: &'a KeyFacts,
    pub call_type: Option<&'a str>,
    pub settings: &'a Settings,
    pub stash: &'a SharedStash,
}

impl Request<'_> {
    pub fn priority(&self) -> Option<Value> {
        self.key.priority.clone()
    }
}

#[derive(Deserialize)]
struct ModelPreference {
    model: String,
    #[serde(default)]
    alternates: Vec<String>,
}

pub fn alternates_for(key: &KeyFacts, model: &str) -> Vec<String> {
    let Some(raw) = key.model_preferences.as_ref().filter(|raw| !raw.is_null()) else {
        return Vec::new();
    };
    let Ok(preferences) = Vec::<ModelPreference>::deserialize(raw) else {
        return Vec::new();
    };
    preferences
        .into_iter()
        .find(|preference| preference.model == model)
        .map(|preference| preference.alternates)
        .unwrap_or_default()
}

fn priced_within(preferred: &GroupInfo, alternate: &GroupInfo) -> bool {
    let (Some(input), Some(output)) = (
        alternate.input_cost_per_token,
        alternate.output_cost_per_token,
    ) else {
        return false;
    };
    input <= preferred.input_cost_per_token.unwrap_or(0.0)
        && output <= preferred.output_cost_per_token.unwrap_or(0.0)
}

pub fn reservation_covers(
    reservation: Option<&Value>,
    access_groups: &[String],
    preferred: &GroupInfo,
    alternate: &GroupInfo,
) -> Result<bool, CapacityError> {
    let Some(reservation) = reservation.filter(|value| !value.is_null()) else {
        return Ok(true);
    };
    let Value::Object(reservation) = reservation else {
        return Err(CapacityError::Config(
            "budget_reservation is not a mapping".into(),
        ));
    };
    if !priced_within(preferred, alternate) {
        return Ok(false);
    }
    let Some(Value::Array(entries)) = reservation.get("entries") else {
        return Ok(true);
    };
    if entries.iter().any(|entry| !entry.is_object()) {
        return Ok(false);
    }
    Ok(entries
        .iter()
        .filter(|entry| {
            entry.get("entity_type").and_then(Value::as_str) == Some("Model access group")
        })
        .all(|entry| {
            entry
                .get("entity_id")
                .and_then(Value::as_str)
                .is_some_and(|id| access_groups.iter().any(|group| group == id))
        }))
}

fn python_optional(value: Option<&str>) -> &str {
    value.unwrap_or("None")
}

fn python_int(value: &Value) -> Result<i64, CapacityError> {
    match value {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|float| float.trunc() as i64))
            .ok_or_else(|| CapacityError::Config(format!("{number} is not an int"))),
        Value::Bool(flag) => Ok(i64::from(*flag)),
        Value::String(text) => pyfmt::numeric_text(text)
            .and_then(|text| text.parse::<i64>().ok())
            .ok_or_else(|| {
                CapacityError::Config(format!(
                    "invalid literal for int() with base 10: {}",
                    pyfmt::str_repr(text)
                ))
            }),
        other => Err(CapacityError::Config(format!("{other} is not an int"))),
    }
}

fn python_float(value: &Value) -> Result<f64, CapacityError> {
    match value {
        Value::Number(number) => number
            .as_f64()
            .ok_or_else(|| CapacityError::Config(format!("{number} is not a float"))),
        Value::Bool(flag) => Ok(f64::from(u8::from(*flag))),
        Value::String(text) => pyfmt::numeric_text(text)
            .and_then(|text| text.parse::<f64>().ok())
            .ok_or_else(|| {
                CapacityError::Config(format!(
                    "could not convert string to float: {}",
                    pyfmt::str_repr(text)
                ))
            }),
        other => Err(CapacityError::Config(format!("{other} is not a float"))),
    }
}

pub fn team_increment(
    descriptor: Option<&Descriptor>,
    amount: i64,
    ttl: i64,
) -> Vec<ReservationOp> {
    let Some(descriptor) = descriptor.filter(|_| amount != 0) else {
        return Vec::new();
    };
    if descriptor.tokens_per_unit().is_none() {
        return Vec::new();
    }
    vec![ReservationOp::Increment {
        counter: descriptor.counter(CounterKind::Tokens),
        amount,
        ttl: Some(ttl),
    }]
}

pub fn window_guarded(
    operations: Vec<ReservationOp>,
    window: Option<&Window>,
) -> Vec<ReservationOp> {
    let Some(window) = window else {
        return operations;
    };
    operations
        .into_iter()
        .map(|operation| match operation {
            ReservationOp::Increment {
                counter,
                amount,
                ttl,
            } if counter == window.counter => ReservationOp::Guarded {
                counter,
                amount,
                ttl,
                expected_window_start: window.start.clone(),
                backend: window.backend.clone(),
            },
            other => other,
        })
        .collect()
}

fn tightest(descriptors: &[Option<&Descriptor>]) -> Option<(Descriptor, i64)> {
    let mut best: Option<(Descriptor, i64)> = None;
    for descriptor in descriptors.iter().flatten() {
        let Some(tokens) = descriptor.tokens_per_unit() else {
            continue;
        };
        if best.as_ref().is_none_or(|(_, least)| tokens < *least) {
            best = Some(((*descriptor).clone(), tokens));
        }
    }
    best
}

pub struct Settlement<'a> {
    pub model_group: Option<&'a str>,
    pub team_id: Option<&'a str>,
    pub team_alias: Option<&'a str>,
}

impl Engine {
    pub async fn saturation(&self, settings: &Settings, model: &str, group: &GroupInfo) -> f64 {
        match self.read_saturation(settings, model, group).await {
            Ok(saturation) => saturation,
            Err(error) => {
                let message = format!(
                    "Error checking saturation for {model}: {}",
                    error.python_message()
                );
                let _ = self.host.log("error", &message).await;
                0.0
            }
        }
    }

    async fn read_saturation(
        &self,
        settings: &Settings,
        model: &str,
        group: &GroupInfo,
    ) -> Result<f64, CapacityError> {
        let mut saturation: f64 = 0.0;
        let ttl = &settings.saturation_check_cache_ttl;
        if let Some(rpm) = group.rpm.filter(|rpm| *rpm > 0) {
            let counter = scope(MODEL_SATURATION_CHECK, model).with_kind(CounterKind::Requests);
            if let Some(value) = self.counter_value(&counter, ttl).await? {
                saturation = saturation.max(python_int(&value)? as f64 / rpm as f64);
            }
        }
        if let Some(tpm) = group.tpm.filter(|tpm| *tpm > 0) {
            let counter = scope(MODEL_SATURATION_CHECK, model).with_kind(CounterKind::Tokens);
            if let Some(value) = self.counter_value(&counter, ttl).await? {
                saturation = saturation.max(python_float(&value)? / tpm as f64);
            }
        }
        Ok(saturation)
    }

    async fn counter_value(
        &self,
        counter: &Counter,
        ttl: &Value,
    ) -> Result<Option<Value>, CapacityError> {
        let value = self.limiter.read_counter(counter, ttl).await?;
        Ok(value.filter(|value| !value.is_null() && !value.is_object()))
    }

    fn model_wide(settings: &Settings, model: &str, group: &GroupInfo) -> Descriptor {
        Descriptor {
            key: MODEL_SATURATION_CHECK.into(),
            value: model.into(),
            rate_limit: Some(RateLimit {
                requests_per_unit: group.rpm.filter(|rpm| *rpm != 0),
                tokens_per_unit: group.tpm.filter(|tpm| *tpm != 0),
                window_size: Some(settings.window_size),
            }),
        }
    }

    async fn class_share(
        &self,
        request: &Request<'_>,
        model: &str,
        weights: &[(Value, f64)],
    ) -> Result<Option<TeamShare>, CapacityError> {
        let settings = request.settings;
        if settings.priority_reservation.is_none() || !settings.class_split {
            return Ok(None);
        }
        let priority = request.priority();
        let class = priority_class(settings, priority.as_ref())?;
        let member = request
            .key
            .team_id
            .as_deref()
            .map_or_else(|| pool_member(&class), team_member);
        let members = match self
            .host
            .touch(model, &member, &class, settings.window_size)
            .await
        {
            Ok(members) => members,
            Err(error) if error.is_registry_unavailable() => return Ok(None),
            Err(error) => return Err(error),
        };
        if request.key.team_id.is_none() {
            return Ok(None);
        }
        Ok(Some(team_share(
            &class,
            &members,
            weights,
            settings.default_priority,
        )))
    }

    async fn priority_descriptors(
        &self,
        request: &Request<'_>,
        model: &str,
        group: &GroupInfo,
        share: Option<&TeamShare>,
    ) -> Result<Vec<Descriptor>, CapacityError> {
        let settings = request.settings;
        if settings.priority_reservation.is_none() {
            return Ok(Vec::new());
        }
        let weights = normalized_weights(settings, group)?;
        let priority = request.priority();
        let allocated = allocation(settings, model, priority.as_ref(), &weights, group)?;
        if allocated.unlicensed {
            self.host.unlicensed_priority().await?;
        }
        let weight = share.map_or(allocated.weight, |share| share.fraction);
        let floor = if share.is_some() { 1 } else { 0 };
        let tokens = group.tpm.map(|tpm| floor.max((tpm as f64 * weight) as i64));
        let requests = group.rpm.map(|rpm| floor.max((rpm as f64 * weight) as i64));
        if tokens.is_none() && requests.is_none() {
            return Ok(Vec::new());
        }
        let rate_limit = RateLimit {
            requests_per_unit: requests,
            tokens_per_unit: tokens,
            window_size: Some(settings.window_size),
        };
        Ok(vec![match share {
            None => Descriptor {
                key: PRIORITY_MODEL.into(),
                value: allocated.key,
                rate_limit: Some(rate_limit),
            },
            Some(_) => Descriptor {
                key: PRIORITY_CLASS_SHARE_AT_SATURATION.into(),
                value: counter_value(model, request.key.team_id.as_deref().unwrap_or("None")),
                rate_limit: Some(rate_limit),
            },
        }])
    }

    pub async fn borrowing_share(
        &self,
        request: &Request<'_>,
        model: &str,
    ) -> Result<Option<BorrowingShare>, CapacityError> {
        let Some(limits) = self.limiter.borrowing_limits(model).await? else {
            return Ok(None);
        };
        if limits.tpm.is_none() && limits.rpm.is_none() {
            return Ok(None);
        }
        let value = format!(
            "{}:{}",
            pyfmt::value(request.key.team_id.clone().map(Value::from).as_ref()),
            limits.group
        );
        let descriptor = |tpm: Option<i64>, rpm: Option<i64>| Descriptor {
            key: TEAM_MODEL_LIMIT_AT_SATURATION.into(),
            value: value.clone(),
            rate_limit: Some(RateLimit {
                requests_per_unit: rpm,
                tokens_per_unit: tpm,
                window_size: Some(request.settings.window_size),
            }),
        };
        let share = descriptor(limits.tpm, limits.rpm);
        let ceiling = limits.ceiling().map(|(tpm, rpm)| descriptor(tpm, rpm));
        Ok(Some(BorrowingShare {
            limits,
            share,
            ceiling,
        }))
    }

    pub async fn token_reservation(
        &self,
        descriptor: &Descriptor,
        model: &str,
        call_type: Option<&str>,
    ) -> Result<i64, CapacityError> {
        let Some(tpm_share) = descriptor.tokens_per_unit() else {
            return Ok(0);
        };
        Ok(
            match self
                .limiter
                .estimate_tokens(tpm_share, model, call_type)
                .await?
            {
                Estimate::NoBody => 0,
                Estimate::Failed => 1,
                Estimate::Tokens(tokens) => tokens.max(1),
            },
        )
    }

    pub async fn admit(
        &self,
        request: &Request<'_>,
        model: &str,
        group: &GroupInfo,
    ) -> Result<(), AdmitError> {
        let saturation = self.saturation(request.settings, model, group).await;
        self.check_rate_limits(request, model, group, saturation)
            .await
    }

    async fn check_rate_limits(
        &self,
        request: &Request<'_>,
        model: &str,
        group: &GroupInfo,
        saturation: f64,
    ) -> Result<(), AdmitError> {
        let settings = request.settings;
        let enforce_priority = saturation >= settings.saturation_threshold;
        let model_wide = Self::model_wide(settings, model, group);
        let weights = normalized_weights(settings, group)?;
        let share = self.class_share(request, model, &weights).await?;
        let priority_descriptors = self
            .priority_descriptors(request, model, group, share.as_ref())
            .await?;
        {
            let mut stash = request.stash.lock();
            stash.priority_class_share = priority_descriptors
                .iter()
                .find(|descriptor| descriptor.key == PRIORITY_CLASS_SHARE_AT_SATURATION)
                .cloned();
            stash.priority = request.priority();
        }
        let team_share = self.borrowing_share(request, model).await?;
        let enforced_team = team_share
            .as_ref()
            .and_then(|share| share.enforced(enforce_priority))
            .cloned();
        request.stash.lock().borrowing_limit = enforced_team.clone();

        let mut enforced = vec![model_wide];
        if enforce_priority {
            enforced.extend(priority_descriptors.iter().cloned());
        }
        let team_reservation = match &enforced_team {
            Some(descriptor) => {
                self.token_reservation(descriptor, model, request.call_type)
                    .await?
            }
            None => 0,
        };
        let mut increments: Vec<Increment> = enforced
            .iter()
            .map(|_| Increment {
                requests: 1,
                tokens: 0,
            })
            .collect();
        if let Some(descriptor) = &enforced_team {
            enforced.push(descriptor.clone());
            increments.push(Increment {
                requests: 1,
                tokens: team_reservation,
            });
        }
        let atomic = self
            .limiter
            .check_and_increment(&enforced, &increments)
            .await?;
        if atomic.over_limit() {
            let priority = request.priority();
            let context = Context {
                model,
                priority: priority.as_ref(),
                team_id: request.key.team_id.as_deref(),
                saturation,
                window_size: settings.window_size,
            };
            return Err(AdmitError::Refused(Self::refusal(
                &context,
                &atomic,
                group,
                settings.saturation_threshold,
                team_share.as_ref(),
                share.as_ref(),
            )));
        }
        {
            let mut stash = request.stash.lock();
            stash.reserved_tokens = team_reservation;
            stash.reserved_window = match (&enforced_team, team_reservation) {
                (Some(descriptor), reserved) if reserved != 0 => {
                    atomic.window_of(&descriptor.counter(CounterKind::Tokens))
                }
                _ => None,
            };
        }
        if let Some(tpm_share) = enforced_team.as_ref().and_then(Descriptor::tokens_per_unit)
            && team_reservation != 0
            && settings.rate_limit_type != crate::model::TokenRateLimitType::Input
        {
            self.limiter
                .cap_output(tpm_share, request.call_type)
                .await?;
        }
        if !priority_descriptors.is_empty() && share.is_some() && !enforce_priority {
            self.limiter.publish(&[atomic.id]).await?;
        } else if !priority_descriptors.is_empty() && !enforce_priority {
            let tracking = self
                .limiter
                .should_rate_limit(&priority_descriptors, false, false)
                .await?;
            self.limiter.publish(&[atomic.id, tracking.id]).await?;
        } else {
            self.limiter.publish(&[atomic.id]).await?;
        }
        Ok(())
    }

    fn refusal(
        context: &Context<'_>,
        atomic: &LimiterResponse,
        group: &GroupInfo,
        threshold: f64,
        team_share: Option<&BorrowingShare>,
        class_share: Option<&TeamShare>,
    ) -> Refusal {
        for status in atomic.statuses.iter().filter(|status| status.over_limit()) {
            match status.descriptor_key.as_str() {
                MODEL_SATURATION_CHECK => return context.model_capacity(status, group),
                PRIORITY_MODEL => return context.priority_pool(status, group),
                TEAM_MODEL_LIMIT_AT_SATURATION => {
                    if let Some(team_share) = team_share {
                        return context.team_share(status, threshold, &team_share.limits);
                    }
                }
                PRIORITY_CLASS_SHARE_AT_SATURATION => {
                    if let Some(share) = class_share {
                        return context.class_share(status, share);
                    }
                }
                _ => {}
            }
        }
        context.unknown(atomic.statuses.iter().find(|status| status.over_limit()))
    }

    pub async fn pre_call(&self, request: &Request<'_>, model: &str, group: &GroupInfo) -> PreCall {
        let steered = self.steer_to_alternate(request, model, group).await;
        match self
            .admit_steered_or_preferred(request, model, group, steered)
            .await
        {
            Ok(None) => PreCall::Allow,
            Ok(Some(steered)) => PreCall::Steer(steered),
            Err(AdmitError::Refused(refusal)) => {
                self.throttled(&refusal, model, request.key).await;
                PreCall::Refuse(refusal)
            }
            Err(AdmitError::Failed(error)) if error.is_http() => PreCall::Raise(error),
            Err(AdmitError::Failed(error)) => {
                let message = format!(
                    "Error in dynamic rate limiter: {}, allowing request",
                    error.python_message()
                );
                let _ = self.host.log("error", &message).await;
                PreCall::Allow
            }
        }
    }

    async fn admit_steered_or_preferred(
        &self,
        request: &Request<'_>,
        model: &str,
        group: &GroupInfo,
        steered: Option<Steered>,
    ) -> Result<Option<Steered>, AdmitError> {
        if let Some(steered) = steered {
            if self
                .admit(request, &steered.model, &steered.group)
                .await
                .is_ok()
            {
                return Ok(Some(steered));
            }
            let message = format!(
                "Alternate {} refused team {}, staying on {model}",
                steered.model,
                python_optional(request.key.team_id.as_deref())
            );
            let _ = self.host.log("info", &message).await;
        }
        self.admit(request, model, group).await?;
        Ok(None)
    }

    async fn steer_to_alternate(
        &self,
        request: &Request<'_>,
        model: &str,
        group: &GroupInfo,
    ) -> Option<Steered> {
        match self.pick_alternate(request, model, group).await {
            Ok(steered) => steered,
            Err(_) => {
                let message = format!(
                    "Alternate check failed for team {}, staying on {model}",
                    python_optional(request.key.team_id.as_deref())
                );
                let _ = self.host.log("warning", &message).await;
                None
            }
        }
    }

    async fn pick_alternate(
        &self,
        request: &Request<'_>,
        model: &str,
        group: &GroupInfo,
    ) -> Result<Option<Steered>, CapacityError> {
        let alternates = alternates_for(request.key, model);
        if alternates.is_empty() {
            return Ok(None);
        }
        let threshold = request.settings.saturation_threshold;
        if self.saturation(request.settings, model, group).await < threshold {
            return Ok(None);
        }
        for alternate in &alternates {
            if alternate == model
                || !self
                    .host
                    .steerable(
                        model,
                        alternate,
                        request.call_type,
                        request.key.team_id.as_deref(),
                    )
                    .await?
            {
                continue;
            }
            if let Some(candidate) = self
                .idle_permitted_alternate(request, alternate, threshold, group)
                .await?
            {
                let message = format!(
                    "Steering team {} from {model} to {alternate}",
                    python_optional(request.key.team_id.as_deref())
                );
                let _ = self.host.log("info", &message).await;
                return Ok(Some(candidate));
            }
        }
        Ok(None)
    }

    async fn idle_permitted_alternate(
        &self,
        request: &Request<'_>,
        alternate: &str,
        threshold: f64,
        preferred: &GroupInfo,
    ) -> Result<Option<Steered>, CapacityError> {
        let Some(group) = self.host.group(alternate).await? else {
            return Ok(None);
        };
        let Some(access_groups) = self.host.authorize(alternate).await? else {
            return Ok(None);
        };
        if !reservation_covers(
            request.key.budget_reservation.as_ref(),
            &access_groups,
            preferred,
            &group,
        )? {
            return Ok(None);
        }
        if self.saturation(request.settings, alternate, &group).await >= threshold {
            return Ok(None);
        }
        Ok(Some(Steered {
            model: alternate.to_owned(),
            group,
            access_groups,
        }))
    }

    async fn throttled(&self, refusal: &Refusal, model: &str, key: &KeyFacts) {
        let (Some(team), Some(reason)) =
            (key.team_id.as_deref(), refusal.header("rate_limited_by"))
        else {
            return;
        };
        let or_default = |value: Option<&Value>, default: &str| match value {
            Some(value) if pyfmt::truthy(Some(value)) => value.clone(),
            _ => Value::from(default),
        };
        let event = Throttle {
            model_group: model.to_owned(),
            team: team.to_owned(),
            team_alias: key.team_alias.clone(),
            priority: or_default(refusal.header("x-litellm-priority"), "default"),
            rate_limit_type: pyfmt::value(Some(&or_default(
                refusal.header("rate_limit_type"),
                "tokens",
            ))),
            reason: pyfmt::value(Some(reason)),
        };
        let _ = self.host.throttled(&event).await;
    }

    pub async fn enforce_frame(
        &self,
        request: &Request<'_>,
        model: &str,
        connection: Option<&SharedStash>,
    ) -> Result<Frame, CapacityError> {
        let Some(group) = self.host.group(model).await? else {
            return Ok(Frame::Allow);
        };
        let settings = request.settings;
        let weights = normalized_weights(settings, &group)?;
        let share = self.class_share(request, model, &weights).await?;
        let share_descriptors = match &share {
            None => Vec::new(),
            Some(share) => {
                self.priority_descriptors(request, model, &group, Some(share))
                    .await?
            }
        };
        let frame_share = share_descriptors
            .iter()
            .find(|descriptor| descriptor.key == PRIORITY_CLASS_SHARE_AT_SATURATION)
            .cloned();
        if let (Some(stash), Some(frame_share)) = (connection, frame_share) {
            stash.lock().priority_class_share = Some(frame_share);
        }
        let borrowing = self.borrowing_share(request, model).await?;
        if share_descriptors.is_empty() && borrowing.is_none() {
            return Ok(Frame::Allow);
        }
        let saturation = self.saturation(settings, model, &group).await;
        let saturated = saturation >= settings.saturation_threshold;
        let enforced_team = borrowing
            .as_ref()
            .and_then(|borrowing| borrowing.enforced(saturated))
            .cloned();
        if let (Some(stash), Some(_)) = (connection, &borrowing) {
            stash.lock().borrowing_limit = enforced_team.clone();
        }
        let mut descriptors = if saturated {
            share_descriptors
        } else {
            Vec::new()
        };
        descriptors.extend(enforced_team);
        if descriptors.is_empty() {
            return Ok(Frame::Allow);
        }
        let windows = socket_turn_tpm_windows(&descriptors);
        let response = match self.limiter.should_rate_limit(&windows, true, true).await {
            Ok(response) => response,
            Err(error) => return Ok(Frame::Unverifiable(error)),
        };
        let Some(over_limit) = response.statuses.iter().find(|status| status.over_limit()) else {
            return Ok(Frame::Allow);
        };
        let priority = request.priority();
        let context = Context {
            model,
            priority: priority.as_ref(),
            team_id: request.key.team_id.as_deref(),
            saturation,
            window_size: settings.window_size,
        };
        let over_class_share = over_limit.descriptor_key == PRIORITY_CLASS_SHARE_AT_SATURATION;
        let refusal = match (&share, &borrowing) {
            (Some(share), _) if over_class_share || borrowing.is_none() => {
                context.class_share(over_limit, share)
            }
            (_, Some(borrowing)) => {
                context.team_share(over_limit, settings.saturation_threshold, &borrowing.limits)
            }
            _ => return Ok(Frame::Allow),
        };
        self.throttled(&refusal, model, request.key).await;
        Ok(Frame::Refuse(refusal))
    }

    pub async fn fallback_target_allowed(
        &self,
        request: &Request<'_>,
        model: &str,
        commit: bool,
    ) -> Result<bool, CapacityError> {
        let team_share = self.borrowing_share(request, model).await?;
        let group = self.host.group(model).await?;
        let (Some(team_share), Some(group)) = (team_share, group) else {
            return Ok(true);
        };
        let settings = request.settings;
        let saturation = self.saturation(settings, model, &group).await;
        let enforced = team_share
            .enforced(saturation >= settings.saturation_threshold)
            .cloned();
        let reservation = match &enforced {
            Some(descriptor) => {
                let call_type = self.limiter.stash_call_type().await?;
                self.token_reservation(descriptor, model, call_type.as_deref())
                    .await?
            }
            None => 0,
        };
        if !commit {
            return match &enforced {
                None => Ok(true),
                Some(descriptor) => self.share_has_room(descriptor, reservation).await,
            };
        }
        let mut descriptors = vec![Self::model_wide(settings, model, &group)];
        let mut increments = vec![Increment {
            requests: 1,
            tokens: 0,
        }];
        if let Some(descriptor) = &enforced {
            descriptors.push(descriptor.clone());
            increments.push(Increment {
                requests: 1,
                tokens: reservation,
            });
        }
        let response = self
            .limiter
            .check_and_increment(&descriptors, &increments)
            .await?;
        if response.over_limit() {
            return Ok(false);
        }
        let window = enforced
            .as_ref()
            .and_then(|descriptor| response.window_of(&descriptor.counter(CounterKind::Tokens)));
        Self::hold_fallback_share(
            request.stash,
            model,
            &team_share.share,
            enforced,
            reservation,
            window,
        );
        Ok(true)
    }

    async fn share_has_room(
        &self,
        descriptor: &Descriptor,
        tokens: i64,
    ) -> Result<bool, CapacityError> {
        let checked = self
            .limiter
            .should_rate_limit(std::slice::from_ref(descriptor), true, false)
            .await?;
        Ok(checked.statuses.iter().all(|status| {
            !status.over_limit()
                && (status.rate_limit_type != "tokens"
                    || status
                        .limit_remaining
                        .as_ref()
                        .and_then(serde_json::Number::as_f64)
                        .unwrap_or(0.0)
                        >= tokens as f64)
        }))
    }

    fn hold_fallback_share(
        stash: &SharedStash,
        model: &str,
        share: &Descriptor,
        charged: Option<Descriptor>,
        reserved_tokens: i64,
        window: Option<Window>,
    ) {
        let mut stash = stash.lock();
        let carried = stash
            .hold(model)
            .filter(|earlier| earlier.window == window)
            .map_or(0, |earlier| earlier.reserved_tokens);
        let hold = Arc::new(Hold {
            descriptor: charged,
            reserved_tokens: reserved_tokens + carried,
            window,
        });
        let group = share
            .value
            .rsplit_once(':')
            .map_or(share.value.as_str(), |(_, tail)| tail);
        stash.put_hold(model, &hold);
        stash.put_hold(group, &hold);
    }

    pub async fn settle_success(
        &self,
        settings: &Settings,
        stash: Option<&SharedStash>,
        settlement: &Settlement<'_>,
        priority: Option<&str>,
        total_tokens: i64,
    ) -> Result<(), CapacityError> {
        let ttl = settings.window_size;
        let model_group = settlement.model_group;
        let (team_tokens, served) = match stash {
            None => (Vec::new(), None),
            Some(stash) => {
                let mut stash = stash.lock();
                let served = model_group.and_then(|group| stash.hold(group));
                let team_tokens = match &served {
                    None => window_guarded(
                        team_increment(
                            stash.borrowing_limit.as_ref(),
                            total_tokens - stash.reserved_tokens,
                            ttl,
                        ),
                        stash.reserved_window.as_ref(),
                    ),
                    Some(hold) => {
                        let mut operations = window_guarded(
                            team_increment(
                                stash.borrowing_limit.as_ref(),
                                -stash.reserved_tokens,
                                ttl,
                            ),
                            stash.reserved_window.as_ref(),
                        );
                        operations.extend(window_guarded(
                            team_increment(
                                hold.descriptor.as_ref(),
                                total_tokens - hold.reserved_tokens,
                                ttl,
                            ),
                            hold.window.as_ref(),
                        ));
                        operations
                    }
                };
                stash.reserved_tokens = 0;
                stash.reserved_window = None;
                if let Some(hold) = &served {
                    stash.drop_hold(hold);
                }
                (team_tokens, served)
            }
        };
        let Some(model_group) = model_group.filter(|group| !group.is_empty()) else {
            if !team_tokens.is_empty() {
                self.limiter.apply(&team_tokens, true).await?;
            }
            return Ok(());
        };
        if total_tokens == 0 && team_tokens.is_empty() {
            return Ok(());
        }
        let priority_key = match settings.reserves_name(priority) {
            Some(explicit) => format!("{model_group}:{explicit}"),
            None => format!("{model_group}:default_pool"),
        };
        let class_share = stash.and_then(|stash| stash.lock().priority_class_share.clone());
        let priority_counter = match &class_share {
            None => scope(PRIORITY_MODEL, &priority_key),
            Some(share) => scope(
                &share.key,
                &settlement.team_id.map_or_else(
                    || share.value.clone(),
                    |team| counter_value(model_group, team),
                ),
            ),
        }
        .with_kind(CounterKind::Tokens);
        let mut operations = Vec::new();
        if total_tokens != 0 {
            operations.push(ReservationOp::Increment {
                counter: scope(MODEL_SATURATION_CHECK, model_group).with_kind(CounterKind::Tokens),
                amount: total_tokens,
                ttl: Some(ttl),
            });
            operations.push(ReservationOp::Increment {
                counter: priority_counter,
                amount: total_tokens,
                ttl: Some(ttl),
            });
        }
        operations.extend(team_tokens);
        if !operations.is_empty() {
            self.limiter.apply(&operations, true).await?;
            self.record_team_share(
                settings,
                stash,
                Some(model_group),
                settlement,
                served.as_deref(),
            )
            .await;
        }
        Ok(())
    }

    pub async fn settle_failure(
        &self,
        settings: &Settings,
        stash: &SharedStash,
        failed_group: Option<&str>,
        used: i64,
    ) -> Result<Option<Arc<Hold>>, CapacityError> {
        let ttl = settings.window_size;
        let (refund, failed_hold) = {
            let mut stash = stash.lock();
            let failed_hold = failed_group.and_then(|group| stash.hold(group));
            let admitted_amount =
                if failed_hold.is_none() { used } else { 0 } - stash.reserved_tokens;
            let mut refund = window_guarded(
                team_increment(stash.borrowing_limit.as_ref(), admitted_amount, ttl),
                stash.reserved_window.as_ref(),
            );
            if let Some(hold) = &failed_hold {
                refund.extend(window_guarded(
                    team_increment(hold.descriptor.as_ref(), used - hold.reserved_tokens, ttl),
                    hold.window.as_ref(),
                ));
            }
            stash.reserved_tokens = 0;
            stash.reserved_window = None;
            if let Some(hold) = &failed_hold {
                stash.drop_hold(hold);
            }
            (refund, failed_hold)
        };
        if !refund.is_empty() {
            self.limiter.apply(&refund, false).await?;
        }
        Ok(failed_hold)
    }

    pub async fn record_team_share(
        &self,
        settings: &Settings,
        stash: Option<&SharedStash>,
        model_group: Option<&str>,
        settlement: &Settlement<'_>,
        hold: Option<&Hold>,
    ) {
        let _ = self
            .try_record_team_share(settings, stash, model_group, settlement, hold)
            .await;
    }

    async fn try_record_team_share(
        &self,
        settings: &Settings,
        stash: Option<&SharedStash>,
        model_group: Option<&str>,
        settlement: &Settlement<'_>,
        hold: Option<&Hold>,
    ) -> Result<(), CapacityError> {
        if !self.host.metrics_enabled().await? {
            return Ok(());
        }
        let (Some(stash), Some(model_group), Some(team)) = (stash, model_group, settlement.team_id)
        else {
            return Ok(());
        };
        let (candidates, priority) = {
            let stash = stash.lock();
            let candidates = match hold {
                None => vec![
                    stash.borrowing_limit.clone(),
                    stash.priority_class_share.clone(),
                ],
                Some(hold) => vec![hold.descriptor.clone()],
            };
            (candidates, stash.priority.clone())
        };
        let Some((descriptor, share_tokens)) =
            tightest(&candidates.iter().map(Option::as_ref).collect::<Vec<_>>())
        else {
            return Ok(());
        };
        let used = self
            .counter_value(
                &descriptor.counter(CounterKind::Tokens),
                &settings.saturation_check_cache_ttl,
            )
            .await?;
        let used_tokens = match used {
            None => 0,
            Some(value) => python_float(&value)? as i64,
        };
        let gauge = crate::seam::ShareGauge {
            model_group: model_group.to_owned(),
            team: team.to_owned(),
            team_alias: settlement.team_alias.map(str::to_owned),
            priority: if pyfmt::truthy(priority.as_ref()) {
                priority.unwrap_or(Value::Null)
            } else {
                Value::from("default")
            },
            share_tokens,
            borrowed_tokens: (used_tokens - share_tokens).max(0),
        };
        self.host.set_share(&gauge).await
    }

    pub fn post_call_headers(key: &KeyFacts) -> Vec<(String, Value)> {
        let header = if pyfmt::truthy(key.priority.as_ref()) {
            key.priority.clone().unwrap_or(Value::Null)
        } else {
            Value::from("default")
        };
        let mut headers = Vec::new();
        if pyfmt::is_latin1(&pyfmt::value(Some(&header))) {
            headers.push(("x-litellm-priority".to_owned(), header));
        }
        headers.push((
            "x-litellm-rate-limiter-version".to_owned(),
            Value::from("v3"),
        ));
        headers
    }
}

pub struct Scope {
    key: String,
    value: String,
}

impl Scope {
    pub fn with_kind(self, kind: CounterKind) -> Counter {
        Counter {
            key: self.key,
            value: self.value,
            kind,
        }
    }
}

pub fn scope(key: &str, value: &str) -> Scope {
    Scope {
        key: key.to_owned(),
        value: value.to_owned(),
    }
}
