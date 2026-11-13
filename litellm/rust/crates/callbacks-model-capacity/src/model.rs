use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};

pub const MODEL_SATURATION_CHECK: &str = "model_saturation_check";
pub const PRIORITY_MODEL: &str = "priority_model";
pub const TEAM_MODEL_LIMIT_AT_SATURATION: &str = "model_per_team_saturated";
pub const PRIORITY_CLASS_SHARE_AT_SATURATION: &str = "priority_class_share_saturated";
pub const TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE: &str = "model_per_team_max_borrow_multiple";
pub const STEERED_FROM_HEADER: &str = "x-litellm-model-steered-from";
pub const DEFAULT_CLASS: &str = "\u{1e}default";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_unit: Option<i64>,
    pub tokens_per_unit: Option<i64>,
    pub window_size: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Descriptor {
    pub key: String,
    pub value: String,
    pub rate_limit: Option<RateLimit>,
}

impl Descriptor {
    pub fn tokens_per_unit(&self) -> Option<i64> {
        self.rate_limit.as_ref()?.tokens_per_unit
    }

    pub fn counter(&self, kind: CounterKind) -> Counter {
        Counter {
            key: self.key.clone(),
            value: self.value.clone(),
            kind,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CounterKind {
    Requests,
    Tokens,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Counter {
    pub key: String,
    pub value: String,
    pub kind: CounterKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Increment {
    pub requests: i64,
    pub tokens: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Window {
    pub counter: Counter,
    pub start: String,
    pub backend: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Status {
    pub code: String,
    pub current_limit: Number,
    pub limit_remaining: Option<Number>,
    pub rate_limit_type: String,
    pub descriptor_key: String,
}

impl Status {
    pub fn over_limit(&self) -> bool {
        self.code == "OVER_LIMIT"
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LimiterResponse {
    pub id: u64,
    pub overall_code: String,
    pub statuses: Vec<Status>,
    #[serde(default)]
    pub reservation_windows: Vec<Window>,
}

impl LimiterResponse {
    pub fn over_limit(&self) -> bool {
        self.overall_code == "OVER_LIMIT"
    }

    pub fn window_of(&self, counter: &Counter) -> Option<Window> {
        self.reservation_windows
            .iter()
            .find(|window| &window.counter == counter)
            .cloned()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ReservationOp {
    Increment {
        counter: Counter,
        amount: i64,
        ttl: Option<i64>,
    },
    Guarded {
        counter: Counter,
        amount: i64,
        ttl: Option<i64>,
        expected_window_start: String,
        backend: String,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GroupInfo {
    pub tpm: Option<i64>,
    pub rpm: Option<i64>,
    pub input_cost_per_token: Option<f64>,
    pub output_cost_per_token: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenRateLimitType {
    Input,
    Output,
    #[default]
    Total,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    pub priority_reservation: Option<Vec<(String, Value)>>,
    pub default_priority: f64,
    pub saturation_threshold: f64,
    pub class_split: bool,
    pub saturation_check_cache_ttl: Value,
    pub window_size: i64,
    pub rate_limit_type: TokenRateLimitType,
}

impl Settings {
    pub fn reserves<'a>(&self, priority: Option<&'a Value>) -> Option<&'a str> {
        let Some(Value::String(priority)) = priority else {
            return None;
        };
        self.priority_reservation
            .as_ref()?
            .iter()
            .any(|(name, _)| name == priority)
            .then_some(priority.as_str())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KeyFacts {
    pub team_id: Option<String>,
    pub team_alias: Option<String>,
    pub priority: Option<Value>,
    pub model_preferences: Option<Value>,
    pub budget_reservation: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BorrowingLimits {
    pub group: String,
    pub tpm: Option<i64>,
    pub rpm: Option<i64>,
    pub max_borrow_multiple: Option<f64>,
}

impl BorrowingLimits {
    pub fn share(&self, rate_limit_type: &str) -> Option<i64> {
        if rate_limit_type == "requests" {
            self.rpm
        } else {
            self.tpm
        }
    }

    pub fn ceiling(&self) -> Option<(Option<i64>, Option<i64>)> {
        let multiple = self.max_borrow_multiple?;
        Some((
            self.tpm.map(|tpm| (tpm as f64 * multiple) as i64),
            self.rpm.map(|rpm| (rpm as f64 * multiple) as i64),
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TeamShare {
    pub fraction: f64,
    pub active_teams_in_class: i64,
    pub active_classes: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageTokens {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
}

impl UsageTokens {
    pub fn settled(usage: Option<&Self>, rate_limit_type: TokenRateLimitType) -> i64 {
        usage.map_or(0, |usage| match rate_limit_type {
            TokenRateLimitType::Output => usage.completion_tokens,
            TokenRateLimitType::Input => usage.prompt_tokens,
            TokenRateLimitType::Total => usage.total_tokens,
        })
    }
}
