use litellm_redis_store::spend_counters::{NotASpendCounter, SpendCounter};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlannedCounter {
    pub counter_key: String,
    pub max_budget: f64,
    pub fallback_spend: f64,
    pub entity_type: String,
    pub entity_id: String,
    #[serde(default)]
    pub spend_log_entity_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub counters: Vec<PlannedCounter>,
    pub reservation_cost: f64,
    pub input_cost: f64,
    #[serde(default)]
    pub input_tokens: Option<i64>,
    #[serde(default)]
    pub throttle_counter: Option<String>,
    #[serde(default)]
    pub fail_closed: bool,
    #[serde(default)]
    pub lease_seconds: Option<f64>,
    pub status_code: u16,
}

#[derive(Clone, Debug)]
pub(crate) struct Counter {
    pub planned: PlannedCounter,
    pub counter: SpendCounter,
}

impl Counter {
    pub fn key(&self) -> &str {
        &self.planned.counter_key
    }
}

impl Plan {
    pub(crate) fn counters(&self) -> Result<Vec<Counter>, NotASpendCounter> {
        self.counters
            .iter()
            .map(|planned| {
                Ok(Counter {
                    counter: SpendCounter::parse(&planned.counter_key)?,
                    planned: planned.clone(),
                })
            })
            .collect()
    }
}

pub(crate) fn entity_type_value(entity_type: &str) -> Option<&'static str> {
    match entity_type {
        "Key" => Some("key"),
        "Team" => Some("team"),
        "TeamMember" => Some("team_member"),
        "User" => Some("user"),
        "EndUser" => Some("end_user"),
        "Tag" => Some("tag"),
        "Model access group" => Some("model_access_group"),
        "Organization" => Some("organization"),
        "Project" => Some("project"),
        _ => None,
    }
}
