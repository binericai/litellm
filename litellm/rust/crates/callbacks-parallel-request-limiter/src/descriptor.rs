use litellm_redis_store::rate_limits::Scope;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

pub const PROJECT_ITPM: &str = "model_per_project_itpm";
pub const PROJECT_OTPM: &str = "model_per_project_otpm";
pub const PTU_TEAM: &str = "model_per_team_ptu";

fn python_int<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<i64>, D::Error> {
    Ok(match Option::<Value>::deserialize(deserializer)? {
        Some(Value::Number(number)) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|float| float.trunc() as i64)),
        Some(Value::Bool(flag)) => Some(i64::from(flag)),
        _ => None,
    })
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Limits {
    #[serde(default, deserialize_with = "python_int")]
    pub requests_per_unit: Option<i64>,
    #[serde(default, deserialize_with = "python_int")]
    pub tokens_per_unit: Option<i64>,
    #[serde(default, deserialize_with = "python_int")]
    pub max_parallel_requests: Option<i64>,
    #[serde(default, deserialize_with = "python_int")]
    pub window_size: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Descriptor {
    pub key: String,
    pub value: String,
    #[serde(default)]
    pub rate_limit: Option<Limits>,
}

impl Descriptor {
    pub fn limits(&self) -> Limits {
        self.rate_limit.clone().unwrap_or_default()
    }

    pub fn scope(&self) -> Scope {
        Scope {
            descriptor: self.key.clone(),
            value: self.value.clone(),
        }
    }

    pub fn window_size(&self, default: i64) -> i64 {
        match self.limits().window_size {
            Some(size) if size != 0 => size,
            _ => default,
        }
    }

    pub fn is_project_io(&self) -> bool {
        self.key == PROJECT_ITPM || self.key == PROJECT_OTPM
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    fn a_descriptor_reads_the_python_typed_dict_with_float_and_missing_limits() {
        let descriptor: Descriptor = serde_json::from_value(json!({
            "key": "api_key", "value": "h",
            "rate_limit": {"requests_per_unit": 5.9, "tokens_per_unit": null, "window_size": 0}
        }))
        .unwrap();
        assert_eq!(descriptor.limits().requests_per_unit, Some(5));
        assert_eq!(descriptor.limits().tokens_per_unit, None);
        assert_eq!(descriptor.window_size(60), 60);
    }

    #[rstest]
    fn a_descriptor_without_limits_has_none() {
        let descriptor: Descriptor =
            serde_json::from_value(json!({"key": "tag", "value": "t", "rate_limit": null}))
                .unwrap();
        assert_eq!(descriptor.limits(), Limits::default());
    }
}
