use litellm_redis_store::rate_limits::Scope;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

pub const PROJECT_ITPM: &str = "model_per_project_itpm";
pub const PROJECT_OTPM: &str = "model_per_project_otpm";
pub const PTU_TEAM: &str = "model_per_team_ptu";

fn int_text(text: &str) -> Option<i64> {
    let trimmed = text.trim();
    let digits = trimmed.strip_prefix(['+', '-']).unwrap_or(trimmed);
    let valid = !digits.is_empty()
        && !digits.starts_with('_')
        && !digits.ends_with('_')
        && !digits.contains("__")
        && digits.chars().all(|c| c.is_ascii_digit() || c == '_');
    if !valid {
        return None;
    }
    trimmed.replace('_', "").parse().ok()
}

fn int_value<E: serde::de::Error>(value: Value) -> Result<i64, E> {
    match value {
        Value::Number(number) => Ok(number
            .as_i64()
            .unwrap_or_else(|| number.as_f64().map_or(0, |float| float.trunc() as i64))),
        Value::Bool(flag) => Ok(i64::from(flag)),
        Value::String(text) => int_text(&text)
            .ok_or_else(|| E::custom(format!("invalid literal for int() with base 10: {text:?}"))),
        other => Err(E::custom(format!(
            "int() argument must be a string or a number, not {other}"
        ))),
    }
}

fn python_int<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<i64>, D::Error> {
    Option::<Value>::deserialize(deserializer)?
        .filter(|value| !value.is_null())
        .map(int_value)
        .transpose()
}

fn truthy_int<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<i64>, D::Error> {
    let value = Option::<Value>::deserialize(deserializer)?;
    let truthy = match &value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(number)) => number.as_f64().is_some_and(|float| float != 0.0),
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(fields)) => !fields.is_empty(),
    };
    value.filter(|_| truthy).map(int_value).transpose()
}

fn python_str<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    Ok(match Value::deserialize(deserializer)? {
        Value::String(text) => text,
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(number) => match number.as_f64() {
            Some(float)
                if !number.is_i64()
                    && !number.is_u64()
                    && float.fract() == 0.0
                    && float.abs() < 1e16 =>
            {
                format!("{float:.1}")
            }
            _ => number.to_string(),
        },
        other => {
            return Err(serde::de::Error::custom(format!(
                "a descriptor value must be a scalar, not {other}"
            )));
        }
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
    #[serde(default, deserialize_with = "truthy_int")]
    pub window_size: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Descriptor {
    pub key: String,
    #[serde(deserialize_with = "python_str")]
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
        self.limits().window_size.unwrap_or(default)
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

    #[rstest]
    #[case(json!("10"), Some(10))]
    #[case(json!(" -1_000 "), Some(-1000))]
    #[case(json!(true), Some(1))]
    #[case(json!(7.9), Some(7))]
    #[case(json!(null), None)]
    fn a_limit_converts_like_python_int(#[case] limit: Value, #[case] expected: Option<i64>) {
        let descriptor: Descriptor = serde_json::from_value(json!({
            "key": "team", "value": "t", "rate_limit": {"requests_per_unit": limit}
        }))
        .unwrap();
        assert_eq!(descriptor.limits().requests_per_unit, expected);
    }

    #[rstest]
    #[case(json!("1.5"))]
    #[case(json!(""))]
    #[case(json!("1__0"))]
    #[case(json!([1]))]
    fn a_limit_python_int_rejects_is_an_error(#[case] limit: Value) {
        let decoded = serde_json::from_value::<Descriptor>(json!({
            "key": "team", "value": "t", "rate_limit": {"tokens_per_unit": limit}
        }));
        assert!(decoded.is_err());
    }

    #[rstest]
    #[case(json!(0), 60)]
    #[case(json!(0.5), 0)]
    #[case(json!("0"), 0)]
    #[case(json!(""), 60)]
    #[case(json!("30"), 30)]
    fn a_window_size_falls_back_only_when_python_reads_it_as_false(
        #[case] window: Value,
        #[case] expected: i64,
    ) {
        let descriptor: Descriptor = serde_json::from_value(json!({
            "key": "team", "value": "t", "rate_limit": {"window_size": window}
        }))
        .unwrap();
        assert_eq!(descriptor.window_size(60), expected);
    }

    #[rstest]
    #[case(json!(123), "123")]
    #[case(json!(1.0), "1.0")]
    #[case(json!(2.5), "2.5")]
    #[case(json!(false), "False")]
    #[case(json!(null), "None")]
    fn a_descriptor_value_is_formatted_like_python_str(
        #[case] value: Value,
        #[case] expected: &str,
    ) {
        let descriptor: Descriptor =
            serde_json::from_value(json!({"key": "team", "value": value})).unwrap();
        assert_eq!(descriptor.value, expected);
    }
}
