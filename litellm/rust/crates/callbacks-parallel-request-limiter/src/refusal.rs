use std::collections::BTreeMap;

use chrono::DateTime;
use litellm_callbacks::{Abort, AbortError};
use serde_json::{Map, Value, json};

use crate::{
    descriptor::Descriptor,
    status::{Code, Response},
};

pub const RATE_LIMITED: &str = "proxy_rate_limit";
pub const UNVERIFIABLE: &str = "rate_limit_unverifiable";

pub fn reset_at(now: f64, window_size: i64) -> String {
    let reset = now + window_size as f64;
    let micros = (reset * 1_000_000.0).round_ties_even() as i64;
    DateTime::from_timestamp_micros(micros)
        .map(|at| at.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_default()
}

pub fn rate_limited(
    response: &Response,
    descriptors: &[Descriptor],
    requested_model: Option<&str>,
    now: f64,
    window_size: i64,
) -> Option<Abort> {
    let status = response
        .statuses
        .iter()
        .find(|status| status.code == Code::OverLimit)?;
    let descriptor_key = status.descriptor_key.as_str();
    let descriptor_value = descriptors
        .iter()
        .find(|descriptor| {
            descriptor.key == descriptor_key
                && status
                    .descriptor_value
                    .as_deref()
                    .is_none_or(|value| descriptor.value == value)
        })
        .map_or("unknown", |descriptor| descriptor.value.as_str());
    let reset = reset_at(now, window_size);
    let kind: &str = status.rate_limit_type.into();
    let detail = format!(
        "Rate limit exceeded for {descriptor_key}: {descriptor_value}. Limit type: {kind}. Current limit: {}, Remaining: {}. Limit resets at: {reset}",
        status.current_limit,
        status.limit_remaining.max(0),
    );
    let headers = BTreeMap::from([
        ("retry-after".to_owned(), window_size.to_string()),
        ("rate_limit_type".to_owned(), kind.to_owned()),
        ("rate_limited_by".to_owned(), descriptor_key.to_owned()),
        ("reset_at".to_owned(), reset),
    ]);
    let header_values: Map<String, Value> = headers
        .iter()
        .map(|(name, value)| (name.clone(), Value::String(value.clone())))
        .collect();
    Some(Abort {
        status: 429,
        message: detail.clone(),
        code: None,
        headers,
        error: Some(AbortError {
            kind: RATE_LIMITED.to_owned(),
            fields: Map::from_iter([
                ("detail".to_owned(), Value::String(detail)),
                ("headers".to_owned(), Value::Object(header_values)),
                ("rate_limit_type".to_owned(), json!(kind)),
                ("descriptor_key".to_owned(), json!(descriptor_key)),
                ("requested_model".to_owned(), json!(requested_model)),
            ]),
        }),
    })
}

pub fn unverifiable() -> Abort {
    Abort {
        error: Some(AbortError {
            kind: UNVERIFIABLE.to_owned(),
            fields: Map::new(),
        }),
        ..Abort::new(503, "rate limit unverifiable")
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::status::{LimitKind, Status};

    #[rstest]
    #[case(1_700_000_000.25, 60, "2023-11-14 22:14:20 UTC")]
    #[case(1_700_000_039.999_999_6, 60, "2023-11-14 22:15:00 UTC")]
    fn the_reset_time_formats_like_strftime_on_utc(
        #[case] now: f64,
        #[case] window: i64,
        #[case] text: &str,
    ) {
        assert_eq!(reset_at(now, window), text);
    }

    #[rstest]
    fn the_refusal_names_the_matching_descriptor_value_and_the_python_detail() {
        let response = Response::over(Status {
            code: Code::OverLimit,
            current_limit: 2,
            limit_remaining: -1,
            rate_limit_type: LimitKind::Requests,
            descriptor_key: "api_key".into(),
            descriptor_value: Some("k2".into()),
        });
        let descriptors: Vec<Descriptor> = serde_json::from_value(serde_json::json!([
            {"key": "api_key", "value": "k1", "rate_limit": null},
            {"key": "api_key", "value": "k2", "rate_limit": null},
        ]))
        .unwrap();
        let abort =
            rate_limited(&response, &descriptors, Some("gpt"), 1_700_000_000.0, 60).unwrap();
        assert_eq!(
            abort.message,
            "Rate limit exceeded for api_key: k2. Limit type: requests. Current limit: 2, Remaining: 0. Limit resets at: 2023-11-14 22:14:20 UTC"
        );
        assert_eq!(abort.headers["retry-after"], "60");
        assert_eq!(abort.headers["rate_limited_by"], "api_key");
    }
}
