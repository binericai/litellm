use serde_json::Value;

use crate::{
    error::CapacityError,
    model::{DEFAULT_CLASS, GroupInfo, Settings, TeamShare},
};

const MEMBER_SEPARATOR: char = '\u{1f}';

fn as_float(value: &Value) -> Result<f64, CapacityError> {
    match value {
        Value::Number(number) => number
            .as_f64()
            .ok_or_else(|| CapacityError::Config(format!("{number} is not a float"))),
        Value::Bool(flag) => Ok(f64::from(u8::from(*flag))),
        Value::String(text) => text.trim().parse::<f64>().map_err(|_| {
            CapacityError::Config(format!("could not convert string to float: {text:?}"))
        }),
        other => Err(CapacityError::Config(format!(
            "float() argument must be a string or a real number, not {other}"
        ))),
    }
}

pub fn convert_priority_to_percent(
    value: &Value,
    group: Option<&GroupInfo>,
) -> Result<f64, CapacityError> {
    let Value::Object(entry) = value else {
        return match value {
            Value::Number(_) | Value::Bool(_) => as_float(value),
            _ => Err(CapacityError::Config(format!(
                "priority reservation value {value} is neither a number nor a mapping"
            ))),
        };
    };
    let kind = entry
        .get("type")
        .cloned()
        .unwrap_or_else(|| Value::String("percent".into()));
    let amount = entry.get("value").cloned().unwrap_or(Value::from(1.0));
    let rpm = group.and_then(|group| group.rpm).filter(|rpm| *rpm > 0);
    let tpm = group.and_then(|group| group.tpm).filter(|tpm| *tpm > 0);
    match (kind.as_str(), rpm, tpm) {
        (Some("rpm"), Some(rpm), _) => Ok(as_float(&amount)? / rpm as f64),
        (Some("tpm"), _, Some(tpm)) => Ok(as_float(&amount)? / tpm as f64),
        _ => as_float(&amount),
    }
}

pub fn normalized_weights(
    settings: &Settings,
    group: &GroupInfo,
) -> Result<Vec<(String, f64)>, CapacityError> {
    let Some(reservation) = &settings.priority_reservation else {
        return Ok(Vec::new());
    };
    let weights = reservation
        .iter()
        .map(|(name, value)| {
            Ok((
                name.clone(),
                convert_priority_to_percent(value, Some(group))?,
            ))
        })
        .collect::<Result<Vec<_>, CapacityError>>()?;
    let total: f64 = weights.iter().map(|(_, weight)| weight).sum();
    if total > 1.0 {
        return Ok(weights
            .into_iter()
            .map(|(name, weight)| (name, weight / total))
            .collect());
    }
    Ok(weights)
}

pub fn weight_of(weights: &[(String, f64)], name: &str) -> Option<f64> {
    weights
        .iter()
        .find(|(candidate, _)| candidate == name)
        .map(|(_, weight)| *weight)
}

pub struct Allocation {
    pub weight: f64,
    pub key: String,
}

pub fn allocation(
    settings: &Settings,
    model: &str,
    priority: Option<&Value>,
    weights: &[(String, f64)],
    group: &GroupInfo,
) -> Result<Allocation, CapacityError> {
    match settings.reserves(priority) {
        Some(explicit) => {
            let fallback = explicit_weight(settings, explicit, group)?;
            Ok(Allocation {
                weight: weight_of(weights, explicit).unwrap_or(fallback),
                key: format!("{model}:{explicit}"),
            })
        }
        None => Ok(Allocation {
            weight: settings.default_priority,
            key: format!("{model}:default_pool"),
        }),
    }
}

fn explicit_weight(
    settings: &Settings,
    name: &str,
    group: &GroupInfo,
) -> Result<f64, CapacityError> {
    let value = settings
        .priority_reservation
        .iter()
        .flatten()
        .find(|(candidate, _)| candidate == name)
        .map(|(_, value)| value);
    match value {
        Some(value) => convert_priority_to_percent(value, Some(group)),
        None => Ok(settings.default_priority),
    }
}

pub fn priority_class(settings: &Settings, priority: Option<&Value>) -> String {
    settings
        .reserves(priority)
        .map_or_else(|| DEFAULT_CLASS.to_owned(), str::to_owned)
}

pub fn pool_member(priority_class: &str) -> String {
    format!("{MEMBER_SEPARATOR}pool{MEMBER_SEPARATOR}{priority_class}")
}

pub fn team_member(team_id: &str) -> String {
    format!("team{MEMBER_SEPARATOR}{team_id}")
}

fn is_pool_member(member: &str) -> bool {
    member.starts_with(&format!("{MEMBER_SEPARATOR}pool{MEMBER_SEPARATOR}"))
}

pub fn counter_value(model: &str, team_id: &str) -> String {
    format!("{}:{model}:{team_id}", model.chars().count())
}

pub fn team_share(
    priority: &str,
    members: &[(String, String)],
    class_weights: &[(String, f64)],
    default_weight: f64,
) -> TeamShare {
    let mut present: Vec<&str> = Vec::new();
    for (_, class) in members {
        if !present.contains(&class.as_str()) {
            present.push(class);
        }
    }
    let weights: Vec<(&str, f64)> = present
        .iter()
        .map(|class| {
            (
                *class,
                weight_of(class_weights, class).unwrap_or(default_weight),
            )
        })
        .collect();
    let total: f64 = weights.iter().map(|(_, weight)| weight).sum();
    let own = weights
        .iter()
        .find(|(class, _)| *class == priority)
        .map_or(0.0, |(_, weight)| *weight);
    let class_fraction = if total > 0.0 {
        own / total
    } else {
        1.0 / present.len() as f64
    };
    let teams_in_class = members
        .iter()
        .filter(|(member, class)| class == priority && !is_pool_member(member))
        .count() as i64;
    TeamShare {
        fraction: class_fraction / teams_in_class.max(1) as f64,
        active_teams_in_class: teams_in_class,
        active_classes: present.len() as i64,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::model::TokenRateLimitType;

    fn settings(reservation: &[(&str, Value)]) -> Settings {
        Settings {
            priority_reservation: Some(
                reservation
                    .iter()
                    .map(|(name, value)| ((*name).to_owned(), value.clone()))
                    .collect(),
            ),
            default_priority: 0.25,
            saturation_threshold: 0.8,
            class_split: false,
            saturation_check_cache_ttl: json!(60),
            window_size: 60,
            rate_limit_type: TokenRateLimitType::Total,
        }
    }

    fn group(tpm: Option<i64>, rpm: Option<i64>) -> GroupInfo {
        GroupInfo {
            tpm,
            rpm,
            ..GroupInfo::default()
        }
    }

    #[rstest]
    #[case(json!(0.9), 0.9)]
    #[case(json!({"type": "percent", "value": 0.4}), 0.4)]
    #[case(json!({"type": "rpm", "value": 25}), 0.25)]
    #[case(json!({"type": "tpm", "value": 500}), 0.5)]
    #[case(json!({"value": 0.3}), 0.3)]
    #[case(json!({"type": "rpm"}), 0.01)]
    fn reservations_convert_to_a_fraction_of_the_group(
        #[case] value: Value,
        #[case] expected: f64,
    ) {
        let converted =
            convert_priority_to_percent(&value, Some(&group(Some(1000), Some(100)))).unwrap();
        assert!(
            (converted - expected).abs() < 1e-12,
            "{converted} != {expected}"
        );
    }

    #[rstest]
    fn an_absolute_reservation_without_that_limit_on_the_group_reads_as_a_percent() {
        let converted = convert_priority_to_percent(
            &json!({"type": "rpm", "value": 0.5}),
            Some(&group(Some(1000), None)),
        )
        .unwrap();
        assert_eq!(converted, 0.5);
    }

    #[rstest]
    fn over_allocated_reservations_scale_down_to_the_whole_group() {
        let weights = normalized_weights(
            &settings(&[("a", json!(0.6)), ("b", json!(0.8))]),
            &group(Some(1000), None),
        )
        .unwrap();
        assert!((weight_of(&weights, "a").unwrap() - 0.6 / 1.4).abs() < 1e-12);
        assert!((weight_of(&weights, "b").unwrap() - 0.8 / 1.4).abs() < 1e-12);
    }

    #[rstest]
    fn reservations_within_the_group_keep_their_weights() {
        let weights = normalized_weights(
            &settings(&[("a", json!(0.6)), ("b", json!(0.4))]),
            &group(Some(1000), None),
        )
        .unwrap();
        assert_eq!(weights, vec![("a".into(), 0.6), ("b".into(), 0.4)]);
    }

    #[rstest]
    fn keys_without_a_reserved_priority_share_the_default_pool() {
        let settings = settings(&[("high", json!(0.9))]);
        let weights = normalized_weights(&settings, &group(Some(1000), None)).unwrap();
        let unknown = allocation(
            &settings,
            "m",
            Some(&json!("gold")),
            &weights,
            &group(Some(1000), None),
        )
        .unwrap();
        assert_eq!(
            (unknown.weight, unknown.key.as_str()),
            (0.25, "m:default_pool")
        );
        let high = allocation(
            &settings,
            "m",
            Some(&json!("high")),
            &weights,
            &group(Some(1000), None),
        )
        .unwrap();
        assert_eq!((high.weight, high.key.as_str()), (0.9, "m:high"));
    }

    #[rstest]
    fn a_class_share_splits_among_its_active_teams_and_absent_classes_give_theirs_away() {
        let members = vec![
            ("team\u{1f}a".to_owned(), "high".to_owned()),
            ("team\u{1f}b".to_owned(), "high".to_owned()),
            ("team\u{1f}c".to_owned(), "low".to_owned()),
        ];
        let weights = vec![
            ("high".to_owned(), 0.6),
            ("low".to_owned(), 0.2),
            ("idle".to_owned(), 0.2),
        ];
        let share = team_share("high", &members, &weights, 0.1);
        assert!((share.fraction - 0.375).abs() < 1e-12);
        assert_eq!((share.active_teams_in_class, share.active_classes), (2, 2));
    }

    #[rstest]
    fn a_pool_member_keeps_its_class_present_without_taking_a_slice() {
        let members = vec![
            ("team\u{1f}a".to_owned(), "high".to_owned()),
            (pool_member("high"), "high".to_owned()),
        ];
        let share = team_share("high", &members, &[("high".to_owned(), 0.5)], 0.1);
        assert_eq!((share.fraction, share.active_teams_in_class), (1.0, 1));
    }

    #[rstest]
    #[case("chat", "prod:victim", "4:chat:prod:victim")]
    #[case("chat:prod", "victim", "9:chat:prod:victim")]
    #[case("é", "t", "1:é:t")]
    fn counter_values_cannot_collide_across_the_model_and_team_boundary(
        #[case] model: &str,
        #[case] team: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(counter_value(model, team), expected);
    }
}
