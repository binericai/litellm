use std::sync::atomic::{AtomicUsize, Ordering};

use litellm_python_compat::repr::float_repr;
use litellm_redis_store::{
    Number,
    spend_counters::{NotASpendCounter, SpendCounter, SpendCounters},
};
use serde::Serialize;

use crate::plan::{Counter, Plan, entity_type_value};

const EPSILON: f64 = 1e-12;

static ENTRY_IDS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Entry {
    pub counter_key: String,
    pub entity_type: String,
    pub entity_id: String,
    pub reserved_cost: f64,
    pub applied_adjustment: f64,
    #[serde(skip)]
    id: usize,
    #[serde(skip)]
    counter: SpendCounter,
}

impl Entry {
    pub fn counter(&self) -> &SpendCounter {
        &self.counter
    }

    fn new(counter: &Counter, reserved_cost: f64) -> Self {
        Self {
            counter_key: counter.planned.counter_key.clone(),
            entity_type: counter.planned.entity_type.clone(),
            entity_id: counter.planned.entity_id.clone(),
            reserved_cost,
            applied_adjustment: 0.0,
            id: ENTRY_IDS.fetch_add(1, Ordering::Relaxed),
            counter: counter.counter.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Reservation {
    pub reserved_cost: f64,
    pub entries: Vec<Entry>,
    pub finalized: bool,
    pub input_cost: f64,
    pub input_tokens: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Exceeded {
    pub current_cost: f64,
    pub max_budget: f64,
    pub message: String,
    pub entity_type: Option<&'static str>,
    pub entity_id: String,
}

#[derive(Debug)]
pub enum EngineError {
    Exceeded(Exceeded),
    Unavailable,
    Inconsistent(String),
    Store(litellm_redis_store::Error),
    Plan(NotASpendCounter),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exceeded(exceeded) => f.write_str(&exceeded.message),
            Self::Unavailable => f.write_str(UNAVAILABLE),
            Self::Inconsistent(message) => f.write_str(message),
            Self::Store(error) => write!(f, "{error}"),
            Self::Plan(error) => write!(f, "{error}"),
        }
    }
}

impl From<litellm_redis_store::Error> for EngineError {
    fn from(error: litellm_redis_store::Error) -> Self {
        Self::Store(error)
    }
}

pub const UNAVAILABLE: &str = "Budget enforcement unavailable: the budget reservation could not be written to the spend counter backend, and fail_closed_budget_enforcement is enabled, so the request was rejected to avoid exceeding the configured budget. Retry shortly.";

fn exceeded(counter: &Counter, current_cost: f64, estimated_cost: Option<f64>) -> EngineError {
    let planned = &counter.planned;
    let estimate = estimated_cost
        .map(|cost| format!("Estimated request cost: {}, ", float_repr(cost)))
        .unwrap_or_default();
    EngineError::Exceeded(Exceeded {
        current_cost,
        max_budget: planned.max_budget,
        message: format!(
            "Budget has been exceeded! {}={} Current cost: {}, {}Max budget: {}",
            planned.entity_type,
            planned.entity_id,
            float_repr(current_cost),
            estimate,
            float_repr(planned.max_budget),
        ),
        entity_type: entity_type_value(&planned.entity_type),
        entity_id: planned
            .spend_log_entity_id
            .clone()
            .unwrap_or_else(|| planned.entity_id.clone()),
    })
}

async fn current_value(store: &SpendCounters, counter: &Counter) -> Result<f64, EngineError> {
    Ok(store
        .read(&counter.counter)
        .await?
        .value()
        .map_or(counter.planned.fallback_spend, Number::as_f64))
}

async fn can_apply(
    store: &SpendCounters,
    counter: &SpendCounter,
    adjustment: f64,
) -> Result<bool, EngineError> {
    Ok(match store.read(counter).await?.value() {
        None => false,
        Some(value) => !(adjustment < 0.0 && value.as_f64() + adjustment < -EPSILON),
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Inconsistent {
    Drop,
    Refuse,
}

pub(crate) async fn apply_actual(
    store: &SpendCounters,
    entries: &mut [Entry],
    actual: f64,
    inconsistent: Inconsistent,
) -> Result<(), EngineError> {
    let mut adjustments = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        let target = actual - entry.reserved_cost;
        let adjustment = target - entry.applied_adjustment;
        if adjustment != 0.0 {
            adjustments.push((index, target, adjustment));
        }
    }
    let mut consistent = Vec::with_capacity(adjustments.len());
    for (index, _, adjustment) in &adjustments {
        consistent.push(can_apply(store, &entries[*index].counter, *adjustment).await?);
    }
    if inconsistent == Inconsistent::Refuse
        && let Some(((index, _, _), _)) = adjustments
            .iter()
            .zip(&consistent)
            .find(|(_, consistent)| !**consistent)
    {
        return Err(EngineError::Inconsistent(format!(
            "Cannot resize budget reservation against inconsistent counter {}",
            entries[*index].counter_key
        )));
    }
    let applicable: Vec<(SpendCounter, Number)> = adjustments
        .iter()
        .zip(&consistent)
        .filter(|(_, consistent)| **consistent)
        .map(|((index, _, adjustment), _)| {
            (entries[*index].counter.clone(), Number::Float(*adjustment))
        })
        .collect();
    if !applicable.is_empty()
        && let Err(error) = store.increment_many(&applicable).await
    {
        let counters: Vec<SpendCounter> = applicable.into_iter().map(|(c, _)| c).collect();
        store.invalidate(&counters).await;
        return Err(error.into());
    }
    for ((index, _, _), _) in adjustments
        .iter()
        .zip(&consistent)
        .filter(|(_, consistent)| !**consistent)
    {
        store
            .invalidate(std::slice::from_ref(&entries[*index].counter))
            .await;
    }
    for (index, target, _) in adjustments {
        entries[index].applied_adjustment = target;
    }
    Ok(())
}

pub(crate) async fn release_best_effort(store: &SpendCounters, entries: &mut [Entry]) {
    for entry in entries {
        if apply_actual(store, std::slice::from_mut(entry), 0.0, Inconsistent::Drop)
            .await
            .is_err()
        {
            tracing::error!(
                counter = entry.counter_key,
                "Failed to release partial budget reservation during exception cleanup"
            );
            store.invalidate(std::slice::from_ref(&entry.counter)).await;
        }
    }
}

struct Admission<'a> {
    store: &'a SpendCounters,
    plan: &'a Plan,
    applied: Vec<Entry>,
    cost: f64,
}

impl Admission<'_> {
    async fn charge(&mut self, counters: &[Counter]) -> Result<(), EngineError> {
        let mut values = Vec::with_capacity(counters.len());
        for counter in counters {
            values.push(current_value(self.store, counter).await?);
        }
        let fits = counters
            .iter()
            .zip(&values)
            .all(|(counter, value)| value + self.cost <= counter.planned.max_budget);
        let groups: Vec<&[Counter]> = if fits {
            vec![counters]
        } else {
            counters.chunks(1).collect()
        };
        for group in groups {
            self.charge_group(group).await?;
        }
        Ok(())
    }

    async fn charge_group(&mut self, group: &[Counter]) -> Result<(), EngineError> {
        let charged = self.cost;
        let entries: Vec<Entry> = group.iter().map(|c| Entry::new(c, charged)).collect();
        let ids: Vec<usize> = entries.iter().map(|entry| entry.id).collect();
        self.applied.extend(entries);
        let increments: Vec<(SpendCounter, Number)> = group
            .iter()
            .map(|counter| (counter.counter.clone(), Number::Float(charged)))
            .collect();
        let totals = match self.store.increment_many(&increments).await {
            Ok(totals) => totals,
            Err(error) => {
                tracing::warn!(%error, "Skipping budget reservation because spend counter reservation failed");
                let counters: Vec<SpendCounter> = increments.into_iter().map(|(c, _)| c).collect();
                self.store.invalidate(&counters).await;
                self.applied.retain(|entry| !ids.contains(&entry.id));
                if self.plan.fail_closed {
                    return Err(EngineError::Unavailable);
                }
                return Ok(());
            }
        };
        for ((counter, id), total) in group.iter().zip(ids).zip(totals) {
            if !self.applied.iter().any(|entry| entry.id == id) {
                continue;
            }
            let current = total.as_f64() - (charged - self.cost);
            if current > counter.planned.max_budget {
                self.over_budget(counter, id, current).await?;
            }
        }
        Ok(())
    }

    async fn over_budget(
        &mut self,
        counter: &Counter,
        id: usize,
        current: f64,
    ) -> Result<(), EngineError> {
        if self.plan.throttle_counter.as_deref() == Some(counter.key()) {
            if let Some(position) = self.applied.iter().position(|entry| entry.id == id) {
                let mut entry = self.applied.remove(position);
                release_best_effort(self.store, std::slice::from_mut(&mut entry)).await;
            }
            return Ok(());
        }
        let remaining = counter.planned.max_budget - (current - self.cost);
        if remaining <= EPSILON {
            return Err(exceeded(counter, current, None));
        }
        if self.plan.fail_closed && current - counter.planned.max_budget > EPSILON {
            return Err(exceeded(counter, current - self.cost, Some(self.cost)));
        }
        apply_actual(
            self.store,
            &mut self.applied,
            remaining,
            Inconsistent::Refuse,
        )
        .await?;
        for entry in &mut self.applied {
            entry.reserved_cost = remaining;
            entry.applied_adjustment = 0.0;
        }
        self.cost = remaining;
        Ok(())
    }
}

pub async fn reserve(
    store: &SpendCounters,
    plan: &Plan,
) -> Result<Option<Reservation>, EngineError> {
    let counters = plan.counters().map_err(EngineError::Plan)?;
    let mut admission = Admission {
        store,
        plan,
        applied: Vec::new(),
        cost: plan.reservation_cost,
    };
    if let Err(error) = admission.charge(&counters).await {
        release_best_effort(store, &mut admission.applied).await;
        return Err(error);
    }
    if admission.applied.is_empty() {
        return Ok(None);
    }
    Ok(Some(Reservation {
        reserved_cost: admission.cost,
        input_cost: plan.input_cost.min(admission.cost),
        input_tokens: plan.input_tokens,
        entries: admission.applied,
        finalized: false,
    }))
}

pub async fn settle(
    store: &SpendCounters,
    reservation: &mut Reservation,
    actual: f64,
    finalize: bool,
) -> Result<(), EngineError> {
    if reservation.finalized {
        return Ok(());
    }
    apply_actual(store, &mut reservation.entries, actual, Inconsistent::Drop).await?;
    if finalize {
        reservation.finalized = true;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use litellm_redis_store::{
        Number,
        spend_counters::{SpendCounter, SpendCounters},
    };
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::plan::{Plan, PlannedCounter};

    fn counter(key: &str, max_budget: f64, entity_type: &str, entity_id: &str) -> PlannedCounter {
        PlannedCounter {
            counter_key: key.to_owned(),
            max_budget,
            fallback_spend: 0.0,
            entity_type: entity_type.to_owned(),
            entity_id: entity_id.to_owned(),
            spend_log_entity_id: None,
        }
    }

    fn plan(counters: Vec<PlannedCounter>, cost: f64) -> Plan {
        Plan {
            counters,
            reservation_cost: cost,
            input_cost: cost / 4.0,
            input_tokens: Some(12),
            throttle_counter: None,
            fail_closed: false,
            lease_seconds: Some(60.0),
            status_code: 400,
        }
    }

    fn store(seed: &[(&str, f64)]) -> Arc<SpendCounters> {
        let store = Arc::new(SpendCounters::new(None));
        for (key, value) in seed {
            store
                .set_local(
                    &SpendCounter::parse(key).unwrap(),
                    Number::Float(*value),
                    None,
                )
                .unwrap();
        }
        store
    }

    fn value(store: &SpendCounters, key: &str) -> Option<f64> {
        store
            .get_local(&SpendCounter::parse(key).unwrap())
            .unwrap()
            .map(Number::as_f64)
    }

    const KEY: &str = "spend:key:hashed";
    const TEAM: &str = "spend:team:t1";

    #[rstest]
    #[tokio::test]
    async fn counters_that_fit_are_charged_the_whole_estimate() {
        let store = store(&[(KEY, 1.0), (TEAM, 2.0)]);
        let plan = plan(
            vec![
                counter(KEY, 10.0, "Key", "hashed"),
                counter(TEAM, 10.0, "Team", "t1"),
            ],
            3.0,
        );
        let reservation = reserve(&store, &plan).await.unwrap().unwrap();
        assert_eq!(reservation.reserved_cost, 3.0);
        assert_eq!(reservation.input_cost, 0.75);
        assert_eq!(value(&store, KEY), Some(4.0));
        assert_eq!(value(&store, TEAM), Some(5.0));
        assert_eq!(
            serde_json::to_value(&reservation).unwrap()["entries"][1],
            json!({"counter_key": TEAM, "entity_type": "Team", "entity_id": "t1", "reserved_cost": 3.0, "applied_adjustment": 0.0})
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_partial_fit_shrinks_every_applied_entry_to_what_is_left() {
        let store = store(&[(KEY, 1.0), (TEAM, 9.0)]);
        let plan = plan(
            vec![
                counter(KEY, 10.0, "Key", "hashed"),
                counter(TEAM, 10.0, "Team", "t1"),
            ],
            3.0,
        );
        let reservation = reserve(&store, &plan).await.unwrap().unwrap();
        assert_eq!(reservation.reserved_cost, 1.0);
        assert_eq!(value(&store, KEY), Some(2.0));
        assert_eq!(value(&store, TEAM), Some(10.0));
    }

    #[rstest]
    #[case::spent(
        10.0,
        false,
        "Budget has been exceeded! Team=t1 Current cost: 11.0, Max budget: 10.0",
        11.0
    )]
    #[case::fail_closed(
        9.5,
        true,
        "Budget has been exceeded! Team=t1 Current cost: 9.5, Estimated request cost: 1.0, Max budget: 10.0",
        9.5
    )]
    #[rstest]
    #[tokio::test]
    async fn an_over_budget_counter_refuses_and_releases_what_was_charged(
        #[case] team_spend: f64,
        #[case] fail_closed: bool,
        #[case] message: &str,
        #[case] current_cost: f64,
    ) {
        let store = store(&[(KEY, 1.0), (TEAM, team_spend)]);
        let plan = Plan {
            fail_closed,
            ..plan(
                vec![
                    counter(KEY, 10.0, "Key", "hashed"),
                    counter(TEAM, 10.0, "Team", "t1"),
                ],
                1.0,
            )
        };
        let Err(EngineError::Exceeded(exceeded)) = reserve(&store, &plan).await else {
            panic!("expected a refusal");
        };
        assert_eq!(exceeded.message, message);
        assert_eq!(exceeded.current_cost, current_cost);
        assert_eq!(exceeded.entity_type, Some("team"));
        assert_eq!(value(&store, KEY), Some(1.0));
        assert_eq!(value(&store, TEAM), Some(team_spend));
    }

    #[rstest]
    #[tokio::test]
    async fn a_throttled_key_releases_its_own_entry_and_keeps_the_rest() {
        let store = store(&[(KEY, 10.0), (TEAM, 0.0)]);
        let plan = Plan {
            throttle_counter: Some(KEY.to_owned()),
            ..plan(
                vec![
                    counter(KEY, 10.0, "Key", "hashed"),
                    counter(TEAM, 10.0, "Team", "t1"),
                ],
                1.0,
            )
        };
        let reservation = reserve(&store, &plan).await.unwrap().unwrap();
        assert_eq!(reservation.entries.len(), 1);
        assert_eq!(reservation.entries[0].counter_key, TEAM);
        assert_eq!(value(&store, KEY), Some(10.0));
        assert_eq!(value(&store, TEAM), Some(1.0));
    }

    #[rstest]
    #[case::success(0.4, 1.4)]
    #[case::refund(0.0, 1.0)]
    #[rstest]
    #[tokio::test]
    async fn settling_moves_each_counter_from_the_hold_to_the_actual_cost(
        #[case] actual: f64,
        #[case] after: f64,
    ) {
        let store = store(&[(KEY, 1.0)]);
        let plan = plan(vec![counter(KEY, 10.0, "Key", "hashed")], 2.0);
        let mut reservation = reserve(&store, &plan).await.unwrap().unwrap();
        settle(&store, &mut reservation, actual, true)
            .await
            .unwrap();
        settle(&store, &mut reservation, 5.0, true).await.unwrap();
        assert_eq!(value(&store, KEY), Some(after));
    }
}
