use std::sync::Arc;

use litellm_callbacks::{
    CallFacts, CallIds, Callback, Field, FieldSet, FieldValues, Hook, HookInput, Verdict,
};
use litellm_callbacks_budgets::{BudgetState, Budgets, Plan, PlannedCounter, Shared};
use litellm_redis_store::{
    Number,
    spend_counters::{SpendCounter, SpendCounters},
};
use rstest::rstest;
use serde_json::json;

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

fn pre_call(plan: &Plan, state: &Arc<BudgetState>) -> HookInput {
    HookInput {
        hook: Hook::PreCall,
        fields: FieldValues::new(FieldSet::of(&[Field::DeprecatedKwargs])).with(
            Field::DeprecatedKwargs,
            json!({"plan": plan, "call_id": "call-1", "model": "gpt"}),
        ),
        call: None,
        state: Some(Arc::clone(state) as _),
    }
}

#[rstest]
#[tokio::test]
async fn the_callback_holds_at_pre_call_and_settles_at_attempt_ended() {
    let store = store(&[(KEY, 1.0)]);
    let budgets = Arc::new(Budgets::new(Arc::clone(&store)));
    let callback = Shared(Arc::clone(&budgets));
    let state = Arc::new(BudgetState::default());
    let plan = plan(vec![counter(KEY, 10.0, "Key", "hashed")], 2.0);
    assert_eq!(
        callback.handle(&pre_call(&plan, &state)).await.unwrap(),
        None
    );
    assert_eq!(value(&store, KEY), Some(3.0));
    assert!(budgets.reservation("call-1").await.is_some());
    let ended = HookInput {
        hook: Hook::AttemptEnded,
        fields: FieldValues::new(FieldSet::of(&[Field::StandardLoggingPayload])).with(
            Field::StandardLoggingPayload,
            json!({"status": "success", "response_cost": 0.5}),
        ),
        call: Some(CallFacts {
            ids: CallIds {
                call_id: "call-1".into(),
                ..CallIds::default()
            },
            ..CallFacts::default()
        }),
        state: Some(state as _),
    };
    assert_eq!(callback.handle(&ended).await.unwrap(), None);
    assert_eq!(value(&store, KEY), Some(1.5));
    assert!(budgets.reservation("call-1").await.is_none());
}

#[rstest]
#[tokio::test]
async fn a_forwarded_settlement_wins_over_the_attempt_ended_cost() {
    let store = store(&[(KEY, 1.0)]);
    let budgets = Arc::new(Budgets::new(Arc::clone(&store)));
    let callback = Shared(Arc::clone(&budgets));
    let state = Arc::new(BudgetState::default());
    let plan = plan(vec![counter(KEY, 10.0, "Key", "hashed")], 2.0);
    callback.handle(&pre_call(&plan, &state)).await.unwrap();
    assert!(budgets.settle("call-1", 0.25, false).await.unwrap());
    let ended = HookInput {
        hook: Hook::AttemptEnded,
        fields: FieldValues::new(FieldSet::of(&[Field::StandardLoggingPayload])).with(
            Field::StandardLoggingPayload,
            json!({"status": "success", "response_cost": 0.5}),
        ),
        call: None,
        state: Some(state as _),
    };
    callback.handle(&ended).await.unwrap();
    assert_eq!(value(&store, KEY), Some(1.25));
}

#[rstest]
#[tokio::test]
async fn a_refusal_aborts_with_the_typed_budget_error() {
    let store = store(&[(KEY, 10.0)]);
    let callback = Shared(Arc::new(Budgets::new(Arc::clone(&store))));
    let state = Arc::new(BudgetState::default());
    let plan = plan(vec![counter(KEY, 10.0, "Key", "hashed")], 2.0);
    let Some(Verdict::Abort(abort)) = callback.handle(&pre_call(&plan, &state)).await.unwrap()
    else {
        panic!("expected an abort");
    };
    assert_eq!(abort.status, 400);
    let error = abort.error.unwrap();
    assert_eq!(error.kind, "budget_exceeded");
    assert_eq!(
        serde_json::Value::Object(error.fields),
        json!({
            "current_cost": 12.0,
            "max_budget": 10.0,
            "message": "Budget has been exceeded! Key=hashed Current cost: 12.0, Max budget: 10.0",
            "entity_type": "key",
            "entity_id": "hashed",
            "model": "gpt",
        })
    );
    assert_eq!(value(&store, KEY), Some(10.0));
}
