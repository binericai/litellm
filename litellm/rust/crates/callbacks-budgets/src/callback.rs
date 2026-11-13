use std::{
    collections::HashMap,
    sync::{Arc, Mutex, PoisonError},
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Abort, AbortError, Activation, Callback, CallbackError, Field, FieldSet, Hook, HookInput,
    HookOutput, Manifest, OptOutBy, RequestState, Selection, Subscription, Verdict,
};
use litellm_redis_store::spend_counters::SpendCounters;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::{
    engine::{self, EngineError, Reservation, UNAVAILABLE},
    plan::Plan,
};

pub const NAME: &str = "budgets";

pub struct Held {
    call_id: String,
    reservation: tokio::sync::Mutex<Reservation>,
}

impl Held {
    pub async fn snapshot(&self) -> Reservation {
        self.reservation.lock().await.clone()
    }
}

#[derive(Default)]
pub struct BudgetState(Mutex<Option<Arc<Held>>>);

impl BudgetState {
    fn set(&self, held: Arc<Held>) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(held);
    }
}

#[derive(Deserialize)]
struct Admission {
    plan: Option<Plan>,
    call_id: Option<String>,
    model: Option<String>,
}

pub struct Budgets {
    store: Arc<SpendCounters>,
    manifest: Manifest,
    held: Mutex<HashMap<String, Arc<Held>>>,
}

fn manifest() -> Manifest {
    Manifest {
        selection: Selection {
            activation: Activation::Required,
            opt_out_by: vec![OptOutBy::Admin],
            ..Selection::named(NAME)
        },
        subscriptions: vec![Subscription {
            reads: FieldSet::of(&[Field::DeprecatedKwargs]),
            may_abort: true,
            ..Subscription::on(Hook::PreCall)
        }],
        has_request_state: true,
    }
}

fn abort(error: &EngineError, status: u16, model: Option<String>) -> Abort {
    let (status, kind, fields) = match error {
        EngineError::Exceeded(exceeded) => (
            status,
            "budget_exceeded",
            Map::from_iter([
                ("current_cost".to_owned(), json!(exceeded.current_cost)),
                ("max_budget".to_owned(), json!(exceeded.max_budget)),
                ("message".to_owned(), json!(exceeded.message)),
                ("entity_type".to_owned(), json!(exceeded.entity_type)),
                ("entity_id".to_owned(), json!(exceeded.entity_id)),
                ("model".to_owned(), json!(model)),
            ]),
        ),
        EngineError::Unavailable => (
            503,
            "budget_unavailable",
            Map::from_iter([("detail".to_owned(), json!(UNAVAILABLE))]),
        ),
        other => (
            401,
            "budget_reservation_failed",
            Map::from_iter([("message".to_owned(), json!(other.to_string()))]),
        ),
    };
    Abort {
        error: Some(AbortError {
            kind: kind.to_owned(),
            fields,
        }),
        ..Abort::new(status, error.to_string())
    }
}

fn decode<T: for<'de> Deserialize<'de>>(value: Option<&Value>) -> Result<Option<T>, CallbackError> {
    value
        .filter(|value| !value.is_null())
        .map(|value| {
            T::deserialize(value).map_err(|error| CallbackError::Failed(error.to_string()))
        })
        .transpose()
}

impl Budgets {
    pub fn new(store: Arc<SpendCounters>) -> Self {
        Self {
            store,
            manifest: manifest(),
            held: Mutex::new(HashMap::new()),
        }
    }

    fn registry(&self) -> std::sync::MutexGuard<'_, HashMap<String, Arc<Held>>> {
        self.held.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn held(&self, call_id: &str) -> Option<Arc<Held>> {
        self.registry().get(call_id).cloned()
    }

    fn forget(&self, held: &Held) {
        let mut registry = self.registry();
        if registry
            .get(&held.call_id)
            .is_some_and(|current| std::ptr::eq(Arc::as_ptr(current), held))
        {
            registry.remove(&held.call_id);
        }
    }

    pub async fn take(&self, call_id: &str) -> Option<Reservation> {
        let held = self.held(call_id)?;
        self.forget(&held);
        Some(held.snapshot().await)
    }

    pub async fn settle(
        &self,
        mut reservation: Reservation,
        actual: f64,
        finalize: bool,
    ) -> Result<Reservation, EngineError> {
        engine::settle(&self.store, &mut reservation, actual, finalize).await?;
        Ok(reservation)
    }

    async fn admit(self: &Arc<Self>, input: &HookInput) -> HookOutput {
        let Some(admission) = decode::<Admission>(input.fields.raw(Field::DeprecatedKwargs)?)?
        else {
            return Ok(None);
        };
        let Some(plan) = admission.plan else {
            return Ok(None);
        };
        let call_id = plan
            .reservation_id
            .clone()
            .or(admission.call_id)
            .or_else(|| input.call.as_ref().map(|call| call.ids.call_id.clone()))
            .unwrap_or_default();
        match engine::reserve(&self.store, &plan).await {
            Ok(None) => Ok(None),
            Ok(Some(reservation)) => {
                let held = Arc::new(Held {
                    call_id: call_id.clone(),
                    reservation: tokio::sync::Mutex::new(reservation),
                });
                if let Some(state) = input.state::<BudgetState>() {
                    state.set(Arc::clone(&held));
                }
                self.registry().insert(call_id, Arc::clone(&held));
                Ok(None)
            }
            Err(error) => Ok(Some(Verdict::Abort(abort(
                &error,
                plan.status_code,
                admission.model,
            )))),
        }
    }
}

pub struct Shared(pub Arc<Budgets>);

impl Callback for Shared {
    fn manifest(&self) -> &Manifest {
        &self.0.manifest
    }

    fn request_state(&self) -> Option<RequestState> {
        Some(Arc::new(BudgetState::default()))
    }

    fn handle<'a>(&'a self, input: &'a HookInput) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            match input.hook {
                Hook::PreCall => self.0.admit(input).await,
                _ => Ok(None),
            }
        })
    }
}
