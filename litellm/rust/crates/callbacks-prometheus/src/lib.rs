mod emit;
pub mod exposition;
mod failure;
pub mod labels;
pub mod ops;
pub mod pyfmt;
pub mod store;
mod success;
mod tracker;

use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Activation, Callback, CallbackError, Field, FieldSet, Hook, HookInput, HookOutput, Manifest,
    OptOutBy, Selection, Subscription,
};

use crate::{emit::Emitter, failure::Failure, success::Success};
pub use crate::{
    store::{Clock, FamilySpec, Kind, ManualClock, Op, Store, StoreError, SystemClock, Target},
    success::AttemptFacts,
    tracker::SeriesLimits,
};

pub const NAME: &str = "prometheus";

pub struct Prometheus {
    store: Arc<Store>,
    manifest: Manifest,
    owns_latency: bool,
}

impl Prometheus {
    pub fn new(store: Arc<Store>, owns_latency: bool) -> Self {
        Self {
            store,
            manifest: Manifest {
                selection: Selection {
                    activation: Activation::DefaultOn,
                    opt_out_by: vec![OptOutBy::Admin, OptOutBy::Caller],
                    honors_no_log: true,
                    ..Selection::named(NAME)
                },
                subscriptions: vec![Subscription {
                    reads: FieldSet::of(&[Field::DeprecatedKwargs]),
                    ..Subscription::on(Hook::AttemptEnded)
                }],
                has_request_state: false,
            },
            owns_latency,
        }
    }

    pub fn store(&self) -> &Arc<Store> {
        &self.store
    }

    fn attempt_ended(&self, input: &HookInput) -> Result<(), CallbackError> {
        let Some(raw) = input.fields.raw(Field::DeprecatedKwargs)? else {
            return Ok(());
        };
        let facts: AttemptFacts = serde_json::from_value(raw.clone())
            .map_err(|error| CallbackError::Failed(format!("attempt facts: {error}")))?;
        self.record_attempt(&facts)
    }

    pub fn record_attempt(&self, facts: &AttemptFacts) -> Result<(), CallbackError> {
        if facts.event == "failure" {
            return Failure {
                emit: Emitter {
                    store: &self.store,
                    settings: &facts.settings,
                },
                facts,
            }
            .run();
        }
        if facts.event != "success" {
            return Ok(());
        }
        Success {
            emit: Emitter {
                store: &self.store,
                settings: &facts.settings,
            },
            facts,
            owns_latency: self.owns_latency,
        }
        .run()
    }
}

impl Callback for Prometheus {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(&'a self, input: &'a HookInput) -> BoxFuture<'a, HookOutput> {
        let outcome = match input.hook {
            Hook::AttemptEnded => self.attempt_ended(input).map(|()| None),
            _ => Ok(None),
        };
        Box::pin(async move { outcome })
    }
}
