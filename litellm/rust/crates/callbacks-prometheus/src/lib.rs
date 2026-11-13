mod emit;
mod failure;
pub mod labels;
pub mod ops;
pub mod pyfmt;
pub mod store;
mod success;
mod tracker;

use std::cell::RefCell;

use litellm_callbacks::CallbackError;
use serde_json::Value;

use crate::{
    emit::Emitter,
    failure::Failure,
    success::{Success, SuccessStep},
};
pub use crate::{
    store::{Clock, FamilySpec, Kind, ManualClock, Op, Store, StoreError, SystemClock, Target},
    success::{AttemptFacts, Level, LogLine, exception_text},
    tracker::SeriesLimits,
};

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr, strum::VariantArray,
)]
pub enum Step {
    #[strum(serialize = "success_head")]
    SuccessHead,
    #[strum(serialize = "success_limits")]
    SuccessLimits,
    #[strum(serialize = "success_latency")]
    SuccessLatency,
    #[strum(serialize = "success_tail")]
    SuccessTail,
    #[strum(serialize = "failure")]
    Failure,
}

pub struct Recorded {
    pub logs: Vec<LogLine>,
    pub outcome: Result<(), CallbackError>,
}

pub fn record(
    store: &Store,
    facts: &AttemptFacts,
    step: Step,
    input_sequence_length: Option<String>,
) -> Recorded {
    let logs = RefCell::new(Vec::new());
    let emit = Emitter {
        store,
        settings: &facts.settings,
    };
    let outcome = match step {
        Step::Failure => Failure {
            emit,
            facts,
            logs: &logs,
        }
        .run(),
        Step::SuccessHead => success(emit, facts, None, &logs).run(SuccessStep::Head),
        Step::SuccessLimits => success(emit, facts, None, &logs).run(SuccessStep::Limits),
        Step::SuccessLatency => {
            success(emit, facts, input_sequence_length, &logs).run(SuccessStep::Latency)
        }
        Step::SuccessTail => success(emit, facts, None, &logs).run(SuccessStep::Tail),
    };
    Recorded {
        logs: logs.into_inner(),
        outcome,
    }
}

fn success<'a>(
    emit: Emitter<'a>,
    facts: &'a AttemptFacts,
    input_sequence_length: Option<String>,
    logs: &'a RefCell<Vec<LogLine>>,
) -> Success<'a> {
    Success {
        emit,
        facts,
        input_sequence_length: input_sequence_length.map_or(Value::Null, Value::from),
        logs,
    }
}
