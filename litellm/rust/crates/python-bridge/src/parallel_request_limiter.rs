use std::sync::Arc;

use litellm_callbacks::{CallFacts, Callback, Field, FieldValues, Hook, HookInput, Verdict};
use litellm_callbacks_parallel_request_limiter::{
    Descriptor, Engine, FailureFacts, LimiterState, ParallelRequestLimiter, PolicyFacts, Settings,
    SuccessFacts,
};
use litellm_host_python::{Pythonized, from_py, json_from_py, run_async_value};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use serde_json::{Value, json};

use crate::redis::store::{rate_limits::NativeRateLimits, request::NativeRequestRedis};

#[pyclass(frozen, name = "NativeParallelRequestLimiter")]
pub(crate) struct NativeParallelRequestLimiter {
    limiter: Arc<ParallelRequestLimiter>,
    manifest: String,
}

#[pyclass(frozen, name = "NativeParallelRequestLimiterState")]
pub(crate) struct NativeParallelRequestLimiterState(Arc<LimiterState>);

fn named<T: serde::de::DeserializeOwned>(name: &str) -> PyResult<T> {
    serde_json::from_value(Value::String(name.to_owned()))
        .map_err(|_| PyValueError::new_err(format!("{name} is not a known name")))
}

fn decoded<T: serde::de::DeserializeOwned>(text: &str) -> PyResult<T> {
    serde_json::from_str(text).map_err(|error| PyValueError::new_err(error.to_string()))
}

fn python_clock(now: Py<PyAny>) -> litellm_callbacks_parallel_request_limiter::Now {
    Arc::new(move || {
        Python::attach(|py| {
            now.call0(py)
                .and_then(|value| value.extract::<f64>(py))
                .unwrap_or_else(|_| system_now())
        })
    })
}

fn system_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs_f64())
        .unwrap_or_default()
}

#[pymethods]
impl NativeParallelRequestLimiter {
    #[new]
    #[pyo3(signature = (rate_limits, window_size, tpm_reservation_enabled, now = None))]
    fn new(
        rate_limits: PyRef<'_, NativeRateLimits>,
        window_size: i64,
        tpm_reservation_enabled: bool,
        now: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let clock = match now {
            Some(now) => python_clock(now),
            None => Arc::new(system_now),
        };
        let engine = Engine::new(
            Arc::clone(&rate_limits.0),
            Settings {
                window_size,
                tpm_reservation_enabled,
            },
            clock,
        );
        let limiter = ParallelRequestLimiter::new(engine);
        limiter
            .manifest()
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let manifest = serde_json::to_string(limiter.manifest())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            limiter: Arc::new(limiter),
            manifest,
        })
    }

    fn manifest(&self) -> &str {
        &self.manifest
    }

    fn request_state(&self) -> NativeParallelRequestLimiterState {
        NativeParallelRequestLimiterState(Arc::new(LimiterState::default()))
    }

    #[pyo3(signature = (hook, fields, call, state, /, request = None))]
    fn handle<'py>(
        &self,
        py: Python<'py>,
        hook: &str,
        fields: &Bound<'py, PyDict>,
        call: Option<&Bound<'py, PyAny>>,
        state: Option<PyRef<'py, NativeParallelRequestLimiterState>>,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let hook: Hook = named(hook)?;
        let subscription = self
            .limiter
            .manifest()
            .subscription(hook)
            .ok_or_else(|| PyValueError::new_err(format!("{hook:?} has no subscription")))?;
        let mut values = FieldValues::new(subscription.reads);
        for (name, value) in fields {
            let field: Field = named(&name.extract::<String>()?)?;
            values.insert(field, json_from_py(&value)?);
        }
        let state = state.map(|state| Arc::clone(&state.0));
        if let Some(state) = &state {
            state.bind_request(request.map(|request| request.0.clone()));
        }
        let input = HookInput {
            hook,
            fields: values,
            call: call.map(from_py::<CallFacts>).transpose()?,
            state: state.map(|state| state as litellm_callbacks::RequestState),
        };
        let limiter = Arc::clone(&self.limiter);
        run_async_value(py, async move {
            let verdict: Option<Verdict> = limiter
                .handle(&input)
                .await
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            Ok(Pythonized(verdict))
        })
    }

    fn owned(&self, state: PyRef<'_, NativeParallelRequestLimiterState>) -> bool {
        state.0.stash().owned
    }

    fn first_pass_admitted(&self, state: PyRef<'_, NativeParallelRequestLimiterState>) -> bool {
        state.0.stash().first_pass_admitted
    }

    fn rate_limit_response(
        &self,
        state: PyRef<'_, NativeParallelRequestLimiterState>,
    ) -> PyResult<Option<String>> {
        let stash = state.0.stash();
        stash
            .rate_limit_response
            .as_ref()
            .map(|response| {
                serde_json::to_string(&json!({
                    "overall_code": response.overall,
                    "statuses": response.statuses,
                }))
                .map_err(|error| PyValueError::new_err(error.to_string()))
            })
            .transpose()
    }

    fn reserved_model(
        &self,
        state: PyRef<'_, NativeParallelRequestLimiterState>,
    ) -> Option<(String, String)> {
        state
            .0
            .stash()
            .reserved_model
            .as_ref()
            .map(|model| (model.requested.clone(), model.group.clone()))
    }

    fn reserved_tokens(&self, state: PyRef<'_, NativeParallelRequestLimiterState>) -> i64 {
        state.0.stash().reserved_tokens
    }

    fn holds_parallel_slot(&self, state: PyRef<'_, NativeParallelRequestLimiterState>) -> bool {
        state.0.stash().parallel_slot.is_some()
    }

    #[pyo3(signature = (state, facts, request = None))]
    fn settle_success<'py>(
        &self,
        py: Python<'py>,
        state: PyRef<'py, NativeParallelRequestLimiterState>,
        facts: &str,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let facts: SuccessFacts = decoded(facts)?;
        let state = Arc::clone(&state.0);
        let request = request.map(|request| request.0.clone());
        let limiter = Arc::clone(&self.limiter);
        run_async_value(py, async move {
            limiter
                .settle_success(&state, facts, request.as_ref())
                .await;
            Ok(())
        })
    }

    #[pyo3(signature = (state, facts, request = None))]
    fn settle_failure<'py>(
        &self,
        py: Python<'py>,
        state: PyRef<'py, NativeParallelRequestLimiterState>,
        facts: &str,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let facts: FailureFacts = decoded(facts)?;
        let state = Arc::clone(&state.0);
        let request = request.map(|request| request.0.clone());
        let limiter = Arc::clone(&self.limiter);
        run_async_value(py, async move {
            limiter
                .settle_failure(&state, facts, request.as_ref())
                .await;
            Ok(())
        })
    }

    fn release_on_disconnect<'py>(
        &self,
        py: Python<'py>,
        state: PyRef<'py, NativeParallelRequestLimiterState>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = Arc::clone(&state.0);
        let limiter = Arc::clone(&self.limiter);
        run_async_value(py, async move {
            limiter.release_on_disconnect(&state).await;
            Ok(())
        })
    }

    #[pyo3(signature = (state, descriptors, model, policy, request = None))]
    fn fallback_target_allowed<'py>(
        &self,
        py: Python<'py>,
        state: Option<PyRef<'py, NativeParallelRequestLimiterState>>,
        descriptors: &str,
        model: String,
        policy: &str,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let descriptors: Vec<Descriptor> = decoded(descriptors)?;
        let policy: PolicyFacts = decoded(policy)?;
        let state = state.map(|state| Arc::clone(&state.0));
        let request = request.map(|request| request.0.clone());
        let limiter = Arc::clone(&self.limiter);
        run_async_value(py, async move {
            let verdict = limiter
                .fallback_target_allowed(
                    state.as_deref(),
                    descriptors,
                    &model,
                    policy,
                    request.as_ref(),
                )
                .await;
            Ok(Pythonized(match verdict {
                Ok(allowed) => json!({"allowed": allowed}),
                Err(abort) => json!({"allowed": false, "abort": Verdict::Abort(*abort)}),
            }))
        })
    }
}
