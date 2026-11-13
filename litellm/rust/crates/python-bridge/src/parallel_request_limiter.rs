use std::sync::Arc;

use litellm_callbacks::{CallFacts, Callback, Field, FieldValues, Hook, HookInput, Verdict};
use litellm_callbacks_parallel_request_limiter::{
    AuthFacts, Backend, Ctx, Descriptor, Engine, EnqueuedReservation, EnqueuedScope, FailureFacts,
    IoReservation, LimiterState, OutputCapFacts, ParallelRequestLimiter, PolicyFacts, Response,
    Settings, SlotAcquisition, SuccessFacts, Unverifiable, WindowIdentity, auth_descriptors,
    output_cap, process_monotonic,
};
use litellm_host_python::{Pythonized, from_py, json_from_py, run_async_value};
use litellm_redis_store::rate_limits::Scope;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use serde_json::{Value, json};

use crate::redis::store::{
    rate_limits::NativeRateLimits, relabel, request::NativeRequestRedis, targeted_call,
};

#[pyclass(frozen, name = "NativeParallelRequestLimiter")]
pub(crate) struct NativeParallelRequestLimiter {
    limiter: Arc<ParallelRequestLimiter>,
    manifest: String,
}

#[pyclass(frozen, name = "NativeParallelRequestLimiterState")]
pub(crate) struct NativeParallelRequestLimiterState(Arc<LimiterState>);

fn limiter_async<'py, T, F>(py: Python<'py>, operation: F) -> PyResult<Bound<'py, PyAny>>
where
    T: for<'a> IntoPyObject<'a> + Send + 'static,
    F: Future<Output = PyResult<T>> + Send + 'static,
{
    let target = py
        .import("litellm.redis.stores.rate_limits.types")?
        .getattr("RATE_LIMITS_TARGET")?;
    let call = targeted_call(py, &target, "eval")?;
    run_async_value(py, call.scope(operation))
}

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

fn windows_json(windows: &[WindowIdentity]) -> Vec<Value> {
    windows
        .iter()
        .map(|identity| {
            let kind: &str = identity.counter.kind.into();
            let backend = match identity.backend {
                Backend::Redis => "redis",
                Backend::Local => "local",
            };
            json!([
                identity.counter.scope.descriptor,
                identity.counter.scope.value,
                kind,
                identity.window_start,
                backend,
            ])
        })
        .collect()
}

fn io_json(reservation: &IoReservation) -> Value {
    json!({
        "tokens": reservation.tokens,
        "scopes": reservation.scopes,
        "windows": windows_json(&reservation.windows),
    })
}

fn reservation_json(reservation: &EnqueuedReservation) -> Value {
    serde_json::to_value(reservation).unwrap_or(Value::Null)
}

fn reply(response: Result<Response, Unverifiable>) -> Value {
    match response {
        Err(Unverifiable) => json!({"unverifiable": true}),
        Ok(response) => {
            let windows: Option<Vec<Value>> = response
                .windows
                .as_ref()
                .map(|windows| windows_json(windows));
            json!({
                "overall_code": response.overall,
                "statuses": response.statuses,
                "windows": windows,
            })
        }
    }
}

#[pymethods]
impl NativeParallelRequestLimiter {
    #[new]
    #[pyo3(signature = (rate_limits, window_size, tpm_reservation_enabled, now = None, monotonic = None))]
    fn new(
        rate_limits: PyRef<'_, NativeRateLimits>,
        window_size: i64,
        tpm_reservation_enabled: bool,
        now: Option<Py<PyAny>>,
        monotonic: Option<Py<PyAny>>,
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
        )
        .with_relabel(Arc::new(relabel));
        let limiter = ParallelRequestLimiter::with_monotonic(
            engine,
            monotonic.map_or_else(process_monotonic, python_clock),
        );
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
        limiter_async(py, async move {
            let verdict: Option<Verdict> = limiter
                .handle(&input)
                .await
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            Ok(Pythonized(verdict))
        })
    }

    fn lock_memory(&self, py: Python<'_>) {
        let limiter = Arc::clone(&self.limiter);
        py.detach(|| limiter.engine().memory_updates().acquire());
    }

    fn unlock_memory(&self) {
        self.limiter.engine().memory_updates().release();
    }

    #[pyo3(signature = (descriptors, read_only, skip_tpm_check, parallel_slot_id, skip_parallel_check, policy, request = None))]
    #[allow(clippy::too_many_arguments)]
    fn should_rate_limit<'py>(
        &self,
        py: Python<'py>,
        descriptors: &str,
        read_only: bool,
        skip_tpm_check: bool,
        parallel_slot_id: Option<String>,
        skip_parallel_check: bool,
        policy: &str,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let descriptors: Vec<Descriptor> = decoded(descriptors)?;
        let policy: PolicyFacts = decoded(policy)?;
        let request = request.map(|request| request.0.clone());
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            let ctx = Ctx {
                policy: policy.into(),
                request: request.as_ref(),
            };
            let response = limiter
                .engine()
                .should_rate_limit(
                    &descriptors,
                    read_only,
                    skip_tpm_check,
                    parallel_slot_id.as_deref(),
                    skip_parallel_check,
                    ctx,
                )
                .await;
            Ok(Pythonized(reply(response)))
        })
    }

    #[pyo3(signature = (descriptors, increments, policy, request = None))]
    fn atomic_check_and_increment<'py>(
        &self,
        py: Python<'py>,
        descriptors: &str,
        increments: Vec<(i64, i64)>,
        policy: &str,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let descriptors: Vec<Descriptor> = decoded(descriptors)?;
        let policy: PolicyFacts = decoded(policy)?;
        let request = request.map(|request| request.0.clone());
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            let ctx = Ctx {
                policy: policy.into(),
                request: request.as_ref(),
            };
            let response = limiter
                .engine()
                .atomic_check_and_increment(&descriptors, &increments, ctx)
                .await;
            Ok(Pythonized(reply(response)))
        })
    }

    fn release_slots<'py>(
        &self,
        py: Python<'py>,
        slot_id: String,
        gauges: Vec<(String, String)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let acquisition = SlotAcquisition {
            slot_id,
            gauges: gauges
                .into_iter()
                .map(|(descriptor, value)| Scope { descriptor, value })
                .collect(),
        };
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            limiter.engine().release_slots(&acquisition).await;
            Ok(())
        })
    }

    fn owner_call_id(&self, state: PyRef<'_, NativeParallelRequestLimiterState>) -> Option<String> {
        state.0.stash().owner_call_id.clone()
    }

    fn identity(&self, state: PyRef<'_, NativeParallelRequestLimiterState>) -> Option<String> {
        state.0.stash().identity.clone()
    }

    fn hand_over(&self, state: PyRef<'_, NativeParallelRequestLimiterState>) -> PyResult<String> {
        let mut stash = state.0.stash();
        stash.owned = false;
        let exported = json!({
            "rate_limit_response": stash.rate_limit_response.as_ref().map(|response| json!({
                "overall_code": response.overall,
                "statuses": response.statuses,
            })),
            "parallel_slot": stash.parallel_slot.as_ref().map(|slot| json!({
                "slot_id": slot.slot_id,
                "gauges": slot.gauges.iter().map(|scope| (&scope.descriptor, &scope.value)).collect::<Vec<_>>(),
            })),
            "reserved_tokens": stash.reserved_tokens,
            "reserved_model": stash.reserved_model.as_ref().map(|model| (&model.requested, &model.group)),
            "reserved_scopes": stash.reserved_scopes,
            "fallback_models": stash.fallback_models,
            "reservation_released": stash.reservation_released,
            "io_reservation_released": stash.io_reservation_released,
            "tpm_limited_tags": stash.tpm_limited_tags,
            "ptu_reserved_tokens": stash.ptu_reserved_tokens,
            "ptu_windows": windows_json(&stash.ptu_windows),
            "itpm": io_json(&stash.itpm),
            "otpm": io_json(&stash.otpm),
            "batch_enqueued_reservation": stash.batch_enqueued.take().map(|reservation| reservation_json(&reservation)),
        });
        serde_json::to_string(&exported).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn enqueued_reserve<'py>(
        &self,
        py: Python<'py>,
        tokens: i64,
        scopes: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let scopes: Vec<EnqueuedScope> = decoded(scopes)?;
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            let outcome = limiter.enqueued().reserve(tokens, scopes).await;
            serde_json::to_string(&outcome)
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })
    }

    fn enqueued_refund<'py>(
        &self,
        py: Python<'py>,
        reservation: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let reservation: EnqueuedReservation = decoded(reservation)?;
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            limiter.enqueued().refund(&reservation).await;
            Ok(())
        })
    }

    fn enqueued_save<'py>(
        &self,
        py: Python<'py>,
        batch_id: String,
        reservation: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let reservation: EnqueuedReservation = decoded(reservation)?;
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            limiter.enqueued().save(&batch_id, &reservation).await;
            Ok(())
        })
    }

    fn enqueued_pop<'py>(&self, py: Python<'py>, batch_id: String) -> PyResult<Bound<'py, PyAny>> {
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            Ok(limiter
                .enqueued()
                .pop(&batch_id)
                .await
                .map(|reservation| reservation_json(&reservation).to_string()))
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

    fn auth_descriptors(&self, facts: &str) -> PyResult<String> {
        let facts: AuthFacts = decoded(facts)?;
        let built =
            auth_descriptors(&facts).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(json!([built.leading, built.organization]).to_string())
    }

    fn output_cap(&self, facts: &str) -> PyResult<Option<i64>> {
        let facts: OutputCapFacts = decoded(facts)?;
        Ok(output_cap(&facts))
    }

    fn take_parallel_slot<'py>(
        &self,
        py: Python<'py>,
        state: PyRef<'py, NativeParallelRequestLimiterState>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = Arc::clone(&state.0);
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            Ok(limiter.take_parallel_slot(&state).await.map(|slot| {
                (
                    slot.slot_id,
                    slot.gauges
                        .into_iter()
                        .map(|scope| (scope.descriptor, scope.value))
                        .collect::<Vec<_>>(),
                )
            }))
        })
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
        limiter_async(py, async move {
            limiter
                .settle_success(&state, facts, request.as_ref())
                .await;
            Ok(())
        })
    }

    #[pyo3(signature = (state, facts, request = None))]
    fn settle_turn<'py>(
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
        limiter_async(py, async move {
            limiter.settle_turn(&state, facts, request.as_ref()).await;
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
        limiter_async(py, async move {
            limiter
                .settle_failure(&state, facts, request.as_ref())
                .await;
            Ok(())
        })
    }

    #[pyo3(signature = (state, call_id, request = None))]
    fn release_for_settlement<'py>(
        &self,
        py: Python<'py>,
        state: PyRef<'py, NativeParallelRequestLimiterState>,
        call_id: Option<String>,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = Arc::clone(&state.0);
        let request = request.map(|request| request.0.clone());
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            limiter
                .release_for_settlement(&state, call_id.as_deref(), request.as_ref())
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
        limiter_async(py, async move {
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
        limiter_async(py, async move {
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
