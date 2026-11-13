use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_callbacks::Callback;
use litellm_callbacks_model_capacity::{
    CapacityError, Delegate, Delegated, Frame, KeyFacts, ModelCapacity,
};
use litellm_callbacks_python::NativeCallback;
use litellm_host_python::{Pythonized, from_py, json_from_py, run_async_value, to_py};
use pyo3::prelude::*;
use serde_json::{Value, json};

struct PyDelegate(Py<PyAny>);

impl Delegate for PyDelegate {
    fn call(&self, method: &'static str, args: Value) -> BoxFuture<'static, Result<Value, String>> {
        let target = Python::attach(|py| self.0.clone_ref(py));
        Box::pin(async move {
            let pending = Python::attach(|py| {
                let args = to_py(py, &args)?;
                let awaitable = target.bind(py).call_method1(method, (args,))?;
                pyo3_async_runtimes::tokio::into_future(awaitable)
            })
            .map_err(|error| error.to_string())?;
            let result = pending.await.map_err(|error| error.to_string())?;
            Python::attach(|py| json_from_py(result.bind(py))).map_err(|error| error.to_string())
        })
    }
}

#[pyclass(frozen, name = "NativeModelCapacity")]
pub struct NativeModelCapacity {
    inner: Arc<ModelCapacity>,
}

#[pymethods]
impl NativeModelCapacity {
    #[new]
    fn new(delegate: Py<PyAny>) -> Self {
        let delegated = Arc::new(Delegated(PyDelegate(delegate)));
        Self {
            inner: Arc::new(ModelCapacity::new(delegated.clone(), delegated)),
        }
    }

    fn callback(&self) -> PyResult<NativeCallback> {
        let callback: Arc<dyn Callback> = self.inner.clone();
        Ok(NativeCallback::new(callback)?)
    }

    #[pyo3(signature = (call_id, key, model, commit))]
    fn fallback_target_allowed<'py>(
        &self,
        py: Python<'py>,
        call_id: Option<String>,
        key: &Bound<'py, PyAny>,
        model: String,
        commit: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let key: KeyFacts = from_py(key)?;
        let inner = Arc::clone(&self.inner);
        run_async_value(py, async move {
            let answer = match inner
                .fallback_target_allowed(call_id.as_deref(), &key, &model, commit)
                .await
            {
                Ok(allowed) => json!({"allowed": allowed}),
                Err(error) => failure(&error),
            };
            Ok(Pythonized(answer))
        })
    }

    #[pyo3(signature = (connection_call_id, key, model))]
    fn enforce_frame<'py>(
        &self,
        py: Python<'py>,
        connection_call_id: Option<String>,
        key: &Bound<'py, PyAny>,
        model: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let key: KeyFacts = from_py(key)?;
        let inner = Arc::clone(&self.inner);
        run_async_value(py, async move {
            let answer = match inner
                .enforce_frame(connection_call_id.as_deref(), &key, &model)
                .await
            {
                Ok(Frame::Allow) => json!({"allowed": true}),
                Ok(Frame::Refuse(refusal)) => json!({"refused": {
                    "detail": refusal.detail,
                    "headers": refusal.headers,
                    "rate_limit_type": refusal.rate_limit_type,
                    "model": refusal.model,
                }}),
                Ok(Frame::Unverifiable(error)) => {
                    let mut answer = failure(&error);
                    answer["unverifiable"] = json!(true);
                    answer
                }
                Err(error) => failure(&error),
            };
            Ok(Pythonized(answer))
        })
    }
}

fn failure(error: &CapacityError) -> Value {
    match error.raised() {
        Some(raised) => json!({"raised": raised.token}),
        None if error.unhashable() => json!({"type_error": error.to_string()}),
        None => json!({"failed": error.to_string()}),
    }
}
