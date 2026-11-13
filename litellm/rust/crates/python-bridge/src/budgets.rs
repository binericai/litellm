use std::sync::Arc;

use litellm_callbacks_budgets::{Budgets, Check, Reservation, Shared, decide};
use litellm_callbacks_python::NativeCallback;
use litellm_host_python::{Pythonized, from_py_argument, run_async_value, to_py};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::redis::store::spend_counters::NativeSpendCounters;

#[pyclass(frozen, name = "NativeBudgets")]
pub(crate) struct NativeBudgets(Arc<Budgets>);

#[pymethods]
impl NativeBudgets {
    #[new]
    fn new(spend_counters: PyRef<'_, NativeSpendCounters>) -> Self {
        Self(Arc::new(Budgets::new(spend_counters.store())))
    }

    fn callback(&self) -> PyResult<NativeCallback> {
        Ok(NativeCallback::new(Arc::new(Shared(Arc::clone(&self.0))))?)
    }

    fn take<'py>(&self, py: Python<'py>, call_id: String) -> PyResult<Bound<'py, PyAny>> {
        let budgets = Arc::clone(&self.0);
        run_async_value(py, async move {
            let reservation: Option<Reservation> = budgets.take(&call_id).await;
            Ok(Pythonized(reservation))
        })
    }

    fn settle<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = from_py_argument)] reservation: Reservation,
        actual: f64,
        finalize: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let budgets = Arc::clone(&self.0);
        run_async_value(py, async move {
            budgets
                .settle(reservation, actual, finalize)
                .await
                .map(Pythonized)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))
        })
    }
}

#[pyfunction]
pub(crate) fn budget_check_decision(
    py: Python<'_>,
    #[pyo3(from_py_with = from_py_argument)] check: Check,
) -> PyResult<Py<PyAny>> {
    to_py(py, &decide(&check))
}
