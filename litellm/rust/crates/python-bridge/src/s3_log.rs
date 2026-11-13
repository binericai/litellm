use std::{collections::HashSet, sync::Arc};

use litellm_callbacks::Callback;
use litellm_callbacks_python::NativeCallback;
use litellm_callbacks_s3_log::{
    Constants, Host, S3Log, S3LogParts, Settings, float_repr, python_backoff, system_clock,
    write_str,
};
use litellm_host_python::{from_py, json_from_py};
use litellm_http::ClientVariant;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PySet, PyString, PyTuple},
};

struct PythonHost(Arc<Py<PyAny>>);

impl Host for PythonHost {
    fn callback_failed(&self) {
        let host = Arc::clone(&self.0);
        let report = move || {
            Python::attach(|py| {
                if let Err(error) = host.call_method0(py, "callback_failed") {
                    error.print(py);
                }
            });
        };
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => drop(runtime.spawn_blocking(report)),
            Err(_) => report(),
        }
    }

    fn cold_storage_is_s3_v2(&self) -> bool {
        Python::attach(|py| {
            self.0
                .call_method0(py, "cold_storage_is_s3_v2")
                .and_then(|answer| answer.extract::<bool>(py))
                .unwrap_or(false)
        })
    }
}

struct Encoder<'py> {
    base_model: Bound<'py, PyAny>,
    safe_json_structure: Bound<'py, PyAny>,
    json_dumps: Bound<'py, PyAny>,
    int_repr: Bound<'py, PyAny>,
    max_depth: usize,
    seen: HashSet<usize>,
}

impl<'py> Encoder<'py> {
    fn new(py: Python<'py>, max_depth: usize) -> PyResult<Self> {
        Ok(Self {
            base_model: py.import("pydantic")?.getattr("BaseModel")?,
            safe_json_structure: py
                .import("litellm.litellm_core_utils.safe_json_dumps")?
                .getattr("safe_json_structure")?,
            json_dumps: py.import("json")?.getattr("dumps")?,
            int_repr: py.get_type::<PyInt>().getattr("__repr__")?,
            max_depth,
            seen: HashSet::new(),
        })
    }

    fn raw(&self, value: &Bound<'py, PyAny>, out: &mut Vec<u8>) -> PyResult<()> {
        let text: String = self.json_dumps.call1((value,))?.extract()?;
        out.extend_from_slice(text.as_bytes());
        Ok(())
    }

    fn text(&self, text: &Bound<'py, PyString>, out: &mut Vec<u8>) -> PyResult<()> {
        match text.to_str() {
            Ok(text) if text.contains('\0') => write_str(out, &text.replace('\0', "")),
            Ok(text) => write_str(out, text),
            Err(_) => self.raw(&text.call_method1("replace", ("\0", ""))?, out)?,
        }
        Ok(())
    }

    fn entries(
        &mut self,
        items: impl Iterator<Item = (Bound<'py, PyAny>, Bound<'py, PyAny>)>,
        depth: usize,
        out: &mut Vec<u8>,
    ) -> PyResult<()> {
        let mut written: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for (key, value) in items {
            let Ok(key) = key.cast_into::<PyString>() else {
                continue;
            };
            let mut encoded_key = Vec::new();
            self.text(&key, &mut encoded_key)?;
            let mut encoded_value = Vec::new();
            self.encode(&value, depth + 1, &mut encoded_value)?;
            match written
                .iter_mut()
                .find(|(present, _)| *present == encoded_key)
            {
                Some((_, slot)) => *slot = encoded_value,
                None => written.push((encoded_key, encoded_value)),
            }
        }
        out.push(b'{');
        for (index, (key, value)) in written.iter().enumerate() {
            if index > 0 {
                out.extend_from_slice(b", ");
            }
            out.extend_from_slice(key);
            out.extend_from_slice(b": ");
            out.extend_from_slice(value);
        }
        out.push(b'}');
        Ok(())
    }

    fn items(
        &mut self,
        items: impl Iterator<Item = Bound<'py, PyAny>>,
        depth: usize,
        out: &mut Vec<u8>,
    ) -> PyResult<()> {
        out.push(b'[');
        for (index, item) in items.enumerate() {
            if index > 0 {
                out.extend_from_slice(b", ");
            }
            self.encode(&item, depth + 1, out)?;
        }
        out.push(b']');
        Ok(())
    }

    fn encode(
        &mut self,
        value: &Bound<'py, PyAny>,
        depth: usize,
        out: &mut Vec<u8>,
    ) -> PyResult<()> {
        if depth > self.max_depth {
            write_str(out, "MaxDepthExceeded");
            return Ok(());
        }
        if let Ok(text) = value.cast::<PyString>() {
            return self.text(text, out);
        }
        if value.is_none() {
            out.extend_from_slice(b"null");
            return Ok(());
        }
        if let Ok(flag) = value.cast::<PyBool>() {
            out.extend_from_slice(if flag.is_true() { b"true" } else { b"false" });
            return Ok(());
        }
        if let Ok(integer) = value.cast::<PyInt>() {
            match integer.extract::<i64>() {
                Ok(integer) => out.extend_from_slice(integer.to_string().as_bytes()),
                Err(_) => {
                    let text: String = self.int_repr.call1((value,))?.extract()?;
                    out.extend_from_slice(text.as_bytes());
                }
            }
            return Ok(());
        }
        if let Ok(float) = value.cast::<PyFloat>() {
            out.extend_from_slice(float_repr(float.value()).as_bytes());
            return Ok(());
        }
        let identity = value.as_ptr() as usize;
        if self.seen.contains(&identity) {
            write_str(out, "CircularReference Detected");
            return Ok(());
        }
        let container = value.cast::<PyDict>().is_ok()
            || value.cast::<PyList>().is_ok()
            || value.cast::<PyTuple>().is_ok()
            || value.is_instance(&self.base_model)?;
        if value.cast::<PySet>().is_ok() {
            let structure = self
                .safe_json_structure
                .call1((value, self.max_depth - depth))?;
            return self.raw(&structure, out);
        }
        if !container {
            match value.str() {
                Ok(text) => self.text(&text, out)?,
                Err(_) => write_str(out, "Unserializable Object"),
            }
            return Ok(());
        }
        self.seen.insert(identity);
        let encoded = if let Ok(dict) = value.cast::<PyDict>() {
            self.entries(dict.iter(), depth, out)
        } else if let Ok(list) = value.cast::<PyList>() {
            self.items(list.iter(), depth, out)
        } else if let Ok(tuple) = value.cast::<PyTuple>() {
            self.items(tuple.iter(), depth, out)
        } else {
            let dumped = value.call_method0("model_dump")?;
            self.encode(&dumped, depth + 1, out)
        };
        self.seen.remove(&identity);
        encoded
    }
}

#[pyclass(frozen, name = "NativeS3Log")]
pub struct NativeS3Log {
    sink: Arc<S3Log>,
    max_depth: usize,
}

#[pymethods]
impl NativeS3Log {
    #[new]
    #[pyo3(signature = (params, constants, host, max_depth))]
    fn new(
        py: Python<'_>,
        params: &Bound<'_, PyDict>,
        constants: &Bound<'_, PyAny>,
        host: Py<PyAny>,
        max_depth: usize,
    ) -> PyResult<Self> {
        let resolved = json_from_py(params.as_any())?;
        let resolved = resolved
            .as_object()
            .ok_or_else(|| PyValueError::new_err("s3 params must be a mapping"))?;
        let constants: Constants = from_py(constants)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("ssl_verify", params.get_item("s3_verify")?)?;
        let config = crate::http::call_config(py, &kwargs, true)?;
        let client = crate::http::pool()
            .client(&config, ClientVariant::Provider)
            .map_err(crate::http::client_error)?;
        let sink = S3Log::new(S3LogParts {
            settings: Settings::from_params(resolved, &constants),
            client,
            environment: Arc::new(|name: &str| std::env::var(name).ok()),
            clock: system_clock,
            backoff: Arc::new(python_backoff),
            host: Arc::new(PythonHost(Arc::new(host))),
        });
        Ok(Self { sink, max_depth })
    }

    fn callback(&self) -> PyResult<NativeCallback> {
        Ok(NativeCallback::new(
            Arc::clone(&self.sink) as Arc<dyn Callback>
        )?)
    }

    #[pyo3(signature = (value, prompts_only))]
    fn encode(
        &self,
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
        prompts_only: bool,
    ) -> PyResult<String> {
        let mut encoder = Encoder::new(py, self.max_depth)?;
        let mut out = Vec::new();
        match value.cast::<PyDict>() {
            Ok(payload) => {
                let copy = payload.copy()?;
                if prompts_only {
                    copy.set_item("response", py.None())?;
                }
                encoder.encode(copy.as_any(), 0, &mut out)?;
            }
            Err(_) => encoder.encode(value, 0, &mut out)?,
        }
        String::from_utf8(out).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn log_prompts_only(&self) -> bool {
        self.sink.log_prompts_only()
    }

    fn partition_granularity(&self) -> &'static str {
        self.sink.partition_granularity().as_str()
    }

    #[getter]
    fn s3_path(&self) -> Option<String> {
        self.sink.settings().path.clone()
    }

    #[getter]
    fn strip_base64_files(&self) -> bool {
        self.sink.settings().strip_base64_files
    }

    fn queued_keys(&self) -> Vec<String> {
        self.sink.queued_keys()
    }

    fn submit_audit<'py>(
        &self,
        py: Python<'py>,
        body: String,
        id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sink = Arc::clone(&self.sink);
        crate::execution::run_async_value(py, async move {
            sink.submit_audit(body, &id).await;
            Ok(())
        })
    }

    fn upload_line_items<'py>(
        &self,
        py: Python<'py>,
        key: String,
        body: String,
        download_name: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sink = Arc::clone(&self.sink);
        crate::execution::run_async_value(py, async move {
            Ok(sink.upload_now(key, body, download_name).await)
        })
    }

    fn download<'py>(&self, py: Python<'py>, key: String) -> PyResult<Bound<'py, PyAny>> {
        let sink = Arc::clone(&self.sink);
        crate::execution::run_async_value(py, async move {
            Ok(sink
                .download(&key)
                .await
                .map(|body| String::from_utf8_lossy(&body).into_owned()))
        })
    }

    fn flush<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let sink = Arc::clone(&self.sink);
        crate::execution::run_async_value(py, async move {
            sink.flush().await;
            Ok(())
        })
    }
}
