use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use bytes::Bytes;
use futures_util::future::BoxFuture;

use litellm_callbacks::Callback;
use litellm_callbacks_python::NativeCallback;
use litellm_callbacks_s3_log::{
    Constants, Host, S3Log, S3LogParts, Settings, float_repr, panic_message, python_backoff,
    system_clock, write_str,
};
use litellm_host_python::{from_py, json_from_py};
use litellm_http::ClientVariant;
use pyo3::{
    prelude::*,
    types::{PyBool, PyCFunction, PyDict, PyFloat, PyInt, PyList, PySet, PyString, PyTuple},
};
use serde_json::{Map, Value};
use tokio::sync::oneshot;

type Payloads = Arc<Mutex<HashMap<u64, Py<PyAny>>>>;
type Rendered = Result<Vec<Bytes>, String>;

tokio::task_local! {
    static RENDER_LOOP: Option<Arc<Py<PyAny>>>;
}

struct PythonHost {
    host: Arc<Py<PyAny>>,
    payloads: Payloads,
    next: AtomicU64,
    max_depth: usize,
}

fn encode_all(py: Python<'_>, payloads: &Payloads, ids: &[u64], max_depth: usize) -> Rendered {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ids.iter()
            .map(|id| {
                let value = payloads
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .get(id)
                    .map(|value| value.clone_ref(py))
                    .ok_or_else(|| "the payload was already released".to_owned())?;
                let mut out = Vec::new();
                Encoder::new(py, max_depth)
                    .and_then(|mut encoder| encoder.encode(value.bind(py), 0, &mut out))
                    .map_err(|error| error.to_string())?;
                Ok(Bytes::from(out))
            })
            .collect()
    }))
    .unwrap_or_else(|panic| Err(panic_message(panic.as_ref())))
}

fn render_on_loop(
    py: Python<'_>,
    event_loop: &Py<PyAny>,
    payloads: Payloads,
    ids: Vec<u64>,
    max_depth: usize,
    reply: oneshot::Sender<Rendered>,
) -> PyResult<()> {
    let reply = Mutex::new(Some(reply));
    let render = PyCFunction::new_closure(
        py,
        None,
        None,
        move |args: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| {
            let rendered = encode_all(args.py(), &payloads, &ids, max_depth);
            if let Some(reply) = reply.lock().unwrap_or_else(PoisonError::into_inner).take() {
                drop(reply.send(rendered));
            }
        },
    )?;
    event_loop
        .bind(py)
        .call_method1("call_soon_threadsafe", (render,))?;
    Ok(())
}

fn running_loop(py: Python<'_>) -> Option<Arc<Py<PyAny>>> {
    py.import("asyncio")
        .and_then(|asyncio| asyncio.call_method0("get_running_loop"))
        .ok()
        .map(|event_loop| Arc::new(event_loop.unbind()))
}

impl PythonHost {
    fn stash(&self, payload: Py<PyAny>) -> u64 {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.payloads
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id, payload);
        id
    }

    fn ask<T>(
        &self,
        method: &str,
        argument: Option<&str>,
        read: impl for<'py> FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
    ) -> Result<T, String> {
        Python::attach(|py| {
            let answer = match argument {
                Some(argument) => self.host.call_method1(py, method, (argument,)),
                None => self.host.call_method0(py, method),
            };
            answer
                .and_then(|answer| read(answer.bind(py)))
                .map_err(|error| error.to_string())
        })
    }
}

impl Host for PythonHost {
    fn callback_failed(&self) {
        let host = Arc::clone(&self.host);
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
            self.host
                .call_method0(py, "cold_storage_is_s3_v2")
                .and_then(|answer| answer.extract::<bool>(py))
                .unwrap_or(false)
        })
    }

    fn render(&self, ids: Vec<u64>) -> BoxFuture<'static, Rendered> {
        let payloads = Arc::clone(&self.payloads);
        let max_depth = self.max_depth;
        let event_loop = RENDER_LOOP.try_with(Clone::clone).ok().flatten();
        Box::pin(async move {
            let Some(event_loop) = event_loop else {
                return tokio::task::spawn_blocking(move || {
                    Python::attach(|py| encode_all(py, &payloads, &ids, max_depth))
                })
                .await
                .map_err(|error| error.to_string())?;
            };
            let (reply, rendered) = oneshot::channel();
            tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    render_on_loop(py, &event_loop, payloads, ids, max_depth, reply)
                        .map_err(|error| error.to_string())
                })
            })
            .await
            .map_err(|error| error.to_string())??;
            rendered
                .await
                .map_err(|_| "the event loop closed before the payload was rendered".to_owned())?
        })
    }

    fn release(&self, payload: u64) {
        let released = self
            .payloads
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&payload);
        drop(released);
    }

    fn region_from_environment(&self) -> Result<Option<String>, String> {
        self.ask("region_from_environment", None, |answer| {
            answer.extract::<Option<String>>()
        })
    }

    fn web_identity_token(&self, reference: &str) -> Result<String, String> {
        self.ask("web_identity_token", Some(reference), |answer| {
            answer.extract::<String>()
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
            self.seen.insert(identity);
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
    host: Arc<PythonHost>,
}

fn client(
    py: Python<'_>,
    params: &Bound<'_, PyDict>,
    connect_timeout_seconds: f64,
) -> PyResult<litellm_http::Client> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("ssl_verify", params.get_item("s3_verify")?)?;
    let mut config = crate::http::call_config(py, &kwargs, true)?;
    config.connect_timeout = Duration::from_secs_f64(connect_timeout_seconds.max(0.0));
    crate::http::pool()
        .client(&config, ClientVariant::Provider)
        .map_err(crate::http::client_error)
}

fn settings_param(value: &Bound<'_, PyAny>) -> Value {
    if let Ok(integer) = value.cast::<PyInt>()
        && !value.is_instance_of::<PyBool>()
        && integer.extract::<i64>().is_err()
        && integer.extract::<u64>().is_err()
    {
        return value
            .str()
            .map_or(Value::Null, |text| Value::String(text.to_string()));
    }
    json_from_py(value).unwrap_or(Value::Null)
}

fn settings_params(params: &Bound<'_, PyDict>) -> Map<String, Value> {
    params
        .iter()
        .filter_map(|(key, value)| {
            let key = key.cast_into::<PyString>().ok()?;
            Some((key.to_str().ok()?.to_owned(), settings_param(&value)))
        })
        .collect()
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
        let resolved = settings_params(params);
        let constants: Constants = from_py(constants)?;
        let host = Arc::new(PythonHost {
            host: Arc::new(host),
            payloads: Payloads::default(),
            next: AtomicU64::new(0),
            max_depth,
        });
        let sink = S3Log::new(S3LogParts {
            settings: Settings::from_params(&resolved, &constants),
            client: client(py, params, constants.upload_attempt_timeout_seconds)?,
            download_client: client(py, params, constants.download_connect_timeout_seconds)?,
            environment: Arc::new(|name: &str| std::env::var(name).ok()),
            clock: system_clock,
            backoff: Arc::new(python_backoff),
            host: Arc::clone(&host) as Arc<dyn Host>,
        });
        Ok(Self { sink, host })
    }

    fn callback(&self) -> PyResult<NativeCallback> {
        Ok(NativeCallback::new(
            Arc::clone(&self.sink) as Arc<dyn Callback>
        )?)
    }

    fn stash(&self, payload: Py<PyAny>) -> u64 {
        self.host.stash(payload)
    }

    fn stashed(&self) -> usize {
        self.host
            .payloads
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
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
        payload: u64,
        id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sink = Arc::clone(&self.sink);
        let event_loop = running_loop(py);
        crate::execution::run_async_value(
            py,
            RENDER_LOOP.scope(event_loop, async move {
                sink.submit_audit(payload, &id).await;
                Ok(())
            }),
        )
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
        let event_loop = running_loop(py);
        crate::execution::run_async_value(
            py,
            RENDER_LOOP.scope(event_loop, async move {
                sink.flush().await;
                Ok(())
            }),
        )
    }
}
