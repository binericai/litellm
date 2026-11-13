use litellm_callbacks::CallbackError;
use std::cell::RefCell;

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{
    emit::{Emitter, failed, py_float, py_int, py_int_or_bool, py_positive_number},
    labels::{
        LabelValues, Settings, combined_custom_metadata, custom_labels_from_metadata, is_py_dict,
    },
    pyfmt::{is_py_number, py_number, py_str, py_truthy},
    store::Op,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuccessStep {
    Head,
    Limits,
    Latency,
    Tail,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub struct Moment {
    pub us: i64,
    pub aware: bool,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct Times {
    pub start_time: Option<Moment>,
    pub end_time: Option<Moment>,
    pub api_call_start_time: Option<Moment>,
    pub completion_start_time: Option<Moment>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AttemptFacts {
    #[serde(default)]
    pub payload: Value,
    #[serde(default)]
    pub model: Value,
    #[serde(default)]
    pub end_user_id: Value,
    #[serde(default)]
    pub service_tier: Value,
    #[serde(default)]
    pub original_model_group: Value,
    #[serde(default)]
    pub queue_time_seconds: Value,
    #[serde(default)]
    pub stream: Value,
    #[serde(default)]
    pub times: Times,
    #[serde(default)]
    pub key: Option<KeyFacts>,
    #[serde(default)]
    pub deployment: Option<DeploymentFacts>,
    pub settings: Settings,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
pub enum Level {
    #[strum(serialize = "debug")]
    Debug,
    #[strum(serialize = "error")]
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogLine {
    pub level: Level,
    pub format: &'static str,
    pub args: Vec<String>,
}

impl LogLine {
    pub const fn new(level: Level, format: &'static str, args: Vec<String>) -> Self {
        Self {
            level,
            format,
            args,
        }
    }
}

pub fn exception_text(error: &CallbackError) -> String {
    let message = match error {
        CallbackError::Failed(message) => message.clone(),
        other => other.to_string(),
    };
    for prefix in ["KeyError: ", "TypeError: ", "ValueError: "] {
        if let Some(rest) = message.strip_prefix(prefix) {
            return rest.to_owned();
        }
    }
    message
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct KeyFacts {
    #[serde(default)]
    pub hashed: Value,
    #[serde(default)]
    pub model_group: Value,
    #[serde(default)]
    pub remaining_requests: Value,
    #[serde(default)]
    pub remaining_tokens: Value,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct DeploymentFacts {
    #[serde(default)]
    pub model_info: Value,
    #[serde(default)]
    pub params_truthy: bool,
    #[serde(default)]
    pub provider: Value,
    #[serde(default)]
    pub params_tpm: Value,
    #[serde(default)]
    pub params_rpm: Value,
    #[serde(default)]
    pub completion_start_present: bool,
}

pub fn duration(start: Option<Moment>, end: Option<Moment>) -> Result<Option<f64>, CallbackError> {
    match (start, end) {
        (Some(start), Some(end)) if start.aware != end.aware => Err(failed(
            "can't subtract offset-naive and offset-aware datetimes",
        )),
        #[allow(clippy::cast_precision_loss)]
        (Some(start), Some(end)) => Ok(Some((end.us - start.us) as f64 / 1_000_000.0)),
        _ => Ok(None),
    }
}

pub fn required<'a>(map: &'a Map<String, Value>, key: &str) -> Result<&'a Value, CallbackError> {
    map.get(key)
        .ok_or_else(|| failed(format!("KeyError: '{key}'")))
}

fn tag_values(tags: &Value) -> Result<Vec<Value>, CallbackError> {
    match tags {
        Value::Array(items) => Ok(items.clone()),
        Value::String(text) => Ok(text.chars().map(|ch| Value::from(ch.to_string())).collect()),
        Value::Object(map) => Ok(map.keys().map(|key| Value::from(key.clone())).collect()),
        other => Err(failed(format!(
            "'{}' object is not iterable",
            py_str(other)
        ))),
    }
}

pub fn get<'a>(map: &'a Map<String, Value>, key: &str) -> &'a Value {
    map.get(key).unwrap_or(&Value::Null)
}

pub struct Success<'a> {
    pub emit: Emitter<'a>,
    pub facts: &'a AttemptFacts,
    pub input_sequence_length: Value,
    pub logs: &'a RefCell<Vec<LogLine>>,
}

impl Success<'_> {
    fn settings(&self) -> &Settings {
        &self.facts.settings
    }

    pub fn run(&self, step: SuccessStep) -> Result<(), CallbackError> {
        let Some(payload) = is_py_dict(&self.facts.payload) else {
            return Err(failed(format!(
                "standard_logging_object is required, got={}",
                py_str(&self.facts.payload)
            )));
        };
        let metadata = required(payload, "metadata")?;
        let metadata =
            is_py_dict(metadata).ok_or_else(|| failed("metadata is not subscriptable"))?;
        let user_id = required(metadata, "user_api_key_user_id")?;
        let hashed_api_key = required(metadata, "user_api_key_hash")?;
        let api_key_alias = required(metadata, "user_api_key_alias")?;
        let team = required(metadata, "user_api_key_team_id")?;
        let team_alias = required(metadata, "user_api_key_team_alias")?;
        let _ = required(payload, "completion_tokens")?;
        let _ = required(payload, "total_tokens")?;
        let response_cost = required(payload, "response_cost")?;
        let combined = combined_custom_metadata(payload);
        let tags = tag_values(required(payload, "request_tags")?)?;
        let model_group = required(payload, "model_group")?;
        let user_email = required(metadata, "user_api_key_user_email")?;
        let model_id = required(payload, "model_id")?;
        let api_base = required(payload, "api_base")?;
        let api_provider = required(payload, "custom_llm_provider")?;
        let model = &self.facts.model;
        let stream = if self.settings().emit_stream_label {
            Value::from(py_str(get(payload, "stream")))
        } else {
            Value::Null
        };
        let values = LabelValues {
            tags,
            custom_metadata_labels: custom_labels_from_metadata(
                &combined,
                &self.settings().custom_metadata_labels,
            ),
            ..LabelValues::default()
        }
        .with("end_user", self.facts.end_user_id.clone())
        .with("hashed_api_key", hashed_api_key.clone())
        .with("api_key_alias", api_key_alias.clone())
        .with("requested_model", model_group.clone())
        .with("model_group", model_group.clone())
        .with("team", team.clone())
        .with("team_alias", team_alias.clone())
        .with(
            "project_id",
            get(metadata, "user_api_key_project_id").clone(),
        )
        .with(
            "project_alias",
            get(metadata, "user_api_key_project_alias").clone(),
        )
        .with("org_id", get(metadata, "user_api_key_org_id").clone())
        .with("org_alias", get(metadata, "user_api_key_org_alias").clone())
        .with("user", user_id.clone())
        .with("user_email", user_email.clone())
        .with("status_code", Value::from("200"))
        .with("model", model.clone())
        .with("litellm_model_name", model.clone())
        .with("model_id", model_id.clone())
        .with("api_base", api_base.clone())
        .with("api_provider", api_provider.clone())
        .with("route", get(metadata, "user_api_key_request_route").clone())
        .with("client_ip", get(metadata, "requester_ip_address").clone())
        .with("user_agent", get(metadata, "user_agent").clone())
        .with("stream", stream)
        .with("service_tier", self.facts.service_tier.clone());
        match step {
            SuccessStep::Head => {
                let context = values.context(self.settings());
                self.emit.inc("litellm_requests_metric", &context, 1.0)?;
                self.emit
                    .inc("litellm_spend_metric", &context, py_float(response_cost)?)?;
                self.zero_cost(payload, &values)?;
                self.tokens(payload, metadata, &values)
            }
            SuccessStep::Limits => {
                self.virtual_key(payload, &values, &combined)?;
                self.rate_limits(payload, &values)
            }
            SuccessStep::Latency => {
                self.latency(&values)?;
                if let Err(error) = self.deployment(payload, &values) {
                    self.log(LogLine::new(
                        Level::Error,
                        "Prometheus Error: set_llm_deployment_success_metrics. Exception occured - %s",
                        vec![exception_text(&error)],
                    ));
                }
                Ok(())
            }
            SuccessStep::Tail => {
                self.cache(payload, metadata, &values)?;
                self.media(metadata, &values)?;
                self.mcp(metadata, &values, response_cost)?;
                self.proxy_total(model_group, &values)
            }
        }
    }

    pub fn zero_cost(
        &self,
        payload: &Map<String, Value>,
        values: &LabelValues,
    ) -> Result<(), CallbackError> {
        let diagnostic = get(payload, "zero_cost_diagnostic");
        if diagnostic.is_null() {
            return Ok(());
        }
        let name = "litellm_zero_cost_requests_total";
        let mut labels = self.emit.labels(name, &values.context(self.settings()));
        if labels.names.iter().any(|label| label == "reason") {
            let reason = is_py_dict(diagnostic)
                .and_then(|map| map.get("reason"))
                .ok_or_else(|| failed("KeyError: 'reason'"))?;
            labels.set("reason", Some(py_str(reason)));
        }
        self.emit.record(name, &labels, Op::Inc(1.0), false)
    }

    fn tokens(
        &self,
        payload: &Map<String, Value>,
        metadata: &Map<String, Value>,
        values: &LabelValues,
    ) -> Result<(), CallbackError> {
        let context = values.context(self.settings());
        for (name, key) in [
            ("litellm_total_tokens_metric", "total_tokens"),
            ("litellm_input_tokens_metric", "prompt_tokens"),
            ("litellm_output_tokens_metric", "completion_tokens"),
        ] {
            self.emit
                .inc(name, &context, py_float(required(payload, key)?)?)?;
        }
        let Some(usage) = metadata.get("usage_object").and_then(is_py_dict) else {
            return Ok(());
        };
        let details = |key: &str| -> Value {
            usage
                .get(key)
                .filter(|value| py_truthy(value))
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new()))
        };
        let prompt = details("prompt_tokens_details");
        let completion = details("completion_tokens_details");
        let from = |details: &Value, key: &str| -> Value {
            is_py_dict(details)
                .and_then(|map| map.get(key))
                .cloned()
                .unwrap_or(Value::Null)
        };
        let cache_write = cache_write_tokens(&prompt).map_or(Value::Null, Value::from);
        let sparse = [
            (
                "litellm_input_cached_tokens_metric",
                from(&prompt, "cached_tokens"),
            ),
            ("litellm_input_cache_creation_tokens_metric", cache_write),
            (
                "litellm_input_audio_tokens_metric",
                from(&prompt, "audio_tokens"),
            ),
            (
                "litellm_output_reasoning_tokens_metric",
                from(&completion, "reasoning_tokens"),
            ),
            (
                "litellm_output_audio_tokens_metric",
                from(&completion, "audio_tokens"),
            ),
        ];
        self.sparse(&sparse, &context)
    }

    fn sparse(
        &self,
        counters: &[(&str, Value)],
        context: &crate::labels::LabelContext,
    ) -> Result<(), CallbackError> {
        for (name, value) in counters {
            if let Some(amount) = py_positive_number(value) {
                self.emit.inc(name, context, amount)?;
            }
        }
        Ok(())
    }

    fn rate_limits(
        &self,
        payload: &Map<String, Value>,
        values: &LabelValues,
    ) -> Result<(), CallbackError> {
        const DESCRIPTORS: [(&str, &str, &str, &str, &str); 8] = [
            (
                "api_key",
                "requests",
                "requests",
                "litellm_api_key_rate_limit_allowed_metric",
                "litellm_api_key_rate_limit_used_metric",
            ),
            (
                "api_key",
                "tokens",
                "tokens",
                "litellm_api_key_rate_limit_allowed_metric",
                "litellm_api_key_rate_limit_used_metric",
            ),
            (
                "team",
                "requests",
                "requests",
                "litellm_team_rate_limit_allowed_metric",
                "litellm_team_rate_limit_used_metric",
            ),
            (
                "team",
                "tokens",
                "tokens",
                "litellm_team_rate_limit_allowed_metric",
                "litellm_team_rate_limit_used_metric",
            ),
            (
                "model_per_project",
                "requests",
                "requests",
                "litellm_project_model_rate_limit_allowed_metric",
                "litellm_project_model_rate_limit_used_metric",
            ),
            (
                "model_per_project",
                "tokens",
                "tokens",
                "litellm_project_model_rate_limit_allowed_metric",
                "litellm_project_model_rate_limit_used_metric",
            ),
            (
                "model_per_project_itpm",
                "tokens",
                "input_tokens",
                "litellm_project_model_rate_limit_allowed_metric",
                "litellm_project_model_rate_limit_used_metric",
            ),
            (
                "model_per_project_otpm",
                "tokens",
                "output_tokens",
                "litellm_project_model_rate_limit_allowed_metric",
                "litellm_project_model_rate_limit_used_metric",
            ),
        ];
        let headers = is_py_dict(get(payload, "hidden_params"))
            .map(|hidden| get(hidden, "additional_headers"))
            .and_then(is_py_dict);
        let header = |name: String| -> Option<i128> {
            headers
                .and_then(|headers| headers.get(&name))
                .and_then(py_int)
        };
        for (descriptor, header_type, rate_limit_type, allowed, used) in DESCRIPTORS {
            let limit = header(format!("x-ratelimit-{descriptor}-limit-{header_type}"));
            let remaining = header(format!("x-ratelimit-{descriptor}-remaining-{header_type}"));
            let context = values
                .clone()
                .with("rate_limit_type", Value::from(rate_limit_type))
                .context(self.settings());
            let labels = self.emit.labels(allowed, &context);
            let (Some(limit), Some(remaining)) = (limit, remaining) else {
                self.emit.remove(allowed, &labels);
                self.emit.remove(used, &labels);
                continue;
            };
            #[allow(clippy::cast_precision_loss)]
            let (limit_value, used_value) = (limit as f64, (limit - remaining) as f64);
            self.emit
                .record(allowed, &labels, Op::Set(limit_value), false)?;
            self.emit
                .record(used, &labels, Op::Set(used_value), false)?;
        }
        Ok(())
    }

    pub fn log(&self, line: LogLine) {
        self.logs.borrow_mut().push(line);
    }

    fn latency(&self, values: &LabelValues) -> Result<(), CallbackError> {
        let times = &self.facts.times;
        let sequenced = if self.input_sequence_length.is_null() {
            values.clone()
        } else {
            values
                .clone()
                .with("input_sequence_length", self.input_sequence_length.clone())
        };
        let context = sequenced.context(self.settings());
        let plain = values.context(self.settings());
        let ttft = duration(times.api_call_start_time, times.completion_start_time)?;
        if let Some(ttft) = ttft
            && self.facts.stream == Value::Bool(true)
        {
            self.emit
                .observe("litellm_llm_api_time_to_first_token_metric", &context, ttft)?;
        }
        if let Some(total) = duration(times.api_call_start_time, times.end_time)? {
            self.emit
                .observe("litellm_llm_api_latency_metric", &context, total)?;
        }
        let queue = &self.facts.queue_time_seconds;
        let queue_seconds = if queue.is_null() {
            None
        } else {
            Some(
                py_number(queue)
                    .ok_or_else(|| failed("'>=' not supported for queue_time_seconds"))?,
            )
        }
        .filter(|seconds| *seconds >= 0.0);
        if let Some(total) = duration(times.start_time, times.end_time)? {
            let observed = queue_seconds.map_or(total, |queue| total + queue);
            self.emit
                .observe("litellm_request_total_latency_metric", &context, observed)?;
        }
        if let Some(queue) = queue_seconds {
            self.emit
                .observe("litellm_request_queue_time_seconds", &plain, queue)?;
        }
        Ok(())
    }

    fn cache(
        &self,
        payload: &Map<String, Value>,
        metadata: &Map<String, Value>,
        values: &LabelValues,
    ) -> Result<(), CallbackError> {
        let context = values.context(self.settings());
        match get(payload, "cache_hit") {
            Value::Null => {}
            Value::Bool(true) => {
                self.emit.inc("litellm_cache_hits_metric", &context, 1.0)?;
                let total = payload
                    .get("total_tokens")
                    .cloned()
                    .unwrap_or(Value::from(0));
                let total = py_number(&total)
                    .or_else(|| total.as_bool().map(|flag| if flag { 1.0 } else { 0.0 }))
                    .ok_or_else(|| failed("'>' not supported for total_tokens"))?;
                if total > 0.0 {
                    self.emit
                        .inc("litellm_cached_tokens_metric", &context, total)?;
                }
            }
            _ => self
                .emit
                .inc("litellm_cache_misses_metric", &context, 1.0)?,
        }
        let Some(usage) = metadata.get("usage_object").and_then(is_py_dict) else {
            return Ok(());
        };
        let (read, creation) = provider_cache_tokens(usage);
        if read > 0.0 {
            self.emit.inc(
                "litellm_provider_cache_read_input_tokens_metric",
                &context,
                read,
            )?;
        }
        if creation > 0.0 {
            self.emit.inc(
                "litellm_provider_cache_creation_input_tokens_metric",
                &context,
                creation,
            )?;
        }
        Ok(())
    }

    fn media(
        &self,
        metadata: &Map<String, Value>,
        values: &LabelValues,
    ) -> Result<(), CallbackError> {
        let Some(usage) = metadata.get("usage_object").and_then(is_py_dict) else {
            return Ok(());
        };
        let context = values.context(self.settings());
        self.sparse(
            &[
                (
                    "litellm_video_duration_seconds_metric",
                    get(usage, "duration_seconds").clone(),
                ),
                (
                    "litellm_images_generated_metric",
                    get(usage, "output_image_count").clone(),
                ),
            ],
            &context,
        )
    }

    fn mcp(
        &self,
        metadata: &Map<String, Value>,
        values: &LabelValues,
        response_cost: &Value,
    ) -> Result<(), CallbackError> {
        let Some(tool) = metadata.get("mcp_tool_call_metadata").and_then(is_py_dict) else {
            return Ok(());
        };
        let mcp = LabelValues::default()
            .with("mcp_tool_name", get(tool, "name").clone())
            .with("mcp_server_name", get(tool, "mcp_server_name").clone())
            .with("hashed_api_key", values.get("hashed_api_key").clone())
            .with("api_key_alias", values.get("api_key_alias").clone())
            .with("team", values.get("team").clone())
            .with("team_alias", values.get("team_alias").clone())
            .with("user", values.get("user").clone())
            .with("end_user", values.get("end_user").clone());
        let context = mcp.context(self.settings());
        self.emit
            .inc("litellm_mcp_tool_calls_total", &context, 1.0)?;
        let cost = py_float(response_cost)?;
        if cost > 0.0 {
            self.emit
                .inc("litellm_mcp_tool_call_spend_metric", &context, cost)?;
        }
        Ok(())
    }

    fn virtual_key(
        &self,
        payload: &Map<String, Value>,
        values: &LabelValues,
        combined: &Map<String, Value>,
    ) -> Result<(), CallbackError> {
        let Some(key) = &self.facts.key else {
            return Ok(());
        };
        let remaining = |given: &Value, kind: &str| -> Result<Value, CallbackError> {
            if !given.is_null() {
                return Ok(given.clone());
            }
            let header = v3_header(
                payload,
                &format!("x-ratelimit-model_per_key-remaining-{kind}"),
            )?;
            Ok(header.map_or_else(|| Value::from(i64::MAX), Value::from))
        };
        let requests = remaining(&key.remaining_requests, "requests")?;
        let tokens = remaining(&key.remaining_tokens, "tokens")?;
        let key_values = LabelValues {
            custom_metadata_labels: custom_labels_from_metadata(
                combined,
                &self.settings().custom_metadata_labels,
            ),
            ..LabelValues::default()
        }
        .with("hashed_api_key", key.hashed.clone())
        .with("api_key_alias", values.get("api_key_alias").clone())
        .with("model", key.model_group.clone())
        .with("model_id", values.get("model_id").clone());
        let context = key_values.context(self.settings());
        for (name, value) in [
            ("litellm_remaining_api_key_requests_for_model", requests),
            ("litellm_remaining_api_key_tokens_for_model", tokens),
        ] {
            let labels = self.emit.labels(name, &context);
            self.emit
                .record(name, &labels, Op::Set(py_float(&value)?), false)?;
        }
        Ok(())
    }

    fn deployment(
        &self,
        payload: &Map<String, Value>,
        values: &LabelValues,
    ) -> Result<(), CallbackError> {
        let Some(deployment) = &self.facts.deployment else {
            return Ok(());
        };
        let api_base = required(payload, "api_base")?;
        let model = &self.facts.model;
        let model_info = &deployment.model_info;
        let info = |key: &str| -> Result<Value, CallbackError> {
            match model_info {
                other if !py_truthy(other) => Ok(Value::Null),
                other => is_py_dict(other)
                    .map(|map| get(map, key).clone())
                    .ok_or_else(|| failed("'model_info' has no attribute 'get'")),
            }
        };
        let model_id = info("id")?;
        if py_truthy(model_info) || deployment.params_truthy {
            let deployment_values = LabelValues::default()
                .with("litellm_model_name", model.clone())
                .with("model_id", model_id.clone())
                .with("api_base", api_base.clone())
                .with("api_provider", deployment.provider.clone())
                .with("model_group", values.get("model_group").clone());
            let context = deployment_values.context(self.settings());
            for (name, key, fallback) in [
                (
                    "litellm_deployment_tpm_limit",
                    "tpm",
                    &deployment.params_tpm,
                ),
                (
                    "litellm_deployment_rpm_limit",
                    "rpm",
                    &deployment.params_rpm,
                ),
            ] {
                let own = info(key)?;
                let limit = if py_truthy(&own) {
                    own
                } else {
                    fallback.clone()
                };
                if limit.is_null() {
                    continue;
                }
                let labels = self.emit.labels(name, &context);
                self.emit
                    .record(name, &labels, Op::Set(py_float(&limit)?), false)?;
            }
        }
        let hidden = required(payload, "hidden_params")?;
        let hidden =
            is_py_dict(hidden).ok_or_else(|| failed("hidden_params is not subscriptable"))?;
        let headers = required(hidden, "additional_headers")?;
        let headers = if py_truthy(headers) {
            is_py_dict(headers)
                .cloned()
                .ok_or_else(|| failed("additional_headers is not a mapping"))?
        } else {
            Map::new()
        };
        let remaining_requests = gauge_reading(get(&headers, "x_ratelimit_remaining_requests"));
        let remaining_tokens = gauge_reading(get(&headers, "x_ratelimit_remaining_tokens"));
        let context = values.context(self.settings());
        let overhead = get(hidden, "litellm_overhead_time_ms");
        if py_truthy(overhead) {
            let seconds = py_divide(overhead, 1000.0)?;
            let name = "litellm_overhead_latency_metric";
            let labels = self.emit.labels(name, &context);
            self.emit
                .record(name, &labels, Op::Observe(seconds), false)?;
        }
        let guardrail = guardrail_overhead_seconds(payload)?;
        if !overhead.is_null() || guardrail > 0.0 {
            let base = if py_truthy(overhead) {
                py_divide(overhead, 1000.0)?
            } else {
                0.0
            };
            let name = "litellm_overhead_with_guardrails_latency_metric";
            let labels = self.emit.labels(name, &context);
            self.emit
                .record(name, &labels, Op::Observe(base + guardrail), false)?;
        }
        for (name, reading) in [
            ("litellm_remaining_requests_metric", remaining_requests),
            ("litellm_remaining_tokens_metric", remaining_tokens),
        ] {
            if let Some(reading) = reading {
                let labels = self.emit.labels(name, &context);
                self.emit.record(name, &labels, Op::Set(reading), false)?;
            }
        }
        set_deployment_state(
            &self.emit,
            0.0,
            &or_text(model),
            &or_text(&model_id),
            &or_text(api_base),
            &or_text(&deployment.provider),
        )?;
        self.emit
            .inc("litellm_deployment_success_responses", &context, 1.0)?;
        self.emit
            .inc("litellm_deployment_total_requests", &context, 1.0)?;
        let times = &self.facts.times;
        let response = duration(times.start_time, times.end_time)?
            .ok_or_else(|| failed("unsupported operand type(s) for -"))?;
        let mut latency = response;
        if self.facts.stream == Value::Bool(true) {
            let first = if deployment.completion_start_present {
                times.completion_start_time.ok_or_else(|| {
                    failed("unsupported operand type(s) for -: 'NoneType' and 'datetime.datetime'")
                })?
            } else {
                times
                    .end_time
                    .ok_or_else(|| failed("unsupported operand type(s) for -"))?
            };
            let ttft = duration(times.start_time, Some(first))?.unwrap_or(0.0);
            if ttft != 0.0 {
                latency = ttft;
            }
        }
        let output_tokens = required(payload, "completion_tokens")?;
        if output_tokens.is_null() {
            return Ok(());
        }
        let tokens = py_number(output_tokens)
            .or_else(|| {
                output_tokens
                    .as_bool()
                    .map(|flag| if flag { 1.0 } else { 0.0 })
            })
            .ok_or_else(|| failed("'>' not supported for completion_tokens"))?;
        if tokens > 0.0 {
            let name = "litellm_deployment_latency_per_output_token";
            let labels = self.emit.labels(name, &context);
            self.emit
                .record(name, &labels, Op::Observe(latency / tokens), false)?;
        }
        Ok(())
    }

    fn proxy_total(&self, model_group: &Value, values: &LabelValues) -> Result<(), CallbackError> {
        let original = &self.facts.original_model_group;
        let requested = match original {
            Value::String(text) if !text.is_empty() => original.clone(),
            _ => model_group.clone(),
        };
        let context = values
            .clone()
            .with("requested_model", requested)
            .context(self.settings());
        self.emit
            .inc("litellm_proxy_total_requests_metric", &context, 1.0)
    }
}

fn or_text(value: &Value) -> Value {
    if py_truthy(value) {
        value.clone()
    } else {
        Value::from("")
    }
}

pub fn set_deployment_state(
    emit: &Emitter<'_>,
    state: f64,
    litellm_model_name: &Value,
    model_id: &Value,
    api_base: &Value,
    api_provider: &Value,
) -> Result<(), CallbackError> {
    let values = LabelValues::default()
        .with("litellm_model_name", litellm_model_name.clone())
        .with("model_id", model_id.clone())
        .with("api_base", api_base.clone())
        .with("api_provider", api_provider.clone());
    let name = "litellm_deployment_state";
    let labels = emit.labels(name, &values.context(emit.settings));
    emit.record(name, &labels, Op::Set(state), false)
}

fn v3_header(payload: &Map<String, Value>, name: &str) -> Result<Option<i64>, CallbackError> {
    let hidden = get(payload, "hidden_params");
    if hidden.is_null() {
        return Ok(None);
    }
    let hidden =
        is_py_dict(hidden).ok_or_else(|| failed("hidden_params has no attribute 'get'"))?;
    let headers = get(hidden, "additional_headers");
    if headers.is_null() {
        return Ok(None);
    }
    let headers =
        is_py_dict(headers).ok_or_else(|| failed("additional_headers has no attribute 'get'"))?;
    Ok(match headers.get(name) {
        Some(Value::Number(number)) if number.is_i64() => number.as_i64(),
        _ => None,
    })
}

fn py_float_text(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    let lowered = trimmed.to_ascii_lowercase();
    let unsigned = lowered.trim_start_matches(['+', '-']);
    if matches!(unsigned, "inf" | "infinity" | "nan") {
        return lowered.parse::<f64>().ok();
    }
    if trimmed.contains('_') {
        let bytes = trimmed.as_bytes();
        let valid = bytes.iter().enumerate().all(|(index, byte)| {
            *byte != b'_'
                || (index > 0
                    && index + 1 < bytes.len()
                    && bytes[index - 1].is_ascii_digit()
                    && bytes[index + 1].is_ascii_digit())
        });
        if !valid {
            return None;
        }
        return trimmed.replace('_', "").parse::<f64>().ok();
    }
    trimmed.parse::<f64>().ok()
}

fn gauge_reading(value: &Value) -> Option<f64> {
    match value {
        _ if is_py_number(value) => py_number(value),
        Value::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
        Value::String(text) => py_float_text(text),
        _ => None,
    }
}

fn py_divide(value: &Value, by: f64) -> Result<f64, CallbackError> {
    match value {
        _ if is_py_number(value) => py_number(value).map(|number| number / by),
        Value::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 } / by),
        _ => None,
    }
    .ok_or_else(|| {
        failed(format!(
            "unsupported operand type(s) for /: '{}' and 'int'",
            crate::pyfmt::py_type_name(value)
        ))
    })
}

fn guardrail_overhead_seconds(payload: &Map<String, Value>) -> Result<f64, CallbackError> {
    let information = get(payload, "guardrail_information");
    let entries: Vec<&Value> = match information {
        Value::Object(_) if is_py_dict(information).is_some() => vec![information],
        Value::Array(items) => items.iter().collect(),
        other if !py_truthy(other) => Vec::new(),
        _ => return Err(failed("guardrail_information is not iterable")),
    };
    let mut total = 0.0;
    for entry in entries {
        let info =
            is_py_dict(entry).ok_or_else(|| failed("guardrail info has no attribute 'get'"))?;
        let mode = get(info, "guardrail_mode");
        let modes: Vec<&Value> = match mode {
            Value::Array(items) => items.iter().collect(),
            other => vec![other],
        };
        let names: Vec<&str> = modes.iter().filter_map(|mode| mode.as_str()).collect();
        let additive = !names.is_empty()
            && names
                .iter()
                .all(|name| matches!(*name, "pre_call" | "post_call"));
        if !additive {
            continue;
        }
        let duration = get(info, "duration");
        if py_truthy(duration) {
            total += py_float(duration)?;
        }
    }
    Ok(total)
}

fn cache_write_tokens(details: &Value) -> Option<i64> {
    let map = is_py_dict(details)?;
    ["cache_write_tokens", "cache_creation_tokens"]
        .iter()
        .find_map(|key| map.get(*key).and_then(Value::as_i64))
}

fn provider_cache_tokens(usage: &Map<String, Value>) -> (f64, f64) {
    let read_raw = get(usage, "cache_read_input_tokens");
    let write_raw = get(usage, "cache_creation_input_tokens");
    let mut read = py_int_or_bool(read_raw).unwrap_or(0.0);
    let mut creation = py_int_or_bool(write_raw).unwrap_or(0.0);
    let prompt = get(usage, "prompt_tokens_details");
    if read_raw.is_null()
        && let Some(cached) = is_py_dict(prompt)
            .and_then(|details| details.get("cached_tokens"))
            .and_then(py_int_or_bool)
    {
        read = cached;
    }
    if write_raw.is_null()
        && let Some(write) = cache_write_tokens(prompt)
    {
        #[allow(clippy::cast_precision_loss)]
        {
            creation = write as f64;
        }
    }
    (read, creation)
}
