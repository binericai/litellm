use litellm_callbacks::CallbackError;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{
    emit::{Emitter, failed, py_float, py_int, py_int_or_bool, py_positive_number},
    labels::{
        LabelValues, Settings, combined_custom_metadata, custom_labels_from_metadata, is_py_dict,
    },
    pyfmt::{py_number, py_str, py_truthy},
    store::Op,
};

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
pub struct ExceptionFacts {
    pub status_code: Option<i64>,
    pub text: String,
    pub deployment_failure: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AttemptFacts {
    pub event: String,
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
    pub exception: Option<ExceptionFacts>,
    pub settings: Settings,
}

const AUTH_ERROR_PATTERNS: [&str; 5] = [
    "virtual key expected",
    "expected to start with 'sk-'",
    "authentication error",
    "invalid api key",
    "api key not valid",
];

pub fn skips_invalid_key(exception: Option<&ExceptionFacts>) -> bool {
    let Some(exception) = exception else {
        return false;
    };
    if exception.deployment_failure {
        return false;
    }
    exception.status_code == Some(401)
        || AUTH_ERROR_PATTERNS
            .iter()
            .any(|pattern| exception.text.contains(pattern))
}

fn duration(start: Option<Moment>, end: Option<Moment>) -> Result<Option<f64>, CallbackError> {
    match (start, end) {
        (Some(start), Some(end)) if start.aware != end.aware => Err(failed(
            "can't subtract offset-naive and offset-aware datetimes",
        )),
        #[allow(clippy::cast_precision_loss)]
        (Some(start), Some(end)) => Ok(Some((end.us - start.us) as f64 / 1_000_000.0)),
        _ => Ok(None),
    }
}

fn required<'a>(map: &'a Map<String, Value>, key: &str) -> Result<&'a Value, CallbackError> {
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

fn get<'a>(map: &'a Map<String, Value>, key: &str) -> &'a Value {
    map.get(key).unwrap_or(&Value::Null)
}

pub struct Success<'a> {
    pub emit: Emitter<'a>,
    pub facts: &'a AttemptFacts,
    pub owns_latency: bool,
}

impl Success<'_> {
    fn settings(&self) -> &Settings {
        &self.facts.settings
    }

    pub fn run(&self) -> Result<(), CallbackError> {
        let Some(payload) = is_py_dict(&self.facts.payload) else {
            return Err(failed(format!(
                "standard_logging_object is required, got={}",
                py_str(&self.facts.payload)
            )));
        };
        if skips_invalid_key(self.facts.exception.as_ref()) {
            return Ok(());
        }
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
        let context = values.context(self.settings());

        self.emit.inc("litellm_requests_metric", &context, 1.0)?;
        self.emit
            .inc("litellm_spend_metric", &context, py_float(response_cost)?)?;
        self.zero_cost(payload, &values)?;
        self.tokens(payload, metadata, &values)?;
        self.rate_limits(payload, &values)?;
        if self.owns_latency {
            self.latency(&values)?;
        }
        self.cache(payload, metadata, &values)?;
        self.media(metadata, &values)?;
        self.mcp(metadata, &values, response_cost)?;
        self.proxy_total(model_group, &values)
    }

    fn zero_cost(
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

    fn latency(&self, values: &LabelValues) -> Result<(), CallbackError> {
        let times = &self.facts.times;
        let context = values.context(self.settings());
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
                .observe("litellm_request_queue_time_seconds", &context, queue)?;
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
