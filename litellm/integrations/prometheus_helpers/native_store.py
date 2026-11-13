from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, TypeVar

import openai
from prometheus_client import metrics as prometheus_metrics
from prometheus_client.metrics_core import Metric
from prometheus_client.registry import Collector
from prometheus_client.utils import (
    floatToGoString,  # pyright: ignore[reportUnknownVariableType]  # the client ships no annotations
)
from pydantic import ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger_v2.enums import Field
from litellm.integrations.custom_logger_v2.native import NativeLogger, native_logger
from litellm.integrations.prometheus_helpers import get_cached_end_user_id_for_cost_tracking
from litellm.litellm_core_utils.core_helpers import (
    env_flag,
    get_litellm_metadata_from_kwargs,  # pyright: ignore[reportUnknownVariableType]  # the helper returns an untyped dict
)
from litellm.litellm_core_utils.service_tier_utils import get_service_tier_from_standard_logging_payload
from litellm.rust_bridge import _native
from litellm.types.integrations.prometheus import NoOpMetric

if TYPE_CHECKING:
    from litellm.integrations.prometheus import PrometheusLogger

_Parsed: Final = TypeVar("_Parsed")
PrometheusKind: TypeAlias = Literal["counter", "gauge", "histogram"]

NATIVE_PROMETHEUS_ENV: Final = "LITELLM_NATIVE_PROMETHEUS"
NativePrometheusMetric: Final = _native.NativePrometheusMetric

NATIVE_SUCCESS_METRICS: Final = (
    "litellm_requests_metric",
    "litellm_spend_metric",
    "litellm_zero_cost_requests_total",
    "litellm_total_tokens_metric",
    "litellm_input_tokens_metric",
    "litellm_output_tokens_metric",
    "litellm_input_cached_tokens_metric",
    "litellm_input_cache_creation_tokens_metric",
    "litellm_input_audio_tokens_metric",
    "litellm_output_reasoning_tokens_metric",
    "litellm_output_audio_tokens_metric",
    "litellm_api_key_rate_limit_allowed_metric",
    "litellm_api_key_rate_limit_used_metric",
    "litellm_team_rate_limit_allowed_metric",
    "litellm_team_rate_limit_used_metric",
    "litellm_project_model_rate_limit_allowed_metric",
    "litellm_project_model_rate_limit_used_metric",
    "litellm_llm_api_time_to_first_token_metric",
    "litellm_llm_api_latency_metric",
    "litellm_request_total_latency_metric",
    "litellm_request_queue_time_seconds",
    "litellm_cache_hits_metric",
    "litellm_cache_misses_metric",
    "litellm_cached_tokens_metric",
    "litellm_provider_cache_read_input_tokens_metric",
    "litellm_provider_cache_creation_input_tokens_metric",
    "litellm_video_duration_seconds_metric",
    "litellm_images_generated_metric",
    "litellm_mcp_tool_calls_total",
    "litellm_mcp_tool_call_spend_metric",
    "litellm_proxy_total_requests_metric",
)

_PAYLOAD_KEYS: Final = (
    "metadata",
    "completion_tokens",
    "total_tokens",
    "prompt_tokens",
    "response_cost",
    "request_tags",
    "model_group",
    "model_id",
    "api_base",
    "custom_llm_provider",
    "stream",
    "zero_cost_diagnostic",
    "cache_hit",
)
_EPOCH_NAIVE: Final = datetime(1970, 1, 1)  # noqa: DTZ001  # callback timestamps use the legacy naive contract
_EPOCH_AWARE: Final = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ONE_MICROSECOND: Final = timedelta(microseconds=1)
_MAPPING: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_MOST_RECENT_MODES: Final = frozenset(("mostrecent", "livemostrecent"))
_STRICT: Final = ConfigDict(strict=True)
_DICT: Final[TypeAdapter[dict[object, object]]] = TypeAdapter(  # mutable-ok: pydantic hands back a fresh dict per call
    dict[object, object], config=_STRICT
)
_LIST: Final[TypeAdapter[list[object]]] = TypeAdapter(  # mutable-ok: pydantic hands back a fresh list per call
    list[object], config=_STRICT
)
_TUPLE: Final[TypeAdapter[tuple[object, ...]]] = TypeAdapter(tuple[object, ...], config=_STRICT)
_KINDS: Final[Mapping[str, PrometheusKind]] = {"counter": "counter", "gauge": "gauge", "histogram": "histogram"}


def native_prometheus_enabled() -> bool:
    return env_flag(os.environ, NATIVE_PROMETHEUS_ENV, verbose_logger) and "PROMETHEUS_MULTIPROC_DIR" not in os.environ


def _parsed(adapter: TypeAdapter[_Parsed], value: object) -> _Parsed | None:
    try:
        return adapter.validate_python(value)
    except ValidationError:
        return None


def _items(value: object) -> Sequence[object] | None:
    listed: Final = _parsed(_LIST, value)
    return listed if listed is not None else _parsed(_TUPLE, value)


def json_safe(value: object) -> object:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    mapping: Final = _parsed(_DICT, value)
    if mapping is not None:
        if all(isinstance(key, str) for key in mapping):
            return {str(key): json_safe(item) for key, item in mapping.items()}
        return {"$py": str(value)}
    items: Final = _items(value)
    if items is not None:
        return [json_safe(item) for item in items]
    return {"$py": str(value)}


def _moment(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, datetime):
        return None
    aware: Final = value.utcoffset() is not None
    return {"us": (value - (_EPOCH_AWARE if aware else _EPOCH_NAIVE)) // _ONE_MICROSECOND, "aware": aware}


def _payload(value: object) -> object:
    payload: Final = _parsed(_DICT, value)
    if payload is None:
        return json_safe(value)
    hidden: Final = payload.get("hidden_params")
    projected: Final = {key: json_safe(payload[key]) for key in _PAYLOAD_KEYS if key in payload}
    if "hidden_params" not in payload:
        return projected
    hidden_mapping: Final = _parsed(_DICT, hidden)
    hidden_view: Final = (
        json_safe(hidden)
        if hidden_mapping is None
        else {"additional_headers": json_safe(hidden_mapping.get("additional_headers"))}
    )
    return {**projected, "hidden_params": hidden_view}


def _settings() -> Mapping[str, object]:
    return {
        "emit_stream_label": litellm.prometheus_emit_stream_label is True,
        "custom_metadata_labels": list(litellm.custom_prometheus_metadata_labels or ()),
        "custom_tags": list(litellm.custom_prometheus_tags or ()),
        "disable_end_user_cost_tracking": bool(litellm.disable_end_user_cost_tracking),
        "end_user_prometheus_only": litellm.enable_end_user_cost_tracking_prometheus_only is True,
        "end_user_max_series": litellm.prometheus_end_user_metrics_max_series_per_metric,
        "end_user_ttl_seconds": litellm.prometheus_end_user_metrics_ttl_seconds,
        "end_user_cleanup_interval_seconds": litellm.prometheus_end_user_metrics_cleanup_interval_seconds,
    }


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    try:
        return _MAPPING.validate_python(value)
    except ValidationError:
        return {}


class AttemptFacts:
    __slots__ = ("_logger",)

    def __init__(self, logger: PrometheusLogger) -> None:
        self._logger: Final = logger

    def _exception(self, kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
        from litellm.integrations.prometheus import (
            _is_deployment_failure,  # pyright: ignore[reportPrivateUsage]  # the logger's own deployment-failure rule keeps the skip identical
        )

        exception: Final = kwargs.get("exception")
        if not isinstance(exception, BaseException):
            return None
        return {
            "status_code": self._logger._extract_status_code(kwargs=dict(kwargs)),  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # the logger's own status extraction keeps the skip identical
            "text": str(exception).lower(),
            "deployment_failure": isinstance(exception, openai.APIError) and _is_deployment_failure(exception, kwargs),
        }

    def __call__(self, kwargs: object) -> object:
        call: Final = _mapping_or_empty(kwargs)
        payload: Final = call.get("standard_logging_object")
        litellm_params: Final = _mapping_or_empty(call.get("litellm_params"))
        request_metadata: Final = {
            **_mapping_or_empty(get_litellm_metadata_from_kwargs(dict(call))),  # pyright: ignore[reportUnknownArgumentType]  # validated to a mapping here
            **_mapping_or_empty(litellm_params.get("metadata")),
        }
        return {
            "event": "failure" if call.get("log_event_type") == "failed_api_call" else "success",
            "payload": _payload(payload),
            "model": json_safe(call.get("model", "")),
            "end_user_id": json_safe(
                get_cached_end_user_id_for_cost_tracking()(dict(litellm_params), service_type="prometheus")
            ),
            "service_tier": json_safe(
                get_service_tier_from_standard_logging_payload(payload)  # pyright: ignore[reportArgumentType]  # the payload is checked to be a dict
                if isinstance(payload, dict)
                else None
            ),
            "original_model_group": json_safe(request_metadata.get("original_model_group")),
            "queue_time_seconds": json_safe(
                _mapping_or_empty(litellm_params.get("metadata")).get("queue_time_seconds")
            ),
            "stream": json_safe(call.get("stream", False)),
            "times": {
                "start_time": _moment(call.get("start_time")),
                "end_time": _moment(call.get("end_time") or datetime.now()),  # noqa: DTZ005  # callback timestamps use the legacy naive contract
                "api_call_start_time": _moment(call.get("api_call_start_time")),
                "completion_start_time": _moment(call.get("completion_start_time")),
            },
            "exception": self._exception(call),
            "settings": _settings(),
        }


def native_companion(
    store: _native.NativePrometheusStore, logger: PrometheusLogger, owns_latency: bool
) -> NativeLogger:
    return native_logger(
        _native.prometheus_callback(store, owns_latency), {Field.deprecated_kwargs: AttemptFacts(logger)}
    )


def _labelnames(kwargs: Mapping[str, object]) -> tuple[str, ...]:
    labelnames: Final = _items(kwargs.get("labelnames") or ())
    return () if labelnames is None else tuple(str(name) for name in labelnames)


def native_metric_factory(
    store: _native.NativePrometheusStore,
    kind: PrometheusKind,
    is_enabled: Callable[[str], bool],
    exclude_labels: frozenset[str],
    limits_enabled: bool,
    shares_overflow_series: bool,
) -> Callable[..., _native.NativePrometheusMetric | NoOpMetric]:
    def factory(
        *args: object,
        **kwargs: object,  # kwargs-ok: mirrors the prometheus_client constructors it replaces
    ) -> _native.NativePrometheusMetric | NoOpMetric:
        name: Final = str(args[0] if args else kwargs.get("name", ""))
        if not is_enabled(name):
            return NoOpMetric()
        documentation: Final = str(args[1] if len(args) > 1 else kwargs.get("documentation", ""))
        original: Final = _labelnames(kwargs)
        kept: Final = tuple(label for label in original if label not in exclude_labels)
        wrapped: Final = bool(original) and (kept != original or limits_enabled)
        mode: Final = str(kwargs.get("multiprocess_mode", "all"))
        if mode not in prometheus_metrics.Gauge._MULTIPROC_MODES:  # pyright: ignore[reportPrivateUsage]  # the client's own mode table
            raise ValueError("Invalid multiprocess mode: " + mode)
        bounds: Final = _items(kwargs.get("buckets", prometheus_metrics.Histogram.DEFAULT_BUCKETS))
        buckets: Final = (
            [] if bounds is None else [float(bound) for bound in bounds]  # pyright: ignore[reportArgumentType]  # prometheus_client calls float() on each bound the same way
        )
        return store.metric(
            kind,
            name,
            documentation,
            list(original),
            [label for label in original if label in exclude_labels],
            buckets if kind == "histogram" else [],
            mode in _MOST_RECENT_MODES,
            wrapped,
            shares_overflow_series,
        )

    return factory


class NativeStoreCollector(Collector):
    def __init__(self, store: _native.NativePrometheusStore) -> None:
        self._store: Final = store

    def describe(self) -> Iterable[Metric]:
        return [Metric(name, documentation, kind) for name, documentation, kind, *_ in self._store.collect()]

    def collect(self) -> Iterable[Metric]:
        created: Final = prometheus_metrics._use_created  # pyright: ignore[reportPrivateUsage]  # the client's own created-series switch
        return [
            _family(name, documentation, kind, labelnames, buckets, series, created)
            for name, documentation, kind, labelnames, buckets, series in self._store.collect()
        ]


def _family(
    name: str,
    documentation: str,
    kind: str,
    labelnames: Sequence[str],
    buckets: Sequence[float],
    series: Sequence[tuple[Sequence[str], float, float, Sequence[float], float]],
    created: bool,
) -> Metric:
    metric: Final = Metric(name, documentation, kind)
    for values, value, total, counts, created_at in series:
        labels = dict(zip(labelnames, values))
        if kind == "counter":
            metric.add_sample(name + "_total", labels, value)
        elif kind == "gauge":
            metric.add_sample(name, labels, value)
        else:
            cumulative = 0.0
            for bound, count in zip(buckets, counts):
                cumulative += count
                metric.add_sample(name + "_bucket", {**labels, "le": floatToGoString(bound)}, cumulative)
            metric.add_sample(name + "_count", labels, cumulative)
            if buckets[0] >= 0:
                metric.add_sample(name + "_sum", labels, total)
        if created and kind != "gauge":
            metric.add_sample(name + "_created", labels, created_at)
    return metric


class NativeStore:
    __slots__ = ("owns_latency", "store")

    def __init__(self, store: _native.NativePrometheusStore) -> None:
        self.store: Final = store
        self.owns_latency = True

    @classmethod
    def open(cls, logger: PrometheusLogger) -> NativeStore:
        limits: Final = logger._series_limits  # pyright: ignore[reportPrivateUsage]  # the logger's resolved caps
        return cls(
            _native.NativePrometheusStore(limits.max_series, limits.ttl_seconds, limits.cleanup_interval_seconds)
        )

    def factory(
        self, metric_class: type[prometheus_metrics.MetricWrapperBase], logger: PrometheusLogger, shares: bool
    ) -> Callable[..., _native.NativePrometheusMetric | NoOpMetric]:
        return native_metric_factory(
            self.store,
            _KINDS[str(metric_class._type)],  # pyright: ignore[reportPrivateUsage]  # the client's own kind name
            logger._is_metric_enabled,  # pyright: ignore[reportPrivateUsage]  # the logger's enable rule
            logger.exclude_labels,
            logger._series_limits.enabled,  # pyright: ignore[reportPrivateUsage]  # the logger's resolved caps
            shares,
        )

    def attach(self, logger: PrometheusLogger) -> NativeLogger:
        from prometheus_client import REGISTRY

        for metric in NATIVE_SUCCESS_METRICS:
            self.store.declare_labels(metric, list(logger.get_labels_for_metric(metric)))  # pyright: ignore[reportArgumentType]  # every name is a defined metric
        REGISTRY.register(NativeStoreCollector(self.store))
        self.owns_latency = not logger._emit_input_sequence_length_label  # pyright: ignore[reportPrivateUsage]  # that label needs the response object, which V2 attempt_ended does not carry
        return native_companion(self.store, logger, self.owns_latency)

    def track_end_user(
        self, metric: str, values: Sequence[object], max_series: int | None, ttl_seconds: float | None
    ) -> None:
        self.store.track_end_user(
            metric,
            [None if value is None else str(value) for value in values],
            max_series,
            ttl_seconds,
            litellm.prometheus_end_user_metrics_cleanup_interval_seconds,
        )


def with_native_companions(callbacks: Sequence[object]) -> list[object]:  # mutable-ok: litellm.callbacks is a list
    def entries(callback: object) -> tuple[object, ...]:
        companion: Final = getattr(callback, "native_companion", None)
        if isinstance(companion, NativeLogger) and companion not in callbacks:
            return (callback, companion)
        return (callback,)

    return [entry for callback in callbacks for entry in entries(callback)]  # comprehension-ok: flattening pairs
