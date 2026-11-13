from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, TypeVar, get_args

from prometheus_client import REGISTRY
from prometheus_client import metrics as prometheus_metrics
from prometheus_client.metrics_core import Metric
from prometheus_client.registry import Collector
from prometheus_client.utils import (
    floatToGoString,  # pyright: ignore[reportUnknownVariableType]  # the client ships no annotations
)
from pydantic import ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.prometheus_helpers import get_cached_end_user_id_for_cost_tracking
from litellm.litellm_core_utils.core_helpers import (
    env_flag,
    get_litellm_metadata_from_kwargs,  # pyright: ignore[reportUnknownVariableType]  # the helper returns an untyped dict
)
from litellm.litellm_core_utils.service_tier_utils import get_service_tier_from_standard_logging_payload
from litellm.rust_bridge import _native
from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS, NoOpMetric, PrometheusMetricLabels

if TYPE_CHECKING:
    from litellm.integrations.prometheus import PrometheusLogger

_Parsed: Final = TypeVar("_Parsed")
PrometheusKind: TypeAlias = Literal["counter", "gauge", "histogram"]
NativeStep: TypeAlias = Literal["success_head", "success_limits", "success_latency", "success_tail", "failure"]
NativeOp: TypeAlias = Literal[
    "deployment_state",
    "deployment_failure",
    "cooled_down",
    "guardrail",
    "managed_batch_created",
    "managed_file_created",
    "managed_file_size",
    "managed_batch_duration",
    "managed_file_deleted",
    "check_batch_cost_run",
    "check_batch_cost_error",
    "callback_failure",
    "provider_budget",
    "spend_capture_rate",
    "dynamic_limiter_share",
    "dynamic_limiter_throttled",
    "fallback",
    "proxy_failure",
    "budget",
    "team_members",
    "counts",
]

NATIVE_PROMETHEUS_ENV: Final = "LITELLM_NATIVE_PROMETHEUS"
NativePrometheusMetric: Final = _native.NativePrometheusMetric

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
    "guardrail_information",
)
_HIDDEN_KEYS: Final = ("additional_headers", "litellm_overhead_time_ms")
_EPOCH_NAIVE: Final = datetime(1970, 1, 1)  # noqa: DTZ001  # callback timestamps use the legacy naive contract
_EPOCH_AWARE: Final = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ONE_MICROSECOND: Final = timedelta(microseconds=1)
_JSON_INTS: Final = (-(2**63), 2**64 - 1)
NUMBER_KEY: Final = "$num"
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
_DEFINED_METRICS: Final = TypeAdapter(tuple[DEFINED_PROMETHEUS_METRICS, ...]).validate_python(
    get_args(DEFINED_PROMETHEUS_METRICS)
)
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
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value if _JSON_INTS[0] <= value <= _JSON_INTS[1] else {NUMBER_KEY: str(value)}
    if isinstance(value, float):
        return value if math.isfinite(value) else {NUMBER_KEY: repr(value)}
    mapping: Final = _parsed(_DICT, value)
    if mapping is not None:
        if all(isinstance(key, str) for key in mapping):
            return {str(key): json_safe(item) for key, item in mapping.items()}
        return {"$py": str(value)}
    items: Final = _items(value)
    if items is not None:
        return [json_safe(item) for item in items]
    return {"$py": str(value)}


def moment(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, datetime):
        return None
    aware: Final = value.utcoffset() is not None
    return {"us": (value - (_EPOCH_AWARE if aware else _EPOCH_NAIVE)) // _ONE_MICROSECOND, "aware": aware}


def wall_moment(value: datetime | None) -> Mapping[str, object] | None:
    return None if value is None else moment(value.replace(tzinfo=None))


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
        else {key: json_safe(hidden_mapping[key]) for key in _HIDDEN_KEYS if key in hidden_mapping}
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


def _hashed_key(key: object) -> object:
    if not isinstance(key, str) or not key.startswith("sk-"):
        return json_safe(key)
    from litellm.proxy.utils import hash_token

    return hash_token(key)


class NativePrometheusError(Exception):
    pass


def attempt_facts(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    payload: Final = kwargs.get("standard_logging_object")
    litellm_params: Final = _mapping_or_empty(kwargs.get("litellm_params"))
    params_metadata: Final = _mapping_or_empty(litellm_params.get("metadata"))
    request_metadata: Final = {
        **_mapping_or_empty(get_litellm_metadata_from_kwargs(dict(kwargs))),  # pyright: ignore[reportUnknownArgumentType]  # validated to a mapping here
        **params_metadata,
    }
    return {
        "payload": _payload(payload),
        "model": json_safe(kwargs.get("model", "")),
        "end_user_id": json_safe(
            get_cached_end_user_id_for_cost_tracking()(dict(litellm_params), service_type="prometheus")
        ),
        "service_tier": json_safe(
            get_service_tier_from_standard_logging_payload(payload)  # pyright: ignore[reportArgumentType]  # the payload is checked to be a dict
            if isinstance(payload, dict)
            else None
        ),
        "original_model_group": json_safe(request_metadata.get("original_model_group")),
        "queue_time_seconds": json_safe(params_metadata.get("queue_time_seconds")),
        "stream": json_safe(kwargs.get("stream", False)),
        "times": {
            "start_time": moment(kwargs.get("start_time")),
            "end_time": moment(kwargs.get("end_time") or datetime.now()),  # noqa: DTZ005  # callback timestamps use the legacy naive contract
            "api_call_start_time": moment(kwargs.get("api_call_start_time")),
            "completion_start_time": moment(kwargs.get("completion_start_time")),
        },
        "key": _key(kwargs, params_metadata, kwargs.get("standard_logging_object")),
        "deployment": _deployment(kwargs, litellm_params),
        "settings": _settings(),
    }


def _key(
    call: Mapping[str, object], request_metadata: Mapping[str, object], payload: object
) -> Mapping[str, object] | None:
    from litellm.proxy.common_utils.callback_utils import (
        get_model_group_from_litellm_kwargs,  # pyright: ignore[reportUnknownVariableType]  # the helper takes untyped kwargs
    )

    parsed: Final = _parsed(_DICT, payload)
    metadata: Final = _parsed(_DICT, parsed.get("metadata")) if parsed is not None else None
    if metadata is None:
        return None
    model_group: Final = get_model_group_from_litellm_kwargs(dict(call))
    return {
        "hashed": _hashed_key(metadata.get("user_api_key_hash")),
        "model_group": json_safe(model_group),
        "remaining_requests": json_safe(request_metadata.get(f"litellm-key-remaining-requests-{model_group}")),
        "remaining_tokens": json_safe(request_metadata.get(f"litellm-key-remaining-tokens-{model_group}")),
    }


def _deployment(call: Mapping[str, object], litellm_params: Mapping[str, object]) -> Mapping[str, object]:
    metadata: Final = _mapping_or_empty(get_litellm_metadata_from_kwargs(dict(call)))  # pyright: ignore[reportUnknownArgumentType]  # validated to a mapping here
    return {
        "model_info": json_safe(metadata.get("model_info")),
        "params_truthy": bool(litellm_params),
        "provider": json_safe(litellm_params.get("custom_llm_provider")),
        "params_tpm": json_safe(litellm_params.get("tpm")),
        "params_rpm": json_safe(litellm_params.get("rpm")),
        "completion_start_present": "completion_start_time" in call,
    }


def record(
    attempt: _native.NativePrometheusAttempt, step: NativeStep, input_sequence_length: str | None = None
) -> None:
    logs, error = attempt.record(step, input_sequence_length)
    for level, message, args in logs:
        (verbose_logger.debug if level == "debug" else verbose_logger.error)(message, *args)
    if error is not None:
        raise NativePrometheusError(error)


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
        metric: Final = store.metric(
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
        REGISTRY.register(NativeFamilyCollector(metric))
        return metric

    return factory


class NativeFamilyCollector(Collector):
    def __init__(self, metric: _native.NativePrometheusMetric) -> None:
        self._metric: Final = metric

    def describe(self) -> Iterable[Metric]:
        name, documentation, kind, *_ = self._metric.collect()
        return [Metric(name, documentation, kind)]

    def collect(self) -> Iterable[Metric]:
        name, documentation, kind, labelnames, buckets, series = self._metric.collect()
        return [
            _family(
                name,
                documentation,
                kind,
                labelnames,
                buckets,
                series,
                prometheus_metrics._use_created,  # pyright: ignore[reportPrivateUsage]  # the client's own created-series switch
            )
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
    __slots__ = ("store",)

    def __init__(self, store: _native.NativePrometheusStore) -> None:
        self.store: Final = store

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

    def attach(self, logger: PrometheusLogger) -> None:
        for metric in _DEFINED_METRICS:
            if hasattr(PrometheusMetricLabels, metric):
                self.store.declare_labels(metric, list(logger.get_labels_for_metric(metric)))

    def attempt(self, kwargs: Mapping[str, object]) -> _native.NativePrometheusAttempt:
        return self.store.attempt(json.dumps(attempt_facts(kwargs)))

    def op(self, name: NativeOp, facts: Mapping[str, object]) -> None:
        self.store.op(
            name, json.dumps({**{key: json_safe(value) for key, value in facts.items()}, "settings": _settings()})
        )

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
