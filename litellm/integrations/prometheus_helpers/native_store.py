from __future__ import annotations

import itertools
import json
import math
import os
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar, Token
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, TypeVar, get_args

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
from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS, NoOpMetric, PrometheusMetricLabels

if TYPE_CHECKING:
    from litellm.integrations.prometheus import PrometheusLogger

_Parsed: Final = TypeVar("_Parsed")
PrometheusKind: TypeAlias = Literal["counter", "gauge", "histogram"]

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
_PENDING_LIMIT: Final = 4096
_Pending: TypeAlias = dict[tuple[object, bool], deque[dict[str, object]]]
_TOKENS: Final = itertools.count(1)
_BUDGET_LOOKUPS: Final = {"api_key": "key", "team": "team", "user": "user", "org": "org", "customer": "customer"}
_Deferred: TypeAlias = list[Mapping[str, object]]
_DEFERRING: Final[ContextVar[_Deferred | None]] = ContextVar("prometheus_native_deferring", default=None)
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


def moment(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, datetime):
        return None
    aware: Final = value.utcoffset() is not None
    return {"us": (value - (_EPOCH_AWARE if aware else _EPOCH_NAIVE)) // _ONE_MICROSECOND, "aware": aware}


def wall_moment(value: datetime | None) -> Mapping[str, object] | None:
    return None if value is None else moment(value.replace(tzinfo=None))


def _pending_key(kwargs: Mapping[str, object]) -> tuple[object, bool]:
    return (json_safe(kwargs.get("litellm_call_id")), kwargs.get("log_event_type") == "failed_api_call")


def budget_lookup_failed(name: str, facts: Mapping[str, object]) -> Mapping[str, object] | None:
    lookup: Final = _BUDGET_LOOKUPS.get(str(facts.get("entity"))) if name == "budget" else None
    if lookup is None:
        return None
    return {
        "level": "debug",
        "format": "[Non-Blocking] Prometheus: Budget metric lookup %s failed: %s",
        "args": [lookup],
    }


def no_log(_name: str, _facts: Mapping[str, object]) -> Mapping[str, object] | None:
    return None


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
        store: Final = self._logger._native_store  # pyright: ignore[reportPrivateUsage]  # the logger owns its store
        pending: Final = None if store is None else store.claim(call)
        if pending is None:
            return {"event": "skipped", "settings": _settings()}
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
                "start_time": moment(call.get("start_time")),
                "end_time": moment(call.get("end_time") or datetime.now()),  # noqa: DTZ005  # callback timestamps use the legacy naive contract
                "api_call_start_time": moment(call.get("api_call_start_time")),
                "completion_start_time": moment(call.get("completion_start_time")),
            },
            "exception": self._exception(call),
            "key": self._key(call, litellm_params, call.get("standard_logging_object")),
            "deployment": self._deployment(call, litellm_params),
            "deployment_failure": self._deployment_failure(call),
            "token": next(_TOKENS),
            **pending,
            "settings": _settings(),
        }

    def _key(
        self, call: Mapping[str, object], litellm_params: Mapping[str, object], payload: object
    ) -> Mapping[str, object] | None:
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_litellm_kwargs,  # pyright: ignore[reportUnknownVariableType]  # the helper takes untyped kwargs
        )

        parsed: Final = _parsed(_DICT, payload)
        metadata: Final = _parsed(_DICT, parsed.get("metadata")) if parsed is not None else None
        if metadata is None:
            return None
        hashed: Final = metadata.get("user_api_key_hash")
        model_group: Final = get_model_group_from_litellm_kwargs(dict(call))
        request_metadata: Final = _mapping_or_empty(litellm_params.get("metadata"))
        return {
            "hashed": _hashed_key(hashed),
            "model_group": json_safe(model_group),
            "remaining_requests": json_safe(request_metadata.get(f"litellm-key-remaining-requests-{model_group}")),
            "remaining_tokens": json_safe(request_metadata.get(f"litellm-key-remaining-tokens-{model_group}")),
        }

    def _deployment(self, call: Mapping[str, object], litellm_params: Mapping[str, object]) -> Mapping[str, object]:
        metadata: Final = _mapping_or_empty(get_litellm_metadata_from_kwargs(dict(call)))  # pyright: ignore[reportUnknownArgumentType]  # validated to a mapping here
        return {
            "model_info": json_safe(metadata.get("model_info")),
            "params_truthy": bool(litellm_params),
            "provider": json_safe(litellm_params.get("custom_llm_provider")),
            "params_tpm": json_safe(litellm_params.get("tpm")),
            "params_rpm": json_safe(litellm_params.get("rpm")),
            "completion_start_present": "completion_start_time" in call,
        }

    def _deployment_failure(self, call: Mapping[str, object]) -> Mapping[str, object] | None:
        if call.get("log_event_type") != "failed_api_call":
            return None
        try:
            labels: Final = self._logger.deployment_failure_labels(dict(call))  # pyright: ignore[reportUnknownMemberType]  # the legacy logging kwargs are untyped
        except Exception as error:  # noqa: BLE001  # the legacy helper logs and drops the same failures
            verbose_logger.debug("Prometheus Error: set_llm_deployment_failure_metrics. Exception occured - %s", error)
            return None
        if labels is None:
            return None
        return {
            "values": json_safe(labels.enum_values.model_dump()),
            "selected": labels.deployment_selected,
            "litellm_model_name": labels.litellm_model_name,
            "model_id": json_safe(labels.model_id),
            "api_base": json_safe(labels.api_base),
            "provider": labels.api_provider,
            "has_exception": labels.has_exception,
        }


class _LoggedNative:
    __slots__ = ("native", "store")

    def __init__(self, native: _native._NativeCallback, store: _native.NativePrometheusStore) -> None:  # pyright: ignore[reportPrivateUsage]  # the stub's protocol for exported callbacks
        self.native: Final = native
        self.store: Final = store

    def manifest(self) -> str:
        return self.native.manifest()

    def request_state(self) -> object:
        return self.native.request_state()

    def handle(
        self, hook: str, fields: Mapping[str, object], call: Mapping[str, object] | None, state: object, /
    ) -> Awaitable[object]:
        return self._logged(self.native.handle(hook, fields, call, state), fields)

    async def _logged(self, running: Awaitable[object], fields: Mapping[str, object]) -> object:
        try:
            return await running
        finally:
            token: Final = _mapping_or_empty(fields.get(Field.deprecated_kwargs.value)).get("token")
            if isinstance(token, int):
                for level, message, args in self.store.take_logs(token):
                    (verbose_logger.debug if level == "debug" else verbose_logger.error)(message, *args)


def native_companion(store: _native.NativePrometheusStore, logger: PrometheusLogger) -> NativeLogger:
    return native_logger(
        _LoggedNative(_native.prometheus_callback(store, True), store),
        {Field.deprecated_kwargs: AttemptFacts(logger)},
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


class _Deferring:
    __slots__ = ("deferred", "entry", "on_error", "reset")

    def __init__(
        self,
        entry: dict[str, object],  # mutable-ok: the legacy handler fills it in
        on_error: Callable[[str, Mapping[str, object]], Mapping[str, object] | None],
    ) -> None:
        self.entry: Final = entry
        self.on_error: Final = on_error
        self.deferred: Final[_Deferred] = []
        self.reset: Token[_Deferred | None] | None = None

    def __enter__(self) -> None:
        self.reset = _DEFERRING.set(self.deferred)

    def __exit__(self, *_exc: object) -> None:
        if self.reset is not None:
            _DEFERRING.reset(self.reset)
        self.entry["deferred"] = [
            {**item, "on_error": self.on_error(str(item["op"]), _mapping_or_empty(item["facts"]))}
            for item in self.deferred
        ]


class NativeStore:
    __slots__ = ("pending", "store")

    def __init__(self, store: _native.NativePrometheusStore) -> None:
        self.store: Final = store
        self.pending: Final[_Pending] = {}

    def expect(self, kwargs: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: the legacy handler fills it in
        entry: Final[dict[str, object]] = {"deferred": []}  # mutable-ok: the legacy handler fills it in
        self.pending.setdefault(_pending_key(kwargs), deque()).append(entry)
        while len(self.pending) > _PENDING_LIMIT:
            del self.pending[next(iter(self.pending))]
        return entry

    def claim(self, kwargs: Mapping[str, object]) -> dict[str, object] | None:  # mutable-ok: handed over once
        key: Final = _pending_key(kwargs)
        waiting: Final = self.pending.get(key)
        if not waiting:
            return None
        entry: Final = waiting.popleft()
        if not waiting:
            del self.pending[key]
        return entry

    def deferring(
        self,
        entry: dict[str, object],  # mutable-ok: the legacy handler fills it in
        on_error: Callable[[str, Mapping[str, object]], Mapping[str, object] | None],
    ) -> _Deferring:
        return _Deferring(entry, on_error)

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

        for metric in _DEFINED_METRICS:
            if hasattr(PrometheusMetricLabels, metric):
                self.store.declare_labels(metric, list(logger.get_labels_for_metric(metric)))
        REGISTRY.register(NativeStoreCollector(self.store))
        return native_companion(self.store, logger)

    def op(self, name: str, facts: Mapping[str, object]) -> None:
        deferred: Final = _DEFERRING.get()
        if deferred is not None:
            deferred.append({"op": name, "facts": {key: json_safe(value) for key, value in facts.items()}})
            return
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


def with_native_companions(callbacks: Sequence[object]) -> list[object]:  # mutable-ok: litellm.callbacks is a list
    def entries(callback: object) -> tuple[object, ...]:
        companion: Final = getattr(callback, "native_companion", None)
        if isinstance(companion, NativeLogger) and companion not in callbacks:
            return (callback, companion)
        return (callback,)

    return [entry for callback in callbacks for entry in entries(callback)]  # comprehension-ok: flattening pairs
