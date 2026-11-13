import copy
import re
import time
from collections.abc import Callable, Generator, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import pytest
from prometheus_client import REGISTRY, CollectorRegistry, generate_latest
from prometheus_client.openmetrics.exposition import generate_latest as generate_openmetrics
from prometheus_client.registry import Collector

import litellm
from litellm.integrations.custom_logger_v2 import Hook
from litellm.integrations.custom_logger_v2.host import get_dispatcher, reset_host_for_tests
from litellm.integrations.custom_logger_v2.legacy import (
    Admit,
    KwargsFields,
    clear_subscribers_cache_for_tests,
    hook_subscribers,
    run_subscribers,
)
from litellm.integrations.custom_logger_v2.offpath import InProcessSink
from litellm.integrations.prometheus import PrometheusLogger
from litellm.integrations.prometheus_helpers.native_store import (
    NATIVE_PROMETHEUS_ENV,
    NativeStoreCollector,
    native_prometheus_enabled,
    with_native_companions,
)
from litellm.litellm_core_utils.litellm_logging import LOGGING_KEYS
from litellm.proxy._types import UserAPIKeyAuth
from litellm.rust_bridge import _native

START: Final = datetime(2026, 11, 13, 9, 0, 0)  # noqa: DTZ001  # callback timestamps use the legacy naive contract
_CREATED: Final = re.compile(r"^(\S+_created(?:\{.*\})?) \S+$", re.MULTILINE)
_SETTINGS: Final = (
    "custom_prometheus_metadata_labels",
    "custom_prometheus_tags",
    "prometheus_emit_stream_label",
    "enable_end_user_cost_tracking_prometheus_only",
    "prometheus_end_user_metrics_max_series_per_metric",
    "prometheus_end_user_metrics_ttl_seconds",
    "prometheus_end_user_metrics_cleanup_interval_seconds",
    "prometheus_exclude_labels",
    "prometheus_exclude_metrics",
    "prometheus_metrics_config",
    "prometheus_metrics_max_series_per_metric",
    "prometheus_deployment_and_latency_caller_identity",
    "prometheus_emit_input_sequence_length_label",
    "prometheus_latency_buckets",
)


import litellm.accounting.metrics  # noqa: E402,F401  # registers its collectors before the snapshot below

_PREEXISTING: Final = frozenset(REGISTRY._collector_to_names)


def _logger_collectors() -> list[Collector]:
    return [collector for collector in list(REGISTRY._collector_to_names) if collector not in _PREEXISTING]


def _clear_registry() -> None:
    for collector in _logger_collectors():
        REGISTRY.unregister(collector)


def _ours() -> CollectorRegistry:
    registry: Final = CollectorRegistry()
    for collector in _logger_collectors():
        registry.register(collector)
    return registry


@pytest.fixture(autouse=True)
def isolated() -> Generator[None]:
    saved: Final = {name: copy.deepcopy(getattr(litellm, name, None)) for name in _SETTINGS}
    reset_host_for_tests()
    clear_subscribers_cache_for_tests()
    _clear_registry()
    yield
    _clear_registry()
    reset_host_for_tests()
    clear_subscribers_cache_for_tests()
    for name, value in saved.items():
        setattr(litellm, name, value)


def _event(
    index: int,
    *,
    user: str = "u",
    end_user: str | None = None,
    stream: bool = False,
    cache_hit: object = False,
    usage_object: object = None,
    tags: list[str] | None = None,
    headers: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    zero_cost: Mapping[str, object] | None = None,
    queue: object = None,
    original_model_group: str | None = None,
    cost: float = 0.0015,
    exception: Exception | None = None,
    aware: bool = False,
    key_hash: str = "hashed",
) -> dict[str, object]:
    base: Final = START.replace(tzinfo=timezone.utc) if aware else START
    start: Final = base + timedelta(seconds=index)
    api_start: Final = start + timedelta(milliseconds=120)
    first_token: Final = api_start + timedelta(milliseconds=333, microseconds=7)
    end: Final = api_start + timedelta(seconds=1, microseconds=123457 * (index + 1))
    payload: Final[dict[str, object]] = {
        "id": f"call-{index}",
        "call_type": "acompletion",
        "status": "success",
        "response_cost": cost,
        "total_tokens": 30 + index,
        "prompt_tokens": 20 + index,
        "completion_tokens": 10,
        "model": "gpt-4o-mini",
        "model_id": f"model-{index % 2}",
        "model_group": "gpt-group",
        "api_base": "https://api.example.com/v1",
        "custom_llm_provider": "openai",
        "request_tags": tags or [],
        "cache_hit": cache_hit,
        "stream": stream,
        "response": None,
        "model_parameters": {},
        "zero_cost_diagnostic": zero_cost,
        "metadata": {
            "user_api_key_hash": key_hash,
            "user_api_key_alias": 'alias "one"\nline',
            "user_api_key_team_id": "team-1",
            "user_api_key_team_alias": "Team One",
            "user_api_key_user_id": user,
            "user_api_key_user_email": f"{user}@example.com",
            "user_api_key_org_id": "org-1",
            "user_api_key_org_alias": "Org",
            "user_api_key_project_id": "p-1",
            "user_api_key_project_alias": None,
            "user_api_key_request_route": "/chat/completions",
            "requester_ip_address": "10.0.0.1",
            "user_agent": "curl/8",
            "user_api_key_end_user_id": end_user,
            "usage_object": usage_object,
            **(metadata or {}),
        },
        "hidden_params": {"litellm_overhead_time_ms": None, "additional_headers": headers},
    }
    litellm_metadata: Final[dict[str, object]] = {"user_api_key_end_user_id": end_user}
    if queue is not None:
        litellm_metadata["queue_time_seconds"] = queue
    if original_model_group is not None:
        litellm_metadata["original_model_group"] = original_model_group
    kwargs: Final[dict[str, object]] = {
        "model": "gpt-4o-mini",
        "standard_logging_object": payload,
        "litellm_params": {"metadata": litellm_metadata, "user_api_key_end_user_id": end_user},
        "start_time": start,
        "end_time": end,
        "api_call_start_time": api_start,
        "completion_start_time": first_token,
        "stream": stream,
        "log_event_type": "successful_api_call",
    }
    if exception is not None:
        kwargs["exception"] = exception
    return kwargs


V3_HEADERS: Final = {
    "x-ratelimit-api_key-limit-requests": 100,
    "x-ratelimit-api_key-remaining-requests": 40,
    "x-ratelimit-team-limit-tokens": 1000,
    "x-ratelimit-team-remaining-tokens": 250,
    "x-ratelimit-model_per_project_itpm-limit-tokens": 500,
    "x-ratelimit-model_per_project_itpm-remaining-tokens": 499,
    "x-ratelimit-api_key-limit-tokens": True,
    "x-ratelimit-api_key-remaining-tokens": 1,
}

CORPUS: Final = (
    _event(0),
    _event(1, stream=True, cache_hit=True, tags=["prod", "User-Agent: curl/8.1"], queue=0.25),
    _event(
        2,
        user="v",
        end_user="customer-a",
        cache_hit=None,
        usage_object={
            "prompt_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 7, "audio_tokens": 2, "cache_creation_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 5, "audio_tokens": 0},
            "duration_seconds": 4.5,
            "output_image_count": 2,
        },
        headers=V3_HEADERS,
        metadata={
            "requester_metadata": {"team_name": "core", "nested": {"tier": "gold"}},
            "spend_logs_metadata": {"app": "billing"},
            "mcp_tool_call_metadata": {"name": "search", "mcp_server_name": "docs"},
        },
        original_model_group="gpt-requested",
    ),
    _event(
        3,
        end_user="customer-b",
        usage_object={
            "cache_read_input_tokens": 11,
            "cache_creation_input_tokens": None,
            "prompt_tokens_details": {"cache_write_tokens": 4},
        },
        zero_cost={"reason": "missing_pricing_key"},
        cost=0.0,
        metadata={"mcp_tool_call_metadata": {"name": "free", "mcp_server_name": "docs"}},
    ),
    _event(4, end_user="customer-a", headers={"x-ratelimit-api_key-limit-requests": 100}, aware=True, queue=-1),
    _event(5, exception=Exception("Authentication Error: invalid api key")),
    _event(6, key_hash="sk-raw-key", stream=True, tags=["dev"], end_user="customer-c", queue=1),
    _event(7, user="w", cache_hit="unknown", end_user="customer-d"),
)


async def _drained() -> None:
    sink: Final = get_dispatcher().sink
    assert isinstance(sink, InProcessSink)
    await sink.drain()


async def _drive(native: bool, monkeypatch: pytest.MonkeyPatch) -> str:
    if native:
        monkeypatch.setenv(NATIVE_PROMETHEUS_ENV, "true")
    else:
        monkeypatch.delenv(NATIVE_PROMETHEUS_ENV, raising=False)
    logger: Final = PrometheusLogger()
    assert (logger.native_companion is not None) is native if hasattr(logger, "native_companion") else not native
    for kwargs in copy.deepcopy(CORPUS):
        await logger.async_log_success_event(kwargs, None, kwargs["start_time"], kwargs["end_time"])
        if native:
            _ = await run_subscribers(
                hook_subscribers(Hook.attempt_ended, [logger.native_companion], admit=Admit.every_entry),
                KwargsFields(kwargs, LOGGING_KEYS),
                lambda _callback, _kwargs: None,
            )
            await _drained()
    outcomes: Final = await _direct_calls(logger)
    return generate_latest(_ours()).decode() + "\n".join(outcomes)


async def _direct_calls(logger: PrometheusLogger) -> list[str]:
    failure: Final = _event(9)
    failure["log_event_type"] = "failed_api_call"
    failure["exception"] = litellm.RateLimitError(message="slow down", llm_provider="openai", model="gpt-4o-mini")
    failure["standard_logging_object"]["status"] = "failure"
    calls: Final = (
        lambda: logger.set_deployment_complete_outage("gpt-4o-mini", "model-0", "https://api.example.com/v1", "openai"),
        lambda: logger.increment_deployment_cooled_down(
            "gpt-4o-mini", "model-0", "https://api.example.com/v1", "openai", "429"
        ),
        lambda: logger._record_guardrail_metrics("pii", 0.42, "success", None, "pre_call"),
        lambda: logger._record_guardrail_metrics("pii", 1.5, "error", "Timeout", "pre_call"),
        lambda: logger.increment_callback_logging_failure("s3_v2"),
        lambda: logger.track_provider_remaining_budget("openai", 3.0, 10.0),
        lambda: logger.set_spend_capture_rate("openai", None),
        lambda: logger.set_spend_capture_rate("anthropic", 0.97),
        lambda: logger.set_dynamic_limiter_share("gpt-group", "team-1", "Team One", "high", 1200, 30),
        lambda: logger.increment_dynamic_limiter_throttled("gpt-group", "team-1", None, "high", "tokens", "team"),
    )
    outcomes: Final = [_outcome(call) for call in calls]
    await logger.async_log_failure_event(failure, None, failure["start_time"], failure["end_time"])
    await logger.async_post_call_failure_hook(
        {"model": "gpt-group", "metadata": {}},
        litellm.RateLimitError(message="slow down", llm_provider="openai", model="gpt-4o-mini"),
        UserAPIKeyAuth(api_key="hashed", user_id="u", team_id="team-1"),
    )
    return outcomes


def _outcome(call: Callable[[], object]) -> str:
    try:
        call()
    except Exception as error:  # noqa: BLE001  # parity compares whatever each side raises
        return f"{type(error).__name__}: {error}"
    return "ok"


def _masked(text: str) -> str:
    return _CREATED.sub(lambda match: f"{match.group(1)} <created>", text)


CONFIGS: Final[Mapping[str, Mapping[str, object]]] = {
    "defaults": {},
    "custom_labels_tags_stream_end_user": {
        "custom_prometheus_metadata_labels": ["metadata.team_name", "nested.tier", "app", "user_api_key_team_alias"],
        "custom_prometheus_tags": ["prod", "User-Agent: curl/*", "absent"],
        "prometheus_emit_stream_label": True,
        "enable_end_user_cost_tracking_prometheus_only": True,
        "prometheus_end_user_metrics_max_series_per_metric": 2,
    },
    "end_user_ttl": {
        "enable_end_user_cost_tracking_prometheus_only": True,
        "prometheus_end_user_metrics_ttl_seconds": 0.000001,
        "prometheus_end_user_metrics_cleanup_interval_seconds": 0,
    },
    "excludes_include_labels_and_series_cap": {
        "prometheus_exclude_labels": ["user_agent", "client_ip"],
        "prometheus_exclude_metrics": ["litellm_cache_misses_metric"],
        "prometheus_metrics_config": [
            {
                "group": "tokens",
                "metrics": ["litellm_total_tokens_metric", "litellm_spend_metric"],
                "include_labels": ["team", "model_id", "end_user"],
            },
            {
                "group": "rest",
                "metrics": [
                    "litellm_requests_metric",
                    "litellm_proxy_total_requests_metric",
                    "litellm_api_key_rate_limit_allowed_metric",
                    "litellm_api_key_rate_limit_used_metric",
                    "litellm_llm_api_latency_metric",
                    "litellm_mcp_tool_calls_total",
                    "litellm_deployment_state",
                    "litellm_guardrail_requests_total",
                ],
            },
        ],
        "prometheus_metrics_max_series_per_metric": 2,
        "enable_end_user_cost_tracking_prometheus_only": True,
    },
    "caller_identity_both_and_buckets": {
        "prometheus_deployment_and_latency_caller_identity": "both",
        "prometheus_latency_buckets": [0.1, 1, 2.5, 600],
    },
    "input_sequence_length_stays_in_python": {"prometheus_emit_input_sequence_length_label": True},
}


@pytest.mark.parametrize("config", list(CONFIGS))
async def test_the_native_store_and_handler_expose_exactly_what_the_python_logger_does(
    config: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in CONFIGS[config].items():
        setattr(litellm, name, copy.deepcopy(value))
    python_text: Final = await _drive(False, monkeypatch)
    _clear_registry()
    reset_host_for_tests()
    clear_subscribers_cache_for_tests()
    native_text: Final = await _drive(True, monkeypatch)

    assert "litellm_spend_metric" in python_text or config.startswith("excludes")
    assert _masked(native_text) == _masked(python_text)


async def test_openmetrics_output_matches_too(monkeypatch: pytest.MonkeyPatch) -> None:
    def openmetrics(native: bool) -> str:
        return _masked(generate_openmetrics(_ours()).decode())

    _ = await _drive(False, monkeypatch)
    python_text: Final = openmetrics(False)
    _clear_registry()
    reset_host_for_tests()
    _ = await _drive(True, monkeypatch)
    assert openmetrics(True) == python_text


async def test_the_rust_encoder_writes_the_python_client_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _ = await _drive(True, monkeypatch)
    collector: Final = next(c for c in REGISTRY._collector_to_names if isinstance(c, NativeStoreCollector))
    store: Final = collector._store
    for created in (True, False):
        monkeypatch.setattr("prometheus_client.metrics._use_created", created)
        rendered = "".join(line for line in generate_latest(_ours()).decode().splitlines(keepends=True))
        assert _masked(store.encode_text(created)) == _masked(rendered)


def test_a_counter_handle_keeps_the_prometheus_client_errors() -> None:
    store: Final = _native.NativePrometheusStore()
    counter: Final = store.metric("counter", "c_total", "doc", ["a"], [], [], False, False, True)
    with pytest.raises(ValueError, match="Incorrect label names"):
        counter.labels(b="1")
    with pytest.raises(ValueError, match="non-negative"):
        counter.labels(a="1").inc(-1)
    with pytest.raises(ValueError, match="missing label values"):
        counter.inc()
    with pytest.raises(KeyError):
        counter.remove("absent")
    gauge: Final = store.metric("gauge", "g", "doc", ["a"], [], [], True, False, False)
    with pytest.raises(RuntimeError, match="mostrecent"):
        gauge.labels("x").inc()
    assert 'c_total{a="1"} 0.0' in store.encode_text(False)
    time.sleep(0)


async def test_the_native_companion_sits_right_after_the_legacy_logger_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NATIVE_PROMETHEUS_ENV, "true")
    logger: Final = PrometheusLogger()
    callbacks: Final = with_native_companions(["before", logger, "after"])

    assert callbacks == ["before", logger, logger.native_companion, "after"]
    assert with_native_companions(callbacks) == callbacks


def test_multiprocess_mode_keeps_the_python_logger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(NATIVE_PROMETHEUS_ENV, "true")
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))

    assert not native_prometheus_enabled()
