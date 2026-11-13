import asyncio
import copy
import logging
import re
from collections.abc import Callable, Generator, Mapping
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import pytest
from prometheus_client import REGISTRY, CollectorRegistry, generate_latest
from prometheus_client.openmetrics.exposition import generate_latest as generate_openmetrics
from prometheus_client.registry import Collector

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.prometheus import PrometheusLogger
from litellm.integrations.prometheus_helpers.native_store import (
    NATIVE_PROMETHEUS_ENV,
    NativeFamilyCollector,
    native_prometheus_enabled,
)
from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, Member, UserAPIKeyAuth
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
    _clear_registry()
    yield
    _clear_registry()
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
    params: Mapping[str, object] | None = None,
    request_metadata: Mapping[str, object] | None = None,
    hidden: Mapping[str, object] | None = None,
    guardrails: object = None,
    tokens: Mapping[str, object] | None = None,
    response: object = None,
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
        "hidden_params": {"litellm_overhead_time_ms": None, "additional_headers": headers, **(hidden or {})},
        "guardrail_information": guardrails,
        **(tokens or {}),
    }
    litellm_metadata: Final[dict[str, object]] = {"user_api_key_end_user_id": end_user}
    if queue is not None:
        litellm_metadata["queue_time_seconds"] = queue
    if original_model_group is not None:
        litellm_metadata["original_model_group"] = original_model_group
    litellm_metadata.update(request_metadata or {})
    kwargs: Final[dict[str, object]] = {
        "model": "gpt-4o-mini",
        "standard_logging_object": payload,
        "litellm_params": {"metadata": litellm_metadata, "user_api_key_end_user_id": end_user, **(params or {})},
        "start_time": start,
        "end_time": end,
        "api_call_start_time": api_start,
        "completion_start_time": first_token,
        "stream": stream,
        "log_event_type": "successful_api_call",
    }
    if exception is not None:
        kwargs["exception"] = exception
    if response is not None:
        kwargs["_response"] = response
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
    _event(
        8,
        stream=True,
        params={"custom_llm_provider": "openai", "tpm": 9000},
        request_metadata={
            "model_group": "gpt-group",
            "model_info": {"id": "model-0", "rpm": 60, "tpm": 0},
            "litellm-key-remaining-requests-gpt-group": 5,
            "litellm-key-remaining-tokens-gpt-group": "250",
        },
        hidden={
            "litellm_overhead_time_ms": 12.5,
            "additional_headers": {"x_ratelimit_remaining_requests": " 77 ", "x_ratelimit_remaining_tokens": 1000},
        },
        guardrails=[
            {"guardrail_mode": "pre_call", "duration": 0.2},
            {"guardrail_mode": ["pre_call", "during_call"], "duration": 5},
            {"guardrail_mode": ["post_call"], "duration": "0.05"},
        ],
    ),
    _event(
        9,
        params={"custom_llm_provider": "azure"},
        hidden={
            "litellm_overhead_time_ms": 0,
            "additional_headers": {"x-ratelimit-model_per_key-remaining-tokens": 17},
        },
        guardrails={"guardrail_mode": "post_call", "duration": 0.4},
    ),
    _event(10, tokens={"prompt_tokens": 0}, response={"usage": {"prompt_tokens": 1500, "completion_tokens": 3}}),
    _event(11, tokens={"completion_tokens": "ten"}, metadata={"user_api_key_team_id": "team-raise"}),
    _event(12, hidden={"litellm_overhead_time_ms": "slow"}, metadata={"user_api_key_team_id": "team-overhead"}),
    _event(
        13, cost=float("nan"), tokens={"completion_tokens": float("inf")}, metadata={"user_api_key_team_id": 10**20}
    ),
    _event(14, queue="slow", metadata={"user_api_key_team_id": "team-queue"}),
    _event(15, tokens={"completion_tokens": [10]}, metadata={"user_api_key_team_id": "team-list"}),
)


async def _drive(native: bool, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr("time.time", lambda: 1_763_000_000.25)
    monkeypatch.setattr("litellm.integrations.prometheus.datetime", _FixedDatetime)
    if native:
        monkeypatch.setenv(NATIVE_PROMETHEUS_ENV, "true")
    else:
        monkeypatch.delenv(NATIVE_PROMETHEUS_ENV, raising=False)
    logger: Final = PrometheusLogger()
    assert (logger._native_store is not None) is native
    lines: Final = _Lines()
    verbose_logger.addHandler(lines)
    try:
        text: Final = await _driven(logger)
    finally:
        verbose_logger.removeHandler(lines)
    return text + "\n".join(lines.seen)


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.seen: Final[list[str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        message: Final = record.getMessage()
        if _LOGGED.match(message):
            self.seen.append(f"{record.levelname} {message}")


_LOGGED: Final = re.compile(
    r"(Prometheus Error|prometheus Layer Error|\[Non-Blocking\] Prometheus: Budget|Error recording)"
)


async def _success_outcome(logger: PrometheusLogger, kwargs: dict[str, object]) -> str:
    response: Final = kwargs.pop("_response", None)
    try:
        await logger.async_log_success_event(kwargs, response, kwargs["start_time"], kwargs["end_time"])
    except Exception as error:  # noqa: BLE001  # parity compares what each side raises to the logging caller
        return f"success raised: {type(error).__name__}: {error}"
    return "success ok"


async def _driven(logger: PrometheusLogger) -> str:
    raised: Final = [await _success_outcome(logger, kwargs) for kwargs in copy.deepcopy(CORPUS)]
    outcomes: Final = await _direct_calls(logger)
    return generate_latest(_ours()).decode() + "\n".join(raised + outcomes) + "\n"


async def _direct_calls(logger: PrometheusLogger) -> list[str]:
    failure: Final = _failure(
        10, litellm.RateLimitError(message="slow down", llm_provider="openai", model="gpt-4o-mini")
    )
    rejected: Final = _failure(11, litellm.BadRequestError(message="bad", model="gpt-4o-mini", llm_provider="openai"))
    rejected["standard_logging_object"]["model_id"] = None
    rejected["standard_logging_object"]["model_group"] = None
    rejected["litellm_params"]["metadata"]["model_group"] = "gpt-group"
    rejected["standard_logging_object"]["zero_cost_diagnostic"] = {"reason": "failed_call"}
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
        lambda: logger.record_managed_batch_created("gpt-4o-mini", "openai", "u", None, "alias"),
        lambda: logger.record_managed_file_created("gpt-4o-mini", None, "u", "u@example.com", None),
        lambda: logger.record_managed_file_size(2048, "batch", "jsonl", None, "openai", "u"),
        lambda: logger.record_managed_batch_duration(12.5, "gpt-4o-mini"),
        lambda: logger.record_managed_file_deleted("blocked"),
        lambda: logger.record_check_batch_cost_run(3, [("gpt-4o-mini", "openai"), (None, None)]),
        lambda: logger.record_check_batch_cost_error("timeout"),
        lambda: logger.set_deployment_healthy("gpt-4o-mini", "model-1", "https://api.example.com/v1", "openai"),
        lambda: logger._record_guardrail_metrics("pii", "slow", "success", None, "pre_call"),  # pyright: ignore[reportArgumentType]  # callers pass untyped values at runtime
        lambda: logger.record_managed_file_size("big", "batch", "jsonl", None, "openai", "u"),  # pyright: ignore[reportArgumentType]  # callers pass untyped values at runtime
        lambda: logger.record_managed_batch_duration("long", "gpt-4o-mini"),  # pyright: ignore[reportArgumentType]  # callers pass untyped values at runtime
        lambda: logger.record_check_batch_cost_run("many", []),  # pyright: ignore[reportArgumentType]  # callers pass untyped values at runtime
        lambda: logger._record_guardrail_metrics("nan-guard", float("nan"), "success", None, "pre_call"),
        lambda: logger.track_provider_remaining_budget("nan-provider", float("nan"), float("inf")),
        lambda: logger.record_managed_file_created("gpt-4o-mini", None, 10**20, None, None),  # pyright: ignore[reportArgumentType]  # callers pass untyped values at runtime
    )
    outcomes: Final = [_outcome(call) for call in calls]
    broken: Final = _failure(
        13,
        litellm.RateLimitError(message="slow", llm_provider="openai", model="gpt-4o-mini"),
        zero_cost={"note": "no reason"},
        metadata={"user_api_key_team_id": "team-broken"},
    )
    for event in (failure, rejected, broken):
        await logger.async_log_failure_event(event, None, event["start_time"], event["end_time"])
    original: Final = litellm.RateLimitError(message="slow", llm_provider="openai", model="gpt-4o-mini")
    fallback_kwargs: Final = {
        "model": "gpt-4o",
        "metadata": {"user_api_key_hash": "hashed", "user_api_key_team_id": "team-1"},
        "tags": ["prod"],
    }
    await logger.log_success_fallback_event("gpt-group", fallback_kwargs, original)
    await logger.log_failure_fallback_event("gpt-group", fallback_kwargs, original)
    await logger.async_post_call_failure_hook(
        {"model": "gpt-group", "metadata": {}},
        litellm.RateLimitError(message="slow down", llm_provider="openai", model="gpt-4o-mini"),
        UserAPIKeyAuth(api_key="hashed", user_id="u", team_id="team-1"),
    )
    return outcomes + _budget_calls(logger) + [await _counts(logger)]


class _FixedClock(type):
    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, datetime)


class _FixedDatetime(datetime, metaclass=_FixedClock):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        if tz is None:
            return START
        return datetime(2026, 11, 13, 9, 0, 0, tzinfo=timezone.utc).astimezone(tz)


def _budget_calls(logger: PrometheusLogger) -> list[str]:
    aware_reset: Final = datetime(2026, 11, 20, 0, 0, 0, tzinfo=timezone.utc)
    shifted_reset: Final = datetime(2026, 11, 14, 3, 30, 0, tzinfo=timezone(timedelta(hours=5)))
    naive_reset: Final = START + timedelta(hours=7, seconds=1)
    dst_reset: Final = datetime(2027, 3, 20, 0, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    team: Final = LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="Team One",
        spend=4.5,
        max_budget=10.0,
        budget_reset_at=aware_reset,
        members_with_roles=[Member(user_id="u", role="admin"), Member(user_id="v", role="user")],
    )
    calls: Final = (
        lambda: logger._set_team_budget_metrics(team),
        lambda: logger._set_team_budget_metrics(LiteLLM_TeamTable(team_id="team-2", spend=None)),
        lambda: logger.set_team_members_metric(team),
        lambda: logger._set_org_budget_metrics("org-1", "", 2.0, None, naive_reset),
        lambda: logger._set_org_budget_metrics("org-2", "Org", 1.0, 3.0, shifted_reset),
        lambda: logger._set_key_budget_metrics(
            UserAPIKeyAuth(token="hashed-1", key_alias="k", spend=1.25, max_budget=None, budget_reset_at=naive_reset)
        ),
        lambda: logger._set_key_budget_metrics(
            UserAPIKeyAuth(token="hashed-2", spend=0.5, max_budget=2.0, budget_reset_at=aware_reset)
        ),
        lambda: logger._set_user_budget_metrics(
            LiteLLM_UserTable(
                user_id="u", user_email="u@example.com", spend=3.0, max_budget=5.0, budget_reset_at=aware_reset
            )
        ),
        lambda: logger._set_user_budget_metrics(LiteLLM_UserTable(user_id="v", spend=0.0)),
        lambda: logger._set_customer_budget_metrics("cust-1", 1.0, 4.0, aware_reset),
        lambda: logger._set_customer_budget_metrics("cust-2", 2.0, None, None),
        lambda: logger._set_customer_budget_metrics("cust-3", 0.5, 1.0, naive_reset),
        lambda: logger._set_org_budget_metrics("org-dst", "Dst", 1.0, 2.0, dst_reset),
        lambda: logger._set_key_budget_metrics(
            UserAPIKeyAuth(token="hashed-dst", spend=0.0, max_budget=1.0, budget_reset_at=dst_reset)
        ),
    )
    return [_outcome(call) for call in calls]


class _Counted:
    def __init__(self, repository: object) -> None:
        self.repository = repository

    async def count(self) -> int:
        return 7 if isinstance(self.repository, _Users) else 3


class _Users:
    def __init__(self, _prisma: object) -> None:
        pass

    async def count_billable_users(self) -> int:
        return 5


class _Teams:
    def __init__(self, _prisma: object) -> None:
        pass


async def _counts(logger: PrometheusLogger) -> str:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("litellm.proxy.proxy_server.prisma_client", object())
        patch.setattr("litellm.integrations.prometheus._paginated_table", _Counted)
        patch.setattr("litellm.integrations.prometheus.UserRepository", _Users)
        patch.setattr("litellm.integrations.prometheus.TeamRepository", _Teams)
        await logger._initialize_user_and_team_count_metrics()
    return "counted"


def _failure(
    index: int,
    exception: Exception,
    zero_cost: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    event: Final = _event(index, zero_cost=zero_cost, metadata=metadata)
    event["log_event_type"] = "failed_api_call"
    event["exception"] = exception
    event["standard_logging_object"]["status"] = "failure"
    return event


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
    "input_sequence_length": {"prometheus_emit_input_sequence_length_label": True},
}


@pytest.mark.parametrize("config", list(CONFIGS))
async def test_the_native_store_and_handler_expose_exactly_what_the_python_logger_does(
    config: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in CONFIGS[config].items():
        setattr(litellm, name, copy.deepcopy(value))
    python_text: Final = await _drive(False, monkeypatch)
    _clear_registry()
    native_text: Final = await _drive(True, monkeypatch)

    assert "litellm_spend_metric" in python_text or config.startswith("excludes")
    assert _masked(native_text) == _masked(python_text)
    assert ('input_sequence_length="1k-4k"' in python_text) is (config == "input_sequence_length")
    if config == "defaults":
        assert "ERROR prometheus Layer Error(): Exception occured - 'reason'" in python_text
        assert "success raised: ValueError: could not convert string to float: 'ten'" in python_text
        assert "success raised: TypeError: '>=' not supported between instances of 'str' and 'int'" in python_text
        assert "TypeError: float() argument must be a string or a real number, not 'list'" in python_text
        assert "ERROR Prometheus Error: set_llm_deployment_success_metrics. Exception occured" in python_text
        assert 'team="team-raise"' in python_text
        assert 'litellm_remaining_team_budget_metric{team="team-raise"' not in python_text
        assert 'org_id="org-dst"} 3044.0' in python_text


async def _openmetrics(native: bool, monkeypatch: pytest.MonkeyPatch) -> str:
    _ = await _drive(native, monkeypatch)
    text: Final = _masked(generate_openmetrics(_ours()).decode())
    _clear_registry()
    return text


async def test_openmetrics_output_matches_too(monkeypatch: pytest.MonkeyPatch) -> None:
    python_text: Final = await _openmetrics(False, monkeypatch)
    assert await _openmetrics(True, monkeypatch) == python_text


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
    assert counter.collect()[5] == [(["1"], 0.0, 0.0, [], counter.collect()[5][0][4])]


_CALLBACK_LISTS: Final = (
    "callbacks",
    "input_callback",
    "success_callback",
    "failure_callback",
    "_async_input_callback",
    "_async_success_callback",
    "_async_failure_callback",
)
_TIMED: Final = re.compile(r"latency|seconds|duration|_time|_created|overhead")


_REGISTRATIONS: Final = {
    "callbacks": ("callbacks",),
    "success_and_failure_callback": ("success_callback", "failure_callback"),
}


async def _completion_text(native: bool, registration: str, monkeypatch: pytest.MonkeyPatch) -> str:
    if native:
        monkeypatch.setenv(NATIVE_PROMETHEUS_ENV, "true")
    else:
        monkeypatch.delenv(NATIVE_PROMETHEUS_ENV, raising=False)
    for name in _CALLBACK_LISTS:
        monkeypatch.setattr(litellm, name, [])
    logger: Final = PrometheusLogger()
    for name in _REGISTRATIONS[registration]:
        monkeypatch.setattr(litellm, name, [logger])
    _ = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="hi there",
        metadata={"user_api_key_hash": "hashed", "user_api_key_team_id": "team-1", "tags": ["prod"]},
    )
    await asyncio.sleep(1)
    text: Final = generate_latest(_ours()).decode()
    return "\n".join(
        line for line in text.splitlines() if not line.startswith("#") and _TIMED.search(line.split("{")[0]) is None
    )


@pytest.mark.parametrize("registration", list(_REGISTRATIONS))
async def test_a_real_acompletion_emits_the_same_metrics_through_the_rust_store(
    registration: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = await _completion_text(False, registration, monkeypatch)
    _clear_registry()
    python_text: Final = await _completion_text(False, registration, monkeypatch)
    _clear_registry()
    native_text: Final = await _completion_text(True, registration, monkeypatch)

    assert any(isinstance(collector, NativeFamilyCollector) for collector in _logger_collectors())
    assert re.search(r"^litellm_total_tokens_metric_total\{.*\} \d+\.0$", native_text, re.MULTILINE)
    assert "litellm_deployment_success_responses_total{" in native_text
    assert "litellm_remaining_team_budget_metric{" in native_text
    assert native_text == python_text


def test_multiprocess_mode_keeps_the_python_logger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(NATIVE_PROMETHEUS_ENV, "true")
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))

    assert not native_prometheus_enabled()
