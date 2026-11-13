import asyncio
import contextvars
import shutil
import socket
import subprocess
import time
from collections.abc import Callable, Coroutine, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final

import pytest
import redis

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.integrations.custom_logger_v2.call import CallContext, CallFacts, CallIds, CallKind, CallTarget, CallTiming
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.native import NativeState
from litellm.integrations.custom_logger_v2.outcomes import Abort
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PARALLEL_REQUEST_SLOT_TTL_SECONDS,
    PROXY_MaxParallelRequestsHandler_v3,
    _request_stash,
    get_request_stash,
)
from litellm.proxy.hooks.parallel_request_limiter_v3_native import (
    ParallelRequestLimiterNativeLogger,
    native_parallel_request_limiter,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.redis.stores.rate_limits import RateLimitScope
from litellm.types.utils import ModelResponse, Usage

pytest.importorskip("litellm.rust_bridge._native")

_READS: Final = frozenset({Field.deprecated_kwargs})


class Clock:
    def __init__(self) -> None:
        self.current = datetime(2026, 11, 13, 12, 0, 0, 250000, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


@dataclass(frozen=True)
class Request:
    auth: UserAPIKeyAuth
    body: Mapping[str, object]
    end: str = "success"
    usage: Usage = field(default_factory=lambda: Usage(prompt_tokens=7, completion_tokens=5, total_tokens=12))
    held: bool = False
    call_type: str = "acompletion"
    fallback: str | None = None


@dataclass
class Side:
    limiter: PROXY_MaxParallelRequestsHandler_v3
    native: ParallelRequestLimiterNativeLogger | None
    remote: "redis.Redis | None" = None
    contexts: dict[int, contextvars.Context] = field(default_factory=dict)
    datas: dict[int, dict[str, object]] = field(default_factory=dict)
    responses: dict[int, ModelResponse] = field(default_factory=dict)
    states: dict[int, object] = field(default_factory=dict)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextmanager
def _redis_server() -> Iterator[int]:
    port: Final = _free_port()
    server: Final = subprocess.Popen(
        ["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        client: Final = redis.Redis(port=port)
        deadline: Final = time.monotonic() + 10
        while True:
            try:
                client.ping()
                break
            except redis.ConnectionError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        yield port
    finally:
        server.terminate()
        server.wait()


@pytest.fixture(params=("memory", "redis"))
def backend(request: pytest.FixtureRequest) -> Iterator[tuple[int | None, int | None]]:
    if request.param == "memory":
        yield (None, None)
        return
    if shutil.which("redis-server") is None:
        pytest.skip("redis-server is not installed")
    with _redis_server() as python_port, _redis_server() as native_port:
        yield (python_port, native_port)


def _side(clock: Clock, native: bool, port: int | None = None) -> Side:
    cache: Final = DualCache() if port is None else DualCache(redis_cache=RedisCache(host="127.0.0.1", port=port))
    limiter: Final = PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(cache), time_provider=clock.now
    )
    assert limiter.rate_limits.has_redis == (port is not None)
    return Side(
        limiter=limiter,
        native=native_parallel_request_limiter(limiter) if native else None,
        remote=None if port is None else redis.Redis(port=port, decode_responses=True),
    )


def _outcome(error: BaseException | None) -> object:
    if error is None:
        return None
    return (
        type(error).__name__,
        getattr(error, "status_code", None),
        getattr(error, "detail", None),
        dict(getattr(error, "headers", None) or {}),
        getattr(error, "rate_limit_type", None),
        getattr(error, "descriptor_key", None),
        getattr(error, "model", None),
    )


def _data(index: int, request: Request) -> dict[str, object]:
    return {
        **request.body,
        "litellm_call_id": f"call-{index}",
        "metadata": {"user_api_key_auth": request.auth},
    }


def _ctx(side: Side, index: int, hook: Hook, request: Request) -> CallContext[object]:
    model: Final = request.body.get("model")
    state: Final = side.states.get(index)
    return CallContext(
        hook=hook,
        call=CallFacts(
            ids=CallIds(call_id=f"call-{index}"),
            kind=CallKind(call_type=request.call_type),
            target=CallTarget(model=model if isinstance(model, str) else ""),
            timing=CallTiming(start=0.0),
        ),
        state=state,
    )


async def _admit(side: Side, index: int, request: Request) -> object:
    _request_stash.set(None)
    data: Final = _data(index, request)
    side.datas[index] = data
    cache: Final = DualCache()
    try:
        if side.native is not None:
            request_state = side.native.request_state
            assert request_state is not None
            side.states[index] = request_state()
            outcome = await side.native.run_native(
                Hook.pre_call,
                _READS,
                SimpleNamespace(deprecated_kwargs=data),
                _ctx(side, index, Hook.pre_call, request),
            )
            state = side.states[index]
            assert isinstance(state, NativeState)
            assert side.native.bridge.owned(state.handle)
            if isinstance(outcome, Abort):
                assert outcome.exception is not None
                raise outcome.exception
        await side.limiter.async_pre_call_hook(request.auth, cache, data, request.call_type)
        if request.fallback is not None:
            return ("fallback", await side.limiter.async_fallback_target_allowed(request.auth, request.fallback, data))
    except Exception as error:  # noqa: BLE001  # the refusal is the observable compared across sides
        return _outcome(error)
    return None


def _logging_kwargs(index: int, request: Request) -> dict[str, object]:
    auth: Final = request.auth
    return {
        "litellm_call_id": f"call-{index}",
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": auth.api_key,
                "user_api_key_user_id": auth.user_id,
                "user_api_key_team_id": auth.team_id,
                "user_api_key_org_id": auth.org_id,
                "user_api_key_end_user_id": auth.end_user_id,
            },
            "hidden_params": {},
        },
        "litellm_params": {"metadata": {"model_group": request.fallback or request.body.get("model")}},
    }


async def _end(side: Side, index: int, request: Request) -> object:
    data: Final = side.datas[index]
    response: Final = ModelResponse(model=str(request.body.get("model")), usage=request.usage)
    side.responses[index] = response
    kwargs: Final = _logging_kwargs(index, request)
    match request.end:
        case "success":
            if side.native is not None:
                await side.native.run_native(
                    Hook.post_call_success,
                    _READS,
                    SimpleNamespace(deprecated_kwargs=data),
                    _ctx(side, index, Hook.post_call_success, request),
                )
            await side.limiter.async_post_call_success_hook(data, request.auth, response)
            if side.native is not None:
                await side.native.run_native(
                    Hook.log_prepare,
                    frozenset({Field.deprecated_kwargs, Field.standard_logging_payload}),
                    SimpleNamespace(
                        deprecated_kwargs=kwargs, standard_logging_payload=kwargs["standard_logging_object"]
                    ),
                    _ctx(side, index, Hook.log_prepare, request),
                )
            await side.limiter.async_logging_hook(kwargs, response, request.call_type)
            await side.limiter.async_log_success_event(kwargs, response, None, None)
        case "failure":
            await side.limiter.async_log_failure_event(kwargs, None, None, None)
        case "proxy_failure":
            if side.native is not None:
                await side.native.run_native(
                    Hook.post_call_failure,
                    _READS,
                    SimpleNamespace(deprecated_kwargs=data),
                    _ctx(side, index, Hook.post_call_failure, request),
                )
            await side.limiter.async_post_call_failure_hook(data, Exception("guardrail"), request.auth)
        case "disconnect":
            await side.limiter.async_release_max_parallel_requests_on_disconnect(request.auth)
    stash: Final = get_request_stash()
    return (
        response._hidden_params.get("additional_headers"),
        kwargs["standard_logging_object"],
        None if stash is None else stash.rate_limit_response,
    )


def _run(
    side: Side, index: int, step: Callable[[Side, int, Request], Coroutine[object, object, object]], request: Request
) -> "asyncio.Task[object]":
    context: Final = side.contexts.setdefault(index, contextvars.copy_context())
    return asyncio.get_running_loop().create_task(step(side, index, request), context=context)


def _snapshot(side: Side, scopes: Sequence[tuple[str, str]]) -> object:
    rows: Final[list[object]] = []
    for key, value in scopes:
        scope = RateLimitScope(key, value)
        local = side.limiter.rate_limits.local(
            (scope.window, scope.counter("requests"), scope.counter("tokens"), scope.gauge)
        )
        gauge = local[3]
        remote = (
            None
            if side.remote is None
            else (
                side.remote.get(f"{{{key}:{value}}}:requests"),
                side.remote.get(f"{{{key}:{value}}}:tokens"),
                side.remote.zcard(f"{{{key}:{value}}}:max_parallel_requests"),
                side.remote.get(f"{{{key}:{value}}}:window"),
            )
        )
        rows.append(
            (
                key,
                value,
                local[0],
                local[1],
                local[2],
                len(gauge) if isinstance(gauge, dict) else gauge,
                remote,
            )
        )
    return rows


async def _differential(
    requests: Sequence[Request],
    scopes: Sequence[tuple[str, str]],
    backend: tuple[int | None, int | None],
    between: Callable[[Clock], None] | None = None,
) -> list[object]:
    clock: Final = Clock()
    python: Final = _side(clock, native=False, port=backend[0])
    native: Final = _side(clock, native=True, port=backend[1])
    observed: Final[list[object]] = []
    held: Final[list[tuple[int, Request]]] = []
    for index, request in enumerate(requests):
        admitted = [await _run(side, index, _admit, request) for side in (python, native)]
        assert admitted[0] == admitted[1], (index, admitted)
        assert python.datas[index] == native.datas[index], index
        observed.append(admitted[0])
        if admitted[0] is None or admitted[0][0] == "fallback":
            if request.held:
                held.append((index, request))
            else:
                ended = [await _run(side, index, _end, request) for side in (python, native)]
                assert ended[0] == ended[1], (index, ended)
        assert _snapshot(python, scopes) == _snapshot(native, scopes), index
        if between is not None:
            between(clock)
    for index, request in held:
        ended = [await _run(side, index, _end, request) for side in (python, native)]
        assert ended[0] == ended[1], (index, ended)
        assert _snapshot(python, scopes) == _snapshot(native, scopes), index
    return observed


_KEY: Final = "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"
_CHAT: Final = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello there, how are you today?"}]}


def _auth(**limits: object) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key=_KEY, **limits)  # pyright: ignore[reportArgumentType]  # pydantic fields by keyword


@pytest.mark.asyncio
async def test_rpm_refusal_matches_python(backend: tuple[int | None, int | None]) -> None:
    observed: Final = await _differential(
        [Request(_auth(rpm_limit=2), _CHAT)] * 3,
        [("api_key", _KEY)],
        backend,
    )
    assert observed[:2] == [None, None]
    assert observed[2] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("max_tokens", (None, 40, 4000))
async def test_tpm_reservation_settles_to_usage_like_python(
    max_tokens: int | None, backend: tuple[int | None, int | None]
) -> None:
    body: Final = {**_CHAT, **({} if max_tokens is None else {"max_tokens": max_tokens})}
    observed: Final = await _differential(
        [Request(_auth(tpm_limit=500), body) for _ in range(6)],
        [("api_key", _KEY)],
        backend,
    )
    assert (observed[0] is None) == (max_tokens != 4000)


@pytest.mark.asyncio
async def test_parallel_slots_hold_and_release_like_python(backend: tuple[int | None, int | None]) -> None:
    held: Final = Request(_auth(max_parallel_requests=2), _CHAT, held=True)
    observed: Final = await _differential(
        [held, held, held, Request(_auth(max_parallel_requests=2), _CHAT)], [("api_key", _KEY)], backend
    )
    assert observed[0] is None and observed[1] is None
    assert observed[2] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ("failure", "proxy_failure", "disconnect"))
async def test_failures_release_and_refund_like_python(end: str, backend: tuple[int | None, int | None]) -> None:
    auth: Final = _auth(tpm_limit=300, max_parallel_requests=1, rpm_limit=10)
    await _differential(
        [Request(auth, {**_CHAT, "max_tokens": 50}, end=end) for _ in range(4)],
        [("api_key", _KEY)],
        backend,
    )


@pytest.mark.asyncio
async def test_team_user_and_member_scopes_match_python(backend: tuple[int | None, int | None]) -> None:
    auth: Final = _auth(
        user_id="u1",
        team_id="t1",
        user_rpm_limit=3,
        user_tpm_limit=400,
        team_rpm_limit=4,
        team_tpm_limit=600,
        team_member_rpm_limit=2,
        team_member_tpm_limit=350,
    )
    await _differential(
        [Request(auth, {**_CHAT, "max_tokens": 30}) for _ in range(5)],
        [("api_key", _KEY), ("user", "u1"), ("team", "t1"), ("team_member", "t1:u1")],
        backend,
    )


@pytest.mark.asyncio
async def test_model_per_key_limits_match_python(backend: tuple[int | None, int | None]) -> None:
    auth: Final = _auth(metadata={"model_rpm_limit": {"gpt-4o": 2}, "model_tpm_limit": {"gpt-4o": 200}})
    await _differential(
        [Request(auth, {**_CHAT, "max_tokens": 20}) for _ in range(4)],
        [("model_per_key", f"{_KEY}:gpt-4o")],
        backend,
    )


@pytest.mark.asyncio
async def test_small_tpm_limit_writes_the_same_implicit_output_cap(backend: tuple[int | None, int | None]) -> None:
    await _differential(
        [Request(_auth(tpm_limit=100), _CHAT) for _ in range(3)],
        [("api_key", _KEY)],
        backend,
    )


@pytest.mark.asyncio
async def test_windows_roll_over_with_the_clock_like_python(backend: tuple[int | None, int | None]) -> None:
    def advance(clock: Clock) -> None:
        clock.advance(31)

    await _differential(
        [Request(_auth(rpm_limit=2, tpm_limit=1000), {**_CHAT, "max_tokens": 10}) for _ in range(6)],
        [("api_key", _KEY)],
        backend,
        advance,
    )


@pytest.mark.asyncio
async def test_slot_ttl_constant_is_shared() -> None:
    assert PARALLEL_REQUEST_SLOT_TTL_SECONDS == 3600


@pytest.mark.asyncio
async def test_a_request_without_auth_metadata_falls_through_to_python() -> None:
    clock: Final = Clock()
    native: Final = _side(clock, native=True)
    assert native.native is not None
    data: Final = {**_CHAT, "litellm_call_id": "call-x"}
    _request_stash.set(None)
    request_state = native.native.request_state
    assert request_state is not None
    state: Final = request_state()
    assert isinstance(state, NativeState)
    outcome: Final = await native.native.run_native(
        Hook.pre_call,
        _READS,
        SimpleNamespace(deprecated_kwargs=data),
        CallContext(hook=Hook.pre_call, call=None, state=state),
    )
    assert outcome is None
    stash: Final = get_request_stash()
    assert stash is None or stash.native_state is None


@pytest.mark.asyncio
async def test_router_fallback_targets_are_admitted_and_settled_like_python(
    backend: tuple[int | None, int | None],
) -> None:
    auth: Final = _auth(metadata={"model_tpm_limit": {"gpt-4o": 300, "gpt-4o-mini": 300}})
    observed: Final = await _differential(
        [Request(auth, {**_CHAT, "max_tokens": 20}, fallback="gpt-4o-mini") for _ in range(4)],
        [("model_per_key", f"{_KEY}:gpt-4o"), ("model_per_key", f"{_KEY}:gpt-4o-mini")],
        backend,
    )
    assert observed[0] == ("fallback", True)


@pytest.mark.asyncio
async def test_post_call_charging_without_reservation_matches_python(
    backend: tuple[int | None, int | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", "false")
    await _differential(
        [Request(_auth(tpm_limit=30, rpm_limit=10), _CHAT) for _ in range(5)],
        [("api_key", _KEY)],
        backend,
    )
