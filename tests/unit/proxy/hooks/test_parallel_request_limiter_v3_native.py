import asyncio
import contextvars
import shutil
import socket
import subprocess
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final

import pytest
import redis
from pydantic import TypeAdapter

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.integrations.custom_logger_v2.call import CallContext, CallFacts, CallIds, CallKind, CallTarget, CallTiming
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.native import NativeState
from litellm.integrations.custom_logger_v2.outcomes import Abort
from litellm.litellm_core_utils.ptu_shares import PTUTeamCeiling
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
    _PROXY_DynamicRateLimitHandlerV3,  # pyright: ignore[reportPrivateUsage]  # the class has no public alias
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PARALLEL_REQUEST_SLOT_TTL_SECONDS,
    PROXY_MaxParallelRequestsHandler_v3,
    _request_stash,
    get_request_stash,
)
from litellm.proxy.hooks.parallel_request_limiter_v3_native import (
    ParallelRequestLimiterNativeLogger,
    native_parallel_request_limiter,
    PreCallInputsScope,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.redis.stores.rate_limits import RateLimitScope
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.utils import CallTypes, ModelResponse, Usage

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
    nested: bool = False
    response: object = None
    endpoint_type: EndpointType = EndpointType.GENERIC
    logged: Mapping[str, object] = field(default_factory=dict)
    logged_metadata: Mapping[str, object] = field(default_factory=dict)
    ceiling_raises_at_end: bool = False


@dataclass
class Side:
    limiter: PROXY_MaxParallelRequestsHandler_v3
    native: ParallelRequestLimiterNativeLogger | None
    remote: "redis.Redis | None" = None
    contexts: dict[int, contextvars.Context] = field(default_factory=dict)
    datas: dict[int, dict[str, object]] = field(default_factory=dict)
    responses: dict[int, object] = field(default_factory=dict)
    states: dict[int, object] = field(default_factory=dict)
    dynamic: _PROXY_DynamicRateLimitHandlerV3 | None = None


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


def _side(clock: Clock, native: bool, port: int | None = None, ceiling: PTUTeamCeiling | None = None) -> Side:
    cache: Final = DualCache() if port is None else DualCache(redis_cache=RedisCache(host="127.0.0.1", port=port))
    limiter: Final = PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(cache),
        time_provider=clock.now,
        ptu_team_ceiling_resolver=lambda team_id, group: ceiling,
    )
    assert limiter.rate_limits.has_redis == (port is not None)
    return Side(
        limiter=limiter,
        native=native_parallel_request_limiter(limiter) if native else None,
        remote=None if port is None else redis.Redis(port=port, decode_responses=True),
    )


def _dynamic(
    monkeypatch: pytest.MonkeyPatch, clock: Clock, native: bool, port: int | None
) -> _PROXY_DynamicRateLimitHandlerV3:
    from litellm import Router
    from litellm.types.utils import PriorityReservationSettings

    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "gpt-4o", "api_key": "k", "api_base": "b", "tpm": 400, "rpm": 6},
            }
        ]
    )
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr("litellm.priority_reservation", {"high": 0.5, "low": 0.5})
    monkeypatch.setattr(
        "litellm.priority_reservation_settings",
        PriorityReservationSettings(saturation_threshold=0.3, saturation_check_cache_ttl=0),
    )
    monkeypatch.setattr("litellm.callbacks", ["dynamic_rate_limiter_v3"])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setenv("LITELLM_NATIVE_PARALLEL_REQUEST_LIMITER", "true" if native else "false")
    handler: Final = _PROXY_DynamicRateLimitHandlerV3(
        DualCache() if port is None else DualCache(redis_cache=RedisCache(host="127.0.0.1", port=port)),
        time_provider=clock.now,
    )
    monkeypatch.delenv("LITELLM_NATIVE_PARALLEL_REQUEST_LIMITER")
    assert (handler.v3_limiter.native_companion is not None) == native
    handler.update_variables(llm_router=router)
    return handler


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
    if request.call_type == "pass_through_endpoint":
        return {**request.body, "litellm_logging_obj": SimpleNamespace(litellm_call_id=f"call-{index}")}
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
    if not request.nested:
        _request_stash.set(None)
    data: Final = _data(index, request)
    side.datas[index] = data
    cache: Final = DualCache()
    try:
        if side.dynamic is not None:
            await side.dynamic.async_pre_call_hook(request.auth, cache, data, request.call_type)  # pyright: ignore[reportArgumentType]  # the call types are the CallTypesLiteral values
        if side.native is not None:
            request_state = side.native.request_state
            assert request_state is not None
            side.states[index] = request_state()
            with PreCallInputsScope(request.auth, request.endpoint_type):
                outcome = await side.native.run_native(
                    Hook.pre_call,
                    _READS,
                    SimpleNamespace(deprecated_kwargs=data),
                    _ctx(side, index, Hook.pre_call, request),
                )
            state = side.states[index]
            assert isinstance(state, NativeState)
            stash = get_request_stash()
            owner = (stash.native_state if stash is not None else None) if request.nested else state.handle
            assert owner is not None or request.nested
            assert owner is None or side.native.bridge.owned(owner)  # pyright: ignore[reportArgumentType]  # the handle is the bridge state
            if isinstance(outcome, Abort):
                assert outcome.exception is not None
                raise outcome.exception
        await side.limiter.async_pre_call_hook(
            request.auth,
            cache,
            data,
            request.call_type,
            endpoint_type=request.endpoint_type,
        )
        if request.fallback is not None:
            return ("fallback", await side.limiter.async_fallback_target_allowed(request.auth, request.fallback, data))
    except Exception as error:  # noqa: BLE001  # the refusal is the observable compared across sides
        return _outcome(error)
    return None


def _logging_kwargs(index: int, request: Request) -> dict[str, object]:
    auth: Final = request.auth
    return {
        "litellm_call_id": f"call-{index}",
        "call_type": request.call_type,
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": auth.api_key,
                "user_api_key_user_id": auth.user_id,
                "user_api_key_team_id": auth.team_id,
                "user_api_key_org_id": auth.org_id,
                "user_api_key_project_id": auth.project_id,
                "user_api_key_end_user_id": auth.end_user_id,
                **request.logged_metadata,
            },
            "hidden_params": {},
        },
        "litellm_params": {"metadata": {"model_group": request.fallback or request.body.get("model")}},
        **({"combined_usage_object": request.usage} if request.end == "partial_failure" else {}),
        **request.logged,
    }


def _raising_ceiling(team_id: str, group: str) -> PTUTeamCeiling | None:
    raise RuntimeError(f"no ceiling for {team_id}:{group}")


async def _end(side: Side, index: int, request: Request) -> object:
    data: Final = side.datas[index]
    response: Final = (
        ModelResponse(model=str(request.body.get("model")), usage=request.usage)
        if request.response is None
        else request.response
    )
    side.responses[index] = response
    kwargs: Final = _logging_kwargs(index, request)
    resolver: Final = side.limiter._ptu_team_ceiling_resolver  # pyright: ignore[reportPrivateUsage]  # the resolver fails only while settling
    if request.ceiling_raises_at_end:
        side.limiter._ptu_team_ceiling_resolver = _raising_ceiling  # pyright: ignore[reportPrivateUsage]  # the resolver fails only while settling
    try:
        return await _ended(side, index, request, data, response, kwargs)
    finally:
        side.limiter._ptu_team_ceiling_resolver = resolver  # pyright: ignore[reportPrivateUsage]  # the resolver fails only while settling


async def _ended(
    side: Side, index: int, request: Request, data: dict[str, object], response: object, kwargs: dict[str, object]
) -> object:
    match request.end:
        case "success":
            if side.dynamic is not None:
                await side.dynamic.async_post_call_success_hook(data, request.auth, response)  # pyright: ignore[reportArgumentType]  # a batch object stands in for the response like the proxy passes it
            if side.native is not None:
                await side.native.run_native(
                    Hook.post_call_success,
                    frozenset({Field.deprecated_kwargs, Field.deprecated_response_obj}),
                    SimpleNamespace(deprecated_kwargs=data, deprecated_response_obj=response),
                    _ctx(side, index, Hook.post_call_success, request),
                )
            await side.limiter.async_post_call_success_hook(data, request.auth, response)  # pyright: ignore[reportArgumentType]  # a batch object stands in for the response like the proxy passes it
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
            if side.dynamic is not None:
                await side.dynamic.async_log_success_event(kwargs, response, None, None)
            await side.limiter.async_log_success_event(kwargs, response, None, None)
        case "log_success":
            await side.limiter.async_logging_hook(kwargs, response, request.call_type)
            await side.limiter.async_log_success_event(kwargs, response, None, None)
        case "failure" | "partial_failure":
            if side.dynamic is not None:
                await side.dynamic.async_log_failure_event(kwargs, None, None, None)
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
    hidden: Final = getattr(response, "_hidden_params", {})
    return (
        hidden.get("additional_headers"),
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
    stores: Final = (side.limiter.rate_limits,) + (
        () if side.dynamic is None else (side.dynamic.v3_limiter.rate_limits,)
    )
    for (key, value), store in ((scope, store) for scope in scopes for store in stores):
        scope = RateLimitScope(key, value)
        local = store.local((scope.window, scope.counter("requests"), scope.counter("tokens"), scope.gauge))
        gauge = local[3]
        remote = (
            None
            if side.remote is None
            else (
                side.remote.get(f"{{{key}:{value}}}:requests"),
                side.remote.get(f"{{{key}:{value}}}:tokens"),
                side.remote.zcard(f"{{{key}:{value}}}:max_parallel_requests"),
                side.remote.exists(f"{{{key}:{value}}}:window"),
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
    ceiling: PTUTeamCeiling | None = None,
    dynamic: pytest.MonkeyPatch | None = None,
    probe: Callable[[Side], Awaitable[object]] | None = None,
) -> list[object]:
    clock: Final = Clock()
    python: Final = _side(clock, native=False, port=backend[0], ceiling=ceiling)
    native: Final = _side(clock, native=True, port=backend[1], ceiling=ceiling)
    if dynamic is not None:
        python.dynamic = _dynamic(dynamic, clock, native=False, port=backend[0])
        native.dynamic = _dynamic(dynamic, clock, native=True, port=backend[1])
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
        if probe is not None:
            assert await probe(python) == await probe(native), index
        if between is not None:
            between(clock)
    for index, request in held:
        ended = [await _run(side, index, _end, request) for side in (python, native)]
        assert ended[0] == ended[1], (index, ended)
        assert _snapshot(python, scopes) == _snapshot(native, scopes), index
        if probe is not None:
            assert await probe(python) == await probe(native), index
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
@pytest.mark.parametrize("end", ("success", "log_success", "failure", "partial_failure"))
@pytest.mark.parametrize(
    "unusual",
    (
        Request(_auth(), _CHAT, logged={"user": 123}),
        Request(_auth(), _CHAT, logged={"user": 1.5}),
        Request(_auth(), _CHAT, logged={"user": True}),
        Request(_auth(), _CHAT, logged_metadata={"session_id": 7, "agent_id": "a1"}),
        Request(_auth(), _CHAT, logged_metadata={"agent_id": ["a1"]}),
        Request(_auth(project_id="p1"), _CHAT, logged_metadata={"user_api_key_project_id": 9}),
        Request(_auth(team_id="t1"), _CHAT, ceiling_raises_at_end=True),
    ),
)
async def test_a_settlement_that_cannot_be_built_still_frees_the_slot_like_python(
    unusual: Request, end: str, backend: tuple[int | None, int | None]
) -> None:
    limits: Final = {"max_parallel_requests": 1, "tpm_limit": 5000, "rpm_limit": 50}
    auth: Final = unusual.auth.model_copy(update=limits)
    first: Final = Request(
        auth,
        {**_CHAT, "max_tokens": 40},
        end=end,
        usage=_CACHED_USAGE,
        logged=unusual.logged,
        logged_metadata=unusual.logged_metadata,
        ceiling_raises_at_end=unusual.ceiling_raises_at_end,
    )
    observed: Final = await _differential(
        [first, Request(auth, {**_CHAT, "max_tokens": 40})],
        [
            ("api_key", _KEY),
            ("team", "t1"),
            ("end_user", "123"),
            ("agent", "a1"),
            ("agent_session", "a1:7"),
            ("model_per_project", "9:gpt-4o"),
        ],
        backend,
    )
    assert observed == [None, None]


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


_CEILING: Final = PTUTeamCeiling(
    model_group="gpt-4o", tpm_limit=900, output_to_input_ratio=4.0, cached_input_ratio=0.25
)
_CACHED_USAGE: Final = Usage(
    prompt_tokens=30,
    completion_tokens=9,
    total_tokens=39,
    prompt_tokens_details={"cached_tokens": 12},  # pyright: ignore[reportArgumentType]  # pydantic coerces the details
)


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ("success", "failure", "partial_failure", "proxy_failure"))
async def test_team_ptu_reservation_settles_in_normalized_tokens_like_python(
    end: str, backend: tuple[int | None, int | None]
) -> None:
    auth: Final = _auth(team_id="t1", tpm_limit=5000)
    observed: Final = await _differential(
        [Request(auth, {**_CHAT, "max_tokens": 60}, end=end, usage=_CACHED_USAGE, held=True) for _ in range(5)],
        [("api_key", _KEY), ("model_per_team_ptu", "t1:gpt-4o")],
        backend,
        ceiling=_CEILING,
    )
    assert observed[0] is None
    assert observed[-1] is not None


@pytest.mark.asyncio
async def test_a_request_over_the_team_ptu_share_is_refused_like_python(
    backend: tuple[int | None, int | None],
) -> None:
    observed: Final = await _differential(
        [Request(_auth(team_id="t1", max_parallel_requests=1), {**_CHAT, "max_tokens": 400}) for _ in range(2)],
        [("api_key", _KEY), ("model_per_team_ptu", "t1:gpt-4o")],
        backend,
        ceiling=_CEILING,
    )
    assert isinstance(observed[0], tuple) and observed[0][1] == 400
    assert observed[1] == observed[0]


@pytest.mark.asyncio
async def test_team_ptu_charges_after_the_call_without_reservation_like_python(
    backend: tuple[int | None, int | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", "false")
    await _differential(
        [Request(_auth(team_id="t1"), _CHAT, usage=_CACHED_USAGE) for _ in range(4)],
        [("model_per_team_ptu", "t1:gpt-4o")],
        backend,
        ceiling=_CEILING,
    )


def _project_auth(itpm: int | None, otpm: int | None, **limits: object) -> UserAPIKeyAuth:
    metadata: Final = {
        **({} if itpm is None else {"model_itpm_limit": {"gpt-4o": itpm}}),
        **({} if otpm is None else {"model_otpm_limit": {"gpt-4o": otpm}}),
        "model_tpm_limit": {"gpt-4o": 4000},
    }
    return _auth(project_id="p1", project_metadata=metadata, **limits)


_PROJECT_SCOPES: Final = (
    ("model_per_project", "p1:gpt-4o"),
    ("model_per_project_itpm", "p1:gpt-4o"),
    ("model_per_project_otpm", "p1:gpt-4o"),
    ("api_key", _KEY),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ("success", "failure", "partial_failure", "proxy_failure"))
@pytest.mark.parametrize(("itpm", "otpm"), ((60, 120), (60, None), (None, 70), (500, 25)))
async def test_project_io_reservations_match_python(
    end: str, itpm: int | None, otpm: int | None, backend: tuple[int | None, int | None]
) -> None:
    await _differential(
        [
            Request(
                _project_auth(itpm, otpm, max_parallel_requests=3),
                {**_CHAT, "max_tokens": 20},
                end=end,
                usage=_CACHED_USAGE,
            )
            for _ in range(6)
        ],
        _PROJECT_SCOPES,
        backend,
    )


@pytest.mark.asyncio
async def test_an_otpm_refusal_refunds_the_tpm_and_ptu_reservations_like_python(
    backend: tuple[int | None, int | None],
) -> None:
    auth: Final = _project_auth(500, 10, team_id="t1", tpm_limit=5000)
    observed: Final = await _differential(
        [Request(auth, {**_CHAT, "max_tokens": 30}) for _ in range(2)],
        (*_PROJECT_SCOPES, ("model_per_team_ptu", "t1:gpt-4o")),
        backend,
        ceiling=_CEILING,
    )
    assert isinstance(observed[0], tuple) and observed[0][1] == 429


@pytest.mark.asyncio
async def test_project_io_without_reservation_still_reserves_like_python(
    backend: tuple[int | None, int | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", "false")
    await _differential(
        [Request(_project_auth(80, 40), _CHAT, usage=_CACHED_USAGE) for _ in range(5)],
        _PROJECT_SCOPES,
        backend,
    )


async def _guarded(call: Coroutine[object, object, object]) -> object:
    try:
        return await call
    except Exception as error:  # noqa: BLE001  # the refusal is the observable compared across sides
        return _outcome(error)


@pytest.mark.asyncio
async def test_body_free_checks_and_request_capacity_run_through_the_engine_like_python(
    backend: tuple[int | None, int | None],
) -> None:
    clock: Final = Clock()
    sides: Final = (_side(clock, native=False, port=backend[0]), _side(clock, native=True, port=backend[1]))
    auth: Final = _auth(rpm_limit=3, max_parallel_requests=2)

    async def capacity(side: Side) -> object:
        async with side.limiter.request_capacity(auth, "gpt-4o"):
            return _snapshot(side, [("api_key", _KEY)])

    observed: Final[list[object]] = []
    for _ in range(4):
        checked = [await _guarded(side.limiter.raise_if_over_limit_before_body(auth)) for side in sides]
        assert checked[0] == checked[1]
        held = [await _guarded(capacity(side)) for side in sides]
        assert held[0] == held[1]
        assert _snapshot(sides[0], [("api_key", _KEY)]) == _snapshot(sides[1], [("api_key", _KEY)])
        observed.append((checked[0], held[0]))
    assert observed[-1][0] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "increments",
    (({"requests": 1, "tokens": 30},), ({"requests": 1, "tokens": 400},), ({"tokens": 0},), ({"tokens": -5},)),
)
async def test_atomic_checks_run_through_the_engine_like_python(
    increments: tuple[dict[str, int], ...], backend: tuple[int | None, int | None]
) -> None:
    clock: Final = Clock()
    sides: Final = (_side(clock, native=False, port=backend[0]), _side(clock, native=True, port=backend[1]))
    descriptors: Final = [
        {"key": "api_key", "value": _KEY, "rate_limit": {"requests_per_unit": 2, "tokens_per_unit": 100}}
    ]
    for _ in range(3):
        responses = [
            await _guarded(
                side.limiter.atomic_check_and_increment_by_n(
                    descriptors,  # pyright: ignore[reportArgumentType]  # the typed dict as JSON callers build it
                    list(increments),  # pyright: ignore[reportArgumentType]  # the typed dict as JSON callers build it
                )
            )
            for side in sides
        ]
        assert responses[0] == responses[1]
        assert _snapshot(sides[0], [("api_key", _KEY)]) == _snapshot(sides[1], [("api_key", _KEY)])


@pytest.mark.asyncio
async def test_mcp_tool_calls_are_limited_natively_like_python(
    backend: tuple[int | None, int | None],
) -> None:
    observed: Final = await _differential(
        [Request(_auth(rpm_limit=2, max_parallel_requests=1), _CHAT, call_type="call_mcp_tool") for _ in range(3)],
        [("api_key", _KEY)],
        backend,
    )
    assert observed[2] is not None


@pytest.mark.asyncio
async def test_the_dynamic_limiters_private_limiter_runs_on_the_engine_when_the_flag_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
        _PROXY_DynamicRateLimitHandlerV3,  # pyright: ignore[reportPrivateUsage]  # the class has no public alias
    )

    monkeypatch.setenv("LITELLM_NATIVE_PARALLEL_REQUEST_LIMITER", "true")
    handler: Final = _PROXY_DynamicRateLimitHandlerV3(DualCache())
    assert handler.v3_limiter.native_companion is not None
    monkeypatch.setenv("LITELLM_NATIVE_PARALLEL_REQUEST_LIMITER", "false")
    assert _PROXY_DynamicRateLimitHandlerV3(DualCache()).v3_limiter.native_companion is None


def _with_dynamic(
    sides: tuple[Side, Side], monkeypatch: pytest.MonkeyPatch, clock: Clock, backend: tuple[int | None, int | None]
) -> tuple[Side, Side]:
    sides[0].dynamic = _dynamic(monkeypatch, clock, native=False, port=backend[0])
    sides[1].dynamic = _dynamic(monkeypatch, clock, native=True, port=backend[1])
    return sides


_DYNAMIC_SCOPES: Final = [("api_key", _KEY), ("model_saturation_check", "gpt-4o"), ("priority_model", "gpt-4o:high")]


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ("success", "failure", "partial_failure", "proxy_failure", "disconnect"))
async def test_with_the_dynamic_limiter_registered_the_native_owns_the_request_and_limits_like_python(
    end: str, backend: tuple[int | None, int | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    auth: Final = UserAPIKeyAuth(
        api_key=_KEY, metadata={"priority": "high"}, rpm_limit=20, tpm_limit=3000, max_parallel_requests=2
    )
    observed: Final = await _differential(
        [Request(auth, {**_CHAT, "max_tokens": 20}, end=end, held=index % 3 == 0) for index in range(7)],
        _DYNAMIC_SCOPES,
        backend,
        dynamic=monkeypatch,
    )
    assert observed.count(None) < len(observed)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_dynamic", (False, True))
@pytest.mark.parametrize("tool_limit", (5, 1))
async def test_a_nested_mcp_tool_call_reuses_the_requests_slot_like_python(
    tool_limit: int, with_dynamic: bool, backend: tuple[int | None, int | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock: Final = Clock()
    built: Final = (_side(clock, native=False, port=backend[0]), _side(clock, native=True, port=backend[1]))
    sides: Final = _with_dynamic(built, monkeypatch, clock, backend) if with_dynamic else built
    outer: Final = Request(
        UserAPIKeyAuth(
            api_key=_KEY, metadata={"priority": "high"}, max_parallel_requests=1, rpm_limit=tool_limit, tpm_limit=3000
        ),
        {**_CHAT, "max_tokens": 20},
    )
    tool: Final = Request(outer.auth, _CHAT, call_type="call_mcp_tool", nested=True)
    for side in sides:
        context = contextvars.copy_context()
        side.contexts[0] = context
        side.contexts[1] = context
    admitted: Final = [(await _run(side, 0, _admit, outer), await _run(side, 1, _admit, tool)) for side in sides]
    assert admitted[0] == admitted[1]
    assert (admitted[0][1] is None) == (tool_limit > 1)
    scopes: Final = _DYNAMIC_SCOPES
    assert _snapshot(sides[0], scopes) == _snapshot(sides[1], scopes)
    ended: Final = [(await _run(side, 1, _end, tool), await _run(side, 0, _end, outer)) for side in sides]
    assert ended[0] == ended[1]
    assert _snapshot(sides[0], scopes) == _snapshot(sides[1], scopes)


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ("pass_through_endpoint", "_arealtime"))
@pytest.mark.parametrize("end", ("success", "failure", "proxy_failure"))
async def test_a_nested_pass_through_or_realtime_call_on_a_native_request_is_limited_like_python(
    call_type: str, end: str, backend: tuple[int | None, int | None]
) -> None:
    clock: Final = Clock()
    sides: Final = (_side(clock, native=False, port=backend[0]), _side(clock, native=True, port=backend[1]))
    auth: Final = _auth(max_parallel_requests=2, rpm_limit=4, tpm_limit=3000)
    outer: Final = Request(auth, {**_CHAT, "max_tokens": 20}, end=end)
    nested: Final = Request(auth, {**_CHAT, "max_tokens": 10}, call_type=call_type, nested=True, end=end)
    scopes: Final = [("api_key", _KEY)]
    for side in sides:
        context = contextvars.copy_context()
        side.contexts.update({0: context, 1: context, 2: context})
    admitted: Final = [
        (await _run(side, 0, _admit, outer), await _run(side, 1, _admit, nested), await _run(side, 2, _admit, nested))
        for side in sides
    ]
    assert admitted[0] == admitted[1]
    assert _snapshot(sides[0], scopes) == _snapshot(sides[1], scopes)
    ended: Final = [(await _run(side, 1, _end, nested), await _run(side, 0, _end, outer)) for side in sides]
    assert ended[0] == ended[1]
    assert _snapshot(sides[0], scopes) == _snapshot(sides[1], scopes)


@pytest.mark.asyncio
async def test_python_and_rust_callers_racing_on_the_memory_tier_lose_no_update() -> None:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import WindowGuardedIncrement
    from litellm.redis.stores.rate_limits.types import CounterIncrement

    side: Final = _side(Clock(), native=True)
    limiter: Final = side.limiter
    descriptor: Final = {"key": "api_key", "value": _KEY, "rate_limit": {"tokens_per_unit": 10**9}}
    seeded: Final = await limiter.atomic_check_and_increment_by_n(
        [descriptor],  # pyright: ignore[reportArgumentType]  # the typed dict as JSON callers build it
        [{"tokens": 1}],
    )
    window: Final = next(iter(seeded.get("reservation_windows", frozenset())))
    counter: Final = RateLimitScope("api_key", _KEY).counter("tokens")
    rounds: Final = 300

    async def rust_atomic() -> None:
        await limiter.atomic_check_and_increment_by_n(
            [descriptor],  # pyright: ignore[reportArgumentType]  # the typed dict as JSON callers build it
            [{"tokens": 1}],
        )

    async def store_increment() -> None:
        await limiter.rate_limits.increment(CounterIncrement(counter=counter, amount=1, ttl=60))

    async def python_read_modify_write() -> None:
        await limiter._apply_local_window_guarded_token_increments(  # pyright: ignore[reportPrivateUsage]  # the Python section under test
            [WindowGuardedIncrement(CounterIncrement(counter=counter, amount=1, ttl=60), window[1], "local")]
        )

    await asyncio.gather(
        *(call() for _ in range(rounds) for call in (rust_atomic, store_increment, python_read_modify_write))
    )
    final: Final = limiter.rate_limits.local((counter,))[0]
    assert isinstance(final, int | float)
    assert final == 1 + 3 * rounds


def _batch_usage(monkeypatch: pytest.MonkeyPatch, total_tokens: int, request_count: int) -> None:
    from litellm.proxy.hooks.batch_rate_limiter import PROXY_BatchRateLimiter, BatchFileUsage

    async def counted(*_: object, **__: object) -> BatchFileUsage:
        return BatchFileUsage(
            total_tokens=total_tokens,
            request_count=request_count,
            output_tokens=total_tokens // 6,
            per_model_usage={"gpt-4o": {"total_tokens": total_tokens, "output_tokens": total_tokens // 6}},
        )

    monkeypatch.setattr(PROXY_BatchRateLimiter, "count_input_file_usage", counted)


_BATCH: Final = {"model": "gpt-4o", "input_file_id": "file-abc", "endpoint": "/v1/chat/completions"}


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ("success", "failure", "proxy_failure"))
@pytest.mark.parametrize(
    "limits",
    (
        {"tpd_limit": 1500, "rpm_limit": 100},
        {"rpm_limit": 15, "tpm_limit": 100000},
        {"tpm_limit": 1300},
        {"project_id": "proj", "project_metadata": {"model_itpm_limit": {"gpt-4o": 1300}}},
    ),
)
async def test_a_batch_is_charged_and_refused_like_python(
    end: str,
    limits: dict[str, object],
    backend: tuple[int | None, int | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _batch_usage(monkeypatch, total_tokens=600, request_count=6)
    auth: Final = _auth(**limits)
    observed: Final = await _differential(
        [Request(auth, _BATCH, end=end, call_type="acreate_batch") for _ in range(3)],
        [("api_key", _KEY), ("api_key_tpd", _KEY), ("model_per_project_itpm", "proj:gpt-4o")],
        backend,
    )
    assert observed[0] is None
    assert observed[2] is not None or end == "proxy_failure"


def _batch_object(batch_id: str, status: str) -> SimpleNamespace:
    return SimpleNamespace(id=batch_id, status=status, object="batch")


_TEAM_ENQUEUED: Final = "t-enqueued"
_RECORD: Final = TypeAdapter(dict[str, object])


def _enqueued_probe(batch_ids: Sequence[str]) -> Callable[[Side], Awaitable[object]]:
    from litellm.proxy.hooks.batch_enqueued_tokens import BatchEnqueuedTokenScope

    scopes: Final = (
        BatchEnqueuedTokenScope(key="api_key", value=_KEY, limit=1),
        BatchEnqueuedTokenScope(key="team", value=_TEAM_ENQUEUED, limit=1),
    )

    async def probe(side: Side) -> object:
        store = side.limiter.batch_enqueued_token_store
        counters = [await store.reserve(10**15, (scope,)) for scope in scopes]
        remote = side.remote
        if remote is None:
            return counters
        records = []
        for batch_id in batch_ids:
            raw = remote.get(f"batch_enqueued_token_reservation:{batch_id}")
            record = {} if not raw else _RECORD.validate_json(str(raw))
            parsed = {name: value for name, value in record.items() if name != "reserved_at_monotonic"}
            records.append((parsed, remote.ttl(f"batch_enqueued_token_reservation:{batch_id}") // 10))
        keys = sorted(str(key) for key in remote.scan_iter("batch_enqueued_tokens:*"))
        return counters, records, [(key, remote.get(key), remote.ttl(key) // 10) for key in keys]

    return probe


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ("success", "failure", "proxy_failure"))
@pytest.mark.parametrize(
    "limits",
    (
        {"metadata": {"batch_enqueued_token_limit": 1500}},
        {
            "metadata": {"batch_enqueued_token_limit": 3000},
            "team_id": _TEAM_ENQUEUED,
            "team_metadata": {"batch_enqueued_token_limit": 1000},
            "rpm_limit": 50,
        },
    ),
)
async def test_batch_enqueued_tokens_are_reserved_saved_and_refunded_like_python(
    end: str,
    limits: dict[str, object],
    backend: tuple[int | None, int | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _batch_usage(monkeypatch, total_tokens=600, request_count=6)
    auth: Final = _auth(**limits)

    def submit(batch_id: str) -> Request:
        return Request(auth, _BATCH, end=end, call_type="acreate_batch", response=_batch_object(batch_id, "validating"))

    def retrieve(batch_id: str, status: str) -> Request:
        return Request(
            auth, {"batch_id": batch_id}, call_type="aretrieve_batch", response=_batch_object(batch_id, status)
        )

    batch_ids: Final = ("batch_a", "batch_b", "batch_c", "batch_d")
    observed: Final = await _differential(
        [
            submit("batch_a"),
            submit("batch_b"),
            submit("batch_c"),
            retrieve("batch_a", "in_progress"),
            retrieve("batch_a", "Completed"),
            submit("batch_d"),
            retrieve("batch_a", "completed"),
            retrieve("batch_b", "cancelled"),
        ],
        [("api_key", _KEY)],
        backend,
        probe=_enqueued_probe(batch_ids),
    )
    assert observed[0] is None
    assert (observed[2] is not None) == (end != "proxy_failure")
    assert observed[2] is None or "enqueued token limit" in str(observed[2])


_PASS_THROUGH: Final = {"model": "gpt-4o", "messages": [{"role": "user", "content": "decide on this case please"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_type", (EndpointType.GENERIC, EndpointType.DECISIONS))
@pytest.mark.parametrize("end", ("success", "failure", "proxy_failure"))
@pytest.mark.parametrize(
    "limits",
    (
        {"tpm_limit": 100, "rpm_limit": 3},
        {"max_parallel_requests": 1, "tpm_limit": 5000},
        {
            "project_id": "p1",
            "project_metadata": {"model_otpm_limit": {"gpt-4o": 600}, "model_tpm_limit": {"gpt-4o": 900}},
        },
    ),
)
async def test_pass_through_requests_are_limited_natively_like_python(
    endpoint_type: EndpointType,
    end: str,
    limits: dict[str, object],
    backend: tuple[int | None, int | None],
) -> None:
    auth: Final = _auth(**limits)
    observed: Final = await _differential(
        [
            Request(auth, _PASS_THROUGH, end=end, call_type="pass_through_endpoint", endpoint_type=endpoint_type)
            for _ in range(5)
        ],
        [("api_key", _KEY), *_PROJECT_SCOPES],
        backend,
    )
    assert observed[0] is None


_REALTIME_EVENTS: Final = [
    {"type": "session.created", "session": {"id": "sess_1"}},
    {
        "type": "response.done",
        "response": {"usage": {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42}},
    },
]


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ("success", "failure", "partial_failure", "proxy_failure", "disconnect"))
@pytest.mark.parametrize(
    "limits",
    (
        {"tpm_limit": 200, "rpm_limit": 4},
        {"max_parallel_requests": 1, "tpm_limit": 5000},
        {
            "project_id": "p1",
            "project_metadata": {"model_itpm_limit": {"gpt-4o": 400}, "model_tpm_limit": {"gpt-4o": 900}},
        },
    ),
)
async def test_realtime_sessions_are_limited_natively_like_python(
    end: str, limits: dict[str, object], backend: tuple[int | None, int | None]
) -> None:
    auth: Final = _auth(**limits)
    observed: Final = await _differential(
        [
            Request(
                auth,
                {"model": "gpt-4o"},
                end=end,
                call_type="_arealtime",
                response=list(_REALTIME_EVENTS),
                held=index == 0,
            )
            for index in range(5)
        ],
        [("api_key", _KEY), *_PROJECT_SCOPES],
        backend,
    )
    assert observed[0] is None


_WS_EVENTS: Final = [
    {"type": "response.created", "response": {"id": "resp_1"}},
    {
        "type": "response.completed",
        "response": {"id": "resp_1", "usage": {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42}},
    },
]


def _turn_kwargs(turn: int, auth: UserAPIKeyAuth, failed: bool) -> dict[str, object]:
    from litellm.constants import RESPONSES_WEBSOCKET_CALL_ID_METADATA_KEY

    return {
        "litellm_call_id": f"turn-{turn}",
        "call_type": CallTypes.aresponses_websocket.value,
        "litellm_params": {
            "metadata": {"model_group": "gpt-4o"},
            "litellm_metadata": {RESPONSES_WEBSOCKET_CALL_ID_METADATA_KEY: "call-0", "model_group": "gpt-4o"},
        },
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": auth.api_key,
                "user_api_key_team_id": auth.team_id,
                "user_api_key_project_id": auth.project_id,
            },
            "hidden_params": {},
        },
        **({"combined_usage_object": _CACHED_USAGE} if failed else {}),
    }


async def _socket_turn(side: Side, turn: int, auth: UserAPIKeyAuth) -> object:
    frame = await _guarded(
        side.limiter.enforce_token_limits_for_frame(user_api_key_dict=auth, requested_model="gpt-4o")
    )
    failed = turn % 3 == 2
    kwargs = _turn_kwargs(turn, auth, failed)
    if failed:
        await side.limiter.async_log_failure_event(kwargs, None, None, None)
    else:
        await side.limiter.async_log_success_event(kwargs, list(_WS_EVENTS), None, None)
    return frame


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ("success", "disconnect"))
@pytest.mark.parametrize(
    ("limits", "ceiling"),
    (
        ({"tpm_limit": 150, "rpm_limit": 20, "max_parallel_requests": 1}, None),
        ({"team_id": "t1", "tpm_limit": 5000}, _CEILING),
        (
            {
                "project_id": "p1",
                "project_metadata": {"model_itpm_limit": {"gpt-4o": 400}, "model_tpm_limit": {"gpt-4o": 120}},
            },
            None,
        ),
    ),
)
async def test_responses_websocket_turns_are_settled_natively_like_python(
    end: str,
    limits: dict[str, object],
    ceiling: PTUTeamCeiling | None,
    backend: tuple[int | None, int | None],
) -> None:
    clock: Final = Clock()
    sides: Final = (
        _side(clock, native=False, port=backend[0], ceiling=ceiling),
        _side(clock, native=True, port=backend[1], ceiling=ceiling),
    )
    auth: Final = _auth(**limits)
    connection: Final = Request(auth, {"model": "gpt-4o"}, call_type=CallTypes.aresponses_websocket.value, end=end)
    scopes: Final = [("api_key", _KEY), ("model_per_team_ptu", "t1:gpt-4o"), *_PROJECT_SCOPES]
    admitted: Final = [await _run(side, 0, _admit, connection) for side in sides]
    assert admitted[0] == admitted[1] == None
    assert sides[0].datas[0] == sides[1].datas[0]
    frames: Final[list[object]] = []
    for turn in range(6):
        outcomes = [
            await asyncio.get_running_loop().create_task(_socket_turn(side, turn, auth), context=side.contexts[0])
            for side in sides
        ]
        assert outcomes[0] == outcomes[1], (turn, outcomes)
        assert _snapshot(sides[0], scopes) == _snapshot(sides[1], scopes), turn
        frames.append(outcomes[0])
    ended: Final = [await _run(side, 0, _end, connection) for side in sides]
    assert ended[0] == ended[1]
    assert _snapshot(sides[0], scopes) == _snapshot(sides[1], scopes)
    assert frames[0] is None


_AUTH_VARIANTS: Final[tuple[Mapping[str, object], ...]] = (
    {"rpm_limit": 10, "tpm_limit": 1000, "max_parallel_requests": 3},
    {"rpm_limit": 10, "tpm_limit": 1000, "budget_throttle_pct": 0.37},
    {"rpm_limit": 1, "tpm_limit": 2, "budget_throttle_pct": 0.01},
    {"max_parallel_requests": 0},
    {"api_key": "", "rpm_limit": 4, "user_id": "u1", "user_rpm_limit": 2},
    {"user_id": "u1", "user_tpm_limit": 300, "team_member_rpm_limit": 5},
    {"user_id": "", "team_member_rpm_limit": 5, "user_rpm_limit": 1},
    {"user_id": "u1", "team_id": "t1", "team_member_tpm_limit": 70, "team_rpm_limit": 9, "team_tpm_limit": 900},
    {"team_id": "t1", "team_tpm_limit": 0},
    {"end_user_id": "e1", "end_user_rpm_limit": 3, "end_user_tpm_limit": None},
    {"end_user_id": "", "end_user_rpm_limit": 3},
    {"org_id": "o1", "organization_rpm_limit": 8, "organization_tpm_limit": 800},
    {"org_id": "", "organization_tpm_limit": 50},
    {"org_id": "o1"},
    {},
)


@pytest.mark.asyncio
@pytest.mark.parametrize("limits", _AUTH_VARIANTS)
@pytest.mark.parametrize("rpm_limit_type", (None, "dynamic", "guaranteed_throughput"))
@pytest.mark.parametrize("tpm_limit_type", (None, "dynamic"))
@pytest.mark.parametrize("model_has_failures", (False, True))
async def test_auth_descriptors_are_built_in_rust_like_python(
    limits: Mapping[str, object], rpm_limit_type: str | None, tpm_limit_type: str | None, model_has_failures: bool
) -> None:
    side: Final = _side(Clock(), native=True)
    assert side.native is not None
    auth: Final = UserAPIKeyAuth(**{"api_key": _KEY, **limits})  # pyright: ignore[reportArgumentType]  # pydantic fields by keyword
    leading, organization = side.native._auth_descriptors(auth)(rpm_limit_type, tpm_limit_type, model_has_failures)  # pyright: ignore[reportPrivateUsage]  # the arithmetic under test
    assert leading == side.limiter._leading_rate_limit_descriptors(  # pyright: ignore[reportPrivateUsage]  # the Python reference
        auth, rpm_limit_type, tpm_limit_type, model_has_failures
    )
    assert side.limiter.create_organization_rate_limit_descriptor(auth, None, organization) == (
        side.limiter.create_organization_rate_limit_descriptor(auth, None)
    )


_CAP_BODIES: Final[tuple[tuple[str, Mapping[str, object]], ...]] = (
    ("acompletion", _CHAT),
    ("acompletion", {**_CHAT, "max_tokens": 0}),
    ("acompletion", {**_CHAT, "max_completion_tokens": "12"}),
    ("aresponses", {"model": "gpt-4o", "input": "hi"}),
    ("aresponses", {"model": "gpt-4o", "input": "hi", "max_output_tokens": 3}),
    ("aembedding", {"model": "text-embedding-3-small", "input": "hi"}),
    ("agenerate_content", {"model": "gemini-2.5-pro", "contents": [{"parts": [{"text": "hi"}]}]}),
    ("agenerate_content", {"model": "gemini-2.5-pro", "contents": [], "generationConfig": {"temperature": 1}}),
    ("", {**_CHAT, "input": "x"}),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("call_type", "body"), _CAP_BODIES)
@pytest.mark.parametrize("min_limit", (None, 1, 40, 100, 4095, 100_000))
@pytest.mark.parametrize("configured_output_tokens", (None, 0, 60, 5000))
@pytest.mark.parametrize("endpoint_type", (EndpointType.GENERIC, EndpointType.DECISIONS))
async def test_the_implicit_output_cap_is_decided_in_rust_like_python(
    call_type: str,
    body: Mapping[str, object],
    min_limit: int | None,
    configured_output_tokens: int | None,
    endpoint_type: EndpointType,
) -> None:
    side: Final = _side(Clock(), native=True)
    assert side.native is not None
    python: Final = dict(body)
    native: Final = dict(body)
    PROXY_MaxParallelRequestsHandler_v3._apply_implicit_output_cap(  # pyright: ignore[reportPrivateUsage]  # the Python reference
        python, min_limit, call_type, configured_output_tokens, endpoint_type
    )
    side.native._apply_output_cap(native, min_limit, call_type, endpoint_type, configured_output_tokens)  # pyright: ignore[reportPrivateUsage]  # the arithmetic under test
    assert native == python


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limits",
    (
        {"rpm_limit": 6, "tpm_limit": 900, "budget_throttle_pct": 0.5},
        {"rpm_limit": 3, "tpm_limit": 400, "metadata": {"rpm_limit_type": "dynamic", "tpm_limit_type": "dynamic"}},
        {"org_id": "o1", "organization_rpm_limit": 3, "organization_tpm_limit": 500, "end_user_id": "e1"},
        {"user_id": "u1", "team_member_tpm_limit": 120, "end_user_id": "e1", "end_user_rpm_limit": 2},
    ),
)
async def test_auth_descriptors_from_rust_limit_requests_like_python(
    limits: dict[str, object], backend: tuple[int | None, int | None]
) -> None:
    auth: Final = _auth(**limits)
    observed: Final = await _differential(
        [Request(auth, _CHAT) for _ in range(4)],
        [("api_key", _KEY), ("organization", "o1"), ("team_member", "None:u1"), ("end_user", "e1")],
        backend,
    )
    assert observed[0] is None
