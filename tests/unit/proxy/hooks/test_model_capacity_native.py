import asyncio
import contextvars
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Final, Protocol

import pytest
from fastapi import HTTPException

import litellm
from litellm import DualCache, Router
from litellm.integrations.custom_logger_v2.call import CallFacts, CallIds, CallKind, CallTarget, CallTiming
from litellm.integrations.custom_logger_v2.enums import Hook
from litellm.integrations.custom_logger_v2.host import request_scope, reset_host_for_tests
from litellm.integrations.custom_logger_v2.legacy import (
    PROXY_REQUEST_KEYS,
    Admit,
    KwargsFields,
    clear_subscribers_cache_for_tests,
    hook_subscribers,
    kwargs_after,
    run_subscribers,
)
from litellm.litellm_core_utils.litellm_logging import LOGGING_KEYS
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import AlternateGrant, PROXY_DynamicRateLimitHandlerV3
from litellm.proxy.hooks.model_capacity_native import NativeDynamicRateLimiter, build_native_dynamic_rate_limiter
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROXY_MaxParallelRequestsHandler_v3,
    get_or_create_request_stash,
)
from litellm.proxy.utils import HookRun, hook_fields, raise_v2_abort
from litellm.integrations.custom_logger_v2.enums import Field
from litellm.redis.stores.rate_limits import RateLimitScope
from litellm.types.utils import ModelResponse, PriorityReservationSettings, Usage

pytestmark = pytest.mark.asyncio


class Clock:
    def __init__(self) -> None:
        self.current = datetime(2026, 11, 13, 12, 0, 0)

    def now(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> Clock:
    reset_host_for_tests()
    clear_subscribers_cache_for_tests()
    clock: Final = Clock()
    monkeypatch.setattr(time, "time", lambda: clock.now().timestamp())
    monkeypatch.setenv("LITELLM_LICENSE", "test-license")
    monkeypatch.setattr(litellm, "priority_reservation", None)
    monkeypatch.setattr(litellm, "priority_reservation_settings", PriorityReservationSettings())
    return clock


async def _grant_everything(_auth: UserAPIKeyAuth, model: str) -> AlternateGrant | None:
    return AlternateGrant(access_groups=(f"group-{model}",))


@dataclass
class Outcome:
    raised: tuple[object, ...] | None = None
    model: object = None
    metadata: object = None
    max_tokens: object = None
    statuses: object = None


def _raised(error: BaseException) -> tuple[object, ...]:
    if isinstance(error, HTTPException):
        return (
            type(error).__name__,
            error.status_code,
            error.detail,
            error.headers,
            getattr(error, "rate_limit_type", None),
            getattr(error, "model", None),
            getattr(error, "llm_provider", None),
        )
    return (type(error).__name__, str(error))


def _outcome(data: Mapping[str, object]) -> Outcome:
    metadata: Final = data.get("metadata")
    stash: Final = get_or_create_request_stash()
    response: Final = stash.rate_limit_response
    return Outcome(
        model=data.get("model"),
        metadata=None
        if not isinstance(metadata, Mapping)
        else {k: v for k, v in metadata.items() if k != "user_api_key_auth"},
        max_tokens=(data.get("max_tokens"), data.get("max_output_tokens")),
        statuses=None if response is None else response["statuses"],
    )


class Side(Protocol):
    limiter: PROXY_MaxParallelRequestsHandler_v3

    def pre_call(self, data: dict[str, object], call_type: str) -> Awaitable[Outcome]: ...

    def post_call(self, data: dict[str, object], response: ModelResponse) -> Awaitable[object]: ...

    def response_headers(self, data: dict[str, object]) -> Awaitable[object]: ...

    def ended(self, kwargs: dict[str, object], response: object, failed: bool) -> Awaitable[None]: ...

    def fallback(
        self, auth: UserAPIKeyAuth, model: str, data: dict[str, object], commit: bool
    ) -> Awaitable[object]: ...


def _auth(data: Mapping[str, object]) -> UserAPIKeyAuth:
    metadata: Final = data["metadata"]
    assert isinstance(metadata, Mapping)
    auth: Final = metadata["user_api_key_auth"]
    assert isinstance(auth, UserAPIKeyAuth)
    return auth


class PythonSide:
    def __init__(self, router: Router, clock: Clock) -> None:
        self.handler: Final = PROXY_DynamicRateLimitHandlerV3(
            internal_usage_cache=DualCache(), time_provider=clock.now, is_model_authorized=_grant_everything
        )
        self.handler.update_variables(llm_router=router)
        self.limiter = self.handler.v3_limiter

    async def pre_call(self, data: dict[str, object], call_type: str) -> Outcome:
        try:
            returned = await self.handler.async_pre_call_hook(_auth(data), DualCache(), data, call_type)  # pyright: ignore[reportArgumentType]
        except Exception as error:
            return Outcome(raised=_raised(error))
        return _outcome(returned if isinstance(returned, dict) else data)

    async def post_call(self, data: dict[str, object], response: ModelResponse) -> object:
        await self.handler.async_post_call_success_hook(data=data, user_api_key_dict=_auth(data), response=response)
        return dict(response._hidden_params.get("additional_headers") or {})

    async def response_headers(self, data: dict[str, object]) -> object:
        return await self.handler.async_post_call_response_headers_hook(data, _auth(data), None)

    async def ended(self, kwargs: dict[str, object], response: object, failed: bool) -> None:
        if failed:
            await self.handler.async_log_failure_event(kwargs, response, None, None)
        else:
            await self.handler.async_log_success_event(kwargs, response, None, None)

    async def fallback(self, auth: UserAPIKeyAuth, model: str, data: dict[str, object], commit: bool) -> object:
        try:
            return await self.handler.async_fallback_target_allowed(auth, model, data, commit=commit)
        except Exception as error:
            return _raised(error)


def _call(data: Mapping[str, object], call_type: str) -> CallFacts:
    return CallFacts(
        ids=CallIds(call_id=str(data.get("litellm_call_id") or "")),
        kind=CallKind(call_type=call_type),
        target=CallTarget(model=str(data.get("model"))),
        timing=CallTiming(start=1.0),
    )


async def _no_legacy(_logger: object, _data: object) -> None:
    return None


class NativeSide:
    def __init__(self, router: Router, clock: Clock) -> None:
        self.logger: Final[NativeDynamicRateLimiter] = build_native_dynamic_rate_limiter(
            DualCache(), is_model_authorized=_grant_everything, time_provider=clock.now
        )
        self.logger.update_variables(router)
        self.limiter = self.logger.v3_limiter
        self.callbacks: Final[list[object]] = [self.logger]

    async def pre_call(self, data: dict[str, object], call_type: str) -> Outcome:
        try:
            result = await run_subscribers(
                hook_subscribers(Hook.pre_call, self.callbacks),
                KwargsFields(data, PROXY_REQUEST_KEYS),
                _no_legacy,
                call=_call(data, call_type),
            )
            after = kwargs_after(raise_v2_abort(result))
        except Exception as error:
            return Outcome(raised=_raised(error))
        return _outcome(after if isinstance(after, dict) else data)

    async def post_call(self, data: dict[str, object], response: ModelResponse) -> object:
        await run_subscribers(
            hook_subscribers(Hook.post_call_success, self.callbacks),
            hook_fields(data, HookRun(response=response), Field.response),
            _no_legacy,
        )
        return dict(response._hidden_params.get("additional_headers") or {})

    async def response_headers(self, data: dict[str, object]) -> object:
        merged: Final[dict[str, str]] = {}
        await run_subscribers(
            hook_subscribers(Hook.response_headers, self.callbacks, admit=Admit.every_entry),
            hook_fields(data, HookRun(response=merged), Field.response_headers),
            _no_legacy,
        )
        return merged or None

    async def ended(self, kwargs: dict[str, object], response: object, failed: bool) -> None:
        await run_subscribers(
            hook_subscribers(Hook.attempt_ended, self.callbacks, admit=Admit.every_entry),
            KwargsFields(kwargs, LOGGING_KEYS),
            _no_legacy,
            call=_call(kwargs, "acompletion"),
        )

    async def fallback(self, auth: UserAPIKeyAuth, model: str, data: dict[str, object], commit: bool) -> object:
        try:
            return await self.logger.async_fallback_target_allowed(auth, model, data, commit=commit)
        except Exception as error:
            return _raised(error)


@dataclass
class Request:
    auth: UserAPIKeyAuth
    model: str
    call_id: str
    messages: Sequence[Mapping[str, object]] = ({"role": "user", "content": "hello there"},)
    call_type: str = "acompletion"
    extra: Mapping[str, object] = field(default_factory=dict)

    def data(self) -> dict[str, object]:
        return {
            "model": self.model,
            "messages": list(self.messages),
            "litellm_call_id": self.call_id,
            "metadata": {"user_api_key_auth": self.auth},
            **self.extra,
        }


def _kwargs(request: Request, model_group: str | None, tokens: tuple[int, int], failed: bool) -> dict[str, object]:
    auth: Final = request.auth
    return {
        "litellm_call_id": request.call_id,
        "call_type": request.call_type,
        "model": request.model,
        "litellm_params": {"metadata": {"model_group": model_group}},
        "standard_logging_object": {
            "status": "failure" if failed else "success",
            "prompt_tokens": tokens[0],
            "completion_tokens": tokens[1],
            "total_tokens": sum(tokens),
            "metadata": {
                "user_api_key_auth_metadata": auth.metadata,
                "user_api_key_team_id": auth.team_id,
                "user_api_key_team_alias": auth.team_alias,
            },
        },
        **({"exception": RuntimeError("provider down")} if failed else {}),
    }


def _response(tokens: tuple[int, int]) -> ModelResponse:
    return ModelResponse(usage=Usage(prompt_tokens=tokens[0], completion_tokens=tokens[1], total_tokens=sum(tokens)))


@dataclass
class Step:
    request: Request
    tokens: tuple[int, int] | None = (10, 20)
    served_group: str | None = None
    failed: bool = False
    partial: tuple[int, int] | None = None
    fallbacks: Sequence[tuple[str, bool]] = ()
    advance: float = 0.0


async def _run_step(side: Side, step: Step) -> list[object]:
    async def lifecycle() -> list[object]:
        data: Final = step.request.data()
        transcript: Final[list[object]] = []
        get_or_create_request_stash()
        with request_scope():
            outcome: Final = await side.pre_call(data, step.request.call_type)
            transcript.append(outcome)
            if outcome.raised is not None:
                return transcript
            transcript.extend(
                [await side.fallback(step.request.auth, model, data, commit) for model, commit in step.fallbacks]
            )
            if step.tokens is None:
                return transcript
            response: Final = _response(step.tokens)
            if not step.failed:
                transcript.append(await side.post_call(data, response))
            transcript.append(await side.response_headers(data))
            served: Final = step.served_group or (outcome.model if isinstance(outcome.model, str) else None)
            kwargs: Final = _kwargs(step.request, served, step.tokens, step.failed)
            if step.partial is not None:
                kwargs["combined_usage_object"] = Usage(
                    prompt_tokens=step.partial[0], completion_tokens=step.partial[1], total_tokens=sum(step.partial)
                )
            await side.ended(kwargs, None if step.failed else response, step.failed)
        return transcript

    return await asyncio.create_task(lifecycle(), context=contextvars.Context())


async def _counters(side: Side, scopes: Sequence[tuple[str, str]]) -> list[object]:
    return [
        (key, value, kind, await side.limiter.rate_limits.get(RateLimitScope(key, value).counter(kind)))
        for key, value in scopes
        for kind in ("requests", "tokens")
    ]


def _router(*groups: tuple[str, int | None, int | None]) -> Router:
    return Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "https://example.invalid",
                    **({} if tpm is None else {"tpm": tpm}),
                    **({} if rpm is None else {"rpm": rpm}),
                    "input_cost_per_token": 0.000001,
                    "output_cost_per_token": 0.000002,
                },
            }
            for name, tpm, rpm in groups
        ]
    )


async def _differential(
    monkeypatch: pytest.MonkeyPatch,
    clock: Clock,
    router: Router,
    steps: Sequence[Step],
    scopes: Sequence[tuple[str, str]],
) -> list[object]:
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    transcripts: Final[list[list[object]]] = []
    for build in (PythonSide, NativeSide):
        clock.current = datetime(2026, 11, 13, 12, 0, 0)
        side = build(router, clock)
        monkeypatch.setattr(litellm, "callbacks", [side.handler if isinstance(side, PythonSide) else side.logger])
        transcript: list[object] = []
        for step in steps:
            clock.advance(step.advance)
            transcript.append(await _run_step(side, step))
        transcript.append(await _counters(side, scopes))
        transcripts.append(transcript)
    python, native = transcripts
    for index, (expected, actual) in enumerate(zip(python, native, strict=True)):
        assert actual == expected, f"step {index} differs"
    return python


def _refusal_prefixes(transcript: Sequence[object]) -> list[str]:
    return [
        str(step[0].raised[2]["error"]).split(" for ")[0].split(".")[0]
        for step in transcript
        if isinstance(step, list) and isinstance(step[0], Outcome) and step[0].raised is not None
    ]


def _key(name: str, **fields: object) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key=f"sk-{name}", **fields)  # pyright: ignore[reportArgumentType]


async def test_the_group_is_held_to_its_deployments_requests(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    router: Final = _router(("capped", None, 3))
    key: Final = _key("a")
    steps: Final = [Step(Request(key, "capped", f"c{i}")) for i in range(5)]
    python: Final = await _differential(monkeypatch, isolated, router, steps, [("model_saturation_check", "capped")])
    refusals: Final = [step[0].raised for step in python[:5] if step[0].raised is not None]
    assert len(refusals) == 2


async def test_busy_groups_hold_priorities_to_their_share_and_tokens_settle(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
):
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.9, "low": 0.1})
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.3)
    router: Final = _router(("shared", 1000, 20))
    high: Final = _key("high", metadata={"priority": "high"})
    low: Final = _key("low", metadata={"priority": "low"})
    plain: Final = _key("plain")
    steps: Final = [
        *(Step(Request(high, "shared", f"h{i}"), tokens=(60, 40)) for i in range(4)),
        *(Step(Request(low, "shared", f"l{i}"), tokens=(10, 10)) for i in range(3)),
        *(Step(Request(plain, "shared", f"p{i}"), tokens=(10, 10)) for i in range(2)),
    ]
    python: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [
            ("model_saturation_check", "shared"),
            ("priority_model", "shared:high"),
            ("priority_model", "shared:low"),
            ("priority_model", "shared:default_pool"),
        ],
    )
    assert _refusal_prefixes(python) == ["Priority-based rate limit exceeded"], python


async def test_over_allocated_and_absolute_reservations_normalize_the_same_way(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
):
    monkeypatch.setattr(
        litellm,
        "priority_reservation",
        {"a": 0.6, "b": {"type": "rpm", "value": 8}, "c": {"type": "tpm", "value": 300}},
    )
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.0)
    router: Final = _router(("norm", 1000, 10))
    steps: Final = [
        Step(Request(_key(name, metadata={"priority": name}), "norm", f"{name}{i}"), tokens=(30, 30))
        for i in range(3)
        for name in ("a", "b", "c")
    ]
    await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [("model_saturation_check", "norm"), *(("priority_model", f"norm:{name}") for name in "abc")],
    )


async def test_team_priority_wins_and_headers_follow_it(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    monkeypatch.setattr(litellm, "priority_reservation", {"gold": 0.5})
    router: Final = _router(("hdr", 1000, 100))
    keys: Final = [
        _key("t", team_id="t1", team_metadata={"priority": "gold"}, metadata={"priority": "other"}),
        _key("u", metadata={"priority": "ünïcødé"}),
        _key("v", metadata={"priority": "日本"}),
    ]
    steps: Final = [Step(Request(key, "hdr", f"h{i}")) for i, key in enumerate(keys)]
    await _differential(monkeypatch, isolated, router, steps, [("priority_model", "hdr:gold")])


async def test_class_split_holds_each_active_team_to_its_slice(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    monkeypatch.setattr(litellm, "priority_reservation", {"interactive": 0.75, "batch": 0.25})
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.2)
    monkeypatch.setattr(
        litellm.priority_reservation_settings, "priority_class_share_splits_equally_among_active_teams", True
    )
    router: Final = _router(("split", 1000, None))
    teams: Final = [
        _key("a", team_id="team-a", team_metadata={"priority": "interactive"}),
        _key("b", team_id="team-b", team_metadata={"priority": "interactive"}),
        _key("c", team_id="team-c", team_metadata={"priority": "batch"}),
        _key("pool", metadata={"priority": "batch"}),
    ]
    steps: Final = [
        *(Step(Request(team, "split", f"w-{index}"), tokens=(50, 40)) for index, team in enumerate(teams)),
        *(Step(Request(teams[0], "split", f"a-{index}"), tokens=(100, 100)) for index in range(4)),
        *(
            Step(Request(team, "split", f"s{round_}-{index}"), tokens=(50, 40))
            for round_ in range(3)
            for index, team in enumerate(teams)
        ),
    ]
    python: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [
            ("model_saturation_check", "split"),
            ("priority_class_share_saturated", "5:split:team-a"),
            ("priority_class_share_saturated", "5:split:team-b"),
            ("priority_class_share_saturated", "5:split:team-c"),
            ("priority_model", "split:batch"),
        ],
    )
    assert "Priority class share reached" in _refusal_prefixes(python), python


def _borrowing(monkeypatch: pytest.MonkeyPatch, threshold: float) -> None:
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", threshold)
    monkeypatch.setattr(
        litellm.priority_reservation_settings, "team_model_limit_borrows_idle_capacity_from_other_teams", True
    )


async def test_borrowing_teams_hold_tokens_settle_and_refund(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    _borrowing(monkeypatch, 0.3)
    router: Final = _router(("borrow", 1000, None))
    team: Final = _key("team", team_id="team-a", team_metadata={"model_tpm_limit": {"borrow": 120}})
    capped: Final = _key(
        "capped",
        team_id="team-b",
        team_metadata={"model_tpm_limit": {"borrow": 60}, "team_model_limit_max_borrow_multiple": 1.5},
    )
    long_prompt: Final = ({"role": "user", "content": "word " * 40},)
    steps: Final = [
        Step(Request(team, "borrow", "b0", messages=long_prompt), tokens=(50, 30)),
        Step(Request(capped, "borrow", "b1"), tokens=(40, 30)),
        Step(Request(capped, "borrow", "b2", extra={"max_tokens": 500}), tokens=(40, 30)),
        Step(Request(team, "borrow", "b3"), tokens=(30, 0), failed=True, partial=(5, 7)),
        Step(Request(team, "borrow", "b4"), tokens=(30, 0), failed=True),
        Step(Request(team, "borrow", "b5", messages=long_prompt), tokens=(200, 200)),
        *(Step(Request(team, "borrow", f"r{i}"), tokens=(100, 100)) for i in range(4)),
        Step(Request(capped, "borrow", "b6"), tokens=(10, 10)),
        Step(Request(team, "borrow", "w0"), tokens=(10, 10), advance=61),
    ]
    python: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [
            ("model_saturation_check", "borrow"),
            ("model_per_team_saturated", "team-a:borrow"),
            ("model_per_team_saturated", "team-b:borrow"),
        ],
    )
    assert set(_refusal_prefixes(python)) == {
        "Team model_tpm_limit borrow cap reached",
        "Team model_tpm_limit reached",
    }, python


async def test_responses_calls_cap_their_output_the_same_way(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    _borrowing(monkeypatch, 0.0)
    router: Final = _router(("resp", 1000, None))
    team: Final = _key("team", team_id="team-r", team_metadata={"model_tpm_limit": {"resp": 80}})
    steps: Final = [
        Step(Request(team, "resp", "r0", call_type="aresponses", extra={"input": "short prompt"}), tokens=(5, 5)),
        Step(Request(team, "resp", "r1", call_type="aresponses", extra={"input": "short", "max_output_tokens": 9})),
    ]
    await _differential(monkeypatch, isolated, router, steps, [("model_per_team_saturated", "team-r:resp")])


async def test_saturated_teams_are_steered_to_an_idle_alternate(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)
    router: Final = _router(("luna", None, 4), ("opus", None, 4), ("pricey", None, 4))
    flexible: Final = _key(
        "flex",
        team_id="flex",
        team_models=["luna", "opus"],
        team_metadata={"model_preferences": [{"model": "luna", "alternates": ["luna", "opus"]}]},
    )
    plain: Final = _key("plain")
    steps: Final = [
        Step(Request(plain, "luna", "p0")),
        Step(Request(plain, "luna", "p1")),
        Step(Request(flexible, "luna", "f0")),
        Step(Request(flexible, "luna", "f1")),
        Step(Request(flexible, "luna", "f2")),
        Step(Request(flexible, "luna", "f3")),
        Step(Request(flexible, "luna", "f4")),
    ]
    python: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [("model_saturation_check", "luna"), ("model_saturation_check", "opus")],
    )
    assert [step[0].model for step in python[:7]] == ["luna", "luna", "opus", "opus", "luna", "luna", None], python


async def test_fallback_targets_are_held_to_the_teams_share_and_settle_there(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
):
    _borrowing(monkeypatch, 0.2)
    router: Final = _router(("primary", 1000, None), ("backup", 500, None))
    team: Final = _key("team", team_id="team-f", team_metadata={"model_tpm_limit": {"primary": 100, "backup": 60}})
    steps: Final = [
        Step(Request(team, "primary", "f0"), fallbacks=(("backup", False), ("backup", True)), served_group="backup"),
        Step(
            Request(team, "primary", "f1"),
            fallbacks=(("backup", False), ("backup", True)),
            served_group="backup",
            failed=True,
            partial=(3, 4),
        ),
        Step(Request(team, "primary", "f2"), tokens=(200, 200), served_group="backup", fallbacks=(("backup", True),)),
        Step(Request(team, "primary", "f3"), fallbacks=(("backup", False), ("backup", True), ("backup", True))),
        Step(
            Request(team, "primary", "f4", messages=({"role": "user", "content": "word " * 80},)),
            fallbacks=(("backup", False), ("backup", True)),
        ),
    ]
    python: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [
            ("model_saturation_check", "primary"),
            ("model_saturation_check", "backup"),
            ("model_per_team_saturated", "team-f:primary"),
            ("model_per_team_saturated", "team-f:backup"),
        ],
    )
    assert python[4][1:3] == [False, False], python[4]


async def test_requests_outside_the_limiter_pass_untouched(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    router: Final = _router(("known", 100, 2))
    key: Final = _key("k")
    steps: Final = [
        Step(Request(key, "unknown-model", "u0")),
        Step(Request(key, "known", "k0"), served_group=None),
        Step(Request(key, "known", "k1"), tokens=(0, 0)),
    ]
    await _differential(monkeypatch, isolated, router, steps, [("model_saturation_check", "known")])


async def test_bad_priority_configuration_fails_open_the_same_way(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    monkeypatch.setattr(litellm, "priority_reservation", {"a": "not a number"})
    router: Final = _router(("open", 100, 1))
    steps: Final = [Step(Request(_key("a", metadata={"priority": "a"}), "open", f"o{i}")) for i in range(3)]
    await _differential(monkeypatch, isolated, router, steps, [("model_saturation_check", "open")])


def _sides() -> Sequence[Callable[[Router, Clock], Side]]:
    return (PythonSide, NativeSide)
