from __future__ import annotations

import gc
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Final, TypeAlias

import pytest

import litellm
from litellm.caching import DualCache
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.native import WireManifest
from litellm.integrations.custom_logger_v2.outcomes import Abort
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking import native_budgets
from litellm.proxy.spend_tracking.budget_reservation import (
    estimate_request_max_cost,
    reconcile_budget_reservation,
    release_budget_reservation,
    release_budget_reservation_on_cancel,
    release_or_invalidate_budget_reservation,
    release_unbound_budget_reservation,
    reserve_budget_for_request,
)
from litellm.proxy.utils import ProxyLogging
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore

ROUTE: Final = "/chat/completions"
BODY: Final = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1000}
TOKEN: Final = "hashed-native"
ESTIMATE: Final = estimate_request_max_cost(request_body=BODY, route=ROUTE, llm_router=None, input_token_counts={})
KEY: Final = f"spend:key:{TOKEN}"
TEAM: Final = "spend:team:team-1"
USER: Final = "spend:user:user-1"
END_USER: Final = "spend:end_user:customer-1"


@dataclass(frozen=True, slots=True)
class Identity:
    key_budget: float | None = 100.0
    team_budget: float | None = None
    user_budget: float | None = None
    end_user_budget: float | None = None
    throttle: bool = False
    fail_closed: bool = False
    spend: Mapping[str, float] = field(default_factory=dict)

    def token(self) -> UserAPIKeyAuth:
        return UserAPIKeyAuth(
            api_key="sk-native",
            token=TOKEN,
            max_budget=self.key_budget,
            spend=0.0,
            team_id="team-1" if self.team_budget is not None else None,
            user_id="user-1",
            end_user_id="customer-1" if self.end_user_budget is not None else None,
            end_user_max_budget=self.end_user_budget,
            tpm_limit=1000 if self.throttle else None,
            metadata={"throttle_on_budget_exceeded": True} if self.throttle else {},
        )

    def team(self) -> LiteLLM_TeamTable | None:
        return None if self.team_budget is None else LiteLLM_TeamTable(team_id="team-1", max_budget=self.team_budget)

    def user(self) -> LiteLLM_UserTable | None:
        return None if self.user_budget is None else LiteLLM_UserTable(user_id="user-1", max_budget=self.user_budget)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[SpendCounterStore]:
    spend_counter_store: Final = SpendCounterStore(connect(None).spend_counters())
    monkeypatch.setattr(proxy_server, "spend_counter_store", spend_counter_store)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(litellm, "budget_throttle_pct", 50, raising=False)
    yield spend_counter_store
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # later tests run without a native owner


def _seed(store: SpendCounterStore, identity: Identity) -> None:
    for counter, value in identity.spend.items():
        store.set_local(counter, value)


def _snapshot(store: SpendCounterStore) -> Mapping[str, float]:
    return {key: round(float(value), 12) for key, value in store.snapshot().items()}


def _arguments(identity: Identity) -> Mapping[str, object]:
    return {
        "request_body": dict(BODY),
        "route": ROUTE,
        "llm_router": None,
        "valid_token": identity.token(),
        "team_object": identity.team(),
        "user_object": identity.user(),
        "prisma_client": None,
        "user_api_key_cache": UserApiKeyCache(),
        "proxy_logging_obj": ProxyLogging(user_api_key_cache=DualCache()),
        "end_user_id": "customer-1" if identity.end_user_budget is not None else None,
        "fail_closed_budget_enforcement": identity.fail_closed,
    }


async def _no_attempt_line(_status: str, _response_cost: float) -> None:
    return None


@dataclass(frozen=True, slots=True)
class Admitted:
    reservation: dict[str, object] | None  # mutable-ok: settlement stamps finalized on the shared reservation
    refusal: Exception | None
    counters: Mapping[str, float]
    attempt_ended: Callable[[str, float], Awaitable[None]] = _no_attempt_line


def _install_native_owner(store: SpendCounterStore) -> native_budgets.NativeBudgetsLogger:
    callbacks: Final[list[object]] = []  # mutable-ok: install_native_budgets inserts into litellm.callbacks
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # each case installs its own owner over its own store
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(native_budgets.NATIVE_BUDGETS_ENV, "true")
        native_budgets.install_native_budgets(store, callbacks)
    logger: Final = callbacks[0]
    assert isinstance(logger, native_budgets.NativeBudgetsLogger)
    return logger


async def _python_admit(store: SpendCounterStore, identity: Identity) -> Admitted:
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # the reference run has no native owner
    _seed(store, identity)
    try:
        reservation: Final = await reserve_budget_for_request(**_arguments(identity))  # pyright: ignore[reportArgumentType]  # the shared argument table is typed per key above
    except litellm.BudgetExceededError as error:
        return Admitted(reservation=None, refusal=error, counters=_snapshot(store))
    return Admitted(reservation=reservation, refusal=None, counters=_snapshot(store))


async def _native_admit(store: SpendCounterStore, identity: Identity, call_id: str) -> Admitted:
    logger: Final = _install_native_owner(store)
    _seed(store, identity)
    plan: Final = await reserve_budget_for_request(**_arguments(identity))  # pyright: ignore[reportArgumentType]  # the shared argument table is typed per key above
    data: Final = {**BODY, "litellm_call_id": call_id}
    view: Final = type("View", (), {Field.deprecated_kwargs.value: data})()
    state_type: Final = type(logger).request_state
    assert state_type is not None
    state: Final = state_type()
    outcome: Final = await logger.run_native(
        Hook.pre_call,
        frozenset({Field.deprecated_kwargs}),
        view,
        CallContext[object](hook=Hook.pre_call, call=None, state=state),
    )

    subscribed: Final = frozenset(
        subscription.hook for subscription in WireManifest.model_validate_json(logger.native.manifest()).subscriptions
    )

    async def attempt_ended(status: str, response_cost: float) -> None:
        if Hook.attempt_ended not in subscribed:
            return
        payload: Final = {"status": status, "response_cost": response_cost, "id": call_id}
        _ = await logger.run_native(
            Hook.attempt_ended,
            frozenset({Field.standard_logging_payload}),
            type("View", (), {Field.standard_logging_payload.value: payload})(),
            CallContext[object](hook=Hook.attempt_ended, call=None, state=state),
        )

    refusal: Final = outcome.exception if isinstance(outcome, Abort) else None
    reservation: Final = plan if plan is not None and plan.get("native_call_id") is not None else None
    return Admitted(reservation=reservation, refusal=refusal, counters=_snapshot(store), attempt_ended=attempt_ended)


def _refusal_fields(error: Exception | None) -> object:
    if error is None:
        return None
    return (
        type(error),
        str(error),
        {name: getattr(error, name, None) for name in ("current_cost", "max_budget", "entity_type", "entity_id")},
        getattr(error, "status_code", None),
    )


def _entries(reservation: Mapping[str, object] | None) -> object:
    if reservation is None:
        return None
    return (
        reservation["reserved_cost"],
        reservation["entries"],
        reservation["input_cost"],
        reservation["input_tokens"],
    )


assert ESTIMATE is not None and ESTIMATE > 0

CASES: Final = {
    "key_and_team_fit": Identity(team_budget=100.0, spend={KEY: 1.0, TEAM: 2.0}),
    "team_shrinks_to_what_is_left": Identity(team_budget=10.0, spend={KEY: 1.0, TEAM: 10.0 - ESTIMATE / 3}),
    "key_spent_refuses": Identity(key_budget=5.0, spend={KEY: 5.0}),
    "team_at_max_refuses": Identity(team_budget=10.0, spend={KEY: 0.0, TEAM: 10.0}),
    "user_on_a_personal_key": Identity(user_budget=50.0, spend={KEY: 1.0, USER: 3.0}),
    "end_user_over_budget": Identity(end_user_budget=2.0, spend={KEY: 1.0, END_USER: 2.0}),
    "throttled_key_releases_its_hold": Identity(key_budget=5.0, team_budget=100.0, throttle=True, spend={KEY: 5.0}),
    "fail_closed_refuses_any_overshoot": Identity(
        team_budget=10.0, fail_closed=True, spend={KEY: 0.0, TEAM: 10.0 - ESTIMATE / 3}
    ),
    "no_budget_reserves_nothing": Identity(key_budget=None),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", CASES.values(), ids=CASES.keys())
async def test_native_admission_matches_python(store: SpendCounterStore, identity: Identity) -> None:
    python: Final = await _python_admit(store, identity)
    store.flush_local()
    native: Final = await _native_admit(store, identity, "call-admit")

    assert _refusal_fields(native.refusal) == _refusal_fields(python.refusal)
    assert native.counters == python.counters
    assert _entries(native.reservation) == _entries(python.reservation)


Settle: TypeAlias = Callable[[dict[str, object]], Awaitable[object]]

SETTLEMENTS: Final = {
    "success_charges_the_actual_cost": lambda reservation: reconcile_budget_reservation(reservation, 0.002),
    "failure_refunds_the_hold": release_or_invalidate_budget_reservation,
    "cancel_keeps_the_input_cost": release_budget_reservation_on_cancel,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("settle", SETTLEMENTS.values(), ids=SETTLEMENTS.keys())
async def test_native_settlement_matches_python(
    store: SpendCounterStore,
    settle: Settle,
) -> None:
    identity: Final = Identity(team_budget=100.0, spend={KEY: 1.0, TEAM: 2.0})
    python: Final = await _python_admit(store, identity)
    assert python.reservation is not None
    await settle(python.reservation)
    await settle(python.reservation)
    settled_python: Final = _snapshot(store)
    store.flush_local()

    native: Final = await _native_admit(store, identity, "call-settle")
    assert native.reservation is not None
    await settle(native.reservation)
    await settle(native.reservation)

    assert _snapshot(store) == settled_python
    assert settled_python != python.counters


async def _db_logger_success(reservation: dict[str, object]) -> None:  # mutable-ok: the shared reservation
    _ = await reconcile_budget_reservation(reservation, 0.002, finalize=False, apply_consistent=False)
    await proxy_server.increment_spend_counters(
        token=TOKEN, team_id="team-1", user_id="user-1", response_cost=0.002, budget_reservation=reservation
    )


@dataclass(frozen=True, slots=True)
class Exit:
    attempt_status: str | None
    settle: Settle


EXITS: Final = {
    "db_logger_charges_the_row": Exit("success", _db_logger_success),
    "db_logger_releases_an_uncharged_row": Exit("success", release_budget_reservation),
    "db_logger_releases_after_a_failed_write": Exit("success", release_or_invalidate_budget_reservation),
    "failure_hook_releases": Exit("failure", release_or_invalidate_budget_reservation),
    "client_disconnect_keeps_the_input_cost": Exit(None, release_budget_reservation_on_cancel),
    "request_end_releases_an_unbound_hold": Exit(None, release_unbound_budget_reservation),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_path", EXITS.values(), ids=EXITS.keys())
async def test_every_exit_settles_the_native_hold_like_python(store: SpendCounterStore, exit_path: Exit) -> None:
    identity: Final = Identity(team_budget=100.0, user_budget=100.0, spend={KEY: 1.0, TEAM: 2.0, USER: 0.5})
    python: Final = await _python_admit(store, identity)
    assert python.reservation is not None
    await exit_path.settle(python.reservation)
    settled_python: Final = _snapshot(store)
    store.flush_local()

    native: Final = await _native_admit(store, identity, "call-exit")
    assert native.reservation is not None
    if exit_path.attempt_status is not None:
        await native.attempt_ended(exit_path.attempt_status, 0.002)
    await exit_path.settle(native.reservation)

    assert _snapshot(store) == settled_python
    assert native.reservation.get("finalized") == python.reservation.get("finalized")


@pytest.mark.asyncio
async def test_a_stream_settles_after_its_request_state_is_gone(store: SpendCounterStore) -> None:
    identity: Final = Identity(team_budget=100.0, spend={KEY: 1.0, TEAM: 2.0})
    python: Final = await _python_admit(store, identity)
    assert python.reservation is not None
    await _db_logger_success(python.reservation)
    settled_python: Final = _snapshot(store)
    store.flush_local()

    native: Final = await _native_admit(store, identity, "call-stream")
    assert native.reservation is not None
    reservation: Final = native.reservation
    del native
    gc.collect()
    await _db_logger_success(reservation)

    assert _snapshot(store) == settled_python


@pytest.mark.asyncio
async def test_a_native_module_without_budgets_leaves_the_flag_off_proxy_reserving_in_python(
    store: SpendCounterStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.rust_bridge import _native

    monkeypatch.delattr(_native, "NativeBudgets")
    monkeypatch.delattr(_native, "budget_check_decision")
    monkeypatch.delenv(native_budgets.NATIVE_BUDGETS_ENV, raising=False)
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # start from a proxy with no native owner
    callbacks: Final[list[object]] = []  # mutable-ok: install_native_budgets inserts into litellm.callbacks
    native_budgets.install_native_budgets(store, callbacks)

    reservation: Final = await reserve_budget_for_request(**_arguments(Identity()))  # pyright: ignore[reportArgumentType]  # the shared argument table is typed per key above

    assert callbacks == []
    assert native_budgets.native_budget_decision({"check": "proxy", "spend": 1.0, "max_budget": 0.5}) is None
    assert reservation is not None and "native_call_id" not in reservation
    held: Final = _snapshot(store)
    entries: Final = reservation["entries"]
    assert isinstance(entries, list) and entries
    assert {entry["counter_key"]: held[entry["counter_key"]] for entry in entries} == {
        entry["counter_key"]: pytest.approx(reservation["reserved_cost"]) for entry in entries
    }
