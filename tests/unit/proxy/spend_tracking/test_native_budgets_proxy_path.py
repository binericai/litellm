from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias

import pytest
from starlette.requests import Request

import litellm
from litellm.proxy import proxy_server
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import cache_key_object
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
from litellm.proxy.spend_tracking import budget_reservation, native_budgets
from litellm.proxy.spend_tracking.budget_reservation import reconcile_budget_reservation
from litellm.proxy.utils import ProxyLogging, hash_token
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore

API_KEY: Final = "sk-native-path"
TOKEN: Final = hash_token(API_KEY)
KEY_COUNTER: Final = f"spend:key:{TOKEN}"
BODY: Final = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}


RequestData: TypeAlias = dict[str, object]


class _NoDatabase:
    pass


@dataclass(frozen=True, slots=True)
class Proxy:
    store: SpendCounterStore
    cache: UserApiKeyCache
    logging: ProxyLogging


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch) -> Iterator[Proxy]:
    store: Final = SpendCounterStore(connect(None).spend_counters())
    cache: Final = UserApiKeyCache()
    logging: Final = ProxyLogging(user_api_key_cache=cache)
    monkeypatch.setattr(proxy_server, "spend_counter_store", store)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", _NoDatabase())
    monkeypatch.setattr(proxy_server, "master_key", "sk-master")
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", logging)
    monkeypatch.setattr(litellm, "callbacks", [])
    yield Proxy(store=store, cache=cache, logging=logging)
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # later tests run without a native owner


def _use_native(proxy: Proxy, native: bool) -> None:
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # each run picks its own side
    litellm.callbacks.clear()
    proxy.store.flush_local()
    if native:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv(native_budgets.NATIVE_BUDGETS_ENV, "true")
            native_budgets.install_native_budgets(proxy.store, litellm.callbacks)


def _request() -> Request:
    raw: Final = json.dumps(BODY).encode()

    async def receive() -> Mapping[str, object]:
        return {"type": "http.request", "body": raw, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "root_path": "",
            "query_string": b"",
            "headers": [(b"authorization", f"Bearer {API_KEY}".encode()), (b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1),
            "server": ("proxy", 4000),
            "scheme": "http",
        },
        receive,
    )


async def _authenticate(proxy: Proxy, max_budget: float) -> UserAPIKeyAuth:
    await cache_key_object(
        hashed_token=TOKEN,
        user_api_key_obj=UserAPIKeyAuth(token=TOKEN, max_budget=max_budget, spend=0.0),
        user_api_key_cache=proxy.cache,
        proxy_logging_obj=None,
    )
    return await user_api_key_auth(
        request=_request(),
        api_key=f"Bearer {API_KEY}",
        azure_api_key_header="",
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        custom_litellm_key_header=None,
    )


async def _admit(
    proxy: Proxy, call_id: str
) -> tuple[UserAPIKeyAuth, dict[str, object]]:  # mutable-ok: the request data the proxy hands its hooks
    auth: Final = await _authenticate(proxy, max_budget=1.0)
    data: Final = {
        **BODY,
        "litellm_call_id": call_id,
        "metadata": {"user_api_key_budget_reservation": auth.budget_reservation},
    }
    _ = await proxy.logging.pre_call_hook(user_api_key_dict=auth, data=data, call_type="completion")
    return auth, data


async def _success(auth: UserAPIKeyAuth, _data: RequestData) -> None:
    reservation: Final = auth.budget_reservation
    _ = await reconcile_budget_reservation(reservation, 0.0007, finalize=False, apply_consistent=False)
    await proxy_server.increment_spend_counters(
        token=TOKEN, team_id=None, user_id=None, response_cost=0.0007, budget_reservation=reservation
    )


async def _failure(auth: UserAPIKeyAuth, data: RequestData) -> None:
    await _ProxyDBLogger().async_post_call_failure_hook(
        request_data=data, original_exception=RuntimeError("upstream failed"), user_api_key_dict=auth
    )


Exit: TypeAlias = Callable[[UserAPIKeyAuth, RequestData], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Run:
    entries_after_pre_call: object
    held_after_pre_call: Mapping[str, float]
    settled: Mapping[str, float]


async def _run(proxy: Proxy, native: bool, exit_path: Exit) -> Run:
    _use_native(proxy, native)
    auth, data = await _admit(proxy, "call-path")
    reservation: Final = auth.budget_reservation
    assert reservation is not None
    entries: Final = [
        {key: entry[key] for key in ("counter_key", "entity_type", "reserved_cost")}
        for entry in reservation["entries"]  # pyright: ignore[reportGeneralTypeIssues]  # the entries list the reservation carries
    ]
    held: Final = dict(proxy.store.snapshot())
    await exit_path(auth, data)
    return Run(entries_after_pre_call=entries, held_after_pre_call=held, settled=dict(proxy.store.snapshot()))


EXITS: Final[Mapping[str, Exit]] = {"success": _success, "failure": _failure}


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_path", EXITS.values(), ids=EXITS.keys())
async def test_a_request_through_auth_and_pre_call_settles_like_python(proxy: Proxy, exit_path: Exit) -> None:
    python: Final = await _run(proxy, native=False, exit_path=exit_path)
    native: Final = await _run(proxy, native=True, exit_path=exit_path)

    assert python.entries_after_pre_call, (
        "the flag-off run must hold a reservation for this comparison to mean anything"
    )
    assert native == python


@pytest.mark.asyncio
async def test_an_over_budget_key_is_refused_by_auth_like_python(proxy: Proxy) -> None:
    refusals: Final[list[tuple[object, ...]]] = []  # mutable-ok: one refusal per side
    for native in (False, True):
        _use_native(proxy, native)
        proxy.store.set_local(KEY_COUNTER, 2.0)
        with pytest.raises(ProxyException) as refused:
            await _authenticate(proxy, max_budget=1.0)
        refusals.append((refused.value.type, refused.value.code, refused.value.message, refused.value.param))

    assert refusals[0][0] == "budget_exceeded"
    assert refusals[1] == refusals[0]


RACERS: Final = 24


async def _race(proxy: Proxy, native: bool, budget: float) -> tuple[int, float]:
    _use_native(proxy, native)
    await cache_key_object(
        hashed_token=TOKEN,
        user_api_key_obj=UserAPIKeyAuth(token=TOKEN, max_budget=budget, spend=0.0),
        user_api_key_cache=proxy.cache,
        proxy_logging_obj=None,
    )

    async def one(index: int) -> bool:
        try:
            _, data = await _admit_with(proxy, budget, f"call-race-{index}")
        except (ProxyException, litellm.BudgetExceededError):
            return False
        metadata: Final = data["metadata"]
        assert isinstance(metadata, dict)
        reservation: Final = metadata["user_api_key_budget_reservation"]
        return isinstance(reservation, dict) and bool(reservation["entries"])

    admitted: Final = await asyncio.gather(*(one(index) for index in range(RACERS)))
    return sum(admitted), round(float(proxy.store.snapshot()[KEY_COUNTER]), 12)


async def _admit_with(
    proxy: Proxy, budget: float, call_id: str
) -> tuple[UserAPIKeyAuth, dict[str, object]]:  # mutable-ok: the request data the proxy hands its hooks
    auth: Final = await user_api_key_auth(
        request=_request(),
        api_key=f"Bearer {API_KEY}",
        azure_api_key_header="",
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        custom_litellm_key_header=None,
    )
    assert auth.max_budget == budget
    data: Final = {
        **BODY,
        "litellm_call_id": call_id,
        "metadata": {"user_api_key_budget_reservation": auth.budget_reservation},
    }
    _ = await proxy.logging.pre_call_hook(user_api_key_dict=auth, data=data, call_type="completion")
    return auth, data


@pytest.mark.asyncio
async def test_racing_requests_for_the_last_of_a_budget_admit_like_python(proxy: Proxy) -> None:
    estimate: Final = 0.00102
    budget: Final = estimate * 5.5

    python: Final = await _race(proxy, native=False, budget=budget)
    native: Final = await _race(proxy, native=True, budget=budget)

    assert python[0] < RACERS, "the budget must run out during the race"
    assert native == python


@pytest.mark.asyncio
async def test_the_reservation_lease_stops_when_the_reservation_is_finalized(
    proxy: Proxy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SpendCounterStore, "redis_ttl", property(lambda _store: 0.01))
    stopped: Final[list[bool]] = []  # mutable-ok: one result per side
    for native in (False, True):
        _use_native(proxy, native)
        before: Final = set(budget_reservation._lease_renewals)  # pyright: ignore[reportPrivateUsage]  # the running lease tasks are the observable
        auth, _ = await _admit(proxy, f"call-lease-{native}")
        leases: Final = set(budget_reservation._lease_renewals) - before  # pyright: ignore[reportPrivateUsage]  # the running lease tasks are the observable
        assert len(leases) == 1 and auth.budget_reservation is not None
        await asyncio.sleep(0.03)
        assert not next(iter(leases)).done()
        auth.budget_reservation["finalized"] = True
        _ = await asyncio.wait_for(asyncio.gather(*leases), timeout=1.0)
        stopped.append(all(lease.done() for lease in leases))

    assert stopped == [True, True]
