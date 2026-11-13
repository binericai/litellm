from __future__ import annotations

import asyncio
import os
import uuid
import weakref
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from fastapi import HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, TypeAdapter

import litellm
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.native import (
    NativeLogger,
    Projection,
    WireAbort,
    native_logger,
    register_abort_error,
)
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch
from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.rust_bridge._native import NativeBudgets

NATIVE_BUDGETS_ENV: Final = "LITELLM_NATIVE_BUDGETS"
NATIVE_CALL_ID_KEY: Final = "native_call_id"


@dataclass(slots=True)
class _PendingPlan:
    reservation: dict[str, object]  # mutable-ok: the shared reservation dict auth handed the request
    identity: UserAPIKeyAuth
    request_task: asyncio.Task[object] | None
    plan: Mapping[str, object] | None
    route: str
    request_data: Mapping[str, object]
    request: Request | None = None

    def claim(self) -> Mapping[str, object] | None:
        plan: Final = self.plan
        self.plan = None
        return plan


_pending_plan: Final[ContextVar[_PendingPlan | None]] = ContextVar("native_budget_plan", default=None)
_admitting: Final[ContextVar[Mapping[str, object] | None]] = ContextVar("native_budget_admission", default=None)
_refusals: Final[weakref.WeakKeyDictionary[Exception, _PendingPlan]] = weakref.WeakKeyDictionary()


def new_reservation_id() -> str:
    return uuid.uuid4().hex


def pending_native_reservation(
    plan: Mapping[str, object],
    identity: UserAPIKeyAuth,
    input_tokens: int | None,
    route: str,
    request_data: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: settlement stamps the reservation it returns
    reservation: Final[dict[str, object]] = {  # mutable-ok: settlement stamps the reservation it returns
        "reserved_cost": 0.0,
        "entries": [],
        "finalized": False,
        "callback_bound": False,
        "input_cost": 0.0,
        "input_tokens": input_tokens,
    }
    _ = _pending_plan.set(
        _PendingPlan(
            reservation=reservation,
            identity=identity.model_copy(),
            request_task=asyncio.current_task(),
            plan=plan,
            route=route,
            request_data={
                **request_data,
                **{
                    name: dict(bucket)
                    for name in ("metadata", "litellm_metadata")
                    if isinstance(bucket := request_data.get(name), Mapping)
                },
            },
        )
    )
    return reservation


_REFUSAL_TYPES: Final = (litellm.BudgetExceededError, HTTPException, ProxyException)


def bind_native_request(reservation: Mapping[str, object] | None, request: Request) -> None:
    pending: Final = _pending_plan.get()
    if pending is not None and reservation is not None and pending.reservation is reservation:
        pending.request = request


async def raise_native_admission_refusal(error: Exception) -> None:
    pending: Final = _refusals.pop(error, None) if isinstance(error, _REFUSAL_TYPES) else None
    if pending is None or pending.request is None:
        return
    from litellm.proxy.auth.auth_exception_handler import UserAPIKeyAuthExceptionHandler

    _ = await UserAPIKeyAuthExceptionHandler.handle_authentication_error(
        e=error,
        request=pending.request,
        request_data=pending.request_data,
        route=pending.route,
        parent_otel_span=pending.identity.parent_otel_span,
        api_key="",
        resolved_identity=pending.identity,
    )


_NATIVE_CALL_TYPES: Final = frozenset(
    {
        CallTypes.acompletion,
        CallTypes.completion,
        CallTypes.atext_completion,
        CallTypes.text_completion,
        CallTypes.aembedding,
        CallTypes.embedding,
        CallTypes.anthropic_messages,
        CallTypes.aanthropic_messages,
    }
)


class _Entry(BaseModel):
    model_config = ConfigDict(frozen=True)

    counter_key: str
    entity_type: str
    entity_id: str
    reserved_cost: float
    applied_adjustment: float


class _Reservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    reserved_cost: float
    entries: tuple[_Entry, ...]
    finalized: bool
    input_cost: float
    input_tokens: int | None


class _BudgetExceededFields(BaseModel):
    model_config = ConfigDict(frozen=True)

    current_cost: float
    max_budget: float
    message: str
    entity_type: str | None
    entity_id: str | None
    model: str | None


class _DetailFields(BaseModel):
    model_config = ConfigDict(frozen=True)

    detail: str


class _MessageFields(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str


_RESERVATION: Final = TypeAdapter(_Reservation | None)
_SETTLED: Final = TypeAdapter(_Reservation)
_STR_MAPPING: Final = TypeAdapter(Mapping[str, object])


def native_budgets_requested() -> bool:
    return os.getenv(NATIVE_BUDGETS_ENV, "").strip().lower() == "true"


def native_budget_route(route: str) -> bool:
    call_types: Final = get_call_types_for_route(route) or ()
    return any(call_type in _NATIVE_CALL_TYPES for call_type in call_types)


def _budget_exceeded(wire: WireAbort) -> Exception:
    fields: Final = _BudgetExceededFields.model_validate(wire.error.fields if wire.error else {})
    return litellm.BudgetExceededError(
        current_cost=fields.current_cost,
        max_budget=fields.max_budget,
        message=fields.message,
        entity_type=fields.entity_type,
        entity_id=fields.entity_id,
    )


def _budget_unavailable(wire: WireAbort) -> Exception:
    fields: Final = _DetailFields.model_validate(wire.error.fields if wire.error else {})
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=fields.detail)


def _reservation_failed(wire: WireAbort) -> Exception:
    fields: Final = _MessageFields.model_validate(wire.error.fields if wire.error else {})
    return ProxyException(
        message="Authentication Error, " + fields.message,
        type=ProxyErrorTypes.auth_error,
        param=None,
        code=status.HTTP_401_UNAUTHORIZED,
    )


def _admission(data: object) -> object:
    request: Final = _STR_MAPPING.validate_python(data) if isinstance(data, Mapping) else {}
    model: Final = request.get("model")
    return {"plan": _admitting.get(), "call_id": None, "model": model if isinstance(model, str) else None}


class NativeBudgetsLogger(NativeLogger):
    budgets: NativeBudgets

    async def run_native(
        self, hook: Hook, reads: frozenset[Field], view: object, ctx: CallContext[object]
    ) -> Patch | Abort | None:
        pending: Final = _pending_plan.get() if hook is Hook.pre_call else None
        plan: Final = None if pending is None else pending.claim()
        if pending is None or plan is None:
            return await super().run_native(hook, reads, view, ctx)
        token: Final = _admitting.set(plan)
        try:
            outcome: Final = await super().run_native(hook, reads, view, ctx)
        finally:
            _admitting.reset(token)
        await self._mirror(pending, plan)
        if isinstance(outcome, Abort) and outcome.exception is not None:
            _refusals[outcome.exception] = pending
        return outcome

    async def _mirror(self, pending: _PendingPlan, plan: Mapping[str, object]) -> None:
        reservation: Final = pending.reservation
        reservation_id: Final = plan.get("reservation_id")
        held: Final = (
            _RESERVATION.validate_python(await self.budgets.take(reservation_id))
            if isinstance(reservation_id, str)
            else None
        )
        if held is None:
            reservation.update({"entries": [], "input_tokens": None})
            return
        reservation.update(
            {
                "reserved_cost": held.reserved_cost,
                "entries": [entry.model_dump() for entry in held.entries],
                "input_cost": held.input_cost,
                "input_tokens": held.input_tokens,
                NATIVE_CALL_ID_KEY: reservation_id,
            }
        )
        from litellm.proxy.spend_tracking.budget_reservation import start_reservation_lease_renewal

        start_reservation_lease_renewal(
            budget_reservation=reservation,
            counter_keys=frozenset(entry.counter_key for entry in held.entries),
            request_task=pending.request_task,
        )


_PROJECTIONS: Final[Mapping[Field, Projection]] = {Field.deprecated_kwargs: _admission}

_installed: Final[list[NativeBudgets]] = []  # mutable-ok: one process-wide native owner, set once at startup


def native_budgets() -> NativeBudgets | None:
    return _installed[0] if _installed else None


def install_native_budgets(store: SpendCounterStore, callbacks: list[object]) -> None:  # mutable-ok: litellm.callbacks
    if _installed or not native_budgets_requested():
        return
    from litellm.rust_bridge._native import NativeBudgets

    budgets: Final = NativeBudgets(store.handle)
    logger: Final = native_logger(budgets.callback(), _PROJECTIONS)
    logger.__class__ = type(type(logger).__name__, (NativeBudgetsLogger, type(logger)), {"budgets": budgets})
    register_abort_error("budget_exceeded", _budget_exceeded)
    register_abort_error("budget_unavailable", _budget_unavailable)
    register_abort_error("budget_reservation_failed", _reservation_failed)
    callbacks.insert(0, logger)
    _installed.append(budgets)


async def settle_native_reservation(
    reservation: dict[str, object],  # mutable-ok: writes the settled entries back
    actual_cost: float,
    finalize: bool,
) -> None:
    budgets: Final = native_budgets()
    if budgets is None or NATIVE_CALL_ID_KEY not in reservation:
        return
    held: Final = _SETTLED.validate_python(reservation)
    settled: Final = _SETTLED.validate_python(await budgets.settle(held.model_dump(), actual_cost, finalize))
    reservation.update({"entries": [entry.model_dump() for entry in settled.entries], "finalized": settled.finalized})


def is_native_reservation(reservation: Mapping[str, object] | None) -> bool:
    return reservation is not None and NATIVE_CALL_ID_KEY in reservation


class BudgetDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: Literal["skip", "pass", "alert", "throttle", "refuse"]
    pct: int | float | None = None
    current_cost: int | float | None = None
    max_budget: int | float | None = None
    message: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None

    @property
    def skipped(self) -> bool:
        return self.decision == "skip"

    @property
    def alerts(self) -> bool:
        return self.decision == "alert"

    @property
    def refused(self) -> bool:
        return self.decision in ("refuse", "throttle")

    def error(self) -> litellm.BudgetExceededError:
        return litellm.BudgetExceededError(
            current_cost=self.current_cost if self.current_cost is not None else 0.0,
            max_budget=self.max_budget if self.max_budget is not None else 0.0,
            message=self.message,
            entity_type=self.entity_type,
            entity_id=self.entity_id,
        )


def native_budget_decision(check: Mapping[str, object]) -> BudgetDecision | None:
    if native_budgets() is None:
        return None
    from litellm.rust_bridge._native import budget_check_decision

    return BudgetDecision.model_validate(budget_check_decision(check))
