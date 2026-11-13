import os
from collections.abc import Mapping
from typing import Final

from fastapi import HTTPException, status
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
from litellm.litellm_core_utils.core_helpers import budget_reservation_from_metadata
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.rust_bridge._native import NativeBudgets
from litellm.types.utils import CallTypes

NATIVE_BUDGETS_ENV: Final = "LITELLM_NATIVE_BUDGETS"
NATIVE_PLAN_KEY: Final = "native_plan"
NATIVE_CALL_ID_KEY: Final = "native_call_id"

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
    from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit

    fields: Final = _BudgetExceededFields.model_validate(wire.error.fields if wire.error else {})
    error: Final = litellm.BudgetExceededError(
        current_cost=fields.current_cost,
        max_budget=fields.max_budget,
        message=fields.message,
        entity_type=fields.entity_type,
        entity_id=fields.entity_id,
    )
    _, error.llm_provider = resolve_llm_provider_for_rate_limit(fields.model)
    error.type = ProxyErrorTypes.budget_exceeded  # pyright: ignore[reportAttributeAccessIssue]  # the route handler reads error.type for the client body, as auth's ProxyException sets it
    error.param = None  # pyright: ignore[reportAttributeAccessIssue]  # auth sends param null for a budget refusal
    return error


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


def _metadata(data: Mapping[str, object]) -> Mapping[str, object]:
    for name in ("litellm_metadata", "metadata"):
        bucket = data.get(name)
        if isinstance(bucket, Mapping) and budget_reservation_from_metadata(_STR_MAPPING.validate_python(bucket)):
            return _STR_MAPPING.validate_python(bucket)
    return {}


def request_reservation(data: object) -> dict[str, object] | None:  # mutable-ok: the shared reservation dict
    return (
        budget_reservation_from_metadata(_metadata(_STR_MAPPING.validate_python(data)))
        if isinstance(data, Mapping)
        else None
    )


def _admission(data: object) -> object:
    reservation: Final = request_reservation(data)
    request: Final = _STR_MAPPING.validate_python(data) if isinstance(data, Mapping) else {}
    model: Final = request.get("model")
    call_id: Final = request.get("litellm_call_id")
    return {
        "plan": None if reservation is None else reservation.get(NATIVE_PLAN_KEY),
        "call_id": call_id if isinstance(call_id, str) else None,
        "model": model if isinstance(model, str) else None,
    }


class NativeBudgetsLogger(NativeLogger):
    budgets: NativeBudgets

    async def run_native(
        self, hook: Hook, reads: frozenset[Field], view: object, ctx: CallContext[object]
    ) -> Patch | Abort | None:
        outcome: Final = await super().run_native(hook, reads, view, ctx)
        if hook is Hook.pre_call:
            await self._mirror(getattr(view, Field.deprecated_kwargs.value, None))
        return outcome

    async def _mirror(self, data: object) -> None:
        reservation: Final = request_reservation(data)
        if reservation is None or reservation.pop(NATIVE_PLAN_KEY, None) is None:
            return
        call_id: Final = _admission(data)
        held: Final = (
            _RESERVATION.validate_python(await self.budgets.take(str(call_id["call_id"])))  # pyright: ignore[reportIndexIssue]  # _admission builds this dict
            if isinstance(call_id, dict) and isinstance(call_id.get("call_id"), str)
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
                NATIVE_CALL_ID_KEY: call_id["call_id"],  # pyright: ignore[reportIndexIssue]  # _admission builds this dict
            }
        )


_PROJECTIONS: Final[Mapping[Field, Projection]] = {Field.deprecated_kwargs: _admission}

_installed: Final[list[NativeBudgets]] = []  # mutable-ok: one process-wide native owner, set once at startup


def native_budgets() -> NativeBudgets | None:
    return _installed[0] if _installed else None


def install_native_budgets(store: SpendCounterStore, callbacks: list[object]) -> None:  # mutable-ok: litellm.callbacks
    if _installed or not native_budgets_requested():
        return
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
    return reservation is not None and (NATIVE_CALL_ID_KEY in reservation or NATIVE_PLAN_KEY in reservation)
