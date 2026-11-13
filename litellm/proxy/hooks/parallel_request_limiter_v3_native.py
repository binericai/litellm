import dataclasses
import json
import os
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from litellm.caching.redis_batch import active_request_redis_batches
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.native import (
    NativeLogger,
    NativeState,
    WireAbort,
    WireManifest,
    _namespace,  # pyright: ignore[reportPrivateUsage]  # the limiter's logger subclasses the contract's generated class
    register_abort_error,
    verdict,
)
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError, map_v3_rate_limit_type
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit
from litellm.types.utils import CallTypes

_IO_KEYS: Final = frozenset({"model_per_project_itpm", "model_per_project_otpm"})
PROJECT_OTPM_DESCRIPTOR_KEY: Final = "model_per_project_otpm"
PTU_TEAM_DESCRIPTOR_KEY: Final = "model_per_team_ptu"

if TYPE_CHECKING:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3
    from litellm.rust_bridge._native import (
        NativeParallelRequestLimiter,
        NativeParallelRequestLimiterState,
        NativeRequestRedis,
    )

NATIVE_PARALLEL_REQUEST_LIMITER_ENV: Final = "LITELLM_NATIVE_PARALLEL_REQUEST_LIMITER"
RATE_LIMITED_KIND: Final = "proxy_rate_limit"
UNVERIFIABLE_KIND: Final = "rate_limit_unverifiable"
PTU_SHARE_EXCEEDED_KIND: Final = "ptu_share_exceeded"
_UNSUPPORTED_CALL_TYPES: Final = frozenset(
    {
        CallTypes.call_mcp_tool.value,
        CallTypes.aresponses_websocket.value,
        "pass_through_endpoint",
        "_arealtime",
    }
)
_HEADERS: Final = TypeAdapter(dict[str, str])
_PTU_FIELDS: Final = TypeAdapter(dict[str, str | int | float])


def native_parallel_request_limiter_enabled() -> bool:
    return os.getenv(NATIVE_PARALLEL_REQUEST_LIMITER_ENV, "false").strip().lower() == "true"


def _rate_limited(wire: WireAbort) -> Exception:
    fields: Final[Mapping[str, object]] = wire.error.fields if wire.error is not None else MappingProxyType({})
    requested_model: Final = fields.get("requested_model")
    resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(
        requested_model if isinstance(requested_model, str) else None
    )
    headers: Final = fields.get("headers")
    rate_limit_type: Final = fields.get("rate_limit_type")
    descriptor_key: Final = fields.get("descriptor_key")
    return ProxyRateLimitError(
        detail=str(fields.get("detail", wire.message)),
        headers=_HEADERS.validate_python(headers) if headers is not None else dict(wire.headers),
        rate_limit_type=map_v3_rate_limit_type(rate_limit_type if isinstance(rate_limit_type, str) else None),
        model=resolved_model,
        llm_provider=llm_provider,
        descriptor_key=descriptor_key if isinstance(descriptor_key, str) else None,
    )


def _unverifiable(wire: WireAbort) -> Exception:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import RateLimitUnverifiableError

    return RateLimitUnverifiableError()


def _ptu_share_exceeded(wire: WireAbort) -> Exception:
    from litellm.litellm_core_utils.ptu_shares import PTUTeamCeiling
    from litellm.proxy.hooks.parallel_request_limiter_v3 import PTUShareExceededError

    fields: Final = _PTU_FIELDS.validate_python(wire.error.fields if wire.error is not None else {})
    ceiling: Final = PTUTeamCeiling(
        model_group=str(fields["model_group"]),
        tpm_limit=int(fields["tpm_limit"]),
        output_to_input_ratio=float(fields["output_to_input_ratio"]),
        cached_input_ratio=float(fields["cached_input_ratio"]),
    )
    return PTUShareExceededError(ceiling, int(fields["needed_tokens"]))


register_abort_error(RATE_LIMITED_KIND, _rate_limited)
register_abort_error(UNVERIFIABLE_KIND, _unverifiable)
register_abort_error(PTU_SHARE_EXCEEDED_KIND, _ptu_share_exceeded)


def raise_wire_abort(returned: object) -> None:
    outcome: Final = verdict(returned)
    if isinstance(outcome, Abort) and outcome.exception is not None:
        raise outcome.exception


def pre_call_request() -> "NativeRequestRedis | None":
    batches: Final = active_request_redis_batches()
    return None if batches is None else batches.native()


def post_call_request() -> "NativeRequestRedis | None":
    batches: Final = active_request_redis_batches()
    return None if batches is None else batches.native_post_call()


def _auth(data: Mapping[str, object]) -> UserAPIKeyAuth | None:
    for key in ("metadata", "litellm_metadata"):
        metadata = data.get(key)
        if isinstance(metadata, dict):
            auth = metadata.get("user_api_key_auth")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # request metadata is untyped
            if isinstance(auth, UserAPIKeyAuth):
                return auth
    return None


def _dynamic_limiter_registered() -> bool:
    import litellm
    from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
        _PROXY_DynamicRateLimitHandlerV3,  # pyright: ignore[reportPrivateUsage]  # the class has no public alias
    )

    return any(isinstance(callback, _PROXY_DynamicRateLimitHandlerV3) for callback in litellm.callbacks)


def limiter_state(value: object) -> "NativeParallelRequestLimiterState | None":
    from litellm.rust_bridge._native import NativeParallelRequestLimiterState

    return value if isinstance(value, NativeParallelRequestLimiterState) else None


def _request_data(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}  # pyright: ignore[reportUnknownVariableType]  # the hook's request data is the untyped proxy body


class ParallelRequestLimiterNativeLogger(NativeLogger):
    limiter: "PROXY_MaxParallelRequestsHandler_v3"

    def __init__(self, native: "NativeParallelRequestLimiter", limiter: "PROXY_MaxParallelRequestsHandler_v3") -> None:
        super().__init__(native)  # pyright: ignore[reportArgumentType]  # the bridge takes its own state type
        self.limiter = limiter
        self.bridge: Final = native

    def _state(self, ctx: CallContext[object]) -> "NativeParallelRequestLimiterState | None":
        from litellm.proxy.hooks.parallel_request_limiter_v3 import get_request_stash

        stash: Final = get_request_stash()
        stashed: Final = None if stash is None else limiter_state(stash.native_state)
        if stashed is not None:
            return stashed
        return limiter_state(ctx.state.handle) if isinstance(ctx.state, NativeState) else None

    async def run_native(
        self, hook: Hook, reads: frozenset[Field], view: object, ctx: CallContext[object]
    ) -> Patch | Abort | None:
        data: Final = _request_data(getattr(view, Field.deprecated_kwargs.value, None))
        match hook:
            case Hook.pre_call:
                return await self._pre_call(data, ctx)
            case Hook.post_call_success:
                return await self._call(hook, {"call_id": _call_id(data)}, ctx)
            case Hook.post_call_failure:
                return await self._post_call_failure(data, ctx)
            case Hook.log_prepare:
                payload: Final = getattr(view, Field.standard_logging_payload.value, None)
                return await self._call(hook, {"call_id": _call_id(data)}, ctx, payload)
            case _:
                return None

    async def _call(
        self,
        hook: Hook,
        facts: Mapping[str, object],
        ctx: CallContext[object],
        payload: object = None,
        state: "NativeParallelRequestLimiterState | None" = None,
        request: "NativeRequestRedis | None" = None,
    ) -> Patch | Abort | None:
        handle: Final = state if state is not None else self._state(ctx)
        if handle is None:
            return None
        fields: Final[dict[str, object]] = {
            Field.deprecated_kwargs.value: dict(facts)
        }  # mutable-ok: the bridge reads a dict
        if payload is not None:
            fields[Field.standard_logging_payload.value] = payload
        returned: Final = await self.bridge.handle(hook.value, fields, None, handle, request=request)
        return verdict(returned)

    async def _post_call_failure(self, data: dict[str, object], ctx: CallContext[object]) -> Patch | Abort | None:
        return await self._call(Hook.post_call_failure, failure_facts(self.limiter, data), ctx)

    def _declines(self, data: Mapping[str, object], call_type: str, auth: UserAPIKeyAuth | None) -> bool:
        from litellm.proxy.hooks.parallel_request_limiter_v3 import get_request_stash

        stash: Final = get_request_stash()
        return (
            auth is None
            or call_type in _UNSUPPORTED_CALL_TYPES
            or self.limiter.get_rate_limiter_for_call_type(call_type) is not None
            or _dynamic_limiter_registered()
            or (
                stash is not None
                and (
                    stash.rate_limit_response is not None
                    or stash.parallel_slot is not None
                    or stash.reserved_tokens != 0
                )
            )
            or not isinstance(data.get("litellm_call_id"), str)
        )

    async def _pre_call(self, data: dict[str, object], ctx: CallContext[object]) -> Patch | Abort | None:
        from litellm.litellm_core_utils.token_counter import offload_token_count
        from litellm.proxy.auth.auth_utils import get_estimated_output_tokens
        from litellm.proxy.hooks.parallel_request_limiter_v3 import claim_request_stash_for_data

        call_type: Final = ctx.call.kind.call_type if ctx.call is not None else ""
        auth: Final = _auth(data)
        state: Final = (
            limiter_state(ctx.state.handle) if isinstance(ctx.state, NativeState) else None
        ) or self.bridge.request_state()
        if auth is None or self._declines(data, call_type, auth):
            return None
        limiter: Final = self.limiter
        stash: Final = claim_request_stash_for_data(data)
        descriptors: Final = await limiter._build_request_rate_limit_descriptors(  # pyright: ignore[reportPrivateUsage]  # the descriptors are the Python limiter's
            user_api_key_dict=auth, data=data, call_type=call_type
        )
        stash.call_type = call_type
        stash.native_state = state
        model_value: Final = data.get("model")
        requested_model: Final = model_value if isinstance(model_value, str) else None
        ceiling: Final = stash.ptu_ceiling
        configured_tpm_limits: Final = tuple(
            ceiling.raw_output_token_limit()
            if descriptor["key"] == PTU_TEAM_DESCRIPTOR_KEY and ceiling is not None
            else int(limit)
            for descriptor in descriptors
            if descriptor["key"] not in _IO_KEYS
            and (limit := (descriptor["rate_limit"] or {}).get("tokens_per_unit")) is not None
        )
        configured_otpm_limits: Final = tuple(
            int(limit)
            for descriptor in descriptors
            if descriptor["key"] == PROJECT_OTPM_DESCRIPTOR_KEY
            and (limit := (descriptor["rate_limit"] or {}).get("tokens_per_unit")) is not None
        )
        min_otpm_limit: Final = min(configured_otpm_limits) if configured_otpm_limits else None
        has_io: Final = any(descriptor["key"] in _IO_KEYS for descriptor in descriptors)
        reserves: Final = bool(configured_tpm_limits) and limiter.tpm_reservation_enabled
        configured_output_tokens: Final = (
            get_estimated_output_tokens(user_api_key_dict=auth, model_name=requested_model) if reserves else None
        )
        probe: Final = dict(data)
        estimated_tokens: int | None = None  # rebind-ok: set only when the request reserves tokens
        ptu_estimate: Mapping[str, object] | None = None  # rebind-ok: set only for a team PTU reservation
        if reserves:
            limiter._apply_implicit_output_cap(  # pyright: ignore[reportPrivateUsage]  # the cap is the Python limiter's
                data=probe,
                min_configured_limit=min(configured_tpm_limits),
                call_type=call_type,
                configured_output_tokens=configured_output_tokens,
            )
            estimated_tokens = await offload_token_count(limiter._estimate_tokens_for_request)(  # pyright: ignore[reportPrivateUsage]  # the estimate is the Python limiter's
                data=probe,
                model=requested_model,
                min_configured_tpm_limit=min(configured_tpm_limits),
                call_type=call_type,
                configured_output_tokens=configured_output_tokens,
                team_id=auth.team_id,
            )
            if ceiling is not None:
                ptu_input, ptu_output = await offload_token_count(limiter._estimate_input_and_output_tokens)(  # pyright: ignore[reportPrivateUsage]  # the estimate is the Python limiter's
                    data=probe,
                    min_configured_tpm_limit=min(configured_tpm_limits),
                    call_type=call_type,
                    configured_output_tokens=configured_output_tokens,
                    team_id=auth.team_id,
                    media_model_resolver=limiter._media_model_resolver,  # pyright: ignore[reportPrivateUsage]  # the estimate is the Python limiter's
                )
                ptu_estimate = {"input_tokens": ptu_input, "output_tokens": ptu_output}
        io_estimate: Mapping[str, object] | None = None  # rebind-ok: set only for project ITPM/OTPM limits
        if has_io:
            _, io_output = limiter._estimate_input_and_output_tokens(  # pyright: ignore[reportPrivateUsage]  # the estimate is the Python limiter's
                data=probe, min_configured_tpm_limit=min_otpm_limit, call_type=call_type, include_media=False
            )
            io_input: Final = await offload_token_count(limiter._estimate_precise_input_tokens)(  # pyright: ignore[reportPrivateUsage]  # the estimate is the Python limiter's
                data=probe, model=requested_model, call_type=call_type, team_id=auth.team_id
            )
            io_estimate = {
                "input_tokens": io_input,
                "output_tokens": io_output,
                "explicit_output_cap": limiter._has_explicit_output_cap(probe, call_type),  # pyright: ignore[reportPrivateUsage]  # the cap is the Python limiter's
            }
        reserved_model: Final = limiter.rate_limited_model(requested_model)
        facts: Final = {
            "call_id": data.get("litellm_call_id"),
            "descriptors": [dict(descriptor) for descriptor in descriptors],
            "requested_model": requested_model,
            "estimated_tokens": estimated_tokens,
            "reserved_model": None
            if reserved_model is None
            else {"requested": reserved_model.requested, "group": reserved_model.group},
            "policy": policy(limiter),
            "ptu_ceiling": None if ceiling is None else dataclasses.asdict(ceiling),
            "ptu_estimate": ptu_estimate,
            "io_estimate": io_estimate,
        }
        outcome: Final = await self._call(Hook.pre_call, facts, ctx, state=state, request=pre_call_request())
        if reserves and self.bridge.first_pass_admitted(state):
            limiter._apply_implicit_output_cap(  # pyright: ignore[reportPrivateUsage]  # the cap is the Python limiter's
                data=data,
                min_configured_limit=min(configured_tpm_limits),
                call_type=call_type,
                configured_output_tokens=configured_output_tokens,
            )
        if self.bridge.io_reached(state):
            limiter._apply_implicit_output_cap(  # pyright: ignore[reportPrivateUsage]  # the cap is the Python limiter's
                data=data, min_configured_limit=min_otpm_limit, call_type=call_type
            )
        mirror_rate_limit_response(self.bridge, state)
        return outcome


def mirror_rate_limit_response(
    bridge: "NativeParallelRequestLimiter", state: "NativeParallelRequestLimiterState"
) -> None:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import get_request_stash

    stash: Final = get_request_stash()
    response: Final = bridge.rate_limit_response(state)
    if stash is None or response is None:
        return
    stash.rate_limit_response = json.loads(response)


def usage_facts(usage: object) -> Mapping[str, int] | None:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        _ReconciledUsage,  # pyright: ignore[reportPrivateUsage]  # the limiter's usage shape
    )

    if not isinstance(usage, _ReconciledUsage):
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cached_tokens": usage.cached_tokens,
    }


def failure_facts(limiter: "PROXY_MaxParallelRequestsHandler_v3", source: Mapping[str, object]) -> Mapping[str, object]:
    recovered: Final = limiter._recovered_partial_usage(source)  # pyright: ignore[reportPrivateUsage]  # the native settles with the limiter's usage recovery
    return {
        "call_id": _call_id(source),
        "tpm_actual": limiter._recovered_partial_usage_tokens(source)[0],  # pyright: ignore[reportPrivateUsage]  # the native settles with the limiter's usage recovery
        "partial_usage": None if recovered is None else usage_facts(limiter._resolve_reconciled_usage(recovered)),  # pyright: ignore[reportPrivateUsage]  # the native settles with the limiter's usage recovery
    }


def policy(limiter: "PROXY_MaxParallelRequestsHandler_v3") -> Mapping[str, bool]:
    return {
        "fail_closed": limiter._fail_closed_resolver(),  # pyright: ignore[reportPrivateUsage]  # read live per call like the Python limiter
        "force_grouping": limiter._force_hash_tag_grouping_resolver(),  # pyright: ignore[reportPrivateUsage]  # read live per call like the Python limiter
    }


def _call_id(data: Mapping[str, object]) -> str | None:
    call_id: Final = data.get("litellm_call_id")
    return call_id if isinstance(call_id, str) else None


def native_engine(limiter: "PROXY_MaxParallelRequestsHandler_v3") -> "NativeParallelRequestLimiter":
    from litellm.rust_bridge._native import NativeParallelRequestLimiter

    clock: Final = limiter._time_provider  # pyright: ignore[reportPrivateUsage]  # the native reads the limiter's clock
    return NativeParallelRequestLimiter(
        limiter.rate_limits._handle,  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # both limiters share one store and memory tier
        limiter.window_size,
        limiter.tpm_reservation_enabled,
        None if clock is datetime.now else (lambda: clock().timestamp()),
    )


def native_parallel_request_limiter(
    limiter: "PROXY_MaxParallelRequestsHandler_v3",
) -> ParallelRequestLimiterNativeLogger:
    bridge: Final = limiter.native_companion or native_engine(limiter)
    manifest: Final = WireManifest.model_validate_json(bridge.manifest())
    cls: Final = type(manifest.selection.name, (ParallelRequestLimiterNativeLogger,), _namespace(bridge, manifest))  # pyright: ignore[reportArgumentType]  # the bridge takes its own state type
    logger: Final = cls(bridge, limiter)
    limiter.native_companion = bridge
    return logger


__all__ = [
    "NATIVE_PARALLEL_REQUEST_LIMITER_ENV",
    "ParallelRequestLimiterNativeLogger",
    "failure_facts",
    "limiter_state",
    "mirror_rate_limit_response",
    "native_engine",
    "native_parallel_request_limiter",
    "native_parallel_request_limiter_enabled",
    "policy",
    "post_call_request",
    "pre_call_request",
    "raise_wire_abort",
    "usage_facts",
]
