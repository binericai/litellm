import asyncio
import json
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from functools import cache
from typing import TYPE_CHECKING, ClassVar, Final, Literal, TypeAlias

import litellm
from litellm._logging import verbose_logger
from litellm.constants import (
    COMPLETION_HTTP_FALLBACK_SECONDS,
    DEFAULT_MAX_RECURSE_DEPTH,
    DEFAULT_S3_BATCH_SIZE,
    DEFAULT_S3_FLUSH_INTERVAL_SECONDS,
    DEFAULT_S3_MAX_ADAPTIVE_CONCURRENCY,
    DEFAULT_S3_MAX_CONCURRENT_UPLOADS,
    HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS,
    S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS,
)
from litellm.integrations.batch_line_item_objects import batch_line_items_ndjson, batch_line_items_object_name
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2.enums import Field
from litellm.integrations.custom_logger_v2.native import NativeLogger, native_logger
from litellm.integrations.s3 import prompts_only_payload
from litellm.integrations.s3_v2 import S3Logger
from litellm.secret_managers.main import get_secret
from litellm.types.integrations.s3_v2 import S3PartitionGranularity
from litellm.types.utils import StandardAuditLogPayload, StandardBatchLineItemsPayload

if TYPE_CHECKING:
    from litellm.rust_bridge._native import NativeS3Log

NATIVE_S3_V2_FLAG: Final = "LITELLM_NATIVE_S3_V2"
CALLBACK_NAME: Final = "s3_v2"

MetadataKind: TypeAlias = Literal["missing", "none", "mapping", "other"]

_MISSING: Final = object()
_FAILED: Final = "failed"


@cache
def native_s3_v2_enabled() -> bool:
    return os.environ.get(NATIVE_S3_V2_FLAG, "").strip().lower() == "true"


class _Host:
    def __init__(self) -> None:
        self.failures: Final = CustomLogger()

    def callback_failed(self) -> None:
        self.failures.handle_callback_failure(callback_name="S3Logger")

    def cold_storage_is_s3_v2(self) -> bool:
        return litellm.cold_storage_custom_logger == "s3_v2"

    def region_from_environment(self) -> str | None:
        region_name: Final = get_secret("AWS_REGION_NAME", None)
        region: Final = get_secret("AWS_REGION", None)
        if isinstance(region, str):
            return region
        return region_name if isinstance(region_name, str) else None

    def web_identity_token(self, reference: str) -> str:
        if reference.startswith("os.environ/") or reference in os.environ:
            raise ValueError("Invalid web identity token reference.")
        token: Final = get_secret(reference)
        if token is None:
            raise ValueError("OIDC token could not be retrieved from secret manager.")
        return str(token)


def _resolved_params(params_source: Mapping[str, object] | None) -> dict[str, object]:  # mutable-ok: handed to Rust
    source: Final = params_source if params_source is not None else (litellm.s3_callback_params or {})
    return {
        key: (litellm.get_secret(value) if isinstance(value, str) and value.startswith("os.environ/") else value)
        for key, value in source.items()
    }


def _constants() -> Mapping[str, object]:
    return {
        "batch_size": DEFAULT_S3_BATCH_SIZE or litellm.DEFAULT_BATCH_SIZE,
        "max_concurrent_uploads": DEFAULT_S3_MAX_CONCURRENT_UPLOADS,
        "max_adaptive_concurrency": DEFAULT_S3_MAX_ADAPTIVE_CONCURRENCY,
        "upload_attempt_timeout_seconds": S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS,
        "download_read_timeout_seconds": COMPLETION_HTTP_FALLBACK_SECONDS,
        "download_connect_timeout_seconds": HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS,
        "web_identity_session_policy": S3Logger._web_identity_session_policy_document(),
    }


def _flush_interval() -> float:
    return DEFAULT_S3_FLUSH_INTERVAL_SECONDS or litellm.DEFAULT_FLUSH_INTERVAL_SECONDS


_STRIPPER: Final = CustomLogger()


def _metadata_kind(metadata: object) -> MetadataKind:
    if metadata is _MISSING:
        return "missing"
    if metadata is None:
        return "none"
    return "mapping" if callable(getattr(metadata, "get", None)) else "other"


def _alias(value: object) -> object:
    if not value:
        return None
    return value if isinstance(value, str) else True


def _start_time(start_time: object, timestamp: object) -> datetime | None:
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        return start_time if isinstance(start_time, datetime) else None
    if isinstance(start_time, datetime) and start_time.timestamp() == timestamp:
        return start_time
    try:
        return datetime.fromtimestamp(timestamp)  # noqa: DTZ006  # naive local time, as the start_time the Python logger formats
    except (OverflowError, OSError, ValueError):
        return None


def _start(start_time: object, timestamp: object) -> list[int] | None:  # mutable-ok: crosses into Rust as a JSON list
    start: Final = _start_time(start_time, timestamp)
    if start is None:
        return None
    return [start.year, start.month, start.day, start.hour, start.minute, start.second, start.microsecond]


def legacy_should_log(kwargs: Mapping[str, object]) -> bool:
    from litellm.litellm_core_utils import litellm_logging

    if litellm.global_disable_no_log_param:
        return True
    litellm_params: Final = kwargs.get("litellm_params", {})
    lookup: Final = getattr(litellm_params, "get", None)
    if callable(lookup) and lookup("no-log", False) is True:
        return False
    controls: Final = litellm_logging.EnterpriseCallbackControls
    return controls is None or not controls.is_callback_disabled_dynamically(
        callback=CALLBACK_NAME,
        litellm_params=litellm_logging._callback_control_params(litellm_params),
        standard_callback_dynamic_params=kwargs.get("standard_callback_dynamic_params") or {},  # pyright: ignore[reportArgumentType]  # the dynamic params dict the legacy path hands the same check
    )


def project_payload(sink: "NativeS3Log", payload: object, start_time: object = None) -> object:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        return _FAILED
    stripped: Final = (
        _STRIPPER._strip_base64_from_messages_sync(payload)  # pyright: ignore[reportArgumentType]  # the shared StandardLoggingPayload, stripped in place as S3Logger does
        if sink.strip_base64_files
        else payload
    )
    metadata: Final = stripped.get("metadata", _MISSING)
    kind: Final = _metadata_kind(metadata)
    get: Final[Callable[[str], object]] = metadata.get if kind == "mapping" else {}.get  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # kind says metadata has a get, as Python's s3_v2 calls it
    payload_id: Final = stripped.get("id")
    cold_storage_object_key: Final = get("cold_storage_object_key") if kind == "mapping" else None
    stored: Final = dict(prompts_only_payload(stripped) if sink.log_prompts_only() else stripped)  # pyright: ignore[reportArgumentType]  # the StandardLoggingPayload copy S3Logger queues
    return {
        "start": _start(start_time, stripped.get("startTime")),
        "id": payload_id if isinstance(payload_id, str) else None,
        "metadata": kind,
        "team_alias": _alias(get("user_api_key_team_alias")),
        "key_alias": _alias(get("user_api_key_alias")),
        "cold_storage_object_key": cold_storage_object_key if isinstance(cold_storage_object_key, str) else None,
        "cold_storage_s3_v2": litellm.cold_storage_custom_logger == "s3_v2",
        "body": sink.stash(stored),
    }


def project_kwargs(sink: "NativeS3Log", kwargs: object) -> object:
    from litellm.litellm_core_utils.litellm_logging import V2_ASYNC_LOGGING_LINE

    if not V2_ASYNC_LOGGING_LINE.get() or not isinstance(kwargs, Mapping):
        return None
    fields: Final[Mapping[str, object]] = kwargs
    if not legacy_should_log(fields):
        return None
    try:
        return project_payload(sink, fields.get("standard_logging_object"), fields.get("start_time"))
    except Exception as e:  # noqa: BLE001  # S3Logger counts a failure to build the element and moves on
        verbose_logger.exception("s3 Layer Error - %s", e)
        return _FAILED


async def _periodic_flush(sink: "NativeS3Log", interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            await sink.flush()
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:  # noqa: BLE001  # a failed flush keeps the queue and the next interval tries again, as flush_queue does
            verbose_logger.exception("CustomLogger: periodic flush failed: %s", e)


class S3LoggerV3(NativeLogger):
    sink: ClassVar["NativeS3Log"]
    periodic_flush_task: ClassVar["asyncio.Task[None]"]

    @property
    def s3_path(self) -> str | None:
        return self.sink.s3_path

    def resolve_partition_granularity(self) -> S3PartitionGranularity:
        return "hour" if self.sink.partition_granularity() == "hour" else "day"

    async def flush_queue(self) -> None:
        await self.sink.flush()

    async def async_log_audit_log_event(self, audit_log: StandardAuditLogPayload) -> None:
        try:
            audit_log_id: Final = audit_log.get("id", "unknown")
            await self.sink.submit_audit(self.sink.stash(dict(audit_log)), f"{audit_log_id}")
        except Exception as e:  # noqa: BLE001  # an audit log failure is logged and swallowed, as in S3Logger
            verbose_logger.exception("S3 audit log error: %s", e)

    async def async_log_batch_line_items(self, payload: StandardBatchLineItemsPayload) -> None:
        object_name: Final = batch_line_items_object_name(payload)
        s3_path: Final = self.s3_path
        s3_object_key: Final = (f"{s3_path.rstrip('/')}/" if s3_path else "") + object_name
        body: Final = await asyncio.to_thread(batch_line_items_ndjson, payload)
        delivered: Final = await self.sink.upload_line_items(
            s3_object_key, body, "-".join(object_name.rsplit("/", 2)[-2:])
        )
        if not delivered:
            verbose_logger.error(
                "s3 logging: the %d line items of part %d of batch %s were not stored at %s",
                len(payload["lines"]),
                payload["part"],
                payload["batch_id"],
                s3_object_key,
            )

    async def _download_object_from_s3(self, s3_object_key: str) -> object:
        try:
            downloaded: Final = await self.sink.download(s3_object_key)
            return None if downloaded is None else json.loads(downloaded)
        except Exception as e:  # noqa: BLE001  # a failed read is None, as in S3Logger
            verbose_logger.exception("Error downloading from S3: %s", e)
            return None

    async def get_proxy_server_request_from_cold_storage_with_object_key(self, object_key: str) -> object:
        try:
            return await self._download_object_from_s3(object_key)
        except Exception as e:  # noqa: BLE001  # a failed read is None, as in S3Logger
            verbose_logger.exception("Error retrieving object %s from cold storage: %s", object_key, e)
            return None


S3_V2_LOGGER_TYPES: Final = (S3Logger, S3LoggerV3)


def new_s3_v2_logger(s3_callback_params_override: Mapping[str, object] | None = None) -> S3Logger | S3LoggerV3:
    if not native_s3_v2_enabled():
        return (
            S3Logger()
            if s3_callback_params_override is None
            else S3Logger(s3_callback_params_override=dict(s3_callback_params_override))
        )
    from litellm.rust_bridge._native import NativeS3Log

    sink: Final = NativeS3Log(
        _resolved_params(s3_callback_params_override), _constants(), _Host(), DEFAULT_MAX_RECURSE_DEPTH
    )
    periodic: Final = asyncio.create_task(_periodic_flush(sink, _flush_interval()))
    generated: Final = type(native_logger(sink.callback()))
    cls: Final = type(S3Logger.__name__, (S3LoggerV3, generated), {"sink": sink, "periodic_flush_task": periodic})
    return cls(sink.callback(), {Field.deprecated_kwargs: lambda kwargs: project_kwargs(sink, kwargs)})


__all__ = [
    "NATIVE_S3_V2_FLAG",
    "S3_V2_LOGGER_TYPES",
    "S3LoggerV3",
    "legacy_should_log",
    "native_s3_v2_enabled",
    "new_s3_v2_logger",
]
