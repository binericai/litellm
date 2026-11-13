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
    DEFAULT_MAX_RECURSE_DEPTH,
    DEFAULT_S3_BATCH_SIZE,
    DEFAULT_S3_FLUSH_INTERVAL_SECONDS,
    DEFAULT_S3_MAX_ADAPTIVE_CONCURRENCY,
    DEFAULT_S3_MAX_CONCURRENT_UPLOADS,
    S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS,
)
from litellm.integrations.batch_line_item_objects import batch_line_items_ndjson, batch_line_items_object_name
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2.enums import Field
from litellm.integrations.custom_logger_v2.native import NativeLogger, native_logger
from litellm.integrations.s3_v2 import S3Logger
from litellm.types.integrations.s3_v2 import S3PartitionGranularity
from litellm.types.utils import StandardAuditLogPayload, StandardBatchLineItemsPayload

if TYPE_CHECKING:
    from litellm.rust_bridge._native import NativeS3Log

NATIVE_S3_V2_FLAG: Final = "LITELLM_NATIVE_S3_V2"

MetadataKind: TypeAlias = Literal["missing", "none", "mapping", "other"]

_MISSING: Final = object()


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


def _resolved_params(params_source: Mapping[str, object] | None) -> dict[str, object]:  # mutable-ok: handed to Rust
    source: Final = params_source if params_source is not None else (litellm.s3_callback_params or {})
    return {
        key: (litellm.get_secret(value) if isinstance(value, str) and value.startswith("os.environ/") else value)
        for key, value in source.items()
    }


def _constants() -> Mapping[str, object]:
    return {
        "flush_interval_seconds": DEFAULT_S3_FLUSH_INTERVAL_SECONDS or litellm.DEFAULT_FLUSH_INTERVAL_SECONDS,
        "batch_size": DEFAULT_S3_BATCH_SIZE or litellm.DEFAULT_BATCH_SIZE,
        "max_concurrent_uploads": DEFAULT_S3_MAX_CONCURRENT_UPLOADS,
        "max_adaptive_concurrency": DEFAULT_S3_MAX_ADAPTIVE_CONCURRENCY,
        "upload_attempt_timeout_seconds": S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS,
    }


_STRIPPER: Final = CustomLogger()


def _stripped(payload: Mapping[str, object]) -> Mapping[str, object]:
    messages: Final = payload.get("messages", [])
    copied: Final = {
        **payload,
        **(
            {"messages": [dict(message) if isinstance(message, dict) else message for message in messages]}
            if isinstance(messages, list)
            else {}
        ),
    }
    return _STRIPPER._strip_base64_from_messages_sync(copied)  # pyright: ignore[reportArgumentType]  # the payload is a StandardLoggingPayload copy


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


def _start(value: object) -> list[int] | None:  # mutable-ok: crosses into Rust as a JSON list
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        start: Final = datetime.fromtimestamp(value)  # noqa: DTZ006  # naive local time, as the start_time the Python logger formats
    except (OverflowError, OSError, ValueError):
        return None
    return [start.year, start.month, start.day, start.hour, start.minute, start.second, start.microsecond]


def project_payload(sink: "NativeS3Log", payload: object) -> object:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        return {"not_a_mapping": True}
    stripped: Final = _stripped(payload) if sink.strip_base64_files else payload
    metadata: Final = stripped.get("metadata", _MISSING)
    kind: Final = _metadata_kind(metadata)
    get: Final[Callable[[str], object]] = metadata.get if kind == "mapping" else {}.get  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # kind says metadata has a get, as Python's s3_v2 calls it
    payload_id: Final = stripped.get("id")
    cold_storage_object_key: Final = get("cold_storage_object_key") if kind == "mapping" else None
    return {
        "body": sink.encode(stripped, sink.log_prompts_only()),
        "start": _start(stripped.get("startTime")),
        "id": payload_id if isinstance(payload_id, str) else None,
        "metadata": kind,
        "team_alias": _alias(get("user_api_key_team_alias")),
        "key_alias": _alias(get("user_api_key_alias")),
        "cold_storage_object_key": cold_storage_object_key if isinstance(cold_storage_object_key, str) else None,
        "cold_storage_s3_v2": litellm.cold_storage_custom_logger == "s3_v2",
    }


class S3LoggerV3(NativeLogger):
    sink: ClassVar["NativeS3Log"]

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
            await self.sink.submit_audit(self.sink.encode(dict(audit_log), False), f"{audit_log_id}")
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
    generated: Final = type(native_logger(sink.callback()))
    cls: Final = type("s3_v2", (S3LoggerV3, generated), {"sink": sink})
    return cls(sink.callback(), {Field.standard_logging_payload: lambda payload: project_payload(sink, payload)})


__all__ = ["NATIVE_S3_V2_FLAG", "S3_V2_LOGGER_TYPES", "S3LoggerV3", "native_s3_v2_enabled", "new_s3_v2_logger"]
