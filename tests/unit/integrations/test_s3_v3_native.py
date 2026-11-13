import asyncio
import copy
import json
import re
import threading
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Final
from unittest.mock import patch
from urllib.parse import parse_qs

import pytest
from pydantic import BaseModel

import litellm
from litellm.integrations import s3_v2, s3_v3
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.host import request_scope
from litellm.integrations.custom_logger_v2.selection import RequestSelection
from litellm.integrations.s3_v2 import S3Logger
from litellm.integrations.s3_v3 import S3LoggerV3, native_s3_v2_enabled, new_s3_v2_logger
from litellm.litellm_core_utils.litellm_logging import V2_ASYNC_LOGGING_LINE
from litellm.rust_bridge import _native

pytestmark = pytest.mark.skipif(not hasattr(_native, "NativeS3Log"), reason="the built _native has no NativeS3Log")

COMPARED_HEADERS: Final = (
    "content-type",
    "content-md5",
    "x-amz-content-sha256",
    "content-language",
    "content-disposition",
    "cache-control",
    "x-amz-server-side-encryption",
    "x-amz-server-side-encryption-aws-kms-key-id",
    "x-amz-security-token",
)
START: Final = datetime(2026, 11, 13, 9, 5, 7, 42)


@dataclass
class Received:
    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes


@dataclass
class FakeBucket:
    received: list[Received] = field(default_factory=list)
    objects: dict[str, bytes] = field(default_factory=dict)
    failures: list[tuple[str, int, str]] = field(default_factory=list)
    port: int = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def puts(self) -> list[Received]:
        return [request for request in self.received if request.method == "PUT"]


def _handler(bucket: FakeBucket) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: object) -> None:
            return

        def _reply(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_PUT(self) -> None:
            body: Final = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            bucket.received.append(Received("PUT", self.path, {k.lower(): v for k, v in self.headers.items()}, body))
            failure: Final = next((f for f in bucket.failures if re.search(f[0], self.path)), None)
            if failure is not None:
                self._reply(failure[1], failure[2].encode())
                return
            bucket.objects[self.path] = body
            self._reply(200, b"")

        def do_GET(self) -> None:
            bucket.received.append(Received("GET", self.path, {k.lower(): v for k, v in self.headers.items()}, b""))
            stored: Final = bucket.objects.get(self.path)
            self._reply(404, b"<Error><Code>NoSuchKey</Code></Error>") if stored is None else self._reply(200, stored)

    return Handler


@pytest.fixture
def buckets() -> Iterator[tuple[FakeBucket, FakeBucket]]:
    pair: Final = (FakeBucket(), FakeBucket())
    servers: Final = []
    for bucket in pair:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(bucket))
        bucket.port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
    yield pair
    for server in servers:
        server.shutdown()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(s3_v2, "_backoff_seconds", lambda _attempt: 0.0)
    monkeypatch.delenv("S3_PARTITION_GRANULARITY", raising=False)
    monkeypatch.delenv("S3_LOG_PROMPTS_ONLY", raising=False)
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", None)
    native_s3_v2_enabled.cache_clear()
    yield
    native_s3_v2_enabled.cache_clear()


def _params(endpoint: str, extra: Mapping[str, object]) -> dict[str, object]:
    return {
        "s3_bucket_name": "logs",
        "s3_region_name": "us-east-1",
        "s3_endpoint_url": endpoint,
        "s3_aws_access_key_id": "AKIDEXAMPLE",
        "s3_aws_secret_access_key": "secret",
        "s3_aws_session_token": "session-token",
        **extra,
    }


async def _loggers(
    monkeypatch: pytest.MonkeyPatch, pair: tuple[FakeBucket, FakeBucket], extra: Mapping[str, object]
) -> tuple[S3Logger, S3LoggerV3]:
    monkeypatch.setattr(litellm, "s3_callback_params", _params(pair[0].url, extra))
    monkeypatch.setenv("LITELLM_NATIVE_S3_V2", "false")
    native_s3_v2_enabled.cache_clear()
    python_logger: Final = new_s3_v2_logger()
    monkeypatch.setattr(litellm, "s3_callback_params", _params(pair[1].url, extra))
    monkeypatch.setenv("LITELLM_NATIVE_S3_V2", "true")
    native_s3_v2_enabled.cache_clear()
    rust_logger: Final = new_s3_v2_logger()
    assert isinstance(python_logger, S3Logger)
    assert isinstance(rust_logger, S3LoggerV3)
    return python_logger, rust_logger


async def _log_native(rust_logger: S3LoggerV3, kwargs: Mapping[str, object]) -> None:
    line: Final = V2_ASYNC_LOGGING_LINE.set(True)
    try:
        await rust_logger.run_native(
            Hook.attempt_ended,
            frozenset({Field.deprecated_kwargs}),
            SimpleNamespace(deprecated_kwargs=kwargs),
            SimpleNamespace(call=None, state=None),  # pyright: ignore[reportArgumentType]  # run_native reads only call and state
        )
    finally:
        V2_ASYNC_LOGGING_LINE.reset(line)


async def _log_each(python_logger: S3Logger, rust_logger: S3LoggerV3, payloads: tuple[object, object]) -> None:
    await python_logger.async_log_success_event({"standard_logging_object": payloads[0]}, None, START, START)
    await _log_native(rust_logger, {"standard_logging_object": payloads[1]})


async def _log(python_logger: S3Logger, rust_logger: S3LoggerV3, payload: object) -> None:
    await _log_each(python_logger, rust_logger, (payload, copy.deepcopy(payload)))


async def _flush(python_logger: S3Logger, rust_logger: S3LoggerV3) -> None:
    await python_logger.flush_queue()
    await rust_logger.flush_queue()


def _signed_headers(request: Received) -> str:
    match: Final = re.search(r"SignedHeaders=([^,]+)", request.headers["authorization"])
    assert match is not None
    return match.group(1)


def _botocore_authorization(
    request: Received,
    port: int,
    credentials: tuple[str, str, str] = ("AKIDEXAMPLE", "secret", "session-token"),
    region: str = "us-east-1",
) -> str:
    from botocore.auth import S3SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    signed: Final = _signed_headers(request).split(";")
    unsigned: Final = {
        name: value
        for name, value in request.headers.items()
        if name in signed and name not in ("host", "x-amz-date", "x-amz-security-token")
    }
    aws_request: Final = AWSRequest(
        method=request.method, url=f"http://127.0.0.1:{port}{request.path}", data=request.body, headers=unsigned
    )
    at: Final = datetime.strptime(request.headers["x-amz-date"], "%Y%m%dT%H%M%SZ")
    with patch("botocore.auth.get_current_datetime", return_value=at):
        S3SigV4Auth(Credentials(*credentials), "s3", region).add_auth(aws_request)
    return aws_request.headers["Authorization"]


def _assert_same_puts(pair: tuple[FakeBucket, FakeBucket], count: int) -> None:
    python_puts, rust_puts = pair[0].puts(), pair[1].puts()
    assert len(python_puts) == len(rust_puts) == count
    for python_put, rust_put in zip(
        sorted(python_puts, key=lambda r: r.path), sorted(rust_puts, key=lambda r: r.path), strict=True
    ):
        assert rust_put.path == python_put.path
        assert rust_put.body == python_put.body
        for name in COMPARED_HEADERS:
            assert rust_put.headers.get(name) == python_put.headers.get(name), name
        assert _signed_headers(rust_put) == _signed_headers(python_put)
        assert rust_put.headers["authorization"] == _botocore_authorization(rust_put, pair[1].port)
        assert python_put.headers["authorization"] == _botocore_authorization(python_put, pair[0].port)


class _Model(BaseModel):
    name: str
    when: datetime


class _Opaque:
    def __str__(self) -> str:
        return "opaque\x00object"


def _payload(response_id: str = "chatcmpl-1", **overrides: object) -> dict[str, object]:
    return {
        "id": response_id,
        "call_type": "acompletion",
        "startTime": START.timestamp(),
        "endTime": START.timestamp() + 1.25,
        "response_cost": 0.000123,
        "metadata": {
            "user_api_key_team_alias": "team-α",
            "user_api_key_alias": "key/one",
            "user_api_key_hash": "88dc28",
        },
        "messages": [{"role": "user", "content": "hello «world» 😀"}],
        "response": {"choices": [{"message": {"content": "hi"}}]},
        **overrides,
    }


def _odd_payload() -> dict[str, object]:
    looped: dict[str, object] = {"name": "loop"}
    looped["self"] = looped
    deep: object = "bottom"
    for _ in range(120):
        deep = [deep]
    return _payload(
        "odd",
        numbers=(1, 2.5, 1e16, 1e-5, -0.0, float("nan"), float("inf"), 10**30, True),
        members={3, 1, 2},
        model=_Model(name="m", when=START),
        opaque=_Opaque(),
        nul="a\x00b",
        looped=looped,
        deep=deep,
        keys={"a\x00": 1, "a": 2, 3: "dropped"},
    )


PARAM_GRID: Final = [
    pytest.param({}, id="defaults"),
    pytest.param({"s3_partition_granularity": "hour", "s3_path": "base/logs/"}, id="hourly-path"),
    pytest.param({"s3_use_team_prefix": True, "s3_use_key_prefix": "false"}, id="team-and-key-prefix"),
    pytest.param({"s3_log_prompts_only": "yes"}, id="prompts-only"),
    pytest.param({"s3_sse_kms_key_id": "kms-1"}, id="sse-kms"),
    pytest.param({"s3_server_side_encryption": "AES256", "s3_sse_kms_key_id": "k"}, id="sse-aes-drops-key"),
    pytest.param({"s3_strip_base64_files": True}, id="strip-base64"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", PARAM_GRID)
async def test_both_loggers_put_the_same_objects(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, extra: Mapping[str, object]
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, extra)
    payloads: Final = [
        _payload(),
        _payload("x" * 400),
        _payload("cmpl:with/odd chars=+&#?%"),
        _payload(
            "with-file",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see data:image/png;base64,AAAA== inline"},
                        {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBER"}},
                    ],
                },
                "not a dict",
            ],
        ),
        _odd_payload(),
    ]
    for payload in payloads:
        await _log(python_logger, rust_logger, payload)
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, len(payloads))


@pytest.mark.asyncio
async def test_a_missing_payload_logs_nothing(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    await _log(python_logger, rust_logger, None)
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(_payload(metadata=None), id="metadata-none"),
        pytest.param(_payload(metadata={"user_api_key_team_alias": 7}), id="alias-not-text"),
        pytest.param({k: v for k, v in _payload().items() if k != "id"}, id="no-id"),
        pytest.param(_payload(id=5), id="id-not-text"),
    ],
)
async def test_both_loggers_skip_and_count_the_same_bad_payloads(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_use_team_prefix": True})
    with (
        patch.object(S3Logger, "handle_callback_failure") as python_failures,
        patch.object(s3_v3._Host, "callback_failed") as rust_failures,
    ):
        await _log(python_logger, rust_logger, payload)
        await _flush(python_logger, rust_logger)
        await asyncio.sleep(0.2)
    assert buckets[0].puts() == buckets[1].puts() == []
    assert python_logger.log_queue == []
    assert rust_logger.sink.queued_keys() == []
    assert python_failures.call_count == rust_failures.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("granularity", ["day", "hour"])
async def test_a_cold_storage_key_is_reused_when_it_matches(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, granularity: str
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_path": "cold"})
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", "s3_v2")
    date: Final = "2026-11-13/" if granularity == "day" else "2026-11-13/09/"
    matching: Final = f"cold/{date}time-09-05-07-000042_chatcmpl-1.json"
    for response_id, key in (("chatcmpl-1", matching), ("chatcmpl-2", "elsewhere/x.json")):
        await _log(python_logger, rust_logger, _payload(response_id, metadata={"cold_storage_object_key": key}))
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, 2)
    assert {r.path for r in buckets[1].puts()} == {
        f"/logs/{matching}",
        "/logs/cold/2026-11-13/time-09-05-07-000042_chatcmpl-2.json",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failures", "extra"),
    [
        pytest.param([("_a\\.json", 503, "")], {}, id="partial-503-keeps-the-failed-one"),
        pytest.param([(".", 500, "")], {}, id="bucket-down-keeps-everything"),
        pytest.param([("_a\\.json", 400, "<Error><Code>EntityTooLarge</Code></Error>")], {}, id="terminal-is-dropped"),
        pytest.param(
            [("_a\\.json", 400, "<Error><Code>EntityTooLarge</Code></Error>")],
            {"s3_drop_on_terminal_error": "false"},
            id="terminal-kept-when-dropping-is-off",
        ),
        pytest.param([(".", 503, "")], {"s3_max_queue_size": 2}, id="queue-cap-trims-oldest"),
        pytest.param([("_a\\.json", 403, "<Error><Code>AccessDenied</Code></Error>")], {}, id="403-retried"),
    ],
)
async def test_both_loggers_keep_and_drop_the_same_failed_uploads(
    buckets: tuple[FakeBucket, FakeBucket],
    monkeypatch: pytest.MonkeyPatch,
    failures: list[tuple[str, int, str]],
    extra: Mapping[str, object],
) -> None:
    for bucket in buckets:
        bucket.failures.extend(failures)
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, extra)
    for response_id in ("a", "b", "c"):
        await _log(python_logger, rust_logger, _payload(response_id))
    await _flush(python_logger, rust_logger)
    python_queue: Final = [element.s3_object_key for element in python_logger.log_queue]
    assert rust_logger.sink.queued_keys() == python_queue
    assert len(buckets[1].puts()) == len(buckets[0].puts())
    for bucket in buckets:
        bucket.failures.clear()
    await _flush(python_logger, rust_logger)
    assert rust_logger.sink.queued_keys() == [element.s3_object_key for element in python_logger.log_queue] == []
    assert sorted(r.path for r in buckets[1].puts()) == sorted(r.path for r in buckets[0].puts())


@pytest.mark.asyncio
async def test_batch_files_hold_the_same_lines(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(
        monkeypatch, buckets, {"s3_batch_file_upload": True, "s3_use_team_prefix": True}
    )
    for response_id, team in (("a", "t1"), ("b", "t2"), ("c", "t1")):
        await _log(python_logger, rust_logger, _payload(response_id, metadata={"user_api_key_team_alias": team}))
    await _flush(python_logger, rust_logger)
    python_puts, rust_puts = buckets[0].puts(), buckets[1].puts()
    assert len(python_puts) == len(rust_puts) == 2
    name: Final = re.compile(r"/batch_\d\d-\d\d-\d\d_[0-9a-f]{32}\.jsonl$")
    for python_put, rust_put in zip(
        sorted(python_puts, key=lambda r: r.path), sorted(rust_puts, key=lambda r: r.path), strict=True
    ):
        assert name.sub("", rust_put.path) == name.sub("", python_put.path)
        assert rust_put.body == python_put.body
        assert rust_put.headers["content-type"] == python_put.headers["content-type"] == "application/x-ndjson"


@pytest.mark.asyncio
async def test_audit_events_and_line_items_land_at_the_same_keys(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_path": "audit-base"})
    audit: Final = {"id": "audit-1", "action": "created", "before_value": None, "updated_at": "2026-11-13"}
    await python_logger.async_log_audit_log_event(audit)  # pyright: ignore[reportArgumentType]  # a partial audit payload is enough here
    await rust_logger.async_log_audit_log_event(audit)  # pyright: ignore[reportArgumentType]  # see above
    await _flush(python_logger, rust_logger)
    line_items: Final = {
        "batch_id": "batch/1",
        "part": 3,
        "finished_at": START.timestamp(),
        "lines": [{"custom_id": "x", "status": "ok", "text": "ünïcode"}],
    }
    await python_logger.async_log_batch_line_items(line_items)  # pyright: ignore[reportArgumentType]  # see above
    await rust_logger.async_log_batch_line_items(line_items)  # pyright: ignore[reportArgumentType]  # see above
    python_puts, rust_puts = buckets[0].puts(), buckets[1].puts()
    assert len(python_puts) == len(rust_puts) == 2
    clock: Final = re.compile(r"/\d\d-\d\d-\d\d_")
    for python_put, rust_put in zip(python_puts, rust_puts, strict=True):
        assert clock.sub("/", rust_put.path) == clock.sub("/", python_put.path)
        assert re.sub(r"/\d{4}-\d\d-\d\d/", "/", rust_put.path) == re.sub(r"/\d{4}-\d\d-\d\d/", "/", python_put.path)
        assert rust_put.body == python_put.body
        for name in COMPARED_HEADERS:
            assert rust_put.headers.get(name) == python_put.headers.get(name), name


@pytest.mark.asyncio
async def test_cold_storage_reads_return_the_same_object(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    await _log(python_logger, rust_logger, _payload())
    await _flush(python_logger, rust_logger)
    key: Final = "2026-11-13/time-09-05-07-000042_chatcmpl-1.json"
    python_object: Final = await python_logger.get_proxy_server_request_from_cold_storage_with_object_key(key)
    rust_object: Final = await rust_logger.get_proxy_server_request_from_cold_storage_with_object_key(key)
    assert python_object is not None
    assert rust_object == python_object
    assert await rust_logger.get_proxy_server_request_from_cold_storage_with_object_key("missing.json") is None
    python_get, rust_get = buckets[0].received[-1], buckets[1].received[-1]
    assert rust_get.headers["x-amz-content-sha256"] == python_get.headers["x-amz-content-sha256"]
    assert _signed_headers(rust_get) == _signed_headers(python_get)


@pytest.mark.asyncio
async def test_the_native_logger_answers_the_python_helpers(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(
        monkeypatch, buckets, {"s3_path": "p", "s3_partition_granularity": "hour"}
    )
    assert rust_logger.s3_path == python_logger.s3_path == "p"
    assert rust_logger.resolve_partition_granularity() == python_logger.resolve_partition_granularity() == "hour"
    selection: Final = type(rust_logger).selection
    assert selection is not None
    assert selection.name == "s3_v2"
    assert json.loads(rust_logger.sink.callback().manifest())["subscriptions"][0]["hook"] == "attempt_ended"


@pytest.mark.asyncio
async def test_stripping_base64_rewrites_the_shared_payload_the_same_way(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_strip_base64_files": True})
    original: Final = _payload(
        "with-file",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "see data:image/png;base64,AAAA== inline"},
                    {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBER"}},
                ],
            }
        ],
    )
    python_payload, rust_payload = copy.deepcopy(original), copy.deepcopy(original)
    await _log_each(python_logger, rust_logger, (python_payload, rust_payload))
    assert rust_payload == python_payload != original
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, 1)


@pytest.mark.asyncio
async def test_a_payload_changed_before_the_flush_is_stored_as_it_is_at_the_flush(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    python_payload, rust_payload = _payload(), _payload()
    await _log_each(python_logger, rust_logger, (python_payload, rust_payload))
    for payload in (python_payload, rust_payload):
        payload["metadata"]["late"] = "written after the event"  # pyright: ignore[reportIndexIssue]  # metadata is the dict _payload builds
        payload["response_cost"] = 99.0
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, 1)
    stored: Final = json.loads(buckets[1].puts()[0].body)
    assert stored["metadata"]["late"] == "written after the event"
    assert stored["response_cost"] == 0.000123
    assert rust_logger.sink.stashed() == 0


def test_both_loggers_need_a_running_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "s3_callback_params", _params("http://127.0.0.1:9", {}))
    for flag in ("false", "true"):
        monkeypatch.setenv("LITELLM_NATIVE_S3_V2", flag)
        native_s3_v2_enabled.cache_clear()
        with pytest.raises(RuntimeError, match="no running event loop"):
            new_s3_v2_logger()


@pytest.mark.asyncio
async def test_the_native_logger_flushes_on_its_own_after_the_interval(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s3_v3, "DEFAULT_S3_FLUSH_INTERVAL_SECONDS", 0.05)
    _, rust_logger = await _loggers(monkeypatch, buckets, {})
    await _log_native(rust_logger, {"standard_logging_object": _payload()})
    for _ in range(100):
        if buckets[1].puts():
            break
        await asyncio.sleep(0.02)
    assert [r.path for r in buckets[1].puts()] == ["/logs/2026-11-13/time-09-05-07-000042_chatcmpl-1.json"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("environment", "region"),
    [
        pytest.param({"AWS_REGION_NAME": "eu-central-1"}, "eu-central-1", id="region-name"),
        pytest.param({"AWS_REGION_NAME": "eu-central-1", "AWS_REGION": "ap-south-1"}, "ap-south-1", id="region-wins"),
        pytest.param({}, "us-west-2", id="default"),
        pytest.param({"AWS_REGION": "Bad_Region"}, None, id="invalid"),
    ],
)
async def test_without_a_region_param_both_sign_with_the_same_environment_region(
    buckets: tuple[FakeBucket, FakeBucket],
    monkeypatch: pytest.MonkeyPatch,
    environment: Mapping[str, str],
    region: str | None,
) -> None:
    for name in ("AWS_REGION_NAME", "AWS_REGION"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_region_name": None})
    with (
        patch.object(S3Logger, "handle_callback_failure") as python_failures,
        patch.object(s3_v3._Host, "callback_failed") as rust_failures,
    ):
        await _log(python_logger, rust_logger, _payload())
        await _flush(python_logger, rust_logger)
        await asyncio.sleep(0.2)
    if region is None:
        assert buckets[0].puts() == buckets[1].puts() == []
        assert python_failures.call_count == rust_failures.call_count == 1
        return
    for bucket in buckets:
        put = bucket.puts()[0]
        assert put.headers["authorization"] == _botocore_authorization(put, bucket.port, region=region)


@dataclass
class FakeSts:
    calls: list[Mapping[str, list[str]]] = field(default_factory=list)
    port: int = 0


STS_REPLY: Final = (
    '<AssumeRoleWithWebIdentityResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">'
    "<AssumeRoleWithWebIdentityResult><Credentials><AccessKeyId>AKIDSTS</AccessKeyId>"
    "<SecretAccessKey>sts-secret</SecretAccessKey><SessionToken>sts-token</SessionToken>"
    "<Expiration>2099-01-01T00:00:00Z</Expiration></Credentials>"
    "<SubjectFromWebIdentityToken>subject</SubjectFromWebIdentityToken>"
    "<AssumedRoleUser><Arn>arn:aws:sts::123456789012:assumed-role/logs/s3</Arn>"
    "<AssumedRoleId>AROA:s3</AssumedRoleId></AssumedRoleUser><PackedPolicySize>10</PackedPolicySize>"
    "</AssumeRoleWithWebIdentityResult></AssumeRoleWithWebIdentityResponse>"
)


@pytest.fixture
def sts() -> Iterator[FakeSts]:
    fake: Final = FakeSts()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: object) -> None:
            return

        def do_POST(self) -> None:
            body: Final = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            fake.calls.append(parse_qs(body.decode()))
            reply: Final = STS_REPLY.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/xml")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    fake.port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield fake
    server.shutdown()


@pytest.mark.asyncio
async def test_web_identity_assumes_the_role_with_the_same_session_policy(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, sts: FakeSts
) -> None:
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_EXTERNAL_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("S3_V3_TEST_OIDC_TOKEN", "oidc-token-value")
    identity: Final = {
        "s3_aws_access_key_id": None,
        "s3_aws_secret_access_key": None,
        "s3_aws_session_token": None,
        "s3_aws_web_identity_token": "oidc/env/S3_V3_TEST_OIDC_TOKEN",
        "s3_aws_role_name": "arn:aws:iam::123456789012:role/logs",
        "s3_aws_session_name": f"s3-{sts.port}",
        "s3_aws_sts_endpoint": f"http://127.0.0.1:{sts.port}",
    }
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, identity)
    await _log(python_logger, rust_logger, _payload())
    await _flush(python_logger, rust_logger)
    assert len(sts.calls) == 2
    python_call, rust_call = sts.calls
    for name in ("Action", "RoleArn", "RoleSessionName", "WebIdentityToken", "DurationSeconds", "Policy"):
        assert rust_call.get(name) == python_call.get(name), name
    assert rust_call["WebIdentityToken"] == ["oidc-token-value"]
    assert json.loads(rust_call["Policy"][0]) == json.loads(S3Logger._web_identity_session_policy_document())
    for bucket in buckets:
        put = bucket.puts()[0]
        assert put.headers["x-amz-security-token"] == "sts-token"
        assert put.headers["authorization"] == _botocore_authorization(
            put, bucket.port, credentials=("AKIDSTS", "sts-secret", "sts-token")
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("global_disable", [False, True])
@pytest.mark.parametrize("no_log", [None, True, "true"])
@pytest.mark.parametrize("disabled_header", [None, "s3_v2", "langfuse, S3_V2"])
async def test_the_native_opt_out_matches_should_run_callback(
    buckets: tuple[FakeBucket, FakeBucket],
    monkeypatch: pytest.MonkeyPatch,
    global_disable: bool,
    no_log: object,
    disabled_header: str | None,
) -> None:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy import proxy_server

    monkeypatch.setattr(litellm, "global_disable_no_log_param", global_disable)
    monkeypatch.setattr(litellm, "allow_dynamic_callback_disabling", True)
    monkeypatch.setattr(proxy_server, "premium_user", True)
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    logging_obj: Final = Logging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="call-1",
        start_time=START,
        function_id="fn-1",
    )
    litellm_params: Final = {
        **({} if no_log is None else {"no-log": no_log}),
        "proxy_server_request": {
            "headers": {} if disabled_header is None else {"x-litellm-disable-callbacks": disabled_header}
        },
    }
    expected: Final = logging_obj.should_run_callback(python_logger, litellm_params, "async_success_handler")
    kwargs: Final = {
        "litellm_params": litellm_params,
        "standard_callback_dynamic_params": logging_obj.standard_callback_dynamic_params,
        "standard_logging_object": _payload(),
    }
    assert s3_v3.legacy_should_log(kwargs) is expected
    await _log_native(rust_logger, kwargs)
    await rust_logger.flush_queue()
    assert len(buckets[1].puts()) == int(expected)


async def _until_queued(python_logger: S3Logger, rust_logger: S3LoggerV3) -> None:
    for _ in range(100):
        if python_logger.log_queue and rust_logger.sink.queued_keys():
            return
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_scope", [False, True])
@pytest.mark.parametrize("global_disable", [False, True])
@pytest.mark.parametrize("no_log", [False, True])
async def test_an_acompletion_reaches_both_loggers_alike(
    buckets: tuple[FakeBucket, FakeBucket],
    monkeypatch: pytest.MonkeyPatch,
    global_disable: bool,
    no_log: bool,
    proxy_scope: bool,
) -> None:
    monkeypatch.setattr(litellm, "global_disable_no_log_param", global_disable)
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_use_team_prefix": True})
    monkeypatch.setattr(litellm, "callbacks", [python_logger, rust_logger])
    with request_scope(RequestSelection(no_log=no_log and not global_disable)) if proxy_scope else nullcontext():
        await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello «world»"}],
            mock_response="hi there",
            metadata={"user_api_key_team_alias": "team-a"},
            **({"no-log": True} if no_log else {}),
        )
    await _until_queued(python_logger, rust_logger)
    await _flush(python_logger, rust_logger)
    expected: Final = 0 if no_log and not global_disable else 1
    _assert_same_puts(buckets, expected)
    assert all(put.path.startswith("/logs/team-a/") for put in buckets[1].puts())


class _Mutating:
    def __init__(self, target: dict[str, object]) -> None:
        self.target: Final = target

    def __str__(self) -> str:
        self.target[f"grown-{len(self.target)}"] = True
        return "mutating"


@pytest.mark.asyncio
async def test_a_payload_that_changes_while_it_is_rendered_is_kept_for_retry_by_both_loggers(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    payloads: Final = (_payload("racing"), _payload("racing"))
    for payload in payloads:
        metadata = payload["metadata"]
        assert isinstance(metadata, dict)
        metadata["mutating"] = _Mutating(metadata)  # pyright: ignore[reportUnknownArgumentType]  # the metadata dict _payload builds
    with (
        patch.object(S3Logger, "handle_callback_failure") as python_failures,
        patch.object(s3_v3._Host, "callback_failed") as rust_failures,
    ):
        await _log_each(python_logger, rust_logger, payloads)
        await _log(python_logger, rust_logger, _payload("steady"))
        await _flush(python_logger, rust_logger)
        await asyncio.sleep(0.2)
    assert rust_logger.sink.queued_keys() == [element.s3_object_key for element in python_logger.log_queue]
    assert len(rust_logger.sink.queued_keys()) == 1
    assert python_failures.call_count == rust_failures.call_count == 1
    _assert_same_puts(buckets, 1)
    assert not rust_logger.periodic_flush_task.done()


class _FlakySink:
    def __init__(self, failures: tuple[BaseException, ...]) -> None:
        self.failures: Final = list(failures)
        self.flushed = 0

    async def flush(self) -> None:
        self.flushed += 1
        if self.failures:
            raise self.failures.pop(0)


class _Panic(BaseException):
    pass


@pytest.mark.asyncio
async def test_the_periodic_flush_keeps_running_after_a_flush_fails() -> None:
    sink: Final = _FlakySink((RuntimeError("bucket gone"), _Panic("a PanicException is a BaseException too")))
    task: Final = asyncio.create_task(s3_v3._periodic_flush(sink, 0.01))  # pyright: ignore[reportArgumentType]  # the loop only calls flush
    for _ in range(100):
        if sink.flushed >= 3:
            break
        await asyncio.sleep(0.01)
    assert sink.flushed >= 3
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _settled(pair: tuple[FakeBucket, FakeBucket], python_logger: S3Logger, rust_logger: S3LoggerV3) -> None:
    for _ in range(3):
        await asyncio.sleep(0.3)
        await _flush(python_logger, rust_logger)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_a_sync_completion_reaches_both_loggers_alike(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    monkeypatch.setattr(litellm, "callbacks", [python_logger, rust_logger])

    def call() -> None:
        response = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            mock_response="hi there",
            stream=stream,
        )
        if stream:
            for _ in response:  # pyright: ignore[reportGeneralTypeIssues]  # a stream when stream=True
                pass

    await asyncio.to_thread(call)
    await _settled(buckets, python_logger, rust_logger)
    _assert_same_puts(buckets, 0)


@pytest.mark.asyncio
async def test_a_sync_line_delivery_is_skipped_by_the_native_logger(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, rust_logger = await _loggers(monkeypatch, buckets, {})
    await rust_logger.run_native(
        Hook.attempt_ended,
        frozenset({Field.deprecated_kwargs}),
        SimpleNamespace(deprecated_kwargs={"standard_logging_object": _payload()}),
        SimpleNamespace(call=None, state=None),  # pyright: ignore[reportArgumentType]  # run_native reads only call and state
    )
    assert rust_logger.sink.queued_keys() == []
    assert rust_logger.sink.stashed() == 0
