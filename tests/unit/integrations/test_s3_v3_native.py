import asyncio
import json
import re
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Final
from unittest.mock import patch

import pytest
from pydantic import BaseModel

import litellm
from litellm.integrations import s3_v2, s3_v3
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.s3_v2 import S3Logger
from litellm.integrations.s3_v3 import S3LoggerV3, native_s3_v2_enabled, new_s3_v2_logger
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


async def _log(python_logger: S3Logger, rust_logger: S3LoggerV3, payload: object) -> None:
    kwargs: Final = {"standard_logging_object": payload}
    await python_logger.async_log_success_event(kwargs, None, START, START)
    await rust_logger.run_native(
        Hook.attempt_ended,
        frozenset({Field.standard_logging_payload}),
        SimpleNamespace(standard_logging_payload=payload),
        SimpleNamespace(call=None, state=None),  # pyright: ignore[reportArgumentType]  # run_native reads only call and state
    )


async def _flush(python_logger: S3Logger, rust_logger: S3LoggerV3) -> None:
    await python_logger.flush_queue()
    await rust_logger.flush_queue()


def _signed_headers(request: Received) -> str:
    match: Final = re.search(r"SignedHeaders=([^,]+)", request.headers["authorization"])
    assert match is not None
    return match.group(1)


def _botocore_authorization(request: Received, port: int) -> str:
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
        S3SigV4Auth(Credentials("AKIDEXAMPLE", "secret", "session-token"), "s3", "us-east-1").add_auth(aws_request)
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
