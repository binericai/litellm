import asyncio
import base64
import binascii
import json
import struct
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from typing import Final, cast  # noqa: TID251  # narrows the untyped Messages entries to their call shape

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.messages.native import handler as python_messages
from tests.test_litellm_rust.messages.data import MESSAGES, MESSAGES_EVENTS, MESSAGES_RESPONSE, RATE_LIMITED
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordedRequest, RecordingServer, ResponseSpec, relaxed

pytestmark = pytest.mark.requires_rust_extension

BEDROCK_MODEL: Final = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
AWS: Final = {
    "aws_access_key_id": "test-access-key",
    "aws_secret_access_key": "test-secret-key",
    "aws_region_name": "us-east-1",
}
GUARDRAIL_TRACE: Final[dict[str, object]] = {
    "guardrail": {
        "input": {"gr-1": {"invocationMetrics": {"usage": {"contentPolicyUnits": 1, "topicPolicyUnits": 1}}}},
        "outputs": [],
    }
}


@pytest.fixture
def server(server: RecordingServer) -> Iterator[RecordingServer]:
    yield from relaxed(server)


@pytest.fixture(autouse=True)
def no_ambient_bedrock_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_BEDROCK_RUNTIME_ENDPOINT", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)


_OBJECT: Final = TypeAdapter(dict[str, object])
_RUST_ENTRY: Final = cast(  # cast-ok: the public entry's call shape, which dispatch sends to Rust
    Callable[..., Awaitable[object]], litellm.anthropic.messages.acreate
)
_PYTHON_ENTRY: Final = cast(  # cast-ok: the Python handler's call shape, which no dispatch reroutes
    Callable[..., Awaitable[object]], python_messages.anthropic_messages
)


async def _rust(server: RecordingServer, **kwargs: object) -> object:
    return await _RUST_ENTRY(**_arguments(server, kwargs))


async def _python(server: RecordingServer, **kwargs: object) -> object:
    return await _PYTHON_ENTRY(**_arguments(server, kwargs))


def _arguments(server: RecordingServer, kwargs: Mapping[str, object]) -> dict[str, object]:
    return {
        "model": BEDROCK_MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 64,
        "api_base": server.base_url,
        "litellm_call_id": str(uuid.uuid4()),
        **AWS,
        **kwargs,
    }


def _signed_headers(request: RecordedRequest) -> str:
    authorization: Final = request.headers["authorization"]
    assert authorization.startswith("AWS4-HMAC-SHA256 ")
    return authorization.split("SignedHeaders=", 1)[1].split(",", 1)[0]


def _public(response: object) -> object:
    return {name: value for name, value in _OBJECT.validate_python(response).items() if name != "_hidden_params"}


@pytest.mark.asyncio
async def test_bedrock_invoke_sends_and_answers_what_python_does(server: RecordingServer) -> None:
    server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)

    python_response: Final = await _python(server)
    rust_response: Final = await _rust(server)

    python_sent, rust_sent = server.requests
    assert rust_sent.path == python_sent.path
    assert rust_sent.body == python_sent.body
    assert _signed_headers(rust_sent) == _signed_headers(python_sent)
    assert _public(rust_response) == _public(python_response)


@pytest.mark.asyncio
async def test_bedrock_invoke_upstream_failure_raises_the_bedrock_exception_python_raises(
    server: RecordingServer,
) -> None:
    server.default_response = ResponseSpec(body=RATE_LIMITED, status=429)

    with pytest.raises(litellm.RateLimitError) as python_raised:
        await _python(server)
    with pytest.raises(litellm.RateLimitError) as rust_raised:
        await _rust(server)

    assert str(rust_raised.value) == str(python_raised.value)
    assert rust_raised.value.llm_provider == python_raised.value.llm_provider == "bedrock"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped
    assert rust_raised.value.status_code == python_raised.value.status_code == 429


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("thread", "error_class"),
    (
        pytest.param({"type": "create"}, litellm.BadRequestError, id="create"),
        pytest.param({"type": "continue", "previous_message_id": "msg_1"}, litellm.NotFoundError, id="continue"),
    ),
)
async def test_bedrock_invoke_refuses_a_thread_as_python_does(
    server: RecordingServer, thread: Mapping[str, str], error_class: type[Exception]
) -> None:
    with pytest.raises(error_class) as python_raised:
        await _python(server, thread=dict(thread))
    with pytest.raises(error_class) as rust_raised:
        await _rust(server, thread=dict(thread))

    assert str(rust_raised.value) == str(python_raised.value)
    assert server.requests == []


@pytest.mark.asyncio
async def test_bedrock_invoke_drops_a_thread_under_drop_params_as_python_does(server: RecordingServer) -> None:
    server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)

    await _python(server, thread={"type": "create"}, drop_params=True)
    await _rust(server, thread={"type": "create"}, drop_params=True)

    python_sent, rust_sent = server.requests
    assert "thread" not in _OBJECT.validate_python(rust_sent.body)
    assert rust_sent.body == python_sent.body


class _GuardrailBilling(CustomLogger):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger's untyped init
        self.billed: list[object] = []

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.billed.append(_OBJECT.validate_python(kwargs.get("standard_logging_object")).get("guardrail_information"))

    async def wait_for(self, count: int) -> None:
        async with asyncio.timeout(5):
            while len(self.billed) < count:
                await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_bedrock_invoke_bills_the_guardrail_trace_as_python_does(server: RecordingServer) -> None:
    server.default_response = ResponseSpec(body={**MESSAGES_RESPONSE, "amazon-bedrock-trace": GUARDRAIL_TRACE})
    guardrail: Final = {"guardrailConfig": {"guardrailIdentifier": "gr-1", "guardrailVersion": "1"}}
    billing: Final = _GuardrailBilling()

    with rebound(litellm, "callbacks", [billing]):
        await _python(server, **guardrail)
        await billing.wait_for(1)
        await _rust(server, **guardrail)
        await billing.wait_for(2)

    python_sent, rust_sent = server.requests
    python_billed, rust_billed = billing.billed
    assert python_billed
    assert rust_billed == python_billed
    assert {name: value for name, value in rust_sent.headers.items() if name.startswith("x-amzn-bedrock-")} == {
        name: value for name, value in python_sent.headers.items() if name.startswith("x-amzn-bedrock-")
    }


def _aws_header(name: str, value: str) -> bytes:
    return bytes((len(name),)) + name.encode() + b"\x07" + struct.pack("!H", len(value)) + value.encode()


def _aws_frame(event: Mapping[str, object]) -> bytes:
    payload: Final = json.dumps({"bytes": base64.b64encode(json.dumps(event).encode()).decode()}).encode()
    headers: Final = (
        _aws_header(":event-type", "chunk")
        + _aws_header(":content-type", "application/json")
        + _aws_header(":message-type", "event")
    )
    prelude: Final = struct.pack("!II", 12 + len(headers) + len(payload) + 4, len(headers))
    prelude_crc: Final = struct.pack("!I", binascii.crc32(prelude) & 0xFFFFFFFF)
    message: Final = prelude + prelude_crc + headers + payload
    return message + struct.pack("!I", binascii.crc32(message) & 0xFFFFFFFF)


def _event_stream(final: Mapping[str, object]) -> bytes:
    *head, (_, stop) = MESSAGES_EVENTS
    events: Final = (*(data for _, data in head), {**_OBJECT.validate_python(stop), **final})
    return b"".join(_aws_frame(event) for event in events)


def _sse_events(chunks: Sequence[bytes]) -> tuple[object, ...]:
    frames: Final = b"".join(chunks).decode().split("\n\n")
    return tuple(json.loads(frame.split("data: ", 1)[1]) for frame in frames if frame)


async def _drain(response: object) -> tuple[bytes, ...]:
    stream: Final = cast(AsyncIterator[bytes], response)  # cast-ok: a streamed Messages answer yields SSE bytes
    return tuple([chunk async for chunk in stream])


@pytest.mark.asyncio
async def test_bedrock_invoke_stream_bills_the_guardrail_trace_of_message_stop_as_python_does(
    server: RecordingServer,
) -> None:
    server.default_response = ResponseSpec(
        body=_event_stream({"amazon-bedrock-trace": GUARDRAIL_TRACE}),
        headers={"content-type": "application/vnd.amazon.eventstream"},
    )
    guardrail: Final = {"guardrailConfig": {"guardrailIdentifier": "gr-1", "guardrailVersion": "1"}}
    billing: Final = _GuardrailBilling()

    with rebound(litellm, "callbacks", [billing]):
        python_chunks: Final = await _drain(await _python(server, stream=True, **guardrail))
        await billing.wait_for(1)
        rust_chunks: Final = await _drain(await _rust(server, stream=True, **guardrail))
        await billing.wait_for(2)

    python_billed, rust_billed = billing.billed
    assert python_billed
    assert rust_billed == python_billed
    assert _sse_events(rust_chunks)[-1] == _sse_events(python_chunks)[-1]


@pytest.fixture
def no_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ROLE_NAME", "AWS_WEB_IDENTITY_TOKEN_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_aws_credentials")
@pytest.mark.parametrize(
    "credentials",
    (
        pytest.param({}, id="no-credentials"),
        pytest.param({"aws_web_identity_token": "os.environ/AWS_WEB_IDENTITY_TOKEN"}, id="environment-reference"),
        pytest.param({"aws_web_identity_token": "bedrock-web-identity-token-not-set"}, id="unresolved-token"),
    ),
)
async def test_bedrock_invoke_credential_failure_raises_what_python_raises(
    server: RecordingServer, credentials: Mapping[str, str]
) -> None:
    web_identity: Final = (
        {"aws_role_name": "arn:aws:iam::123456789012:role/bedrock", "aws_session_name": "litellm-session"}
        if credentials
        else {}
    )
    unsigned: Final = {"aws_access_key_id": None, "aws_secret_access_key": None, **credentials, **web_identity}

    with pytest.raises(Exception) as python_raised:
        await _python(server, **unsigned)
    with pytest.raises(Exception) as rust_raised:
        await _rust(server, **unsigned)

    assert type(rust_raised.value) is type(python_raised.value)
    assert str(rust_raised.value) == str(python_raised.value)
    assert getattr(rust_raised.value, "status_code", None) == getattr(python_raised.value, "status_code", None)
    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment",
    (
        pytest.param(
            {"model_id": "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"},
            id="model-id-arn",
        ),
        pytest.param(
            {
                "model": "bedrock/invoke/arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
            },
            id="invoke-arn",
        ),
        pytest.param({"stream": False, "drop_params": True, "temperature": 0.2}, id="sampling"),
    ),
)
async def test_bedrock_invoke_sends_the_url_and_body_python_sends(
    server: RecordingServer, deployment: Mapping[str, object]
) -> None:
    server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)

    await _python(server, **deployment)
    await _rust(server, **deployment)

    python_sent, rust_sent = server.requests
    assert rust_sent.path == python_sent.path
    assert rust_sent.body == python_sent.body


@pytest.mark.asyncio
async def test_bedrock_invoke_refuses_an_invalid_guardrail_config_as_python_does(server: RecordingServer) -> None:
    invalid: Final = {"guardrailConfig": {"guardrailVersion": "1"}}

    with pytest.raises(Exception) as python_raised:
        await _python(server, **invalid)
    with pytest.raises(Exception) as rust_raised:
        await _rust(server, **invalid)

    assert str(rust_raised.value) == str(python_raised.value)
    assert type(rust_raised.value) is type(python_raised.value)
    assert getattr(rust_raised.value, "status_code", None) == getattr(python_raised.value, "status_code", None)
    assert server.requests == []


@pytest.mark.asyncio
async def test_bedrock_invoke_refuses_a_web_search_tool_as_python_does(server: RecordingServer) -> None:
    web_search: Final = {"tools": [{"type": "web_search_20250305", "name": "web_search"}]}

    with pytest.raises(Exception) as python_raised:
        await _python(server, **web_search)
    with pytest.raises(Exception) as rust_raised:
        await _rust(server, **web_search)

    assert type(rust_raised.value) is type(python_raised.value)
    assert str(rust_raised.value) == str(python_raised.value)
    assert rust_raised.value.llm_provider == python_raised.value.llm_provider  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the public error's provider
    assert rust_raised.value.model == python_raised.value.model  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the public error's model
    assert server.requests == []
