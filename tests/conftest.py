"""Shared test scaffolding.

Two harnesses drive the clients offline against their *real* implementations:

* ``http_mock`` / ``tracking_transport`` — :class:`httpx.MockTransport` builders.
  Feed the transport to a real SDK client (``http_client=AsyncClient(transport=...)``)
  or the built-in client (``client._transport = ...``); ``Recorder`` captures what
  was sent so tests can assert the request the adapter produced.
* ``bedrock_stub`` — a real ``bedrock-runtime`` client with ``invoke_model`` stubbed
  via :class:`botocore.stub.Stubber`, returning a canned ``StreamingBody``.

Plus a Protocol-satisfying :class:`MockLLMClient` for the orchestration tests (which
shouldn't care about any wire protocol) and the data factories.
"""

import asyncio
import io
import uuid
from collections.abc import Callable
from typing import Any, Protocol

import boto3
import httpx
import orjson
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

from tournament_eval import GenerationResponse, StructuredResponse
from tournament_eval.models import GenerationResult, GenerationTask, RankingTask

# --------------------------------------------------------------------------- #
# httpx MockTransport — drives the real httpx-based clients/SDKs offline        #
# --------------------------------------------------------------------------- #


class Recorder:
    """Captures the requests an httpx-based client sends."""

    def __init__(self) -> None:
        self.calls = 0
        self.request: httpx.Request | None = None

    @property
    def json(self) -> dict[str, Any]:
        """The parsed JSON body of the most recent request."""
        assert self.request is not None, "no request was captured"
        parsed: dict[str, Any] = orjson.loads(self.request.content)
        return parsed


_Response = dict[str, Any] | tuple[int, dict[str, Any]]
_HttpMock = Callable[..., tuple[httpx.MockTransport, Recorder]]


@pytest.fixture
def http_mock() -> _HttpMock:
    """Build a ``(MockTransport, Recorder)`` from canned responses.

    Each positional arg is a body dict (-> ``200``) or ``(status, body)``; successive
    requests get successive responses, the last repeating.
    """

    def _make(*responses: _Response) -> tuple[httpx.MockTransport, Recorder]:
        specs: list[tuple[int, dict[str, Any]]] = [(200, r) if isinstance(r, dict) else r for r in (responses or ({},))]
        recorder = Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.request = request
            recorder.calls += 1
            status, body = specs[min(recorder.calls - 1, len(specs) - 1)]
            return httpx.Response(status, json=body)

        return httpx.MockTransport(handler), recorder

    return _make


@pytest.fixture
def tracking_transport() -> tuple[httpx.MockTransport, dict[str, int]]:
    """An async MockTransport (chat-completions body) recording peak concurrency."""
    state = {"inflight": 0, "peak": 0}

    async def handler(_request: httpx.Request) -> httpx.Response:
        state["inflight"] += 1
        state["peak"] = max(state["peak"], state["inflight"])
        await asyncio.sleep(0.01)
        state["inflight"] -= 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    return httpx.MockTransport(handler), state


# --------------------------------------------------------------------------- #
# botocore Stubber — drives the real bedrock-runtime client offline             #
# --------------------------------------------------------------------------- #

_BedrockStub = Callable[..., tuple[Any, Stubber]]


@pytest.fixture
def bedrock_stub() -> _BedrockStub:
    """Build a real ``bedrock-runtime`` client with ``invoke_model`` stubbed.

    ``response`` is serialized to the canned ``StreamingBody``.  Use inside
    ``with stubber:`` so the stub is active for the call.
    """

    def _make(response: object, *, expected_params: dict[str, Any] | None = None) -> tuple[Any, Stubber]:
        client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        data = orjson.dumps(response)
        stubber = Stubber(client)
        stubber.add_response(
            "invoke_model",
            {"body": StreamingBody(io.BytesIO(data), len(data)), "contentType": "application/json"},
            expected_params,
        )
        return client, stubber

    return _make


# --------------------------------------------------------------------------- #
# Protocol-satisfying mock client — for the orchestration tests                 #
# --------------------------------------------------------------------------- #


class MockLLMClient:
    """A canned-response client; satisfies the LLMClient protocol structurally."""

    def __init__(
        self,
        name: str,
        *,
        generate_responses: dict[str, str] | None = None,
        structured_responses: dict[str, dict[str, object]] | None = None,
        fail_on: set[str] | None = None,
    ) -> None:
        self._name = name
        self._generate = generate_responses or {}
        self._structured = structured_responses or {}
        self._fail_on = fail_on or set()

    @property
    def name(self) -> str:
        return self._name

    async def generate(self, prompt: str) -> GenerationResponse:
        if prompt in self._fail_on:
            raise RuntimeError(f"Mock generate failure: {prompt}")
        if prompt not in self._generate:
            raise RuntimeError(f"No mock response for prompt: {prompt!r}")
        return GenerationResponse(text=self._generate[prompt], reasoning=None)

    async def generate_structured(self, prompt: str, _schema: dict[str, object]) -> StructuredResponse:
        if prompt in self._fail_on:
            raise RuntimeError(f"Mock structured failure: {prompt}")
        if prompt not in self._structured:
            raise RuntimeError(f"No mock structured response for prompt: {prompt!r}")
        data = self._structured[prompt]
        return StructuredResponse(data=data, raw=str(data))


@pytest.fixture
def make_client() -> Callable[..., MockLLMClient]:
    """Return a factory for :class:`MockLLMClient` instances."""

    def _factory(
        name: str = "mock",
        *,
        generate_responses: dict[str, str] | None = None,
        structured_responses: dict[str, dict[str, object]] | None = None,
        fail_on: set[str] | None = None,
    ) -> MockLLMClient:
        return MockLLMClient(
            name=name,
            generate_responses=generate_responses,
            structured_responses=structured_responses,
            fail_on=fail_on,
        )

    return _factory


# --------------------------------------------------------------------------- #
# Data factories                                                                #
# --------------------------------------------------------------------------- #


class _MakeTask(Protocol):
    def __call__(self, prompt: str = "test prompt") -> GenerationTask: ...


class _MakeGeneration(Protocol):
    def __call__(
        self,
        *,
        author: str = "test-model",
        task_id: uuid.UUID | None = None,
        output: str = "some output",
        reasoning: str | None = None,
    ) -> GenerationResult: ...


class _MakeRankingTask(Protocol):
    def __call__(
        self,
        *,
        ranking_prompt: str = "Rank.",
        generations: dict[str, uuid.UUID] | None = None,
        generation_task_id: uuid.UUID | None = None,
    ) -> RankingTask: ...


@pytest.fixture
def make_task() -> _MakeTask:
    def _factory(prompt: str = "test prompt") -> GenerationTask:
        return GenerationTask(id=uuid.uuid4(), generation_prompt=prompt)

    return _factory


@pytest.fixture
def make_generation() -> _MakeGeneration:
    def _factory(
        *,
        author: str = "test-model",
        task_id: uuid.UUID | None = None,
        output: str = "some output",
        reasoning: str | None = None,
    ) -> GenerationResult:
        return GenerationResult(
            id=uuid.uuid4(),
            task_id=task_id or uuid.uuid4(),
            generation_prompt="test prompt",
            output=output,
            reasoning=reasoning,
            author=author,
        )

    return _factory


@pytest.fixture
def make_ranking_task() -> _MakeRankingTask:
    def _factory(
        *,
        ranking_prompt: str = "Rank.",
        generations: dict[str, uuid.UUID] | None = None,
        generation_task_id: uuid.UUID | None = None,
    ) -> RankingTask:
        return RankingTask(
            id=uuid.uuid4(),
            generation_task_id=generation_task_id or uuid.uuid4(),
            ranking_prompt=ranking_prompt,
            generations=generations or {},
        )

    return _factory
