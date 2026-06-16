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

from tournament_eval.llm.base import GenerationResponse, LLMClient, StructuredResponse
from tournament_eval.models import (
    GenerationFailure,
    GenerationResult,
    GenerationTask,
    RankingFailure,
    RankingResult,
    RankingTask,
)


class FakeTransport(httpx.MockTransport):
    def __init__(self) -> None:
        self.call_count = 0
        self.handler = self._handler
        self.error_code = 429
        self.return_body = {"choices": [{"message": {"content": "ok"}}]}
        self.fail_amount = 0
        self.raise_error = False

    def _handler(self, _: httpx.Request) -> httpx.Response:
        if self.fail_amount == self.call_count:
            self.call_count += 1
            return httpx.Response(200, json=self.return_body)
        else:
            self.call_count += 1
            if self.raise_error:
                raise httpx.TransportError("boom")
            return httpx.Response(self.error_code, json=self.return_body)


class MockLLMClient(LLMClient):
    def __init__(self) -> None:
        self.generation_response = GenerationResponse(text="ok", reasoning="because")
        self.structured_response = StructuredResponse(data={"a": 1}, raw='{"a": 1}')
        self.model = "test-model"
        self.raise_on_enter = False
        self.raise_on_exit = False

    @property
    def name(self) -> str:
        return self.model

    async def generate(self, prompt: str) -> GenerationResponse:  # noqa: ARG002
        return self.generation_response

    async def generate_structured(self, prompt: str, schema: dict[str, object]) -> StructuredResponse:  # noqa: ARG002
        return self.structured_response

    async def __aenter__(self) -> MockLLMClient:
        if self.raise_on_enter:
            raise RuntimeError("Enter boom")
        return self

    async def __aexit__(self, *args: Any, **kwargs: Any) -> None:
        if self.raise_on_exit:
            raise RuntimeError("Exit boom")


@pytest.fixture
def test_transport() -> Callable[..., FakeTransport]:
    def _make() -> FakeTransport:
        return FakeTransport()

    return _make


@pytest.fixture
def bedrock_stub() -> Callable[[Any], tuple[Any, Stubber]]:
    """A real ``bedrock-runtime`` client with ``invoke_model`` stubbed.

    Pass the JSON the model should "return"; get back the client and an un-entered
    :class:`~botocore.stub.Stubber` (drive it with ``with stubber: ...``).  The payload
    is wrapped in a :class:`~botocore.response.StreamingBody` exactly as boto hands it
    back, so the transport's ``body.read()`` path is exercised for real.
    """

    def _make(response_body: Any) -> tuple[Any, Stubber]:
        client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        stubber = Stubber(client)
        payload = orjson.dumps(response_body)
        stubber.add_response(
            "invoke_model",
            {"body": StreamingBody(io.BytesIO(payload), len(payload)), "contentType": "application/json"},
        )
        return client, stubber

    return _make


class GenerationTaskFactory(Protocol):
    def __call__(self, *, id: uuid.UUID | None = None, prompt: str = "test prompt") -> GenerationTask: ...


@pytest.fixture
def make_generation_task() -> GenerationTaskFactory:
    def _factory(*, id: uuid.UUID | None = None, prompt: str = "test prompt") -> GenerationTask:
        return GenerationTask(id=id or uuid.uuid4(), generation_prompt=prompt)

    return _factory


class GenerationResultFactory(Protocol):
    def __call__(
        self,
        *,
        id: uuid.UUID | None = None,
        generation_task_id: uuid.UUID | None = None,
        generation_prompt: str = "test prompt",
        output: str = "some output",
        reasoning: str | None = "some reasoning",
        author: str = "test-model",
        metadata: dict[str, object] | None = None,
    ) -> GenerationResult: ...


class GenerationFailureFactory(Protocol):
    def __call__(
        self,
        *,
        generation_task_id: uuid.UUID | None = None,
        author: str = "test-model",
        error_type: str = "RuntimeError",
        message: str = "boom!",
    ) -> GenerationFailure: ...


@pytest.fixture
def make_generation_failure() -> GenerationFailureFactory:
    def _factory(
        *,
        generation_task_id: uuid.UUID | None = None,
        author: str = "test-model",
        error_type: str = "RuntimeError",
        message: str = "boom!",
    ) -> GenerationFailure:
        return GenerationFailure(
            generation_task_id=generation_task_id or uuid.uuid4(),
            author=author,
            error_type=error_type,
            message=message,
        )

    return _factory


@pytest.fixture
def make_generation_result() -> GenerationResultFactory:
    def _factory(
        *,
        id: uuid.UUID | None = None,
        generation_task_id: uuid.UUID | None = None,
        generation_prompt: str = "test prompt",
        output: str = "some output",
        reasoning: str | None = "some reasoning",
        author: str = "test-model",
        metadata: dict[str, object] | None = None,
    ) -> GenerationResult:
        return GenerationResult(
            id=id or uuid.uuid4(),
            generation_task_id=generation_task_id or uuid.uuid4(),
            generation_prompt=generation_prompt,
            output=output,
            reasoning=reasoning,
            author=author,
            metadata=metadata or {},
        )

    return _factory


class RankingTaskFactory(Protocol):
    def __call__(
        self,
        *,
        id: uuid.UUID | None = None,
        generation_task_id: uuid.UUID | None = None,
        ranking_prompt: str = "Rank.",
        generations: dict[str, uuid.UUID] | None = None,
    ) -> RankingTask: ...


@pytest.fixture
def make_ranking_task() -> RankingTaskFactory:
    def _factory(
        *,
        id: uuid.UUID | None = None,
        generation_task_id: uuid.UUID | None = None,
        ranking_prompt: str = "Rank.",
        generations: dict[str, uuid.UUID] | None = None,
    ) -> RankingTask:
        return RankingTask(
            id=id or uuid.uuid4(),
            generation_task_id=generation_task_id or uuid.uuid4(),
            ranking_prompt=ranking_prompt,
            generations=generations or {},
        )

    return _factory


class RankingFailureFactory(Protocol):
    def __call__(
        self,
        *,
        ranking_task_id: uuid.UUID | None = None,
        author: str = "test-model",
        error_type: str = "RuntimeError",
        message: str = "boom!",
    ) -> RankingFailure: ...


@pytest.fixture
def make_ranking_failure() -> RankingFailureFactory:
    def _factory(
        *,
        ranking_task_id: uuid.UUID | None = None,
        author: str = "test-model",
        error_type: str = "RuntimeError",
        message: str = "boom!",
    ) -> RankingFailure:
        return RankingFailure(
            ranking_task_id=ranking_task_id or uuid.uuid4(),
            author=author,
            error_type=error_type,
            message=message,
        )

    return _factory


class RankingResultFactory(Protocol):
    def __call__(
        self,
        *,
        id: uuid.UUID | None = None,
        ranking_task_id: uuid.UUID | None = None,
        ranking_prompt: str = "Rank.",
        author: str = "test-model",
        raw_model_ranking: list[str] | None = None,
        ranking: list[uuid.UUID] | None = None,
        reasoning: str | None = None,
        raw_response: str = "some raw response",
        metadata: dict[str, object] | None = None,
    ) -> RankingResult: ...


@pytest.fixture
def make_ranking_result() -> RankingResultFactory:
    def _factory(
        *,
        id: uuid.UUID | None = None,
        ranking_task_id: uuid.UUID | None = None,
        ranking_prompt: str = "Rank.",
        author: str = "test-model",
        raw_model_ranking: list[str] | None = None,
        ranking: list[uuid.UUID] | None = None,
        reasoning: str | None = None,
        raw_response: str = "some raw response",
        metadata: dict[str, object] | None = None,
    ) -> RankingResult:
        return RankingResult(
            id=id or uuid.uuid4(),
            ranking_task_id=ranking_task_id or uuid.uuid4(),
            ranking_prompt=ranking_prompt,
            author=author,
            raw_model_ranking=raw_model_ranking or [],
            ranking=ranking or [],
            reasoning=reasoning,
            raw_response=raw_response,
            metadata=metadata or {},
        )

    return _factory
