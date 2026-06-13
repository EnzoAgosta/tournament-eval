"""Shared pytest fixtures, fake clients, and HTTP handlers.

All reusable test scaffolding lives here so test modules contain only tests:

* :class:`FakeHTTPLLMClient` / ``make_http_client`` — a trivial concrete
  :class:`HTTPLLMClient` for exercising the shared request/template/lifecycle.
* ``make_handler`` / ``tracking_handler`` — ``httpx.MockTransport`` handlers.
* ``make_openai_client`` / ``make_ollama_client`` — real provider clients.
* :class:`MockLLMClient` / ``make_client`` — a canned-response client for the
  orchestration tests.
* ``make_task`` / ``make_generation`` / ``make_ranking_task`` — data factories.
"""

import asyncio
import contextlib
import json
import uuid
from collections.abc import Callable
from typing import Protocol

import httpx
import pytest

from tournament_eval.llm import (
    AnthropicLLMClient,
    AnthropicModelConfig,
    BedrockLLMClient,
    BedrockModelConfig,
    HTTPLLMClient,
    HTTPModelConfig,
    LLMClient,
    OllamaLLMClient,
    OllamaModelConfig,
    OpenAICompatibleLLMClient,
    OpenAICompatibleModelConfig,
    StructuredResponse,
)
from tournament_eval.models import GenerationResult, GenerationTask, RankingTask


class FakeHTTPLLMClient(HTTPLLMClient):
    """A minimal concrete :class:`HTTPLLMClient` for exercising shared machinery.

    The hooks are deliberately trivial: a fixed endpoint, plain JSON headers, a
    payload that just carries the chat messages (plus ``schema`` when present),
    and text read from a ``"text"`` field in the response body.  This lets the
    generic request/template/lifecycle behaviour be tested without leaning on any
    real provider's wire format.
    """

    @property
    def _endpoint_url(self) -> str:
        return "http://test.local/v1/chat"

    @property
    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _build_payload(
        self,
        prompt: str,
        *,
        schema: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {"messages": self._messages(prompt)}
        if schema is not None:
            payload["schema"] = schema
        return payload

    def _extract_text(self, body: dict[str, object]) -> str:
        return str(body["text"])


@pytest.fixture
def make_http_client() -> Callable[..., FakeHTTPLLMClient]:
    """Return a factory that builds :class:`FakeHTTPLLMClient` instances.

    Pass an httpx handler to wire up a ``MockTransport``; any other keyword
    becomes an :class:`HTTPModelConfig` field; ``semaphore`` is forwarded to the
    client.
    """

    def _factory(
        handler: Callable[[httpx.Request], httpx.Response] | None = None,
        *,
        semaphore: object | None = None,
        **config_kwargs: object,
    ) -> FakeHTTPLLMClient:
        config_kwargs.setdefault("model_name", "fake")
        client = FakeHTTPLLMClient(
            HTTPModelConfig(**config_kwargs),  # type: ignore[arg-type]
            semaphore=semaphore,  # type: ignore[arg-type]
        )
        if handler is not None:
            client._transport = httpx.MockTransport(handler)
        return client

    return _factory


@pytest.fixture
def make_handler() -> Callable[
    ..., tuple[Callable[[httpx.Request], httpx.Response], dict[str, object]]
]:
    """Return a factory for ``MockTransport`` handlers used with the fake client.

    Each positional argument is one response, returned on successive calls (the
    last repeats once exhausted):

    * ``str``            -> ``200`` with body ``{"text": <str>}``
    * ``(status, str)``  -> ``<status>`` with body ``{"text": <str>}``

    Returns ``(handler, record)`` where ``record`` tracks ``"calls"`` and the
    JSON body of the most recent request under ``"captured"``.
    """

    def _factory(
        *responses: str | tuple[int, str],
    ) -> tuple[Callable[[httpx.Request], httpx.Response], dict[str, object]]:
        specs: list[tuple[int, dict[str, str]]] = []
        for response in responses or ("ok",):
            if isinstance(response, tuple):
                status, text = response
            else:
                status, text = 200, response
            specs.append((status, {"text": text}))

        record: dict[str, object] = {"calls": 0, "captured": {}}

        def handler(request: httpx.Request) -> httpx.Response:
            record["calls"] = int(record["calls"]) + 1  # type: ignore[call-overload]
            with contextlib.suppress(json.JSONDecodeError):
                record["captured"] = json.loads(request.content)
            status, body = specs[min(int(record["calls"]) - 1, len(specs) - 1)]  # type: ignore[call-overload]
            return httpx.Response(status, json=body)

        return handler, record

    return _factory


@pytest.fixture
def tracking_handler() -> tuple[
    Callable[[httpx.Request], httpx.Response], dict[str, int]
]:
    """An async handler that records peak in-flight concurrency.

    Returns ``(handler, state)`` where ``state`` tracks ``"inflight"`` and
    ``"peak"``.  Share one handler across clients to measure a global cap.
    """
    state = {"inflight": 0, "peak": 0}

    async def handler(_request: httpx.Request) -> httpx.Response:
        state["inflight"] += 1
        state["peak"] = max(state["peak"], state["inflight"])
        await asyncio.sleep(0.01)
        state["inflight"] -= 1
        return httpx.Response(200, json={"text": "ok"})

    return handler, state  # type: ignore[return-value]


@pytest.fixture
def make_openai_client() -> Callable[..., OpenAICompatibleLLMClient]:
    """Return a factory that builds :class:`OpenAICompatibleLLMClient` instances."""

    def _factory(**config_kwargs: object) -> OpenAICompatibleLLMClient:
        config_kwargs.setdefault("model_name", "gpt-4o")
        return OpenAICompatibleLLMClient(OpenAICompatibleModelConfig(**config_kwargs))  # type: ignore[arg-type]

    return _factory


@pytest.fixture
def make_ollama_client() -> Callable[..., OllamaLLMClient]:
    """Return a factory that builds :class:`OllamaLLMClient` instances."""

    def _factory(**config_kwargs: object) -> OllamaLLMClient:
        config_kwargs.setdefault("model_name", "llama3")
        return OllamaLLMClient(OllamaModelConfig(**config_kwargs))  # type: ignore[arg-type]

    return _factory


@pytest.fixture
def make_anthropic_client() -> Callable[..., AnthropicLLMClient]:
    """Return a factory that builds :class:`AnthropicLLMClient` instances."""

    def _factory(**config_kwargs: object) -> AnthropicLLMClient:
        config_kwargs.setdefault("model_name", "claude-opus-4-8")
        return AnthropicLLMClient(AnthropicModelConfig(**config_kwargs))  # type: ignore[arg-type]

    return _factory


@pytest.fixture
def make_bedrock_client() -> Callable[..., BedrockLLMClient]:
    """Return a factory that builds :class:`BedrockLLMClient` instances."""

    def _factory(**config_kwargs: object) -> BedrockLLMClient:
        config_kwargs.setdefault("model_name", "us.anthropic.claude-sonnet-4-6")
        return BedrockLLMClient(BedrockModelConfig(**config_kwargs))  # type: ignore[arg-type]

    return _factory


class MockLLMClient(LLMClient):
    """A fake LLM client for testing that returns pre-configured responses."""

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

    async def generate(self, prompt: str) -> str:
        if prompt in self._fail_on:
            raise RuntimeError(f"Mock generate failure: {prompt}")
        if prompt not in self._generate:
            raise RuntimeError(f"No mock response for prompt: {prompt!r}")
        return self._generate[prompt]

    async def generate_structured(
        self,
        prompt: str,
        _schema: dict[str, object],
    ) -> StructuredResponse:
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


class _MakeTask(Protocol):
    def __call__(self, prompt: str = "test prompt") -> GenerationTask: ...


class _MakeGeneration(Protocol):
    def __call__(
        self,
        *,
        author: str = "test-model",
        task_id: uuid.UUID | None = None,
        output: str = "some output",
    ) -> GenerationResult: ...


class _MakeRankingTask(Protocol):
    def __call__(
        self,
        *,
        ranking_prompt: str = "Rank.",
        generations: dict[str, uuid.UUID] | None = None,
    ) -> RankingTask: ...


@pytest.fixture
def make_task() -> _MakeTask:
    """Return a factory that creates GenerationTasks."""

    def _factory(prompt: str = "test prompt") -> GenerationTask:
        return GenerationTask(id=uuid.uuid4(), generation_prompt=prompt)

    return _factory


@pytest.fixture
def make_generation() -> _MakeGeneration:
    """Return a factory that creates GenerationResults."""

    def _factory(
        *,
        author: str = "test-model",
        task_id: uuid.UUID | None = None,
        output: str = "some output",
    ) -> GenerationResult:
        return GenerationResult(
            id=uuid.uuid4(),
            task_id=task_id or uuid.uuid4(),
            generation_prompt="test prompt",
            raw_response="raw",
            output=output,
            author=author,
        )

    return _factory


@pytest.fixture
def make_ranking_task() -> _MakeRankingTask:
    """Return a factory that creates RankingTasks."""

    def _factory(
        *,
        ranking_prompt: str = "Rank.",
        generations: dict[str, uuid.UUID] | None = None,
    ) -> RankingTask:
        return RankingTask(
            id=uuid.uuid4(),
            ranking_prompt=ranking_prompt,
            generations=generations or {},
        )

    return _factory
