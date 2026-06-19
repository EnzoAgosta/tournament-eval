"""Shared test scaffolding.

The client/wire-format tests are gone — pydantic-ai owns the LLM layer now, and we
test our *orchestration* logic against pydantic-ai's offline test models:

* :class:`pydantic_ai.models.test.TestModel` — auto-generates valid output from the
  agent's ``output_type`` (text via ``custom_output_text``, structured via
  ``custom_output_args``), no network.  We wrap it in tiny builders so each test
  gets an agent with a controllable author (the model name) and output.

Plus the data factories used by the model/persistence/orchestration tests.
"""

import uuid
from typing import Protocol

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from tournament_eval.models import (
    GenerationFailure,
    GenerationResult,
    GenerationTask,
    RankingFailure,
    RankingResult,
    RankingTask,
)


def generation_agent(
    *,
    text: str = "ok",
    author: str = "test-model",
) -> Agent[None, str]:
    """A generation agent (``output_type=str``) backed by a :class:`TestModel`.

    ``author`` becomes the model name (and thus the result ``author``); ``text`` is
    what it returns.  No network.
    """
    return Agent(
        TestModel(custom_output_text=text, model_name=author),
        output_type=str,
        name=None,  # leave author resolution to fall back to the model name
    )


def _raising_fn(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    """A :class:`FunctionModel` body that always raises — backs the failure-path agents."""
    raise RuntimeError("boom")


def raising_generation_agent(*, author: str = "test-model") -> Agent[None, str]:
    """A generation agent whose ``run`` always raises ``RuntimeError("boom")``."""
    return Agent(FunctionModel(_raising_fn, model_name=author), output_type=str, name=None)


def raising_ranking_agent(
    *, author: str = "test-model", response_model: type[BaseModel] | None = None
) -> Agent[None, BaseModel]:
    """A ranking agent whose ``run`` always raises ``RuntimeError("boom")``."""
    from tournament_eval.ranking import RankingResponse

    return Agent(
        FunctionModel(_raising_fn, model_name=author),
        output_type=response_model or RankingResponse,
        name=None,
    )


def ranking_agent(
    *,
    output: BaseModel | None = None,
    output_args: dict[str, object] | None = None,
    author: str = "test-model",
    response_model: type[BaseModel] | None = None,
) -> Agent[None, BaseModel]:
    """A ranking agent backed by a :class:`TestModel` returning structured output.

    Pass either a ready ``output`` model instance (its dict is sent as the tool
    args) or raw ``output_args``.  ``response_model`` defaults to
    :class:`~tournament_eval.ranking.RankingResponse`.
    """
    from tournament_eval.ranking import RankingResponse

    model = response_model or RankingResponse
    args: dict[str, object] | BaseModel
    if output is not None:
        args = output.model_dump()
    elif output_args is not None:
        args = output_args
    else:
        args = {"ranking": ["A"], "reasoning": "ok"}
    return Agent(
        TestModel(custom_output_args=args, model_name=author),
        output_type=model,
        name=None,
    )


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
