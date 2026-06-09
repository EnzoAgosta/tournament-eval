"""Shared pytest fixtures."""

import uuid
from typing import Protocol

import pytest

from tournament_eval.models import GenerationResult, GenerationTask, RankingTask


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
