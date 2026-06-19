"""Tests for the pure data models — chiefly ``from_json`` (str -> UUID) rebuilding."""

import uuid
from collections.abc import Callable

import pytest

from tournament_eval.models import (
    GenerationFailure,
    GenerationFailureDict,
    GenerationResult,
    GenerationResultDict,
    GenerationTask,
    GenerationTaskDict,
    RankingFailure,
    RankingFailureDict,
    RankingResult,
    RankingResultDict,
    RankingTask,
    RankingTaskDict,
)


@pytest.mark.parametrize(
    ("method", "data", "expected"),
    [
        (
            GenerationTask.from_json,
            {"id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "generation_prompt": "test prompt"},
            GenerationTask(id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"), generation_prompt="test prompt"),
        ),
        (
            GenerationResult.from_json,
            {
                "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "generation_task_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "generation_prompt": "test prompt",
                "output": "some output",
                "reasoning": "some reasoning",
                "author": "test-model",
                "metadata": {"cost": 0.1, "latency": 0.2},
            },
            GenerationResult(
                id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                generation_task_id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                generation_prompt="test prompt",
                output="some output",
                reasoning="some reasoning",
                author="test-model",
                metadata={"cost": 0.1, "latency": 0.2},
            ),
        ),
        (
            GenerationFailure.from_json,
            {
                "generation_task_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "author": "test-model",
                "error_type": "RuntimeError",
                "message": "boom!",
            },
            GenerationFailure(
                generation_task_id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                author="test-model",
                error_type="RuntimeError",
                message="boom!",
            ),
        ),
        (
            RankingTask.from_json,
            {
                "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "generation_task_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "ranking_prompt": "Rank.",
                "generations": {
                    "A": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "B": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                },
            },
            RankingTask(
                id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                generation_task_id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                ranking_prompt="Rank.",
                generations={
                    "A": uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                    "B": uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                },
            ),
        ),
        (
            RankingResult.from_json,
            {
                "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "ranking_task_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "generation_task_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "ranking_prompt": "Rank.",
                "author": "test-model",
                "raw_model_ranking": ["B", "A"],
                "ranking": ["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"],
                "ranking_reasoning": "some reasoning",
                "reasoning": "some thinking trace",
                "raw_response": "some raw response",
                "metadata": {"cost": 0.1, "latency": 0.2},
            },
            RankingResult(
                id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                ranking_task_id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                generation_task_id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                ranking_prompt="Rank.",
                author="test-model",
                raw_model_ranking=["B", "A"],
                ranking=[
                    uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                    uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                ],
                ranking_reasoning="some reasoning",
                reasoning="some thinking trace",
                raw_response="some raw response",
                metadata={"cost": 0.1, "latency": 0.2},
            ),
        ),
        (
            RankingFailure.from_json,
            {
                "ranking_task_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "generation_task_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "author": "test-model",
                "error_type": "RuntimeError",
                "message": "boom!",
                "ranking_prompt": "the rendered prompt",
            },
            RankingFailure(
                ranking_task_id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                generation_task_id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                author="test-model",
                error_type="RuntimeError",
                message="boom!",
                ranking_prompt="the rendered prompt",
            ),
        ),
    ],
)
def test_from_json_rebuilds[T](method: Callable[..., T], data: dict[str, object], expected: T) -> None:
    assert method(data) == expected


@pytest.mark.parametrize(
    "method",
    [
        GenerationTask.from_json,
        GenerationResult.from_json,
        RankingTask.from_json,
        RankingResult.from_json,
    ],
)
def test_from_json_fails_on_invalid_uuid(method: Callable[..., object]) -> None:
    with pytest.raises(ValueError, match="UUID"):
        method({"id": "not a uuid"})


def test_generation_failure_from_json_fails_on_invalid_uuid() -> None:
    with pytest.raises(ValueError, match="UUID"):
        GenerationFailure.from_json({"generation_task_id": "not a uuid"})  # type: ignore[typeddict-item]


def test_ranking_failure_from_json_fails_on_invalid_uuid() -> None:
    with pytest.raises(ValueError, match="UUID"):
        RankingFailure.from_json({"ranking_task_id": "not a uuid"})  # type: ignore[typeddict-item]


@pytest.mark.parametrize(
    "missing_arg",
    [
        "id",
        "generation_prompt",
    ],
)
def test_generation_task_from_json_fails_on_missing_arg(missing_arg: str) -> None:
    data: GenerationTaskDict = {"id": str(uuid.uuid4()), "generation_prompt": "test prompt"}
    data.pop(missing_arg)  # type: ignore[misc]
    with pytest.raises(KeyError, match=missing_arg):
        GenerationTask.from_json(data)


@pytest.mark.parametrize(
    "missing_arg",
    [
        "generation_task_id",
        "author",
        "error_type",
        "message",
    ],
)
def test_generation_failure_from_json_fails_on_missing_arg(missing_arg: str) -> None:
    data: GenerationFailureDict = {
        "generation_task_id": str(uuid.uuid4()),
        "author": "test-model",
        "error_type": "RuntimeError",
        "message": "boom!",
    }
    data.pop(missing_arg)  # type: ignore[misc]
    with pytest.raises(KeyError, match=missing_arg):
        GenerationFailure.from_json(data)


@pytest.mark.parametrize(
    "missing_arg",
    [
        "id",
        "generation_task_id",
        "generation_prompt",
        "output",
        "reasoning",
        "author",
        "metadata",
    ],
)
def test_generation_result_from_json_fails_on_missing_arg(missing_arg: str) -> None:
    data: GenerationResultDict = {
        "id": str(uuid.uuid4()),
        "generation_task_id": str(uuid.uuid4()),
        "generation_prompt": "test prompt",
        "output": "some output",
        "reasoning": "some reasoning",
        "author": "test-model",
        "metadata": {"cost": 0.1, "latency": 0.2},
    }
    data.pop(missing_arg)  # type: ignore[misc]
    with pytest.raises(KeyError, match=missing_arg):
        GenerationResult.from_json(data)


@pytest.mark.parametrize(
    "missing_arg",
    [
        "id",
        "generation_task_id",
        "ranking_prompt",
        "generations",
    ],
)
def test_ranking_task_from_json_fails_on_missing_arg(missing_arg: str) -> None:
    data: RankingTaskDict = {
        "id": str(uuid.uuid4()),
        "generation_task_id": str(uuid.uuid4()),
        "ranking_prompt": "Rank.",
        "generations": {"A": str(uuid.uuid4()), "B": str(uuid.uuid4())},
    }
    data.pop(missing_arg)  # type: ignore[misc]
    with pytest.raises(KeyError, match=missing_arg):
        RankingTask.from_json(data)


@pytest.mark.parametrize(
    "missing_arg",
    [
        "id",
        "ranking_task_id",
        "generation_task_id",
        "ranking_prompt",
        "author",
        "raw_model_ranking",
        "ranking",
        "ranking_reasoning",
        "reasoning",
        "raw_response",
        "metadata",
    ],
)
def test_ranking_result_from_json_fails_on_missing_arg(missing_arg: str) -> None:
    data: RankingResultDict = {
        "id": str(uuid.uuid4()),
        "ranking_task_id": str(uuid.uuid4()),
        "generation_task_id": str(uuid.uuid4()),
        "ranking_prompt": "Rank.",
        "author": "test-model",
        "raw_model_ranking": ["B", "A"],
        "ranking": ["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"],
        "ranking_reasoning": "some reasoning",
        "reasoning": "some thinking trace",
        "raw_response": "some raw response",
        "metadata": {"cost": 0.1, "latency": 0.2},
    }
    data.pop(missing_arg)  # type: ignore[misc]
    with pytest.raises(KeyError, match=missing_arg):
        RankingResult.from_json(data)


@pytest.mark.parametrize(
    "missing_arg",
    [
        "ranking_task_id",
        "generation_task_id",
        "author",
        "error_type",
        "message",
        "ranking_prompt",
    ],
)
def test_ranking_failure_from_json_fails_on_missing_arg(missing_arg: str) -> None:
    data: RankingFailureDict = {
        "ranking_task_id": str(uuid.uuid4()),
        "generation_task_id": str(uuid.uuid4()),
        "author": "test-model",
        "error_type": "RuntimeError",
        "message": "boom!",
        "ranking_prompt": "the rendered prompt",
    }
    data.pop(missing_arg)  # type: ignore[misc]
    with pytest.raises(KeyError, match=missing_arg):
        RankingFailure.from_json(data)
