"""Tests for the pure data models — chiefly ``from_json`` (str -> UUID) rebuilding."""

import uuid

from tournament_eval.models import (
    GenerationFailure,
    GenerationResult,
    GenerationTask,
    RankingFailure,
    RankingResult,
    RankingTask,
)


def test_generation_task_from_json() -> None:
    task_id = uuid.uuid4()
    task = GenerationTask.from_json({"id": str(task_id), "generation_prompt": "translate"})
    assert task.id == task_id
    assert task.generation_prompt == "translate"


def test_generation_failure_from_json() -> None:
    task_id = uuid.uuid4()
    failure = GenerationFailure.from_json(
        {"task_id": str(task_id), "author": "m", "error_type": "RuntimeError", "message": "boom"}
    )
    assert failure.task_id == task_id
    assert (failure.author, failure.error_type, failure.message) == ("m", "RuntimeError", "boom")


def test_generation_result_from_json() -> None:
    rid, task_id = uuid.uuid4(), uuid.uuid4()
    result = GenerationResult.from_json(
        {
            "id": str(rid),
            "task_id": str(task_id),
            "generation_prompt": "p",
            "output": "o",
            "reasoning": "thought about it",
            "author": "gpt",
            "metadata": {"latency": 1.5},
        }
    )
    assert result.id == rid
    assert result.task_id == task_id
    assert result.output == "o"
    assert result.reasoning == "thought about it"
    assert result.metadata == {"latency": 1.5}


def test_ranking_task_from_json() -> None:
    rt_id, gen_a, gen_b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    gen_task_id = uuid.uuid4()
    task = RankingTask.from_json(
        {
            "id": str(rt_id),
            "generation_task_id": str(gen_task_id),
            "ranking_prompt": "rank",
            "generations": {"A": str(gen_a), "B": str(gen_b)},
        }
    )
    assert task.id == rt_id
    assert task.generation_task_id == gen_task_id
    assert task.generations == {"A": gen_a, "B": gen_b}


def test_ranking_failure_from_json() -> None:
    rt_id = uuid.uuid4()
    failure = RankingFailure.from_json(
        {"ranking_task_id": str(rt_id), "author": "judge", "error_type": "ValueError", "message": "bad"}
    )
    assert failure.ranking_task_id == rt_id
    assert failure.author == "judge"


def test_ranking_result_from_json() -> None:
    rid, rt_id, g1, g2 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    result = RankingResult.from_json(
        {
            "id": str(rid),
            "ranking_task_id": str(rt_id),
            "ranking_prompt": "rp",
            "author": "judge",
            "raw_model_ranking": ["B", "A"],
            "ranking": [str(g1), str(g2)],
            "reasoning": None,
            "raw_response": "{}",
            "metadata": {},
        }
    )
    assert result.id == rid
    assert result.ranking == [g1, g2]
    assert result.raw_model_ranking == ["B", "A"]
    assert result.reasoning is None
