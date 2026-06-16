import contextlib
import uuid
from pathlib import Path
from typing import Any

import orjson
import pytest

from tests.conftest import (
    GenerationResultFactory,
    GenerationTaskFactory,
    MockLLMClient,
    RankingTaskFactory,
)
from tournament_eval.llm.base import GenerationResponse, StructuredResponse
from tournament_eval.models import (
    GenerationFailure,
    GenerationResult,
    RankingFailure,
    RankingResult,
    RankingTask,
)
from tournament_eval.orchestration import (
    _open_clients,
    build_ranking_task,
    build_ranking_tasks,
    generate_all,
    generate_one,
    rank_all,
    rank_one,
)


async def test_generate_one_returns_generation_result(make_generation_task: GenerationTaskFactory) -> None:
    client = MockLLMClient()
    task = make_generation_task()

    result = await generate_one(task, client)

    assert isinstance(result, GenerationResult)
    assert isinstance(result.id, uuid.UUID)
    assert result.generation_task_id == task.id
    assert result.generation_prompt == task.generation_prompt
    assert result.output == client.generation_response.text
    assert result.reasoning == client.generation_response.reasoning
    assert result.author == client.name


async def test_generate_one_returns_generation_failure_on_error(make_generation_task: GenerationTaskFactory) -> None:
    client = MockLLMClient()
    task = make_generation_task()

    async def raise_error(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom")

    client.generate = raise_error  # type: ignore[method-assign]

    result = await generate_one(task, client)

    assert isinstance(result, GenerationFailure)
    assert result.generation_task_id == task.id
    assert result.author == client.name
    assert result.error_type == "RuntimeError"
    assert result.message == "boom"


async def test_generate_one_saves_to_results_path(
    make_generation_task: GenerationTaskFactory,
    tmp_path: Path,
) -> None:
    client = MockLLMClient()
    task = make_generation_task()
    results_path = tmp_path / "results.jsonl"

    result = await generate_one(task, client, results_path=results_path)

    assert isinstance(result, GenerationResult)
    assert isinstance(result.id, uuid.UUID)
    assert result.generation_task_id == task.id
    assert result.generation_prompt == task.generation_prompt
    assert result.output == client.generation_response.text
    assert result.reasoning == client.generation_response.reasoning
    assert result.author == client.name

    with open(results_path, "rb") as f:
        assert f.read() == orjson.dumps(result) + b"\n"


async def test_generate_one_saves_to_failures_path(
    make_generation_task: GenerationTaskFactory,
    tmp_path: Path,
) -> None:
    client = MockLLMClient()
    task = make_generation_task()
    failures_path = tmp_path / "failures.jsonl"

    async def raise_error(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom")

    client.generate = raise_error  # type: ignore[method-assign]

    result = await generate_one(task, client, failures_path=failures_path)

    assert isinstance(result, GenerationFailure)
    assert result.generation_task_id == task.id
    assert result.author == client.name
    assert result.error_type == "RuntimeError"
    assert result.message == "boom"

    with open(failures_path, "rb") as f:
        assert f.read() == orjson.dumps(result) + b"\n"


async def test_generate_all_returns_list_of_generation_results(
    make_generation_task: GenerationTaskFactory,
) -> None:
    client1 = MockLLMClient()
    client2 = MockLLMClient()
    task = make_generation_task()
    client1.generation_response = GenerationResponse(text="one", reasoning="because")
    client2.generation_response = GenerationResponse(text="two", reasoning="because")
    client1.model = "client1"
    client2.model = "client2"

    results, failures = await generate_all([task], clients=[client1, client2])

    assert len(failures) == 0
    assert len(results) == 2
    result1, result2 = results
    assert isinstance(result1, GenerationResult)
    assert isinstance(result2, GenerationResult)
    assert result1.generation_task_id == result2.generation_task_id == task.id
    assert result1.generation_prompt == result2.generation_prompt == task.generation_prompt
    assert result1.output == "one"
    assert result2.output == "two"
    assert result1.reasoning == result2.reasoning == "because"
    assert result1.author == client1.name
    assert result2.author == client2.name


async def test_generate_all_saves_to_results_path(
    make_generation_task: GenerationTaskFactory,
    tmp_path: Path,
) -> None:
    client1 = MockLLMClient()
    client2 = MockLLMClient()
    task = make_generation_task()
    results_path = tmp_path / "results.jsonl"
    client1.generation_response = GenerationResponse(text="one", reasoning="because")
    client2.generation_response = GenerationResponse(text="two", reasoning="because")
    client1.model = "client1"
    client2.model = "client2"

    results, failures = await generate_all([task], clients=[client1, client2], results_path=results_path)

    assert len(failures) == 0
    assert len(results) == 2
    result1, result2 = results
    assert isinstance(result1, GenerationResult)
    assert isinstance(result2, GenerationResult)
    assert result1.generation_task_id == result2.generation_task_id == task.id
    assert result1.generation_prompt == result2.generation_prompt == task.generation_prompt
    assert result1.output == "one"
    assert result2.output == "two"
    assert result1.reasoning == result2.reasoning == "because"
    assert result1.author == client1.name
    assert result2.author == client2.name

    with open(results_path, "rb") as f:
        assert f.read() == orjson.dumps(result1) + b"\n" + orjson.dumps(result2) + b"\n"


async def test_generate_all_saves_to_failures_path(
    make_generation_task: GenerationTaskFactory,
    tmp_path: Path,
) -> None:
    failure_client = MockLLMClient()
    result_client = MockLLMClient()
    task = make_generation_task()
    failures_path = tmp_path / "failures.jsonl"
    failure_client.generation_response = GenerationResponse(text="one", reasoning="because")
    result_client.generation_response = GenerationResponse(text="two", reasoning="because")
    failure_client.model = "client1"
    result_client.model = "client2"

    def raise_error(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom")

    failure_client.generate = raise_error  # type: ignore[method-assign]

    results, failures = await generate_all([task], clients=[failure_client, result_client], failures_path=failures_path)

    assert len(results) == 1
    assert len(failures) == 1
    result, failure = results[0], failures[0]
    assert isinstance(result, GenerationResult)
    assert isinstance(failure, GenerationFailure)
    assert result.generation_task_id == task.id
    assert result.generation_prompt == task.generation_prompt
    assert result.output == "two"
    assert result.reasoning == "because"
    assert result.author == result_client.name
    assert failure.generation_task_id == task.id
    assert failure.author == failure_client.name
    assert failure.error_type == "RuntimeError"
    assert failure.message == "boom"

    with open(failures_path, "rb") as f:
        assert f.read() == orjson.dumps(failure) + b"\n"


async def test_generate_all_saves_to_results_and_failures_path(
    make_generation_task: GenerationTaskFactory,
    tmp_path: Path,
) -> None:
    failure_client = MockLLMClient()
    result_client = MockLLMClient()
    task = make_generation_task()
    results_path = tmp_path / "results.jsonl"
    failures_path = tmp_path / "failures.jsonl"
    failure_client.generation_response = GenerationResponse(text="one", reasoning="because")
    result_client.generation_response = GenerationResponse(text="two", reasoning="because")
    failure_client.model = "client1"
    result_client.model = "client2"

    def raise_error(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom")

    failure_client.generate = raise_error  # type: ignore[method-assign]

    results, failures = await generate_all(
        [task], clients=[failure_client, result_client], results_path=results_path, failures_path=failures_path
    )

    assert len(results) == 1
    assert len(failures) == 1
    result, failure = results[0], failures[0]
    assert isinstance(result, GenerationResult)
    assert isinstance(failure, GenerationFailure)
    assert result.generation_task_id == task.id
    assert result.generation_prompt == task.generation_prompt
    assert result.output == "two"
    assert result.reasoning == "because"
    assert result.author == result_client.name
    assert failure.generation_task_id == task.id
    assert failure.author == failure_client.name
    assert failure.error_type == "RuntimeError"
    assert failure.message == "boom"

    with open(results_path, "rb") as f:
        assert f.read() == orjson.dumps(result) + b"\n"
    with open(failures_path, "rb") as f:
        assert f.read() == orjson.dumps(failure) + b"\n"


async def test_generate_all_reads_generation_results_from_results_path(
    make_generation_task: GenerationTaskFactory,
    tmp_path: Path,
) -> None:
    client1 = MockLLMClient()
    client2 = MockLLMClient()
    task = make_generation_task()
    results_path = tmp_path / "results.jsonl"
    client1.generation_response = GenerationResponse(text="one", reasoning="because")
    client2.generation_response = GenerationResponse(text="two", reasoning="because")
    client1.model = "client1"
    client2.model = "client2"

    await generate_one(task, client1, results_path=results_path)

    client1.generation_response = GenerationResponse(text="three", reasoning="because")

    results, failures = await generate_all([task], clients=[client1, client2], results_path=results_path)

    assert len(failures) == 0
    assert len(results) == 2
    result1, result2 = results
    assert isinstance(result1, GenerationResult)
    assert isinstance(result2, GenerationResult)
    assert result1.generation_task_id == result2.generation_task_id == task.id
    assert result1.generation_prompt == result2.generation_prompt == task.generation_prompt
    assert result1.output == "one"
    assert result2.output == "two"
    assert result1.reasoning == result2.reasoning == "because"
    assert result1.author == client1.name
    assert result2.author == client2.name


async def test_generate_all_opens_and_closes_clients(make_generation_task: GenerationTaskFactory) -> None:
    client1 = MockLLMClient()
    client1.raise_on_enter = True

    with pytest.raises(RuntimeError, match="Enter boom"):
        await generate_all([make_generation_task()], clients=[client1])

    client1.raise_on_exit = True
    client1.raise_on_enter = False

    with pytest.raises(RuntimeError, match="Exit boom"):
        await generate_all([make_generation_task()], clients=[client1])


async def test_open_clients_only_opens_async_context_manager() -> None:
    client = object()
    async with contextlib.AsyncExitStack() as stack:
        opened = await _open_clients([client], stack)  # type: ignore[list-item]
    assert opened == [client]
    assert opened[0] is client


async def test_generate_all_returns_all_generation_results(
    make_generation_task: GenerationTaskFactory,
    tmp_path: Path,
) -> None:
    client1 = MockLLMClient()
    client2 = MockLLMClient()
    client3 = MockLLMClient()
    task = make_generation_task()
    results_path = tmp_path / "results.jsonl"
    client1.generation_response = GenerationResponse(text="one", reasoning="because one")
    client2.generation_response = GenerationResponse(text="two", reasoning="because two")
    client3.generation_response = GenerationResponse(text="three", reasoning="because three")
    client1.model = "client1"
    client2.model = "client2"
    client3.model = "client3"

    results, _ = await generate_all([task], clients=[client1, client2], results_path=results_path)
    assert len(results) == 2

    client1.generation_response = GenerationResponse(text="four", reasoning="because four")
    client2.generation_response = GenerationResponse(text="five", reasoning="because five")

    results, _ = await generate_all([task], clients=[client1, client2, client3], results_path=results_path)
    assert len(results) == 3

    result1, result2, result3 = results
    assert all(isinstance(result, GenerationResult) for result in results)
    assert all(result.generation_task_id == task.id for result in results)
    assert all(result.generation_prompt == task.generation_prompt for result in results)
    assert result1.output == "one"
    assert result2.output == "two"
    assert result3.output == "three"
    assert result1.author == client1.name
    assert result2.author == client2.name
    assert result3.author == client3.name
    assert result1.reasoning == "because one"
    assert result2.reasoning == "because two"
    assert result3.reasoning == "because three"


async def test_generate_all_ignores_records_outside_current_universe(
    tmp_path: Path, make_generation_task: GenerationTaskFactory
) -> None:
    task = make_generation_task(prompt="p")
    path = tmp_path / "generations.jsonl"
    client = MockLLMClient()
    client.generation_response = GenerationResponse(text="old", reasoning="because out")
    await generate_one(task, client, results_path=path)
    client.generation_response = GenerationResponse(text="new", reasoning="because out")
    results, _ = await generate_all([task], [client], results_path=path)
    assert {r.author for r in results} == {"test-model"}


def test_build_ranking_task(make_generation_result: GenerationResultFactory) -> None:
    task_id = uuid.uuid4()
    results = [make_generation_result(generation_task_id=task_id) for _ in range(3)]
    task = build_ranking_task(results=results, ranking_prompt="test")
    assert isinstance(task, RankingTask)
    assert isinstance(task.id, uuid.UUID)
    assert task.generation_task_id == task_id
    assert task.ranking_prompt == "test"
    assert set(task.generations.values()) == {result.id for result in results}
    assert set(task.generations.keys()) == {"A", "B", "C"}


def test_build_ranking_tasks(make_generation_result: GenerationResultFactory) -> None:
    results = [make_generation_result() for _ in range(3)]
    with pytest.raises(ValueError, match="must all be for the same task"):
        build_ranking_task(results=results, ranking_prompt="test")


def test_build_ranking_task_shuffles_deterministically(make_generation_result: GenerationResultFactory) -> None:
    id = uuid.uuid4()
    results = [make_generation_result(generation_task_id=id) for _ in range(10)]
    task1 = build_ranking_task(results=results, ranking_prompt="test", random_seed=42)
    task2 = build_ranking_task(results=results, ranking_prompt="test", random_seed=42)
    assert task1.generations == task2.generations


def test_build_ranking_task_saves_to_tasks_path(
    make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    id = uuid.uuid4()
    results = [make_generation_result(generation_task_id=id) for _ in range(3)]
    tasks_path = tmp_path / "tasks.jsonl"
    task = build_ranking_task(results=results, ranking_prompt="test", tasks_path=tasks_path)
    assert isinstance(task, RankingTask)
    assert isinstance(task.id, uuid.UUID)
    assert task.generation_task_id == id
    assert task.ranking_prompt == "test"
    assert set(task.generations.values()) == {result.id for result in results}
    assert set(task.generations.keys()) == {"A", "B", "C"}

    with open(tasks_path, "rb") as f:
        assert f.read() == orjson.dumps(task) + b"\n"


def test_build_ranking_task_fails_on_no_results() -> None:
    with pytest.raises(ValueError, match="build_ranking_task needs at least one GenerationResult"):
        build_ranking_task(results=[], ranking_prompt="test")


def test_build_ranking_tasks_return_ranking_tasks(
    make_generation_task: GenerationTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    tasks = [make_generation_task() for _ in range(3)]
    results = [make_generation_result(generation_task_id=task.id) for _ in range(3) for task in tasks]
    ranking_tasks = build_ranking_tasks(tasks=tasks, generation_results=results, ranking_prompt="Test")

    assert len(ranking_tasks) == 3
    assert all(isinstance(task, RankingTask) for task in ranking_tasks)
    assert all(task.ranking_prompt == "Test" for task in ranking_tasks)


def test_build_ranking_tasks_fails_resuses_existing(
    make_generation_task: GenerationTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    tasks = make_generation_task()
    old_id, new_id = uuid.uuid4(), uuid.uuid4()
    old_result = make_generation_result(generation_task_id=tasks.id, id=old_id)
    path = tmp_path / "tasks.jsonl"
    ranking_task = build_ranking_task(results=[old_result], ranking_prompt="Test", tasks_path=path)
    new_result = make_generation_result(generation_task_id=tasks.id, id=new_id)
    ranking_tasks = build_ranking_tasks(
        tasks=[tasks], generation_results=[new_result], ranking_prompt="Test", tasks_path=path
    )
    assert len(ranking_tasks) == 1
    new_ranking_task = ranking_tasks[0]
    assert new_ranking_task.id == ranking_task.id
    assert new_ranking_task.generation_task_id == tasks.id
    assert new_ranking_task.ranking_prompt == "Test"
    assert new_ranking_task.generations == ranking_task.generations
    assert new_ranking_task.generations["A"] == old_result.id


def test_build_ranking_tasks_fails_skips_tasks_without_results(make_generation_task: GenerationTaskFactory) -> None:
    tasks = make_generation_task()
    ranking_task = build_ranking_tasks(tasks=[tasks], generation_results=[], ranking_prompt="Test")
    assert len(ranking_task) == 0


async def test_rank_one_returns_ranking_result(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    client = MockLLMClient()
    gen_a, gen_b = make_generation_result(), make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen_a.id, "B": gen_b.id})
    lookup = {gen_a.id: gen_a, gen_b.id: gen_b}
    ranking = {"ranking": ["B", "A"], "reasoning": "b is better"}
    client.structured_response = StructuredResponse(data=ranking, raw=orjson.dumps(ranking).decode())

    result = await rank_one(ranking_task, client, lookup)

    assert isinstance(result, RankingResult)
    assert isinstance(result.id, uuid.UUID)
    assert result.ranking_task_id == ranking_task.id
    assert result.author == client.name
    assert result.raw_model_ranking == ["B", "A"]
    assert result.ranking == [gen_b.id, gen_a.id]
    assert result.reasoning == "b is better"
    assert result.raw_response == client.structured_response.raw
    assert ranking_task.ranking_prompt in result.ranking_prompt


async def test_rank_one_returns_ranking_failure_on_error(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    client = MockLLMClient()
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})

    async def raise_error(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom")

    client.generate_structured = raise_error  # type: ignore[method-assign]

    result = await rank_one(ranking_task, client, {gen.id: gen})

    assert isinstance(result, RankingFailure)
    assert result.ranking_task_id == ranking_task.id
    assert result.author == client.name
    assert result.error_type == "RuntimeError"
    assert result.message == "boom"


async def test_rank_one_returns_ranking_failure_on_malformed_ranking(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    client = MockLLMClient()
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    ranking = {"ranking": ["Z"]}  # alias not in the task
    client.structured_response = StructuredResponse(data=ranking, raw=orjson.dumps(ranking).decode())

    result = await rank_one(ranking_task, client, {gen.id: gen})

    assert isinstance(result, RankingFailure)
    assert result.ranking_task_id == ranking_task.id
    assert result.author == client.name
    assert result.error_type == "ValueError"
    assert "Z" in result.message


async def test_rank_one_saves_to_results_path(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    client = MockLLMClient()
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    ranking = {"ranking": ["A"], "reasoning": "ok"}
    client.structured_response = StructuredResponse(data=ranking, raw=orjson.dumps(ranking).decode())
    results_path = tmp_path / "rankings.jsonl"

    result = await rank_one(ranking_task, client, {gen.id: gen}, results_path=results_path)

    assert isinstance(result, RankingResult)
    with open(results_path, "rb") as f:
        assert f.read() == orjson.dumps(result) + b"\n"


async def test_rank_one_saves_to_failures_path(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    client = MockLLMClient()
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    failures_path = tmp_path / "ranking_failures.jsonl"

    async def raise_error(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom")

    client.generate_structured = raise_error  # type: ignore[method-assign]

    result = await rank_one(ranking_task, client, {gen.id: gen}, failures_path=failures_path)

    assert isinstance(result, RankingFailure)
    with open(failures_path, "rb") as f:
        assert f.read() == orjson.dumps(result) + b"\n"


async def test_rank_all_returns_list_of_ranking_results(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    client1, client2 = MockLLMClient(), MockLLMClient()
    client1.model, client2.model = "client1", "client2"
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    ranking1 = {"ranking": ["A"], "reasoning": "one"}
    ranking2 = {"ranking": ["A"], "reasoning": "two"}
    client1.structured_response = StructuredResponse(data=ranking1, raw=orjson.dumps(ranking1).decode())
    client2.structured_response = StructuredResponse(data=ranking2, raw=orjson.dumps(ranking2).decode())

    results, failures = await rank_all([ranking_task], [gen], clients=[client1, client2])

    assert len(failures) == 0
    assert len(results) == 2
    result1, result2 = results
    assert isinstance(result1, RankingResult)
    assert isinstance(result2, RankingResult)
    assert result1.ranking_task_id == result2.ranking_task_id == ranking_task.id
    assert result1.ranking == result2.ranking == [gen.id]
    assert result1.author == client1.name
    assert result2.author == client2.name
    assert result1.reasoning == "one"
    assert result2.reasoning == "two"


async def test_rank_all_saves_to_results_path(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    client1, client2 = MockLLMClient(), MockLLMClient()
    client1.model, client2.model = "client1", "client2"
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    ranking1 = {"ranking": ["A"], "reasoning": "one"}
    ranking2 = {"ranking": ["A"], "reasoning": "two"}
    client1.structured_response = StructuredResponse(data=ranking1, raw=orjson.dumps(ranking1).decode())
    client2.structured_response = StructuredResponse(data=ranking2, raw=orjson.dumps(ranking2).decode())
    results_path = tmp_path / "rankings.jsonl"

    results, failures = await rank_all([ranking_task], [gen], clients=[client1, client2], results_path=results_path)

    assert len(failures) == 0
    assert len(results) == 2
    result1, result2 = results
    with open(results_path, "rb") as f:
        assert f.read() == orjson.dumps(result1) + b"\n" + orjson.dumps(result2) + b"\n"


async def test_rank_all_saves_to_failures_path(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    failure_client, result_client = MockLLMClient(), MockLLMClient()
    failure_client.model, result_client.model = "client1", "client2"
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    ranking = {"ranking": ["A"], "reasoning": "ok"}
    result_client.structured_response = StructuredResponse(data=ranking, raw=orjson.dumps(ranking).decode())
    failures_path = tmp_path / "ranking_failures.jsonl"

    def raise_error(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom")

    failure_client.generate_structured = raise_error  # type: ignore[method-assign]

    results, failures = await rank_all(
        [ranking_task], [gen], clients=[failure_client, result_client], failures_path=failures_path
    )

    assert len(results) == 1
    assert len(failures) == 1
    result, failure = results[0], failures[0]
    assert isinstance(result, RankingResult)
    assert isinstance(failure, RankingFailure)
    assert result.author == result_client.name
    assert failure.ranking_task_id == ranking_task.id
    assert failure.author == failure_client.name
    assert failure.error_type == "RuntimeError"
    assert failure.message == "boom"

    with open(failures_path, "rb") as f:
        assert f.read() == orjson.dumps(failure) + b"\n"


async def test_rank_all_saves_to_results_and_failures_path(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    failure_client, result_client = MockLLMClient(), MockLLMClient()
    failure_client.model, result_client.model = "client1", "client2"
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    ranking = {"ranking": ["A"], "reasoning": "ok"}
    result_client.structured_response = StructuredResponse(data=ranking, raw=orjson.dumps(ranking).decode())
    results_path = tmp_path / "rankings.jsonl"
    failures_path = tmp_path / "ranking_failures.jsonl"

    def raise_error(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom")

    failure_client.generate_structured = raise_error  # type: ignore[method-assign]

    results, failures = await rank_all(
        [ranking_task],
        [gen],
        clients=[failure_client, result_client],
        results_path=results_path,
        failures_path=failures_path,
    )

    assert len(results) == 1
    assert len(failures) == 1
    result, failure = results[0], failures[0]
    with open(results_path, "rb") as f:
        assert f.read() == orjson.dumps(result) + b"\n"
    with open(failures_path, "rb") as f:
        assert f.read() == orjson.dumps(failure) + b"\n"


async def test_rank_all_reads_ranking_results_from_results_path(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    client1, client2 = MockLLMClient(), MockLLMClient()
    client1.model, client2.model = "client1", "client2"
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    results_path = tmp_path / "rankings.jsonl"
    ranking1 = {"ranking": ["A"], "reasoning": "one"}
    ranking2 = {"ranking": ["A"], "reasoning": "two"}
    client1.structured_response = StructuredResponse(data=ranking1, raw=orjson.dumps(ranking1).decode())
    client2.structured_response = StructuredResponse(data=ranking2, raw=orjson.dumps(ranking2).decode())

    await rank_one(ranking_task, client1, {gen.id: gen}, results_path=results_path)

    # client1 already succeeded; a re-run must skip it (resume) rather than re-rank.
    changed = {"ranking": ["A"], "reasoning": "changed"}
    client1.structured_response = StructuredResponse(data=changed, raw=orjson.dumps(changed).decode())

    results, failures = await rank_all([ranking_task], [gen], clients=[client1, client2], results_path=results_path)

    assert len(failures) == 0
    assert len(results) == 2
    by_author = {result.author: result for result in results}
    assert by_author[client1.name].reasoning == "one"  # loaded, not re-ranked
    assert by_author[client2.name].reasoning == "two"


async def test_rank_all_ignores_records_outside_current_universe(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    client = MockLLMClient()
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    path = tmp_path / "rankings.jsonl"
    old = {"ranking": ["A"], "reasoning": "old"}
    client.structured_response = StructuredResponse(data=old, raw=orjson.dumps(old).decode())

    # Record a result for a *different* ranking task — it must not count toward this run.
    await rank_one(make_ranking_task(generations={"A": gen.id}), client, {gen.id: gen}, results_path=path)
    new = {"ranking": ["A"], "reasoning": "new"}
    client.structured_response = StructuredResponse(data=new, raw=orjson.dumps(new).decode())

    results, failures = await rank_all([ranking_task], [gen], clients=[client], results_path=path)

    assert len(failures) == 0
    assert len(results) == 1
    assert results[0].ranking_task_id == ranking_task.id
    assert results[0].reasoning == "new"  # ranked this run, not the stale loaded record


async def test_rank_all_opens_and_closes_clients(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    client = MockLLMClient()
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    ranking = {"ranking": ["A"], "reasoning": "ok"}
    client.structured_response = StructuredResponse(data=ranking, raw=orjson.dumps(ranking).decode())
    client.raise_on_enter = True

    with pytest.raises(RuntimeError, match="Enter boom"):
        await rank_all([ranking_task], [gen], clients=[client])

    client.raise_on_enter = False
    client.raise_on_exit = True

    with pytest.raises(RuntimeError, match="Exit boom"):
        await rank_all([ranking_task], [gen], clients=[client])
