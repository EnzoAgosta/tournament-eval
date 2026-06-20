"""Tests for the orchestration stages.

The LLM layer is pydantic-ai's; these tests exercise *our* logic — author
resolution, resume, persistence streaming, failure partitioning, alias shuffling —
against pydantic-ai's offline test models (see ``conftest``).  No network.
"""

import json
import uuid
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelResponse,
    RequestUsage,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UnexpectedModelBehavior,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from tests.conftest import (
    GenerationResultFactory,
    GenerationTaskFactory,
    RankingResultFactory,
    RankingTaskFactory,
    generation_agent,
    raising_generation_agent,
    raising_ranking_agent,
    ranking_agent,
)
from tournament_eval.models import GenerationFailure, GenerationResult, RankingFailure, RankingResult, RankingTask
from tournament_eval.orchestration import (
    _extract_ranking_failure_details,
    _resolve_author,
    build_generation_task,
    build_generation_tasks,
    build_ranking_task,
    build_ranking_tasks,
    deanonymize_ranking,
    generate_all,
    generate_one,
    rank_all,
    rank_one,
)
from tournament_eval.persistence import _dumps, read_ranking_failure_file
from tournament_eval.ranking import DefaultRankingTemplate, RankingResponse


async def test_generate_one_returns_generation_result(make_generation_task: GenerationTaskFactory) -> None:
    agent = generation_agent(text="hello", author="gen-model")
    task = make_generation_task()

    result = await generate_one(task, agent)

    assert isinstance(result, GenerationResult)
    assert isinstance(result.id, uuid.UUID)
    assert result.generation_task_id == task.id
    assert result.generation_prompt == task.generation_prompt
    assert result.output == "hello"
    assert result.author == "gen-model"
    assert isinstance(result.metadata["input_tokens"], int)
    assert "output_tokens" in result.metadata


async def test_generate_one_returns_generation_failure_on_error(
    make_generation_task: GenerationTaskFactory,
) -> None:
    agent = raising_generation_agent(author="boom-model")
    task = make_generation_task()

    result = await generate_one(task, agent)

    assert isinstance(result, GenerationFailure)
    assert result.generation_task_id == task.id
    assert result.author == "boom-model"
    assert result.error_type == "RuntimeError"
    assert result.message == "boom"


async def test_generate_one_saves_to_results_path(
    make_generation_task: GenerationTaskFactory,
    tmp_path: Path,
) -> None:
    agent = generation_agent(text="hello", author="gen-model")
    task = make_generation_task()
    results_path = tmp_path / "results.jsonl"

    result = await generate_one(task, agent, results_path=results_path)

    assert isinstance(result, GenerationResult)
    with open(results_path, "rb") as f:
        assert f.read() == _dumps(result) + b"\n"


async def test_generate_one_saves_to_failures_path(
    make_generation_task: GenerationTaskFactory,
    tmp_path: Path,
) -> None:
    agent = raising_generation_agent(author="boom-model")
    task = make_generation_task()
    failures_path = tmp_path / "failures.jsonl"

    result = await generate_one(task, agent, failures_path=failures_path)

    assert isinstance(result, GenerationFailure)
    with open(failures_path, "rb") as f:
        assert f.read() == _dumps(result) + b"\n"


async def test_generate_all_returns_list_of_generation_results(
    make_generation_task: GenerationTaskFactory,
) -> None:
    agent1 = generation_agent(text="one", author="client1")
    agent2 = generation_agent(text="two", author="client2")
    task = make_generation_task()

    results, failures = await generate_all([task], agents=[agent1, agent2])

    assert len(failures) == 0
    assert len(results) == 2
    by_author = {r.author: r for r in results}
    assert {r.generation_task_id for r in results} == {task.id}
    assert by_author["client1"].output == "one"
    assert by_author["client2"].output == "two"


async def test_generate_all_saves_to_results_path(
    make_generation_task: GenerationTaskFactory,
    tmp_path: Path,
) -> None:
    agent1 = generation_agent(text="one", author="client1")
    agent2 = generation_agent(text="two", author="client2")
    task = make_generation_task()
    results_path = tmp_path / "results.jsonl"

    results, failures = await generate_all([task], agents=[agent1, agent2], results_path=results_path)

    assert len(failures) == 0
    assert len(results) == 2
    on_disk = results_path.read_bytes()
    assert on_disk == _dumps(results[0]) + b"\n" + _dumps(results[1]) + b"\n"


async def test_generate_all_saves_to_failures_path(
    make_generation_task: GenerationTaskFactory,
    tmp_path: Path,
) -> None:
    failure_agent = raising_generation_agent(author="client1")
    result_agent = generation_agent(text="two", author="client2")
    task = make_generation_task()
    failures_path = tmp_path / "failures.jsonl"

    results, failures = await generate_all([task], agents=[failure_agent, result_agent], failures_path=failures_path)

    assert len(results) == 1
    assert len(failures) == 1
    result, failure = results[0], failures[0]
    assert isinstance(result, GenerationResult)
    assert isinstance(failure, GenerationFailure)
    assert result.author == "client2"
    assert failure.author == "client1"
    assert failure.error_type == "RuntimeError"
    assert failure.message == "boom"

    with open(failures_path, "rb") as f:
        assert f.read() == _dumps(failure) + b"\n"


async def test_generate_all_saves_to_results_and_failures_path(
    make_generation_task: GenerationTaskFactory,
    tmp_path: Path,
) -> None:
    failure_agent = raising_generation_agent(author="client1")
    result_agent = generation_agent(text="two", author="client2")
    task = make_generation_task()
    results_path = tmp_path / "results.jsonl"
    failures_path = tmp_path / "failures.jsonl"

    results, failures = await generate_all(
        [task], agents=[failure_agent, result_agent], results_path=results_path, failures_path=failures_path
    )

    assert len(results) == 1
    assert len(failures) == 1
    result, failure = results[0], failures[0]
    with open(results_path, "rb") as f:
        assert f.read() == _dumps(result) + b"\n"
    with open(failures_path, "rb") as f:
        assert f.read() == _dumps(failure) + b"\n"


async def test_generate_all_reads_generation_results_from_results_path(
    make_generation_task: GenerationTaskFactory,
    tmp_path: Path,
) -> None:
    agent1 = generation_agent(text="one", author="client1")
    task = make_generation_task()
    results_path = tmp_path / "results.jsonl"

    await generate_one(task, agent1, results_path=results_path)

    agent2 = generation_agent(text="two", author="client2")
    results, failures = await generate_all([task], agents=[agent1, agent2], results_path=results_path)

    assert len(failures) == 0
    assert len(results) == 2
    by_author = {r.author: r for r in results}
    assert by_author["client1"].output == "one"
    assert by_author["client2"].output == "two"


async def test_generate_all_returns_all_generation_results(
    make_generation_task: GenerationTaskFactory,
    tmp_path: Path,
) -> None:
    agent1 = generation_agent(text="one", author="client1")
    agent2 = generation_agent(text="two", author="client2")
    task = make_generation_task()
    results_path = tmp_path / "results.jsonl"

    results, _ = await generate_all([task], agents=[agent1, agent2], results_path=results_path)
    assert len(results) == 2

    agent3 = generation_agent(text="three", author="client3")
    results, _ = await generate_all([task], agents=[agent1, agent2, agent3], results_path=results_path)
    assert len(results) == 3
    by_author = {r.author: r for r in results}
    assert by_author["client1"].output == "one"
    assert by_author["client2"].output == "two"
    assert by_author["client3"].output == "three"


async def test_generate_all_ignores_records_outside_current_universe(
    tmp_path: Path, make_generation_task: GenerationTaskFactory
) -> None:
    task = make_generation_task(prompt="p")
    path = tmp_path / "generations.jsonl"
    agent = generation_agent(text="old", author="test-model")
    await generate_one(task, agent, results_path=path)
    agent2 = generation_agent(text="new", author="other-model")
    results, _ = await generate_all([task], [agent2], results_path=path)
    assert {r.author for r in results} == {"other-model"}


async def test_generate_all_rejects_duplicate_authors(
    make_generation_task: GenerationTaskFactory,
) -> None:

    agent1 = generation_agent(text="one", author="dup")
    agent2 = generation_agent(text="two", author="dup")
    with pytest.raises(ValueError, match="Duplicate agent authors"):
        await generate_all([make_generation_task()], agents=[agent1, agent2])


async def test_generate_all_distinct_name_overrides_model_name(
    make_generation_task: GenerationTaskFactory,
) -> None:

    a_hot = Agent(TestModel(custom_output_text="hot", model_name="gpt-4o"), output_type=str, name="gpt-4o-hot")
    a_cold = Agent(TestModel(custom_output_text="cold", model_name="gpt-4o"), output_type=str, name="gpt-4o-cold")
    results, failures = await generate_all([make_generation_task()], agents=[a_hot, a_cold])
    assert len(failures) == 0
    assert len(results) == 2
    assert {r.author for r in results} == {"gpt-4o-hot", "gpt-4o-cold"}


def test_build_generation_task_stamps_fresh_id() -> None:
    task = build_generation_task("translate me")
    assert isinstance(task.id, uuid.UUID)
    assert task.generation_prompt == "translate me"


def test_build_generation_task_uses_explicit_id() -> None:
    explicit = uuid.uuid4()
    task = build_generation_task("translate me", id=explicit)
    assert task.id == explicit


def test_build_generation_task_saves_to_tasks_path(tmp_path: Path) -> None:
    tasks_path = tmp_path / "generation_tasks.jsonl"
    task = build_generation_task("translate me", tasks_path=tasks_path)
    with open(tasks_path, "rb") as f:
        assert f.read() == _dumps(task) + b"\n"


def test_build_generation_task_no_resume_always_new_id(tmp_path: Path) -> None:

    tasks_path = tmp_path / "generation_tasks.jsonl"
    t1 = build_generation_task("same prompt", tasks_path=tasks_path)
    t2 = build_generation_task("same prompt", tasks_path=tasks_path)
    assert t1.id != t2.id
    assert tasks_path.read_text().count("\n") == 2


def test_build_generation_tasks_returns_tasks_in_order(tmp_path: Path) -> None:
    prompts = ["one", "two", "three"]
    tasks = build_generation_tasks(prompts, tasks_path=tmp_path / "generation_tasks.jsonl")
    assert [t.generation_prompt for t in tasks] == prompts
    assert all(isinstance(t.id, uuid.UUID) for t in tasks)


def test_build_generation_tasks_accepts_an_iterable(tmp_path: Path) -> None:

    path = tmp_path / "sentences.txt"
    path.write_text("alpha\nbeta\ngamma\n")
    with open(path) as f:
        tasks = build_generation_tasks((line.rstrip() for line in f), tasks_path=tmp_path / "tasks.jsonl")
    assert [t.generation_prompt for t in tasks] == ["alpha", "beta", "gamma"]


def test_build_generation_tasks_requires_tasks_path() -> None:

    with pytest.raises(TypeError):
        build_generation_tasks(["one"])  # type: ignore[call-arg]


def test_build_generation_tasks_collapses_duplicate_prompts(tmp_path: Path) -> None:
    tasks_path = tmp_path / "generation_tasks.jsonl"
    tasks = build_generation_tasks(["hello", "world", "hello"], tasks_path=tasks_path)
    assert len(tasks) == 2
    assert [t.generation_prompt for t in tasks] == ["hello", "world"]
    assert tasks_path.read_text().count("\n") == 2


def test_build_generation_tasks_reuses_existing_on_rerun(tmp_path: Path) -> None:

    tasks_path = tmp_path / "generation_tasks.jsonl"
    first = build_generation_tasks(["hello", "world"], tasks_path=tasks_path)
    second = build_generation_tasks(["hello", "world"], tasks_path=tasks_path)
    assert [t.id for t in second] == [t.id for t in first]
    assert tasks_path.read_text().count("\n") == 2


def test_build_generation_tasks_adds_new_prompts_without_touching_existing(tmp_path: Path) -> None:
    tasks_path = tmp_path / "generation_tasks.jsonl"
    first = build_generation_tasks(["hello", "world"], tasks_path=tasks_path)
    second = build_generation_tasks(["hello", "world", "bye"], tasks_path=tasks_path)

    by_prompt = {t.generation_prompt: t for t in second}
    assert by_prompt["hello"].id == next(t.id for t in first if t.generation_prompt == "hello")
    assert by_prompt["world"].id == next(t.id for t in first if t.generation_prompt == "world")
    assert isinstance(by_prompt["bye"].id, uuid.UUID)
    assert tasks_path.read_text().count("\n") == 3


async def test_build_generation_tasks_resume_end_to_end(tmp_path: Path) -> None:

    tasks_path = tmp_path / "generation_tasks.jsonl"
    results_path = tmp_path / "generations.jsonl"
    failures_path = tmp_path / "generation_failures.jsonl"
    prompts = ["translate: one", "translate: two"]
    agent = generation_agent(text="out", author="m")

    tasks = build_generation_tasks(prompts, tasks_path=tasks_path)
    results, _ = await generate_all(tasks, [agent], results_path=results_path, failures_path=failures_path)
    assert len(results) == 2
    first_ids = {r.id for r in results}

    tasks_again = build_generation_tasks(prompts, tasks_path=tasks_path)
    results_again, _ = await generate_all(tasks_again, [agent], results_path=results_path, failures_path=failures_path)
    assert {r.id for r in results_again} == first_ids


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


def test_build_ranking_task_rejects_mixed_tasks(make_generation_result: GenerationResultFactory) -> None:
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
    with open(tasks_path, "rb") as f:
        assert f.read() == _dumps(task) + b"\n"


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


def test_build_ranking_tasks_reuses_existing(
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
    assert new_ranking_task.generations == ranking_task.generations
    assert new_ranking_task.generations["A"] == old_result.id


def test_build_ranking_tasks_skips_tasks_without_results(make_generation_task: GenerationTaskFactory) -> None:
    tasks = make_generation_task()
    ranking_task = build_ranking_tasks(tasks=[tasks], generation_results=[], ranking_prompt="Test")
    assert len(ranking_task) == 0


async def test_rank_one_returns_ranking_result(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    gen_a, gen_b = make_generation_result(), make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen_a.id, "B": gen_b.id})
    lookup = {gen_a.id: gen_a, gen_b.id: gen_b}
    agent = ranking_agent(output_args={"ranking": ["B", "A"], "reasoning": "b is better"}, author="ranker")

    result = await rank_one(ranking_task, agent, lookup)

    assert isinstance(result, RankingResult)
    assert isinstance(result.id, uuid.UUID)
    assert result.ranking_task_id == ranking_task.id
    assert result.generation_task_id == ranking_task.generation_task_id
    assert result.author == "ranker"
    assert result.raw_model_ranking == ["B", "A"]
    assert result.ranking == [gen_b.id, gen_a.id]
    assert result.ranking_reasoning == "b is better"
    assert result.reasoning is None
    assert json.loads(result.raw_response) == {"ranking": ["B", "A"], "reasoning": "b is better"}
    assert ranking_task.ranking_prompt in result.ranking_prompt
    assert "output_tokens" in result.metadata


async def test_rank_one_returns_ranking_failure_on_error(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    agent = raising_ranking_agent(author="boom-ranker")

    result = await rank_one(ranking_task, agent, {gen.id: gen})

    assert isinstance(result, RankingFailure)
    assert result.ranking_task_id == ranking_task.id
    assert result.generation_task_id == ranking_task.generation_task_id
    assert result.author == "boom-ranker"
    assert result.error_type == "RuntimeError"
    assert result.message == "boom"
    assert result.ranking_prompt


async def test_rank_one_returns_ranking_failure_on_unknown_alias(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})

    agent = ranking_agent(output_args={"ranking": ["Z"], "reasoning": "ok"}, author="ranker")

    result = await rank_one(ranking_task, agent, {gen.id: gen})

    assert isinstance(result, RankingFailure)
    assert result.ranking_task_id == ranking_task.id
    assert result.generation_task_id == ranking_task.generation_task_id
    assert result.author == "ranker"
    assert result.error_type == "ValueError"
    assert "Z" in result.message
    assert result.ranking_prompt


async def test_rank_one_saves_to_results_path(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    agent = ranking_agent(output_args={"ranking": ["A"], "reasoning": "ok"}, author="ranker")
    results_path = tmp_path / "rankings.jsonl"

    result = await rank_one(ranking_task, agent, {gen.id: gen}, results_path=results_path)

    assert isinstance(result, RankingResult)
    with open(results_path, "rb") as f:
        assert f.read() == _dumps(result) + b"\n"


async def test_rank_one_saves_to_failures_path(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    agent = raising_ranking_agent(author="boom-ranker")
    failures_path = tmp_path / "ranking_failures.jsonl"

    result = await rank_one(ranking_task, agent, {gen.id: gen}, failures_path=failures_path)

    assert isinstance(result, RankingFailure)
    with open(failures_path, "rb") as f:
        assert f.read() == _dumps(result) + b"\n"


async def test_rank_one_failure_details_captures_model_output_when_alias_check_fails(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:

    gen_a = make_generation_result(author="a")
    gen_b = make_generation_result(author="b")
    ranking_task = make_ranking_task(generations={"A": gen_a.id, "B": gen_b.id})
    # Valid RankingResponse shape, but duplicates "A" and omits "B" — fails parse().
    agent = ranking_agent(output_args={"ranking": ["A", "A"], "reasoning": "tied them"}, author="ranker")

    result = await rank_one(ranking_task, agent, {gen_a.id: gen_a, gen_b.id: gen_b})

    assert isinstance(result, RankingFailure)
    assert result.error_type == "ValueError"
    assert result.details is not None
    model_output = result.details["model_output"]
    assert isinstance(model_output, str)
    assert json.loads(model_output) == {"ranking": ["A", "A"], "reasoning": "tied them"}


async def test_rank_one_failure_details_captures_validation_errors_on_pydantic_failure(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:

    gen = make_generation_result(author="a")
    ranking_task = make_ranking_task(generations={"A": gen.id})

    def emit_malformed(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        # 'ranking' as a string instead of a list — fails pydantic validation every call.
        return ModelResponse(parts=[ToolCallPart("final_result", {"ranking": "not-a-list", "reasoning": "x"})])

    agent = Agent(
        FunctionModel(emit_malformed, model_name="bad"),
        output_type=DefaultRankingTemplate().response_model,
        name="bad",
        retries=1,
    )

    result = await rank_one(ranking_task, agent, {gen.id: gen})

    assert isinstance(result, RankingFailure)
    assert result.error_type == "UnexpectedModelBehavior"
    assert result.details is not None
    assert result.details["cause_type"] == "ValidationError"
    errors = result.details["validation_errors"]
    assert isinstance(errors, list)
    assert len(errors) >= 1

    assert errors[0]["input"] == "not-a-list"
    assert errors[0]["loc"] == ["ranking"]


async def test_rank_one_failure_details_captures_cause_on_misc_failure(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:

    gen = make_generation_result(author="a")
    ranking_task = make_ranking_task(generations={"A": gen.id})

    def emit_empty(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[])

    agent = Agent(
        FunctionModel(emit_empty, model_name="empty"),
        output_type=DefaultRankingTemplate().response_model,
        name="empty",
        retries=1,
    )

    result = await rank_one(ranking_task, agent, {gen.id: gen})

    assert isinstance(result, RankingFailure)
    assert result.details is not None
    assert "cause_type" in result.details
    assert "cause_message" in result.details
    # No pydantic validation involved, so no validation_errors key.
    assert "validation_errors" not in result.details
    assert "model_output" not in result.details


async def test_rank_one_failure_details_captures_cause_even_on_a_bare_runtime_error(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:

    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    agent = raising_ranking_agent(author="boom-ranker")

    result = await rank_one(ranking_task, agent, {gen.id: gen})

    assert isinstance(result, RankingFailure)
    assert result.error_type == "RuntimeError"
    assert result.details is not None
    assert result.details["cause_type"] == "ExceptionGroup"

    assert "validation_errors" not in result.details
    assert "model_output" not in result.details


async def test_rank_one_failure_details_round_trips_through_persistence(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:

    gen = make_generation_result(author="a")
    ranking_task = make_ranking_task(generations={"A": gen.id})

    def emit_malformed(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("final_result", {"ranking": "not-a-list", "reasoning": "x"})])

    agent = Agent(
        FunctionModel(emit_malformed, model_name="bad"),
        output_type=DefaultRankingTemplate().response_model,
        name="bad",
        retries=1,
    )
    failures_path = tmp_path / "ranking_failures.jsonl"

    result = await rank_one(ranking_task, agent, {gen.id: gen}, failures_path=failures_path)
    assert isinstance(result, RankingFailure)
    assert result.details is not None

    [restored] = read_ranking_failure_file(failures_path)
    assert restored == result
    assert restored.details == result.details
    assert restored.details is not None
    assert restored.details["cause_type"] == "ValidationError"


def test_extract_ranking_failure_details_returns_none_for_a_truly_bare_exception() -> None:

    assert _extract_ranking_failure_details(RuntimeError("boom")) is None


def test_extract_ranking_failure_details_captures_unexpected_model_behavior_body() -> None:

    exc = UnexpectedModelBehavior("unexpected status 500", body='{"error":"rate limited"}')
    details = _extract_ranking_failure_details(exc)
    assert details is not None

    assert json.loads(details["body"]) == {"error": "rate limited"}  # type: ignore[arg-type]


def test_extract_ranking_failure_details_skips_validation_errors_that_wont_report() -> None:

    # validation_errors is simply omitted rather than populated with garbage.

    class BrokenValidationError(ValueError):
        def errors(self) -> list[dict[str, object]]:
            raise RuntimeError("this validation error can't report")

    cause = BrokenValidationError("broken")
    exc = RuntimeError("wrapped")
    exc.__cause__ = cause
    details = _extract_ranking_failure_details(exc)
    assert details is not None
    assert details["cause_type"] == "BrokenValidationError"
    assert "validation_errors" not in details


async def test_rank_all_returns_list_of_ranking_results(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    agent1 = ranking_agent(output_args={"ranking": ["A"], "reasoning": "one"}, author="client1")
    agent2 = ranking_agent(output_args={"ranking": ["A"], "reasoning": "two"}, author="client2")

    results, failures = await rank_all([ranking_task], [gen], agents=[agent1, agent2])

    assert len(failures) == 0
    assert len(results) == 2
    by_author = {r.author: r for r in results}
    assert by_author["client1"].ranking_reasoning == "one"
    assert by_author["client2"].ranking_reasoning == "two"
    assert all(r.ranking == [gen.id] for r in results)


async def test_rank_all_saves_to_results_path(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    agent1 = ranking_agent(output_args={"ranking": ["A"], "reasoning": "one"}, author="client1")
    agent2 = ranking_agent(output_args={"ranking": ["A"], "reasoning": "two"}, author="client2")
    results_path = tmp_path / "rankings.jsonl"

    results, failures = await rank_all([ranking_task], [gen], agents=[agent1, agent2], results_path=results_path)

    assert len(failures) == 0
    assert len(results) == 2
    on_disk = results_path.read_bytes()
    assert on_disk == _dumps(results[0]) + b"\n" + _dumps(results[1]) + b"\n"


async def test_rank_all_saves_to_failures_path(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    failure_agent = raising_ranking_agent(author="client1")
    result_agent = ranking_agent(output_args={"ranking": ["A"], "reasoning": "ok"}, author="client2")
    failures_path = tmp_path / "ranking_failures.jsonl"

    results, failures = await rank_all(
        [ranking_task], [gen], agents=[failure_agent, result_agent], failures_path=failures_path
    )

    assert len(results) == 1
    assert len(failures) == 1
    result, failure = results[0], failures[0]
    assert isinstance(result, RankingResult)
    assert isinstance(failure, RankingFailure)
    assert result.author == "client2"
    assert failure.author == "client1"
    assert failure.error_type == "RuntimeError"
    assert failure.message == "boom"

    with open(failures_path, "rb") as f:
        assert f.read() == _dumps(failure) + b"\n"


async def test_rank_all_saves_to_results_and_failures_path(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    failure_agent = raising_ranking_agent(author="client1")
    result_agent = ranking_agent(output_args={"ranking": ["A"], "reasoning": "ok"}, author="client2")
    results_path = tmp_path / "rankings.jsonl"
    failures_path = tmp_path / "ranking_failures.jsonl"

    results, failures = await rank_all(
        [ranking_task],
        [gen],
        agents=[failure_agent, result_agent],
        results_path=results_path,
        failures_path=failures_path,
    )

    assert len(results) == 1
    assert len(failures) == 1
    result, failure = results[0], failures[0]
    with open(results_path, "rb") as f:
        assert f.read() == _dumps(result) + b"\n"
    with open(failures_path, "rb") as f:
        assert f.read() == _dumps(failure) + b"\n"


async def test_rank_all_reads_ranking_results_from_results_path(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    results_path = tmp_path / "rankings.jsonl"
    agent1 = ranking_agent(output_args={"ranking": ["A"], "reasoning": "one"}, author="client1")
    agent2 = ranking_agent(output_args={"ranking": ["A"], "reasoning": "two"}, author="client2")

    await rank_one(ranking_task, agent1, {gen.id: gen}, results_path=results_path)

    agent1_again = ranking_agent(output_args={"ranking": ["A"], "reasoning": "changed"}, author="client1")
    results, failures = await rank_all([ranking_task], [gen], agents=[agent1_again, agent2], results_path=results_path)

    assert len(failures) == 0
    assert len(results) == 2
    by_author = {result.author: result for result in results}
    assert by_author["client1"].ranking_reasoning == "one"
    assert by_author["client2"].ranking_reasoning == "two"


async def test_rank_all_ignores_records_outside_current_universe(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    path = tmp_path / "rankings.jsonl"
    agent = ranking_agent(output_args={"ranking": ["A"], "reasoning": "old"}, author="test-model")

    await rank_one(make_ranking_task(generations={"A": gen.id}), agent, {gen.id: gen}, results_path=path)
    agent_new = ranking_agent(output_args={"ranking": ["A"], "reasoning": "new"}, author="test-model")

    results, failures = await rank_all([ranking_task], [gen], agents=[agent_new], results_path=path)

    assert len(failures) == 0
    assert len(results) == 1
    assert results[0].ranking_task_id == ranking_task.id
    assert results[0].ranking_reasoning == "new"


async def test_rank_all_rejects_duplicate_authors(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    agent1 = ranking_agent(author="dup")
    agent2 = ranking_agent(author="dup")
    with pytest.raises(ValueError, match="Duplicate agent authors"):
        await rank_all([ranking_task], [gen], agents=[agent1, agent2])


async def test_rank_all_rejects_agent_output_type_mismatch(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:

    class OtherResponse(BaseModel):
        winner: str

    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    agent = ranking_agent(author="mismatch", response_model=OtherResponse, output_args={"winner": "A"})
    with pytest.raises(ValueError, match="output_type"):
        await rank_all([ranking_task], [gen], agents=[agent])


def test_author_prefers_explicit_name() -> None:
    agent = generation_agent(text="x", author="the-model")
    agent.name = "explicit-label"
    assert _resolve_author(agent) == "explicit-label"


def test_author_falls_back_to_string_model_name() -> None:

    agent = Agent("some-model-id", defer_model_check=True, output_type=str)
    assert _resolve_author(agent) == "some-model-id"


def test_author_raises_when_unresolvable() -> None:
    agent = Agent(model=None, output_type=str)
    with pytest.raises(ValueError, match="no resolvable author"):
        _resolve_author(agent)


async def test_generate_one_captures_reasoning_trace(make_generation_task: GenerationTaskFactory) -> None:

    def fn(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ThinkingPart(content="let me think"), TextPart(content="answer")],
            usage=RequestUsage(input_tokens=1, output_tokens=1),
        )

    agent = Agent(FunctionModel(fn, model_name="thinker"), output_type=str)
    result = await generate_one(make_generation_task(), agent)
    assert isinstance(result, GenerationResult)
    assert result.output == "answer"
    assert result.reasoning == "let me think"


async def test_rank_one_captures_reasoning_trace(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:

    def fn(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ThinkingPart(content="weighing fluency vs accuracy"),
                ToolCallPart("final_result", {"ranking": ["A"], "reasoning": "A is clearer"}),
            ],
            usage=RequestUsage(input_tokens=1, output_tokens=1),
        )

    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    agent = Agent(FunctionModel(fn, model_name="ranker"), output_type=RankingResponse)
    result = await rank_one(ranking_task, agent, {gen.id: gen})
    assert isinstance(result, RankingResult)
    assert result.ranking_reasoning == "A is clearer"
    assert result.reasoning == "weighing fluency vs accuracy"


def test_deanonymize_ranking_maps_ids_to_authors(
    make_generation_result: GenerationResultFactory, make_ranking_result: RankingResultFactory
) -> None:
    gen_a = make_generation_result(author="alpha")
    gen_b = make_generation_result(author="beta")
    gen_c = make_generation_result(author="gamma")

    result = make_ranking_result(ranking=[gen_b.id, gen_a.id, gen_c.id])

    deanon = deanonymize_ranking(result, [gen_a, gen_b, gen_c])

    assert deanon == ["beta", "alpha", "gamma"]


def test_deanonymize_ranking_accepts_any_iterable(
    make_generation_result: GenerationResultFactory, make_ranking_result: RankingResultFactory
) -> None:
    gen_a = make_generation_result(author="alpha")
    gen_b = make_generation_result(author="beta")
    result = make_ranking_result(ranking=[gen_a.id, gen_b.id])

    deanon = deanonymize_ranking(result, (g for g in [gen_a, gen_b]))

    assert deanon == ["alpha", "beta"]


def test_deanonymize_ranking_raises_on_unknown_id(
    make_generation_result: GenerationResultFactory, make_ranking_result: RankingResultFactory
) -> None:
    gen_a = make_generation_result(author="alpha")
    other = make_generation_result(author="orphan")
    result = make_ranking_result(ranking=[gen_a.id, other.id])

    with pytest.raises(KeyError):
        deanonymize_ranking(result, [gen_a])


async def test_generate_all_fires_on_result_for_each_success(
    make_generation_task: GenerationTaskFactory,
) -> None:
    agent1 = generation_agent(text="one", author="c1")
    agent2 = generation_agent(text="two", author="c2")
    task = make_generation_task()

    seen: list[GenerationResult] = []
    results, failures = await generate_all([task], agents=[agent1, agent2], on_result=seen.append)

    assert failures == []
    assert len(results) == 2

    assert len(seen) == 2
    assert all(isinstance(r, GenerationResult) for r in seen)
    assert {r.output for r in seen} == {"one", "two"}


async def test_generate_all_fires_on_failure_for_each_failure(
    make_generation_task: GenerationTaskFactory,
) -> None:
    bad = raising_generation_agent(author="bad")
    good = generation_agent(text="ok", author="good")
    task = make_generation_task()

    results_seen: list[GenerationResult] = []
    failures_seen: list[GenerationFailure] = []
    results, failures = await generate_all(
        [task],
        agents=[bad, good],
        on_result=results_seen.append,
        on_failure=failures_seen.append,
    )

    assert len(results) == 1
    assert len(failures) == 1
    assert len(results_seen) == 1
    assert len(failures_seen) == 1
    assert isinstance(failures_seen[0], GenerationFailure)
    assert failures_seen[0].author == "bad"


async def test_generate_all_callback_fires_after_persistence(
    make_generation_task: GenerationTaskFactory, tmp_path: Path
) -> None:
    agent = generation_agent(text="ok", author="c1")
    task = make_generation_task()
    results_path = tmp_path / "results.jsonl"

    def on_result(_r: GenerationResult) -> None:
        assert results_path.exists()
        assert results_path.read_bytes().count(b"\n") >= 1

    await generate_all([task], agents=[agent], results_path=results_path, on_result=on_result)


async def test_generate_all_resume_does_not_fire_callback_for_loaded_successes(
    make_generation_task: GenerationTaskFactory, tmp_path: Path
) -> None:
    task = make_generation_task()
    results_path = tmp_path / "results.jsonl"
    agent = generation_agent(text="ok", author="c1")

    seen_first: list[GenerationResult] = []
    await generate_all([task], agents=[agent], results_path=results_path, on_result=seen_first.append)
    assert len(seen_first) == 1

    seen_second: list[GenerationResult] = []
    await generate_all([task], agents=[agent], results_path=results_path, on_result=seen_second.append)
    assert seen_second == []


async def test_generate_all_swallows_raising_callback_as_warning(
    make_generation_task: GenerationTaskFactory,
) -> None:
    def bad_hook(_r: GenerationResult) -> None:
        raise RuntimeError("hook is broken")

    agent = generation_agent(text="ok", author="c1")
    task = make_generation_task()

    with pytest.warns(UserWarning, match="progress callback raised"):
        results, failures = await generate_all([task], agents=[agent], on_result=bad_hook)

    assert failures == []
    assert len(results) == 1


async def test_rank_all_fires_on_result_for_each_success(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    agent1 = ranking_agent(output_args={"ranking": ["A"], "reasoning": "one"}, author="c1")
    agent2 = ranking_agent(output_args={"ranking": ["A"], "reasoning": "two"}, author="c2")

    seen: list[RankingResult] = []
    results, failures = await rank_all([ranking_task], [gen], agents=[agent1, agent2], on_result=seen.append)

    assert failures == []
    assert len(results) == 2
    assert len(seen) == 2
    assert all(isinstance(r, RankingResult) for r in seen)
    assert {r.ranking_reasoning for r in seen} == {"one", "two"}


async def test_rank_all_fires_on_failure_for_each_failure(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    bad = raising_ranking_agent(author="bad")
    good = ranking_agent(output_args={"ranking": ["A"], "reasoning": "ok"}, author="good")

    failures_seen: list[RankingFailure] = []
    results, failures = await rank_all([ranking_task], [gen], agents=[bad, good], on_failure=failures_seen.append)

    assert len(results) == 1
    assert len(failures) == 1
    assert len(failures_seen) == 1
    assert isinstance(failures_seen[0], RankingFailure)
    assert failures_seen[0].author == "bad"


async def test_rank_all_swallows_raising_callback_as_warning(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    def bad_hook(_r: RankingResult) -> None:
        raise RuntimeError("hook is broken")

    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    agent = ranking_agent(output_args={"ranking": ["A"], "reasoning": "ok"}, author="c1")

    with pytest.warns(UserWarning, match="progress callback raised"):
        results, failures = await rank_all([ranking_task], [gen], agents=[agent], on_result=bad_hook)

    assert failures == []
    assert len(results) == 1
