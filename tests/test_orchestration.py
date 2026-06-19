"""Tests for the orchestration stages.

The LLM layer is pydantic-ai's; these tests exercise *our* logic — author
resolution, resume, persistence streaming, failure partitioning, alias shuffling —
against pydantic-ai's offline test models (see ``conftest``).  No network.
"""

import json
import uuid
from pathlib import Path

import pytest
from pydantic_ai import Agent, ModelMessage, ModelResponse, RequestUsage, TextPart, ThinkingPart, ToolCallPart
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
    _resolve_author,
    build_generation_task,
    build_generation_tasks,
    build_ranking_task,
    build_ranking_tasks,
    deanonimize_ranking,
    generate_all,
    generate_one,
    rank_all,
    rank_one,
)
from tournament_eval.persistence import _dumps


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

    # client1 already succeeded; a re-run with client2 must skip client1 (resume).
    agent2 = generation_agent(text="two", author="client2")
    results, failures = await generate_all([task], agents=[agent1, agent2], results_path=results_path)

    assert len(failures) == 0
    assert len(results) == 2
    by_author = {r.author: r for r in results}
    assert by_author["client1"].output == "one"  # loaded, not re-run
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
    # Two agents whose model names collide — same author, so resume would corrupt.
    agent1 = generation_agent(text="one", author="dup")
    agent2 = generation_agent(text="two", author="dup")
    with pytest.raises(ValueError, match="Duplicate agent authors"):
        await generate_all([make_generation_task()], agents=[agent1, agent2])


async def test_generate_all_distinct_name_overrides_model_name(
    make_generation_task: GenerationTaskFactory,
) -> None:
    # Same model name, distinct agent.name → distinct authors (the temperature-ablation case).

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
    # Singular builder never resumes — two calls produce distinct ids even for the
    # same prompt (resume is a batch concern; use build_generation_tasks for that).
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
    # An open text file (single-pass iterable) works — the typical naive usage.
    path = tmp_path / "sentences.txt"
    path.write_text("alpha\nbeta\ngamma\n")
    with open(path) as f:
        tasks = build_generation_tasks((line.rstrip() for line in f), tasks_path=tmp_path / "tasks.jsonl")
    assert [t.generation_prompt for t in tasks] == ["alpha", "beta", "gamma"]


def test_build_generation_tasks_requires_tasks_path() -> None:
    # tasks_path is required — persistence is the whole point (guards the naive
    # footgun where forgetting the path silently breaks resume).
    with pytest.raises(TypeError):
        build_generation_tasks(["one"])  # type: ignore[call-arg]


def test_build_generation_tasks_collapses_duplicate_prompts(tmp_path: Path) -> None:
    tasks_path = tmp_path / "generation_tasks.jsonl"
    tasks = build_generation_tasks(["hello", "world", "hello"], tasks_path=tasks_path)
    assert len(tasks) == 2  # duplicate "hello" collapsed to one task
    assert [t.generation_prompt for t in tasks] == ["hello", "world"]
    assert tasks_path.read_text().count("\n") == 2  # only unique prompts persisted


def test_build_generation_tasks_reuses_existing_on_rerun(tmp_path: Path) -> None:
    # The whole point: ids stay stable across reruns, so generate_all's resume matches.
    tasks_path = tmp_path / "generation_tasks.jsonl"
    first = build_generation_tasks(["hello", "world"], tasks_path=tasks_path)
    second = build_generation_tasks(["hello", "world"], tasks_path=tasks_path)
    assert [t.id for t in second] == [t.id for t in first]  # same ids, reused
    assert tasks_path.read_text().count("\n") == 2  # nothing new appended


def test_build_generation_tasks_adds_new_prompts_without_touching_existing(tmp_path: Path) -> None:
    tasks_path = tmp_path / "generation_tasks.jsonl"
    first = build_generation_tasks(["hello", "world"], tasks_path=tasks_path)
    second = build_generation_tasks(["hello", "world", "bye"], tasks_path=tasks_path)
    # hello/world reused (same ids); bye is new.
    by_prompt = {t.generation_prompt: t for t in second}
    assert by_prompt["hello"].id == next(t.id for t in first if t.generation_prompt == "hello")
    assert by_prompt["world"].id == next(t.id for t in first if t.generation_prompt == "world")
    assert isinstance(by_prompt["bye"].id, uuid.UUID)
    assert tasks_path.read_text().count("\n") == 3


async def test_build_generation_tasks_resume_end_to_end(tmp_path: Path) -> None:
    # The flagship promise: build → generate → rerun the exact same script → resume.
    tasks_path = tmp_path / "generation_tasks.jsonl"
    results_path = tmp_path / "generations.jsonl"
    failures_path = tmp_path / "generation_failures.jsonl"
    prompts = ["translate: one", "translate: two"]
    agent = generation_agent(text="out", author="m")

    tasks = build_generation_tasks(prompts, tasks_path=tasks_path)
    results, _ = await generate_all(tasks, [agent], results_path=results_path, failures_path=failures_path)
    assert len(results) == 2
    first_ids = {r.id for r in results}

    # Rerun the exact same script — tasks reused (stable ids), generation skipped.
    tasks_again = build_generation_tasks(prompts, tasks_path=tasks_path)
    results_again, _ = await generate_all(tasks_again, [agent], results_path=results_path, failures_path=failures_path)
    assert {r.id for r in results_again} == first_ids  # loaded, not regenerated


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
    assert new_ranking_task.id == ranking_task.id  # frozen: reused, not rebuilt
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
    assert result.reasoning is None  # TestModel produces no thinking parts
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
    assert result.ranking_prompt  # the rendered prompt that triggered the failure


async def test_rank_one_returns_ranking_failure_on_unknown_alias(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    # The model returns an alias not in the task — passes pydantic validation (it's a
    # list[str]) but fails the dynamic alias check in the template.
    agent = ranking_agent(output_args={"ranking": ["Z"]}, author="ranker")

    result = await rank_one(ranking_task, agent, {gen.id: gen})

    assert isinstance(result, RankingFailure)
    assert result.ranking_task_id == ranking_task.id
    assert result.generation_task_id == ranking_task.generation_task_id
    assert result.author == "ranker"
    assert result.error_type == "ValueError"
    assert "Z" in result.message
    assert result.ranking_prompt  # the rendered prompt that triggered the failure


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

    # client1 already succeeded; re-run must skip it (resume) rather than re-rank.
    agent1_again = ranking_agent(output_args={"ranking": ["A"], "reasoning": "changed"}, author="client1")
    results, failures = await rank_all([ranking_task], [gen], agents=[agent1_again, agent2], results_path=results_path)

    assert len(failures) == 0
    assert len(results) == 2
    by_author = {result.author: result for result in results}
    assert by_author["client1"].ranking_reasoning == "one"  # loaded, not re-ranked
    assert by_author["client2"].ranking_reasoning == "two"


async def test_rank_all_ignores_records_outside_current_universe(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory, tmp_path: Path
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    path = tmp_path / "rankings.jsonl"
    agent = ranking_agent(output_args={"ranking": ["A"], "reasoning": "old"}, author="test-model")

    # Record a result for a *different* ranking task — it must not count toward this run.
    await rank_one(make_ranking_task(generations={"A": gen.id}), agent, {gen.id: gen}, results_path=path)
    agent_new = ranking_agent(output_args={"ranking": ["A"], "reasoning": "new"}, author="test-model")

    results, failures = await rank_all([ranking_task], [gen], agents=[agent_new], results_path=path)

    assert len(failures) == 0
    assert len(results) == 1
    assert results[0].ranking_task_id == ranking_task.id
    assert results[0].ranking_reasoning == "new"  # ranked this run, not the stale loaded record


async def test_rank_all_rejects_duplicate_authors(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    gen = make_generation_result()
    ranking_task = make_ranking_task(generations={"A": gen.id})
    agent1 = ranking_agent(author="dup")
    agent2 = ranking_agent(author="dup")
    with pytest.raises(ValueError, match="Duplicate agent authors"):
        await rank_all([ranking_task], [gen], agents=[agent1, agent2])


def test_author_prefers_explicit_name() -> None:
    agent = generation_agent(text="x", author="the-model")
    agent.name = "explicit-label"  # an explicit name wins over the model name
    assert _resolve_author(agent) == "explicit-label"


def test_author_falls_back_to_string_model_name() -> None:
    # A deferred string/KnownModelName model: agent.model stays a str until first run.
    agent = Agent("some-model-id", defer_model_check=True, output_type=str)
    assert _resolve_author(agent) == "some-model-id"


def test_author_raises_when_unresolvable() -> None:
    agent = Agent(model=None, output_type=str)  # no name, no model
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
    # The ranker's thinking trace is captured separately from its ranking justification.
    from tournament_eval.ranking import RankingResponse

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
    assert result.ranking_reasoning == "A is clearer"  # the structured-output justification
    assert result.reasoning == "weighing fluency vs accuracy"  # the thinking trace


def test_deanonimize_ranking_maps_ids_to_authors(
    make_generation_result: GenerationResultFactory, make_ranking_result: RankingResultFactory
) -> None:
    gen_a = make_generation_result(author="alpha")
    gen_b = make_generation_result(author="beta")
    gen_c = make_generation_result(author="gamma")
    # ranking is best-first in id space: [gen_b, gen_a, gen_c]
    result = make_ranking_result(ranking=[gen_b.id, gen_a.id, gen_c.id])

    deanon = deanonimize_ranking(result, [gen_a, gen_b, gen_c])

    assert deanon == ["beta", "alpha", "gamma"]


def test_deanonimize_ranking_accepts_any_iterable(
    make_generation_result: GenerationResultFactory, make_ranking_result: RankingResultFactory
) -> None:
    gen_a = make_generation_result(author="alpha")
    gen_b = make_generation_result(author="beta")
    result = make_ranking_result(ranking=[gen_a.id, gen_b.id])

    # A generator, not a list — the helper materialises it into the lookup.
    deanon = deanonimize_ranking(result, (g for g in [gen_a, gen_b]))

    assert deanon == ["alpha", "beta"]


def test_deanonimize_ranking_raises_on_unknown_id(
    make_generation_result: GenerationResultFactory, make_ranking_result: RankingResultFactory
) -> None:
    gen_a = make_generation_result(author="alpha")
    other = make_generation_result(author="orphan")  # not in the ranking's candidate set
    result = make_ranking_result(ranking=[gen_a.id, other.id])

    # A ranking referencing an id absent from `generations` is a data-integrity error.
    with pytest.raises(KeyError):
        deanonimize_ranking(result, [gen_a])
