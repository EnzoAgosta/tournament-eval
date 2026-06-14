"""Tests for the orchestration pipeline.

These use Protocol-satisfying mock clients (canned behaviour, no wire protocol) —
the right altitude for testing fan-out, lifecycle, failure isolation, resume,
and persistence, none of which should care what a client is backed by.
"""

from pathlib import Path
from typing import Any

import pytest

from tests.conftest import _MakeGeneration, _MakeRankingTask, _MakeTask
from tournament_eval import GenerationResponse, StructuredResponse
from tournament_eval.models import (
    GenerationFailure,
    GenerationResult,
    RankingFailure,
    RankingResult,
)
from tournament_eval.orchestration import (
    build_ranking_task,
    build_ranking_tasks,
    generate_all,
    generate_one,
    rank_all,
    rank_one,
)
from tournament_eval.persistence import (
    read_generation_failure_file,
    read_generation_result_file,
    read_ranking_failure_file,
    read_ranking_result_file,
    read_ranking_task_file,
)


class _ManagedClient:
    """A mock client that IS an async context manager (like the built-in client)."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.opened = False
        self.closed = False

    @property
    def name(self) -> str:
        return self._name

    async def __aenter__(self) -> _ManagedClient:
        self.opened = True
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.closed = True

    async def generate(self, prompt: str) -> GenerationResponse:
        assert self.opened, "used before being opened"
        assert not self.closed, "used after being closed"
        return GenerationResponse(text=f"{self._name}:{prompt}", reasoning=None)

    async def generate_structured(self, _prompt: str, _schema: dict[str, object]) -> StructuredResponse:
        raise AssertionError("not exercised")


class _RankingClient:
    """A mock judge: returns a fixed structured verdict regardless of prompt."""

    def __init__(self, name: str = "judge", *, ranking: list[str] | None = None, fail: bool = False) -> None:
        self._name = name
        self._ranking = ranking if ranking is not None else ["A", "B"]
        self._fail = fail

    @property
    def name(self) -> str:
        return self._name

    async def generate(self, _prompt: str) -> GenerationResponse:
        raise AssertionError("not exercised")

    async def generate_structured(self, _prompt: str, _schema: dict[str, object]) -> StructuredResponse:
        if self._fail:
            raise RuntimeError("judge boom")
        return StructuredResponse(data={"ranking": self._ranking}, raw="{}")


# --------------------------------------------------------------------------- #
# generate_one / generate_all                                                   #
# --------------------------------------------------------------------------- #


class TestGenerateOne:
    async def test_success(self, make_client: Any, make_task: _MakeTask) -> None:
        result = await generate_one(make_task("p"), make_client("a", generate_responses={"p": "out"}))
        assert isinstance(result, GenerationResult)
        assert (result.author, result.output, result.reasoning) == ("a", "out", None)

    async def test_captures_reasoning_from_response(self, make_task: _MakeTask) -> None:
        class _Reasoner:
            name = "r"

            async def generate(self, _prompt: str) -> GenerationResponse:
                return GenerationResponse(text="answer", reasoning="because")

            async def generate_structured(self, _p: str, _s: dict[str, object]) -> StructuredResponse:
                raise AssertionError("not exercised")

        result = await generate_one(make_task("p"), _Reasoner())
        assert isinstance(result, GenerationResult)
        assert (result.output, result.reasoning) == ("answer", "because")

    async def test_failure_is_returned_not_raised(self, make_client: Any, make_task: _MakeTask) -> None:
        result = await generate_one(make_task("p"), make_client("a", fail_on={"p"}))
        assert isinstance(result, GenerationFailure)
        assert (result.author, result.error_type) == ("a", "RuntimeError")

    async def test_persists_result(self, tmp_path: Path, make_client: Any, make_task: _MakeTask) -> None:
        path = tmp_path / "generations.jsonl"
        client = make_client("a", generate_responses={"p": "out"})
        result = await generate_one(make_task("p"), client, results_path=path)
        assert read_generation_result_file(path) == [result]

    async def test_persists_failure(self, tmp_path: Path, make_client: Any, make_task: _MakeTask) -> None:
        path = tmp_path / "generation_failures.jsonl"
        result = await generate_one(make_task("p"), make_client("a", fail_on={"p"}), failures_path=path)
        assert read_generation_failure_file(path) == [result]


class TestGenerateAll:
    async def test_splits_results_and_failures(self, make_client: Any, make_task: _MakeTask) -> None:
        task = make_task("p")
        good = make_client("a", generate_responses={"p": "out"})
        bad = make_client("b", fail_on={"p"})
        results, failures = await generate_all([task], [good, bad])
        assert {r.author for r in results} == {"a"}
        assert {f.author for f in failures} == {"b"}

    async def test_opens_and_closes_context_manager_clients(self, make_client: Any, make_task: _MakeTask) -> None:
        managed = _ManagedClient("cm")
        plain = make_client("plain", generate_responses={"p": "x"})
        results, _ = await generate_all([make_task("p")], [managed, plain])
        assert managed.opened
        assert managed.closed
        assert {r.author for r in results} == {"cm", "plain"}

    async def test_streams_to_results_path(self, tmp_path: Path, make_client: Any, make_task: _MakeTask) -> None:
        task = make_task("p")
        path = tmp_path / "generations.jsonl"
        results, _ = await generate_all(
            [task], [make_client("a", generate_responses={"p": "out"})], results_path=path
        )
        assert read_generation_result_file(path) == results

    async def test_empty(self) -> None:
        assert await generate_all([], []) == ([], [])

    async def test_resume_skips_done_and_returns_complete_set(
        self, tmp_path: Path, make_client: Any, make_task: _MakeTask
    ) -> None:
        task = make_task("p")
        path = tmp_path / "generations.jsonl"
        # First run: 'a' succeeds and is persisted; 'b' fails (nothing persisted for it).
        a1 = make_client("a", generate_responses={"p": "out"})
        b1 = make_client("b", fail_on={"p"})
        run1_results, run1_failures = await generate_all([task], [a1, b1], results_path=path)
        assert {r.author for r in run1_results} == {"a"}
        assert {f.author for f in run1_failures} == {"b"}

        # Second run: 'a' is skipped (loaded from disk), 'b' is retried and now succeeds.
        a2 = make_client("a", fail_on={"p"})  # would fail if re-run — proves it's skipped
        b2 = make_client("b", generate_responses={"p": "fixed"})
        run2_results, run2_failures = await generate_all([task], [a2, b2], results_path=path)
        assert run2_failures == []
        # Returns the COMPLETE set: 'a' loaded from disk + 'b' produced this call.
        assert {(r.author, r.output) for r in run2_results} == {("a", "out"), ("b", "fixed")}

    async def test_resume_ignores_records_outside_current_universe(
        self, tmp_path: Path, make_client: Any, make_task: _MakeTask
    ) -> None:
        task = make_task("p")
        path = tmp_path / "generations.jsonl"
        # Persist a result for a client no longer in the pool.
        await generate_one(task, make_client("stale", generate_responses={"p": "old"}), results_path=path)
        results, _ = await generate_all([task], [make_client("a", generate_responses={"p": "out"})], results_path=path)
        assert {r.author for r in results} == {"a"}  # 'stale' filtered out


# --------------------------------------------------------------------------- #
# build_ranking_task(s)                                                         #
# --------------------------------------------------------------------------- #


class TestBuildRankingTasks:
    def test_assigns_aliases(self, make_generation: _MakeGeneration) -> None:
        results = [make_generation(), make_generation(), make_generation()]
        task = build_ranking_task(results, "Rank.")
        assert list(task.generations.keys()) == ["A", "B", "C"]
        assert set(task.generations.values()) == {r.id for r in results}
        assert task.ranking_prompt == "Rank."

    def test_seed_makes_shuffle_deterministic(self, make_generation: _MakeGeneration) -> None:
        results = [make_generation() for _ in range(5)]
        a = build_ranking_task(results, "R", random_seed=7)
        b = build_ranking_task(results, "R", random_seed=7)
        assert list(a.generations.values()) == list(b.generations.values())

    def test_does_not_mutate_input(self, make_generation: _MakeGeneration) -> None:
        results = [make_generation(), make_generation()]
        before = list(results)
        build_ranking_task(results, "R", random_seed=1)
        assert results == before

    def test_records_generation_task_id(self, make_task: _MakeTask, make_generation: _MakeGeneration) -> None:
        t = make_task()
        task = build_ranking_task([make_generation(task_id=t.id), make_generation(task_id=t.id)], "R")
        assert task.generation_task_id == t.id

    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            build_ranking_task([], "R")

    def test_persists(self, tmp_path: Path, make_generation: _MakeGeneration) -> None:
        path = tmp_path / "ranking_tasks.jsonl"
        task = build_ranking_task([make_generation()], "R", tasks_path=path)
        assert read_ranking_task_file(path) == [task]

    def test_groups_by_task(self, make_task: _MakeTask, make_generation: _MakeGeneration) -> None:
        t1, t2 = make_task(), make_task()
        r1, r2, r3 = (
            make_generation(task_id=t1.id),
            make_generation(task_id=t1.id),
            make_generation(task_id=t2.id),
        )
        tasks = build_ranking_tasks([t1, t2], [r1, r2, r3], "R")
        assert len(tasks) == 2
        assert set(tasks[0].generations.values()) == {r1.id, r2.id}
        assert set(tasks[1].generations.values()) == {r3.id}

    def test_skips_tasks_with_no_results(self, make_task: _MakeTask, make_generation: _MakeGeneration) -> None:
        t1, t2 = make_task(), make_task()
        tasks = build_ranking_tasks([t1, t2], [make_generation(task_id=t1.id)], "R")
        assert len(tasks) == 1

    def test_resume_reuses_persisted_tasks_frozen(
        self, tmp_path: Path, make_task: _MakeTask, make_generation: _MakeGeneration
    ) -> None:
        t = make_task()
        path = tmp_path / "ranking_tasks.jsonl"
        first = build_ranking_tasks([t], [make_generation(task_id=t.id)], "R", tasks_path=path, random_seed=1)
        # Rebuild with a *different* candidate set + seed: the persisted task is reused as-is, not rebuilt.
        second = build_ranking_tasks(
            [t],
            [make_generation(task_id=t.id), make_generation(task_id=t.id)],
            "R",
            tasks_path=path,
            random_seed=2,
        )
        assert second == first
        assert read_ranking_task_file(path) == first  # nothing new appended


# --------------------------------------------------------------------------- #
# rank_one / rank_all                                                           #
# --------------------------------------------------------------------------- #


class TestRankOne:
    async def test_success_maps_aliases_to_ids(
        self, make_generation: _MakeGeneration, make_ranking_task: _MakeRankingTask
    ) -> None:
        a, b = make_generation(), make_generation()
        task = make_ranking_task(generations={"A": a.id, "B": b.id})
        result = await rank_one(task, _RankingClient(ranking=["B", "A"]), {a.id: a, b.id: b})
        assert isinstance(result, RankingResult)
        assert result.author == "judge"
        assert result.raw_model_ranking == ["B", "A"]
        assert result.ranking == [b.id, a.id]

    async def test_malformed_ranking_is_failure(
        self, make_generation: _MakeGeneration, make_ranking_task: _MakeRankingTask
    ) -> None:
        a, b = make_generation(), make_generation()
        task = make_ranking_task(generations={"A": a.id, "B": b.id})
        result = await rank_one(task, _RankingClient(ranking=["A"]), {a.id: a, b.id: b})  # missing B
        assert isinstance(result, RankingFailure)
        assert "Missing aliases" in result.message

    async def test_client_error_is_failure(
        self, make_generation: _MakeGeneration, make_ranking_task: _MakeRankingTask
    ) -> None:
        a = make_generation()
        task = make_ranking_task(generations={"A": a.id})
        result = await rank_one(task, _RankingClient(ranking=["A"], fail=True), {a.id: a})
        assert isinstance(result, RankingFailure)
        assert result.error_type == "RuntimeError"

    async def test_persists(
        self, tmp_path: Path, make_generation: _MakeGeneration, make_ranking_task: _MakeRankingTask
    ) -> None:
        a = make_generation()
        task = make_ranking_task(generations={"A": a.id})
        path = tmp_path / "rankings.jsonl"
        result = await rank_one(task, _RankingClient(ranking=["A"]), {a.id: a}, results_path=path)
        assert read_ranking_result_file(path) == [result]

    async def test_persists_failure(
        self, tmp_path: Path, make_generation: _MakeGeneration, make_ranking_task: _MakeRankingTask
    ) -> None:
        a = make_generation()
        task = make_ranking_task(generations={"A": a.id})
        path = tmp_path / "ranking_failures.jsonl"
        result = await rank_one(task, _RankingClient(fail=True), {a.id: a}, failures_path=path)
        assert isinstance(result, RankingFailure)
        assert read_ranking_failure_file(path) == [result]


class TestRankAll:
    async def test_runs_every_judge(
        self, make_generation: _MakeGeneration, make_ranking_task: _MakeRankingTask
    ) -> None:
        a, b = make_generation(), make_generation()
        task = make_ranking_task(generations={"A": a.id, "B": b.id})
        results, failures = await rank_all([task], [a, b], [_RankingClient("j", ranking=["A", "B"])])
        assert not failures
        assert results[0].ranking == [a.id, b.id]

    async def test_resume_skips_done_and_returns_complete_set(
        self, tmp_path: Path, make_generation: _MakeGeneration, make_ranking_task: _MakeRankingTask
    ) -> None:
        a = make_generation()
        task = make_ranking_task(generations={"A": a.id})
        path = tmp_path / "rankings.jsonl"
        run1, _ = await rank_all([task], [a], [_RankingClient("j", ranking=["A"])], results_path=path)
        assert [r.author for r in run1] == ["j"]
        # Rerun: 'j' would fail if re-invoked, but it's loaded from disk and skipped.
        run2, failures2 = await rank_all([task], [a], [_RankingClient("j", fail=True)], results_path=path)
        assert failures2 == []
        assert run2 == run1

    async def test_failing_judge_becomes_a_failure(
        self, make_generation: _MakeGeneration, make_ranking_task: _MakeRankingTask
    ) -> None:
        a = make_generation()
        task = make_ranking_task(generations={"A": a.id})
        results, failures = await rank_all([task], [a], [_RankingClient("j", fail=True)])
        assert results == []
        assert [f.author for f in failures] == ["j"]
