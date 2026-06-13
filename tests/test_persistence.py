"""Tests for directory-based JSONL persistence."""

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tournament_eval.models import (
    GenerationFailure,
    GenerationResult,
    GenerationTask,
    RankingFailure,
    RankingResult,
    RankingTask,
)
from tournament_eval.orchestration import (
    build_ranking_tasks,
    generate_all,
    rank_all,
)
from tournament_eval.persistence import (
    GENERATION_FAILURE_DIR,
    GENERATION_RESULT_DIR,
    GENERATION_TASK_FILE,
    RANKING_FAILURE_DIR,
    RANKING_RESULT_DIR,
    RANKING_TASK_FILE,
    _append_line,
    _sanitize_author,
    append_generation_failure,
    append_generation_result,
    append_generation_task,
    append_ranking_failure,
    append_ranking_result,
    append_ranking_task,
    read_generation_failure_file,
    read_generation_result_file,
    read_generation_task_file,
    read_ranking_failure_file,
    read_ranking_result_file,
    read_ranking_task_file,
)
from tournament_eval.ranking import DefaultRankingTemplate

if TYPE_CHECKING:
    from tests.conftest import MockLLMClient

_MakeTask = Callable[..., GenerationTask]
_MakeClient = Callable[..., "MockLLMClient"]


def _result(task_id: uuid.UUID, author: str, output: str = "o") -> GenerationResult:
    return GenerationResult(
        id=uuid.uuid4(),
        task_id=task_id,
        generation_prompt="p",
        raw_response="r",
        output=output,
        author=author,
    )


def _ranking_result(author: str, reasoning: str | None) -> RankingResult:
    return RankingResult(
        id=uuid.uuid4(),
        ranking_task_id=uuid.uuid4(),
        ranking_prompt="rp",
        author=author,
        raw_model_ranking=["A", "B"],
        ranking=[uuid.uuid4(), uuid.uuid4()],
        reasoning=reasoning,
        raw_response="{}",
        metadata={"cost": 0.1},
    )


class TestSanitizeAuthor:
    def test_unsafe_chars(self) -> None:
        assert _sanitize_author("meta-llama/Llama-3:8b") == "meta-llama_Llama-3_8b"

    def test_empty_falls_back(self) -> None:
        assert _sanitize_author("") == "_"


class TestRoundTrip:
    def test_generation_task(self, tmp_path: Path, make_task: _MakeTask) -> None:
        task = make_task("translate this")
        append_generation_task(tmp_path, task)
        assert read_generation_task_file(tmp_path / GENERATION_TASK_FILE) == [task]

    def test_generation_result_full_fidelity(self, tmp_path: Path) -> None:
        result = GenerationResult(
            id=uuid.uuid4(),
            task_id=uuid.uuid4(),
            generation_prompt="p",
            raw_response="r",
            output="o",
            author="org/model:tag",  # unsafe filename chars
            metadata={"latency": 1.5, "tokens": 12},
        )
        append_generation_result(tmp_path, result)
        # written under a sanitized per-model filename
        path = tmp_path / GENERATION_RESULT_DIR / "org_model_tag.jsonl"
        assert path.exists()
        assert read_generation_result_file(path) == [result]

    def test_generation_failure(self, tmp_path: Path) -> None:
        failure = GenerationFailure(
            task_id=uuid.uuid4(), author="m", error_type="RuntimeError", message="boom"
        )
        append_generation_failure(tmp_path, failure)
        path = tmp_path / GENERATION_FAILURE_DIR / "m.jsonl"
        assert read_generation_failure_file(path) == [failure]

    def test_ranking_task(self, tmp_path: Path) -> None:
        rt = RankingTask(
            id=uuid.uuid4(),
            ranking_prompt="rank",
            generations={"A": uuid.uuid4(), "B": uuid.uuid4()},
        )
        append_ranking_task(tmp_path, rt)
        assert read_ranking_task_file(tmp_path / RANKING_TASK_FILE) == [rt]

    @pytest.mark.parametrize("reasoning", ["because", None])
    def test_ranking_result(self, tmp_path: Path, reasoning: str | None) -> None:
        result = _ranking_result("judge", reasoning)
        append_ranking_result(tmp_path, result)
        path = tmp_path / RANKING_RESULT_DIR / "judge.jsonl"
        assert read_ranking_result_file(path) == [result]

    def test_ranking_failure(self, tmp_path: Path) -> None:
        failure = RankingFailure(
            ranking_task_id=uuid.uuid4(),
            author="judge",
            error_type="ValueError",
            message="bad",
        )
        append_ranking_failure(tmp_path, failure)
        path = tmp_path / RANKING_FAILURE_DIR / "judge.jsonl"
        assert read_ranking_failure_file(path) == [failure]


class TestWriteReadBehaviour:
    def test_appends_accumulate(self, tmp_path: Path) -> None:
        task_id = uuid.uuid4()
        g1, g2 = _result(task_id, "m", "o1"), _result(task_id, "m", "o2")
        append_generation_result(tmp_path, g1)
        append_generation_result(tmp_path, g2)
        path = tmp_path / GENERATION_RESULT_DIR / "m.jsonl"
        assert read_generation_result_file(path) == [g1, g2]

    def test_append_ranking_tasks_accumulate(self, tmp_path: Path) -> None:
        rt1 = RankingTask(
            id=uuid.uuid4(), ranking_prompt="r", generations={"A": uuid.uuid4()}
        )
        rt2 = RankingTask(
            id=uuid.uuid4(), ranking_prompt="r", generations={"A": uuid.uuid4()}
        )
        append_ranking_task(tmp_path, rt1)
        append_ranking_task(tmp_path, rt2)
        assert read_ranking_task_file(tmp_path / RANKING_TASK_FILE) == [rt1, rt2]

    def test_reader_skips_blank_lines(self, tmp_path: Path) -> None:
        result = _result(uuid.uuid4(), "m")
        append_generation_result(tmp_path, result)
        path = tmp_path / GENERATION_RESULT_DIR / "m.jsonl"
        path.write_bytes(path.read_bytes() + b"\n   \n")  # stray blank lines
        assert read_generation_result_file(path) == [result]

    def test_readers_return_empty_for_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.jsonl"
        assert read_generation_task_file(missing) == []
        assert read_generation_result_file(missing) == []
        assert read_generation_failure_file(missing) == []
        assert read_ranking_task_file(missing) == []
        assert read_ranking_result_file(missing) == []
        assert read_ranking_failure_file(missing) == []

    def test_append_line_rejects_directory(self, tmp_path: Path) -> None:
        with pytest.raises(IsADirectoryError):
            _append_line(tmp_path, {"x": 1})


class TestPipelinePersistence:
    async def test_generate_all_streams_per_model(
        self, tmp_path: Path, make_client: _MakeClient, make_task: _MakeTask
    ) -> None:
        task = make_task("p1")
        good = make_client("a", generate_responses={"p1": "out-a"})
        bad = make_client("b", generate_responses={}, fail_on={"p1"})

        results, failures = await generate_all([task], [good, bad], output=tmp_path)

        assert {r.author for r in results} == {"a"}
        assert {f.author for f in failures} == {"b"}
        assert (
            read_generation_result_file(tmp_path / GENERATION_RESULT_DIR / "a.jsonl")
            == results
        )
        assert (
            read_generation_failure_file(tmp_path / GENERATION_FAILURE_DIR / "b.jsonl")
            == failures
        )

    async def test_resume_via_skip(
        self, tmp_path: Path, make_client: _MakeClient, make_task: _MakeTask
    ) -> None:
        task = make_task("p1")
        a = make_client("a", generate_responses={"p1": "out-a"})
        await generate_all([task], [a], output=tmp_path)

        # derive the done set from what was persisted (no helper needed)
        persisted = read_generation_result_file(
            tmp_path / GENERATION_RESULT_DIR / "a.jsonl"
        )
        done = {(r.task_id, r.author) for r in persisted}

        a_boom = make_client("a", generate_responses={}, fail_on={"p1"})  # would raise
        b = make_client("b", generate_responses={"p1": "out-b"})
        results, _ = await generate_all([task], [a_boom, b], output=tmp_path, skip=done)
        assert {r.author for r in results} == {"b"}

    async def test_build_ranking_tasks_writes_and_rank_all_streams(
        self, tmp_path: Path, make_client: _MakeClient, make_task: _MakeTask
    ) -> None:
        task = make_task("p1")
        gens = [_result(task.id, "a"), _result(task.id, "b")]
        ranking_tasks = build_ranking_tasks([task], gens, "rank", output=tmp_path)
        assert read_ranking_task_file(tmp_path / RANKING_TASK_FILE) == ranking_tasks

        lookup = {g.id: g for g in gens}
        candidates = {a: lookup[gid] for a, gid in ranking_tasks[0].generations.items()}
        prompt = DefaultRankingTemplate().render(ranking_tasks[0], candidates)
        aliases = list(ranking_tasks[0].generations.keys())
        judge = make_client(
            "judge", structured_responses={prompt: {"ranking": aliases}}
        )

        results, _ = await rank_all(ranking_tasks, gens, [judge], output=tmp_path)
        assert (
            read_ranking_result_file(tmp_path / RANKING_RESULT_DIR / "judge.jsonl")
            == results
        )

    async def test_rank_all_streams_failures(
        self, tmp_path: Path, make_client: _MakeClient, make_task: _MakeTask
    ) -> None:
        task = make_task("p1")
        gens = [_result(task.id, "a")]
        ranking_tasks = build_ranking_tasks([task], gens, "rank")
        lookup = {g.id: g for g in gens}
        candidates = {a: lookup[gid] for a, gid in ranking_tasks[0].generations.items()}
        prompt = DefaultRankingTemplate().render(ranking_tasks[0], candidates)
        judge = make_client("judge", structured_responses={}, fail_on={prompt})

        _, failures = await rank_all(ranking_tasks, gens, [judge], output=tmp_path)
        assert (
            read_ranking_failure_file(tmp_path / RANKING_FAILURE_DIR / "judge.jsonl")
            == failures
        )
