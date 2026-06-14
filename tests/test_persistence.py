"""Tests for flat-file JSONL persistence: one type-agnostic writer, typed readers."""

import uuid
from pathlib import Path

import pytest

from tests.conftest import _MakeTask
from tournament_eval.models import (
    GenerationFailure,
    GenerationResult,
    RankingFailure,
    RankingResult,
    RankingTask,
)
from tournament_eval.persistence import (
    append_record,
    read_generation_failure_file,
    read_generation_result_file,
    read_generation_task_file,
    read_ranking_failure_file,
    read_ranking_result_file,
    read_ranking_task_file,
)


def _result(task_id: uuid.UUID, author: str, output: str = "o") -> GenerationResult:
    return GenerationResult(
        id=uuid.uuid4(),
        task_id=task_id,
        generation_prompt="p",
        output=output,
        reasoning=None,
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


class TestRoundTrip:
    def test_generation_task(self, tmp_path: Path, make_task: _MakeTask) -> None:
        task = make_task("translate this")
        path = tmp_path / "generation_tasks.jsonl"
        append_record(path, task)
        assert read_generation_task_file(path) == [task]

    def test_generation_result_full_fidelity(self, tmp_path: Path) -> None:
        result = GenerationResult(
            id=uuid.uuid4(),
            task_id=uuid.uuid4(),
            generation_prompt="p",
            output="o",
            reasoning="thinking out loud",
            author="org/model:tag",  # unsafe chars no longer matter — author is a field, not a filename
            metadata={"latency": 1.5, "tokens": 12},
        )
        path = tmp_path / "generations.jsonl"
        append_record(path, result)
        assert read_generation_result_file(path) == [result]

    def test_generation_failure(self, tmp_path: Path) -> None:
        failure = GenerationFailure(task_id=uuid.uuid4(), author="m", error_type="RuntimeError", message="boom")
        path = tmp_path / "generation_failures.jsonl"
        append_record(path, failure)
        assert read_generation_failure_file(path) == [failure]

    def test_ranking_task(self, tmp_path: Path) -> None:
        task = RankingTask(
            id=uuid.uuid4(),
            generation_task_id=uuid.uuid4(),
            ranking_prompt="rank",
            generations={"A": uuid.uuid4()},
        )
        path = tmp_path / "ranking_tasks.jsonl"
        append_record(path, task)
        assert read_ranking_task_file(path) == [task]

    @pytest.mark.parametrize("reasoning", ["because", None])
    def test_ranking_result(self, tmp_path: Path, reasoning: str | None) -> None:
        result = _ranking_result("judge", reasoning)
        path = tmp_path / "rankings.jsonl"
        append_record(path, result)
        assert read_ranking_result_file(path) == [result]

    def test_ranking_failure(self, tmp_path: Path) -> None:
        failure = RankingFailure(ranking_task_id=uuid.uuid4(), author="judge", error_type="ValueError", message="bad")
        path = tmp_path / "ranking_failures.jsonl"
        append_record(path, failure)
        assert read_ranking_failure_file(path) == [failure]


class TestWriterBehaviour:
    def test_appends_accumulate(self, tmp_path: Path) -> None:
        task_id = uuid.uuid4()
        g1, g2 = _result(task_id, "m", "o1"), _result(task_id, "m", "o2")
        path = tmp_path / "generations.jsonl"
        append_record(path, g1)
        append_record(path, g2)
        assert read_generation_result_file(path) == [g1, g2]

    def test_creates_missing_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deeper" / "generations.jsonl"
        append_record(path, _result(uuid.uuid4(), "m"))
        assert path.exists()

    def test_rejects_directory(self, tmp_path: Path) -> None:
        with pytest.raises(IsADirectoryError):
            append_record(tmp_path, {"x": 1})


class TestReadBehaviour:
    def test_reader_skips_blank_lines(self, tmp_path: Path) -> None:
        result = _result(uuid.uuid4(), "m")
        path = tmp_path / "generations.jsonl"
        append_record(path, result)
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
