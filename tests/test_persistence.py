"""Tests for directory-based JSONL persistence."""

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


class TestSanitizeAuthor:
    def test_unsafe_chars_replaced(self) -> None:
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
            output="o",
            reasoning="thinking out loud",
            author="org/model:tag",  # unsafe filename chars
            metadata={"latency": 1.5, "tokens": 12},
        )
        append_generation_result(tmp_path, result)
        path = tmp_path / GENERATION_RESULT_DIR / "org_model_tag.jsonl"  # sanitized stem
        assert path.exists()
        assert read_generation_result_file(path) == [result]

    def test_generation_failure(self, tmp_path: Path) -> None:
        failure = GenerationFailure(task_id=uuid.uuid4(), author="m", error_type="RuntimeError", message="boom")
        append_generation_failure(tmp_path, failure)
        assert read_generation_failure_file(tmp_path / GENERATION_FAILURE_DIR / "m.jsonl") == [failure]

    def test_ranking_task(self, tmp_path: Path) -> None:
        task = RankingTask(id=uuid.uuid4(), ranking_prompt="rank", generations={"A": uuid.uuid4()})
        append_ranking_task(tmp_path, task)
        assert read_ranking_task_file(tmp_path / RANKING_TASK_FILE) == [task]

    @pytest.mark.parametrize("reasoning", ["because", None])
    def test_ranking_result(self, tmp_path: Path, reasoning: str | None) -> None:
        result = _ranking_result("judge", reasoning)
        append_ranking_result(tmp_path, result)
        assert read_ranking_result_file(tmp_path / RANKING_RESULT_DIR / "judge.jsonl") == [result]

    def test_ranking_failure(self, tmp_path: Path) -> None:
        failure = RankingFailure(ranking_task_id=uuid.uuid4(), author="judge", error_type="ValueError", message="bad")
        append_ranking_failure(tmp_path, failure)
        assert read_ranking_failure_file(tmp_path / RANKING_FAILURE_DIR / "judge.jsonl") == [failure]


class TestReadBehaviour:
    def test_appends_accumulate(self, tmp_path: Path) -> None:
        task_id = uuid.uuid4()
        g1, g2 = _result(task_id, "m", "o1"), _result(task_id, "m", "o2")
        append_generation_result(tmp_path, g1)
        append_generation_result(tmp_path, g2)
        assert read_generation_result_file(tmp_path / GENERATION_RESULT_DIR / "m.jsonl") == [g1, g2]

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
