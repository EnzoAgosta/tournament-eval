from pathlib import Path

import pytest

from tests.conftest import (
    GenerationFailureFactory,
    GenerationResultFactory,
    GenerationTaskFactory,
    RankingFailureFactory,
    RankingResultFactory,
    RankingTaskFactory,
)
from tournament_eval.persistence import (
    _read_file,
    append_record,
    read_generation_failure_file,
    read_generation_result_file,
    read_generation_task_file,
    read_ranking_failure_file,
    read_ranking_result_file,
    read_ranking_task_file,
)


def test_append_record_fails_on_directory(tmp_path: Path) -> None:
    path = tmp_path
    path.mkdir(exist_ok=True)
    with pytest.raises(IsADirectoryError):
        append_record(path, {"x": 1})


def test_append_record_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "generations.jsonl"
    append_record(path, {"x": 1})
    assert path.exists()


def test_append_record_does_not_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    append_record(path, {"x": 1})
    append_record(path, {"x": 2})
    assert (tmp_path / "generations.jsonl").read_text() == '{"x":1}\n{"x":2}\n'


def test_read_file_returns_empty_list_for_missing_file() -> None:
    assert _read_file("nope.jsonl", lambda x: x) == []


def test_read_file_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    path.write_text("1\n\n\n\n2\n\n\n3\n")
    assert _read_file(path, lambda x: x) == [1, 2, 3]


def test_read_file_converts_json(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    path.write_text('{"x": 1}\n{"y": 2}\n')
    assert _read_file(path, lambda x: x) == [{"x": 1}, {"y": 2}]


def test_read_generation_task_file_returns_list_of_generation_tasks(
    tmp_path: Path, make_generation_task: GenerationTaskFactory
) -> None:
    path = tmp_path / "generation_tasks.jsonl"
    task1 = make_generation_task()
    task2 = make_generation_task()
    append_record(path, task1)
    append_record(path, task2)
    assert read_generation_task_file(path) == [task1, task2]


def test_read_generation_result_file_returns_list_of_generation_results(
    tmp_path: Path, make_generation_result: GenerationResultFactory
) -> None:
    path = tmp_path / "generations.jsonl"
    result1 = make_generation_result()
    result2 = make_generation_result()
    append_record(path, result1)
    append_record(path, result2)
    assert read_generation_result_file(path) == [result1, result2]


def test_read_generation_failure_file_returns_list_of_generation_failures(
    tmp_path: Path, make_generation_failure: GenerationFailureFactory
) -> None:
    path = tmp_path / "generation_failures.jsonl"
    failure1 = make_generation_failure()
    failure2 = make_generation_failure()
    append_record(path, failure1)
    append_record(path, failure2)
    assert read_generation_failure_file(path) == [failure1, failure2]


def test_read_generation_failure_file_returns_list_of_ranking_task(
    tmp_path: Path, make_ranking_task: RankingTaskFactory
) -> None:
    path = tmp_path / "generation_failures.jsonl"
    task1 = make_ranking_task()
    task2 = make_ranking_task()
    append_record(path, task1)
    append_record(path, task2)
    assert read_ranking_task_file(path) == [task1, task2]


def test_read_ranking_task_file_returns_list_of_ranking_tasks(
    tmp_path: Path, make_ranking_task: RankingTaskFactory
) -> None:
    path = tmp_path / "ranking_tasks.jsonl"
    task1 = make_ranking_task()
    task2 = make_ranking_task()
    append_record(path, task1)
    append_record(path, task2)
    assert read_ranking_task_file(path) == [task1, task2]


def test_read_ranking_failure_file_returns_list_of_ranking_failures(
    tmp_path: Path, make_ranking_failure: RankingFailureFactory
) -> None:
    path = tmp_path / "ranking_failures.jsonl"
    failure1 = make_ranking_failure()
    failure2 = make_ranking_failure()
    append_record(path, failure1)
    append_record(path, failure2)
    assert read_ranking_failure_file(path) == [failure1, failure2]


def test_read_ranking_result_file_returns_list_of_ranking_results(
    tmp_path: Path, make_ranking_result: RankingResultFactory
) -> None:
    path = tmp_path / "ranking_results.jsonl"
    result1 = make_ranking_result()
    result2 = make_ranking_result()
    append_record(path, result1)
    append_record(path, result2)
    assert read_ranking_result_file(path) == [result1, result2]


def test_append_record_raises_on_non_serialisable(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    with pytest.raises(TypeError, match="not JSON serialisable"):
        append_record(path, object())
