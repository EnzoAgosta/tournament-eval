"""JSONL persistence: append records as they're produced, read them back later.

A run is a set of JSON Lines files whose paths *you* choose — typically one file
per stream::

    generation_tasks.jsonl     # your inputs, appended by you
    generations.jsonl          # GenerationResults, streamed as they land
    generation_failures.jsonl  # GenerationFailures
    ranking_tasks.jsonl        # RankingTasks, built once
    rankings.jsonl             # RankingResults
    ranking_failures.jsonl     # RankingFailures

There is no directory layout or filename convention here: nothing is sharded by
author, because each record already carries its ``author``/``task_id``, so a flat
file is fully reconstructable — group or filter on read.  Where the pipeline
writes is whatever path you pass to ``generate_all`` / ``rank_all`` /
``build_ranking_tasks``.

Writing is type-agnostic: orjson encodes :class:`uuid.UUID` and dataclasses
natively, so one :func:`append_record` serialises every record type.  Reading is
per-type (it rebuilds the dataclass: ``str`` → ``UUID``, dict → dataclass), so
each stream has its own reader.

These readers are also what powers resume: :func:`~tournament_eval.orchestration.generate_all`
/ :func:`~tournament_eval.orchestration.rank_all` read their results file back to
skip the ``(task, author)`` pairs already done.

This module depends only on :mod:`tournament_eval.models` — the pipeline calls
*into* it, never the other way around.

A note on safety: :func:`append_record` does one synchronous ``open → write →
close`` with no ``await`` in between.  Under a single asyncio event loop that
makes each line atomic with respect to other in-flight calls — do not "optimise"
the write into a threaded or async one, or concurrent appends could interleave.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import orjson

from tournament_eval.models import (
    GenerationFailure,
    GenerationResult,
    GenerationTask,
    RankingFailure,
    RankingResult,
    RankingTask,
)


def append_record(path: str | Path, record: object) -> None:
    """Append one record as a JSON line to ``path``, creating parent dirs.

    Type-agnostic: orjson serialises any dataclass/UUID, so the same function
    persists results, failures, and tasks alike.  Pair it on read with the typed
    reader for whatever you wrote (e.g. :func:`read_generation_result_file`).
    """
    path = Path(path)
    if path.is_dir():
        raise IsADirectoryError(f"{path} is a directory, not a file")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        f.write(orjson.dumps(record))
        f.write(b"\n")


def _read_file[T](path: str | Path, from_json: Callable[[Any], T]) -> list[T]:
    """Read one JSONL file, rebuilding each line via ``from_json``.

    A missing file reads as empty; blank lines are skipped.
    """
    path = Path(path)
    if not path.exists():
        return []
    with path.open("rb") as f:
        return [from_json(orjson.loads(line)) for line in f if line.strip()]


def read_generation_task_file(path: str | Path) -> list[GenerationTask]:
    """Read generation tasks from a single file."""
    return _read_file(path, GenerationTask.from_json)


def read_generation_result_file(path: str | Path) -> list[GenerationResult]:
    """Read generation results from a single file."""
    return _read_file(path, GenerationResult.from_json)


def read_generation_failure_file(path: str | Path) -> list[GenerationFailure]:
    """Read generation failures from a single file."""
    return _read_file(path, GenerationFailure.from_json)


def read_ranking_task_file(path: str | Path) -> list[RankingTask]:
    """Read ranking tasks from a single file."""
    return _read_file(path, RankingTask.from_json)


def read_ranking_result_file(path: str | Path) -> list[RankingResult]:
    """Read ranking results from a single file."""
    return _read_file(path, RankingResult.from_json)


def read_ranking_failure_file(path: str | Path) -> list[RankingFailure]:
    """Read ranking failures from a single file."""
    return _read_file(path, RankingFailure.from_json)
