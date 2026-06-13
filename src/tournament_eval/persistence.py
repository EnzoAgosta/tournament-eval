"""JSONL persistence: dump results as they're produced, read them back later.

A run is a *directory* of JSON Lines files, split by record type and (for the
per-model streams) by model::

    <dir>/
      generation_tasks.jsonl              # shared inputs, appended by the caller
      generations/<model>.jsonl           # appended as each result lands
      generation_failures/<model>.jsonl
      ranking_tasks.jsonl                 # shared inputs, appended as built
      rankings/<model>.jsonl              # per ranking model
      ranking_failures/<model>.jsonl

Each line is one entity. orjson encodes :class:`uuid.UUID` and dataclasses
natively, so writing is free; only the readers convert back (``str`` → ``UUID``,
dict → dataclass).

This module depends only on :mod:`tournament_eval.models` — the pipeline calls
*into* it (via the ``output=`` argument on ``generate_one`` / ``rank_one``),
never the other way around.

A note on safety: the append helpers do one synchronous ``open → write → close``
with no ``await`` in between.  Under a single asyncio event loop that makes each
line atomic with respect to other in-flight calls — do not "optimise" the write
into a threaded or async one, or concurrent appends could interleave.
"""

import re
from pathlib import Path

import orjson

from tournament_eval.models import (
    GenerationFailure,
    GenerationResult,
    GenerationTask,
    RankingFailure,
    RankingResult,
    RankingTask,
)

GENERATION_TASK_FILE = "generation_tasks.jsonl"
RANKING_TASK_FILE = "ranking_tasks.jsonl"
GENERATION_RESULT_DIR = "generations"
GENERATION_FAILURE_DIR = "generation_failures"
RANKING_RESULT_DIR = "rankings"
RANKING_FAILURE_DIR = "ranking_failures"

_UNSAFE = re.compile(r"[^A-Za-z0-9\._-]")


def _sanitize_author(author: str) -> str:
    """Make a model name safe to use as a filename stem.

    The true name always survives in each record's ``author`` field, so this is
    only a shard label — a sanitisation collision merely co-locates two models'
    records in one file, still separable by ``author``.
    """
    return _UNSAFE.sub("_", author) or "_"


def _append_line(path: Path, entity: object) -> None:
    # More of a sanity check than anything else
    # We always check before but in case the user decides
    # to call this directly, we'll check again.
    if path.is_dir():
        raise IsADirectoryError(f"{path} is a directory, not a file")
    with path.open("ab") as f:
        f.write(orjson.dumps(entity))
        f.write(b"\n")


def append_generation_task(output_dir: str | Path, task: GenerationTask) -> None:
    """Append one generation task to the run's shared generation-tasks file.

    Tasks are inputs the caller owns: build them from your dataset and append
    each one here.  (Re-running this against the same directory appends again —
    persist a given task set once.)
    """
    output_dir = Path(output_dir)
    path = output_dir / GENERATION_TASK_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    _append_line(path, task)


def append_generation_result(output_dir: str | Path, result: GenerationResult) -> None:
    """Append one generation result to its model's stream."""
    output_dir = Path(output_dir)
    path = (
        output_dir / GENERATION_RESULT_DIR / f"{_sanitize_author(result.author)}.jsonl"
    )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    _append_line(path, result)


def append_generation_failure(
    output_dir: str | Path, failure: GenerationFailure
) -> None:
    """Append one generation failure to its model's stream."""
    output_dir = Path(output_dir)
    path = (
        output_dir
        / GENERATION_FAILURE_DIR
        / f"{_sanitize_author(failure.author)}.jsonl"
    )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    _append_line(path, failure)


def append_ranking_task(output_dir: str | Path, task: RankingTask) -> None:
    """Append one ranking task to the run's shared ranking-tasks file.

    Ranking tasks carry a random alias assignment, so build them once and reuse
    the persisted set on resume (``read_ranking_task_file``) rather than
    rebuilding — a rebuild would re-shuffle and orphan everything already ranked.
    (Re-running a build against the same directory appends again.)
    """
    output_dir = Path(output_dir)
    path = output_dir / RANKING_TASK_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    _append_line(path, task)


def append_ranking_result(output_dir: str | Path, result: RankingResult) -> None:
    """Append one ranking result to its ranking model's stream."""
    output_dir = Path(output_dir)
    path = output_dir / RANKING_RESULT_DIR / f"{_sanitize_author(result.author)}.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    _append_line(path, result)


def append_ranking_failure(output_dir: str | Path, failure: RankingFailure) -> None:
    """Append one ranking failure to its ranking model's stream."""
    output_dir = Path(output_dir)
    path = (
        output_dir / RANKING_FAILURE_DIR / f"{_sanitize_author(failure.author)}.jsonl"
    )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    _append_line(path, failure)


def read_ranking_task_file(path: str | Path) -> list[RankingTask]:
    """Read ranking tasks from a single file."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("rb") as f:
        d = [orjson.loads(line) for line in f if line.strip()]
    return [RankingTask.from_json(task) for task in d]


def read_ranking_result_file(path: str | Path) -> list[RankingResult]:
    """Read ranking results from a single file."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("rb") as f:
        d = [orjson.loads(line) for line in f if line.strip()]
    return [RankingResult.from_json(result) for result in d]


def read_ranking_failure_file(path: str | Path) -> list[RankingFailure]:
    """Read ranking failures from a single file."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("rb") as f:
        d = [orjson.loads(line) for line in f if line.strip()]
    return [RankingFailure.from_json(failure) for failure in d]


def read_generation_result_file(path: str | Path) -> list[GenerationResult]:
    """Read generation results from a single file."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("rb") as f:
        d = [orjson.loads(line) for line in f if line.strip()]
    return [GenerationResult.from_json(result) for result in d]


def read_generation_failure_file(path: str | Path) -> list[GenerationFailure]:
    """Read generation failures from a single file."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("rb") as f:
        d = [orjson.loads(line) for line in f if line.strip()]
    return [GenerationFailure.from_json(failure) for failure in d]


def read_generation_task_file(path: str | Path) -> list[GenerationTask]:
    """Read generation tasks from a single file."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("rb") as f:
        d = [orjson.loads(line) for line in f if line.strip()]
    return [GenerationTask.from_json(task) for task in d]
