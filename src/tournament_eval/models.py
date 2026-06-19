"""The pipeline's data model: the tasks, results, and failures it passes around.

Plain frozen dataclasses with no behaviour beyond ``from_json`` — the inverse of the
orjson serialization in :mod:`tournament_eval.persistence`.  Each is paired with a
``*Dict`` TypedDict describing its on-disk JSON shape.  The ranking *strategy* lives
in :mod:`tournament_eval.ranking`; this module is data only.
"""

import dataclasses
import uuid
from typing import TypedDict


class GenerationTaskDict(TypedDict):
    id: str
    generation_prompt: str


@dataclasses.dataclass(frozen=True, slots=True)
class GenerationTask:
    """A user-created description of what to evaluate.

    Each GenerationTask will eventually be used to create a RankingTask once
    all participating models have generated outputs for it.
    """

    id: uuid.UUID
    """Unique identifier for this task."""
    generation_prompt: str
    """The creative task to be given to each model."""

    @classmethod
    def from_json(cls, data: GenerationTaskDict) -> GenerationTask:
        """Rebuild from a JSON-decoded dict (e.g. a line read from a run file)."""
        return cls(
            id=uuid.UUID(data["id"]),
            generation_prompt=data["generation_prompt"],
        )


class GenerationFailureDict(TypedDict):
    generation_task_id: str
    author: str
    error_type: str
    message: str


@dataclasses.dataclass(frozen=True, slots=True)
class GenerationFailure:
    """A single model's failure to generate a :class:`GenerationResult`."""

    generation_task_id: uuid.UUID
    """The :class:`GenerationTask` this failure is for."""
    author: str
    """The model identifier that failed to generate this result."""
    error_type: str
    """The type of error that occurred."""
    message: str
    """The error message."""

    @classmethod
    def from_json(cls, data: GenerationFailureDict) -> GenerationFailure:
        """Rebuild from a JSON-decoded dict (e.g. a line read from a run file)."""
        return cls(
            generation_task_id=uuid.UUID(data["generation_task_id"]),
            author=data["author"],
            error_type=data["error_type"],
            message=data["message"],
        )


class GenerationResultDict(TypedDict):
    id: str
    generation_task_id: str
    generation_prompt: str
    output: str
    reasoning: str | None
    author: str
    metadata: dict[str, object]


@dataclasses.dataclass(frozen=True, slots=True)
class GenerationResult:
    """A single model's output based on a :class:`GenerationTask`."""

    id: uuid.UUID
    """Unique identifier for this generation."""
    generation_task_id: uuid.UUID
    """The :class:`GenerationTask` that was used to produce this generation."""
    generation_prompt: str
    """The exact prompt string sent to the model, verbatim."""
    output: str
    """The model's answer text, as returned (no cleaning applied)."""
    reasoning: str | None
    """The model's reasoning trace, if the provider surfaced one; ``None`` otherwise."""
    author: str
    """The model identifier that produced this output (e.g. ``"gpt-4o"``)."""
    metadata: dict[str, object] = dataclasses.field(default_factory=dict)
    """Usage/cost data from the agent run (token counts, request count, ...).
    Populated by the orchestration from pydantic-ai's ``RunUsage``; a kitchen-sink
    dict so it can grow (cost via ``genai-prices``, latency, ...) without changing
    the record shape."""

    @classmethod
    def from_json(cls, data: GenerationResultDict) -> GenerationResult:
        """Rebuild from a JSON-decoded dict (e.g. a line read from a run file)."""
        return cls(
            id=uuid.UUID(data["id"]),
            generation_task_id=uuid.UUID(data["generation_task_id"]),
            generation_prompt=data["generation_prompt"],
            output=data["output"],
            reasoning=data["reasoning"],
            author=data["author"],
            metadata=data["metadata"],
        )


class RankingTaskDict(TypedDict):
    id: str
    generation_task_id: str
    ranking_prompt: str
    generations: dict[str, str]


@dataclasses.dataclass(frozen=True, slots=True)
class RankingTask:
    """A description of a ranking task.

    Each RankingTask is used by every ranking model to produce one
    :class:`RankingResult`.
    """

    id: uuid.UUID
    """Unique identifier for this ranking task."""
    generation_task_id: uuid.UUID
    """The :class:`GenerationTask` whose outputs this ranks.  Records provenance
    (so a ranking traces straight back to its source task) and is the key used to
    reuse a persisted ranking task on resume instead of rebuilding it."""
    ranking_prompt: str
    """The ranking instructions to be used when ranking the candidates."""
    generations: dict[str, uuid.UUID]
    """Mapping of anonymized aliases (e.g. ``"A"``, ``"B"``) to
    :class:`GenerationResult` IDs."""

    @classmethod
    def from_json(cls, data: RankingTaskDict) -> RankingTask:
        """Rebuild from a JSON-decoded dict (e.g. a line read from a run file)."""
        return cls(
            id=uuid.UUID(data["id"]),
            generation_task_id=uuid.UUID(data["generation_task_id"]),
            ranking_prompt=data["ranking_prompt"],
            generations={alias: uuid.UUID(gid) for alias, gid in data["generations"].items()},
        )


class RankingFailureDict(TypedDict):
    ranking_task_id: str
    generation_task_id: str
    author: str
    error_type: str
    message: str
    ranking_prompt: str


@dataclasses.dataclass(frozen=True, slots=True)
class RankingFailure:
    """A single model's failure to rank a :class:`RankingResult`."""

    ranking_task_id: uuid.UUID
    """The :class:`RankingTask` this failure is for."""
    generation_task_id: uuid.UUID
    """The :class:`GenerationTask` whose outputs were being ranked.  Records
    provenance so a failure traces back to its source task without a join through
    the :class:`RankingTask`."""
    author: str
    """The model identifier that failed to rank this result."""
    error_type: str
    """The type of error that occurred."""
    message: str
    """The error message."""
    ranking_prompt: str
    """The exact ranking prompt sent to the model, verbatim.  Carries the rendered
    template output (candidates, rubric, ...), so a failure is debuggable without
    rebuilding the :class:`RankingTask` and its candidates — useful for auditing
    template/anonymization issues."""

    @classmethod
    def from_json(cls, data: RankingFailureDict) -> RankingFailure:
        """Rebuild from a JSON-decoded dict (e.g. a line read from a run file)."""
        return cls(
            ranking_task_id=uuid.UUID(data["ranking_task_id"]),
            generation_task_id=uuid.UUID(data["generation_task_id"]),
            author=data["author"],
            error_type=data["error_type"],
            message=data["message"],
            ranking_prompt=data["ranking_prompt"],
        )


class RankingResultDict(TypedDict):
    id: str
    ranking_task_id: str
    generation_task_id: str
    ranking_prompt: str
    author: str
    raw_model_ranking: list[str]
    ranking: list[str]
    ranking_reasoning: str | None
    reasoning: str | None
    raw_response: str
    metadata: dict[str, object]


@dataclasses.dataclass(frozen=True, slots=True)
class RankingResult:
    """A single model's ranking of candidates, based on a :class:`RankingTask`."""

    id: uuid.UUID
    """Unique identifier for this ranking."""
    ranking_task_id: uuid.UUID
    """The :class:`RankingTask` that was used to produce this ranking."""
    generation_task_id: uuid.UUID
    """The :class:`GenerationTask` whose outputs were ranked.  Records provenance
    so a ranking traces straight back to its source task without a join through
    the :class:`RankingTask`."""
    ranking_prompt: str
    """The exact ranking prompt sent to the model, verbatim."""
    author: str
    """The model identifier that performed this ranking (e.g. ``"gpt-4o"``)."""
    raw_model_ranking: list[str]
    """Raw aliases in ranked order from best to worst, e.g. ``["B", "A", "C"]``.
    Produced by the model."""
    ranking: list[uuid.UUID]
    """Cleaned-up ranking in ranked order from best to worst.
    Parsed from ``raw_model_ranking`` by matching against the generation IDs
    in the :class:`RankingTask`."""
    ranking_reasoning: str | None
    """The ranking justification the model wrote in its structured response's
    ``reasoning`` field, if any.  This is *output*, not the model's thinking
    trace — see :attr:`reasoning` for that."""
    reasoning: str | None
    """The model's reasoning/thinking trace from the ranking run, if the provider
    surfaced one (mirrors :attr:`GenerationResult.reasoning`).  ``None`` when the
    model produced no thinking parts."""
    raw_response: str
    """The complete structured JSON response returned by the LLM."""
    metadata: dict[str, object] = dataclasses.field(default_factory=dict)
    """Usage/cost data from the agent run (token counts, request count, ...).
    Populated by the orchestration from pydantic-ai's ``RunUsage``; a kitchen-sink
    dict so it can grow (cost via ``genai-prices``, latency, ...) without changing
    the record shape."""

    @classmethod
    def from_json(cls, data: RankingResultDict) -> RankingResult:
        """Rebuild from a JSON-decoded dict (e.g. a line read from a run file)."""
        return cls(
            id=uuid.UUID(data["id"]),
            ranking_task_id=uuid.UUID(data["ranking_task_id"]),
            generation_task_id=uuid.UUID(data["generation_task_id"]),
            ranking_prompt=data["ranking_prompt"],
            author=data["author"],
            raw_model_ranking=data["raw_model_ranking"],
            ranking=[uuid.UUID(gid) for gid in data["ranking"]],
            ranking_reasoning=data["ranking_reasoning"],
            reasoning=data["reasoning"],
            raw_response=data["raw_response"],
            metadata=data["metadata"],
        )
