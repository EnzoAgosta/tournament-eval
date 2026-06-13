import abc
import dataclasses
import uuid
from collections.abc import Mapping
from typing import ClassVar, TypedDict


class LetterGenerator:
    """Simple generator for letters, Excel-style: A, B, C, ..., AA, AB, AC, ..."""

    def __init__(self) -> None:
        self.n = 0

    def get_next_letter(self) -> str:
        self.n += 1
        result = ""
        temp = self.n
        while temp > 0:
            temp, remainder = divmod(temp - 1, 26)
            result = chr(65 + remainder) + result
        return result


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
    task_id: str
    author: str
    error_type: str
    message: str


@dataclasses.dataclass(frozen=True, slots=True)
class GenerationFailure:
    """A single model's failure to generate a :class:`GenerationResult`."""

    task_id: uuid.UUID
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
            task_id=uuid.UUID(data["task_id"]),
            author=data["author"],
            error_type=data["error_type"],
            message=data["message"],
        )


class GenerationResultDict(TypedDict):
    id: str
    task_id: str
    generation_prompt: str
    raw_response: str
    output: str
    author: str
    metadata: dict[str, object]


@dataclasses.dataclass(frozen=True, slots=True)
class GenerationResult:
    """A single model's output based on a :class:`GenerationTask`."""

    id: uuid.UUID
    """Unique identifier for this generation."""
    task_id: uuid.UUID
    """The :class:`GenerationTask` that was used to produce this generation."""
    generation_prompt: str
    """The exact prompt string sent to the model, verbatim."""
    raw_response: str
    """The direct, unmodified response returned by the LLM."""
    output: str
    """The cleaned, ready-to-use text (e.g. stripped of markdown fences)."""
    author: str
    """The model identifier that produced this output (e.g. ``"gpt-4o"``)."""
    metadata: dict[str, object] = dataclasses.field(default_factory=dict)
    """Arbitrary extra data such as cost, latency, or token count."""

    @classmethod
    def from_json(cls, data: GenerationResultDict) -> GenerationResult:
        """Rebuild from a JSON-decoded dict (e.g. a line read from a run file)."""
        return cls(
            id=uuid.UUID(data["id"]),
            task_id=uuid.UUID(data["task_id"]),
            generation_prompt=data["generation_prompt"],
            raw_response=data["raw_response"],
            output=data["output"],
            author=data["author"],
            metadata=data["metadata"],
        )


class RankingTaskDict(TypedDict):
    id: str
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
            ranking_prompt=data["ranking_prompt"],
            generations={
                alias: uuid.UUID(gid) for alias, gid in data["generations"].items()
            },
        )


class RankingFailureDict(TypedDict):
    ranking_task_id: str
    author: str
    error_type: str
    message: str


@dataclasses.dataclass(frozen=True, slots=True)
class RankingFailure:
    """A single model's failure to rank a :class:`RankingResult`."""

    ranking_task_id: uuid.UUID
    """The :class:`RankingTask` this failure is for."""
    author: str
    """The model identifier that failed to rank this result."""
    error_type: str
    """The type of error that occurred."""
    message: str
    """The error message."""

    @classmethod
    def from_json(cls, data: RankingFailureDict) -> RankingFailure:
        """Rebuild from a JSON-decoded dict (e.g. a line read from a run file)."""
        return cls(
            ranking_task_id=uuid.UUID(data["ranking_task_id"]),
            author=data["author"],
            error_type=data["error_type"],
            message=data["message"],
        )


class RankingResultDict(TypedDict):
    id: str
    ranking_task_id: str
    ranking_prompt: str
    author: str
    raw_model_ranking: list[str]
    ranking: list[str]
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
    reasoning: str | None
    """The reasoning provided by the model, if any."""
    raw_response: str
    """The complete structured JSON response returned by the LLM."""
    metadata: dict[str, object] = dataclasses.field(default_factory=dict)
    """Arbitrary extra data such as cost, latency, or token count."""

    @classmethod
    def from_json(cls, data: RankingResultDict) -> RankingResult:
        """Rebuild from a JSON-decoded dict (e.g. a line read from a run file)."""
        return cls(
            id=uuid.UUID(data["id"]),
            ranking_task_id=uuid.UUID(data["ranking_task_id"]),
            ranking_prompt=data["ranking_prompt"],
            author=data["author"],
            raw_model_ranking=data["raw_model_ranking"],
            ranking=[uuid.UUID(gid) for gid in data["ranking"]],
            reasoning=data["reasoning"],
            raw_response=data["raw_response"],
            metadata=data["metadata"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ParsedRanking:
    """A ranking model's validated verdict, as aliases.

    The output of :meth:`RankingTemplate.parse`, still in alias space (``"A"``,
    ``"B"``); the orchestration maps these back to :class:`GenerationResult` IDs
    when building the :class:`RankingResult`.
    """

    ranking: list[str]
    """Aliases in ranked order, best first."""
    reasoning: str | None
    """The ranking model's explanation, if it provided one."""


class RankingTemplate(abc.ABC):
    """The ranking contract: how a ranking model is prompted, constrained, and parsed.

    Bundles the three pieces that must agree with each other — they can't be
    varied independently:

    * :meth:`render` — the full prompt the ranking model sees.
    * :meth:`schema` — the JSON schema its response is constrained to.
    * :meth:`parse` — validation of that response into a :class:`ParsedRanking`.

    Subclass to customise ranking.  The common case — injecting context such as
    the original generation prompt — is a one-method override of :meth:`render`;
    each candidate is passed as its full :class:`GenerationResult`, so ``.output``,
    ``.generation_prompt`` and ``.metadata`` are all in reach.  Changing the
    *shape* of the verdict (e.g. allowing ties) means overriding all three.

    See :class:`DefaultRankingTemplate` for the built-in strict-total-order ranker.
    """

    @abc.abstractmethod
    def render(
        self,
        ranking_task: RankingTask,
        candidates: Mapping[str, GenerationResult],
    ) -> str:
        """Build the full prompt sent to a ranking model.

        ``candidates`` maps each alias to its :class:`GenerationResult` (resolved
        from ``ranking_task.generations``), so derive any context you need from it.
        Note it exposes ``.author``: rendering that into the prompt would defeat the
        anonymisation, so a custom template should read only ``.output`` and
        task-level fields like ``.generation_prompt``.
        """
        ...

    @property
    @abc.abstractmethod
    def schema(self) -> dict[str, object]:
        """The JSON schema ``generate_structured`` constrains the model's reply to."""
        ...

    @abc.abstractmethod
    def parse(
        self,
        data: dict[str, object],
        valid_aliases: set[str],
    ) -> ParsedRanking:
        """Validate a ranking model's parsed JSON reply and extract the ranking.

        ``valid_aliases`` are the aliases that must appear; raise
        :class:`ValueError` on anything malformed so the orchestration records it
        as a :class:`RankingFailure`.
        """
        ...


class DefaultRankingTemplate(RankingTemplate):
    """The built-in ranking model: a strict total order over the candidates, no ties.

    Prompts with the ranking instruction plus the anonymised candidate outputs,
    constrains the reply to ``{"ranking": [...], "reasoning": ...}``, and validates
    that every alias appears exactly once.
    """

    _SCHEMA: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            "ranking": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Aliases in ranked order, best first.",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Optional step-by-step analysis and explanation "
                    "behind overall ranking and tie breakers."
                ),
            },
        },
        "required": ["ranking"],
        "additionalProperties": False,
    }

    def render(
        self,
        ranking_task: RankingTask,
        candidates: Mapping[str, GenerationResult],
    ) -> str:
        lines: list[str] = [ranking_task.ranking_prompt, "", "Candidates:"]
        for alias, result in candidates.items():
            lines.append(f"{alias}. {result.output}")

        lines.extend(
            [
                "",
                "Respond ONLY with a JSON object in this exact format:",
                '{"ranking": ["A", "B", "C"], "reasoning": "..."}',
                "",
                'The "ranking" field must list every candidate alias exactly once,',
                'from best to worst. The "reasoning" field is optional.',
                (
                    "NO TIES ARE ALLOWED. If two outputs appear equal, "
                    "break the tie as you see fit."
                ),
                'and explain why using the "reasoning" field.',
            ]
        )
        return "\n".join(lines)

    @property
    def schema(self) -> dict[str, object]:
        return self._SCHEMA

    def parse(
        self,
        data: dict[str, object],
        valid_aliases: set[str],
    ) -> ParsedRanking:
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        ranking_raw = data.get("ranking")
        if not isinstance(ranking_raw, list):
            raise ValueError(
                f'"ranking" must be a list, got {type(ranking_raw).__name__}'
            )

        ranking: list[str] = []
        seen: set[str] = set()
        for alias in ranking_raw:
            if not isinstance(alias, str):
                raise ValueError(
                    f"Ranking entry must be a string, got {type(alias).__name__}"
                )
            if alias not in valid_aliases:
                raise ValueError(f"Unknown alias in ranking: {alias!r}")
            if alias in seen:
                raise ValueError(f"Duplicate alias in ranking: {alias!r}")
            seen.add(alias)
            ranking.append(alias)

        missing = valid_aliases - seen
        if missing:
            raise ValueError(f"Missing aliases in ranking: {sorted(missing)}")

        reasoning = data.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError(
                f'"reasoning" must be a string, got {type(reasoning).__name__}'
            )

        return ParsedRanking(ranking=ranking, reasoning=reasoning)
