import dataclasses
import uuid


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
    metadata: dict = dataclasses.field(default_factory=dict)
    """Arbitrary extra data such as cost, latency, or token count."""


@dataclasses.dataclass(frozen=True, slots=True)
class RankingTask:
    """A description of a ranking task.

    Each RankingTask is used by every judge model to produce one
    :class:`RankingResult`.
    """

    id: uuid.UUID
    """Unique identifier for this ranking task."""
    ranking_prompt: str
    """The judging instructions to be used when ranking the candidates."""
    generations: dict[str, uuid.UUID]
    """Mapping of anonymized aliases (e.g. ``"A"``, ``"B"``) to
    :class:`GenerationResult` IDs."""


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
    metadata: dict = dataclasses.field(default_factory=dict)
    """Arbitrary extra data such as cost, latency, or token count."""
