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


class TaskManager:
    """Holds the full state of an evaluation run.

    Provides UUID-lookup accessors for every artifact produced.
    """

    _generation_tasks: dict[uuid.UUID, GenerationTask]
    """Mapping of :class:`GenerationTask` IDs to objects."""
    _ranking_tasks: dict[uuid.UUID, RankingTask]
    """Mapping of :class:`RankingTask` IDs to objects."""
    _generation_results: dict[uuid.UUID, GenerationResult]
    """Mapping of :class:`GenerationResult` IDs to objects."""
    _ranking_results: dict[uuid.UUID, RankingResult]
    """Mapping of :class:`RankingResult` IDs to objects."""

    def __init__(self) -> None:
        self._generation_tasks: dict[uuid.UUID, GenerationTask] = {}
        self._ranking_tasks: dict[uuid.UUID, RankingTask] = {}
        self._generation_results: dict[uuid.UUID, GenerationResult] = {}
        self._ranking_results: dict[uuid.UUID, RankingResult] = {}

    def create_generation_task(self, prompt: str) -> GenerationTask:
        """Create a :class:`GenerationTask`, store it, and return it.

        Parameters
        ----------
        prompt : str
            The creative task given to candidate models.

        Returns
        -------
        GenerationTask
            The newly created instance.
        """
        task = GenerationTask(id=uuid.uuid4(), generation_prompt=prompt)
        self._generation_tasks[task.id] = task
        return task

    def append_generation_task[T: GenerationTask](self, task: T) -> T:
        """Append an existing :class:`GenerationTask` and return it.

        Parameters
        ----------
        task : GenerationTask
            The task to append.

        Returns
        -------
        GenerationTask
            The same instance, now stored.

        Raises
        ------
        ValueError
            If the task ID already exists.
        """
        if task.id in self._generation_tasks:
            raise ValueError(f"Task {task.id} already exists.")
        self._generation_tasks[task.id] = task
        return task

    def create_ranking_task(
        self, ranking_prompt: str, generations: list[uuid.UUID]
    ) -> RankingTask:
        """Create a :class:`RankingTask`, store it, and return it.

        Aliases are assigned automatically (A, B, C, ...).

        Parameters
        ----------
        ranking_prompt : str
            The exact ranking prompt to be sent to the model.
        generations : list[uuid.UUID]
            The :class:`GenerationResult` IDs to be ranked.

        Returns
        -------
        RankingTask
            The newly created instance.

        Raises
        ------
        ValueError
            If any generation ID does not exist.
            If any generation references a generation task that does not exist.
        """
        anonymized_map: dict[str, uuid.UUID] = {}
        letter_generator = LetterGenerator()
        for gen_id in generations:
            if gen_id not in self._generation_results:
                raise ValueError(f"Generation {gen_id} does not exist.")
            if self._generation_results[gen_id].task_id not in self._generation_tasks:
                raise ValueError(
                    f"Generation {gen_id} references a generation task that does not exist."
                )
            anonymized_map[letter_generator.get_next_letter()] = gen_id
        ranking_task = RankingTask(
            id=uuid.uuid4(), ranking_prompt=ranking_prompt, generations=anonymized_map
        )
        self._ranking_tasks[ranking_task.id] = ranking_task
        return ranking_task

    def append_ranking_task[T: RankingTask](self, ranking_task: T) -> T:
        """Append an existing :class:`RankingTask` and return it.

        Parameters
        ----------
        ranking_task : RankingTask
            The task to append.

        Returns
        -------
        RankingTask
            The same instance, now stored.

        Raises
        ------
        ValueError
            If the task ID already exists.
            If any generation ID does not exist.
            If any generation references a generation task that does not exist.
        """
        if ranking_task.id in self._ranking_tasks:
            raise ValueError(f"Task {ranking_task.id} already exists.")

        for generation_id in ranking_task.generations.values():
            if generation_id not in self._generation_results:
                raise ValueError(f"Generation {generation_id} does not exist.")
            if (
                self._generation_results[generation_id].task_id
                not in self._generation_tasks
            ):
                raise ValueError(
                    f"Generation {generation_id} references a generation task that does not exist."
                )

        self._ranking_tasks[ranking_task.id] = ranking_task
        return ranking_task
