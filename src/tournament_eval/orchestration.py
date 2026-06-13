"""Pure-function orchestration for the tournament evaluation flow.

Typical usage::

    from tournament_eval.models import GenerationTask
    from tournament_eval.llm import OpenAICompatibleLLMClient, AnthropicLLMClient
    from tournament_eval.orchestration import (
        build_ranking_tasks,
        generate_all,
        rank_all,
    )

    tasks = [GenerationTask(...), GenerationTask(...)]

    # Clients are async context managers (each owns a connection pool).
    async with OpenAICompatibleLLMClient(...) as a, OpenAICompatibleLLMClient(...) as b:
        clients = [a, b]

        generation_results, _failures = await generate_all(tasks, clients)

        ranking_tasks = build_ranking_tasks(
            tasks=tasks,
            generation_results=generation_results,
            ranking_prompt="Rank these translations by fluency and accuracy.",
        )

        ranking_results, _failures = await rank_all(
            ranking_tasks=ranking_tasks,
            generation_results=generation_results,
            clients=clients,
        )
"""

import asyncio
import random
import uuid
from collections.abc import Collection, Iterable
from pathlib import Path

from tournament_eval import persistence
from tournament_eval.llm import LLMClient, ModelConfig
from tournament_eval.models import (
    DefaultRankingTemplate,
    GenerationFailure,
    GenerationResult,
    GenerationTask,
    LetterGenerator,
    RankingFailure,
    RankingResult,
    RankingTask,
    RankingTemplate,
)

_DEFAULT_TEMPLATE: RankingTemplate = DefaultRankingTemplate()


def _build_generation_lookup(
    generation_results: list[GenerationResult],
) -> dict[uuid.UUID, GenerationResult]:
    """Index GenerationResults by their ID for O(1) lookup."""
    return {result.id: result for result in generation_results}


async def generate_one(
    task: GenerationTask,
    client: LLMClient[ModelConfig],
    *,
    output: str | Path | None = None,
) -> GenerationResult | GenerationFailure:
    """Call a single client for a single task.

    Returns a :class:`GenerationResult`, or a :class:`GenerationFailure` if the
    client raised — a caller looping over :func:`generate_one` directly tells the
    two apart by type.

    Concurrency is the client's concern: if the client was given a
    ``max_concurrency`` (or a shared semaphore), this call self-throttles.  That
    means a hand-rolled loop over :func:`generate_one` gets the same bounding as
    :func:`generate_all` for free.

    If ``output`` is a directory, the result or failure is appended to that
    run's JSONL files as soon as it is produced (see
    :mod:`tournament_eval.persistence`).  Only the model call is guarded, so a
    persistence error propagates rather than masquerading as a failure.
    """
    result: GenerationResult | GenerationFailure
    try:
        raw = await client.generate(task.generation_prompt)
        result = GenerationResult(
            id=uuid.uuid4(),
            task_id=task.id,
            generation_prompt=task.generation_prompt,
            raw_response=raw,
            output=raw,  # no cleaning applied by default
            author=client.name,
        )
    except Exception as err:
        result = GenerationFailure(
            task_id=task.id,
            author=client.name,
            error_type=type(err).__name__,
            message=str(err),
        )

    if output is not None:
        if isinstance(result, GenerationResult):
            persistence.append_generation_result(output, result)
        else:
            persistence.append_generation_failure(output, result)
    return result


async def generate_all(
    tasks: list[GenerationTask],
    clients: Iterable[LLMClient[ModelConfig]],
    *,
    output: str | Path | None = None,
    skip: Collection[tuple[uuid.UUID, str]] = (),
) -> tuple[list[GenerationResult], list[GenerationFailure]]:
    """Run every client against every task to produce GenerationResults.

    All calls are fired concurrently.

    Parameters
    ----------
    tasks : list[GenerationTask]
        The creative tasks to evaluate.
    clients : Iterable[LLMClient]
        The models that will generate outputs.  Any iterable (list, tuple, set,
        generator); it is materialised once, so a single-pass iterator is safe
        even though every task is paired with every client.
    output : str | Path | None
        If set, a run directory: each result or failure is streamed to its
        per-model JSONL file as it lands.  Persisting the *tasks* is the
        caller's responsibility (see ``persistence.append_generation_task``);
        this writes only the pipeline's own output.
    skip : Collection[tuple[uuid.UUID, str]]
        ``(task_id, author)`` pairs to leave un-run (an author is a client's
        ``name``).  To resume an interrupted run, pass the pairs already
        recorded under ``output`` — read them back with
        ``persistence.read_generation_result_file`` /
        ``read_generation_failure_file`` and take ``(rec.task_id, rec.author)``.

    Returns
    -------
    tuple[list[GenerationResult], list[GenerationFailure]]
        The results and failures for the pairs run *this call* (i.e. excluding
        anything in ``skip``).  One :class:`GenerationResult` per successful
        (task, client) pair — its ``output`` field mirrors ``raw_response``, no
        cleaning applied — and one :class:`GenerationFailure` per pair whose
        client raised.  When ``output`` is set, the same records are also
        streamed to the run directory.
    """
    skip_set = set(skip)
    clients = list(clients)  # materialise: clients is re-iterated once per task
    coros = [
        generate_one(task, client, output=output)
        for task in tasks
        for client in clients
        if (task.id, client.name) not in skip_set
    ]
    outcomes = await asyncio.gather(*coros)

    generation_results: list[GenerationResult] = []
    generation_failures: list[GenerationFailure] = []
    for outcome in outcomes:
        if isinstance(outcome, GenerationResult):
            generation_results.append(outcome)
        else:
            generation_failures.append(outcome)
    return generation_results, generation_failures


def build_ranking_task(
    results: list[GenerationResult],
    ranking_prompt: str,
    *,
    output: str | Path | None = None,
    random_seed: int | None = None,
) -> RankingTask:
    """Build one RankingTask from a single task's GenerationResults.

    The ``results`` are shuffled before aliases (A, B, C...) are assigned, so a
    model's position bias (e.g. always picking "A") doesn't track authorship.

    Parameters
    ----------
    results : list[GenerationResult]
        The outputs for one GenerationTask.  Must be non-empty.
    ranking_prompt : str
        The ranking instructions to embed in the RankingTask.
    output : str | Path | None
        If a directory, the task is appended to the run's ranking-tasks file as
        soon as it is built (mirrors ``generate_one``).
    random_seed : int | None
        Seed for the alias shuffle; ``None`` for nondeterministic.

    Returns
    -------
    RankingTask
        Maps aliases to the GenerationResult IDs, in shuffled order.
    """
    shuffled = list(results)
    rng = random.Random(random_seed)
    rng.shuffle(shuffled)

    letter_gen = LetterGenerator()
    alias_map: dict[str, uuid.UUID] = {
        letter_gen.get_next_letter(): result.id for result in shuffled
    }
    ranking_task = RankingTask(
        id=uuid.uuid4(),
        ranking_prompt=ranking_prompt,
        generations=alias_map,
    )
    if output is not None:
        persistence.append_ranking_task(output, ranking_task)
    return ranking_task


def build_ranking_tasks(
    tasks: list[GenerationTask],
    generation_results: list[GenerationResult],
    ranking_prompt: str,
    *,
    output: str | Path | None = None,
    random_seed: int | None = None,
) -> list[RankingTask]:
    """Group GenerationResults by task and create RankingTasks with aliases.

    A thin fan-out over :func:`build_ranking_task` — one RankingTask per task that
    produced at least one output.

    Parameters
    ----------
    tasks : list[GenerationTask]
        The original generation tasks (used for ordering and task IDs).
    generation_results : list[GenerationResult]
        The outputs produced by ``generate_all``.
    ranking_prompt : str
        The ranking instructions to embed in every RankingTask.
    output : str | Path | None
        If set, each task is appended to this run directory as it is built (via
        :func:`build_ranking_task`).  Because their IDs and alias assignment are
        random, build them once and reuse the persisted set on resume
        (``persistence.read_ranking_task_file``) rather than calling this again.
    random_seed : int | None
        Seed for the per-task alias shuffle; ``None`` for nondeterministic.

    Returns
    -------
    list[RankingTask]
        One RankingTask per task that has at least one GenerationResult.
        Each RankingTask maps aliases (A, B, C...) to GenerationResult IDs.
    """
    results_by_task: dict[uuid.UUID, list[GenerationResult]] = {}
    for result in generation_results:
        results_by_task.setdefault(result.task_id, []).append(result)

    return [
        build_ranking_task(
            results, ranking_prompt, output=output, random_seed=random_seed
        )
        for task in tasks
        if (results := results_by_task.get(task.id, []))
    ]


async def rank_one(
    ranking_task: RankingTask,
    client: LLMClient[ModelConfig],
    generation_lookup: dict[uuid.UUID, GenerationResult],
    *,
    template: RankingTemplate | None = None,
    output: str | Path | None = None,
) -> RankingResult | RankingFailure:
    """Call a single client as a ranking model for a single RankingTask.

    Returns a :class:`RankingResult`, or a :class:`RankingFailure` if the ranking model
    raised or returned a malformed ranking.

    The ``template`` (a :class:`RankingTemplate`, default
    :class:`DefaultRankingTemplate`) owns the prompt, the response schema, and the
    validation — subclass it to
    inject context or change the ranking contract.

    Like :func:`generate_one`, concurrency is bounded by the client itself, so a
    custom loop over :func:`rank_one` is throttled the same as :func:`rank_all`.
    If ``output`` is a directory, the result or failure is appended to that
    run's JSONL files as soon as it is produced.  Only the ranking call is
    guarded, so a persistence error propagates rather than masquerading as a
    failure.
    """
    template = template or _DEFAULT_TEMPLATE
    candidates = {
        alias: generation_lookup[gen_id]
        for alias, gen_id in ranking_task.generations.items()
    }
    prompt = template.render(ranking_task, candidates)
    result: RankingResult | RankingFailure

    try:
        structured = await client.generate_structured(prompt, template.schema)
        parsed = template.parse(
            structured.data,
            set(ranking_task.generations.keys()),
        )
        uuid_ranking = [ranking_task.generations[alias] for alias in parsed.ranking]
        result = RankingResult(
            id=uuid.uuid4(),
            ranking_task_id=ranking_task.id,
            ranking_prompt=prompt,
            author=client.name,
            raw_model_ranking=parsed.ranking,
            ranking=uuid_ranking,
            reasoning=parsed.reasoning,
            raw_response=structured.raw,
        )
    except Exception as exc:
        result = RankingFailure(
            ranking_task_id=ranking_task.id,
            author=client.name,
            error_type=type(exc).__name__,
            message=str(exc),
        )

    if output is not None:
        if isinstance(result, RankingResult):
            persistence.append_ranking_result(output, result)
        else:
            persistence.append_ranking_failure(output, result)
    return result


async def rank_all(
    ranking_tasks: list[RankingTask],
    generation_results: list[GenerationResult],
    clients: Iterable[LLMClient[ModelConfig]],
    *,
    template: RankingTemplate | None = None,
    output: str | Path | None = None,
    skip: Collection[tuple[uuid.UUID, str]] = (),
) -> tuple[list[RankingResult], list[RankingFailure]]:
    """Run every client as a ranking model against every RankingTask.

    For each RankingTask, the full ranking model prompt is built by ``template`` (a
    :class:`RankingTemplate`, default :class:`DefaultRankingTemplate`) from the
    ``ranking_prompt`` and the candidates.  Each client receives the assembled
    prompt and returns a structured ranking.

    Parameters
    ----------
    ranking_tasks : list[RankingTask]
        The scenarios to be ranked.
    generation_results : list[GenerationResult]
        The outputs to be ranked (looked up by the aliases in each
        RankingTask).
    clients : Iterable[LLMClient]
        The models that will act as ranking models.  Any iterable (list, tuple,
        set, generator); it is materialised once, so a single-pass iterator is
        safe even though every ranking task is paired with every client.
    template : RankingTemplate | None
        The ranking contract (prompt, schema, validation).  ``None`` uses
        :class:`DefaultRankingTemplate` — a strict total order with no ties.
    output : str | Path | None
        If set, a run directory: each result or failure is streamed to its
        each ranking model's JSONL file as it lands.  The ranking *tasks* are persisted
        when built (``build_ranking_tasks(..., output=...)``), not here — on
        resume pass those persisted tasks (``persistence.read_ranking_task_file``),
        since they carry random IDs and a shuffled alias map that must not be
        rebuilt.
    skip : Collection[tuple[uuid.UUID, str]]
        ``(ranking_task_id, author)`` pairs to leave un-run (an author is a
        client's ``name``).  To resume, pass the pairs already recorded under
        ``output`` — read them back with ``persistence.read_ranking_result_file``
        / ``read_ranking_failure_file`` and take ``(rec.ranking_task_id,
        rec.author)``.

    Returns
    -------
    tuple[list[RankingResult], list[RankingFailure]]
        The results and failures for the pairs run *this call*.  One
        :class:`RankingResult` per successful (ranking_task, client) pair, and
        one :class:`RankingFailure` per pair whose ranking model raised or returned a
        malformed ranking.  When ``output`` is set, the same records are also
        streamed to the run directory.
    """
    generation_lookup = _build_generation_lookup(generation_results)

    skip_set = set(skip)
    clients = list(clients)  # materialise: clients is re-iterated once per task
    coros = [
        rank_one(
            ranking_task, client, generation_lookup, template=template, output=output
        )
        for ranking_task in ranking_tasks
        for client in clients
        if (ranking_task.id, client.name) not in skip_set
    ]
    outcomes = await asyncio.gather(*coros)

    ranking_results: list[RankingResult] = []
    ranking_failures: list[RankingFailure] = []
    for outcome in outcomes:
        if isinstance(outcome, RankingResult):
            ranking_results.append(outcome)
        else:
            ranking_failures.append(outcome)
    return ranking_results, ranking_failures
