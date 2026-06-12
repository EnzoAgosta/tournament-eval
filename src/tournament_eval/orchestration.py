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
from collections.abc import Collection, Sequence
from pathlib import Path

from tournament_eval import persistence
from tournament_eval.llm import LLMClient
from tournament_eval.models import (
    GenerationFailure,
    GenerationResult,
    GenerationTask,
    LetterGenerator,
    RankingFailure,
    RankingResult,
    RankingTask,
)


def _build_generation_lookup(
    generation_results: list[GenerationResult],
) -> dict[uuid.UUID, GenerationResult]:
    """Index GenerationResults by their ID for O(1) lookup."""
    return {result.id: result for result in generation_results}


def _build_judge_prompt(
    ranking_task: RankingTask,
    generation_lookup: dict[uuid.UUID, GenerationResult],
) -> str:
    """Assemble the full prompt sent to a judge model.

    Combines the ranking_prompt, the anonymised candidate outputs, and a JSON
    schema reminder.
    """
    lines: list[str] = [ranking_task.ranking_prompt, "", "Candidates:"]
    for alias, gen_id in ranking_task.generations.items():
        output = generation_lookup[gen_id].output
        lines.append(f"{alias}. {output}")

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


_DEFAULT_RANKING_SCHEMA: dict[str, object] = {
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


def _parse_ranking_response(
    data: dict[str, object],
    valid_aliases: set[str],
) -> tuple[list[str], str | None]:
    """Validate and extract ranking + reasoning from a structured response.

    Parameters
    ----------
    data : dict
        The parsed JSON returned by the model.
    valid_aliases : set[str]
        The aliases that must appear exactly once in the ranking.

    Returns
    -------
    tuple[list[str], str | None]
        The validated alias ranking and optional reasoning text.

    Raises
    ------
    ValueError
        If the response is malformed, contains unknown aliases, duplicates,
        or misses required aliases.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    ranking_raw = data.get("ranking")
    if not isinstance(ranking_raw, list):
        raise ValueError(f'"ranking" must be a list, got {type(ranking_raw).__name__}')

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

    return ranking, reasoning


async def generate_one(
    task: GenerationTask,
    client: LLMClient,
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
    clients: Sequence[LLMClient],
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
    clients : list[LLMClient]
        The models that will generate outputs.
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


def build_ranking_tasks(
    tasks: list[GenerationTask],
    generation_results: list[GenerationResult],
    ranking_prompt: str,
    *,
    output: str | Path | None = None,
    random_seed: int | None = None,
) -> list[RankingTask]:
    """Group GenerationResults by task and create RankingTasks with aliases.

    Parameters
    ----------
    tasks : list[GenerationTask]
        The original generation tasks (used for ordering and task IDs).
    generation_results : list[GenerationResult]
        The outputs produced by ``generate_all``.
    ranking_prompt : str
        The judging instructions to embed in every RankingTask.
    output : str | Path | None
        If set, the built ranking tasks are written to this run directory.
        Because their IDs and alias assignment are random, build them once and
        reuse the persisted set on resume (``persistence.read_ranking_task_file``)
        rather than calling this again.
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

    ranking_tasks: list[RankingTask] = []
    for task in tasks:
        results = results_by_task.get(task.id, [])
        if not results:
            continue

        alias_map: dict[str, uuid.UUID] = {}
        letter_gen = LetterGenerator()
        # to fight models whi just like to choose A all the time
        rng = random.Random(random_seed)
        rng.shuffle(results)
        for result in results:
            alias_map[letter_gen.get_next_letter()] = result.id

        ranking_tasks.append(
            RankingTask(
                id=uuid.uuid4(),
                ranking_prompt=ranking_prompt,
                generations=alias_map,
            )
        )
    if output is not None:
        persistence.write_ranking_tasks(output, ranking_tasks)
    return ranking_tasks


async def rank_one(
    ranking_task: RankingTask,
    client: LLMClient,
    generation_lookup: dict[uuid.UUID, GenerationResult],
    *,
    output: str | Path | None = None,
) -> RankingResult | RankingFailure:
    """Call a single client as judge for a single RankingTask.

    Returns a :class:`RankingResult`, or a :class:`RankingFailure` if the judge
    raised or returned a malformed ranking.

    Like :func:`generate_one`, concurrency is bounded by the client itself, so a
    custom loop over :func:`rank_one` is throttled the same as :func:`rank_all`.
    If ``output`` is a directory, the result or failure is appended to that
    run's JSONL files as soon as it is produced.  Only the judging call is
    guarded, so a persistence error propagates rather than masquerading as a
    failure.
    """
    prompt = _build_judge_prompt(ranking_task, generation_lookup)
    result: RankingResult | RankingFailure

    try:
        structured = await client.generate_structured(prompt, _DEFAULT_RANKING_SCHEMA)
        raw_ranking, reasoning = _parse_ranking_response(
            structured.data,
            set(ranking_task.generations.keys()),
        )
        uuid_ranking = [ranking_task.generations[alias] for alias in raw_ranking]
        result = RankingResult(
            id=uuid.uuid4(),
            ranking_task_id=ranking_task.id,
            ranking_prompt=prompt,
            author=client.name,
            raw_model_ranking=raw_ranking,
            ranking=uuid_ranking,
            reasoning=reasoning,
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
    clients: Sequence[LLMClient],
    *,
    output: str | Path | None = None,
    skip: Collection[tuple[uuid.UUID, str]] = (),
) -> tuple[list[RankingResult], list[RankingFailure]]:
    """Run every client as a judge against every RankingTask.

    For each RankingTask, the full judge prompt is built from the
    ``ranking_prompt`` and the actual candidate output texts.  Each client
    receives the assembled prompt and returns a structured ranking.

    Parameters
    ----------
    ranking_tasks : list[RankingTask]
        The scenarios to be ranked.
    generation_results : list[GenerationResult]
        The outputs to be ranked (looked up by the aliases in each
        RankingTask).
    clients : list[LLMClient]
        The models that will act as judges.
    output : str | Path | None
        If set, a run directory: each result or failure is streamed to its
        per-judge JSONL file as it lands.  The ranking *tasks* are persisted
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
        one :class:`RankingFailure` per pair whose judge raised or returned a
        malformed ranking.  When ``output`` is set, the same records are also
        streamed to the run directory.
    """
    generation_lookup = _build_generation_lookup(generation_results)

    skip_set = set(skip)
    coros = [
        rank_one(ranking_task, client, generation_lookup, output=output)
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
