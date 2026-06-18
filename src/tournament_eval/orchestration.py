"""Pure-function orchestration for the tournament evaluation flow.

Three stages, each fired concurrently and each returning ``(results, failures)`` so
one failing model never sinks a run:

1. :func:`generate_all` — every client produces an output for every task.
2. :func:`build_ranking_tasks` — group the outputs per task and assign anonymised
   aliases (A, B, C...) so authorship is hidden from the rankers.
3. :func:`rank_all` — every client ranks every task's candidates.

:func:`generate_all` / :func:`rank_all` manage client lifecycles themselves — they
open any client that owns resources (e.g. the built-in httpx client) for the batch
and close it afterwards — so the caller just passes clients, no ``async with``::

    from tournament_eval import OpenAICompatibleClient
    from tournament_eval.models import GenerationTask
    from tournament_eval.orchestration import build_ranking_tasks, generate_all, rank_all

    tasks = [GenerationTask(id=..., generation_prompt="Translate ... into French."), ...]
    clients = [OpenAICompatibleClient(model_id="...", base_url="...", api_key="...")]

    generations, gen_failures = await generate_all(
        tasks, clients, results_path="run/generations.jsonl", failures_path="run/generation_failures.jsonl"
    )

    ranking_tasks = build_ranking_tasks(
        tasks, generations, "Rank by fluency and accuracy.", tasks_path="run/ranking_tasks.jsonl"
    )
    rankings, rank_failures = await rank_all(
        ranking_tasks, generations, clients,
        results_path="run/rankings.jsonl", failures_path="run/ranking_failures.jsonl",
    )

**Resume is automatic.**  When a ``results_path`` is given, :func:`generate_all` /
:func:`rank_all` read it back first and skip every ``(task, author)`` pair that
already succeeded, running only what's left and returning the *complete* set
(loaded plus newly produced).  So a partial run is finished by just running the
script again — no skip lists to thread through.  Successes are skipped; failures
are always retried (fix the cause, rerun).  Per call the return is a clean
partition: ``results`` is every pair with a success, ``failures`` every pair still
without one.

Aggregation — collapsing the per-ranker rankings into a leaderboard (Borda, Elo, ...) —
is intentionally out of scope; the pipeline hands you validated ranked data to score
however you like.
"""

import asyncio
import contextlib
import random
import uuid
from collections.abc import Iterable
from pathlib import Path

from tournament_eval import persistence
from tournament_eval.llm import LLMClient
from tournament_eval.models import (
    GenerationFailure,
    GenerationResult,
    GenerationTask,
    RankingFailure,
    RankingResult,
    RankingTask,
)
from tournament_eval.ranking import (
    DefaultRankingTemplate,
    RankingTemplate,
    alias_for_index,
)

_DEFAULT_TEMPLATE: RankingTemplate = DefaultRankingTemplate()


async def _open_clients(clients: list[LLMClient], stack: contextlib.AsyncExitStack) -> list[LLMClient]:
    """Enter the clients that are async context managers; pass the rest through.

    The built-in :class:`~tournament_eval.llm.OpenAICompatibleClient` owns an httpx
    pool and must be opened; the SDK-backed clients aren't context managers (they
    don't own the injected SDK client's lifecycle), so they're used as-is.  This is
    what lets :func:`generate_all` / :func:`rank_all` manage the batch's lifecycles
    so the caller doesn't have to wrap clients in their own ``async with``.
    """
    opened: list[LLMClient] = []
    for client in clients:
        if isinstance(client, contextlib.AbstractAsyncContextManager):
            opened.append(await stack.enter_async_context(client))
        else:
            opened.append(client)
    return opened


async def generate_one(
    task: GenerationTask,
    client: LLMClient,
    *,
    results_path: str | Path | None = None,
    failures_path: str | Path | None = None,
) -> GenerationResult | GenerationFailure:
    """Call a single client for a single task.

    Returns a :class:`GenerationResult`, or a :class:`GenerationFailure` if the
    client raised — a caller looping over :func:`generate_one` directly tells the
    two apart by type.

    Concurrency is the client's concern: if the client was given a
    ``max_concurrency`` (or a shared semaphore), this call self-throttles.  That
    means a hand-rolled loop over :func:`generate_one` gets the same bounding as
    :func:`generate_all` for free.

    Unlike :func:`generate_all`, this does not manage the client's lifecycle (open
    a resource-owning client yourself, ``async with client: ...``) and does **not**
    resume — it always runs.  Resume is a batch concern; see :func:`generate_all`.

    If ``results_path`` / ``failures_path`` is given, the produced
    :class:`GenerationResult` / :class:`GenerationFailure` is appended to that file
    as soon as it lands (see :mod:`tournament_eval.persistence`).  Only the model
    call is guarded, so a persistence error propagates rather than masquerading as
    a failure.
    """
    result: GenerationResult | GenerationFailure
    try:
        response = await client.generate(task.generation_prompt)
        result = GenerationResult(
            id=uuid.uuid4(),
            generation_task_id=task.id,
            generation_prompt=task.generation_prompt,
            output=response.text,
            reasoning=response.reasoning,
            author=client.name,
        )
    except Exception as err:
        result = GenerationFailure(
            generation_task_id=task.id,
            author=client.name,
            error_type=type(err).__name__,
            message=str(err),
        )

    if isinstance(result, GenerationResult):
        if results_path is not None:
            persistence.append_record(results_path, result)
    elif failures_path is not None:
        persistence.append_record(failures_path, result)
    return result


async def generate_all(
    tasks: list[GenerationTask],
    clients: Iterable[LLMClient],
    *,
    results_path: str | Path | None = None,
    failures_path: str | Path | None = None,
) -> tuple[list[GenerationResult], list[GenerationFailure]]:
    """Run every client against every task to produce GenerationResults.

    All calls are fired concurrently.  Clients that are async context managers (the
    built-in :class:`~tournament_eval.llm.OpenAICompatibleClient`) are opened for the
    duration of the batch and closed on exit; SDK-backed clients pass through.  So no
    caller-side ``async with`` is needed — pass clients and go.

    **Resume is automatic** when ``results_path`` is set: the file is read back
    first and every ``(generation_task_id, author)`` pair already recorded there is
    skipped, so re-running the script finishes an interrupted run.  Records loaded for
    a task or client absent from the current run are ignored.  Only *successes* are
    skipped; a pair that previously failed is retried.

    Parameters
    ----------
    tasks : list[GenerationTask]
        The creative tasks to evaluate.
    clients : Iterable[LLMClient]
        The models that will generate outputs.  Any iterable (list, tuple, set,
        generator); it is materialised once, so a single-pass iterator is safe
        even though every task is paired with every client.
    results_path : str | Path | None
        If set, the GenerationResults file: each success is streamed to it as it
        lands, and on entry it is read back to skip pairs already done (resume).
        Persisting the *tasks* is yours (``persistence.append_record``).
    failures_path : str | Path | None
        If set, the GenerationFailures file: each failure is appended as it lands.
        It is a write-only log — not read for resume — so across reruns it may hold
        several entries for a pair that kept failing.

    Returns
    -------
    tuple[list[GenerationResult], list[GenerationFailure]]
        A partition of every (task, client) pair: ``results`` is every pair that
        now has a success (loaded from ``results_path`` plus produced this call),
        and ``failures`` is every pair still without one.  A result's ``output`` is the
        model's answer verbatim (no cleaning); ``reasoning`` carries the trace when
        the provider surfaced one.
    """
    clients = list(clients)  # In case we get given a generator.
    loaded = persistence.read_generation_result_file(results_path) if results_path is not None else []
    valid_ids = {task.id for task in tasks}
    valid_authors = {client.name for client in clients}
    done: dict[tuple[uuid.UUID, str], GenerationResult] = {
        (result.generation_task_id, result.author): result
        for result in loaded
        if result.generation_task_id in valid_ids and result.author in valid_authors
    }
    async with contextlib.AsyncExitStack() as stack:
        # Open any clients that own resources (e.g. the built-in httpx pool) for
        # the duration of the batch, closing them all on exit; SDK clients pass through.
        live = await _open_clients(clients, stack)
        coros = [
            generate_one(task, client, results_path=results_path, failures_path=failures_path)
            for task in tasks
            for client in live
            if (task.id, client.name) not in done
        ]
        outcomes = await asyncio.gather(*coros)

    generation_results: list[GenerationResult] = list(done.values())
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
    tasks_path: str | Path | None = None,
    random_seed: int | None = None,
) -> RankingTask:
    """Build one RankingTask from a single task's GenerationResults.

    The ``results`` are shuffled before aliases (A, B, C...) are assigned, so a
    model's position bias (e.g. always picking "A") doesn't track authorship.  The
    originating task is taken from ``results[0].generation_task_id`` (all results must be for
    the same task) and recorded as the RankingTask's ``generation_task_id``.

    Parameters
    ----------
    results : list[GenerationResult]
        The outputs for one GenerationTask.  Must be non-empty.
    ranking_prompt : str
        The ranking instructions to embed in the RankingTask.
    tasks_path : str | Path | None
        If set, the RankingTask is appended to this ranking-tasks file as soon as
        it is built (mirrors ``generate_one``).
    random_seed : int | None
        Seed for the alias shuffle; ``None`` for nondeterministic.

    Returns
    -------
    RankingTask
        Maps aliases to the GenerationResult IDs, in shuffled order.
    """
    if not results:
        raise ValueError("build_ranking_task needs at least one GenerationResult")
    ids = {result.generation_task_id for result in results}
    if len(ids) != 1:
        raise ValueError(f"GenerationResults must all be for the same task; got {ids}")

    shuffled = list(results)
    rng = random.Random(random_seed)
    rng.shuffle(shuffled)

    alias_map: dict[str, uuid.UUID] = {alias_for_index(i): result.id for i, result in enumerate(shuffled)}
    ranking_task = RankingTask(
        id=uuid.uuid4(),
        generation_task_id=results[0].generation_task_id,
        ranking_prompt=ranking_prompt,
        generations=alias_map,
    )
    if tasks_path is not None:
        persistence.append_record(tasks_path, ranking_task)
    return ranking_task


def build_ranking_tasks(
    tasks: list[GenerationTask],
    generation_results: list[GenerationResult],
    ranking_prompt: str,
    *,
    tasks_path: str | Path | None = None,
    random_seed: int | None = None,
) -> list[RankingTask]:
    """Group GenerationResults by task and create RankingTasks with aliases.

    A thin fan-out over :func:`build_ranking_task` — one RankingTask per task that
    produced at least one output.

    **Resume is per-task** when ``tasks_path`` is set: the file is read back and any
    task that already has a persisted RankingTask reuses it as-is; only tasks
    without one are (re)built and appended.  This is deliberately *frozen* — a
    task's RankingTask, once built, is never rebuilt, because rebuilding would
    re-shuffle the aliases and orphan every ranking already collected against it.

    The consequence: a generation that lands *after* its task's RankingTask was
    built (e.g. a flaky model that only succeeded on a later resume) will **not**
    be added to that task's candidate set.  So build ranking tasks only once
    generation is fully done — run :func:`generate_all` until its ``failures`` are
    empty before calling this.  To deliberately rebuild, delete the ranking-tasks
    file (and any rankings already collected) first.

    Parameters
    ----------
    tasks : list[GenerationTask]
        The original generation tasks (used for ordering and task IDs).
    generation_results : list[GenerationResult]
        The outputs produced by ``generate_all``.
    ranking_prompt : str
        The ranking instructions to embed in every RankingTask.
    tasks_path : str | Path | None
        If set, RankingTasks are streamed here as built and read back for per-task
        resume (see above).
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
        results_by_task.setdefault(result.generation_task_id, []).append(result)

    existing = {}
    if tasks_path is not None:
        existing = {rt.generation_task_id: rt for rt in persistence.read_ranking_task_file(tasks_path)}

    ranking_tasks: list[RankingTask] = []
    for task in tasks:
        results = results_by_task.get(task.id, [])
        if not results:
            continue
        if task.id in existing:
            ranking_tasks.append(existing[task.id])  # frozen: reuse, never rebuild
        else:
            ranking_tasks.append(
                build_ranking_task(results, ranking_prompt, tasks_path=tasks_path, random_seed=random_seed)
            )
    return ranking_tasks


async def rank_one(
    ranking_task: RankingTask,
    client: LLMClient,
    generation_lookup: dict[uuid.UUID, GenerationResult],
    *,
    template: RankingTemplate | None = None,
    results_path: str | Path | None = None,
    failures_path: str | Path | None = None,
) -> RankingResult | RankingFailure:
    """Call a single client as a ranking model for a single RankingTask.

    Returns a :class:`RankingResult`, or a :class:`RankingFailure` if the ranking model
    raised or returned a malformed ranking.

    The ``template`` (a :class:`RankingTemplate`, default
    :class:`DefaultRankingTemplate`) owns the prompt, the response schema, and the
    validation — subclass it to inject context or change the ranking contract.

    As with :func:`generate_one`, concurrency is bounded by the client itself, the
    client's lifecycle is the caller's (:func:`rank_all` manages it for a batch),
    and there is no resume — it always runs.  If ``results_path`` / ``failures_path``
    is given, the produced record is appended to that file as soon as it lands.
    Only the ranking call is guarded, so a persistence error propagates rather than
    masquerading as a failure.
    """
    template = template or _DEFAULT_TEMPLATE
    candidates = {alias: generation_lookup[gen_id] for alias, gen_id in ranking_task.generations.items()}
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

    if isinstance(result, RankingResult):
        if results_path is not None:
            persistence.append_record(results_path, result)
    elif failures_path is not None:
        persistence.append_record(failures_path, result)
    return result


async def rank_all(
    ranking_tasks: list[RankingTask],
    generation_results: list[GenerationResult],
    clients: Iterable[LLMClient],
    *,
    template: RankingTemplate | None = None,
    results_path: str | Path | None = None,
    failures_path: str | Path | None = None,
) -> tuple[list[RankingResult], list[RankingFailure]]:
    """Run every client as a ranking model against every RankingTask.

    For each RankingTask, the full ranking model prompt is built by ``template`` (a
    :class:`RankingTemplate`, default :class:`DefaultRankingTemplate`) from the
    ``ranking_prompt`` and the candidates.  Each client receives the assembled
    prompt and returns a structured ranking.  Like :func:`generate_all`, resource-owning
    clients are opened for the batch and closed on exit (no caller-side ``async with``).

    **Resume is automatic** when ``results_path`` is set, exactly as in
    :func:`generate_all` but keyed on ``(ranking_task_id, author)``: already-recorded
    successes are skipped and returned alongside what's produced this call.  Records
    loaded for a ranking task or client absent from the current run are ignored.

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
    results_path : str | Path | None
        If set, the RankingResults file: each success is streamed to it as it
        lands, and on entry it is read back to skip pairs already done (resume).
        The ranking *tasks* are persisted separately when built
        (``build_ranking_tasks(..., tasks_path=...)``).
    failures_path : str | Path | None
        If set, the RankingFailures file: each failure is appended as it lands.
        A write-only log (not read for resume).

    Returns
    -------
    tuple[list[RankingResult], list[RankingFailure]]
        A partition of every (ranking_task, client) pair: ``results`` is every pair
        that now has a success (loaded from ``results_path`` plus produced this
        call), and ``failures`` every pair still without one (the ranking model raised or
        returned a malformed ranking).
    """
    clients = list(clients)
    generation_lookup = {result.id: result for result in generation_results}
    loaded = persistence.read_ranking_result_file(results_path) if results_path is not None else []
    valid_ids = {ranking_task.id for ranking_task in ranking_tasks}
    valid_authors = {client.name for client in clients}
    done: dict[tuple[uuid.UUID, str], RankingResult] = {
        (result.ranking_task_id, result.author): result
        for result in loaded
        if result.ranking_task_id in valid_ids and result.author in valid_authors
    }

    async with contextlib.AsyncExitStack() as stack:
        # Open resource-owning clients for the batch (see generate_all); SDK clients pass through.
        live = await _open_clients(clients, stack)
        coros = [
            rank_one(
                ranking_task,
                client,
                generation_lookup,
                template=template,
                results_path=results_path,
                failures_path=failures_path,
            )
            for ranking_task in ranking_tasks
            for client in live
            if (ranking_task.id, client.name) not in done
        ]
        outcomes = await asyncio.gather(*coros)

    ranking_results: list[RankingResult] = list(done.values())
    ranking_failures: list[RankingFailure] = []
    for outcome in outcomes:
        if isinstance(outcome, RankingResult):
            ranking_results.append(outcome)
        else:
            ranking_failures.append(outcome)
    return ranking_results, ranking_failures
