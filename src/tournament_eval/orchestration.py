"""Pure-async orchestration for the tournament evaluation flow.

Three stages, each fired concurrently and each returning ``(results, failures)`` so
one failing agent never sinks a run:

1. :func:`generate_all` - every agent produces an output for every task.
2. :func:`build_ranking_tasks` - group the outputs per task and assign anonymised
   aliases (A, B, C...) so authorship is hidden from the rankers.
3. :func:`rank_all` - every agent ranks every task's candidates.

Build your generation tasks first with :func:`build_generation_tasks` (rerun-stable;
see below) or construct :class:`~tournament_eval.models.GenerationTask` directly,
then run the three stages.

The pipeline is a thin wrapper over `pydantic-ai <https://ai.pydantic.dev>`_.
You construct and own the :class:`pydantic_ai.Agent` instances - configuring their
model, settings, thinking, retries, and concurrency the pydantic-ai way - and hand
them in.  ``generate_all`` / ``rank_all`` call ``agent.run`` and map each
:class:`pydantic_ai.AgentRunResult` onto the tournament data model (output,
reasoning trace, usage → metadata).  No client lifecycle to manage: ``agent.run``
is a plain coroutine, so there's no ``async with`` and nothing to open or close.

Generation agents produce ``str``; ranking agents produce the ranking
:class:`~tournament_eval.ranking.RankingTemplate`'s ``response_model`` (a pydantic
model).  Contestants and rankers are separate lists, so the two roles use agents
with different output types — wire a ranking agent with
``Agent(model, output_type=template.response_model)``.

**Resume is automatic.**  When a ``results_path`` is given, :func:`generate_all` /
:func:`rank_all` read it back first and skip every ``(task, author)`` pair that
already succeeded, running only what's left and returning the *complete* set
(loaded plus newly produced).  So a partial run is finished by just running the
script again - no skip lists to thread through.  Successes are skipped; failures
are always retried (fix the cause, rerun).  Per call the return is a clean
partition: ``results`` is every pair with a success, ``failures`` every pair still
without one.

**Author labels are load-bearing.**  Each result's ``author`` is resolved from the
agent (``agent.name`` if set, else the model name) and - together with the task id
- is the resume key, so two agents in one run must resolve to distinct authors.
:func:`generate_all` / :func:`rank_all` check this up front and raise on
collisions; set distinct ``agent.name`` values (or use distinct models) to disambiguate
e.g. one model run at two temperatures.

Aggregation - collapsing the per-ranker rankings into a leaderboard (Borda, Elo,
...) - is intentionally out of scope; the pipeline hands you validated ranked data
to score however you like.
"""

import asyncio
import random
import uuid
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ThinkingPart
from pydantic_ai.models import Model
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RunUsage

from tournament_eval import persistence
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

# Type aliases: generation agents produce ``str``; ranking agents produce the
# template's response model.  Kept loose (``Agent[None, object]``) at the batch
# boundary so a list of agents with a shared response model still type-checks.
_GenAgent = Agent[None, str]
_RankAgent = Agent[None, BaseModel]


def _resolve_author[DepsT, OutT](agent: Agent[DepsT, OutT]) -> str:
    """Resolve an agent's stable author label: ``agent.name`` if set, else the model name.

    ``agent.name`` is the explicit, stable label - use it when two agents share a
    model id (e.g. one model at two temperatures) and must stay distinct in the
    resume key.  Otherwise the model name dedups naturally: distinct models get
    distinct authors.  Raises if neither is available, since the label is
    load-bearing for resume and de-anonymisation.
    """
    name = agent.name
    if name:
        return name
    model = agent.model
    if isinstance(model, Model):
        return model.model_name
    if model:  # a str / KnownModelName literal (e.g. deferred-model case)
        return str(model)
    raise ValueError(
        f"Agent has no resolvable author: set agent.name= or give it a model. (name={name!r}, model={model!r})"
    )


def _check_distinct_authors[DepsT, OutT](
    agents: Iterable[Agent[DepsT, OutT]],
) -> list[str]:
    """Resolve every agent's author and raise on duplicates.

    Duplicate authors would silently overwrite each other in the resume map and
    corrupt de-anonymisation, so a collision is a hard error with a clear message
    rather than silent data loss.
    """
    authors = [_resolve_author(agent) for agent in agents]
    seen: dict[str, int] = {}
    dups: list[str] = []
    for a in authors:
        seen[a] = seen.get(a, 0) + 1
        if seen[a] == 2:
            dups.append(a)
    if dups:
        raise ValueError(
            f"Duplicate agent authors {sorted(dups)!r} - two agents resolve to the same "
            "author label, which would corrupt resume and de-anonymisation. "
            "Set distinct agent.name= (or use distinct models)."
        )
    return authors


def _extract_reasoning(messages: list[ModelMessage]) -> str | None:
    """Pull the reasoning trace out of a run's new messages.

    pydantic-ai normalises each provider's thinking into :class:`ThinkingPart`
    objects on the model responses; we join their ``content``.  ``None`` when the
    model produced no thinking parts (or they carried no content - some providers
    surface raw reasoning only in ``provider_details`` and we don't chase that
    yet).
    """
    chunks: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ThinkingPart) and part.content:
                    chunks.append(part.content)
    return "\n".join(chunks) if chunks else None


def _usage_to_metadata(usage: RunUsage) -> dict[str, object]:
    """Stamp a run's usage into the result ``metadata`` kitchen-sink dict.

    Token/request counts come from pydantic-ai's :class:`RunUsage`; cost ($) is a
    later addition (``RunUsage`` implements ``genai_prices``' ``AbstractUsage``,
    so it's reachable).  Keys are stable across reruns so persisted records stay
    readable.
    """
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "requests": usage.requests,
    }


def build_generation_task(
    prompt: str,
    *,
    id: uuid.UUID | None = None,
    tasks_path: str | Path | None = None,
) -> GenerationTask:
    """Build one GenerationTask from a prompt.

    Mirrors :func:`build_ranking_task` for the generation side: a fresh
    :class:`uuid.UUID` (or the given ``id``) is stamped on the task and, if
    ``tasks_path`` is set, the task is appended to that generation-tasks file as
    soon as it is built.

    **No resume** — unlike :func:`build_generation_tasks`, this always builds a
    new task (with a fresh id when ``id`` is unset).  Resume is a batch concern;
    use :func:`build_generation_tasks` when you want rerun stability.

    Parameters
    ----------
    prompt : str
        The generation prompt for this task.
    id : uuid.UUID | None
        Optional explicit id; defaults to a fresh ``uuid4``.  Pass an explicit
        id only if you want to opt out of the library's id management — note that
        a random id here makes :func:`generate_all` resume silently no-op for this
        task across reruns (the id won't match), so prefer
        :func:`build_generation_tasks` for rerun-stable ids.
    tasks_path : str | Path | None
        If set, the GenerationTask is appended to this generation-tasks file as
        soon as it is built (mirrors ``build_ranking_task``).

    Returns
    -------
    GenerationTask
        A single GenerationTask with ``id`` and ``generation_prompt`` set.
    """
    task = GenerationTask(id=id or uuid.uuid4(), generation_prompt=prompt)
    if tasks_path is not None:
        persistence.append_record(tasks_path, task)
    return task


def build_generation_tasks(
    prompts: Iterable[str],
    tasks_path: str | Path,
) -> list[GenerationTask]:
    """Build GenerationTasks from prompts, rerun-stable by matching on prompt.

    Mirrors :func:`build_ranking_tasks` for the generation side.  ``tasks_path`` is
    **required** — persistence is the whole point: on a rerun the file is read back
    and any prompt that already has a persisted task reuses it (and its id) as-is;
    only prompts without one are built (with a fresh ``uuid4``) and appended.

    This is what makes the naive "just rerun the script" pattern actually resume:
    ids stay stable across runs because they come from the persisted file, so
    :func:`generate_all`'s existing ``(task_id, author)`` resume matches and skips
    already-done pairs instead of silently regenerating everything.

    Tasks are matched to prompts **by their ``generation_prompt``**.  The
    consequence: duplicate prompts collapse to a single task (and a single
    GenerationResult per author) — two identical prompts are one task, not two,
    and you don't want two identical candidates behind two aliases in a ranking.
    For deliberate distinct duplicates, use :func:`build_generation_task` with an
    explicit ``id`` (or construct :class:`GenerationTask` directly).

    Parameters
    ----------
    prompts : Iterable[str]
        The generation prompts.  Any iterable; materialised once, so an open text
        file (yielding lines) is fine — strip the trailing newlines yourself if
        they shouldn't be part of the prompt.
    tasks_path : str | Path
        The generation-tasks file: tasks are streamed here as built and read back
        for per-prompt resume.  Required.

    Returns
    -------
    list[GenerationTask]
        One GenerationTask per input prompt (duplicates collapsed to the first
        occurrence's task), ordered to match ``prompts``.
    """
    existing = {task.generation_prompt: task for task in persistence.read_generation_task_file(tasks_path)}

    built: list[GenerationTask] = []
    seen: set[str] = set()
    for prompt in prompts:
        if prompt in seen:
            # Duplicate within this call: reuse the first occurrence's task rather
            # than emitting a second one — keeps one task per unique prompt.
            continue
        seen.add(prompt)
        if prompt in existing:
            built.append(existing[prompt])  # frozen: reuse, never rebuild
        else:
            built.append(build_generation_task(prompt, tasks_path=tasks_path))
    return built


async def generate_one(
    task: GenerationTask,
    agent: _GenAgent,
    *,
    results_path: str | Path | None = None,
    failures_path: str | Path | None = None,
) -> GenerationResult | GenerationFailure:
    """Call a single generation agent for a single task.

    Returns a :class:`GenerationResult`, or a :class:`GenerationFailure` if the
    agent raised - a caller looping over :func:`generate_one` directly tells the
    two apart by type.

    Concurrency is the agent's concern: if it was given ``max_concurrency`` (or a
    shared limiter), this call self-throttles.  That means a hand-rolled loop over
    :func:`generate_one` gets the same bounding as :func:`generate_all` for free.

    Unlike :func:`generate_all`, this does **not** resume - it always runs.  Resume
    is a batch concern; see :func:`generate_all`.

    If ``results_path`` / ``failures_path`` is given, the produced
    :class:`GenerationResult` / :class:`GenerationFailure` is appended to that file
    as soon as it lands (see :mod:`tournament_eval.persistence`).  Only the
    ``agent.run`` call is guarded, so a persistence error propagates rather than
    masquerading as a failure.
    """
    result: GenerationResult | GenerationFailure
    try:
        run = await agent.run(task.generation_prompt, infer_name=False)
        result = _generation_result_from_run(task, agent, run)
    except Exception as err:
        result = GenerationFailure(
            generation_task_id=task.id,
            author=_resolve_author(agent),
            error_type=type(err).__name__,
            message=str(err),
        )

    if isinstance(result, GenerationResult):
        if results_path is not None:
            persistence.append_record(results_path, result)
    elif failures_path is not None:
        persistence.append_record(failures_path, result)
    return result


def _generation_result_from_run(
    task: GenerationTask,
    agent: _GenAgent,
    run: AgentRunResult[str],
) -> GenerationResult:
    return GenerationResult(
        id=uuid.uuid4(),
        generation_task_id=task.id,
        generation_prompt=task.generation_prompt,
        output=run.output,
        reasoning=_extract_reasoning(run.new_messages()),
        author=_resolve_author(agent),
        metadata=_usage_to_metadata(run.usage),
    )


async def generate_all(
    tasks: list[GenerationTask],
    agents: Iterable[_GenAgent],
    *,
    results_path: str | Path | None = None,
    failures_path: str | Path | None = None,
) -> tuple[list[GenerationResult], list[GenerationFailure]]:
    """Run every generation agent against every task to produce GenerationResults.

    All calls are fired concurrently with :func:`asyncio.gather`; per-agent
    bounding is the agent's ``max_concurrency``.  No lifecycle management is
    needed - ``agent.run`` is a plain coroutine.

    **Resume is automatic** when ``results_path`` is set: the file is read back
    first and every ``(generation_task_id, author)`` pair already recorded there is
    skipped, so re-running the script finishes an interrupted run.  Records loaded
    for a task or agent absent from the current run are ignored.  Only *successes*
    are skipped; a pair that previously failed is retried.

    Parameters
    ----------
    tasks : list[GenerationTask]
        The creative tasks to evaluate.
    agents : Iterable[Agent[None, str]]
        The generation agents (``output_type=str``).  Any iterable; materialised
        once, so a single-pass iterator is safe.  Each must resolve to a distinct
        author (``agent.name`` or model name) - a collision raises.
    results_path : str | Path | None
        If set, the GenerationResults file: each success is streamed to it as it
        lands, and on entry it is read back to skip pairs already done (resume).
        Persisting the *tasks* is yours (``persistence.append_record``).
    failures_path : str | Path | None
        If set, the GenerationFailures file: each failure is appended as it lands.
        It is a write-only log - not read for resume - so across reruns it may hold
        several entries for a pair that kept failing.

    Returns
    -------
    tuple[list[GenerationResult], list[GenerationFailure]]
        A partition of every (task, agent) pair: ``results`` is every pair that
        now has a success (loaded from ``results_path`` plus produced this call),
        and ``failures`` is every pair still without one.
    """
    agents_list = list(agents)
    _check_distinct_authors(agents_list)
    loaded = persistence.read_generation_result_file(results_path) if results_path is not None else []
    valid_ids = {task.id for task in tasks}
    valid_authors = {_resolve_author(agent) for agent in agents_list}
    done: dict[tuple[uuid.UUID, str], GenerationResult] = {
        (result.generation_task_id, result.author): result
        for result in loaded
        if result.generation_task_id in valid_ids and result.author in valid_authors
    }
    coros = [
        generate_one(task, agent, results_path=results_path, failures_path=failures_path)
        for task in tasks
        for agent in agents_list
        if (task.id, _resolve_author(agent)) not in done
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

    A thin fan-out over :func:`build_ranking_task` - one RankingTask per task that
    produced at least one output.

    **Resume is per-task** when ``tasks_path`` is set: the file is read back and any
    task that already has a persisted RankingTask reuses it as-is; only tasks
    without one are (re)built and appended.  This is deliberately *frozen* - a
    task's RankingTask, once built, is never rebuilt, because rebuilding would
    re-shuffle the aliases and orphan every ranking already collected against it.

    The consequence: a generation that lands *after* its task's RankingTask was
    built (e.g. a flaky model that only succeeded on a later resume) will **not**
    be added to that task's candidate set.  So build ranking tasks only once
    generation is fully done - run :func:`generate_all` until its ``failures`` are
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
    agent: _RankAgent,
    generation_lookup: dict[uuid.UUID, GenerationResult],
    *,
    template: RankingTemplate | None = None,
    results_path: str | Path | None = None,
    failures_path: str | Path | None = None,
) -> RankingResult | RankingFailure:
    """Call a single ranking agent for a single RankingTask.

    Returns a :class:`RankingResult`, or a :class:`RankingFailure` if the ranking
    agent raised or returned a malformed ranking (failed pydantic validation, or
    failed the dynamic alias check).

    The ``template`` (a :class:`RankingTemplate`, default
    :class:`~tournament_eval.ranking.DefaultRankingTemplate`) owns the prompt, the
    response model (the agent's ``output_type``), and the dynamic alias
    validation.  The agent's ``output_type`` must match ``template.response_model``.

    As with :func:`generate_one`, concurrency is bounded by the agent itself, there
    is no resume - it always runs - and if ``results_path`` / ``failures_path`` is
    given, the produced record is appended as soon as it lands.  Only the ranking
    call is guarded, so a persistence error propagates rather than masquerading as
    a failure.
    """
    template = template or _DEFAULT_TEMPLATE
    candidates = {alias: generation_lookup[gen_id] for alias, gen_id in ranking_task.generations.items()}
    prompt = template.render(ranking_task, candidates)
    result: RankingResult | RankingFailure

    try:
        run = await agent.run(prompt, infer_name=False)
        parsed = template.parse(run.output, set(ranking_task.generations.keys()))
        uuid_ranking = [ranking_task.generations[alias] for alias in parsed.ranking]
        result = RankingResult(
            id=uuid.uuid4(),
            ranking_task_id=ranking_task.id,
            ranking_prompt=prompt,
            author=_resolve_author(agent),
            raw_model_ranking=parsed.ranking,
            ranking=uuid_ranking,
            reasoning=parsed.reasoning,
            raw_response=run.output.model_dump_json(),
            metadata=_usage_to_metadata(run.usage),
        )
    except Exception as exc:
        result = RankingFailure(
            ranking_task_id=ranking_task.id,
            author=_resolve_author(agent),
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
    agents: Iterable[_RankAgent],
    *,
    template: RankingTemplate | None = None,
    results_path: str | Path | None = None,
    failures_path: str | Path | None = None,
) -> tuple[list[RankingResult], list[RankingFailure]]:
    """Run every ranking agent against every RankingTask.

    For each RankingTask, the full ranking prompt is built by ``template`` (a
    :class:`RankingTemplate`, default
    :class:`~tournament_eval.ranking.DefaultRankingTemplate`) from the
    ``ranking_prompt`` and the candidates.  Each agent receives the assembled
    prompt and returns a structured ranking (its ``output_type`` must match
    ``template.response_model``).  No lifecycle management - ``agent.run`` is a
    plain coroutine.

    **Resume is automatic** when ``results_path`` is set, exactly as in
    :func:`generate_all` but keyed on ``(ranking_task_id, author)``: already-recorded
    successes are skipped and returned alongside what's produced this call.  Records
    loaded for a ranking task or agent absent from the current run are ignored.

    Parameters
    ----------
    ranking_tasks : list[RankingTask]
        The scenarios to be ranked.
    generation_results : list[GenerationResult]
        The outputs to be ranked (looked up by the aliases in each
        RankingTask).
    agents : Iterable[Agent[None, BaseModel]]
        The ranking agents (``output_type=template.response_model``).  Any
        iterable; materialised once.  Each must resolve to a distinct author.
    template : RankingTemplate | None
        The ranking contract (prompt, response model, validation).  ``None`` uses
        :class:`~tournament_eval.ranking.DefaultRankingTemplate` - a strict total
        order with no ties.
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
        A partition of every (ranking_task, agent) pair: ``results`` is every pair
        that now has a success (loaded from ``results_path`` plus produced this
        call), and ``failures`` every pair still without one.
    """
    agents_list = list(agents)
    _check_distinct_authors(agents_list)
    generation_lookup = {result.id: result for result in generation_results}
    loaded = persistence.read_ranking_result_file(results_path) if results_path is not None else []
    valid_ids = {ranking_task.id for ranking_task in ranking_tasks}
    valid_authors = {_resolve_author(a) for a in agents_list}
    done: dict[tuple[uuid.UUID, str], RankingResult] = {
        (result.ranking_task_id, result.author): result
        for result in loaded
        if result.ranking_task_id in valid_ids and result.author in valid_authors
    }
    coros = [
        rank_one(
            ranking_task,
            agent,
            generation_lookup,
            template=template,
            results_path=results_path,
            failures_path=failures_path,
        )
        for ranking_task in ranking_tasks
        for agent in agents_list
        if (ranking_task.id, _resolve_author(agent)) not in done
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
