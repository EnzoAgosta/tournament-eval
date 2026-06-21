"""Pure-async orchestration for the tournament evaluation flow.

Three stages, each fired concurrently and each returning ``(results, failures)``
so one failing agent never sinks a run:

1. :func:`generate_all` - every agent produces an output for every task.
2. :func:`build_ranking_tasks` - group the outputs per task and assign anonymised
   aliases (A, B, C...) so authorship is hidden from the rankers.
3. :func:`rank_all` - every agent ranks every task's candidates.

Build generation tasks first with :func:`build_generation_tasks` (rerun-stable) or
construct :class:`~tournament_eval.models.GenerationTask` directly, then run the
three stages.

A thin wrapper over `pydantic-ai <https://ai.pydantic.dev>`_: you construct and
own the :class:`pydantic_ai.Agent` instances - model, settings, thinking, retries,
concurrency all configured the pydantic-ai way - and hand them in.
``generate_all`` / ``rank_all`` call ``agent.run`` and map each
:class:`pydantic_ai.AgentRunResult` onto the tournament data model (output,
reasoning trace, usage → metadata).  No client lifecycle: ``agent.run`` is a plain
coroutine, so there's no ``async with``.

Generation agents produce ``str``; ranking agents produce the ranking
:class:`~tournament_eval.ranking.RankingTemplate`'s ``response_model`` (a pydantic
model) — wire one with ``Agent(model, output_type=template.response_model)``.

**Resume is automatic.**  When a ``results_path`` is given, :func:`generate_all` /
:func:`rank_all` read it back first and skip every ``(task, author)`` pair that
already succeeded, running only what's left and returning the *complete* set
(loaded plus newly produced).  A partial run is finished by just running the
script again.  Successes are skipped; failures are always retried.  The return is
a clean partition: ``results`` is every pair with a success, ``failures`` every
pair still without one.

**Author labels are load-bearing.**  Each result's ``author`` is resolved from the
agent (``agent.name`` if set, else the model name) and, with the task id, is the
resume key, so two agents in a run must resolve to distinct authors.
:func:`generate_all` / :func:`rank_all` check this up front and raise on collision;
set distinct ``agent.name`` values (or use distinct models) to disambiguate e.g.
one model run at two temperatures.

Aggregation - collapsing the per-ranker rankings into a leaderboard - lives in
:mod:`tournament_eval.aggregation` and is intentionally a convenience, not a
mandate; the pipeline hands you validated ranked data to score however you like.
"""

import asyncio
import random
import uuid
import warnings
from collections.abc import Callable, Iterable
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


_GenAgent = Agent[None, str]
_RankAgent = Agent[None, BaseModel]


def _resolve_author[DepsT, OutT](agent: Agent[DepsT, OutT]) -> str:
    """Resolve an agent's stable author label: ``agent.name`` if set, else the model name.

    Use ``agent.name`` when two agents share a model id (e.g. one model at two
    temperatures) and must stay distinct in the resume key; otherwise the model
    name dedups naturally.  Raises if neither is available — the label is
    load-bearing for resume and de-anonymisation.
    """
    name = agent.name
    if name:
        return name
    model = agent.model
    if isinstance(model, Model):
        return model.model_name
    if model:
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


def _check_output_types(agents: Iterable[_RankAgent], template: RankingTemplate) -> None:
    """Raise if any ranking agent's ``output_type`` isn't the template's ``response_model``.

    The template owns the response shape and its parser; an agent wired to a
    different ``output_type`` would only surface later as a per-task
    :class:`RankingFailure`.  Checking up front makes it one loud error at the
    call boundary.
    """
    expected = template.response_model
    mismatched = [_resolve_author(agent) for agent in agents if agent.output_type is not expected]
    if mismatched:
        raise ValueError(
            f"Ranking agents {sorted(mismatched)!r} have an output_type that isn't the template's "
            f"response_model ({expected.__name__}). Wire each with Agent(model, output_type=template.response_model)."
        )


def _extract_reasoning(messages: list[ModelMessage]) -> str | None:
    """Pull the reasoning trace out of a run's new messages.

    pydantic-ai normalises each provider's thinking into :class:`ThinkingPart`
    objects on the model responses; we join their ``content``.  ``None`` when the
    model produced no thinking parts.  Some providers surface raw reasoning only in
    ``provider_details``; we don't chase that yet.
    """
    chunks: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ThinkingPart) and part.content:
                    chunks.append(part.content)
    return "\n".join(chunks) if chunks else None


def _usage_to_metadata(usage: RunUsage) -> dict[str, object]:
    """Stamp a run's usage into the result ``metadata`` dict.

    Token/request counts come from pydantic-ai's :class:`RunUsage`; cost ($) is a
    later addition (``RunUsage`` implements ``genai_prices``' ``AbstractUsage``).
    Keys are stable across reruns so persisted records stay readable.
    """
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "requests": usage.requests,
    }


def _notify[OutT](hook: Callable[[OutT], None] | None, outcome: OutT, *, stage: str) -> None:
    """Invoke a per-outcome progress callback, swallowing a buggy hook's error.

    A progress hook is the user's concern, not the run's, so a broken one mustn't
    sink a batch: a raised exception becomes a :class:`UserWarning` and the run
    continues.  ``stage`` labels the warning so it's clear which stage's hook broke.
    """
    if hook is None:
        return
    try:
        hook(outcome)
    except Exception as err:
        warnings.warn(
            f"{stage} progress callback raised {type(err).__name__}: {err}; "
            "the run continues but the hook is broken and should be fixed.",
            stacklevel=2,
        )


def _extract_ranking_failure_details(
    exc: BaseException,
    *,
    model_output: BaseModel | None = None,
) -> dict[str, object] | None:
    """Pull whatever structured diagnostic payload a ranking failure exposes.

    Failures reach here from two places inside :func:`rank_one`:

    * the agent run itself raised (pydantic validation exhausted its retries, the
      model returned an empty response, called the wrong tool, hit an API error, …);
    * the agent run *succeeded* but the template's dynamic alias check raised — in
      which case ``model_output`` is the validated response the model actually
      emitted, and we capture it so a user can see what it answered.

    The exception chain is best-effort probed; any attribute that's missing or
    raises leaves its key out rather than failing.  Returns ``None`` when nothing
    structured was found (the plain ``message`` is then the only breadcrumb).

    Keys, populated when available: ``model_output`` (the emitted ranking, as
    ``model_dump_json``), ``validation_errors`` (pydantic's error list, each with
    the failing field's ``input``), ``cause_type`` / ``cause_message`` (the wrapped
    exception), ``body`` (pydantic-ai's response body when it surfaces one).
    """
    details: dict[str, object] = {}

    if model_output is not None:
        details["model_output"] = model_output.model_dump_json()

    cause = exc.__cause__ or exc.__context__
    if cause is not None:
        details["cause_type"] = type(cause).__name__
        details["cause_message"] = str(cause)

        errors = getattr(cause, "errors", None)
        if callable(errors):
            try:
                normalised: list[dict[str, object]] = []
                for err in errors():
                    err = dict(err)
                    loc = err.get("loc")
                    if isinstance(loc, tuple):
                        err["loc"] = list(loc)
                    normalised.append(err)
                details["validation_errors"] = normalised
            except Exception:
                pass

    body = getattr(exc, "body", None)
    if body is not None:
        details["body"] = body

    return details or None


def _per_task_seed(base_seed: int, generation_task_id: uuid.UUID) -> int:
    """Derive a distinct, reproducible shuffle seed for one ranking task.

    The base seed is mixed with the task's ``generation_task_id`` so two tasks
    built from the same base seed and the same candidate ordering still get
    distinct shuffles.  Without this, alias ``A`` would map to the same author in
    every task, leaking authorship to a ranker that sees several tasks.  Task ids
    are the resume-stability anchor (stable across rebuilds), so deriving from
    them keeps the shuffle reproducible: delete the ranking-tasks file and rebuild
    and the per-task shuffles come back identical.  XOR with a fixed operand is
    injective, so distinct task ids yield distinct seeds.
    """
    return base_seed ^ generation_task_id.int


def build_generation_task(
    prompt: str,
    *,
    id: uuid.UUID | None = None,
    tasks_path: str | Path | None = None,
) -> GenerationTask:
    """Build one GenerationTask from a prompt.

    Stamps a fresh :class:`uuid.UUID` (or the given ``id``) and, if ``tasks_path``
    is set, appends the task to that file as soon as it's built.

    **No resume** — always builds a new task (fresh id when ``id`` is unset).  Use
    :func:`build_generation_tasks` for rerun stability; a random id here makes
    :func:`generate_all` resume silently no-op for this task across reruns.

    Parameters
    ----------
    prompt : str
        The generation prompt for this task.
    id : uuid.UUID | None
        Optional explicit id; defaults to a fresh ``uuid4``.
    tasks_path : str | Path | None
        If set, the task is appended to this generation-tasks file as built.

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

    ``tasks_path`` is **required** — persistence is the whole point: on a rerun
    the file is read back and any prompt that already has a persisted task reuses
    it (and its id); only prompts without one are built (fresh ``uuid4``) and
    appended.  This is what makes "just rerun the script" actually resume: ids
    stay stable, so :func:`generate_all`'s ``(task_id, author)`` resume matches and
    skips already-done pairs.

    Tasks match prompts **by ``generation_prompt``**, so duplicate prompts
    collapse to one task (and one GenerationResult per author) — two identical
    prompts are one task, not two, and you don't want two identical candidates
    behind two aliases in a ranking.  For deliberate distinct duplicates, use
    :func:`build_generation_task` with an explicit ``id``.

    Parameters
    ----------
    prompts : Iterable[str]
        The generation prompts.  Any iterable; materialised once, so an open text
        file is fine — strip trailing newlines yourself if they shouldn't be part
        of the prompt.
    tasks_path : str | Path
        The generation-tasks file: streamed here as built, read back for resume.

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
            continue
        seen.add(prompt)
        if prompt in existing:
            built.append(existing[prompt])
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
    agent raised — tell them apart by type.  Does **not** resume (always runs);
    resume is a batch concern, see :func:`generate_all`.

    Concurrency is the agent's: if it has ``max_concurrency`` (or a shared
    limiter), this call self-throttles, so a hand-rolled loop over
    :func:`generate_one` gets the same bounding as :func:`generate_all`.

    If ``results_path`` / ``failures_path`` is given, the produced record is
    appended as soon as it lands.  Only the ``agent.run`` call is guarded, so a
    persistence error propagates rather than masquerading as a failure.
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


async def _generate_and_notify(
    task: GenerationTask,
    agent: _GenAgent,
    *,
    results_path: str | Path | None,
    failures_path: str | Path | None,
    on_result: Callable[[GenerationResult], None] | None,
    on_failure: Callable[[GenerationFailure], None] | None,
) -> GenerationResult | GenerationFailure:
    """Run :func:`generate_one` and fire the matching progress callback as it lands.

    The callback fires *after* ``generate_one`` returns, which is *after* the
    record is appended to disk (``append_record`` is sync/await-free), so a
    notified outcome is always a persisted one.  A raising hook is swallowed by
    :func:`_notify`.  Internal: :func:`generate_one` stays callback-free (its
    caller already gets the value synchronously); this wrapper is the per-coroutine
    seam that makes :func:`generate_all`'s ``asyncio.gather`` observable
    outcome-by-outcome.
    """
    outcome = await generate_one(task, agent, results_path=results_path, failures_path=failures_path)
    if isinstance(outcome, GenerationResult):
        _notify(on_result, outcome, stage="generate_all")
    else:
        _notify(on_failure, outcome, stage="generate_all")
    return outcome


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
    on_result: Callable[[GenerationResult], None] | None = None,
    on_failure: Callable[[GenerationFailure], None] | None = None,
) -> tuple[list[GenerationResult], list[GenerationFailure]]:
    """Run every generation agent against every task to produce GenerationResults.

    All calls fire concurrently with :func:`asyncio.gather`; per-agent bounding is
    the agent's ``max_concurrency``.  No lifecycle — ``agent.run`` is a plain
    coroutine.

    **Resume is automatic** when ``results_path`` is set: the file is read back
    first and every ``(generation_task_id, author)`` pair already recorded there is
    skipped, so re-running finishes an interrupted run.  Records for a task or
    agent absent from the current run are ignored.  Only *successes* are skipped;
    a pair that previously failed is retried.

    Parameters
    ----------
    tasks : list[GenerationTask]
        The creative tasks to evaluate.
    agents : Iterable[Agent[None, str]]
        The generation agents (``output_type=str``).  Any iterable; materialised
        once.  Each must resolve to a distinct author — a collision raises.
    results_path : str | Path | None
        If set, the GenerationResults file: each success is streamed to it as it
        lands, and on entry it's read back to skip pairs already done (resume).
        Persisting the *tasks* is yours (``persistence.append_record``).
    failures_path : str | Path | None
        If set, the GenerationFailures file: each failure is appended as it lands.
        A write-only log — not read for resume — so across reruns it may hold
        several entries for a pair that kept failing.
    on_result : Callable[[GenerationResult], None] | None
        Optional sync callback fired with each :class:`GenerationResult` as it
        lands, *after* it's persisted to ``results_path`` (so "notified" means
        "safely on disk").  Use it for progress reporting (print, tqdm).  Not fired
        for successes loaded from ``results_path`` on resume.  A raising hook is
        swallowed into a :class:`UserWarning` so a broken callback can't sink the
        batch.
    on_failure : Callable[[GenerationFailure], None] | None
        Optional sync callback fired with each :class:`GenerationFailure` as it
        lands (after it's appended to ``failures_path``).  Same semantics as
        ``on_result`` for resume and error-swalling.

    Returns
    -------
    tuple[list[GenerationResult], list[GenerationFailure]]
        A partition of every (task, agent) pair: ``results`` is every pair that
        now has a success (loaded plus produced this call), ``failures`` every
        pair still without one.
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
        _generate_and_notify(
            task,
            agent,
            results_path=results_path,
            failures_path=failures_path,
            on_result=on_result,
            on_failure=on_failure,
        )
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
    ranker's position bias (e.g. always picking "A") doesn't track authorship.
    The originating task is taken from ``results[0].generation_task_id`` (all
    results must be for the same task).

    Parameters
    ----------
    results : list[GenerationResult]
        The outputs for one GenerationTask.  Must be non-empty.
    ranking_prompt : str
        The ranking instructions to embed in the RankingTask.
    tasks_path : str | Path | None
        If set, the RankingTask is appended to this file as soon as it's built.
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

    **Resume is per-task** when ``tasks_path`` is set: the file is read back and
    any task that already has a persisted RankingTask reuses it as-is; only tasks
    without one are (re)built and appended.  This is deliberately *frozen* — a
    task's RankingTask, once built, is never rebuilt, since rebuilding would
    re-shuffle the aliases and orphan every ranking already collected against it.

    The consequence: a generation that lands *after* its task's RankingTask was
    built (e.g. a flaky model that only succeeded on a later resume) will **not**
    be added to that task's candidate set.  So build ranking tasks only once
    generation is fully done — run :func:`generate_all` until its ``failures`` are
    empty first.  To deliberately rebuild, delete the ranking-tasks file (and any
    rankings already collected) first.

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
        resume.
    random_seed : int | None
        Base seed for the per-task alias shuffle; ``None`` for nondeterministic.
        When set, a distinct seed is derived per task from this and the task's
        ``generation_task_id`` (see :func:`_per_task_seed`) so a fixed base seed
        doesn't make alias ``A`` track the same author across tasks.

    Returns
    -------
    list[RankingTask]
        One RankingTask per task that has at least one GenerationResult, each
        mapping aliases (A, B, C...) to GenerationResult IDs.
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
            ranking_tasks.append(existing[task.id])
        else:
            per_task_seed = None if random_seed is None else _per_task_seed(random_seed, task.id)
            ranking_tasks.append(
                build_ranking_task(results, ranking_prompt, tasks_path=tasks_path, random_seed=per_task_seed)
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

    Returns a :class:`RankingResult`, or a :class:`RankingFailure` if the agent
    raised or returned a malformed ranking (failed pydantic validation, or failed
    the dynamic alias check).

    The ``template`` (default :class:`~tournament_eval.ranking.DefaultRankingTemplate`)
    owns the prompt, the response model (the agent's ``output_type``), and the
    alias validation; the agent's ``output_type`` must match
    ``template.response_model``.

    As with :func:`generate_one`: concurrency is bounded by the agent, there is no
    resume (always runs), and if ``results_path`` / ``failures_path`` is given the
    record is appended as it lands.  Only the ranking call is guarded, so a
    persistence error propagates rather than masquerading as a failure.
    """
    template = template or _DEFAULT_TEMPLATE
    candidates = {alias: generation_lookup[gen_id] for alias, gen_id in ranking_task.generations.items()}
    prompt = template.render(ranking_task, candidates)
    result: RankingResult | RankingFailure

    run: AgentRunResult[BaseModel] | None = None
    try:
        run = await agent.run(prompt, infer_name=False)
        parsed = template.parse(run.output, set(ranking_task.generations.keys()))
        uuid_ranking = [ranking_task.generations[alias] for alias in parsed.ranking]
        result = RankingResult(
            id=uuid.uuid4(),
            ranking_task_id=ranking_task.id,
            generation_task_id=ranking_task.generation_task_id,
            ranking_prompt=prompt,
            author=_resolve_author(agent),
            raw_model_ranking=parsed.ranking,
            ranking=uuid_ranking,
            ranking_reasoning=parsed.reasoning,
            reasoning=_extract_reasoning(run.new_messages()),
            raw_response=run.output.model_dump_json(),
            metadata=_usage_to_metadata(run.usage),
        )
    except Exception as exc:
        result = RankingFailure(
            ranking_task_id=ranking_task.id,
            generation_task_id=ranking_task.generation_task_id,
            author=_resolve_author(agent),
            error_type=type(exc).__name__,
            message=str(exc),
            ranking_prompt=prompt,
            details=_extract_ranking_failure_details(exc, model_output=run.output if run is not None else None),
        )

    if isinstance(result, RankingResult):
        if results_path is not None:
            persistence.append_record(results_path, result)
    elif failures_path is not None:
        persistence.append_record(failures_path, result)
    return result


async def _rank_and_notify(
    ranking_task: RankingTask,
    agent: _RankAgent,
    generation_lookup: dict[uuid.UUID, GenerationResult],
    *,
    template: RankingTemplate,
    results_path: str | Path | None,
    failures_path: str | Path | None,
    on_result: Callable[[RankingResult], None] | None,
    on_failure: Callable[[RankingFailure], None] | None,
) -> RankingResult | RankingFailure:
    """Run :func:`rank_one` and fire the matching progress callback as it lands.

    Mirrors :func:`_generate_and_notify` for the ranking stage: the callback fires
    *after* :func:`rank_one` returns (so after the record is persisted), and a
    raising hook is swallowed by :func:`_notify`.  Internal — :func:`rank_one`
    stays callback-free.
    """
    outcome = await rank_one(
        ranking_task,
        agent,
        generation_lookup,
        template=template,
        results_path=results_path,
        failures_path=failures_path,
    )
    if isinstance(outcome, RankingResult):
        _notify(on_result, outcome, stage="rank_all")
    else:
        _notify(on_failure, outcome, stage="rank_all")
    return outcome


async def rank_all(
    ranking_tasks: list[RankingTask],
    generation_results: list[GenerationResult],
    agents: Iterable[_RankAgent],
    *,
    template: RankingTemplate | None = None,
    results_path: str | Path | None = None,
    failures_path: str | Path | None = None,
    on_result: Callable[[RankingResult], None] | None = None,
    on_failure: Callable[[RankingFailure], None] | None = None,
) -> tuple[list[RankingResult], list[RankingFailure]]:
    """Run every ranking agent against every RankingTask.

    For each RankingTask, ``template`` (default
    :class:`~tournament_eval.ranking.DefaultRankingTemplate`) builds the full
    prompt from ``ranking_prompt`` and the candidates.  Each agent receives it and
    returns a structured ranking (its ``output_type`` must match
    ``template.response_model``).  No lifecycle — ``agent.run`` is a plain
    coroutine.

    **Resume is automatic** when ``results_path`` is set, as in :func:`generate_all`
    but keyed on ``(ranking_task_id, author)``: already-recorded successes are
    skipped and returned alongside what's produced this call.  Records for a
    ranking task or agent absent from the current run are ignored.

    Parameters
    ----------
    ranking_tasks : list[RankingTask]
        The scenarios to be ranked.
    generation_results : list[GenerationResult]
        The outputs to be ranked (looked up by the aliases in each RankingTask).
    agents : Iterable[Agent[None, BaseModel]]
        The ranking agents (``output_type=template.response_model``).  Any
        iterable; materialised once.  Each must resolve to a distinct author.
    template : RankingTemplate | None
        The ranking contract (prompt, response model, validation).  ``None`` uses
        :class:`~tournament_eval.ranking.DefaultRankingTemplate` — a strict total
        order with no ties.
    results_path : str | Path | None
        If set, the RankingResults file: each success is streamed to it as it
        lands, and on entry it's read back to skip pairs already done (resume).
        The ranking *tasks* are persisted separately
        (``build_ranking_tasks(..., tasks_path=...)``).
    failures_path : str | Path | None
        If set, the RankingFailures file: each failure is appended as it lands.
        A write-only log (not read for resume).
    on_result : Callable[[RankingResult], None] | None
        Optional sync callback fired with each :class:`RankingResult` as it lands,
        *after* it's persisted to ``results_path``.  Same semantics as
        :func:`generate_all`'s ``on_result``: not fired for resume-loaded
        successes, a raising hook swallowed into a :class:`UserWarning`.
    on_failure : Callable[[RankingFailure], None] | None
        Optional sync callback fired with each :class:`RankingFailure` as it lands
        (after it's appended to ``failures_path``); see ``on_result``.

    Returns
    -------
    tuple[list[RankingResult], list[RankingFailure]]
        A partition of every (ranking_task, agent) pair: ``results`` is every pair
        that now has a success (loaded plus produced this call), ``failures``
        every pair still without one.
    """
    template = template or _DEFAULT_TEMPLATE
    agents_list = list(agents)
    _check_distinct_authors(agents_list)
    _check_output_types(agents_list, template)
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
        _rank_and_notify(
            ranking_task,
            agent,
            generation_lookup,
            template=template,
            results_path=results_path,
            failures_path=failures_path,
            on_result=on_result,
            on_failure=on_failure,
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


def deanonymize_ranking(
    ranking_result: RankingResult,
    generations: Iterable[GenerationResult],
) -> list[str]:
    """Resolve a :class:`RankingResult`'s alias-space ranking into author names.

    Maps ``ranking_result.ranking`` (a list of :class:`GenerationResult` ids, best
    first) to each one's ``author`` via ``generations``.  This is the
    de-anonymization step before aggregation, and the inverse of the aliasing
    :func:`build_ranking_task` did at ranking-task build time.

    Pure: no state, no I/O, no mutation.  Raises :class:`KeyError` if a ranked id
    isn't among ``generations`` — a ranking should only reference ids from its own
    task's candidate set, so that's a data-integrity error worth surfacing loudly.

    Parameters
    ----------
    ranking_result : RankingResult
        The ranking to de-anonymize.
    generations : Iterable[GenerationResult]
        The generation outputs (e.g. what :func:`generate_all` returned, or what
        ``read_generation_result_file`` loaded back).

    Returns
    -------
    list[str]
        Authors in ranked order, best first — e.g.
        ``["gpt-4o", "claude-sonnet-4-6", ...]``.
    """
    author_by_id = {generation.id: generation.author for generation in generations}
    return [author_by_id[generation_id] for generation_id in ranking_result.ranking]
