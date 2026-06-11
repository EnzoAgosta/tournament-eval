"""Pure-function orchestration for the tournament evaluation flow.

Typical usage::

    from tournament_eval.models import GenerationTask
    from tournament_eval.llm import OpenAILLMClient, AnthropicLLMClient
    from tournament_eval.orchestration import (
        build_ranking_tasks,
        generate_all,
        rank_all,
    )

    tasks = [GenerationTask(...), GenerationTask(...)]
    clients = [OpenAILLMClient(...), AnthropicLLMClient(...)]

    generation_results = await generate_all(tasks, clients)

    ranking_tasks = build_ranking_tasks(
        tasks=tasks,
        generation_results=generation_results,
        ranking_prompt="Rank these translations by fluency and accuracy.",
    )

    ranking_results = await rank_all(
        ranking_tasks=ranking_tasks,
        generation_results=generation_results,
        clients=clients,
    )
"""

import asyncio
import random
import uuid
from collections.abc import Sequence

from tournament_eval.llm import LLMClient
from tournament_eval.models import (
    GenerationResult,
    GenerationTask,
    LetterGenerator,
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


async def _generate_one(
    task: GenerationTask,
    client: LLMClient,
) -> tuple[uuid.UUID, str, GenerationResult | BaseException]:
    """Call a single client for a single task.

    Returns ``(task_id, client_name, result_or_error)`` so the caller can
    correlate failures even when exceptions are returned alongside successes.
    """
    try:
        raw = await client.generate(task.generation_prompt)
    except BaseException as exc:
        return task.id, client.name, exc
    return (
        task.id,
        client.name,
        GenerationResult(
            id=uuid.uuid4(),
            task_id=task.id,
            generation_prompt=task.generation_prompt,
            raw_response=raw,
            output=raw,  # no cleaning applied by default
            author=client.name,
        ),
    )


async def generate_all(
    tasks: list[GenerationTask],
    clients: Sequence[LLMClient],
) -> tuple[list[GenerationResult], dict[tuple[uuid.UUID, str], BaseException]]:
    """Run every client against every task to produce GenerationResults.

    All calls are fired concurrently.

    Parameters
    ----------
    tasks : list[GenerationTask]
        The creative tasks to evaluate.
    clients : list[LLMClient]
        The models that will generate outputs.

    Returns
    -------
    tuple[list[GenerationResult], dict[tuple[uuid.UUID, str], BaseException]]
        A pair of ``(results, failures)``.

        ``results`` contains one :class:`GenerationResult` per successful
        (task, client) pair.  ``output`` is set to ``raw_response``; no
        cleaning is applied.

        ``failures`` maps ``(task_id, client_name)`` to the exception that
        was raised so the caller can decide what to do.
    """
    coros = [_generate_one(task, client) for task in tasks for client in clients]
    outcomes = await asyncio.gather(*coros)

    generation_results: list[GenerationResult] = []
    failures: dict[tuple[uuid.UUID, str], BaseException] = {}
    for task_id, client_name, result in outcomes:
        if isinstance(result, BaseException):
            failures[(task_id, client_name)] = result
            continue
        generation_results.append(result)
    return generation_results, failures


def build_ranking_tasks(
    tasks: list[GenerationTask],
    generation_results: list[GenerationResult],
    ranking_prompt: str,
    *,
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

    return ranking_tasks


async def _rank_one(
    ranking_task: RankingTask,
    client: LLMClient,
    generation_lookup: dict[uuid.UUID, GenerationResult],
) -> tuple[uuid.UUID, str, RankingResult | BaseException]:
    """Call a single client as judge for a single RankingTask.

    Returns ``(ranking_task_id, client_name, result_or_error)``.
    """
    prompt = _build_judge_prompt(ranking_task, generation_lookup)

    try:
        structured = await client.generate_structured(prompt, _DEFAULT_RANKING_SCHEMA)
    except BaseException as exc:
        return ranking_task.id, client.name, exc

    try:
        raw_ranking, reasoning = _parse_ranking_response(
            structured.data,
            set(ranking_task.generations.keys()),
        )
    except BaseException as exc:
        return ranking_task.id, client.name, exc

    uuid_ranking = [ranking_task.generations[alias] for alias in raw_ranking]

    return (
        ranking_task.id,
        client.name,
        RankingResult(
            id=uuid.uuid4(),
            ranking_task_id=ranking_task.id,
            ranking_prompt=prompt,
            author=client.name,
            raw_model_ranking=raw_ranking,
            ranking=uuid_ranking,
            reasoning=reasoning,
            raw_response=structured.raw,
        ),
    )


async def rank_all(
    ranking_tasks: list[RankingTask],
    generation_results: list[GenerationResult],
    clients: Sequence[LLMClient],
) -> tuple[list[RankingResult], dict[tuple[uuid.UUID, str], BaseException]]:
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

    Returns
    -------
    tuple[list[RankingResult], dict[tuple[uuid.UUID, str], BaseException]]
        A pair of ``(results, failures)``.

        ``results`` contains one :class:`RankingResult` per successful
        (ranking_task, client) pair.

        ``failures`` maps ``(ranking_task_id, client_name)`` to the
        exception or validation error that was raised.
    """
    generation_lookup = _build_generation_lookup(generation_results)

    coros = [
        _rank_one(ranking_task, client, generation_lookup)
        for ranking_task in ranking_tasks
        for client in clients
    ]
    outcomes = await asyncio.gather(*coros)

    ranking_results: list[RankingResult] = []
    failures: dict[tuple[uuid.UUID, str], BaseException] = {}
    for task_id, client_name, result in outcomes:
        if isinstance(result, BaseException):
            failures[(task_id, client_name)] = result
            continue
        ranking_results.append(result)
    return ranking_results, failures
