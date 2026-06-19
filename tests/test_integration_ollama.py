"""Integration smoke test against a real local Ollama server.

Skipped by default — it's marked ``integration``, which ``pyproject.toml``
deselects by default (``addopts = ["-m", "not integration"]``) so a normal
``uv run pytest`` stays hermetic and fast. It also makes real (local, free)
model calls and needs a running Ollama server with the models it names pulled.

Opt in with one of::

    uv run pytest -m integration                           # run only the integration tests
    uv run pytest -m "" tests/test_integration_ollama.py   # run it explicitly, clearing the deselect

If you opt in without a server reachable, the test skips with a clear reason
rather than hanging or erroring on connection failures.

Run it manually to exercise the full pydantic-ai wiring end-to-end against live
models. It is intentionally tolerant of flaky rankings: small local 7-8B models
are often unreliable at strict structured output, so a ranking that fails
pydantic validation or the dynamic alias check lands in ``RankingFailure`` by
design. The assertions target *wiring* (does the pipeline run end-to-end and
produce the right record shapes) rather than *model quality* (do the models rank
well). If generation or ranking produces zero successes across all models,
that's a wiring break and the test fails; a mix of successes and failures is
expected and fine.

Requires the ``openai`` package on the path (pydantic-ai's Ollama provider
routes through the OpenAI-compatible ``/v1/chat/completions`` endpoint). Install
it with ``uv sync --all-extras`` or ``pip install "pydantic-ai-slim[openai]"``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ollama = pytest.importorskip("pydantic_ai.models.ollama")
pytest.importorskip("pydantic_ai.providers.ollama")

from pydantic import BaseModel  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.ollama import OllamaModel  # noqa: E402
from pydantic_ai.providers.ollama import OllamaProvider  # noqa: E402

from tournament_eval import (  # noqa: E402
    DefaultRankingTemplate,
    build_generation_tasks,
    build_ranking_tasks,
    deanonymize_ranking,
    generate_all,
    rank_all,
)
from tournament_eval.models import GenerationResult, RankingFailure, RankingResult  # noqa: E402

# A real Ollama server is a hard requirement, and the models must be pulled.
_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_MODELS = ["qwen2.5:7b", "llama3.1:8b"]
# Retries bumped: small models frequently flub structured output on first try.
_RANKING_RETRIES = 3


def _ollama_reachable() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as resp:
            return bool(resp.status == 200)
    except OSError:
        return False


def _gen_agent(model_name: str) -> Agent[None, str]:
    return Agent(
        OllamaModel(model_name, provider=OllamaProvider(base_url=_OLLAMA_BASE_URL)),
        output_type=str,
        name=model_name,
    )


def _rank_agent(model_name: str, response_model: type[BaseModel]) -> Agent[None, BaseModel]:
    return Agent(
        OllamaModel(model_name, provider=OllamaProvider(base_url=_OLLAMA_BASE_URL)),
        output_type=response_model,
        name=model_name,
        retries=_RANKING_RETRIES,
    )


@pytest.mark.integration
async def test_end_to_end_tournament_against_ollama(tmp_path: Path) -> None:
    """A full build → generate → rank → de-anonymize run against live local models.

    Asserts the wiring holds: results are the right record types, authors resolve
    to the model names, usage metadata is populated, ``generation_task_id``
    propagates from ranking tasks through to ranking results, and
    ``deanonymize_ranking`` resolves ids to authors. Tolerates flaky rankings
    (some ``RankingFailure`` is expected from small models) but requires at least
    one successful generation and at least one successful ranking — zero of either
    is a wiring break, not a model-quality issue.
    """
    if not _ollama_reachable():
        pytest.skip("no Ollama server reachable at http://localhost:11434 (start `ollama serve` and pull the models)")

    template = DefaultRankingTemplate()

    # 1. Build generation tasks (rerun-stable, persisted).
    tasks = build_generation_tasks(
        [
            "Translate to French. Reply with only the translation, no commentary:\n\nThe cat sleeps.",
            "Translate to French. Reply with only the translation, no commentary:\n\nGood morning, friend.",
        ],
        tasks_path=tmp_path / "generation_tasks.jsonl",
    )

    # 2. Generate with both models as contestants.
    gen_agents = [_gen_agent(m) for m in _MODELS]
    generations, gen_failures = await generate_all(
        tasks,
        gen_agents,
        results_path=tmp_path / "generations.jsonl",
        failures_path=tmp_path / "generation_failures.jsonl",
    )

    # Wiring: at least one model must generate something. (Both may fail only if
    # the server is broken — that's the wiring break we're catching.)
    assert gen_failures == [] or len(generations) > 0, (
        f"all generations failed — wiring break? failures={[(f.author, f.error_type) for f in gen_failures]}"
    )
    assert len(generations) > 0, "no generations succeeded across either model"
    for gen in generations:
        assert isinstance(gen, GenerationResult)
        assert gen.author in _MODELS, f"author {gen.author!r} not a model name — resolution broken"
        assert gen.output, f"empty output from {gen.author}"
        assert isinstance(gen.metadata["input_tokens"], int)
        assert isinstance(gen.metadata["requests"], int)
        assert gen.metadata["requests"] >= 1

    # 3. Build ranking tasks from the generations.
    ranking_tasks = build_ranking_tasks(
        tasks,
        generations,
        ranking_prompt="Rank the candidates by translation quality — fluency and accuracy.",
        tasks_path=tmp_path / "ranking_tasks.jsonl",
    )
    assert len(ranking_tasks) >= 1

    # 4. Rank with both models as rankers.
    rank_agents = [_rank_agent(m, template.response_model) for m in _MODELS]
    rankings, rank_failures = await rank_all(
        ranking_tasks,
        generations,
        rank_agents,
        template=template,
        results_path=tmp_path / "rankings.jsonl",
        failures_path=tmp_path / "ranking_failures.jsonl",
    )

    # Wiring: rankings are RankingResult or RankingFailure, never anything else.
    for r in [*rankings, *rank_failures]:
        assert isinstance(r, RankingResult | RankingFailure)
    # Tolerate flaky models, but require at least one successful ranking — every
    # ranking failing across both models and all tasks would indicate a wiring
    # break (e.g. structured output never working at all).
    assert len(rankings) > 0, (
        f"no rankings succeeded — wiring break? "
        f"failures={[(f.author, f.error_type, f.message[:120]) for f in rank_failures]}"
    )

    for ranking in rankings:
        assert isinstance(ranking, RankingResult)
        assert ranking.author in _MODELS
        # Provenance: generation_task_id propagates task → ranking task → ranking result.
        assert ranking.generation_task_id in {t.id for t in tasks}
        # The dynamic alias check passed, so ranking is a permutation of the task's candidate ids.
        ranked_task = next(rt for rt in ranking_tasks if rt.id == ranking.ranking_task_id)
        assert set(ranking.ranking) == set(ranked_task.generations.values())
        # deanonymize_ranking resolves ids back to the contestants' authors.
        authors_best_first = deanonymize_ranking(ranking, generations)
        assert set(authors_best_first).issubset(set(_MODELS))
        assert len(authors_best_first) == len(ranking.ranking)
        # Usage metadata is populated on the ranking run too.
        assert isinstance(ranking.metadata["input_tokens"], int)

    # The ranking reasoning split: ranking_reasoning is the structured-output
    # justification; reasoning is the thinking trace (None for non-reasoning models).
    for ranking in rankings:
        assert isinstance(ranking, RankingResult)
        assert ranking.ranking_reasoning is None or isinstance(ranking.ranking_reasoning, str)
        assert ranking.reasoning is None or isinstance(ranking.reasoning, str)
