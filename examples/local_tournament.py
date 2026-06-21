"""A self-contained circular tournament against a local Ollama server.

A runnable example, not a test: every model in the field both **generates** a
response and **ranks** the others' — the full peer-review case. It exercises the
whole pipeline end to end (build → generate → build ranking tasks → rank →
de-anonymize → aggregate) and then renders the results both ways we ship:

* ``presentation.console`` — aligned text tables (leaderboard, pairwise matrix,
  ballot summary).
* ``presentation.plots`` — matplotlib figures (leaderboard bar chart, pairwise
  heatmap, rank distribution), saved as PNGs.

Run it, then open ``examples/run/figures/*.png`` to look at the plots.

Prerequisites
-------------
* A local Ollama server (``ollama serve``) reachable at ``http://localhost:11434``.
* The models in :data:`MODELS` pulled (``ollama list``). This example uses a
  ~26B model (17 GB on disk) alongside several ~7-14B ones, so it targets a
  machine with ~32 GB of memory.
* The ``openai`` package on the path — pydantic-ai's Ollama provider routes
  through the OpenAI-compatible ``/v1`` endpoint. ``uv sync --all-extras``
  installs it (and the ``plotting`` extra for the figures).

    uv run python examples/local_tournament.py

Everything streams to ``examples/run/*.jsonl`` as it lands, so a crash means just
rerunning the script — already-succeeded ``(task, author)`` pairs are skipped and
only what's left runs again.
"""

import asyncio
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.concurrency import ConcurrencyLimiter
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from tournament_eval import (
    DefaultRankingTemplate,
    GenerationFailure,
    GenerationResult,
    RankingFailure,
    RankingResult,
    build_generation_tasks,
    build_ranking_tasks,
    generate_all,
    rank_all,
)
from tournament_eval.aggregation import ballots_from_rankings, pairwise, positional
from tournament_eval.aggregation.ballots import Ballot
from tournament_eval.orchestration import deanonymize_ranking
from tournament_eval.presentation import console, plots

OLLAMA_BASE_URL = "http://localhost:11434/v1"
"""The OpenAI-compatible endpoint of the local Ollama server."""

MODELS = ["gemma4:26b", "qwen2.5:14b", "qwen2.5:7b", "llama3.1:8b", "mistral:7b"]
"""The field — five pulled models. Same pool generates and ranks (circular)."""

TASK_PROMPTS = [
    # Translation — subjectively graded, easy to compare at a glance.
    "Translate this English sentence to French. Keep the tone light and idiomatic; "
    "no commentary, no notes:\n\n"
    "It's raining cats and dogs, so we'd better stay in.",
    # ELI5 — constrained length, rewards clarity and intuition.
    "Explain what a black hole is to a five-year-old, in exactly two sentences.",
    # Creative — short, subjective, no single right answer.
    "Write a four-line poem about the feeling of a Monday morning. No title, just the four lines.",
    # Analogy — rewards a tight, illuminating metaphor over jargon.
    "Explain the idea of 'recursion' to someone who has never programmed, using a single "
    "everyday metaphor. Keep it to two sentences.",
]
"""Four short, subjective generation tasks — cheap to generate, easy to compare."""

RANKING_PROMPT = (
    "Rank these responses from best to worst. Judge by overall quality — how well it "
    "answers the task, clarity, and creativity or elegance. The best response is the one "
    "you'd most want to receive. Break any tie deliberately and say why in `reasoning`."
)

RUN_DIR = Path(__file__).parent / "run"
"""Where the JSONL streams and figures land. Gitignored — the script is committed, its data isn't."""

RANKING_RETRIES = 3
"""Retries on the ranking agents — small local models frequently flub structured output on first try."""

# A shared cap across every agent: at most 2 models in flight at once. The 26B
# model is 17 GB on disk; unbounded fan-out could keep ~5 models resident at once
# (~33 GB), too close to the ceiling. Cap 2 keeps peak memory comfortable and
# lets Ollama swap models in and out without thrashing.
_CONCURRENCY = ConcurrencyLimiter(max_running=2, name="ollama-pool")


def _gen_agent(model_name: str) -> Agent[None, str]:
    """A generation agent (``output_type=str``) named after its model."""
    return Agent(
        OllamaModel(model_name, provider=OllamaProvider(base_url=OLLAMA_BASE_URL)),
        output_type=str,
        name=model_name,
        max_concurrency=_CONCURRENCY,
    )


def _rank_agent(model_name: str, template: DefaultRankingTemplate) -> Agent[None, BaseModel]:
    """A ranking agent whose ``output_type`` is the template's response model."""
    return Agent(
        OllamaModel(model_name, provider=OllamaProvider(base_url=OLLAMA_BASE_URL)),
        output_type=template.response_model,
        name=model_name,
        retries=RANKING_RETRIES,
        max_concurrency=_CONCURRENCY,
    )


def _generation_tasks_path() -> Path:
    return RUN_DIR / "generation_tasks.jsonl"


def _generations_path() -> Path:
    return RUN_DIR / "generations.jsonl"


def _generation_failures_path() -> Path:
    return RUN_DIR / "generation_failures.jsonl"


def _ranking_tasks_path() -> Path:
    return RUN_DIR / "ranking_tasks.jsonl"


def _rankings_path() -> Path:
    return RUN_DIR / "rankings.jsonl"


def _ranking_failures_path() -> Path:
    return RUN_DIR / "ranking_failures.jsonl"


def _figures_dir() -> Path:
    return RUN_DIR / "figures"


async def run() -> None:
    """Run the full tournament and render the results to console + figures."""
    template = DefaultRankingTemplate()
    gen_agents = [_gen_agent(name) for name in MODELS]
    rank_agents = [_rank_agent(name, template) for name in MODELS]

    print(f"\n=== Circular tournament: {len(MODELS)} models, {len(TASK_PROMPTS)} tasks ===\n")

    # 1. Build generation tasks (rerun-stable: prompt-matched ids survive reruns).
    tasks = build_generation_tasks(TASK_PROMPTS, tasks_path=_generation_tasks_path())

    # 2. Generate. Resume is automatic against _generations_path; rerun to finish.
    print("Generating...")
    generations, gen_failures = await generate_all(
        tasks,
        gen_agents,
        results_path=_generations_path(),
        failures_path=_generation_failures_path(),
        on_result=lambda result: print(f"{result.author} generated {result.output}", flush=True),
        on_failure=lambda failure: print(f"{failure.author} failed to generate {failure.message}", flush=True),
    )
    print(f"\n  {len(generations)} generations, {len(gen_failures)} failures")
    _report_generation_failures(gen_failures)

    # 3. Build ranking tasks (freezes once built: let generation finish first).
    print("\nBuilding ranking tasks...")
    ranking_tasks = build_ranking_tasks(
        tasks,
        generations,
        ranking_prompt=RANKING_PROMPT,
        tasks_path=_ranking_tasks_path(),
    )
    print(f"  {len(ranking_tasks)} ranking tasks")

    # 4. Rank. Resume is automatic against _rankings_path; rerun to finish.
    print("\nRanking...")
    rankings, rank_failures = await rank_all(
        ranking_tasks,
        generations,
        rank_agents,
        template=template,
        results_path=_rankings_path(),
        failures_path=_ranking_failures_path(),
        on_result=lambda result: print(f"{result.author} ranked {result.ranking}", flush=True),
        on_failure=lambda failure: print(f"{failure.author} failed to rank {failure.message}", flush=True),
    )
    print(f"\n  {len(rankings)} rankings, {len(rank_failures)} failures")
    _report_ranking_failures(rank_failures)

    # 5. Aggregate + present.
    print("\n=== Results ===\n")
    _present(generations, rankings)


def _report_generation_failures(failures: list[GenerationFailure]) -> None:
    if not failures:
        return
    print("  Generation failures:")
    for f in failures:
        print(f"    {f.author} on {f.generation_task_id}: {f.error_type}: {f.message}")


def _report_ranking_failures(failures: list[RankingFailure]) -> None:
    if not failures:
        return
    print("  Ranking failures:")
    for f in failures:
        print(f"    {f.author} on {f.ranking_task_id}: {f.error_type}: {f.message}")


def _present(generations: list[GenerationResult], rankings: list[RankingResult]) -> None:
    """Aggregate the rankings and render console tables + figure PNGs."""
    ballots = ballots_from_rankings(rankings, generations)

    print("--- Ballot summary ---")
    print(console.ballot_summary(ballots))
    print()

    borda = positional.borda(ballots)
    normalized_borda = positional.normalized_borda(ballots)
    copeland_scores = pairwise.copeland(ballots)

    print("--- Borda (raw; assumes full participation) ---")
    print(console.leaderboard(borda))
    print()

    print("--- Normalized Borda (mean per-ballot score in [0, 1]) ---")
    print(console.leaderboard(normalized_borda))
    print()

    print("--- Copeland (wins - losses; a Condorcet winner tops it) ---")
    print(console.leaderboard(copeland_scores))
    print()

    print("--- Pairwise head-to-head ---")
    tally = pairwise.matrix(ballots)
    print(console.pairwise_matrix(tally))
    print()

    print("--- Rank distribution per contestant (from a sample) ---")
    # Show a couple of rankings in author space so the aliasing is visible.
    for ranking in rankings[:3]:
        authors = deanonymize_ranking(ranking, generations)
        print(f"  {ranking.author}: {' > '.join(authors)}")
    print()

    _save_figures(normalized_borda, tally, ballots)


def _save_figures(
    normalized_borda: dict[str, float],
    tally: pairwise.PairwiseTally,
    ballots: list[Ballot],
) -> None:
    figures = _figures_dir()
    figures.mkdir(parents=True, exist_ok=True)

    leaderboard_fig = plots.leaderboard_bar(normalized_borda)
    leaderboard_fig.savefig(figures / "leaderboard.png", dpi=150)

    heatmap_fig = plots.pairwise_heatmap(tally)
    heatmap_fig.savefig(figures / "pairwise_heatmap.png", dpi=150)

    distribution_fig = plots.rank_distribution(ballots)
    distribution_fig.savefig(figures / "rank_distribution.png", dpi=150)

    print(f"Figures saved to {figures}/*.png")


if __name__ == "__main__":
    asyncio.run(run())
