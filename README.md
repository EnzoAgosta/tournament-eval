# tournament-eval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![Coverage: 100%](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#tested-minimal-extensible)

Circular tournament evaluation for language models: every model is both a contestant and a judge.

## Why this exists

I built this while fine-tuning models for translation. Translation is a task with no single correct answer — two fluent translations can both be "right," and a BLEU score won't tell you which one a human would prefer. I wanted a way to rank my fine-tunes that didn't cost a fortune in API calls to a single "grader" model, and that didn't quietly inherit that one grader's blind spots.

So instead of trusting one judge, I let the contestants judge each other. That turned out to be a more interesting idea than I expected, and it grew into this.

## The idea

You have a set of models and a set of tasks where quality is subjective — translation, summarization, style transfer, creative writing. There's no gold answer to grade against.

`tournament-eval` runs a circular tournament:

1. **Every model generates** an output for every task.
2. **Every model ranks** every model's outputs for each task — including its own, anonymized behind aliases (`A`, `B`, `C`...).
3. You get a complete graph of rankings, each with the judge's full reasoning preserved.

Five models judging five outputs is 25 independent rankings per task. No model's verdict is load-bearing on its own.

Judges work independently — there's no debate, no iteration, no consensus step. They never see each other's verdicts. Collapsing the rankings into a single result comes later, and it's yours to define.

## Why peer ranking instead of a single judge

**Bias becomes a signal, not a problem to hide.**
When a model overrates its own output relative to how its peers rate it, that's a measurable fact about its calibration. When it systematically underrates a competitor, that's a visible preference pattern. These biases aren't corrected away — they're recorded, and they become part of your dataset. A single judge gives you its biases with no way to see them. A room full of judges lets you measure them against each other.

**No single point of failure.**
If one model hallucinates, its noise is diluted by the others. If one model is biased, that bias is legible against the consensus rather than silently baked into every score.

**Model-agnostic.**
The framework only cares about ordinal rankings. The contestants can be GPT, Claude, a local Llama, or the checkpoint you fine-tuned an hour ago. They don't need to agree, and they don't need to know anything about each other.

**Order-invariant aggregation.**
There's no Elo, no match history, no score that drifts as judgments accumulate. Every ranking is independent. Once you have the set of rankings, aggregating them is order-invariant — Borda count, Condorcet, Bradley–Terry, whatever you choose, the result doesn't depend on the order you feed the rankings in.

> **A note on determinism.** Generation itself is *not* reproducible — models are sampled at a temperature, so two runs can differ. The order-invariance is a property of the aggregation math over a fixed set of rankings, not of the model outputs. If you need reproducible generation, pin temperature/seed at the client level.

**Full reasoning capture.**
Every judge's deliberation is preserved alongside its ranking, so you can run the tournament once and analyze it many ways afterward:

- Which model values fluency over accuracy?
- Which model is most self-consistent?
- Which model's reasoning tracks human preference?

The rankings are the output. The reasoning is the audit trail.

## Not Elo, not pairwise

If you've ranked models before, you've probably reached for one of two tools. This is neither.

**Elo — and its descendants Glicko, TrueSkill, arena leaderboards — is sequential by construction.** Every match nudges a running score up or down, so a rating depends on match history and the order games were played in. There is no running score here to nudge. Every judgment is independent and stateless, so there is nothing for an Elo update to attach to. You *can't* build an Elo on top of this — and that's deliberate. Feed the rankings in any order and the aggregate is identical.

**Pairwise comparison collapses every decision to "A vs B."** To order N candidates you collect a pile of binary votes and reconstruct a global order from them — which can contradict itself (A beats B beats C beats A). Here, each judge ranks the entire field at once and returns one complete ordinal order, reasoned over all candidates together, rather than stitched back together from fragments.

The design target is the opposite of a two-player match: **many contestants, and as many judges as contestants.** Five models producing five outputs and ranking all five is the natural shape — 25 whole-field rankings per task, not 25 coin flips.

### Contestants and judges are separate axes

The circular case — every contestant is also a judge — is the interesting default, but it isn't a requirement. `generate_all` and `rank_all` each take their own client list, so the two roles are fully decoupled:

- **2 candidates, 15 judges** — a small head-to-head settled by a large, diverse panel.
- **15 candidates, 2 judges** — a wide field filtered by a couple of trusted graders.
- **N candidates judged by the same N models** — the full circular tournament.

Same pipeline, same data model. Who generates and who judges are just two lists you pass in.

## Quickstart

Requires Python 3.14+ and a running [Ollama](https://ollama.com) server. Ollama is the only client implemented today (see [Status](#status)).

```bash
git clone https://github.com/<you>/tournament-eval.git
cd tournament-eval
uv sync
```

```python
import asyncio
import uuid

from tournament_eval import (
    GenerationTask,
    OllamaLLMClient,
    OllamaModelConfig,
    build_ranking_tasks,
    generate_all,
    rank_all,
)


async def main() -> None:
    # The contestants — also the judges.
    clients = [
        OllamaLLMClient(OllamaModelConfig(model_name="llama3.2")),
        OllamaLLMClient(OllamaModelConfig(model_name="gemma3")),
    ]

    # The tasks. Subjective by design — no gold answer.
    tasks = [
        GenerationTask(
            id=uuid.uuid4(),
            generation_prompt="Translate to French: The quick brown fox jumps over the lazy dog.",
        ),
    ]

    # 1. Every model generates an output for every task.
    generations, gen_failures = await generate_all(tasks, clients)

    # 2. Group outputs per task and assign anonymized aliases (A, B, ...).
    ranking_tasks = build_ranking_tasks(
        tasks=tasks,
        generation_results=generations,
        ranking_prompt="Rank these French translations by fluency and accuracy, best first.",
    )

    # 3. Every model ranks every output, including its own.
    rankings, rank_failures = await rank_all(
        ranking_tasks=ranking_tasks,
        generation_results=generations,
        clients=clients,
    )

    for r in rankings:
        print(f"{r.author} ranked: {r.raw_model_ranking}")
        if r.reasoning:
            print(f"  reasoning: {r.reasoning[:120]}...")


asyncio.run(main())
```

See [`test_e2e.py`](test_e2e.py) for a slightly fuller version.

## How it works

The pipeline is three pure-ish async functions, each a clean stage:

| Stage | Function | In → Out |
|-------|----------|----------|
| Generate | `generate_all` | tasks × clients → `GenerationResult`s |
| Build | `build_ranking_tasks` | generations grouped per task, aliased → `RankingTask`s |
| Rank | `rank_all` | ranking tasks × clients → `RankingResult`s |

Every stage fans its work out concurrently with `asyncio.gather`. Failures aren't thrown — each stage returns a `(results, failures)` pair, where `failures` maps `(task_id, model_name)` to the exception that was raised. One model timing out or returning garbage doesn't sink the run; it just shows up in the failures map for you to handle.

Judge responses are validated strictly: the ranking must list every candidate exactly once — no unknown aliases, no duplicates, no missing entries, no ties. A malformed ranking is a failure, not a silent best-guess.

## Status

This is an early, honest-about-it project.

- **Ollama** is fully implemented and is what the quickstart uses.
- **OpenAI** and **Anthropic** have config classes in place but their clients aren't implemented yet. The `LLMClient` abstract base is small and the Ollama client is a ~80-line reference, so adding a provider is mostly a matter of wiring up its HTTP call.
- **Aggregation math is intentionally out of scope.** The framework hands you the rankings and the reasoning. You own the question of how to collapse them into a verdict — and that choice is a real methodological decision, not a detail to bury in a library.

## Tested, minimal, extensible

- 100% source coverage, fully type-checked (`mypy --strict`) and linted (`ruff`).
- Pure async functions, no hidden global state.
- One concrete client (Ollama); an abstract base ready for OpenAI, Anthropic, and beyond.

```bash
uv run pytest          # tests
uv run mypy            # type check
uv run ruff check      # lint
```

## License

MIT. See [LICENSE](LICENSE).
