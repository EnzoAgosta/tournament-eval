# tournament-eval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![Coverage: 100%](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#tested-and-typed)

Circular ranking for language models — and a clean set of primitives for wiring many different LLMs together to do it.

## Why this exists

I built this while fine-tuning models for translation. Translation is a task with no single correct answer — two fluent translations can both be "right," and a BLEU score won't tell you which one a human would prefer. I wanted a way to rank my fine-tunes that didn't cost a fortune in API calls to a single "ranker" model, and that didn't quietly inherit that one ranker's blind spots.

So instead of trusting one ranker, I let a panel of models rank each other's work. That turned out to be a more interesting idea than I expected — and getting many different LLMs to actually cooperate cleanly turned out to be most of the work. This is both: the ranking methodology, and the plumbing to run it.

## The idea

You have a set of models and a set of tasks where quality is subjective — translation, summarization, style transfer, creative writing. There's no gold answer to measure against.

`tournament-eval` runs the evaluation as two passes over a pool of models:

1. **Generation** — every contestant produces an output for every task.
2. **Ranking** — every ranker orders the whole field of outputs for each task, with authorship hidden behind aliases (`A`, `B`, `C`...).

The contestants and the rankers are just two lists you pass in. Point them at the **same** pool and you get the circular case — a peer review where the models being ranked are also the ones doing the ranking. Point them at **different** pools and you get a panel of trusted rankers scoring a field of candidates. Same pipeline either way.

Each ranker works independently — no debate, no consensus step, they never see each other's rankings. You get a set of complete, reasoned rankings; collapsing them into a single result comes later, and it's yours to define.

## Why peer ranking instead of a single ranker

**Bias becomes a signal, not a problem to hide.**
When a model overrates a translation that its peers rate poorly, that's a measurable fact about its calibration. When it systematically underrates a competitor, that's a visible preference pattern. These biases aren't corrected away — they're recorded, and they become part of your dataset. A single ranker gives you its biases with no way to see them; a panel lets you measure them against each other.

**No single point of failure.**
If one model hallucinates, its noise is diluted by the others. If one model is biased, that bias is legible against the panel rather than silently baked into every score.

**Model-agnostic.**
The framework only cares about ordinal rankings. The models can be GPT, Claude, a local Llama, or the checkpoint you fine-tuned an hour ago. They don't need to agree, and they don't need to know anything about each other.

**Order-invariant aggregation.**
There's no Elo, no match history, no score that drifts as rankings accumulate. Every ranking is independent. Once you have the set of rankings, aggregating them is order-invariant — Borda, Copeland, whatever you choose — and the result doesn't depend on the order you feed the rankings in.

> **A note on determinism.** Generation itself is *not* reproducible — models are sampled at a temperature, so two runs can differ. The order-invariance is a property of the aggregation math over a fixed set of rankings, not of the model outputs. If you need reproducible generation, pin temperature/seed on your agent's model settings.

**Full reasoning capture.**
Every ranker's reasoning is preserved alongside its ranking — both the justification it wrote in its structured response and the thinking trace the model produced while ranking (when the provider surfaces one). So you can run the tournament once and analyze it many ways — which model values fluency over accuracy, which is most self-consistent, which one's reasoning tracks human preference. The rankings are the output; the reasoning is the audit trail.

## Not Elo, not pairwise

If you've ranked models before, you've probably reached for one of two tools. This is neither.

**Elo — and its descendants Glicko, TrueSkill, arena leaderboards — is sequential by construction.** Every match nudges a running score, so a rating depends on match history and the order games were played in. There's no running score here to nudge: every ranking is independent and stateless. Feed the rankings in any order and the aggregate is identical — by design.

**Pairwise comparison collapses every decision to "A vs B."** To order N candidates you collect a pile of binary votes and reconstruct a global order, which can contradict itself (A beats B beats C beats A). Here each ranker orders the entire field at once and returns one complete ordinal order, reasoned over all candidates together, rather than stitched back from fragments.

### Contestants and rankers are separate axes

`generate_all` and `rank_all` each take their own list of agents, so the two roles are fully decoupled:

- **N candidates ranked by the same N models** — the full circular tournament.
- **2 candidates, 15 rankers** — a small head-to-head settled by a large, diverse panel.
- **15 candidates, 2 rankers** — a wide field filtered by a couple of trusted rankers.
- **5 fine-tunes ranked by 3 frontier models** — contestants and rankers from entirely different tiers.

Same pipeline, same data model. Who generates and who ranks are just two lists.

## Install

```bash
uv add tournament-eval
```

That's the whole install. The only core dependency is [`pydantic-ai-slim`](https://ai.pydantic.dev), the lightweight core of [Pydantic AI](https://ai.pydantic.dev) — no provider SDKs are pulled in by default.

### Adding providers

`tournament-eval` talks to models through **Pydantic AI agents**, so the providers available to you are exactly Pydantic AI's. Each provider needs its SDK on your path, which you add as an *extra* matching the Pydantic AI provider group:

```bash
uv add tournament-eval --extra anthropic    # + the Anthropic SDK (Claude)
uv add tournament-eval --extra openai       # + the OpenAI SDK
uv add tournament-eval --extra google       # + Google GenAI (Gemini)
uv add tournament-eval --extra bedrock      # + Amazon Bedrock (boto3)
```

Pydantic AI supports more (Groq, Mistral, Cohere, xAI, OpenRouter, Hugging Face, Ollama, Cerebras, Outlines, Deepseek, …). Add the SDK you need to your project the way Pydantic AI documents it and the corresponding model class is reachable — `tournament-eval` doesn't gate the list.

## The LLM layer is Pydantic AI

This library owns the **methodology** — the tasks, the aliasing, the resume, the persistence, the ranking contract — and leans on Pydantic AI for everything LLM: providers, auth, retries, streaming, structured output, reasoning/thinking, usage tracking. You don't configure models through `tournament-eval`; you construct and own `pydantic_ai.Agent` instances the Pydantic AI way and hand them in.

The pipeline needs two kinds of agent:

- **Generation agents** — `Agent(model, output_type=str)`. They produce the contestant outputs.
- **Ranking agents** — `Agent(model, output_type=template.response_model)`. They produce a ranking. The `output_type` is the ranking template's pydantic model (see [Customizing the ranking](#customizing-the-ranking)).

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.bedrock import BedrockProvider

import boto3

from tournament_eval import DefaultRankingTemplate

template = DefaultRankingTemplate()

# A generation agent — plain text out.
claude_gen = Agent(
    AnthropicModel("claude-sonnet-4-6", provider=AnthropicProvider(api_key="...")),
    output_type=str,
)

# A ranking agent — its output_type is the template's response model.
claude_rank = Agent(
    AnthropicModel("claude-sonnet-4-6", provider=AnthropicProvider(api_key="...")),
    output_type=template.response_model,
)

# Claude on Bedrock: the model is configured the Pydantic AI way (Converse API),
# so auth is boto's credential chain and every Bedrock family is covered.
bedrock_rank = Agent(
    BedrockConverseModel(
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        provider=BedrockProvider(bedrock_client=boto3.client("bedrock-runtime", region_name="us-east-1")),
    ),
    output_type=template.response_model,
)
```

Everything about the model — temperature, max tokens, system prompt/instructions, thinking/`reasoning_effort`, retries, concurrency — is configured on the agent/model the Pydantic AI way and is outside this library's surface. See the [Pydantic AI docs](https://ai.pydantic.dev) for the full set.

### Local models (Ollama)

A local Ollama server works through the OpenAI-compatible endpoint. Two knobs are worth setting explicitly against small local models: `retries` (structured ranking output is where weak models flub, and a retry often clears it) and `max_concurrency` (Ollama's default `OLLAMA_NUM_PARALLEL` is low, so unbounded fan-out can queue or time out).

```python
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

OLLAMA = "http://localhost:11434/v1"
MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"]

template = DefaultRankingTemplate()

gen_agents = [
    Agent(OllamaModel(m, provider=OllamaProvider(base_url=OLLAMA)), output_type=str, name=m, max_concurrency=2)
    for m in MODELS
]
rank_agents = [
    Agent(OllamaModel(m, provider=OllamaProvider(base_url=OLLAMA)), output_type=template.response_model, name=m, retries=3, max_concurrency=2)
    for m in MODELS
]
```

`name=m` makes the model name the `author` label (see [Author labels](#author-labels)).

### Author labels

Each result carries an `author` — the label used for de-anonymization and the resume key. It's resolved as **`agent.name` if set, else the model name**. Distinct models get distinct authors automatically; if you run the *same* model twice with different config (e.g. one model at two temperatures), set distinct `agent.name=` on each so they don't collide:

```python
hot = Agent(model, output_type=str, name="gpt-4o-temp0.7")
cold = Agent(model, output_type=str, name="gpt-4o-temp0.0")
```

`generate_all` / `rank_all` check for author collisions up front and raise rather than silently overwriting results.

## How it works

The pipeline is three async functions, each a clean stage:

| Stage | Function | In → Out |
|-------|----------|----------|
| Generate | `generate_all(tasks, agents)` | tasks × agents → `GenerationResult`s |
| Build | `build_ranking_tasks(tasks, generations, prompt)` | generations grouped per task, aliased → `RankingTask`s |
| Rank | `rank_all(ranking_tasks, generations, agents, template=...)` | ranking tasks × agents → `RankingResult`s |

Each stage fans its work out concurrently with `asyncio.gather` and calls `agent.run` once per pair. There's no client lifecycle to manage — `agent.run` is a plain coroutine. **Failures aren't thrown** — each stage returns a `(results, failures)` pair, where every failure is a typed record (`GenerationFailure` / `RankingFailure`) carrying the failing id, the `author`, and the error type and message. One agent timing out or returning garbage doesn't sink the run; it lands in `failures` for you to inspect or retry.

Ranking responses are validated strictly: the static shape of the response is validated by Pydantic AI at the provider (against the template's `response_model`), and the *dynamic* rule — the ranking must list every candidate alias exactly once, no unknowns, no duplicates, no missing entries, no ties — is checked by the template afterward. A malformed ranking is a failure, not a silent best guess.

A `RankingFailure` is debuggable, not just a dead end: it carries the rendered `ranking_prompt` (what was asked) and a `details` payload extracted best-effort from the exception chain (what the model answered wrong). `details` is a kitchen-sink dict (like `metadata`) populated with whatever the failure path exposes — `validation_errors` (pydantic's per-field errors, each carrying the failing `input`) when the response failed validation; `model_output` (the full emitted ranking) when the run succeeded but the dynamic alias check rejected it; `cause_type`/`cause_message` for the wrapped exception; `body` when Pydantic AI surfaces a response body. `None` when nothing structured was captured, leaving `message` as the only breadcrumb.

Every result carries `metadata` with the run's usage (`input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `requests`) from Pydantic AI's `RunUsage` — a kitchen-sink dict that can grow to hold cost (via [`genai-prices`](https://github.com/pydantic/genai-prices)), latency, and more without changing the record shape.

## Building generation tasks

`generate_all`'s resume is keyed on `(generation_task_id, author)`, so a task's id must stay **stable across reruns** for "just rerun the script" to actually resume. The naive pattern — constructing `GenerationTask(id=uuid.uuid4(), prompt=...)` in a loop — gives every run fresh ids, so resume matches nothing and silently regenerates everything.

`build_generation_tasks` fixes that the same way `build_ranking_tasks` does for ranking: it persists tasks to a file and, on a rerun, reuses the persisted task for any prompt it already has one for (matching by prompt), so ids stay stable. `tasks_path` is **required** — persistence is the whole point.

```python
from tournament_eval import build_generation_tasks, generate_all

# An open text file is a fine iterable of prompts (strip newlines if they
# shouldn't be part of the prompt).
with open("sentences.txt") as f:
    tasks = build_generation_tasks(
        (f"Translate this sentence to French. No commentary, no note:\n\n{line.rstrip()}" for line in f),
        tasks_path="run/generation_tasks.jsonl",
    )

results, failures = await generate_all(
    tasks, agents,
    results_path="run/generations.jsonl",
    failures_path="run/generation_failures.jsonl",
)
```

Rerun that exact script and `build_generation_tasks` reuses the persisted tasks (stable ids), `generate_all` skips the `(task, author)` pairs already done, and only anything new or previously-failed runs again.

Tasks are matched **by prompt**, so duplicate prompts collapse to one task (two identical prompts are one task, not two — and you don't want two identical candidates behind two aliases in a ranking). For deliberate distinct duplicates, construct `GenerationTask` directly (or `build_generation_task` with an explicit `id=`), noting that a random id opts out of resume for that task.

## Customizing the ranking

How a ranking model is prompted, constrained, and validated lives in a `RankingTemplate` — a single object owning the three pieces that must agree with each other:

- `response_model` — the pydantic `BaseModel` a ranking agent's `output_type` is set to. Pydantic AI constrains the model's reply to this schema and validates it, so the *static* shape of the verdict is enforced at the provider.
- `render(ranking_task, candidates)` — the full prompt the ranking model sees.
- `parse(instance, valid_aliases)` — the *dynamic* alias check (every alias exactly once, drawn from the task's candidates), which can't be expressed in a static schema.

`rank_all` / `rank_one` take a `template=` argument (default `DefaultRankingTemplate`, a strict total order with no ties). By default it shows the ranker the **original task the models answered** — shared across candidates, so it leaks no authorship — followed by the anonymized outputs. To also surface each candidate's **reasoning trace**, flip one flag (off by default, since traces are long and a verbose reasoner can look more thorough than it is):

```python
rankings, failures = await rank_all(
    ranking_tasks, generations, agents, template=DefaultRankingTemplate(include_reasoning=True)
)
```

For anything more, subclass. Adding context (a rubric, domain notes) is a one-method override of `render` — each candidate is passed as its full `GenerationResult`, so `.output`, `.reasoning`, `.generation_prompt`, and `.metadata` are all in reach:

```python
from collections.abc import Mapping
from tournament_eval import DefaultRankingTemplate, GenerationResult, RankingTask


class RubricTemplate(DefaultRankingTemplate):
    def render(self, ranking_task: RankingTask, candidates: Mapping[str, GenerationResult]) -> str:
        return "Ranking rubric: prioritise factual accuracy over fluency.\n\n" + super().render(
            ranking_task, candidates
        )


rankings, failures = await rank_all(ranking_tasks, generations, agents, template=RubricTemplate())
```

Changing the *shape* of the ranking (e.g. allowing ties) means overriding all three — `response_model`, `render`, and `parse` — so the prompt, the schema, and the parser stay consistent. (`candidates` exposes `.author` — don't render it into the prompt, or you defeat the anonymization.)

## Concurrency

Every stage fans out with `asyncio.gather`, so by default **every request fires at once**. The concurrency limit lives on the *agent* (Pydantic AI's `max_concurrency`), not the orchestration — because rate limits belong to the provider, not to the tournament.

```python
from pydantic_ai import Agent
from pydantic_ai.concurrency import ConcurrencyLimiter

# Per-agent limit: at most 4 in-flight runs of this agent.
agent = Agent(model, output_type=str, max_concurrency=4)

# Shared cap: hand the same limiter to several agents and they draw from one budget.
limiter = ConcurrencyLimiter(max_running=8, name="anthropic-pool")
agents = [
    Agent(model_a, output_type=str, max_concurrency=limiter),
    Agent(model_b, output_type=str, max_concurrency=limiter),
]
```

`max_concurrency` defaults to unbounded — fine for a local server you control, but against a rate-limited hosted API you'll want a conservative value, per-agent or as a shared limiter. The same limit applies whether you call the batch `generate_all` / `rank_all` or the single-shot `generate_one` / `rank_one`.

### Progress callbacks

A batch fans out concurrently and returns only when everything's done, so a big run is a blank screen until the end. `generate_all` and `rank_all` take two optional sync callbacks — `on_result` and `on_failure` — fired with each `GenerationResult`/`RankingResult` or `GenerationFailure`/`RankingFailure` as it lands, **after** it's been persisted to disk (so "notified" means "safely on disk"). They're `None` by default and not fired for successes loaded from `results_path` on resume (those were never run this call). A callback that raises is swallowed into a `UserWarning` — a broken progress hook can't sink a batch of model calls.

```python
from tqdm.auto import tqdm

pbar = tqdm(total=len(tasks) * len(gen_agents))
results, failures = await generate_all(
    tasks, gen_agents,
    results_path="run/generations.jsonl",
    on_result=lambda _: pbar.update(1),
    on_failure=lambda _: pbar.update(1),
)
pbar.close()
```

`total` here is the pre-resume count, so on a rerun the bar overshoots by the number of pairs skipped from `results_path` — cosmetic, since callbacks only fire for pairs actually run this call.

The callbacks are deliberately typed and minimal (one concrete type each, no dict/event protocol, no index/count — the caller already knows the total). `generate_one` / `rank_one` stay callback-free: their caller already receives the outcome synchronously.

## Persistence & resume

A tournament is expensive — many model calls — so the pipeline streams every result and failure to disk *as it lands*. You choose the file paths; there's no directory layout or naming convention to learn. Typically one flat JSONL file per stream:

```python
results, failures = await generate_all(
    tasks, agents,
    results_path="run/generations.jsonl",
    failures_path="run/generation_failures.jsonl",
)
```

Each write is a single synchronous append, so a process that dies mid-run keeps everything finished so far. There's no per-model sharding — every record carries its own `author` and `generation_task_id`/`ranking_task_id`, so a flat file is fully reconstructable; group or filter on read.

**Resume is automatic — just run the script again.** When you pass a `results_path`, `generate_all` reads it back first and skips every `(task, author)` pair that already succeeded, running only what's left and returning the **complete** set (loaded plus newly produced). So the return value is always a clean partition: `results` is every pair that now has a success, `failures` every pair still without one. Successes are skipped; **failures are always retried** — fix the cause (rate limit, API key, a flaky endpoint) and rerun, and only the still-broken pairs go out again.

```python
# Run once, get interrupted, run the exact same call again — it picks up where it left off.
results, failures = await generate_all(
    tasks, agents,
    results_path="run/generations.jsonl",
    failures_path="run/generation_failures.jsonl",
)
```

`rank_all` works identically, keyed on `(ranking_task, author)` against its own `results_path`.

Reading a run back is per-file and typed — `persistence.read_generation_result_file`, `persistence.read_ranking_result_file`, and friends each restore the original dataclasses, UUIDs and all. Writing anything yourself (e.g. persisting your tasks) is the one type-agnostic `persistence.append_record(path, record)`.

**One caveat — ranking tasks freeze.** A `RankingTask` carries a random alias shuffle, so `build_ranking_tasks(..., tasks_path="run/ranking_tasks.jsonl")` builds each task once and, on a rerun, reuses the persisted one rather than rebuilding (a rebuild would re-shuffle and orphan every ranking already collected). The consequence: a generation that only succeeds on a *later* resume won't be added to an already-built ranking task. So **let generation finish before you start ranking** — run `generate_all` until its `failures` are empty, then build ranking tasks. To deliberately rebuild, delete the ranking-tasks file (and any rankings) first.

## Aggregation

Collapsing per-ranker rankings into a verdict is a real methodological choice, so the `aggregation` package treats it as a **convenience, not a mandate**: every method is a pure function over plain *ballots* (one ranker's verdict as author labels, best-first), so you can use one, several, or none — and still do your own math over the data on disk. One invariant holds throughout: **ballots are strict total orders, never ties** (the shape `DefaultRankingTemplate` guarantees).

The package is organised by aggregation style, and you reach for the style you want:

- `tournament_eval.aggregation.positional` — Borda and its normalized variant. Pure Python, **no `numpy` needed**. Scores depend on *where* each ballot ranks a contestant.
- `tournament_eval.aggregation.pairwise` — the head-to-head tally (`matrix`) and the methods built on it (Copeland, with more to come). Need `numpy` (the `analysis` extra) **at call time only** — importing the module doesn't pull it in, so a plain tournament run stays dependency-free until you actually call a pairwise method.

`ballots_from_rankings` bridges a run to either style, de-anonymizing each ranking's ids back to author labels:

```python
from tournament_eval.aggregation import ballots_from_rankings, positional, pairwise

ballots = ballots_from_rankings(rankings, generations)
positional.borda(ballots)              # raw Borda points; assumes full participation
positional.normalized_borda(ballots)   # mean per-ballot score in [0, 1]; fair under unequal participation
```

The pairwise methods need `numpy` (the `analysis` extra) — but only when you call them, not when you import:

```python
# pip install "tournament-eval[analysis]"
from tournament_eval.aggregation import pairwise

pairwise.matrix(ballots)               # the head-to-head count matrix; the shared primitive
pairwise.copeland(ballots, expected_contestants={"gpt-4o", "claude-sonnet-4-6"})  # wins − losses; a Condorcet winner tops it
```

`pairwise.matrix` is the shared primitive the non-positional methods build on; `expected_contestants` declares the full field so a contestant some ballots omit is still scored (treated as not-compared, not penalized). Calling a pairwise method without the `analysis` extra raises a clear `ImportError` naming the package and the install command — importing the module never fails. More methods (Schulze, Bradley–Terry, …) will land in `pairwise` over time — conservatively, only ones whose results we're confident are accurate.

Every method that would otherwise silently gloss over something (a sub-two-candidate ballot, an omitted expected contestant) takes a `strict=False` flag: flip it to `True` to raise instead. `strict` means one thing across the package — "raise on anything that would otherwise be handled silently."

Prefer to roll your own? `tournament_eval.orchestration.deanonymize_ranking(ranking_result, generations)` returns one ranking's authors best-first — a pure function over the `generations` you already have (from `generate_all` or `persistence.read_generation_result_file`), composing with whatever aggregation you choose.

## Presentation

A run hands you data; `tournament_eval.presentation` hands you the handful of shapes everyone reaches for at the end of a run — text tables and plots over the aggregation results. A *convenience* layer in the same spirit as `aggregation`: pure functions you opt into, never a mandate. Not re-exported at the top level — reach for it via its submodule, the way `aggregation` is reached.

### Console (no extra dependencies)

`tournament_eval.presentation.console` renders aligned text tables — pure Python, returns `str` (never prints), so the output composes with your own surrounding text:

```python
from tournament_eval.presentation import console

print(console.leaderboard(positional.normalized_borda(ballots)))
print(console.pairwise_matrix(pairwise.matrix(ballots)))
print(console.ballot_summary(ballots))
```

### Plots (the `plotting` extra)

`tournament_eval.presentation.plots` renders matplotlib figures — `leaderboard_bar`, `pairwise_heatmap`, and `rank_distribution`. Each builds and returns a `Figure`; it never calls `plt.show` or touches your global pyplot state, so call `fig.savefig(...)` or `fig.show()` yourself.

`rank_distribution` is the one most worth reaching for: a per-contestant × position heatmap of how often each model landed in each slot, surfacing the panel's agreement-or-split that the aggregate score hides — the methodology's "bias as signal" story made visual.

```bash
uv add tournament-eval --extra plotting      # + matplotlib
```

```python
from tournament_eval.presentation import plots

fig = plots.rank_distribution(ballots)
fig.savefig("rank_distribution.png")
```

Importing the module pulls in neither matplotlib nor numpy — only *calling* a plot function does. A missing `plotting` extra raises a clear `ImportError` naming the package and the install command, the same pattern as the pairwise methods and `numpy`.

## Tested and typed

- 100% source coverage; `mypy --strict` and `ruff` clean.
- The orchestration is exercised offline against Pydantic AI's built-in test models (`TestModel`, `FunctionModel`) — no live network, no SDK mocking.
- Pure async orchestration, no hidden global state.

```bash
uv run pytest          # tests
uv run mypy            # type check
uv run ruff check      # lint
```

## License

MIT. See [LICENSE](LICENSE).
