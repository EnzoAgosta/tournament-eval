# tournament-eval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![Coverage: 100%](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#tested-and-typed)

Circular ranking for language models — and a clean set of primitives for wiring many different LLMs together to do it.

## Why this exists

I built this while fine-tuning models for translation. Translation is a task with no single correct answer — two fluent translations can both be "right," and a BLEU score won't tell you which one a human would prefer. I wanted a way to rank my fine-tunes that didn't cost a fortune in API calls to a single "grader" model, and that didn't quietly inherit that one grader's blind spots.

So instead of trusting one judge, I let a panel of models rank each other's work. That turned out to be a more interesting idea than I expected — and getting many different LLMs to actually cooperate cleanly turned out to be most of the work. This is both: the ranking methodology, and the plumbing to run it.

## The idea

You have a set of models and a set of tasks where quality is subjective — translation, summarization, style transfer, creative writing. There's no gold answer to grade against.

`tournament-eval` runs the evaluation as two passes over a pool of models:

1. **Generation** — every contestant produces an output for every task.
2. **Ranking** — every judge ranks the whole field of outputs for each task, with authorship hidden behind aliases (`A`, `B`, `C`...).

The contestants and the judges are just two lists you pass in. Point them at the **same** pool and you get the circular case — a peer review where the models being ranked are also the ones doing the ranking. Point them at **different** pools and you get a panel of trusted graders scoring a field of candidates. Same pipeline either way.

Each judge ranks independently — no debate, no consensus step, they never see each other's verdicts. You get a set of complete, reasoned rankings; collapsing them into a single result comes later, and it's yours to define.

## Why peer ranking instead of a single judge

**Bias becomes a signal, not a problem to hide.**
When a model overrates a translation that its peers rate poorly, that's a measurable fact about its calibration. When it systematically underrates a competitor, that's a visible preference pattern. These biases aren't corrected away — they're recorded, and they become part of your dataset. A single judge gives you its biases with no way to see them; a panel lets you measure them against each other.

**No single point of failure.**
If one model hallucinates, its noise is diluted by the others. If one model is biased, that bias is legible against the panel rather than silently baked into every score.

**Model-agnostic.**
The framework only cares about ordinal rankings. The models can be GPT, Claude, a local Llama, or the checkpoint you fine-tuned an hour ago. They don't need to agree, and they don't need to know anything about each other.

**Order-invariant aggregation.**
There's no Elo, no match history, no score that drifts as judgments accumulate. Every ranking is independent. Once you have the set of rankings, aggregating them is order-invariant — Borda count, Condorcet, Bradley–Terry, whatever you choose — and the result doesn't depend on the order you feed the rankings in.

> **A note on determinism.** Generation itself is *not* reproducible — models are sampled at a temperature, so two runs can differ. The order-invariance is a property of the aggregation math over a fixed set of rankings, not of the model outputs. If you need reproducible generation, pin temperature/seed at the client level.

**Full reasoning capture.**
Every judge's reasoning is preserved alongside its ranking, so you can run the tournament once and analyze it many ways — which model values fluency over accuracy, which is most self-consistent, which one's reasoning tracks human preference. The rankings are the output; the reasoning is the audit trail.

## Not Elo, not pairwise

If you've ranked models before, you've probably reached for one of two tools. This is neither.

**Elo — and its descendants Glicko, TrueSkill, arena leaderboards — is sequential by construction.** Every match nudges a running score, so a rating depends on match history and the order games were played in. There's no running score here to nudge: every judgment is independent and stateless. Feed the rankings in any order and the aggregate is identical — by design.

**Pairwise comparison collapses every decision to "A vs B."** To order N candidates you collect a pile of binary votes and reconstruct a global order, which can contradict itself (A beats B beats C beats A). Here each judge ranks the entire field at once and returns one complete ordinal order, reasoned over all candidates together, rather than stitched back from fragments.

### Contestants and judges are separate axes

`generate_all` and `rank_all` each take their own list of clients, so the two roles are fully decoupled:

- **N candidates judged by the same N models** — the full circular tournament.
- **2 candidates, 15 judges** — a small head-to-head settled by a large, diverse panel.
- **15 candidates, 2 judges** — a wide field filtered by a couple of trusted graders.
- **5 fine-tunes judged by 3 frontier models** — contestants and judges from entirely different tiers.

Same pipeline, same data model. Who generates and who judges are just two lists.

## Install

```bash
uv add tournament-eval                     # core — the built-in httpx client, no SDKs
uv add tournament-eval --extra openai      # + the official OpenAI SDK client
uv add tournament-eval --extra anthropic   # + the Anthropic SDK client
uv add tournament-eval --extra ollama      # + the Ollama SDK client
uv add tournament-eval --extra bedrock     # + Amazon Bedrock (boto3)
```

The core install talks to anything speaking the OpenAI `/chat/completions` protocol over HTTP — local servers included — with no extra dependencies. Reach for an extra only when you want a specific provider's official SDK.

## Quickstart

Requires Python 3.14+. This example ranks two local models served by [Ollama](https://ollama.com) through its OpenAI-compatible endpoint, so it needs no extras.

```python
import asyncio

from tournament_eval import (
    OpenAICompatibleClient,
    build_ranking_tasks,
    generate_all,
    generation_tasks,
    rank_all,
    ranking_by_author,
)


async def main() -> None:
    # A pool of models. The built-in client speaks the OpenAI /chat/completions
    # protocol — point base_url at any local server (Ollama here, or mlx_lm, vLLM...).
    # Local servers ignore the API key, so any non-empty string works.
    clients = [
        OpenAICompatibleClient(model_id="llama3.2", base_url="http://localhost:11434/v1", api_key="local"),
        OpenAICompatibleClient(model_id="gemma3", base_url="http://localhost:11434/v1", api_key="local"),
    ]

    # One GenerationTask per sentence, sharing a base prompt. (Or pull the lines
    # from a text file with generation_tasks_from_file(base_prompt, "sentences.txt").)
    tasks = generation_tasks(
        "Translate into French. Output only the translation, no notes.",
        ["The quick brown fox jumps over the lazy dog.", "She sells seashells by the sea shore."],
    )

    # 1. Every model produces an output for every task.
    generations, gen_failures = await generate_all(tasks, clients)

    # 2. Group outputs per task and hide authorship behind aliases (A, B, ...).
    ranking_tasks = build_ranking_tasks(
        tasks, generations, "Rank these French translations by fluency and accuracy, best first."
    )

    # 3. Every judge ranks the whole field. Same clients here = the circular case.
    rankings, rank_failures = await rank_all(ranking_tasks, generations, clients)

    # Read each judge's verdict back as author names, best first.
    for r in rankings:
        print(f"{r.author} ranked: {ranking_by_author(r, generations)}")
        if r.reasoning:
            print(f"  reasoning: {r.reasoning[:120]}...")


asyncio.run(main())
```

No `async with` to manage: `generate_all` / `rank_all` open and close any clients that own resources (like the built-in httpx client) for the duration of the batch.

## The clients

Everything the pipeline needs from a model is one tiny contract — the `LLMClient` **Protocol**: a `name`, and async `generate` / `generate_structured`. Anything that satisfies it works; there's no base class to inherit.

The design principle is **lean on the providers' own SDKs**. Rather than re-implement each provider's auth, endpoints, retries, and wire quirks, the SDK-backed clients wrap a native client *you* construct and own — so you configure the provider its canonical way, and the adapter just maps it onto the contract.

| Client | Install | Wraps | Notes |
|--------|---------|-------|-------|
| `OpenAICompatibleClient` | core | — (httpx) | any OpenAI `/chat/completions` server: local (Ollama, mlx_lm, vLLM, llama.cpp) or the OpenAI API |
| `OpenAIClient` | `[openai]` | `openai.AsyncOpenAI` | the official SDK, via the Responses API |
| `AnthropicClient` | `[anthropic]` | `anthropic.AsyncAnthropic` | Messages API; native json-schema structured output |
| `OllamaClient` | `[ollama]` | `ollama.AsyncClient` | structured output via Ollama's grammar-constrained `format` |
| `AnthropicBedrockClient`, `NovaBedrockClient`, `TitanBedrockClient`, `LlamaBedrockClient`, `MistralBedrockClient`, `CohereBedrockClient`, `JambaBedrockClient`, `PalmyraBedrockClient`, `GptOssBedrockClient` | `[bedrock]` | a boto3 `bedrock-runtime` client | one per model family — **Bedrock = boto**, auth is the AWS credential chain |

The built-in client takes plain keyword config; the SDK-backed ones take an injected client plus the same shared generation knobs:

```python
import boto3
from anthropic import AsyncAnthropic
from tournament_eval import AnthropicClient, AnthropicBedrockClient

# Anthropic's first-party API, via the official SDK (you own auth + config):
claude = AnthropicClient(AsyncAnthropic(api_key="..."), model_id="claude-sonnet-4-6")

# Claude on Bedrock, via boto3 (auth = the AWS credential chain, handled by boto):
claude_bedrock = AnthropicBedrockClient(
    boto3.client("bedrock-runtime", region_name="us-east-1"),
    model_id="anthropic.claude-sonnet-4-5-20250929-v1:0",
)
```

All clients share the same generation config — `model_id`, optional `name` (the label used as `author`), `temperature`, `max_tokens`, `system_prompt`, `max_concurrency`, `reasoning_effort`. `reasoning_effort` (`low`/`medium`/`high`/`xhigh`/`max`) is the provider-agnostic reasoning dial: each client maps it onto its backend (Anthropic adaptive thinking + effort, OpenAI Responses reasoning effort, Ollama think mode, the OpenAI-compatible `reasoning_effort` field — best-effort), and turning it on is what populates the captured `reasoning` trace. `xhigh`/`max` are Anthropic's full range; other backends clamp down to their ceiling. A model with no reliable structured output (most Bedrock families over the Invoke API fall back to best-effort prompt-injection) makes a fine *contestant* even if it's a flaky *judge* — which is exactly why the two roles are separate lists.

Adding a provider is implementing the three-method Protocol — no framework surgery.

## How it works

The pipeline is three async functions, each a clean stage:

| Stage | Function | In → Out |
|-------|----------|----------|
| Generate | `generate_all(tasks, clients)` | tasks × clients → `GenerationResult`s |
| Build | `build_ranking_tasks(tasks, generations, prompt)` | generations grouped per task, aliased → `RankingTask`s |
| Rank | `rank_all(ranking_tasks, generations, clients)` | ranking tasks × clients → `RankingResult`s |

Each stage fans its work out concurrently with `asyncio.gather`. **Failures aren't thrown** — each stage returns a `(results, failures)` pair, where every failure is a typed record (`GenerationFailure` / `RankingFailure`) carrying the failing id, the `author`, and the error type and message. One model timing out or returning garbage doesn't sink the run; it lands in `failures` for you to inspect or retry.

Ranking responses are validated strictly: the ranking must list every candidate exactly once — no unknown aliases, no duplicates, no missing entries, no ties. A malformed ranking is a failure, not a silent best guess.

## Customizing the ranking

How a ranking model is prompted, constrained, and validated lives in a `RankingTemplate` — a single object owning the three pieces that must agree with each other:

- `render(ranking_task, candidates)` — the full prompt the ranking model sees
- `schema` — the JSON schema its response is constrained to
- `parse(data, valid_aliases)` — validation into a `ParsedRanking`

`rank_all` / `rank_one` take a `template=` argument (default `DefaultRankingTemplate`, a strict total order with no ties). By default it shows the ranker the **original task the models answered** — shared across candidates, so it leaks no authorship — followed by the anonymized outputs. To also surface each candidate's **reasoning trace**, flip one flag (off by default, since traces are long and a verbose reasoner can look more thorough than it is):

```python
rankings, failures = await rank_all(
    ranking_tasks, generations, clients, template=DefaultRankingTemplate(include_reasoning=True)
)
```

For anything more, subclass. Adding context (a rubric, domain notes) is a one-method override of `render` — each candidate is passed as its full `GenerationResult`, so `.output`, `.reasoning`, `.generation_prompt`, and `.metadata` are all in reach:

```python
from collections.abc import Mapping
from tournament_eval import DefaultRankingTemplate, GenerationResult, RankingTask


class RubricTemplate(DefaultRankingTemplate):
    def render(self, ranking_task: RankingTask, candidates: Mapping[str, GenerationResult]) -> str:
        return "Grading rubric: prioritise factual accuracy over fluency.\n\n" + super().render(
            ranking_task, candidates
        )


rankings, failures = await rank_all(ranking_tasks, generations, clients, template=RubricTemplate())
```

Changing the *shape* of the verdict (e.g. allowing ties) means overriding all three methods so the prompt, schema, and parser stay consistent. (`candidates` exposes `.author` — don't render it into the prompt, or you defeat the anonymization.)

## Concurrency

Every stage fans out with `asyncio.gather`, so by default **every request fires at once**. The concurrency limit lives on the *client*, not the orchestration — because rate limits belong to the provider, not to the tournament.

```python
import asyncio

# Per-client limit: at most 4 in-flight requests to this model.
client = OpenAICompatibleClient(model_id="gpt-4o", base_url="...", api_key="...", max_concurrency=4)

# Shared cap: hand the same semaphore to several clients and they draw from one budget.
sem = asyncio.Semaphore(8)
clients = [
    OpenAICompatibleClient(model_id="gpt-4o", base_url="...", api_key="...", semaphore=sem),
    OpenAICompatibleClient(model_id="gpt-4o-mini", base_url="...", api_key="...", semaphore=sem),
]
```

`max_concurrency` defaults to unbounded — fine for a local server you control, but against a rate-limited hosted API you'll want a conservative value, per-client or as a shared semaphore. The same limit applies whether you call the batch `generate_all` / `rank_all` or the single-shot `generate_one` / `rank_one`.

## Persistence & resume

A tournament is expensive — many model calls — so the pipeline streams every result and failure to disk *as it lands*. You choose the file paths; there's no directory layout or naming convention to learn. Typically one flat JSONL file per stream:

```python
results, failures = await generate_all(
    tasks, clients,
    results_path="run/generations.jsonl",
    failures_path="run/generation_failures.jsonl",
)
```

Each write is a single synchronous append, so a process that dies mid-run keeps everything finished so far. There's no per-model sharding — every record carries its own `author` and `task_id`, so a flat file is fully reconstructable; group or filter on read.

**Resume is automatic — just run the script again.** When you pass a `results_path`, `generate_all` reads it back first and skips every `(task, author)` pair that already succeeded, running only what's left and returning the **complete** set (loaded plus newly produced). So the return value is always a clean partition: `results` is every pair that now has a success, `failures` every pair still without one. Successes are skipped; **failures are always retried** — fix the cause (rate limit, API key, a flaky endpoint) and rerun, and only the still-broken pairs go out again.

```python
# Run once, get interrupted, run the exact same call again — it picks up where it left off.
results, failures = await generate_all(
    tasks, clients,
    results_path="run/generations.jsonl",
    failures_path="run/generation_failures.jsonl",
)
```

`rank_all` works identically, keyed on `(ranking_task, judge)` against its own `results_path`.

Reading a run back is per-file and typed — `read_generation_result_file`, `read_ranking_result_file`, and friends each restore the original dataclasses, UUIDs and all. Writing anything yourself (e.g. persisting your tasks) is the one type-agnostic `append_record(path, record)`.

**One caveat — ranking tasks freeze.** A `RankingTask` carries a random alias shuffle, so `build_ranking_tasks(..., tasks_path="run/ranking_tasks.jsonl")` builds each task once and, on a rerun, reuses the persisted one rather than rebuilding (a rebuild would re-shuffle and orphan every ranking already collected). The consequence: a generation that only succeeds on a *later* resume won't be added to an already-built ranking task. So **let generation finish before you start ranking** — run `generate_all` until its `failures` are empty, then build ranking tasks. To deliberately rebuild, delete the ranking-tasks file (and any rankings) first.

## Scope

**Aggregation is intentionally out of scope.** The framework hands you the rankings and the reasoning; collapsing them into a verdict — Borda, Condorcet, Bradley–Terry, Elo over the pairwise implications, whatever fits — is a real methodological decision, not a detail to bury in a library. It's all ordinal-ranking math over data you already have on disk, so it's yours for now (and may arrive later as an opt-in convenience).

Resolving the data is fair game, though — `ranking_by_author(ranking_result, generations)` turns one judge's id-based ranking back into author names (best first). That's de-anonymization for analysis, the line before aggregation begins.

## Tested and typed

- 100% source coverage; `mypy --strict` and `ruff` clean.
- The client adapters are tested against their **real** SDKs offline — httpx-based clients through `httpx.MockTransport`, Bedrock through botocore's `Stubber` — so the tests exercise actual request serialization and response parsing, not stand-ins.
- Pure async orchestration, no hidden global state.

```bash
uv run pytest          # tests
uv run mypy            # type check
uv run ruff check      # lint
```

## License

MIT. See [LICENSE](LICENSE).
