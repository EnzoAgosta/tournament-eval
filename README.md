# tournament-eval

Circular tournament evaluation for language models.

## What it is

A framework where every model being evaluated also acts as a judge. Each model generates outputs for a shared set of tasks, then every model independently ranks every other model's outputs—including its own. The result is a complete graph of rankings with full reasoning capture.

## Why this matters

**Bias is not a bug, it's a signal.**

When a model overrates its own output compared to how peers rate it, that tells you something about its calibration and confidence. When it systematically underrates a competitor, that reveals a preference pattern. These biases are measured, not corrected away. They become part of your evaluation dataset.

**Deterministic by design.**

There is no Elo, no sequential history, no "learning" across judgments. Every evaluation is independent. Feed the same data in any order and you get the same result. The math is order-invariant: Borda Count, Condorcet, Bradley-Terry—pick your aggregation, the underlying rankings never change.

**Model-agnostic.**

The framework only cares about rankings. The models can be GPT-4, Claude, a local Llama, or a fine-tuned checkpoint you trained yesterday. They don't need to agree. They don't even need to know what the others are. Each produces an ordinal ranking; the aggregation method handles the rest.

**No single point of failure.**

Five models judging five outputs each means 25 independent rankings per task. If one model hallucinates, its noise is diluted by the other four. If one model is biased, its bias is visible and measurable against the consensus. You are not relying on any single model's judgment.

**Full reasoning capture.**

Every judge's deliberation text is preserved. This means you can run the evaluation once, then analyze:
- Which model values fluency over accuracy?
- Which model is most self-consistent?
- Which model's reasoning correlates with human preference?

The rankings are the output. The reasoning is the audit trail.

## What it's not

- Not pairwise comparison. Full ordinal rankings, not binary "A vs B" votes.
- Not stateful. No match history, no drifting scores. Each run is self-contained.
- Not a consensus builder. Models don't debate or iterate. They independently judge; aggregation happens post-hoc.
- Not a benchmark platform. It's evaluation methodology for your specific models on your specific task.

## Tested, minimal, extensible

- 58 tests, 100% source coverage
- Pure async functions, no hidden state
- One concrete client (Ollama), abstract base ready for OpenAI, Anthropic, etc.
- Aggregation math is intentionally out of scope: you own the rankings, compute what you want