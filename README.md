# Circular Tournament Evaluation Framework — Briefing

## Intent

Build an **open-source ranking aggregation system** for evaluating multiple language models on subjective quality tasks (translation, code review, content evaluation, etc.). The system allows practitioners to:
1. Run comparative evaluation on their own models
2. Measure and capture judge bias as meaningful data
3. Get multiple perspectives on quality without relying on a single judge
4. Validate fine-tuned models before production deployment

## What Separates It

**Circular evaluation design:** Every model being evaluated *also* acts as a judge. This creates a complete graph where all candidates assess all outputs, including their own.

**Order-invariant ranking aggregation:** Uses mathematically sound methods (Borda Count primary, Condorcet + Bradley-Terry as alternatives) that produce identical results regardless of input order or judge sequence. This is deliberate—no artificial "state" or learning across judgments.

**Explicit bias measurement:** Self-bias becomes a first-class output. When Model A judges itself differently than others judge it, that's captured and reported. Not hidden, not corrected—measured.

**Full reasoning capture:** Doesn't just extract "winner," captures all judges' deliberations and rankings. This enables post-hoc analysis of judge preferences (e.g., "Sonnet prioritizes fluency, GPT prioritizes accuracy").

## What It's NOT

**Not pairwise comparison.** It takes full ordinal rankings (1st, 2nd, 3rd, etc.) and aggregates them, not binary "A vs B" matchups.

**Not sequential/stateful.** Unlike Elo, it doesn't assume model strength changes over time or that match history matters. Each evaluation is independent.

**Not single-judge evaluation.** Avoids the bias and brittleness of LLM-as-judge approaches that rely on one model's perspective.

**Not debate-based consensus.** Models don't iterate or persuade each other; they independently rank, then aggregation happens. Cleaner, deterministic.

**Not benchmark/leaderboard infrastructure.** It's evaluation methodology for your specific models on your specific task, not a public ranking system.

## Key Benefits

**Multi-perspective signal:** Five judges × 500 test cases = 2,500 independent ranking observations. Signal is robust to individual judge idiosyncrasies.

**Bias as data:** Self-bias reveals model confidence. A model that overrates itself vs. underrates competitors gives you insight into its "personality." This is useful for understanding model behavior, not a flaw to eliminate.

**Post-hoc analysis flexibility:** Run the evaluation *once*, then compute Borda, Condorcet, Bradley-Terry, or future ranking methods on the same data without re-running. Experiment with aggregation approaches free.

**Cost-effective for production validation:** ~$130 for full multi-model evaluation beats hiring annotators ($5k+) or running human evaluation loops.

**Suited for subjective quality tasks:** Translation, localization, code review, content quality—domains where there's no ground truth, only "better/worse." Perfect fit.

**Transparency:** All judges' reasoning is visible. Can correlate final rankings with human validator feedback to see which judges aligned best with human preference.