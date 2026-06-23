"""Aggregation: collapse a tournament's per-ranker orderings into a verdict.

The pipeline (:mod:`tournament_eval.orchestration`) hands you validated per-ranker
rankings; this package turns a pile of those into a leaderboard.  It is a
*convenience*, not a mandate — every function here is pure and operates on plain
ballots, so you can use one, several, or none and do your own math over the data on
disk instead.

**The one invariant: there are never ties.**  Every ballot is a *strict* total
order, best-first, with each contestant appearing at most once — the shape the
default ranking template guarantees.  Every method here assumes this; none model
tie-groups.  A ballot that ranks the same contestant twice is malformed (not a tie)
and is rejected.

**Two styles, two modules — reach for the one you want:**

* :mod:`tournament_eval.aggregation.positional` — Borda and its normalized variant.
  Pure Python, **no ``numpy`` needed**.  Scores depend on *where* each ballot ranks a
  contestant.
* :mod:`tournament_eval.aggregation.pairwise` — the head-to-head tally
  (:func:`~tournament_eval.aggregation.pairwise.matrix`) and the methods built on it
  (Copeland, Schulze, with more to come).  Need ``numpy`` (the ``analysis`` extra) **at call
  time only** — importing the module does not pull it in, so this package and a plain
  tournament run stay dependency-free until you actually call a pairwise method.
* :mod:`tournament_eval.aggregation.bias` — ranker preference patterns the aggregate
  scores hide: a ranker x contestant mean-placement grid and the per-ranker
  self-preference number.  Pure Python (no ``numpy``), and the one module that reads
  the ballot's ``author`` — ranker identity *is* the signal there.

The shared bridge from the tournament data model to these methods lives in
:mod:`tournament_eval.aggregation.ballots`:

    from tournament_eval.aggregation import ballots_from_rankings, positional, pairwise

    ballots = ballots_from_rankings(rankings, generations)
    positional.borda(ballots)              # raw Borda; assumes full participation
    positional.normalized_borda(ballots)   # mean per-ballot score in [0, 1]; fair under unequal participation
    pairwise.copeland(ballots)             # wins - losses; a Condorcet winner tops it

Every method that would otherwise silently gloss over something (a sub-two-candidate
ballot, an omitted expected contestant) takes a ``strict=False`` flag: flip it to
``True`` to raise instead.  The package-wide ``strict`` means one thing everywhere —
"raise on anything that would otherwise be handled silently."
"""

from . import bias, pairwise, positional
from .ballots import Ballot, ballots_from_rankings

__all__ = [
    "Ballot",
    "ballots_from_rankings",
    "bias",
    "pairwise",
    "positional",
]
