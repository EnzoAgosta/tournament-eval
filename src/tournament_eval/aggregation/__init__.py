"""Aggregation: collapse a tournament's per-ranker orderings into a verdict.

The tournament pipeline (:mod:`tournament_eval.orchestration`) hands you validated,
per-ranker rankings; this package turns a pile of those into a leaderboard.  It is a
*convenience*, not a mandate — every function here is pure and operates on plain
ballots, so you can use one, several, or none and do your own math over the data on
disk instead.

**The one invariant: there are never ties.**  Every ballot is a *strict* total order,
best-first, with each contestant appearing at most once — the shape the default
ranking template guarantees (see :class:`~tournament_eval.ranking.DefaultRankingTemplate`).
Every method here assumes this and will keep assuming it until further notice; none of
them model tie-groups.  A ballot that ranks the same contestant twice is malformed (not
a tie) and is rejected.
"""

from .ballots import Ballot, ballots_from_rankings
from .borda import borda, normalized_borda

__all__ = [
    "Ballot",
    "ballots_from_rankings",
    "borda",
    "normalized_borda",
]
