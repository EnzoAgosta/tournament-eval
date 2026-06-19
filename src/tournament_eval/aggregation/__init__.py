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

**Optional heavy dependencies.**  The positional methods (:func:`borda`,
:func:`normalized_borda`) are pure counting, always available, and re-exported here.  The
pairwise methods need ``numpy`` from the ``analysis`` extra and live in their own
submodules — import them directly::

    from tournament_eval.aggregation.pairwise import pairwise_matrix
    from tournament_eval.aggregation.copeland import copeland

This package root deliberately re-exports only the dependency-free methods, so importing
it (and ``tournament_eval``) never pulls in numpy; the submodule import path is where you
opt into the extra.
"""

from .ballots import Ballot, ballots_from_rankings
from .borda import borda, normalized_borda

__all__ = [
    "Ballot",
    "ballots_from_rankings",
    "borda",
    "normalized_borda",
]
