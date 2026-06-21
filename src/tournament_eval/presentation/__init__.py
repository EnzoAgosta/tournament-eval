"""Presentation: render aggregation results as text or plots.

A *convenience* layer over :mod:`tournament_eval.aggregation`, in the same spirit —
pure functions you opt into, never a mandate.  The pipeline and the math hand you
data; this package hands you the handful of shapes everyone reaches for at the end
of a run.

**Two modules, mirroring :mod:`tournament_eval.aggregation`:**

* :mod:`tournament_eval.presentation.console` — aligned text tables (leaderboard,
  pairwise grid, ballot summary).  Pure Python, **no extra dependencies**.
* :mod:`tournament_eval.presentation.plots` — matplotlib figures (leaderboard bar
  chart, pairwise heatmap, rank distribution).  Need the ``plotting`` extra **at
  call time only** — importing the module won't pull matplotlib in, same pattern as
  :mod:`tournament_eval.aggregation.pairwise` and ``numpy``.

Everything here returns the rendered artifact (a ``str`` or, later, a
:class:`matplotlib.figure.Figure`) rather than printing or calling ``plt.show`` — so
the output is composable (write to a file, embed in a report) and testable without
touching your global pyplot state.

Not re-exported at the top level — reach for it via its submodule, the way
``aggregation`` is reached::

    from tournament_eval.presentation import console, plots

    print(console.leaderboard(scores))
    print(console.pairwise_matrix(tally))
    fig = plots.leaderboard_bar(scores)
    fig.savefig("leaderboard.png")
"""

from . import console, plots
from .console import ballot_summary, leaderboard, pairwise_matrix
from .plots import leaderboard_bar, pairwise_heatmap, rank_distribution

__all__ = [
    "ballot_summary",
    "console",
    "leaderboard",
    "leaderboard_bar",
    "pairwise_heatmap",
    "pairwise_matrix",
    "plots",
    "rank_distribution",
]
