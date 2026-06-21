"""Plot presentation: render aggregation results as matplotlib figures.

A *convenience* layer over :mod:`tournament_eval.aggregation`, the plot
counterpart to :mod:`tournament_eval.presentation.console`.  Every function here
is pure, builds a :class:`matplotlib.figure.Figure`, and returns it — never
calls ``plt.show()`` or ``plt.savefig()`` and never touches the caller's global
pyplot state beyond creating the figure.  Call ``fig.savefig(...)`` or
``fig.show()`` yourself, or embed the figure in a notebook/report.

**``matplotlib`` is required at call time, not import time.**  Importing this
module — and referencing its return type — pulls in no heavy dependency:
``matplotlib`` is referenced only under ``TYPE_CHECKING`` (via the PEP 695
:data:`_Figure` alias, which the runtime never evaluates), and imported lazily
inside each function via :func:`~tournament_eval._lazy_imports.require`.  So a
plain ``import tournament_eval.presentation.plots`` stays dependency-free;
calling any function here without the ``plotting`` extra raises a clear
:class:`ImportError` naming the package and the install command.

No ``numpy`` is used here: heatmaps are fed plain Python lists of lists (with
``float('nan')`` for masked cells), and bar values are plain Python floats.  A
:class:`~tournament_eval.aggregation.pairwise.PairwiseTally`'s ``above`` array
is read element-by-element rather than via numpy ops, so this module never
imports ``numpy`` — only the call that *built* the tally did.

Scope is deliberately small — a leaderboard bar chart, the head-to-head
heatmap, and a rank-position distribution.  The methodology's "bias as signal"
story lives in the distribution of rankings, not the aggregate score, so
:func:`rank_distribution` is the one most worth reaching for.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from tournament_eval._lazy_imports import require
from tournament_eval.aggregation.ballots import Ballot
from tournament_eval.aggregation.pairwise import PairwiseTally

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

type _Figure = Figure
"""The return type of every plot function — a PEP 695 alias so the annotation
costs no ``matplotlib`` import at module load (the alias's value is never
evaluated at runtime).  Resolved to :class:`matplotlib.figure.Figure` only under
``TYPE_CHECKING``."""

type _Axes = Axes
"""A matplotlib :class:`~matplotlib.axes.Axes` — same PEP 695 lazy-alias trick as
:data:`_Figure`, used only to type :func:`_new_figure`'s return."""


def _new_figure(figsize: tuple[float, float]) -> tuple[_Figure, _Axes]:
    """Build a fresh single-axes figure via pyplot and return ``(fig, ax)``.

    Centralised so every plot function gets the figure the same way and the
    ``matplotlib`` import lives in one place.  Uses ``plt.subplots`` (the
    conventional entry point) so the returned :class:`_Figure` works with
    ``fig.savefig``, ``fig.show``, and notebook embedding; we don't call
    ``plt.show`` or otherwise finalize — that's the caller's call.
    """
    require("matplotlib", extra="plotting")
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=figsize)


def leaderboard_bar(scores: dict[str, float], *, precision: int = 2) -> _Figure:
    """Render a score map as a horizontal bar chart, best at the top.

    One horizontal bar per contestant, sorted so the highest score sits at the
    top of the axes (worst at the bottom) — the visual analogue of
    :func:`~tournament_eval.presentation.console.leaderboard`.  Bars are labelled
    at their end with the score formatted to ``precision`` decimals.

    Parameters
    ----------
    scores : dict[str, float]
        One entry per contestant (e.g. from
        :func:`~tournament_eval.aggregation.positional.borda` or
        :func:`~tournament_eval.aggregation.pairwise.copeland`).  Higher is better.
    precision : int
        Decimal places for the end-of-bar score labels.  Default ``2``, matching
        :func:`~tournament_eval.presentation.console.leaderboard`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure — call ``fig.savefig(...)`` or ``fig.show()`` to render it.

    Notes
    -----
    Requires ``matplotlib`` (the ``plotting`` extra) at call time; importing this
    module does not.  A missing extra raises :class:`ImportError` with the install
    command.
    """
    fig, ax = _new_figure(figsize=(8, max(1.0, 0.5 * len(scores) + 0.6)))
    if scores:
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        labels = [name for name, _ in ranked]
        values = [score for _, score in ranked]
        bars = ax.barh(labels, values)
        ax.invert_yaxis()  # best-first (highest score) at the top.
        ax.bar_label(bars, fmt=f"%.{precision}f", padding=3)
    ax.set_xlabel("score")
    ax.set_title("Leaderboard")
    fig.tight_layout()
    return fig


def pairwise_heatmap(tally: PairwiseTally) -> _Figure:
    """Render a :class:`~tournament_eval.aggregation.pairwise.PairwiseTally` as a heatmap.

    One cell per ordered pair ``(i, j)`` showing ``above[i, j]`` — how many
    ballots rank ``contestants[i]`` over ``contestants[j]`` — with the
    self-matchup diagonal masked (a contestant doesn't play itself, so a count
    there would be meaningless).  Rows and columns are labelled with the
    contestant names, and each cell is annotated with its count.

    Parameters
    ----------
    tally : PairwiseTally
        The head-to-head counts from
        :func:`~tournament_eval.aggregation.pairwise.matrix`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure — call ``fig.savefig(...)`` or ``fig.show()`` to render it.

    Notes
    -----
    Reads only an already-built tally, so this call needs no ``numpy`` — building
    one (via :func:`~tournament_eval.aggregation.pairwise.matrix`) does.  Requires
    ``matplotlib`` (the ``plotting`` extra) at call time; importing this module
    does not.
    """
    contestants = tally.contestants
    n = len(contestants)
    fig, ax = _new_figure(figsize=(max(4.0, 0.8 * n + 1.6), max(3.2, 0.8 * n + 1.0)))
    if n > 0:
        # Pure-Python grid; NaN on the diagonal so the self-matchup cell renders
        # blank instead of showing a misleading 0 count.
        data: list[list[float]] = []
        for i in range(n):
            row: list[float] = []
            for j in range(n):
                if i == j:
                    row.append(float("nan"))
                else:
                    row.append(float(int(tally.above[i, j])))
            data.append(row)
        image = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(n), contestants, rotation=45, ha="right")
        ax.set_yticks(range(n), contestants)
        ax.set_xlabel("column contestant")
        ax.set_ylabel("row contestant")
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ax.text(j, i, str(int(tally.above[i, j])), ha="center", va="center")
        fig.colorbar(image, ax=ax, label="ballots ranking row over column")
    ax.set_title("Pairwise head-to-head")
    fig.tight_layout()
    return fig


def rank_distribution(ballots: Iterable[Ballot]) -> _Figure:
    """Render how often each contestant landed in each position, as a heatmap.

    Rows are contestants, columns are 1-indexed positions (``1`` = best, the top
    of a ballot), and each cell is the number of ballots that placed that
    contestant in that position.  A contestant that consistently tops the field
    shows a hot cell in column 1; one that bimodally wins-or-loses shows weight
    in both column 1 and the last column — the "did the panel agree or split"
    story the aggregate score hides, which is the methodology's whole point.

    Contestants are sorted by appearances descending, then mean position ascending
    (the same order :func:`~tournament_eval.presentation.console.ballot_summary`
    uses), so the most-participating, best-placing contestants sit at the top.
    The position-axis spans ``1..max_field_size``; ballots of unequal size leave
    the upper cells for shorter ballots at zero, which is correct (a contestant
    can't be placed 5th in a 3-candidate ballot).

    Parameters
    ----------
    ballots : Iterable[Ballot]
        The ranked verdicts, each contestant labels best-first (e.g. from
        :func:`~tournament_eval.aggregation.ballots.ballots_from_rankings`).
        Materialized once, so a single-pass iterator is fine.

    Returns
    -------
    matplotlib.figure.Figure
        The figure — call ``fig.savefig(...)`` or ``fig.show()`` to render it.

    Notes
    -----
    Requires ``matplotlib`` (the ``plotting`` extra) at call time; importing this
    module does not.  No ``numpy`` is used — counts are plain Python ints fed to
    ``imshow`` as a list of lists.
    """
    materialized = list(ballots)

    positions: dict[str, list[int]] = {}
    for ballot in materialized:
        for index, label in enumerate(ballot):
            positions.setdefault(label, []).append(index + 1)

    rows: list[tuple[str, int, float]] = [
        (label, len(placements), sum(placements) / len(placements)) for label, placements in positions.items()
    ]
    rows.sort(key=lambda r: (-r[1], r[2], r[0]))

    max_field_size = max((len(ballot) for ballot in materialized), default=0)
    labels = [label for label, _, _ in rows]

    fig, ax = _new_figure(figsize=(max(4.0, 0.8 * max_field_size + 1.6), max(3.0, 0.5 * len(labels) + 1.0)))
    if labels and max_field_size > 0:
        grid: list[list[float]] = []
        for label, _, _ in rows:
            placements = positions[label]
            counts = [0.0] * max_field_size
            for position in placements:
                counts[position - 1] += 1.0
            grid.append(counts)
        image = ax.imshow(grid, cmap="Blues")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xticks(range(max_field_size), [str(p) for p in range(1, max_field_size + 1)])
        ax.set_xlabel("position (1 = best)")
        ax.set_ylabel("contestant")
        for i in range(len(labels)):
            for j in range(max_field_size):
                count = int(grid[i][j])
                if count:
                    ax.text(j, i, str(count), ha="center", va="center")
        fig.colorbar(image, ax=ax, label="ballots placing contestant in position")
    ax.set_title("Rank distribution")
    fig.tight_layout()
    return fig
