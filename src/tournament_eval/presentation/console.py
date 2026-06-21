"""Console presentation: render aggregation results as plain, aligned text.

A *convenience* layer over :mod:`tournament_eval.aggregation`, in the same spirit:
every function here is pure, returns a ``str``, and never touches ``stdout`` — call
``print(console.leaderboard(scores))`` yourself.  Returning strings (not printing)
keeps these composable (write to a file, pipe to a logger, embed in a bigger report)
and testable without capturing stdout.

**No heavy dependencies.**  Nothing here needs ``numpy``: :func:`pairwise_matrix`
only *reads* a :class:`~tournament_eval.aggregation.pairwise.PairwiseTally` someone
already built (which did need ``numpy``), so importing this module pulls in nothing
heavy.  The plot counterparts (matplotlib) live in :mod:`tournament_eval.presentation.plots`.

Scope is deliberately small — a leaderboard, the head-to-head grid, and a ballot
sanity summary.  Everything users compose themselves from the data on disk stays
theirs; this is the handful of shapes everyone reaches for at the end of a run.
"""

import statistics
from collections.abc import Iterable

from tournament_eval.aggregation.ballots import Ballot
from tournament_eval.aggregation.pairwise import PairwiseTally

_COLUMN_GAP = "  "
"""Gap between columns in the rendered tables — wide enough to read, narrow enough
to keep a wide field on one line."""

_DIAGONAL = "-"
"""Marker for the self-matchup cell of the pairwise matrix — a contestant doesn't
play itself, so a count there would be meaningless."""


def leaderboard(scores: dict[str, float], *, precision: int = 2) -> str:
    """Render a score map as a ranked, aligned leaderboard string.

    Sorts best-first (highest score first); ties break by contestant name ascending
    for deterministic output.  The score column is right-aligned and formatted to
    ``precision`` decimals, the rank column right-aligned to the rank count's width.
    No header — the rank, name, score layout is self-explanatory and keeps the output
    composable with your own surrounding text.

    Parameters
    ----------
    scores : dict[str, float]
        One entry per contestant (e.g. from
        :func:`~tournament_eval.aggregation.positional.borda` or
        :func:`~tournament_eval.aggregation.pairwise.copeland`).  Higher is better.
    precision : int
        Decimal places for the score column.  Default ``2`` — uniform and predictable
        across Borda's raw points, normalized Borda's ``[0, 1]``, and Copeland's
        integer-valued floats.

    Returns
    -------
    str
        One line per contestant, best first, e.g.
        ``"1  gpt-4o  0.72\\n2  claude-sonnet-4-6  0.68\\n3  llama-3.1  0.41"``.
        Empty string when ``scores`` is empty.
    """
    if not scores:
        return ""

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    name_width = max(len(name) for name, _ in ranked)
    rank_width = len(str(len(ranked)))
    score_strings = [format(score, f".{precision}f") for _, score in ranked]
    score_width = max(len(s) for s in score_strings)

    lines: list[str] = []
    for rank, ((name, _), score_str) in enumerate(zip(ranked, score_strings, strict=True), start=1):
        lines.append(f"{rank:>{rank_width}}{_COLUMN_GAP}{name:<{name_width}}{_COLUMN_GAP}{score_str:>{score_width}}")
    return "\n".join(lines)


def pairwise_matrix(tally: PairwiseTally) -> str:
    """Render a :class:`~tournament_eval.aggregation.pairwise.PairwiseTally` as an aligned grid.

    One header row of contestant labels, then one row per contestant with its
    head-to-head counts: ``above[i, j]`` — how many ballots rank the row contestant
    over the column contestant.  The self-matchup diagonal renders as ``"-"`` (a
    contestant doesn't play itself).  Per-column widths size to the wider of the
    header label and the column's counts, so long author names don't clip and short
    counts don't pad excessively.

    Parameters
    ----------
    tally : PairwiseTally
        The head-to-head counts from
        :func:`~tournament_eval.aggregation.pairwise.matrix`.

    Returns
    -------
    str
        The grid; e.g. for three contestants::

               a  b  c
            a  -  1  2
            b  1  -  2
            c  0  0  -

        Empty string when the tally has no contestants.

    Notes
    -----
    Reads only an already-built tally, so this call needs no ``numpy`` — building one
    (via :func:`~tournament_eval.aggregation.pairwise.matrix`) does.
    """
    contestants = tally.contestants
    n = len(contestants)
    if n == 0:
        return ""

    label_width = max(len(c) for c in contestants)
    cells: list[list[str]] = []
    for i in range(n):
        cell_row: list[str] = []
        for j in range(n):
            if i == j:
                cell_row.append(_DIAGONAL)
            else:
                cell_row.append(str(int(tally.above[i, j])))
        cells.append(cell_row)

    column_widths: list[int] = []
    for j in range(n):
        width = len(contestants[j])
        for i in range(n):
            width = max(width, len(cells[i][j]))
        column_widths.append(width)

    header = _COLUMN_GAP.join(f"{contestants[j]:>{column_widths[j]}}" for j in range(n))
    lines = [f"{'':<{label_width}}{_COLUMN_GAP}{header}"]
    for i in range(n):
        row = _COLUMN_GAP.join(f"{cells[i][j]:>{column_widths[j]}}" for j in range(n))
        lines.append(f"{contestants[i]:<{label_width}}{_COLUMN_GAP}{row}")
    return "\n".join(lines)


def ballot_summary(ballots: Iterable[Ballot]) -> str:
    """Summarize a set of ballots: turnout, field sizes, and per-contestant placement.

    Two things a sanity check wants before trusting an aggregate: whether every
    ballot ranked the full field (unequal participation biases raw Borda), and where
    each contestant actually placed (a high appearance count with a poor mean position
    is a different story from a high count with a good one).

    Renders a header line describing turnout and field-size spread, then a per-
    contestant table of appearances and mean position.  Mean position is 1-indexed
    (``1`` = best, the top of a ballot) and averaged over only the ballots a
    contestant appears in, so an absent contestant doesn't drag an average toward a
    field-size number.  Rows sort by appearances descending, then mean position
    ascending, then name — the most-participating, best-placing contestants first,
    deterministic on ties.

    Parameters
    ----------
    ballots : Iterable[Ballot]
        The ranked verdicts, each contestant labels best-first (e.g. from
        :func:`~tournament_eval.aggregation.ballots.ballots_from_rankings`).
        Materialized once, so a single-pass iterator is fine.

    Returns
    -------
    str
        A header line, a blank line, and a per-contestant table — e.g.::

            3 ballots, field size 3 (all equal)

            Contestant  Appearances  Mean position
            a                     3          1.33
            b                     3          2.00
            c                     3          2.67

        When field sizes vary, the header reports the spread instead (``min ...,
        median ..., max ...``).  When there are no ballots, returns ``"0 ballots"``;
        when ballots exist but every one is empty (no contestants ranked), returns
        just the header line.
    """
    materialized = list(ballots)
    n_ballots = len(materialized)
    if n_ballots == 0:
        return "0 ballots"

    sizes = [len(ballot) for ballot in materialized]
    if len(set(sizes)) == 1:
        size_line = f"{n_ballots} ballots, field size {sizes[0]} (all equal)"
    else:
        size_line = (
            f"{n_ballots} ballots, field size min {min(sizes)}, median {statistics.median(sizes)}, max {max(sizes)}"
        )

    positions: dict[str, list[int]] = {}
    for ballot in materialized:
        for index, label in enumerate(ballot):
            positions.setdefault(label, []).append(index + 1)

    rows: list[tuple[str, int, float]] = []
    for label, placements in positions.items():
        appearances = len(placements)
        mean_position = sum(placements) / appearances
        rows.append((label, appearances, mean_position))
    if not rows:
        return size_line

    rows.sort(key=lambda r: (-r[1], r[2], r[0]))
    name_width = max(len(label) for label, _, _ in rows)
    appearances_width = max(len("Appearances"), max(len(str(a)) for _, a, _ in rows))
    mean_width = max(len("Mean position"), max(len(f"{m:.2f}") for _, _, m in rows))

    lines = [size_line, ""]
    lines.append(
        f"{'Contestant':<{name_width}}{_COLUMN_GAP}{'Appearances':>{appearances_width}}"
        f"{_COLUMN_GAP}{'Mean position':>{mean_width}}"
    )
    for label, appearances, mean_position in rows:
        lines.append(
            f"{label:<{name_width}}{_COLUMN_GAP}{appearances:>{appearances_width}}"
            f"{_COLUMN_GAP}{mean_position:>{mean_width}.2f}"
        )
    return "\n".join(lines)
