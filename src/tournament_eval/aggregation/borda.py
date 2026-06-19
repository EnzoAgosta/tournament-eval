"""Borda count and its participation-normalized variant.

Each ballot is a strict ranking of *k* contestants, best-first.  A ballot awards
``k - 1`` points to its top contestant, ``k - 2`` to the next, down to ``0`` for the
last.

* :func:`borda` sums those raw points across every ballot; higher is better.  It only
  compares fairly under **full participation** (every contestant in every ballot).
* :func:`normalized_borda` rescales each ballot's points to ``[0, 1]`` and averages a
  contestant over only the ballots it *appears in*, so **unequal participation** — a
  contestant absent from some ballots because, say, its generation failed for that task
  — no longer penalizes it for not competing rather than for losing.

**No ties.**  Like everything in :mod:`tournament_eval.aggregation`, both functions
assume every ballot is a strict total order.  A contestant appearing twice in one ballot
is malformed (not a tie) and raises :class:`ValueError`.
"""

from collections.abc import Iterable

from tournament_eval.aggregation.ballots import Ballot


def _reject_ties(ballot: Ballot) -> None:
    """Raise if a ballot lists any contestant more than once (a malformed, non-tie ballot)."""
    if len(set(ballot)) != len(ballot):
        raise ValueError(f"Ballot lists a contestant more than once (no ties allowed): {ballot!r}")


def borda(ballots: Iterable[Ballot]) -> dict[str, float]:
    """Borda-count a collection of ballots into a raw score per contestant.

    Parameters
    ----------
    ballots : Iterable[Ballot]
        The ranked verdicts, each a list of contestant labels best-first (e.g. from
        :func:`~tournament_eval.aggregation.ballots.ballots_from_rankings`).  A ballot
        of *k* contestants awards ``k - 1 .. 0`` points top-to-bottom.

    Returns
    -------
    dict[str, float]
        Total Borda points per contestant label, summed over every ballot.  Higher is
        better; sort it yourself for a leaderboard.  Scores are floats so this composes
        with :func:`normalized_borda`.

    Raises
    ------
    ValueError
        If any single ballot lists a contestant more than once — a malformed ballot
        that would silently double-count.  (Distinct contestants tying is not
        representable in a ballot and so cannot occur; see the module docstring.)

    Notes
    -----
    Assumes **full participation**: the raw sum rewards a contestant for appearing in
    more ballots, so it only compares fairly when every contestant is in every ballot.
    For unequal participation, use :func:`normalized_borda`.
    """
    scores: dict[str, float] = {}
    for ballot in ballots:
        _reject_ties(ballot)
        last_index = len(ballot) - 1
        for position, label in enumerate(ballot):
            scores[label] = scores.get(label, 0.0) + (last_index - position)
    return scores


def normalized_borda(ballots: Iterable[Ballot]) -> dict[str, float]:
    """Borda count fair under unequal participation: each contestant's mean ballot score.

    Within a ballot of *k* contestants, raw Borda points (``k - 1 .. 0``) are rescaled to
    ``[0, 1]`` by dividing by ``k - 1`` — so the top contestant scores ``1.0`` and the
    bottom ``0.0`` regardless of how many candidates the ballot held.  A contestant's
    score is then the **mean** of those rescaled values over the ballots it appears in.
    Because it averages rather than sums, appearing in more ballots can't inflate a
    score; a contestant is judged on how it placed, not how often it competed.

    Parameters
    ----------
    ballots : Iterable[Ballot]
        The ranked verdicts, each contestant labels best-first.

    Returns
    -------
    dict[str, float]
        Mean rescaled Borda score in ``[0, 1]`` per contestant, higher is better.
        Contestants that appear *only* in ballots carrying no comparative information
        (fewer than two candidates) are omitted, since such ballots are skipped — there
        is nothing to compare, and inventing a score would be dishonest.

    Raises
    ------
    ValueError
        If any single ballot lists a contestant more than once (as in :func:`borda`).

    Notes
    -----
    Normalization fixes *participation* bias, not *opponent-strength* bias: a contestant
    that only ever faced weak fields can still score highly.  Accounting for who beat
    whom (strength of schedule) is the domain of pairwise methods (Bradley-Terry, Elo),
    not Borda.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for ballot in ballots:
        _reject_ties(ballot)
        last_index = len(ballot) - 1
        if last_index < 1:
            # A 0- or 1-candidate ballot carries no comparative information; skip it
            # rather than divide by zero or fabricate a score.
            continue
        for position, label in enumerate(ballot):
            totals[label] = totals.get(label, 0.0) + (last_index - position) / last_index
            counts[label] = counts.get(label, 0) + 1
    return {label: totals[label] / counts[label] for label in totals}
