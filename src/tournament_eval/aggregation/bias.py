"""Bias analysis: measure ranker preference patterns the aggregate scores hide.

The methodology's headline claim is that *bias becomes a signal*: a peer ranker that
overrates itself, or systematically over- or under-places a competitor, is a measurable
fact about its calibration — not noise to correct away.  The aggregate scores
(:mod:`tournament_eval.aggregation.positional`, :mod:`tournament_eval.aggregation.pairwise`)
collapse the panel into a leaderboard and throw ranker identity away; this module keeps
it, turning the per-ranker preference structure into numbers you can read.

**No heavy dependencies.**  Pure Python, like the positional methods — a plain tournament
run and its bias analysis never need the ``analysis`` extra.

Two functions, both pure and operating on :class:`~tournament_eval.aggregation.ballots.Ballot`\\ s:

* :func:`preference_matrix` — a ranker x contestant grid of mean placement: the raw
  "who over- or under-rates whom" surface.  Every off-diagonal cell is a ranker's lean
  toward a competitor; the diagonal is a ranker's placement of itself.
* :func:`self_preference` — the headline self-bias number per ranker that is also a
  contestant: how much higher (or lower) it places itself than the rest of the panel does.

Both require ballots with ``author`` set — :func:`~tournament_eval.aggregation.ballots.ballots_from_rankings`
always populates it; hand-built ballots in tests need to too.  An authorless ballot
carries no ranker signal, so it's skipped by default and raises on ``strict=True``,
mirroring the package-wide ``strict`` convention ("raise on anything that would
otherwise be handled silently").
"""

import dataclasses
from collections.abc import Iterable

from tournament_eval.aggregation.ballots import Ballot, reject_ties


@dataclasses.dataclass(frozen=True, slots=True)
class PreferenceMatrix:
    """A ranker x contestant grid of mean placement.

    ``placement[i][j]`` is the mean 1-indexed position that ``rankers[i]`` placed
    ``contestants[j]`` at, averaged over the ballots authored by ``rankers[i]`` that
    include ``contestants[j]``.  ``None`` when ``rankers[i]`` never ranked
    ``contestants[j]`` — absence is not a placement, so it isn't averaged in as a
    worst-rank.  Lower is better (``1`` = top of a ballot).

    ``rankers`` and ``contestants`` are sorted for determinism, so index ``i`` and row
    ``i`` always line up, and likewise for ``j`` / column ``j``.  Mirrors
    :class:`~tournament_eval.aggregation.pairwise.PairwiseTally`'s shape so the two
    compose naturally with a future presentation layer.
    """

    rankers: list[str]
    """The ranker label at each matrix row, sorted."""
    contestants: list[str]
    """The contestant label at each matrix column, sorted."""
    placement: list[list[float | None]]
    """``placement[i][j]`` = mean position ``rankers[i]`` gave ``contestants[j]`` (1 =
    best, lower is better), or ``None`` when ``rankers[i]`` never ranked
    ``contestants[j]``."""


def _check_authored(ballots: list[Ballot], *, strict: bool) -> None:
    """Reject ballots with no author under ``strict`` — they can't be ranker-keyed.

    Bias analysis is keyed on ranker identity, so a ballot with ``author is None``
    carries no signal here.  The package-wide ``strict`` convention: skip silently by
    default, raise on the first one when ``strict=True``.
    """
    if not strict:
        return
    missing = [ballot for ballot in ballots if ballot.author is None]
    if missing:
        raise ValueError(
            f"{len(missing)} ballot(s) have no author; bias analysis needs ranker identity. "
            "Build ballots via ballots_from_rankings, or pass strict=False to skip them."
        )


def preference_matrix(
    ballots: Iterable[Ballot],
    *,
    strict: bool = False,
) -> PreferenceMatrix:
    """Build a ranker x contestant grid of mean placement from authored ballots.

    Each ballot contributes its 1-indexed positions (best-first) to its author's row.
    A cell ``placement[i][j]`` is the mean position ``rankers[i]`` placed
    ``contestants[j]`` at, over the ballots authored by ``rankers[i]`` that include
    ``contestants[j]``; ``None`` when ``rankers[i]`` never ranked ``contestants[j]``.

    The off-diagonal cells are the "who over- or under-rates whom" surface — a ranker
    that systematically places a competitor higher than the rest of the panel does
    shows a low mean position in that competitor's column.  The diagonal is a ranker's
    placement of itself; :func:`self_preference` turns that into the headline
    self-bias number.

    Ballots with ``author is None`` are skipped by default (they carry no ranker
    signal); pass ``strict=True`` to raise on the first one instead.

    Parameters
    ----------
    ballots : Iterable[Ballot]
        The ranked verdicts, each contestant labels best-first (e.g. from
        :func:`~tournament_eval.aggregation.ballots.ballots_from_rankings`).  Materialized
        once, so a single-pass iterator is fine.
    strict : bool
        If ``True``, raise :class:`ValueError` on the first ballot with no ``author``
        instead of skipping it.  Default ``False`` (skip silently).

    Returns
    -------
    PreferenceMatrix
        The sorted ranker and contestant labels and the ``placement`` grid.  Empty
        rankers/contestants (no ballots, or only authorless ballots) yield a ``0 x 0``
        grid.

    Raises
    ------
    ValueError
        If any single ballot lists a contestant more than once (the package-wide
        no-ties guard), or — when ``strict`` is set — if any ballot has no ``author``.
    """
    materialized = list(ballots)
    for ballot in materialized:
        reject_ties(ballot)
    _check_authored(materialized, strict=strict)

    rankers = sorted({ballot.author for ballot in materialized if ballot.author is not None})
    contestants = sorted({label for ballot in materialized for label in ballot})
    ranker_index = {ranker: i for i, ranker in enumerate(rankers)}
    contestant_index = {contestant: j for j, contestant in enumerate(contestants)}

    # positions[i][j] = every position rankers[i] placed contestants[j] at.
    positions: list[list[list[int]]] = [[[] for _ in contestants] for _ in rankers]
    for ballot in materialized:
        if ballot.author is None:
            continue
        row = positions[ranker_index[ballot.author]]
        for position, label in enumerate(ballot, start=1):
            row[contestant_index[label]].append(position)

    placement: list[list[float | None]] = []
    for row in positions:
        placement_row: list[float | None] = [sum(p) / len(p) if p else None for p in row]
        placement.append(placement_row)

    return PreferenceMatrix(rankers=rankers, contestants=contestants, placement=placement)


def self_preference(
    ballots: Iterable[Ballot],
    *,
    strict: bool = False,
) -> dict[str, float]:
    """Measure how much each ranker over- or under-rates itself versus the panel.

    For every author that is *both* a ranker (authored ballots) and a contestant
    (appears in some ballot) **and** ranked itself in at least one of its own ballots,
    the self-preference score is::

        panel_mean_placement(self) - self_mean_placement(self)

    where ``self_mean`` is the mean 1-indexed position the ranker placed itself at (over
    its own ballots that include itself), and ``panel_mean`` is the mean position the
    *rest of the panel* — every ballot whose author is not this ranker — placed it at
    (over those ballots that include it).  The ranker's own ballots are excluded from
    the panel baseline so a self-ranker can't inflate its own reference point.

    **Positive = self-preference**: the ranker placed itself higher (a lower position
    number) than its peers did.  Negative = self-deprecation.  Magnitudes are on the
    position scale (``1`` = one rank apart), so a score of ``0.7`` means the ranker
    places itself about two-thirds of a rank above where the panel places it on average.

    A ranker that never ranked itself (e.g. its own generation failed for the tasks it
    judged, so it wasn't a candidate in its own ballots) is excluded — there's no
    self-placement to compare.  A ranker the panel never ranked (it appears only in its
    own ballots) is excluded too — the panel baseline is undefined.  Ballots with
    ``author is None`` are skipped by default; ``strict=True`` raises on the first one.

    Parameters
    ----------
    ballots : Iterable[Ballot]
        The ranked verdicts, each contestant labels best-first (e.g. from
        :func:`~tournament_eval.aggregation.ballots.ballots_from_rankings`).  Materialized
        once, so a single-pass iterator is fine.
    strict : bool
        If ``True``, raise :class:`ValueError` on the first ballot with no ``author``
        instead of skipping it.  Default ``False`` (skip silently).

    Returns
    -------
    dict[str, float]
        Self-preference score per ranker-that-is-also-a-contestant, keyed by author
        label, sorted by label.  Positive = ranks self higher than the panel does;
        negative = ranks self lower.  Absent entries are rankers with no
        self-placement or no panel baseline (see above).

    Raises
    ------
    ValueError
        If any single ballot lists a contestant more than once (the package-wide
        no-ties guard), or — when ``strict`` is set — if any ballot has no ``author``.
    """
    materialized = list(ballots)
    for ballot in materialized:
        reject_ties(ballot)
    _check_authored(materialized, strict=strict)

    # Pair each ballot with its (now non-None) author so the rest of the function
    # works with a narrow `str` rather than re-narrowing `ballot.author` everywhere.
    authored: list[tuple[str, Ballot]] = [
        (ballot.author, ballot) for ballot in materialized if ballot.author is not None
    ]
    rankers = {author for author, _ in authored}
    contestants = {label for _, ballot in authored for label in ballot}
    both = rankers & contestants

    self_positions: dict[str, list[int]] = {author: [] for author in both}
    panel_positions: dict[str, list[int]] = {author: [] for author in both}
    for author, ballot in authored:
        for position, label in enumerate(ballot, start=1):
            if label not in both:
                continue
            if author == label:
                self_positions[label].append(position)
            else:
                panel_positions[label].append(position)

    scores: dict[str, float] = {}
    for author in sorted(both):
        self_placements = self_positions[author]
        if not self_placements:
            continue  # ranker never ranked itself; no self-placement to compare.
        panel_placements = panel_positions[author]
        if not panel_placements:
            continue  # nobody else ranked this contestant; panel baseline undefined.
        self_mean = sum(self_placements) / len(self_placements)
        panel_mean = sum(panel_placements) / len(panel_placements)
        scores[author] = panel_mean - self_mean
    return scores
