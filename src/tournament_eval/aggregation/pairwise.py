"""Pairwise aggregation: methods that look at who beat whom, head-to-head.

Every non-positional aggregation method — Copeland, Schulze, minimax, Bradley-Terry,
… — is a function of one shared object: the head-to-head matrix counting, for every
ordered pair ``(i, j)``, how many ballots rank contestant ``i`` above contestant ``j``.
:func:`matrix` builds that tally once so the methods built on it don't each re-derive it;
:func:`copeland` and :func:`schulze` are the first two such methods.

**``numpy`` is required at call time, not import time.**  Importing this module — and
even defining :class:`PairwiseTally` with its numpy-typed field — pulls in no
heavy dependency: ``numpy`` is referenced only under ``TYPE_CHECKING`` (for the type
aliases) and imported lazily inside each function via
:func:`~tournament_eval._lazy_imports.require`.  So a plain ``import
tournament_eval.aggregation.pairwise`` stays dependency-free; calling any function here
without the ``analysis`` extra raises a clear :class:`ImportError` naming the package and
the install command.

**Pairwise ties are allowed in the aggregate.**  The package-wide no-ties invariant is
about individual *ballots*; two contestants can still split the rankers evenly
head-to-head (equal counts above and below).  Such a matchup is a draw — neither a win
nor a loss for either side (standard Copeland; the ``+0.5`` fractional-win convention is
not used here).
"""

import dataclasses
from collections.abc import Iterable
from typing import TYPE_CHECKING

from tournament_eval._lazy_imports import require
from tournament_eval.aggregation.ballots import Ballot, reject_ties

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

type _IntMatrix = npt.NDArray[np.int_]
"""A numpy int matrix — ``numpy`` is only referenced under ``TYPE_CHECKING``, so this
PEP 695 alias is never evaluated at runtime and the annotation costs no import."""


@dataclasses.dataclass(frozen=True, slots=True)
class PairwiseTally:
    """The head-to-head counts over a set of ballots.

    ``above[i, j]`` is the number of ballots that rank ``contestants[i]`` above
    ``contestants[j]`` (counting only ballots where *both* appear).  The diagonal is
    zero.  ``contestants`` gives the label for each matrix index, sorted for
    determinism, so ``contestants[i]`` and row/column ``i`` always line up.
    """

    contestants: list[str]
    """The contestant label at each matrix index, sorted."""
    above: _IntMatrix
    """``above[i, j]`` = number of ballots ranking ``contestants[i]`` over ``contestants[j]``."""


def matrix(
    ballots: Iterable[Ballot],
    *,
    expected_contestants: Iterable[str] | None = None,
    strict: bool = False,
) -> PairwiseTally:
    """Tally, for every ordered pair of contestants, how often one is ranked above the other.

    Each ballot is a strict ranking, so for any two contestants it appears in, exactly one
    is above the other; that increments one off-diagonal cell.

    By default the contestant universe is inferred from the ballots, and a contestant a
    ballot omits is simply **not compared** in that ballot — it takes part in no matchups
    there, so its counts with contestants it never meets stay zero on both sides (absence
    is not a loss).  Pass ``expected_contestants`` to declare the universe explicitly: the
    matrix then spans exactly that set (an omitted contestant still gets a zero
    row/column rather than vanishing), letting you account for who *could* have been
    ranked.

    The count itself is vectorized per ballot via numpy: a ballot whose ranked
    contestants sit at indices ``idx`` (best-first) contributes the strict upper triangle
    of an all-ones ``k x k`` matrix scattered in at ``np.ix_(idx, idx)`` — one numpy
    statement where a hand-rolled ``O(k²)`` Python loop would otherwise do the same work
    slower.  Validation (the no-ties guard, the expected-set checks) stays in Python,
    where it's cheap and readable.

    Parameters
    ----------
    ballots : Iterable[Ballot]
        The ranked verdicts, each contestant labels best-first.
    expected_contestants : Iterable[str] | None
        The full universe of contestants, if known.  When given, the matrix is built over
        exactly this set and every ballot is checked against it (see ``strict`` and
        *Raises*).  When ``None`` (default), the universe is whatever the ballots mention.
    strict : bool
        Only meaningful with ``expected_contestants``.  If ``True``, a ballot that *omits*
        any expected contestant raises :class:`ValueError` instead of leaving that
        contestant not-compared for that ballot.  Default ``False`` (omissions allowed).

    Returns
    -------
    PairwiseTally
        The sorted contestant labels and the ``above`` count matrix.  When
        ``expected_contestants`` is given, ``contestants`` is exactly that set, sorted.

    Raises
    ------
    ValueError
        If any single ballot lists a contestant more than once (the package-wide no-ties
        guard; see :func:`~tournament_eval.aggregation.ballots.reject_ties`); or, when
        ``expected_contestants`` is given, if a ballot contains a contestant outside it
        (an *extra* — a contradiction of the declared universe, always rejected); or, when
        ``strict`` is also set, if a ballot omits any expected contestant.

    Notes
    -----
    Requires ``numpy`` (the ``analysis`` extra) at call time; importing this module does
    not.  A missing extra raises :class:`ImportError` with the install command.
    """
    require("numpy", extra="analysis")
    import numpy as np

    expected = set(expected_contestants) if expected_contestants is not None else None

    materialized: list[Ballot] = []
    seen: set[str] = set()
    for ballot in ballots:
        reject_ties(ballot)
        if expected is not None:
            labels = set(ballot)
            extra = labels - expected
            if extra:
                raise ValueError(
                    f"Ballot contains contestant(s) outside the expected set {sorted(extra)!r}: {ballot!r}"
                )
            if strict:
                missing = expected - labels
                if missing:
                    raise ValueError(f"Ballot omits expected contestant(s) {sorted(missing)!r}: {ballot!r}")
        materialized.append(ballot)
        seen.update(ballot)

    contestants = sorted(expected) if expected is not None else sorted(seen)
    index = {label: i for i, label in enumerate(contestants)}
    above = np.zeros((len(contestants), len(contestants)), dtype=int)

    for ballot in materialized:
        idx = [index[label] for label in ballot]
        k = len(idx)
        if k > 1:
            above[np.ix_(idx, idx)] += np.triu(np.ones((k, k), dtype=int), 1)

    return PairwiseTally(contestants=contestants, above=above)


def copeland(
    ballots: Iterable[Ballot],
    *,
    expected_contestants: Iterable[str] | None = None,
    strict: bool = False,
) -> dict[str, float]:
    """Copeland-score a collection of ballots: head-to-head wins minus losses per contestant.

    A contestant *beats* another when more ballots rank it above the other than below — a
    head-to-head majority.  Its Copeland score is the number of contestants it beats minus
    the number that beat it.  Higher is better; the top scorer is the Condorcet winner
    whenever one exists (someone who beats every other contestant).

    Unlike the positional methods (:mod:`tournament_eval.aggregation.positional`), this
    looks at *who beat whom* rather than average position, so it's robust to the size of
    the field and to lopsided ballots.  It reads straight off the
    :class:`PairwiseTally`, and the scoring is a couple of numpy array operations rather
    than hand-rolled loops.

    Parameters
    ----------
    ballots : Iterable[Ballot]
        The ranked verdicts, each contestant labels best-first (e.g. from
        :func:`~tournament_eval.aggregation.ballots.ballots_from_rankings`).
    expected_contestants : Iterable[str] | None
        The full universe of contestants, forwarded to :func:`matrix`.  When given, every
        such contestant is scored even if some ballots omit it (an omitted contestant is
        not-compared, so with no wins or losses it scores ``0``).
    strict : bool
        Forwarded to :func:`matrix`: with ``expected_contestants`` set, raise on a ballot
        that omits an expected contestant.

    Returns
    -------
    dict[str, float]
        Net pairwise score per contestant: ``(opponents beaten) - (opponents lost to)``.
        Ranges over ``[-(n-1), n-1]`` for ``n`` contestants; higher is better.  A
        contestant that beats every other scores ``n - 1`` and is the Condorcet winner.

    Raises
    ------
    ValueError
        If any single ballot lists a contestant more than once, or for the
        ``expected_contestants`` / ``strict`` cases documented on :func:`matrix`.

    Notes
    -----
    Requires ``numpy`` (the ``analysis`` extra) at call time, via :func:`matrix`;
    importing this module does not.  A missing extra raises :class:`ImportError` with
    the install command.
    """
    tally = matrix(ballots, expected_contestants=expected_contestants, strict=strict)
    above = tally.above

    beats = above > above.T
    beaten = above < above.T
    net = beats.sum(axis=1) - beaten.sum(axis=1)
    return {label: float(net[i]) for i, label in enumerate(tally.contestants)}


def schulze(
    ballots: Iterable[Ballot],
    *,
    expected_contestants: Iterable[str] | None = None,
    strict: bool = False,
) -> dict[str, float]:
    """Schulze-score a collection of ballots via strongest-beatpath widths.

    The Schulze method resolves a set of strict rankings by computing, for every
    ordered pair of contestants ``(i, j)``, the *strongest beatpath* width from
    ``i`` to ``j`` — the maximum, over every chain
    ``i = x_0 \u2192 x_1 \u2192 \u2026 \u2192 x_m = j`` where each link beats the next
    head-to-head, of the weakest link in that chain (a bottleneck/flooding-style
    max-min path).
    Contestant ``i`` beats ``j`` in the Schulze sense when ``i``\u2019s beatpath to
    ``j`` is wider than ``j``\u2019s back to ``i``; a contestant\u2019s score is how
    many others it beats that way.  Higher is better; the top scorer is the
    Schulze winner, which coincides with the Condorcet winner whenever one exists
    (a Condorcet winner\u2019s direct win is the widest possible beatpath, so no
    one beats it back) and resolves Condorcet cycles deterministically otherwise.

    Three properties make it a strong default over Copeland for ranked ballots:
    it is **Condorcet-consistent** (a head-to-head majority winner always tops
    the result), **monotone** (ranking a contestant higher on a ballot can never
    *lower* its Schulze standing), and **cloneproof** (adding a near-identical
    \u201cclone\u201d of a contestant doesn\u2019t change the non-clones\u2019
    relative order or hand the clone-group an advantage).  Copeland shares the
    first but not the other two, and routinely *ties* on cycles (every cycle
    member scores 0); Schulze breaks those ties via the beatpath widths, so it
    produces a strict order where Copeland declares a dead heat.

    The score itself \u2014 *number of opponents beatpath-beaten* \u2014 makes the
    method drop-in with :func:`copeland` and the positional methods for
    :func:`~tournament_eval.presentation.console.leaderboard`: one float per
    contestant, higher is better.  Read it the same way as Copeland (range
    ``[0, n-1]`` for ``n`` contestants; the winner scores ``n - 1`` when the
    beatpath order is strict).  When two contestants beatpath-tie (their
    beatpath widths are equal both ways) they score the same \u2014 a *Schulze
    tie*, rarer than a Copeland tie but possible on symmetric profiles (e.g.
    an even head-to-head split between two contestants, or a perfectly symmetric
    multi-way cycle).  It is not a ballot tie (ballots stay strict); it is an
    aggregate tie the method doesn\u2019t manufacture on ordinary data.

    Parameters
    ----------
    ballots : Iterable[Ballot]
        The ranked verdicts, each contestant labels best-first (e.g. from
        :func:`~tournament_eval.aggregation.ballots.ballots_from_rankings`).
    expected_contestants : Iterable[str] | None
        The full universe of contestants, forwarded to :func:`matrix`.  When
        given, every such contestant is scored even if some ballots omit it
        (an omitted contestant is not-compared, so it has no beatpaths and
        beats no one \u2014 it scores ``0``).
    strict : bool
        Forwarded to :func:`matrix`: with ``expected_contestants`` set, raise on
        a ballot that omits an expected contestant.

    Returns
    -------
    dict[str, float]
        Number of opponents each contestant beatpath-beats, range ``[0, n-1]``
        for ``n`` contestants.  Higher is better; ``n - 1`` is the Schulze winner
        when the beatpath order is strict.  Sort it yourself for a leaderboard,
        the same way as :func:`copeland`.

    Raises
    ------
    ValueError
        If any single ballot lists a contestant more than once, or for the
        ``expected_contestants`` / ``strict`` cases documented on :func:`matrix`.

    Notes
    -----
    Requires ``numpy`` (the ``analysis`` extra) at call time, via :func:`matrix`;
    importing this module does not.  A missing extra raises :class:`ImportError`
    with the install command.
    """
    tally = matrix(ballots, expected_contestants=expected_contestants, strict=strict)
    require("numpy", extra="analysis")
    import numpy as np

    above = tally.above
    n = len(tally.contestants)

    # Strongest-beatpath widths via the Floyd-Warshall-style update:
    #   strengths[i, j] = max(strengths[i, j],
    #                         min(strengths[i, k], strengths[k, j]))
    # over every intermediate k.  The pairwise head-to-head counts `above`
    # seed the widths (a direct win d[i,j] is a one-link beatpath of that
    # width); the diagonal is fixed at 0 (a contestant doesn't beat itself).
    strengths = above.astype(int).copy()
    np.fill_diagonal(strengths, 0)
    for k in range(n):
        via_k = np.minimum(strengths[:, k, None], strengths[k, None, :])
        strengths = np.maximum(strengths, via_k)
        np.fill_diagonal(strengths, 0)

    beats = strengths > strengths.T
    scores = beats.sum(axis=1)
    return {label: float(scores[i]) for i, label in enumerate(tally.contestants)}
