"""The pairwise tally: how often each contestant is ranked above each other one.

Most non-positional aggregation methods — Copeland, minimax, Schulze, … — are
functions of one shared object: the head-to-head matrix counting, for every ordered
pair ``(i, j)``, how many ballots rank contestant ``i`` above contestant ``j``.  This
module builds that matrix once so those methods don't each re-derive it.

The first piece of the package to need ``numpy`` (the ``analysis`` extra): the
counting loop is plain Python, but the matrix is a :class:`numpy.ndarray` so
downstream methods express their logic as reviewable array operations rather than
hand-rolled loops — leaning on numpy where getting it right by hand is easy to get
subtly wrong.
"""

import dataclasses
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from tournament_eval.aggregation.ballots import Ballot, reject_ties


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
    above: npt.NDArray[np.int_]
    """``above[i, j]`` = number of ballots ranking ``contestants[i]`` over ``contestants[j]``."""


def pairwise_matrix(
    ballots: Iterable[Ballot],
    *,
    expected_contestants: Iterable[str] | None = None,
    fail_fast: bool = False,
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

    Parameters
    ----------
    ballots : Iterable[Ballot]
        The ranked verdicts, each contestant labels best-first.
    expected_contestants : Iterable[str] | None
        The full universe of contestants, if known.  When given, the matrix is built over
        exactly this set and every ballot is checked against it (see ``fail_fast`` and
        *Raises*).  When ``None`` (default), the universe is whatever the ballots mention.
    fail_fast : bool
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
        ``fail_fast`` is also set, if a ballot omits any expected contestant.
    """
    expected = set(expected_contestants) if expected_contestants is not None else None

    # One pass to validate, retain (the input may be a single-use iterator), and gather
    # the contestant universe; ballots are read-only, so no per-ballot copy is needed.
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
            if fail_fast:
                missing = expected - labels
                if missing:
                    raise ValueError(f"Ballot omits expected contestant(s) {sorted(missing)!r}: {ballot!r}")
        materialized.append(ballot)
        seen.update(ballot)

    contestants = sorted(expected) if expected is not None else sorted(seen)
    index = {label: i for i, label in enumerate(contestants)}
    above = np.zeros((len(contestants), len(contestants)), dtype=int)

    for ballot in materialized:
        positions = [index[label] for label in ballot]
        # Every earlier position outranks every later one: increment (higher, lower).
        for rank, higher in enumerate(positions):
            for lower in positions[rank + 1 :]:
                above[higher, lower] += 1

    return PairwiseTally(contestants=contestants, above=above)
