"""Copeland's method: score each contestant by its head-to-head wins minus losses.

A contestant *beats* another when more ballots rank it above the other than below — a
head-to-head majority.  Its Copeland score is the number of contestants it beats minus
the number that beat it.  Higher is better; the top scorer is the Condorcet winner
whenever one exists (someone who beats every other contestant).

Unlike Borda, this looks at *who beat whom* rather than average position, so it's robust
to the size of the field and to lopsided ballots.  It reads straight off the
:class:`~tournament_eval.aggregation.pairwise.PairwiseTally`, and the scoring is a couple
of numpy array operations rather than hand-rolled loops.

**Pairwise ties are allowed in the aggregate.**  The no-ties invariant is about
individual *ballots*; two contestants can still split the rankers evenly head-to-head
(equal counts above and below).  Such a matchup is a draw — neither a win nor a loss for
either side (standard Copeland; the ``+0.5`` fractional-win convention is not used here).
"""

from collections.abc import Iterable

from tournament_eval.aggregation.ballots import Ballot
from tournament_eval.aggregation.pairwise import pairwise_matrix


def copeland(
    ballots: Iterable[Ballot],
    *,
    expected_contestants: Iterable[str] | None = None,
    fail_fast: bool = False,
) -> dict[str, float]:
    """Copeland-score a collection of ballots: head-to-head wins minus losses per contestant.

    Parameters
    ----------
    ballots : Iterable[Ballot]
        The ranked verdicts, each contestant labels best-first (e.g. from
        :func:`~tournament_eval.aggregation.ballots.ballots_from_rankings`).
    expected_contestants : Iterable[str] | None
        The full universe of contestants, forwarded to
        :func:`~tournament_eval.aggregation.pairwise.pairwise_matrix`.  When given, every
        such contestant is scored even if some ballots omit it (an omitted contestant is
        not-compared, so with no wins or losses it scores ``0``).
    fail_fast : bool
        Forwarded to :func:`~tournament_eval.aggregation.pairwise.pairwise_matrix`: with
        ``expected_contestants`` set, raise on a ballot that omits an expected contestant.

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
        ``expected_contestants`` / ``fail_fast`` cases documented on
        :func:`~tournament_eval.aggregation.pairwise.pairwise_matrix`.
    """
    tally = pairwise_matrix(ballots, expected_contestants=expected_contestants, fail_fast=fail_fast)
    above = tally.above
    # beats[i, j] is True when i takes the head-to-head majority over j (and so loses to
    # nobody on the diagonal, where above == above.T). net wins = (beats) - (beaten).
    beats = above > above.T
    beaten = above < above.T
    net = beats.sum(axis=1).astype(int) - beaten.sum(axis=1).astype(int)
    return {label: float(net[i]) for i, label in enumerate(tally.contestants)}
