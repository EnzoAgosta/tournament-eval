"""The bridge between the tournament data model and the aggregation math.

A *ballot* is the unit every aggregation method consumes: a single ranker's verdict as
an ordered list of contestant labels, best-first.  :func:`ballots_from_rankings` is the
only seam where this package reaches into
:class:`~tournament_eval.models.RankingResult`; the math modules (e.g.
:mod:`tournament_eval.aggregation.borda`) depend only on the :data:`Ballot` shape, never
on the data model, so they stay pure and trivially testable with hand-written ballots.
"""

from collections.abc import Iterable

from tournament_eval.models import GenerationResult, RankingResult
from tournament_eval.orchestration import deanonymize_ranking

type Ballot = list[str]
"""One ranker's verdict: contestant labels (authors) in ranked order, best first.

A strict total order — each label appears at most once, no ties (the invariant the
whole :mod:`tournament_eval.aggregation` package holds to)."""


def ballots_from_rankings(
    ranking_results: Iterable[RankingResult],
    generation_results: Iterable[GenerationResult],
) -> list[Ballot]:
    """Turn the tournament's :class:`RankingResult`\\ s into de-anonymized ballots.

    Each :class:`~tournament_eval.models.RankingResult` ranks
    :class:`~tournament_eval.models.GenerationResult` *ids* (in anonymized alias space);
    this resolves every one back to its contestant ``author`` via
    :func:`~tournament_eval.orchestration.deanonymize_ranking`, yielding one
    :data:`Ballot` per ranking.  The result is exactly what :func:`~tournament_eval.aggregation.borda.borda`
    and the other methods expect.

    This is the wire from the tournament side to the analysis side: run the pipeline,
    collect ``ranking_results`` and ``generation_results``, pass both here, hand the
    ballots to an aggregation method.

    Parameters
    ----------
    ranking_results : Iterable[RankingResult]
        The rankings produced by :func:`~tournament_eval.orchestration.rank_all` (or
        loaded back from disk).  One ballot is produced per result, in order.
    generation_results : Iterable[GenerationResult]
        The generations being ranked, used to map each ranked id to its ``author``.
        Materialized once into a lookup, so a single-pass iterator is fine.

    Returns
    -------
    list[Ballot]
        One ballot per ranking result, each contestant labels best-first.

    Raises
    ------
    KeyError
        If a ranking references a generation id absent from ``generation_results`` —
        a data-integrity error surfaced loudly rather than silently dropped (this is
        :func:`~tournament_eval.orchestration.deanonymize_ranking`'s behaviour).
    """
    generations = list(generation_results)
    return [deanonymize_ranking(ranking_result, generations) for ranking_result in ranking_results]
