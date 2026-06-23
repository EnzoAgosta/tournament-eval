"""The bridge between the tournament data model and the aggregation math.

A *ballot* is the unit every aggregation method consumes: a single ranker's verdict as
an ordered list of contestant labels, best-first.  :func:`ballots_from_rankings` is the
only seam where this package reaches into :class:`~tournament_eval.models.RankingResult`;
the math modules depend only on the :class:`Ballot` shape, never on the data model, so
they stay pure and testable with hand-written ballots.

:class:`Ballot` also carries the ranker's ``author`` — optional, but always populated by
:func:`ballots_from_rankings`.  The positional and pairwise math reads only the ranking
(via the sequence dunders) and ignores the author; the bias analysis
(:mod:`tournament_eval.aggregation.bias`) reads the author, since ranker identity *is*
the signal there.
"""

import dataclasses
from collections.abc import Iterable, Iterator

from tournament_eval.models import GenerationResult, RankingResult


@dataclasses.dataclass(frozen=True, slots=True)
class Ballot:
    """One ranker's verdict: contestant labels (authors) in ranked order, best first.

    A strict total order — each label appears at most once, no ties (the invariant the
    whole :mod:`tournament_eval.aggregation` package holds to).  ``author`` is the
    ranker that produced this ballot (``None`` when unknown); the positional and pairwise
    math ignore it, the bias analysis (:mod:`tournament_eval.aggregation.bias`) requires
    it.  :func:`ballots_from_rankings` always populates it from
    :attr:`~tournament_eval.models.RankingResult.author`.

    Acts like a read-only sequence of labels via ``__iter__`` / ``__len__`` /
    ``__getitem__`` so the math modules read a ballot the same way they read a plain
    ``list[str]`` — ``for label in ballot``, ``len(ballot)``, ``enumerate(ballot)``,
    ``set(ballot)``, ``ballot[i]`` all work.  ``frozen=True`` makes the ballot
    identity-aware (two ballots over the same ranking but from different authors
    compare unequal — load-bearing for bias analysis), and prevents reassigning
    ``ranking``; the backing list could still be mutated in place by a determined caller,
    but :func:`reject_ties` guards every entry point the way it always has.  Not
    hashable — ``ranking`` is a list — but nothing here needs ballot hashing.
    """

    ranking: list[str]
    """Contestant labels best-first — a strict total order, no repeats."""
    author: str | None = None
    """The ranker that produced this ballot (``None`` when unknown).  Ignored by the
    positional/pairwise math; required by the bias analysis."""

    def __iter__(self) -> Iterator[str]:
        return iter(self.ranking)

    def __len__(self) -> int:
        return len(self.ranking)

    def __getitem__(self, index: int) -> str:
        return self.ranking[index]


def reject_ties(ballot: Ballot) -> None:
    """Raise if a ballot lists any contestant more than once.

    The package-wide guard for the no-ties invariant: a repeated label is a malformed
    ballot (not a representable tie) that would silently double-count.  Shared by every
    aggregation method so the rule is enforced identically everywhere.
    """
    if len(set(ballot)) != len(ballot):
        raise ValueError(f"Ballot lists a contestant more than once (no ties allowed): {ballot!r}")


def ballots_from_rankings(
    ranking_results: Iterable[RankingResult],
    generation_results: Iterable[GenerationResult],
) -> list[Ballot]:
    """Turn the tournament's :class:`RankingResult`\\ s into de-anonymized ballots.

    Each ranking ranks :class:`~tournament_eval.models.GenerationResult` *ids* (in
    anonymized alias space); this resolves every one back to its contestant ``author``
    with the same id-to-author lookup as
    :func:`~tournament_eval.orchestration.deanonymize_ranking`, built **once** here and
    reused across every ranking, yielding one :class:`Ballot` per ranking — exactly what
    :func:`~tournament_eval.aggregation.positional.borda`
    and the other methods expect.  The ballot's ``author`` is the ranking ranker's
    author, so bias analysis downstream of this adapter has ranker identity in reach
    without a second pass over the data model.

    Parameters
    ----------
    ranking_results : Iterable[RankingResult]
        The rankings produced by :func:`~tournament_eval.orchestration.rank_all` (or
        loaded back from disk).  One ballot per result, in order.
    generation_results : Iterable[GenerationResult]
        The generations being ranked, used to map each ranked id to its ``author``.
        Materialized once into a lookup, so a single-pass iterator is fine.

    Returns
    -------
    list[Ballot]
        One ballot per ranking result, each contestant labels best-first and carrying
        its ranker's ``author``.

    Raises
    ------
    KeyError
        If a ranking references a generation id absent from ``generation_results`` —
        a data-integrity error surfaced loudly (this is
        :func:`~tournament_eval.orchestration.deanonymize_ranking`'s behaviour).
    """
    author_by_id = {generation.id: generation.author for generation in generation_results}
    return [
        Ballot(
            ranking=[author_by_id[generation_id] for generation_id in ranking_result.ranking],
            author=ranking_result.author,
        )
        for ranking_result in ranking_results
    ]
