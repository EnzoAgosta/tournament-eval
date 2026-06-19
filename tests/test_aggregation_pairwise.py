"""Known-answer tests for the numpy-backed pairwise tally + Copeland.

The expected values here are computed by hand from the ballots, not derived from the
implementation, so they pin the *meaning* of the methods rather than their current code.
"""

import pytest

pytest.importorskip("numpy")  # the pairwise methods live behind the 'analysis' extra

from tournament_eval.aggregation.copeland import copeland
from tournament_eval.aggregation.pairwise import pairwise_matrix


def test_pairwise_matrix_counts_head_to_head_above_relationships() -> None:
    # "a" is ranked above "b" on two ballots, "b" above "a" on one.
    tally = pairwise_matrix([["a", "b"], ["b", "a"], ["a", "b"]])

    assert tally.contestants == ["a", "b"]
    assert tally.above.tolist() == [
        [0, 2],  # a over b twice
        [1, 0],  # b over a once
    ]


def test_pairwise_matrix_of_one_full_ordering_fills_the_upper_triangle() -> None:
    # a > b > c: a beats b and c, b beats c, once each; nothing in the lower triangle.
    tally = pairwise_matrix([["a", "b", "c"]])

    assert tally.contestants == ["a", "b", "c"]
    assert tally.above.tolist() == [
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ]


def test_pairwise_matrix_sorts_contestants_regardless_of_ballot_order() -> None:
    tally = pairwise_matrix([["c", "a", "b"]])

    assert tally.contestants == ["a", "b", "c"]


def test_pairwise_matrix_of_no_ballots_is_empty() -> None:
    tally = pairwise_matrix([])

    assert tally.contestants == []
    assert tally.above.shape == (0, 0)


def test_pairwise_matrix_raises_on_a_contestant_appearing_twice_in_one_ballot() -> None:
    with pytest.raises(ValueError, match="more than once"):
        pairwise_matrix([["a", "b", "a"]])


def test_pairwise_matrix_spans_the_full_expected_set_treating_omissions_as_not_compared() -> None:
    # "c" is never ranked, but the declared universe includes it: it gets a zero row and
    # column (not-compared) rather than vanishing from the matrix.
    tally = pairwise_matrix([["a", "b"]], expected_contestants={"a", "b", "c"})

    assert tally.contestants == ["a", "b", "c"]
    assert tally.above.tolist() == [
        [0, 1, 0],  # a over b once; never meets c
        [0, 0, 0],
        [0, 0, 0],  # c compared to nobody
    ]


def test_pairwise_matrix_rejects_a_ballot_with_a_contestant_outside_the_expected_set() -> None:
    with pytest.raises(ValueError, match="outside the expected set"):
        pairwise_matrix([["a", "b", "z"]], expected_contestants={"a", "b"})


def test_pairwise_matrix_fail_fast_rejects_a_ballot_that_omits_an_expected_contestant() -> None:
    with pytest.raises(ValueError, match="omits expected contestant"):
        pairwise_matrix([["a", "b"]], expected_contestants={"a", "b", "c"}, fail_fast=True)


def test_pairwise_matrix_fail_fast_accepts_a_complete_ballot() -> None:
    tally = pairwise_matrix([["a", "b", "c"]], expected_contestants={"a", "b", "c"}, fail_fast=True)

    assert tally.contestants == ["a", "b", "c"]


def test_copeland_scores_an_omitted_expected_contestant_as_zero() -> None:
    # a beats b on both ballots; c is in the universe but never ranked, so it's
    # not-compared everywhere and nets zero — yet still appears in the leaderboard.
    scores = copeland([["a", "b"], ["a", "b"]], expected_contestants={"a", "b", "c"})

    assert scores == {"a": 1.0, "b": -1.0, "c": 0.0}


def test_copeland_fail_fast_rejects_a_ballot_that_omits_an_expected_contestant() -> None:
    with pytest.raises(ValueError, match="omits expected contestant"):
        copeland([["a", "b"]], expected_contestants={"a", "b", "c"}, fail_fast=True)


def test_copeland_gives_a_condorcet_winner_the_top_score() -> None:
    # "a" wins its head-to-head against both b and c, so it beats 2 and loses 0 -> +2.
    # b beats c but loses to a -> 0. c loses to both -> -2.
    scores = copeland([["a", "b", "c"], ["a", "c", "b"], ["b", "a", "c"]])

    assert scores == {"a": 2.0, "b": 0.0, "c": -2.0}


def test_copeland_ties_everyone_at_zero_on_a_condorcet_cycle() -> None:
    # The Condorcet paradox: a>b>c, b>c>a, c>a>b. Each contestant beats exactly one other
    # and loses to exactly one, so every net score is zero — the cycle is visible as a tie.
    scores = copeland([["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]])

    assert scores == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_copeland_treats_an_even_head_to_head_split_as_a_draw() -> None:
    # Two rankers each way: neither takes the majority, so it's neither a win nor a loss.
    scores = copeland([["a", "b"], ["b", "a"]])

    assert scores == {"a": 0.0, "b": 0.0}


def test_copeland_scores_disjoint_groups_independently() -> None:
    # a and b never meet c and d. Within each pair the winner is +1, the loser -1; the
    # cross-pair matchups never happen and count as neither win nor loss.
    scores = copeland([["a", "b"], ["c", "d"]])

    assert scores == {"a": 1.0, "b": -1.0, "c": 1.0, "d": -1.0}


def test_copeland_of_no_ballots_is_empty() -> None:
    assert copeland([]) == {}


def test_copeland_raises_on_a_contestant_appearing_twice_in_one_ballot() -> None:
    with pytest.raises(ValueError, match="more than once"):
        copeland([["a", "b", "a"]])
