"""Known-answer tests for the numpy-backed pairwise tally + Copeland.

The expected values here are computed by hand from the ballots, not derived from the
implementation, so they pin the *meaning* of the methods rather than their current code.

``numpy`` is a dev dependency (and an end-user ``analysis`` extra), so these run as part
of the normal suite — they exercise the lazy-import path: importing
``tournament_eval.aggregation.pairwise`` pulls in no numpy, and calling any function here
imports it on demand.
"""

import pytest

from tournament_eval._lazy_imports import require
from tournament_eval.aggregation import pairwise
from tournament_eval.aggregation.ballots import Ballot


def test_require_raises_a_clear_importerror_for_a_missing_module() -> None:

    with pytest.raises(ImportError, match="is required here but isn't installed") as exc_info:
        require("definitely_not_a_real_module_xyzzy", extra="analysis")
    assert "analysis" in str(exc_info.value)


def test_pairwise_matrix_raises_a_clear_importerror_when_numpy_is_missing() -> None:

    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, importlib.abc\n"
                "class _Block(importlib.abc.MetaPathFinder):\n"
                "    def find_spec(self, name, path, target=None):\n"
                "        if name == 'numpy' or name.startswith('numpy.'):\n"
                "            raise ImportError('numpy blocked for test')\n"
                "        return None\n"
                "sys.meta_path.insert(0, _Block())\n"
                "from tournament_eval.aggregation import pairwise\n"
                "from tournament_eval.aggregation.ballots import Ballot\n"
                "pairwise.matrix([Ballot(['a', 'b'])])\n"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0

    assert "'numpy' is required here" in result.stderr
    assert "analysis" in result.stderr


def test_pairwise_matrix_counts_head_to_head_above_relationships() -> None:
    # "a" is ranked above "b" on two ballots, "b" above "a" on one.
    tally = pairwise.matrix([Ballot(["a", "b"]), Ballot(["b", "a"]), Ballot(["a", "b"])])

    assert tally.contestants == ["a", "b"]
    assert tally.above.tolist() == [
        [0, 2],  # a over b twice
        [1, 0],  # b over a once
    ]


def test_pairwise_matrix_of_one_full_ordering_fills_the_upper_triangle() -> None:
    # a > b > c: a beats b and c, b beats c, once each; nothing in the lower triangle.
    tally = pairwise.matrix([Ballot(["a", "b", "c"])])

    assert tally.contestants == ["a", "b", "c"]
    assert tally.above.tolist() == [
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ]


def test_pairwise_matrix_sorts_contestants_regardless_of_ballot_order() -> None:
    tally = pairwise.matrix([Ballot(["c", "a", "b"])])

    assert tally.contestants == ["a", "b", "c"]


def test_pairwise_matrix_of_no_ballots_is_empty() -> None:
    tally = pairwise.matrix([])

    assert tally.contestants == []
    assert tally.above.shape == (0, 0)


def test_pairwise_matrix_raises_on_a_contestant_appearing_twice_in_one_ballot() -> None:
    with pytest.raises(ValueError, match="more than once"):
        pairwise.matrix([Ballot(["a", "b", "a"])])


def test_pairwise_matrix_spans_the_full_expected_set_treating_omissions_as_not_compared() -> None:
    # "c" is never ranked, but the declared universe includes it: it gets a zero row and
    # column (not-compared) rather than vanishing from the matrix.
    tally = pairwise.matrix([Ballot(["a", "b"])], expected_contestants={"a", "b", "c"})

    assert tally.contestants == ["a", "b", "c"]
    assert tally.above.tolist() == [
        [0, 1, 0],  # a over b once; never meets c
        [0, 0, 0],
        [0, 0, 0],  # c compared to nobody
    ]


def test_pairwise_matrix_rejects_a_ballot_with_a_contestant_outside_the_expected_set() -> None:
    with pytest.raises(ValueError, match="outside the expected set"):
        pairwise.matrix([Ballot(["a", "b", "z"])], expected_contestants={"a", "b"})


def test_pairwise_matrix_strict_rejects_a_ballot_that_omits_an_expected_contestant() -> None:
    with pytest.raises(ValueError, match="omits expected contestant"):
        pairwise.matrix([Ballot(["a", "b"])], expected_contestants={"a", "b", "c"}, strict=True)


def test_pairwise_matrix_strict_accepts_a_complete_ballot() -> None:
    tally = pairwise.matrix([Ballot(["a", "b", "c"])], expected_contestants={"a", "b", "c"}, strict=True)

    assert tally.contestants == ["a", "b", "c"]


def test_copeland_scores_an_omitted_expected_contestant_as_zero() -> None:
    # a beats b on both ballots; c is in the universe but never ranked, so it's
    # not-compared everywhere and nets zero — yet still appears in the leaderboard.
    scores = pairwise.copeland([Ballot(["a", "b"]), Ballot(["a", "b"])], expected_contestants={"a", "b", "c"})

    assert scores == {"a": 1.0, "b": -1.0, "c": 0.0}


def test_copeland_strict_rejects_a_ballot_that_omits_an_expected_contestant() -> None:
    with pytest.raises(ValueError, match="omits expected contestant"):
        pairwise.copeland([Ballot(["a", "b"])], expected_contestants={"a", "b", "c"}, strict=True)


def test_copeland_gives_a_condorcet_winner_the_top_score() -> None:
    # "a" wins its head-to-head against both b and c, so it beats 2 and loses 0 -> +2.
    # b beats c but loses to a -> 0. c loses to both -> -2.
    scores = pairwise.copeland([Ballot(["a", "b", "c"]), Ballot(["a", "c", "b"]), Ballot(["b", "a", "c"])])

    assert scores == {"a": 2.0, "b": 0.0, "c": -2.0}


def test_copeland_ties_everyone_at_zero_on_a_condorcet_cycle() -> None:
    # The Condorcet paradox: a>b>c, b>c>a, c>a>b. Each contestant beats exactly one other
    # and loses to exactly one, so every net score is zero — the cycle is visible as a tie.
    scores = pairwise.copeland([Ballot(["a", "b", "c"]), Ballot(["b", "c", "a"]), Ballot(["c", "a", "b"])])

    assert scores == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_copeland_treats_an_even_head_to_head_split_as_a_draw() -> None:
    # Two rankers each way: neither takes the majority, so it's neither a win nor a loss.
    scores = pairwise.copeland([Ballot(["a", "b"]), Ballot(["b", "a"])])

    assert scores == {"a": 0.0, "b": 0.0}


def test_copeland_scores_disjoint_groups_independently() -> None:
    # a and b never meet c and d. Within each pair the winner is +1, the loser -1; the
    # cross-pair matchups never happen and count as neither win nor loss.
    scores = pairwise.copeland([Ballot(["a", "b"]), Ballot(["c", "d"])])

    assert scores == {"a": 1.0, "b": -1.0, "c": 1.0, "d": -1.0}


def test_copeland_of_no_ballots_is_empty() -> None:
    assert pairwise.copeland([]) == {}


def test_copeland_raises_on_a_contestant_appearing_twice_in_one_ballot() -> None:
    with pytest.raises(ValueError, match="more than once"):
        pairwise.copeland([Ballot(["a", "b", "a"])])


# --- schulze -----------------------------------------------------------------
# Known-answer tests for the Schulze method, mirroring the Copeland tests above.
# Every expected value is computed by hand from the ballots (verified against a
# scratch Floyd-Warschulze run), so the tests pin the *meaning* of the method —
# Condorcet consistency, cycle resolution via beatpath widths, and tie behaviour —
# rather than its current code.


def test_schulze_gives_a_condorcet_winner_the_top_score() -> None:
    # a beats b (2:1) and c (3:0); b beats c (2:1). a is the Condorcet winner, so
    # it beatpath-beats both -> score 2; b beats c -> 1; c beats nobody -> 0.
    scores = pairwise.schulze([Ballot(["a", "b", "c"]), Ballot(["a", "c", "b"]), Ballot(["b", "a", "c"])])

    assert scores == {"a": 2.0, "b": 1.0, "c": 0.0}


def test_schulze_resolves_a_condorcet_cycle_where_copeland_ties() -> None:
    # The Condorcet paradox: a>b>c, b>c>a, c>a>b. Every head-to-head is 2:1, so
    # Copeland ties everyone at 0. Schulze breaks the cycle via beatpath widths:
    # each contestant's strongest beatpath to every other is 2 (e.g. a->b direct
    # is 2; a->c runs a->b->c, min(2,2)=2), so all three pairwise beatpath
    # comparisons tie 2-vs-2 and everyone scores 0 — a *Schulze tie* on a
    # perfectly symmetric cycle, which the method legitimately declares rather
    # than manufacturing an arbitrary winner.
    scores = pairwise.schulze([Ballot(["a", "b", "c"]), Ballot(["b", "c", "a"]), Ballot(["c", "a", "b"])])

    assert scores == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_schulze_breaks_a_copeland_tie_via_beatpath_widths() -> None:
    # A profile where Copeland ties (it counts only direct head-to-head wins) but
    # Schulze breaks the tie via beatpath widths.
    #
    # Ballots (5, over 4 contestants):
    #   [a,b,c,d], [a,b,c,d], [b,c,d,a], [c,d,a,b], [d,a,b,c]
    # Direct head-to-head wins:
    #   a>b 4:1  a>c 3:2  a>d 2:3 (d wins)
    #   b>c 4:1  b>d 3:2
    #   c>d 4:1
    # Copeland: a wins 2 loses 1 = +1; b wins 2 loses 1 = +1 (a,b tied);
    #           c wins 1 loses 2 = -1; d wins 1 loses 2 = -1 (c,d tied).
    # Schulze beatpath widths P[i,j] (strongest = max over chains of the min link):
    #   P[a,b]=4 (direct 4:1 is the widest one-link path); P[b,a]=3 (b->c->d->a
    #     = min(4,4,3)=3). 4 > 3, so a beatpath-beats b.
    #   The same 4-vs-3 pattern holds for every ordered pair (a direct 4:1 win
    #     somewhere up the chain vs a 3-wide indirect path back), yielding a
    #     strict beatpath order a > b > c > d.
    ballots = [
        Ballot(["a", "b", "c", "d"]),
        Ballot(["a", "b", "c", "d"]),
        Ballot(["b", "c", "d", "a"]),
        Ballot(["c", "d", "a", "b"]),
        Ballot(["d", "a", "b", "c"]),
    ]

    schulze_scores = pairwise.schulze(ballots)
    copeland_scores = pairwise.copeland(ballots)

    # Copeland ties a=b and c=d (ignores beatpath width).
    assert copeland_scores == {"a": 1.0, "b": 1.0, "c": -1.0, "d": -1.0}
    # Schulze breaks both ties into a strict order a > b > c > d.
    assert schulze_scores == {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0}


def test_schulze_treats_an_even_head_to_head_split_as_a_draw() -> None:
    # Two rankers each way: each contestant's beatpath to the other is 1, so
    # neither beatpath-beats the other -> both score 0 (a Schulze tie, not a win).
    scores = pairwise.schulze([Ballot(["a", "b"]), Ballot(["b", "a"])])

    assert scores == {"a": 0.0, "b": 0.0}


def test_schulze_scores_a_single_contestant_zero() -> None:
    # One contestant beats no one (no opponents), so it scores 0 — the empty-field
    # baseline, matching Copeland's behaviour.
    scores = pairwise.schulze([Ballot(["a"])])

    assert scores == {"a": 0.0}


def test_schulze_of_no_ballots_is_empty() -> None:
    assert pairwise.schulze([]) == {}


def test_schulze_scores_an_omitted_expected_contestant_as_zero() -> None:
    # a beats b on both ballots; c is in the universe but never ranked, so it has
    # no beatpaths and beats no one -> 0, yet still appears in the leaderboard.
    scores = pairwise.schulze([Ballot(["a", "b"]), Ballot(["a", "b"])], expected_contestants={"a", "b", "c"})

    assert scores == {"a": 1.0, "b": 0.0, "c": 0.0}


def test_schulze_strict_rejects_a_ballot_that_omits_an_expected_contestant() -> None:
    with pytest.raises(ValueError, match="omits expected contestant"):
        pairwise.schulze([Ballot(["a", "b"])], expected_contestants={"a", "b", "c"}, strict=True)


def test_schulze_raises_on_a_contestant_appearing_twice_in_one_ballot() -> None:
    with pytest.raises(ValueError, match="more than once"):
        pairwise.schulze([Ballot(["a", "b", "a"])])
