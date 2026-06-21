"""Tests for the aggregation package: the pure positional math and the ballot adapter.

The numpy-backed pairwise methods live in :mod:`tests.test_aggregation_pairwise`.
"""

import subprocess
import sys

import pytest

from tests.conftest import GenerationResultFactory, RankingResultFactory
from tournament_eval.aggregation import ballots_from_rankings, positional
from tournament_eval.aggregation.ballots import Ballot


def test_importing_tournament_eval_does_not_pull_in_numpy() -> None:

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import tournament_eval\n"
                "import tournament_eval.aggregation\n"
                "import tournament_eval.aggregation.pairwise\n"
                "assert 'numpy' not in sys.modules, 'numpy was imported at module load!'\n"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_borda_scores_a_single_ballot_top_to_bottom() -> None:
    # A ballot of k=3 awards 2, 1, 0 from best to worst.
    scores = positional.borda([Ballot(["x", "y", "z"])])

    assert scores == {"x": 2.0, "y": 1.0, "z": 0.0}


def test_borda_sums_points_across_ballots() -> None:
    # Three rankers, the same three contestants in different orders.
    ballots = [
        Ballot(["a", "b", "c"]),  # a:2 b:1 c:0
        Ballot(["b", "a", "c"]),  # b:2 a:1 c:0
        Ballot(["a", "c", "b"]),  # a:2 c:1 b:0
    ]

    scores = positional.borda(ballots)

    assert scores == {"a": 5.0, "b": 3.0, "c": 1.0}


def test_borda_of_no_ballots_is_empty() -> None:
    assert positional.borda([]) == {}


def test_borda_raises_on_a_contestant_appearing_twice_in_one_ballot() -> None:
    # A duplicate within a ballot is malformed (not a tie) and would double-count.
    with pytest.raises(ValueError, match="more than once"):
        positional.borda([Ballot(["a", "b", "a"])])


def test_normalized_borda_rescales_a_single_ballot_to_zero_one() -> None:
    # k=3: raw points 2,1,0 divided by (k-1)=2 -> 1.0, 0.5, 0.0.
    scores = positional.normalized_borda([Ballot(["x", "y", "z"])])

    assert scores == {"x": 1.0, "y": 0.5, "z": 0.0}


def test_normalized_borda_averages_a_contestant_over_its_ballots() -> None:
    # "a" tops one ballot and bottoms the other: mean of 1.0 and 0.0 is 0.5.
    scores = positional.normalized_borda([Ballot(["a", "b"]), Ballot(["b", "a"])])

    assert scores == {"a": 0.5, "b": 0.5}


def test_normalized_borda_does_not_reward_more_frequent_participation() -> None:
    # x wins its one and only ballot; y wins both of its two. Averaging (not summing)
    # puts them level at 1.0 — frequency of appearance gives no advantage.
    ballots = [
        Ballot(["x", "loser_one"]),
        Ballot(["y", "loser_two"]),
        Ballot(["y", "loser_three"]),
    ]

    scores = positional.normalized_borda(ballots)

    assert scores["x"] == 1.0
    assert scores["y"] == 1.0


def test_normalized_borda_silently_skips_ballots_with_no_comparison() -> None:
    # A lone-candidate ballot carries no comparative information: "solo" is dropped
    # silently (no warning), and the size-2 ballot is scored normally.
    scores = positional.normalized_borda([Ballot(["solo"]), Ballot(["a", "b"])])

    assert scores == {"a": 1.0, "b": 0.0}


def test_normalized_borda_strict_raises_on_a_no_comparison_ballot() -> None:
    with pytest.raises(ValueError, match="fewer than two candidates"):
        positional.normalized_borda([Ballot(["solo"]), Ballot(["a", "b"])], strict=True)


def test_normalized_borda_of_no_ballots_is_empty() -> None:
    assert positional.normalized_borda([]) == {}


def test_normalized_borda_raises_on_a_contestant_appearing_twice_in_one_ballot() -> None:
    with pytest.raises(ValueError, match="more than once"):
        positional.normalized_borda([Ballot(["a", "b", "a"])])


def test_ballots_from_rankings_resolves_aliases_back_to_authors(
    make_generation_result: GenerationResultFactory, make_ranking_result: RankingResultFactory
) -> None:
    gen_alpha = make_generation_result(author="alpha")
    gen_beta = make_generation_result(author="beta")
    gen_gamma = make_generation_result(author="gamma")
    # The ranking is in id space, best-first: beta, then alpha, then gamma.
    ranking_result = make_ranking_result(ranking=[gen_beta.id, gen_alpha.id, gen_gamma.id])

    ballots = ballots_from_rankings([ranking_result], [gen_alpha, gen_beta, gen_gamma])

    assert len(ballots) == 1
    assert ballots[0].ranking == ["beta", "alpha", "gamma"]
    assert ballots[0].author == "test-model"


def test_ballots_from_rankings_produces_one_ballot_per_ranking_in_order(
    make_generation_result: GenerationResultFactory, make_ranking_result: RankingResultFactory
) -> None:
    gen_alpha = make_generation_result(author="alpha")
    gen_beta = make_generation_result(author="beta")
    first = make_ranking_result(ranking=[gen_alpha.id, gen_beta.id])
    second = make_ranking_result(ranking=[gen_beta.id, gen_alpha.id])

    ballots = ballots_from_rankings([first, second], [gen_alpha, gen_beta])

    assert [b.ranking for b in ballots] == [["alpha", "beta"], ["beta", "alpha"]]
    assert [b.author for b in ballots] == ["test-model", "test-model"]


def test_ballots_from_rankings_accepts_single_pass_iterators(
    make_generation_result: GenerationResultFactory, make_ranking_result: RankingResultFactory
) -> None:
    gen_alpha = make_generation_result(author="alpha")
    gen_beta = make_generation_result(author="beta")
    ranking_result = make_ranking_result(ranking=[gen_alpha.id, gen_beta.id])

    ballots = ballots_from_rankings(
        (r for r in [ranking_result]),
        (g for g in [gen_alpha, gen_beta]),
    )

    assert len(ballots) == 1
    assert ballots[0].ranking == ["alpha", "beta"]
    assert ballots[0].author == "test-model"


def test_ballots_from_rankings_raises_on_a_ranked_id_with_no_generation(
    make_generation_result: GenerationResultFactory, make_ranking_result: RankingResultFactory
) -> None:
    gen_alpha = make_generation_result(author="alpha")
    orphan = make_generation_result(author="orphan")  # deliberately not passed in
    ranking_result = make_ranking_result(ranking=[gen_alpha.id, orphan.id])

    # A ranking referencing an id absent from the generations is a data-integrity error.
    with pytest.raises(KeyError):
        ballots_from_rankings([ranking_result], [gen_alpha])


def test_ballots_feed_straight_into_borda_for_a_leaderboard(
    make_generation_result: GenerationResultFactory, make_ranking_result: RankingResultFactory
) -> None:
    # The whole wire: rankings -> ballots -> Borda scores. alpha wins both ballots.
    gen_alpha = make_generation_result(author="alpha")
    gen_beta = make_generation_result(author="beta")
    gen_gamma = make_generation_result(author="gamma")
    generations = [gen_alpha, gen_beta, gen_gamma]
    rankings = [
        make_ranking_result(ranking=[gen_alpha.id, gen_beta.id, gen_gamma.id]),
        make_ranking_result(ranking=[gen_alpha.id, gen_gamma.id, gen_beta.id]),
    ]

    scores = positional.borda(ballots_from_rankings(rankings, generations))

    assert scores == {"alpha": 4.0, "beta": 1.0, "gamma": 1.0}


def test_ballot_acts_as_a_read_only_sequence_of_labels() -> None:
    # The math modules read a ballot the same way they read a list[str]: iteration,
    # len, membership, indexing. author is preserved and identity-aware on equality.
    ballot = Ballot(["a", "b", "c"], author="r1")

    assert list(ballot) == ["a", "b", "c"]
    assert len(ballot) == 3
    assert set(ballot) == {"a", "b", "c"}
    assert ballot[0] == "a"
    assert ballot[-1] == "c"


def test_ballot_is_identity_aware_two_ballots_with_the_same_ranking_but_different_authors_differ() -> None:
    # frozen=True gives equality over both ranking and author — load-bearing for bias
    # analysis, where two rankers producing the same order must not be conflated.
    assert Ballot(["a", "b"], author="r1") != Ballot(["a", "b"], author="r2")
    assert Ballot(["a", "b"], author="r1") == Ballot(["a", "b"], author="r1")
