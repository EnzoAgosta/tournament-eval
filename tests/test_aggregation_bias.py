"""Known-answer tests for the bias analysis module.

Every expected value here is computed by hand from the ballots, not derived from
the implementation, so the tests pin the *meaning* of the methods.  Pure Python —
no ``numpy`` or ``matplotlib`` is needed, and a separate test pins that importing
the module pulls neither in.
"""

import subprocess
import sys

import pytest

from tournament_eval.aggregation import bias
from tournament_eval.aggregation.ballots import Ballot


def test_importing_bias_does_not_pull_in_heavy_deps() -> None:
    # bias is pure Python like the positional methods; importing it must not drag
    # numpy or matplotlib into sys.modules.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import tournament_eval.aggregation.bias\n"
                "assert 'numpy' not in sys.modules, 'numpy was imported at module load!'\n"
                "assert 'matplotlib' not in sys.modules, 'matplotlib was imported at module load!'\n"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# --- preference_matrix -------------------------------------------------------


def test_preference_matrix_records_mean_placement_per_ranker_contestant() -> None:
    # r1 ranks a,b,c (positions 1,2,3); r2 ranks b,a,c (positions for a,b,c: 2,1,3).
    ballots = [
        Ballot(["a", "b", "c"], author="r1"),
        Ballot(["b", "a", "c"], author="r2"),
    ]

    matrix = bias.preference_matrix(ballots)

    assert matrix.rankers == ["r1", "r2"]
    assert matrix.contestants == ["a", "b", "c"]
    assert matrix.placement == [
        [1.0, 2.0, 3.0],  # r1: a=1, b=2, c=3
        [2.0, 1.0, 3.0],  # r2: a=2, b=1, c=3
    ]


def test_preference_matrix_averages_a_rankers_multiple_ballots() -> None:
    # r1 ranks a then b, then b then a: mean position of each is (1+2)/2 = 1.5.
    ballots = [
        Ballot(["a", "b"], author="r1"),
        Ballot(["b", "a"], author="r1"),
    ]

    matrix = bias.preference_matrix(ballots)

    assert matrix.rankers == ["r1"]
    assert matrix.contestants == ["a", "b"]
    assert matrix.placement == [[1.5, 1.5]]


def test_preference_matrix_none_when_ranker_never_ranked_contestant() -> None:
    # r1 and r2 rank disjoint fields: each ranker's row is None for the contestants
    # it never met — absence is not averaged in as a worst placement.
    ballots = [
        Ballot(["a", "b"], author="r1"),
        Ballot(["c", "d"], author="r2"),
    ]

    matrix = bias.preference_matrix(ballots)

    assert matrix.rankers == ["r1", "r2"]
    assert matrix.contestants == ["a", "b", "c", "d"]
    assert matrix.placement == [
        [1.0, 2.0, None, None],
        [None, None, 1.0, 2.0],
    ]


def test_preference_matrix_empty_ballots_yields_empty_grid() -> None:
    matrix = bias.preference_matrix([])

    assert matrix.rankers == []
    assert matrix.contestants == []
    assert matrix.placement == []


def test_preference_matrix_skips_authorless_ballots_by_default() -> None:
    # An authorless ballot carries no ranker signal; it's dropped silently, and
    # only the authored ballot contributes.
    ballots = [
        Ballot(["a", "b"]),
        Ballot(["a", "b"], author="r1"),
    ]

    matrix = bias.preference_matrix(ballots)

    assert matrix.rankers == ["r1"]
    assert matrix.placement == [[1.0, 2.0]]


def test_preference_matrix_strict_raises_on_authorless_ballot() -> None:
    with pytest.raises(ValueError, match="no author"):
        bias.preference_matrix([Ballot(["a", "b"])], strict=True)


def test_preference_matrix_raises_on_a_contestant_appearing_twice_in_one_ballot() -> None:
    with pytest.raises(ValueError, match="more than once"):
        bias.preference_matrix([Ballot(["a", "b", "a"], author="r1")])


# --- self_preference ---------------------------------------------------------


def test_self_preference_positive_when_ranker_places_self_above_panel() -> None:
    # Each ranker puts itself first; the other puts it second. panel_mean (2) -
    # self_mean (1) = +1.0: positive = self-preference.
    ballots = [
        Ballot(["alpha", "beta"], author="alpha"),
        Ballot(["beta", "alpha"], author="beta"),
    ]

    assert bias.self_preference(ballots) == {"alpha": 1.0, "beta": 1.0}


def test_self_preference_negative_when_ranker_places_self_below_panel() -> None:
    # Each ranker puts itself second; the other puts it first. panel_mean (1) -
    # self_mean (2) = -1.0: negative = self-deprecation.
    ballots = [
        Ballot(["beta", "alpha"], author="alpha"),
        Ballot(["alpha", "beta"], author="beta"),
    ]

    assert bias.self_preference(ballots) == {"alpha": -1.0, "beta": -1.0}


def test_self_preference_zero_when_panel_agrees_with_self() -> None:
    # Both rankers agree on the order alpha > beta, so each ranker's self-placement
    # matches where the panel places it: score 0.0.
    ballots = [
        Ballot(["alpha", "beta"], author="alpha"),
        Ballot(["alpha", "beta"], author="beta"),
    ]

    assert bias.self_preference(ballots) == {"alpha": 0.0, "beta": 0.0}


def test_self_preference_excludes_ranker_from_panel_baseline() -> None:
    # alpha places itself at 2; beta and gamma (the rest of the panel) place alpha
    # at 1. panel_mean (1) - self_mean (2) = -1.0. If alpha's own ballot were in the
    # panel baseline the mean would be (2+1+1)/3 = 1.33 and the score would differ,
    # so the expected -1.0 pins the exclude-self behaviour.
    # beta places itself at 2; alpha and gamma place beta at 1 and 2 -> mean 1.5,
    # score 1.5 - 2 = -0.5. gamma is a ranker but never a contestant -> excluded.
    ballots = [
        Ballot(["beta", "alpha"], author="alpha"),
        Ballot(["alpha", "beta"], author="beta"),
        Ballot(["alpha", "beta"], author="gamma"),
    ]

    assert bias.self_preference(ballots) == {"alpha": -1.0, "beta": -0.5}


def test_self_preference_averages_a_rankers_multiple_self_placements() -> None:
    # alpha ranks itself first in two ballots (self_mean 1); the panel ranks alpha
    # last in two ballots (panel_mean 3): score 3 - 1 = 2.0.
    ballots = [
        Ballot(["alpha", "beta", "gamma"], author="alpha"),
        Ballot(["alpha", "gamma", "beta"], author="alpha"),
        Ballot(["gamma", "beta", "alpha"], author="beta"),
        Ballot(["beta", "gamma", "alpha"], author="gamma"),
    ]

    scores = bias.self_preference(ballots)
    assert scores["alpha"] == 2.0


def test_self_preference_excludes_a_ranker_that_never_ranked_itself() -> None:
    # alpha and beta are both rankers and contestants, but neither's own ballot
    # contains itself — so there's no self-placement to compare and both are absent.
    ballots = [
        Ballot(["beta", "gamma"], author="alpha"),
        Ballot(["alpha", "gamma"], author="beta"),
    ]

    assert bias.self_preference(ballots) == {}


def test_self_preference_excludes_a_ranker_the_panel_never_ranked() -> None:
    # alpha ranks itself, but no other ranker ever ranks alpha — the panel baseline
    # is undefined, so alpha is excluded rather than scored against an empty panel.
    ballots = [
        Ballot(["alpha", "beta"], author="alpha"),
    ]

    assert bias.self_preference(ballots) == {}


def test_self_preference_returns_scores_sorted_by_author() -> None:
    # Constructed so all three score the same (each puts self first, panel second):
    # the dict's key order must be sorted by author label, not insertion order.
    ballots = [
        Ballot(["zeta", "alpha"], author="zeta"),
        Ballot(["alpha", "zeta"], author="alpha"),
    ]

    scores = bias.self_preference(ballots)
    assert list(scores.keys()) == ["alpha", "zeta"]


def test_self_preference_skips_authorless_ballots_by_default() -> None:
    # The authorless ballot is dropped; only alpha's authored ballot counts, and
    # with no panel baseline for alpha the result is empty (not an error).
    ballots = [
        Ballot(["alpha", "beta"]),
        Ballot(["alpha", "beta"], author="alpha"),
    ]

    assert bias.self_preference(ballots) == {}


def test_self_preference_strict_raises_on_authorless_ballot() -> None:
    with pytest.raises(ValueError, match="no author"):
        bias.self_preference([Ballot(["alpha", "beta"])], strict=True)


def test_self_preference_raises_on_a_contestant_appearing_twice_in_one_ballot() -> None:
    with pytest.raises(ValueError, match="more than once"):
        bias.self_preference([Ballot(["alpha", "beta", "alpha"], author="alpha")])
