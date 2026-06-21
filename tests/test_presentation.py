"""Tests for the console presentation layer.

These assert on the returned strings (the functions never print), so no stdout
capturing is needed.  The pairwise grid test builds a real :class:`PairwiseTally`
via :func:`tournament_eval.aggregation.pairwise.matrix`, which needs ``numpy`` —
fine, it's a dev dependency; only the *call* needs it, not importing
:mod:`tournament_eval.presentation.console`.
"""

import subprocess
import sys

import numpy as np

from tournament_eval.aggregation import pairwise
from tournament_eval.aggregation.ballots import Ballot
from tournament_eval.presentation import console


def test_importing_presentation_does_not_pull_in_numpy() -> None:
    # console reads an already-built tally; importing it must not drag numpy in.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import tournament_eval.presentation\n"
                "import tournament_eval.presentation.console\n"
                "assert 'numpy' not in sys.modules, 'numpy was imported at module load!'\n"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_leaderboard_sorts_descending_with_name_tiebreak() -> None:
    out = console.leaderboard({"b": 0.9, "a": 0.5, "c": 0.5})
    lines = out.splitlines()

    assert lines[0].startswith("1")
    assert "b" in lines[0]
    assert "0.90" in lines[0]
    # Tied scores break by name ascending: a before c.
    assert "a" in lines[1]
    assert "c" in lines[2]


def test_leaderboard_right_aligns_scores() -> None:
    # A wide integer-valued score and a small one: the score column aligns to the
    # widest rendered score, so digits line up regardless of magnitude.
    out = console.leaderboard({"big": 123.0, "small": 0.5})
    lines = out.splitlines()

    big_line, small_line = lines
    assert big_line.endswith("123.00")
    assert small_line.endswith("0.50")
    # Right-aligned: the score column's right edge is the same column in both rows,
    # so the lines are equal length and each score sits flush at the end.
    assert len(big_line) == len(small_line)


def test_leaderboard_respects_precision() -> None:
    out = console.leaderboard({"a": 0.123456}, precision=4)
    assert "0.1235" in out


def test_leaderboard_empty_returns_empty_string() -> None:
    assert console.leaderboard({}) == ""


def test_pairwise_matrix_renders_grid_with_diagonal_dash() -> None:
    # Two ballots over {a, b, c}:
    #   ballot 1: a > b > c   -> a beats b, a beats c, b beats c
    #   ballot 2: b > a > c   -> b beats a, b beats c, a beats c
    # above[i, j] = ballots ranking i over j:
    #   a over b: 1   a over c: 2
    #   b over a: 1   b over c: 2
    #   c over a: 0   c over b: 0
    tally = pairwise.matrix([Ballot(["a", "b", "c"]), Ballot(["b", "a", "c"])])
    out = console.pairwise_matrix(tally)
    lines = out.splitlines()

    assert len(lines) == 4  # header + 3 rows
    assert lines[0].strip() == "a  b  c"
    assert lines[1] == "a  -  1  2"
    assert lines[2] == "b  1  -  2"
    assert lines[3] == "c  0  0  -"


def test_pairwise_matrix_sizes_columns_to_long_labels() -> None:
    tally = pairwise.matrix([Ballot(["long-name", "x"])])
    out = console.pairwise_matrix(tally)
    lines = out.splitlines()

    # The label column pads to the longest label's width (9 for "long-name"), so
    # the short row label "x" left-pads to the same width and columns line up.
    long_row = next(line for line in lines if line.startswith("long-name"))
    x_row = next(line for line in lines if line.lstrip().startswith("x"))
    assert long_row[: len("long-name")] == "long-name"
    assert x_row[: len("long-name")] == "x" + " " * (len("long-name") - 1)


def test_pairwise_matrix_empty_tally_returns_empty_string() -> None:
    empty = pairwise.PairwiseTally(contestants=[], above=np.zeros((0, 0), dtype=int))
    assert console.pairwise_matrix(empty) == ""


def test_ballot_summary_reports_appearances_and_mean_position() -> None:
    # a: positions 1, 2, 1 -> mean (1+2+1)/3 = 1.33
    # b: positions 2, 1, 3 -> mean 2.00
    # c: positions 3, 3, 2 -> mean 2.67
    out = console.ballot_summary([Ballot(["a", "b", "c"]), Ballot(["b", "a", "c"]), Ballot(["a", "c", "b"])])
    lines = out.splitlines()

    assert lines[0] == "3 ballots, field size 3 (all equal)"
    assert lines[1] == ""
    assert lines[2] == "Contestant  Appearances  Mean position"
    # Sorted by appearances desc (all 3), then mean position asc: a, b, c.
    assert "a" in lines[3]
    assert "1.33" in lines[3]
    assert "2.00" in lines[4]
    assert "2.67" in lines[5]
    # And the rows are in the a, b, c order.
    assert lines[3].startswith("a")
    assert lines[4].startswith("b")
    assert lines[5].startswith("c")


def test_ballot_summary_reports_size_spread_when_unequal() -> None:
    out = console.ballot_summary([Ballot(["a", "b", "c"]), Ballot(["a", "b"])])
    header = out.splitlines()[0]

    assert header == "2 ballots, field size min 2, median 2.5, max 3"


def test_ballot_summary_sorts_by_appearances_then_position() -> None:
    # d appears once (1 ballot) at position 1; a appears in both ballots.
    # Order: a (2 appearances) before d (1), regardless of d's top placement.
    out = console.ballot_summary([Ballot(["a", "b"]), Ballot(["a", "d"])])
    rows = out.splitlines()[2:]  # skip header + table header
    # Drop the table-header line if it sorted into the slice; find the data rows.
    data = [line for line in rows if line and not line.startswith("Contestant")]

    assert data[0].startswith("a")
    assert data[-1].startswith("d")


def test_ballot_summary_no_ballots() -> None:
    assert console.ballot_summary([]) == "0 ballots"


def test_ballot_summary_empty_ballots_returns_just_header() -> None:
    # Ballots exist but none rank anyone: turnout line only, no contestant table.
    assert console.ballot_summary([Ballot([]), Ballot([])]) == "2 ballots, field size 0 (all equal)"
