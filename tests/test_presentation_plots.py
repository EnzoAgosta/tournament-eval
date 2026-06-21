"""Tests for the plot presentation layer.

Each function builds a :class:`matplotlib.figure.Figure` and returns it (never
calls ``plt.show``), so these assert on the figure's structure — axes, bars,
image arrays, tick labels, annotations — without rendering to a file or a
display.  ``matplotlib`` is a dev dependency, so it's available in the test env;
the lazy-import invariant (importing the module pulls in neither matplotlib nor
numpy) is checked separately in a subprocess.

The backend is forced to ``Agg`` before any pyplot import so the suite runs
headless on CI without a display or GUI warnings.  ``fig.axes[0]`` is the main
axes throughout: a colorbar adds a second axes, so unpacking ``[ax] = fig.axes``
would break on the heatmap/distribution figures.
"""

import subprocess
import sys
from typing import cast

import matplotlib

matplotlib.use("Agg")  # before pyplot is imported (lazily, inside the functions)

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from tournament_eval.aggregation import pairwise
from tournament_eval.presentation import plots


def test_importing_presentation_plots_does_not_pull_in_heavy_deps() -> None:
    # The module references matplotlib only under TYPE_CHECKING and numpy not at
    # all; importing it must drag neither into sys.modules.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import tournament_eval.presentation.plots\n"
                "assert 'matplotlib' not in sys.modules, "
                "'matplotlib was imported at module load!'\n"
                "assert 'numpy' not in sys.modules, 'numpy was imported at module load!'\n"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_leaderboard_bar_returns_a_figure_with_one_bar_per_contestant() -> None:
    scores = {"a": 0.9, "b": 0.5, "c": 0.1}

    fig = plots.leaderboard_bar(scores)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    bars = ax.patches
    assert len(bars) == 3
    # Best (a) sits at the top after invert_yaxis: its bar is the first y-tick.
    ytick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert ytick_labels[0] == "a"
    # Bar widths match the scores, in the plotted (y-axis) order.
    plotted_values = [cast(Rectangle, bar).get_width() for bar in bars]
    assert plotted_values == [0.9, 0.5, 0.1]


def test_leaderboard_bar_empty_scores_yields_empty_axes() -> None:
    fig = plots.leaderboard_bar({})

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(ax.patches) == 0


def test_leaderboard_bar_precision_formats_bar_labels() -> None:
    fig = plots.leaderboard_bar({"a": 0.123456}, precision=4)

    ax = fig.axes[0]
    # bar_label adds one Annotation per bar to the axes' text artists.
    label_texts = [text.get_text() for text in ax.texts]
    assert label_texts == ["0.1235"]


def test_pairwise_heatmap_renders_an_n_by_n_grid_with_diagonal_masked() -> None:
    # Same ballots as the console grid test: above[i,j] = a/b:1, a/c:2, b/c:2.
    tally = pairwise.matrix([["a", "b", "c"], ["b", "a", "c"]])

    fig = plots.pairwise_heatmap(tally)

    # The colorbar adds a second axes; the main one is first and holds the image.
    assert len(fig.axes) == 2
    ax = fig.axes[0]
    image = ax.images[0]
    data = np.asarray(image.get_array())
    assert data.shape == (3, 3)
    # Diagonal masked as NaN.
    assert np.all(np.isnan(np.diag(data)))
    # Off-diagonals carry the counts: above[0,1]=1, above[0,2]=2, above[1,2]=2.
    assert data[0, 1] == 1.0
    assert data[0, 2] == 2.0
    assert data[1, 2] == 2.0
    # Both axes are labelled with the contestants (sorted by the tally).
    xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    ytick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert xtick_labels == ["a", "b", "c"]
    assert ytick_labels == ["a", "b", "c"]


def test_pairwise_heatmap_empty_tally_yields_empty_axes() -> None:
    empty = pairwise.PairwiseTally(contestants=[], above=np.zeros((0, 0), dtype=int))

    fig = plots.pairwise_heatmap(empty)

    # No image and no colorbar when the tally is empty.
    assert len(fig.axes) == 1
    assert len(fig.axes[0].images) == 0


def test_rank_distribution_counts_positions_per_contestant() -> None:
    # a: positions 1, 2, 1 -> {1:2, 2:1}; b: 2, 1, 3 -> {1:1, 2:1, 3:1}; c: 3, 3, 2 -> {2:1, 3:2}
    ballots = [["a", "b", "c"], ["b", "a", "c"], ["a", "c", "b"]]

    fig = plots.rank_distribution(ballots)

    assert len(fig.axes) == 2  # main + colorbar
    ax = fig.axes[0]
    data = np.asarray(ax.images[0].get_array())
    assert data.shape == (3, 3)  # 3 contestants x 3 positions
    # Row order: sorted by appearances desc (all 3), then mean position asc: a, b, c.
    ytick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert ytick_labels == ["a", "b", "c"]
    # a: position 1 twice, position 2 once.
    assert data[0, 0] == 2.0
    assert data[0, 1] == 1.0
    assert data[0, 2] == 0.0
    # c: position 3 twice, position 2 once.
    assert data[2, 1] == 1.0
    assert data[2, 2] == 2.0
    xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert xtick_labels == ["1", "2", "3"]


def test_rank_distribution_handles_unequal_field_sizes() -> None:
    # A 3-candidate ballot and a 2-candidate ballot: max_field_size = 3, so the
    # grid is 3 columns; the shorter ballot can only place contestants in cols 1-2.
    ballots = [["a", "b", "c"], ["a", "b"]]

    fig = plots.rank_distribution(ballots)

    ax = fig.axes[0]
    data = np.asarray(ax.images[0].get_array())
    assert data.shape[1] == 3
    ytick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    # 'a' is placed 1st in both ballots — find its row by label, not position.
    a_row = next(row for row, label in zip(data, ytick_labels, strict=False) if label == "a")
    assert a_row[0] == 2.0


def test_rank_distribution_no_ballots_yields_empty_axes() -> None:
    fig = plots.rank_distribution([])

    assert len(fig.axes) == 1
    assert len(fig.axes[0].images) == 0


def test_plot_functions_raise_clear_importerror_when_matplotlib_is_missing() -> None:
    # A subprocess with matplotlib poisoned must surface the plotting extra, not
    # a bare ModuleNotFoundError.  Covers the shared _new_figure import path.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "sys.modules['matplotlib'] = None  # poison the import\n"
                "from tournament_eval.presentation import plots\n"
                "try:\n"
                "    plots.leaderboard_bar({'a': 1.0})\n"
                "except ImportError as err:\n"
                "    assert 'matplotlib' in str(err), err\n"
                "    assert 'plotting' in str(err), err\n"
                "else:\n"
                "    raise SystemExit('expected ImportError')\n"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
