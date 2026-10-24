import json
from pathlib import Path
import subprocess
import sys

import pytest

from research.scoring import leaderboard_score


@pytest.mark.parametrize(
    ("rmse", "macro_f1", "expected"),
    [
        (20, 0.4, {"regression_points": 0, "classification_points": 0, "score": 0}),
        (25, 0.3, {"regression_points": 0, "classification_points": 0, "score": 0}),
        (0, 1, {"regression_points": 100, "classification_points": 100, "score": 100}),
        (10, 0.7, {"regression_points": 50, "classification_points": 50, "score": 50}),
        (0, 0.3, {"regression_points": 100, "classification_points": 0, "score": 50}),
        (25, 1, {"regression_points": 0, "classification_points": 100, "score": 50}),
    ],
)
def test_appendix_three_examples(rmse, macro_f1, expected):
    assert leaderboard_score(rmse, 20, macro_f1, 0.4) == pytest.approx(expected)


def test_unequal_improvements_have_equal_task_weights():
    assert leaderboard_score(12, 20, 0.9, 0.5) == pytest.approx(
        {"regression_points": 40, "classification_points": 80, "score": 60}
    )


def test_zero_baseline_f1_is_valid():
    assert leaderboard_score(1, 2, 0.3, 0)["classification_points"] == pytest.approx(30)


def test_near_perfect_baseline_does_not_change_normalization():
    assert leaderboard_score(1, 2, 1, 0.999999999999)["classification_points"] == 100


@pytest.mark.parametrize("position", range(4))
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), "0.5", None, True])
def test_rejects_non_finite_or_non_numeric_arguments(position, invalid):
    args = [1, 2, 0.8, 0.5]
    args[position] = invalid
    with pytest.raises(ValueError, match="finite number"):
        leaderboard_score(*args)


@pytest.mark.parametrize(
    ("args", "argument"),
    [
        ((-1, 2, 0.8, 0.5), "rmse"),
        ((1, 0, 0.8, 0.5), "baseline_rmse"),
        ((1, -1, 0.8, 0.5), "baseline_rmse"),
        ((1, 2, -0.1, 0.5), "macro_f1"),
        ((1, 2, 1.1, 0.5), "macro_f1"),
        ((1, 2, 0.8, -0.1), "baseline_f1"),
        ((1, 2, 0.8, 1), "baseline_f1"),
        ((1, 2, 0.8, 1.1), "baseline_f1"),
    ],
)
def test_rejects_invalid_metric_ranges(args, argument):
    with pytest.raises(ValueError, match=argument):
        leaderboard_score(*args)


def test_scores_are_monotonic_and_bounded():
    regression = [leaderboard_score(rmse, 10, 0.8, 0.4)["score"] for rmse in [20, 10, 5, 0]]
    classification = [leaderboard_score(5, 10, f1, 0.4)["score"] for f1 in [0, 0.4, 0.7, 1]]
    assert regression == sorted(regression)
    assert classification == sorted(classification)
    assert all(0 <= score <= 100 for score in regression + classification)


def test_cli_prints_json():
    result = subprocess.run(
        [sys.executable, "-m", "research.scoring", "--rmse", "12", "--baseline-rmse", "20", "--f1", "0.9", "--baseline-f1", "0.5"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout) == pytest.approx(
        {"regression_points": 40, "classification_points": 80, "score": 60}
    )
    assert result.stderr == ""


def test_cli_rejects_invalid_baseline_without_json_output():
    result = subprocess.run(
        [sys.executable, "-m", "research.scoring", "--rmse", "12", "--baseline-rmse", "0", "--f1", "0.9", "--baseline-f1", "0.5"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "baseline_rmse must be positive" in result.stderr
