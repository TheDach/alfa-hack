import argparse
import json
import math
from numbers import Real


def _finite_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def leaderboard_score(
    rmse: float,
    baseline_rmse: float,
    macro_f1: float,
    baseline_f1: float,
) -> dict[str, float]:
    rmse = _finite_number("rmse", rmse)
    baseline_rmse = _finite_number("baseline_rmse", baseline_rmse)
    macro_f1 = _finite_number("macro_f1", macro_f1)
    baseline_f1 = _finite_number("baseline_f1", baseline_f1)
    if rmse < 0:
        raise ValueError("rmse must be non-negative")
    if baseline_rmse <= 0:
        raise ValueError("baseline_rmse must be positive")
    if not 0 <= macro_f1 <= 1:
        raise ValueError("macro_f1 must be between 0 and 1")
    if not 0 <= baseline_f1 < 1:
        raise ValueError("baseline_f1 must be between 0 (inclusive) and 1 (exclusive)")
    regression_points = 100.0 * max(0.0, 1.0 - rmse / baseline_rmse)
    classification_points = 100.0 * max(
        0.0, (macro_f1 - baseline_f1) / (1.0 - baseline_f1)
    )
    return {
        "regression_points": regression_points,
        "classification_points": classification_points,
        "score": (regression_points + classification_points) / 2.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate the Appendix 3 leaderboard score.")
    parser.add_argument("--rmse", required=True, type=float)
    parser.add_argument("--baseline-rmse", required=True, type=float)
    parser.add_argument("--f1", required=True, type=float)
    parser.add_argument("--baseline-f1", required=True, type=float)
    args = parser.parse_args()
    try:
        result = leaderboard_score(args.rmse, args.baseline_rmse, args.f1, args.baseline_f1)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
