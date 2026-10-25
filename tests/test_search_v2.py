from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import engine
from research.engine import (detect_forward_time, ensemble_weights, final_reserve,
                             fit_oof_candidate, make_oof_splits, make_splits)
from research.metrics import evaluate
from research.search import CORE_COVERAGE, candidates, coverage_key, coverage_pending


def profile(kind="classification"):
    result = {"kind": kind, "metric": "macro_f1" if kind == "classification" else "rmse",
              "target_quantiles": {0.: 1., 1.: 10.}}
    return result


def test_mandatory_search_covers_available_families_even_after_failures(monkeypatch):
    monkeypatch.setattr("research.search.importlib.util.find_spec", lambda module: object())
    history = []
    observed = []
    for index in range(len(CORE_COVERAGE)):
        menu = candidates(profile(), history)
        config = menu[0]
        observed.append(coverage_key(config))
        assert coverage_pending(history, menu, profile()) == coverage_key(config)
        history.append({"status": "timeout" if index % 2 else "ok", "metric": .5,
                        "config": config, "elapsed_seconds": 1.})
    assert observed == list(CORE_COVERAGE)
    assert coverage_pending(history, candidates(profile(), history), profile()) is None


def test_entity_history_is_covered_early_and_only_when_available(monkeypatch):
    monkeypatch.setattr("research.search.importlib.util.find_spec", lambda module: object())
    p = profile("regression") | {"entity_groups": [["category", "amount"]]}
    history = []
    observed = []
    for _ in range(3):
        config = candidates(p, history)[0]
        observed.append(coverage_key(config))
        history.append({"status": "ok", "metric": 1., "config": config, "elapsed_seconds": 1.})
    assert observed == ["catboost", "lightgbm:native", "lightgbm:native:groups"]
    assert history[-1]["config"]["group_features"] == p["entity_groups"]
    assert history[-1]["config"]["history_residual"] is True
    assert not any(config.get("group_features") for config in candidates(profile(), []))


def test_forward_shift_infers_discrete_year_and_rejects_ordinary_features():
    train = pd.DataFrame({"Year_Factor": np.repeat(np.arange(1, 6), 10),
                          "amount": np.arange(50)})
    test = pd.DataFrame({"Year_Factor": [6] * 5, "amount": np.arange(100, 105)})
    assert detect_forward_time(train, test) == "Year_Factor"
    assert detect_forward_time(train.drop(columns="Year_Factor"), test.drop(columns="Year_Factor")) is None
    test["Year_Factor"] = 3
    assert detect_forward_time(train, test) is None


def test_discrete_time_split_keeps_entire_years_separate():
    x = pd.DataFrame({"Year_Factor": np.repeat(np.arange(1, 6), 10)})
    fit, cal, valid, scheme = make_splits(x, np.arange(50), "regression", 42, .2, time_column="Year_Factor")
    assert scheme == "time"
    assert set(x.iloc[fit].Year_Factor) == {1, 2, 3}
    assert set(x.iloc[cal].Year_Factor) == {4}
    assert set(x.iloc[valid].Year_Factor) == {5}
    assert sorted(np.concatenate([fit, cal, valid])) == list(range(50))


@pytest.mark.parametrize("kind", ["regression", "classification"])
def test_oof_each_row_is_held_out_once_without_group_overlap(kind):
    x = pd.DataFrame({"group": np.repeat(np.arange(12), 3), "x": np.arange(36)})
    y = np.tile([0, 1, 2], 12) if kind == "classification" else np.arange(36) * .5
    splits, scheme = make_oof_splits(x, y, kind, 42, group="group")
    assert scheme == "group_oof"
    assert len(splits) == 3
    assert sorted(np.concatenate([valid for _, valid in splits])) == list(range(36))
    for fit, valid in splits:
        assert not set(x.iloc[fit].group) & set(x.iloc[valid].group)
        assert not set(fit) & set(valid)


def test_oof_singleton_class_is_skipped():
    x = pd.DataFrame({"x": np.arange(9)})
    splits, reason = make_oof_splits(x, np.array([0] * 8 + [1]), "classification", 42)
    assert splits == []
    assert reason == "singleton_class"


def test_greedy_ensemble_is_convex_and_captures_complementary_errors():
    y = np.array([1., 2., 4., 8.])
    values = [y + 1., y - 1., y + 100.]
    weights, score = ensemble_weights("rmse", y, values)
    assert weights.sum() == pytest.approx(1.)
    assert (weights >= 0).all()
    assert weights[2] == 0
    assert score < min(evaluate("rmse", y, value) for value in values)
    assert score == pytest.approx(0.)


def test_final_reserve_scales_to_slow_selected_models():
    rows = [{"elapsed_seconds": 900., "metric": 1.}, {"elapsed_seconds": 700., "metric": 2.}]
    reserve = final_reserve(10000, rows, 1000, 680, "rmse")
    assert 3500 <= reserve <= 6500
    assert reserve > 3500
    assert final_reserve(60, [], 100, 68, "rmse") == 15


@pytest.mark.parametrize("kind", ["regression", "classification"])
def test_oof_workers_never_receive_their_heldout_labels_as_training(tmp_path, monkeypatch, kind):
    n = 60
    x = pd.DataFrame({"row_key": np.arange(n), "signal": np.tile([0., 1.], n // 2)})
    y = x.signal.to_numpy(dtype=int) if kind == "classification" else x.row_key.to_numpy() * 2.
    test = pd.DataFrame({"row_key": [100, 101], "signal": [0., 1.]})
    bundle = {"train": x, "test": test, "y": y, "kind": kind,
              "metric": "macro_f1" if kind == "classification" else "rmse",
              "n_classes": 2 if kind == "classification" else 0, "threads": 1, "seed": 42,
              "categorical": [], "fit_indices": np.arange(40),
              "cal_indices": np.arange(40, 50), "validation_indices": np.arange(50, 60)}
    splits, _ = make_oof_splits(x, y, kind, 42)
    seen = []
    def worker(data_path, config, folder, timeout, final, cwd, memory):
        data = joblib.load(data_path)
        assert final is True
        assert config["class_weights"] == [1.] * bundle["n_classes"]
        held = data["test"].iloc[:-2]
        assert not set(data["train"].row_key) & set(held.row_key)
        assert set(data["train"].row_key) | set(held.row_key) == set(range(n))
        assert len(data["cal_indices"]) == 0
        expected_y = (data["train"].signal.to_numpy(dtype=int) if kind == "classification"
                      else data["train"].row_key.to_numpy() * 2.)
        np.testing.assert_equal(data["y"], expected_y)
        seen.extend(held.row_key.tolist())
        if kind == "classification":
            p = .1 + data["test"].signal.to_numpy() * .8
            prediction = np.column_stack([1-p, p])
        else:
            prediction = data["test"].row_key.to_numpy() * 2.
        np.savez_compressed(folder / "predictions.npz", test=prediction)
        return {"status": "ok", "iterations": 20, "elapsed_seconds": .01}
    monkeypatch.setattr(engine, "run_worker", worker)
    row = {"config": {"family": "lightgbm", "iterations": 20},
           "iterations": 20, "number": 0, "class_weights": [2., 3.]}
    result = fit_oof_candidate(bundle, row, splits, tmp_path, time.monotonic() + 60, 1, tmp_path)
    assert result["status"] == "ok"
    assert sorted(seen) == list(range(n))
    assert len(result["models"]) == 3
    assert sum(model["weight"] for model in result["models"]) == pytest.approx(1)
    with np.load(tmp_path / "oof_predictions.npz") as saved:
        if kind == "regression":
            np.testing.assert_equal(saved["oof"], y)
            np.testing.assert_equal(saved["test"], [200., 202.])
        else:
            assert saved["oof"].argmax(axis=1).tolist() == y.tolist()
            assert saved["test"].argmax(axis=1).tolist() == [0, 1]


def test_entity_metadata_is_normalized_and_history_candidates_keep_distinct_hypotheses(monkeypatch):
    monkeypatch.setattr("research.search.importlib.util.find_spec", lambda module: object())
    metadata = [{"columns": ["category", "amount"], "train_duplicate_fraction": .8, "test_seen_fraction": .7},
                {"columns": ["category", "amount"], "train_duplicate_fraction": .8, "test_seen_fraction": .7},
                ["category", "size"]]
    p = profile("regression") | {"entity_groups": metadata}
    menu = candidates(p, [])
    grouped = [config for config in menu if config.get("group_features")]
    assert len(grouped) == 5
    expected = [["category", "amount"], ["category", "size"]]
    assert all(config["group_features"] == expected for config in grouped)
    assert all(not isinstance(item, dict) for config in grouped for item in config["group_features"])
    direct = next(config for config in grouped if config.get("history_residual") is False)
    stable = next(config for config in grouped if config.get("stable_features"))
    assert direct["num_leaves"] == stable["num_leaves"] == 31
    assert direct["min_child_samples"] == stable["min_child_samples"] == 40
    assert direct["l2"] == stable["l2"] == 15
    assert stable["history_residual"] is True

