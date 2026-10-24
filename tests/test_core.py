from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from research.engine import make_splits, split_indices, write_submission
from research.features import FeatureBuilder, NumericEncoder
from research.metrics import calibrate_weights, evaluate
from research.task import infer_metric, load_task, validation_columns
from research.worker import train_model


def fixture_task(folder, kind="classification", numeric_classes=False):
    folder.mkdir(parents=True, exist_ok=True)
    target = "unknown_outcome"
    values = [10, 20, 10, 30, 20, 30] if numeric_classes else ["red", "blue", "red", "green", "blue", "green"]
    if kind == "regression":
        values = [.25, 2.2, -1.2, 3.5, 8.7, 12.1]
    pd.DataFrame({"id": range(6), "a": [1, 2, 3, 4, None, 6], "cat": ["a", "b", None, "a", "b", "a"], target: values}).to_csv(folder / "train.csv", index=False)
    pd.DataFrame({"id": [70, 71], "a": [9, None], "cat": ["unseen", "b"]}).to_csv(folder / "test.csv", index=False)
    pd.DataFrame({"id": [70, 71], target: [0, 0]}).to_csv(folder / "sample_submission.csv", index=False)
    pd.DataFrame({"feature": ["id", "a", "cat", target], "role": ["id", "feature", "feature", "target"], "dtype": ["numeric", "numeric", "category", "numeric" if kind == "regression" else "category"]}).to_csv(folder / "data_dictionary.csv", index=False)
    metric = "RMSE" if kind == "regression" else "Macro F1"
    (folder / "task.md").write_text(f"{kind}. Primary metric: {metric}. `train.csv` `test.csv` `sample_submission.csv` `data_dictionary.csv`", encoding="utf-8")
    return load_task(folder / "task.md")


@pytest.mark.parametrize("kind,numeric", [("regression", False), ("classification", False), ("classification", True)])
def test_task_infers_arbitrary_target(tmp_path, kind, numeric):
    task = fixture_task(tmp_path / "task", kind, numeric)
    assert task.target == "unknown_outcome"
    assert task.kind == kind
    assert task.features == ["a", "cat"]
    assert task.categorical == ["cat"]


def test_primary_metric_overrides_reference_metrics():
    text = "Classification. Reference uses Macro F1. Primary metric: Accuracy. RMSE is unsuitable."
    assert infer_metric(text, "classification") == "accuracy"


def test_explicit_validation_column_in_description(tmp_path):
    task = fixture_task(tmp_path / "task", "regression")
    task.description += "\nUse a time split on `a`."
    assert validation_columns(task) == (None, "a")


def test_features_fit_only_on_training_and_new_categories():
    fit = pd.DataFrame({"x": [1., 2., 3.], "c": ["a", "a", "b"]})
    builder = FeatureBuilder(extended=True).fit(fit)
    transformed = builder.transform(pd.DataFrame({"x": [1000.], "c": ["new"]}))
    assert transformed["__freq_c"].iloc[0] == 0
    assert builder.frequencies["c"] == {"a": 2 / 3, "b": 1 / 3}
    encoder = NumericEncoder().fit(builder.transform(fit))
    assert np.isfinite(encoder.transform(transformed)).all()


def test_feature_operations_are_finite_after_encoder():
    x = pd.DataFrame({"a": [1., 3.], "b": [0., 2.]})
    builder = FeatureBuilder(operations=[{"op": "ratio", "left": "a", "right": "b"}]).fit(x)
    frame = builder.transform(x)
    assert np.isnan(frame["__research_0"].iloc[0])
    assert np.isfinite(NumericEncoder().fit(frame).transform(frame)).all()


def test_rare_classes_remain_in_training():
    y = np.array([0] * 20 + [1] * 4 + [2])
    train, valid = split_indices(y, "classification", .2, 42)
    assert set(y[train]) == {0, 1, 2}
    assert not set(train) & set(valid)
    assert len(train) + len(valid) == len(y)


def test_duplicate_groups_do_not_cross_validation():
    x = pd.DataFrame({"x": np.repeat(np.arange(40), 3)})
    y = np.repeat(np.arange(40) * 1.5, 3)
    fit, cal, val, scheme = make_splits(x, y, "regression", 42, .2)
    assert scheme == "duplicate_groups"
    assert not set(x.iloc[fit].x) & set(x.iloc[val].x)
    assert not set(x.iloc[cal].x) & set(x.iloc[val].x)
    assert not set(x.iloc[fit].x) & set(x.iloc[cal].x)


def test_submission_keeps_labels_and_ids(tmp_path):
    task = fixture_task(tmp_path / "task")
    classes = np.array(["blue", "green", "red"])
    write_submission(task, np.array([[.1, .2, .7], [.6, .3, .1]]), classes)
    result = pd.read_csv(task.output)
    assert result.columns.tolist() == ["id", task.target]
    assert result.id.tolist() == [70, 71]
    assert result[task.target].tolist() == ["red", "blue"]


def test_rejects_changed_sample_order(tmp_path):
    task = fixture_task(tmp_path / "task")
    task.sample.iloc[::-1].to_csv(task.path.parent / "sample_submission.csv", index=False)
    with pytest.raises(ValueError, match="order"):
        load_task(task.path)


@pytest.mark.parametrize("family", ["catboost", "lightgbm", "extra_trees", "linear"])
@pytest.mark.parametrize("kind", ["regression", "classification"])
def test_model_families_handle_missing_and_unseen_values(family, kind):
    rng = np.random.default_rng(42)
    x = pd.DataFrame({"x": rng.normal(size=120), "cat": np.tile(["a", "b", "c"], 40)})
    x.loc[::7, "x"] = np.nan
    x.loc[::11, "cat"] = None
    y = np.nan_to_num(x.x.to_numpy()) * 2 + rng.normal(size=120) * .1
    if kind == "classification":
        y = (y > 0).astype(int)
    bundle = {"train": x, "y": y, "fit_indices": np.arange(80), "cal_indices": np.arange(80, 100),
              "categorical": ["cat"], "kind": kind, "metric": "rmse" if kind == "regression" else "macro_f1",
              "n_classes": 0 if kind == "regression" else 2, "threads": 1, "seed": 42}
    config = {"family": family, "iterations": 20, "depth": 4, "features": "extended"}
    model, _ = train_model(bundle, config)
    result = model.predict(pd.DataFrame({"x": [None, 2.], "cat": ["unseen", "a"]}))
    assert len(result) == 2
    assert np.isfinite(result).all()
    if kind == "classification":
        assert result.shape == (2, 2)
        assert np.allclose(result.sum(axis=1), 1)


def test_macro_f1_counts_missing_classes():
    assert evaluate("macro_f1", [0, 0], [0, 0], 2) == .5


def test_calibration_does_not_reduce_macro_f1():
    rng = np.random.default_rng(42)
    probabilities = rng.dirichlet([3, 1, .5], 100)
    y = np.repeat([0, 1, 2], [60, 30, 10])
    weights = calibrate_weights(y, probabilities, 3)
    assert evaluate("macro_f1", y, probabilities * weights, 3) >= evaluate("macro_f1", y, probabilities, 3)
