import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.engine import make_splits, run, write_submission
from research.features import FeatureBuilder, NumericEncoder
from research.metrics import evaluate
from research.task import infer_metric, load_task
from research.worker import FittedModel


def task_files(root, y, description="Regression. Metric RMSE."):
    root.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame({"key": np.arange(len(y)), "amount": np.arange(len(y), dtype=float), "segment": ["a", "b"] * (len(y) // 2) + (["a"] if len(y) % 2 else []), "outcome": y})
    test = pd.DataFrame({"key": [101, 205], "amount": [0.5, 2.5], "segment": ["new", None]})
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    pd.DataFrame({"key": [101, 205], "outcome": [0, 0]}).to_csv(root / "sample_submission.csv", index=False)
    pd.DataFrame({"feature": ["key", "amount", "segment", "outcome"], "role": ["id", "feature", "feature", "target"], "dtype": ["integer", "float", "category", "float"]}).to_csv(root / "data_dictionary.csv", index=False)
    path = root / "task.md"
    path.write_text(description, encoding="utf-8")
    return path


def runner_args(path, **overrides):
    values = dict(task=str(path), target=None, task_type=None, metric=None, output=None,
                  work_dir=str(path.parent / "runs"), seed=42, validation_fraction=.2,
                  group_column=None, time_column=None, time_budget=60, threads=1,
                  max_trials=0, trial_timeout=10, memory_gb=1, keep_data=False)
    return argparse.Namespace(**(values | overrides))


@pytest.fixture(autouse=True)
def disable_api(monkeypatch):
    for key in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]:
        monkeypatch.delenv(key, raising=False)


def test_task_infers_arbitrary_target_and_dictionary_category(tmp_path):
    task = load_task(task_files(tmp_path, [1.5, 2.2, 4.7, 8.5]))
    assert task.target == "outcome"
    assert task.kind == "regression"
    assert task.metric == "rmse"
    assert task.features == ["amount", "segment"]
    assert task.categorical == ["segment"]


@pytest.mark.parametrize("labels", [[2, 9, 2, 9], ["normal", "attack", "normal", "attack"]])
def test_class_labels_roundtrip_submission_without_ordinal_replacement(tmp_path, labels):
    task = load_task(task_files(tmp_path, labels, "Multiclass classification. Macro F1."))
    classes = np.sort(np.unique(labels))
    write_submission(task, np.array([[.8, .2], [.1, .9]]), classes)
    output = pd.read_csv(task.output)
    assert output.columns.tolist() == ["key", "outcome"]
    assert output.key.tolist() == [101, 205]
    assert output.outcome.tolist() == classes.tolist()


def test_sample_order_mismatch_is_rejected(tmp_path):
    path = task_files(tmp_path, [1., 2., 3., 4.])
    pd.DataFrame({"key": [205, 101], "outcome": [0, 0]}).to_csv(tmp_path / "sample_submission.csv", index=False)
    with pytest.raises(ValueError, match="order"):
        load_task(path)


def test_frequency_and_imputation_fit_ignore_holdout_distribution():
    training = pd.DataFrame({"cat": ["a", "a", "b"], "value": [1., 3., np.nan]})
    holdout = pd.DataFrame({"cat": ["holdout", "a"], "value": [np.nan, 100000.]})
    builder = FeatureBuilder(extended=True).fit(training)
    transformed = builder.transform(holdout)
    assert transformed["__freq_cat"].tolist() == [0, 2 / 3]
    encoder = NumericEncoder().fit(builder.transform(training))
    result = encoder.transform(transformed)
    assert result[0, transformed.columns.get_loc("cat")] == -1
    assert result[0, transformed.columns.get_loc("value")] == 2


def test_feature_operations_handle_zero_infinity_and_negative_inputs():
    frame = pd.DataFrame({"left": [-2., 0., np.inf], "right": [0., 2., 1.]})
    builder = FeatureBuilder(operations=[{"op": "ratio", "left": "left", "right": "right"}, {"op": "log1p", "left": "left"}]).fit(frame)
    transformed = builder.transform(frame)
    assert np.isnan(transformed["__research_0"].iloc[0])
    assert np.isnan(transformed["__research_0"].iloc[2])
    assert transformed["__research_1"].iloc[0] == pytest.approx(-np.log(3))


def test_classification_splits_keep_singletons_and_all_classes_in_fit():
    y = np.array([0] * 10 + [1] * 4 + [2])
    x = pd.DataFrame({"feature": np.arange(len(y))})
    fit, cal, valid, scheme = make_splits(x, y, "classification", 42, .2)
    assert set(y[fit]) == {0, 1, 2}
    assert 14 in fit
    assert not (set(fit) & set(cal) or set(fit) & set(valid) or set(cal) & set(valid))
    assert sorted(np.concatenate([fit, cal, valid])) == list(range(len(y)))
    assert scheme == "stratified"


def test_duplicate_rows_never_cross_fit_calibration_and_validation():
    x = pd.DataFrame({"feature": np.repeat(np.arange(25), 3)})
    y = np.repeat(np.arange(25) * 1.2, 3)
    fit, cal, valid, scheme = make_splits(x, y, "regression", 42, .2)
    groups = [set(x.iloc[index].feature) for index in [fit, cal, valid]]
    assert scheme == "duplicate_groups"
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_time_split_obeys_chronology():
    x = pd.DataFrame({"event_time": np.arange(100)[::-1]})
    fit, cal, valid, scheme = make_splits(x, np.arange(100), "regression", 42, .2, time_column="event_time")
    assert x.iloc[fit].event_time.max() < x.iloc[cal].event_time.min()
    assert x.iloc[cal].event_time.max() < x.iloc[valid].event_time.min()
    assert scheme == "time"


def test_macro_f1_keeps_absent_validation_classes_in_denominator():
    assert evaluate("macro_f1", np.array([0, 0]), np.array([[1, 0], [1, 0]]), 2) == .5


def test_model_probability_columns_align_missing_estimator_classes():
    frame = pd.DataFrame({"value": [1., 2.]})
    class Estimator:
        classes_ = np.array([0, 2])
        def predict_proba(self, x):
            return np.tile([.2, .8], (len(x), 1))
    fitted = FittedModel(FeatureBuilder().fit(frame), None, Estimator(), "classification", 3, "none", np.ones(3))
    result = fitted.predict(frame)
    assert result.shape == (2, 3)
    assert result[:, 2] == pytest.approx([.8, .8])
    assert result.sum(axis=1) == pytest.approx([1., 1.])


def test_dummy_validation_mean_uses_only_fit_targets(tmp_path):
    y = np.array([1., 3., 7., 15., 31., 63., 127., 255., 511., 1023.])
    path = task_files(tmp_path, y)
    report = run(runner_args(path), time.monotonic())
    splits = np.load(Path(report["run_dir"]) / "splits.npz")
    expected = evaluate("rmse", y[splits["validation"]], np.full(len(splits["validation"]), y[splits["fit"]].mean()))
    assert report["dummy_validation_metric"] == pytest.approx(expected)


@pytest.mark.parametrize("metric", ["macro_f1", "rmse"])
def test_constant_target_writes_complete_fallback_without_training(tmp_path, metric):
    path = task_files(tmp_path, [7., 7., 7., 7.], "Classification. Macro F1." if metric == "macro_f1" else "Regression. RMSE.")
    report = run(runner_args(path), time.monotonic())
    assert report["status"] == "constant_target"
    assert pd.read_csv(tmp_path / "submission.csv").outcome.tolist() == [7., 7.]


@pytest.mark.parametrize("flag", ["--trial-timeout", "--memory-gb"])
@pytest.mark.parametrize("value", ["nan", "inf"])
def test_resource_arguments_reject_nonfinite_limits(monkeypatch, flag, value):
    spec = importlib.util.spec_from_file_location("reviewed_solution", Path(__file__).resolve().parents[1] / "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["solution.py", flag, value])
    with pytest.raises(SystemExit) as error:
        module.arguments()
    assert error.value.code == 2


def test_small_r2_run_preserves_submission_and_exits_cleanly(tmp_path):
    path = task_files(tmp_path, [1., 3., 7., 15.], "Regression. Metric R2.")
    report = run(runner_args(path), time.monotonic())
    assert pd.read_csv(tmp_path / "submission.csv").outcome.notna().all()
    json.dumps(report, allow_nan=False)
