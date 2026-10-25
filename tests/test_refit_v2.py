from pathlib import Path
import importlib.util
import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("refit_under_test", ROOT / "tools" / "refit.py")
refit_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refit_tool)


def oof_report():
    a = [f"D:/old/bagged_0/fold_{index}/model.joblib" for index in range(3)]
    b = [f"D:/old/bagged_1/fold_{index}/model.joblib" for index in range(3)]
    inactive = ["D:/old/bagged_2/fold_0/model.joblib"]
    return {
        "target": "outcome", "kind": "classification", "metric_name": "macro_f1",
        "classes": ["no", "yes"], "seed": 42,
        "history": [{"number": 0, "config": {"family": "linear", "iterations": 900},
                     "iterations": 41, "class_weights": [5., 1.]}],
        # Deliberately disagrees with the final OOF ensemble.
        "selected": [{"trial": 0, "weight": 1.}],
        "models": [{"path": path.replace("/", "\\").upper() if index == 0 else path,
                    "weight": .2} for index, path in enumerate(a)] +
                  [{"path": path, "weight": .4 / 3} for path in b],
        "oof": {"candidates": [
            {"status": "ok", "source_trial": 1,
             "config": {"family": "lightgbm", "iterations": 99, "class_weights": [1., 1.]},
             "class_weights": [.7, 1.3], "models": [{"path": path} for path in a]},
            {"status": "ok", "source_trial": 2,
             "config": {"family": "xgboost", "iterations": 133, "class_weights": [1., 1.]},
             "class_weights": [1.1, .9], "models": [{"path": path} for path in b]},
            {"status": "ok", "source_trial": 3, "config": {"family": "catboost", "iterations": 999},
             "class_weights": [1., 1.], "models": [{"path": path} for path in inactive]},
        ]},
    }


def task_files(tmp_path):
    pd.DataFrame({"id": np.arange(12), "signal": np.arange(12) * .1,
                  "outcome": ["no", "yes"] * 6}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "signal": [.3, .8]}).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "outcome": ["", ""]}).to_csv(tmp_path / "sample_submission.csv", index=False)
    pd.DataFrame({"feature": ["id", "signal", "outcome"], "role": ["id", "feature", "target"],
                  "dtype": ["numeric", "numeric", "category"]}).to_csv(tmp_path / "data_dictionary.csv", index=False)
    task = tmp_path / "task.md"
    task.write_text("Classification. Primary metric: Macro F1.", encoding="utf8")
    report_path = tmp_path / "source_report.json"
    report_path.write_text(json.dumps(oof_report()), encoding="utf8")
    return task, report_path


def test_oof_refit_uses_delivered_ensemble_and_calibration_not_initial_selected():
    plan = refit_tool.selected_refit_plan(oof_report())
    assert [item["trial"] for item in plan] == [1, 2]
    assert [item["weight"] for item in plan] == pytest.approx([.6, .4])
    assert [item["config"]["family"] for item in plan] == ["lightgbm", "xgboost"]
    assert [item["config"]["iterations"] for item in plan] == [99, 133]
    assert [item["config"]["class_weights"] for item in plan] == [[.7, 1.3], [1.1, .9]]
    assert all(item["source"] == "oof" for item in plan)


def test_temporal_or_legacy_selection_retains_existing_refit_behavior():
    report = oof_report()
    report["oof"] = {"status": "skipped", "reason": "temporal_validation"}
    plan = refit_tool.selected_refit_plan(report)
    assert len(plan) == 1
    assert plan[0]["source"] == "validation"
    assert plan[0]["trial"] == 0
    assert plan[0]["weight"] == 1.
    assert plan[0]["config"] == {"family": "linear", "iterations": 45, "class_weights": [5., 1.]}


@pytest.mark.parametrize("missing", ["calibration", "model_path"])
def test_oof_refit_rejects_missing_final_selection_evidence(missing):
    report = oof_report()
    if missing == "calibration":
        del report["oof"]["candidates"][0]["class_weights"]
        match = "calibration"
    else:
        report["models"].append({"path": "D:/unknown/model.joblib", "weight": .1})
        match = "cannot be matched"
    with pytest.raises(ValueError, match=match):
        refit_tool.selected_refit_plan(report)


def test_all_training_rows_and_oof_weights_reach_full_refit_workers(tmp_path, monkeypatch):
    task, report_path = task_files(tmp_path)
    calls = []
    def worker(bundle_path, config, destination, timeout, final, cwd, memory_gb):
        bundle = joblib.load(bundle_path)
        assert final is True
        assert len(bundle["train"]) == 12
        assert len(bundle["fit_indices"]) == 12
        assert len(bundle["cal_indices"]) == 0
        assert len(bundle["validation_indices"]) == 0
        assert memory_gb == 1.25
        assert 0 < timeout <= 30
        assert config["class_weights"] == ([.7, 1.3] if config["family"] == "lightgbm" else [1.1, .9])
        destination.mkdir(parents=True)
        prediction = ([[.8, .2], [.1, .9]] if config["family"] == "lightgbm" else [[.3, .7], [.9, .1]])
        np.savez_compressed(destination / "predictions.npz", test=prediction)
        (destination / "model.joblib").write_bytes(b"mock model")
        calls.append(dict(config))
        return {"status": "ok", "elapsed_seconds": .01}
    monkeypatch.setattr(refit_tool, "run_worker", worker)
    result = refit_tool.refit(task, report_path, time_budget=30, threads=1, memory_gb=1.25)
    assert [config["iterations"] for config in calls] == [99, 133]
    assert result["training_rows"] == 12
    assert [entry["weight"] for entry in result["models"]] == pytest.approx([.6, .4])
    output = pd.read_csv(tmp_path / "submission.csv")
    assert output.outcome.tolist() == ["no", "yes"]
    assert output.id.tolist() == [100, 101]
    manifests = list((tmp_path / "refit").glob("*/model_manifest.json"))
    assert len(manifests) == 1
    saved = json.loads(manifests[0].read_text(encoding="utf8"))
    assert [entry["weight"] for entry in saved["models"]] == pytest.approx([.6, .4])
    assert not list((tmp_path / "refit").glob("*/data.joblib"))


def test_failed_later_refit_keeps_existing_submission_and_cleans_training_bundle(tmp_path, monkeypatch):
    task, report_path = task_files(tmp_path)
    old_output = b"id,outcome\n100,yes\n101,no\n"
    (tmp_path / "submission.csv").write_bytes(old_output)
    calls = []
    def worker(bundle_path, config, destination, timeout, final, cwd, memory_gb):
        calls.append(config["family"])
        destination.mkdir(parents=True)
        if len(calls) == 2:
            return {"status": "memory_limit", "elapsed_seconds": .01}
        np.savez_compressed(destination / "predictions.npz", test=[[.8, .2], [.1, .9]])
        return {"status": "ok", "elapsed_seconds": .01}
    monkeypatch.setattr(refit_tool, "run_worker", worker)
    with pytest.raises(RuntimeError, match="memory_limit"):
        refit_tool.refit(task, report_path, time_budget=30, threads=1, memory_gb=1)
    assert (tmp_path / "submission.csv").read_bytes() == old_output
    assert not list((tmp_path / "refit").glob("*/data.joblib"))
    assert not list((tmp_path / "refit").glob("*/model_manifest.json"))

