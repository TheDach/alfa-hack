import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from research.engine import atomic_json, run_worker, write_submission
from research.task import load_task


def _model_key(value):
    # Reports may be inspected on a host using different path separators.
    return str(value).replace("\\", "/").rstrip("/").casefold()


def selected_refit_plan(report):
    """Recover the ensemble actually delivered, including OOF class calibration."""
    candidates = report.get("oof", {}).get("candidates", [])
    successful = [candidate for candidate in candidates if candidate.get("status") == "ok"]
    plan = []
    if successful:
        model_weights = {}
        for entry in report.get("models", []):
            weight = float(entry["weight"])
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Invalid selected OOF model weight")
            key = _model_key(entry["path"])
            model_weights[key] = model_weights.get(key, 0.) + weight
        accounted = set()
        for candidate in successful:
            keys = {_model_key(model["path"]) for model in candidate.get("models", [])}
            if accounted & keys:
                raise ValueError("OOF candidates share a model path")
            accounted |= keys
            weight = sum(model_weights.get(key, 0.) for key in keys)
            if weight <= 0:
                continue
            config = dict(candidate["config"])
            classes = report.get("classes", [])
            if classes and "class_weights" not in candidate:
                raise ValueError("Selected OOF candidate lacks its class calibration")
            calibration = candidate.get("class_weights", [])
            if (len(calibration) != len(classes) or
                    any(not math.isfinite(float(value)) or float(value) <= 0 for value in calibration)):
                raise ValueError("Invalid OOF class calibration")
            # Candidate config already contains the fixed iteration count used
            # for every OOF fold. Do not increase it a second time.
            if not isinstance(config.get("iterations"), int) or config["iterations"] < 1:
                raise ValueError("Selected OOF candidate lacks a valid iteration count")
            config["class_weights"] = [float(value) for value in calibration]
            plan.append({"source": "oof", "trial": candidate["source_trial"],
                         "weight": weight, "config": config})
        if any(weight > 0 and key not in accounted for key, weight in model_weights.items()):
            raise ValueError("Delivered model cannot be matched to an OOF candidate")
        if not plan:
            raise ValueError("No delivered OOF models found in report")
    else:
        history = {row["number"]: row for row in report["history"]}
        for item in report.get("selected", []):
            row = history[item["trial"]]
            config = dict(row["config"])
            config.update(iterations=max(30, int(row["iterations"] * 1.1)),
                          class_weights=row["class_weights"])
            plan.append({"source": "validation", "trial": item["trial"],
                         "weight": float(item["weight"]), "config": config})
    if not plan:
        raise ValueError("Report has no selected configurations to refit")
    total = sum(item["weight"] for item in plan)
    if (not math.isfinite(total) or total <= 0 or
            any(not math.isfinite(item["weight"]) or item["weight"] < 0 for item in plan)):
        raise ValueError("Invalid selected ensemble weights")
    if not math.isclose(total, 1., rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Selected ensemble weights do not sum to one")
    for item in plan:
        item["weight"] /= total
    return plan


def refit(task_path, report_path, time_budget=600., threads=8, memory_gb=27.):
    if (not math.isfinite(time_budget) or time_budget <= 0 or threads < 1 or
            not math.isfinite(memory_gb) or not 0 < memory_gb <= 29):
        raise ValueError("Invalid refit resource limits")
    task = load_task(task_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if (task.target, task.kind, task.metric) != (report["target"], report["kind"], report["metric_name"]):
        raise ValueError("Report belongs to another task")
    label_encoder = LabelEncoder().fit(task.train[task.target]) if task.kind == "classification" else None
    classes = label_encoder.classes_ if label_encoder else np.array([])
    if classes.tolist() != report["classes"]:
        raise ValueError("Report class schema differs")
    selected = selected_refit_plan(report)
    y = label_encoder.transform(task.train[task.target]) if label_encoder else task.train[task.target].to_numpy(dtype=float)
    folder = task.path.parent / "refit" / (time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    folder.mkdir(parents=True)
    bundle = {"train": task.train[task.features], "test": task.test[task.features], "y": y,
              "fit_indices": np.arange(len(y)), "cal_indices": np.array([], dtype=int),
              "validation_indices": np.array([], dtype=int), "categorical": task.categorical,
              "kind": task.kind, "metric": task.metric, "n_classes": len(classes),
              "threads": min(threads, 8), "seed": report["seed"]}
    bundle_path = folder / "data.joblib"
    joblib.dump(bundle, bundle_path)
    deadline = time.monotonic() + time_budget
    predictions, models = [], []
    try:
        for index, item in enumerate(selected):
            remaining = deadline - time.monotonic()
            if remaining <= 1:
                raise TimeoutError("Refit time budget exhausted; previous submission retained")
            config = item["config"]
            destination = folder / f"final_{index}"
            result = run_worker(bundle_path, config, destination, remaining / (len(selected) - index),
                                True, task.path.parent, memory_gb)
            if result["status"] != "ok":
                raise RuntimeError(result)
            with np.load(destination / "predictions.npz") as values:
                predictions.append(item["weight"] * values["test"])
            models.append({"path": str(destination / "model.joblib"), "weight": item["weight"]})
            print(f"refit {index + 1}/{len(selected)}: {config['family']}, {result['elapsed_seconds']:.1f}s",
                  flush=True)
        write_submission(task, np.sum(predictions, axis=0), classes)
        atomic_json(folder / "model_manifest.json",
                    {"models": models, "classes": classes.tolist(), "features": task.features,
                     "kind": task.kind, "metric": task.metric})
        result = {"source_report": str(Path(report_path).resolve()), "output": str(task.output),
                  "training_rows": len(y), "test_rows": len(task.test), "models": models,
                  "selected_refit_plan": selected}
        atomic_json(folder / "refit.json", result)
    finally:
        bundle_path.unlink(missing_ok=True)
    print(task.output, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Refit the selected validation or OOF ensemble on all public training rows")
    parser.add_argument("task")
    parser.add_argument("report")
    parser.add_argument("--time-budget", type=float, default=600)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-gb", type=float, default=27)
    args = parser.parse_args()
    refit(args.task, args.report, args.time_budget, args.threads, args.memory_gb)


if __name__ == "__main__":
    main()
