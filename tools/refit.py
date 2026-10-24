import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description="Refit configurations selected on a local training subset")
    parser.add_argument("task")
    parser.add_argument("report")
    parser.add_argument("--time-budget", type=float, default=600)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    task = load_task(args.task)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if (task.target, task.kind, task.metric) != (report["target"], report["kind"], report["metric_name"]):
        raise ValueError("Report belongs to another task")
    label_encoder = LabelEncoder().fit(task.train[task.target]) if task.kind == "classification" else None
    classes = label_encoder.classes_ if label_encoder else np.array([])
    if classes.tolist() != report["classes"]:
        raise ValueError("Report class schema differs")
    y = label_encoder.transform(task.train[task.target]) if label_encoder else task.train[task.target].to_numpy(dtype=float)
    folder = task.path.parent / "refit" / (time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    folder.mkdir(parents=True)
    bundle = {"train": task.train[task.features], "test": task.test[task.features], "y": y,
              "fit_indices": np.arange(len(y)), "cal_indices": np.array([], dtype=int),
              "validation_indices": np.array([], dtype=int), "categorical": task.categorical,
              "kind": task.kind, "metric": task.metric, "n_classes": len(classes),
              "threads": min(args.threads, 8), "seed": report["seed"]}
    bundle_path = folder / "data.joblib"
    joblib.dump(bundle, bundle_path)
    deadline = time.monotonic() + args.time_budget
    predictions, models = [], []
    selected = report["selected"]
    for index, item in enumerate(selected):
        row = next(row for row in report["history"] if row["number"] == item["trial"])
        config = dict(row["config"])
        config.update(iterations=max(30, int(row["iterations"] * 1.1)), class_weights=row["class_weights"])
        destination = folder / f"final_{index}"
        result = run_worker(bundle_path, config, destination, max(1, (deadline - time.monotonic()) / (len(selected) - index)),
                            True, task.path.parent, 27)
        if result["status"] != "ok":
            raise RuntimeError(result)
        with np.load(destination / "predictions.npz") as values:
            predictions.append(item["weight"] * values["test"])
        models.append({"path": str(destination / "model.joblib"), "weight": item["weight"]})
        print(f"refit {index + 1}/{len(selected)}: {config['family']}, {result['elapsed_seconds']:.1f}s", flush=True)
    write_submission(task, np.sum(predictions, axis=0), classes)
    atomic_json(folder / "model_manifest.json", {"models": models, "classes": classes.tolist(),
                                               "features": task.features, "kind": task.kind, "metric": task.metric})
    atomic_json(folder / "refit.json", {"source_report": str(Path(args.report).resolve()), "output": str(task.output),
                                       "training_rows": len(y), "test_rows": len(task.test), "models": models})
    bundle_path.unlink()
    print(task.output, flush=True)


if __name__ == "__main__":
    main()
