import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from research.engine import atomic_json, split_indices
from research.features import FeatureBuilder, NumericEncoder
from research.metrics import error_summary, evaluate
from research.scoring import leaderboard_score
from research.task import load_task


def benchmark_one(name, args):
    task = load_task(ROOT / "data" / name / "task.md")
    folder = Path(args.output).resolve() / name
    folder.mkdir(parents=True, exist_ok=True)
    labels = LabelEncoder().fit_transform(task.train[task.target]) if task.kind == "classification" else task.train[task.target].to_numpy()
    train_idx, holdout_idx = split_indices(labels, task.kind, .2, args.seed)
    train = task.train.iloc[train_idx].reset_index(drop=True)
    holdout = task.train.iloc[holdout_idx].reset_index(drop=True)
    train.to_csv(folder / "train.csv", index=False)
    holdout.drop(columns=[task.target]).to_csv(folder / "test.csv", index=False)
    template = holdout[task.sample.columns].copy()
    template[task.target] = 0 if task.kind == "regression" else train[task.target].iloc[0]
    template.to_csv(folder / "sample_submission.csv", index=False)
    task.dictionary.to_csv(folder / "data_dictionary.csv", index=False)
    (folder / "task.md").write_text(task.description, encoding="utf-8")
    atomic_json(folder / "outer_split.json", {"seed": args.seed, "source_rows": len(task.train),
                                              "train_indices": train_idx.tolist(), "holdout_indices": holdout_idx.tolist(),
                                              "source_sha256": hashlib.sha256(task.train.to_csv(index=False).encode()).hexdigest()})
    pd.DataFrame({task.target: holdout[task.target]}).to_csv(folder / "holdout_truth.csv", index=False)
    x = train[task.features]
    builder = FeatureBuilder(task.categorical).fit(x)
    tx = builder.transform(x)
    encoder = NumericEncoder().fit(tx)
    label_encoder = LabelEncoder().fit(train[task.target]) if task.kind == "classification" else None
    y_train = label_encoder.transform(train[task.target]) if label_encoder else train[task.target].to_numpy()
    cls = RandomForestClassifier if label_encoder else RandomForestRegressor
    model = cls(n_estimators=200, max_depth=12, min_samples_leaf=2, n_jobs=args.threads, random_state=42)
    start = time.monotonic()
    model.fit(encoder.transform(tx), y_train)
    hx = encoder.transform(builder.transform(holdout[task.features]))
    baseline = model.predict_proba(hx) if label_encoder else model.predict(hx)
    truth = label_encoder.transform(holdout[task.target]) if label_encoder else holdout[task.target].to_numpy()
    base_value = evaluate(task.metric, truth, baseline, len(label_encoder.classes_) if label_encoder else 0)
    print(f"{name} reference {task.metric}={base_value:.8f}", flush=True)
    baseline_seconds = time.monotonic() - start
    command = [sys.executable, str(ROOT / "solution.py"), str(folder / "task.md"),
               "--time-budget", str(args.time_budget), "--max-trials", str(args.max_trials),
               "--threads", str(args.threads), "--trial-timeout", str(args.trial_timeout),
               "--offline",
               "--work-dir", str(folder / "runs")]
    start = time.monotonic()
    with (folder / "agent.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.time_budget + 60)
    if result.returncode:
        raise RuntimeError((folder / "agent.log").read_text(encoding="utf-8")[-6000:])
    prediction = pd.read_csv(folder / "submission.csv")
    if not all(np.array_equal(prediction[c], holdout[c]) for c in task.identifiers):
        raise AssertionError("Submission identifiers differ from holdout")
    predicted = label_encoder.transform(prediction[task.target]) if label_encoder else prediction[task.target].to_numpy()
    value = evaluate(task.metric, truth, predicted, len(label_encoder.classes_) if label_encoder else 0)
    latest = json.loads((folder / "runs" / "latest.json").read_text())
    report = json.loads(Path(latest["report"]).read_text(encoding="utf-8"))
    probabilities = np.eye(len(label_encoder.classes_))[predicted] if label_encoder else predicted
    summary = {"task": name, "metric_name": task.metric, "reference_metric": base_value, "agent_metric": value,
               "reference_model": "RandomForest(n_estimators=200,max_depth=12,min_samples_leaf=2,random_state=42)",
               "reference_seconds": baseline_seconds, "agent_seconds": time.monotonic() - start,
               "train_rows": len(train), "holdout_rows": len(holdout), "seed": args.seed,
               "status": report["status"], "report": latest["report"],
               "trials": len(report["history"]), "llm_enabled": report["llm_enabled"],
               "diagnostics": error_summary(task.kind, truth, probabilities, len(label_encoder.classes_) if label_encoder else 0)}
    atomic_json(folder / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["regression", "classification"])
    parser.add_argument("--output", default=str(ROOT / "reports" / "benchmark"))
    parser.add_argument("--time-budget", type=float, default=600)
    parser.add_argument("--max-trials", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--trial-timeout", type=float, default=180)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    results = {name: benchmark_one(name, args) for name in args.tasks}
    if {"regression", "classification"} <= set(results):
        score = leaderboard_score(results["regression"]["agent_metric"], results["regression"]["reference_metric"],
                                  results["classification"]["agent_metric"], results["classification"]["reference_metric"])
        combined = {"evaluation": "local holdout; surrogate reference, not organizer baseline", "tasks": results,
                    "points": score, "official_score": None,
                    "limitations": ["Official hidden test targets and baseline metric values are unavailable.",
                                    "LLM integration requires organizer-provided Qwen credentials."]}
        atomic_json(Path(args.output) / "summary.json", combined)
        print(json.dumps(combined, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
