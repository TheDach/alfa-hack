import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from research.metrics import PROBABILITY, calibrate_weights, evaluate, utility
from research.planner import QwenPlanner
from research.search import candidates, coverage_pending
from research.task import load_task, profile_task, validation_columns


def atomic_json(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, allow_nan=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def split_indices(y, kind, fraction, seed):
    rng = np.random.default_rng(seed)
    if kind == "classification":
        train, valid = [], []
        for cls in np.unique(y):
            indices = np.flatnonzero(y == cls)
            rng.shuffle(indices)
            n = min(len(indices) - 1, max(1, int(round(len(indices) * fraction)))) if len(indices) > 1 else 0
            valid.extend(indices[:n])
            train.extend(indices[n:])
        if not valid:
            raise ValueError("Validation requires a repeated class")
        return np.asarray(train, dtype=int), np.asarray(valid, dtype=int)
    indices = rng.permutation(len(y))
    n = min(len(y) - 1, max(1, int(round(len(y) * fraction))))
    return indices[n:], indices[:n]


def make_splits(x, y, kind, seed, validation, group=None, time_column=None):
    if group:
        if group not in x:
            raise ValueError(f"Unknown group column: {group}")
        fit, valid = next(GroupShuffleSplit(n_splits=1, test_size=validation, random_state=seed).split(x, y, x[group].fillna("__missing__")))
        local, calibration = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed + 1).split(x.iloc[fit], y[fit], x.iloc[fit][group].fillna("__missing__")))
        return fit[local], fit[calibration], valid, "group"
    if time_column:
        if time_column not in x or x[time_column].isna().any():
            raise ValueError("Time split needs a complete time column")
        values = x[time_column].to_numpy()
        levels = np.unique(values)
        if 3 <= len(levels) <= 30:
            valid_count = max(1, min(len(levels) - 2, int(round(len(levels) * validation))))
            cal_count = max(1, int(round((len(levels) - valid_count) * .15)))
            fit_levels = levels[:len(levels) - valid_count - cal_count]
            cal_levels = levels[len(fit_levels):len(levels) - valid_count]
            valid_levels = levels[-valid_count:]
            return (np.flatnonzero(np.isin(values, fit_levels)),
                    np.flatnonzero(np.isin(values, cal_levels)),
                    np.flatnonzero(np.isin(values, valid_levels)), "time")
        order = np.argsort(values, kind="stable")
        cut = max(1, int(len(y) * (1 - validation)))
        cal_cut = max(1, int(cut * .85))
        return order[:cal_cut], order[cal_cut:cut], order[cut:], "time"
    hashes = pd.util.hash_pandas_object(x, index=False).to_numpy()
    if len(np.unique(hashes)) < .98 * len(hashes) and len(np.unique(hashes)) >= 10:
        fit, valid = next(GroupShuffleSplit(n_splits=1, test_size=validation, random_state=seed).split(x, y, hashes))
        local, cal = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed + 1).split(x.iloc[fit], y[fit], hashes[fit]))
        if kind == "regression" or len(np.unique(y[fit[local]])) == len(np.unique(y)):
            return fit[local], fit[cal], valid, "duplicate_groups"
    fit, valid = split_indices(y, kind, validation, seed)
    try:
        local, cal = split_indices(y[fit], kind, .15, seed + 1)
    except ValueError:
        local, cal = np.arange(len(fit)), np.array([], dtype=int)
    return fit[local], fit[cal], valid, "stratified" if kind == "classification" else "random"


def detect_forward_time(train, test):
    """Infer only a clear forward shift of an explicitly temporal discrete feature."""
    candidates = []
    for column in train:
        if not re.search(r"time|year|date|period|factor", str(column), re.IGNORECASE):
            continue
        if not pd.api.types.is_numeric_dtype(train[column]) or not pd.api.types.is_numeric_dtype(test[column]):
            continue
        left = train[column].dropna()
        right = test[column].dropna()
        if len(left) != len(train) or len(right) != len(test) or not len(right):
            continue
        levels = left.nunique()
        if 3 <= levels <= 30 and right.nunique() <= 5 and right.min() > left.max():
            candidates.append((levels, column))
    return max(candidates)[1] if candidates else None


def ensemble_weights(metric, y, predictions, n_classes=0, max_models=4):
    """Greedy convex averaging, selected exclusively with supplied validation labels."""
    if not predictions:
        raise ValueError("Ensembling requires at least one prediction array")
    scores = [evaluate(metric, y, pred, n_classes) for pred in predictions]
    first = max(range(len(scores)), key=lambda index: utility(metric, scores[index]))
    weights = np.zeros(len(predictions))
    weights[first] = 1.
    current = predictions[first].copy()
    value = scores[first]
    for _ in range(max(0, max_models - 1)):
        choice = None
        for index, pred in enumerate(predictions):
            if weights[index] > 0:
                continue
            for fraction in [.1, .2, .35, .5]:
                mixed = (1 - fraction) * current + fraction * pred
                score = evaluate(metric, y, mixed, n_classes)
                if utility(metric, score) > utility(metric, value) + 1e-5:
                    if choice is None or utility(metric, score) > utility(metric, choice[0]):
                        choice = (score, index, fraction, mixed)
        if choice is None:
            break
        value, index, fraction, current = choice
        weights *= 1 - fraction
        weights[index] += fraction
    return weights, value


def final_reserve(time_budget, successful, training_rows, fit_rows, metric, temporal=False):
    base = max(15., time_budget * (.25 if temporal else (.35 if time_budget >= 300 else .25)))
    if not successful:
        return base
    # Three fold fits cost roughly 3*(2/3)/(fit_fraction) validation fits.
    best = sorted(successful, key=lambda row: utility(metric, row["metric"]), reverse=True)[:2]
    if temporal:
        # Trials already include early-stopping fit plus a larger calibration refit;
        # temporal finalization does not train three OOF folds.
        estimate = sum(row["elapsed_seconds"] * training_rows / fit_rows * .5 for row in best) + 15
        return max(base, min(time_budget * .45, estimate))
    estimate = sum(row["elapsed_seconds"] * training_rows / fit_rows * 2.2 for row in best) + 15
    return max(base, min(time_budget * .65, estimate))


def make_oof_splits(x, y, kind, seed, group=None, duplicate_groups=False):
    if group or duplicate_groups:
        groups = (x[group].fillna("__missing__").astype(str).to_numpy() if group
                  else pd.util.hash_pandas_object(x, index=False).to_numpy())
        count = min(3, len(np.unique(groups)))
        if count < 2:
            return [], "insufficient_groups"
        splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed) if kind == "classification" else GroupKFold(n_splits=count)
        splits = list(splitter.split(x, y, groups))
        scheme = "group_oof"
    elif kind == "classification":
        count = min(3, int(np.bincount(y).min()))
        if count < 2:
            return [], "singleton_class"
        splits = list(StratifiedKFold(n_splits=count, shuffle=True, random_state=seed).split(x, y))
        scheme = "stratified_oof"
    else:
        count = min(3, len(y))
        if count < 2:
            return [], "insufficient_rows"
        splits = list(KFold(n_splits=count, shuffle=True, random_state=seed).split(x))
        scheme = "random_oof"
    if kind == "classification" and any(len(np.unique(y[fit])) != len(np.unique(y)) for fit, _ in splits):
        return [], "group_split_removed_class"
    return splits, scheme


def calibrated_probabilities(prediction, weights):
    value = prediction * weights
    return value / np.maximum(value.sum(axis=1, keepdims=True), 1e-30)


def fit_oof_candidate(bundle, row, splits, folder, deadline, memory_gb, cwd):
    """Refit fixed iterations on each fold. A fold's labels are never used in its fit."""
    n_rows, n_test = len(bundle["y"]), len(bundle["test"])
    shape = (n_rows, bundle["n_classes"]) if bundle["kind"] == "classification" else (n_rows,)
    oof = np.full(shape, np.nan)
    test_predictions, models, fold_results = [], [], []
    config = dict(row["config"])
    config["iterations"] = max(20, int(row["iterations"] * 1.1))
    config["class_weights"] = np.ones(bundle["n_classes"]).tolist()
    for index, (fit, valid) in enumerate(splits):
        remaining = deadline - time.monotonic() - 5
        if remaining < 5:
            return {"status": "timeout", "folds": fold_results}
        fold_folder = folder / f"fold_{index}"
        fold_folder.mkdir(parents=True, exist_ok=True)
        folded = dict(bundle)
        folded.update(train=bundle["train"].iloc[fit].reset_index(drop=True), y=bundle["y"][fit],
                      test=pd.concat([bundle["train"].iloc[valid], bundle["test"]], ignore_index=True),
                      fit_indices=np.arange(len(fit)), cal_indices=np.array([], dtype=int),
                      validation_indices=np.array([], dtype=int), seed=bundle["seed"] + 17 * index)
        data_path = fold_folder / "fold_data.joblib"
        joblib.dump(folded, data_path, compress=0)
        del folded
        # Leave a fair share for every remaining fold; partial OOF is never selected.
        timeout = max(2., remaining / (len(splits) - index))
        result = run_worker(data_path, config, fold_folder, timeout, True, cwd, memory_gb)
        data_path.unlink(missing_ok=True)
        fold_results.append(result | {"fit_rows": len(fit), "validation_rows": len(valid)})
        if result["status"] != "ok":
            return {"status": result["status"], "folds": fold_results}
        with np.load(fold_folder / "predictions.npz") as saved:
            values = saved["test"].copy()
        if len(values) != len(valid) + n_test:
            return {"status": "failed", "error": "Invalid OOF prediction length", "folds": fold_results}
        oof[valid] = values[:len(valid)]
        test_predictions.append(values[len(valid):])
        models.append({"path": str(fold_folder / "model.joblib"), "weight": 1 / len(splits)})
    if not np.isfinite(oof).all():
        return {"status": "failed", "error": "OOF predictions do not cover training rows", "folds": fold_results}
    weights = np.ones(bundle["n_classes"])
    if bundle["kind"] == "classification" and bundle["metric"] == "macro_f1":
        weights = calibrate_weights(bundle["y"], oof, bundle["n_classes"])
        oof = calibrated_probabilities(oof, weights)
        test_predictions = [calibrated_probabilities(pred, weights) for pred in test_predictions]
        # Save the very same postprocessing in each model so the normal manifest
        # prediction path exactly reproduces the delivered CSV.
        if not np.allclose(weights, 1):
            for model in models:
                fitted = joblib.load(model["path"])
                fitted.weights = weights.copy()
                joblib.dump(fitted, model["path"], compress=1)
                del fitted
    test = np.mean(test_predictions, axis=0)
    np.savez_compressed(folder / "oof_predictions.npz", oof=oof, test=test)
    return {"status": "ok", "config": config, "metric": evaluate(bundle["metric"], bundle["y"], oof, bundle["n_classes"]),
            "class_weights": weights.tolist(), "models": models, "folds": fold_results,
            "directory": str(folder), "source_trial": row["number"]}


def kill_process(process):
    import psutil
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
        psutil.wait_procs(children + [parent], timeout=3)
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=5)


def run_worker(bundle_path, config, folder, timeout, final, cwd, memory_gb):
    folder.mkdir(parents=True, exist_ok=True)
    spec_path = folder / "experiment.json"
    atomic_json(spec_path, {"bundle": str(bundle_path), "config": config, "output": str(folder), "final": final})
    source = ("from pathlib import Path\nimport sys\n"
              f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent if __package__ else Path(__file__).resolve().parent)!r})\n"
              "from research.worker import execute\n"
              f"execute({str(spec_path)!r})\n")
    script = folder / "experiment.py"
    script.write_text(source, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI_")}
    env["PYTHONIOENCODING"] = "utf-8"
    start = time.monotonic()
    status = "ok"
    with (folder / "process.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", str(script)], cwd=cwd, env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                   start_new_session=os.name != "nt")
        try:
            import psutil
            while process.poll() is None:
                if time.monotonic() - start >= timeout:
                    status = "timeout"
                    kill_process(process)
                    break
                try:
                    parent = psutil.Process(process.pid)
                    resident = sum(p.memory_info().rss for p in [parent] + parent.children(recursive=True))
                    if resident > memory_gb * 1024 ** 3:
                        status = "memory_limit"
                        kill_process(process)
                        break
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.25)
        except BaseException:
            kill_process(process)
            raise
    if process.returncode and status == "ok":
        status = "failed"
    if status != "ok":
        return {"status": status, "elapsed_seconds": time.monotonic() - start,
                "error": (folder / "process.log").read_text(encoding="utf-8")[-4000:]}
    result_path = folder / "result.json"
    if not result_path.exists():
        return {"status": "failed", "error": "Missing worker result", "elapsed_seconds": time.monotonic() - start}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(status="ok", elapsed_seconds=time.monotonic() - start)
    return result


def write_submission(task, predictions, classes):
    result = task.sample.copy()
    for col in task.identifiers:
        result[col] = task.test[col].to_numpy()
    if task.kind == "classification":
        if task.metric in PROBABILITY:
            if len(classes) != 2:
                raise ValueError("Probability submission currently requires two classes")
            values = predictions[:, 1]
        else:
            values = classes[np.argmax(predictions, axis=1)]
    else:
        values = np.asarray(predictions).reshape(-1)
        if task.metric == "rmsle":
            values = np.maximum(values, 0)
    if len(values) != len(task.test) or pd.isna(values).any():
        raise ValueError("Invalid submission length or missing predictions")
    if pd.api.types.is_numeric_dtype(values) and not np.isfinite(values).all():
        raise ValueError("Nonfinite submission")
    result[task.target] = values
    task.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = task.output.with_suffix(".csv.tmp")
    result.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(task.output)


def run(args, started=None):
    started = started or time.monotonic()
    deadline = started + args.time_budget
    task = load_task(args.task, args.target, args.task_type, args.metric, args.output)
    profile = profile_task(task)
    x = task.train[task.features]
    encoder = LabelEncoder().fit(task.train[task.target]) if task.kind == "classification" else None
    y = encoder.transform(task.train[task.target]) if encoder else task.train[task.target].to_numpy(dtype=float)
    classes = encoder.classes_ if encoder else np.array([])
    if encoder and task.metric in PROBABILITY and len(classes) != 2:
        raise ValueError("Multiclass probability CSV needs an explicit per-class submission schema")
    if encoder:
        prior = np.bincount(y, minlength=len(classes)) / len(y)
        fallback = np.tile(prior, (len(task.test), 1))
    else:
        constant = np.median(y) if task.metric in {"mae", "mape"} else np.mean(y)
        fallback = np.full(len(task.test), constant)
    write_submission(task, fallback, classes)
    run_root = Path(args.work_dir).resolve() if args.work_dir else task.path.parent / "research_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    folder = run_root / (time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    folder.mkdir()
    planner = QwenPlanner()
    report = {"task": str(task.path), "kind": task.kind, "metric_name": task.metric, "target": task.target,
              "output": str(task.output), "classes": classes.tolist(), "profile": profile,
              "history": [], "status": "fallback", "llm_enabled": planner.enabled,
              "llm_disabled_reason": planner.disabled_reason, "seed": args.seed,
              "time_budget_seconds": args.time_budget, "run_dir": str(folder)}
    def save():
        report.update(elapsed_seconds=round(time.monotonic() - started, 3), llm_usage=planner.usage, llm_events=planner.events, llm_enabled=planner.enabled, llm_disabled_reason=planner.disabled_reason)
        atomic_json(folder / "report.json", report)
        atomic_json(run_root / "latest.json", {"report": str(folder / "report.json")})
    save()
    if np.unique(y).size == 1:
        report["status"] = "constant_target"
        save()
        return report
    if task.metric == "r2" and len(y) < 4:
        report.update(status="insufficient_validation", reason="R2 requires at least two independent validation rows")
        save()
        return report
    validation_fraction = max(args.validation_fraction, 2 / len(y)) if task.metric == "r2" else args.validation_fraction
    automatic_group, automatic_time = validation_columns(task)
    group_column, time_column = args.group_column, args.time_column
    if not group_column and not time_column:
        group_column, time_column = automatic_group, automatic_time
        if not group_column and not time_column:
            time_column = detect_forward_time(x, task.test[task.features])
            if time_column:
                report["automatic_time_column"] = time_column
    if group_column and time_column:
        raise ValueError("Task requests incompatible group and time validation schemes")
    try:
        fit, cal, valid, scheme = make_splits(x, y, task.kind, args.seed, validation_fraction, group_column, time_column)
    except ValueError as error:
        if task.kind == "classification" and len(y) <= len(classes) + 1:
            report.update(status="insufficient_validation", reason=str(error))
            save()
            return report
        raise
    if not len(fit) or not len(valid):
        raise ValueError("Empty training or validation split")
    if encoder and len(np.unique(y[fit])) != len(classes):
        raise ValueError("Validation scheme removed a class from training")
    np.savez_compressed(folder / "splits.npz", fit=fit, calibration=cal, validation=valid)
    report["validation"] = {"scheme": scheme, "fit_rows": len(fit), "calibration_rows": len(cal), "validation_rows": len(valid), "refit_before_validation": scheme == "time",
                            "selection_training_rows": len(fit) + (len(cal) if scheme == "time" else 0)}
    bundle = {"train": x, "test": task.test[task.features], "y": y, "fit_indices": fit,
              "cal_indices": cal, "validation_indices": valid, "categorical": task.categorical,
              "kind": task.kind, "metric": task.metric, "n_classes": len(classes),
              "threads": args.threads, "seed": args.seed, "refit_before_validation": scheme == "time"}
    bundle_path = folder / "data.joblib"
    joblib.dump(bundle, bundle_path, compress=0)
    validation_constant = np.median(y[fit]) if task.metric in {"mae", "mape"} else np.mean(y[fit])
    dummy_valid = np.tile(np.bincount(y[fit], minlength=len(classes)) / len(fit), (len(valid), 1)) if encoder else np.full(len(valid), validation_constant)
    if task.metric in {"roc_auc", "average_precision"} and len(np.unique(y[valid])) != len(classes):
        report.update(status="insufficient_validation", reason="Probability metric requires every class in validation")
        save()
        bundle_path.unlink(missing_ok=True)
        return report
    best_value = evaluate(task.metric, y[valid], dummy_valid, len(classes))
    report["dummy_validation_metric"] = best_value
    best = None
    successful = []
    trial_limit = min(args.max_trials, 16) if task.kind == "classification" else args.max_trials
    for number in range(trial_limit):
        remaining = deadline - time.monotonic()
        reserve = final_reserve(args.time_budget, successful, len(y), len(fit), task.metric, temporal=scheme == "time")
        if remaining <= reserve + 5:
            break
        allowed = candidates(profile, report["history"])
        if not allowed:
            break
        pending = coverage_pending(report["history"], allowed, profile)
        menu = allowed[:1] if pending else allowed
        if not pending and len(report["history"]) >= 2:
            recent = [row["config"]["family"] for row in report["history"][-2:]]
            counts = {family: sum(row["config"]["family"] == family for row in report["history"])
                      for family in {config["family"] for config in menu}}
            capped = [config for config in menu if counts[config["family"]] < ((12 if config["family"] == "lightgbm" else 6) if task.kind == "regression" else 5) and
                      not (recent[0] == recent[1] == config["family"])]
            if capped:
                menu = capped
        config = menu[0]
        if number > 0 and planner.enabled and remaining > reserve + 30:
            proposal = planner.propose(profile, report["history"], menu, remaining - reserve)
            if proposal:
                config = proposal
        trial_dir = folder / f"trial_{number:02d}"
        timeout = min(args.trial_timeout, deadline - time.monotonic() - reserve)
        if timeout <= 2:
            break
        print(f"experiment {number + 1}: {config['family']}, {config['hypothesis']}", flush=True)
        row = run_worker(bundle_path, config, trial_dir, timeout, False, task.path.parent, args.memory_gb)
        row.update(config=config, directory=str(trial_dir), number=number)
        report["history"].append(row)
        if row["status"] == "ok":
            successful.append(row)
            print(f"{task.metric}={row['metric']:.7f}, seconds={row['elapsed_seconds']:.1f}", flush=True)
            if utility(task.metric, row["metric"]) > utility(task.metric, best_value):
                best, best_value = row, row["metric"]
                with np.load(trial_dir / "predictions.npz") as data:
                    write_submission(task, data["test"], classes)
                report.update(status="validated_checkpoint", best_validation_metric=best_value, best_config=config)
        else:
            print(f"experiment {number + 1}: {row['status']}", flush=True)
        save()
    if best:
        arrays = []
        for row in successful:
            with np.load(Path(row["directory"]) / "predictions.npz") as data:
                arrays.append(data["validation"].copy())
        mix_weights, blend_value = ensemble_weights(task.metric, y[valid], arrays, len(classes))
        selected = [(row, float(weight)) for row, weight in zip(successful, mix_weights) if weight > 0]
        checkpoint = []
        for row, weight in selected:
            with np.load(Path(row["directory"]) / "predictions.npz") as data:
                checkpoint.append(weight * data["test"])
        write_submission(task, np.sum(checkpoint, axis=0), classes)
        report.update(best_validation_metric=blend_value,
                      selected=[{"trial": row["number"], "weight": weight} for row, weight in selected])
        save()

        # Fixed model configurations were selected only on the initial validation.
        # Fresh OOF predictions use all labeled rows without using held-out labels in fit.
        oof_results = []
        oof_splits, oof_scheme = ([], "temporal_validation") if scheme == "time" else make_oof_splits(
            x, y, task.kind, args.seed, group_column, bool(x.duplicated().any()))
        can_bag = args.time_budget >= 300 and bool(oof_splits)
        if can_bag:
            ranked = sorted(successful, key=lambda row: utility(task.metric, row["metric"]), reverse=True)
            ordered = [row for row, _ in sorted(selected, key=lambda item: item[1], reverse=True)] + ranked
            chosen = []
            for row in ordered:
                if row not in chosen:
                    chosen.append(row)
                if len(chosen) >= 2:
                    break
            report["oof"] = {"scheme": oof_scheme, "fold_count": len(oof_splits), "candidates": []}
            for number, row in enumerate(chosen):
                remaining = deadline - time.monotonic() - 5
                estimate = row["elapsed_seconds"] * len(y) / len(fit) * 2.2 + 10
                if remaining < max(15, estimate * .75):
                    report["oof"]["candidates"].append(
                        {"source_trial": row["number"], "status": "skipped", "reason": "insufficient_time"})
                    break
                try:
                    result = fit_oof_candidate(bundle, row, oof_splits, folder / f"bagged_{number}",
                                               deadline, args.memory_gb, task.path.parent)
                except Exception as error:
                    result = {"status": "failed", "source_trial": row["number"],
                              "error": type(error).__name__ + ": " + str(error)[:500]}
                report["oof"]["candidates"].append(result)
                save()
                if result["status"] != "ok":
                    continue
                oof_results.append(result)
                oof_arrays, test_arrays = [], []
                for fitted in oof_results:
                    with np.load(Path(fitted["directory"]) / "oof_predictions.npz") as saved:
                        oof_arrays.append(saved["oof"].copy())
                        test_arrays.append(saved["test"].copy())
                weights, score = ensemble_weights(task.metric, y, oof_arrays, len(classes), max_models=2)
                final_models, final_predictions = [], []
                for fitted, weight, prediction in zip(oof_results, weights, test_arrays):
                    if weight <= 0:
                        continue
                    final_predictions.append(weight * prediction)
                    final_models.extend({"path": model["path"], "weight": float(weight * model["weight"])}
                                        for model in fitted["models"])
                write_submission(task, np.sum(final_predictions, axis=0), classes)
                report.update(status="complete", models=final_models, oof_metric=score)
                atomic_json(folder / "model_manifest.json",
                            {"models": final_models, "classes": classes.tolist(), "features": task.features,
                             "kind": task.kind, "metric": task.metric, "training_scheme": oof_scheme})
                save()
        else:
            report["oof"] = {"scheme": oof_scheme, "status": "skipped",
                             "reason": "small_budget" if args.time_budget < 300 else oof_scheme}

        if not oof_results:
            fitted_groups = {}
            def commit_full_models():
                if len(fitted_groups) != len(selected):
                    return
                predictions, manifest_models = [], []
                for row, weight in selected:
                    group = fitted_groups[row["number"]]
                    predictions.append(weight * np.mean(group["predictions"], axis=0))
                    manifest_models.extend({"path": path, "weight": weight / len(group["models"])}
                                           for path in group["models"])
                write_submission(task, np.sum(predictions, axis=0), classes)
                report.update(status="complete", models=manifest_models)
                atomic_json(folder / "model_manifest.json",
                            {"models": manifest_models, "classes": classes.tolist(),
                             "features": task.features, "kind": task.kind, "metric": task.metric})
                save()

            for index, (row, weight) in enumerate(selected):
                remaining = deadline - time.monotonic() - 3
                estimate = row["elapsed_seconds"] * len(y) / len(fit) * 1.4 + 5
                if remaining < min(estimate, 15):
                    break
                config = dict(row["config"])
                config["iterations"] = max(20, int(row["iterations"] * 1.1))
                config["class_weights"] = row["class_weights"]
                final_dir = folder / f"final_{index}"
                result = run_worker(bundle_path, config, final_dir, remaining / (len(selected) - index), True,
                                    task.path.parent, args.memory_gb)
                report.setdefault("final_fits", []).append(result | {"config": config, "directory": str(final_dir)})
                if result["status"] != "ok":
                    break
                with np.load(final_dir / "predictions.npz") as data:
                    prediction = data["test"].copy()
                fitted_groups[row["number"]] = {"predictions": [prediction],
                                                "models": [str(final_dir / "model.joblib")]}
            commit_full_models()
            # Random folds are inappropriate under a forward shift. If all base
            # refits finished, reduce the strongest model's variance with new seeds
            # trained on every labeled row, while retaining the validated blend weights.
            if scheme == "time" and args.time_budget >= 300 and len(fitted_groups) == len(selected):
                row, _ = max(selected, key=lambda item: item[1])
                config = dict(row["config"])
                config["iterations"] = max(20, int(row["iterations"] * 1.1))
                config["class_weights"] = row["class_weights"]
                estimate = row["elapsed_seconds"] * len(y) / len(fit) * 1.4 + 5
                for extra in [1, 2]:
                    remaining = deadline - time.monotonic() - 5
                    if remaining < max(15, estimate * 1.1):
                        break
                    final_dir = folder / f"temporal_seed_{extra}"
                    final_dir.mkdir(parents=True, exist_ok=True)
                    data_path = final_dir / "seed_data.joblib"
                    joblib.dump(dict(bundle, seed=args.seed + extra * 17), data_path, compress=0)
                    result = run_worker(data_path, config, final_dir, remaining, True,
                                        task.path.parent, args.memory_gb)
                    data_path.unlink(missing_ok=True)
                    report.setdefault("temporal_seed_fits", []).append(
                        result | {"source_trial": row["number"], "seed": args.seed + extra * 17,
                                  "directory": str(final_dir)})
                    if result["status"] != "ok":
                        break
                    with np.load(final_dir / "predictions.npz") as data:
                        fitted_groups[row["number"]]["predictions"].append(data["test"].copy())
                    fitted_groups[row["number"]]["models"].append(str(final_dir / "model.joblib"))
                    commit_full_models()
    save()
    if not args.keep_data:
        bundle_path.unlink(missing_ok=True)
    print(f"submission: {task.output}", flush=True)
    print(f"report: {folder / 'report.json'}", flush=True)
    return report
