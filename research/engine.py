import json
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from research.metrics import PROBABILITY, evaluate, utility
from research.planner import QwenPlanner
from research.search import candidates
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
        order = np.argsort(x[time_column].to_numpy(), kind="stable")
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
              f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})\n"
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
        report.update(elapsed_seconds=round(time.monotonic() - started, 3), llm_usage=planner.usage, llm_events=planner.events)
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
    report["validation"] = {"scheme": scheme, "fit_rows": len(fit), "calibration_rows": len(cal), "validation_rows": len(valid)}
    bundle = {"train": x, "test": task.test[task.features], "y": y, "fit_indices": fit,
              "cal_indices": cal, "validation_indices": valid, "categorical": task.categorical,
              "kind": task.kind, "metric": task.metric, "n_classes": len(classes),
              "threads": args.threads, "seed": args.seed}
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
    for number in range(args.max_trials):
        remaining = deadline - time.monotonic()
        reserve = max(15, min(1200, args.time_budget * .25))
        if remaining <= reserve + 5:
            break
        allowed = candidates(profile, report["history"])
        if not allowed:
            break
        config = allowed[0]
        if number > 0 and planner.enabled and remaining > reserve + 30:
            proposal = planner.propose(profile, report["history"], allowed, remaining - reserve)
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
        selected = [(best, 1.0)]
        best_arrays = np.load(Path(best["directory"]) / "predictions.npz")
        best_val, best_test = best_arrays["validation"].copy(), best_arrays["test"].copy()
        best_arrays.close()
        blend_value = best_value
        for other in sorted(successful, key=lambda item: utility(task.metric, item["metric"]), reverse=True)[:4]:
            if other is best:
                continue
            with np.load(Path(other["directory"]) / "predictions.npz") as data:
                for weight in [.2, .35, .5]:
                    combined = (1 - weight) * best_val + weight * data["validation"]
                    score = evaluate(task.metric, y[valid], combined, len(classes))
                    if utility(task.metric, score) > utility(task.metric, blend_value) + 1e-5:
                        blend_value = score
                        selected = [(best, 1 - weight), (other, weight)]
        if len(selected) > 1:
            with np.load(Path(selected[1][0]["directory"]) / "predictions.npz") as data:
                write_submission(task, selected[0][1] * best_test + selected[1][1] * data["test"], classes)
        report.update(best_validation_metric=blend_value, selected=[{"trial": row["number"], "weight": weight} for row, weight in selected])
        final_predictions = []
        final_models = []
        for index, (row, weight) in enumerate(selected):
            remaining = deadline - time.monotonic() - 3
            estimate = row["elapsed_seconds"] * len(y) / len(fit) * 1.4 + 5
            if remaining < min(estimate, 15):
                break
            config = dict(row["config"])
            config["iterations"] = max(30, int(row["iterations"] * 1.1))
            config["class_weights"] = row["class_weights"]
            final_dir = folder / f"final_{index}"
            result = run_worker(bundle_path, config, final_dir, remaining / (len(selected) - index), True,
                                task.path.parent, args.memory_gb)
            report.setdefault("final_fits", []).append(result | {"config": config, "directory": str(final_dir)})
            if result["status"] != "ok":
                break
            with np.load(final_dir / "predictions.npz") as data:
                final_predictions.append(weight * data["test"])
            final_models.append({"path": str(final_dir / "model.joblib"), "weight": weight})
        if len(final_predictions) == len(selected):
            write_submission(task, np.sum(final_predictions, axis=0), classes)
            report.update(status="complete", models=final_models)
            atomic_json(folder / "model_manifest.json", {"models": final_models, "classes": classes.tolist(),
                                                       "features": task.features, "kind": task.kind, "metric": task.metric})
    save()
    if not args.keep_data:
        bundle_path.unlink(missing_ok=True)
    print(f"submission: {task.output}", flush=True)
    print(f"report: {folder / 'report.json'}", flush=True)
    return report
