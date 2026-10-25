import argparse
import math
import os
from pathlib import Path
import time


def arguments():
    parser = argparse.ArgumentParser(description="Autonomous tabular ML research runner")
    parser.add_argument("task", nargs="?", default="task.md")
    parser.add_argument("--output")
    parser.add_argument("--work-dir")
    parser.add_argument("--time-budget", type=float, default=float(os.getenv("TIME_BUDGET_SECONDS", "10320")))
    parser.add_argument("--max-trials", type=int, default=24)
    parser.add_argument("--trial-timeout", type=float, default=900)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-gb", type=float, default=27)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--task-type", choices=["classification", "regression"])
    parser.add_argument("--metric", choices=["rmse", "rmsle", "mae", "mape", "r2", "macro_f1", "weighted_f1", "accuracy", "roc_auc", "log_loss", "average_precision"])
    parser.add_argument("--target")
    parser.add_argument("--group-column")
    parser.add_argument("--time-column")
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--env-file")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if not 0 < args.time_budget <= 10620:
        parser.error("--time-budget must be positive and at most 10620 seconds")
    if not 0 < args.validation_fraction < .5:
        parser.error("--validation-fraction must be between 0 and .5")
    if (args.threads < 1 or args.max_trials < 0 or not math.isfinite(args.memory_gb)
            or not 0 < args.memory_gb <= 29 or not math.isfinite(args.trial_timeout) or args.trial_timeout <= 0):
        parser.error("Invalid resource limits")
    if args.group_column and args.time_column:
        parser.error("Choose either a group split or a time split")
    args.threads = min(args.threads, 8)
    return args


def main():
    started = time.monotonic()
    args = arguments()
    cache = Path(args.task).resolve().parent / ".automl_cache" / "matplotlib"
    if not os.environ.get("MPLCONFIGDIR"):
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache)
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = str(args.threads)
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(args.env_file) if args.env_file else Path(__file__).with_name(".env"), override=False)
    except ImportError:
        if args.env_file:
            raise RuntimeError("python-dotenv is needed for --env-file")
    if args.offline:
        os.environ.pop("OPENAI_API_KEY", None)
    from research.engine import run
    run(args, started)


if __name__ == "__main__":
    main()
