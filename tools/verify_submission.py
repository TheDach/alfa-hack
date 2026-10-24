import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from research.metrics import PROBABILITY
from research.task import load_task


def verify(task_path, submission=None):
    task = load_task(task_path)
    path = Path(submission) if submission else task.output
    result = pd.read_csv(path)
    if result.columns.tolist() != task.sample.columns.tolist() or len(result) != len(task.test):
        raise ValueError("Submission schema or row count differs from the sample")
    if result.isna().any().any():
        raise ValueError("Submission contains missing values")
    for col in task.identifiers:
        if not result[col].is_unique or not np.array_equal(result[col].to_numpy(), task.test[col].to_numpy()):
            raise ValueError("Identifier duplication or order mismatch")
    if task.kind == "classification" and task.metric not in PROBABILITY:
        if not set(result[task.target]) <= set(task.train[task.target]):
            raise ValueError("Predicted label is absent from the training label schema")
    elif not np.isfinite(result[task.target]).all():
        raise ValueError("Predictions must be finite numbers")
    if task.metric in PROBABILITY and not result[task.target].between(0, 1).all():
        raise ValueError("Probabilities must be between zero and one")
    return {"task": str(task.path), "submission": str(path), "rows": len(result), "columns": result.columns.tolist(), "valid": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("submission", nargs="?")
    args = parser.parse_args()
    print(json.dumps(verify(args.task, args.submission), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
