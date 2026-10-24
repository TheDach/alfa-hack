from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass
class Task:
    path: Path
    description: str
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame
    dictionary: pd.DataFrame
    target: str
    identifiers: list[str]
    kind: str
    metric: str
    output: Path
    categorical: list[str]

    @property
    def features(self):
        return [c for c in self.test if c not in self.identifiers and c != self.target]


def find_file(root, text, name):
    candidates = []
    for match in re.findall(r"`([^`\n]+)`", text):
        if Path(match.replace("\\", "/")).name == name:
            candidates.append(root / match)
    candidates.append(root / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    found = list(root.rglob(name))
    if len(found) == 1:
        return found[0].resolve()
    raise ValueError(f"Cannot locate an unambiguous {name} relative to {root}")


def infer_metric(text, kind):
    text = text.lower().replace("**", "").replace("_", " ")
    priorities = [("rmsle", "rmsle"), ("rmse", "rmse"),
                  ("macro f1", "macro_f1"), ("f1 macro", "macro_f1"),
                  ("roc auc", "roc_auc"), ("roc-auc", "roc_auc"),
                  ("log loss", "log_loss"), ("logloss", "log_loss"),
                  ("average precision", "average_precision"),
                  ("mape", "mape"), ("mae", "mae"), ("r2", "r2"),
                  ("weighted f1", "weighted_f1"), ("accuracy", "accuracy")]
    regression = {"rmse", "rmsle", "mae", "mape", "r2"}
    priorities = [(token, metric) for token, metric in priorities if (metric in regression) == (kind == "regression")]
    sections = re.findall(r"(?:основн\w*\s+метрик\w*|primary\s+metric|метрик\w*\s+оценки)\s*[:—–-]?([^\n]{0,160})", text)
    for section in sections + [text]:
        found = []
        for token, metric in priorities:
            match = re.search(r"(?<![a-z])" + re.escape(token) + r"(?![a-z])", section)
            if match:
                found.append((match.start(), metric))
        if found:
            return min(found)[1]
    return "rmse" if kind == "regression" else "macro_f1"


def validation_columns(task):
    text = task.description.lower()
    time_column = group_column = None
    patterns = [
        ("time", r"(?:time[- ](?:based )?split|temporal validation|временн\w*\s+валидац\w*)[^\n`]{0,80}`([^`]+)`"),
        ("group", r"(?:group(?:ed)?[- ](?:split|validation)|groupkfold|группов\w*\s+валидац\w*)[^\n`]{0,80}`([^`]+)`")]
    lookup = {column.lower(): column for column in task.features}
    for mode, pattern in patterns:
        matches = re.findall(pattern, text)
        for column in matches:
            if column in lookup:
                if mode == "time":
                    time_column = lookup[column]
                else:
                    group_column = lookup[column]
    return group_column, time_column


def load_task(path, target=None, kind=None, metric=None, output=None):
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8-sig")
    root = path.parent
    train = pd.read_csv(find_file(root, text, "train.csv"))
    test = pd.read_csv(find_file(root, text, "test.csv"))
    sample = pd.read_csv(find_file(root, text, "sample_submission.csv"))
    dictionary = pd.read_csv(find_file(root, text, "data_dictionary.csv"))
    if not train.columns.is_unique or not test.columns.is_unique:
        raise ValueError("Duplicate column names")
    if target is None and {"feature", "role"} <= set(dictionary):
        targets = dictionary.loc[dictionary.role.str.lower().isin(["target", "таргет"]), "feature"].tolist()
        if len(targets) == 1:
            target = targets[0]
    if target is None:
        targets = [c for c in train if c not in test and c in sample]
        if len(targets) != 1:
            raise ValueError("Expected one supervised target; specify --target")
        target = targets[0]
    if target not in train or target in test:
        raise ValueError("Target must occur only in train.csv")
    identifiers = [c for c in sample if c != target and c in test]
    if target not in sample or set(sample) != {target, *identifiers}:
        raise ValueError("Unsupported submission schema")
    if len(sample) != len(test):
        raise ValueError("Sample and test row counts differ")
    for col in identifiers:
        if test[col].isna().any() or not test[col].is_unique:
            raise ValueError(f"Invalid identifier: {col}")
        if not np.array_equal(sample[col].to_numpy(), test[col].to_numpy()):
            raise ValueError("Sample identifiers must match test order")
    train = train.loc[train[target].notna()].reset_index(drop=True)
    if len(train) < 2 or not len(test):
        raise ValueError("Need at least two labeled rows and a nonempty test")
    y = train[target]
    low = text.lower()
    if kind is None:
        if any(x in low for x in ["классификац", "classification", "macro f1", "roc_auc", "roc-auc"]):
            kind = "classification"
        elif any(x in low for x in ["регресси", "regression", "rmse", "rmsle"]):
            kind = "regression"
        else:
            numeric = pd.api.types.is_numeric_dtype(y)
            discrete = numeric and np.allclose(y, np.round(y)) and y.nunique() <= min(30, max(2, len(y) // 20))
            kind = "classification" if not numeric or discrete else "regression"
    metric = metric or infer_metric(text, kind)
    regression_metrics = {"rmse", "rmsle", "mae", "mape", "r2"}
    if (kind == "regression") != (metric in regression_metrics):
        raise ValueError(f"Metric {metric} is incompatible with {kind}")
    if kind == "regression" and (not pd.api.types.is_numeric_dtype(y) or not np.isfinite(y).all()):
        raise ValueError("Regression target must be finite and numeric")
    categorical = []
    if {"feature", "dtype"} <= set(dictionary):
        categorical = dictionary.loc[dictionary.dtype.astype(str).str.contains("categor|object|string", case=False), "feature"].tolist()
    features = [c for c in test if c not in identifiers and c != target]
    if not features or not set(features) <= set(train):
        raise ValueError("Train/test feature schema mismatch")
    destination = Path(output).resolve() if output else root / "submission.csv"
    return Task(path, text, train, test, sample, dictionary, target, identifiers,
                kind, metric, destination, [c for c in categorical if c in features])


def profile_task(task):
    x = task.train[task.features]
    numeric = [c for c in x.select_dtypes(include="number") if c not in task.categorical]
    profile = {"kind": task.kind, "metric": task.metric, "target": task.target,
               "train_rows": len(x), "test_rows": len(task.test), "feature_count": len(x.columns),
               "numeric_columns": numeric, "categorical_columns": [c for c in x if c not in numeric],
               "missing_fraction": x.isna().mean().round(4).to_dict(),
               "cardinality": x.nunique().to_dict(), "description": task.description[:14000]}
    if task.kind == "classification":
        profile["class_counts"] = {str(k): int(v) for k, v in task.train[task.target].value_counts().items()}
    else:
        profile["target_quantiles"] = task.train[task.target].quantile([0, .01, .5, .99, 1]).to_dict()
    return profile
