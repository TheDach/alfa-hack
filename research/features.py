import numpy as np
import pandas as pd


class FeatureBuilder:
    def __init__(self, categorical=(), extended=False, operations=()):
        self.categorical = list(categorical)
        self.extended = extended
        self.operations = list(operations)

    def fit(self, frame):
        self.columns = frame.columns.tolist()
        self.cats = [c for c in frame if c in self.categorical or not pd.api.types.is_numeric_dtype(frame[c])]
        self.numeric = [c for c in frame if c not in self.cats]
        self.frequencies = {c: frame[c].fillna("__missing__").astype(str).value_counts(normalize=True).to_dict() for c in self.cats}
        self.log_columns = [c for c in self.numeric if frame[c].nunique() > 20 and abs(frame[c].skew()) > 2]
        self.active = [c for c in frame if frame[c].nunique(dropna=False) > 1]
        if not self.active:
            self.active = self.columns[:1]
        return self

    def transform(self, frame):
        x = frame[self.columns].copy()
        for c in self.cats:
            x[c] = x[c].fillna("__missing__").astype(str)
        for c in self.numeric:
            x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
        if self.extended:
            x["__missing_count"] = frame.isna().sum(axis=1).astype(float)
            for c in self.log_columns:
                x[f"__log_{c}"] = np.sign(x[c]) * np.log1p(x[c].abs())
            for c in self.cats:
                x[f"__freq_{c}"] = x[c].map(self.frequencies[c]).fillna(0).astype(float)
            if len(self.numeric) >= 4:
                values = x[self.numeric]
                x["__numeric_zero_count"] = values.eq(0).sum(axis=1).astype(float)
        for i, spec in enumerate(self.operations):
            left = x[spec["left"]]
            right = x[spec["right"]] if spec.get("right") else None
            op = spec["op"]
            if op == "ratio":
                value = left / right.replace(0, np.nan)
            elif op == "difference":
                value = left - right
            elif op == "sum":
                value = left + right
            elif op == "product":
                value = left * right
            elif op == "log1p":
                value = np.sign(left) * np.log1p(left.abs())
            elif op == "sqrt":
                value = np.sqrt(left.clip(lower=0))
            else:
                raise ValueError(f"Unsupported feature operation: {op}")
            x[f"__research_{i}"] = value.replace([np.inf, -np.inf], np.nan)
        added = [c for c in x if c not in self.columns]
        return x[self.active + added]


class NumericEncoder:
    def fit(self, frame):
        self.columns = frame.columns.tolist()
        self.maps = {c: {v: i for i, v in enumerate(frame[c].unique())}
                     for c in frame if not pd.api.types.is_numeric_dtype(frame[c])}
        numeric = frame.drop(columns=list(self.maps))
        self.medians = numeric.median().fillna(0).to_dict()
        return self

    def transform(self, frame):
        x = frame[self.columns].copy()
        for c, mapping in self.maps.items():
            x[c] = x[c].map(mapping).fillna(-1)
        for c, value in self.medians.items():
            x[c] = x[c].fillna(value)
        return x.to_numpy(dtype=np.float32)
