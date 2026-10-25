import numpy as np
import pandas as pd


class FeatureBuilder:
    def __init__(self, categorical=(), extended=False, operations=(), interactions=False, groups=(), history_detail=False, history_neighbors=False):
        self.categorical = list(categorical)
        self.extended = extended
        self.operations = list(operations)
        self.interactions = interactions
        self.groups = [list(group) for group in groups]
        self.history_detail = history_detail
        self.history_neighbors = history_neighbors

    @staticmethod
    def group_keys(frame, columns):
        normalized = pd.DataFrame(index=frame.index)
        for column in columns:
            values = frame[column]
            if pd.api.types.is_numeric_dtype(values):
                values = pd.to_numeric(values, errors="coerce").astype(float)
            normalized[column] = values.astype("string").fillna("__missing__")
        return pd.util.hash_pandas_object(normalized, index=False).to_numpy()

    def fit(self, frame, y=None):
        self.columns = frame.columns.tolist()
        self.cats = [c for c in frame if c in self.categorical or not pd.api.types.is_numeric_dtype(frame[c])]
        self.numeric = [c for c in frame if c not in self.cats]
        self.frequencies = {c: frame[c].fillna("__missing__").astype(str).value_counts(normalize=True).to_dict() for c in self.cats}
        self.log_columns = [c for c in self.numeric if frame[c].nunique() > 20 and abs(frame[c].skew()) > 2]
        self.active = [c for c in frame if frame[c].nunique(dropna=False) > 1]
        if not self.active:
            self.active = self.columns[:1]
        self.crosses = []
        if self.interactions:
            eligible = [c for c in self.cats if 1 < frame[c].nunique() <= 100]
            for i, left in enumerate(eligible[:6]):
                for right in eligible[i + 1:6]:
                    self.crosses.append((left, right))
            self.cats += [f"__cross_{i}" for i in range(len(self.crosses))]
        self.history = []
        self.history_details = []
        self.neighbor_history = None
        if self.groups and y is not None:
            target = np.asarray(y, dtype=float)
            self.target_sum = float(target.sum())
            self.target_count = len(target)
            for columns in self.groups:
                keys = self.group_keys(frame, columns)
                stats = pd.DataFrame({"key": keys, "y": target}).groupby("key").y.agg(["sum", "count"])
                self.history.append((columns, stats))
                if self.history_detail:
                    detail = pd.DataFrame({"key": keys, "square": target ** 2}).groupby("key").square.sum()
                    covariates = []
                    centered_y = target - pd.Series(keys).map(stats["sum"] / stats["count"]).to_numpy()
                    for column in self.numeric:
                        if column in columns or frame[column].nunique() < 10:
                            continue
                        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
                        local = pd.DataFrame({"key": keys, "value": values}).groupby("key").value.agg(["sum", "count"])
                        delta = values - pd.Series(keys).map(local["sum"] / local["count"]).to_numpy()
                        finite = np.isfinite(delta) & np.isfinite(centered_y)
                        if finite.sum() < 30 or np.std(delta[finite]) < 1e-10:
                            continue
                        correlation = np.corrcoef(delta[finite], centered_y[finite])[0, 1]
                        if np.isfinite(correlation) and abs(correlation) > .05:
                            covariates.append((abs(correlation), column, local))
                    covariates.sort(key=lambda item: item[0], reverse=True)
                    self.history_details.append((detail, [(col, local) for _, col, local in covariates[:6]]))
        if self.history_neighbors and self.groups and y is not None:
            self.neighbor_history = NeighborHistory(self.groups[0], self.numeric).fit(frame, y)
        return self

    def fit_transform(self, frame, y=None):
        return self.fit(frame, y).transform(frame, training_y=y)

    def transform(self, frame, training_y=None):
        x = frame[self.columns].copy()
        for c in self.cats:
            if c in x:
                x[c] = x[c].fillna("__missing__").astype(str)
        for c in self.numeric:
            x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
        if self.extended:
            x["__missing_count"] = frame.isna().sum(axis=1).astype(float)
            for c in self.log_columns:
                x[f"__log_{c}"] = np.sign(x[c]) * np.log1p(x[c].abs())
            for c in self.cats:
                if c in self.frequencies:
                    x[f"__freq_{c}"] = x[c].map(self.frequencies[c]).fillna(0).astype(float)
            if len(self.numeric) >= 4:
                x["__numeric_zero_count"] = x[self.numeric].eq(0).sum(axis=1).astype(float)
        if self.interactions:
            for i, (left, right) in enumerate(self.crosses):
                x[f"__cross_{i}"] = x[left].str.len().astype(str) + ":" + x[left] + x[right]
            if self.numeric:
                x["__numeric_mean"] = x[self.numeric].mean(axis=1)
                x["__numeric_std"] = x[self.numeric].std(axis=1)
        for i, (columns, stats) in enumerate(getattr(self, "history", [])):
            if getattr(self, "history_detail", False):
                x = x.copy()
            keys = pd.Series(self.group_keys(frame, columns), index=frame.index)
            counts = keys.map(stats["count"]).fillna(0).to_numpy(dtype=float)
            sums = keys.map(stats["sum"]).fillna(0).to_numpy(dtype=float)
            prior = np.full(len(frame), self.target_sum / self.target_count)
            if training_y is not None:
                target = np.asarray(training_y, dtype=float)
                counts -= 1
                sums -= target
            x[f"__history_{i}_mean"] = np.divide(sums, counts, out=prior.copy(), where=counts > 0)
            x[f"__history_{i}_count"] = np.maximum(counts, 0)
            if i < len(getattr(self, "history_details", [])):
                squares, covariates = self.history_details[i]
                sq = keys.map(squares).fillna(0).to_numpy(dtype=float)
                if training_y is not None:
                    sq -= target ** 2
                means = x[f"__history_{i}_mean"].to_numpy()
                variance = np.maximum(np.divide(sq, counts, out=np.zeros(len(frame)), where=counts > 0) - means ** 2, 0)
                x[f"__history_{i}_std"] = np.sqrt(variance)
                for j, (column, local) in enumerate(covariates):
                    cc = keys.map(local["count"]).fillna(0).to_numpy(float)
                    ss = keys.map(local["sum"]).fillna(0).to_numpy(float)
                    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
                    if training_y is not None:
                        cc -= np.isfinite(values)
                        ss -= np.nan_to_num(values)
                    mm = np.divide(ss, cc, out=np.full(len(frame), np.nan), where=cc > 0)
                    x[f"__history_{i}_cov_{j}_mean"] = mm
                    x[f"__history_{i}_cov_{j}_delta"] = values - mm
        if getattr(self, "neighbor_history", None) is not None:
            memory, count, deviation = self.neighbor_history.transform(frame, training=training_y is not None)
            x["__neighbor_mean"] = memory
            x["__neighbor_count"] = count
            x["__neighbor_std"] = deviation
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
        self.medians = frame.drop(columns=list(self.maps)).median().fillna(0).to_dict()
        return self

    def transform(self, frame):
        x = frame[self.columns].copy()
        for c, mapping in self.maps.items():
            x[c] = x[c].map(mapping).fillna(-1)
        for c, value in self.medians.items():
            x[c] = x[c].fillna(value)
        return x.to_numpy(dtype=np.float32)


class CategoricalEncoder:
    def fit(self, frame):
        self.columns = frame.columns.tolist()
        self.categories = {c: pd.CategoricalDtype(categories=frame[c].dropna().unique().tolist())
                           for c in frame if not pd.api.types.is_numeric_dtype(frame[c])}
        return self

    def transform(self, frame):
        x = frame[self.columns].copy()
        for c, dtype in self.categories.items():
            x[c] = x[c].astype(dtype)
        x.columns = [f"f{i}" for i in range(len(self.columns))]
        return x


class NeighborHistory:
    """Link repeated entities despite small changes in one continuous attribute."""
    def __init__(self, columns, numeric):
        self.columns, self.numeric = list(columns), list(numeric)

    def fit(self, frame, y):
        from sklearn.neighbors import NearestNeighbors
        self.area = next((c for c in self.columns if c in self.numeric and frame[c].nunique() > 30), None)
        self.models = {}
        self.target = np.asarray(y, float)
        self.prior = float(self.target.mean())
        if self.area is None:
            return self
        self.anchors = [c for c in self.columns if c != self.area]
        self.values = pd.to_numeric(frame[self.area], errors="coerce").to_numpy(float)
        self.conditioner = None
        exact = FeatureBuilder.group_keys(frame, self.columns)
        target_center = self.target - pd.Series(self.target).groupby(exact).transform("mean").to_numpy()
        ranked = []
        for column in self.numeric:
            if column in self.columns or frame[column].nunique() < 10:
                continue
            value = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            delta = value - pd.Series(value).groupby(exact).transform("mean").to_numpy()
            ok = np.isfinite(delta) & np.isfinite(target_center)
            if ok.sum() >= 30 and np.std(delta[ok]) > 1e-10:
                corr = np.corrcoef(delta[ok], target_center[ok])[0, 1]
                if np.isfinite(corr):
                    ranked.append((abs(corr), column))
        if ranked and max(ranked)[0] > .15:
            self.conditioner = max(ranked)[1]
            self.condition_values = pd.to_numeric(frame[self.conditioner], errors="coerce").to_numpy(float)
            quartiles = np.nanquantile(self.condition_values, [.25, .75])
            self.condition_scale = max(float((quartiles[1] - quartiles[0]) / 3), 1e-6)
        keys = FeatureBuilder.group_keys(frame, self.anchors)
        for key, indices in pd.Series(np.arange(len(frame))).groupby(keys).groups.items():
            ids = np.asarray(indices)
            ids = ids[np.isfinite(self.values[ids]) & (self.values[ids] > 0)]
            if len(ids):
                model = NearestNeighbors(n_neighbors=min(16, len(ids)), algorithm="kd_tree", n_jobs=1).fit(np.log1p(self.values[ids])[:, None])
                self.models[key] = (model, ids)
        return self

    def transform(self, frame, training=False):
        mean = np.full(len(frame), self.prior)
        count = np.zeros(len(frame))
        deviation = np.zeros(len(frame))
        if self.area is None:
            return mean, count, deviation
        keys = FeatureBuilder.group_keys(frame, self.anchors)
        values = pd.to_numeric(frame[self.area], errors="coerce").to_numpy(float)
        for key, indices in pd.Series(np.arange(len(frame))).groupby(keys).groups.items():
            if key not in self.models:
                continue
            query = np.asarray(indices)
            query = query[np.isfinite(values[query]) & (values[query] > 0)]
            if not len(query):
                continue
            model, ids = self.models[key]
            distance, nearest = model.kneighbors(np.log1p(values[query])[:, None])
            source = ids[nearest]
            weights = np.where(distance < .03, np.exp(-distance / .01), 0.)
            if self.conditioner is not None:
                current = pd.to_numeric(frame.iloc[query][self.conditioner], errors="coerce").to_numpy(float)
                difference = abs(current[:, None] - self.condition_values[source])
                weights *= np.where(np.isfinite(difference), np.exp(-difference / self.condition_scale), 1.)
            if training:
                weights[source == query[:, None]] = 0.
            total = weights.sum(axis=1)
            yy = self.target[source]
            mu = np.divide((weights * yy).sum(axis=1), total, out=np.full(len(query), self.prior), where=total > 0)
            var = np.divide((weights * yy ** 2).sum(axis=1), total, out=np.zeros(len(query)), where=total > 0) - mu ** 2
            mean[query], count[query], deviation[query] = mu, total, np.sqrt(np.maximum(var, 0))
        return mean, count, deviation
