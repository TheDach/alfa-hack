import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.features import FeatureBuilder, NumericEncoder
from research.metrics import calibrate_weights, error_summary, evaluate


class FittedModel:
    def __init__(self, builder, encoder, estimator, kind, n_classes, transform, weights):
        self.builder = builder
        self.encoder = encoder
        self.estimator = estimator
        self.kind = kind
        self.n_classes = n_classes
        self.transform = transform
        self.weights = weights

    def predict(self, frame):
        x = self.builder.transform(frame)
        if self.encoder:
            x = self.encoder.transform(x)
        if self.kind == "regression":
            result = np.asarray(self.estimator.predict(x)).reshape(-1)
            if self.transform == "log1p":
                result = np.expm1(np.minimum(result, 30))
            return result
        local = np.asarray(self.estimator.predict_proba(x))
        result = np.full((len(frame), self.n_classes), 1e-12)
        for j, cls in enumerate(np.asarray(self.estimator.classes_).reshape(-1)):
            result[:, int(cls)] = local[:, j]
        result *= self.weights
        return result / result.sum(axis=1, keepdims=True)


def train_model(bundle, config, final=False):
    train_idx = np.arange(len(bundle["train"])) if final else bundle["fit_indices"]
    cal_idx = bundle["cal_indices"]
    x, y = bundle["train"], bundle["y"]
    builder = FeatureBuilder(bundle["categorical"], config.get("features") == "extended", config.get("feature_ops", []))
    builder.fit(x.iloc[train_idx])
    tx = builder.transform(x.iloc[train_idx])
    cx = builder.transform(x.iloc[cal_idx])
    ty = y[train_idx]
    transform = config.get("target_transform", "none")
    if bundle["kind"] == "regression" and transform == "log1p":
        if np.min(ty) < 0:
            raise ValueError("log1p requires nonnegative targets")
        ty = np.log1p(ty)
    else:
        transform = "none"
    kind = bundle["kind"]
    classes = bundle["n_classes"]
    family = config["family"]
    encoder = None
    threads, seed = bundle["threads"], bundle["seed"]
    count = int(config.get("iterations", 600))
    balanced = config.get("balanced", False)
    cats = [c for c in tx if c in builder.cats]
    if (kind == "classification" and len(np.unique(ty)) < 2) or np.std(ty) == 0:
        estimator = DummyRegressor() if kind == "regression" else DummyClassifier(strategy="prior")
        encoder = NumericEncoder().fit(tx)
        estimator.fit(encoder.transform(tx), ty)
    elif family == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        kwargs = dict(iterations=count, depth=int(config.get("depth", 6)),
                      learning_rate=config.get("learning_rate", .06), l2_leaf_reg=config.get("l2", 5),
                      random_seed=seed, thread_count=threads, verbose=False, allow_writing_files=False,
                      cat_features=cats, loss_function="RMSE" if kind == "regression" else "MultiClass",
                      used_ram_limit="20gb")
        if kind == "regression" and bundle["metric"] == "mae":
            kwargs["loss_function"] = "MAE"
        if kind == "classification" and balanced:
            kwargs["auto_class_weights"] = "SqrtBalanced"
        cls = CatBoostRegressor if kind == "regression" else CatBoostClassifier
        estimator = cls(**kwargs)
        fit_args = {}
        if not final and len(cal_idx):
            cy = y[cal_idx]
            if transform == "log1p":
                cy = np.log1p(cy)
            fit_args = dict(eval_set=(cx, cy), early_stopping_rounds=100, use_best_model=True)
        estimator.fit(tx, ty, **fit_args)
    elif family == "lightgbm":
        import lightgbm as lgb
        encoder = NumericEncoder().fit(tx)
        xt = encoder.transform(tx)
        xc = encoder.transform(cx)
        kwargs = dict(n_estimators=count, learning_rate=config.get("learning_rate", .045),
                      num_leaves=min(2 ** int(config.get("depth", 6)), 127),
                      max_depth=-1, min_child_samples=25, reg_lambda=config.get("l2", 5),
                      colsample_bytree=.9, subsample=.85, subsample_freq=1,
                      random_state=seed, n_jobs=threads, verbosity=-1,
                      deterministic=True, force_col_wise=True)
        if kind == "classification":
            kwargs["class_weight"] = "balanced" if balanced else None
        cls = lgb.LGBMRegressor if kind == "regression" else lgb.LGBMClassifier
        estimator = cls(**kwargs)
        fit_args = {}
        if not final and len(cal_idx):
            cy = np.log1p(y[cal_idx]) if transform == "log1p" else y[cal_idx]
            fit_args = dict(eval_set=[(xc, cy)], callbacks=[lgb.early_stopping(100, verbose=False)])
        estimator.fit(xt, ty, **fit_args)
    elif family == "extra_trees":
        encoder = NumericEncoder().fit(tx)
        cls = ExtraTreesRegressor if kind == "regression" else ExtraTreesClassifier
        kwargs = dict(n_estimators=min(count, 500), min_samples_leaf=2, max_features=.9,
                      n_jobs=threads, random_state=seed)
        if kind == "classification":
            kwargs["class_weight"] = "balanced" if balanced else None
        estimator = cls(**kwargs).fit(encoder.transform(tx), ty)
    elif family == "linear":
        encoder = NumericEncoder().fit(tx)
        model = Ridge(alpha=config.get("l2", 10)) if kind == "regression" else LogisticRegression(
            C=1 / config.get("l2", 1), max_iter=600, class_weight="balanced" if balanced else None)
        estimator = make_pipeline(StandardScaler(), model).fit(encoder.transform(tx), ty)
    else:
        raise ValueError(f"Unknown model family: {family}")
    fitted = FittedModel(builder, encoder, estimator, kind, classes, transform, np.ones(classes))
    if kind == "classification":
        if final:
            fitted.weights = np.asarray(config.get("class_weights", np.ones(classes)))
        elif bundle["metric"] == "macro_f1" and len(cal_idx):
            fitted.weights = calibrate_weights(y[cal_idx], fitted.predict(x.iloc[cal_idx]), classes)
    best_iteration = getattr(estimator, "best_iteration_", None)
    if family == "catboost":
        best_iteration = getattr(estimator, "tree_count_", count)
    return fitted, int(best_iteration or count)


def execute(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    bundle = joblib.load(spec["bundle"])
    fitted, iterations = train_model(bundle, spec["config"], spec.get("final", False))
    test_pred = fitted.predict(bundle["test"])
    result = {"iterations": iterations, "class_weights": fitted.weights.tolist()}
    predictions = {"test": test_pred}
    if not spec.get("final", False):
        indices = bundle["validation_indices"]
        pred = fitted.predict(bundle["train"].iloc[indices])
        result["metric"] = evaluate(bundle["metric"], bundle["y"][indices], pred, bundle["n_classes"])
        result["errors"] = error_summary(bundle["kind"], bundle["y"][indices], pred, bundle["n_classes"])
        predictions["validation"] = pred
    if not np.isfinite(test_pred).all():
        raise ValueError("Nonfinite predictions")
    joblib.dump(fitted, out / "model.joblib", compress=1)
    np.savez_compressed(out / "predictions.npz", **predictions)
    result["elapsed_seconds"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    execute(sys.argv[1])
