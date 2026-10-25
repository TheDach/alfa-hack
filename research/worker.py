import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from research.features import CategoricalEncoder, FeatureBuilder, NumericEncoder
from research.metrics import calibrate_weights, error_summary, evaluate


class ClassificationMetric:
    def __init__(self, metric, n_classes, xgboost=False):
        self.metric, self.n_classes, self.xgboost = metric, n_classes, xgboost
        self.__name__ = "validation_" + metric

    def __call__(self, y, predictions):
        p = np.asarray(predictions)
        if p.ndim == 1:
            p = np.column_stack([1 - p, p]) if self.n_classes == 2 else p.reshape(-1, self.n_classes, order="F")
        if self.metric in {"macro_f1", "weighted_f1", "accuracy"}:
            matrix = np.bincount(np.asarray(y, dtype=int) * self.n_classes + p.argmax(axis=1),
                                 minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes)
            if self.metric == "accuracy":
                value = np.trace(matrix) / max(1, matrix.sum())
            else:
                denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
                f1 = np.divide(2 * np.diag(matrix), denominator, out=np.zeros(self.n_classes), where=denominator > 0)
                value = np.mean(f1) if self.metric == "macro_f1" else np.average(f1, weights=matrix.sum(axis=1))
        else:
            value = evaluate(self.metric, y, p, self.n_classes)
        return float(1 - value) if self.xgboost else (self.metric, float(value), True)


class FittedModel:
    def __init__(self, builder, encoder, estimator, kind, n_classes, transform, weights, residual_column=None):
        self.builder, self.encoder, self.estimator = builder, encoder, estimator
        self.kind, self.n_classes, self.transform, self.weights = kind, n_classes, transform, weights
        self.residual_column = residual_column

    def predict(self, frame):
        x = self.builder.transform(frame)
        baseline = x[self.residual_column].to_numpy() if getattr(self, "residual_column", None) else 0
        if self.encoder:
            x = self.encoder.transform(x)
        if self.kind == "regression":
            result = np.asarray(self.estimator.predict(x)).reshape(-1)
            if self.transform == "log1p":
                result = np.expm1(np.minimum(result, 30))
            return result + baseline
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
    kind, classes, family = bundle["kind"], bundle["n_classes"], config["family"]
    groups = config.get("group_features", []) if kind == "regression" else []
    selected_columns = list(x.columns)
    if config.get("stable_features") and kind == "regression":
        from scipy.stats import ks_2samp
        reference = x.iloc[cal_idx] if not final and len(cal_idx) else bundle.get("test", x.iloc[train_idx])
        protected = {column for group in groups for column in group}
        dropped = []
        for column in selected_columns:
            if column in protected or column in bundle["categorical"] or not np.issubdtype(x[column].dtype, np.number):
                continue
            left = x.iloc[train_idx][column].dropna()
            right = reference[column].dropna()
            if len(left) > 100 and len(right) > 100:
                left = left.sample(min(len(left), 10000), random_state=bundle["seed"])
                right = right.sample(min(len(right), 10000), random_state=bundle["seed"])
                if ks_2samp(left, right).statistic > .5:
                    dropped.append(column)
        if len(dropped) < len(selected_columns):
            selected_columns = [column for column in selected_columns if column not in dropped]
    x = x[selected_columns]
    mode = config.get("features", "basic")
    feature_ops = [op for op in config.get("feature_ops", [])
                   if op.get("left") in selected_columns and (not op.get("right") or op["right"] in selected_columns)]
    builder = FeatureBuilder(bundle["categorical"], mode != "basic", feature_ops,
                             interactions=mode == "interactions", groups=groups, history_detail=config.get("history_detail", False),
                             history_neighbors=config.get("history_neighbors", False))
    tx = builder.fit_transform(x.iloc[train_idx], y[train_idx] if groups else None)
    cx = builder.transform(x.iloc[cal_idx])
    ty, cy = y[train_idx].copy(), y[cal_idx].copy()
    residual_column = "__history_0_mean" if config.get("history_residual", False) and groups else None
    if residual_column:
        ty = ty - tx[residual_column].to_numpy()
        cy = cy - cx[residual_column].to_numpy()
    transform = config.get("target_transform", "none")
    if kind == "regression" and bundle["metric"] == "rmsle":
        transform = "log1p"
    if kind == "regression" and transform == "log1p" and not residual_column:
        if np.min(ty) < 0:
            raise ValueError("log1p requires nonnegative targets")
        ty, cy = np.log1p(ty), np.log1p(cy)
    else:
        transform = "none"
    encoder = None
    threads, seed = bundle["threads"], bundle["seed"]
    count = int(config.get("iterations", 600))
    balanced = config.get("balanced", False)
    power = config.get("balance_power", (.5 if family == "catboost" else 1.) if balanced else 0.)
    sample_weight = None
    if kind == "classification" and power:
        counts = np.bincount(ty, minlength=classes)
        weight = np.maximum(counts, 1).astype(float) ** -float(power)
        sample_weight = weight[ty]
        sample_weight /= sample_weight.mean()
    cats = [c for c in tx if c in builder.cats]
    if (kind == "classification" and len(np.unique(ty)) < 2) or np.std(ty) == 0 or tx.nunique(dropna=False).le(1).all():
        estimator = DummyRegressor() if kind == "regression" else DummyClassifier(strategy="prior")
        encoder = NumericEncoder().fit(tx)
        estimator.fit(encoder.transform(tx), ty)
    elif family == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        loss = "RMSE" if kind == "regression" else ("Logloss" if classes == 2 else "MultiClass")
        if kind == "regression" and transform == "none" and bundle["metric"] in {"mae", "mape"}:
            loss = "MAE" if bundle["metric"] == "mae" else "MAPE"
        eval_metric = loss
        if kind == "classification" and config.get("stopping_metric") != "log_loss":
            eval_metric = {"macro_f1": "TotalF1:average=Macro;use_weights=false",
                           "weighted_f1": "TotalF1:average=Weighted;use_weights=false",
                           "accuracy": "Accuracy:use_weights=false", "roc_auc": "AUC"}.get(bundle["metric"], loss)
        if sample_weight is None:
            eval_metric = eval_metric.replace(";use_weights=false", "").replace(":use_weights=false", "")
        kwargs = dict(iterations=count, depth=int(config.get("depth", 6)),
                      learning_rate=config.get("learning_rate", .06), l2_leaf_reg=config.get("l2", 5),
                      random_seed=seed, thread_count=threads, verbose=False, allow_writing_files=False,
                      cat_features=cats, loss_function=loss, eval_metric=eval_metric, used_ram_limit="20gb")
        cls = CatBoostRegressor if kind == "regression" else CatBoostClassifier
        estimator = cls(**kwargs)
        fit_args = {}
        if not final and len(cal_idx):
            fit_args = dict(eval_set=(cx, cy), early_stopping_rounds=150, use_best_model=True)
        estimator.fit(tx, ty, sample_weight=sample_weight, **fit_args)
    elif family == "lightgbm":
        import lightgbm as lgb
        encoder_type = NumericEncoder if config.get("categorical_encoding") == "ordinal" else CategoricalEncoder
        encoder = encoder_type().fit(tx)
        xt, xc = encoder.transform(tx), encoder.transform(cx)
        kwargs = dict(n_estimators=count, learning_rate=config.get("learning_rate", .045),
                      num_leaves=int(config.get("num_leaves", min(2 ** int(config.get("depth", 6)), 127))),
                      max_depth=-1, min_child_samples=int(config.get("min_child_samples", 25)),
                      reg_lambda=config.get("l2", 5), colsample_bytree=config.get("colsample_bytree", .9),
                      subsample=config.get("subsample", .85), subsample_freq=1,
                      max_bin=int(config.get("max_bin", 255)), random_state=seed, n_jobs=threads,
                      verbosity=-1, deterministic=True, force_col_wise=True)
        if kind == "regression" and transform == "none":
            kwargs["objective"] = {"mae": "regression_l1", "mape": "mape"}.get(bundle["metric"], "regression")
        cls = lgb.LGBMRegressor if kind == "regression" else lgb.LGBMClassifier
        metric = None
        if kind == "classification" and config.get("stopping_metric") != "log_loss" and bundle["metric"] in {"macro_f1", "weighted_f1", "accuracy", "average_precision"}:
            kwargs["metric"] = "None"
            metric = ClassificationMetric(bundle["metric"], classes)
        elif kind == "classification" and bundle["metric"] == "roc_auc" and classes == 2:
            kwargs["metric"] = "auc"
        estimator = cls(**kwargs)
        fit_args = {}
        if not final and len(cal_idx):
            fit_args = dict(eval_set=[(xc, cy)], callbacks=[lgb.early_stopping(150, verbose=False)])
            if metric:
                fit_args["eval_metric"] = metric
        estimator.fit(xt, ty, sample_weight=sample_weight, **fit_args)
    elif family == "xgboost":
        import xgboost as xgb
        encoder = NumericEncoder().fit(tx)
        kwargs = dict(n_estimators=count, learning_rate=config.get("learning_rate", .055),
                      max_depth=int(config.get("depth", 6)), min_child_weight=3,
                      reg_lambda=config.get("l2", 5), subsample=config.get("subsample", .85),
                      colsample_bytree=config.get("colsample_bytree", .9), tree_method="hist",
                      max_bin=int(config.get("max_bin", 255)), n_jobs=threads, random_state=seed)
        if kind == "regression":
            kwargs["objective"] = "reg:absoluteerror" if bundle["metric"] == "mae" else "reg:squarederror"
            kwargs["eval_metric"] = "mae" if bundle["metric"] == "mae" else "rmse"
            cls = xgb.XGBRegressor
        else:
            kwargs["objective"] = "multi:softprob" if classes > 2 else "binary:logistic"
            kwargs["eval_metric"] = ClassificationMetric(bundle["metric"], classes, True) if bundle["metric"] != "log_loss" else ("mlogloss" if classes > 2 else "logloss")
            cls = xgb.XGBClassifier
        if not final and len(cal_idx):
            kwargs["early_stopping_rounds"] = 150
        estimator = cls(**kwargs)
        fit_args = dict(eval_set=[(encoder.transform(cx), cy)], verbose=False) if not final and len(cal_idx) else {}
        estimator.fit(encoder.transform(tx), ty, sample_weight=sample_weight, **fit_args)
    elif family == "extra_trees":
        encoder = NumericEncoder().fit(tx)
        cls = ExtraTreesRegressor if kind == "regression" else ExtraTreesClassifier
        kwargs = dict(n_estimators=min(count, 500), min_samples_leaf=2, max_features=.9,
                      n_jobs=threads, random_state=seed)
        if kind == "regression" and bundle["metric"] == "mae":
            kwargs["criterion"] = "absolute_error"
        estimator = cls(**kwargs).fit(encoder.transform(tx), ty, sample_weight=sample_weight)
    elif family == "linear":
        numeric = [c for c in tx if c not in cats]
        encoder = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", max_categories=128, min_frequency=2), cats)
        ], sparse_threshold=1.).fit(tx)
        model = Ridge(alpha=config.get("l2", 10), solver="lsqr") if kind == "regression" else LogisticRegression(
            C=1 / config.get("l2", 1), max_iter=600, solver="lbfgs")
        estimator = model.fit(encoder.transform(tx), ty, sample_weight=sample_weight)
    else:
        raise ValueError(f"Unknown model family: {family}")
    fitted = FittedModel(builder, encoder, estimator, kind, classes, transform, np.ones(classes), residual_column)
    if kind == "classification":
        if final:
            fitted.weights = np.asarray(config.get("class_weights", np.ones(classes)))
        elif bundle["metric"] == "macro_f1" and len(cal_idx):
            fitted.weights = calibrate_weights(y[cal_idx], fitted.predict(x.iloc[cal_idx]), classes)
    best_iteration = getattr(estimator, "best_iteration_", None)
    if family == "catboost":
        best_iteration = getattr(estimator, "tree_count_", count)
    elif family == "xgboost" and not final and len(cal_idx):
        best_iteration = getattr(estimator, "best_iteration", count - 1) + 1
    return fitted, int(best_iteration or count)


def execute(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    bundle = joblib.load(spec["bundle"])
    fitted, iterations = train_model(bundle, spec["config"], spec.get("final", False))
    early_stop_iterations = iterations
    if not spec.get("final", False) and bundle.get("refit_before_validation") and len(bundle["cal_indices"]):
        # Calibration labels determine the iteration count, then join training.
        # Validation labels stay exclusively outside both fits.
        indices = np.sort(np.r_[bundle["fit_indices"], bundle["cal_indices"]])
        refit_bundle = dict(bundle, train=bundle["train"].iloc[indices].reset_index(drop=True),
                            y=bundle["y"][indices], fit_indices=np.arange(len(indices)),
                            cal_indices=np.array([], dtype=int))
        refit_config = dict(spec["config"], iterations=max(20, iterations), class_weights=fitted.weights.tolist())
        fitted, iterations = train_model(refit_bundle, refit_config, True)
    test_pred = fitted.predict(bundle["test"])
    result = {"iterations": iterations, "early_stop_iterations": early_stop_iterations, "class_weights": fitted.weights.tolist()}
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
