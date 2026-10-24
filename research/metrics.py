import numpy as np
from sklearn import metrics


MINIMIZE = {"rmse", "rmsle", "mae", "mape", "log_loss"}
PROBABILITY = {"roc_auc", "log_loss", "average_precision"}


def evaluate(metric, y, pred, n_classes=0):
    y, pred = np.asarray(y), np.asarray(pred)
    if metric == "rmse":
        return float(np.sqrt(metrics.mean_squared_error(y, pred)))
    if metric == "rmsle":
        return float(np.sqrt(metrics.mean_squared_log_error(y, np.maximum(pred, 0))))
    if metric == "mae":
        return float(metrics.mean_absolute_error(y, pred))
    if metric == "mape":
        return float(metrics.mean_absolute_percentage_error(y, pred))
    if metric == "r2":
        if len(y) < 2:
            raise ValueError("R2 validation requires at least two rows")
        return float(metrics.r2_score(y, pred))
    if pred.ndim == 2:
        if metric == "log_loss":
            return float(metrics.log_loss(y, pred, labels=np.arange(n_classes)))
        if metric == "roc_auc":
            return float(metrics.roc_auc_score(y, pred[:, 1] if n_classes == 2 else pred,
                                             multi_class="ovr", average="macro"))
        if metric == "average_precision":
            if n_classes != 2:
                raise ValueError("average_precision requires a binary task")
            return float(metrics.average_precision_score(y, pred[:, 1]))
        pred = pred.argmax(axis=1)
    if metric in {"macro_f1", "weighted_f1"}:
        return float(metrics.f1_score(y, pred, labels=np.arange(n_classes),
                                     average="macro" if metric == "macro_f1" else "weighted", zero_division=0))
    if metric == "accuracy":
        return float(metrics.accuracy_score(y, pred))
    raise ValueError(f"Unsupported metric: {metric}")


def utility(metric, value):
    return -value if metric in MINIMIZE else value


def calibrate_weights(y, probabilities, n_classes):
    weights = np.ones(n_classes)
    if len(y) < max(30, n_classes * 4):
        return weights
    best = evaluate("macro_f1", y, probabilities, n_classes)
    for _ in range(2):
        changed = False
        for col in np.argsort(np.bincount(y, minlength=n_classes)):
            current = weights[col]
            for factor in [.65, .8, 1.25, 1.6, 2.0]:
                candidate = weights.copy()
                candidate[col] = np.clip(current * factor, .2, 5)
                score = evaluate("macro_f1", y, probabilities * candidate, n_classes)
                if score > best + .0001:
                    weights, best, changed = candidate, score, True
        if not changed:
            break
    return weights


def error_summary(kind, y, pred, n_classes):
    if kind == "classification":
        labels = pred.argmax(axis=1)
        return {"per_class_f1": metrics.f1_score(y, labels, labels=np.arange(n_classes), average=None, zero_division=0).tolist(),
                "accuracy": float(metrics.accuracy_score(y, labels)),
                "weighted_f1": float(metrics.f1_score(y, labels, average="weighted", zero_division=0)),
                "confusion_matrix": metrics.confusion_matrix(y, labels, labels=np.arange(n_classes)).tolist()}
    residual = np.asarray(y) - pred
    return {"residual_quantiles": np.quantile(residual, [0, .1, .5, .9, 1]).tolist(),
            "rmse": evaluate("rmse", y, pred), "mae": evaluate("mae", y, pred),
            "target_prediction_correlation": float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 0 and np.std(y) > 0 else None}
