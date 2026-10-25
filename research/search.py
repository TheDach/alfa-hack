import importlib.util
import json


CORE_COVERAGE = ("catboost", "lightgbm:native", "lightgbm:ordinal", "xgboost", "extra_trees", "linear")


def coverage_key(config):
    family = config["family"]
    if family == "lightgbm":
        suffix = config.get("categorical_encoding", "native")
        if config.get("group_features"):
            suffix += ":groups"
            if config.get("stable_features"):
                suffix += ":stable"
            if config.get("history_detail"):
                suffix += ":detail"
            if config.get("history_neighbors"):
                suffix += ":neighbors"
        if config.get("stopping_metric") == "log_loss":
            suffix += ":logloss"
        if config.get("balanced") or config.get("balance_power", 0) > 0:
            suffix += ":balanced"
        return family + ":" + suffix
    return family


def coverage_pending(history, options, profile=None):
    """Failures count as attempts so an unavailable family cannot stall research."""
    attempted = {coverage_key(row["config"]) for row in history}
    available = {coverage_key(config) for config in options}
    priority = list(CORE_COVERAGE)
    if profile and profile.get("entity_groups") and profile["kind"] == "regression":
        priority.insert(2, "lightgbm:native:groups")
        if profile.get("temporal_shift"):
            priority = ["lightgbm:native:groups:stable", "lightgbm:native:groups:stable:neighbors",
                        "lightgbm:native:groups:stable:detail"] + priority
    counts = list((profile or {}).get("class_counts", {}).values())
    if (profile or {}).get("metric") == "macro_f1" and counts and max(counts) / max(1, min(counts)) >= 5:
        priority.insert(2, "lightgbm:native:balanced")
    return next((key for key in priority if key in available and key not in attempted), None)


def configuration_key(config):
    return json.dumps({k: v for k, v in config.items()
                       if k not in {"hypothesis", "class_weights"}}, sort_keys=True)


def candidates(profile, history):
    kind = profile["kind"]
    entity_groups = []
    for item in profile.get("entity_groups", []):
        columns = item.get("columns") if isinstance(item, dict) else item
        if isinstance(columns, (list, tuple)) and columns and all(isinstance(column, str) for column in columns):
            if list(columns) not in entity_groups:
                entity_groups.append(list(columns))
    options = []
    def add(family, hypothesis, **kwargs):
        module = {"linear": "sklearn", "extra_trees": "sklearn"}.get(family, family)
        if importlib.util.find_spec(module):
            options.append(dict(family=family, hypothesis=hypothesis, features="basic",
                                target_transform="none", depth=6, learning_rate=.065,
                                iterations=900, l2=5, balanced=False) | kwargs)
    add("catboost", "Native categorical splits establish a nonlinear reference.", iterations=500)
    add("lightgbm", "Leaf-wise trees test local interactions.", iterations=1400, learning_rate=.04)
    if kind == "regression" and entity_groups:
        add("lightgbm", "Train-only entity history tests repeated-entity signal across periods.",
            group_features=entity_groups, history_residual=True, features="extended",
            iterations=1800, learning_rate=.04, num_leaves=63, min_child_samples=20)
        add("lightgbm", "Direct history features with stronger regularization test entity-level signal.",
            group_features=entity_groups, history_residual=False, features="extended",
            iterations=1800, learning_rate=.04, num_leaves=31, min_child_samples=40, l2=15)
        add("lightgbm", "Stable covariates and regularized entity residuals test forward-shift robustness.",
            group_features=entity_groups, history_residual=True, stable_features=True, features="extended",
            iterations=1800, learning_rate=.04, num_leaves=31, min_child_samples=40, l2=15)
        add("lightgbm", "Entity covariance changes and history dispersion test conditional annual changes.",
            group_features=entity_groups, history_residual=True, history_detail=True,
            stable_features=True, features="extended", iterations=2000, learning_rate=.04,
            num_leaves=31, min_child_samples=40, l2=15)
        add("lightgbm", "Approximate entity matches test small changes of continuous identity measurements.",
            group_features=entity_groups, history_residual=True, history_neighbors=True,
            stable_features=True, features="extended", iterations=2000, learning_rate=.04,
            num_leaves=31, min_child_samples=40, l2=15)
    add("lightgbm", "Compare ordinal category encoding with nominal splits.",
        categorical_encoding="ordinal", iterations=1400, learning_rate=.04)
    add("xgboost", "Histogram boosting supplies a complementary regularized tree partition.",
        iterations=1100, learning_rate=.05, depth=6)
    add("extra_trees", "Randomized trees supply a different bias for blending.", iterations=250)
    add("linear", "One-hot linear effects test a complementary additive model.", features="extended", l2=10)
    if kind == "classification":
        if profile["metric"] != "macro_f1":
            add("lightgbm", "Smoother partially balanced trees stopped by log loss test calibration quality.",
                balanced=True, balance_power=.5, features="interactions", num_leaves=31,
                min_child_samples=60, iterations=2400, learning_rate=.04, stopping_metric="log_loss")
            add("lightgbm", "Unweighted longer boosting tests probability quality before decision calibration.",
                features="extended", num_leaves=63, min_child_samples=25,
                iterations=2400, learning_rate=.035, stopping_metric="log_loss")
            add("lightgbm", "Ordinal long boosting with partial balance tests a distinct class boundary.",
                categorical_encoding="ordinal", features="extended", balanced=True, balance_power=.5,
                num_leaves=31, min_child_samples=40, iterations=2400, learning_rate=.035, stopping_metric="log_loss")
        add("catboost", "Square-root class balancing tests minority recall.",
            balanced=True, balance_power=.5, depth=7, iterations=1100)
        add("lightgbm", "Class-balanced leaves test minority recall.", balanced=True,
            balance_power=1., iterations=1300)
        add("lightgbm", "Partial class balancing tests the precision-recall tradeoff.",
            balanced=True, balance_power=.5, iterations=1600, learning_rate=.04)
        add("lightgbm", "Ordinal categories with balancing test an alternative rare-label boundary.",
            categorical_encoding="ordinal", balanced=True, iterations=1300)
    add("catboost", "Deeper trees test interactions behind validation errors.",
        depth=8, iterations=1600, learning_rate=.045)
    add("catboost", "Frequency and skew transforms expose long-tail structure.",
        features="extended", depth=7, iterations=1400)
    add("lightgbm", "More leaves and extended features test underfitting.",
        features="extended", depth=7, iterations=1800, learning_rate=.035)
    add("lightgbm", "Frequency features with ordinal categories test another partition.",
        categorical_encoding="ordinal", features="extended", depth=7, iterations=1800, learning_rate=.035)
    add("lightgbm", "Smaller leaves test local structure with stronger row and feature sampling.",
        depth=8, num_leaves=127, min_child_samples=10, subsample=.8,
        colsample_bytree=.8, iterations=2200, learning_rate=.035)
    add("lightgbm", "Larger leaves test smoother structure and reduce rare-category overfitting.",
        depth=6, num_leaves=31, min_child_samples=60, l2=12, iterations=1800, learning_rate=.04)
    add("xgboost", "Shallower histogram trees test a smoother regularized model.",
        depth=4, l2=12, iterations=1800, learning_rate=.04)
    add("catboost", "Categorical crosses and row statistics test generic interactions.",
        features="interactions", depth=7, iterations=1600, learning_rate=.045)
    add("lightgbm", "Categorical crosses and row statistics test missing interactions.",
        features="interactions", depth=7, iterations=1800, learning_rate=.04)
    if kind == "regression" and min(profile["target_quantiles"].values()) >= 0:
        add("catboost", "Log-target training tests whether high-tail noise dominates squared error.",
            target_transform="log1p", depth=7, iterations=1300)
    add("catboost", "Stronger regularization tests depth overfitting.",
        depth=8, l2=20, iterations=1800, learning_rate=.035)
    add("catboost", "Shallower long boosting tests a smoother response surface.",
        depth=5, l2=8, iterations=2400, learning_rate=.035)
    successful = [row for row in history if row.get("status") == "ok"]
    if successful:
        from research.metrics import utility
        best = max(successful, key=lambda row: utility(profile["metric"], row["metric"]))
        config = dict(best["config"])
        config.update(iterations=min(3500, int(config["iterations"] * 1.5)),
                      learning_rate=max(.02, config["learning_rate"] * .8),
                      hypothesis="Extend the best observed model with a smaller learning rate.")
        options.append(config)
    used = {configuration_key(row["config"]) for row in history}
    options = [config for config in options if configuration_key(config) not in used]
    pending = coverage_pending(history, options, profile)
    if pending:
        options.sort(key=lambda config: coverage_key(config) != pending)
        return options
    if successful:
        counts = {family: sum(row["config"]["family"] == family for row in history)
                  for family in {config["family"] for config in options}}
        recent = [row["config"]["family"] for row in history[-2:]]
        repeated = recent[0] if len(recent) == 2 and recent[0] == recent[1] else None
        options.sort(key=lambda item: (item["family"] == repeated,
                                      counts[item["family"]] >= 5,
                                      item["family"] != best["config"]["family"]))
    return options
