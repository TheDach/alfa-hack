import importlib.util
import json


def configuration_key(config):
    return json.dumps({k: v for k, v in config.items() if k not in {"hypothesis", "class_weights"}}, sort_keys=True)


def candidates(profile, history):
    kind = profile["kind"]
    options = []
    def add(family, hypothesis, **kwargs):
        if importlib.util.find_spec({"linear": "sklearn", "extra_trees": "sklearn"}.get(family, family)):
            options.append(dict(family=family, hypothesis=hypothesis, features="basic", target_transform="none",
                                depth=6, learning_rate=.065, iterations=900, l2=5, balanced=False, **{} ) | kwargs)
    add("catboost", "Native categorical splits establish a nonlinear reference.", iterations=500)
    add("lightgbm", "Leaf-wise trees may separate local interactions missed by symmetric trees.", iterations=1400, learning_rate=.04)
    if kind == "classification":
        add("catboost", "Reduce majority-class dominance with square-root balancing.", balanced=True, depth=7, iterations=1100)
        add("lightgbm", "Class-balanced leaf-wise trees may improve minority recall.", balanced=True, iterations=1300)
    add("catboost", "Deeper trees can capture interactions behind the largest validation errors.", depth=8, iterations=1600, learning_rate=.045)
    add("catboost", "Frequency and skew transforms can expose rare-category and long-tail structure.", features="extended", depth=7, iterations=1400)
    add("lightgbm", "More leaves and generated features can reduce underfitting.", features="extended", depth=7, iterations=1800, learning_rate=.035)
    add("extra_trees", "Randomized trees provide a different bias and a candidate for blending.", iterations=250)
    add("linear", "A regularized linear model tests whether global additive structure is sufficient.", features="extended", l2=10)
    if kind == "regression" and min(profile["target_quantiles"].values()) >= 0:
        add("catboost", "Log-target training tests whether high-tail noise dominates the squared-error objective.", target_transform="log1p", depth=7, iterations=1300)
    add("catboost", "Stronger regularization tests whether previous depth increases overfit.", depth=8, l2=20, iterations=1800, learning_rate=.035)
    add("catboost", "A shallow, longer boosting run tests a smoother response surface.", depth=5, l2=8, iterations=2400, learning_rate=.035)
    successful = [row for row in history if row.get("status") == "ok"]
    if successful:
        from research.metrics import utility
        best = max(successful, key=lambda row: utility(profile["metric"], row["metric"]))
        config = dict(best["config"])
        config.update(iterations=min(3500, int(config["iterations"] * 1.5)), learning_rate=max(.02, config["learning_rate"] * .8),
                      hypothesis="Extend the best observed model with a smaller learning rate.")
        options.append(config)
        if len(successful) >= 3:
            options.sort(key=lambda item: (item["family"] != best["config"]["family"], item.get("balanced", False) is False))
    used = {configuration_key(row["config"]) for row in history}
    return [config for config in options if configuration_key(config) not in used]
