import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from research.features import FeatureBuilder
from research.worker import ClassificationMetric, train_model
from research.metrics import evaluate
from research.planner import QwenPlanner


def test_entity_statistics_exclude_the_training_rows_own_target():
    x = pd.DataFrame({"area": [100.,100.,200.,200.], "kind": ["a","a","b","b"], "value": [1.,2.,3.,4.]})
    groups = [["area","kind"]]
    first = FeatureBuilder(groups=groups).fit_transform(x, [10.,20.,30.,40.])
    changed = FeatureBuilder(groups=groups).fit_transform(x, [999.,20.,30.,40.])
    assert first["__history_0_mean"].tolist() == [20.,10.,40.,30.]
    assert first["__history_0_mean"].iloc[0] == changed["__history_0_mean"].iloc[0]
    assert first["__history_0_count"].tolist() == [1.,1.,1.,1.]


def test_entity_statistics_do_not_use_unseen_rows_or_future_labels():
    fit = pd.DataFrame({"area":[100.,100.], "kind":["a","a"]})
    query = pd.DataFrame({"area":[100,999], "kind":["a","new"]})
    builder = FeatureBuilder(groups=[["area","kind"]]).fit(fit, [10.,20.])
    first = builder.transform(query)
    assert first["__history_0_mean"].tolist() == [15.,15.]
    assert first["__history_0_count"].tolist() == [2.,0.]
    pd.testing.assert_frame_equal(first, builder.transform(query))


@pytest.mark.parametrize("metric", ["macro_f1", "weighted_f1", "accuracy", "roc_auc", "average_precision"])
def test_backend_classification_metrics_match_target_metric(metric):
    y = np.array([0,0,1,1,0,1])
    p = np.array([[.8,.2],[.3,.7],[.1,.9],[.6,.4],[.7,.3],[.2,.8]])
    name, actual, higher = ClassificationMetric(metric,2)(y,p)
    assert name == metric and higher
    assert actual == pytest.approx(evaluate(metric,y,p,2))
    custom = ClassificationMetric(metric,2,True)
    assert custom.__name__
    assert custom(y,p) == pytest.approx(1-actual)


@pytest.mark.parametrize("family", ["catboost","lightgbm","xgboost","linear","extra_trees"])
def test_history_residual_pipeline_survives_refit_and_reload(family, tmp_path):
    rng=np.random.default_rng(7)
    x=pd.DataFrame({"area": np.repeat(np.arange(30),4), "kind":np.tile(["a","b"],60), "x":rng.normal(size=120)})
    y=20+x.area.to_numpy()*3+x.x.to_numpy()
    bundle=dict(train=x, test=x.iloc[:3], y=y, fit_indices=np.arange(80),cal_indices=np.arange(80,100),
        categorical=["kind"],kind="regression",metric="rmse",n_classes=0,threads=1,seed=42)
    config=dict(family=family,features="extended",iterations=15,depth=3,
                group_features=[["area","kind"]],history_residual=True)
    model,_=train_model(bundle,config,True)
    before=model.predict(x.iloc[100:])
    joblib.dump(model,tmp_path/"model.joblib")
    assert np.isfinite(before).all()
    np.testing.assert_allclose(before,joblib.load(tmp_path/"model.joblib").predict(x.iloc[100:]))


@pytest.mark.parametrize("family", ["catboost", "xgboost"])
def test_classification_custom_metric_and_constant_features(family):
    rng=np.random.default_rng(9)
    x=pd.DataFrame({"x":rng.normal(size=120),"kind":np.tile(["a","b"],60)})
    y=(x.x.to_numpy()>0).astype(int)
    bundle=dict(train=x, y=y, fit_indices=np.arange(80),cal_indices=np.arange(80,100),
        categorical=["kind"],kind="classification",metric="macro_f1",n_classes=2,threads=1,seed=42)
    model,_=train_model(bundle,dict(family=family,iterations=10,depth=3))
    assert model.predict(x.iloc[100:]).shape == (20,2)
    bundle["train"] = pd.DataFrame({"x":np.zeros(120),"kind":["constant"]*120})
    model,_=train_model(bundle,dict(family=family,iterations=10,depth=3))
    np.testing.assert_allclose(model.predict(bundle["train"].iloc[:5]).sum(axis=1),1)


def test_stability_filter_is_saved_in_model_and_preserves_group_keys():
    x=pd.DataFrame({"area":np.tile([1.,2.],225),"kind":np.tile(["a","b"],225),
                    "period":np.repeat([1.,2.,3.],150),"x":np.arange(450,dtype=float)})
    y=np.sin(x.x.to_numpy())+x.area.to_numpy()
    bundle=dict(train=x, test=x.iloc[300:],y=y,fit_indices=np.arange(150),cal_indices=np.arange(150,300),
        categorical=["kind"],kind="regression",metric="rmse",n_classes=0,threads=1,seed=42)
    model,_=train_model(bundle,dict(family="lightgbm",iterations=10,stable_features=True,
                                  group_features=[["area","kind"]],history_residual=True,
                                  feature_ops=[{"op":"log1p","left":"period"}]))
    assert not model.builder.operations
    assert "period" not in model.builder.columns
    assert {"area","kind"} <= set(model.builder.columns)
    assert np.isfinite(model.predict(x.iloc[160:])) .all()

