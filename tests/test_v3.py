import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from research.features import NeighborHistory, FeatureBuilder
from research import worker
from research.planner import QwenPlanner


def test_approximate_history_excludes_self_and_survives_serialization(tmp_path):
    x = pd.DataFrame({'size': [100.,101.,102.,1000.], 'type': ['a','a','a','b']})
    y = np.array([10.,20.,30.,80.])
    memory = NeighborHistory(['size','type'], ['size']).fit(x,y)
    # Tiny test has low cardinality: no approximate linker is activated.
    assert np.isfinite(memory.transform(x)[0]).all()
    x = pd.DataFrame({'size': np.arange(100.,180.), 'type': ['a']*80})
    y = np.arange(80.)
    first = NeighborHistory(['size','type'], ['size']).fit(x,y)
    train = first.transform(x,training=True)[0]
    changed = y.copy();changed[20] = 100000
    second = NeighborHistory(['size','type'], ['size']).fit(x,changed)
    assert train[20] == pytest.approx(second.transform(x,training=True)[0][20])
    query = pd.DataFrame({'size': [120.1,130.1,3000.], 'type':['a','a','new']})
    prediction = first.transform(query)
    assert prediction[1][0] > 0 and prediction[1][-1] == 0
    joblib.dump(first,tmp_path/'history.joblib')
    np.testing.assert_allclose(prediction[0],joblib.load(tmp_path/'history.joblib').transform(query)[0])


def test_history_variance_excludes_own_label():
    x=pd.DataFrame({'size':[100.,100.,100.,200.],'type':['a','a','a','b']})
    a=FeatureBuilder(groups=[['size','type']],history_detail=True).fit_transform(x,[10.,20.,30.,40.])
    b=FeatureBuilder(groups=[['size','type']],history_detail=True).fit_transform(x,[999.,20.,30.,40.])
    assert a.loc[0,'__history_0_std']==pytest.approx(5.)
    assert a.loc[0,'__history_0_std']==b.loc[0,'__history_0_std']


class PredictionModel:
    weights=np.array([])
    def predict(self,frame):return frame['value'].to_numpy()*2


def test_temporal_refit_uses_calibration_but_never_validation_labels(tmp_path,monkeypatch):
    x=pd.DataFrame({'value':np.arange(12.)})
    data=dict(train=x,y=x.value.to_numpy()*2,test=x.iloc[:2],fit_indices=np.arange(6),cal_indices=np.arange(6,9),validation_indices=np.arange(9,12),refit_before_validation=True,kind='regression',metric='rmse',n_classes=0)
    path=tmp_path/'data.joblib';joblib.dump(data,path)
    spec=dict(bundle=str(path),output=str(tmp_path/'run'),final=False,config=dict(family='lightgbm',iterations=500))
    specpath=tmp_path/'experiment.json';specpath.write_text(json.dumps(spec))
    calls=[]
    def train(bundle,config,final):
        rows=np.arange(len(bundle['train'])) if final else bundle['fit_indices']
        calls.append((bundle['train'].iloc[rows].value.tolist(),bundle['y'][rows].tolist(),dict(config),final))
        return PredictionModel(),30
    monkeypatch.setattr(worker,'train_model',train)
    worker.execute(specpath)
    assert calls[0][0]==list(np.arange(6.)) and calls[0][-1] is False
    assert calls[1][0]==list(np.arange(9.)) and calls[1][-1] is True
    assert calls[1][2]['iterations']==30
    assert all(max(call[0])<9 for call in calls)
    assert json.loads((tmp_path/'run/result.json').read_text())['metric']==0.


def test_new_configuration_fields_validate_strictly():
    cfg=dict(family='lightgbm',group_features=[['size','type']],history_neighbors=True,history_detail=True,stopping_metric='log_loss')
    assert QwenPlanner._configuration(cfg) is not None
    assert QwenPlanner._configuration(dict(cfg,history_neighbors='true')) is None
    assert QwenPlanner._configuration(dict(cfg,stopping_metric='shell')) is None
    assert QwenPlanner._configuration(dict(cfg,group_features=[])) is None
from research.search import candidates
from research.engine import final_reserve


def test_forward_shift_prioritizes_history_before_expensive_families(monkeypatch):
    monkeypatch.setattr('research.search.importlib.util.find_spec',lambda name:object())
    profile=dict(kind='regression',metric='rmse',target_quantiles={0.:1.,1.:10.},entity_groups=[['size','type']],temporal_shift=[dict(column='period')])
    first=candidates(profile,[])[0]
    assert first['group_features']==[['size','type']] and first['stable_features']


def test_macro_f1_rejects_regressed_candidate_but_probability_tasks_can_search_it(monkeypatch):
    monkeypatch.setattr('research.search.importlib.util.find_spec',lambda name:object())
    profile=dict(kind='classification',metric='macro_f1',class_counts={'major':1000,'minor':10})
    menu=candidates(profile,[])
    assert menu[0]['family']=='catboost'
    assert not any(c.get('stopping_metric')=='log_loss' for c in menu)
    assert any(c.get('stopping_metric')=='log_loss' for c in candidates(dict(profile,metric='log_loss'),[]))


def test_temporal_reserve_does_not_charge_for_nonexistent_oof_folds():
    rows=[dict(elapsed_seconds=40.,metric=1.)]
    assert final_reserve(600,rows,35000,11000,'rmse',temporal=True)<final_reserve(600,rows,35000,11000,'rmse')