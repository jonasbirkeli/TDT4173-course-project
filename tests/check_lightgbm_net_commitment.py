"""Check frozen features, DP, year isolation and held-label handling without fitting."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import copy
import importlib.util
import io
import itertools
import json

import lightgbm as lgb
import nbformat
import numpy as np


def forbidden(*args, **kwargs):
    raise AssertionError("Real classifier fitting or pretrained loading is forbidden in this check.")


def main():
    source=Path("analysis/lightgbm_net_commitment_20261102.ipynb")
    document=nbformat.read(source,as_version=4)
    lgb.train=forbidden
    lgb.LGBMClassifier.fit=forbidden
    booster_init=lgb.Booster.__init__
    def guarded_init(self,*args,**kwargs):
        if kwargs.get("model_file") is not None or kwargs.get("model_str") is not None:
            forbidden()
        booster_init(self,*args,**kwargs)
    lgb.Booster.__init__=guarded_init
    ns={"__name__":"nonfitting_net_check"}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in document.cells[:-1]:
            if cell.cell_type=="code":exec(compile(cell.source,str(source),"exec"),ns)
    c=ns["SELECTED_CONFIGURATION"]
    assert isinstance(c,dict), "Publish a measured frozen configuration first."
    ns["guard_selected"](c)
    changed=copy.deepcopy(c)
    changed["iterations"][ns["GENERATORS"][0]]+=1
    try:ns["guard_selected"](changed)
    except ValueError:pass
    else:raise AssertionError("A modified frozen tree budget was accepted.")
    original_step=ns["LP_STEP"]
    ns["LP_STEP"]=12
    try:
        try:ns["guard_selected"](c)
        except ValueError:pass
        else:raise AssertionError("An altered LP grid was accepted.")
    finally:ns["LP_STEP"]=original_step
    rng=np.random.default_rng(17)
    profit=rng.normal(size=(1,6,2))
    start,stop=np.array([.2,.7]),np.array([.4,.1])
    path,gap=ns["commitment_dp"](profit,start,stop)
    for scenario in range(2):
        paths=[]
        for states in itertools.product([0,1],repeat=6):
            value=sum(profit[0,h,scenario]*states[h] for h in range(6))
            value-=sum(start[scenario] for h in range(1,6) if states[h] and not states[h-1])
            value-=sum(stop[scenario] for h in range(1,6) if not states[h] and states[h-1])
            paths.append((states,value))
        selected=tuple(path[0,:,scenario].astype(int))
        assert abs(next(v for s,v in paths if s==selected)-max(v for _,v in paths))<1e-10
        for h in range(6):
            expected=max(v for s,v in paths if s[h])-max(v for s,v in paths if not s[h])
            assert abs(gap[0,h,scenario]-expected)<1e-10
    cases=ns["train_cases"].iloc[[0,365,1826,2557]]
    frame=ns["feature_matrix"](cases[["Run No","starttime"]])
    inventory={}
    for g in ns["GENERATORS"]:
        x=ns["model_features"](frame,g,c["variant"])
        assert x.columns.is_unique and np.isfinite(x.to_numpy()).all()
        inventory[g]=list(x.columns)
    for variant in dict.fromkeys([c["variant"],"net_dp"]):
        chosen=cases.iloc[[2]]
        ns["NET_FEATURE_CACHE"].clear()
        expected=ns["model_features"](ns["feature_matrix"](chosen),"Lio_G1",variant)
        original=ns["tables"]["volumes"].copy()
        try:
            ns["tables"]["volumes"].loc[ns["tables"]["volumes"].index>chosen.starttime.iloc[0]]*=.25
            ns["LP_CACHE"].clear();ns["NET_FEATURE_CACHE"].clear()
            actual=ns["model_features"](ns["feature_matrix"](chosen),"Lio_G1",variant)
            np.testing.assert_array_equal(expected.to_numpy(),actual.to_numpy())
        finally:
            ns["tables"]["volumes"]=original
            ns["LP_CACHE"].clear();ns["NET_FEATURE_CACHE"].clear()
        # Perturb every exogenous timestamp outside this case's allowed window.
        # Terminal values are available only at S+168h; observed storage only at S.
        preserved={key:table.copy() for key,table in ns["tables"].items()}
        start=chosen.starttime.iloc[0]
        end=start+ns["pd"].Timedelta(hours=168)
        try:
            for key,table in ns["tables"].items():
                outside=(table.index!=start) if key=="volumes" else (table.index!=end) if key=="water_values" else ((table.index<start)|(table.index>=end))
                table.loc[outside]=table.loc[outside]+17
            ns["LP_CACHE"].clear();ns["NET_FEATURE_CACHE"].clear()
            actual=ns["model_features"](ns["feature_matrix"](chosen),"Lio_G1",variant)
            np.testing.assert_array_equal(expected.to_numpy(),actual.to_numpy())
        finally:
            ns["tables"].update(preserved)
            ns["LP_CACHE"].clear();ns["NET_FEATURE_CACHE"].clear()
        flipped=chosen.copy()
        flipped[ns["LABELS"]]=1-flipped[ns["LABELS"]]
        actual=ns["model_features"](ns["feature_matrix"](flipped),"Lio_G1",variant)
        np.testing.assert_array_equal(expected.to_numpy(),actual.to_numpy())
        ns["NET_FEATURE_CACHE"].clear()
        together=ns["model_features"](frame,"Lio_G1",variant).iloc[336:504]
        np.testing.assert_array_equal(expected.to_numpy(),together.to_numpy())
    ids=[]
    for year in range(2015,2023):
        fit,score=ns["loyo_partition"](year)
        start_year=ns["pd"].Timestamp(f"{year}-01-01",tz="UTC")
        end_year=ns["pd"].Timestamp(f"{year+1}-01-01",tz="UTC")
        assert not ((fit.starttime<end_year)&(fit.starttime+ns["pd"].Timedelta(hours=168)>start_year)).any()
        assert ((score.starttime>=start_year)&(score.starttime+ns["pd"].Timedelta(hours=168)<=end_year)).all()
        ids.extend(score["Run No"].tolist())
    assert len(ids)==len(set(ids))==2874
    output=Path("outputs/lightgbm_net_checks")/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True,exist_ok=False)
    # Exercise the actual fit adapter with one fake learner: fixed-budget mode
    # must build exactly one fitting Dataset and no held-label evaluation Dataset.
    fitting=ns["train_cases"].iloc[[0,400,1500]]
    scoring=ns["train_cases"].iloc[[1826,1830]]
    unit=ns["GENERATORS"][0]
    datasets=[]
    class Dataset:
        def __init__(self,data,label,**kwargs):
            self.data=data;self.label=np.asarray(label);datasets.append(self)
    calls=[]
    def fake_train(parameters,dataset,num_boost_round,**kwargs):
        assert len(datasets)==1
        np.testing.assert_array_equal(dataset.label,ns["labels_for"](fitting,unit))
        assert kwargs["valid_sets"] is None and kwargs["valid_names"] is None
        assert not kwargs.get("init_model")
        assert num_boost_round==c["iterations"][unit]
        calls.append(parameters)
        class Model:
            best_iteration=num_boost_round
            def current_iteration(self):return num_boost_round
            def predict(self,x,**kw):return np.full(len(x),.5)
            def save_model(self,path,**kw):Path(path).write_text("Mock only; no fitted classifier weights.")
            def feature_importance(self,*a,**kw):return np.zeros(dataset.data.shape[1])
        return Model()
    real_dataset=lgb.Dataset
    lgb.Dataset=Dataset;lgb.train=fake_train
    try:
        ns["fit_units"](fitting,scoring,c["variant"],output/"mock_fixed_fit",rounds=c["iterations"],units=[unit])
    finally:
        lgb.Dataset=real_dataset;lgb.train=forbidden
    assert len(calls)==1
    spec=importlib.util.spec_from_file_location("course_metric",Path("kaggle_metric.py"))
    metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
    predictions={g:rng.random((len(cases),168)).astype(np.float32) for g in ns["GENERATORS"]}
    expected=metric.score(cases[["Run No",*ns["LABELS"]]],ns["assemble"](cases,predictions))
    assert ns["official_score"](cases,predictions)==expected
    result=dict(status="passed",classifier_fits=0,mock_train_calls=1,fixed_fit_has_no_held_label_dataset=True,
                dp_matches_exhaustive_search=True,future_storage_invariance=True,label_invariance=True,
                outside_case_exogenous_invariance=True,case_isolation=True,loyo_unique_scoring_cases=2874,frozen_guard=True,official_metric_matches=True,
                features={g:len(v) for g,v in inventory.items()},output=str(output))
    ns["write_json"](output/"check.json",result)
    Path("outputs/lightgbm_net_checks/latest.json").write_text(json.dumps(result,indent=2))
    print(result)


if __name__=="__main__":main()
