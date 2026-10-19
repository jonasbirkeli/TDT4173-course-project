"""Non-fitting audit of XGBoost topology, physics, history and submission alignment."""
import contextlib
import io
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kaggle_metric import score

def main():
    nb=nbformat.read('xgboost.ipynb',as_version=4)
    ns={}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in nb.cells:
            if cell.cell_type=='code' and cell.id not in {'run','interpretation'}:
                exec(compile(cell.source,cell.id,'exec'),ns)
    assert ns['RECEIVERS']['Tokke']==['Bandak']
    assert ns['RECEIVERS']['Byrte']==['Byrtevatn']
    assert ns['RECEIVERS']['Lio']==['Bandak']
    assert ns['RECEIVERS']['Hogga']==[]
    assert 'Kjela' in ns['UPSTREAM_PLANTS']['Vinje']
    assert 'Tokke' in ns['DOWNSTREAM_PLANTS']['Songa']
    cases=pd.concat([ns['train_cases'].iloc[[0,1,500]],ns['test_cases'].iloc[[0,-1]]],ignore_index=True)
    records=[]
    for preset,spec in ns['EXPERIMENTS'].items():
        ns['FEATURE_VERSION']=spec['version']
        ns['ADDITIONAL_GROUPS']=set(spec['groups'])
        base=ns['feature_matrix'](cases)
        single=ns['feature_matrix'](cases.iloc[[0]])
        schema=None
        for generator in ns['GENERATORS']:
            frame=ns['generator_features'](base,generator)
            alone=ns['generator_features'](single,generator)
            np.testing.assert_allclose(frame.iloc[:168].to_numpy(),alone.to_numpy(),rtol=0,atol=0)
            assert np.isfinite(frame.to_numpy()).all()
            assert not set(['Run No','starttime','total_calculation_time']) & set(frame)
            if schema is None:schema=frame.columns.tolist()
            assert schema==frame.columns.tolist()
        records.append({'preset':preset,'features':len(schema),'feature_signature':__import__('hashlib').sha256(json.dumps(schema).encode()).hexdigest()})
        if preset=='decision_reference':
            assert len(schema)==216
            assert records[-1]['feature_signature']=='413b34b792c90dc7fb586250679bd5438ef0dc4e90ef8b07a8bdb6331b6c2af9'
        # One case's features must not depend on observed storage after its start.
        focus=ns['train_cases'].iloc[[500]]
        before=ns['generator_features'](ns['feature_matrix'](focus),'Tokke_G1')
        volumes=ns['tables']['volumes'].copy()
        mask=ns['tables']['volumes'].index>focus.starttime.iloc[0]
        ns['tables']['volumes'].loc[mask]*=0.5
        after=ns['generator_features'](ns['feature_matrix'](focus),'Tokke_G1')
        np.testing.assert_array_equal(before.to_numpy(),after.to_numpy())
        ns['tables']['volumes']=volumes
        print(preset,'passed',len(schema),'features',flush=True)
    ns['FEATURE_VERSION']='xgb-topology-physics-v1'
    ns['ADDITIONAL_GROUPS']={'topology','physics'}
    probe=ns['feature_matrix'](cases)
    frame=ns['generator_features'](probe,'Tokke_G1')
    # Energy/discharge conversion: 1 Mm3 at factor MWh/Mm3 must satisfy P/Q.
    q=frame.physics_minimum_discharge_m3s.to_numpy()
    factor=frame.physics_minimum_energy_MWh_per_Mm3.to_numpy()
    power=float(ns['GEN_SPECS']['Tokke_G1']['p_min'])
    np.testing.assert_allclose(factor,power*1e6/(q*3600),rtol=2e-6)
    # Supply curve interpolation and receiving tailwater at S only.
    expected=np.maximum(ns['tables']['volumes'].reindex(pd.DatetimeIndex(cases.starttime))['Bandak'].map(lambda v:np.interp(v,*ns['LEVEL_CURVES']['Bandak'])),72)
    np.testing.assert_allclose(frame.topo_initial_tailwater_m.to_numpy().reshape(-1,168)[:,0],expected,rtol=2e-6)
    dummy={g:np.full((725,168),0.37,dtype=np.float32) for g in ns['GENERATORS']}
    mapped=ns['assemble_predictions'](ns['test_cases'],dummy)
    assert mapped.shape==ns['template'].shape and mapped.columns.equals(ns['template'].columns)
    np.testing.assert_array_equal(mapped['Run No'],ns['template']['Run No'])
    fitting,validation=ns['chronological_split']('2021-01-01','2022-01-01')
    assert (fitting.starttime+pd.Timedelta(hours=168)).max()<=validation.starttime.min()
    focus=validation.iloc[:3]
    rng=np.random.default_rng(42)
    values={g:rng.uniform(.01,.99,(3,168)).astype(np.float32) for g in ns['GENERATORS']}
    measured,_,output=ns['evaluate'](focus,values)
    assert abs(measured-score(focus[['Run No']+ns['LABELS']].copy(),output.copy()))<1e-12
    Path('outputs/xgboost_development/feature_checks.json').write_text(json.dumps({'status':'passed','presets':records,'models_fitted':False},indent=2),encoding='utf-8')
    print('Topology, case isolation, future-volume invariance, physical units, metric parity and725-row mapping passed.')

if __name__=='__main__':main()
