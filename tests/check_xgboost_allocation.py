"""Benchmark and check the deterministic allocation feature calculation without classifier fitting."""
import contextlib
import io
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import nbformat

def main():
    nb=nbformat.read('xgboost.ipynb',as_version=4)
    ns={}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in nb.cells:
            if cell.cell_type=='code' and cell.id in {'configuration','schema','raw-inputs','features'}:
                source=cell.source.replace('EXPERIMENT = \'decision_reference\'','EXPERIMENT = \'topology_allocation\'').replace('EXPERIMENT = "decision_reference"','EXPERIMENT = "topology_allocation"')
                exec(compile(source,cell.id,'exec'),ns)
    examples=pd.concat([ns['train_cases'].iloc[np.linspace(0,2550,18,dtype=int)],ns['test_cases'].iloc[[0,-1]]],ignore_index=True)
    started=perf_counter()
    result=[ns['allocation_case'](s) for s in examples.starttime]
    seconds=perf_counter()-started
    assert all(r['success'] for r in result)
    assert max(r['equality_error'] for r in result)<1e-5
    assert all(np.isfinite(r['storage']).all() and (r['storage']>=-1e-8).all() for r in result)
    assert all(np.isfinite(r['production_fraction']).all() and (r['production_fraction']>=-1e-8).all() and (r['production_fraction']<=1+1e-7).all() for r in result)
    target=examples.iloc[[6]]
    base=ns['feature_matrix'](target)
    before=ns['generator_features'](base,'Tokke_G1')
    original=ns['tables']['volumes'].copy()
    mask=ns['tables']['volumes'].index>target.starttime.iloc[0]
    ns['tables']['volumes'].loc[mask]*=0.5
    ns['ALLOCATION_CACHE'].clear()
    after=ns['generator_features'](ns['feature_matrix'](target),'Tokke_G1')
    np.testing.assert_array_equal(before.to_numpy(),after.to_numpy())
    ns['tables']['volumes']=original
    ns['ALLOCATION_CACHE'].clear()
    large=ns['generator_features'](ns['feature_matrix'](examples),'Tokke_G1')
    ns['ALLOCATION_CACHE'].clear()
    alone=ns['generator_features'](ns['feature_matrix'](examples.iloc[[0]]),'Tokke_G1')
    np.testing.assert_array_equal(large.iloc[:168].to_numpy(),alone.to_numpy())
    summary={'status':'passed','cases':len(examples),'seconds':seconds,'seconds_per_case':seconds/len(examples),'features':len(before.columns),'max_water_balance_residual_Mm3':max(r['equality_error'] for r in result),'constraint_violation_cases':sum(np.any(r['violations']>1e-7) for r in result),'classifiers_fitted':False,'model':'12-hour linear flow relaxation; initial-head production factors, approximate global terminal value, soft storage limits; no startup cost or exact routing/head-loss physics.'}
    summary['constraint_violation_cases'] = int(summary['constraint_violation_cases'])
    Path('outputs/xgboost_development/allocation_checks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
