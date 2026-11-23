"""Same-period ablation of the M7 workload gate using saved paired paths."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from ..finite_cache import data
from ..m4_carry.analysis import write_csv
from ..m4_carry.core import key_seed
from ..m4_carry.run import ROOT,line
from .analysis import derived,mean_valid,quantile
from .audit import ledger
from .engine import Branch, Context, execute
from .metrics import utility


def run(source, root, out):
    out.mkdir(parents=True,exist_ok=False)
    read=lambda name:json.loads((source/name).read_text())
    conf,split,frozen=read('config.json'),read('split.json'),read('frozen_routes.json')
    assert data.input_hashes(root,conf['templates'])==read('input_hashes.json')
    records=data.load(root,conf['templates'])
    actual=defaultdict(dict)
    with (source/'metrics.jsonl').open() as f:
        for text in f:
            row=json.loads(text)
            if row['method']=='M7-D':
                actual[row['template'],row['B']][row['subject'],row['repeat']]=row
    output=[]
    with (out/'matched_final_paths.jsonl').open('w') as log:
        for template in conf['templates']:
            ctx=Context(template,data.partition(records,split,'train',template),conf['noise_seed'])
            held=sorted(data.partition(records,split,'test',template),key=lambda rec:str(rec['subject_id']))
            rng=np.random.default_rng(key_seed('m7-bootstrap',template))
            size,R=len(held),conf['repeats']
            n=conf['bootstrap']
            p=rng.integers(size,size=(n,size,1))
            r=rng.integers(R,size=(n,size,R))
            for B in conf['budgets']:
                plan=Branch(**frozen[template]['routes']['workload']['branches'][str(B)][1])
                control=replace(plan,gate='calibrated_final')
                comparison=[]
                for rec in held:
                    result=execute(rec,B,control,ctx,conf['eval_phase'],R,True)
                    ledger(result,B,control)
                    by_field=utility(ctx,rec,result)
                    rows=[{k:float(v[j]) for k,v in by_field.items()} for j in range(R)]
                    comparison.append(rows)
                    for j,row in enumerate(rows):
                        line(log,dict(row,template=template,B=B,subject=str(rec['subject_id']),repeat=j))
                pairs=actual[template,B]
                fixed=[[pairs[str(rec['subject_id']),r] for r in range(R)] for rec in held]
                names=sorted(comparison[0][0])

                def metrics(paths):
                    panel=np.array([[[np.nan if q.get(k) is None else q[k] for k in names]
                                     for q in rows] for rows in paths],float)
                    point=dict(zip(names,mean_valid(panel,axis=(0,1))))
                    sims=dict(zip(names,mean_valid(panel[p,r],axis=(1,2)).T))
                    return derived(point,ctx),derived(sims,ctx)

                a,z=metrics(fixed)
                b,x=metrics(comparison)
                for k in ('mae','wmape','rce','intermediate_wmape','final_query_wmape','query_objective','refreshes','spent'):
                    if k not in a or not np.isfinite(a[k]) or not np.isfinite(b[k]):
                        continue
                    lo,hi=quantile(z[k]-x[k])
                    output.append(dict(template=template,B=B,period=plan.period,metric=k,workload_gate=float(a[k]),
                        final_gate=float(b[k]),delta=float(a[k]-b[k]),lo95=lo,hi95=hi,patients=size,repeats=R))
    write_csv(out/'matched_gate.csv',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=ROOT/'data_mine/finite_cache')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.data,args.output)
