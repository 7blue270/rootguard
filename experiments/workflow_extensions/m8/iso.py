"""Scan the original 54 budgets for the follow-up policies on the main new cohort."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .study import (OUT,HERE,OLD,REF,SEED,ORIGINAL,OLDPLANS,Context,data,sim,prep,Branch,BUDGETS,write)
def main():
    archive=OLD.parent/'physionet_fusion_20261120'
    protocols=json.loads((archive/'protocol.json').read_text())
    src=Path('data_mine/finite_cache_physionet2019')
    assert protocols['dataset']==data.input_hashes(src,['BUN_CR'])
    by={r['subject_id']:r for r in data.load(src,['BUN_CR'])}
    split=json.loads((src/'split.json').read_text()); ids=sorted(s for s in split if split[s]=='test')
    ctx=Context('BUN_CR',[by[s] for s in protocols['references']],SEED);prep(ctx,[by[s] for s in ids])
    frozen=json.loads((OUT/'PhysioNet-frozen.json').read_text()); known=defaultdict(dict)
    with (archive/'grid.csv').open() as f:
        for row in csv.DictReader(f):
            known[float(row['B']),row['subject']][row['method']]=float(row['mae'])
    vals=defaultdict(list)
    for B in protocols['grid']:
        near=min(BUDGETS,key=lambda a:abs(np.log(a/B)))
        per=defaultdict(list)
        for s in ids:
            r=by[s];key=f"{near}/{int(len(r['events'])>=8)}"
            old=known[B,s]; arms=ORIGINAL[key]
            cache={Branch(**arms[m]):old[m] for m in ('static','periodic','route')}
            cache[Branch(**OLDPLANS['BUN_CR'][str(B)]['branches']['R-new'])]=old['oldroute']
            for m in ('static','periodic','oldroute','route'):per[m].append(old[m])
            base=Branch(**frozen[key]['m8'])
            for m,b in [('m8',base),('ceiling',Branch(**arms['dynamic']) if near==960 else base)]:
                if b not in cache:
                    res=sim(ctx,r,B,b,4,False)
                    cache[b]=abs(ctx.work.from_cache(res['caches'])[ctx.work.target]-r['_y']).mean()
                per[m].append(cache[b])
        for k,v in per.items():vals[k].append(v)
    thresholds=json.loads((REF/'targets.json').read_text())['BUN_CR']
    grid=np.array(protocols['grid']);rows=[]
    for group in ('all','high'):
        mask=np.array([group=='all' or by[s]['_span']>protocols['threshold'] for s in ids])
        for m,values in vals.items():
            curve=np.array(values)[:,mask].mean(axis=1)
            for reg,t in thresholds.items():
                for side in ('lo','hi'):
                    tau=t['tau_'+side];hits=grid[curve<=tau]
                    rows.append(dict(group=group,method=m,regime=reg,side=side,tau=tau,B_min=hits[0] if len(hits) else None))
    write(OUT/'iso.csv',rows)
if __name__=='__main__':main()
