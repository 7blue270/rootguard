"""Use identical paid grid releases to audit exact sampler and replay wiring."""
import json,hashlib
from fractions import Fraction
from unittest.mock import patch
import numpy as np
from .study import OUT,COHORTS,setup,sim,dump,BUDGETS,DEFAULT_SPLIT
from pathlib import Path
from .policy import run,estimate,plans
def main():
    rows=[];inputs={}
    for name in COHORTS:
        ctx,(train,_,_),_=setup(name);cfg=json.loads((OUT/name/'frozen.json').read_text())
        folder,t=COHORTS[name];src=Path('data_mine')/folder
        split=DEFAULT_SPLIT if name.startswith('old') else src/'split.json'
        files=[src/(t+'.jsonl'),split,OUT/name/'frozen.json']
        inputs[name]={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        r=train[0];public=r['_public'];truth=ctx.work.denormalize(r['_x'])
        for arm,b in plans(ctx,cfg['package']).items():
            if arm=='period':continue
            for B in (80.,960.):
                ps=sim(ctx,r,B,b,1,True)
                actions=[(1,ctx.roots,ps['anchor_cache'][0])]+[(h['turn'],tuple(np.flatnonzero(h['selected'][0])),h['cache_after'][0]) for h in ps['histories']]
                permitted=[(t,[ctx.work.roots[i] for i in ids]) for t,ids,_ in actions]
                expected=[round(raw[i]*999) for _,ids,raw in actions for i in ids]
                def reader(turn,names):
                    assert permitted.pop(0)==(turn,names)
                    return [float(truth[turn-1,ctx.work.roots.index(n)]) for n in names]
                conf={**cfg,'route':{str(B):{'0':arm,'1':arm}}}
                with patch('experiments.workflow_extensions.m9.policy.sample',side_effect=expected):
                    res=run(ctx,conf,B,public,reader)
                assert not permitted and res['spent']==Fraction(str(B))
                co=cfg['models'][str(B)]['static' if arm=='static' else 'period']
                eff=estimate(ctx,ps,{bool(int(k)):np.asarray(v) for k,v in co.items()},np.asarray(cfg['median']))[0]
                error=np.max(abs(eff-res['cache']))
                assert error<1e-8,(name,arm,error)
                rows.append(dict(cohort=name,arm=arm,B=B,error=error))
        # One real, non-seeded execution validates the exact-kernel entry point.
        run(ctx,cfg,80,public,lambda turn,names:[float(truth[turn-1,ctx.work.roots.index(n)]) for n in names])
    dump(OUT/'input_hashes.json',inputs)
    dump(OUT/'audit.json',dict(cases=len(rows),real_runs=len(COHORTS),max_cache_delta=max(v['error'] for v in rows)))
if __name__=='__main__':main()
