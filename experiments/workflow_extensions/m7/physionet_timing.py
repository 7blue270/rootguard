"""Warm single-session CPU-side latency on development fixtures."""
from time import perf_counter_ns
import numpy as np
from .physionet_study import DIR,OUT,SEED,BUDGETS,read,order,write
from .physionet_policy import choose,run
from .engine import Context,public_view
from ..finite_cache import data

def main():
    frozen=read(OUT/'frozen.json'); cfg=read(OUT/'protocol.json')
    by={r['subject_id']:r for r in data.load(DIR,['BUN_CR'])}
    ctx=Context('BUN_CR',[by[s] for s in cfg['references']],SEED)
    rows=[]
    for density in (0,1):
        rec=next(by[s] for s in cfg['references'] if (len(by[s]['events'])>=8)==density)
        public=public_view(rec); x=data.replay(rec)
        for B in BUDGETS:
            session=[]; decision=[]
            for i in range(21):
                start=perf_counter_ns()
                branch=choose(frozen,B,public)
                delta=perf_counter_ns()-start
                start=perf_counter_ns()
                result=run(ctx,frozen,B,public,lambda t,names:[x[t-1,ctx.work.roots.index(n)] for n in names])
                ms=(perf_counter_ns()-start)/1e6
                assert abs(B-result['spent'])<1e-8
                if i:session.append(ms); decision.append(delta/1e6)
            rows.append(dict(B=B,density=density,form='static' if branch.period is None else 'dynamic',
                decision_p50_ms=np.median(decision),session_p50_ms=np.median(session),session_p95_ms=np.quantile(session,.95)))
    write(OUT/'live_timing.csv',rows)

if __name__=='__main__':main()
