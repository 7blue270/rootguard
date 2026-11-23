"""Live client timing on public train records, with synthetic private read fixtures."""
from time import perf_counter_ns
from unittest.mock import patch
import numpy as np
from . import engine
from .appointment_study import OUT,OLD,DAT,STATIC,SEED,Branch,Context,setup,read,write,public_view
from .routing import Route

def main():
    parts,_,_=setup(DAT)
    plan=read(OUT/'frozen.json'); old=read(OLD/'frozen.json')
    result=[]
    for t,p in parts.items():
        ctx=Context(t,p['train'],SEED)
        pub=public_view(p['train'][0]); B=960.
        win=Branch(**plan[t][str(B)]['win'])
        follow=read(OUT.parent/'appointment_gate_20261119/frozen.json')[t][str(B)]
        gated=Branch(**follow['win'])
        for method,branch in [('gate',gated),('scheduled',win),('static',STATIC),('periodic',Branch(**old[t][str(B)]['branches']['P-cal'])),
                              ('oldroute',Branch(**old[t][str(B)]['branches']['R-new']))]:
            route=Route({B:(STATIC,branch)},{B:True},'profile')
            stats={k:[] for k in ('route','gate','release','calibrate','session')}
            def timed(fn,label):
                def call(*a,**kw):
                    tic=perf_counter_ns()
                    answer=fn(*a,**kw)
                    stats[label].append((perf_counter_ns()-tic)/1e6)
                    return answer
                return call
            def fixture(turn,names):
                return [ctx.work.lo[ctx.work.roots.index(n)]+.4*ctx.work.width[ctx.work.roots.index(n)] for n in names]
            with patch.object(Route,'choose',timed(Route.choose,'route')),patch.object(ctx,'gate',timed(ctx.gate,'gate')),patch.object(engine,'release',timed(engine.release,'release')),patch.object(ctx,'calibrated',timed(ctx.calibrated,'calibrate')):
                for i in range(21):
                    session=timed(engine.run_live,'session')(ctx,route,B,pub,fixture)
                    assert session['spent']<=B+1e-9
                    if i==0:
                        for v in stats.values():v.clear()
                for metric,values in stats.items():
                    result.append(dict(template=t,method=method,component=metric,calls=len(values),
                        p50_ms=np.median(values) if values else None,p95_ms=np.quantile(values,.95) if values else None))
    write(OUT/'live_timing.csv',result)

if __name__=='__main__':
    main()
