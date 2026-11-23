"""Audit exact sampler, paid reads, action schedule, marginal laws and prefix charges."""
from dataclasses import asdict
from fractions import Fraction
from collections import Counter
import json
import numpy as np
from ..m7.engine import Context,cdf,public_view
from ..m7.audit import ledger
from ..m7.appointment_study import STATIC
from ..finite_cache import data
from .policy import sample,run
from .study import HERE,OUT,SEED,ORIGINAL,prep,sim,Branch,dump
from pathlib import Path

def main():
    src=Path('data_mine/finite_cache_physionet2019')
    records=data.load(src,['BUN_CR'])
    by={r['subject_id']:r for r in records}
    refs=json.loads((HERE/'original_protocol.json').read_text())['references']
    ctx=Context('BUN_CR',[by[s] for s in refs],SEED)
    details=[]
    for n in (5,10):
        rec=next(r for r in records if len(r['events'])==n)
        prep(ctx,[rec]); x=data.replay(rec)
        for B in (80.,651.22,960.):
            calls=[]
            def reader(t,names):
                calls.append((t,tuple(names)))
                return [x[t-1,ctx.work.roots.index(name)] for name in names]
            result=run(ctx,ORIGINAL,B,rec['_public'],reader)
            assert calls==[(a['turn'],tuple(a['roots'])) for a in result['actions']]
            assert sum(sum(a['epsilon']) for a in result['actions'])==Fraction(B)
            assert np.isfinite(result['cache']).all()
            branch=Branch(**ORIGINAL[f'{B}/{int(n>=8)}']['route'])
            ordinary=sim(ctx,rec,B,branch,4,True);ledger(ordinary,B,branch)
            assert ordinary['schedule']==[a['turn'] for a in result['actions'][1:]]
            details.append(dict(B=B,n=n,actions=len(calls)))
    moments=[]
    for val,e in [(Fraction(1,3),8),(Fraction(9,10),80)]:
        draws=np.array([sample(val,Fraction(e)) for _ in range(2000)])/999
        pmf=np.diff(cdf(float(val),e),prepend=0);grid=np.arange(1000)/999
        mean=pmf@grid;se=np.sqrt(pmf@(grid-mean)**2/len(draws))
        assert abs(draws.mean()-mean)<6*se
        moments.append(dict(mean_exact_draws=draws.mean(),mean_theory=mean,se=se))
    # On the original secret 52-bit uniform lattice an output may be impossible for x=0.
    # Bin zero at x=1 has positive theoretical mass but lies below the smallest available uniform.
    floor=2**-53
    masses=[float(cdf(x,160.)[0]) for x in (0.,1.)]
    assert masses[0]>floor>masses[1]
    dump(OUT/'dp_audit.json',dict(prefix_checks=details,sampler_moments=moments,
         old_bin0_probabilities=masses,old_uniform_floor=floor,
         exact_randomness='private uniform integers via secrets.randbelow; ideal private random-bit model'))
    print('exact-law moment checks and paid callbacks passed',flush=True)
if __name__=='__main__':main()
