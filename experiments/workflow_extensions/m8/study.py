"""Autoresearch replay of M8 and drift-weighted development rules."""
import csv
import hashlib
import json
from dataclasses import asdict,replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from ..finite_cache import data
from ..m4_carry.core import key_seed
from ..m7.engine import Context,Branch,branches,_execute,public_view
from ..m7.metrics import utility,attack_result
from ..m7.appointment_study import cumulative,write,STATIC,OLD,REF,SEED,BUDGETS
from ..m7.audit import ledger
from ..m7.run import DEFAULT_SPLIT
from .policy import choose

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'outputs/m8/replay_20261120'
ORIGINAL=json.loads((HERE/'original_bun.json').read_text())
OLDPLANS=json.loads((OLD/'frozen.json').read_text())
def dump(path,obj):path.write_text(json.dumps(obj,indent=2))
def order(r):return hashlib.sha256(r['subject_id'].encode()).digest()
def prep(ctx,recs):
    for r in recs:
        x=data.replay(r)
        r['_x']=ctx.work.normalize(x)
        r['_y']=ctx.work.evaluate(x)[ctx.work.target]
        r['_span']=np.ptp(r['_y'])
        r['_public']=public_view(r)

def sim(ctx,r,B,b,R,trace=False):
    seed=key_seed(SEED,'physionet2020',r['subject_id'],B)
    u=np.random.default_rng(seed).laplace(size=(r['T']+1,R,len(ctx.roots)))
    return _execute(r['_public'],B,b,ctx,lambda t,i:r['_x'][t-1,list(i)],u,trace)

def losses(ctx,rows,B,b):
    return np.array([abs(ctx.work.from_cache(sim(ctx,r,B,b,4)['caches'])[ctx.work.target]-r['_y']).mean() for r in rows])

def weighted(x,high):
    # Give high-drift performance a fixed 75% weight where both strata exist.
    return .75*x[high].mean()+.25*x[~high].mean() if high.any() and (~high).any() else x.mean()

def study(label,template,refs,dev,val,test,threshold,phys=False,transfer=False):
    ctx=Context(template,refs,SEED)
    prep(ctx,dev+val+test)
    local=[]; scores=[]; privacy=[]; frozen={}
    groups=sorted({int(len(r['events'])>=8) for r in dev+val+test})
    for B in BUDGETS:
        for g in groups:
            group=lambda rows:[r for r in rows if int(len(r['events'])>=8)==g]
            tr,va=group(dev),group(val)
            base=ORIGINAL[f'{B}/{g}']
            d=Branch(**base['dynamic'])
            init=Branch(**base['route'])
            if phys:
                periodic=Branch(**base['periodic'])
            elif template in OLDPLANS:
                periodic=Branch(**OLDPLANS[template][str(B)]['branches']['P-cal'])
            else:
                periodic=replace(STATIC,anchor=.5,period=4)
            old=Branch(**OLDPLANS[template][str(B)]['branches']['R-new']) if template in OLDPLANS else branches('portable',template,'workload')[1]
            best=init
            sel=init
            if tr and va and not transfer:
                @lru_cache(None)
                def risk(b):return losses(ctx,tr,B,b)
                @lru_cache(None)
                def validation(b):return losses(ctx,va,B,b)
                hi=np.array([r['_span']>threshold for r in tr]); hv=np.array([r['_span']>threshold for r in va])
                # First adapt the original selector to the template without tail weighting.
                pool=[STATIC,d,replace(d,refresh_weights='mopt'),replace(d,refresh_count=2,fusion=True)]
                best=min(pool,key=lambda b:risk(b).mean())
                if validation(best).mean()>=validation(init).mean():best=init
                sel=best
                if B>=651.22:
                    baseplan=replace(d,period=1,fusion=True)
                    pool=[best,d]+[replace(baseplan,anchor=a,refresh_count=k,fixed_roots=mask)
                        for a in (.65,.85) for k in (1,2)
                        for mask in [ctx.roots,*[(i,) for i in ctx.roots]]]
                    def objective(b):
                        v=weighted(risk(b),hi)
                        local.append(dict(dataset=label,B=B,density=g,stage='high-budget',branch=json.dumps(asdict(b)),score=v))
                        return v
                    contender=min(pool,key=objective)
                    if weighted(validation(contender)-validation(best),hv)<0:sel=contender
                local.append(dict(dataset=label,B=B,density=g,stage='freeze',branch=json.dumps(asdict(sel)),score=validation(sel).mean()))
            methods=dict(static=STATIC,periodic=periodic,m7=old,m8_original=init,m8_mean=best,m8=sel,forced=d,
                         native_static=Branch(1.),native_periodic=replace(periodic,calibration='none',shrink=0.))
            frozen[f'{B}/{g}']={k:asdict(b) for k,b in {'route':sel,**methods}.items()}
    dump(OUT/(label+'-frozen.json'),frozen)
    for B in BUDGETS:
        for i,r in enumerate(test):
            memo={}
            plan=frozen[f"{B}/{int(len(r['events'])>=8)}"]
            for m,spec in plan.items():
                if m=='route':continue
                b=Branch(**spec)
                if b not in memo:
                    o=sim(ctx,r,B,b,4 if phys else 20,True)
                    ledger(o,B,b)
                    u={k:float(v.mean()) for k,v in utility(ctx,r,o).items()}
                    priv={}
                    if i<32 and b.calibration=='marginal':
                        c=cumulative(ctx,r,o)
                        priv={k:float(v.mean()) for k,v in c.items() if k.endswith('all_hit')}
                        attacks=attack_result(ctx,r,o,b)
                        for z in attacks:
                            key=z['attack']+'_'+z['prior']
                            priv[key]=priv.get(key,0)+100*z['success_005']/len(o['spent'])
                    memo[b]=u,priv
                u,pr=memo[b]
                meta=dict(dataset=label,template=template,B=B,subject=r['subject_id'],method=m,high=int(r['_span']>threshold))
                scores.append({**meta,**u})
                if pr:privacy.append({**meta,**pr})
    for suffix,rows in [('paths',scores),('privacy',privacy),('iterations',local)]:
        if rows:write(OUT/f'{label}-{suffix}.csv',rows)
    print(label,len(test),'complete',flush=True)

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    dump(OUT/'protocol.json',dict(primary='B=960, high task range test subset, paired MAE vs strong periodic and original M8; report low-range cost',other='4 anchor budgets; MIMIC legacy split; NLR LOPO; no parameter change after test',rule='mean-stage then 75/25 high/low risk-stage at two top anchors; independent validation for acceptance',seeds=SEED))
    path=Path('data_mine/finite_cache_physionet2019'); ds=sorted(data.load(path,['BUN_CR']),key=order)
    sp=json.loads((path/'split.json').read_text())
    train,valid,test=[[r for r in ds if sp[r['subject_id']]==role] for role in ('train','validation','test')]
    cfg=json.loads((HERE/'original_protocol.json').read_text()); assert [r['subject_id'] for r in train[:48]]==cfg['references']
    # Exclude validation patients used to choose the first version when available.
    def slices(rows,a,b):
        return [r for g in (0,1) for r in [x for x in rows if int(len(x['events'])>=8)==g][a:b]]
    rest=slices(valid,120,None)
    # Dense stratum's validation can be exhausted: reserve disjoint A training patients for adjudication.
    extra=slices(train[48:],120,220)
    va=rest+[r for r in extra if not any((len(z['events'])>=8)==(len(r['events'])>=8) for z in rest)]
    study('PhysioNet','BUN_CR',train[:48],slices(train[48:],0,120),va,test,cfg['threshold'],True)
    sp=json.loads(DEFAULT_SPLIT.read_text())
    for t in ('MELD_NA','ANEMIA','FIB4','BUN_CR','TYG','NLR'):
        ds=data.load(Path('data_mine/finite_cache'),[t])
        if t=='NLR':
            for i,r in enumerate(ds):
                study(f'NLR-fold{i}',t,[ds[1-i]],[],[],[r],float('inf'),transfer=True)
            continue
        train,va,te=[[r for r in ds if sp.get(r['subject_id'])==role] for role in ('train','validation','test')]
        ctx=Context(t,train,SEED)
        threshold=float(np.median(np.ptp(ctx.bank_tasks[:,:,ctx.target],axis=1)))
        study(t,t,train,train,va,te,threshold,transfer=t=='TYG')

if __name__=='__main__':main()
