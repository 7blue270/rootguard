"""Train on existing folds, freeze the public route, then replay all held-out stays."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from ..m8.study import (data,sim,prep,order,DEFAULT_SPLIT,Context,SEED,BUDGETS,Branch,ORIGINAL,dump,write)
from ..m8.policy import choose
from ..m7.appointment_study import REF
from ..m7.audit import ledger
from ..state_estimation.window_study import references
from ..physionet2012.study import segments,inputs,OUT as PN12
from ..m6.kernels import grid_logZ
from .policy import plans,pick,estimate
OUT=Path(__file__).resolve().parent/'results'
COHORTS={'2019_BUN':('finite_cache_physionet2019','BUN_CR'),'2012_BUN':('finite_cache_physionet2012','BUN_CR'),'2012_FIB4':('finite_cache_physionet2012','FIB4'),
         **{'old_'+t:('finite_cache',t) for t in ('BUN_CR','ANEMIA','FIB4','MELD_NA')}}
ARMS=('static','full','package','early','anchor75','equal','combined')

def setup(name):
    folder,t=COHORTS[name];folder=Path('data_mine')/folder
    rows=sorted(data.load(folder,[t]),key=order)
    split=json.loads(((folder/'split.json') if not name.startswith('old') else DEFAULT_SPLIT).read_text())
    parts=[[r for r in rows if split[r['subject_id']]==s] for s in ('train','validation','test')]
    refs=references(t,t=='BUN_CR' and not name.startswith('old'))
    ctx=Context(t,refs,SEED);prep(ctx,rows)
    return ctx,parts,refs

def fit(ctx,rows,B,b,median):
    blocks={}
    for r in rows:
        ps=sim(ctx,r,B,b,8,True)
        for seg in segments(ctx,ps):
            st,en=seg[:2];x,f=inputs(ctx,seg,ps['anchor_service'],median,False)
            times=st+np.arange(len(x))*(en-st)//len(x)
            blocks.setdefault(bool(st),[]).append((x,f,r['_y'][times],np.full(len(x),en-st)))
    model={}
    for phase,items in blocks.items():
        base,f,y,w=map(np.concatenate,zip(*items));w=w/w.sum()
        shape=(f.shape[1],base.shape[1]);lo=np.log(ctx.work.lo);hi=np.log(ctx.work.hi)
        def task(z):return ctx.work.evaluate(np.exp(z))[ctx.work.target]
        norm=max(np.sum(w*abs(task(base)-y)),1e-8)
        def loss(v):
            raw=base+f@v.reshape(shape);z=np.clip(raw,lo,hi);err=task(z)-y
            grad=[]
            for i in ctx.roots:
                step=np.zeros(len(ctx.roots));step[i]=1e-5
                grad.append((task(z+step)-task(z-step))/2e-5)
            g=f.T@((w*np.sign(err))[:,None]*np.stack(grad,axis=1)*((raw>=lo)&(raw<=hi)))
            return w@abs(err)/norm+.06*(v@v),g.ravel()/norm+.12*v
        opt=minimize(loss,np.zeros(np.prod(shape)),jac=True,method='L-BFGS-B',bounds=[(-.6,.6)]*np.prod(shape),options={'maxiter':90})
        model[phase]=opt.x.reshape(shape)
    return model

def scores(ctx,r,c,ps,jump,privacy=None):
    pred=ctx.work.from_cache(c)[ctx.work.target];y=r['_y'];err=abs(pred-y)
    e=np.flatnonzero(abs(np.diff(y))>=jump)+1;ev=np.unique(np.minimum(len(y)-1,np.r_[e,e+1]))
    cl=ctx.work.classes(y)
    return dict(mae=err.mean(),den=abs(y).mean(),p95=np.quantile(err,.95,axis=1).mean(),response=err[:,ev].mean() if len(ev) else None,
                rce=50*abs(ctx.work.classes(pred)-cl).mean() if cl is not None else None,refresh=ps['refreshes'].mean(),roots=ps['roots'].mean(),spent=ps['spent'].mean(),map_error=privacy)

def reconstruction(ctx,r,ps):
    # Offline stronger observer has all raw paid grid values, not unselected true roots.
    logp=np.zeros((len(ps['anchor_cache']),len(ctx.bank)))
    obs=[(0,ps['anchor_cache'],ps['anchor_epsilon'][0])]+[(h['turn']-1,h['cache_after'],h['actual_epsilon'][0]) for h in ps['histories']]
    for turn,z,eps in obs:
        for i in np.flatnonzero(eps>0):
            prior=ctx.bank[:,turn,i];e=eps[i]
            logp+=-e*abs(z[:,None,i]-prior)/2-grid_logZ(prior,e,0,1,1000)
    return abs(ctx.bank[logp.argmax(axis=1)]-r['_x']).mean()

def run(name):
    dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    ctx,(train,val,test),refs=setup(name);t=ctx.work.template
    cut=np.median([r['_span'] for r in train])
    change=np.concatenate([abs(np.diff(r['_y'])) for r in train]);jump=np.median(change[change>0])
    sample=train[:96];dev=val[:120]
    drift=abs(np.diff(ctx.bank,axis=1)).mean(axis=(0,1))
    rank=np.argsort(-(ctx.rms*drift),kind='stable')
    package=sorted(map(int,rank[:max(1,len(rank)//2)]))
    pool=plans(ctx,package)
    median=np.median(np.log(ctx.work.denormalize(np.stack([r['_x'] for r in sample]))),axis=(0,1))
    models={};route={};pooled={};forced={};iterations=[]
    archive=json.loads((PN12/(t+'_frozen.json')).read_text()) if name.startswith('2012') else None
    if archive:median=np.array(archive['median'])
    for B in BUDGETS:
        models[B]={}
        for label in ('static','period'):
            if archive:
                models[B][label]={bool(int(k[0])):np.asarray(v) for k,v in archive['models'][f'{B}/{label}'].items() if k.endswith('/task')}
            else:models[B][label]=fit(ctx,sample,B,pool[label],median)
        costs=[]
        for r in dev:
            line=[]
            for a in ARMS:
                ps=sim(ctx,r,B,pool[a],4,True)
                c=estimate(ctx,ps,models[B]['static' if a=='static' else 'period'],median)
                line.append(abs(ctx.work.from_cache(c)[ctx.work.target]-r['_y']).mean())
            costs.append(line)
        costs=np.array(costs);dense=np.array([len(r['events'])>=8 for r in dev])
        route[str(B)]={};forced[str(B)]={}
        pooled[str(B)]=ARMS[np.argmin(costs.mean(0))]
        for g in (0,1):
            subset=costs[dense==g] if (dense==g).any() else costs
            avg=subset.mean(0)
            route[str(B)][str(g)]=ARMS[np.argmin(avg)]
            forced[str(B)][str(g)]=ARMS[1+np.argmin(avg[1:])]
            for a,v in zip(ARMS,avg):iterations.append(dict(B=B,dense=g,validation_n=int((dense==g).sum()),arm=a,mae=v))
    cfg=dict(route=route,pooled=pooled,forced=forced,median=median.tolist(),package=package,models={str(B):{l:{str(int(p)):v.tolist() for p,v in mod.items()} for l,mod in pair.items()} for B,pair in models.items()},
             references=[r['subject_id'] for r in refs],fitting=archive['train'] if archive else [r['subject_id'] for r in sample],validation=[r['subject_id'] for r in dev],split_counts=list(map(len,(train,val,test))),span=float(cut),jump=float(jump))
    dump(dest/'frozen.json',cfg);write(dest/'iterations.csv',iterations)
    grid=sorted(set(map(float,BUDGETS))|{160.,480.})
    with (dest/'paths.csv').open('w') as f:
        writer=None
        for r in test:
            for B in grid:
                near=min(BUDGETS,key=lambda b:abs(np.log(b/B)));key=str(near);public=r['_public']
                arms={a:a for a in (*ARMS,'period')}
                arms.update(m9=pick(route,B,public),pooled=pooled[key],force_dynamic=pick(forced,B,public))
                cache={}
                for m,a in arms.items():
                    if a not in cache:
                        ps=sim(ctx,r,B,pool[a],4,True);ledger(ps,B,pool[a])
                        model=models[near]['static' if a=='static' else 'period']
                        c=estimate(ctx,ps,model,median)
                        priv=reconstruction(ctx,r,ps)
                        cache[a]=(scores(ctx,r,c,ps,jump,priv),ps)
                    met,ps=cache[a]
                    row=dict(subject=r['subject_id'],B=B,method=m,arm=a,high=int(r['_span']>cut),**met)
                    if writer is None:writer=csv.DictWriter(f,fieldnames=row);writer.writeheader()
                    writer.writerow(row)
                a=arms['m9'];met,ps=cache[a]
                for m,b in [('no_estimator',None),('rootguard',Branch(1.)),('m8',choose(ORIGINAL,B,public))]:
                    if b is None:c=ps['caches'];p=ps;priv=met['map_error']
                    else:
                        p=sim(ctx,r,B,b,4,True);c=p['caches'];priv=reconstruction(ctx,r,p)
                    writer.writerow(dict(subject=r['subject_id'],B=B,method=m,arm=a if b is None else m,high=int(r['_span']>cut),**scores(ctx,r,c,p,jump,priv)))
    print(name,len(test),'finished',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('cohorts',nargs='*');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    dump(OUT/'protocol.json',dict(cohorts=COHORTS,anchors=BUDGETS,iso_grid=sorted(set(BUDGETS)|{160.,480.}),thresholds=str(REF/'targets.json'),selection='min validation patient mean MAE per budget and public density cell; empty cell uses all validation; no test selection',paired_seed=SEED,repeats=4,privacy='raw transcript reference-bank full-trajectory MAP normalized root MAE',primary='four anchor equal-weight patient trajectory MAE; compare m8, static and forced dynamic; exploratory historical test'))
    for name in args.cohorts or COHORTS:run(name)
