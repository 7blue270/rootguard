import argparse,csv,json,hashlib,platform,subprocess,time
from pathlib import Path
import numpy as np
import scipy
from adapter import old,original,PREVIOUS
from dataset import make,observed,split,diagnosis
from release import run,metrics


def candidates(cfg,H):
    out=[dict(method='period',period=p) for p in cfg['periods'] if p<=H]
    for mode in ['attribute','task']:
        for C in cfg['trigger_caps']:
            for f in cfg['trigger_fractions']:
                for th in cfg['trigger_thresholds']:
                    out.append(dict(method='svt',mode=mode,C=C,fraction=f,threshold=th))
    return out


def loss(row,profile):
    # Fixed, interpretable exploratory objective, not a clinical acceptance rule.
    if profile=='accuracy':return row['wmape']
    return row['wmape']+.1*(100-row['tcr2'] if row['tcr2'] is not None else 0)+.1*(row['false_change'] or 0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(args.config.read_text());out=args.out
    out.mkdir(parents=True,exist_ok=False);(out/'config.json').write_text(json.dumps(cfg,indent=2))
    start=time.time();datasets=[]
    for name in cfg['templates']:
        for H in cfg['horizons']:
            for scenario in cfg['scenarios']:
                datasets.append((f'syn_{name}_{scenario}_H{H}',make(name,H,scenario,cfg['amplitude'])))
            if name=='ANEMIA':datasets.append((f'syn_{name}_irrelevant_H{H}',make(name,H,'irrelevant',cfg['amplitude'])))
    datasets.append(('real_ANEMIA_H8',observed()))
    meta=dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,datasets={})
    repo=original.REPO
    sources=list(Path(__file__).parent.rglob('*.py'))+[args.config,PREVIOUS/'mechanisms.py',PREVIOUS/'data.py',PREVIOUS.parent/'target_mismatch/core.py',repo/'utils/utils.py',repo/'utils/risk_class.py',repo/'preempt/sanitizer.py',repo/'data/nhanes_benchmark_200.json',repo/'data/holdout_population_means.json',Path('/home/rootguard_research_audit/datasets/mimic-iv-demo-2.2/labevents.csv.gz')]
    meta['source_sha256']={str(p.resolve()):original.sha(p) for p in sources}
    meta['privacy']='public finite horizon/layout; d_inf of 0..999 indices; metric B; sufficient pure trajectory E=999B; oracle not certified'
    diag=[];validation=[];selection=[];cells=[];patients=[]
    tracks=[('metric',b,b) for b in cfg['budgets']]+[('standard',e,e/999) for e in cfg['standard_eps']]
    # Tune across all synthetic scenarios for each template/horizon. Never
    # choose a separate best method for each synthetic test scenario.
    pools={}
    for di,(did,d) in enumerate(datasets):
        val,test=split(d);group=('real' if did.startswith('real') else 'synthetic')+'_'+d['name']+'_H'+str(d['x'].shape[1])
        pools.setdefault(group,[]).append((di,did,val,test))
        meta['datasets'][did]=dict(d['audit'],validation_ids=val['ids'],test_ids=test['ids'])
        for part,pd in [('validation',val),('test',test)]:diag.append(dict(dataset=did,split=part,**diagnosis(pd)))
    for group,items in pools.items():
        H=items[0][2]['x'].shape[1];params=candidates(cfg,H)
        for track,label,B in tracks:
            estimates={}
            for pi,param in enumerate(params):
                sub=[]
                for di,did,val,test in items:
                    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],di,0]))
                    N,_,k=val['x'].shape;R=cfg['validation_repeats']
                    u=rng.random((N,R,H,k));q=rng.laplace(size=(N,R,H));t=rng.laplace(size=(N,R,4))
                    z,l=run(val,B,param,u,q,t);a,_=metrics(val,z,l)
                    validation.append(dict(group=group,dataset=did,track=track,budget=label,param=param,**a));sub.append(a)
                estimates[pi]={profile:float(np.mean([loss(a,profile) for a in sub])) for profile in ['accuracy','balanced']}
            chosen=[]
            for family in ['period','attribute','task']:
                indices=[i for i,p in enumerate(params) if p['method']==family or p.get('mode')==family]
                for profile in ['accuracy','balanced']:
                    index=min(indices,key=lambda i:(estimates[i][profile],i));param=params[index]
                    chosen.append((family+'_'+profile,param))
                    selection.append(dict(group=group,track=track,budget=label,family=family,profile=profile,param=param,validation_loss=estimates[index][profile]))
            chosen=[('static',dict(method='static')),('public_roots',dict(method='root'))]+chosen+[(f'oracle_C{C}',dict(method='oracle',C=C)) for C in [2,4]]
            for di,did,val,test in items:
                rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],di,1]))
                N,_,k=test['x'].shape;R=cfg['test_repeats']
                u=rng.random((N,R,H,k));q=rng.laplace(size=(N,R,H));t=rng.laplace(size=(N,R,4))
                for method,param in chosen:
                    z,l=run(test,B,param,u,q,t);a,pr=metrics(test,z,l)
                    cid=f'{did}|{track}|{label}|{method}'
                    cells.append(dict(cell=cid,dataset=did,group=group,method=method,param=param,track=track,budget=label,B=B,n=N,H=H,**a))
                    for i,pid in enumerate(test['ids']):patients.append(dict(cell=cid,patient=pid,**{key:float(value[i]) for key,value in pr.items()}))
                    if a['certified']:assert a['spent_max']<=B+1e-8
            print(group,track,label,'complete',flush=True)
    for file,value in [('metadata',meta),('diagnosis',diag),('validation',validation),('selection',selection),('cells',cells)]:
        (out/(file+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False))
    with (out/'patients.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(patients[0]));w.writeheader();w.writerows(patients)
    meta.update(seconds=time.time()-start,cells=len(cells),validation_cells=len(validation))
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(dict(cells=len(cells),validation=len(validation),seconds=meta['seconds'])),flush=True)


if __name__=='__main__':main()
