import argparse,csv,json,time,platform,subprocess,shutil
from pathlib import Path
import numpy as np
import scipy
from common import original,HERE,events
from data_cases import inventory,split
from publishing import quota_period,oracle_quota,scheduled,deployment,candidates,frozen_schedule_pair
from event_metrics import measure


def select(rows,params,family,target,false_max):
    indices=[i for i,p in enumerate(params) if p['method']==family or p.get('mode')==family]
    summaries=[]
    for i in indices:
        rr=rows[i];ev=[r['recall1'] for r in rr if r['recall1'] is not None]
        summaries.append(dict(index=i,recall=None if not ev else float(np.mean(ev)),rce=float(np.mean([r['rce'] for r in rr])),
                              cmr=float(np.mean([r['cmr'] for r in rr])),false=float(np.mean([r['false_change'] or 0 for r in rr]))))
    feasible=[s for s in summaries if (s['recall'] is None or s['recall']>=target) and s['false']<=false_max]
    if feasible:chosen=min(feasible,key=lambda s:(s['rce'],s['cmr'],s['index']))
    else:chosen=min(summaries,key=lambda s:(-(s['recall'] or 0),s['false'],s['rce'],s['index']))
    return params[chosen['index']],dict(validation_feasible=bool(feasible),constraint=target,**chosen)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(args.config.read_text());out=args.out;out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2));start=time.time()
    all_cases=inventory(cfg);cases=[(i,*r) for i,r in enumerate(all_cases) if 'only_cases' not in cfg or r[0] in cfg['only_cases']]
    meta=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),datasets={})
    sources=[HERE/f for f in ['common.py','data_cases.py','event_metrics.py','publishing.py','run.py']]+list((HERE/'tests').rglob('*.py'))+[args.config,HERE/'previous_sources.json',original.REPO/'data/nhanes_benchmark_200.json',original.REPO/'data/holdout_population_means.json',Path('/home/rootguard_research_audit/datasets/mimic-iv-demo-2.2/labevents.csv.gz')]
    meta['sources_sha256']={str(p.resolve()):original.sha(p) for p in sources}
    for p in HERE.rglob('*.py'):
        target=out/'source_snapshot'/p.relative_to(HERE);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    meta['privacy']='T8/16 public schedules/layout; d_inf of normalized 0..999 root indices; B-metric trajectory DP; sufficient standard E=999B; all Oracle and exact arms non-DP'
    cells=[];validations=[];selections=[];patients=[]
    def record(did,source,d,B,comparison,method,param,cache,ledger,extra=None):
        row,pr=measure(d,cache,ledger)
        cid=f'{did}|{B}|{comparison}|{method}'
        row=dict(cell=cid,dataset=did,source=source,template=d['name'],T=d['x'].shape[1],n=len(d['ids']),B=B,comparison=comparison,method=method,param=param,**row)
        if comparison=='A2':row.update({('constraint' if key=='constraint' else key if key.startswith('validation_') else 'validation_'+key):value for key,value in (extra or {}).items()})
        else:row.update(extra or {})
        cells.append(row)
        for j,pid in enumerate(d['ids']):patients.append(dict(cell=cid,patient=pid,**{k:float(v[j]) for k,v in pr.items()}))
        if row['dp']:assert row['spent_max']<=B+1e-8
    pools={}
    for di,did,d,source in cases:
        v,t=split(d,source);group=f'{"real" if source=="observed" else "synthetic"}_{d["name"]}_H{d["x"].shape[1]}'
        pools.setdefault(group,[]).append((di,did,d,source,v,t))
        y,c,e=events(d)
        meta['datasets'][did]=dict(source=source,template=d['name'],T=d['x'].shape[1],audit=d['audit'],validation_ids=v['ids'],test_ids=t['ids'],
            test_events=int(events(t)[2].sum()),test_event_patients=int((events(t)[2].sum(axis=1)>0).sum()))
    for group,items in pools.items():
        T=items[0][4]['x'].shape[1];params=candidates(T)
        for B in cfg['budgets']:
            vr={i:[] for i in range(len(params))}
            for di,did,d,source,v,t in items:
                rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],di,0]));N,_,k=v['x'].shape;R=cfg['validation_repeats']
                u=rng.random((N,R,T,k));q=rng.laplace(size=(N,R,T));th=rng.laplace(size=(N,R,4))
                for i,p in enumerate(params):
                    z,l=deployment(v,B,p,u,q,th);a,_=measure(v,z,l);vr[i].append(a)
                    validations.append(dict(group=group,dataset=did,B=B,param=p,**a))
            selected=[]
            for family in ['period','attribute','task']:
                for floor in cfg['recall_constraints']:
                    p,s=select(vr,params,family,floor,cfg['false_change_constraint']);selected.append((f'{family}_target{floor}',p,s))
                    selections.append(dict(group=group,B=B,family=family,param=p,**s))
            for di,did,d,source,v,t in items:
                N,_,k=t['x'].shape;R=cfg['test_repeats'];rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],di,1]))
                u=rng.random((N,R,T,k));q=rng.laplace(size=(N,R,T));th=rng.laplace(size=(N,R,4))
                # Full-budget deployment comparison. Test grid is descriptive;
                # selected deployment policies only use validation choices.
                methods=[('static',dict(method='static'),{}),('public_roots',dict(method='root'),{})]
                methods += [(f'period_grid{P}',dict(method='period',period=P),{}) for P in [1,2,4,8,16] if P<=T]
                methods += selected
                evaluated={}
                for method,p,s in methods:
                    key=json.dumps(p,sort_keys=True)
                    if key not in evaluated:evaluated[key]=deployment(t,B,p,u,q,th)
                    z,l=evaluated[key];record(did,source,t,B,'A2',method,p,z,l,s)
                # Pure timing diagnosis: same K ACTUAL refreshes, same allocation,
                # common uniforms by ordinal release. Phase fixed on validation.
                for K in cfg['K']:
                    ordinal=u[:,:,:K+1,:]
                    val_rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],di,2]));Nv=len(v['ids']);uv=val_rng.random((Nv,cfg['validation_repeats'],K+1,k))
                    phase_rows=[]
                    for offset in [-1,0,1]:
                        plan=quota_period(Nv,T,K,offset);z,l=scheduled(v,B,K,plan,uv);a,_=measure(v,z,l)
                        phase_rows.append((offset,a))
                    ok=[r for r in phase_rows if (r[1]['recall1'] is None or r[1]['recall1']>=20) and (r[1]['false_change'] or 0)<=20]
                    chosen=min(ok,key=lambda r:r[1]['rce']) if ok else min(phase_rows,key=lambda r:(-(r[1]['recall1'] or 0),r[1]['rce']))
                    offset=chosen[0]
                    for method,plan,oracle in [('period',quota_period(N,T,K,offset),False),('oracle',oracle_quota(t,K),True)]:
                        z,l=scheduled(t,B,K,plan,ordinal,oracle=oracle)
                        record(did,source,t,B,'A1',f'{method}_K{K}',dict(K=K,offset=offset if not oracle else None),z,l)
                    if did in cfg['B_cases'] and K in cfg['B_K']:
                        plan=oracle_quota(t,K)
                        for ie,re in [(True,True),(True,False),(False,True),(False,False)]:
                            z,l=scheduled(t,B,K,plan,ordinal,initial_exact=ie,refresh_exact=re,oracle=True)
                            label=f'initial_{"exact" if ie else "noisy"}_refresh_{"exact" if re else "noisy"}_K{K}'
                            record(did,source,t,B,'B',label,dict(K=K,initial_exact=ie,refresh_exact=re),z,l)
                if did in cfg['B_cases']:
                    for family in ['attribute','task']:
                        chosen=next(r for r in selected if r[0]==f'{family}_target20')
                        p=chosen[1];C=p['C'];fraction=p['fraction'];ordinal=u[:,:,:C+1,:]
                        # Same Oracle times/number/uniforms, change only the
                        # reserved publication fraction: diagnostic cost of
                        # allocating privacy budget to detection.
                        plan=oracle_quota(t,C)
                        for tag,f in [('full',1.),('reserved',1-fraction)]:
                            z,l=scheduled(t,B,C,plan,ordinal,oracle=True,publish_fraction=f)
                            record(did,source,t,B,'B_fee',f'{family}_{tag}',dict(K=C,fraction=fraction),z,l)
                        z,l=evaluated[json.dumps(p,sort_keys=True)]
                        actual,ideal=frozen_schedule_pair(t,l)
                        for tag,pair in [('svt_frozen',actual),('oracle_count_matched',ideal)]:
                            record(did,source,t,B,'B_schedule',f'{family}_{tag}',p,*pair,
                                   extra={'matched_actual_refreshes':True,'exact_outputs':True})
                print(did,B,'complete',flush=True)
    for name,data in [('cells',cells),('validation',validations),('selection',selections)]:
        (out/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False))
    with (out/'patients.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(patients[0]));writer.writeheader();writer.writerows(patients)
    meta.update(seconds=time.time()-start,cells=len(cells),validation_cells=len(validations))
    (out/'metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
    print(json.dumps(dict(cells=len(cells),validation=len(validations),seconds=meta['seconds'])),flush=True)


if __name__=='__main__':main()
