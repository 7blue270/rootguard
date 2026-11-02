"""Patient-paired exploratory intervals and source/budget audit."""
import argparse,csv,json,hashlib,subprocess
from pathlib import Path
import numpy as np
from common import HERE,original


def ratio(num,den):
    if np.sum(den)==0:return None
    return float(100*np.sum(num)/np.sum(den))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args();out=args.run.resolve()
    cfg=json.loads((out/'config.json').read_text());meta=json.loads((out/'metadata.json').read_text());cells=json.loads((out/'cells.json').read_text())
    previous=json.loads((HERE/'previous_sources.json').read_text())
    changed=[p for p,h in previous.items() if original.sha(Path(p))!=h]
    mismatch=[p for p,h in meta['sources_sha256'].items() if original.sha(Path(p))!=h]
    assert not changed and not mismatch
    records={};ids={}
    with (out/'patients.csv').open() as f:
        for row in csv.DictReader(f):
            cell=row.pop('cell');pid=row.pop('patient');ids.setdefault(cell,[]).append(pid)
            bucket=records.get(cell)
            if bucket is None:bucket={k:[] for k in row};records[cell]=bucket
            for k,v in row.items():bucket[k].append(float(v))
    arrays={};output=[];boots={}
    for cell in list(records):arrays[cell]={k:np.array(v) for k,v in records.pop(cell).items()}
    for c in cells:
        if c['dp']:assert c['spent_max']<=c['B']+1e-8 and c['budget_bound']==c['B']
        if c['comparison']=='A1':assert abs(c['refreshes']-c['param']['K']-1)<1e-10
        if c['comparison'].startswith('B') or (c['comparison']=='A1' and c['method'].startswith('oracle')):
            assert not c['dp'] and c['spent_max'] is None
        a=arrays[c['cell']];n=len(ids[c['cell']]);key=(c['dataset'],n)
        if key not in boots:boots[key]=np.random.default_rng(202613).integers(0,n,(cfg['bootstrap'],n))
        idx=boots[key];r=dict(c)
        samples={'rce':100*a['rce'][idx].mean(axis=1),'cmr':100*a['cmr'][idx].mean(axis=1),
            'wmape':100*a['error'][idx].sum(axis=1)/a['truth'][idx].sum(axis=1)}
        den=a['event_den'][idx].sum(axis=1)
        for lag in range(3):samples['recall'+str(lag)]=None if not den.max() else (100*a['hit'+str(lag)][idx].sum(axis=1)[den>0]/den[den>0])
        for metric,v in samples.items():r[metric+'_ci95']=None if v is None else np.quantile(v,[.025,.975]).tolist()
        if c['comparison']=='A1' and c['method'].startswith('oracle'):
            ref=c['cell'].replace('|oracle_K','|period_K');b=arrays[ref];assert ids[ref]==ids[c['cell']]
            for lag in range(3):
                r['gap_recall'+str(lag)+'_vs_period']=ratio(a['hit'+str(lag)]-b['hit'+str(lag)],a['event_den'])
                values=None if not den.max() else 100*(a['hit'+str(lag)]-b['hit'+str(lag)])[idx].sum(axis=1)[den>0]/den[den>0]
                r['gap_recall'+str(lag)+'_ci95']=None if values is None else np.quantile(values,[.025,.975]).tolist()
            r['gap_rce_vs_period']=float(100*(a['rce']-b['rce']).mean())
            r['gap_rce_ci95']=np.quantile(100*(a['rce']-b['rce'])[idx].mean(axis=1),[.025,.975]).tolist()
        output.append(r)
    # Macro condition means with shared patient draws across repeated scenes
    # and both horizons. No scene, horizon or noise replication is an extra
    # independent patient. Templates are reported separately.
    aggregate=[]
    for source in ['legacy_synthetic','stress','observed']:
        for name in cfg['templates']:
            for B in cfg['budgets']:
                for K in cfg['K']:
                    rr=[r for r in output if r['source']==source and r['template']==name and r['B']==B and r['comparison']=='A1' and r['method']==f'oracle_K{K}' and r['recall1'] is not None]
                    if not rr:continue
                    union=sorted({p for r in rr for p in ids[r['cell']]});mapping={p:i for i,p in enumerate(union)}
                    draws=np.random.default_rng(202613).integers(0,len(union),(cfg['bootstrap'],len(union)))
                    weights=np.stack([np.bincount(x,minlength=len(union)) for x in draws])
                    estimates=[]
                    for r in rr:
                        a=arrays[r['cell']];ref=r['cell'].replace('|oracle_K','|period_K');b=arrays[ref]
                        w=weights[:,[mapping[p] for p in ids[r['cell']]]]
                        den=w@a['event_den'];num=w@(a['hit1']-b['hit1'])
                        estimates.append(np.divide(100*num,den,out=np.full(len(den),np.nan),where=den>0))
                    matrix=np.array(estimates);valid=np.all(np.isfinite(matrix),axis=0)
                    combined=matrix[:,valid].mean(axis=0)
                    aggregate.append(dict(source=source,template=name,B=B,K=K,conditions=len(rr),unique_patients=len(union),
                        mean_gap_recall1=float(np.mean([r['gap_recall1_vs_period'] for r in rr])),
                        gap_recall1_ci95=np.quantile(combined,[.025,.975]).tolist(),macro_valid_draws=int(valid.sum()),
                        min_condition_gap=min(r['gap_recall1_vs_period'] for r in rr),max_condition_gap=max(r['gap_recall1_vs_period'] for r in rr)))
    (out/'analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    (out/'A1_macro_effects.json').write_text(json.dumps(aggregate,indent=2,allow_nan=False))
    fields=['dataset','source','template','T','n','B','comparison','method','param','dp','events','event_patients','recall0','recall1','recall2',
        'precision2','f1_2','delay_success2','failure2','rce','cmr','false_change','stable_output_change_rate','false_events','wmape','refreshes','spent_max',
        'recall1_ci95','rce_ci95','gap_recall1_vs_period','gap_recall1_ci95','gap_rce_vs_period','gap_rce_ci95',
        'validation_feasible','constraint','recovery1','adjacent_recall2','threshold_near_recall2','persistent_recall2','aoii']
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(output)
    audit=dict(previous_files_checked=len(previous),previous_files_changed=changed,run_sources_mismatch=mismatch,
        dp_budget_caps_pass=True,A1_actual_counts_equal=True,oracle_not_claimed_DP=True,cells=len(cells),
        sources_sha256={str(p):original.sha(p) for p in HERE.rglob('*.py')})
    root_sources={}
    for rel in ['utils/utils.py','utils/risk_class.py','preempt/sanitizer.py']:
        p=original.REPO/rel;head=subprocess.check_output(['git','show',meta['git_head']+':'+rel],cwd=original.REPO)
        assert hashlib.sha256(head).hexdigest()==original.sha(p)
        root_sources[str(p)]=original.sha(p)
    audit['original_root_sources_match_recorded_git_head']=True
    audit['original_root_sources_sha256']=root_sources
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    from reports import build
    build(out,output,aggregate,meta,cfg)
    print(out/'SUMMARY.md');print(json.dumps(dict(cells=len(cells),macro_groups=len(aggregate))))


if __name__=='__main__':main()
