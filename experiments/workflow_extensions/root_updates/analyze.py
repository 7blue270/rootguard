"""Patient-level paired bootstrap, plus fixed-cohort noise uncertainty."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

def analyze(path):
    cfg=json.loads((path/'config.json').read_text());cells=json.loads((path/'cells.json').read_text())
    rows={}
    with (path/'patients.csv').open() as f:
        for r in csv.DictReader(f):rows.setdefault(r['cell'],[]).append(r)
    references={(c['dataset'],c['budget_label'],c['method']):c for c in cells}
    boots={};output=[]
    for c in cells:
        rr=rows[c['cell']];n=len(rr)
        if c['dataset'] not in boots:
            boots[c['dataset']]=np.random.default_rng(cfg['seed']+200).integers(0,n,(cfg['bootstrap'],n))
        idx=boots[c['dataset']]
        error=np.array([float(r['mean_abs_error']) for r in rr]);truth=np.array([float(r['truth_weight']) for r in rr]);w=np.array([float(r['wrong_fraction']) for r in rr])
        result=dict(c)
        denominator=truth[idx].sum(axis=1)
        result['wmape_ci95']=np.quantile(100*error[idx].sum(axis=1)/denominator,[.025,.975]).tolist()
        result['rce_ci95']=np.quantile(100*w[idx].mean(axis=1),[.025,.975]).tolist()
        for method in ['static','full_opt','root_opt']:
            ref=references[c['dataset'],c['budget_label'],method];rr2=rows[ref['cell']]
            assert [r['patient'] for r in rr]==[r['patient'] for r in rr2]
            de=error-np.array([float(r['mean_abs_error']) for r in rr2]);dw=w-np.array([float(r['wrong_fraction']) for r in rr2])
            result['delta_wmape_vs_'+method]=float(100*de.sum()/truth.sum())
            result['delta_wmape_vs_'+method+'_ci95']=np.quantile(100*de[idx].sum(axis=1)/denominator,[.025,.975]).tolist()
            result['delta_rce_vs_'+method]=float(100*dw.mean())
            result['delta_rce_vs_'+method+'_ci95']=np.quantile(100*dw[idx].mean(axis=1),[.025,.975]).tolist()
            dr=np.array(c['wmape_per_repeat'])-ref['wmape_per_repeat']
            margin=student_t.ppf(.975,len(dr)-1)*dr.std(ddof=1)/np.sqrt(len(dr))
            result['delta_wmape_vs_'+method+'_mc_ci95']=[float(dr.mean()-margin),float(dr.mean()+margin)]
        output.append(result)
    (path/'analysis.json').write_text(json.dumps(output,indent=2))
    fields=['dataset','source','template','H','n','privacy_track','budget_label','B','method','same_budget','wmape','rce',
       'root_mae_pct_domain','change_miss_rate','false_change_rate','delay_versions_capped','spent_mean','spent_max',
       'release_roots_mean','delta_wmape_vs_full_opt','delta_wmape_vs_full_opt_ci95','delta_wmape_vs_root_opt',
       'delta_wmape_vs_root_opt_ci95','delta_wmape_vs_root_opt_mc_ci95']
    with (path/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(output)
    text=['# Dynamic roots run summary','',f'{len(cells)} cells; patient-paired pointwise bootstrap, no multiplicity correction.','',
          '| Dataset | budget | method | wMAPE (%) | RCE (%) | same budget |','|---|---:|---|---:|---:|---|']
    for c in output:
        if c['source']=='real':text.append(f"| {c['dataset']} | {c['budget_label']} | {c['method']} | {c['wmape']:.3f} | {c['rce']:.3f} | {c['same_budget']} |")
    (path/'SUMMARY.md').write_text('\n'.join(text)+'\n')
    print(path,len(cells),'cells analyzed')
    return output

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);analyze(ap.parse_args().run)
