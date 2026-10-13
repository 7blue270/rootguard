"""Broad dynamic-root exploration; all raw secrets stay in offline evaluator."""
import argparse
import csv
import datetime
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import scipy
from mechanisms import original,run,metrics
from data import synthetic,real,MIMIC_DIR

def datasets(cfg):
    for ti,name in enumerate(cfg['templates']):
        for H in cfg['horizons']:
            for scenario in cfg['scenarios']:
                amps=[0.] if scenario in ['no_updates','constant'] else cfg['amplitudes']
                for amp in amps:
                    d=synthetic(name,cfg['synthetic_patients'],H,scenario,amp,cfg.get('data_seed',481516)+ti)
                    yield f'syn_{name}_{scenario}_H{H}_a{amp}',d
            if name=='ANEMIA':
                for scenario in ['irrelevant','correlated']:
                    for amp in cfg['amplitudes']:
                        yield f'syn_{name}_{scenario}_H{H}_a{amp}',synthetic(name,cfg['synthetic_patients'],H,scenario,amp,cfg.get('data_seed',481516)+ti)
    for H in cfg['horizons']:
        yield f'real_ANEMIA_H{H}',real(H,cfg['real_patients'],Path(cfg.get('mimic_dir',str(MIMIC_DIR))))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    cfg=json.loads(args.config.read_text())
    if args.out.exists():raise ValueError('Use a new run ID, outputs are never overwritten')
    args.out.mkdir(parents=True)
    (args.out/'config.json').write_text(json.dumps(cfg,indent=2))
    start=time.time()
    input_paths=[original.REPO/'data/nhanes_benchmark_200.json',original.REPO/'data/holdout_population_means.json',
                 original.REPO/'utils/utils.py',original.REPO/'preempt/sanitizer.py',original.REPO/'utils/risk_class.py',
                 Path(__file__).parent.parent/'target_mismatch/core.py',
                 Path(cfg.get('mimic_dir',str(MIMIC_DIR)))/'labevents.csv.gz',args.config.resolve()]
    input_paths+=sorted(Path(__file__).parent.rglob('*.py'))
    meta={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'git_head':subprocess.check_output(['git','-C',str(original.REPO),'rev-parse','HEAD'],text=True).strip(),
          'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
          'sources_sha256':{str(p):original.sha(p) for p in input_paths},'datasets':{},
          'privacy_contract':'fixed public layout; d_inf of normalized root indices; B-mDP; E=(m-1)*B sufficient standard trajectory LDP'}
    cells=[]; saved={};task_draws=0
    patient_fields=['cell','patient','truth_weight','mean_abs_error','wrong_fraction','root_error',
                    'miss_num','change_den','false_num','stable_den']
    with (args.out/'patients.csv').open('w',newline='') as pf:
        writer=csv.DictWriter(pf,fieldnames=patient_fields);writer.writeheader()
        for di,(dataset_id,d) in enumerate(datasets(cfg)):
            name=d['name']; x=d['x'];n,H,k=x.shape
            truth=original.compute(name,x);gt=original.classes(name,truth)
            old=np.repeat(truth[:,0:1],H,axis=1)
            audit=dict(d['audit'],n=n,H=H,root_counts=d['plan'].sum(axis=0).tolist(),
                class_counts=np.bincount(gt.ravel(),minlength=3).tolist(),
                real_class_changes=int((gt[:,1:]!=gt[:,:-1]).sum()),
                noiseless_stale_wmape=float(100*np.abs(old-truth).sum()/np.abs(truth).sum()),
                noiseless_stale_rce=float(100*(original.classes(name,old)!=gt).mean()))
            meta['datasets'][dataset_id]=audit
            rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],di]))
            u=rng.random((n,cfg['repeats'],H,k))
            lap=rng.laplace(size=(n,cfg['repeats'],H))
            # Public/evaluator oracle for change diagnosis, not a deployable
            # change detector. All evaluated release plans are public or DP.
            for budget_label in cfg.get('standard_eps',cfg['budgets']):
                B=budget_label/(cfg['grid_size']-1) if 'standard_eps' in cfg else budget_label
                for method in cfg['methods']:
                    period=2 if method=='period2' else 4 if method=='period4' else 1
                    threshold=100. if method=='predictive100' else 30.
                    actual_B=B*H if method=='fixed_per_version' else B
                    implementation='full_opt' if method=='fixed_per_version' else method
                    cache,ledger=run(name,x,d['plan'],d['mu'],actual_B,implementation,u,lap,cfg['grid_size'],period=period,threshold=threshold)
                    result=metrics(name,x,cache)
                    error=result['error'];wrong=result['wrong']
                    pe=error.mean(axis=(1,2));pw=wrong.mean(axis=(1,2));pr=result['root_error'].mean(axis=(1,2,3))
                    denom=np.abs(truth).mean(axis=1)
                    cid=f'{dataset_id}|{budget_label}|{method}'
                    spent=ledger.get('spent')
                    calls=ledger['release_roots']
                    c={'cell':cid,'dataset':dataset_id,'source':'real' if dataset_id.startswith('real') else 'synthetic',
                       'template':name,'H':H,'n':n,'budget_label':budget_label,'B':B,
                       'actual_budget_bound':None if not ledger['certified'] else actual_B,
                       'same_budget':ledger['certified'] and actual_B<=B+1e-10,
                       'standard_eps_bound':None if not ledger['certified'] else (cfg['grid_size']-1)*actual_B,'privacy_track':'standard' if 'standard_eps' in cfg else 'metric',
                       'method':method,'certified':ledger['certified'],
                       'wmape':float(100*pe.sum()/denom.sum()),'rce':float(100*pw.mean()),'root_mae_pct_domain':float(100*pr.mean()),
                       'change_miss_rate':None if result['change_den'].sum()==0 else float(100*result['miss_num'].sum()/result['change_den'].sum()),
                       'false_change_rate':None if result['stable_den'].sum()==0 else float(100*result['false_num'].sum()/result['stable_den'].sum()),
                       'delay_versions_capped':result['delay_mean'],
                       'spent_mean':None if spent is None else float(np.mean(spent)),
                       'spent_max':None if spent is None else float(np.max(spent)),
                       'release_roots_mean':float(np.mean(calls)),
                       'refresh_fraction':float(ledger['refresh_flags'].mean()),
                       'caps':ledger.get('cap'),'counts':ledger.get('counts'),
                       'wmape_per_repeat':(100*error.sum(axis=(0,2))/np.abs(truth).sum()).tolist(),
                       'rce_per_repeat':(100*wrong.mean(axis=(0,2))).tolist()}
                    if c['certified'] and c['spent_max']>actual_B+1e-8:raise RuntimeError('Budget exceeded')
                    cells.append(c)
                    for i,pid in enumerate(d['ids']):
                        writer.writerow(dict(cell=cid,patient=pid,truth_weight=denom[i],mean_abs_error=pe[i],wrong_fraction=pw[i],
                            root_error=pr[i],miss_num=result['miss_num'][i],change_den=result['change_den'][i],
                            false_num=result['false_num'][i],stable_den=result['stable_den'][i]))
                    task_draws+=n*cfg['repeats']*H
                    if (dataset_id=='real_ANEMIA_H8' or dataset_id=='syn_ANEMIA_jump_H8_a0.05') and budget_label==.3:
                        saved[cid+':true']=truth
                        saved[cid+':target']=original.compute(name,cache).astype(np.float32)
                        saved[cid+':root_cache']=cache.astype(np.float32)
                        saved[cid+':refresh']=ledger['refresh_flags']
            print(f'{dataset_id}: n={n}, class transitions={audit["real_class_changes"]}, complete',flush=True)
    np.savez_compressed(args.out/'selected_trajectories.npz',**saved)
    (args.out/'cells.json').write_text(json.dumps(cells,indent=2))
    meta.update(elapsed_seconds=time.time()-start,cell_count=len(cells),task_evaluations=task_draws)
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({'cells':len(cells),'task_evaluations':task_draws,'seconds':meta['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
