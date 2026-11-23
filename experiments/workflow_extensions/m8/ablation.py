"""Frozen, patient-paired attribution study for original M8."""
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..finite_cache import data
from ..m7.audit import ledger
from ..m7.engine import Branch, public_view
from . import load_original
from .original_study import DIR, OLD, BUDGETS, order, noise_for, simulate

OUT = Path(__file__).resolve().parent.parent / 'outputs/m8/attribution_20261120'
ARCHIVE = OLD.parent / 'physionet_fusion_20261120'
METHODS = ('rootguard', 'cal_static', 'cal_periodic', 'fixed_dynamic', 'dynamic_fusion',
           'm8', 'forced_plan', 'm8_no_fusion')
CONTRASTS = [('cal_static', 'rootguard'), ('cal_periodic', 'cal_static'),
             ('fixed_dynamic', 'cal_periodic'), ('dynamic_fusion', 'fixed_dynamic'),
             ('m8', 'dynamic_fusion'), ('forced_plan', 'fixed_dynamic'),
             ('m8', 'forced_plan'), ('m8', 'm8_no_fusion'), ('m8', 'rootguard'),
             ('m8', 'cal_static'), ('m8', 'cal_periodic')]

def write_csv(filename, rows):
    with (OUT / filename).open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def methods(plan):
    static, periodic, dynamic, routed = (Branch(**plan[m]) for m in ('static','periodic','dynamic','route'))
    return (replace(static, calibration='none', shrink=0.), static, periodic,
            replace(dynamic, fusion=False), replace(dynamic, fusion=True), routed,
            dynamic, replace(routed, fusion=False))

def main():
    OUT.mkdir(exist_ok=True, parents=True)
    ctx, plans = load_original(DIR)
    split = json.loads((DIR/'split.json').read_text())
    test = sorted(data.partition(data.load(DIR,['BUN_CR']), split, 'test', 'BUN_CR'), key=order)
    cfg = json.loads((Path(__file__).with_name('original_protocol.json')).read_text())
    assert len(test) == cfg['test']
    (OUT/'protocol.json').write_text(json.dumps(dict(
        goal='Attribute original M8 MAE at fixed total budget', skill='AutoResearch plan and bounded verification',
        dataset=cfg['dataset'], budgets=BUDGETS, repeats=4, patients=len(test),
        sampler='archived native float replay; common noise by patient, budget, turn, root and repeat',
        contrasts=CONTRASTS, high_cut=cfg['threshold'],
        stopping_rule='Complete six prescribed arms and two matched routing controls; verify archive parity'),
        indent=2))
    refs = {}
    with (ARCHIVE/'paths.csv').open() as handle:
        for r in csv.DictReader(handle):
            refs[float(r['B']),r['subject'],r['method']] = float(r['mae'])
    names = {'rootguard':'native_static','cal_static':'static','cal_periodic':'periodic','m8':'route','forced_plan':'dynamic','fixed_dynamic':'matched_latest'}
    err = np.empty((len(test),4,4,len(METHODS),2))
    totals = np.empty((len(test),4,len(METHODS),2))
    scales = np.empty(len(test))
    high = np.zeros(len(test), bool)
    dense = np.zeros(len(test), bool)
    max_archive_diff = 0.
    for i,r in enumerate(test):
        x = data.replay(r)
        truth = ctx.work.evaluate(x)['ratio']
        r['_raw'] = ctx.work.normalize(x)
        r['_public'] = public_view(r)
        scales[i] = np.abs(truth).mean()
        high[i] = np.ptp(truth) > cfg['threshold']
        dense[i] = len(r['events']) >= 8
        for j,B in enumerate(BUDGETS):
            branches = methods(plans[f'{B}/{int(dense[i])}'])
            noise = noise_for(r,B,4)
            memo = {}
            for k,(name,branch) in enumerate(zip(METHODS,branches)):
                if branch not in memo:
                    result = simulate(ctx,r,B,branch,noise,True)
                    ledger(result,B,branch)
                    assert np.allclose(result['spent'],B,rtol=0,atol=1e-9)
                    e = abs(ctx.work.from_cache(result['caches'])['ratio']-truth)
                    memo[branch] = (np.stack([e.mean(1),np.quantile(e,.95,axis=1)],axis=1),
                                    [result['refreshes'].mean(),result['roots'].mean()])
                err[i,j,:,k], totals[i,j,k] = memo[branch]
                if name in names:
                    diff = abs(err[i,j,:,k,0].mean()-refs[B,r['subject_id'],names[name]])
                    max_archive_diff = max(max_archive_diff,diff)
        if (i+1)%400 == 0: print(f'{i+1}/{len(test)} patients complete',flush=True)
    assert max_archive_diff < 1e-10
    np.savez_compressed(OUT/'paired.npz', errors=err, costs=totals, denom=scales, high=high, dense=dense,
                        subjects=[r['subject_id'] for r in test], methods=METHODS, budgets=BUDGETS)
    rng = np.random.default_rng(819153)
    contrasts=[]; summary=[]
    for group,mask in [('all',np.ones(len(test),bool)),('high',high),('low',~high),('high_sparse',high&~dense)]:
        e = err[mask]
        n = len(e)
        stats = np.empty((1000,4,len(METHODS),3))
        for i in range(len(stats)):
            subject = rng.integers(n,size=(n,1,1))
            repeat = rng.integers(4,size=(n,4,4))
            sampled = e[subject,np.arange(4)[None,:,None],repeat].mean(axis=(0,2))
            stats[i,:,:,:2] = sampled
            stats[i,:,:,2] = 100*sampled[:,:,0]/scales[mask][subject[:,0,0]].mean()
        center = e.mean(axis=(0,2))
        point = np.concatenate([center,100*center[:,:,:1]/scales[mask].mean()],axis=2)
        cost = totals[mask].mean(0)
        for label,j in [('avg',slice(None)),*((str(b),slice(k,k+1)) for k,b in enumerate(BUDGETS))]:
            means=point[j].mean(0); boot=stats[:,j].mean(1)
            for k,m in enumerate(METHODS):
                summary.append(dict(group=group,B=label,n=n,method=m,mae=means[k,0],p95=means[k,1],wmape=means[k,2],
                                    refreshes=cost[j,k,0].mean(),roots=cost[j,k,1].mean()))
            for a,b in CONTRASTS:
                x,y=METHODS.index(a),METHODS.index(b)
                for k,metric in enumerate(('mae','p95','wmape')):
                    delta=boot[:,x,k]-boot[:,y,k]
                    lo,hi=np.quantile(delta,[.025,.975])
                    contrasts.append(dict(group=group,B=label,n=n,method=a,reference=b,metric=metric,
                                          delta=means[x,k]-means[y,k],lo=lo,hi=hi))
    write_csv('summary.csv',summary)
    write_csv('paired.csv',contrasts)
    (OUT/'verification.json').write_text(json.dumps(dict(patient_budget_repetitions=len(test)*4*4,
        max_archive_mae_difference=max_archive_diff, full_budget_assertion=True),indent=2))
    print('Attribution completed',flush=True)

if __name__ == '__main__':
    main()
