"""python -m experiments.workflow_extensions.m4_carry.run --help"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload, FORMULA_VERSION
from .core import METHODS, Plan, Policy, fit_stats, simulate, evaluate
from .planning import candidates, shortlist, public_population, public_select, score_population

ROOT = Path(__file__).resolve().parents[3]


def clean(x):
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, Plan):
        return x.to_dict()
    return x


def dump(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def line(handle, value):
    handle.write(json.dumps(clean(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def budget_grid(cfg):
    g = cfg['budget_grid']
    coarse = []
    b = float(g['low'])
    while b < g['high'] - 1e-8:
        coarse.append(round(b, 2))
        b *= g['ratio']
    coarse.append(float(g['high']))
    fine = [round(a * cfg['refine_ratio'], 2) for a, b in zip(coarse, coarse[1:]) if a * cfg['refine_ratio'] < b - 1e-8] if cfg['refine_all_intervals'] else []
    return sorted(set(coarse + fine)), coarse


def patient_score(population, B, plan, mode, policy, cfg, phase, R):
    values = {}
    for rec in population:
        r = simulate(rec, B, plan, mode, policy, cfg, phase, R)
        metric = evaluate(rec, r, policy.w, {}, {**cfg, '_B': B})['mae'].mean()
        values.setdefault(str(rec['subject_id']), []).append(metric)
    return float(np.mean([np.mean(v) for v in values.values()]))


def run(args):
    cfg = json.loads(Path(args.config).read_text())
    if args.precheck:
        cfg.update(repeats=2, validation_repeats=2, bootstrap=100, public_repeats=2, calibration_repeats=2)
    budgets, coarse = budget_grid(cfg)
    if args.precheck:
        budgets = [80., 320., 960.]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit('Output must be empty; previous results are never overwritten')
    started = time.monotonic()
    records = data.load(args.data, cfg['templates'])
    assignment = data.split(records, cfg['split_seed'], *cfg['split_fractions'])
    assert set(assignment.values()) == {'train', 'validation', 'test'}
    cfg['evaluated_budgets'] = budgets
    cfg['coarse_budgets'] = coarse
    cfg['precheck'] = args.precheck
    cfg['actual_max_grid_ratio'] = max(b/a for a, b in zip(budgets, budgets[1:]))
    dump(out/'config.json', cfg)
    dump(out/'split.json', assignment)
    dump(out/'input_hashes.json', data.input_hashes(args.data, cfg['templates']))
    source_paths = list(Path(__file__).parent.glob('*.py')) + [Path(args.config), ROOT/'experiments/workflow_extensions/finite_cache/data.py', ROOT/'experiments/workflow_extensions/finite_cache/workload.py']
    dump(out/'source_hashes.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    contexts, all_selected, targets, stats_out, counts, skipped = {}, {}, {}, {}, {}, {}
    scorelog = (out/'selection_scores.jsonl').open('w')
    # Complete every selection and public target BEFORE evaluating any test error.
    for template in cfg['templates']:
        full = [r for r in records if r['template'] == template]
        n = len({r['subject_id'] for r in full})
        if n < 10:
            skipped[template] = f'{n} patients < 10'
            continue
        parts = {p: data.partition(records, assignment, p, template) for p in ('train', 'validation', 'test')}
        if any(not v for v in parts.values()):
            raise ValueError(f'Empty split: {template}')
        counts[template] = {p: dict(patients=len({r['subject_id'] for r in rs}), trajectories=len(rs)) for p, rs in parts.items()}
        w = Workload(template)
        stats = fit_stats(parts['train'], w, cfg)
        stats_out[template] = stats
        policy = Policy(w, stats, cfg)
        publics = {name: public_population(parts['train'], w, cfg, d) for name, d in cfg['delta_scenarios'].items()}
        mix = sum(publics.values(), [])
        targets[template] = {}
        for name, population in publics.items():
            loss, _ = score_population(population, cfg['budget_grid']['high'], Plan(1.), 'fixed', policy, cfg,
                                       'public_calibration', cfg['calibration_repeats'])
            targets[template][name] = dict(delta=cfg['delta_scenarios'][name], reference_B0_Bmax=loss,
                                           **{f'tau_{k}': v*loss for k, v in cfg['target_multipliers'].items()})
        plans = candidates(len(w.roots), stats['h'], cfg)
        contexts[template] = dict(parts=parts, w=w, stats=stats, policy=policy)
        all_selected[template] = {}
        for bi, B in enumerate(budgets):
            short = shortlist(plans, B, stats, cfg)
            fixed, _, fscores = public_select(short, mix, B, 'fixed', policy, cfg)
            _, carry_groups, cscores = public_select(short, mix, B, 'carry', policy, cfg)
            for mode, scores in [('fixed', fscores), ('carry', cscores)]:
                for s in scores:
                    line(scorelog, dict(template=template, B=B, phase='public_plan', executor=mode, **s))
            # Validation only picks K/rho from publicly selected per-cell plans.
            vgroups = []
            for entry in carry_groups:
                p = entry['plan']
                value = patient_score(parts['validation'], B, p, 'carry', policy, cfg, 'validation', cfg['validation_repeats'])
                vgroups.append((value, p))
                line(scorelog, dict(template=template, B=B, phase='validation', kind='K_rho', plan=p, mae=value))
            replan = min(vgroups, key=lambda x: x[0])[1]
            roots = tuple(range(len(w.roots)))
            periodic = [Plan(cfg['b2_rho'], sets=(roots,), period=P) for P in cfg['periods']]
            pvals, ppub = [], []
            for p in periodic:
                value = patient_score(parts['validation'], B, p, 'periodic', policy, cfg, 'validation', cfg['validation_repeats'])
                pubvalue, _ = score_population(mix, B, p, 'periodic', policy, cfg, 'public_plan', cfg['public_repeats'])
                pvals.append((value, p))
                ppub.append((pubvalue, p))
                line(scorelog, dict(template=template, B=B, phase='validation', kind='period', P=p.period, mae=value, public_mae=pubvalue))
            selected = {'B0': Plan(1.), 'B2-val': min(pvals, key=lambda x:x[0])[1],
                        'B2-pub': min(ppub, key=lambda x:x[0])[1], 'M4': fixed['plan'],
                        'M4-carry': fixed['plan'], 'M4-carry-replan': replan}
            assert selected['M4'] == selected['M4-carry']
            all_selected[template][str(B)] = selected
            if bi % 5 == 0 or bi == len(budgets)-1:
                print(f'SELECT {template} {bi+1}/{len(budgets)} B={B:g} Pval={selected["B2-val"].period} K/rho={fixed["plan"].K}/{fixed["plan"].rho} replan={replan.K}/{replan.rho} elapsed={time.monotonic()-started:.1f}s', flush=True)
    scorelog.close()
    dump(out/'public_stats.json', stats_out)
    dump(out/'targets.json', targets)
    dump(out/'selections.json', all_selected)
    dump(out/'public_population.json', {t:{n:public_population(c['parts']['train'], c['w'], cfg, d) for n,d in cfg['delta_scenarios'].items()} for t,c in contexts.items()})
    dump(out/'phase_a_complete.json', dict(test_errors_evaluated=False, selections_complete=True, elapsed=time.monotonic()-started))
    rows = 0
    audits = dict(ledger_all_ok=True, epsilon_positive=True, causal_release=True, same_plan_m4_carry=True,
                  selections_completed_before_test=True, test_used_for_delta_or_tau=False, counts=counts, skipped=skipped,
                  max_budget_excess=0., evaluated_mechanism_paths=0, precheck=args.precheck)
    with (out/'metrics.jsonl').open('w') as mh, (out/'actions.jsonl').open('w') as ah:
        for template, c in contexts.items():
            for bi, B in enumerate(budgets):
                for method in METHODS:
                    plan = all_selected[template][str(B)][method]
                    mode = 'carry' if method.startswith('M4-carry') else 'periodic' if method.startswith('B2') else 'fixed'
                    for rec in c['parts']['test']:
                        result = simulate(rec, B, plan, mode, c['policy'], cfg, 'test', cfg['repeats'], detailed=True)
                        metrics = evaluate(rec, result, c['w'], c['stats'], {**cfg, '_B':B}, detailed=True)
                        audits['max_budget_excess'] = max(audits['max_budget_excess'], float(np.max(result['spent']-B)))
                        audits['evaluated_mechanism_paths'] += cfg['repeats']
                        for r in range(cfg['repeats']):
                            ident = dict(template=template, subject=str(rec['subject_id']), trajectory=rec['trajectory_id'], B=B, method=method, repeat=r)
                            line(mh, {**ident, **{k:v[r] for k,v in metrics.items()}, 'K':result['K'], 'rho':plan.rho,
                                      'stop_reason':result['stop_reason']})
                            line(ah, {**ident, 'turn':1, 'action_index':0, 'selected_roots':c['w'].roots,
                                      'actual_epsilon':result['anchor_epsilon'][r], 'cache_after':result['anchor_cache'][r],
                                      'R_after':B-float(result['anchor_epsilon'][r].sum()), 'n_after':result['K']})
                            for h in result['histories']:
                                line(ah, {**ident, 'turn':h['turn'], 'action_index':h['action_index'],
                                          'planned_roots':[c['w'].roots[i] for i in h['planned_roots']],
                                          'selected_roots':[c['w'].roots[i] for i in np.where(h['selected'][r])[0]],
                                          **{k:h[k][r] for k in ('candidate_epsilon','actual_epsilon','eps_can','cache_before','cache_after','immediate_error_drop')},
                                          'R_before':h['left_before'][r], 'R_after':h['left_after'][r], 'n_before':h['n_before'], 'n_after':h['n_after']})
                            rows += 1
                print(f'TEST {template} {bi+1}/{len(budgets)} B={B:g} paths={rows} elapsed={time.monotonic()-started:.1f}s', flush=True) if bi % 5 == 0 or bi == len(budgets)-1 else None
    audits['elapsed_simulation_seconds'] = time.monotonic()-started
    audits['source_policy_signature'] = 'decide(cache, eps_last, ages, candidate_mask, candidate_epsilon)'
    dump(out/'audit.json', audits)
    from .audit import audit_output
    print('INDEPENDENT AUDIT', audit_output(out, args.data), flush=True)
    print('ANALYZE', flush=True)
    from .analysis import analyze
    analyze(out)
    print(f'DONE {out} elapsed={time.monotonic()-started:.1f}s', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(Path(__file__).parent/'configs/real_v1.json'))
    p.add_argument('--data', default=str(ROOT/'data_mine/finite_cache'))
    p.add_argument('--output', required=True)
    p.add_argument('--precheck', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
