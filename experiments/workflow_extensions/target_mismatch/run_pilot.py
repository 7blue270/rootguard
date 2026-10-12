"""Run deterministic mechanism-level function-mismatch pilot, no LLM required."""
from __future__ import annotations
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
from core import REPO, ROOTS, TARGETS, DOMAINS, compute, classes, sensitivities, allocate, noisy_roots, proxy, sha

def load_data(name, n):
    records = json.loads((REPO / 'data/nhanes_benchmark_200.json').read_text())[name][:n]
    rows, stored, ids = [], [], []
    for record in records:
        vals = {}
        for turn in record['turns']:
            vals.update(turn.get('private_value', {}))
        rows.append([float(vals[r]) for r in ROOTS[name]])
        stored.append(float(vals[TARGETS[name]]))
        ids.append(record['id'])
    x = np.asarray(rows)
    y = compute(name, x)
    if len(set(ids)) != len(ids) or not np.allclose(y, stored, rtol=1e-8, atol=1e-8):
        raise RuntimeError(f'Duplicate IDs or ground-truth formula mismatch: {name}')
    mu_data = json.loads((REPO / 'data/holdout_population_means.json').read_text())
    mu = np.array([mu_data['per_template_means'][name][r] for r in ROOTS[name]])
    return x, y, mu, ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/pilot.json')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.out.exists():
        raise RuntimeError('Output exists; use a new run ID to preserve provenance')
    args.out.mkdir(parents=True)
    (args.out / 'config.json').write_text(json.dumps(cfg, indent=2))
    start = time.time()
    source_paths = [REPO / 'utils/utils.py', REPO / 'utils/risk_class.py', REPO / 'preempt/sanitizer.py',
                    REPO / 'utils/med_domain/all_templates.py', REPO / 'data/nhanes_benchmark_200.json',
                    REPO / 'data/holdout_population_means.json'] + sorted(Path(__file__).parent.rglob('*.py')) + [args.config.resolve()]
    metadata = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'git_head': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                'source_sha256': {str(p.relative_to(REPO)): sha(p) for p in source_paths},
                'mechanism': 'original RQ1 unrounded discrete exponential PMF',
                'optimizer_indices': 'original precomputation default grid midpoint',
                'data_audit': {}}
    cell_rows, patient_rows, arrays = [], [], {}
    for ti, name in enumerate(cfg['templates']):
        x, y, mu, ids = load_data(name, cfg['patients'])
        k = len(ROOTS[name])
        widths = np.array([DOMAINS[name][r][1] - DOMAINS[name][r][0] for r in ROOTS[name]])
        in_domain = np.ones(len(x), bool)
        outside = {}
        for j, r in enumerate(ROOTS[name]):
            lo, hi = DOMAINS[name][r]
            mask = (x[:, j] < lo) | (x[:, j] > hi)
            outside[r] = int(mask.sum())
            in_domain &= ~mask
        metadata['data_audit'][name] = {'n': len(x), 'unique_ids': len(set(ids)), 'out_of_domain_by_root': outside,
            'in_domain_n': int(in_domain.sum()), 'mu': dict(zip(ROOTS[name], mu.tolist())),
            'bounds': {r: DOMAINS[name][r] for r in ROOTS[name]},
            'class_counts': np.bincount(classes(name, y), minlength=3).tolist(),
            'no_noise_max_abs_error': float(np.max(np.abs(y - compute(name, x))))}
        uniforms = np.random.default_rng(np.random.SeedSequence([cfg['seed'], ti])).random((len(x), cfg['repeats'], k))
        true_s = sensitivities(name, mu)
        conditions = [('roots', '', '', 0.), ('opt_true', '', '', 0.)]
        for family in cfg['families']:
            for root in ROOTS[name]:
                for level in cfg['levels']:
                    if true_s[ROOTS[name].index(root)] == 0 and level < 0:
                        continue
                    conditions.append(('opt_proxy', family, root, level))
        cache = {}
        for B in cfg['budgets']:
            for method, family, root, level in conditions:
                if method == 'roots':
                    sens, eps, yh = true_s, np.full(k, B / k), None
                elif method == 'opt_true':
                    sens, yh = true_s, None
                    eps = allocate(name, sens, B, cfg['grid_size'], cfg['epsilon_min'])
                else:
                    yh, sens = proxy(name, x, mu, family, ROOTS[name].index(root), level)
                    eps = allocate(name, sens, B, cfg['grid_size'], cfg['epsilon_min'])
                cid = f'{name}|{B}|{method}|{family}|{root}|{level}'
                ck = (B, tuple(np.round(eps, 12)))
                if ck not in cache:
                    z = noisy_roots(name, x, eps, uniforms, cfg['grid_size'])
                    est = compute(name, z)
                    errors = np.abs(est - y[:, None])
                    wrong = classes(name, est) != classes(name, y)[:, None]
                    cache[ck] = (errors, wrong)
                errors, wrong = cache[ck]
                pe, pw = errors.mean(axis=1), wrong.mean(axis=1)
                proxy_wmape = None if yh is None else float(100 * np.abs(yh - y).sum() / np.abs(y).sum())
                importance = sens * widths
                reference = true_s * widths
                row = {'cell': cid, 'template': name, 'B': B, 'method': method, 'family': family,
                       'root': root, 'level': level, 'wmape': float(100 * pe.sum() / np.abs(y).sum()),
                       'rce': float(100 * pw.mean()), 'proxy_wmape': proxy_wmape,
                       'importance_l1': float(np.abs(importance/importance.sum() - reference/reference.sum()).sum()),
                       'eps': dict(zip(ROOTS[name], eps.tolist())), 'sens': dict(zip(ROOTS[name], sens.tolist()))}
                cell_rows.append(row)
                arrays[cid + ':error'] = errors.astype(np.float32)
                arrays[cid + ':wrong'] = wrong
                for i, pid in enumerate(ids):
                    patient_rows.append({'cell': cid, 'patient': pid, 'truth': float(y[i]), 'mean_abs_error': float(pe[i]),
                                         'wrong_fraction': float(pw[i]), 'in_domain': bool(in_domain[i])})
            print(f'{name} B={B}: {len(conditions)} cells complete', flush=True)
    (args.out / 'cells.json').write_text(json.dumps(cell_rows, indent=2))
    with (args.out / 'patients.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(patient_rows[0]))
        writer.writeheader()
        writer.writerows(patient_rows)
    np.savez_compressed(args.out / 'draws.npz', **arrays)
    metadata['elapsed_seconds'] = time.time() - start
    metadata['cell_count'] = len(cell_rows)
    metadata['task_draw_count'] = len(patient_rows) * cfg['repeats']
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'cells': len(cell_rows), 'task_draws': metadata['task_draw_count'], 'seconds': metadata['elapsed_seconds']}), flush=True)

if __name__ == '__main__':
    main()
