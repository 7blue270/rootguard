"""Recompute wMAPE from RQ1 adversarial results (no rerunning experiments).

Reads from results_rq1_adversarial_tN/, computes wMAPE + bootstrap SE, writes to
results_rq1_adversarial_tN_wmape/ with the same directory structure.

Usage:
    python recompute_wmape_rq1_adv.py [--turns N]
"""
import json
import os
import sys
import argparse
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--turns", type=int, default=6)
parser.add_argument("--budget_mode", choices=["fixed", "kplus1", "2kplus1", "3kplus1"], default="fixed")
_args = parser.parse_args()

if _args.budget_mode == "kplus1":
    SRC_BASE = "./results_rq1_adversarial_kplus1"
    DST_BASE = "./results_rq1_adversarial_kplus1_wmape"
elif _args.budget_mode == "2kplus1":
    SRC_BASE = "./results_rq1_adversarial_2kplus1"
    DST_BASE = "./results_rq1_adversarial_2kplus1_wmape"
elif _args.budget_mode == "3kplus1":
    SRC_BASE = "./results_rq1_adversarial_3kplus1"
    DST_BASE = "./results_rq1_adversarial_3kplus1_wmape"
else:
    SRC_BASE = "./results_rq1_adversarial_t{}".format(_args.turns)
    DST_BASE = "./results_rq1_adversarial_t{}_wmape".format(_args.turns)
EPSILONS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
TEMPLATES = ['ANEMIA', 'AIP', 'CONICITY', 'VASCULAR', 'FIB4', 'TYG', 'HOMA', 'NLR']
METHODS = [
    'exp_all', 'exp_roots', 'exp_opt',
    'blap_all', 'blap_roots', 'blap_opt',
    'stair_all', 'stair_roots', 'stair_opt',
]
N_BOOTSTRAP = 1000
RNG = np.random.RandomState(42)


def compute_wmape(gt, san):
    """wMAPE = sum(|san - gt|) / sum(|gt|) * 100."""
    abs_errors = np.abs(san - gt)
    abs_actuals = np.abs(gt)
    denom = np.sum(abs_actuals)
    if denom == 0:
        return 0.0
    return float(np.sum(abs_errors) / denom * 100)


def bootstrap_se(gt, san, n_boot=N_BOOTSTRAP):
    """Bootstrap standard error for wMAPE."""
    N = len(gt)
    abs_err = np.abs(san - gt)
    abs_gt = np.abs(gt)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.randint(0, N, size=N)
        denom = np.sum(abs_gt[idx])
        if denom == 0:
            boots[b] = 0.0
        else:
            boots[b] = np.sum(abs_err[idx]) / denom * 100
    return float(np.std(boots))


n_processed = 0
n_skipped = 0

for eps in EPSILONS:
    print("\n{}\nEpsilon = {}\n{}".format("=" * 60, eps, "=" * 60), flush=True)

    eps_summary = {"epsilon": eps, "templates": {}}

    for tmpl in TEMPLATES:
        eps_summary["templates"][tmpl] = {}

        for method in METHODS:
            src_file = "{}/epsilon_{}/{}/{}_results.json".format(SRC_BASE, eps, tmpl, method)
            if not os.path.exists(src_file):
                print("  SKIP (not found): {}".format(src_file), flush=True)
                n_skipped += 1
                continue

            with open(src_file) as f:
                data = json.load(f)

            errors = data['per_sample_errors']
            gt = np.array([e['gt_target'] for e in errors])
            san = np.array([e['sanitized_target'] for e in errors])

            wmape = compute_wmape(gt, san)
            se = bootstrap_se(gt, san)

            old_summary = data['summary']
            new_summary = {
                'wmape': round(wmape, 4),
                'wmape_se': round(se, 4),
                'raw_mae': old_summary['raw_mae'],
                'raw_std': old_summary['raw_std'],
                'risk_mae': old_summary['risk_mae'],
                'risk_std': old_summary['risk_std'],
                'n_samples': old_summary['n_samples'],
            }

            dst_dir = "{}/epsilon_{}/{}".format(DST_BASE, eps, tmpl)
            os.makedirs(dst_dir, exist_ok=True)
            dst_file = "{}/{}_results.json".format(dst_dir, method)

            out_data = {
                'method': data['method'],
                'epsilon': data['epsilon'],
                'template': data['template'],
                'total_budget': data.get('total_budget', 6 * data['epsilon']),
                'summary': new_summary,
                'per_sample_errors': data['per_sample_errors'],
            }
            if 'allocation_info' in data:
                out_data['allocation_info'] = data['allocation_info']

            with open(dst_file, 'w') as f:
                json.dump(out_data, f, indent=2, default=str)

            eps_summary["templates"][tmpl]["{}_wmape".format(method)] = round(wmape, 4)
            eps_summary["templates"][tmpl]["{}_wmape_se".format(method)] = round(se, 4)
            eps_summary["templates"][tmpl]["{}_raw_mae".format(method)] = old_summary['raw_mae']
            eps_summary["templates"][tmpl]["{}_risk_mae".format(method)] = old_summary['risk_mae']

            n_processed += 1

        # Print template summary
        best_method = None
        best_wmape = float('inf')
        for method in METHODS:
            key = "{}_wmape".format(method)
            if key in eps_summary["templates"][tmpl]:
                val = eps_summary["templates"][tmpl][key]
                if val < best_wmape:
                    best_wmape = val
                    best_method = method
        if best_method:
            print("  {:12s}  best={}({:.2f}%)".format(tmpl, best_method, best_wmape), flush=True)

    summary_path = "{}/epsilon_{}/summary.json".format(DST_BASE, eps)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(eps_summary, f, indent=2)

print("\nDone! Processed {} files, skipped {}.".format(n_processed, n_skipped))
print("Results saved to {}/".format(DST_BASE))
