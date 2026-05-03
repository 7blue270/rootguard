"""RQ1 Adversarial: Target utility under worst-case 6-turn adversary.

Threat model:
  - t=6 turns, budget B = 6*epsilon
  - M-All: sanitize target T(x) directly with epsilon (single draw)
  - M-Roots: sanitize roots uniformly with 6*epsilon/k each, post-process target
  - M-Opt: allocate 6*epsilon optimally across roots, post-process target

Usage:
  python run_rq1_adversarial.py --method exp_all
  python run_rq1_adversarial.py --method blap_roots
  python run_rq1_adversarial.py --method stair_opt
"""
# _REPO_ROOT_BOOTSTRAP: ensure repo root is on sys.path so that
# 'from utils.*' / 'from preempt.*' imports resolve when this script
# is run from its own subfolder.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, sys, os, time, argparse, math, random
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

METHODS = [
    "exp_all", "exp_roots", "exp_opt",
    "blap_all", "blap_roots", "blap_opt",
    "stair_all", "stair_roots", "stair_opt",
]

parser = argparse.ArgumentParser()
parser.add_argument("--method", required=True, choices=METHODS)
parser.add_argument("--turns", type=int, default=6, help="Number of adversarial turns (B = turns*epsilon)")
parser.add_argument("--budget_mode", choices=["fixed", "kplus1", "2kplus1", "3kplus1"], default="fixed",
                    help="fixed: B=turns*eps; kplus1: B=(k+1)*eps; 2kplus1: B=(2k+1)*eps; 3kplus1: B=(3k+1)*eps")
args = parser.parse_args()

METHOD = args.method
T = args.turns
BUDGET_MODE = args.budget_mode
if BUDGET_MODE == "fixed":
    RESULTS_BASE = "./results_rq1_adversarial_t{}".format(T)
elif BUDGET_MODE == "kplus1":
    RESULTS_BASE = "./results_rq1_adversarial_kplus1"
elif BUDGET_MODE == "2kplus1":
    RESULTS_BASE = "./results_rq1_adversarial_2kplus1"
else:
    RESULTS_BASE = "./results_rq1_adversarial_3kplus1"
EPSILONS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
N = 200
NUM_CANDIDATES = 1000

print("Method: {}".format(METHOD), flush=True)

# ── Load data ────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
with open('../data/nhanes_benchmark_200.json', 'r') as f:
    samples = json.load(f)

from utils.med_domain.all_templates import (
    template_nodes, template_edges, template_expressions,
    template_target_keys, CDC_POPULATION_MEANS,
)
from utils.utils import get_topological_order
from preempt.sanitizer import Sanitizer, MEDICAL_DOMAINS, set_template_domains
from utils.risk_class import get_risk_class

with open('../data/holdout_population_means.json', 'r') as f:
    _holdout_data = json.load(f)
HOLDOUT_MEANS = _holdout_data['per_template_means']

SKIP_NODES = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
              "ppi_risk", "tyg_class", "homa_class", "nlr_class"]

fields = list(samples.keys())
for key in fields:
    samples[key] = samples[key][:N]

# Pre-extract ground truth values
gt_values = {}
for key in fields:
    gt_values[key] = [get_topological_order(samples[key][i], template_edges)[1]
                      for i in range(N)]

# ── Noise functions ──────────────────────────────────────────────────────
_san = Sanitizer()

def noise_exp(value, epsilon, lo, hi):
    return _san.M_exponential_discrete(value, epsilon, lo, hi, NUM_CANDIDATES)

from baselines.bounded_laplace_baseline import bounded_laplace_noise
def noise_blap(value, epsilon, lo, hi):
    return bounded_laplace_noise(value, epsilon, lo, hi, NUM_CANDIDATES)

from baselines.staircase_baseline import staircase_noise
def noise_stair(value, epsilon, lo, hi):
    return staircase_noise(value, epsilon, lo, hi, NUM_CANDIDATES)

NOISE_FN = {"exp": noise_exp, "blap": noise_blap, "stair": noise_stair}

# ── Allocation functions ─────────────────────────────────────────────────
from utils.utils import (
    optimal_budget_allocation_exact,
    optimal_budget_allocation_blap,
    optimal_budget_allocation_staircase,
    compute_population_mean_sensitivities,
)

ALLOC_FN = {
    "exp": optimal_budget_allocation_exact,
    "blap": optimal_budget_allocation_blap,
    "stair": optimal_budget_allocation_staircase,
}


# ── Template info helpers ────────────────────────────────────────────────
def get_template_info(template_name):
    """Return (target_node, root_nodes, derived_order) for a template."""
    target_node = template_target_keys[template_name]
    root_nodes = [n for n, parents in template_nodes.items()
                  if parents is None and n in _template_roots(template_name)]
    # Derived nodes in topological order (excluding roots and skip nodes)
    all_nodes = _template_all_nodes(template_name)
    derived_order = [n for n in all_nodes if n not in root_nodes and n not in SKIP_NODES]
    return target_node, root_nodes, derived_order


def _template_roots(template_name):
    """Return set of root nodes used by this template."""
    TEMPLATE_ROOTS = {
        "ANEMIA": ["hb", "hct", "rbc"],
        "FIB4": ["age", "ast", "alt", "plt"],
        "AIP": ["tc", "hdl", "tg"],
        "CONICITY": ["wt", "ht", "waist"],
        "VASCULAR": ["sbp", "dbp"],
        "TYG": ["tg", "glu"],
        "HOMA": ["glu", "ins"],
        "NLR": ["neu", "lym"],
    }
    return TEMPLATE_ROOTS[template_name]


def _template_all_nodes(template_name):
    """Return all non-skip nodes for this template in topo order."""
    TEMPLATE_NODES = {
        "ANEMIA": ["hb", "hct", "rbc", "mcv", "mch", "mchc"],
        "FIB4": ["age", "ast", "alt", "plt", "fib4_prod", "fib4_denom", "fib4"],
        "AIP": ["tc", "hdl", "tg", "non_hdl", "ldl", "aip"],
        "CONICITY": ["wt", "ht", "waist", "bmi", "wthr", "conicity"],
        "VASCULAR": ["sbp", "dbp", "pp", "map", "mbp", "ppi"],
        "TYG": ["tg", "glu", "tyg_prod", "tyg_half", "tyg"],
        "HOMA": ["glu", "ins", "homa_prod", "homa"],
        "NLR": ["neu", "lym", "nlr_sum", "nlr_diff", "nlr"],
    }
    return TEMPLATE_NODES[template_name]


def postprocess_derived(gt_vals, noised_roots, derived_order):
    """Compute derived nodes from noised roots using DAG expressions."""
    values = dict(noised_roots)
    for node in derived_order:
        expr = template_expressions.get(node)
        if expr is None:
            values[node] = gt_vals.get(node, 0)
            continue
        try:
            values[node] = eval(
                expr,
                {"random": random, "round": round, "np": np, "math": math},
                values
            )
        except Exception:
            values[node] = gt_vals.get(node, 0)
    return values


# ── Run method ───────────────────────────────────────────────────────────
def run_method_single(method, gt_vals, template_name, epsilon):
    """Run one method on one sample. Returns (sanitized_target, allocation_info_or_None)."""
    mech = method.split("_")[0]       # "exp", "blap", "stair"
    variant = method.split("_", 1)[1]  # "all", "roots", "opt"
    noise_fn = NOISE_FN[mech]
    target_node, root_nodes, derived_order = get_template_info(template_name)
    k = len(root_nodes)
    if BUDGET_MODE == "kplus1":
        B = (k + 1) * epsilon
    elif BUDGET_MODE == "2kplus1":
        B = (2 * k + 1) * epsilon
    elif BUDGET_MODE == "3kplus1":
        B = (3 * k + 1) * epsilon
    else:
        B = T * epsilon

    if variant == "all":
        # M-All: sanitize target directly with epsilon
        lo, hi = MEDICAL_DOMAINS.get(target_node, MEDICAL_DOMAINS["_default"])
        target_gt = float(gt_vals[target_node])
        target_noised = noise_fn(target_gt, epsilon, lo, hi)
        return target_noised, None

    elif variant == "roots":
        # M-Roots: uniform allocation B/k per root
        k = len(root_nodes)
        eps_per_root = B / k
        noised_roots = {}
        for root in root_nodes:
            lo, hi = MEDICAL_DOMAINS.get(root, MEDICAL_DOMAINS["_default"])
            noised_roots[root] = noise_fn(float(gt_vals[root]), eps_per_root, lo, hi)
        values = postprocess_derived(gt_vals, noised_roots, derived_order)
        return float(values[target_node]), None

    elif variant == "opt":
        # M-Opt: optimal allocation of B across roots
        topo_order = root_nodes + derived_order
        holdout = HOLDOUT_MEANS[template_name]
        sensitivities = compute_population_mean_sensitivities(
            target_node, root_nodes, template_expressions,
            SKIP_NODES, topo_order, template_nodes, holdout)
        domain_widths = {}
        for r in root_nodes:
            lo, hi = MEDICAL_DOMAINS.get(r, MEDICAL_DOMAINS["_default"])
            domain_widths[r] = hi - lo

        alloc_fn = ALLOC_FN[mech]
        allocation = alloc_fn(
            sensitivities, B, epsilon_min=0.001,
            domain_widths=domain_widths, num_candidates=NUM_CANDIDATES)

        noised_roots = {}
        for root in root_nodes:
            lo, hi = MEDICAL_DOMAINS.get(root, MEDICAL_DOMAINS["_default"])
            noised_roots[root] = noise_fn(float(gt_vals[root]), allocation[root], lo, hi)
        values = postprocess_derived(gt_vals, noised_roots, derived_order)

        alloc_info = {
            "root_nodes": root_nodes,
            "target_node": target_node,
            "sensitivities": {r: round(sensitivities[r], 6) for r in root_nodes},
            "allocation": {r: round(allocation[r], 6) for r in root_nodes},
            "total_budget": round(B, 6),
        }
        return float(values[target_node]), alloc_info


# ── Metrics ──────────────────────────────────────────────────────────────
def evaluate_per_sample(gt_list, san_targets, template_name):
    """Compute per-sample errors and aggregates."""
    target_key = template_target_keys.get(template_name)
    max_classes = {"ANEMIA": 3, "AIP": 3, "FIB4": 3,
                   "CONICITY": 2, "VASCULAR": 2, "TYG": 2, "HOMA": 2, "NLR": 2}
    max_dist = max_classes.get(template_name, 2) - 1

    per_sample = []
    raw_errors = []
    risk_errors = []

    for i in range(len(gt_list)):
        gt_val = float(gt_list[i].get(target_key, 0))
        san_val = float(san_targets[i])

        gt_class = get_risk_class(template_name, gt_val)
        san_class = get_risk_class(template_name, san_val)

        if abs(gt_val) > 1e-10:
            raw_err = abs(san_val - gt_val) / abs(gt_val) * 100.0
        else:
            raw_err = abs(san_val - gt_val) * 100.0

        risk_err = abs(san_class - gt_class) / max_dist * 100.0

        per_sample.append({
            "gt_target": gt_val,
            "sanitized_target": san_val,
            "raw_error_pct": round(raw_err, 4),
            "gt_risk_class": gt_class,
            "sanitized_risk_class": san_class,
            "risk_error_pct": round(risk_err, 4),
        })
        raw_errors.append(raw_err)
        risk_errors.append(risk_err)

    summary = {
        "raw_mae": round(float(np.mean(raw_errors)), 4),
        "raw_std": round(float(np.std(raw_errors)), 4),
        "risk_mae": round(float(np.mean(risk_errors)), 4),
        "risk_std": round(float(np.std(risk_errors)), 4),
        "n_samples": len(raw_errors),
    }
    return per_sample, summary


# ── Main loop ────────────────────────────────────────────────────────────
print("Loaded {} templates, {} samples each".format(len(fields), N), flush=True)
print("Epsilons: {}".format(EPSILONS), flush=True)
print("Budget: B = {}*epsilon".format(T), flush=True)

for eps in EPSILONS:
    print("\n{}\nEpsilon = {}\n{}".format("=" * 60, eps, "=" * 60), flush=True)

    for key in fields:
        set_template_domains(key)

        out_dir = "{}/epsilon_{}/{}".format(RESULTS_BASE, eps, key)
        out_file = "{}/{}_results.json".format(out_dir, METHOD)
        if os.path.exists(out_file):
            print("  {:12s}  already done, skipping".format(key), flush=True)
            continue

        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()

        san_targets = []
        alloc_info = None
        for i in range(N):
            gt = gt_values[key][i]
            target_noised, ai = run_method_single(METHOD, gt, key, eps)
            san_targets.append(target_noised)
            if ai is not None and alloc_info is None:
                alloc_info = ai  # same for all samples (data-independent)

        per_sample, summary = evaluate_per_sample(gt_values[key], san_targets, key)

        elapsed = time.time() - t0
        print("  {:12s}  raw={:8.1f}%  risk={:5.1f}%  ({:.1f}s)".format(
            key, summary['raw_mae'], summary['risk_mae'], elapsed), flush=True)

        result = {
            "method": METHOD,
            "epsilon": eps,
            "template": key,
            "total_budget": round(
                (len(_template_roots(key)) + 1) * eps if BUDGET_MODE == "kplus1"
                else (2 * len(_template_roots(key)) + 1) * eps if BUDGET_MODE == "2kplus1"
                else (3 * len(_template_roots(key)) + 1) * eps if BUDGET_MODE == "3kplus1"
                else T * eps, 6),
            "summary": summary,
            "per_sample_errors": per_sample,
        }
        if alloc_info is not None:
            result["allocation_info"] = alloc_info

        with open(out_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

print("\nDone! Results saved to {}/".format(RESULTS_BASE), flush=True)
