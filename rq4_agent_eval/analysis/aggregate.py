"""Aggregate sweep_v2 sessions into per-cell stats with bootstrap SEs.

A "cell" is one (template, config, B_setting, eps) tuple. For each cell we
compute wMAPE, RCE, per-root MAP MAE, per-root naive MAE, and LLM-compliance
counters, all with 1000-sample bootstrap standard errors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SWEEP = REPO / "results" / "sweep_v2"
# The artifact ships a pre-built aggregates.json at the rq4_agent_eval/ root
# so reviewers can run the analysis pipeline without rerunning the LLM sweep.
# Re-running aggregate.py overwrites that file in place.
DEFAULT_OUT = REPO / "aggregates.json"
B_BOOTSTRAP = 1000


def _eps_from_filename(path: str) -> float | None:
    """Extract ε from filename like '..._eps0.1.json'."""
    base = os.path.basename(path)
    m = base.split("_eps")
    if len(m) < 2:
        return None
    return float(m[-1].rsplit(".json", 1)[0])


def bootstrap_se(values: np.ndarray, B: int = B_BOOTSTRAP,
                 statistic=np.mean) -> float:
    """Standard bootstrap SE of `statistic` on `values` (np array)."""
    n = len(values)
    if n == 0:
        return float("nan")
    rng = np.random.default_rng(seed=0)
    samples = rng.choice(values, size=(B, n), replace=True)
    stats = np.array([statistic(s) for s in samples])
    return float(np.std(stats, ddof=1))


def wmape_with_se(targets_est: np.ndarray, targets_truth: np.ndarray,
                  B: int = B_BOOTSTRAP) -> tuple[float, float]:
    """wMAPE = 100 * sum(|est - truth|) / sum(|truth|), with bootstrap SE."""
    abs_err = np.abs(targets_est - targets_truth)
    abs_truth = np.abs(targets_truth)
    if abs_truth.sum() == 0:
        return float("nan"), float("nan")
    point = 100.0 * abs_err.sum() / abs_truth.sum()
    n = len(abs_err)
    rng = np.random.default_rng(seed=0)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        e = abs_err[idx].sum()
        t = abs_truth[idx].sum()
        if t > 0:
            boots.append(100.0 * e / t)
    return float(point), float(np.std(boots, ddof=1))


def rce_with_se(correct: np.ndarray, B: int = B_BOOTSTRAP) -> tuple[float, float]:
    """RCE = 100 * (1 - accuracy), with bootstrap SE."""
    n = len(correct)
    if n == 0:
        return float("nan"), float("nan")
    point = 100.0 * (1.0 - correct.mean())
    rng = np.random.default_rng(seed=0)
    boots = [100.0 * (1.0 - rng.choice(correct, n).mean()) for _ in range(B)]
    return float(point), float(np.std(boots, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=str(DEFAULT_SWEEP))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n-patients-cap", type=int, default=None,
                    help="If set, drop sessions with patient_idx >= this cap. "
                         "Used to make N uniform across ε levels.")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    files = sorted(sweep_dir.glob("*.json"))
    print(f"loading {len(files)} session JSONs from {sweep_dir}")

    # Group by (template, config, B_setting, eps)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    n_dropped = 0
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if args.n_patients_cap is not None and d.get("patient_idx", 0) >= args.n_patients_cap:
            n_dropped += 1
            continue
        eps = float(d.get("eps_in") or _eps_from_filename(str(f)) or float("nan"))
        key = (d["template"], d["config"], d["B_setting"], eps)
        groups[key].append(d)

    if args.n_patients_cap is not None:
        print(f"dropped {n_dropped} sessions with patient_idx >= {args.n_patients_cap}")
    print(f"grouped into {len(groups)} (template, config, B, ε) cells")

    cells = []
    for (tmpl, cfg, B, eps), recs in sorted(groups.items()):
        n = len(recs)
        targets_est = np.array([
            r["target_estimate"] for r in recs
            if r["target_estimate"] == r["target_estimate"]
        ])
        targets_truth = np.array([
            r["target_truth"] for r in recs
            if r["target_estimate"] == r["target_estimate"]
        ])
        n_valid = len(targets_est)
        wmape_mean, wmape_se = wmape_with_se(targets_est, targets_truth)
        correct = np.array([1.0 if r["diagnosis_correct"] else 0.0 for r in recs])
        rce_mean, rce_se = rce_with_se(correct)
        # per-root MAEs
        roots = list(recs[0]["raw"].keys())
        per_root_map = {}
        per_root_naive = {}
        for r in roots:
            map_vals = np.array([
                rec["map_mae"][r] for rec in recs
                if rec["map_mae"][r] == rec["map_mae"][r]
            ])
            naive_vals = np.array([
                rec["naive_mae"][r] for rec in recs
                if rec["naive_mae"][r] == rec["naive_mae"][r]
            ])
            per_root_map[r] = {
                "mean": float(map_vals.mean()) if len(map_vals) else float("nan"),
                "se": bootstrap_se(map_vals) if len(map_vals) else float("nan"),
                "n": int(len(map_vals)),
            }
            per_root_naive[r] = {
                "mean": float(naive_vals.mean()) if len(naive_vals) else float("nan"),
                "se": bootstrap_se(naive_vals) if len(naive_vals) else float("nan"),
                "n": int(len(naive_vals)),
            }
        # compliance
        n_pf_sess = sum(1 for r in recs if sum(r["parse_failures"].values()) > 0)
        n_pf_obs = sum(sum(r["parse_failures"].values()) for r in recs)
        n_obs = sum(sum(len(v) for v in r["wire_observations"].values()) for r in recs)
        n_rg_nz = sum(
            1 for r in recs for vs in r["rounding_gap"].values() for v in vs
            if abs(v) > 1e-9
        )
        n_tx = sum(1 for r in recs if r.get("transcript"))
        n_nan_target = sum(
            1 for r in recs if r["target_estimate"] != r["target_estimate"]
        )

        cells.append({
            "template": tmpl,
            "config": cfg,
            "B_setting": B,
            "eps": eps,
            "n": n,
            "n_valid_for_wmape": n_valid,
            "wmape_mean": wmape_mean,
            "wmape_se": wmape_se,
            "rce_mean": rce_mean,
            "rce_se": rce_se,
            "per_root_map_mae": per_root_map,
            "per_root_naive_mae": per_root_naive,
            "compliance": {
                "parse_failure_sessions": int(n_pf_sess),
                "parse_failure_obs": int(n_pf_obs),
                "obs_total": int(n_obs),
                "rounding_gap_nonzero_obs": int(n_rg_nz),
                "transcripts_saved": int(n_tx),
                "nan_target_sessions": int(n_nan_target),
            },
        })

    out = {
        "sweep_dir": str(sweep_dir),
        "n_sessions_loaded": len(files),
        "n_cells": len(cells),
        "bootstrap_resamples": B_BOOTSTRAP,
        "cells": cells,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
