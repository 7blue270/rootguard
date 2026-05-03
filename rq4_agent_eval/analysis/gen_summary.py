"""Emit summary.md for the agent_eval RQ1 sweep."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_AGG = REPO / "aggregates.json"
DEFAULT_OUT = REPO / "SUMMARY.md"

CONFIGS = ("all", "roots", "opt")
CONFIG_LABEL = {"all": "M-All", "roots": "M-Roots", "opt": "M-Opt"}
B_SETTINGS = ("k+1", "2k+1", "3k+1")
TEMPLATES = ("HOMA", "ANEMIA", "FIB4", "AIP", "CONICITY", "VASCULAR", "TYG", "NLR")
EPS_LIST = (0.01, 0.05, 0.1)


def cell(cells, **kw):
    for c in cells:
        if all(c.get(k) == v for k, v in kw.items()):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default=str(DEFAULT_AGG))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    with open(args.agg) as f:
        agg = json.load(f)
    cells = agg["cells"]

    L = []
    L.append("# agent_eval RQ1 sweep — summary")
    L.append("")
    L.append("## Setup")
    L.append("")
    L.append(f"- {agg['n_sessions_loaded']:,} sessions across {agg['n_cells']} cells "
             "= 8 templates × 3 configs × 3 budgets × 3 ε levels.")
    L.append("- Patient counts per template by ε: 200 at ε=0.1; 100 at ε=0.05 and ε=0.01.")
    L.append("- Mechanism: discrete Exponential. Per-call ε for M-All; per-root budgets "
             "from `data/allocations.csv` for M-Roots / M-Opt.")
    L.append("- Adversary protocol: t-1 gather queries (round-robin, one root per turn) "
             "+ 1 final turn requesting all roots; t = (k+1, 2k+1, 3k+1) for the three budgets.")
    L.append("- Bootstrap SEs (1,000 resamples) on every metric.")
    L.append("")
    L.append("## Cross-template aggregate wMAPE (mean across 8 templates)")
    L.append("")
    L.append("| ε | config | B=k+1 | B=2k+1 | B=3k+1 |")
    L.append("|---|---|---|---|---|")
    for eps in EPS_LIST:
        for cfg in CONFIGS:
            row = [f"{eps}", CONFIG_LABEL[cfg]]
            for B in B_SETTINGS:
                vals = [cell(cells, template=t, config=cfg, B_setting=B, eps=eps)
                        for t in TEMPLATES]
                means = [c["wmape_mean"] for c in vals if c]
                if means:
                    row.append(f"{np.mean(means):.2f}%")
                else:
                    row.append("--")
            L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("## Headline numbers (best-case M-Opt per template at ε=0.1, B=3k+1)")
    L.append("")
    L.append("| template | wMAPE | RCE |")
    L.append("|---|---|---|")
    for t in TEMPLATES:
        c = cell(cells, template=t, config="opt", B_setting="3k+1", eps=0.1)
        if c:
            L.append(f"| {t} | {c['wmape_mean']:.2f} ± {c['wmape_se']:.2f}% | "
                     f"{c['rce_mean']:.1f} ± {c['rce_se']:.1f}% |")
    L.append("")
    L.append("## M-Opt vs M-All / M-Roots (wMAPE wins out of 9 (B, ε) cells per template)")
    L.append("")
    L.append("| template | M-Opt < M-All | M-Opt < M-Roots |")
    L.append("|---|---|---|")
    for t in TEMPLATES:
        wins_all, wins_roots, total = 0, 0, 0
        for B in B_SETTINGS:
            for eps in EPS_LIST:
                c_all = cell(cells, template=t, config="all", B_setting=B, eps=eps)
                c_roots = cell(cells, template=t, config="roots", B_setting=B, eps=eps)
                c_opt = cell(cells, template=t, config="opt", B_setting=B, eps=eps)
                if not all((c_all, c_roots, c_opt)):
                    continue
                if c_opt["wmape_mean"] < c_all["wmape_mean"]:
                    wins_all += 1
                if c_opt["wmape_mean"] < c_roots["wmape_mean"]:
                    wins_roots += 1
                total += 1
        L.append(f"| {t} | {wins_all}/{total} | {wins_roots}/{total} |")
    L.append("")
    L.append("## LLM compliance")
    L.append("")
    pf_sess, pf_obs, n_obs, rg_nz, n_nan, n_tx = 0, 0, 0, 0, 0, 0
    for c in cells:
        comp = c["compliance"]
        pf_sess += comp["parse_failure_sessions"]
        pf_obs += comp["parse_failure_obs"]
        n_obs += comp["obs_total"]
        rg_nz += comp["rounding_gap_nonzero_obs"]
        n_nan += comp["nan_target_sessions"]
        n_tx += comp["transcripts_saved"]
    n_sess = agg["n_sessions_loaded"]
    L.append(f"- Sessions with any parse failure: **{pf_sess} / {n_sess}**.")
    L.append(f"- Observation-level parse failures: **{pf_obs} / {n_obs}**.")
    L.append(f"- NaN target sessions: **{n_nan}**.")
    L.append(f"- Rounding gaps (text reply ≠ wire-format sanitize output): "
             f"**{rg_nz} / {n_obs}** observations.")
    L.append(f"- Transcripts saved: **{n_tx} / {n_sess}** "
             f"({100*n_tx/n_sess:.1f}%). The remainder are pre-alias-parser "
             "ε=0.1 sessions that ran before transcript saving was added.")
    L.append("")
    L.append("## Caveats")
    L.append("")
    L.append("1. **M-All is *root-space* sanitization**, not RQ1's *target-space* "
             "single-shot. Numbers under M-All are not directly comparable to "
             "RQ1's `exp_all` cells; the cross-comparison in `compare_to_rq1.py` "
             "is the canonical reference for that question.")
    L.append("2. **Non-uniform N across ε**: ε=0.1 has 200 patients per cell; "
             "ε=0.05 and ε=0.01 have 100. Bootstrap SEs reflect this — "
             "ε=0.05 / 0.01 cells have ~√2 wider intervals.")
    L.append("3. **One mechanism only** (Exponential). Bounded Laplace and "
             "Staircase rows exist in `data/allocations.csv` but were not "
             "swept; they would need a separate run.")
    L.append("4. **Patients drawn from NHANES holdout** (`data/nhanes_benchmark_200.json`); "
             "always the same first-N patients per template (deterministic seeding).")
    L.append("")
    L.append("## Files in this analysis")
    L.append("")
    L.append("- `aggregates.json` — every per-cell stat with bootstrap SE")
    L.append("- `tables/` — LaTeX (`\\pmstd{...}` style) for paper inclusion")
    L.append("- `plots/` — PDF figures (8-panel grids + summary)")
    L.append("- `summary.md` — this file")
    L.append("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
