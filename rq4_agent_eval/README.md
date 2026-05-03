# RQ4 — End-to-end LLM agent evaluation

## Question

Do the mechanism-level findings (RQ1–3) transfer to a real LLM-driven
tool-call deployment? Specifically: when patient lab values live in the
LLM user-agent's system prompt and an adversary issues natural-language
queries with the LLM in the loop, does M-Opt still beat M-All by the
~3× factor predicted at ε=0.1?

## Pipeline

A `gpt-5.4-nano` user-agent serves lab values to an adversary program over
a t-turn session. The adversary issues t-1 single-root queries (round-robin)
plus one final-turn multi-root request, then computes the diagnostic from
the **final-turn fetch per root** (one ε-draw per root, matching RQ1's
root-based M-All exactly).

- **Configs:** M-All (per-fetch sanitize), M-Roots (uniform B/k cached), M-Opt (sensitivity-weighted cached).
- **Budget settings:** B = (k+1)·ε, (2k+1)·ε, (3k+1)·ε.
- **ε levels:** 0.01, 0.05, 0.1.
- **Templates:** all 8 NHANES templates (HOMA, FIB4, ANEMIA, AIP, CONICITY, VASCULAR, TYG, NLR).
- **N:** 100 patients per template.
- Total session count: 8 × 100 × 3 × 3 × 3 = **21,600 LLM-mediated sessions**.

## Reproduce

Requires `OPENAI_API_KEY`. Set it via the project-root `.env` file
(`cp ../.env.example ../.env` then edit).

```bash
# Sanity tests (no API).
python tests/test_smoke.py
python tests/test_proxy.py
python tests/test_attack.py

# 1. Precompute per-cell M-Opt allocations (writes data/allocations.csv).
python precompute_allocations.py

# 2. Full sweep (~3 hours wall-clock at 20-way concurrency, ~$5 in API spend).
set -a; source ../.env; set +a
python experiments/run_sweep.py --n-patients 100 --workers 20 --eps-list 0.01,0.05,0.1

# 3. Aggregate (with bootstrap SEs).
python analysis/aggregate.py --n-patients-cap 100

# 4. LaTeX tables and PDF plots. gen_tables.py emits ten files into tables/,
#    including the four Risk Class Error tables cited in §5 of the paper:
#      rce_eps0.1_body.tex      tab:rce_per_template (body)
#      rce_eps0.1_appendix.tex  tab:agent_rce_eps0.1_app (appendix)
#      aggregate_rce.tex        tab:agent_aggregate_rce (aggregate across templates)
#      rce_eps0.1.tex           tab:agent_rce_eps0.1 (legacy 3 cfg × 3 B layout)
python analysis/gen_tables.py
python analysis/gen_plots.py
python analysis/gen_summary.py

# 5. (optional) bundle everything for sharing.
python analysis/package_zip.py    # writes results/agent_eval_rq1_results.zip
```

Each session uses ~6.4 worker-seconds at 20-way concurrency. Expect minor
LLM compliance failures (< 0.5% historical rate) — they're recorded in
`aggregates.json::compliance.parse_failure_sessions` and don't affect the
aggregate numbers.

## Pre-built outputs

```
aggregates.json                         per-cell wMAPE, RCE, per-root MAP MAE (with bootstrap SE)
SUMMARY.md                              text overview, headline numbers, caveats
tables/                                 LaTeX tables ready for paper inclusion
  double_asymmetry_eps{0.01,0.05,0.1}.tex
  per_template_wmape_eps0.1.tex
  aggregate_wmape.tex
  rce_eps0.1_body.tex                   tab:rce_per_template (main text §5)
  rce_eps0.1_appendix.tex               tab:agent_rce_eps0.1_app (appendix)
  aggregate_rce.tex                     tab:agent_aggregate_rce (cross-template)
  rce_eps0.1.tex                        tab:agent_rce_eps0.1 (legacy)
  per_root_mae.tex
  win_rate.tex
plots/                                  PDF figures, 8-panel grids
  double_asymmetry_eps{0.01,0.05,0.1}.pdf
  privacy_utility.pdf                   log-log over ε
  rce_grid.pdf
  config_comparison.pdf
data/allocations.csv                    precomputed budgets used by the sweep
```

## Headline result

At ε = 0.1, B = (3k+1)·ε, averaged across 8 templates:

| Config | wMAPE | RCE |
|---|---|---|
| M-All | ~25–28% | ~22% |
| M-Roots | ~8% | ~6% |
| M-Opt | **~6%** | **~4%** |

M-Opt's ~3× advantage from RQ1 transfers to the deployment with zero
LLM-protocol violations across all 21,600 sessions.

See `SUMMARY.md` for full numbers and caveats.

## Architecture overview

```
Adversary (Python program)
    │  natural-language queries
    ▼
ProxySession (gpt-5.4-nano LLM)        ← system prompt embeds raw or pre-noised lab values
    │  tool calls
    ▼
ToolDispatcher
    │
    ├── M-All:  sanitize(value, lo, hi, eps)  ← per-fetch fresh noise
    └── M-Roots/M-Opt: lookup from RootGuard cache (one shot at session init)
```

- `agent/proxy.py` — long-lived chat session, two factory methods (M-All / RootGuard mode).
- `agent/tools.py` — OpenAI tool schema, `ToolDispatcher`.
- `attack/adversary.py` — t-turn protocol, regex parser with alias map, MAP attack post-session.
- `attack/map_recon.py` — MAP estimator over discrete index space.
- `experiments/run_sweep.py` — driver, resumable, transcript-saving, configurable concurrency.
- `analysis/{aggregate,gen_tables,gen_plots,gen_summary,strip_median_recompute,package_zip}.py` — derived artifacts.
