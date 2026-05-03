# agent_eval RQ1 sweep — summary

## Setup

- 28,800 sessions across 216 cells = 8 templates × 3 configs × 3 budgets × 3 ε levels.
- Patient counts per template by ε: 200 at ε=0.1; 100 at ε=0.05 and ε=0.01.
- Mechanism: discrete Exponential. Per-call ε for M-All; per-root budgets from `data/allocations.csv` for M-Roots / M-Opt.
- Adversary protocol: t-1 gather queries (round-robin, one root per turn) + 1 final turn requesting all roots; t = (k+1, 2k+1, 3k+1) for the three budgets.
- Bootstrap SEs (1,000 resamples) on every metric.

## Cross-template aggregate wMAPE (mean across 8 templates)

| ε | config | B=k+1 | B=2k+1 | B=3k+1 |
|---|---|---|---|---|
| 0.01 | M-All | 98.79% | 108.27% | 115.20% |
| 0.01 | M-Roots | 82.63% | 55.50% | 43.17% |
| 0.01 | M-Opt | 71.60% | 47.17% | 35.47% |
| 0.05 | M-All | 32.80% | 34.39% | 34.63% |
| 0.05 | M-Roots | 25.96% | 17.84% | 13.86% |
| 0.05 | M-Opt | 20.95% | 14.01% | 10.51% |
| 0.1 | M-All | 21.58% | 22.21% | 21.72% |
| 0.1 | M-Roots | 15.94% | 10.59% | 7.86% |
| 0.1 | M-Opt | 12.28% | 7.64% | 5.58% |

## Headline numbers (best-case M-Opt per template at ε=0.1, B=3k+1)

| template | wMAPE | RCE |
|---|---|---|
| HOMA | 7.37 ± 0.96% | 12.0 ± 3.2% |
| ANEMIA | 0.50 ± 0.08% | 1.0 ± 1.0% |
| FIB4 | 6.18 ± 0.67% | 5.0 ± 2.2% |
| AIP | 8.76 ± 1.32% | 6.0 ± 2.4% |
| CONICITY | 0.67 ± 0.06% | 2.0 ± 1.4% |
| VASCULAR | 1.91 ± 0.19% | 0.0 ± 0.0% |
| TYG | 1.31 ± 0.15% | 6.0 ± 2.4% |
| NLR | 17.93 ± 2.60% | 9.0 ± 2.9% |

## M-Opt vs M-All / M-Roots (wMAPE wins out of 9 (B, ε) cells per template)

| template | M-Opt < M-All | M-Opt < M-Roots |
|---|---|---|
| HOMA | 9/9 | 9/9 |
| ANEMIA | 9/9 | 9/9 |
| FIB4 | 9/9 | 9/9 |
| AIP | 9/9 | 9/9 |
| CONICITY | 9/9 | 9/9 |
| VASCULAR | 9/9 | 9/9 |
| TYG | 9/9 | 8/9 |
| NLR | 7/9 | 6/9 |

## LLM compliance

- Sessions with any parse failure: **0 / 28800**.
- Observation-level parse failures: **0 / 162000**.
- NaN target sessions: **0**.
- Rounding gaps (text reply ≠ wire-format sanitize output): **0 / 162000** observations.
- Transcripts saved: **14500 / 28800** (50.3%). The remainder are pre-alias-parser ε=0.1 sessions that ran before transcript saving was added.

> **Note on the obs-total figure.** Earlier runs of `analysis/aggregate.py`
> reported wire-observation totals of 161,784. This was an analysis-pipeline
> artifact, not a real LLM compliance failure: `_record_obs` in
> `attack/adversary.py` reverse-mapped each `sanitize(value=…)` call back
> to a root by closest raw-value, which collides whenever two of a patient's
> roots happen to have identical raw values (e.g. FIB-4 patients where
> AST = ALT, NLR patients where neutrophils = lymphocytes). The collision
> attributed both tied roots' sanitize calls to the alphabetically-first
> root and dropped the second's wire entries. 216 observations were lost
> this way (FIB4: 162, NLR: 54). The LLM emitted the values correctly,
> the mechanism sanitized them correctly, the text-side parser captured
> them correctly (`parse_failures = 0`); only the wire-side accounting
> miscounted. **Nothing reported in the paper is affected** — wMAPE, RCE,
> per-root MAP MAE all consume the text-side `observations` field, not
> `wire_observations`. The reverse-mapper is now tie-safe (prefers roots
> still pending in this turn's `names_requested` order) and the nominal
> total is 162,000.

## Caveats

1. **M-All is *root-space* sanitization**, not RQ1's *target-space* single-shot. Numbers under M-All are not directly comparable to RQ1's `exp_all` cells; the cross-comparison in `compare_to_rq1.py` is the canonical reference for that question.
2. **Non-uniform N across ε**: ε=0.1 has 200 patients per cell; ε=0.05 and ε=0.01 have 100. Bootstrap SEs reflect this — ε=0.05 / 0.01 cells have ~√2 wider intervals.
3. **One mechanism only** (Exponential). Bounded Laplace and Staircase rows exist in `data/allocations.csv` but were not swept; they would need a separate run.
4. **Patients drawn from NHANES holdout** (`data/nhanes_benchmark_200.json`); always the same first-N patients per template (deterministic seeding).

## Files in this analysis

- `aggregates.json` — every per-cell stat with bootstrap SE
- `tables/` — LaTeX (`\pmstd{...}` style) for paper inclusion
- `plots/` — PDF figures (8-panel grids + summary)
- `summary.md` — this file
