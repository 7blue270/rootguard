# RQ3 v3 — Reconstruction Under Adversarial Queries (root-only)

This folder contains the deliverables for the RQ3 **v3** experiment (sweep launched and completed 2026-04-28). v3 differs from v2 in one critical way: **the legitimate utility release ("target release") is no longer fed into the MAP likelihood**. The adversary's last turn directly requests root values, contributing one extra root release on the affected root(s). All MAP estimation uses **root observations only**.

## Files

| File | What it is |
|---|---|
| `fig_mae_vs_q_eps0.1_{uniform,informed}.pdf/.png` | Main figure (both priors): wMAPE vs q for Exponential at ε_r = 0.1. Two panels (Strategy A, Strategy B). Three lines (M-All, M-Roots, M-Opt). Mean across 8 templates. |
| `main_table_{uniform,informed}.tex` | Main-text tables: ε_r = 0.1, Strategy B, Exponential. Rows = 8 templates, columns = q ∈ {1, 4, 8, 16} × {M-All, M-Roots, M-Opt}. |
| `appendix.tex` | Full appendix: aggregate tables for every (mechanism × strategy × prior) combination across all ε_r ∈ {0.01, 0.05, 0.1, 0.5, 1.0} and q ∈ {1, 4, 8, 16}, per-template breakdowns for the informed prior under Strategy A and Strategy B, and a Strategy-A metadata table showing r* and ε*^Opt per template. |
| `per_template/` | 80 files: 16 per-template Exponential plots + 12 grid overviews + 12 all-eps log-scale grids (each ×2 priors). |
| `per_epsilon/` | 44 files: per-epsilon plots, per-mechanism all-ε grids, wMAPE-vs-ε log-log plots (×2 priors). |

Raw results: `../results_rq3_adversarial_v3/epsilon_{ε}/{template}/{method}_{prior}_{strategy}_q{q}_results.json` — 5360 JSON files. Allocations reused unchanged from `../allocations_v2/`.

## What changed from v2

### Threat model

- **v2**: the adversary observed q noisy root queries plus a **target release** that included the legitimate utility output and noisy values for *every* root and *every* derived node (computed independently at ε_r each). The MAP combined all of these via a joint optimizer.
- **v3**: the adversary observes q noisy root queries plus **one extra root release** on the affected roots only. No target on derived nodes. MAP is per-root 1-D argmax. The adversary cannot exploit the legitimate utility release to gain "free" observations on roots through derived-node coupling.

### Observation count per affected root

- Strategy A: r* gets q + 1 = q+1 observations.
- Strategy B: every root gets q + 1 observations.
- Other roots in Strategy A: zero observations (MAP returns prior mean).

### MAP simplification

v3 MAP collapses to **per-root 1-D argmax** in every cell (no joint optimizer, no derived-node coupling). For cached variants (M-Roots, M-Opt), the q+1 obs are q+1 copies of the same cached value — argmax invariant in q.

### RNG seeding

Method-agnostic seeding `hash((template, root, sample_idx, draw_idx))` is preserved. M-All draw 0 ≡ M-Roots cached draw at the same (template, root, sample, ε_r) within a single interpreter session.

## Implementation

`run_rq3_adversarial_v3.py` was a copy of v2 with the following changes:
1. `generate_target_release` is no longer called from the main loop; target-release plumbing dropped from worker, JSON output, and naive baseline.
2. `generate_adversarial_obs` now appends a single extra root release (draw_idx = q for M-All; cached value replayed for M-Roots/M-Opt).
3. `map_estimate_multi_obs` rewritten as **per-root 1-D argmax** with optional Gaussian prior. The whole joint-MAP optimizer block deleted (~150 lines).
4. **Vectorized MAP**: a precomputed `(log_terms[1000,1000], log_Z[1000])` matrix is built once per `(mechanism, ε_r)` per config. Per-sample MAP is `log_terms[:, obs_indices].sum(axis=1) - K * log_Z` + argmax — pure numpy, microseconds per sample. The ProcessPoolExecutor was removed because vectorized single-threaded MAP is faster than the pool overhead.

End-to-end runtime: **~10 minutes** for the full 5360-config sweep on a single core (vs ~11 h for v2 with 16 workers).

## Verification

`run_rq3_adversarial_v3.py --verify` passes 6 sanity checks:

1. PMF normalization sums to 1 for all three mechanisms.
2. M-All draw 0 ≡ M-Roots cached draw (method-agnostic seed).
3. `obs[r]` has length q+1 on affected roots; cached methods have q+1 identical copies.
4. M-Roots / M-Opt MAP invariant across q ∈ {1, 4, 8, 16}.
5. Uniform + cached + Strategy B: per-root MAP equals the cached value.
6. M-All q-scaling: MAE generally decreases with q (soft check).

## Sweep parameters (unchanged from v2)

- ε_r ∈ {0.01, 0.05, 0.1, 0.5, 1.0}
- q ∈ {1, 4, 8, 16}
- Strategies A and B
- Priors uniform and informed
- Mechanisms: Exponential, Bounded Laplace, Staircase
- Methods per mechanism: M-All, M-Roots, M-Opt
- Templates (8): ANEMIA, FIB4, AIP, CONICITY, VASCULAR, TYG, HOMA, NLR
- 200 samples per template

Total valid configs: **5360** (5760 planned − 400 staircase skips at ε_r > 0.962). The 400 skips are: 3 staircase methods × 8 templates × 2 strategies × 2 priors × 4 q values = 384 at ε=1.0, plus 16 staircase_roots_opt configs at ε=0.5 / AIP where the max per-root allocation exceeds 0.962.

## v2 vs v3 — high-level comparison

Spot-check: at ε_r = 0.1, Exponential, informed prior:

| Cell | v2 wMAPE | v3 wMAPE | Note |
|---|---|---|---|
| ANEMIA M-Roots q=1 | 1.78% | 1.82% | within 1% (cached methods unchanged) |
| ANEMIA M-Opt q=1 | 1.18% | 1.21% | within 1% |
| FIB4 M-Roots q=1 | 21.91% | 18.89% | v3 better by 3% |
| AIP M-All q=1 | 11.29% | 25.74% | v2 better by 14% — see below |
| AIP M-Roots q=1 | 39.08% | 35.40% | v3 better by 3.7% |

Aggregate over all 5360 configs (per-config wMAPE, threshold 0.5%):
- v3 better: 833 cells (15.5%)
- v2 better: 1529 cells (28.5%)
- Tied:      2998 cells (55.9%)

### Why v3 looks worse than v2 in M-All informed cells (and better in M-All uniform A)

- **M-All uniform Strategy A**: v3 *better* (e.g. mean wMAPE Δ = -6.6% for vanilla). v2 in this cell did 1-D MAP on r* with **only the q adv obs, no target term**. v3 adds the +1 extra release, giving the adversary q+1 obs on r*.
- **M-All informed (any strategy)**: v3 *worse* in some templates. v2 ran a joint MAP over all k roots that included a target release on every root + every derived node, all at ε_r each. The adversary effectively got multiple noisy observations of every root via derived-node formulas. v3 correctly excludes this — the adversary now only has q+1 direct observations on each affected root.
- **M-All Strategy B (any prior)**: v3 slightly worse for the same reason — v2's joint MAP exploited the derived-node target release; v3 does not.

The takeaway: **v2's higher reconstruction success against M-All under informed/B was partly an artifact of the legitimate utility release leaking root information through derived-node coupling.** v3 measures the pure root-observation attack — a more honest model of the adversary's threat surface. The reported wMAPEs are more conservative for that reason; cached methods (M-Roots, M-Opt) are essentially unchanged because they never leaned on the target release in the first place (cached values dominate the likelihood).

## Reproducing

```bash
# 1. Allocations (same as v2)
python precompute_mopt_allocations_v2.py

# 2. Sanity checks
python run_rq3_adversarial_v3.py --verify

# 3. Full sweep (~10 min single-threaded)
python run_rq3_adversarial_v3.py 2>&1 | tee logs/rq3_v3_full.log

# 4. Generate deliverables
python gen_rq3_v3_main_figure.py
python gen_rq3_v3_main_table.py
python gen_rq3_v3_appendix.py
python gen_rq3_v3_per_template_plots.py
python gen_rq3_v3_per_epsilon_plots.py
```

## Compared with v2

v2 results remain in `../rq3_v2_deliverables/` and `../results_rq3_adversarial_v2/` for reference. The script `compare_v2_v3.py` reports per-cell deltas.
