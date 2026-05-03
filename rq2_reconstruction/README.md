# RQ2 — Reconstruction attacks

## Question

Under a MAP adversary with repeated queries at matched per-root noise, how
much does M-All's privacy degrade compared to RootGuard (M-Roots / M-Opt)?

## What's compared

- **M-All**: per-release sanitization; q noisy releases per root, MAP averages them.
- **M-Roots**: root-only with uniform allocation; cached value, q-invariant.
- **M-Opt**: root-only with sensitivity-weighted allocation; cached, q-invariant.

Reported metric is **reconstruction wMAPE (%)** as a function of adversary
query count q ∈ {1, 4, 8, 16}, at fixed per-root noise.

## Reproduce

```bash
# 1. Precompute M-Opt allocations (this caches them under allocations_v2/).
python precompute_allocations.py

# 2. Run the reconstruction adversary across the full grid.
#    ε ∈ {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0},
#    q ∈ {1, 4, 8, 16}, 8 templates, 3 mechanisms, 3 strategies × 2 priors.
python run.py    # ~30 min on 8 cores

# 3. Build paper figures and tables. gen_paper_tables.py emits every
#    label referenced by the paper text: tab:recon_main (main text §5)
#    and tab:rq2_per_template, tab:rq2_exp_{A,B}_{uniform,informed},
#    tab:rq2_blap_agg, tab:rq2_stair_agg (appendix).
python analysis/gen_paper_tables.py
python analysis/gen_main_figure.py     # writes fig_mae_vs_q_eps0.1_*.pdf
python analysis/gen_per_epsilon_plots.py
python analysis/gen_per_template_plots.py
```

## Pre-built outputs

```
tables/
  recon_main.tex             tab:recon_main             (main text §5)
  rq2_per_template.tex       tab:rq2_per_template       (appendix)
  rq2_exp_A_uniform.tex      tab:rq2_exp_A_uniform      (appendix)
  rq2_exp_A_informed.tex     tab:rq2_exp_A_informed     (appendix)
  rq2_exp_B_uniform.tex      tab:rq2_exp_B_uniform      (appendix)
  rq2_exp_B_informed.tex     tab:rq2_exp_B_informed     (appendix)
  rq2_blap_agg.tex           tab:rq2_blap_agg           (appendix)
  rq2_stair_agg.tex          tab:rq2_stair_agg          (appendix)
plots/
  main/
    fig_mae_vs_q_eps0.1_uniform.pdf       Figure: paper main-text recon plot
    fig_mae_vs_q_eps0.1_informed.pdf      Figure: same with informed prior
  per_epsilon/                            grid of per-ε plots for appendix
  per_template/                           per-template breakdowns
SUMMARY.md                                quick text summary of results
```

## Headline result

At ε = 0.1 (paper Figure 2): M-All's reconstruction wMAPE drops from
~30% at q=1 to ~3% at q=16 (degrades with q). M-Roots and M-Opt are
**flat** with q (q-invariance: cached values give the adversary no
additional information from repeated queries).

See `SUMMARY.md` for full numbers.
