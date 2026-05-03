# RQ1 — Target utility under adversarial queries

## Question

How much does root-only noising with dependency-aware budget allocation
reduce target-node error compared to independent per-release noising?

## What's compared

Three configurations, evaluated under three DP mechanisms, three adversary
budget settings, and ten ε levels:

| Config | Description |
|---|---|
| **M-All** | Independent per-release noising. Two variants:<br>(a) **target-space** — sanitize the target T = g(X) directly at ε (single shot, total leakage = ε). Implemented in `run_target_space.py`.<br>(b) **root-space** — sanitize each root once at ε; compute T deterministically from noised roots (total leakage = k·ε). Implemented in `run_root_space.py`. |
| **M-Roots** | Root-only noising with uniform per-root allocation εᵢ = B/k where B = (B_setting)·ε. |
| **M-Opt** | Root-only noising with sensitivity-weighted allocation εᵢ ∝ √(\|h_i\|·D_i), summing to B. |

- **Mechanisms:** Exponential (`exp`), Bounded Laplace (`blap`), Staircase (`stair`).
- **Budget settings:** B = (k+1)·ε, (2k+1)·ε, (3k+1)·ε.
- **ε grid:** {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0}.
- **Templates:** 8 NHANES diagnostics — HOMA, FIB4, ANEMIA, AIP, CONICITY, VASCULAR, TYG, NLR.
- **N = 200** patients per template.

Total: 8 templates × 9 methods × 10 ε × 3 budgets ≈ 2,160 cells with 200 samples each.

## Reproduce

From this folder. **Note on directory naming:** the sweep scripts emit raw
per-call MAE into `results_rq1_adversarial_<budget>/` (target-space) and
`results_rq1_adversarial_<budget>_wmape_root_based/` (root-space). The
target-space analysis scripts read from `*_wmape/` directories, so a
`recompute_wmape.py` pass converts the raw target-space output into the
wMAPE-summarised form. The root-space sweep already emits the `_wmape_*`
form directly (no recompute needed there).

```bash
# 1. Run all 27 (mechanism × variant × budget) cells of the target-space
#    M-All variant (paper canonical). ~10 min on 8 cores.
for budget in kplus1 2kplus1 3kplus1; do
  for method in exp_all exp_roots exp_opt blap_all blap_roots blap_opt stair_all stair_roots stair_opt; do
    python run_target_space.py --method $method --budget_mode $budget &
  done
  wait
done

# 2. Convert target-space raw MAE into the wMAPE summaries the analysis
#    scripts read. Writes results_rq1_adversarial_<budget>_wmape/.
for budget in kplus1 2kplus1 3kplus1; do
  python recompute_wmape.py --budget_mode $budget
done

# 3. Same for the root-space M-All variant (alongside, not replacement).
#    run_root_space.py emits the _wmape_root_based/ form directly, no
#    recompute pass needed.
for budget in kplus1 2kplus1 3kplus1; do
  for method in exp_all exp_roots exp_opt blap_all blap_roots blap_opt stair_all stair_roots stair_opt; do
    python run_root_space.py --method $method --budget_mode $budget &
  done
  wait
done

# 4. Build the main-text "double asymmetry" table + appendix tables
#    (target-space, legacy comparison set).
python analysis/gen_double_asymmetry_table.py
python analysis/gen_appendix_summary.py
# (gen_main_tables.py is a legacy parametrized-turns generator; not part
# of the paper-canonical pipeline — invoke with `--turns N` only if you
# specifically want the per-T target-space breakdown.)

# 5. Build the root-space wMAPE + Risk Class Error tables (paper-canonical).
#    Produces tables/root_space/:
#      rq1_double_asymmetry.tex          tab:double_asymmetry      (body §5)
#      rq1_rce_per_template.tex          tab:rce_per_template_rq1  (body §5)
#      rq1_{k,2k,3k}plus1_wmape.tex      tab:rq1_*_wmape           (appendix)
#      rq1_{k,2k,3k}plus1_rce.tex        tab:rq1_*_rce             (appendix)
#      rq1_{k,2k,3k}plus1_per_tmpl_wmape.tex tab:rq1_*_per_tmpl_wmape (appendix)
python analysis/gen_rce_tables.py

# 6. Build figures (target-space).
python analysis/gen_plots.py

# 7. Build figures (root-space variant).
python analysis/gen_plots_root_based.py
```

## Pre-built outputs (verify without rerunning)

```
tables/
  target_space/                  paper-canonical tables
    latex_rq1_double_asymmetry.tex   tab:double_asymmetry (main text)
    latex_rq1_adversarial_summary.tex appendix tables
    latex_rq1_new.tex
  root_space/                    target-from-noised-roots variant (paper-canonical)
    rq1_double_asymmetry.tex     tab:double_asymmetry (main text §5)
    rq1_rce_per_template.tex     tab:rce_per_template_rq1 (main text §5)
    rq1_{k,2k,3k}plus1_wmape.tex tab:rq1_{k,2k,3k}plus1_wmape (appendix)
    rq1_{k,2k,3k}plus1_rce.tex   tab:rq1_{k,2k,3k}plus1_rce (appendix)
    rq1_{k,2k,3k}plus1_per_tmpl_wmape.tex tab:rq1_{k,2k,3k}plus1_per_tmpl_wmape (appendix)
plots/
  target_space/
    rq1_double_asymmetry.pdf     Figure 1: budget scaling
    rq1_privacy_utility.pdf      Figure 2: ε sweep at 2k+1
    rq1_per_template_grid.pdf    Appendix: 8-panel per-template
  root_space/
    same three files, root-based M-All
```

## Headline result

At ε = 0.1, B = (3k+1)·ε (paper Table 1):

| Template | M-All wMAPE | M-Roots wMAPE | M-Opt wMAPE | Opt vs All |
|---|---|---|---|---|
| HOMA | 47.5% | 9.8% | 9.0% | 5.3× |
| FIB4 | 52.1% | 8.3% | 5.9% | 8.8× |
| AIP | (compressing) | … | … | 5.1× |

(Full table in `tables/target_space/latex_rq1_double_asymmetry.tex`.)
