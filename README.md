# RootGuard

Dependency-aware sanitization for multi-turn agentic LLM interactions on
private numeric data. RootGuard noises the *roots* of a computation graph
once and then derives every downstream value deterministically from the
noised roots, so privacy depends only on the initial sanitization
regardless of adversary function choices, query count, or turn order.

This repository contains the implementation, four self-contained
research-question (RQ) folders, and pre-built tables + plots for each.

## Layout

```
rootguard/
├── README.md                     this file
├── INSTALL.md                    setup instructions
├── .env.example                  template for environment variables (RQ4 only)
├── pyproject.toml                pinned Python dependencies
├── data/                         NHANES benchmark + population stats (input data)
│   ├── nhanes_benchmark_200.json
│   └── holdout_population_means.json
├── preempt/                      core sanitizer module (NER, FF3 cipher, sanitize tool)
├── utils/                        shared utilities (sensitivity, allocation, topo order)
├── baselines/                    Bounded-Laplace and Staircase mechanism implementations
├── rq1_target_utility/           RQ1 — target utility under adversarial queries
├── rq2_reconstruction/           RQ2 — MAP reconstruction attacks
├── rq3_structural_analysis/      RQ3 — structural per-template analysis
└── rq4_agent_eval/               RQ4 — end-to-end LLM agent evaluation
```

## Quick start

```bash
python3.12 -m venv .venv          # NB: not python3.13 (see INSTALL.md)
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Each `rq*/` folder is **self-contained**: it has a `README.md`, a runner
(`run.py` or `run_root_space.py`), an `analysis/` directory of table/plot
generators, and pre-built `tables/` + `plots/` so you can inspect outputs
without rerunning any experiment.

| RQ | Pipeline (run from the `rq*/` folder) | Wall-clock | Notes |
|---|---|---|---|
| RQ1 — Target utility | `python run_target_space.py ...` → `python run_root_space.py ...` (all 27 cells) → `python analysis/gen_rce_tables.py` → `python analysis/gen_double_asymmetry_table.py` → `python analysis/gen_appendix_summary.py` → `python analysis/gen_plots.py` | ~10 min | Pure CPU |
| RQ2 — Reconstruction attacks | `python precompute_allocations.py` → `python run.py` → `python analysis/gen_paper_tables.py` → `python analysis/gen_main_figure.py` | ~30 min | Pure CPU |
| RQ3 — Structural analysis | `python analysis/gen_allocation_plot.py` → `python analysis/gen_allocation_plot_2kplus1.py` → `python analysis/gen_per_template_summary.py` → `python analysis/gen_template_metadata_table.py` | <5 min | Reads RQ1's allocations + root-space results |
| RQ4 — End-to-end LLM agent eval | `python precompute_allocations.py` → `python experiments/run_sweep.py --n-patients 100 --workers 20 --eps-list 0.01,0.05,0.1` → `python analysis/aggregate.py` → `python analysis/gen_tables.py` → `python analysis/gen_plots.py` | ~3 hours | Requires `OPENAI_API_KEY` (see `.env.example`) |

See each RQ's `README.md` for full options and expected outputs.

## Inspect pre-built outputs

Pre-generated tables and figures live under each `rq*/tables/` and
`rq*/plots/` so you can browse them without rerunning anything:

```bash
ls rq1_target_utility/tables/ rq1_target_utility/plots/
ls rq2_reconstruction/tables/ rq2_reconstruction/plots/main/
ls rq3_structural_analysis/plots/
ls rq4_agent_eval/tables/ rq4_agent_eval/plots/
cat rq2_reconstruction/SUMMARY.md
cat rq4_agent_eval/SUMMARY.md
```

## System requirements

- Python 3.10, 3.11, or 3.12 (not 3.13 — see [`INSTALL.md`](INSTALL.md) for the resolver-level reason)
- ~2 GB free disk for results (RQ1–3) plus ~200 MB for RQ4 raw JSONs
- 8+ CPU cores recommended for the parallel cells
- For RQ4 only: a working OpenAI API key (any tier) and ~$2–5 of API budget

See [`INSTALL.md`](INSTALL.md) for full install instructions and
troubleshooting.
