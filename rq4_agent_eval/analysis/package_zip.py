"""Package the agent_eval RQ1 deliverable as a zip."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "results" / "agent_eval_rq1_results.zip"

README_TEMPLATE = """# agent_eval RQ1 results bundle

Produced from the `agent_eval/agent_eval/` pipeline.

## Layout

- `analysis/` — the four scripts that produced everything below:
    - `aggregate.py`   compute per-cell stats from raw sessions
    - `gen_tables.py`  emit LaTeX tables
    - `gen_plots.py`   emit PDF plots
    - `gen_summary.py` emit summary.md
    - `package_zip.py` (this packager)
- `data/allocations.csv` — per-(template, mechanism, B_setting, ε, config, root) budgets used by the sweep
- `results/sweep_v2/` — 28,800 raw session JSONs (8 templates × {200, 100, 100} patients × 3 configs × 3 budgets × 3 ε)
- `results/sweep_v2_analysis/`
    - `aggregates.json` — every per-cell stat with bootstrap SE
    - `tables/*.tex`    — LaTeX tables ready for paper inclusion
    - `plots/*.pdf`     — figures
    - `summary.md`      — overview report

## How to reproduce / extend

1. The sweep itself was run with:
    ```
    python experiments/run_sweep.py --n-patients 200 --workers 5 --eps-list 0.1
    python experiments/run_sweep.py --n-patients 100 --workers 20 --eps-list 0.01,0.05
    ```
2. Re-run the analysis (deterministic given the JSONs):
    ```
    python analysis/aggregate.py
    python analysis/gen_tables.py
    python analysis/gen_plots.py
    python analysis/gen_summary.py
    ```

## Caveats

See `results/sweep_v2_analysis/summary.md` for: M-All mechanism difference vs RQ1, non-uniform N across ε, one-mechanism-only sweep, and LLM-compliance stats.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--staging", default=None,
                    help="Staging dir; defaults to a fresh tempfile.mkdtemp().")
    args = ap.parse_args()

    if args.staging is None:
        import tempfile
        staging = Path(tempfile.mkdtemp(prefix='rootguard_pkg_'))
    else:
        staging = Path(args.staging)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

    # Copy what we want into the staging dir.
    print(f"staging at {staging}")

    # 1. analysis scripts
    (staging / "analysis").mkdir()
    for f in (REPO / "analysis").glob("*.py"):
        shutil.copy2(f, staging / "analysis" / f.name)

    # 2. allocations CSV
    (staging / "data").mkdir()
    shutil.copy2(REPO / "data" / "allocations.csv", staging / "data" / "allocations.csv")

    # 3. raw sessions + analysis outputs
    (staging / "results").mkdir()
    print("copying 28,800 session JSONs (this takes a moment)...")
    shutil.copytree(REPO / "results" / "sweep_v2", staging / "results" / "sweep_v2")
    shutil.copytree(REPO / "results" / "sweep_v2_analysis",
                    staging / "results" / "sweep_v2_analysis")

    # 4. README
    (staging / "README.md").write_text(README_TEMPLATE)

    # zip it
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    print(f"zipping -> {out}")
    # use shutil.make_archive (uses zipfile under the hood, deflate)
    base = str(out.with_suffix(""))
    shutil.make_archive(base, "zip", root_dir=str(staging))
    sz = out.stat().st_size / (1024 * 1024)
    print(f"wrote {out} ({sz:.1f} MB)")
    # cleanup staging
    shutil.rmtree(staging)


if __name__ == "__main__":
    main()
