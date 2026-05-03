"""
Allocation loader: reads precomputed per-root privacy budgets.

Three configurations, all evaluated in the experiment:

  - "all"   M-All: independent noising per release. Each tool-call response
            is sanitized at the SAME eps. There is no per-root allocation
            — the agent receives an `eps_per_call` scalar.
  - "roots" M-Roots: root-only noising with uniform allocation eps_i = B/k.
            Returns a dict {root_name: B/k}.
  - "opt"   M-Opt: root-only noising with sensitivity-weighted allocation
            eps_i ∝ sqrt(|h_i| * delta_i), with eps_min clamping and
            redistribution. Returns a dict {root_name: precomputed_eps}.

The format of the precomputed file is left to whichever convention the
existing Sec 5 codebase uses; the loader below is a minimal CSV reader. If
your precomputed data is JSON, pickle, or something else, replace the body
of `load_allocation` and keep the signature.

Expected CSV schema (one row per (template, mechanism, B_setting, config, root)):

    template,mechanism,B_setting,config,root,eps
    FIB4,exp,2k+1,opt,age,0.0234
    FIB4,exp,2k+1,opt,AST,0.0512
    ...
    FIB4,exp,2k+1,roots,age,0.05625
    FIB4,exp,2k+1,roots,AST,0.05625
    ...

For the "all" configuration, a single row per (template, mechanism, B_setting)
with root="*" gives the per-call epsilon:

    FIB4,exp,2k+1,all,*,0.05
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional


# Default location for the precomputed allocations file. Override with
# the `csv_path` argument if your data lives elsewhere.
DEFAULT_ALLOCATIONS_PATH = Path("data/allocations.csv")


def load_allocation(
    template_name: str,
    mechanism: str,
    B_setting: str,
    config: str,
    csv_path: Optional[Path] = None,
    eps_in: Optional[float] = None,
) -> Dict[str, float]:
    """
    Load precomputed per-root budgets for a given (template, mechanism,
    B_setting, config) cell.

    `eps_in` is the input epsilon level the allocation was computed at
    (the per-cell privacy parameter the paper writes as ε). If the CSV
    contains an `eps_in` column, this argument is required to disambiguate
    between rows for the same cell at different epsilons. If the column is
    absent (legacy schema), `eps_in` is ignored.

    Returns
    -------
    Dict[str, float]
        For config in {"roots", "opt"}: a mapping {root_name: eps_i}.
        For config == "all":             a singleton {"*": eps_per_call}.
    """
    if csv_path is None:
        csv_path = DEFAULT_ALLOCATIONS_PATH
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Allocations file not found: {csv_path}. "
            "Drop your precomputed budgets in data/allocations.csv "
            "(see allocations.py docstring for schema)."
        )
    out: Dict[str, float] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        required_fields = {"template", "mechanism", "B_setting", "config", "root", "eps"}
        if not required_fields.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Allocations CSV missing required columns. "
                f"Got {reader.fieldnames}, need {sorted(required_fields)}"
            )
        has_eps_in = "eps_in" in (reader.fieldnames or [])
        if has_eps_in and eps_in is None:
            raise ValueError(
                "CSV has eps_in column but caller did not pass eps_in; "
                "specify which input eps level to load."
            )
        for row in reader:
            if (
                row["template"] == template_name
                and row["mechanism"] == mechanism
                and row["B_setting"] == B_setting
                and row["config"] == config
                and (not has_eps_in
                     or float(row["eps_in"]) == float(eps_in))
            ):
                out[row["root"]] = float(row["eps"])
    if not out:
        raise KeyError(
            f"No rows matched (template={template_name}, mechanism={mechanism}, "
            f"B_setting={B_setting}, config={config}, eps_in={eps_in}) in {csv_path}"
        )
    return out


def load_per_call_eps(
    template_name: str,
    mechanism: str,
    B_setting: str,
    csv_path: Optional[Path] = None,
    eps_in: Optional[float] = None,
) -> float:
    """Convenience for the M-All configuration: returns the scalar eps used per release."""
    alloc = load_allocation(template_name, mechanism, B_setting, "all",
                            csv_path, eps_in=eps_in)
    if list(alloc.keys()) != ["*"]:
        raise ValueError(
            f"Expected M-All allocation to have a single '*' row; got {list(alloc.keys())}"
        )
    return alloc["*"]
