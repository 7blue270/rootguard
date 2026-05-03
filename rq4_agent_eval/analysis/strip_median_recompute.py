"""Recompute target_estimate / final_class / diagnosis_correct using only the
*final-turn* fetch per root (i.e. observations[r][-1]) instead of the median
of all q+1 observations.

Saves the previous (median-based) values into _median side-keys before
overwriting the canonical fields, so nothing is lost.

Operates in place on the session JSONs in sweep_v2/.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SWEEP = REPO / "results" / "sweep_v2"

# Per-template formula + risk thresholds (kept self-contained so this script
# doesn't depend on importing the template classes).
TEMPLATE_FORMULA = {
    "HOMA":     lambda v: v["Glu"] * v["Ins"] / 405.0,
    "ANEMIA":   lambda v: 100.0 * v["Hb"] / v["Hct"],
    "FIB4":     lambda v: (v["age"] * v["AST"]) / (v["PLT"] * math.sqrt(v["ALT"])),
    "AIP":      lambda v: math.log10(v["tg"] / v["hdl"]),
    "CONICITY": lambda v: v["waist"] / (0.109 * math.sqrt(v["wt"] / v["ht"])),
    "VASCULAR": lambda v: (v["sbp"] - v["dbp"]) / v["sbp"],
    "TYG":      lambda v: math.log((v["tg"] * v["glu"]) / 2.0),
    "NLR":      lambda v: v["neu"] / v["lym"],
}

def _hcl(t):  # HOMA risk_class
    return 0 if t < 1.0 else (1 if t < 2.5 else 2)
def _acl(t):
    return 0 if t < 32.0 else (1 if t <= 36.0 else 2)
def _fcl(t):
    return 0 if t < 1.30 else (1 if t <= 2.67 else 2)
def _aipcl(t):
    return 2 if t > 0.21 else (1 if t > 0.11 else 0)
def _cocl(t):
    return 1 if t > 1.25 else 0
def _vacl(t):
    return 1 if t > 0.60 else 0
def _tycl(t):
    return 1 if t > 8.5 else 0
def _ncl(t):
    return 1 if t >= 3.0 else 0

TEMPLATE_RISK = {
    "HOMA": _hcl, "ANEMIA": _acl, "FIB4": _fcl, "AIP": _aipcl,
    "CONICITY": _cocl, "VASCULAR": _vacl, "TYG": _tycl, "NLR": _ncl,
}


def recompute_one(d: dict) -> dict:
    """Return a new record with target / class fields recomputed from the
    final-turn fetch per root."""
    obs = d.get("observations") or {}
    roots = list(d.get("raw", {}).keys())

    # Capture old (median-based) values once.
    if "target_estimate_median" not in d:
        d["target_estimate_median"] = d.get("target_estimate")
        d["final_class_median"] = d.get("final_class")
        d["diagnosis_correct_median"] = d.get("diagnosis_correct")

    final_values = {}
    for r in roots:
        vs = obs.get(r) or []
        if not vs:
            final_values = None
            break
        final_values[r] = float(vs[-1])

    if final_values is None or d.get("template") not in TEMPLATE_FORMULA:
        d["target_estimate"] = float("nan")
        d["final_class"] = -1
        d["diagnosis_correct"] = False
        return d

    tmpl = d["template"]
    target = float(TEMPLATE_FORMULA[tmpl](final_values))
    cls = int(TEMPLATE_RISK[tmpl](target))
    truth = int(d.get("truth_class", -1))

    d["target_estimate"] = target
    d["final_class"] = cls
    d["diagnosis_correct"] = (cls == truth)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=str(DEFAULT_SWEEP))
    args = ap.parse_args()
    sweep = Path(args.sweep_dir)
    files = sorted(sweep.glob("*.json"))
    print(f"recomputing {len(files)} session JSONs (final-turn fetch per root)...")
    n_changed, n_unchanged = 0, 0
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if "target_estimate_median" in d:
            n_unchanged += 1
            continue  # already converted, idempotent skip
        d = recompute_one(d)
        with open(f, "w") as fh:
            json.dump(d, fh, indent=2, default=str)
        n_changed += 1
    print(f"updated {n_changed}, already-converted {n_unchanged}")


if __name__ == "__main__":
    main()
