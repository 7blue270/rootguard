"""WP5: 1,350-session sweep at ε=0.1 across all templates / configs / B_settings.

Layout:  3 templates × 50 patients × 3 configs × 3 B_settings = 1350 sessions.
Concurrency: ThreadPoolExecutor (OpenAI calls are I/O-bound).
Output: one JSON per session under results/sweep/, plus a final summary.
Resumable: a session whose result file already exists is skipped.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import json
import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from openai import OpenAI

from sanitizers.allocations import load_allocation, load_per_call_eps
from templates.aip import AIP
from templates.anemia import ANEMIA
from templates.conicity import CONICITY
from templates.fib4 import FIB4
from templates.homa import HOMA
from templates.nlr import NLR
from templates.tyg import TYG
from templates.vascular import VASCULAR

from agent.proxy import ProxySession
from agent.runner import (
    CONFIG_ALL, CONFIG_OPT, CONFIG_ROOTS, build_session_config,
)
from agent.tools import MODE_M_ALL, MODE_ROOTGUARD, ToolDispatcher

from attack.adversary import Adversary


REPO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO_BASE, "data", "allocations.csv")
BENCHMARK = "../data/nhanes_benchmark_200.json"
DEFAULT_OUT = os.path.join(REPO_BASE, "results", "sweep_v2")

TEMPLATES = {
    "HOMA": HOMA, "ANEMIA": ANEMIA, "FIB4": FIB4,
    "AIP": AIP, "CONICITY": CONICITY, "VASCULAR": VASCULAR,
    "TYG": TYG, "NLR": NLR,
}

# private_value keys (lowercase) → display root names used by the agent_eval templates.
ROOT_ALIAS = {
    "HOMA":     {"glu": "Glu", "ins": "Ins"},
    "ANEMIA":   {"hb": "Hb", "hct": "Hct", "rbc": "RBC"},
    "FIB4":     {"age": "age", "ast": "AST", "alt": "ALT", "plt": "PLT"},
    "AIP":      {"tg": "tg", "hdl": "hdl"},
    "CONICITY": {"waist": "waist", "wt": "wt", "ht": "ht"},
    "VASCULAR": {"sbp": "sbp", "dbp": "dbp"},
    "TYG":      {"tg": "tg", "glu": "glu"},
    "NLR":      {"neu": "neu", "lym": "lym"},
}

B_SETTINGS = ("k+1", "2k+1", "3k+1")
CONFIGS = (CONFIG_ALL, CONFIG_ROOTS, CONFIG_OPT)
MECHANISM = "exp"
DEFAULT_EPS_LIST = (0.1,)   # default operating point if --eps-list not given.

_print_lock = threading.Lock()
def log(msg):
    with _print_lock:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Patient loader
# ---------------------------------------------------------------------------

def load_patients(template_name: str, n: int) -> list[dict]:
    """Pull the first n patients for a template from the NHANES benchmark.

    Each entry is {"id": <patient_id>, "roots": {root_name: value}}.
    """
    benchmark = json.load(open(BENCHMARK))
    aliases = ROOT_ALIAS[template_name]
    out = []
    for sample in benchmark[template_name][:n]:
        roots: dict = {}
        for turn in sample.get("turns", []):
            for k, v in (turn.get("private_value") or {}).items():
                if k in aliases:
                    roots[aliases[k]] = float(v)
        # Sanity: every root must be present.
        if set(roots.keys()) != set(aliases.values()):
            continue  # skip incomplete samples
        out.append({"id": sample["id"], "roots": roots})
    return out


# ---------------------------------------------------------------------------
# Per-session worker
# ---------------------------------------------------------------------------

def t_for(template, B_setting: str) -> int:
    k = len(template.roots)
    return {"k+1": k + 1, "2k+1": 2 * k + 1, "3k+1": 3 * k + 1}[B_setting]


def session_filename(template_name: str, patient_idx: int,
                     config: str, B_setting: str, eps_in: float) -> str:
    return (f"{template_name}_p{patient_idx:03d}_"
            f"{config}_{B_setting.replace('+', 'plus')}_"
            f"eps{eps_in}.json")


def run_one(spec: dict, client: OpenAI, out_dir: str) -> dict:
    """Run a single session. Skips if a result file already exists."""
    out_path = os.path.join(out_dir, session_filename(
        spec["template_name"], spec["patient_idx"],
        spec["config"], spec["B_setting"], spec["eps_in"],
    ))
    if os.path.exists(out_path):
        return {"status": "skipped", "path": out_path}

    template = TEMPLATES[spec["template_name"]]()
    patient = spec["patient"]
    cfg = spec["config"]
    B_setting = spec["B_setting"]
    eps_in = spec["eps_in"]
    t = t_for(template, B_setting)
    seed = spec["seed"]

    try:
        sess = build_session_config(
            template=template, patient=patient, config=cfg,
            mechanism=MECHANISM, B_setting=B_setting, csv_path=CSV,
            eps_in=eps_in,
            rng_seed=seed, rootguard_seed=seed,
        )
        dispatcher = ToolDispatcher(
            template=template, patient=patient, mode=sess.mode,
            eps_per_call=sess.eps_per_call, rootguard=sess.rootguard,
            mechanism=sess.mechanism, m=sess.m,
            rng=np.random.default_rng(seed=seed),
        )
        if cfg == CONFIG_ALL:
            eps_per_obs = {r: sess.eps_per_call for r in template.roots}
            proxy = ProxySession.for_m_all(
                template=template, dispatcher=dispatcher,
                raw_values=patient, eps_per_call=sess.eps_per_call,
                client=client,
            )
        else:
            eps_per_obs = dict(sess.eps_per_root)
            proxy = ProxySession.for_rootguard(
                template=template, dispatcher=dispatcher,
                cached_values=sess.rootguard.all_cached(),
                client=client,
            )
        adv = Adversary(template=template, t=t)
        result = adv.run(proxy, mechanism=MECHANISM,
                         eps_per_obs=eps_per_obs, raw=patient)

        record = {
            "patient_id": spec["patient_id"],
            "patient_idx": spec["patient_idx"],
            "template": spec["template_name"],
            "config": cfg,
            "B_setting": B_setting,
            "t": t,
            "mechanism": MECHANISM,
            "eps_in": eps_in,
            "eps_per_obs": eps_per_obs,
            "seed": seed,
            "raw": patient,
            "observations": result.observations,         # text-parsed (attack input)
            "wire_observations": result.wire_observations,  # truth (analysis only)
            "rounding_gap": result.rounding_gap,
            "parse_failures": result.parse_failures,
            "map_estimate": result.map_estimate,
            "map_mae": result.map_mae,
            "naive_mae": result.naive_mae,
            "target_estimate": result.target_estimate,
            "target_truth": result.target_truth,
            "final_class": result.final_class,
            "truth_class": result.truth_class,
            "diagnosis_correct": result.diagnosis_correct,
            "n_tool_calls": len(result.tool_call_log),
            "n_sanitize_calls": sum(1 for c in result.tool_call_log
                                     if c.get("name") == "sanitize"),
            # Always-on: transcripts and the full tool-call log are
            # required for retroactive re-parsing / auditing per the
            # "Record all intermediate values" rule. Do not drop these
            # for size — disk is cheap, rerunning is not.
            "transcript": result.transcript,
            "tool_call_log": result.tool_call_log,
        }
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2, default=str)
        return {"status": "ok", "path": out_path}
    except Exception as e:
        err = traceback.format_exc()
        with open(out_path + ".err", "w") as f:
            f.write(err)
        return {"status": "error", "path": out_path, "error": str(e)}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_work_list(n_patients: int, eps_list) -> list[dict]:
    work = []
    for tn in TEMPLATES:
        patients = load_patients(tn, n_patients)
        for eps_in in eps_list:
            for i, p in enumerate(patients):
                for cfg in CONFIGS:
                    for B in B_SETTINGS:
                        work.append({
                            "template_name": tn,
                            "patient_idx": i,
                            "patient_id": p["id"],
                            "patient": p["roots"],
                            "config": cfg,
                            "B_setting": B,
                            "eps_in": float(eps_in),
                            "seed": i,
                        })
    return work


def _parse_eps_list(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=50)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--eps-list", type=_parse_eps_list,
                    default=list(DEFAULT_EPS_LIST),
                    help="Comma-separated ε values (e.g. '0.01,0.05,0.1').")
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    work = build_work_list(args.n_patients, args.eps_list)
    log(f"Sweep: {len(work)} sessions across ε={args.eps_list}, "
        f"{args.workers} concurrent workers, results -> {args.out_dir}")

    client = OpenAI()
    counts = {"ok": 0, "skipped": 0, "error": 0}
    t0 = time.time()

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, spec, client, args.out_dir): spec
                   for spec in work}
        for i, fut in enumerate(cf.as_completed(futures)):
            spec = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"status": "error", "error": str(e)}
            counts[r.get("status", "error")] = counts.get(r.get("status", "error"), 0) + 1
            done = i + 1
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (len(work) - done) / max(rate, 1e-9)
            tag = (f"{spec['template_name']:6s} p{spec['patient_idx']:03d} "
                   f"{spec['config']:5s} {spec['B_setting']:5s} "
                   f"eps{spec['eps_in']}")
            status = r.get("status", "?")
            extra = (": " + r["error"]) if status == "error" else ""
            log(f"[{done}/{len(work)}] {tag} -> {status}{extra}  "
                f"(rate={rate:.2f}/s, eta={eta/60:.1f}m)")

    log(f"Done: ok={counts.get('ok',0)}, skipped={counts.get('skipped',0)}, "
        f"error={counts.get('error',0)}, total={len(work)}, "
        f"wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
