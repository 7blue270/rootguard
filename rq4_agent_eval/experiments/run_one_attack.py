"""Run one HOMA patient through the t-turn attack pipeline under all
3 configs (M-All / M-Roots / M-Opt) x 3 budget settings (k+1, 2k+1, 3k+1).

Prints a 9-row summary: per-root MAP-reconstruction MAE, naive (single-obs)
MAE for comparison, and final-diagnosis correctness.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from openai import OpenAI

from sanitizers.allocations import load_allocation, load_per_call_eps
from templates.homa import HOMA

from agent.proxy import ProxySession
from agent.runner import (
    CONFIG_ALL, CONFIG_OPT, CONFIG_ROOTS,
    build_session_config,
)

from attack.adversary import Adversary


CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "allocations.example.csv",
)
MECHANISM = "exp"

# B_setting -> t (total turns). For HOMA k=2, t = (B_setting * k) + 1 has the
# right shape for 1, 2, 3 queries per root in the gather phase.
B_TO_T = {"k+1": 3, "2k+1": 5, "3k+1": 7}


def _eps_per_obs_for(config, template, B_setting):
    """Return the per-observation eps each fetch_lab call was sanitized at,
    keyed by root name. For M-All it's the same per-call scalar for every
    root; for M-Roots / M-Opt it's the per-root cached eps."""
    if config == CONFIG_ALL:
        eps = load_per_call_eps(template.name, MECHANISM, B_setting,
                                csv_path=CSV)
        return {r: eps for r in template.roots}
    return load_allocation(template.name, MECHANISM, B_setting, config,
                           csv_path=CSV)


def main():
    template = HOMA()
    patient = {"Glu": 100.0, "Ins": 8.1}
    truth_target = template.compute(patient)
    truth_class = template.risk_class(truth_target)

    print("=" * 84)
    print(f"  HOMA patient {patient}  →  target={truth_target:.4f}, class={truth_class}")
    print("=" * 84)
    print()

    client = OpenAI()
    rows = []

    for config in (CONFIG_ALL, CONFIG_ROOTS, CONFIG_OPT):
        for B_setting in ("k+1", "2k+1", "3k+1"):
            t = B_TO_T[B_setting]
            try:
                sess = build_session_config(
                    template=template, patient=patient, config=config,
                    mechanism=MECHANISM, B_setting=B_setting,
                    csv_path=CSV, rng_seed=42, rootguard_seed=42,
                )
            except KeyError as e:
                print("-" * 84)
                print(f"  config={config:5s}  B_setting={B_setting:5s}  t={t}  "
                      f"SKIP: {e}")
                print()
                continue
            # Build a dispatcher mirroring the SessionConfig and wrap in proxy.
            from agent.tools import (
                MODE_M_ALL, MODE_ROOTGUARD, ToolDispatcher,
            )
            dispatcher = ToolDispatcher(
                template=template, patient=patient, mode=sess.mode,
                eps_per_call=sess.eps_per_call, rootguard=sess.rootguard,
                mechanism=sess.mechanism, m=sess.m,
                rng=np.random.default_rng(seed=42),
            )
            proxy = ProxySession(template=template, dispatcher=dispatcher,
                                 client=client)
            adv = Adversary(template=template, t=t)

            try:
                eps_per_obs = _eps_per_obs_for(config, template, B_setting)
                result = adv.run(proxy, mechanism=MECHANISM,
                                 eps_per_obs=eps_per_obs, raw=patient)
            except Exception:
                traceback.print_exc()
                print(f"  config={config} B={B_setting} t={t} FAILED")
                continue

            print("-" * 84)
            print(f"  config={config:5s}  B_setting={B_setting:5s}  t={t}")
            print(f"    eps_per_obs : {eps_per_obs}")
            print(f"    observations: " + "; ".join(
                f"{r}={[round(v, 2) for v in result.observations[r]]}"
                for r in template.roots
            ))
            print(f"    MAP estimate: " + "; ".join(
                f"{r}={result.map_estimate[r]:.3f} (true {patient[r]:.1f}, "
                f"MAE={result.map_mae[r]:.3f})" for r in template.roots
            ))
            print(f"    naive MAE   : " + "; ".join(
                f"{r}={result.naive_mae[r]:.3f}" for r in template.roots
            ))
            print(f"    diagnosis   : final_class={result.final_class} "
                  f"(truth={result.truth_class})  "
                  f"correct={result.diagnosis_correct}")
            rows.append((config, B_setting, t, result))
            print()

    # Summary table
    print("=" * 84)
    print("  SUMMARY")
    print("=" * 84)
    header = f"  {'config':<8} {'B':<6} {'t':>2}  " \
             f"{'MAE Glu':>10} {'MAE Ins':>10}  " \
             f"{'naive Glu':>10} {'naive Ins':>10}  " \
             f"{'class':>5} {'truth':>5}  ok"
    print(header)
    for config, B, t, r in rows:
        print(f"  {config:<8} {B:<6} {t:>2}  "
              f"{r.map_mae['Glu']:>10.3f} {r.map_mae['Ins']:>10.3f}  "
              f"{r.naive_mae['Glu']:>10.3f} {r.naive_mae['Ins']:>10.3f}  "
              f"{r.final_class:>5} {r.truth_class:>5}  "
              f"{'Y' if r.diagnosis_correct else 'N'}")


if __name__ == "__main__":
    main()
