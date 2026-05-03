"""Tests for WP4 attack pipeline (option A: prompt-embedded data).

Covers:
  - MAP estimator (mechanism math)
  - Value parser (LLM text → numbers)
  - Adversary end-to-end with ScriptedClient under M-All and RootGuard
  - Q-invariance under RootGuard
  - Dual logging: text vs wire observations + parse_failures
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from sanitizers.mechanisms import EXPONENTIAL, BLAPLACE, STAIRCASE, sanitize
from sanitizers.rootguard import RootGuard
from templates.homa import HOMA

from agent.proxy import ProxySession
from agent.tools import MODE_M_ALL, MODE_ROOTGUARD, ToolDispatcher
from agent.runner import ScriptedClient, make_response

from attack.map_recon import map_estimate
from attack.adversary import Adversary, parse_values_from_reply


# ---------------------------------------------------------------------------
# MAP estimator
# ---------------------------------------------------------------------------

def test_map_estimate_single_observation_returns_observation():
    for mech in (EXPONENTIAL, BLAPLACE, STAIRCASE):
        est = map_estimate([50.0], lo=0.0, hi=100.0, eps=0.5, mechanism=mech)
        assert abs(est - 50.0) < 0.2, f"{mech}: got {est}"


def test_map_estimate_identical_observations_returns_value():
    for mech in (EXPONENTIAL, BLAPLACE):
        est = map_estimate([42.0] * 5, lo=0.0, hi=100.0, eps=0.5, mechanism=mech)
        assert abs(est - 42.0) < 0.2


def test_map_estimate_multi_observations_converges_to_truth():
    rng = np.random.default_rng(seed=0)
    lo, hi, eps = 0.0, 100.0, 0.5
    truth = 50.0
    obs = [sanitize(truth, lo, hi, eps, mechanism=EXPONENTIAL, rng=rng)
           for _ in range(50)]
    est = map_estimate(obs, lo, hi, eps, EXPONENTIAL)
    assert abs(est - truth) <= abs(obs[0] - truth) + 0.5


def test_map_estimate_empty_raises():
    try:
        map_estimate([], lo=0.0, hi=100.0, eps=0.5, mechanism=EXPONENTIAL)
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty observations")


# ---------------------------------------------------------------------------
# Value parser
# ---------------------------------------------------------------------------

def test_parse_values_basic():
    out = parse_values_from_reply("Glu: 102.5\nIns: 7.81")
    assert out == {"Glu": 102.5, "Ins": 7.81}


def test_parse_values_handles_equals_and_extra_text():
    out = parse_values_from_reply("Here are your values:\nAST = 12.34\n  PLT: 200")
    assert out["AST"] == 12.34
    assert out["PLT"] == 200.0


def test_parse_values_empty_reply():
    assert parse_values_from_reply("") == {}
    assert parse_values_from_reply("nothing useful here") == {}


# ---------------------------------------------------------------------------
# Helpers — build scripts for M-All / RootGuard sessions
# ---------------------------------------------------------------------------

def _m_all_turn_script(value_raw: float, root_name: str, lo: float, hi: float,
                       eps: float, reply_value: float, call_id: str):
    """One M-All gather turn: sanitize tool_call + text reply."""
    return [
        make_response(tool_calls=[
            {"id": call_id, "name": "sanitize",
             "arguments": {"value": value_raw, "lo": lo, "hi": hi, "eps": eps}},
        ]),
        make_response(content=f"{root_name}: {reply_value}"),
    ]


def _m_all_final_script(template, raw, eps, reply_values):
    """M-All final turn: parallel sanitize for every root + multi-line reply."""
    tcs = []
    for i, r in enumerate(template.roots):
        lo, hi = template.bounds[r]
        tcs.append({"id": f"f{i}", "name": "sanitize",
                    "arguments": {"value": raw[r], "lo": lo, "hi": hi, "eps": eps}})
    text = "\n".join(f"{r}: {reply_values[r]}" for r in template.roots)
    return [
        make_response(tool_calls=tcs),
        make_response(content=text),
    ]


def _rootguard_text_response(root_name: str, value: float):
    """One RootGuard turn (no tool calls): direct text reply."""
    return [make_response(content=f"{root_name}: {value!r}")]


def _rootguard_final_script(template, cached):
    text = "\n".join(f"{r}: {cached[r]!r}" for r in template.roots)
    return [make_response(content=text)]


# ---------------------------------------------------------------------------
# Adversary plumbing
# ---------------------------------------------------------------------------

def test_adversary_gather_plan_round_robin():
    adv = Adversary(template=HOMA(), t=5)  # t-1 = 4, k=2 → 2 per root
    plan = adv.gather_plan()
    assert plan == ["Glu", "Ins", "Glu", "Ins"]


def test_adversary_t_too_small_raises():
    try:
        Adversary(template=HOMA(), t=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for t < 2")


def test_adversary_runs_full_session_m_all():
    """End-to-end M-All: t=3, gather Glu, gather Ins, final all-roots."""
    template = HOMA()
    raw = {"Glu": 100.0, "Ins": 8.1}
    eps = 0.1
    disp = ToolDispatcher(
        template=template, patient=raw, mode=MODE_M_ALL,
        eps_per_call=eps, rng=np.random.default_rng(seed=0),
    )
    # Reply values pretend the LLM gives us full-precision noised numbers.
    # Use values close to raw so diagnosis stays at class 1.
    script = []
    script += _m_all_turn_script(100.0, "Glu", 60.0, 350.0, eps, 99.5, "g0")
    script += _m_all_turn_script(8.1, "Ins", 1.0, 250.0, eps, 8.0, "g1")
    script += _m_all_final_script(template, raw, eps,
                                   {"Glu": 100.5, "Ins": 8.2})
    proxy = ProxySession.for_m_all(
        template=template, dispatcher=disp,
        raw_values=raw, eps_per_call=eps,
        client=ScriptedClient(script),
    )
    adv = Adversary(template=template, t=3)
    result = adv.run(proxy, mechanism=EXPONENTIAL,
                     eps_per_obs={"Glu": eps, "Ins": eps}, raw=raw)
    # Two text observations per root (one gather + one final)
    assert result.observations["Glu"] == [99.5, 100.5]
    assert result.observations["Ins"] == [8.0, 8.2]
    # Two wire observations per root (the actual sanitize results)
    assert len(result.wire_observations["Glu"]) == 2
    assert len(result.wire_observations["Ins"]) == 2
    # Rounding gap recorded for both turns of both roots
    assert len(result.rounding_gap["Glu"]) == 2
    # No parse failures
    assert result.parse_failures == {"Glu": 0, "Ins": 0}
    # Diagnosis succeeded (target ~ 100*8/405 ~= 1.97 -> class 1)
    assert result.final_class == result.truth_class
    assert result.diagnosis_correct


def test_adversary_rootguard_q_invariance():
    """Under RootGuard, all q observations of a root are identical, and the
    MAP estimate is that cached value regardless of t."""
    template = HOMA()
    raw = {"Glu": 100.0, "Ins": 8.1}
    rg = RootGuard.initialize(
        raw, template, {"Glu": 0.05, "Ins": 0.05},
        rng=np.random.default_rng(seed=42),
    )
    cached = rg.all_cached()
    disp = ToolDispatcher(
        template=template, patient=raw, mode=MODE_ROOTGUARD,
        rootguard=rg, rng=np.random.default_rng(seed=99),
    )
    # t=5 → gather plan ["Glu","Ins","Glu","Ins"] + final → 3 obs per root.
    plan = ["Glu", "Ins", "Glu", "Ins"]
    script = []
    for r in plan:
        script += _rootguard_text_response(r, cached[r])
    script += _rootguard_final_script(template, cached)

    proxy = ProxySession.for_rootguard(
        template=template, dispatcher=disp,
        cached_values=cached, client=ScriptedClient(script),
    )
    adv = Adversary(template=template, t=5)
    result = adv.run(proxy, mechanism=EXPONENTIAL,
                     eps_per_obs={"Glu": 0.05, "Ins": 0.05}, raw=raw)
    assert all(v == cached["Glu"] for v in result.observations["Glu"])
    assert all(v == cached["Ins"] for v in result.observations["Ins"])
    assert abs(result.map_estimate["Glu"] - cached["Glu"]) < 1e-6
    assert abs(result.map_estimate["Ins"] - cached["Ins"]) < 1e-6
    # Wire-observations populated from the RootGuard cache (constant)
    assert all(v == cached["Glu"] for v in result.wire_observations["Glu"])


def test_adversary_records_parse_failure_when_regex_misses():
    """If the LLM rounds/paraphrases past the regex, parse_failures increments."""
    template = HOMA()
    raw = {"Glu": 100.0, "Ins": 8.1}
    disp = ToolDispatcher(
        template=template, patient=raw, mode=MODE_M_ALL,
        eps_per_call=10.0, rng=np.random.default_rng(seed=0),
    )
    # First gather turn replies with un-parseable text (no "Glu: <num>")
    script = []
    script.append(make_response(tool_calls=[
        {"id": "g0", "name": "sanitize",
         "arguments": {"value": 100.0, "lo": 60.0, "hi": 350.0, "eps": 10.0}},
    ]))
    script.append(make_response(content="The glucose appears normal."))
    # Second gather: ok parse
    script += _m_all_turn_script(8.1, "Ins", 1.0, 250.0, 10.0, 8.05, "g1")
    # Final: ok
    script += _m_all_final_script(template, raw, 10.0,
                                   {"Glu": 100.1, "Ins": 8.05})
    proxy = ProxySession.for_m_all(
        template=template, dispatcher=disp,
        raw_values=raw, eps_per_call=10.0,
        client=ScriptedClient(script),
    )
    adv = Adversary(template=template, t=3)
    result = adv.run(proxy, mechanism=EXPONENTIAL,
                     eps_per_obs={"Glu": 10.0, "Ins": 10.0}, raw=raw)
    # The first Glu turn missed → parse_failures Glu == 1, observations Glu has 1 entry from final
    assert result.parse_failures["Glu"] == 1
    assert result.parse_failures["Ins"] == 0
    assert result.observations["Glu"] == [100.1]  # only final-turn parse
    # Wire still has both turns logged
    assert len(result.wire_observations["Glu"]) == 2


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_map_estimate_single_observation_returns_observation,
        test_map_estimate_identical_observations_returns_value,
        test_map_estimate_multi_observations_converges_to_truth,
        test_map_estimate_empty_raises,
        test_parse_values_basic,
        test_parse_values_handles_equals_and_extra_text,
        test_parse_values_empty_reply,
        test_adversary_gather_plan_round_robin,
        test_adversary_t_too_small_raises,
        test_adversary_runs_full_session_m_all,
        test_adversary_rootguard_q_invariance,
        test_adversary_records_parse_failure_when_regex_misses,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ERROR {test.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed")
