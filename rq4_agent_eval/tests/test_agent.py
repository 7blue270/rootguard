"""Tests for WP3 (agent harness): prompts, tools, runner.

These tests use the in-process ScriptedClient to drive run_session without
hitting the OpenAI API. Real-API integration is exercised by the sweep
runner (WP5).
"""

import json
import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from sanitizers.allocations import load_allocation, load_per_call_eps
from sanitizers.rootguard import RootGuard
from templates.fib4 import FIB4
from templates.homa import HOMA
from templates.anemia import ANEMIA

from agent.prompts import SYSTEM_PROMPT, task_prompt
from agent.tools import (
    MODE_M_ALL,
    MODE_ROOTGUARD,
    ToolDispatcher,
    make_tool_specs,
)
from agent.runner import (
    CONFIG_ALL,
    CONFIG_OPT,
    CONFIG_ROOTS,
    SessionConfig,
    ScriptedClient,
    build_session_config,
    make_response,
    parse_final_class,
    run_session,
)


EXAMPLE_ALLOC = pathlib.Path(__file__).resolve().parent.parent / "data" / "allocations.example.csv"


# ---------------------------------------------------------------------------
# Prompt + schema tests
# ---------------------------------------------------------------------------

def test_system_prompt_mentions_tools():
    for tool in ("fetch_lab", "compute_target", "classify_risk", "FINAL_CLASS"):
        assert tool in SYSTEM_PROMPT, f"system prompt missing {tool!r}"


def test_task_prompt_lists_roots():
    p = task_prompt(FIB4())
    for r in FIB4.roots:
        assert r in p, f"task prompt missing root {r}"
    assert "FIB4" in p


def test_tool_specs_shape():
    specs = make_tool_specs(FIB4())
    names = [s["function"]["name"] for s in specs]
    assert names == ["fetch_lab", "compute_target", "classify_risk"]
    # fetch_lab has enum constrained to roots
    enum = specs[0]["function"]["parameters"]["properties"]["name"]["enum"]
    assert sorted(enum) == sorted(FIB4.roots)
    # compute_target requires every root by name
    ct_props = specs[1]["function"]["parameters"]["properties"]
    assert sorted(ct_props.keys()) == sorted(FIB4.roots)
    assert sorted(specs[1]["function"]["parameters"]["required"]) == sorted(FIB4.roots)
    # classify_risk takes one numeric value
    assert "value" in specs[2]["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# Dispatcher tests (M-All)
# ---------------------------------------------------------------------------

def test_dispatcher_m_all_fetch_lab_uses_sanitize():
    """In M-All mode, fetch_lab returns a sanitized value within bounds and
    each call may differ (different RNG state)."""
    template = FIB4()
    patient = {"age": 50.0, "AST": 25.0, "PLT": 200.0, "ALT": 30.0}
    disp = ToolDispatcher(
        template=template, patient=patient, mode=MODE_M_ALL,
        eps_per_call=0.5, rng=np.random.default_rng(seed=1),
    )
    out1 = disp.fetch_lab("AST")
    out2 = disp.fetch_lab("AST")
    assert out1["name"] == "AST"
    lo, hi = template.bounds["AST"]
    assert lo <= out1["value"] <= hi
    assert lo <= out2["value"] <= hi
    # Different per-call sanitization -> at least sometimes differs.
    # We don't assert inequality on a single pair (could collide), but the
    # value must come from sanitize at least.
    assert disp.fetched["AST"] == out2["value"]


def test_dispatcher_m_all_requires_eps():
    try:
        ToolDispatcher(
            template=FIB4(), patient={"age": 50, "AST": 25, "PLT": 200, "ALT": 30},
            mode=MODE_M_ALL,
        )
    except ValueError:
        return
    raise AssertionError("ToolDispatcher should reject M-All without eps_per_call")


def test_dispatcher_unknown_lab_raises():
    disp = ToolDispatcher(
        template=FIB4(),
        patient={"age": 50.0, "AST": 25.0, "PLT": 200.0, "ALT": 30.0},
        mode=MODE_M_ALL, eps_per_call=0.5,
        rng=np.random.default_rng(seed=0),
    )
    try:
        disp.fetch_lab("not_a_lab")
    except ValueError:
        return
    raise AssertionError("fetch_lab should reject unknown lab names")


# ---------------------------------------------------------------------------
# Dispatcher tests (RootGuard)
# ---------------------------------------------------------------------------

def test_dispatcher_rootguard_returns_cached():
    template = HOMA()
    patient = {"Glu": 100.0, "Ins": 10.0}
    rg = RootGuard.initialize(
        patient, template, {"Glu": 0.05, "Ins": 0.05},
        rng=np.random.default_rng(seed=42),
    )
    disp = ToolDispatcher(
        template=template, patient=patient, mode=MODE_ROOTGUARD, rootguard=rg,
        rng=np.random.default_rng(seed=99),
    )
    a = disp.fetch_lab("Glu")
    b = disp.fetch_lab("Glu")
    assert a["value"] == b["value"]
    # Matches the underlying RootGuard cache exactly
    assert a["value"] == rg.get("Glu")


def test_dispatcher_rootguard_requires_instance():
    try:
        ToolDispatcher(template=HOMA(), patient={"Glu": 100, "Ins": 10},
                       mode=MODE_ROOTGUARD)
    except ValueError:
        return
    raise AssertionError("RootGuard mode without instance should raise")


# ---------------------------------------------------------------------------
# Dispatcher: compute_target / classify_risk
# ---------------------------------------------------------------------------

def test_dispatcher_compute_target_flat_args():
    template = FIB4()
    disp = ToolDispatcher(
        template=template,
        patient={"age": 66, "AST": 12.0, "PLT": 170.0, "ALT": 10.0},
        mode=MODE_M_ALL, eps_per_call=1.0,
        rng=np.random.default_rng(seed=0),
    )
    out = disp.compute_target({"age": 66, "AST": 12.0, "PLT": 170.0, "ALT": 10.0})
    expected = (66 * 12.0) / (170.0 * math.sqrt(10.0))
    assert abs(out["target"] - expected) < 1e-9


def test_dispatcher_compute_target_missing_root():
    disp = ToolDispatcher(
        template=FIB4(),
        patient={"age": 50, "AST": 25.0, "PLT": 200.0, "ALT": 30.0},
        mode=MODE_M_ALL, eps_per_call=1.0,
        rng=np.random.default_rng(seed=0),
    )
    try:
        disp.compute_target({"age": 50, "AST": 25.0})
    except ValueError as e:
        assert "missing" in str(e).lower()
        return
    raise AssertionError("compute_target should reject incomplete root sets")


def test_dispatcher_classify_risk():
    disp = ToolDispatcher(
        template=FIB4(), patient={"age": 50, "AST": 25.0, "PLT": 200.0, "ALT": 30.0},
        mode=MODE_M_ALL, eps_per_call=1.0, rng=np.random.default_rng(seed=0),
    )
    assert disp.classify_risk(0.5)["risk_class"] == 0
    assert disp.classify_risk(2.0)["risk_class"] == 1
    assert disp.classify_risk(5.0)["risk_class"] == 2


def test_dispatch_records_calls():
    disp = ToolDispatcher(
        template=HOMA(), patient={"Glu": 100.0, "Ins": 8.1},
        mode=MODE_M_ALL, eps_per_call=10.0,  # very high eps -> ~no noise
        rng=np.random.default_rng(seed=0),
    )
    disp.dispatch("fetch_lab", {"name": "Glu"})
    disp.dispatch("compute_target", {"Glu": 100.0, "Ins": 8.1})
    disp.dispatch("classify_risk", {"value": 2.0})
    names = [c["name"] for c in disp.calls]
    assert names == ["fetch_lab", "compute_target", "classify_risk"]


# ---------------------------------------------------------------------------
# Final-class parser
# ---------------------------------------------------------------------------

def test_parse_final_class_basic():
    assert parse_final_class("FINAL_CLASS: 1") == 1
    assert parse_final_class("blah\nFINAL_CLASS: 2\nmore") == 2
    assert parse_final_class("final_class = 0") == 0
    assert parse_final_class("nothing here") is None
    assert parse_final_class(None) is None


# ---------------------------------------------------------------------------
# End-to-end runner with ScriptedClient
# ---------------------------------------------------------------------------

def _homa_script():
    """Standard HOMA tool-call sequence: fetch Glu, fetch Ins, compute, classify, finalize."""
    return [
        make_response(tool_calls=[
            {"id": "c0", "name": "fetch_lab", "arguments": {"name": "Glu"}},
        ]),
        make_response(tool_calls=[
            {"id": "c1", "name": "fetch_lab", "arguments": {"name": "Ins"}},
        ]),
        make_response(tool_calls=[
            {"id": "c2", "name": "compute_target",
             "arguments": {"Glu": 100.0, "Ins": 8.1}},
        ]),
        make_response(tool_calls=[
            {"id": "c3", "name": "classify_risk", "arguments": {"value": 2.0}},
        ]),
        make_response(content="Done. FINAL_CLASS: 1"),
    ]


def test_run_session_m_all_happy_path():
    template = HOMA()
    patient = {"Glu": 100.0, "Ins": 8.1}
    config = SessionConfig(
        template=template, patient=patient, mode=MODE_M_ALL,
        eps_per_call=10.0, rng_seed=7,
    )
    client = ScriptedClient(_homa_script())
    result = run_session(config, client=client)

    assert not result.failed, result.failure_reason
    assert result.final_class == 1
    # final_target is what compute_target returned
    assert result.final_target is not None
    assert abs(result.final_target - 2.0) < 1e-9
    # Both roots must have been fetched
    assert set(result.fetched_values.keys()) == set(template.roots)
    # All five model turns consumed
    assert len(client.requests) == 5
    # Tool-call log matches the script
    assert [c["name"] for c in result.tool_calls] == \
        ["fetch_lab", "fetch_lab", "compute_target", "classify_risk"]


def test_run_session_rootguard_uses_cache():
    template = HOMA()
    patient = {"Glu": 100.0, "Ins": 8.1}
    rg = RootGuard.initialize(
        patient, template, {"Glu": 0.05, "Ins": 0.05},
        rng=np.random.default_rng(seed=42),
    )
    cached_glu = rg.get("Glu")
    cached_ins = rg.get("Ins")

    config = SessionConfig(
        template=template, patient=patient, mode=MODE_ROOTGUARD,
        rootguard=rg, rng_seed=7,
    )
    # Build a script where the agent fetches Glu twice and Ins once;
    # all three reads must equal the cached values.
    script = [
        make_response(tool_calls=[
            {"id": "c0", "name": "fetch_lab", "arguments": {"name": "Glu"}},
        ]),
        make_response(tool_calls=[
            {"id": "c1", "name": "fetch_lab", "arguments": {"name": "Glu"}},
        ]),
        make_response(tool_calls=[
            {"id": "c2", "name": "fetch_lab", "arguments": {"name": "Ins"}},
        ]),
        make_response(tool_calls=[
            {"id": "c3", "name": "compute_target",
             "arguments": {"Glu": cached_glu, "Ins": cached_ins}},
        ]),
        make_response(tool_calls=[
            {"id": "c4", "name": "classify_risk",
             "arguments": {"value": (cached_glu * cached_ins) / 405.0}},
        ]),
        make_response(content="FINAL_CLASS: 0"),
    ]
    result = run_session(config, client=ScriptedClient(script))

    # Every fetch_lab returned the cached value
    fetch_calls = [c for c in result.tool_calls if c["name"] == "fetch_lab"]
    assert len(fetch_calls) == 3
    glu_results = [c["result"]["value"] for c in fetch_calls if c["arguments"]["name"] == "Glu"]
    assert glu_results == [cached_glu, cached_glu]
    ins_results = [c["result"]["value"] for c in fetch_calls if c["arguments"]["name"] == "Ins"]
    assert ins_results == [cached_ins]


def test_run_session_no_final_class_marks_failed():
    template = HOMA()
    config = SessionConfig(
        template=template, patient={"Glu": 100.0, "Ins": 8.1}, mode=MODE_M_ALL,
        eps_per_call=10.0, rng_seed=0,
    )
    # Single response with no tool calls and no FINAL_CLASS line.
    client = ScriptedClient([make_response(content="I'm not sure.")])
    result = run_session(config, client=client)
    assert result.failed
    assert result.final_class is None
    assert "FINAL_CLASS" in result.failure_reason


def test_run_session_max_iterations_caps_loop():
    """If the agent keeps tool-calling forever, runner must terminate."""
    template = HOMA()
    config = SessionConfig(
        template=template, patient={"Glu": 100.0, "Ins": 8.1}, mode=MODE_M_ALL,
        eps_per_call=10.0, rng_seed=0, max_iterations=3,
    )
    # Three responses, all tool-calling; runner stops after iteration 3.
    script = [
        make_response(tool_calls=[
            {"id": f"c{i}", "name": "fetch_lab", "arguments": {"name": "Glu"}},
        ])
        for i in range(3)
    ]
    result = run_session(config, client=ScriptedClient(script))
    assert result.failed
    assert result.n_iterations == 3
    assert "max_iterations" in result.failure_reason


def test_run_session_dispatcher_error_surfaces_to_model():
    """Tool errors should be returned as JSON to the model rather than crashing."""
    template = HOMA()
    config = SessionConfig(
        template=template, patient={"Glu": 100.0, "Ins": 8.1}, mode=MODE_M_ALL,
        eps_per_call=10.0, rng_seed=0,
    )
    script = [
        make_response(tool_calls=[
            {"id": "c0", "name": "fetch_lab", "arguments": {"name": "DOES_NOT_EXIST"}},
        ]),
        make_response(content="FINAL_CLASS: 0"),
    ]
    result = run_session(config, client=ScriptedClient(script))
    # First tool message should carry an error payload
    tool_msgs = [m for m in result.transcript if m.get("role") == "tool"]
    assert len(tool_msgs) >= 1
    payload = json.loads(tool_msgs[0]["content"])
    assert "error" in payload
    # And the run still completed
    assert result.final_class == 0


def test_run_session_default_model_is_nano():
    template = HOMA()
    config = SessionConfig(
        template=template, patient={"Glu": 100.0, "Ins": 8.1}, mode=MODE_M_ALL,
        eps_per_call=10.0, rng_seed=0,
    )
    client = ScriptedClient([make_response(content="FINAL_CLASS: 0")])
    run_session(config, client=client)
    assert client.requests[0]["model"] == "gpt-5.4-nano"


# ---------------------------------------------------------------------------
# build_session_config: three experiment configs from the allocations CSV
# ---------------------------------------------------------------------------

def test_build_session_config_all():
    template = HOMA()
    patient = {"Glu": 100.0, "Ins": 8.1}
    cfg = build_session_config(
        template, patient, CONFIG_ALL,
        mechanism="exp", B_setting="k+1", csv_path=EXAMPLE_ALLOC, rng_seed=0,
    )
    assert cfg.mode == MODE_M_ALL
    assert cfg.rootguard is None
    assert cfg.eps_per_call == 0.1
    assert cfg.eps_per_root is None
    assert cfg.config == CONFIG_ALL


def test_build_session_config_roots_uniform():
    template = HOMA()
    patient = {"Glu": 100.0, "Ins": 8.1}
    cfg = build_session_config(
        template, patient, CONFIG_ROOTS,
        mechanism="exp", B_setting="k+1", csv_path=EXAMPLE_ALLOC, rng_seed=42,
    )
    assert cfg.mode == MODE_ROOTGUARD
    assert cfg.eps_per_call is None
    assert cfg.rootguard is not None
    # Uniform: both roots get the same eps
    assert cfg.eps_per_root == {"Glu": 0.15, "Ins": 0.15}
    # RootGuard cache covers every root and stays within domain
    for r in template.roots:
        v = cfg.rootguard.get(r)
        lo, hi = template.bounds[r]
        assert lo <= v <= hi
    assert cfg.config == CONFIG_ROOTS


def test_build_session_config_opt_matches_csv():
    template = HOMA()
    patient = {"Glu": 100.0, "Ins": 8.1}
    cfg = build_session_config(
        template, patient, CONFIG_OPT,
        mechanism="exp", B_setting="k+1", csv_path=EXAMPLE_ALLOC, rng_seed=42,
    )
    assert cfg.mode == MODE_ROOTGUARD
    expected = load_allocation("HOMA", "exp", "k+1", "opt", csv_path=EXAMPLE_ALLOC)
    assert cfg.eps_per_root == expected
    # Sanity: M-Opt for HOMA k+1 puts more on Ins than Glu
    assert cfg.eps_per_root["Ins"] > cfg.eps_per_root["Glu"]
    assert cfg.config == CONFIG_OPT


def test_build_session_config_anemia_opt_clamps_rbc():
    """ANEMIA M-Opt should reflect the eps_min clamp on the zero-sensitivity RBC root."""
    template = ANEMIA()
    patient = {"Hb": 14.0, "Hct": 42.0, "RBC": 5.0}
    cfg = build_session_config(
        template, patient, CONFIG_OPT,
        mechanism="exp", B_setting="2k+1", csv_path=EXAMPLE_ALLOC, rng_seed=0,
    )
    assert cfg.eps_per_root["RBC"] < 0.01
    assert cfg.eps_per_root["Hb"] > cfg.eps_per_root["RBC"] * 100


def test_build_session_config_unknown_config_raises():
    try:
        build_session_config(HOMA(), {"Glu": 100.0, "Ins": 8.1}, "not_a_config",
                             csv_path=EXAMPLE_ALLOC)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown config name")


def test_build_session_config_then_run_m_opt():
    """End-to-end: build M-Opt config, drive with ScriptedClient."""
    template = HOMA()
    patient = {"Glu": 100.0, "Ins": 8.1}
    cfg = build_session_config(
        template, patient, CONFIG_OPT,
        mechanism="exp", B_setting="k+1", csv_path=EXAMPLE_ALLOC, rng_seed=42,
    )
    glu = cfg.rootguard.get("Glu")
    ins = cfg.rootguard.get("Ins")
    target = (glu * ins) / 405.0
    expected_class = template.risk_class(target)
    script = [
        make_response(tool_calls=[
            {"id": "c0", "name": "fetch_lab", "arguments": {"name": "Glu"}},
            {"id": "c1", "name": "fetch_lab", "arguments": {"name": "Ins"}},
        ]),
        make_response(tool_calls=[
            {"id": "c2", "name": "compute_target",
             "arguments": {"Glu": glu, "Ins": ins}},
        ]),
        make_response(tool_calls=[
            {"id": "c3", "name": "classify_risk", "arguments": {"value": target}},
        ]),
        make_response(content=f"FINAL_CLASS: {expected_class}"),
    ]
    result = run_session(cfg, client=ScriptedClient(script))
    assert not result.failed, result.failure_reason
    assert result.config == CONFIG_OPT
    assert result.final_class == expected_class
    assert abs(result.final_target - target) < 1e-9
    # All RootGuard reads matched the cache
    fetches = [c for c in result.tool_calls if c["name"] == "fetch_lab"]
    assert {f["arguments"]["name"]: f["result"]["value"] for f in fetches} == {
        "Glu": glu, "Ins": ins,
    }


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_system_prompt_mentions_tools,
        test_task_prompt_lists_roots,
        test_tool_specs_shape,
        test_dispatcher_m_all_fetch_lab_uses_sanitize,
        test_dispatcher_m_all_requires_eps,
        test_dispatcher_unknown_lab_raises,
        test_dispatcher_rootguard_returns_cached,
        test_dispatcher_rootguard_requires_instance,
        test_dispatcher_compute_target_flat_args,
        test_dispatcher_compute_target_missing_root,
        test_dispatcher_classify_risk,
        test_dispatch_records_calls,
        test_parse_final_class_basic,
        test_run_session_m_all_happy_path,
        test_run_session_rootguard_uses_cache,
        test_run_session_no_final_class_marks_failed,
        test_run_session_max_iterations_caps_loop,
        test_run_session_dispatcher_error_surfaces_to_model,
        test_run_session_default_model_is_nano,
        test_build_session_config_all,
        test_build_session_config_roots_uniform,
        test_build_session_config_opt_matches_csv,
        test_build_session_config_anemia_opt_clamps_rbc,
        test_build_session_config_unknown_config_raises,
        test_build_session_config_then_run_m_opt,
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
            print(f"  ERROR {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed")
