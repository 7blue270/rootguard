"""Tests for the option-A user-agent ProxySession (prompt-embedded data)."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from sanitizers.rootguard import RootGuard
from templates.homa import HOMA

from agent.proxy import (
    ProxySession,
    build_m_all_system_prompt,
    build_rootguard_system_prompt,
)
from agent.tools import (
    MODE_M_ALL,
    MODE_ROOTGUARD,
    ToolDispatcher,
    make_sanitize_only_tool_specs,
)
from agent.runner import ScriptedClient, make_response


# ---------------------------------------------------------------------------
# Tool spec helper
# ---------------------------------------------------------------------------

def test_sanitize_only_tool_specs():
    specs = make_sanitize_only_tool_specs()
    names = [s["function"]["name"] for s in specs]
    assert names == ["sanitize"]
    props = specs[0]["function"]["parameters"]["properties"]
    assert sorted(props.keys()) == ["eps", "hi", "lo", "value"]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def test_m_all_prompt_embeds_raw_values_and_bounds_and_eps():
    template = HOMA()
    raw = {"Glu": 100.0, "Ins": 8.1}
    bounds = {r: tuple(template.bounds[r]) for r in template.roots}
    p = build_m_all_system_prompt(template, raw, bounds, 0.1)
    # Raw values present
    assert "100.0" in p and "8.1" in p
    # Bounds present
    assert "60.0" in p  # Glu lo
    assert "350.0" in p  # Glu hi
    # eps present
    assert "0.1" in p
    # Hard rules present
    assert "Never reply with the raw value" in p
    assert "FULL PRECISION" in p
    # Tool name mentioned
    assert "sanitize" in p


def test_rootguard_prompt_embeds_cached_only():
    template = HOMA()
    cached = {"Glu": 103.25, "Ins": 7.73}
    p = build_rootguard_system_prompt(template, cached)
    assert "103.25" in p and "7.73" in p
    # No sanitize tool reference
    assert "sanitize" not in p
    assert "FULL PRECISION" in p
    assert "no rounding" in p.lower() or "no rounding" in p


# ---------------------------------------------------------------------------
# ProxySession runtime behavior — M-All
# ---------------------------------------------------------------------------

def _disp_m_all(seed=0):
    return ToolDispatcher(
        template=HOMA(),
        patient={"Glu": 100.0, "Ins": 8.1},
        mode=MODE_M_ALL,
        eps_per_call=10.0,
        rng=np.random.default_rng(seed=seed),
    )


def test_proxy_for_m_all_calls_sanitize_then_replies():
    disp = _disp_m_all()
    script = [
        # LLM looks at the raw value (100.0) in its prompt and calls sanitize
        make_response(tool_calls=[
            {"id": "c0", "name": "sanitize",
             "arguments": {"value": 100.0, "lo": 60.0, "hi": 350.0, "eps": 0.1}},
        ]),
        # LLM reports the noised value
        make_response(content="Glu: 99.123456789"),
    ]
    proxy = ProxySession.for_m_all(
        template=HOMA(), dispatcher=disp,
        raw_values={"Glu": 100.0, "Ins": 8.1}, eps_per_call=0.1,
        client=ScriptedClient(script),
    )
    reply = proxy.ask("What is the patient's Glu value?")
    assert "Glu" in reply
    san_calls = [c for c in disp.calls if c["name"] == "sanitize"]
    assert len(san_calls) == 1
    assert san_calls[0]["arguments"]["value"] == 100.0


def test_proxy_for_m_all_no_fetch_lab_tool():
    """Confirm the LLM is NOT given fetch_lab."""
    disp = _disp_m_all()
    proxy = ProxySession.for_m_all(
        template=HOMA(), dispatcher=disp,
        raw_values={"Glu": 100.0, "Ins": 8.1}, eps_per_call=0.1,
        client=ScriptedClient([make_response(content="ok")]),
    )
    tool_names = [t["function"]["name"] for t in proxy.tools]
    assert tool_names == ["sanitize"], tool_names


# ---------------------------------------------------------------------------
# ProxySession runtime behavior — RootGuard
# ---------------------------------------------------------------------------

def test_proxy_for_rootguard_no_tools_no_calls():
    template = HOMA()
    patient = {"Glu": 100.0, "Ins": 8.1}
    rg = RootGuard.initialize(
        patient, template, {"Glu": 0.05, "Ins": 0.05},
        rng=np.random.default_rng(seed=42),
    )
    disp = ToolDispatcher(
        template=template, patient=patient, mode=MODE_ROOTGUARD,
        rootguard=rg, rng=np.random.default_rng(seed=99),
    )
    cached_glu = rg.get("Glu")
    cached_ins = rg.get("Ins")
    proxy = ProxySession.for_rootguard(
        template=template, dispatcher=disp,
        cached_values={"Glu": cached_glu, "Ins": cached_ins},
        client=ScriptedClient([
            make_response(content=f"Glu: {cached_glu!r}"),
        ]),
    )
    assert proxy.tools == []
    reply = proxy.ask("What is the patient's Glu?")
    assert str(cached_glu) in reply
    # No tool calls were made
    assert disp.calls == []


def test_proxy_long_lived_keeps_history():
    """Two consecutive asks share message history."""
    disp = _disp_m_all()
    script = [
        make_response(tool_calls=[
            {"id": "c0", "name": "sanitize",
             "arguments": {"value": 100.0, "lo": 60.0, "hi": 350.0, "eps": 0.1}},
        ]),
        make_response(content="Glu: 99.5"),
        make_response(tool_calls=[
            {"id": "c1", "name": "sanitize",
             "arguments": {"value": 8.1, "lo": 1.0, "hi": 250.0, "eps": 0.1}},
        ]),
        make_response(content="Ins: 8.0"),
    ]
    client = ScriptedClient(script)
    proxy = ProxySession.for_m_all(
        template=HOMA(), dispatcher=disp,
        raw_values={"Glu": 100.0, "Ins": 8.1}, eps_per_call=0.1,
        client=client,
    )
    proxy.ask("What is Glu?")
    n_after_first = len(client.requests)
    proxy.ask("What is Ins?")
    msgs = client.requests[n_after_first]["messages"]
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert [m["content"] for m in user_msgs] == [
        "What is Glu?", "What is Ins?",
    ]
    roles = [m["role"] for m in msgs]
    assert "assistant" in roles and "tool" in roles


def test_proxy_max_iterations_raises():
    disp = _disp_m_all()
    script = [
        make_response(tool_calls=[
            {"id": f"c{i}", "name": "sanitize",
             "arguments": {"value": 100.0, "lo": 60.0, "hi": 350.0, "eps": 0.1}},
        ])
        for i in range(5)
    ]
    proxy = ProxySession.for_m_all(
        template=HOMA(), dispatcher=disp,
        raw_values={"Glu": 100.0, "Ins": 8.1}, eps_per_call=0.1,
        client=ScriptedClient(script), max_iterations=3,
    )
    try:
        proxy.ask("What is Glu?")
    except RuntimeError as e:
        assert "max_iterations" in str(e)
        return
    raise AssertionError("expected RuntimeError after max_iterations")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_sanitize_only_tool_specs,
        test_m_all_prompt_embeds_raw_values_and_bounds_and_eps,
        test_rootguard_prompt_embeds_cached_only,
        test_proxy_for_m_all_calls_sanitize_then_replies,
        test_proxy_for_m_all_no_fetch_lab_tool,
        test_proxy_for_rootguard_no_tools_no_calls,
        test_proxy_long_lived_keeps_history,
        test_proxy_max_iterations_raises,
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
