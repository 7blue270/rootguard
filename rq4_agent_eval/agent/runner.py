"""Single-session agent runner.

Wires together the prompts, the tool dispatcher, and an OpenAI tool-calling
loop to produce one transcript + final risk class for one patient under one
mode (M-All or RootGuard).

Public API:
    run_session(config, client=None) -> SessionResult

If `client` is None we instantiate `openai.OpenAI()`; otherwise the caller
provides a client (real or scripted -- see ScriptedClient below) that exposes
the OpenAI 1.x interface: client.chat.completions.create(...).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from sanitizers.allocations import load_allocation, load_per_call_eps
from sanitizers.mechanisms import EXPONENTIAL
from sanitizers.rootguard import RootGuard

from agent.prompts import SYSTEM_PROMPT, task_prompt
from agent.tools import (
    MODE_M_ALL,
    MODE_ROOTGUARD,
    VALID_MODES,
    ToolDispatcher,
    make_tool_specs,
)


DEFAULT_MODEL = "gpt-5.4-nano"

CONFIG_ALL = "all"
CONFIG_ROOTS = "roots"
CONFIG_OPT = "opt"
VALID_CONFIGS = (CONFIG_ALL, CONFIG_ROOTS, CONFIG_OPT)


# ---------------------------------------------------------------------------
# Config & result types
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    """Inputs for a single agent session.

    `mode` is the runtime backend (per-call sanitize vs. cached RootGuard).
    `config` is the experiment-layer label ("all" / "roots" / "opt") used to
    tag transcripts; build_session_config() sets it automatically.
    """
    template: Any
    patient: Dict[str, float]
    mode: str  # "m_all" or "rootguard"
    eps_per_call: Optional[float] = None      # required for m_all
    rootguard: Optional[RootGuard] = None     # required for rootguard
    mechanism: str = EXPONENTIAL
    m: int = 1000
    rng_seed: int = 0
    model: str = DEFAULT_MODEL
    max_iterations: int = 20
    config: Optional[str] = None              # "all" / "roots" / "opt"
    eps_per_root: Optional[Dict[str, float]] = None  # for "roots"/"opt"

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Unknown mode {self.mode!r}; expected one of {VALID_MODES}"
            )
        if self.config is not None and self.config not in VALID_CONFIGS:
            raise ValueError(
                f"Unknown config {self.config!r}; expected one of {VALID_CONFIGS}"
            )


@dataclass
class SessionResult:
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    final_class: Optional[int] = None
    final_target: Optional[float] = None
    fetched_values: Dict[str, float] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    n_iterations: int = 0
    failed: bool = False
    failure_reason: str = ""
    config: Optional[str] = None  # propagated from SessionConfig.config


# ---------------------------------------------------------------------------
# Final-class parser
# ---------------------------------------------------------------------------

_FINAL_RE = re.compile(r"FINAL_CLASS\s*[:=]\s*(-?\d+)", re.IGNORECASE)


def parse_final_class(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = _FINAL_RE.search(text)
    if not m:
        return None
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Single tool-call turn (reused by run_session and ProxySession)
# ---------------------------------------------------------------------------

def take_turn(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[dict],
    dispatcher: ToolDispatcher,
    transcript: List[Dict[str, Any]],
) -> tuple:
    """Run one model turn: call the API, append the assistant message,
    dispatch any tool calls, append tool replies. Mutates `messages` and
    `transcript` in place.

    Returns (done: bool, dispatched: list[(name, args, result)]).
    `done=True` iff the assistant emitted text with no tool calls.
    """
    create_kwargs: Dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        create_kwargs["tools"] = tools
        create_kwargs["tool_choice"] = "auto"
    response = client.chat.completions.create(**create_kwargs)
    msg = response.choices[0].message

    assistant_entry: Dict[str, Any] = {
        "role": "assistant",
        "content": getattr(msg, "content", None),
    }
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        assistant_entry["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    messages.append(assistant_entry)
    transcript.append(assistant_entry)

    if not tool_calls:
        return True, []

    dispatched = []
    for tc in tool_calls:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError as e:
            args = {}
            result: Dict[str, Any] = {"error": f"invalid JSON arguments: {e}"}
        else:
            try:
                result = dispatcher.dispatch(tc.function.name, args)
            except Exception as e:  # surface errors to the model
                result = {"error": f"{type(e).__name__}: {e}"}

        tool_msg = {
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.function.name,
            "content": json.dumps(result),
        }
        messages.append(tool_msg)
        transcript.append(tool_msg)
        dispatched.append((tc.function.name, args, result))

    return False, dispatched


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_session(config: SessionConfig, client: Any = None) -> SessionResult:
    """Run a single agent session end-to-end.

    Returns a SessionResult with the full transcript, the parsed final class,
    and all values the agent fetched along the way.
    """
    if client is None:
        # Lazy import so the module is importable without openai installed
        # (e.g. from tests that always pass a scripted client).
        from openai import OpenAI
        client = OpenAI()

    rng = np.random.default_rng(config.rng_seed)
    dispatcher = ToolDispatcher(
        template=config.template,
        patient=dict(config.patient),
        mode=config.mode,
        eps_per_call=config.eps_per_call,
        rootguard=config.rootguard,
        mechanism=config.mechanism,
        m=config.m,
        rng=rng,
    )
    tools = make_tool_specs(config.template)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt(config.template)},
    ]
    transcript: List[Dict[str, Any]] = list(messages)

    final_class: Optional[int] = None
    final_target: Optional[float] = None

    for iteration in range(config.max_iterations):
        done, dispatched = take_turn(
            client, config.model, messages, tools, dispatcher, transcript,
        )
        for name, _args, result in dispatched:
            if name == "compute_target" and "target" in result:
                final_target = float(result["target"])
        if done:
            parsed = parse_final_class(messages[-1].get("content"))
            if parsed is not None:
                final_class = parsed
            failed = final_class is None
            return SessionResult(
                transcript=transcript,
                final_class=final_class,
                final_target=final_target,
                fetched_values=dict(dispatcher.fetched),
                tool_calls=list(dispatcher.calls),
                n_iterations=iteration + 1,
                failed=failed,
                failure_reason="" if not failed else "no FINAL_CLASS in final message",
                config=config.config,
            )

    return SessionResult(
        transcript=transcript,
        final_class=final_class,
        final_target=final_target,
        fetched_values=dict(dispatcher.fetched),
        tool_calls=list(dispatcher.calls),
        n_iterations=config.max_iterations,
        failed=True,
        failure_reason="max_iterations reached without final answer",
        config=config.config,
    )


# ---------------------------------------------------------------------------
# Config builder: drive the three experiment configs from the allocations CSV
# ---------------------------------------------------------------------------

def build_session_config(
    template: Any,
    patient: Dict[str, float],
    config: str,
    *,
    mechanism: str = EXPONENTIAL,
    B_setting: str = "k+1",
    csv_path: Any = None,
    eps_in: Optional[float] = None,
    rng_seed: int = 0,
    rootguard_seed: Optional[int] = None,
    m: int = 1000,
    model: str = DEFAULT_MODEL,
    max_iterations: int = 20,
) -> SessionConfig:
    """Build a SessionConfig for one of the three experiment configs.

    For config="all": loads eps_per_call from the CSV and returns an M-All
    SessionConfig. For config="roots"/"opt": loads the per-root eps dict,
    initializes a RootGuard at `rootguard_seed` (defaults to rng_seed), and
    returns a RootGuard SessionConfig.

    The per-root dict is also stored on SessionConfig.eps_per_root so that
    downstream code can inspect what allocation was used.
    """
    if config not in VALID_CONFIGS:
        raise ValueError(
            f"Unknown config {config!r}; expected one of {VALID_CONFIGS}"
        )

    if config == CONFIG_ALL:
        eps = load_per_call_eps(template.name, mechanism, B_setting,
                                csv_path=csv_path, eps_in=eps_in)
        return SessionConfig(
            template=template,
            patient=dict(patient),
            mode=MODE_M_ALL,
            eps_per_call=eps,
            mechanism=mechanism,
            m=m,
            rng_seed=rng_seed,
            model=model,
            max_iterations=max_iterations,
            config=CONFIG_ALL,
            eps_per_root=None,
        )

    eps_per_root = load_allocation(
        template.name, mechanism, B_setting, config,
        csv_path=csv_path, eps_in=eps_in,
    )
    rg_seed = rng_seed if rootguard_seed is None else rootguard_seed
    rg = RootGuard.initialize(
        patient=dict(patient),
        template=template,
        eps_per_root=eps_per_root,
        mechanism=mechanism,
        m=m,
        rng=np.random.default_rng(seed=rg_seed),
    )
    return SessionConfig(
        template=template,
        patient=dict(patient),
        mode=MODE_ROOTGUARD,
        rootguard=rg,
        mechanism=mechanism,
        m=m,
        rng_seed=rng_seed,
        model=model,
        max_iterations=max_iterations,
        config=config,
        eps_per_root=dict(eps_per_root),
    )


# ---------------------------------------------------------------------------
# Scripted client (for tests and offline replays)
# ---------------------------------------------------------------------------

@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction
    type: str = "function"


@dataclass
class _FakeMessage:
    content: Optional[str] = None
    tool_calls: Optional[List[_FakeToolCall]] = None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: List[_FakeChoice]


def make_response(
    content: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> _FakeResponse:
    """Build a fake OpenAI response.

    `tool_calls` is a list of {"id": str, "name": str, "arguments": dict-or-str}.
    """
    fake_calls: Optional[List[_FakeToolCall]] = None
    if tool_calls:
        fake_calls = []
        for i, tc in enumerate(tool_calls):
            tc_id = tc.get("id") or f"call_{i}"
            args = tc["arguments"]
            args_str = args if isinstance(args, str) else json.dumps(args)
            fake_calls.append(
                _FakeToolCall(
                    id=tc_id,
                    function=_FakeFunction(name=tc["name"], arguments=args_str),
                )
            )
    return _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content=content, tool_calls=fake_calls))]
    )


class ScriptedClient:
    """Drop-in replacement for openai.OpenAI() that replays a fixed script.

    Usage:
        client = ScriptedClient([
            make_response(tool_calls=[{"name": "fetch_lab", "arguments": {"name": "AST"}}]),
            ...
            make_response(content="FINAL_CLASS: 1"),
        ])
        result = run_session(config, client=client)
    """

    def __init__(self, responses: List[_FakeResponse]):
        self._responses = list(responses)
        self.requests: List[Dict[str, Any]] = []
        # Mirror client.chat.completions.create:
        self.chat = self
        self.completions = self

    def create(self, **kwargs) -> _FakeResponse:
        self.requests.append(kwargs)
        if not self._responses:
            raise RuntimeError("ScriptedClient ran out of scripted responses")
        return self._responses.pop(0)
