"""Tool schemas + dispatcher for the medical-screening agent.

Three tools are exposed to the model:

  - fetch_lab(name): per-call sanitized lab read (M-All) OR a cached read from
    a preloaded RootGuard instance.
  - compute_target(<roots...>): pure backend; applies the template's formula.
  - classify_risk(value): pure backend; returns the integer risk class.

The dispatcher routes calls to the right backend based on the session mode.
The two modes ("m_all" and "rootguard") are mutually exclusive: M-All needs
the raw patient + per-call eps; RootGuard needs a preinitialized RootGuard.

This module is independent of the OpenAI SDK -- runner.py is the only place
that imports openai. Tests can construct ToolDispatcher directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from sanitizers.mechanisms import sanitize, EXPONENTIAL
from sanitizers.rootguard import RootGuard


MODE_M_ALL = "m_all"
MODE_ROOTGUARD = "rootguard"
VALID_MODES = (MODE_M_ALL, MODE_ROOTGUARD)


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI structured tool-calling format)
# ---------------------------------------------------------------------------

def make_tool_specs(template) -> List[dict]:
    """Build the OpenAI `tools` array for a given template.

    fetch_lab restricts `name` to the template's roots; compute_target's
    parameters are exactly the template's roots (one numeric parameter each).
    """
    roots = list(template.roots)
    return [
        {
            "type": "function",
            "function": {
                "name": "fetch_lab",
                "description": (
                    "Retrieve one privacy-protected lab value for the patient. "
                    "Returns {\"name\": str, \"value\": number}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": roots,
                            "description": "Name of the root variable to fetch.",
                        },
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compute_target",
                "description": (
                    "Apply the template's diagnostic formula to a complete set of "
                    "lab values. Returns {\"target\": number}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        r: {"type": "number", "description": f"Lab value for {r}."}
                        for r in roots
                    },
                    "required": roots,
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "classify_risk",
                "description": (
                    "Map a target value to its clinical risk class. "
                    "Returns {\"risk_class\": integer}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "number",
                            "description": "The target value returned by compute_target.",
                        },
                    },
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        },
    ]


TOOL_NAMES = ("fetch_lab", "compute_target", "classify_risk")


def make_value_service_tool_specs(template) -> List[dict]:
    """OpenAI tool schema with only `fetch_lab` exposed.

    Used by the user-agent ProxySession: the LLM is a value-serving lab
    service, not a diagnostician. compute_target / classify_risk live with
    the adversary program in attack/.
    """
    full = make_tool_specs(template)
    return [t for t in full if t["function"]["name"] == "fetch_lab"]


def make_sanitize_only_tool_specs() -> List[dict]:
    """Tool schemas for the prompt-embedded user-agent (option A, M-All).

    The patient values live in the system prompt. M-All sessions expose only
    `sanitize(value, lo, hi, eps)` so the LLM can noise raw values before
    replying. RootGuard sessions get NO tools (values in the prompt are
    already pre-noised — the LLM just looks them up).
    """
    return [{
        "type": "function",
        "function": {
            "name": "sanitize",
            "description": (
                "Apply privacy-preserving noise to a single value. Returns "
                "{\"sanitized_value\": number}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number",
                               "description": "Raw value to sanitize."},
                    "lo": {"type": "number",
                           "description": "Lower bound of the value's domain."},
                    "hi": {"type": "number",
                           "description": "Upper bound of the value's domain."},
                    "eps": {"type": "number",
                            "description": "Privacy parameter (per-call)."},
                },
                "required": ["value", "lo", "hi", "eps"],
                "additionalProperties": False,
            },
        },
    }]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

@dataclass
class ToolDispatcher:
    """Backend for the agent's three tools, parameterized by mode.

    For mode='m_all', fetch_lab calls sanitize() with the raw patient value
    and the per-call epsilon. For mode='rootguard', fetch_lab returns the
    cached value from the preloaded RootGuard.
    """

    template: Any
    patient: Dict[str, float]
    mode: str
    eps_per_call: Optional[float] = None
    rootguard: Optional[RootGuard] = None
    mechanism: str = EXPONENTIAL
    m: int = 1000
    rng: Optional[np.random.Generator] = None

    fetched: Dict[str, float] = field(default_factory=dict)
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Unknown mode {self.mode!r}; expected one of {VALID_MODES}"
            )
        if self.mode == MODE_M_ALL:
            if self.eps_per_call is None or self.eps_per_call <= 0:
                raise ValueError("M-All mode requires eps_per_call > 0")
            if not self.patient:
                raise ValueError("M-All mode requires a non-empty patient dict")
        elif self.mode == MODE_ROOTGUARD:
            if self.rootguard is None:
                raise ValueError("RootGuard mode requires a preloaded RootGuard")
        if self.rng is None:
            self.rng = np.random.default_rng()

    # -- individual tools ---------------------------------------------------

    def fetch_lab(self, name: str) -> Dict[str, Any]:
        if name not in self.template.roots:
            raise ValueError(
                f"Unknown lab {name!r}; must be one of {self.template.roots}"
            )
        if self.mode == MODE_ROOTGUARD:
            value = self.rootguard.get(name)
        else:  # MODE_M_ALL
            if name not in self.patient:
                raise KeyError(f"Patient missing root {name!r}")
            lo, hi = self.template.bounds[name]
            value = sanitize(
                value=self.patient[name],
                lo=lo,
                hi=hi,
                eps=self.eps_per_call,
                mechanism=self.mechanism,
                m=self.m,
                rng=self.rng,
            )
        self.fetched[name] = float(value)
        return {"name": name, "value": float(value)}

    def compute_target(self, values: Dict[str, float]) -> Dict[str, float]:
        missing = [r for r in self.template.roots if r not in values]
        if missing:
            raise ValueError(
                f"compute_target missing roots: {missing}"
            )
        target = self.template.compute(
            {r: float(values[r]) for r in self.template.roots}
        )
        return {"target": float(target)}

    def classify_risk(self, value: float) -> Dict[str, int]:
        return {"risk_class": int(self.template.risk_class(float(value)))}

    def sanitize_value(self, value: float, lo: float, hi: float,
                       eps: float) -> Dict[str, float]:
        """Apply mechanism-specific noise to one value (option A, M-All).

        The mechanism is fixed at dispatcher construction time; the LLM
        provides the bounds and eps it was told to use in the system prompt.
        """
        sanitized = sanitize(
            value=float(value),
            lo=float(lo),
            hi=float(hi),
            eps=float(eps),
            mechanism=self.mechanism,
            m=self.m,
            rng=self.rng,
        )
        return {"sanitized_value": float(sanitized)}

    # -- dispatch ----------------------------------------------------------

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route a tool call to its backend; record the call for the transcript."""
        if name == "fetch_lab":
            result = self.fetch_lab(arguments["name"])
        elif name == "compute_target":
            # Accept either the flat form ({root: v, ...}) or a nested
            # {"values": {...}} form.
            if "values" in arguments and isinstance(arguments["values"], dict):
                values = arguments["values"]
            else:
                values = arguments
            result = self.compute_target(values)
        elif name == "classify_risk":
            result = self.classify_risk(arguments["value"])
        elif name == "sanitize":
            result = self.sanitize_value(
                value=arguments["value"],
                lo=arguments["lo"],
                hi=arguments["hi"],
                eps=arguments["eps"],
            )
        else:
            raise ValueError(f"Unknown tool {name!r}")
        self.calls.append({"name": name, "arguments": arguments, "result": result})
        return result
