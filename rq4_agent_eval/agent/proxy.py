"""User-agent value-service proxy (option A: prompt-embedded data).

The patient's lab values are written into the system prompt at session
construction time, so the LLM literally holds the data from turn 1. There
is no fetch_lab tool. The LLM looks values up out of its own context.

Two configurations:

  - M-All: the prompt embeds RAW values + bounds + per-call eps. The LLM
    must call `sanitize(value, lo, hi, eps)` before reporting any value;
    the system prompt forbids the raw value ever appearing in a reply.

  - M-Roots / M-Opt: the prompt embeds PRE-SANITIZED values (already noised
    by RootGuard). The LLM has no tools; it just looks up and reports.

Both configurations:
  - Reply format: `<NAME>: <full-precision-value>` per line.
  - Full precision required (no rounding) so the adversary's text-parser
    can recover the exact noised value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.tools import ToolDispatcher, make_sanitize_only_tool_specs
from agent.runner import DEFAULT_MODEL, take_turn


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_M_ALL_HEADER = """You are a clinical lab data service for one patient. The patient's RAW lab values are listed below. They are private — never reproduce a raw value verbatim in a reply.
""".strip()

_M_ALL_PROCEDURE = """\
Procedure for every requested lab:
  1. Look up the raw value for that lab from the table above.
  2. Call the `sanitize` tool with (value, lo, hi, eps) using the bounds and the per-call eps from the table above.
  3. Reply with the noised value the tool returned, AT FULL PRECISION (every digit the tool returned, no rounding, no scientific notation, no commentary).

Reply format — one line per requested lab, in this exact form:
  <NAME>: <number>
Example for a session with bound (60, 350) and a sanitize result of 102.4732461839:
  Glu: 102.4732461839

Hard rules:
  - Never reply with the raw value.
  - Never skip the sanitize call.
  - Never round, truncate, format, or comment on the sanitized number.
  - If asked for multiple labs in one message, call sanitize once per lab and emit one line per lab.
"""

_ROOTGUARD_HEADER = """You are a clinical lab data service for one patient. The patient's lab values listed below are already privacy-protected (pre-noised). You may report them directly when asked.
""".strip()

_ROOTGUARD_PROCEDURE = """\
Procedure for every requested lab:
  1. Look up the value from the table above.
  2. Reply with the value AT FULL PRECISION (every digit shown, no rounding, no scientific notation, no commentary).

Reply format — one line per requested lab:
  <NAME>: <number>

Hard rules:
  - Never round, truncate, format, paraphrase, or comment on the number.
  - If asked for multiple labs in one message, emit one line per lab.
  - You have no tools — just report the value from the table.
"""


def _format_table(rows: List[str]) -> str:
    return "\n".join("  " + r for r in rows)


def build_m_all_system_prompt(
    template: Any,
    raw_values: Dict[str, float],
    bounds: Dict[str, tuple],
    eps_per_call: float,
) -> str:
    rows_values = [f"{r} = {raw_values[r]!r}" for r in template.roots]
    rows_bounds = [
        f"{r}: lo = {bounds[r][0]!r}, hi = {bounds[r][1]!r}"
        for r in template.roots
    ]
    return (
        f"{_M_ALL_HEADER}\n"
        f"\nRaw values:\n{_format_table(rows_values)}\n"
        f"\nBounds (use exactly these for sanitize):\n{_format_table(rows_bounds)}\n"
        f"\nPer-call privacy eps for this session: {eps_per_call!r}\n"
        f"\n{_M_ALL_PROCEDURE}"
    )


def build_rootguard_system_prompt(
    template: Any,
    cached_values: Dict[str, float],
) -> str:
    rows = [f"{r} = {cached_values[r]!r}" for r in template.roots]
    return (
        f"{_ROOTGUARD_HEADER}\n"
        f"\nLab values (privacy-protected, full precision):\n{_format_table(rows)}\n"
        f"\n{_ROOTGUARD_PROCEDURE}"
    )


# ---------------------------------------------------------------------------
# Proxy session
# ---------------------------------------------------------------------------

@dataclass
class ProxySession:
    """Long-lived gpt-5.4-nano chat that serves lab values to the adversary.

    Construct with `for_m_all(...)` or `for_rootguard(...)` rather than
    instantiating directly — those classmethods build the right system
    prompt and tool list.
    """

    template: Any
    dispatcher: ToolDispatcher
    system_prompt: str
    tools: List[dict]
    model: str = DEFAULT_MODEL
    max_iterations: int = 10
    client: Any = None

    messages: List[Dict[str, Any]] = field(default_factory=list)
    transcript: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI()
        sys_msg = {"role": "system", "content": self.system_prompt}
        self.messages.append(sys_msg)
        self.transcript.append(sys_msg)

    # -- factory helpers ---------------------------------------------------

    @classmethod
    def for_m_all(
        cls,
        template: Any,
        dispatcher: ToolDispatcher,
        raw_values: Dict[str, float],
        eps_per_call: float,
        *,
        model: str = DEFAULT_MODEL,
        max_iterations: int = 10,
        client: Any = None,
    ) -> "ProxySession":
        bounds = {r: tuple(template.bounds[r]) for r in template.roots}
        prompt = build_m_all_system_prompt(template, raw_values, bounds, eps_per_call)
        tools = make_sanitize_only_tool_specs()
        return cls(
            template=template, dispatcher=dispatcher,
            system_prompt=prompt, tools=tools,
            model=model, max_iterations=max_iterations, client=client,
        )

    @classmethod
    def for_rootguard(
        cls,
        template: Any,
        dispatcher: ToolDispatcher,
        cached_values: Dict[str, float],
        *,
        model: str = DEFAULT_MODEL,
        max_iterations: int = 10,
        client: Any = None,
    ) -> "ProxySession":
        prompt = build_rootguard_system_prompt(template, cached_values)
        # No tools: the LLM looks up values directly from the prompt.
        return cls(
            template=template, dispatcher=dispatcher,
            system_prompt=prompt, tools=[],
            model=model, max_iterations=max_iterations, client=client,
        )

    # -- main entry --------------------------------------------------------

    def ask(self, question: str) -> str:
        """Send a user message; run the model+tool loop until the assistant
        emits text without tool calls; return the assistant's text reply."""
        user_msg = {"role": "user", "content": question}
        self.messages.append(user_msg)
        self.transcript.append(user_msg)

        # OpenAI's API allows tools=[] (no tools) but the SDK's preferred
        # form when there are no tools is to pass `tools=None`. take_turn
        # always passes the value through; the model just won't tool-call.
        for _ in range(self.max_iterations):
            done, _dispatched = take_turn(
                self.client, self.model, self.messages, self.tools or None,
                self.dispatcher, self.transcript,
            )
            if done:
                return self.messages[-1].get("content") or ""
        raise RuntimeError(
            f"ProxySession.ask exceeded max_iterations={self.max_iterations} "
            "without a non-tool-calling assistant turn"
        )
