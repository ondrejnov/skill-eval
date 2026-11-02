"""Grader `tool_used` (počet volání toolu v intervalu [min, max]) a společné párování volání pro `tool_order`."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from skilleval.grading.regexjs import compile_js_regex
from skilleval.models import GraderSpec, RunArtifacts, ToolCallRecord

EVIDENCE_CHARS = 400


def input_json(call: ToolCallRecord) -> str:
    """JSON vstupu tak, jak ho vidí `input_match` (kompaktní jako JSON.stringify, ne-ASCII ponechané)."""
    return json.dumps(call.input, ensure_ascii=False, separators=(",", ":"))


class CallMatcher:
    """Název toolu (přesný) + volitelný `input_match` (JS regex nad JSON vstupu)."""

    def __init__(self, tool: Any, input_match: Any = None):
        if not isinstance(tool, str) or not tool:
            raise ValueError("`tool` must be a non-empty tool name")
        self.tool = tool
        self.input_match = str(input_match) if input_match not in (None, "") else None
        self.rx: re.Pattern[str] | None = compile_js_regex(self.input_match) if self.input_match else None

    @classmethod
    def parse(cls, value: Any, what: str) -> CallMatcher:
        """Název toolu nebo {tool, input_match}."""
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, dict):
            extra = set(value) - {"tool", "input_match"}
            if extra:
                raise ValueError(f"`{what}` has unknown key(s): {', '.join(sorted(extra))}")
            return cls(value.get("tool"), value.get("input_match"))
        raise ValueError(f"`{what}` must be a tool name or {{tool, input_match}}")

    def matches(self, call: ToolCallRecord) -> bool:
        if call.name != self.tool:
            return False
        return self.rx is None or self.rx.search(input_json(call)) is not None

    def label(self) -> str:
        return f"{self.tool} /{self.input_match}/" if self.input_match else self.tool


def _bound(value: Any, default: float, what: str) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, str) and value.strip().lower() in ("inf", "infinity", ".inf", "∞", "unlimited"):
        return math.inf
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"`{what}` must be a number, got {value!r}") from None
    if f < 0:
        raise ValueError(f"`{what}` must be >= 0")
    return f


def _fmt(n: float) -> str:
    return "∞" if math.isinf(n) else str(int(n)) if n == int(n) else str(n)


def grade_tool_used(spec: GraderSpec, art: RunArtifacts) -> tuple[bool, str, str | None]:
    cfg = spec.config
    matcher = CallMatcher(cfg.get("tool"), cfg.get("input_match"))
    lo = _bound(cfg.get("min"), 1, "min")
    hi = _bound(cfg.get("max"), math.inf, "max")
    same_tool = [c for c in art.tool_calls if c.name == matcher.tool]
    matching = [c for c in same_tool if matcher.matches(c)]
    n = len(matching)
    passed = lo <= n <= hi
    explanation = f"{n} matching call(s), expected [{_fmt(lo)}, {_fmt(hi)}]"
    if matcher.input_match and len(same_tool) > n:
        explanation += f" ({len(same_tool) - n} other {matcher.tool} call(s) did not match input_match)"
    # evidence: vstupy shodných volání (nebo volání téhož toolu, když žádné nesedí)
    shown = matching or same_tool
    evidence = "\n".join(f"{c.name} {input_json(c)}" for c in shown[:5])[:EVIDENCE_CHARS] or None
    return passed, explanation, evidence
