"""Grader `tool_order`: oba tooly volány a první shodné `before` volání předchází prvnímu shodnému `after`."""

from __future__ import annotations

from skilleval.grading.tool_used import CallMatcher, input_json
from skilleval.models import GraderSpec, RunArtifacts


def _first(matcher: CallMatcher, art: RunArtifacts) -> int | None:
    return next((i for i, c in enumerate(art.tool_calls) if matcher.matches(c)), None)


def grade_tool_order(spec: GraderSpec, art: RunArtifacts) -> tuple[bool, str, str | None]:
    cfg = spec.config
    if cfg.get("before") is None or cfg.get("after") is None:
        raise ValueError("tool_order grader requires `before` and `after`")
    before = CallMatcher.parse(cfg["before"], "before")
    after = CallMatcher.parse(cfg["after"], "after")
    ib, ia = _first(before, art), _first(after, art)
    missing = [m.label() for m, i in ((before, ib), (after, ia)) if i is None]
    if missing:
        return False, f"not called: {', '.join(missing)}", None
    assert ib is not None and ia is not None
    cb, ca = art.tool_calls[ib], art.tool_calls[ia]
    evidence = f"#{ib + 1} {cb.name} {input_json(cb)}\n#{ia + 1} {ca.name} {input_json(ca)}"[:400]
    if ib < ia:
        return True, f"first {before.label()} (call #{ib + 1}) precedes first {after.label()} (call #{ia + 1})", evidence
    return False, (f"first {after.label()} (call #{ia + 1}) does not come after first {before.label()} "
                   f"(call #{ib + 1})"), evidence
