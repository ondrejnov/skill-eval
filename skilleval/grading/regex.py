"""Grader `regex`: JS regex nad targetem – contains (default) | not_contains | count:N."""

from __future__ import annotations

import re

from skilleval.grading.regexjs import compile_js_regex, js_finditer
from skilleval.grading.targets import TargetError, target_text
from skilleval.models import GraderSpec, RunArtifacts

EVIDENCE_CHARS = 400
_COUNT_RE = re.compile(r"count:\s*(\d+)")


def parse_match(match: object) -> tuple[str, int | None]:
    """`match` → (mode, N). Neplatná hodnota = ValueError."""
    m = str(match or "contains").strip()
    if m in ("contains", "not_contains"):
        return m, None
    cm = _COUNT_RE.fullmatch(m)
    if cm:
        return "count", int(cm.group(1))
    raise ValueError(f"invalid match {match!r} (contains | not_contains | count:N)")


def snippet(text: str, start: int, end: int, limit: int = EVIDENCE_CHARS) -> str:
    """Výřez kolem shody (max `limit` znaků)."""
    if end - start >= limit:
        return text[start:start + limit] + "…"
    pad = (limit - (end - start)) // 2
    a, b = max(0, start - pad), min(len(text), end + pad)
    return ("…" if a > 0 else "") + text[a:b] + ("…" if b < len(text) else "")


def pattern_of(spec: GraderSpec) -> str:
    pattern = spec.config.get("pattern")
    if pattern is None:
        pattern = spec.markdown.strip()
    if not isinstance(pattern, str) or pattern == "":
        raise ValueError("regex grader requires `pattern`")
    return pattern


def grade_regex(spec: GraderSpec, art: RunArtifacts) -> tuple[bool, str, str | None]:
    """→ (passed, explanation, evidence)."""
    cfg = spec.config
    rx = compile_js_regex(pattern_of(spec), str(cfg.get("flags") or ""))
    mode, expected = parse_match(cfg.get("match"))
    target = cfg.get("target")
    try:
        text, _ = target_text(target, art, for_llm=False)
    except TargetError as ex:
        return False, str(ex), None

    matches = list(js_finditer(rx, text)) if mode == "count" else []
    first = matches[0] if matches else (rx.search(text) if mode != "count" else None)
    evidence = snippet(text, first.start(), first.end()) if first else None
    if evidence is None and text and target != "trace":
        evidence = text[:EVIDENCE_CHARS] + ("…" if len(text) > EVIDENCE_CHARS else "")

    if mode == "contains":
        return (True, "pattern found", evidence) if first else (False, "pattern not found", evidence)
    if mode == "not_contains":
        return (False, "pattern found", evidence) if first else (True, "pattern not found", evidence)
    n = len(matches)
    word = "match" if n == 1 else "matches"
    return n == expected, f"found {n} {word}, expected {expected}", evidence
