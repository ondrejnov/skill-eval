"""`skilleval compare A.json B.json` – změny skóre a Δ po cases (regrese skillu/modelu)."""

from __future__ import annotations

from typing import Any

from skilleval.results.summary import one_line

EPS = 1e-9


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def _fmt_signed(v: float | None) -> str:
    return "—" if v is None else f"{v:+.2f}"


def _label(doc: dict[str, Any]) -> str:
    """Krátký popis běhu: čas, model, runner."""
    bits = [str(doc.get("startedAt") or "?")]
    models = doc.get("models") or {}
    model = models.get("agent") or (doc.get("suite") or {}).get("modelOverride")
    if model:
        bits.append(str(model))
    runner = doc.get("runner") or {}
    if runner:
        bits.append(f"{runner.get('name', 'skilleval')} {runner.get('version', '')}".strip())
    elif doc.get("claudeVersion"):
        bits.append(f"Claude Code {doc['claudeVersion']}")
    if doc.get("partial"):
        bits.append(f"PARTIAL ({doc.get('partialReason') or '?'})")
    return " · ".join(bits)


def _cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(c.get("name")): (c.get("aggregates") or {}) for c in doc.get("cases") or []}


def _change(a: float | None, b: float | None) -> tuple[str, float | None]:
    if a is None and b is None:
        return "—", None
    if a is None:
        return "added", None
    if b is None:
        return "removed", None
    d = b - a
    if d > EPS:
        return "improved", d
    if d < -EPS:
        return "regressed", d
    return "same", d


def compare(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Tabulka `CASE  A  B  CHANGE  ΔA  ΔB` + souhrn regresí/zlepšení (A = starší/referenční, B = nový)."""
    ca, cb = _cases(a), _cases(b)
    names = list(ca) + [n for n in cb if n not in ca]
    rows = [["CASE", "A", "B", "CHANGE", "Δ(A)", "Δ(B)", ""]]
    counts = {"improved": 0, "regressed": 0, "same": 0, "added": 0, "removed": 0}
    regressions: list[str] = []
    for n in names:
        sa = ca[n].get("score") if n in ca else None
        sb = cb[n].get("score") if n in cb else None
        kind, d = _change(sa, sb)
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "regressed":
            regressions.append(n)
        rows.append([one_line(n, 40), _fmt(sa), _fmt(sb), _fmt_signed(d) if d is not None else "",
                     _fmt_signed(ca.get(n, {}).get("delta")), _fmt_signed(cb.get(n, {}).get("delta")),
                     kind if kind != "same" else ""])

    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = [f"A: {_label(a)}", f"B: {_label(b)}", ""]
    for r in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)).rstrip())

    ta, tb = a.get("aggregates") or {}, b.get("aggregates") or {}
    oa, ob = ta.get("overallScore"), tb.get("overallScore")
    summary = [f"{len(names)} case(s)",
               f"improved {counts['improved']}", f"regressed {counts['regressed']}", f"same {counts['same']}"]
    if counts["added"] or counts["removed"]:
        summary += [f"added {counts['added']}", f"removed {counts['removed']}"]
    lines += ["", " · ".join(summary)]
    overall = f"overall score {_fmt(oa)} → {_fmt(ob)}"
    if oa is not None and ob is not None:
        overall += f" ({ob - oa:+.2f})"
    passed = f"cases passed {ta.get('casesPassed', '—')}/{ta.get('casesTotal', '—')} → " \
             f"{tb.get('casesPassed', '—')}/{tb.get('casesTotal', '—')}"
    parts = [overall, passed]
    if "meanDelta" in ta or "meanDelta" in tb:
        parts.append(f"mean Δ {_fmt_signed(ta.get('meanDelta'))} → {_fmt_signed(tb.get('meanDelta'))}")
    lines.append(" · ".join(parts))
    if regressions:
        lines.append("Regressions: " + ", ".join(regressions))
    return "\n".join(lines)


__all__ = ["compare"]
