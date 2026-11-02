"""Textový výstup do terminálu: řádek progressu na run a souhrnná tabulka (jako `claude plugin eval`)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from skilleval.models import RunResult

NOTES_MAX = 70

_RESET = "\x1b[0m"
_COLORS = {"green": "\x1b[32m", "red": "\x1b[31m", "yellow": "\x1b[33m", "dim": "\x1b[2m", "bold": "\x1b[1m"}


def _c(text: str, color: str, enabled: bool) -> str:
    return f"{_COLORS[color]}{text}{_RESET}" if enabled else text


def one_line(text: str, limit: int) -> str:
    """Sloučí bílé znaky a zkrátí na `limit` znaků (…)."""
    s = re.sub(r"\s+", " ", text or "").strip()
    return s if len(s) <= limit else s[: max(limit - 1, 0)].rstrip() + "…"


def fmt_cost(usd: float, decimals: int = 2) -> str:
    return f"${usd:.{decimals}f}"


def fmt_signed(value: float) -> str:
    return f"{value:+.2f}"


# ---------------------------------------------------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------------------------------------------------


def progress_line(case: str, arm: str, idx: int, total: int, run: RunResult, *, color: bool) -> str:
    """`case [with 1/3] 0.80 · 5 turns · 43s · $0.036 · 01-a ✓ 02-b ✗ (skill ✓) · error: …`."""
    score_color = "green" if run.passed else ("red" if run.score == 0 else "yellow")
    parts = [
        f"{run.score:.2f}",
        f"{run.turns} turns",
        f"{round(run.duration_seconds)}s",
        fmt_cost(run.cost_usd + run.judge_cost_usd, 3),
    ]
    head = f"{case} {_c(f'[{arm} {idx}/{total}]', 'dim', color)} {_c(parts[0], score_color, color)}"
    segs = [head, *(_c(p, "dim", color) for p in parts[1:])]

    marks: list[str] = []
    for g in run.graders:
        mark = _c("✓", "green", color) if g.passed else _c("✗", "red", color)
        item = f"{g.name} {mark}"
        marks.append(item if g.scored else f"({item})")
    if marks:
        segs.append(" ".join(marks))
    if run.skipped_paid_graders:
        segs.append(_c("judge graders skipped (cost ceiling)", "yellow", color))
    if run.aborted:
        a = run.aborted
        where = "/".join(str(a[k]) for k in ("server", "tool") if a.get(k))
        segs.append(_c(f"aborted: {where}{': ' if where else ''}{a.get('reason', '')}", "red", color))
    for notice in run.notices:
        segs.append(_c(notice, "yellow", color))
    if run.error:
        segs.append(_c(f"error: {one_line(run.error, 200)}", "red", color))
    return " · ".join(segs)


# ---------------------------------------------------------------------------------------------------------------------
# Souhrnná tabulka
# ---------------------------------------------------------------------------------------------------------------------


def _all_runs(case: dict[str, Any]) -> list[dict[str, Any]]:
    arms = case.get("arms") or {}
    return [*(arms.get("with") or []), *(arms.get("without") or [])]


def case_cost(case: dict[str, Any]) -> float:
    return sum((r.get("costUsd") or 0) + (r.get("judgeCostUsd") or 0) for r in _all_runs(case))


def case_notes(case: dict[str, Any], limit: int = NOTES_MAX) -> str:
    """Vysvětlení nejtěžšího neúspěšného scored graderu z with-armu, jinak chyba/abort runu."""
    best: tuple[float, str] | None = None
    fallback = ""
    for run in (case.get("arms") or {}).get("with") or []:
        for g in run.get("graders") or []:
            if g.get("scored", True) and not g.get("passed"):
                w = float(g.get("weight", 1) or 0)
                if best is None or w > best[0]:
                    best = (w, f"{g.get('name')}: {g.get('explanation') or 'fail'}")
        if not fallback:
            if run.get("error"):
                fallback = f"error: {run['error']}"
            elif run.get("aborted"):
                fallback = f"aborted: {(run['aborted'] or {}).get('reason', '')}"
    if best is not None:
        return one_line(best[1], limit)
    if fallback:
        return one_line(fallback, limit)
    if case.get("singleArmReason"):
        return one_line(f"single-arm: {case['singleArmReason']}", limit)
    return ""


def is_two_arm(aggregate: dict[str, Any]) -> bool:
    for c in aggregate.get("cases") or []:
        if (c.get("arms") or {}).get("without") or "delta" in (c.get("aggregates") or {}):
            return True
    return False


def _fmt_score(v: Any) -> str:
    return "—" if v is None else f"{float(v):.2f}"


def _render(rows: list[list[str]]) -> list[str]:
    """Zarovnání sloupců (min. 2 mezery u prvního, 1 u ostatních); poslední sloupec bez paddingu."""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for r in rows:
        cells = []
        for i, cell in enumerate(r):
            if i == len(r) - 1:
                cells.append(cell)
            else:
                pad = widths[i] + (2 if i == 0 else 1)
                cells.append(cell.ljust(pad))
        out.append("".join(cells).rstrip())
    return out


def summary_table(aggregate: dict[str, Any], *, report_path: Path | None) -> str:
    """Tabulka cases + řádek souhrnu + `Report: <path>`."""
    cases = aggregate.get("cases") or []
    two_arm = is_two_arm(aggregate)
    if two_arm:
        rows = [["CASE", "WITH ", "W/OUT", "Δ     ", "RUNS", "COST   ", "NOTES"]]
    else:
        rows = [["CASE", "SCORE", "PASS%", "RUNS", "COST   ", "NOTES"]]
    for c in cases:
        a = c.get("aggregates") or {}
        runs = str(len(_all_runs(c)))
        cost = fmt_cost(case_cost(c))
        if two_arm:
            delta = fmt_signed(a["delta"]) if "delta" in a else "—"
            rows.append([c.get("name", "?"), _fmt_score(a.get("score")), _fmt_score(a.get("scoreWithout")), delta,
                         runs, cost, case_notes(c)])
        else:
            rows.append([c.get("name", "?"), _fmt_score(a.get("score")), f"{round(float(a.get('passRate', 0)) * 100)}%",
                         runs, cost, case_notes(c)])
    lines = _render(rows) if cases else []

    tot = aggregate.get("aggregates") or {}
    foot = [f"{len(cases)} case(s)"]
    if two_arm:
        foot.append(f"mean Δ {fmt_signed(tot['meanDelta'])}" if "meanDelta" in tot else "mean Δ —")
    else:
        foot.append(f"{tot.get('casesPassed', 0)}/{tot.get('casesTotal', len(cases))} passed")
    foot.append(f"{round(float(aggregate.get('durationSeconds') or 0))}s")
    foot.append(fmt_cost(float(aggregate.get("costUsd") or 0)))
    lines += ["", " · ".join(foot)]
    if aggregate.get("partial"):
        lines.append(f"Partial: {aggregate.get('partialReason') or 'unknown'}")
    if report_path is not None:
        lines.append(f"Report: {report_path}")
    return "\n".join(lines).lstrip("\n")


__all__ = ["case_cost", "case_notes", "fmt_cost", "is_two_arm", "one_line", "progress_line", "summary_table"]
