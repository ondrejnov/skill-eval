"""report.html – jeden soběstačný soubor (inline CSS/JS, žádné externí requesty) z aggregate-result.json.

Načte i výstup Claude Code (bez `runner`/`models`); trace se čte z `tracePath` (chybějící soubor → „not available“).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from skilleval.results.trace_view import TraceView, load_trace

REPORT_FILE = "report.html"
TEMPLATES = Path(__file__).parent / "templates"
_EPS = 1e-9

PARTIAL_REASONS = {
    "cost_ceiling": "The --max-cost-usd ceiling was hit; runs that had not started were skipped.",
    "interrupted": "The run was interrupted before every case finished.",
    "auth_failed": "The model credential was rejected before or at the first run.",
}


# ---------------------------------------------------------------------------------------------------------------------
# Formátování
# ---------------------------------------------------------------------------------------------------------------------


def pct(x: float | None) -> str:
    """0.1333 → '13.3%', 0.2 → '20%'."""
    if x is None:
        return "—"
    s = f"{float(x) * 100:.1f}"
    return (s[:-2] if s.endswith(".0") else s) + "%"


def pts(delta: float) -> str:
    v = float(delta) * 100
    if abs(v) < 0.05:
        return "→ 0"
    s = f"{abs(v):.1f}"
    s = s[:-2] if s.endswith(".0") else s
    return f"↑ +{s}" if v > 0 else f"↓ −{s}"


def delta_cls(delta: float | None) -> str:
    if delta is None:
        return "delta-zero"
    return "delta-pos" if delta > _EPS else ("delta-neg" if delta < -_EPS else "delta-zero")


def money(usd: float | None) -> str:
    usd = float(usd or 0)
    return f"${usd:.3f}" if 0 < usd < 0.01 else f"${usd:.2f}"


def duration(seconds: float | None) -> str:
    s = round(float(seconds or 0))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def when(value: str | None, fmt: str) -> str:
    dt = _parse_iso(value)
    return dt.strftime(fmt) + " UTC" if dt else (value or "")


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


# ---------------------------------------------------------------------------------------------------------------------
# View model
# ---------------------------------------------------------------------------------------------------------------------


def _is_api_error(run: dict[str, Any]) -> bool:
    return bool(run.get("error")) and not run.get("graders") and not run.get("aborted")


def _is_perfect(run: dict[str, Any]) -> bool:
    gs = run.get("graders") or []
    return bool(gs) and all(g.get("passed") for g in gs)


def _grader_vm(g: dict[str, Any]) -> dict[str, Any]:
    votes = g.get("judgeVotes")
    if votes is None and isinstance(g.get("votes"), list):
        votes = g["votes"]
    passed = bool(g.get("passed"))
    return {
        "name": g.get("name", "?"),
        "passed": passed,
        "scored": g.get("scored", True),
        "with_only": bool(g.get("withOnly")),
        "weight": g.get("weight", 1),
        "explanation": g.get("explanation") or "",
        "votes": [bool(v) for v in votes] if isinstance(votes, list) else None,
        "evidence": g.get("evidence"),
        "open": not passed,
    }


def _run_vm(run: dict[str, Any], idx: int, out_dir: Path | None) -> dict[str, Any]:
    meta = [_plural(int(run.get("turns") or 0), "turn"), money(run.get("costUsd"))]
    if run.get("judgeCostUsd"):
        meta[-1] += f" + {money(run.get('judgeCostUsd'))} judge"
    meta.append(duration(run.get("durationSeconds")))
    if run.get("startedAt"):
        meta.append(when(run["startedAt"], "%H:%M:%S"))
    aborted = run.get("aborted")
    aborted_text = None
    if aborted:
        where = "/".join(str(aborted[k]) for k in ("server", "tool") if aborted.get(k))
        aborted_text = (f"{where}: " if where else "") + str(aborted.get("reason", ""))
    trace: TraceView = load_trace(run.get("tracePath"), out_dir)
    return {
        "title": f"Run {idx}",
        "score": float(run.get("score") or 0),
        "passed": bool(run.get("passed")),
        "meta": " · ".join(meta),
        "error": run.get("error"),
        "api_error": _is_api_error(run),
        "aborted": aborted_text,
        "skipped_paid": bool(run.get("skippedPaidGraders")),
        "notices": list(run.get("notices") or []),
        "graders": [_grader_vm(g) for g in run.get("graders") or []],
        "trace": trace,
    }


def _arm_vm(label: str, cls: str, runs: list[dict[str, Any]], out_dir: Path | None) -> dict[str, Any]:
    score = _mean([float(r.get("score") or 0) for r in runs])
    perfect = _mean([1.0 if _is_perfect(r) else 0.0 for r in runs])
    return {"label": label, "cls": cls, "score": score or 0.0, "perfect": perfect or 0.0,
            "runs": [_run_vm(r, i, out_dir) for i, r in enumerate(runs, 1)]}


def _config_items(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    out = []
    for k, v in (cfg or {}).items():
        if k == "criteria":
            continue
        out.append((k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)))
    return out


def _extension_labels(ext: dict[str, Any] | None) -> list[str]:
    if not ext:
        return []
    labels = []
    if ext.get("mcpProfile"):
        labels.append(f"MCP profile: {ext['mcpProfile']}")
    if ext.get("faults"):
        labels.append(_plural(len(ext["faults"]), "fault"))
    if ext.get("setup") or ext.get("teardown"):
        labels.append("setup/teardown hooks")
    if ext.get("simulatedUser"):
        labels.append("simulated user")
    if ext.get("tagsExtra"):
        labels.append("tags: " + ", ".join(map(str, ext["tagsExtra"])))
    known = {"mcpProfile", "faults", "setup", "teardown", "simulatedUser", "tagsExtra"}
    labels += [str(k) for k in ext if k not in known]
    return labels


def _case_vm(case: dict[str, Any], idx: int, threshold: float, out_dir: Path | None) -> dict[str, Any]:
    agg = case.get("aggregates") or {}
    arms_src = case.get("arms") or {}
    arms = [_arm_vm("With plugin", "m-accent", arms_src.get("with") or [], out_dir)]
    if arms_src.get("without"):
        arms.append(_arm_vm("Baseline (no plugin)", "m-base", arms_src["without"], out_dir))
    delta = agg.get("delta")
    score = float(agg.get("score") or 0)
    return {
        "id": f"case-{idx}",
        "name": case.get("name", "?"),
        "dir": case.get("dir", ""),
        "score": score,
        "passed": score >= threshold - _EPS,
        "score_without": agg.get("scoreWithout"),
        "delta": delta,
        "regressed": delta is not None and delta < -_EPS,
        "single_arm_reason": case.get("singleArmReason"),
        "extensions": _extension_labels(case.get("extensions")),
        "extensions_json": json.dumps(case["extensions"], ensure_ascii=False, indent=2) if case.get("extensions")
        else None,
        "arms": arms,
        "prompt": case.get("promptMarkdown") or "",
        "graders": [{"name": g.get("name", "?"), "type": g.get("type", "?"), "weight": g.get("weight", 1),
                     "markdown": g.get("graderMarkdown") or (g.get("config") or {}).get("criteria") or "",
                     "config": _config_items(g.get("config") or {})}
                    for g in case.get("graders") or []],
        "meta": f"{case.get('runsPerCase', '?')} runs/arm · max {case.get('maxTurns', '?')} turns · "
                f"timeout {case.get('timeoutSeconds', '?')}s · source {case.get('source', '?')}",
    }


def build_view(aggregate: dict[str, Any], out_dir: Path | None) -> dict[str, Any]:
    suite = aggregate.get("suite") or {}
    totals = aggregate.get("aggregates") or {}
    threshold = float(suite.get("threshold", 1) if suite.get("threshold") is not None else 1)
    cases_src = aggregate.get("cases") or []
    cases = [_case_vm(c, i, threshold, out_dir) for i, c in enumerate(cases_src, 1)]

    with_runs = [r for c in cases_src for r in (c.get("arms") or {}).get("with") or []]
    all_runs = with_runs + [r for c in cases_src for r in (c.get("arms") or {}).get("without") or []]
    deltas = [c["delta"] for c in cases if c["delta"] is not None]
    two_arm = bool(deltas) or any(len(c["arms"]) > 1 for c in cases)
    mean_delta = totals.get("meanDelta", _mean(deltas))
    baseline = _mean([float(c["score_without"]) for c in cases if c["score_without"] is not None])
    overall = float(totals.get("overallScore") or 0)
    n_cases = int(totals.get("casesTotal", len(cases)))
    passed = int(totals.get("casesPassed", sum(1 for c in cases if c["passed"])))
    perfect = _mean([1.0 if _is_perfect(r) else 0.0 for r in with_runs])

    improved = sum(1 for d in deltas if d > _EPS)
    regressed = sum(1 for d in deltas if d < -_EPS)
    flat = len(deltas) - improved - regressed
    verdict: dict[str, Any] = {"two_arm": two_arm and mean_delta is not None}
    if verdict["two_arm"]:
        verdict.update(delta=pts(mean_delta) + " pts", cls=delta_cls(mean_delta),
                       rest=f"vs baseline, improved {improved}, flat {flat}, regressed {regressed} of "
                            f"{_plural(len(deltas), 'case')}")
    else:
        verdict.update(text=f"Suite score {pct(overall)} · {passed} of {_plural(n_cases, 'case')} "
                            f"≥ {pct(threshold)} threshold")

    tiles = [{"label": "Suite score · with plugin" if two_arm else "Suite score", "value": pct(overall),
              "sub": "mean of per-case scores", "hero": True, "cls": ""}]
    if two_arm:
        tiles.append({"label": "Ablation Δ", "value": pts(mean_delta) if mean_delta is not None else "—",
                      "sub": f"score points vs baseline, {len(deltas)} of {_plural(len(cases), 'case')}",
                      "cls": delta_cls(mean_delta), "hero": False})
        tiles.append({"label": "Baseline score", "value": pct(baseline), "sub": "without the plugin", "cls": "", "hero": False})
    tiles.append({"label": "Cases", "value": str(n_cases), "sub": f"{passed} of {n_cases} ≥ {pct(threshold)} threshold",
                  "cls": "", "hero": False})
    tiles.append({"label": "Perfect runs", "value": pct(perfect) if perfect is not None else "—",
                  "sub": "with-plugin runs where every grader passed", "cls": "", "hero": False})

    banners = []
    if aggregate.get("partial"):
        reason = aggregate.get("partialReason") or ""
        banners.append(("⚠ Partial results", PARTIAL_REASONS.get(reason, reason or "The suite did not finish.")))
    api_errors = sum(1 for r in all_runs if _is_api_error(r))
    errored = sum(1 for r in all_runs if r.get("error")) - api_errors
    if api_errors:
        banners.append(("⚠ API errors", f"{_plural(api_errors, 'run')} ended on an API error and scored 0 without "
                                        "grading."))
    if errored:
        banners.append(("⚠ Averages include non-judgments",
                        f"{_plural(errored, 'run')} errored — their scores count toward the averages shown."))
    if any(r.get("skippedPaidGraders") for r in all_runs):
        banners.append(("⚠ Judge graders skipped", "The cost ceiling skipped judge graders in some runs; "
                                                   "their scores are not comparable."))

    runner = aggregate.get("runner") or {}
    models = aggregate.get("models") or {}
    if runner:
        runner_label = f"{runner.get('name', 'skilleval')} v{runner.get('version', '?')}"
    elif aggregate.get("claudeVersion"):
        runner_label = f"Claude Code v{aggregate['claudeVersion']}"
    else:
        runner_label = ""
    meta = [m for m in [
        runner_label,
        f"agent {models['agent']}" if models.get("agent") else
        (f"model {suite['modelOverride']}" if suite.get("modelOverride") else ""),
        f"judge {models.get('judge') or suite.get('judgeModel') or 'default'}",
        when(aggregate.get("startedAt"), "%Y-%m-%d %H:%M"),
        duration(aggregate.get("durationSeconds")),
        money(aggregate.get("costUsd")),
        _plural(len(all_runs), "run"),
        f"ablation {suite.get('ablation', '?')}",
        f"threshold {pct(threshold)}",
        f"filtered: --case {suite['caseFilter']}" if suite.get("caseFilter") else "",
    ] if m]

    plugins = [p.get("name") for p in suite.get("plugins") or [] if p.get("name")]
    title = ", ".join(plugins) or Path(str(suite.get("root") or "suite")).name
    return {"title": title, "root": suite.get("root", ""), "meta": meta, "banners": banners, "verdict": verdict,
            "tiles": tiles, "cases": cases, "threshold": threshold, "two_arm": two_arm,
            "runner_label": runner_label or "skilleval", "schema": aggregate.get("schemaVersion", 1)}


# ---------------------------------------------------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------------------------------------------------


def _env() -> Environment:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True, undefined=StrictUndefined,
                      trim_blocks=True, lstrip_blocks=True)
    env.filters.update(pct=pct, pts=pts, delta_cls=delta_cls, money=money)
    env.globals["width"] = lambda x: f"{max(0.0, min(1.0, float(x or 0))) * 100:.1f}%"
    return env


def render_html(aggregate: dict[str, Any], out_dir: Path | None = None) -> str:
    return _env().get_template("report.html.j2").render(v=build_view(aggregate, out_dir))


def render_report(aggregate: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPORT_FILE
    path.write_text(render_html(aggregate, out_dir), encoding="utf-8")
    return path


__all__ = ["REPORT_FILE", "build_view", "render_html", "render_report"]
