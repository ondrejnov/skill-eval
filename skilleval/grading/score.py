"""Hodnocení runu: spuštění graderů, vyloučení ze skóre (ablace) a vážené skóre."""

from __future__ import annotations

import asyncio
from pathlib import Path

from skilleval.grading.baseline import grade_baseline
from skilleval.grading.file_exists import grade_file_exists
from skilleval.grading.llm import grade_llm
from skilleval.grading.regex import grade_regex
from skilleval.grading.tool_order import grade_tool_order
from skilleval.grading.tool_used import grade_tool_used
from skilleval.models import CaseSpec, GraderResult, GraderSpec, Judge, RunArtifacts

PAID_TYPES = frozenset({"llm", "baseline"})
SKIPPED_PAID = "skipped: cost ceiling"


def mocked_servers(case: CaseSpec) -> set[str]:
    """Servery s mocky = podadresáře `mocks/` (sada i case) obsahující aspoň jeden `.md` soubor."""
    out: set[str] = set()
    for d in case.mock_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for sub in d.iterdir():
            if sub.is_dir() and not sub.name.startswith(".") and any(sub.glob("*.md")):
                out.add(sub.name)
    return out


def excluded_graders(case: CaseSpec, *, two_arm: bool, plugin_servers: set[str]) -> set[str]:
    """Názvy graderů vyloučených ze skóre (scored=False) v obou armech dvou-armového běhu.

    Vyloučené: `tool_used` s `tool: Skill`; `regex target: mock_calls` a `llm focus: mock_calls`, když všechny
    mockované servery case patří pluginu; `arm: with-only`. `arm: both` nikdy. Když by byly vyloučené všechny,
    hodnotí se normálně. Jednoarmový běh (`--ablation none`) nevylučuje nic."""
    if not two_arm:
        return set()
    mock_calls_plugin_only = all(s in plugin_servers for s in mocked_servers(case))
    excluded: set[str] = set()
    for g in case.graders:
        if g.arm == "both":
            continue
        if (
            g.arm == "with-only"
            or (g.type == "tool_used" and g.config.get("tool") == "Skill")
            or (g.type == "regex" and g.config.get("target") == "mock_calls" and mock_calls_plugin_only)
            or (g.type == "llm" and g.config.get("focus") == "mock_calls" and mock_calls_plugin_only)
        ):
            excluded.add(g.name)
    if excluded and excluded >= {g.name for g in case.graders}:
        return set()
    return excluded


def run_score(results: list[GraderResult]) -> float:
    """Vážený podíl prošlých scored graderů; žádný scored → 0."""
    total = sum(r.weight for r in results if r.scored)
    if total <= 0:
        return 0.0
    return sum(r.weight for r in results if r.scored and r.passed) / total


async def _grade_one(case: CaseSpec, spec: GraderSpec, art: RunArtifacts, judge: Judge | None, *, scored: bool,
                     skip_paid: bool, votes: int) -> GraderResult:
    res = GraderResult(name=spec.name, passed=False, weight=spec.weight, explanation="",
                       with_only=spec.arm == "with-only", scored=scored)
    try:
        if spec.type in PAID_TYPES:
            if skip_paid:
                res.explanation = SKIPPED_PAID
                return res
            if judge is None:
                raise RuntimeError("no judge model configured")
            if spec.type == "llm":
                out = await grade_llm(spec, art, judge, votes=votes, task_prompt=case.prompt)
            else:
                out = await grade_baseline(case, spec, art, judge, votes=votes, task_prompt=case.prompt)
            res.passed, res.explanation, res.evidence, res.judge_votes, res.judge_cost_usd = out
            if not res.judge_votes:
                res.judge_votes = None
            return res
        fn = {"regex": grade_regex, "tool_used": grade_tool_used, "tool_order": grade_tool_order,
              "file_exists": grade_file_exists}.get(spec.type)
        if fn is None:
            raise ValueError(f"unknown grader type {spec.type!r}")
        res.passed, res.explanation, res.evidence = fn(spec, art)
    except Exception as ex:  # jako Claude Code: chyba graderu = fail s vysvětlením
        res.passed = False
        res.explanation = f"grader threw: {ex}"
        res.judge_cost_usd = float(getattr(ex, "judge_cost", 0.0) or 0.0)
    return res


async def grade_run(case: CaseSpec, art: RunArtifacts, judge: Judge | None, *, excluded: set[str],
                    skip_paid: bool = False, votes: int = 3) -> tuple[list[GraderResult], float, float]:
    """→ (výsledky v pořadí case.graders, skóre runu, náklady soudce).

    Run s API chybou nebo abortovaný mockem se nehodnotí: ([], 0.0, 0.0)."""
    if art.api_error or art.aborted:
        return [], 0.0, 0.0
    results = await asyncio.gather(*(
        _grade_one(case, g, art, judge, scored=g.name not in excluded, skip_paid=skip_paid, votes=votes)
        for g in case.graders
    ))
    results = list(results)
    return results, run_score(results), sum(r.judge_cost_usd for r in results)
