"""aggregate-result.json – dokument kompatibilní s `claude plugin eval` (schemaVersion 1, camelCase).

Pořadí klíčů kopíruje Claude Code; místo `claudeVersion` je `runner` + `models`, u case navíc `extensions`
a `singleArmReason`. Celá čísla zapisujeme jako int (JS nerozlišuje 1 a 1.0 → CC píše `1`).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from skilleval.models import CaseResult, CaseSpec, GraderSpec, RunResult, Suite

SCHEMA_VERSION = 1
RUNNER_NAME = "skilleval"
RESULT_FILE = "aggregate-result.json"
_EPS = 1e-9


# ---------------------------------------------------------------------------------------------------------------------
# Run / case agregace
# ---------------------------------------------------------------------------------------------------------------------


def is_api_error_run(run: RunResult) -> bool:
    """Run skončil chybou API modelu (skóre 0 bez hodnocení). CC: error + žádné výsledky graderů."""
    if getattr(run, "api_error", False):  # případné budoucí pole RunResult
        return True
    return bool(run.error) and not run.graders and not run.aborted


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score(runs: list[RunResult]) -> float:
    return _mean([r.score for r in runs])


def _pass_rate(runs: list[RunResult]) -> float:
    return _mean([1.0 if r.passed else 0.0 for r in runs])


def case_aggregates(cr: CaseResult, threshold: float) -> dict[str, Any]:
    """{score, passRate[, scoreWithout, passRateWithout, delta]}.

    Skóre/passRate armu = průměr přes všechny jeho runy (API chyba = 0, jako CC). Without-arm a delta
    (= score − scoreWithout) jen když oba army mají aspoň jeden run bez API chyby (jinak nesrovnatelné)."""
    del threshold  # passRate vychází z run.passed (práh aplikoval runner)
    agg: dict[str, Any] = {"score": _score(cr.with_runs), "passRate": _pass_rate(cr.with_runs)}
    if cr.without_runs:
        w_ok = any(not is_api_error_run(r) for r in cr.with_runs)
        wo_ok = any(not is_api_error_run(r) for r in cr.without_runs)
        if w_ok and wo_ok:
            agg["scoreWithout"] = _score(cr.without_runs)
            agg["passRateWithout"] = _pass_rate(cr.without_runs)
            agg["delta"] = agg["score"] - agg["scoreWithout"]
    return agg


def case_passed(cr: CaseResult, threshold: float) -> bool:
    """Case prošel, když průměrné skóre with-armu ≥ práh."""
    return bool(cr.with_runs) and _score(cr.with_runs) >= threshold - _EPS


# ---------------------------------------------------------------------------------------------------------------------
# JSON části
# ---------------------------------------------------------------------------------------------------------------------


def grader_def_json(spec: GraderSpec) -> dict[str, Any]:
    """graders[] definice jako CC: config = volby z souboru; llm/baseline mají criteria první, focus jen když je."""
    cfg = {k: v for k, v in spec.config.items() if v is not None}
    if spec.type in ("llm", "baseline"):
        criteria = cfg.pop("criteria", None) or spec.markdown
        cfg = {"criteria": criteria, **cfg}
    if spec.arm:
        cfg["arm"] = spec.arm
    return {"name": spec.name, "type": spec.type, "weight": spec.weight, "graderMarkdown": spec.markdown,
            "config": cfg}


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest)


def _is_empty(v: Any) -> bool:
    return v is None or (isinstance(v, (list, dict, str)) and not v)


def _model_to_camel(m: BaseModel) -> dict[str, Any]:
    """Pydantic model → dict s camelCase klíči polí; vynechá prázdné hodnoty a výchozí hodnoty (bez šumu)."""
    out: dict[str, Any] = {}
    for fname, finfo in type(m).model_fields.items():
        v = getattr(m, fname)
        if _is_empty(v):
            continue
        if not finfo.is_required() and finfo.default is not None and v == finfo.default:
            continue
        if isinstance(v, BaseModel):
            v = _model_to_camel(v)
        elif isinstance(v, list):
            v = [_model_to_camel(x) if isinstance(x, BaseModel) else x for x in v]
        if not _is_empty(v):
            out[_camel(fname)] = v
    return out


def case_extensions(case: CaseSpec) -> dict[str, Any]:
    """Použitá rozšíření ze skilleval.yaml (prázdný dict = žádná)."""
    return _model_to_camel(case.extension) if case.extension else {}


def case_json(cr: CaseResult, threshold: float) -> dict[str, Any]:
    case = cr.case
    arms: dict[str, Any] = {"with": [r.to_json() for r in cr.with_runs]}
    if cr.without_runs:  # CC vynechá `without`, když žádný baseline run neproběhl
        arms["without"] = [r.to_json() for r in cr.without_runs]
    d: dict[str, Any] = {
        "name": case.name,
        "dir": case.rel_dir,
        "source": case.source,
        "promptMarkdown": case.prompt,
        "runsPerCase": cr.runs_per_case,
        "timeoutSeconds": case.timeout_seconds,
        "maxTurns": case.max_turns,
        "graders": [grader_def_json(g) for g in case.graders],
        "arms": arms,
        "aggregates": case_aggregates(cr, threshold),
    }
    ext = case_extensions(case)
    if ext:
        d["extensions"] = ext
    if cr.single_arm_reason:
        d["singleArmReason"] = cr.single_arm_reason
    return d


def suite_plugins(suite: Suite, case_results: list[CaseResult]) -> list[dict[str, str]]:
    """plugins[{name, path}] – plugin pod testem + další pluginy z cases (--plugin / pole `plugins`)."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(name: str, path: Path) -> None:
        key = str(Path(path).resolve()) if Path(path).is_absolute() else str(path)
        if key in seen:
            return
        seen.add(key)
        out.append({"name": name, "path": str(path)})

    if suite.plugin:
        add(suite.plugin.name, suite.plugin.root)
    for cr in case_results:
        for p in cr.case.plugins:
            add(Path(p).name, p)
    return out


def _js_numbers(obj: Any) -> Any:
    """Celočíselné floaty → int (JSON jako z JS: `1`, ne `1.0`)."""
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    if isinstance(obj, dict):
        return {k: _js_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_js_numbers(v) for v in obj]
    return obj


def build_aggregate(*, suite: Suite, case_results: list[CaseResult], started_at: str, duration_seconds: float,
                    threshold: float, ablation: str, concurrency: int, model_override: str | None,
                    case_filter: str | None, judge_model: str, agent_model: str, partial: bool = False,
                    partial_reason: str | None = None, runner_version: str = "0.1.0",
                    extra_suite: dict[str, Any] | None = None) -> dict[str, Any]:
    """Sestaví celý dokument aggregate-result.json."""
    cases = [case_json(cr, threshold) for cr in case_results]
    cost = sum(r.cost_usd + r.judge_cost_usd
               for cr in case_results for r in [*cr.with_runs, *(cr.without_runs or [])])

    suite_d: dict[str, Any] = {"root": str(suite.root), "ablation": ablation}
    if model_override:
        suite_d["modelOverride"] = model_override
    if judge_model:
        suite_d["judgeModel"] = judge_model
    if case_filter:
        suite_d["caseFilter"] = case_filter
    suite_d["threshold"] = threshold
    suite_d["concurrency"] = concurrency
    suite_d["plugins"] = suite_plugins(suite, case_results)
    for k, v in (extra_suite or {}).items():
        if v is not None:
            suite_d[k] = v

    aggs = [c["aggregates"] for c in cases]
    deltas = [a["delta"] for a in aggs if "delta" in a]
    totals: dict[str, Any] = {
        "casesTotal": len(cases),
        "casesPassed": sum(1 for cr in case_results if case_passed(cr, threshold)),
        "overallScore": _mean([a["score"] for a in aggs]),
        "overallPassRate": _mean([a["passRate"] for a in aggs]),
    }
    if deltas:
        totals["meanDelta"] = _mean(deltas)

    doc: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "runner": {"name": RUNNER_NAME, "version": runner_version},
        "models": {"agent": agent_model, "judge": judge_model},
        "startedAt": started_at,
        "durationSeconds": round(duration_seconds),
        "costUsd": cost,
        "partial": partial,
    }
    if partial and partial_reason:
        doc["partialReason"] = partial_reason
    doc["suite"] = suite_d
    doc["cases"] = cases
    doc["aggregates"] = totals
    return _js_numbers(doc)


# ---------------------------------------------------------------------------------------------------------------------
# Soubory
# ---------------------------------------------------------------------------------------------------------------------


def write_results(out_dir: Path, aggregate: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RESULT_FILE
    path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def timestamp_dirname(started: datetime) -> str:
    """2026-11-01T08-14-03-914Z (UTC, ms) – jako Claude Code."""
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    s = started.astimezone(UTC)
    return s.strftime("%Y-%m-%dT%H-%M-%S-") + f"{s.microsecond // 1000:03d}Z"


def default_results_dir(eval_dir: Path, started: datetime) -> Path:
    return eval_dir / "results" / timestamp_dirname(started)


def load_aggregate(path: Path) -> dict[str, Any]:
    """Načte aggregate-result.json (skilleval i Claude Code). `path` může být i adresář výsledků."""
    if path.is_dir():
        path = path / RESULT_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(f"{path}: není aggregate-result.json (chybí pole cases)")
    version = data.get("schemaVersion")
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path}: nepodporovaná schemaVersion {version!r} (očekávána {SCHEMA_VERSION})")
    return data


__all__ = ["RESULT_FILE", "SCHEMA_VERSION", "build_aggregate", "case_aggregates", "case_extensions", "case_json",
           "case_passed", "default_results_dir", "grader_def_json", "is_api_error_run", "load_aggregate",
           "timestamp_dirname", "write_results"]
