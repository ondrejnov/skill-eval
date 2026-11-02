"""aggregate-result.json: výpočty, tvar a shoda klíčů se vzorem Claude Code."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from skilleval.models import (
    CaseResult,
    CaseSpec,
    ExtensionSpec,
    FaultSpec,
    GraderResult,
    GraderSpec,
    PluginInfo,
    RunResult,
    Suite,
)
from skilleval.results.aggregate import (
    build_aggregate,
    case_aggregates,
    case_passed,
    default_results_dir,
    load_aggregate,
    write_results,
)

# Volitelné vzory z `claude plugin eval`: adresář evals/results; bez proměnné se testy přeskočí.
CC_RESULTS = Path(os.environ.get("SKILLEVAL_CC_RESULTS", "/nonexistent"))
CC_TWO_ARM = CC_RESULTS / "2026-11-01T07-59-25-294Z" / "aggregate-result.json"
CC_SINGLE = CC_RESULTS / "2026-11-01T08-14-03-914Z" / "aggregate-result.json"

# ---------------------------------------------------------------------------------------------------------------------
# Továrny (používají i ostatní test_results_*)
# ---------------------------------------------------------------------------------------------------------------------


def make_case(name: str = "zapis", tmp: Path = Path("/tmp/plug"), extension: ExtensionSpec | None = None) -> CaseSpec:
    return CaseSpec(
        name=name, dir=tmp / "evals" / name, rel_dir=f"evals/{name}", source="mixed",
        prompt="Přepni objednávku 71781 do stavu <b>„vyřizuje se“</b>.", max_turns=25, timeout_seconds=900,
        graders=[
            GraderSpec(name="01-llm", type="llm", markdown="Agent načte stav.", config={"criteria": "Agent načte stav.",
                                                                                         "focus": "trace"}),
            GraderSpec(name="02-llm", type="llm", weight=2, markdown="Agent shrne.",
                       config={"criteria": "Agent shrne.", "focus": None}),
            GraderSpec(name="skill", type="tool_used", config={"tool": "Skill"}, markdown=""),
        ],
        extension=extension,
    )


def g(name: str, passed: bool, *, scored: bool = True, weight: float = 1, votes: list[bool] | None = None,
      expl: str | None = None) -> GraderResult:
    return GraderResult(name=name, passed=passed, weight=weight, scored=scored,
                        explanation=expl or ("ok" if passed else f"{name} failed"), judge_votes=votes,
                        evidence="výřez" if votes is not None else None)


def run(score: float, *, passed: bool | None = None, error: str | None = None, graders: list[GraderResult] | None = None,
        cost: float = 0.1, judge: float = 0.01, trace: str | None = None, aborted: dict | None = None,
        notices: list[str] | None = None) -> RunResult:
    if graders is None:
        graders = [g("01-llm", score > 0, votes=[score > 0] * 3), g("02-llm", score >= 1, weight=2),
                   g("skill", True, scored=False)]
    return RunResult(score=score, passed=score >= 1 if passed is None else passed, turns=4, cost_usd=cost,
                     judge_cost_usd=judge, duration_seconds=37.4, started_at="2026-11-01T07:59:25.325Z", error=error,
                     trace_path=trace, graders=graders, aborted=aborted, notices=notices or [])


def api_err() -> RunResult:
    return run(0, error="API error 429: rate limited", graders=[])


def make_suite(tmp: Path = Path("/tmp/plug")) -> Suite:
    return Suite(root=tmp, eval_dir=tmp / "evals", plugin=PluginInfo(name="webstore-core", root=tmp))


def build(case_results: list[CaseResult], *, ablation: str = "with-without", **kw) -> dict:
    args = dict(suite=make_suite(), case_results=case_results, started_at="2026-11-01T07:59:25.294Z",
                duration_seconds=186.4, threshold=1.0, ablation=ablation, concurrency=1, model_override="haiku",
                case_filter="zapis", judge_model="anthropic/claude-haiku-5.5", agent_model="anthropic/claude-haiku-5.5")
    args.update(kw)
    return build_aggregate(**args)


def two_arm_result() -> CaseResult:
    return CaseResult(case=make_case(), runs_per_case=3, with_runs=[run(1), run(1), run(0.2)],
                      without_runs=[run(0.2), run(0), run(0.2)])


# ---------------------------------------------------------------------------------------------------------------------
# Výpočty
# ---------------------------------------------------------------------------------------------------------------------


def test_case_aggregates_two_arm():
    agg = case_aggregates(two_arm_result(), 1.0)
    assert list(agg) == ["score", "passRate", "scoreWithout", "passRateWithout", "delta"]
    assert agg["score"] == pytest.approx(2.2 / 3)
    assert agg["passRate"] == pytest.approx(2 / 3)
    assert agg["scoreWithout"] == pytest.approx(0.4 / 3)
    assert agg["passRateWithout"] == 0
    assert agg["delta"] == pytest.approx(2.2 / 3 - 0.4 / 3)


def test_case_aggregates_single_arm_and_api_errors():
    single = CaseResult(case=make_case(), runs_per_case=2, with_runs=[run(1), run(0)])
    assert case_aggregates(single, 1.0) == {"score": 0.5, "passRate": 0.5}
    # baseline jen s API chybami → bez delta
    cr = CaseResult(case=make_case(), runs_per_case=2, with_runs=[run(1), run(1)], without_runs=[api_err(), api_err()])
    assert "delta" not in case_aggregates(cr, 1.0)
    # API chyba v with-armu: skóre ji počítá jako 0, delta = score − scoreWithout
    cr = CaseResult(case=make_case(), runs_per_case=2, with_runs=[run(1), api_err()], without_runs=[run(0.2), run(0.2)])
    agg = case_aggregates(cr, 1.0)
    assert agg["score"] == 0.5
    assert agg["delta"] == pytest.approx(0.3)
    # celý with-arm API chyby → bez delta
    cr = CaseResult(case=make_case(), runs_per_case=1, with_runs=[api_err()], without_runs=[run(0.2)])
    assert "scoreWithout" not in case_aggregates(cr, 1.0)
    # timeout s gradery není API chyba
    cr = CaseResult(case=make_case(), runs_per_case=1, with_runs=[run(0.5, error="timed out after 300s")],
                    without_runs=[run(0)])
    assert case_aggregates(cr, 1.0)["delta"] == 0.5


def test_case_passed_threshold():
    cr = CaseResult(case=make_case(), runs_per_case=3, with_runs=[run(0.8), run(0.8), run(0.8)])
    assert case_passed(cr, 0.8)
    assert not case_passed(cr, 0.9)
    assert not case_passed(CaseResult(case=make_case(), runs_per_case=1), 0.0)


def test_build_aggregate_two_arm():
    doc = build([two_arm_result()])
    assert list(doc) == ["schemaVersion", "runner", "models", "startedAt", "durationSeconds", "costUsd", "partial",
                         "suite", "cases", "aggregates"]
    assert doc["schemaVersion"] == 1
    assert doc["runner"] == {"name": "skilleval", "version": "0.1.0"}
    assert doc["models"]["judge"] == "anthropic/claude-haiku-5.5"
    assert doc["durationSeconds"] == 186
    assert doc["costUsd"] == pytest.approx(6 * 0.11)
    assert doc["suite"] == {"root": "/tmp/plug", "ablation": "with-without", "modelOverride": "haiku",
                            "judgeModel": "anthropic/claude-haiku-5.5", "caseFilter": "zapis", "threshold": 1,
                            "concurrency": 1, "plugins": [{"name": "webstore-core", "path": "/tmp/plug"}]}
    c = doc["cases"][0]
    assert c["dir"] == "evals/zapis" and c["promptMarkdown"].startswith("Přepni")
    assert c["timeoutSeconds"] == 900 and c["maxTurns"] == 25 and c["runsPerCase"] == 3
    assert set(c["arms"]) == {"with", "without"}
    assert c["arms"]["with"][0]["durationSeconds"] == 37
    assert "extensions" not in c and "singleArmReason" not in c
    agg = doc["aggregates"]
    assert agg["casesTotal"] == 1 and agg["casesPassed"] == 0
    assert agg["overallScore"] == pytest.approx(2.2 / 3)
    assert agg["meanDelta"] == pytest.approx(0.6)
    # celočíselné floaty jako int (jako JS)
    assert isinstance(doc["suite"]["threshold"], int)
    assert isinstance(c["arms"]["with"][0]["score"], int)


def test_build_aggregate_grader_config_like_cc():
    graders = build([two_arm_result()])["cases"][0]["graders"]
    assert graders[0] == {"name": "01-llm", "type": "llm", "weight": 1, "graderMarkdown": "Agent načte stav.",
                          "config": {"criteria": "Agent načte stav.", "focus": "trace"}}
    assert graders[1]["config"] == {"criteria": "Agent shrne."}  # focus None vynechán
    assert graders[1]["weight"] == 2
    assert graders[2]["config"] == {"tool": "Skill"}


def test_build_aggregate_single_arm_partial_extensions():
    ext = ExtensionSpec(mcp_profile="limited", faults=[FaultSpec(tool="orders_update", call=1, error=True,
                                                                  when={"dry_run": False}, body="{}")],
                        tags_extra=["db-write"])
    cr = CaseResult(case=make_case(extension=ext), runs_per_case=1, with_runs=[run(1)],
                    single_arm_reason="history_file")
    doc = build([cr], ablation="none", partial=True, partial_reason="cost_ceiling", model_override=None,
                case_filter=None)
    assert doc["partial"] is True and doc["partialReason"] == "cost_ceiling"
    assert "modelOverride" not in doc["suite"] and "caseFilter" not in doc["suite"]
    c = doc["cases"][0]
    assert set(c["arms"]) == {"with"}
    assert set(c["aggregates"]) == {"score", "passRate"}
    assert "meanDelta" not in doc["aggregates"]
    assert c["singleArmReason"] == "history_file"
    assert c["extensions"] == {"mcpProfile": "limited", "faults": [{"tool": "orders_update", "call": 1,
                                                                     "when": {"dry_run": False}, "error": True,
                                                                     "body": "{}"}],
                               "tagsExtra": ["db-write"]}
    assert doc["aggregates"]["casesPassed"] == 1 and doc["aggregates"]["overallPassRate"] == 1


def test_partial_reason_omitted_when_not_partial():
    assert "partialReason" not in build([two_arm_result()], partial_reason="interrupted")


def test_extra_suite_merged():
    doc = build([two_arm_result()], extra_suite={"plugins": [{"name": "x", "path": "/x"}], "mocks": "off", "y": None})
    assert doc["suite"]["plugins"] == [{"name": "x", "path": "/x"}]
    assert doc["suite"]["mocks"] == "off" and "y" not in doc["suite"]


def test_case_plugins_added(tmp_path: Path):
    case = make_case()
    case.plugins = [Path("/tmp/plug"), tmp_path / "webstore-orders"]
    doc = build([CaseResult(case=case, runs_per_case=1, with_runs=[run(1)])])
    assert [p["name"] for p in doc["suite"]["plugins"]] == ["webstore-core", "webstore-orders"]


# ---------------------------------------------------------------------------------------------------------------------
# Soubory
# ---------------------------------------------------------------------------------------------------------------------


def test_write_and_load_roundtrip(tmp_path: Path):
    doc = build([two_arm_result()])
    path = write_results(tmp_path / "out", doc)
    assert path.name == "aggregate-result.json"
    text = path.read_text(encoding="utf-8")
    assert "Přepni" in text and text.startswith('{\n  "schemaVersion": 1')
    assert load_aggregate(path) == doc
    assert load_aggregate(tmp_path / "out") == doc
    (tmp_path / "bad.json").write_text('{"schemaVersion": 2, "cases": []}')
    with pytest.raises(ValueError):
        load_aggregate(tmp_path / "bad.json")


def test_default_results_dir():
    started = datetime(2026, 11, 1, 8, 14, 3, 914_000, tzinfo=UTC)
    assert default_results_dir(Path("/e"), started) == Path("/e/results/2026-11-01T08-14-03-914Z")


# ---------------------------------------------------------------------------------------------------------------------
# Shoda klíčů s výstupem Claude Code
# ---------------------------------------------------------------------------------------------------------------------

ALLOWED_EXTRA = {"runner", "models", "extensions", "notices", "singleArmReason", "judgeModel"}
ALLOWED_MISSING = {"claudeVersion"}


def _keysets(doc: dict) -> dict[str, set[str]]:
    case = doc["cases"][0]
    run_ = case["arms"]["with"][0]
    out = {"top": set(doc), "suite": set(doc["suite"]), "aggregates": set(doc["aggregates"]), "case": set(case),
           "caseAggregates": set(case["aggregates"]), "graderDef": set(case["graders"][0]), "run": set(run_),
           "plugin": set(doc["suite"]["plugins"][0])}
    gres = [gr for gr in run_["graders"] if "judgeVotes" in gr]
    plain = [gr for gr in run_["graders"] if "judgeVotes" not in gr]
    if gres:
        out["graderResultLlm"] = set(gres[0])
    if plain:
        out["graderResult"] = set(plain[0])
    return out


def _check_keys(ours: dict, cc: dict) -> None:
    a, b = _keysets(ours), _keysets(cc)
    for level in b:
        if level not in a:
            continue
        extra = a[level] - b[level] - ALLOWED_EXTRA
        missing = b[level] - a[level] - ALLOWED_MISSING
        assert not extra, f"{level}: navíc {extra}"
        assert not missing, f"{level}: chybí {missing}"


@pytest.mark.skipif(not CC_TWO_ARM.exists(), reason="vzor Claude Code není k dispozici")
def test_keys_match_claude_code_two_arm():
    cc = json.loads(CC_TWO_ARM.read_text(encoding="utf-8"))
    ours = build([two_arm_result()], partial=True, partial_reason="interrupted", model_override=None)
    _check_keys(ours, cc)
    # gradery bez hlasů soudce: klíče jako v CC (podmnožina klíčů llm výsledku)
    plain = next(gr for gr in ours["cases"][0]["arms"]["with"][0]["graders"] if "judgeVotes" not in gr)
    cc_llm = cc["cases"][0]["arms"]["with"][0]["graders"][0]
    assert set(plain) == set(cc_llm) - {"judgeVotes", "evidence"}


@pytest.mark.skipif(not CC_SINGLE.exists(), reason="vzor Claude Code není k dispozici")
def test_keys_match_claude_code_single_arm():
    cc = json.loads(CC_SINGLE.read_text(encoding="utf-8"))
    cr = CaseResult(case=make_case(), runs_per_case=1, with_runs=[run(0.2)])
    _check_keys(build([cr], ablation="none"), cc)
    assert load_aggregate(CC_SINGLE)["schemaVersion"] == 1
