"""report.html: soběstačnost, escapování, prohlížeč trace, načtení výstupu Claude Code."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from test_results_aggregate import CC_SINGLE, CC_TWO_ARM, api_err, build, g, make_case, run, two_arm_result

from skilleval.models import CaseResult
from skilleval.results.aggregate import load_aggregate
from skilleval.results.report import render_report
from skilleval.results.trace_view import load_trace, parse_trace_lines, resolve_trace_path
from skilleval.trace import TraceWriter

EXTERNAL_RE = re.compile(r"""<(script|link|img|iframe|source|object|embed)\b[^>]*\b(src|href)\s*=\s*["']?(https?:)?//""",
                         re.I)


def assert_self_contained(html: str) -> None:
    assert not EXTERNAL_RE.search(html), EXTERNAL_RE.search(html)
    assert "@import" not in html
    assert not re.search(r"url\(\s*['\"]?(https?:)?//", html)
    assert "<script src" not in html


def write_trace(path: Path) -> str:
    tw = TraceWriter(cwd="/tmp/ws", model="anthropic/claude-haiku-5.5", tools=["Skill", "mcp__plugin_c_s__orders_get"],
                     mcp_servers=[{"name": "plugin_c_s", "status": "connected"}], skills=["c:core"], plugins=[])
    tw.user_prompt("Přepni objednávku <script>alert(1)</script>")
    tw.assistant(msg_id="m1", text="Načtu objednávku.", thinking="hmm",
                 tool_uses=[{"id": "t1", "name": "mcp__plugin_c_s__orders_get", "input": {"id": 71781}},
                            {"id": "t2", "name": "mcp__plugin_c_s__orders_update", "input": {"dry_run": False}}])
    tw.tool_result(tool_use_id="t1", content='{"status": 1}' + "x" * 3000, source="mock")
    tw.system("skilleval_fault", tool="orders_update", call=1)
    tw.tool_result(tool_use_id="t2", content='{"error":{"code":"conflict"}}', is_error=True, source="fault")
    tw.assistant(msg_id="m2", text="Hotovo.", tool_uses=[])
    tw.result(final_text="Hotovo.", turns=2, cost_usd=0.01, duration_ms=4200, error=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tw.text(), encoding="utf-8")
    return str(path)


def test_trace_parsing(tmp_path: Path):
    p = write_trace(tmp_path / "traces" / "zapis" / "with-1.jsonl")
    view = load_trace(p, tmp_path)
    assert view.available and view.tool_calls == 2
    kinds = [e.kind for e in view.events]
    assert kinds == ["init", "prompt", "thinking", "text", "tool", "tool", "skilleval", "text", "result"]
    t1, t2 = [e for e in view.events if e.kind == "tool"]
    assert t1.title == "orders_get" and t1.server == "plugin_c_s" and t1.source == "mock" and t1.highlight
    assert t1.output is not None and t1.output.full is not None  # dlouhý výstup → náhled + rozbalení
    assert t2.is_error and t2.source == "fault" and not t2.pending
    assert view.highlights == 3
    # nečitelný řádek a CC tool_result se seznamem bloků
    evs = parse_trace_lines(["not json", '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"a",'
                             '"name":"Read","input":{}}]}}', '{"type":"user","message":{"content":[{"type":'
                             '"tool_result","tool_use_id":"a","content":[{"type":"text","text":"obsah"}]}]}}'])
    assert evs[0].kind == "raw" and evs[1].output is not None and evs[1].output.preview == "obsah"


def test_trace_path_resolution(tmp_path: Path):
    p = write_trace(tmp_path / "traces" / "c" / "with-1.jsonl")
    assert resolve_trace_path(p, None) == Path(p)
    assert resolve_trace_path("traces/c/with-1.jsonl", tmp_path) == Path(p)
    # adresář výsledků přesunut → najde se podle …/traces/…
    assert resolve_trace_path("/old/place/traces/c/with-1.jsonl", tmp_path) == Path(p)
    assert resolve_trace_path("/tmp/claude-eval-XXXX/out/trace.jsonl", tmp_path) is None
    assert not load_trace(None, tmp_path).available


def test_render_report_own(tmp_path: Path):
    out = tmp_path / "results" / "2026-11-01T08-14-03-914Z"
    trace = write_trace(out / "traces" / "zapis" / "with-1.jsonl")
    with_runs = [run(1, trace=trace), run(0.2, graders=[g("01-llm", False, votes=[False, True, False],
                                                          expl="<b>chybí</b> orders_get"),
                                                        g("skill", True, scored=False)],
                                         notices=["mocked: plugin_c_s"]),
                 api_err()]
    regressed = CaseResult(case=make_case("regrese"), runs_per_case=1, with_runs=[run(0)], without_runs=[run(1)])
    doc = build([CaseResult(case=make_case(), runs_per_case=3, with_runs=with_runs,
                            without_runs=[run(0.2), run(0), run(0.2)]), two_arm_result(), regressed],
                partial=True, partial_reason="cost_ceiling")
    doc["cases"][1]["name"] = "druhy"
    path = render_report(doc, out)
    assert path == out / "report.html"
    html = path.read_text(encoding="utf-8")
    assert_self_contained(html)
    assert html.startswith("<!doctype html>")
    # verdikt + dlaždice
    assert "Plugin effect:" in html and "improved 2, flat 0, regressed 1 of 3 cases" in html
    for label in ("Suite score · with plugin", "Ablation Δ", "Baseline score", "Cases", "Perfect runs"):
        assert label in html
    assert "Partial results" in html and "API errors" in html
    # case karty
    assert html.count('class="case case-regressed"') == 1
    assert 'class="tick" style="left:100.0%"' in html
    # gradery: chip, hlasy, evidence, indikátor, escapování
    assert "plugin-fired indicator" in html
    assert "&lt;b&gt;chybí&lt;/b&gt; orders_get" in html and "<b>chybí</b>" not in html
    assert "Judge votes" in html and "Evidence" in html
    assert "mocked: plugin_c_s" in html
    # trace prohlížeč
    assert '<details class="trace">' in html
    assert "orders_get" in html and "src-fault" in html and "src-mock" in html and "ev-hl" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html and "<script>alert(1)" not in html
    assert "Trace not available" in html  # runy bez tracePath
    # prompt a gradery dole
    assert "Přepni objednávku 71781" in html and "&lt;b&gt;„vyřizuje se“&lt;/b&gt;" in html
    assert 'Graders — what "good" means' in html


def test_render_report_single_arm(tmp_path: Path):
    doc = build([CaseResult(case=make_case(), runs_per_case=1, with_runs=[run(1)], single_arm_reason="history_file")],
                ablation="none")
    html = render_report(doc, tmp_path).read_text(encoding="utf-8")
    assert_self_contained(html)
    assert "Plugin effect" not in html and "Ablation Δ" not in html
    assert "Suite score 100% · 1 of 1 case ≥ 100% threshold" in html
    assert "Single arm (no Δ): history_file" in html


def test_render_report_empty(tmp_path: Path):
    html = render_report(build([], ablation="none"), tmp_path).read_text(encoding="utf-8")
    assert "No cases." in html


@pytest.mark.parametrize("src", [CC_TWO_ARM, CC_SINGLE], ids=["two-arm", "single"])
def test_render_report_from_claude_code(src: Path, tmp_path: Path):
    if not src.exists():
        pytest.skip("vzor Claude Code není k dispozici")
    doc = load_aggregate(src)
    html = render_report(doc, tmp_path).read_text(encoding="utf-8")
    assert_self_contained(html)
    assert "Claude Code v" in html
    assert "Trace not available" in html  # tracePath do /tmp, který už neexistuje
    assert "zapis-bez-dotazu" in html
    if "meanDelta" in doc["aggregates"]:
        assert "Plugin effect:" in html and "Baseline (no plugin)" in html
