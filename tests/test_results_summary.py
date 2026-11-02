"""Progress řádek a souhrnná tabulka."""

from __future__ import annotations

from pathlib import Path

from test_results_aggregate import api_err, build, g, make_case, run, two_arm_result

from skilleval.models import CaseResult
from skilleval.results.summary import case_notes, progress_line, summary_table


def test_progress_line_plain():
    r = run(0.8, cost=0.03, judge=0.006, graders=[g("01-tvrzeni", True), g("02-tvrzeni", False),
                                                    g("skill", True, scored=False)])
    line = progress_line("zapis-bez-dotazu", "with", 1, 3, r, color=False)
    assert line == ("zapis-bez-dotazu [with 1/3] 0.80 · 4 turns · 37s · $0.036 · "
                    "01-tvrzeni ✓ 02-tvrzeni ✗ (skill ✓)")
    assert "\x1b[" not in line


def test_progress_line_error_notices_color():
    r = run(0, error="timed out\nafter 300s", notices=["mocked: webstore"],
            aborted={"server": "webstore", "tool": "orders_get", "reason": "expect failed"})
    line = progress_line("c", "without", 2, 3, r, color=True)
    assert "\x1b[31m" in line and "\x1b[0m" in line
    plain = progress_line("c", "without", 2, 3, r, color=False)
    assert plain.endswith("· mocked: webstore · error: timed out after 300s")
    assert "aborted: webstore/orders_get: expect failed" in plain


def test_summary_table_two_arm():
    doc = build([two_arm_result()])
    out = summary_table(doc, report_path=Path("/r/report.html"))
    lines = out.splitlines()
    assert lines[0].split() == ["CASE", "WITH", "W/OUT", "Δ", "RUNS", "COST", "NOTES"]
    assert lines[0].startswith("CASE   WITH  W/OUT Δ      RUNS COST    NOTES")
    row = lines[1]
    assert row.startswith("zapis  0.73  0.13  +0.60  6    $0.66   ")
    assert "02-llm: 02-llm failed" in row  # nejtěžší neúspěšný scored grader (weight 2)
    assert lines[-2] == "1 case(s) · mean Δ +0.60 · 186s · $0.66"
    assert lines[-1] == "Report: /r/report.html"


def test_summary_table_layout_matches_cc_example():
    # CC: "CASE        WITH  W/OUT Δ      RUNS COST    NOTES" / "first-case  1.00  0.33  +0.67  6    $0.41"
    cr = CaseResult(case=make_case("first-case"), runs_per_case=3, with_runs=[run(1, cost=0.06, judge=0)] * 3,
                    without_runs=[run(1, cost=0.0766, judge=0), run(0, cost=0.0767, judge=0),
                                  run(0, cost=0.0767, judge=0)])
    lines = summary_table(build([cr]), report_path=None).splitlines()
    assert lines[0] == "CASE        WITH  W/OUT Δ      RUNS COST    NOTES"
    assert lines[1] == "first-case  1.00  0.33  +0.67  6    $0.41"
    assert not any(line.startswith("Report:") for line in lines)


def test_summary_table_missing_delta_dash():
    cr = CaseResult(case=make_case(), runs_per_case=1, with_runs=[run(1)], without_runs=[api_err()])
    lines = summary_table(build([cr]), report_path=None).splitlines()
    assert lines[1].split()[1:4] == ["1.00", "—", "—"]
    assert lines[-1].startswith("1 case(s) · mean Δ —")


def test_summary_table_single_arm_and_partial():
    cr = CaseResult(case=make_case(), runs_per_case=3, with_runs=[run(1), run(0.2), run(1)])
    doc = build([cr], ablation="none", partial=True, partial_reason="interrupted")
    lines = summary_table(doc, report_path=None).splitlines()
    assert lines[0].split() == ["CASE", "SCORE", "PASS%", "RUNS", "COST", "NOTES"]
    assert lines[1].split()[:5] == ["zapis", "0.73", "67%", "3", "$0.33"]
    assert lines[-2] == "1 case(s) · 0/1 passed · 186s · $0.33"
    assert lines[-1] == "Partial: interrupted"


def test_case_notes_error_and_truncation():
    cr = CaseResult(case=make_case(), runs_per_case=1, with_runs=[api_err()])
    case = build([cr])["cases"][0]
    assert case_notes(case) == "error: API error 429: rate limited"
    cr = CaseResult(case=make_case(), runs_per_case=1,
                    with_runs=[run(0, graders=[g("x", False, expl="slovo " * 50)])])
    note = case_notes(build([cr])["cases"][0])
    assert len(note) <= 70 and note.endswith("…")


def test_summary_table_empty():
    doc = build([], ablation="none")
    assert summary_table(doc, report_path=None) == "0 case(s) · 0/0 passed · 186s · $0.00"
