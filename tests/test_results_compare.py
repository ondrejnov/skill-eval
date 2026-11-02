"""compare A.json B.json."""

from __future__ import annotations

import copy

from test_results_aggregate import build, two_arm_result

from skilleval.results.compare import compare


def test_compare_regression_and_improvement():
    a = build([two_arm_result()])
    b = copy.deepcopy(a)
    b["cases"][0]["aggregates"]["score"] = 0.5
    b["cases"][0]["aggregates"]["delta"] = 0.3
    b["aggregates"]["overallScore"] = 0.5
    extra = copy.deepcopy(a["cases"][0])
    extra["name"] = "novy"
    b["cases"].append(extra)
    out = compare(a, b)
    lines = out.splitlines()
    assert lines[0].startswith("A: 2026-11-01T07:59:25.294Z · anthropic/claude-haiku-5.5 · skilleval 0.1.0")
    header = next(line for line in lines if line.startswith("CASE"))
    assert header.split() == ["CASE", "A", "B", "CHANGE", "Δ(A)", "Δ(B)"]
    row = next(line for line in lines if line.startswith("zapis"))
    assert row.split() == ["zapis", "0.73", "0.50", "-0.23", "+0.60", "+0.30", "regressed"]
    assert next(line for line in lines if line.startswith("novy")).split()[-1] == "added"
    assert "2 case(s) · improved 0 · regressed 1 · same 0 · added 1 · removed 0" in out
    assert "overall score 0.73 → 0.50 (-0.23)" in out
    assert "mean Δ +0.60 → +0.60" in out
    assert lines[-1] == "Regressions: zapis"


def test_compare_same():
    a = build([two_arm_result()])
    out = compare(a, copy.deepcopy(a))
    assert "improved 0 · regressed 0 · same 1" in out
    assert "Regressions" not in out
