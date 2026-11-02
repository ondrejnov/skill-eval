"""Jednotlivé gradery: regex, tool_used, tool_order, file_exists, llm, baseline (FakeJudge, bez sítě)."""

from __future__ import annotations

import json

import pytest
from test_grading_helpers import MCP_ORDERS_GET, MCP_SET_STATUS, FakeJudge, g, make_art, make_case

from skilleval.grading import grade_run
from skilleval.grading.file_exists import glob_match
from skilleval.grading.llm import JUDGE_SYSTEM
from skilleval.models import McpCallRecord, ToolCallRecord


async def one(tmp_path, spec, art=None, judge=None, **kw):
    case = make_case(tmp_path, [spec])
    results, score, cost = await grade_run(case, art or make_art(tmp_path), judge, excluded=set(), **kw)
    return results[0], score, cost


# --- regex --------------------------------------------------------------------------------------------------------


async def test_regex_contains_default_last_message(tmp_path):
    r, score, _ = await one(tmp_path, g("r", "regex", pattern=r"71781"))
    assert r.passed and r.explanation == "pattern found" and score == 1.0
    assert "71781" in r.evidence
    r, score, _ = await one(tmp_path, g("r", "regex", pattern=r"faktura"))
    assert not r.passed and r.explanation == "pattern not found" and score == 0.0


async def test_regex_pattern_from_markdown_and_flags(tmp_path):
    r, *_ = await one(tmp_path, g("r", "regex", markdown="OBJEDNÁVKA\n", flags="i"))
    assert r.passed
    r, *_ = await one(tmp_path, g("r", "regex", pattern="OBJEDNÁVKA"))
    assert not r.passed


async def test_regex_not_contains(tmp_path):
    r, *_ = await one(tmp_path, g("r", "regex", pattern="storno", match="not_contains"))
    assert r.passed and r.explanation == "pattern not found"
    r, *_ = await one(tmp_path, g("r", "regex", pattern="71781", match="not_contains"))
    assert not r.passed and r.explanation == "pattern found"


async def test_regex_count(tmp_path):
    art = make_art(tmp_path, final="a1 a2 a3")
    r, *_ = await one(tmp_path, g("r", "regex", pattern=r"a\d", match="count:3"), art)
    assert r.passed and r.explanation == "found 3 matches, expected 3"
    r, *_ = await one(tmp_path, g("r", "regex", pattern=r"a\d", match="count:2"), art)
    assert not r.passed and r.explanation == "found 3 matches, expected 2"
    r, *_ = await one(tmp_path, g("r", "regex", pattern=r"a1", match="count:1"), art)
    assert r.passed and r.explanation == "found 1 match, expected 1"


async def test_regex_trace_json_escaping(tmp_path):
    # v trace jsou uvozovky escapované → regex musí matchnout \" (v JS patternu \\")
    r, *_ = await one(tmp_path, g("r", "regex", target="trace", pattern=r'stavu \\"vyřizuje se\\"'))
    assert r.passed
    r, *_ = await one(tmp_path, g("r", "regex", target="trace", pattern=r'stavu "vyřizuje se"'))
    assert not r.passed
    # vstup tool_use je v trace jako objekt (bez escapování)
    r, *_ = await one(tmp_path, g("r", "regex", target="trace", pattern=r'"name":"[^"]*orders_set_status".*"dry_run":false'))
    assert r.passed


async def test_regex_targets_files_file_mock_calls(tmp_path):
    art = make_art(tmp_path, files={"out/report.md": "Obrat: 1 234 Kč"},
                   mock_calls=[McpCallRecord(server="webstore", tool="orders_get", input={"id": 5}, output="{}",
                                             is_error=False, source="mock")])
    r, *_ = await one(tmp_path, g("r", "regex", target="files", pattern=r"^out/report\.md$", flags="m"), art)
    assert r.passed
    r, *_ = await one(tmp_path, g("r", "regex", target={"source": "file", "path": "out/report.md"},
                                  pattern=r"Obrat: [\d ]+ Kč"), art)
    assert r.passed
    r, *_ = await one(tmp_path, g("r", "regex", target="mock_calls", pattern=r'"tool":"orders_get","input":\{"id":5\}'),
                      art)
    assert r.passed


async def test_regex_file_target_missing_fails_with_explanation(tmp_path):
    r, *_ = await one(tmp_path, g("r", "regex", target={"source": "file", "path": "x.md"}, pattern="a"))
    assert not r.passed and r.explanation == "file not found in workspace: x.md"


async def test_regex_invalid_pattern_and_match_throw(tmp_path):
    r, *_ = await one(tmp_path, g("r", "regex", pattern="(?i)abc"))
    assert not r.passed and r.explanation.startswith("grader threw: invalid regex /(?i)abc/: inline flags")
    r, *_ = await one(tmp_path, g("r", "regex", pattern="a", match="count:x"))
    assert r.explanation.startswith("grader threw: invalid match")
    r, *_ = await one(tmp_path, g("r", "regex"))
    assert r.explanation == "grader threw: regex grader requires `pattern`"


async def test_regex_evidence_is_short_snippet(tmp_path):
    art = make_art(tmp_path, final="x" * 2000 + "NEEDLE" + "y" * 2000)
    r, *_ = await one(tmp_path, g("r", "regex", pattern="NEEDLE"), art)
    assert "NEEDLE" in r.evidence and len(r.evidence) <= 402


# --- tool_used ----------------------------------------------------------------------------------------------------


async def test_tool_used_defaults(tmp_path):
    r, *_ = await one(tmp_path, g("t", "tool_used", tool=MCP_ORDERS_GET))
    assert r.passed and r.explanation == "1 matching call(s), expected [1, ∞]"
    r, *_ = await one(tmp_path, g("t", "tool_used", tool="Read"))
    assert not r.passed and r.explanation == "0 matching call(s), expected [1, ∞]"


async def test_tool_used_input_match_json(tmp_path):
    r, *_ = await one(tmp_path, g("t", "tool_used", tool=MCP_SET_STATUS, input_match=r'"status":2,"dry_run":false'))
    assert r.passed
    assert '"note":"vyřízení"' in r.evidence  # ensure_ascii=False, kompaktní separátory
    r, *_ = await one(tmp_path, g("t", "tool_used", tool=MCP_SET_STATUS, input_match=r'"note":"vyřízení"'))
    assert r.passed
    r, *_ = await one(tmp_path, g("t", "tool_used", tool=MCP_SET_STATUS, input_match=r'"dry_run":true'))
    assert not r.passed and "did not match input_match" in r.explanation


async def test_tool_used_min_max(tmp_path):
    calls = [ToolCallRecord(id=str(i), name="Read", input={"file_path": f"/x/{i}"}) for i in range(3)]
    art = make_art(tmp_path, tool_calls=calls)
    r, *_ = await one(tmp_path, g("t", "tool_used", tool="Read", min=2, max=3), art)
    assert r.passed and r.explanation == "3 matching call(s), expected [2, 3]"
    r, *_ = await one(tmp_path, g("t", "tool_used", tool="Read", max=2), art)
    assert not r.passed
    # „nikdy nevoláno“
    r, *_ = await one(tmp_path, g("t", "tool_used", tool="Write", min=0, max=0), art)
    assert r.passed and r.explanation == "0 matching call(s), expected [0, 0]"
    r, *_ = await one(tmp_path, g("t", "tool_used", tool="Read", min=0, max=0), art)
    assert not r.passed


async def test_tool_used_exact_name_only(tmp_path):
    r, *_ = await one(tmp_path, g("t", "tool_used", tool="orders_get"))  # bez prefixu mcp__… nesedí
    assert not r.passed
    r, *_ = await one(tmp_path, g("t", "tool_used", tool="Skill", input_match='"skill":"webstore-core"'))
    assert r.passed


async def test_tool_used_bad_config(tmp_path):
    r, *_ = await one(tmp_path, g("t", "tool_used"))
    assert r.explanation.startswith("grader threw:")
    r, *_ = await one(tmp_path, g("t", "tool_used", tool="Read", min="many"))
    assert r.explanation == "grader threw: `min` must be a number, got 'many'"


# --- tool_order ---------------------------------------------------------------------------------------------------


async def test_tool_order_pass_and_fail(tmp_path):
    r, *_ = await one(tmp_path, g("o", "tool_order", before=MCP_ORDERS_GET, after=MCP_SET_STATUS))
    assert r.passed and "call #2" in r.explanation and "call #3" in r.explanation
    r, *_ = await one(tmp_path, g("o", "tool_order", before=MCP_SET_STATUS, after=MCP_ORDERS_GET))
    assert not r.passed and "does not come after" in r.explanation


async def test_tool_order_not_called(tmp_path):
    r, *_ = await one(tmp_path, g("o", "tool_order", before="Read", after=MCP_SET_STATUS))
    assert not r.passed and r.explanation == "not called: Read"
    r, *_ = await one(tmp_path, g("o", "tool_order", before="Read", after="Write"))
    assert r.explanation == "not called: Read, Write"


async def test_tool_order_with_input_match(tmp_path):
    calls = [
        ToolCallRecord(id="1", name=MCP_SET_STATUS, input={"id": 1, "dry_run": True}),
        ToolCallRecord(id="2", name=MCP_ORDERS_GET, input={"id": 1}),
        ToolCallRecord(id="3", name=MCP_SET_STATUS, input={"id": 1, "dry_run": False}),
    ]
    art = make_art(tmp_path, tool_calls=calls)
    spec = g("o", "tool_order", before=MCP_ORDERS_GET, after={"tool": MCP_SET_STATUS, "input_match": '"dry_run":false'})
    r, *_ = await one(tmp_path, spec, art)
    assert r.passed
    r, *_ = await one(tmp_path, g("o", "tool_order", before=MCP_ORDERS_GET, after=MCP_SET_STATUS), art)
    assert not r.passed  # první set_status (dry_run) je před orders_get
    r, *_ = await one(tmp_path, g("o", "tool_order", before={"tool": "X", "bogus": 1}, after="Y"), art)
    assert r.explanation.startswith("grader threw: `before` has unknown key")
    r, *_ = await one(tmp_path, g("o", "tool_order", before="X"), art)
    assert r.explanation == "grader threw: tool_order grader requires `before` and `after`"


# --- file_exists --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pattern,path,ok", [
    ("report.md", "report.md", True),
    ("./report.md", "report.md", True),
    ("*.md", "report.md", True),
    ("*.md", "out/report.md", False),
    ("**/*.md", "report.md", True),
    ("**/*.md", "out/sub/report.md", True),
    ("out/**", "out/a/b.txt", True),
    ("out/*.{md,txt}", "out/a.txt", True),
    ("out/*.{md,txt}", "out/a.csv", False),
    ("out/?.md", "out/a.md", True),
    ("out/[ab].md", "out/c.md", False),
    ("out/[!ab].md", "out/c.md", True),
])
def test_glob_match(pattern, path, ok):
    assert glob_match(pattern, path) is ok


async def test_file_exists(tmp_path):
    art = make_art(tmp_path, created=["out/report.md", "notes.txt"])
    r, *_ = await one(tmp_path, g("f", "file_exists", path="out/*.md"), art)
    assert r.passed and "out/report.md" in r.explanation
    r, *_ = await one(tmp_path, g("f", "file_exists", path="*.pdf"), art)
    assert not r.passed and r.explanation == "no file matching *.pdf was created"
    r, *_ = await one(tmp_path, g("f", "file_exists", path="*.pdf", exists=False), art)
    assert r.passed
    r, *_ = await one(tmp_path, g("f", "file_exists", path="**/*.md", exists=False), art)
    assert not r.passed and "unexpected" in r.explanation
    r, *_ = await one(tmp_path, g("f", "file_exists", path=str(art.workspace / "notes.txt")), art)
    assert r.passed


async def test_file_exists_only_created_files(tmp_path):
    # soubor existuje ve workspace (scaffold/upravený), ale není ve files_created → nepočítá se
    art = make_art(tmp_path, files={"pre.md": "x"}, created=[])
    r, *_ = await one(tmp_path, g("f", "file_exists", path="pre.md"), art)
    assert not r.passed


# --- llm ----------------------------------------------------------------------------------------------------------


async def test_llm_three_votes_majority(tmp_path):
    judge = FakeJudge([True, True, False], cost=0.002)
    r, score, cost = await one(tmp_path, g("l", "llm", markdown="Agent uvede ref.url.\n"), judge=judge)
    assert r.passed and score == 1.0
    assert r.explanation.startswith("judge votes: PASS PASS FAIL — vote 2 failed")
    assert r.judge_votes == [True, True, False]
    assert cost == pytest.approx(0.006) and r.judge_cost_usd == pytest.approx(0.006)
    assert len(judge.calls) == 3
    call = judge.calls[0]
    assert call["system"] == JUDGE_SYSTEM
    assert "Agent uvede ref.url." in call["user"] and "final response" in call["user"]
    assert "Přepni objednávku 71781" in call["user"]  # zadání jako kontext
    assert r.evidence == 'Objednávka 71781 je ve stavu "vyřizuje se".'
    assert r.to_json()["judgeVotes"] == [True, True, False]


async def test_llm_all_pass_explanation_exact(tmp_path):
    r, *_ = await one(tmp_path, g("l", "llm", criteria="ok"), judge=FakeJudge([True]))
    assert r.explanation == "judge votes: PASS PASS PASS"


async def test_llm_fail_majority(tmp_path):
    r, score, _ = await one(tmp_path, g("l", "llm", criteria="ok"), judge=FakeJudge([False, True, False]))
    assert not r.passed and score == 0.0
    assert r.explanation.startswith("judge votes: FAIL PASS FAIL — vote 0 failed")


async def test_llm_dissent_truncated(tmp_path):
    judge = FakeJudge([False], explanation="x" * 1000)
    r, *_ = await one(tmp_path, g("l", "llm", criteria="ok"), judge=judge)
    assert len(r.explanation) < 400 and r.explanation.endswith("…")


async def test_llm_focus_trace_head_tail_and_evidence_cap(tmp_path):
    lines = [json.dumps({"type": "user", "n": i, "pad": "p" * 1000}) for i in range(40)]
    art = make_art(tmp_path, trace_lines=lines)
    judge = FakeJudge([True])
    r, *_ = await one(tmp_path, g("l", "llm", criteria="ok", focus="trace"), art, judge)
    assert "[…16 messages elided…]" in judge.calls[0]["user"]
    assert '"n": 12,' not in judge.calls[0]["user"] and '"n": 28,' in judge.calls[0]["user"]
    assert len(r.evidence) <= 20_001 and r.evidence.startswith(lines[0])


async def test_llm_focus_image_file(tmp_path):
    art = make_art(tmp_path, files={"chart.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 8})
    judge = FakeJudge([True])
    r, *_ = await one(tmp_path, g("l", "llm", criteria="Graf má osu.", focus={"source": "file", "path": "chart.png"}),
                      art, judge)
    assert r.passed and judge.calls[0]["images"][0]["media_type"] == "image/png"


async def test_llm_binary_focus_fails_without_judge_call(tmp_path):
    art = make_art(tmp_path, files={"deck.pptx": b"PK\x03\x04\x00binary"})
    judge = FakeJudge([True])
    r, _, cost = await one(tmp_path, g("l", "llm", criteria="x", focus={"source": "file", "path": "deck.pptx"}),
                           art, judge)
    assert not r.passed and "binary" in r.explanation and judge.calls == [] and cost == 0


async def test_llm_judge_error_is_grader_threw(tmp_path):
    judge = FakeJudge([True], fail=RuntimeError("Request was aborted."), fail_on=1)
    r, _, cost = await one(tmp_path, g("l", "llm", criteria="x"), judge=judge)
    assert not r.passed and r.explanation == "grader threw: Request was aborted."
    assert cost == pytest.approx(0.002)  # náklady úspěšných hlasů se započítají


async def test_llm_without_criteria_or_judge(tmp_path):
    r, *_ = await one(tmp_path, g("l", "llm"), judge=FakeJudge())
    assert r.explanation == "grader threw: llm grader requires `criteria` (file body)"
    r, *_ = await one(tmp_path, g("l", "llm", criteria="x"), judge=None)
    assert r.explanation == "grader threw: no judge model configured"


async def test_llm_votes_param(tmp_path):
    judge = FakeJudge([True])
    r, *_ = await one(tmp_path, g("l", "llm", criteria="x"), judge=judge, votes=1)
    assert r.judge_votes == [True] and len(judge.calls) == 1


# --- baseline -----------------------------------------------------------------------------------------------------


async def test_baseline(tmp_path):
    case = make_case(tmp_path, [g("b", "baseline", baseline_file="ref.jsonl", markdown="Shrne výsledek.")])
    ref = [json.dumps({"type": "assistant", "i": i}) for i in range(30)]
    (case.dir / "ref.jsonl").write_text("\n".join(ref) + "\n\n", encoding="utf-8")
    judge = FakeJudge([True, False, True])
    results, score, cost = await grade_run(case, make_art(tmp_path), judge, excluded=set())
    r = results[0]
    assert r.passed and score == 1.0 and cost == pytest.approx(0.003)
    assert r.explanation.startswith("judge votes: PASS FAIL PASS — ")
    user = judge.calls[0]["user"]
    assert "Shrne výsledek." in user and "[…6 messages elided…]" in user and "<reference>" in user
    assert "orders_set_status" in user  # trace runu
    assert r.evidence.startswith('{"type":"system","subtype":"init"')


async def test_baseline_errors(tmp_path):
    judge = FakeJudge([True])
    for cfg, msg in [({}, "requires `baseline_file`"), ({"baseline_file": "ref.txt"}, "must be a .jsonl"),
                     ({"baseline_file": "../x.jsonl"}, "inside the case directory")]:
        case = make_case(tmp_path, [g("b", "baseline", criteria="x", **cfg)])
        results, *_ = await grade_run(case, make_art(tmp_path), judge, excluded=set())
        assert results[0].explanation.startswith("grader threw:") and msg in results[0].explanation
    case = make_case(tmp_path, [g("b", "baseline", criteria="x", baseline_file="missing.jsonl")])
    results, *_ = await grade_run(case, make_art(tmp_path), judge, excluded=set())
    assert results[0].explanation == "baseline_file not found: missing.jsonl"
    assert judge.calls == []
