"""Skóre runu, vyloučení graderů ze skóre (ablace), api_error/aborted, skip_paid, výjimky."""

from __future__ import annotations

import pytest
from test_grading_helpers import MCP_ORDERS_GET, FakeJudge, g, make_art, make_case

from skilleval.grading import SKIPPED_PAID, excluded_graders, grade_run, mocked_servers

PLUGIN = {"webstore"}


def test_excluded_skill_tool_used(tmp_path):
    case = make_case(tmp_path, [g("skill", "tool_used", tool="Skill"), g("ans", "regex", pattern="x"),
                                g("read", "tool_used", tool="Read")])
    assert excluded_graders(case, two_arm=True, plugin_servers=PLUGIN) == {"skill"}


def test_excluded_with_only_and_both(tmp_path):
    case = make_case(tmp_path, [g("a", "regex", pattern="x", arm="with-only"),
                                g("skill0", "tool_used", tool="Skill", min=0, max=0, arm="both"),
                                g("c", "regex", pattern="y")])
    assert excluded_graders(case, two_arm=True, plugin_servers=PLUGIN) == {"a"}


def test_excluded_mock_calls_when_all_mocked_servers_belong_to_plugin(tmp_path):
    graders = [g("rx", "regex", target="mock_calls", pattern="x"),
               g("judge", "llm", focus="mock_calls", criteria="x"),
               g("trace", "regex", target="trace", pattern="x"),
               g("last", "llm", criteria="x")]
    case = make_case(tmp_path, graders, mocks={"webstore": ["orders_get.md"]})
    assert mocked_servers(case) == {"webstore"}
    assert excluded_graders(case, two_arm=True, plugin_servers=PLUGIN) == {"rx", "judge"}


def test_not_excluded_when_some_mocked_server_is_foreign(tmp_path):
    graders = [g("rx", "regex", target="mock_calls", pattern="x"), g("last", "llm", criteria="x")]
    case = make_case(tmp_path, graders, mocks={"webstore": ["orders_get.md"], "github": ["_server.md"]})
    assert mocked_servers(case) == {"webstore", "github"}
    assert excluded_graders(case, two_arm=True, plugin_servers=PLUGIN) == set()


def test_mocked_servers_ignores_hidden_and_empty_dirs(tmp_path):
    case = make_case(tmp_path, [g("a", "regex", pattern="x")], mocks={"webstore": ["a.md"]})
    md = case.mock_dirs[0]
    (md / ".replay").mkdir()
    (md / ".replay" / "x.md").write_text("x")
    (md / "empty").mkdir()
    (md / "jsononly").mkdir()
    (md / "jsononly" / "_tools.json").write_text("[]")
    assert mocked_servers(case) == {"webstore"}


def test_suite_level_mock_dir(tmp_path):
    suite_mocks = tmp_path / "evals" / "mocks" / "github"
    suite_mocks.mkdir(parents=True)
    (suite_mocks / "x.md").write_text("ok")
    case = make_case(tmp_path, [g("rx", "regex", target="mock_calls", pattern="x"), g("b", "regex", pattern="y")],
                     mocks={"webstore": ["a.md"]})
    case.mock_dirs.insert(0, tmp_path / "evals" / "mocks")
    assert excluded_graders(case, two_arm=True, plugin_servers=PLUGIN) == set()
    assert excluded_graders(case, two_arm=True, plugin_servers={"webstore", "github"}) == {"rx"}


def test_all_excluded_means_none(tmp_path):
    case = make_case(tmp_path, [g("skill", "tool_used", tool="Skill"), g("w", "regex", pattern="x", arm="with-only")])
    assert excluded_graders(case, two_arm=True, plugin_servers=PLUGIN) == set()


def test_ablation_none_excludes_nothing(tmp_path):
    case = make_case(tmp_path, [g("skill", "tool_used", tool="Skill"), g("a", "regex", pattern="x")])
    assert excluded_graders(case, two_arm=False, plugin_servers=PLUGIN) == set()


async def test_weighted_score_and_scored_flags(tmp_path):
    graders = [
        g("skill", "tool_used", tool="Skill"),  # pass, vyloučený
        g("get", "tool_used", tool=MCP_ORDERS_GET, weight=3),  # pass
        g("faktura", "regex", pattern="faktura", weight=1),  # fail
        g("wo", "regex", pattern="nic", arm="with-only"),  # fail, vyloučený
    ]
    case = make_case(tmp_path, graders)
    excluded = excluded_graders(case, two_arm=True, plugin_servers=PLUGIN)
    assert excluded == {"skill", "wo"}
    results, score, cost = await grade_run(case, make_art(tmp_path), None, excluded=excluded)
    assert [r.name for r in results] == ["skill", "get", "faktura", "wo"]
    assert [r.scored for r in results] == [False, True, True, False]
    assert [r.with_only for r in results] == [False, False, False, True]
    assert score == pytest.approx(0.75) and cost == 0.0
    j = results[0].to_json()
    assert j == {"name": "skill", "passed": True, "weight": 1, "explanation": "1 matching call(s), expected [1, ∞]",
                 "withOnly": False, "scored": False, "evidence": 'Skill {"skill":"webstore-core"}'}


async def test_no_scored_graders_score_zero(tmp_path):
    case = make_case(tmp_path, [g("a", "regex", pattern="71781")])
    results, score, _ = await grade_run(case, make_art(tmp_path), None, excluded={"a"})
    assert results[0].passed and not results[0].scored and score == 0.0


async def test_api_error_and_aborted_not_graded(tmp_path):
    judge = FakeJudge([True])
    case = make_case(tmp_path, [g("a", "regex", pattern="71781"), g("l", "llm", criteria="x")])
    assert await grade_run(case, make_art(tmp_path, api_error=True), judge, excluded=set()) == ([], 0.0, 0.0)
    aborted = {"server": "webstore", "tool": "orders_get", "reason": "expect id: number"}
    assert await grade_run(case, make_art(tmp_path, aborted=aborted), judge, excluded=set()) == ([], 0.0, 0.0)
    assert judge.calls == []


async def test_run_error_is_still_graded(tmp_path):
    case = make_case(tmp_path, [g("a", "regex", pattern="71781")])
    results, score, _ = await grade_run(case, make_art(tmp_path, error="max turns reached (10)"), None, excluded=set())
    assert score == 1.0 and results[0].passed


async def test_skip_paid(tmp_path):
    judge = FakeJudge([True])
    graders = [g("a", "regex", pattern="71781"), g("l", "llm", criteria="x"),
               g("b", "baseline", baseline_file="r.jsonl", criteria="x"), g("w", "llm", criteria="y", arm="with-only")]
    case = make_case(tmp_path, graders)
    results, score, cost = await grade_run(case, make_art(tmp_path), judge, excluded={"w"}, skip_paid=True)
    assert judge.calls == [] and cost == 0.0
    assert [(r.passed, r.explanation, r.scored) for r in results[1:]] == [
        (False, SKIPPED_PAID, True), (False, SKIPPED_PAID, True), (False, SKIPPED_PAID, False)]
    assert results[0].passed and score == pytest.approx(1 / 3)


async def test_grader_exception_does_not_break_others(tmp_path):
    judge = FakeJudge([True], fail=ValueError("boom"))
    graders = [g("bad", "regex", pattern="(?i)x"), g("l", "llm", criteria="x"), g("ok", "regex", pattern="71781"),
               g("ft", "file_exists")]
    case = make_case(tmp_path, graders)
    results, score, _ = await grade_run(case, make_art(tmp_path), judge, excluded=set())
    assert [r.passed for r in results] == [False, False, True, False]
    assert results[1].explanation == "grader threw: boom"
    assert results[3].explanation == "grader threw: file_exists grader requires `path`"
    assert score == pytest.approx(0.25)


async def test_paid_graders_run_concurrently_in_order(tmp_path):
    judge = FakeJudge([True, True, False])
    graders = [g(f"l{i}", "llm", criteria=f"kritérium {i}") for i in range(3)]
    case = make_case(tmp_path, graders)
    results, _, cost = await grade_run(case, make_art(tmp_path), judge, excluded=set())
    assert [r.name for r in results] == ["l0", "l1", "l2"]
    assert len(judge.calls) == 9 and cost == pytest.approx(0.009)
