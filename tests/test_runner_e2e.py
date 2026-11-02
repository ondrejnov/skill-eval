"""Integrace celého běhu (runner + CLI vrstvy) s fake modelem a fake soudcem – bez sítě."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from test_agent_graph import FakeChat
from test_grading_helpers import FakeJudge

from skilleval.runner import Runner, RunOptions

MCP_SERVER = Path(__file__).parent / "fixtures" / "mcp" / "server.py"
TOOL = "mcp__plugin_demo_tracker__create_issue"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def plugin(tmp_path: Path) -> Path:
    root = tmp_path / "demo"
    write(root / ".claude-plugin/plugin.json", json.dumps({"name": "demo"}))
    write(root / "skills/greeter/SKILL.md", "---\nname: greeter\ndescription: Greets.\n---\n\nSay ahoj.\n")
    write(root / ".mcp.json", json.dumps({"mcpServers": {"tracker": {"type": "http", "url": "http://127.0.0.1:9/x"}}}))
    write(root / "evals/mocks/tracker/create_issue.md",
          "---\nexpect:\n  title: string\n---\n\nCreated issue: {{input.title}}\n")
    case = root / "evals/c1"
    write(case / "prompt.md", "---\nmax_turns: 5\n---\n\nZaloz issue Bug.\n")
    write(case / "graders/skill.md", '---\ntype: tool_used\ntool: Skill\ninput_match: greeter\n---\n')
    write(case / "graders/reply.md", "---\ntype: regex\npattern: hotovo\nflags: i\n---\n")
    write(case / "graders/issue.md", f"---\ntype: tool_used\ntool: {TOOL}\ninput_match: Bug\n---\n")
    write(case / "graders/calls.md", "---\ntype: regex\ntarget: mock_calls\npattern: Created issue\n---\n")
    write(case / "graders/judge.md", "---\ntype: llm\nfocus: trace\n---\n\nPASS if an issue was created.\n")
    return root


@pytest.fixture
def config(tmp_path: Path) -> str:
    p = tmp_path / "skilleval.toml"
    p.write_text('[models]\ndefault = "test/fake"\njudge = "test/judge"\n', encoding="utf-8")
    return str(p)


def call(name: str, args: dict[str, Any], id_: str = "t1", cost: float = 0.0) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_, "type": "tool_call"}],
                     response_metadata={"token_usage": {"cost": cost}})


def script_with(_case: Any, arm: str, _idx: int) -> FakeChat:
    if arm == "with":
        return FakeChat(script=[call("Skill", {"skill": "greeter"}, "t0"), call(TOOL, {"title": "Bug"}), "Hotovo."])
    return FakeChat(script=["Nemám nástroje."])


def run(plugin: Path, config: str, factory: Any = script_with, judge: Any = None,
        **opts: Any) -> tuple[int, dict[str, Any], str, str]:
    import asyncio

    out, err = io.StringIO(), io.StringIO()
    o = RunOptions(target=str(plugin), config=config, output_dir=str(plugin / "out"), runs=1, **opts)
    runner = Runner(o, out=out, err=err, model_factory=factory, judge=judge or FakeJudge())
    rc = asyncio.run(runner.run())
    path = plugin / "out" / "aggregate-result.json"
    agg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return rc, agg, out.getvalue(), err.getvalue()


def test_single_arm_mocks(plugin: Path, config: str) -> None:
    rc, agg, out, _ = run(plugin, config, ablation="none")
    assert rc == 0, out
    case = agg["cases"][0]
    assert case["aggregates"]["score"] == 1
    r = case["arms"]["with"][0]
    assert "without" not in case["arms"]
    assert [g["name"] for g in r["graders"]] == ["calls", "issue", "judge", "reply", "skill"]
    assert all(g["passed"] and g["scored"] for g in r["graders"])  # --ablation none nic nevylučuje
    assert (plugin / "out/report.html").is_file()
    lines = [json.loads(x) for x in Path(r["tracePath"]).read_text(encoding="utf-8").splitlines()]
    assert lines[0]["subtype"] == "init" and TOOL in lines[0]["tools"] and "demo:greeter" in lines[0]["skills"]
    assert "SCORE" in out and "Report:" in out


def test_two_arm_without_has_no_plugin(plugin: Path, config: str) -> None:
    rc, agg, out, _ = run(plugin, config)
    case = agg["cases"][0]
    wo = case["arms"]["without"][0]
    init = json.loads(Path(wo["tracePath"]).read_text(encoding="utf-8").splitlines()[0])
    assert init["skills"] == [] and init["mcp_servers"] == []
    assert "Skill" not in init["tools"] and TOOL not in init["tools"]
    scored = {g["name"]: g["scored"] for g in wo["graders"]}
    assert scored["skill"] is False and scored["calls"] is False and scored["issue"] is True
    assert case["aggregates"]["delta"] == case["aggregates"]["score"] - case["aggregates"]["scoreWithout"]
    assert case["aggregates"]["delta"] > 0
    assert "W/OUT" in out and rc == 0  # Δ exit kód neovlivňuje


def test_expect_violation_aborts(plugin: Path, config: str) -> None:
    factory = lambda *_: FakeChat(script=[call(TOOL, {"title": 5}), "Hotovo."])  # noqa: E731
    rc, agg, _, _ = run(plugin, config, factory, ablation="none")
    r = agg["cases"][0]["arms"]["with"][0]
    assert rc == 1 and r["score"] == 0 and r["error"] is None and r["graders"] == []
    assert r["aborted"]["tool"] == "create_issue"


def test_fault_on_mock(plugin: Path, config: str) -> None:
    write(plugin / "evals/c1/skilleval.yaml",
          "faults:\n  - tool: create_issue\n    call: 1\n    error: true\n    body: conflict {{input.title}}\n")
    factory = lambda *_: FakeChat(script=[call(TOOL, {"title": "Bug"}, "a"), call(TOOL, {"title": "Bug"}, "b"), "Hotovo"])  # noqa: E731
    rc, agg, _, _ = run(plugin, config, factory, ablation="none")
    r = agg["cases"][0]["arms"]["with"][0]
    results = [json.loads(x) for x in Path(r["tracePath"]).read_text(encoding="utf-8").splitlines()]
    tr = [x for x in results if x["type"] == "user" and isinstance(x["message"]["content"], list)]
    assert "conflict Bug" in tr[0]["message"]["content"][0]["content"] and tr[0]["skilleval_source"] == "fault"
    assert "Created issue: Bug" in tr[1]["message"]["content"][0]["content"]
    assert any(x.get("subtype") == "skilleval_fault" for x in results)
    assert agg["cases"][0]["extensions"]["faults"]


def _status_error(status: int) -> openai.APIStatusError:
    req = httpx.Request("POST", "http://x")
    return openai.APIStatusError(f"Error code: {status} - boom", response=httpx.Response(status, request=req), body=None)


def test_api_error_scores_zero_without_grading(plugin: Path, config: str) -> None:
    factory = lambda *_: FakeChat(script=[_status_error(500)])  # noqa: E731
    rc, agg, _, err = run(plugin, config, factory, ablation="none")
    r = agg["cases"][0]["arms"]["with"][0]
    assert rc == 1 and r["score"] == 0 and r["graders"] == [] and r["error"]
    assert "API error" in err and agg["partial"] is False


def test_auth_error_is_partial(plugin: Path, config: str) -> None:
    write(plugin / "evals/c2/prompt.md", "Dalsi.\n")
    write(plugin / "evals/c2/graders/a.md", "---\ntype: regex\npattern: x\n---\n")
    factory = lambda *_: FakeChat(script=[_status_error(401)])  # noqa: E731
    rc, agg, _, _ = run(plugin, config, factory, ablation="none")
    assert rc == 2 and agg["partial"] is True and agg["partialReason"] == "auth_failed"
    assert agg["cases"][1]["arms"]["with"] == []


def test_cost_ceiling_is_partial(plugin: Path, config: str) -> None:
    write(plugin / "evals/c2/prompt.md", "Dalsi.\n")
    write(plugin / "evals/c2/graders/a.md", "---\ntype: regex\npattern: x\n---\n")
    factory = lambda *_: FakeChat(script=[call(TOOL, {"title": "Bug"}, cost=1.0), "Hotovo"])  # noqa: E731
    rc, agg, _, _ = run(plugin, config, factory, ablation="none", max_cost_usd=0.5)
    assert rc == 2 and agg["partialReason"] == "cost_ceiling"
    first = agg["cases"][0]["arms"]["with"][0]
    assert first["skippedPaidGraders"] is True
    assert agg["cases"][1]["arms"]["with"] == []


def test_threshold_and_load_error(plugin: Path, config: str) -> None:
    write(plugin / "evals/broken/prompt.md", "---\nbogus: 1\n---\n\nX\n")
    rc, agg, _, err = run(plugin, config, ablation="none")
    assert rc == 1 and "bogus" in err and len(agg["cases"]) == 1  # platný case přesto běží


def test_no_cases(plugin: Path, config: str) -> None:
    rc, _, _, err = run(plugin, config, ablation="none", case_globs=["nic-*"])
    assert rc == 1 and "No eval cases found" in err


def test_eval_dir_is_hidden_from_agent(plugin: Path, config: str) -> None:
    secret = str(plugin / "evals/c1/graders/judge.md")
    factory = lambda *_: FakeChat(script=[call("Read", {"file_path": secret}), "Hotovo"])  # noqa: E731
    _, agg, _, _ = run(plugin, config, factory, ablation="none")
    r = agg["cases"][0]["arms"]["with"][0]
    text = Path(r["tracePath"]).read_text(encoding="utf-8")
    assert "PASS if an issue" not in text and "is_error" in text


def test_scaffold_only_with_flag(plugin: Path, config: str) -> None:
    case = plugin / "evals/c1"
    write(case / "case.yaml", 'schema_version: "1.1"\nname: c1\ncontext:\n  scaffold_script: s.sh\n')
    write(case / "s.sh", "echo ahoj > seed.txt\n")
    read = lambda *_: FakeChat(script=[call("Read", {"file_path": "seed.txt"}), "Hotovo"])  # noqa: E731
    _, agg, _, _ = run(plugin, config, read, ablation="none")
    r = agg["cases"][0]["arms"]["with"][0]
    assert "scaffold skipped: --scaffold off" in r["notices"]
    _, agg, _, _ = run(plugin, config, read, ablation="none", scaffold=True)
    text = Path(agg["cases"][0]["arms"]["with"][0]["tracePath"]).read_text(encoding="utf-8")
    assert "ahoj" in text


def test_write_only_with_grant_and_file_exists(plugin: Path, config: str) -> None:
    case = plugin / "evals/c1"
    write(case / "graders/file.md", "---\ntype: file_exists\npath: out.txt\n---\n")
    factory = lambda *_: FakeChat(script=[call("Write", {"file_path": "out.txt", "content": "x"}), "Hotovo"])  # noqa: E731
    _, agg, _, _ = run(plugin, config, factory, ablation="none")
    g = {x["name"]: x for x in agg["cases"][0]["arms"]["with"][0]["graders"]}
    assert g["file"]["passed"] is False
    _, agg, _, _ = run(plugin, config, factory, ablation="none", allow_tools=["Write"])
    g = {x["name"]: x for x in agg["cases"][0]["arms"]["with"][0]["graders"]}
    assert g["file"]["passed"] is True


def test_json_stdout_is_quiet(plugin: Path, config: str) -> None:
    rc, agg, out, err = run(plugin, config, ablation="none", json_out="-")
    assert rc == 0 and json.loads(out)["schemaVersion"] == 1 and err == ""


def test_real_stdio_server_mocks_off(plugin: Path, config: str, tmp_path: Path) -> None:
    write(plugin / ".mcp.json", json.dumps({"mcpServers": {"tracker": {
        "type": "stdio", "command": sys.executable, "args": [str(MCP_SERVER)]}}}))
    case = plugin / "evals/c1"
    for name in ("issue", "calls", "skill", "reply", "judge"):
        (case / f"graders/{name}.md").unlink()
    write(case / "graders/add.md", "---\ntype: tool_used\ntool: mcp__plugin_demo_tracker__add\n---\n")
    write(case / "graders/out.md", "---\ntype: regex\ntarget: trace\npattern: 'result.{0,4}:3'\n---\n")
    factory = lambda *_: FakeChat(script=[call("mcp__plugin_demo_tracker__add", {"a": 1, "b": 2}), "Hotovo"])  # noqa: E731
    rc, agg, _, _ = run(plugin, config, factory, ablation="none", mocks="off")
    assert rc == 0, agg["cases"][0]["arms"]["with"][0]


def test_report_keeps_foreign_report(tmp_path: Path, plugin: Path, config: str) -> None:
    from skilleval.cli import main

    _, agg, _, _ = run(plugin, config, ablation="none")
    foreign = tmp_path / "cc"
    foreign.mkdir()
    agg.pop("runner")
    (foreign / "aggregate-result.json").write_text(json.dumps(agg), encoding="utf-8")
    (foreign / "report.html").write_text("original", encoding="utf-8")
    with pytest.raises(SystemExit) as ex:
        main(["report", str(foreign)])
    assert ex.value.code == 0
    assert (foreign / "report.html").read_text(encoding="utf-8") == "original"
    assert (foreign / "report-skilleval.html").is_file()
