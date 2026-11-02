import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from skilleval.config import Settings
from skilleval.mcp.session import BUDGET_EXCEEDED, McpSession
from skilleval.models import CaseSpec, ExtensionSpec, FaultSpec, GraderSpec, McpServerConfig

SERVER = Path(__file__).parent / "fixtures" / "mcp" / "server.py"
PLUGIN = "my-plugin"


def name(server: str, tool: str) -> str:
    return f"mcp__plugin_{PLUGIN}_{server}__{tool}"


def w(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_case(tmp_path: Path, *, mock_dirs: list[Path] | None = None, faults: list[dict] | None = None,
              max_turns: int = 10) -> CaseSpec:
    case_dir = tmp_path / "case"
    case_dir.mkdir(exist_ok=True)
    ext = ExtensionSpec(faults=[FaultSpec(**f) for f in faults]) if faults else None
    return CaseSpec(name="c", dir=case_dir, rel_dir="evals/c", source="prompt", prompt="x", max_turns=max_turns,
                    graders=[GraderSpec(name="g", type="regex", config={"pattern": "x"})],
                    mock_dirs=mock_dirs or [], extension=ext)


def stdio_cfg(server: str = "fx", **kw: Any) -> McpServerConfig:
    return McpServerConfig(name=server, plugin=PLUGIN, type="stdio", command=sys.executable,
                           args=[str(SERVER), "stdio"], **kw)


class FakeJudge:
    def __init__(self, replies: list[str] | None = None):
        self.replies = list(replies or [])
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    async def vote(self, system, user, images=None):  # pragma: no cover - nepoužito
        raise NotImplementedError

    async def complete(self, system: str, messages: list[dict[str, str]]) -> tuple[str, float]:
        self.calls.append((system, messages))
        return (self.replies.pop(0) if self.replies else "ok"), 0.01


def session(servers, case, *, mode="record", allow_real=False, allow_tools=None, strict=False, judge=None,
            hooks=False, recordings_dir=None, run_env=None, settings=None, events=None) -> McpSession:
    return McpSession(servers, case=case, mode=mode, allow_real=allow_real, allow_tools=allow_tools or [],
                      strict_grants=strict, judge=judge, settings=settings or Settings(), hooks=hooks,
                      recordings_dir=recordings_dir, run_env=run_env or {},
                      on_event=(lambda k, d: events.append((k, d))) if events is not None else None)


# ---------------------------------------------------------------------------------------------------------------------
# fixed mocky
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def tracker_mocks(tmp_path):
    suite = tmp_path / "suite-mocks"
    w(suite / "tracker" / "create_issue.md",
      "---\nexpect:\n  title: string\n  priority: [low, medium, high]\n---\n\nCreated issue #4821: {{input.title}}\n")
    w(suite / "tracker" / "get_issue.md", "Issue {{input.id}}: {{file:fixtures/{input.id}.json}}\n")
    w(suite / "tracker" / "fixtures" / "7.json", '{"title": "Seven"}')
    w(suite / "tracker" / "broken.md", "---\nerror: true\n---\nService unavailable\n")
    return suite


async def test_fixed_mock_record_mode(tmp_path, tracker_mocks):
    case = make_case(tmp_path, mock_dirs=[tracker_mocks])
    servers = [stdio_cfg("tracker"), stdio_cfg("fx")]
    async with session(servers, case) as s:
        assert sorted(t.name for t in s.tool_defs) == sorted(
            name("tracker", t) for t in ("create_issue", "get_issue", "broken"))
        assert all(t.kind == "mcp" and t.server == "tracker" and t.plugin == PLUGIN for t in s.tool_defs)
        assert "mocked: plugin_my-plugin_tracker[3 tools]" in s.notices
        assert "mocked: plugin_my-plugin_fx[not started: no mock]" in s.notices
        assert s.instructions == {}

        o = await s.call(name("tracker", "create_issue"), {"title": "Widget", "priority": "low"})
        assert (o.content, o.is_error, o.source, o.aborted) == ("Created issue #4821: Widget", False, "mock", None)
        o = await s.call(name("tracker", "get_issue"), {"id": 7})
        assert o.content == 'Issue 7: {"title": "Seven"}'
        o = await s.call(name("tracker", "get_issue"), {"id": 8})
        assert o.is_error and "fixtures/8.json not found" in o.content
        o = await s.call(name("tracker", "broken"), {})
        assert o.is_error and o.content == "Service unavailable"
        o = await s.call(name("tracker", "create_issue"), {"title": "X", "priority": "urgent"})
        assert o.aborted == {"server": "tracker", "tool": "create_issue",
                             "reason": 'priority: "urgent" is not one of ["low", "medium", "high"]'}
        o = await s.call(name("fx", "echo"), {"text": "x"})
        assert o.is_error and o.source == "denied"
    assert [(c.tool, c.source, c.is_error) for c in s.calls] == [
        ("create_issue", "mock", False), ("get_issue", "mock", False), ("get_issue", "mock", True),
        ("broken", "mock", True), ("create_issue", "mock", True)]
    assert s.calls[0].input == {"title": "Widget", "priority": "low"} and s.calls[0].server == "tracker"


async def test_case_mocks_override_suite(tmp_path, tracker_mocks):
    case_mocks = tmp_path / "case-mocks"
    w(case_mocks / "tracker" / "create_issue.md", "Case override {{input.title}}")
    case = make_case(tmp_path, mock_dirs=[tracker_mocks, case_mocks])
    async with session([stdio_cfg("tracker")], case) as s:
        o = await s.call(name("tracker", "create_issue"), {"title": "A", "priority": "nope"})  # bez expect
        assert o.content == "Case override A" and o.aborted is None
        assert (await s.call(name("tracker", "get_issue"), {"id": 7})).content.startswith("Issue 7")


async def test_undeclared_mock_server(tmp_path):
    mocks = tmp_path / "m"
    w(mocks / "ghost" / "x.md", "boo")
    case = make_case(tmp_path, mock_dirs=[mocks])
    async with session([], case) as s:
        assert s.tool_defs == [] and any("ghost" in n for n in s.notices)
    s2 = McpSession([], case=case, mode="record", allow_real=False, allow_tools=[], strict_grants=False, judge=None,
                    settings=Settings(), hooks=False, recordings_dir=None, run_env={}, default_plugin="p")
    async with s2:
        assert [t.name for t in s2.tool_defs] == ["mcp__plugin_p_ghost__x"]


# ---------------------------------------------------------------------------------------------------------------------
# agent mocky
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def agent_mocks(tmp_path):
    m = tmp_path / "mocks"
    w(m / "crm" / "find.md", "---\ntype: agent\nexpect:\n  q: string\n---\nYou are a CRM. Customer Novák has id 5.\n")
    w(m / "crm" / "_server.md", "---\ntype: agent\ntools: [update, delete]\nabort_when: |\n  The agent deletes a "
                                "customer without asking.\n---\nCRM that stores customers.\n")
    w(m / "crm" / "_tools.json", json.dumps({"tools": [{"name": "find", "description": "Find customers",
                                                         "inputSchema": {"type": "object"}}]}))
    return m


async def test_agent_mock_history_recording_and_replay(tmp_path, agent_mocks):
    rec_dir = tmp_path / "results" / "mock-recordings"
    judge = FakeJudge(["[{\"id\": 5}]", "```\nupdated\n```", "ERROR: not found"])
    case = make_case(tmp_path, mock_dirs=[agent_mocks])
    async with session([stdio_cfg("crm")], case, judge=judge, recordings_dir=rec_dir) as s:
        assert {t.name: t.description for t in s.tool_defs}[name("crm", "find")] == "Find customers"
        o = await s.call(name("crm", "find"), {"q": "Novák"})
        assert (o.content, o.source, o.cost_usd) == ('[{"id": 5}]', "agent-mock", 0.01)
        o = await s.call(name("crm", "update"), {"id": 5})
        assert o.content == "updated"
        o = await s.call(name("crm", "update"), {"id": 6})
        assert o.is_error and o.content == "not found"
        assert s.mock_cost_usd == pytest.approx(0.03)
    system, messages = judge.calls[1]
    assert "CRM that stores customers." in system and '"crm"' in system and '"update"' in system
    assert "The agent deletes a customer without asking." in system and "ABORT:" in system
    assert "Never abort" in judge.calls[0][0] and "Find customers" in judge.calls[0][0]
    assert messages[0] == {"role": "user", "content": '{"tool": "find", "input": {"q": "Novák"}}'}
    assert messages[1] == {"role": "assistant", "content": '[{"id": 5}]'}
    assert messages[-1]["content"] == '{"tool": "update", "input": {"id": 5}}'
    assert len(list(rec_dir.glob("crm/*.json"))) == 3
    assert s.recordings and all(p.exists() for p in s.recordings)

    # adopce nahrávky → replay bez modelu
    from skilleval.mcp.mocks import write_adopt
    adopt = write_adopt(rec_dir.parent)
    assert adopt is not None and str(agent_mocks / ".replay" / "crm") in adopt.read_text()
    replay_dir = agent_mocks / ".replay" / "crm"
    replay_dir.mkdir(parents=True)
    for f in rec_dir.glob("crm/find-*.json"):
        (replay_dir / f.name).write_text(f.read_text())
    judge2 = FakeJudge()
    async with session([stdio_cfg("crm")], make_case(tmp_path, mock_dirs=[agent_mocks]), judge=judge2) as s2:
        o = await s2.call(name("crm", "find"), {"q": "Novák"})
        assert (o.content, o.source) == ('[{"id": 5}]', "replay")
        assert judge2.calls == []
        assert s2.agent_calls == 1  # replay se počítá do rozpočtu


async def test_agent_mock_discard_recordings(tmp_path, agent_mocks):
    rec_dir = tmp_path / "rec"
    async with session([stdio_cfg("crm")], make_case(tmp_path, mock_dirs=[agent_mocks]), judge=FakeJudge(),
                       recordings_dir=rec_dir) as s:
        await s.call(name("crm", "find"), {"q": "x"})
    assert len(s.recordings) == 1
    s.discard_recordings()
    assert list(rec_dir.glob("*/*.json")) == []


async def test_agent_mock_budget(tmp_path, agent_mocks):
    case = make_case(tmp_path, mock_dirs=[agent_mocks], max_turns=1)  # rozpočet 4
    async with session([stdio_cfg("crm")], case, judge=FakeJudge()) as s:
        for i in range(4):
            assert (await s.call(name("crm", "find"), {"q": str(i)})).aborted is None
        o = await s.call(name("crm", "find"), {"q": "5"})
        assert o.aborted == {"server": "crm", "tool": "find", "reason": BUDGET_EXCEEDED}


async def test_agent_mock_abort_when(tmp_path, agent_mocks):
    case = make_case(tmp_path, mock_dirs=[agent_mocks])
    judge = FakeJudge(["ABORT: deleted customer 5 without asking", "ABORT: nope"])
    async with session([stdio_cfg("crm")], case, judge=judge) as s:
        o = await s.call(name("crm", "delete"), {"id": 5})
        assert o.aborted == {"server": "crm", "tool": "delete", "reason": "deleted customer 5 without asking"}
        o = await s.call(name("crm", "find"), {"q": "x"})  # find nemá abort_when → ABORT je obyčejná odpověď
        assert o.aborted is None and o.content == "ABORT: nope"


async def test_agent_mock_expect_and_no_judge(tmp_path, agent_mocks):
    case = make_case(tmp_path, mock_dirs=[agent_mocks])
    async with session([stdio_cfg("crm")], case, judge=None) as s:
        o = await s.call(name("crm", "find"), {"q": 1})
        assert o.aborted and o.aborted["reason"].startswith("q: expected string")
        o = await s.call(name("crm", "find"), {"q": "x"})
        assert o.is_error and "judge" in o.content


# ---------------------------------------------------------------------------------------------------------------------
# reálné servery: off, hybrid, granty, stdio env, chyby
# ---------------------------------------------------------------------------------------------------------------------


async def test_mode_off_real_stdio(tmp_path, tracker_mocks):
    w(tracker_mocks / "fx" / "echo.md", "MOCKED")
    case = make_case(tmp_path, mock_dirs=[tracker_mocks])
    s = session([stdio_cfg("fx", env={"EXTRA": "e"})], case, mode="off", run_env={"EVAL_X": "hello"},
                settings=Settings(max_tool_result_chars=50))
    async with s:
        names = [t.name for t in s.tool_defs]
        assert len(names) == 8 and name("fx", "echo") in names  # 4 stránky tools/list
        assert s.instructions == {"fx": "Fixture server: use echo for everything."}
        assert (await s.call(name("fx", "echo"), {"text": "hi"})).content == "hi"  # mock ignorován
        assert (await s.call(name("fx", "add"), {"a": 1, "b": 2})).content == '{"result":3}'  # structuredContent
        o = await s.call(name("fx", "fail"), {"msg": "boom"})
        assert o.is_error and "boom" in o.content and o.source == "real"
        assert (await s.call(name("fx", "link"), {})).content == "[resource] webstore://export/1.pdf (faktura.pdf)"
        assert (await s.call(name("fx", "env"), {"name": "EVAL_X"})).content == "hello"
        assert (await s.call(name("fx", "env"), {"name": "EXTRA"})).content == "e"
        assert (await s.call(name("fx", "counter"), {})).content == "1"
        assert (await s.call(name("fx", "counter"), {})).content == "2"  # jedno spojení na run
        o = await s.call(name("fx", "big"), {"n": 80})
        assert o.content == "x" * 50 + "\n… [truncated 30 chars]"
        o = await s.call(name("fx", "add"), {"a": "x"})
        assert o.is_error
    assert all(c.source == "real" for c in s.calls)


async def test_hybrid_allow_real(tmp_path):
    mocks = tmp_path / "mocks"
    w(mocks / "fx" / "echo.md", "MOCKED {{input.text}}")
    case = make_case(tmp_path, mock_dirs=[mocks])
    async with session([stdio_cfg("fx"), stdio_cfg("fy")], case, allow_real=True) as s:
        names = {t.name: t for t in s.tool_defs}
        assert len([n for n in names if "__plugin_my-plugin_fx__" in n]) == 8
        assert len([n for n in names if "__plugin_my-plugin_fy__" in n]) == 8
        assert names[name("fx", "echo")].description == "Echo the text back."  # schéma z reálného serveru
        assert "text" in names[name("fx", "echo")].input_schema["properties"]
        assert (await s.call(name("fx", "echo"), {"text": "a"})).content == "MOCKED a"
        assert (await s.call(name("fx", "add"), {"a": 2, "b": 2})).content == '{"result":4}'
        assert (await s.call(name("fy", "echo"), {"text": "b"})).content == "b"
        assert "mocked: plugin_my-plugin_fx[1 tool + 7 real]" in s.notices
    assert [c.source for c in s.calls] == ["mock", "real", "real"]


async def test_strict_grants(tmp_path):
    case = make_case(tmp_path)
    async with session([stdio_cfg("fx")], case, mode="off", strict=True,
                       allow_tools=[name("fx", "ec*"), "mcp__plugin_my-plugin_fx__add"]) as s:
        assert sorted(t.name for t in s.tool_defs) == [name("fx", "add"), name("fx", "echo")]
        ng = [n for n in s.notices if n.startswith("not granted: ")]
        assert len(ng) == 1 and name("fx", "fail") in ng[0]
    async with session([stdio_cfg("fx")], case, mode="off") as s:  # default: vše povoleno
        assert len(s.tool_defs) == 8 and not any(n.startswith("not granted") for n in s.notices)


async def test_server_connect_failure(tmp_path):
    bad = McpServerConfig(name="bad", plugin=PLUGIN, type="stdio", command="/nonexistent/server-binary")
    http_bad = McpServerConfig(name="hb", plugin=PLUGIN, type="http", url=f"http://127.0.0.1:{free_port()}/mcp")
    async with session([bad, http_bad, stdio_cfg("fx")], make_case(tmp_path), mode="off") as s:
        assert len(s.tool_defs) == 8
        assert any(n.startswith("server bad failed:") for n in s.notices)
        assert any(n.startswith("server hb failed:") for n in s.notices)


# ---------------------------------------------------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def http_server():
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(SERVER), "http", str(port)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            break
        except OSError:
            time.sleep(0.1)
    else:  # pragma: no cover
        proc.kill()
        pytest.fail("fixture MCP server did not start")
    yield f"http://127.0.0.1:{port}/mcp"
    proc.terminate()
    proc.wait(timeout=10)


async def test_http_transport_headers_and_masking(tmp_path, http_server):
    token = "Bearer sk-test-secret-1234567890"
    cfg = McpServerConfig(name="web", plugin=PLUGIN, type="http", url=http_server,
                          headers={"Authorization": token, "X-Plain": "plain-value-123"})
    settings = Settings(env={"MY_API_TOKEN": "env-secret-abcdef"})
    async with session([cfg], make_case(tmp_path), mode="off", settings=settings) as s:
        assert len(s.tool_defs) == 8 and s.instructions["web"].startswith("Fixture server")
        assert (await s.call(name("web", "echo"), {"text": "via http"})).content == "via http"
        o = await s.call(name("web", "header"), {"name": "authorization"})
        assert o.content == "***" and "sk-test" not in o.content
        o = await s.call(name("web", "echo"), {"text": "leak env-secret-abcdef"})
        assert o.content == "leak ***"
        assert (await s.call(name("web", "header"), {"name": "x-plain"})).content == "***"


# ---------------------------------------------------------------------------------------------------------------------
# fault injection
# ---------------------------------------------------------------------------------------------------------------------

CONFLICT = '{"error":{"code":"conflict","message":"Záznam {{input.id}} byl mezitím změněn."}}'


async def test_fault_call_n_with_when_on_real_server(tmp_path):
    faults = [{"tool": "echo", "call": 2, "when": {"text": "/^w/"}, "error": True, "body": CONFLICT.replace(
        "{{input.id}}", "{{input.text}}")}]
    events: list = []
    async with session([stdio_cfg("fx")], make_case(tmp_path, faults=faults), mode="off", events=events) as s:
        e = name("fx", "echo")
        assert (await s.call(e, {"text": "w1"})).content == "w1"
        assert (await s.call(e, {"text": "other"})).content == "other"  # nesplňuje when → nepočítá se
        o = await s.call(e, {"text": "w2"})
        assert o.is_error and o.source == "fault" and "Záznam w2 byl" in o.content
        assert (await s.call(e, {"text": "w3"})).content == "w3"  # passthrough_after (default)
    assert [c.source for c in s.calls] == ["real", "real", "fault", "real"]
    assert events == [("skilleval_fault", {"tool": e, "server": "fx", "fault": 0, "call": 2, "action": "response",
                                           "is_error": True})]


async def test_fault_every_calls_full_name_and_sticky(tmp_path, tracker_mocks):
    faults = [
        {"tool": name("tracker", "get_issue"), "every": True, "body": "fault {{input.id}}"},
        {"tool": "create_issue", "calls": [1, 3], "body": "boom", "error": True},
        {"tool": "broken", "call": 1, "body": "sticky", "passthrough_after": False},
    ]
    case = make_case(tmp_path, mock_dirs=[tracker_mocks], faults=faults)
    async with session([stdio_cfg("tracker")], case) as s:  # faulty platí i nad mocky
        g = name("tracker", "get_issue")
        assert [(await s.call(g, {"id": i})).content for i in (1, 2)] == ["fault 1", "fault 2"]
        c = name("tracker", "create_issue")
        args = {"title": "T", "priority": "low"}
        assert [(await s.call(c, args)).source for _ in range(3)] == ["fault", "mock", "fault"]
        b = name("tracker", "broken")
        assert [(await s.call(b, {})).content for _ in range(3)] == ["sticky"] * 3


async def test_fault_before_call_hooks(tmp_path):
    case = make_case(tmp_path, faults=[{"tool": "counter", "before_call": 2,
                                        "run": ["bash", "-c", "echo ran >> marker.txt"]}])
    events: list = []
    async with session([stdio_cfg("fx")], case, mode="off", hooks=True, events=events) as s:
        for _ in range(3):
            await s.call(name("fx", "counter"), {})
    assert (case.dir / "marker.txt").read_text() == "ran\n"
    assert [d["action"] for _, d in events] == ["run"] and events[0][1]["call"] == 2
    assert [c.source for c in s.calls] == ["real"] * 3

    (case.dir / "marker.txt").unlink()
    events.clear()
    async with session([stdio_cfg("fx")], case, mode="off", hooks=False, events=events) as s:
        for _ in range(3):
            await s.call(name("fx", "counter"), {})
    assert not (case.dir / "marker.txt").exists()
    assert "fault run skipped: --hooks off" in s.notices
    assert [d["action"] for _, d in events] == ["run_skipped"]


async def test_fault_run_failure_notice(tmp_path):
    case = make_case(tmp_path, faults=[{"tool": "echo", "before_call": 1, "run": ["bash", "-c", "exit 3"]}])
    async with session([stdio_cfg("fx")], case, mode="off", hooks=True) as s:
        assert (await s.call(name("fx", "echo"), {"text": "a"})).content == "a"
    assert any(n.startswith("fault run failed:") and "code 3" in n for n in s.notices)


async def test_prefer_structured_content_setting(tmp_path):
    async with session([stdio_cfg("fx")], make_case(tmp_path), mode="off",
                       settings=Settings(prefer_structured_content=False)) as s:
        assert (await s.call(name("fx", "add"), {"a": 1, "b": 2})).content == "3"


def test_convert_result_structured():
    from mcp import types

    from skilleval.mcp.client import convert_result

    res = types.CallToolResult(
        content=[types.TextContent(type="text", text="Uloženo: objednávka 71781"),
                 types.ResourceLink(type="resource_link", uri="webstore://x.pdf", name="x.pdf"),
                 types.ImageContent(type="image", data="aGk=", mimeType="image/png")],
        structuredContent={"ok": True, "dry_run": False, "název": "č"})
    text, images, err = convert_result(res)
    assert text == '{"ok":true,"dry_run":false,"název":"č"}\n[resource] webstore://x.pdf (x.pdf)'
    assert images == [{"media_type": "image/png", "data": "aGk="}] and not err
    text, _, _ = convert_result(res, prefer_structured=False)
    assert text == "Uloženo: objednávka 71781\n[resource] webstore://x.pdf (x.pdf)"
    only_sc = types.CallToolResult(content=[], structuredContent={"a": 1}, isError=True)
    assert convert_result(only_sc, prefer_structured=False) == ('{"a":1}', [], True)
    plain = types.CallToolResult(content=[types.TextContent(type="text", text="hi")])
    assert convert_result(plain) == ("hi", [], False)
