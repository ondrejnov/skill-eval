"""Odložené MCP tooly a ToolSearch (skilleval.agent.deferred + graph)."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from test_agent_graph import FakeChat, FakeProvider, call, new_trace, rows, run, settings

from skilleval.agent.deferred import TOOL_SEARCH, ToolSearch, deferred_tool_names, estimate_tokens, should_defer
from skilleval.config import Settings
from skilleval.harness.prompt import build_system_prompt
from skilleval.models import ToolDef

PREFIX = "mcp__plugin_webstore-core_webstore__"


def mcp(tool: str, desc: str = "", big: bool = False) -> ToolDef:
    schema: dict[str, Any] = {"type": "object", "properties": {"id": {"type": "integer", "description": "ID"}}}
    if big:
        schema["properties"].update({f"p{i}": {"type": "string", "description": "x" * 200} for i in range(10)})
    return ToolDef(name=PREFIX + tool, description=desc or f"Tool {tool}", input_schema=schema, kind="mcp",
                   server="webstore", tool=tool, plugin="webstore-core")


TOOLS = [mcp("core_whoami", "Kdo jsem a co smím."), mcp("orders_get", "Detail objednávky podle ID."),
         mcp("orders_list", "Seznam objednávek s filtry."), mcp("invoices_get", "Detail faktury."),
         mcp("analytics_documents_consistency_check_for_long_names", "Kontrola konzistence dokladů.")]


def names(model: FakeChat) -> list[str]:
    return [t["function"]["name"] for t in model.tools]


# ------------------------------------------------------------------------------------------------ rozhodnutí


def test_should_defer_modes():
    small, big = [mcp("a")], [mcp(f"t{i}", big=True) for i in range(30)]
    s = Settings()
    assert estimate_tokens(big) > s.defer_tools_min_tokens > estimate_tokens(small)
    assert should_defer(s, big) and not should_defer(s, small)
    assert should_defer(s, small, True) and not should_defer(s, big, False)
    assert should_defer(Settings(defer_tools="on"), small) and not should_defer(Settings(defer_tools="off"), big)
    assert not should_defer(Settings(defer_tools="on"), [ToolDef("Read", "", {})])  # žádné MCP tooly
    assert should_defer(Settings(defer_tools_min_tokens=1), small)
    assert deferred_tool_names(s, [ToolDef("Read", "", {}), *big]) == [d.name for d in big]


def test_settings_defer_from_toml(tmp_path):
    from skilleval.config import load_settings

    cfg = tmp_path / "skilleval.toml"
    cfg.write_text('[limits]\ndefer_tools = "on"\ndefer_tools_min_tokens = 500\n', encoding="utf-8")
    s = load_settings(config=str(cfg))
    assert s.defer_tools == "on" and s.defer_tools_min_tokens == 500


def test_search_select_and_keywords():
    ts = ToolSearch(TOOLS)
    found, missing = ts.select([PREFIX + "core_whoami", "orders_get", " nope ", ""])
    assert found == [PREFIX + "core_whoami", PREFIX + "orders_get"] and missing == ["nope"]
    assert ts.keyword("objednávky detail", 3)[0] == PREFIX + "orders_get"  # description
    assert ts.keyword("orders", 5) == [PREFIX + "orders_get", PREFIX + "orders_list"]
    assert ts.keyword("+invoices detail", 5) == [PREFIX + "invoices_get"]
    assert ts.keyword("+orders list", 1) == [PREFIX + "orders_list"]
    assert ts.keyword("zzz", 5) == []
    out, refs = ts.run({"query": "select:core_whoami,missing_tool"})
    assert refs == [{"type": "tool_reference", "tool_name": PREFIX + "core_whoami"}]
    assert "Not found: missing_tool" in out.content and not out.is_error
    assert ts.loaded == [PREFIX + "core_whoami"] and not ts.is_hidden(PREFIX + "core_whoami")
    assert ts.is_hidden(PREFIX + "orders_get")
    out, refs = ts.run({"query": "zzz"})
    assert refs == [] and "No matching deferred tools" in out.content


# ------------------------------------------------------------------------------------------------- smyčka


async def test_toolsearch_select_then_call_loaded_tool():
    long_full = PREFIX + "analytics_documents_consistency_check_for_long_names"

    def turn1(messages: list[BaseMessage], m: FakeChat) -> AIMessage:
        assert names(m) == ["Read", TOOL_SEARCH]  # MCP tooly nebindované
        sys = messages[0].content
        assert "<available-deferred-tools>" in sys and long_full in sys
        return AIMessage(content="", tool_calls=[call(TOOL_SEARCH, {
            "query": f"select:{PREFIX}core_whoami,{long_full}", "max_results": 3}, "s1")])

    def turn2(messages: list[BaseMessage], m: FakeChat) -> AIMessage:
        bound = names(m)
        assert bound[:2] == ["Read", PREFIX + "core_whoami"] and TOOL_SEARCH in bound and len(bound) == 4
        alias = next(t["function"]["name"] for t in m.tools if t["function"]["description"].startswith("Kontrola"))
        assert alias != long_full and len(alias) <= 64
        tm = messages[-1]
        assert isinstance(tm, ToolMessage) and f"{alias} (= {long_full})" in tm.content
        return AIMessage(content="", tool_calls=[call(PREFIX + "core_whoami", {}, "c1"), call(alias, {"id": 1}, "c2")])

    model = FakeChat(script=[turn1, turn2, "hotovo"])
    prov = FakeProvider([ToolDef("Read", "Read", {"type": "object"}), *TOOLS])
    trace = new_trace()
    out, trace = await run(model, [prov], trace=trace, defer_tools=True)
    assert out.error is None and out.turns == 3
    assert [c[0] for c in prov.calls] == [PREFIX + "core_whoami", long_full]
    assert [r.name for r in out.tool_calls] == [TOOL_SEARCH, PREFIX + "core_whoami", long_full]
    lines = rows(trace)
    assert TOOL_SEARCH in lines[0]["tools"]
    res = lines[3]["message"]["content"][0]
    assert res["tool_use_id"] == "s1" and res["content"] == [
        {"type": "tool_reference", "tool_name": PREFIX + "core_whoami"},
        {"type": "tool_reference", "tool_name": long_full}]
    assert '"name":"ToolSearch","input":{"query":"select:' in trace.lines[2]
    assert out.tool_calls[0].output and out.tool_calls[0].output.startswith("Loaded 2 tool(s)")


async def test_keyword_search_and_unloaded_call_executes():
    def turn1(messages: list[BaseMessage], m: FakeChat) -> AIMessage:
        return AIMessage(content="", tool_calls=[call(TOOL_SEARCH, {"query": "+orders detail"}, "s1"),
                                                 call(PREFIX + "invoices_get", {"id": 5}, "c0")])  # nenačtený

    def turn2(messages: list[BaseMessage], m: FakeChat) -> AIMessage:
        assert PREFIX + "orders_get" in names(m) and PREFIX + "invoices_get" not in names(m)
        tms = [x for x in messages if isinstance(x, ToolMessage)]
        assert tms[1].status == "success"  # odložený nenačtený tool se provede normálně
        return AIMessage(content="ok")

    model = FakeChat(script=[turn1, turn2])
    prov = FakeProvider(TOOLS)
    out, trace = await run(model, [prov], defer_tools=True)
    assert out.error is None and prov.calls == [(PREFIX + "invoices_get", {"id": 5})]
    blocks = [r["message"]["content"][0] for r in rows(trace) if r["type"] == "user" and isinstance(r["message"]["content"], list)]
    assert blocks[0]["content"][0]["tool_name"] == PREFIX + "orders_get"


async def test_auto_threshold_and_off():
    big = [mcp(f"t{i}", big=True) for i in range(30)]
    model = FakeChat(script=["ok"])
    await run(model, [FakeProvider(big)])  # auto, nad prahem
    assert names(model) == [TOOL_SEARCH]
    model = FakeChat(script=["ok"])
    await run(model, [FakeProvider(big)], settings=Settings(env={}, defer_tools="off"))
    assert len(names(model)) == 30
    model = FakeChat(script=["ok"])
    trace = new_trace()
    await run(model, [FakeProvider(TOOLS)], trace=trace)  # auto, pod prahem
    assert TOOL_SEARCH not in names(model) and TOOL_SEARCH not in json.loads(trace.lines[0])["tools"]
    assert "<available-deferred-tools>" not in model.calls[0][0].content
    model = FakeChat(script=["ok"])
    await run(model, [FakeProvider(TOOLS)], settings=Settings(env={}, defer_tools="on"))
    assert names(model) == [TOOL_SEARCH]


async def test_prompt_section_not_duplicated():
    prompt = build_system_prompt(cwd="/w", skills=[], tool_defs=TOOLS, mcp_instructions={}, append=None,
                                 today=date(2026, 11, 1), model="m", deferred_tools=[d.name for d in TOOLS])
    assert "<available-deferred-tools>\n" + "\n".join(d.name for d in TOOLS) + "\n</available-deferred-tools>" in prompt
    assert "ToolSearch" in prompt
    model = FakeChat(script=["ok"])
    from skilleval.agent.graph import run_agent

    await run_agent(settings=settings(), model_slug="x/y", system_prompt=prompt, prompt="p", providers=[FakeProvider(TOOLS)],
                    trace=new_trace(), max_turns=2, timeout_seconds=10, model=model, defer_tools=True)
    assert model.calls[0][0].content.count("<available-deferred-tools>") == 1
