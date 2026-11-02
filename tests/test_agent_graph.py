"""Smyčka agenta (skilleval.agent.graph) s fake chat modelem a fake ToolProviderem – bez sítě."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import openai
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from skilleval.agent.graph import AgentOutcome, run_agent, system_message
from skilleval.agent.simulated_user import NOT_AVAILABLE, looks_like_question, make_ask_user_hook
from skilleval.agent.tools import VALID_NAME_RE, ToolIndex, make_alias
from skilleval.config import Settings
from skilleval.models import SimulatedUserSpec, ToolDef, ToolOutcome
from skilleval.trace import TraceWriter

SECRET = "sk-or-v1-supersecret-0123456789"
SLUG = "test/fake-model"


# --------------------------------------------------------------------------------------------------- fakes


class FakeChat(BaseChatModel):
    """Skriptovaný model: položka = AIMessage | str | výjimka | callable(messages, model) → jedno z nich."""

    script: list[Any] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    delay: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeChat:  # type: ignore[override]
        self.tools = list(tools)
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        raise NotImplementedError

    async def _agenerate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                         **kw: Any) -> ChatResult:
        self.calls.append(list(messages))
        if self.delay:
            await asyncio.sleep(self.delay)
        item = self.script.pop(0) if self.script else AIMessage(content="done")
        if callable(item):
            item = item(messages, self)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            item = AIMessage(content=item)
        return ChatResult(generations=[ChatGeneration(message=item)])

    def tool_alias(self, full: str) -> str:
        """Název toolu, jak ho model dostal (alias) – podle description `Tool <plný název>` (viz tdef)."""
        for t in self.tools:
            if t["function"]["description"] == f"Tool {full}":
                return t["function"]["name"]
        raise AssertionError(f"tool {full} nenabídnut")


Handler = Callable[[str, dict[str, Any]], Awaitable[ToolOutcome]]


class FakeProvider:
    def __init__(self, defs: list[ToolDef], handler: Handler | None = None):
        self._defs = defs
        self.handler = handler
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def tool_defs(self) -> list[ToolDef]:
        return self._defs

    async def call(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        self.calls.append((name, args))
        if self.handler:
            return await self.handler(name, args)
        return ToolOutcome(content=f"{name} ok {json.dumps(args, ensure_ascii=False)}")


def tdef(name: str, kind: str = "builtin") -> ToolDef:
    return ToolDef(name=name, description=f"Tool {name}", kind=kind,  # type: ignore[arg-type]
                   input_schema={"type": "object", "properties": {"q": {"type": "string"}}})


def call(name: str, args: dict[str, Any] | None = None, id: str | None = None) -> dict[str, Any]:
    return {"name": name, "args": args or {}, "id": id or f"toolu_{name[-6:]}_{len(name)}", "type": "tool_call"}


def settings(**kw: Any) -> Settings:
    return Settings(env={"OPENROUTER_API_KEY": SECRET}, **kw)


def new_trace() -> TraceWriter:
    return TraceWriter(cwd="/tmp/ws", model=SLUG, tools=["Read"], mcp_servers=[], skills=[], plugins=[])


async def run(model: FakeChat, providers: list[Any], *, trace: TraceWriter | None = None, max_turns: int = 10,
              timeout: int = 30, **kw: Any) -> tuple[AgentOutcome, TraceWriter]:
    trace = trace or new_trace()
    out = await run_agent(settings=kw.pop("settings", settings()), model_slug=kw.pop("slug", SLUG),
                          system_prompt="SYSTEM", prompt="Jaké je počasí v Praze?", providers=providers, trace=trace,
                          max_turns=max_turns, timeout_seconds=timeout, model=model, **kw)
    return out, trace


def rows(trace: TraceWriter) -> list[dict[str, Any]]:
    return [json.loads(line) for line in trace.lines]


# --------------------------------------------------------------------------------------------------- smyčka


async def test_loop_tool_then_answer():
    model = FakeChat(script=[AIMessage(content="Zjistím to.", tool_calls=[call("get_weather", {"q": "Praha"}, "t1")]),
                             AIMessage(content="V Praze je 20 °C a \"jasno\".")])
    prov = FakeProvider([tdef("get_weather")])
    out, trace = await run(model, [prov])

    assert out.error is None and not out.api_error and out.aborted is None
    assert out.turns == 2
    assert out.final_message == 'V Praze je 20 °C a "jasno".'
    assert prov.calls == [("get_weather", {"q": "Praha"})]
    assert [(r.name, r.input, r.is_error) for r in out.tool_calls] == [("get_weather", {"q": "Praha"}, False)]
    assert out.tool_calls[0].output.startswith("get_weather ok")
    # model dostal system, prompt; ve 2. tahu výsledek toolu
    assert isinstance(model.calls[0][0], SystemMessage) and model.calls[0][0].content == "SYSTEM"
    assert isinstance(model.calls[0][1], HumanMessage)
    tm = model.calls[1][-1]
    assert isinstance(tm, ToolMessage) and tm.tool_call_id == "t1" and tm.status == "success"
    assert model.tools[0] == {"type": "function", "function": {
        "name": "get_weather", "description": "Tool get_weather",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}

    lines = rows(trace)
    types = [(r["type"], r.get("subtype")) for r in lines]
    assert types == [("system", "init"), ("user", None), ("assistant", None), ("assistant", None), ("user", None),
                     ("assistant", None), ("result", "success")]
    assert lines[1]["message"] == {"role": "user", "content": "Jaké je počasí v Praze?"}
    assert lines[2]["message"]["content"] == [{"type": "text", "text": "Zjistím to."}]
    assert lines[3]["message"]["content"] == [{"type": "tool_use", "id": "t1", "name": "get_weather",
                                               "input": {"q": "Praha"}}]
    assert lines[2]["message"]["id"] == lines[3]["message"]["id"]
    tr = lines[4]["message"]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "t1" and "is_error" not in tr
    assert lines[-1]["num_turns"] == 2 and lines[-1]["result"] == out.final_message and lines[-1]["error"] is None
    # regex nad trace vidí escapované uvozovky a ne-ASCII jako CC
    assert '20 °C a \\"jasno\\"' in trace.lines[5]


async def test_parallel_tool_calls_run_concurrently_results_in_order():
    running = 0
    peak = 0

    async def handler(name: str, args: dict[str, Any]) -> ToolOutcome:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.2 if args["q"] == "a" else 0.05)  # první skončí později
        running -= 1
        return ToolOutcome(content=f"result {args['q']}")

    model = FakeChat(script=[AIMessage(content="", tool_calls=[call("t_x", {"q": "a"}, "c1"), call("t_x", {"q": "b"}, "c2")]),
                             "hotovo"])
    out, trace = await run(model, [FakeProvider([tdef("t_x")], handler)])
    assert peak == 2
    msgs = model.calls[1]
    assert [m.tool_call_id for m in msgs if isinstance(m, ToolMessage)] == ["c1", "c2"]
    results = [r["message"]["content"][0] for r in rows(trace) if r["type"] == "user" and isinstance(r["message"]["content"], list)]
    assert [(b["tool_use_id"], b["content"]) for b in results] == [("c1", "result a"), ("c2", "result b")]
    assert [r.output for r in out.tool_calls] == ["result a", "result b"]


async def test_max_turns():
    model = FakeChat(script=[AIMessage(content=f"krok {i}", tool_calls=[call("t_x", {"q": str(i)}, f"c{i}")])
                             for i in range(10)])
    out, trace = await run(model, [FakeProvider([tdef("t_x")])], max_turns=2)
    assert out.error == "max turns reached (2)"
    assert out.turns == 2 and len(model.calls) == 2
    assert len(out.tool_calls) == 2
    assert out.final_message == "krok 1"
    last = rows(trace)[-1]
    assert last["type"] == "result" and last["subtype"] == "error" and last["error"] == "max turns reached (2)"


async def test_timeout_keeps_tool_calls_and_trace():
    async def slow(name: str, args: dict[str, Any]) -> ToolOutcome:
        if args["q"] == "slow":
            await asyncio.sleep(10)
        return ToolOutcome(content="ok")

    model = FakeChat(script=[AIMessage(content="", tool_calls=[call("t_x", {"q": "fast"}, "c1")]),
                             AIMessage(content="", tool_calls=[call("t_x", {"q": "slow"}, "c2")])])
    out, trace = await run(model, [FakeProvider([tdef("t_x")], slow)], timeout=1)
    assert out.error == "timed out after 1s"
    assert not out.api_error
    assert [r.id for r in out.tool_calls] == ["c1", "c2"]
    assert out.tool_calls[0].output == "ok" and out.tool_calls[1].output is None
    assert rows(trace)[-1]["error"] == "timed out after 1s"


async def test_api_error_masked():
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    err = openai.APIStatusError(f"Rate limited for key {SECRET}", response=httpx.Response(429, request=req), body=None)
    model = FakeChat(script=[err])
    out, trace = await run(model, [])
    assert out.api_error is True
    assert out.error and "Rate limited" in out.error and SECRET not in out.error and "***" in out.error
    assert SECRET not in trace.text()
    assert out.turns == 1


async def test_internal_error_does_not_raise():
    model = FakeChat(script=[RuntimeError("boom")])
    out, _ = await run(model, [])
    assert out.error == "internal error: RuntimeError: boom" and not out.api_error


async def test_aborted_stops_run():
    async def handler(name: str, args: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(content="aborted", is_error=True, source="mock",
                           aborted={"server": "webstore", "tool": "orders_get", "reason": "expect: id"})

    model = FakeChat(script=[AIMessage(content="", tool_calls=[call("t_x", {"q": "1"}, "c1")]), "nikdy"])
    out, trace = await run(model, [FakeProvider([tdef("t_x")], handler)])
    assert out.aborted == {"server": "webstore", "tool": "orders_get", "reason": "expect: id"}
    assert out.error is None and len(model.calls) == 1
    lines = rows(trace)
    assert any(r["type"] == "system" and r["subtype"] == "skilleval_aborted" for r in lines)
    tr = [r for r in lines if r.get("skilleval_source") == "mock"]
    assert tr and tr[0]["message"]["content"][0]["is_error"] is True


async def test_long_tool_names_aliased_full_names_in_trace():
    full = "mcp__plugin_webstore-core_webstore__" + "analytics_documents_consistency_with_a_very_long_suffix"
    assert len(full) > 64
    dotted = "mcp__plugin_x.y_srv__tool"  # neplatný znak → alias i pod limitem
    prov = FakeProvider([tdef(full, "mcp"), tdef(dotted, "mcp"), tdef("Read")])

    def first(messages: list[BaseMessage], m: FakeChat) -> AIMessage:
        names = [t["function"]["name"] for t in m.tools]
        assert all(len(n) <= 64 and VALID_NAME_RE.match(n) for n in names)
        assert "Read" in names and full not in names and dotted not in names
        return AIMessage(content="", tool_calls=[call(m.tool_alias(full), {"q": "x"}, "c1"),
                                                 call(m.tool_alias(dotted), {}, "c2")])

    model = FakeChat(script=[first, "ok"])
    out, trace = await run(model, [prov])
    assert [c[0] for c in prov.calls] == [full, dotted]
    assert [r.name for r in out.tool_calls] == [full, dotted]
    uses = [r["message"]["content"][0] for r in rows(trace) if r["type"] == "assistant"
            and r["message"]["content"][0]["type"] == "tool_use"]
    assert [u["name"] for u in uses] == [full, dotted]


def test_alias_unique_and_stable():
    a = "mcp__plugin_p_s__" + "x" * 80
    b = "mcp__plugin_p_s__" + "x" * 79 + "y"
    taken: set[str] = set()
    aa = make_alias(a, 64, taken)
    taken.add(aa)
    bb = make_alias(b, 64, taken)
    assert aa != bb and len(aa) <= 64 and len(bb) <= 64
    assert make_alias(a, 64, set()) == aa
    idx = ToolIndex.build([FakeProvider([tdef(a), tdef(b), tdef(a)])], 64)
    assert len(idx.defs) == 2 and idx.full_name(idx.alias(a)) == a and idx.full_name(idx.alias(b)) == b


async def test_unknown_tool_and_tool_error_wrapped():
    async def handler(name: str, args: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(content="conflict: version 12 → 13", is_error=True)

    model = FakeChat(script=[AIMessage(content="", tool_calls=[call("Nope", {}, "c1"), call("t_x", {}, "c2")]), "konec"])
    out, trace = await run(model, [FakeProvider([tdef("t_x")], handler)])
    tms = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert tms[0].content == "<tool_use_error>No such tool available: Nope</tool_use_error>" and tms[0].status == "error"
    assert tms[1].content == "<tool_use_error>conflict: version 12 → 13</tool_use_error>"
    assert out.tool_calls[0].is_error and out.tool_calls[0].source == "denied"
    assert out.tool_calls[1].output == "conflict: version 12 → 13"
    blocks = [r["message"]["content"][0] for r in rows(trace) if r["type"] == "user" and isinstance(r["message"]["content"], list)]
    assert all(b["is_error"] is True for b in blocks)
    assert blocks[1]["content"].startswith("<tool_use_error>")


async def test_provider_exception_is_tool_error_and_images():
    async def handler(name: str, args: dict[str, Any]) -> ToolOutcome:
        if args.get("q") == "boom":
            raise ValueError(f"bad {SECRET}")
        return ToolOutcome(content="obrázek", images=[{"media_type": "image/png", "data": "iVBORw0KGgo="}])

    model = FakeChat(script=[AIMessage(content="", tool_calls=[call("t_x", {"q": "boom"}, "c1"), call("t_x", {}, "c2")]), "ok"])
    out, trace = await run(model, [FakeProvider([tdef("t_x")], handler)])
    assert out.error is None
    assert out.tool_calls[0].is_error and SECRET not in (out.tool_calls[0].output or "")
    tm = [m for m in model.calls[1] if isinstance(m, ToolMessage)][1]
    assert tm.content[0] == {"type": "text", "text": "obrázek"}
    assert tm.content[1]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_invalid_tool_call_and_missing_ids_and_thinking():
    msg = AIMessage(content=[{"type": "thinking", "thinking": "Hmm, nejdřív počasí."}, {"type": "text", "text": "Jdu na to."}],
                    tool_calls=[{"name": "t_x", "args": {"q": "1"}, "id": None, "type": "tool_call"}],
                    invalid_tool_calls=[{"name": "t_x", "args": "{bad", "id": "bad1", "error": "JSON", "type": "invalid_tool_call"}])
    model = FakeChat(script=[msg, "ok"])
    out, trace = await run(model, [FakeProvider([tdef("t_x")])])
    assert len(out.tool_calls) == 2 and out.tool_calls[0].id.startswith("toolu_")
    assert out.tool_calls[1].is_error and "InputValidationError" in (out.tool_calls[1].output or "")
    tms = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tms] == [out.tool_calls[0].id, "bad1"]
    lines = rows(trace)
    assert lines[2]["message"]["content"] == [{"type": "thinking", "thinking": "Hmm, nejdřív počasí."}]
    assert lines[3]["message"]["content"] == [{"type": "text", "text": "Jdu na to."}]


async def test_cost_sums_model_and_tool_costs():
    def priced(text: str, tool_calls: list[dict[str, Any]] | None = None) -> AIMessage:
        return AIMessage(content=text, tool_calls=tool_calls or [],
                         response_metadata={"token_usage": {"cost": 0.001, "prompt_tokens": 100, "completion_tokens": 10,
                                                            "prompt_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 0}},
                                            "id": "gen-1"})

    async def handler(name: str, args: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(content="mock", source="agent-mock", cost_usd=0.0005)

    model = FakeChat(script=[priced("", [call("t_x", {}, "c1")]), priced("hotovo")])
    out, trace = await run(model, [FakeProvider([tdef("t_x")], handler)])
    assert out.cost_usd == pytest.approx(0.0025)
    lines = rows(trace)
    assert lines[2]["message"]["usage"] == {"input_tokens": 40, "output_tokens": 10, "cache_read_input_tokens": 60,
                                            "cache_creation_input_tokens": 0}
    assert lines[2]["message"]["id"] == "gen-1"
    assert lines[-1]["total_cost_usd"] == pytest.approx(0.0025)
    assert lines[-1]["usage"]["cache_read_input_tokens"] == 120


# --------------------------------------------------------------------------------------- simulovaný uživatel


async def test_simulated_user_replies():
    model = FakeChat(script=["Mám vystavit dobropis, nebo storno?", "Dobře, vystavím dobropis.", "nepoužito"])
    spec = SimulatedUserSpec(max_replies=1, replies=["Chci dobropis."])
    out, trace = await run(model, [], simulated_user=spec)
    assert out.turns == 2 and out.final_message == "Dobře, vystavím dobropis."
    assert isinstance(model.calls[1][-1], HumanMessage) and model.calls[1][-1].content == "Chci dobropis."
    lines = rows(trace)
    sim = [r for r in lines if r.get("subtype") == "skilleval_simulated_user"]
    assert sim and sim[0]["reply"] == "Chci dobropis."
    prompts = [r["message"]["content"] for r in lines if r["type"] == "user" and isinstance(r["message"]["content"], str)]
    assert prompts == ["Jaké je počasí v Praze?", "Chci dobropis."]


async def test_simulated_user_budget_exhausted_and_no_question():
    model = FakeChat(script=["Otázka 1?", "Otázka 2?", "x"])
    out, _ = await run(model, [], simulated_user=SimulatedUserSpec(max_replies=1, replies=["a"]))
    assert out.turns == 2 and out.final_message == "Otázka 2?"
    model = FakeChat(script=["Hotovo, faktura vystavena.", "x"])
    out, _ = await run(model, [], simulated_user=SimulatedUserSpec(max_replies=2, replies=["a"]))
    assert out.turns == 1


class FakeJudge:
    def __init__(self, answer: str = "Chci dobropis, děkuji."):
        self.answer = answer
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    async def vote(self, system: str, user: str, images: Any = None) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def complete(self, system: str, messages: list[dict[str, str]]) -> tuple[str, float]:
        self.calls.append((system, messages))
        return self.answer, 0.002


async def test_simulated_user_judge_and_shared_hook():
    judge = FakeJudge()
    spec = SimulatedUserSpec(max_replies=2, agent="Jsi účetní, chceš vždy dobropis.")
    hook = make_ask_user_hook(spec, judge)
    out = await hook({"questions": [{"question": "Dobropis nebo storno?", "header": "Doklad",
                                     "options": [{"label": "Dobropis"}, {"label": "Storno"}], "multiSelect": False}]})
    assert out.startswith('User has answered your questions: "Dobropis nebo storno?"="Chci dobropis, děkuji."')
    assert "Jsi účetní" in judge.calls[0][0] and "- Dobropis" in judge.calls[0][1][0]["content"]

    model = FakeChat(script=["Mám pokračovat?", "Hotovo.", "x"])
    res, trace = await run(model, [], simulated_user=hook, judge=judge)
    assert res.turns == 2
    assert "ASSISTANT: Mám pokračovat?" in judge.calls[1][1][0]["content"]
    assert res.cost_usd == pytest.approx(0.004)  # obě repliky soudce
    # rozpočet vyčerpán → AskUserQuestion vrací neinteraktivní odpověď
    assert await hook({"questions": [{"question": "Ještě něco?"}]}) == NOT_AVAILABLE


@pytest.mark.parametrize("text,expected", [
    ("Chcete vystavit dobropis?", True),
    ("Která varianta?\n\n**Poznámka:** x", True),
    ("Mohu pokračovat? **", True),
    ("Let me know which option you prefer.", True),
    ("Faktura 86177 byla vystavena.", False),
    ("Nemám přístup k objednávce.", False),
    ("", False),
])
def test_looks_like_question(text: str, expected: bool):
    assert looks_like_question(text) is expected


# ------------------------------------------------------------------------------------ historie, prompt caching


async def test_history_prefix_alias_and_repair():
    full = "mcp__plugin_webstore-core_webstore__" + "orders_get_extremely_long_name_for_alias_testing"
    history: list[BaseMessage] = [
        HumanMessage(content="Najdi objednávku 71781"),
        AIMessage(content="", tool_calls=[call(full, {"id": 71781}, "h1"), call("Read", {}, "h2")]),
        ToolMessage(content="{...}", tool_call_id="h1"),
        ToolMessage(content="osiřelý", tool_call_id="zzz"),
        AIMessage(content="Objednávka nalezena."),
    ]
    model = FakeChat(script=["ok"])
    trace = new_trace()
    await run(model, [FakeProvider([tdef(full, "mcp")])], history=history, trace=trace)
    sent = model.calls[0]
    assert isinstance(sent[0], SystemMessage) and sent[1].content == "Najdi objednávku 71781"
    ai = sent[2]
    assert isinstance(ai, AIMessage) and ai.tool_calls[0]["name"] == model.tools[0]["function"]["name"] != full
    tms = [m for m in sent if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tms] == ["h1", "h2"] and tms[1].status == "error"
    assert sent[-1].content == "Jaké je počasí v Praze?"
    # historie se do trace nepíše – jen prompt
    assert [r["type"] for r in rows(trace)][:2] == ["system", "user"]


def test_system_message_cache_control():
    m = system_message("SYS", "anthropic/claude-haiku-5.5")
    assert m.content == [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    assert system_message("SYS", "openai/gpt-5").content == "SYS"


def test_openrouter_model_payload_and_reasoning():
    from langchain_openai import ChatOpenAI

    from skilleval.agent.model import OpenRouterChatOpenAI, upgrade_model

    m = upgrade_model(ChatOpenAI(model="anthropic/claude-haiku-5.5", api_key="x", base_url="http://localhost:1"))
    assert isinstance(m, OpenRouterChatOpenAI)
    rd = [{"type": "reasoning.text", "text": "t", "signature": "s"}]
    msgs = [system_message("SYS", "anthropic/claude-haiku-5.5"), HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[call("t_x", {}, "c1")], additional_kwargs={"reasoning_details": rd}),
            ToolMessage(content="výsledek", tool_call_id="c1")]
    payload = m._get_request_payload(msgs)
    out = payload["messages"]
    assert out[0]["content"] == [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    assert out[2]["reasoning_details"] == rd
    assert out[3]["content"] == [{"type": "text", "text": "výsledek", "cache_control": {"type": "ephemeral"}}]
    assert out[1]["content"] == "hi"

    res = m._create_chat_result({"id": "gen-1", "model": "x", "choices": [{"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": "ahoj", "reasoning": "přemýšlím", "reasoning_details": rd}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.1}})
    ai = res.generations[0].message
    assert ai.additional_kwargs["reasoning"] == "přemýšlím" and ai.additional_kwargs["reasoning_details"] == rd

    # ne-Anthropic model: bez cache breakpointu
    o = upgrade_model(ChatOpenAI(model="openai/gpt-5", api_key="x", base_url="http://localhost:1"))
    assert o._get_request_payload([HumanMessage(content="a"), ToolMessage(content="b", tool_call_id="c")])["messages"][-1]["content"] == "b"
