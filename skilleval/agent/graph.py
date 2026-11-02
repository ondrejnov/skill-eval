"""Smyčka agenta pod testem (LangGraph `StateGraph`): agent ↔ tools (+ simulovaný uživatel), trace stream-json.

- Tah = jedno volání modelu; před každým voláním kontrola `max_turns` (dosažení = error, run se hodnotí).
- Tool calls jedné odpovědi běží souběžně, výsledky (ToolMessage i trace) v pořadí volání.
- Názvy toolů pro model mohou být aliasy (dlouhé/neplatné názvy); trace a ToolCallRecord mají vždy plný název.
- Stav runu (trace, tool_calls, náklady, chyba) žije v `_Run` mimo graf, takže timeout ani chyba nic neztratí.
"""

from __future__ import annotations

import asyncio
import operator
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from skilleval import llm
from skilleval.agent.deferred import TOOL_SEARCH, ToolSearch, should_defer
from skilleval.agent.model import CACHE_CONTROL, aclose_model, is_anthropic, upgrade_model
from skilleval.agent.simulated_user import SimulatedUser, looks_like_question
from skilleval.agent.tools import ToolIndex
from skilleval.config import Settings
from skilleval.harness.prompt import deferred_tools_section
from skilleval.models import Judge, SimulatedUserSpec, ToolCallRecord, ToolDef, ToolOutcome, ToolProvider
from skilleval.trace import TraceWriter

NOT_EXECUTED = "Tool call was not executed."
INVALID_ARGS = "InputValidationError: tool arguments are not a valid JSON object"


@dataclass
class AgentOutcome:
    final_message: str
    tool_calls: list[ToolCallRecord]
    turns: int
    error: str | None
    aborted: dict | None
    api_error: bool
    cost_usd: float


class _State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    stop: bool


@dataclass
class _Run:
    settings: Settings
    slug: str
    model: Any  # model s navázanými tooly (aktuální sada)
    system: SystemMessage
    index: ToolIndex
    trace: TraceWriter
    max_turns: int
    sim: SimulatedUser | None = None
    base_model: Any = None  # model bez toolů
    search: ToolSearch | None = None  # odložené tooly (None = neodkládá se)
    bound_key: tuple[str, ...] | None = None  # načtené odložené tooly, pro které je `model` nabindovaný
    trace_content: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # call id → bloky do trace
    turns: int = 0
    cost_usd: float = 0.0
    final_text: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    error: str | None = None
    aborted: dict | None = None
    api_error: bool = False
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0,
                                                           "cache_read_input_tokens": 0,
                                                           "cache_creation_input_tokens": 0})


# ------------------------------------------------------------------------------------------------- pomocné


def message_text(content: Any) -> str:
    """Text odpovědi z `AIMessage.content` (str nebo seznam bloků; thinking/reasoning bloky se vynechají)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text", "")))
        return "".join(parts)
    return ""


def message_thinking(msg: AIMessage) -> str | None:
    """Reasoning modelu: OpenRouter `reasoning` (OpenRouterChatOpenAI) nebo bloky thinking/reasoning v content."""
    r = msg.additional_kwargs.get("reasoning") or msg.additional_kwargs.get("reasoning_content")
    if isinstance(r, str) and r.strip():
        return r.strip()
    if isinstance(msg.content, list):
        parts = [str(b.get("thinking") or b.get("reasoning") or b.get("text") or "") for b in msg.content
                 if isinstance(b, dict) and b.get("type") in ("thinking", "reasoning")]
        text = "\n".join(p for p in parts if p).strip()
        return text or None
    return None


def _usage(msg: AIMessage) -> dict[str, int]:
    """Usage ve tvaru Anthropic/Claude Code (input bez cache, cache read/creation zvlášť)."""
    meta = msg.response_metadata or {}
    tu = meta.get("token_usage") or meta.get("usage") or {}
    if isinstance(tu, dict) and tu.get("prompt_tokens") is not None:
        details = tu.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        written = int(details.get("cache_write_tokens") or 0)
        return {"input_tokens": max(0, int(tu.get("prompt_tokens") or 0) - cached - written),
                "output_tokens": int(tu.get("completion_tokens") or 0),
                "cache_read_input_tokens": cached, "cache_creation_input_tokens": written}
    um = msg.usage_metadata or {}
    if um:
        details = um.get("input_token_details") or {}
        cached, written = int(details.get("cache_read") or 0), int(details.get("cache_creation") or 0)
        return {"input_tokens": max(0, int(um.get("input_tokens") or 0) - cached - written),
                "output_tokens": int(um.get("output_tokens") or 0),
                "cache_read_input_tokens": cached, "cache_creation_input_tokens": written}
    return {}


def _tool_error(text: str) -> str:
    text = text or ""
    return text if text.startswith("<tool_use_error>") else f"<tool_use_error>{text}</tool_use_error>"


def _truncate(text: str, limit: int) -> str:
    if limit > 0 and len(text) > limit:
        return text[:limit] + f"\n\n[… output truncated: {len(text) - limit} more characters]"
    return text


def _tool_message(outcome: ToolOutcome, call_id: str, name: str, limit: int) -> tuple[ToolMessage, str]:
    """ToolMessage pro model + text do trace (stejný, co vidí model)."""
    text = _truncate(outcome.content or "", limit)
    if outcome.is_error:
        text = _tool_error(text)
    content: Any = text
    if outcome.images:
        content = [{"type": "text", "text": text}] + [
            {"type": "image_url", "image_url": {"url": f"data:{im.get('media_type', 'image/png')};base64,{im['data']}"}}
            for im in outcome.images if im.get("data")]
    return ToolMessage(content=content, tool_call_id=call_id, name=name,
                       status="error" if outcome.is_error else "success"), text


def prepare_history(history: list[BaseMessage], index: ToolIndex) -> list[BaseMessage]:
    """Historie pro model: názvy toolů → aliasy, doplnění chybějících výsledků, vyřazení osiřelých ToolMessage."""
    out: list[BaseMessage] = []
    pending: list[str] = []

    def flush() -> None:
        for cid in pending:
            out.append(ToolMessage(content=_tool_error(NOT_EXECUTED), tool_call_id=cid, status="error"))
        pending.clear()

    for m in history:
        if isinstance(m, ToolMessage):
            if m.tool_call_id in pending:
                pending.remove(m.tool_call_id)
                out.append(m)
            continue
        flush()
        if isinstance(m, AIMessage) and m.tool_calls:
            calls = [{**tc, "name": index.alias(tc["name"]), "id": tc.get("id") or _new_id("toolu_")}
                     for tc in m.tool_calls]
            out.append(AIMessage(content=m.content, tool_calls=calls))
            pending.extend(tc["id"] for tc in calls)
        else:
            out.append(m)
    flush()
    return out


def _new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:24]


def _run(config: RunnableConfig) -> _Run:
    return config["configurable"]["skilleval_run"]


# ------------------------------------------------------------------------------------------------------ uzly


def _bind(run: _Run) -> None:
    """Naváže tooly na model; s odloženými tooly jen nenačtené vynechá a po ToolSearch přebinduje."""
    key = tuple(run.search.loaded) if run.search is not None else ()
    if run.model is not None and key == run.bound_key:
        return
    specs = run.index.specs(run.search.is_hidden if run.search is not None else None)
    run.model = run.base_model.bind_tools(specs) if specs else run.base_model
    run.bound_key = key


async def _agent_node(state: _State, config: RunnableConfig) -> dict[str, Any]:
    run = _run(config)
    if run.turns >= run.max_turns:
        run.error = f"max turns reached ({run.max_turns})"
        return {"stop": True}
    run.turns += 1
    _bind(run)
    raw = await run.model.ainvoke([run.system, *state["messages"]])
    if not isinstance(raw, AIMessage):
        raw = AIMessage(content=getattr(raw, "content", str(raw)))
    run.cost_usd += llm.message_cost(run.settings, run.slug, raw)
    usage = _usage(raw)
    for k, v in usage.items():
        run.usage[k] = run.usage.get(k, 0) + v

    calls = [{**tc, "id": tc.get("id") or _new_id("toolu_")} for tc in raw.tool_calls]
    invalid = [{**tc, "id": tc.get("id") or _new_id("toolu_")} for tc in raw.invalid_tool_calls]
    msg = AIMessage(content=raw.content, tool_calls=calls, invalid_tool_calls=invalid,
                    additional_kwargs=raw.additional_kwargs, response_metadata=raw.response_metadata,
                    usage_metadata=raw.usage_metadata, id=raw.id)
    text = message_text(msg.content)
    if text.strip():
        run.final_text = text
        if run.sim is not None:
            run.sim.conversation.append({"role": "assistant", "content": text})

    tool_uses = []
    for tc in calls:
        full = run.index.full_name(tc["name"])
        tool_uses.append({"id": tc["id"], "name": full, "input": tc.get("args") or {}})
        run.tool_calls.append(ToolCallRecord(id=tc["id"], name=full, input=tc.get("args") or {}))
    for tc in invalid:
        full = run.index.full_name(tc.get("name") or "")
        tool_uses.append({"id": tc["id"], "name": full, "input": {}})
        run.tool_calls.append(ToolCallRecord(id=tc["id"], name=full, input={}))
    meta = raw.response_metadata or {}
    msg_id = str(meta.get("id") or raw.id or _new_id("msg_"))
    run.trace.assistant(msg_id=msg_id, text=text or None, tool_uses=tool_uses, usage=usage,
                        thinking=message_thinking(raw))
    return {"messages": [msg]}


async def _call_tool(run: _Run, tc: dict[str, Any], valid: bool) -> ToolOutcome:
    full = run.index.full_name(tc.get("name") or "")
    if not valid:
        detail = tc.get("error") or ""
        return ToolOutcome(content=f"{INVALID_ARGS}{f' ({detail})' if detail else ''}", is_error=True,
                           source="denied")
    provider = run.index.providers.get(full)
    if provider is not None and provider is run.search:
        outcome, refs = run.search.run(tc.get("args") or {}, run.index.to_model)
        if refs:
            run.trace_content[tc["id"]] = refs
        return outcome
    if provider is None:
        return ToolOutcome(content=f"No such tool available: {full}", is_error=True, source="denied")
    try:
        return await provider.call(full, tc.get("args") or {})
    except asyncio.CancelledError:
        raise
    except Exception as ex:  # chyba provideru = tool error, run pokračuje
        return ToolOutcome(content=run.settings.mask(f"Error: {type(ex).__name__}: {ex}"), is_error=True)


async def _tools_node(state: _State, config: RunnableConfig) -> dict[str, Any]:
    run = _run(config)
    last = state["messages"][-1]
    assert isinstance(last, AIMessage)
    calls = [(tc, True) for tc in last.tool_calls] + [(tc, False) for tc in last.invalid_tool_calls]
    records = {r.id: r for r in run.tool_calls}
    results: list[ToolOutcome | None] = [None] * len(calls)
    out: list[BaseMessage] = []
    emitted = 0

    async def one(i: int) -> None:
        tc, valid = calls[i]
        results[i] = await _call_tool(run, tc, valid)

    def emit_ready() -> None:
        """Výsledky v pořadí volání – do trace a zpráv jen souvislý hotový prefix."""
        nonlocal emitted
        while emitted < len(calls) and results[emitted] is not None:
            tc, _ = calls[emitted]
            outcome = results[emitted]
            assert outcome is not None
            tm, text = _tool_message(outcome, tc["id"], tc.get("name") or "", run.settings.max_tool_result_chars)
            out.append(tm)
            run.trace.tool_result(tool_use_id=tc["id"], content=run.trace_content.pop(tc["id"], None) or text,
                                  is_error=outcome.is_error, source=outcome.source)
            run.cost_usd += outcome.cost_usd or 0.0
            if rec := records.get(tc["id"]):
                rec.output, rec.is_error, rec.source = outcome.content, outcome.is_error, outcome.source
            emitted += 1

    tasks = [asyncio.create_task(one(i)) for i in range(len(calls))]
    aborted: dict | None = None
    try:
        pending: set[asyncio.Task] = set(tasks)
        while pending:
            _done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in _done:
                if (ex := t.exception()) is not None:
                    raise ex
            emit_ready()
            aborted = next((r.aborted for r in results if r is not None and r.aborted), None)
            if aborted:
                break
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        if any(not t.done() for t in tasks):
            await asyncio.gather(*tasks, return_exceptions=True)
        emit_ready()

    if aborted:
        run.aborted = aborted
        run.trace.system("skilleval_aborted", **aborted)
        return {"messages": out, "stop": True}
    return {"messages": out}


async def _user_node(state: _State, config: RunnableConfig) -> dict[str, Any]:
    run = _run(config)
    assert run.sim is not None
    reply = await run.sim.reply(run.sim.conversation)
    if reply is None:
        return {"stop": True}
    run.trace.system("skilleval_simulated_user", reply=reply)
    run.trace.user_prompt(reply)
    run.sim.conversation.append({"role": "user", "content": reply})
    return {"messages": [HumanMessage(content=reply)]}


def _after_agent(state: _State, config: RunnableConfig) -> str:
    if state.get("stop"):
        return END
    last = state["messages"][-1] if state["messages"] else None
    if not isinstance(last, AIMessage):
        return END
    if last.tool_calls or last.invalid_tool_calls:
        return "tools"
    run = _run(config)
    if run.sim is not None and run.sim.can_reply() and looks_like_question(message_text(last.content)):
        return "user"
    return END


def _after_step(state: _State) -> str:
    return END if state.get("stop") else "agent"


def build_graph():
    g = StateGraph(_State)
    g.add_node("agent", _agent_node)
    g.add_node("tools", _tools_node)
    g.add_node("user", _user_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", _after_agent, {"tools": "tools", "user": "user", END: END})
    g.add_conditional_edges("tools", _after_step, {"agent": "agent", END: END})
    g.add_conditional_edges("user", _after_step, {"agent": "agent", END: END})
    return g.compile()


_GRAPH = build_graph()


# ------------------------------------------------------------------------------------------------------- run


def system_message(system_prompt: str, slug: str) -> SystemMessage:
    """U Anthropic modelů breakpoint prompt cache na konci system promptu (cachuje definice toolů + system)."""
    if is_anthropic(slug):
        return SystemMessage(content=[{"type": "text", "text": system_prompt, "cache_control": CACHE_CONTROL}])
    return SystemMessage(content=system_prompt)


async def run_agent(*, settings: Settings, model_slug: str, system_prompt: str, prompt: str,
                    providers: list[ToolProvider], trace: TraceWriter, max_turns: int, timeout_seconds: int,
                    history: list[BaseMessage] | None = None,
                    simulated_user: SimulatedUserSpec | SimulatedUser | None = None, judge: Judge | None = None,
                    model: Any = None, defer_tools: bool | None = None) -> AgentOutcome:
    """Jeden run agenta. Chyby nevyhazuje (kromě zrušení zvenku) – vrací je v `AgentOutcome.error`.

    `defer_tools`: True/False vynutí odkládání MCP toolů (ToolSearch), None = `settings.defer_tools` (auto podle
    odhadu tokenů). Když system prompt neobsahuje `<available-deferred-tools>`, sekce se doplní na konec."""
    started = time.monotonic()
    all_defs = [d for p in providers for d in p.tool_defs]
    search: ToolSearch | None = None
    if should_defer(settings, all_defs, defer_tools):
        first: dict[str, ToolDef] = {}  # stejný název u víc providerů → první vyhrává (jako ToolIndex)
        for d in all_defs:
            first.setdefault(d.name, d)
        deferred = [d for d in first.values() if d.kind == "mcp"]
        search = ToolSearch(deferred)
        providers = [*providers, search]
        if "<available-deferred-tools>" not in system_prompt:
            system_prompt = system_prompt.rstrip("\n") + "\n" + deferred_tools_section([d.name for d in deferred])
        trace.add_init_tools([TOOL_SEARCH])
    index = ToolIndex.build(providers, settings.max_tool_name_len)
    if search is not None:
        search.aliases = index.to_full
    if isinstance(simulated_user, SimulatedUser):
        sim: SimulatedUser | None = simulated_user
    elif simulated_user is not None:
        sim = SimulatedUser(simulated_user, judge)
    else:
        sim = None
    if sim is not None:
        sim.trace = trace
        sim.conversation.append({"role": "user", "content": prompt})

    run = _Run(settings=settings, slug=model_slug, model=None, system=system_message(system_prompt, model_slug),
               index=index, trace=trace, max_turns=max_turns, sim=sim, search=search)
    messages = prepare_history(list(history or []), index) + [HumanMessage(content=prompt)]
    trace.user_prompt(prompt)

    cm = asyncio.timeout(timeout_seconds)
    own_model: Any = None  # model vytvořený zde (zavřít HTTP klienta)
    try:
        async with cm:
            if model is None:
                model = own_model = upgrade_model(llm.make_chat_model(settings, model_slug))
            run.base_model = model
            cfg: RunnableConfig = {"configurable": {"skilleval_run": run},
                                   "recursion_limit": max_turns * 3 + 20}
            await _GRAPH.ainvoke({"messages": messages, "stop": False}, config=cfg)
    except TimeoutError:
        if cm.expired():
            run.error = f"timed out after {timeout_seconds}s"
        else:
            run.error = "internal error: TimeoutError"
    except Exception as ex:
        if llm.is_api_error(ex):
            run.api_error = True
            run.error = settings.mask(str(ex) or type(ex).__name__)
        else:
            run.error = settings.mask(f"internal error: {type(ex).__name__}: {ex}")
    finally:
        if own_model is not None:
            await aclose_model(own_model)

    cost = run.cost_usd + (sim.cost_usd if sim is not None else 0.0)
    trace.result(final_text=run.final_text, turns=run.turns, cost_usd=cost,
                 duration_ms=int((time.monotonic() - started) * 1000), error=run.error, usage=dict(run.usage))
    return AgentOutcome(final_message=run.final_text, tool_calls=run.tool_calls, turns=run.turns, error=run.error,
                        aborted=run.aborted, api_error=run.api_error, cost_usd=cost)


__all__ = ["AgentOutcome", "build_graph", "message_text", "prepare_history", "run_agent", "system_message"]
