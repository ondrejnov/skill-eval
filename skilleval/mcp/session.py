"""MCP tooly jednoho runu: mocky (fixed/agent/replay), reálné servery, granty a fault injection.

Režimy (`--mocks`):
- `record` – reálné servery se nestartují; tool s mock souborem odpovídá z mocku (grant netřeba), tool bez mocku
  není dostupný, server bez mocků → `mocked: plugin_<p>_<s>[not started: no mock]`. S `allow_real` se pro
  nemockované tooly/servery navíc spustí reálný server (hybrid po toolech).
- `off` – mocky se ignorují, všechno reálně.
Reálné tooly jsou povolené všechny (rozšíření skilleval); `strict_grants` = jen ty, které matchují `allow_tools`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from skilleval.config import Settings
from skilleval.harness.sandbox import run_hook
from skilleval.mcp.client import RealServer, ServerError, convert_result, describe_exception
from skilleval.mcp.expect import check_expect
from skilleval.mcp.faults import FaultEngine, FaultHit
from skilleval.mcp.mocks import MockError, MockSet, MockSpec, render_body, write_recording
from skilleval.mcp.naming import match_any, mcp_tool_name, server_label
from skilleval.models import CaseSpec, Judge, McpCallRecord, McpServerConfig, ToolDef, ToolOutcome

BUDGET_EXCEEDED = "mock call budget exceeded"
AGENT_MOCK_RULES = """\
You are acting as the MCP server "{server}" and are answering a call to its tool "{tool}" during an automated
evaluation of an AI agent. The instructions above describe how this server behaves. Earlier calls to this server in
the same session are given as conversation history; keep your answers consistent with them.
Reply with ONLY the tool result exactly as the real server would return it - no commentary, no explanations, no
markdown code fences unless the real server would return them.
If the call must fail the way the real server would fail, start your reply with "ERROR:" followed by the error
message."""
ABORT_RULES = """
You may abort the whole evaluation run ONLY under these conditions:
{abort_when}
To abort, reply with a single line "ABORT: <reason>". Never abort for any other reason."""
NO_ABORT_RULES = "\nNever abort the run; always answer as the server."

OnEvent = Callable[[str, dict[str, Any]], None]


@dataclass
class _Entry:
    name: str  # název nabízený modelu (= plný název jako v Claude Code)
    full_name: str
    server: str
    tool: str
    plugin: str
    kind: Literal["mock", "real"]
    description: str
    input_schema: dict[str, Any]


class McpSession:
    """Implementace `ToolProvider` pro MCP tooly pluginů; async context manager (spojení na dobu runu)."""

    def __init__(self, servers: list[McpServerConfig], *, case: CaseSpec, mode: Literal["record", "off"],
                 allow_real: bool, allow_tools: list[str], strict_grants: bool, judge: Judge | None,
                 settings: Settings, hooks: bool, recordings_dir: Path | None, run_env: dict[str, str],
                 on_event: OnEvent | None = None, default_plugin: str | None = None,
                 connect_timeout: float = 30.0, call_timeout: float = 300.0):
        self.servers = {s.name: s for s in servers}
        self.case = case
        self.mode = mode
        self.allow_real = allow_real
        self.allow_tools = list(allow_tools)
        self.strict_grants = strict_grants
        self.judge = judge
        self.settings = settings
        self.hooks = hooks
        self.recordings_dir = recordings_dir
        self.run_env = dict(run_env)
        self.on_event = on_event
        self.default_plugin = default_plugin
        self.connect_timeout = connect_timeout
        self.call_timeout = call_timeout

        self.mocks = MockSet(case.mock_dirs if mode == "record" else [])
        self.faults = FaultEngine(case.extension.faults if case.extension else None)
        self.budget = 4 * case.max_turns
        self.agent_calls = 0
        self.tool_defs: list[ToolDef] = []
        self.instructions: dict[str, str] = {}
        self.calls: list[McpCallRecord] = []
        self.notices: list[str] = []
        self.mock_cost_usd = 0.0
        self.recordings: list[Path] = []
        self._entries: dict[str, _Entry] = {}
        self._real: dict[str, RealServer] = {}
        self._secrets = self._collect_secrets()
        self._hook_skip_noticed = False

    # --- životní cyklus -----------------------------------------------------------------------------------------------

    async def __aenter__(self) -> McpSession:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def _collect_secrets(self) -> list[str]:
        out = []
        for cfg in self.servers.values():
            for v in list(cfg.headers.values()) + list(cfg.env.values()):
                if isinstance(v, str) and len(v) >= 8:
                    out.append(v)
                    if " " in v:  # "Bearer <token>" → i samotný token
                        out.append(v.split(" ", 1)[1])
        return sorted(set(out), key=len, reverse=True)

    def mask(self, text: str) -> str:
        for s in self._secrets:
            text = text.replace(s, "***")
        return self.settings.mask(text)

    def _notice(self, text: str) -> None:
        text = self.mask(text)
        if text not in self.notices:
            self.notices.append(text)

    def _event(self, kind: str, data: dict[str, Any]) -> None:
        if self.on_event:
            self.on_event(kind, data)

    async def start(self) -> None:
        for e in self.mocks.errors:
            self._notice(f"mock error: {e}")
        names = list(self.servers)
        for s in sorted(self.mocks.servers()):
            if s not in self.servers:
                if self.default_plugin:
                    names.append(s)
                else:
                    self._notice(f"mocks/{s}: no plugin loaded in this run, mocks ignored")

        # reálné servery – paralelní start
        to_start: list[str] = []
        for name in names:
            cfg = self.servers.get(name)
            if cfg is None:
                continue
            mocked = self.mocks.tool_names(name)
            if self.mode == "off" or (self.allow_real and (cfg is not None)):
                to_start.append(name)
            elif not mocked:
                self._notice(f"mocked: {server_label(cfg.plugin, name)}[not started: no mock]")
        results = await asyncio.gather(*(self._start_real(n) for n in to_start))
        started = {n for n, ok in zip(to_start, results, strict=True) if ok}

        not_granted: list[str] = []
        for name in names:
            cfg = self.servers.get(name)
            plugin = cfg.plugin if cfg else (self.default_plugin or "")
            mocked = self.mocks.tools_for(name)
            real = self._real.get(name) if name in started else None
            real_tools = {t.name: t for t in real.tools} if real else {}
            for mt in mocked:
                rt = real_tools.get(mt.name)
                listed = self.mocks.listed_tool(name, mt.name)
                desc = mt.description if listed or rt is None else (rt.description or "")
                schema = mt.input_schema if listed or rt is None else dict(rt.inputSchema)
                self._add(name, mt.name, plugin, "mock", desc, schema)
            mocked_names = {mt.name for mt in mocked}
            n_real = 0
            for rt in real_tools.values():
                if rt.name in mocked_names:
                    continue
                full = mcp_tool_name(plugin, name, rt.name)
                if self.strict_grants and not match_any(self.allow_tools, full):
                    not_granted.append(full)
                    continue
                self._add(name, rt.name, plugin, "real", rt.description or "", dict(rt.inputSchema))
                n_real += 1
            if mocked:
                label = server_label(plugin, name)
                extra = f" + {n_real} real" if n_real else ""
                self._notice(f"mocked: {label}[{len(mocked)} tool{'s' if len(mocked) != 1 else ''}{extra}]")
        if not_granted:
            shown = ", ".join(not_granted[:10]) + (f" … (+{len(not_granted) - 10})" if len(not_granted) > 10 else "")
            self._notice(f"not granted: {shown}")

    async def _start_real(self, name: str) -> bool:
        cfg = self.servers[name]
        srv = RealServer(cfg, env=self.run_env, connect_timeout=self.connect_timeout, call_timeout=self.call_timeout)
        try:
            await srv.start()
        except ServerError as ex:
            self._notice(f"server {name} failed: {ex}")
            return False
        except Exception as ex:  # noqa: BLE001
            self._notice(f"server {name} failed: {describe_exception(ex)}")
            return False
        self._real[name] = srv
        if srv.instructions:
            self.instructions[name] = srv.instructions
        return True

    def _add(self, server: str, tool: str, plugin: str, kind: Literal["mock", "real"], description: str,
             schema: dict[str, Any]) -> None:
        # plný název jako v CC; aliasy pro limit délky názvu v API řeší agent (skilleval.agent.tools.ToolIndex)
        full = name = mcp_tool_name(plugin, server, tool)
        if name in self._entries:
            return
        if not isinstance(schema, dict) or schema.get("type") != "object":
            schema = {"type": "object", "properties": {}, "additionalProperties": True}
        e = _Entry(name, full, server, tool, plugin, kind, description, schema)
        self._entries[name] = e
        self.tool_defs.append(ToolDef(name=name, description=description, input_schema=schema, kind="mcp",
                                      server=server, tool=tool, plugin=plugin))

    async def close(self) -> None:
        real, self._real = self._real, {}
        await asyncio.gather(*(s.close() for s in real.values()), return_exceptions=True)

    # --- volání -------------------------------------------------------------------------------------------------------

    def _finish(self, e: _Entry, args: dict[str, Any], out: ToolOutcome) -> ToolOutcome:
        out.content = self.mask(out.content)
        limit = self.settings.max_tool_result_chars
        if limit and len(out.content) > limit:
            cut = len(out.content) - limit
            out.content = out.content[:limit] + f"\n… [truncated {cut} chars]"
        self.calls.append(McpCallRecord(server=e.server, tool=e.tool, input=args, output=out.content,
                                        is_error=out.is_error, source=out.source))
        return out

    async def call(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        e = self._entries.get(name)
        if e is None:
            return ToolOutcome(f"Error: No such tool available: {name}", is_error=True, source="denied")
        args = dict(args or {})

        runs, response = self.faults.on_call(tool=e.tool, full_name=e.full_name, input=args)
        for hit in runs:
            await self._fault_run(e, hit)
        if response is not None:
            return self._finish(e, args, self._fault_response(e, hit=response, args=args))

        if e.kind == "mock":
            spec = self.mocks.get(e.server, e.tool)
            assert spec is not None
            out = await self._mock_call(e, spec, args)
        else:
            out = await self._real_call(e, args)
        return self._finish(e, args, out)

    # --- faulty -------------------------------------------------------------------------------------------------------

    async def _fault_run(self, e: _Entry, hit: FaultHit) -> None:
        data = {"tool": e.full_name, "server": e.server, "fault": hit.index, "call": hit.call,
                "command": list(hit.fault.run or [])}
        if not self.hooks:
            self._event("skilleval_fault", {**data, "action": "run_skipped"})
            if not self._hook_skip_noticed:
                self._notice("fault run skipped: --hooks off")
                self._hook_skip_noticed = True
            return
        err = await run_hook(list(hit.fault.run or []), case=self.case, env=self.run_env)
        self._event("skilleval_fault", {**data, "action": "run", "error": self.mask(err) if err else None})
        if err:
            self._notice(f"fault run failed: {err}")

    def _fault_response(self, e: _Entry, *, hit: FaultHit, args: dict[str, Any]) -> ToolOutcome:
        f = hit.fault
        try:
            text = render_body(f.body or "", args, [self.case.dir]).strip("\n")
            is_error = f.error
        except MockError as ex:
            text, is_error = f"Fault body error: {ex}", True
        self._event("skilleval_fault", {"tool": e.full_name, "server": e.server, "fault": hit.index,
                                        "call": hit.call, "action": "response", "is_error": is_error})
        return ToolOutcome(text, is_error=is_error, source="fault")

    # --- mocky --------------------------------------------------------------------------------------------------------

    async def _mock_call(self, e: _Entry, spec: MockSpec, args: dict[str, Any]) -> ToolOutcome:
        violation = check_expect(spec.expect, args)
        if violation:
            src = "agent-mock" if spec.type == "agent" else "mock"
            return ToolOutcome(f"Mock expectation failed: {violation}", is_error=True, source=src,
                               aborted={"server": e.server, "tool": e.tool, "reason": violation})
        if spec.type == "fixed":
            try:
                body = render_body(spec.body, args, self.mocks.fixture_roots(e.server))
            except MockError as ex:
                return ToolOutcome(f"Mock error: {ex}", is_error=True, source="mock")
            return ToolOutcome(body.strip("\n"), is_error=spec.error, source="mock")
        return await self._agent_call(e, spec, args)

    def _history(self, server: str) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        for c in self.calls:
            if c.server != server:
                continue
            msgs.append({"role": "user", "content": json.dumps({"tool": c.tool, "input": c.input},
                                                               ensure_ascii=False)})
            msgs.append({"role": "assistant", "content": ("ERROR: " if c.is_error else "") + c.output})
        return msgs

    def _agent_system(self, e: _Entry, spec: MockSpec) -> str:
        parts = [spec.body.strip(), "---", AGENT_MOCK_RULES.format(server=e.server, tool=e.tool)]
        listed = self.mocks.listed_tool(e.server, e.tool)
        if listed:
            parts.append(f"Tool description: {listed.get('description') or ''}\n"
                         f"Input schema: {json.dumps(listed.get('inputSchema') or {}, ensure_ascii=False)}")
        if spec.tool is None:
            parts.append(f"This server answers these tools: {', '.join(spec.tools)}.")
        parts.append(ABORT_RULES.format(abort_when=spec.abort_when.strip()) if spec.abort_when else NO_ABORT_RULES)
        return "\n\n".join(p for p in parts if p)

    async def _agent_call(self, e: _Entry, spec: MockSpec, args: dict[str, Any]) -> ToolOutcome:
        if self.agent_calls >= self.budget:
            return ToolOutcome(f"Error: {BUDGET_EXCEEDED} ({self.budget} calls)", is_error=True, source="agent-mock",
                               aborted={"server": e.server, "tool": e.tool, "reason": BUDGET_EXCEEDED})
        self.agent_calls += 1
        rec = self.mocks.replay(e.server, e.tool, args)
        if rec is not None:
            return ToolOutcome(str(rec.get("output", "")), is_error=bool(rec.get("is_error")), source="replay")
        if self.judge is None:
            return ToolOutcome("Error: agent mock needs a judge model", is_error=True, source="agent-mock")
        messages = self._history(e.server)
        messages.append({"role": "user", "content": json.dumps({"tool": e.tool, "input": args}, ensure_ascii=False)})
        try:
            text, cost = await self.judge.complete(self._agent_system(e, spec), messages)
        except Exception as ex:  # noqa: BLE001 – chyba soudce = chyba toolu, run pokračuje
            self._notice(f"agent mock {e.server}/{e.tool} failed: {describe_exception(ex)}")
            return ToolOutcome(f"Error: agent mock failed: {describe_exception(ex)}", is_error=True,
                               source="agent-mock")
        self.mock_cost_usd += cost
        text = (text or "").strip()
        if text.startswith("```") and text.endswith("```") and text.count("```") == 2:
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip() if "\n" in text else text.strip("`")
        if spec.abort_when and text.upper().startswith("ABORT:"):
            reason = text[6:].strip().splitlines()[0] if text[6:].strip() else "aborted by agent mock"
            return ToolOutcome(f"Aborted: {reason}", is_error=True, source="agent-mock", cost_usd=cost,
                               aborted={"server": e.server, "tool": e.tool, "reason": reason})
        is_error = text.upper().startswith("ERROR:")
        if is_error:
            text = text[6:].strip()
        if self.recordings_dir is not None:
            self.recordings.append(write_recording(self.recordings_dir, spec=spec, server=e.server, tool=e.tool,
                                                   input=args, output=self.mask(text), is_error=is_error))
        return ToolOutcome(text, is_error=is_error, source="agent-mock", cost_usd=cost)

    def discard_recordings(self) -> None:
        """Run neskončil čistě → nahrávky tohoto runu smazat (do ADOPT.txt patří jen čisté runy)."""
        for p in self.recordings:
            p.unlink(missing_ok=True)
        self.recordings.clear()

    # --- reálné servery -----------------------------------------------------------------------------------------------

    async def _real_call(self, e: _Entry, args: dict[str, Any]) -> ToolOutcome:
        srv = self._real.get(e.server)
        if srv is None:
            return ToolOutcome(f"Error: MCP server {e.server} is not connected", is_error=True, source="real")
        try:
            result = await srv.call(e.tool, args)
        except Exception as ex:  # noqa: BLE001
            return ToolOutcome(f"Error: {describe_exception(ex)}", is_error=True, source="real")
        text, images, is_error = convert_result(result, prefer_structured=self.settings.prefer_structured_content)
        return ToolOutcome(text, is_error=is_error, source="real", images=images)
