"""Trace runu ve formátu stream-json `claude -p` (jedna JSON zpráva na řádek).

Gradery `regex target: trace` a `llm focus: trace` pracují nad `TraceWriter.lines` stejně jako v Claude Code:
JSON bez mezer (separators=(",", ":")), ne-ASCII znaky ponechané, uvozovky uvnitř řetězců escapované (\\").
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + f"{datetime.now(UTC).microsecond // 1000:03d}Z"


class TraceWriter:
    def __init__(self, *, cwd: str, model: str, tools: list[str], mcp_servers: list[dict[str, Any]],
                 skills: list[str], plugins: list[dict[str, str]]):
        self.session_id = str(uuid.uuid4())
        self.model = model
        self.lines: list[str] = []
        self._emit({"type": "system", "subtype": "init", "cwd": cwd, "session_id": self.session_id, "tools": tools,
                    "mcp_servers": mcp_servers, "model": model, "permissionMode": "dontAsk", "skills": skills,
                    "plugins": plugins, "runner": "skilleval"})

    def _emit(self, obj: dict[str, Any]) -> None:
        obj.setdefault("session_id", self.session_id)
        obj.setdefault("uuid", str(uuid.uuid4()))
        self.lines.append(dumps(obj))

    def user_prompt(self, text: str) -> None:
        self._emit({"type": "user", "message": {"role": "user", "content": text}, "timestamp": now_iso()})

    def assistant(self, *, msg_id: str, text: str | None, tool_uses: list[dict[str, Any]],
                  usage: dict[str, Any] | None = None, thinking: str | None = None) -> None:
        """Jedna odpověď modelu. Jako Claude Code: každý content blok na vlastní řádek se stejným message.id."""
        blocks: list[dict[str, Any]] = []
        if thinking:
            blocks.append({"type": "thinking", "thinking": thinking})
        if text:
            blocks.append({"type": "text", "text": text})
        for tu in tool_uses:
            blocks.append({"type": "tool_use", "id": tu["id"], "name": tu["name"], "input": tu["input"]})
        for b in blocks or [{"type": "text", "text": ""}]:
            self._emit({"type": "assistant", "message": {"model": self.model, "id": msg_id, "type": "message",
                                                         "role": "assistant", "content": [b], "usage": usage or {}}})

    def tool_result(self, *, tool_use_id: str, content: str | list[dict[str, Any]], is_error: bool = False,
                    source: str | None = None) -> None:
        """`content` = text, nebo seznam bloků jako CC (např. `tool_reference` u ToolSearch)."""
        block: dict[str, Any] = {"tool_use_id": tool_use_id, "type": "tool_result", "content": content}
        if is_error:
            block["is_error"] = True
        obj: dict[str, Any] = {"type": "user", "message": {"role": "user", "content": [block]},
                               "parent_tool_use_id": None, "timestamp": now_iso()}
        if source and source not in ("builtin", "real"):
            obj["skilleval_source"] = source  # mock | agent-mock | replay | fault | denied
        self._emit(obj)

    def add_init_tools(self, names: list[str]) -> None:
        """Doplní tooly do `tools` init řádku (např. ToolSearch, o kterém rozhodne až agent)."""
        if not self.lines:
            return
        init = json.loads(self.lines[0])
        if init.get("type") != "system" or init.get("subtype") != "init":
            return
        tools = list(init.get("tools") or [])
        added = [n for n in names if n not in tools]
        if added:
            init["tools"] = tools + added
            self.lines[0] = dumps(init)

    def system(self, subtype: str, **data: Any) -> None:
        """Rozšíření skilleval (skilleval_fault, skilleval_simulated_user, skilleval_notice …)."""
        self._emit({"type": "system", "subtype": subtype, **data})

    def result(self, *, final_text: str, turns: int, cost_usd: float, duration_ms: int, error: str | None,
               usage: dict[str, Any] | None = None) -> None:
        self._emit({"type": "result", "subtype": "success" if not error else "error", "is_error": bool(error),
                    "duration_ms": duration_ms, "num_turns": turns, "result": final_text, "total_cost_usd": cost_usd,
                    "usage": usage or {}, "error": error})

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"
