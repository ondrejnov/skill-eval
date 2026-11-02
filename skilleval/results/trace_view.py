"""Načtení trace (stream-json `claude -p`, .jsonl) do časové osy pro report.html."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PREVIEW_CHARS = 1200  # výstup toolu / text: kolik je vidět hned
MAX_CHARS = 30000  # strop jednoho bloku v reportu (víc se zkrátí)
MAX_EVENTS = 2000  # strop událostí jednoho trace


@dataclass
class Clip:
    """Text s náhledem; `full` jen když je delší než náhled."""

    preview: str
    full: str | None = None
    truncated: bool = False  # zkráceno i ve `full` (MAX_CHARS)

    @classmethod
    def of(cls, text: str, preview: int = PREVIEW_CHARS) -> Clip:
        truncated = len(text) > MAX_CHARS
        if truncated:
            text = text[:MAX_CHARS]
        if len(text) <= preview:
            return cls(preview=text, truncated=truncated)
        head = text[:preview].rstrip() + " …" if preview > 0 else ""
        return cls(preview=head, full=text, truncated=truncated)


@dataclass
class TraceEvent:
    kind: str  # init | prompt | text | thinking | tool | system | skilleval | result | raw
    title: str = ""
    body: Clip | None = None
    # tool call
    tool_id: str | None = None
    server: str | None = None
    input_json: Clip | None = None
    output: Clip | None = None
    is_error: bool = False
    source: str | None = None  # mock | agent-mock | replay | fault | denied (jen ne-reálné)
    pending: bool = False  # tool_use bez tool_result
    highlight: bool = False
    meta: list[str] = field(default_factory=list)


@dataclass
class TraceView:
    available: bool
    path: str | None
    reason: str = ""
    events: list[TraceEvent] = field(default_factory=list)
    tool_calls: int = 0
    highlights: int = 0
    truncated: bool = False


def resolve_trace_path(trace_path: str | None, out_dir: Path | None) -> Path | None:
    """Absolutní cesta, relativní k adresáři výsledků, nebo `…/traces/…` po přesunu adresáře výsledků."""
    if not trace_path:
        return None
    p = Path(trace_path)
    candidates = [p] if p.is_absolute() else []
    if out_dir is not None:
        if not p.is_absolute():
            candidates.append(out_dir / p)
        parts = p.parts
        if "traces" in parts:
            candidates.append(out_dir.joinpath(*parts[parts.index("traces"):]))
    for c in candidates:
        if c.is_file():
            return c
    return None


def _pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(obj)


def _content_text(content: Any) -> str:
    """tool_result.content: řetězec nebo seznam bloků (text/image)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    out.append(str(b.get("text", "")))
                elif b.get("type") == "image":
                    out.append("[image]")
                else:
                    out.append(_pretty(b))
            else:
                out.append(str(b))
        return "\n".join(out)
    return "" if content is None else _pretty(content)


def split_tool_name(name: str) -> tuple[str, str | None]:
    """mcp__plugin_x_srv__orders_get → ("orders_get", "plugin_x_srv")."""
    if name.startswith("mcp__"):
        rest = name[5:]
        server, sep, tool = rest.rpartition("__")
        if sep:
            return tool, server
    return name, None


def _system_event(obj: dict[str, Any]) -> TraceEvent:
    subtype = str(obj.get("subtype") or "")
    data = {k: v for k, v in obj.items() if k not in ("type", "subtype", "session_id", "uuid")}
    if subtype == "init":
        meta = []
        if obj.get("model"):
            meta.append(f"model {obj['model']}")
        if isinstance(obj.get("tools"), list):
            meta.append(f"{len(obj['tools'])} tools")
        servers = obj.get("mcp_servers")
        if isinstance(servers, list) and servers:
            names = [f"{s.get('name')}" + (f" ({s.get('status')})" if s.get("status") else "")
                     for s in servers if isinstance(s, dict)]
            meta.append("MCP: " + ", ".join(names))
        return TraceEvent(kind="init", title="init", body=Clip.of(_pretty(data), 0), meta=meta)
    is_ext = subtype.startswith("skilleval")
    return TraceEvent(kind="skilleval" if is_ext else "system", title=subtype or "system",
                      body=Clip.of(_pretty(data), 600), highlight=is_ext)


def parse_trace_lines(lines: list[str]) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    tools: dict[str, TraceEvent] = {}
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            events.append(TraceEvent(kind="raw", title="unparsable line", body=Clip.of(raw)))
            continue
        if not isinstance(obj, dict):
            events.append(TraceEvent(kind="raw", title="line", body=Clip.of(raw)))
            continue
        typ = obj.get("type")
        if typ == "system":
            events.append(_system_event(obj))
        elif typ == "assistant":
            content = (obj.get("message") or {}).get("content") or []
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            for b in content:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    if str(b.get("text", "")).strip():
                        events.append(TraceEvent(kind="text", title="assistant", body=Clip.of(str(b["text"]))))
                elif bt in ("thinking", "redacted_thinking"):
                    events.append(TraceEvent(kind="thinking", title="thinking",
                                             body=Clip.of(str(b.get("thinking") or "[redacted]"), 400)))
                elif bt == "tool_use":
                    name, server = split_tool_name(str(b.get("name", "?")))
                    ev = TraceEvent(kind="tool", title=name, server=server, tool_id=b.get("id"),
                                    input_json=Clip.of(_pretty(b.get("input", {})), 800), pending=True)
                    events.append(ev)
                    if ev.tool_id:
                        tools[ev.tool_id] = ev
        elif typ == "user":
            msg = obj.get("message") or {}
            content = msg.get("content")
            source = obj.get("skilleval_source")
            if isinstance(content, str):
                events.append(TraceEvent(kind="prompt", title="user", body=Clip.of(content)))
                continue
            for b in content or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    out = Clip.of(_content_text(b.get("content")))
                    ev = tools.get(str(b.get("tool_use_id")))
                    if ev is None:  # výsledek bez tool_use – samostatně
                        ev = TraceEvent(kind="tool", title="tool_result", tool_id=b.get("tool_use_id"))
                        events.append(ev)
                    ev.output, ev.pending = out, False
                    ev.is_error = bool(b.get("is_error"))
                    if source:
                        ev.source, ev.highlight = str(source), True
                elif b.get("type") == "text" and str(b.get("text", "")).strip():
                    events.append(TraceEvent(kind="prompt", title="user", body=Clip.of(str(b["text"]))))
        elif typ == "result":
            meta = []
            if obj.get("num_turns") is not None:
                meta.append(f"{obj['num_turns']} turns")
            if obj.get("duration_ms") is not None:
                meta.append(f"{round(float(obj['duration_ms']) / 1000)}s")
            if obj.get("total_cost_usd") is not None:
                meta.append(f"${float(obj['total_cost_usd']):.3f}")
            err = obj.get("error") if obj.get("is_error") else None
            events.append(TraceEvent(kind="result", title=str(obj.get("subtype") or "result"), meta=meta,
                                     body=Clip.of(str(err)) if err else None, is_error=bool(obj.get("is_error"))))
        else:
            events.append(TraceEvent(kind="raw", title=str(typ or "?"), body=Clip.of(_pretty(obj), 400)))
    return events


def load_trace(trace_path: str | None, out_dir: Path | None) -> TraceView:
    if not trace_path:
        return TraceView(available=False, path=None, reason="the run has no trace")
    path = resolve_trace_path(trace_path, out_dir)
    if path is None:
        return TraceView(available=False, path=trace_path, reason="file no longer exists")
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as ex:
        return TraceView(available=False, path=trace_path, reason=f"cannot read: {ex}")
    events = parse_trace_lines(lines)
    truncated = len(events) > MAX_EVENTS
    if truncated:
        events = events[:MAX_EVENTS]
    return TraceView(available=True, path=str(path), events=events,
                     tool_calls=sum(1 for e in events if e.kind == "tool"),
                     highlights=sum(1 for e in events if e.highlight), truncated=truncated)
