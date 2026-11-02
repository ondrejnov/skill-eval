"""Převod transcriptu Claude Code (`.jsonl` – session soubor i stream-json `claude -p`) na zprávy LangChain.

- Řádky `type: user|assistant` s `message.content` (string nebo bloky text/image/tool_use/tool_result/thinking).
- Ostatní typy (system, result, attachment, summary, queue-operation …), sidechain (subagenti) a `isMeta` se ignorují.
- Claude Code zapisuje každý content blok odpovědi na vlastní řádek se stejným `message.id` → slučují se do jedné
  `AIMessage` (text + tool_calls).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def _block_text(content: Any) -> str:
    """Text z obsahu tool_result (string nebo seznam bloků)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text", "")))
            elif isinstance(b, dict) and b.get("type") == "image":
                parts.append("[image]")
        return "\n".join(parts)
    return "" if content is None else str(content)


def _image_block(b: dict[str, Any]) -> dict[str, Any] | None:
    src = b.get("source") or {}
    if src.get("type") == "base64" and src.get("data"):
        return {"type": "image_url", "image_url": {"url": f"data:{src.get('media_type', 'image/png')};base64,{src['data']}"}}
    if src.get("type") == "url" and src.get("url"):
        return {"type": "image_url", "image_url": {"url": src["url"]}}
    return None


def _user_messages(content: Any) -> list[BaseMessage]:
    if isinstance(content, str):
        return [HumanMessage(content=content)]
    if not isinstance(content, list):
        return []
    out: list[BaseMessage] = []
    human: list[dict[str, Any]] = []
    for b in content:
        if isinstance(b, str):
            human.append({"type": "text", "text": b})
        elif not isinstance(b, dict):
            continue
        elif b.get("type") == "tool_result":
            out.append(ToolMessage(content=_block_text(b.get("content")), tool_call_id=str(b.get("tool_use_id", "")),
                                   status="error" if b.get("is_error") else "success"))
        elif b.get("type") == "text":
            human.append({"type": "text", "text": str(b.get("text", ""))})
        elif b.get("type") == "image" and (img := _image_block(b)):
            human.append(img)
    if human:  # text uživatele až za výsledky toolů (API vyžaduje tool výsledky hned za voláním)
        if all(h["type"] == "text" for h in human):
            out.append(HumanMessage(content="\n\n".join(h["text"] for h in human)))
        else:
            out.append(HumanMessage(content=human))
    return out


def load_history(path: Path) -> list[BaseMessage]:
    """Transcript Claude Code → [HumanMessage | AIMessage(tool_calls) | ToolMessage]."""
    out: list[BaseMessage] = []
    ai_index: dict[str, int] = {}  # message.id → index AIMessage v `out` (bloky odpovědi mohou být proložené tool_result)
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("isSidechain") or row.get("isMeta"):
            continue
        typ, msg = row.get("type"), row.get("message")
        if typ not in ("user", "assistant") or not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if typ == "user":
            out.extend(_user_messages(content))
            continue

        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
        texts = [str(b.get("text", "")) for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        calls = [{"name": str(b.get("name", "")), "args": b.get("input") or {}, "id": str(b.get("id", "")),
                  "type": "tool_call"}
                 for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        text = "\n\n".join(t for t in texts if t)
        if not text and not calls:
            continue  # jen thinking / prázdný blok
        msg_id = msg.get("id")
        idx = ai_index.get(msg_id) if msg_id else None
        if idx is not None:
            prev = out[idx]
            merged = "\n\n".join(t for t in (str(prev.content or ""), text) if t)
            out[idx] = AIMessage(content=merged, tool_calls=[*prev.tool_calls, *calls])
        else:
            out.append(AIMessage(content=text, tool_calls=calls))
            if msg_id:
                ai_index[msg_id] = len(out) - 1
    return out


__all__ = ["load_history"]
