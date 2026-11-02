"""load_history: transcript Claude Code (.jsonl) → zprávy LangChain."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from skilleval.agent.history import load_history


def write_jsonl(path: Path, rows: list[dict | str]) -> Path:
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    return path


def test_session_transcript(tmp_path: Path):
    p = write_jsonl(tmp_path / "h.jsonl", [
        {"type": "queue-operation", "operation": "enqueue"},
        {"type": "system", "subtype": "init", "cwd": "/x"},
        {"type": "user", "message": {"role": "user", "content": "Najdi objednávku 71781"}},
        {"type": "user", "isMeta": True, "message": {"role": "user", "content": "<local-command-caveat>"}},
        {"type": "assistant", "message": {"id": "m1", "role": "assistant", "content": [{"type": "thinking", "thinking": "..."}]}},
        {"type": "assistant", "message": {"id": "m1", "role": "assistant", "content": [{"type": "text", "text": "Hledám."}]}},
        {"type": "assistant", "message": {"id": "m1", "role": "assistant", "content": [
            {"type": "tool_use", "id": "tu1", "name": "mcp__plugin_webstore-core_webstore__core_search", "input": {"q": "71781"}}]}},
        # výsledek prvního toolu proložený mezi bloky téže odpovědi
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": [{"type": "text", "text": "nalezeno"}]}]}},
        {"type": "assistant", "message": {"id": "m1", "role": "assistant", "content": [
            {"type": "tool_use", "id": "tu2", "name": "Read", "input": {"file_path": "/a"}}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu2", "content": "<tool_use_error>nope</tool_use_error>", "is_error": True}]}},
        {"type": "assistant", "isSidechain": True, "message": {"id": "s1", "content": [{"type": "text", "text": "subagent"}]}},
        {"type": "assistant", "message": {"id": "m2", "role": "assistant", "content": [{"type": "text", "text": "Objednávka 71781 je ve stavu 2."}]}},
        {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "Díky"},
                                                                 {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}}]}},
        {"type": "result", "subtype": "success", "result": "x"},
        "not json",
    ])
    msgs = load_history(p)
    assert [type(m) for m in msgs] == [HumanMessage, AIMessage, ToolMessage, ToolMessage, AIMessage, HumanMessage]
    assert msgs[0].content == "Najdi objednávku 71781"
    ai = msgs[1]
    assert ai.content == "Hledám."
    assert [(tc["name"], tc["id"], tc["args"]) for tc in ai.tool_calls] == [
        ("mcp__plugin_webstore-core_webstore__core_search", "tu1", {"q": "71781"}), ("Read", "tu2", {"file_path": "/a"})]
    assert msgs[2].tool_call_id == "tu1" and msgs[2].content == "nalezeno" and msgs[2].status == "success"
    assert msgs[3].status == "error"
    assert msgs[4].content == "Objednávka 71781 je ve stavu 2." and not msgs[4].tool_calls
    assert msgs[5].content[0] == {"type": "text", "text": "Díky"}
    assert msgs[5].content[1]["image_url"]["url"] == "data:image/png;base64,AAA"


def test_stream_json_trace(tmp_path: Path):
    """Formát stream-json `claude -p` (i trace skilleval): user prompt string, bloky po řádcích."""
    p = write_jsonl(tmp_path / "t.jsonl", [
        {"type": "system", "subtype": "init"},
        {"type": "user", "message": {"role": "user", "content": "Ahoj"}},
        {"type": "assistant", "message": {"id": "a", "content": [{"type": "text", "text": "Zdravím"}]}},
        {"type": "user", "message": {"role": "user", "content": "Dobropis"}},
        {"type": "assistant", "message": {"id": "b", "content": "Hotovo"}},
        {"type": "result", "result": "Hotovo"},
    ])
    msgs = load_history(p)
    assert [(type(m).__name__, m.content) for m in msgs] == [
        ("HumanMessage", "Ahoj"), ("AIMessage", "Zdravím"), ("HumanMessage", "Dobropis"), ("AIMessage", "Hotovo")]
