"""Pomocné funkce pro testy graderů (bez testů) – deterministický FakeJudge, stavba CaseSpec/RunArtifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skilleval.models import CaseSpec, GraderSpec, JudgeVote, McpCallRecord, RunArtifacts, ToolCallRecord
from skilleval.trace import TraceWriter

MCP_ORDERS_GET = "mcp__plugin_webstore-core_webstore__orders_get"
MCP_SET_STATUS = "mcp__plugin_webstore-core_webstore__orders_set_status"


class FakeJudge:
    """Deterministický soudce: hlasy dle skriptu (cyklicky), zaznamenává volání."""

    def __init__(self, verdicts: list[bool] | None = None, cost: float = 0.001, explanation: str = "",
                 fail: Exception | None = None, fail_on: int | None = None):
        self.verdicts = verdicts or [True]
        self.cost = cost
        self.explanation = explanation
        self.fail = fail
        self.fail_on = fail_on
        self.calls: list[dict[str, Any]] = []

    async def vote(self, system: str, user: str, images: list[dict[str, Any]] | None = None) -> tuple[JudgeVote, float]:
        n = len(self.calls)
        self.calls.append({"system": system, "user": user, "images": images})
        if self.fail is not None and (self.fail_on is None or self.fail_on == n):
            raise self.fail
        passed = self.verdicts[n % len(self.verdicts)]
        expl = self.explanation or ("looks good" if passed else f"vote {n} failed: missing ref.url")
        return JudgeVote(passed=passed, explanation=expl), self.cost

    async def complete(self, system: str, messages: list[dict[str, str]]) -> tuple[str, float]:
        return "", 0.0


def make_trace(final_text: str = "Objednávka 71781 je ve stavu \"vyřizuje se\".") -> tuple[list[str], list[ToolCallRecord]]:
    """Realistický trace jako Claude Code: Skill → orders_get → orders_set_status → shrnutí."""
    tw = TraceWriter(cwd="/tmp/ws", model="anthropic/claude-haiku-5.5",
                     tools=["Read", "Skill", MCP_ORDERS_GET, MCP_SET_STATUS],
                     mcp_servers=[{"name": "plugin_webstore-core_webstore", "status": "connected"}],
                     skills=["webstore-core"], plugins=[{"name": "webstore-core", "path": "/x"}])
    tw.user_prompt("Přepni objednávku 71781 do stavu vyřizuje se.")
    calls = [
        ToolCallRecord(id="t1", name="Skill", input={"skill": "webstore-core"}, output="# Webstore core"),
        ToolCallRecord(id="t2", name=MCP_ORDERS_GET, input={"id": 71781}, output='{"id":71781,"status":1}',
                       source="real"),
        ToolCallRecord(id="t3", name=MCP_SET_STATUS,
                       input={"id": 71781, "status": 2, "dry_run": False, "expected_status": 1,
                              "note": "vyřízení"},
                       output='{"ok":true,"ref":{"url":"http://127.0.0.1:5174/orders/71781"}}', source="real"),
    ]
    for i, c in enumerate(calls):
        tw.assistant(msg_id=f"msg_{i}", text=None, tool_uses=[{"id": c.id, "name": c.name, "input": c.input}])
        tw.tool_result(tool_use_id=c.id, content=c.output or "", source=c.source)
    tw.assistant(msg_id="msg_end", text=final_text, tool_uses=[])
    tw.result(final_text=final_text, turns=4, cost_usd=0.004, duration_ms=7000, error=None)
    return tw.lines, calls


def make_art(tmp_path: Path, *, final: str | None = None, files: dict[str, str | bytes] | None = None,
             created: list[str] | None = None, mock_calls: list[McpCallRecord] | None = None,
             **kw: Any) -> RunArtifacts:
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    for rel, content in (files or {}).items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")
    lines, calls = make_trace(final) if final is not None else make_trace()
    return RunArtifacts(final_message=final if final is not None else "Objednávka 71781 je ve stavu \"vyřizuje se\".",
                        trace_lines=kw.pop("trace_lines", lines), tool_calls=kw.pop("tool_calls", calls),
                        files_created=created if created is not None else list((files or {}).keys()),
                        workspace=ws, mock_calls=mock_calls or [], **kw)


def g(name: str, type_: str, *, weight: float = 1, arm: str | None = None, markdown: str = "",
      **config: Any) -> GraderSpec:
    return GraderSpec(name=name, type=type_, weight=weight, arm=arm, config=config, markdown=markdown)  # type: ignore[arg-type]


def make_case(tmp_path: Path, graders: list[GraderSpec], *, mocks: dict[str, list[str]] | None = None,
              prompt: str = "Přepni objednávku 71781 do stavu vyřizuje se.") -> CaseSpec:
    case_dir = tmp_path / "evals" / "case1"
    case_dir.mkdir(parents=True, exist_ok=True)
    mock_dirs = []
    if mocks is not None:
        md = case_dir / "mocks"
        md.mkdir(exist_ok=True)
        for server, tools in mocks.items():
            (md / server).mkdir(exist_ok=True)
            for t in tools:
                (md / server / t).write_text("ok", encoding="utf-8")
        mock_dirs.append(md)
    return CaseSpec(name="case1", dir=case_dir, rel_dir="evals/case1", source="prompt", prompt=prompt,
                    graders=graders, mock_dirs=mock_dirs)
