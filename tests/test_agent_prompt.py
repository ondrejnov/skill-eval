"""build_system_prompt (skilleval.harness.prompt)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from skilleval.harness.prompt import build_system_prompt
from skilleval.models import SkillInfo, ToolDef


def skill(name: str, desc: str, plugin: str = "webstore-core") -> SkillInfo:
    return SkillInfo(name=name, description=desc, body="# body", path=Path(f"/p/{name}"), plugin=plugin)


TOOLS = [ToolDef(name="Skill", description="", input_schema={}), ToolDef(name="Read", description="", input_schema={}),
         ToolDef(name="mcp__plugin_webstore-core_webstore__core_whoami", description="", input_schema={}, kind="mcp",
                 server="webstore", tool="core_whoami", plugin="webstore-core")]


def build(**kw):
    args = dict(cwd="/tmp/ws/workspace", skills=[], tool_defs=TOOLS, mcp_instructions={}, append=None,
                today=date(2026, 11, 1), model="anthropic/claude-haiku-5.5")
    args.update(kw)
    return build_system_prompt(**args)


def test_basics_without_skills_and_mcp():
    p = build()
    assert "non-interactive" in p and "/tmp/ws/workspace" in p and "2026-11-01" in p
    assert "anthropic/claude-haiku-5.5" in p
    assert "parallel" in p and "same language" in p
    assert "# Skills" not in p and "<available_skills>" not in p
    assert "MCP Server Instructions" not in p
    assert "Skill, Read, MCP server `plugin_webstore-core_webstore` (1 tools" in p


def test_skills_catalog_mcp_instructions_append():
    p = build(skills=[skill("webstore-core", "Základ pro\n  práci s administrací."),
                      skill("webstore-orders", "Objednávky {a} a reklamace.", plugin="webstore-orders")],
              mcp_instructions={"webstore": "Nejdřív zavolej core_whoami.\n", "empty": "  "},
              append="Odpovídej česky.")
    assert "<available_skills>\n- webstore-core:webstore-core: Základ pro práci s administrací.\n" \
           "- webstore-orders:webstore-orders: Objednávky {a} a reklamace.\n</available_skills>" in p
    assert "invoke the Skill tool" in p and "base directory" in p
    assert "# MCP Server Instructions" in p and "## webstore\nNejdřív zavolej core_whoami." in p
    assert "## empty" not in p
    assert p.rstrip().endswith("Odpovídej česky.")
    assert p.index("<available_skills>") < p.index("# MCP Server Instructions") < p.index("Odpovídej česky.")


def test_skills_section_needs_skill_tool():
    p = build(skills=[skill("webstore-core", "x")], tool_defs=[t for t in TOOLS if t.name != "Skill"])
    assert "<available_skills>" not in p
