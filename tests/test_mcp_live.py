"""Živý test proti MCP serveru (opt-in `SKILLEVAL_LIVE=1` a `SKILLEVAL_LIVE_MCP_JSON`). Volá jen čtecí tool `core_whoami`."""

import json
import os
from pathlib import Path

import pytest

from skilleval.config import Settings
from skilleval.mcp.session import McpSession
from skilleval.models import CaseSpec, GraderSpec, McpServerConfig

MCP_JSON = Path(os.environ.get("SKILLEVAL_LIVE_MCP_JSON", "/nonexistent"))  # <plugin>/.mcp.json

pytestmark = pytest.mark.skipif(os.environ.get("SKILLEVAL_LIVE") != "1" or not MCP_JSON.is_file(),
                                reason="live test: SKILLEVAL_LIVE=1, SKILLEVAL_LIVE_MCP_JSON")


async def test_live_whoami(tmp_path):
    plugin = MCP_JSON.resolve().parent.name
    server, raw = next(iter(json.loads(MCP_JSON.read_text())["mcpServers"].items()))
    prefix = f"mcp__plugin_{plugin}_{server}__core_"
    cfg = McpServerConfig(name=server, plugin=plugin, **raw)
    token = cfg.headers.get("Authorization", "")
    case = CaseSpec(name="live", dir=tmp_path, rel_dir="evals/live", source="prompt", prompt="x",
                    graders=[GraderSpec(name="g", type="regex", config={"pattern": "x"})])
    session = McpSession([cfg], case=case, mode="off", allow_real=False, strict_grants=True,
                         allow_tools=[prefix + "*"], judge=None, settings=Settings(),
                         hooks=False, recordings_dir=None, run_env={})
    async with session as s:
        names = [t.name for t in s.tool_defs]
        assert prefix + "whoami" in names, s.notices
        assert all(n.startswith(prefix) for n in names)
        assert s.instructions.get(server)
        out = await s.call(prefix + "whoami", {})
        assert not out.is_error, out.content[:300]
        assert out.content.strip()
        for secret in (token, token.removeprefix("Bearer ").strip()):
            if secret:
                assert secret not in out.content
                assert all(secret not in n for n in s.notices)
