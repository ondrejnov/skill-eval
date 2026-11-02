"""Testovací MCP server (FastMCP) pro testy skilleval.mcp: `python server.py stdio` | `python server.py http <port>`.

tools/list stránkuje po 2 toolech (test cursoru). Textové tooly bez structuredContent, `add` se structuredContent.
"""

from __future__ import annotations

import os
import sys

from mcp import types
from mcp.server.fastmcp import Context, FastMCP

server = FastMCP("fixture", instructions="Fixture server: use echo for everything.", log_level="WARNING")
_COUNTER = {"n": 0}


@server.tool(description="Echo the text back.", structured_output=False)
def echo(text: str) -> str:
    return text


@server.tool(description="Add two numbers.")
def add(a: int, b: int) -> int:
    return a + b


@server.tool(description="Always fails.", structured_output=False)
def fail(msg: str) -> str:
    raise ValueError(msg)


@server.tool(description="Returns a resource link.", structured_output=False)
def link() -> list[types.ContentBlock]:
    return [types.ResourceLink(type="resource_link", uri="webstore://export/1.pdf", name="faktura.pdf")]


@server.tool(description="Returns n characters.", structured_output=False)
def big(n: int) -> str:
    return "x" * n


@server.tool(description="Returns a request header (http only).", structured_output=False)
def header(name: str, ctx: Context) -> str:
    req = ctx.request_context.request
    return req.headers.get(name, "") if req is not None else ""


@server.tool(description="Returns an environment variable.", structured_output=False)
def env(name: str) -> str:
    return os.environ.get(name, "")


@server.tool(description="Counts calls on this connection/process.", structured_output=False)
def counter() -> str:
    _COUNTER["n"] += 1
    return str(_COUNTER["n"])


_orig = server._mcp_server.request_handlers[types.ListToolsRequest]


async def _paged(req: types.ListToolsRequest) -> types.ServerResult:
    res = await _orig(req)
    tools = res.root.tools
    start = int(req.params.cursor) if req.params and req.params.cursor else 0
    nxt = str(start + 2) if start + 2 < len(tools) else None
    return types.ServerResult(types.ListToolsResult(tools=tools[start:start + 2], nextCursor=nxt))


server._mcp_server.request_handlers[types.ListToolsRequest] = _paged

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "http":
        server.settings.port = int(sys.argv[2])
        server.run("streamable-http")
    else:
        server.run("stdio")
