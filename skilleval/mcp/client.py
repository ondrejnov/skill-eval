"""Připojení k reálnému MCP serveru (oficiální `mcp` SDK): http (Streamable HTTP), sse, stdio.

Spojení vlastní samostatná asyncio úloha (anyio cancel scopy transportu se otevírají i zavírají ve stejné úloze),
volání toolů z jiných úloh jdou přes `ClientSession` (paměťové streamy).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

import anyio
import httpx
from mcp import ClientSession, types
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.exceptions import McpError

from skilleval.models import McpServerConfig

CLIENT_INFO = types.Implementation(name="skilleval", version="0.1.0")


class ServerError(Exception):
    pass


def describe_exception(ex: BaseException) -> str:
    """Čitelný popis chyby (rozbalí ExceptionGroup, HTTP status)."""
    if isinstance(ex, BaseExceptionGroup):
        return "; ".join(dict.fromkeys(describe_exception(e) for e in ex.exceptions))
    if isinstance(ex, httpx.HTTPStatusError):
        return f"HTTP {ex.response.status_code} {ex.response.reason_phrase}".strip()
    if isinstance(ex, TimeoutError):
        return "timed out"
    if isinstance(ex, (httpx.ConnectError, ConnectionError, OSError)):
        return f"connection failed: {ex}" if str(ex) else "connection failed"
    if isinstance(ex, McpError):
        return f"MCP error {ex.error.code}: {ex.error.message}"
    return f"{type(ex).__name__}: {ex}" if str(ex) else type(ex).__name__


def convert_result(result: types.CallToolResult, *, prefer_structured: bool = True
                   ) -> tuple[str, list[dict[str, Any]], bool]:
    """CallToolResult → (text pro model, obrázky, is_error) jako Claude Code.

    Je-li `structuredContent` (a `prefer_structured`), model dostane jeho kompaktní JSON místo textových bloků
    (ověřeno z CC trace); resource_link/obrázky/ostatní bloky se připojí jako obvykle.
    """
    structured = prefer_structured and result.structuredContent is not None
    texts: list[str] = []
    if structured:
        texts.append(json.dumps(result.structuredContent, ensure_ascii=False, separators=(",", ":")))
    images: list[dict[str, Any]] = []
    for block in result.content:
        if isinstance(block, types.TextContent):
            if not structured:
                texts.append(block.text)
        elif isinstance(block, types.ImageContent):
            images.append({"media_type": block.mimeType, "data": block.data})
        elif isinstance(block, types.ResourceLink):
            texts.append(f"[resource] {block.uri} ({block.name})")
        elif isinstance(block, types.EmbeddedResource):
            res = block.resource
            if isinstance(res, types.TextResourceContents):
                texts.append(res.text)
            else:
                texts.append(f"[resource] {res.uri} ({res.mimeType or 'binary'})")
        elif isinstance(block, types.AudioContent):
            texts.append(f"[audio] ({block.mimeType})")
    if not texts and result.structuredContent is not None:
        texts.append(json.dumps(result.structuredContent, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(texts), images, bool(result.isError)


class RealServer:
    """Jedno spojení na MCP server po dobu runu."""

    def __init__(self, config: McpServerConfig, *, env: dict[str, str], connect_timeout: float = 30.0,
                 call_timeout: float = 300.0):
        self.config = config
        self.env = env
        self.connect_timeout = connect_timeout
        self.call_timeout = call_timeout
        self.tools: list[types.Tool] = []
        self.instructions: str | None = None
        self.error: str | None = None
        self._session: ClientSession | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._errlog: Any = None

    @property
    def name(self) -> str:
        return self.config.name

    async def start(self) -> None:
        """Připojí se, `initialize` + `tools/list` (se stránkováním). Chyba → ServerError."""
        self._task = asyncio.create_task(self._run(), name=f"mcp-{self.name}")
        await self._ready.wait()
        if self.error:
            await self.close()
            raise ServerError(self.error)

    async def _open(self, stack: AsyncExitStack) -> tuple[Any, Any]:
        c = self.config
        if c.type == "http":
            if not c.url:
                raise ServerError("missing url")
            client = create_mcp_http_client(headers=dict(c.headers),
                                            timeout=httpx.Timeout(self.connect_timeout, read=self.call_timeout))
            await stack.enter_async_context(client)
            read, write, _ = await stack.enter_async_context(streamable_http_client(c.url, http_client=client))
            return read, write
        if c.type == "sse":
            if not c.url:
                raise ServerError("missing url")
            read, write = await stack.enter_async_context(
                sse_client(c.url, headers=dict(c.headers), timeout=self.connect_timeout,
                           sse_read_timeout=self.call_timeout))
            return read, write
        if not c.command:
            raise ServerError("missing command")
        env = {**self.env, **c.env}
        env.setdefault("PATH", os.environ.get("PATH", "/usr/bin:/bin"))
        self._errlog = stack.enter_context(tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace"))
        params = StdioServerParameters(command=c.command, args=list(c.args), env=env, cwd=c.cwd)
        read, write = await stack.enter_async_context(stdio_client(params, errlog=self._errlog))
        return read, write

    def _stderr_tail(self, n: int = 600) -> str:
        if self._errlog is None:
            return ""
        with contextlib.suppress(Exception):
            self._errlog.flush()
            self._errlog.seek(0)
            text = self._errlog.read().strip()
            return text[-n:]
        return ""

    async def _run(self) -> None:
        try:
            async with AsyncExitStack() as stack:
                read, write = await self._open(stack)
                session = await stack.enter_async_context(ClientSession(read, write, client_info=CLIENT_INFO))
                # timeout jen kolem požadavků – cancel scope nesmí obalit vstup do kontextů ve stacku
                with anyio.fail_after(self.connect_timeout):
                    init = await session.initialize()
                    self.instructions = init.instructions
                    cursor: str | None = None
                    while True:
                        page = await session.list_tools(cursor=cursor)
                        self.tools.extend(page.tools)
                        cursor = page.nextCursor
                        if not cursor:
                            break
                self._session = session
                self._ready.set()
                await self._stop.wait()
                self._session = None
        except BaseException as ex:  # noqa: BLE001 – chyba spojení se hlásí jako notice, run pokračuje
            if not self._ready.is_set():
                msg = describe_exception(ex)
                tail = self._stderr_tail()
                self.error = msg + (f" (stderr: {tail})" if tail else "")
            if isinstance(ex, asyncio.CancelledError):
                raise
        finally:
            self._session = None
            self._ready.set()

    async def call(self, tool: str, args: dict[str, Any]) -> types.CallToolResult:
        if self._session is None:
            raise ServerError(f"server {self.name} is not connected")
        return await self._session.call_tool(tool, args, read_timeout_seconds=timedelta(seconds=self.call_timeout))

    async def close(self) -> None:
        self._stop.set()
        task, self._task = self._task, None
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=10)
        except (TimeoutError, Exception):  # noqa: BLE001
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
