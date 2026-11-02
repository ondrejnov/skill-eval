"""Odložené MCP tooly (jako Claude Code): model vidí jen názvy, schémata si načte toolem `ToolSearch`.

- `should_defer` – rozhodnutí (parametr run_agent / Settings.defer_tools auto|on|off, práh odhadu tokenů).
- `ToolSearch` – vestavěný tool: `select:a,b` (přesné názvy, i bez prefixu `mcp__…__` nebo aliasem), jinak keyword
  search nad názvem a description (`+slovo` = povinné v názvu). Načtené tooly se nabindují od dalšího tahu.
"""

from __future__ import annotations

import json
import re
from typing import Any

from skilleval.config import Settings
from skilleval.models import ToolDef, ToolOutcome

TOOL_SEARCH = "ToolSearch"

TOOL_SEARCH_DESCRIPTION = """\
Fetches full schema definitions for deferred tools so they can be called.

Deferred tools appear by name in <available-deferred-tools> in the system prompt. Until fetched, only the name is \
known — there is no parameter schema, so the tool cannot be invoked reliably. This tool takes a query, matches it \
against the deferred tool list, and loads the matched tools: their full definitions become available and callable \
from your next response.

Query forms:
- "select:Read,Edit,Grep" — fetch these exact tools by name
- "notebook jupyter" — keyword search, up to max_results best matches
- "+slack send" — require "slack" in the name, rank by remaining terms

Load every tool you are likely to need in a single call (select: accepts a comma-separated list)."""

TOOL_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string",
                  "description": 'Query to find deferred tools. Use "select:<tool_name>[,<tool_name>...]" for direct '
                                 'selection, or keywords to search.'},
        "max_results": {"type": "integer", "default": 5, "description": "Maximum number of results (default 5)."},
    },
    "required": ["query"],
}

TOOL_SEARCH_DEF = ToolDef(name=TOOL_SEARCH, description=TOOL_SEARCH_DESCRIPTION, input_schema=TOOL_SEARCH_SCHEMA)


def estimate_tokens(defs: list[ToolDef]) -> int:
    """Hrubý odhad tokenů definic toolů (znaky / 4)."""
    chars = sum(len(d.name) + len(d.description or "") + len(json.dumps(d.input_schema or {}, ensure_ascii=False))
                for d in defs)
    return chars // 4


def should_defer(settings: Settings, tool_defs: list[ToolDef], defer_tools: bool | None = None) -> bool:
    """True/False = vynucené; None = podle `settings.defer_tools` (auto: MCP schémata > defer_tools_min_tokens)."""
    mcp = [d for d in tool_defs if d.kind == "mcp"]
    if not mcp:
        return False
    if defer_tools is not None:
        return defer_tools
    mode = (settings.defer_tools or "auto").lower()
    if mode == "on":
        return True
    if mode == "off":
        return False
    return estimate_tokens(mcp) > settings.defer_tools_min_tokens


def deferred_tool_names(settings: Settings, tool_defs: list[ToolDef], defer_tools: bool | None = None) -> list[str]:
    """Názvy odložených toolů (prázdné = neodkládá se) – pro `build_system_prompt(deferred_tools=…)`."""
    if not should_defer(settings, tool_defs, defer_tools):
        return []
    return [d.name for d in tool_defs if d.kind == "mcp"]


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]


class ToolSearch:
    """Stav ToolSearch jednoho runu: odložené tooly, načtené tooly (pořadí načtení)."""

    def __init__(self, deferred: list[ToolDef], aliases: dict[str, str] | None = None):
        self.deferred = {d.name: d for d in deferred}
        self.aliases = aliases or {}  # název pro model → plný název
        self.loaded: list[str] = []

    @property
    def tool_defs(self) -> list[ToolDef]:
        return [TOOL_SEARCH_DEF]

    def is_hidden(self, name: str) -> bool:
        """Odložený a dosud nenačtený tool (nebinduje se)."""
        return name in self.deferred and name not in self.loaded

    def _resolve(self, raw: str) -> str | None:
        name = raw.strip().strip("`'\"")
        if not name:
            return None
        if name in self.deferred:
            return name
        full = self.aliases.get(name)
        if full in self.deferred:
            return full
        low = name.lower()
        for n, d in self.deferred.items():  # bez prefixu: původní název MCP toolu nebo konec plného názvu
            if (d.tool and d.tool.lower() == low) or n.lower().endswith("__" + low) or n.lower() == low:
                return n
        return None

    def select(self, names: list[str]) -> tuple[list[str], list[str]]:
        found, missing = [], []
        for raw in names:
            if not raw.strip():
                continue
            n = self._resolve(raw)
            if n is None:
                missing.append(raw.strip())
            elif n not in found:
                found.append(n)
        return found, missing

    def keyword(self, query: str, max_results: int) -> list[str]:
        required: list[str] = []
        terms: list[str] = []
        for tok in query.split():
            if tok.startswith("+") and len(tok) > 1:
                required += _words(tok[1:])
            else:
                terms += _words(tok)
        scored: list[tuple[float, str]] = []
        for n, d in self.deferred.items():
            name_l = n.lower()
            if any(r not in name_l for r in required):
                continue
            name_words = set(_words(d.tool or n))
            desc_l = (d.description or "").lower()
            score = 0.0
            for t in terms:
                if t in name_words:
                    score += 5
                elif t in name_l:
                    score += 3
                if t in desc_l:
                    score += 1
            if required and not terms:
                score += 1
            if score > 0:
                scored.append((score, n))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [n for _, n in scored[:max(1, max_results)]]

    def run(self, args: dict[str, Any], model_name: dict[str, str] | None = None) -> tuple[ToolOutcome, list[dict[str, Any]]]:
        """Provede dotaz → (výsledek pro model, bloky `tool_reference` do trace). Nalezené tooly označí jako načtené."""
        query = str(args.get("query") or "").strip()
        try:
            max_results = int(args.get("max_results") or 5)
        except (TypeError, ValueError):
            max_results = 5
        if not query:
            return ToolOutcome(content="InputValidationError: query is required", is_error=True), []
        missing: list[str] = []
        if query.lower().startswith("select:"):
            found, missing = self.select(query[len("select:"):].split(","))
        else:
            found = self.keyword(query, max_results)
        for n in found:
            if n not in self.loaded:
                self.loaded.append(n)
        names = model_name or {}
        lines = []
        if found:
            lines.append(f"Loaded {len(found)} tool(s). Their full definitions are now available; call them by the "
                         f"name shown (from your next response):")
            for n in found:
                desc = (self.deferred[n].description or "").strip().splitlines()
                call_as = names.get(n, n)
                label = call_as if call_as == n else f"{call_as} (= {n})"
                lines.append(f"- {label}" + (f": {desc[0][:200]}" if desc else ""))
        else:
            lines.append(f"No matching deferred tools found for query {query!r}.")
        if missing:
            lines.append("Not found: " + ", ".join(missing))
        refs = [{"type": "tool_reference", "tool_name": n} for n in found]
        return ToolOutcome(content="\n".join(lines)), refs

    async def call(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        """Rozhraní ToolProvider (bez aliasů a trace bloků) – graf volá přímo `run`."""
        return self.run(args)[0]


__all__ = ["TOOL_SEARCH", "TOOL_SEARCH_DEF", "ToolSearch", "deferred_tool_names", "estimate_tokens", "should_defer"]
