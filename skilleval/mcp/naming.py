"""Pojmenování MCP toolů přesně jako Claude Code a globy pro `--allow-tools`.

Claude Code registruje server pluginu pod názvem `plugin:<plugin>:<server>` a název toolu skládá jako
`mcp__<normalize(server)>__<normalize(tool)>`, kde normalize nahradí každý znak mimo `[A-Za-z0-9_-]` za `_`
(dvojtečky → `_`, pomlčky zůstávají): `mcp__plugin_webstore-core_webstore__orders_get`.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re

_INVALID_RE = re.compile(r"[^A-Za-z0-9_-]")


def normalize(name: str) -> str:
    """Normalizace jako CC `normalizeNameForMCP`: znaky mimo [A-Za-z0-9_-] → `_`."""
    return _INVALID_RE.sub("_", name)


def server_label(plugin: str, server: str) -> str:
    """`plugin_<plugin>_<server>` – název serveru v progress řádcích (`mocked:`) a prefix toolů."""
    return normalize(f"plugin:{plugin}:{server}")


def mcp_tool_name(plugin: str, server: str, tool: str) -> str:
    """Plný název MCP toolu pluginu: `mcp__plugin_<plugin>_<server>__<tool>`."""
    return f"mcp__{server_label(plugin, server)}__{normalize(tool)}"


def fit_tool_name(name: str, limit: int) -> str:
    """Zkrátí název nad limit API (OpenAI/Anthropic: 64 znaků) deterministicky s hashem (unikátnost)."""
    if limit <= 0 or len(name) <= limit:
        return name
    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
    return name[: limit - 9] + "_" + digest


def match_tool_glob(pattern: str, name: str) -> bool:
    """Glob z `--allow-tools` (`mcp__plugin_x_y__*`) nebo přesná shoda; rozlišuje velikost písmen.

    Vzory s pravidlem v závorce (`Bash(npm test *)`) se porovnávají jen názvem toolu před závorkou.
    """
    pattern = pattern.strip()
    if not pattern:
        return False
    if "(" in pattern and pattern.endswith(")"):
        pattern = pattern[: pattern.index("(")]
    if pattern == name:
        return True
    return fnmatch.fnmatchcase(name, pattern)


def match_any(patterns: list[str], name: str) -> bool:
    return any(match_tool_glob(p, name) for p in patterns)
