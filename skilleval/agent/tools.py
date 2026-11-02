"""Tooly pro model: sloučení ToolProviderů, aliasy dlouhých/neplatných názvů, OpenAI function specifikace."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from skilleval.models import ToolDef, ToolProvider

VALID_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _hash(name: str, n: int) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:n]


def make_alias(name: str, max_len: int, taken: set[str]) -> str:
    """Zkrácený alias: platné znaky z názvu (prefix) + `_` + krátký hash; unikátní vůči `taken`."""
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    for n in (8, 12, 16, 40):
        h = _hash(name, n)
        alias = f"{clean[:max(1, max_len - n - 1)]}_{h}"[:max_len]
        if alias not in taken:
            return alias
    i = 2  # pragma: no cover - kolize sha1 prakticky nenastane
    while f"t{i}_{_hash(name, 8)}" in taken:
        i += 1
    return f"t{i}_{_hash(name, 8)}"


def needs_alias(name: str, max_len: int) -> bool:
    return len(name) > max_len or not VALID_NAME_RE.match(name)


@dataclass
class ToolIndex:
    """Plný název ↔ název pro model, plný název → provider. První provider s daným názvem vyhrává."""

    max_len: int
    defs: list[ToolDef] = field(default_factory=list)
    providers: dict[str, ToolProvider] = field(default_factory=dict)
    to_model: dict[str, str] = field(default_factory=dict)
    to_full: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, providers: list[ToolProvider], max_len: int) -> ToolIndex:
        idx = cls(max_len=max_len)
        for p in providers:
            for d in p.tool_defs:
                if d.name in idx.providers:
                    continue
                idx.defs.append(d)
                idx.providers[d.name] = p
        taken = {d.name for d in idx.defs if not needs_alias(d.name, max_len)}
        for d in idx.defs:
            idx.alias(d.name, taken)
        return idx

    def alias(self, full: str, taken: set[str] | None = None) -> str:
        """Název pro model (i pro názvy z historie, které nejsou mezi tooly)."""
        if full in self.to_model:
            return self.to_model[full]
        if taken is None:
            taken = set(self.to_full)
        name = make_alias(full, self.max_len, taken) if needs_alias(full, self.max_len) else full
        taken.add(name)
        self.to_model[full] = name
        self.to_full[name] = full
        return name

    def full_name(self, model_name: str) -> str:
        """Název od modelu → plný název (model může poslat i plný název)."""
        return self.to_full.get(model_name, model_name)

    def specs(self, hidden: Callable[[str], bool] | None = None) -> list[dict[str, Any]]:
        """OpenAI function specifikace; `hidden(plný název)` = nebindovat (odložené nenačtené tooly)."""
        out = []
        for d in self.defs:
            if hidden is not None and hidden(d.name):
                continue
            schema = dict(d.input_schema or {})
            schema.setdefault("type", "object")
            if schema.get("type") == "object":
                schema.setdefault("properties", {})
            out.append({"type": "function", "function": {"name": self.to_model[d.name],
                                                         "description": d.description or "", "parameters": schema}})
        return out
