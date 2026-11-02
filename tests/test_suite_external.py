"""Volitelná reálná sada: všechny cases v `$SKILLEVAL_EXTERNAL_SKILLS/*/evals` se musí načíst bez chyb (bez expanze MCP)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from skilleval.config import load_settings
from skilleval.plugin import find_plugin_root
from skilleval.suite.loader import load_suite
from skilleval.suite.validate import validate_suite

SKILLS = Path(os.environ.get("SKILLEVAL_EXTERNAL_SKILLS", "/nonexistent"))
PLUGINS = sorted(p for p in SKILLS.glob("*") if (p / "evals").is_dir()) if SKILLS.is_dir() else []

pytestmark = pytest.mark.skipif(not PLUGINS, reason="externí sada není k dispozici (SKILLEVAL_EXTERNAL_SKILLS)")


def test_all_external_cases_load():
    settings = load_settings()  # mcp_profile z cases se ověřuje proti lokálnímu skilleval.toml
    total = 0
    for root in PLUGINS:
        s = load_suite(root, settings, resolve_mcp=False)
        assert s.load_errors == [], (root.name, s.load_errors)
        assert validate_suite(s, settings) == [], root.name
        assert s.plugin is not None and s.plugin.name == root.name
        for c in s.cases:
            assert c.rel_dir.startswith("evals/") and "results" not in c.rel_dir
            assert c.plugins == [root.resolve()]
            assert c.prompt and c.graders
            assert find_plugin_root(c.dir) == root.resolve()
        total += len(s.cases)
    assert total > 0
