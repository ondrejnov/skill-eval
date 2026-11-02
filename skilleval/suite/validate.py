"""Hlubší validace načtené sady (`skilleval validate`): soubory z context, regexy graderů, mocky, skilleval.yaml."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from skilleval.config import Settings
from skilleval.models import CaseSpec, LoadError, Suite
from skilleval.plugin import is_plugin_dir
from skilleval.suite.frontmatter import FrontmatterError, read_frontmatter

MOCK_KEYS = frozenset({"type", "expect", "error", "abort_when"})
SERVER_MOCK_KEYS = MOCK_KEYS | {"tools"}
MOCK_TYPES = ("fixed", "agent")


def _regex_compiler() -> Callable[[str, str], Any] | None:
    try:
        from skilleval.grading.regexjs import compile_js_regex
    except ImportError:
        return None
    return compile_js_regex


def _expect_validator() -> Callable[[dict], list[str]] | None:
    try:
        from skilleval.mcp.expect import validate_expect
    except ImportError:
        return None
    return validate_expect


def _inside(path: Path, base: Path) -> bool:
    return path.resolve().is_relative_to(base.resolve())


class _Collector:
    def __init__(self) -> None:
        self.errors: list[LoadError] = []

    def add(self, path: Path | str, message: str) -> None:
        self.errors.append(LoadError(path=str(path), message=message))


def _check_case_file(c: _Collector, case: CaseSpec, rel: str, what: str, *, want_dir: bool = False) -> None:
    p = case.dir / rel
    if Path(rel).is_absolute() or not _inside(p, case.dir):
        c.add(case.dir, f"{what}: cesta {rel!r} vede mimo adresář case")
    elif want_dir and not p.is_dir():
        c.add(case.dir, f"{what}: adresář {rel!r} neexistuje")
    elif not want_dir and not p.is_file():
        c.add(case.dir, f"{what}: soubor {rel!r} neexistuje")


def _check_regex(c: _Collector, compile_js: Callable[[str, str], Any] | None, where: Path, label: str,
                 pattern: Any, flags: Any = "") -> None:
    if compile_js is None or not isinstance(pattern, str):
        return
    try:
        compile_js(pattern, flags if isinstance(flags, str) else "")
    except (ValueError, re.error) as e:
        c.add(where, f"{label}: neplatný regex {pattern!r}: {e}")


def _validate_graders(c: _Collector, case: CaseSpec, compile_js: Callable[[str, str], Any] | None) -> None:
    for g in case.graders:
        where = case.dir / "graders" / f"{g.name}.md"
        where = where if where.is_file() else case.dir
        cfg = g.config
        label = f"grader {g.name}"
        if g.type == "regex":
            _check_regex(c, compile_js, where, f"{label}: pattern", cfg.get("pattern"), cfg.get("flags") or "")
        elif g.type == "tool_used":
            _check_regex(c, compile_js, where, f"{label}: input_match", cfg.get("input_match"))
        elif g.type == "tool_order":
            for key in ("before", "after"):
                ref = cfg.get(key)
                if isinstance(ref, dict):
                    _check_regex(c, compile_js, where, f"{label}: {key}.input_match", ref.get("input_match"))
        elif g.type == "baseline":
            bf = cfg.get("baseline_file")
            if isinstance(bf, str):
                _check_case_file(c, case, bf, f"{label}: baseline_file")


def _validate_mock_file(c: _Collector, path: Path, validate_expect: Callable[[dict], list[str]] | None) -> None:
    is_server = path.name == "_server.md"
    try:
        fm, _ = read_frontmatter(path)
    except FrontmatterError as e:
        c.add(path, str(e))
        return
    fm = fm or {}
    allowed = SERVER_MOCK_KEYS if is_server else MOCK_KEYS
    bad = [str(k) for k in fm if k not in allowed]
    if bad:
        c.add(path, f"neznámý klíč {', '.join(repr(b) for b in bad)} (povolené: {', '.join(sorted(allowed))})")
    typ = fm.get("type", "fixed")
    if typ not in MOCK_TYPES:
        c.add(path, f"type musí být fixed nebo agent, ne {typ!r}")
    if "error" in fm and not isinstance(fm["error"], bool):
        c.add(path, "error musí být true/false")
    if "abort_when" in fm and fm["abort_when"] is not None and not isinstance(fm["abort_when"], str):
        c.add(path, "abort_when musí být text")
    if is_server:
        if "type" in fm and typ != "agent":
            c.add(path, "_server.md musí být type: agent")
        tools = fm.get("tools")
        if not isinstance(tools, list) or not tools or not all(isinstance(t, str) and t for t in tools):
            c.add(path, "_server.md: tools musí být neprázdný seznam názvů toolů")
            tools = []
        if "expect" in fm and len(tools) != 1:
            c.add(path, "_server.md: expect je povolené jen když tools obsahuje právě jeden tool "
                        "(guard dej do <tool>.md)")
    if "expect" in fm:
        exp = fm["expect"]
        if not isinstance(exp, dict):
            c.add(path, "expect musí být mapa (cesta → typ / /regex/ / literál / seznam literálů)")
        elif validate_expect is not None:
            for msg in validate_expect(exp):  # zprávy už obsahují cestu (expect.<pole>: …)
                c.add(path, msg)


def _validate_mock_dir(c: _Collector, mock_dir: Path, validate_expect: Callable[[dict], list[str]] | None) -> None:
    for server in sorted(d for d in mock_dir.iterdir() if d.is_dir() and not d.name.startswith(".")):
        for f in sorted(p for p in server.iterdir() if p.is_file()):
            if f.suffix == ".md":
                _validate_mock_file(c, f, validate_expect)
            elif f.name == "_tools.json":
                try:
                    json.loads(f.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    c.add(f, f"neplatný JSON: {e}")


def _validate_extension(c: _Collector, case: CaseSpec, settings: Settings | None) -> None:
    ext = case.extension
    if ext is None:
        return
    path = case.dir / "skilleval.yaml"
    if ext.mcp_profile and settings is not None and ext.mcp_profile not in settings.mcp_profiles:
        known = ", ".join(sorted(settings.mcp_profiles)) or "žádné"
        c.add(path, f"mcp_profile {ext.mcp_profile!r} není v [mcp.profiles] konfigurace (známé: {known})")
    for i, f in enumerate(ext.faults):
        where = f"faults[{i}] ({f.tool})"
        modes = [m for m, on in (("call", f.call is not None), ("calls", f.calls is not None), ("every", f.every))
                 if on]
        if not modes and f.before_call is None:
            c.add(path, f"{where}: chybí call / calls / every / before_call")
        elif len(modes) > 1:
            c.add(path, f"{where}: call / calls / every se vylučují (zadáno: {', '.join(modes)})")
        nums = ([f.call] if f.call is not None else []) + (f.calls or []) + (
            [f.before_call] if f.before_call is not None else [])
        if any(n < 1 for n in nums):
            c.add(path, f"{where}: pořadí volání je 1-based (≥ 1)")
        if f.before_call is not None and not f.run:
            c.add(path, f"{where}: before_call vyžaduje run")
        if f.run and f.before_call is None:
            c.add(path, f"{where}: run je povolené jen s before_call")
        if modes and f.body is None and not f.error:
            c.add(path, f"{where}: fault bez body i error nemá žádný účinek")
    sim = ext.simulated_user
    if sim is not None:
        if sim.max_replies < 1:
            c.add(path, "simulated_user.max_replies musí být ≥ 1")
        if not sim.replies and not sim.agent:
            c.add(path, "simulated_user: zadej replies nebo agent")


def validate_suite(suite: Suite, settings: Settings | None = None) -> list[LoadError]:
    """Chyby načtení + hlubší kontroly. Regex dialekt a `expect:` se kontrolují, jen když jsou moduly k dispozici."""
    c = _Collector()
    c.errors.extend(suite.load_errors)
    compile_js = _regex_compiler()
    validate_expect = _expect_validator()
    seen_mock_dirs: set[Path] = set()

    for case in suite.cases:
        ctx = case.context
        if ctx.scaffold_script:
            _check_case_file(c, case, ctx.scaffold_script, "context.scaffold_script")
        if ctx.history_file:
            _check_case_file(c, case, ctx.history_file, "context.history_file")
        for d in ctx.add_dirs:
            _check_case_file(c, case, d, "context.add_dirs", want_dir=True)
        for p in case.plugins:
            if not is_plugin_dir(p):
                c.add(case.dir, f"plugins: {p} není plugin (chybí plugin.json, SKILL.md nebo skills/<x>/SKILL.md)")
        _validate_graders(c, case, compile_js)
        for md in case.mock_dirs:
            r = md.resolve()
            if r not in seen_mock_dirs:
                seen_mock_dirs.add(r)
                _validate_mock_dir(c, md, validate_expect)
        _validate_extension(c, case, settings)
    return c.errors
