"""Mocky MCP serverů ve formátu Claude Code plugin evals.

Layout (v každém adresáři `mocks/` – suite, pak case; pozdější přebíjí soubor po souboru):

    mocks/<server>/<tool>.md        fixed (default) nebo agent mock jednoho toolu
    mocks/<server>/_server.md       agent mock pro víc toolů (`tools: [...]`), `<tool>.md` má přednost
    mocks/<server>/_tools.json      uložený `tools/list` ({"tools": [...]}) → description a inputSchema
    mocks/<server>/fixtures/…       soubory pro `{{file:fixtures/…}}`
    mocks/.replay/<server>/*.json   nahrávky agent mocků → odpověď na identické volání bez modelu
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from skilleval.mcp.expect import lookup, validate_expect

PERMISSIVE_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": True}
TOOL_KEYS = {"type", "expect", "error", "abort_when"}
SERVER_KEYS = {"type", "expect", "abort_when", "tools"}
REPLAY_DIR = ".replay"
RECORDINGS_DIRNAME = "mock-recordings"


class MockError(Exception):
    """Chyba při vyhodnocení mocku (chybějící fixture …) → tool error."""


@dataclass
class MockSpec:
    server: str
    tool: str | None  # None = _server.md
    path: Path
    mock_dir: Path  # kořenový adresář `mocks/`, odkud soubor pochází (cíl .replay)
    type: Literal["fixed", "agent"] = "fixed"
    body: str = ""
    expect: dict[str, Any] | None = None
    error: bool = False
    abort_when: str | None = None
    tools: list[str] = field(default_factory=list)  # jen _server.md


@dataclass
class MockTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    title: str | None = None


# ---------------------------------------------------------------------------------------------------------------------
# Pomocné funkce
# ---------------------------------------------------------------------------------------------------------------------


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """YAML frontmatter mezi `---` řádky → (dict, tělo). Bez frontmatteru → ({}, text)."""
    if text.startswith("﻿"):
        text = text[1:]
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines(keepends=True)
    if lines[0].strip() != "---":
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            raw = "".join(lines[1:i])
            data = yaml.safe_load(raw) if raw.strip() else {}
            if data is None:
                data = {}
            if not isinstance(data, dict):
                raise ValueError("frontmatter must be a YAML map")
            body = "".join(lines[i + 1 :])
            return data, body.lstrip("\n")
    raise ValueError("unterminated frontmatter (missing closing ---)")


def canonical_input(input: dict[str, Any]) -> str:
    return json.dumps(input, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def input_hash(input: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_input(input).encode()).hexdigest()[:8]


def _value_text(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _lookup_text(input: dict[str, Any], path: str) -> str:
    found, value = lookup(input, path)
    return _value_text(value) if found else ""


_SUBST_RE = re.compile(r"\{\{\s*(?:file:((?:[^{}]|\{[^{}]*\})+?)|input(?:\.([^{}\s]+))?)\s*\}\}")
_INNER_RE = re.compile(r"\{\s*input\.([^{}\s]+)\s*\}")


def render_body(body: str, input: dict[str, Any], fixture_root: list[Path]) -> str:
    """Náhrady `{{input.<path>}}`, `{{input}}` a `{{file:<path s {input.x}>}}` (cesta relativně k `fixture_root`).

    Chybějící hodnota vstupu → prázdný řetězec; chybějící soubor → MockError.
    """

    def repl(m: re.Match[str]) -> str:
        file_tpl, path = m.group(1), m.group(2)
        if file_tpl is not None:
            rel = _INNER_RE.sub(lambda im: _lookup_text(input, im.group(1)), file_tpl.strip())
            return read_fixture(rel, fixture_root)
        if path is None:
            return canonical_input(input)
        return _lookup_text(input, path)

    return _SUBST_RE.sub(repl, body)


def read_fixture(rel: str, roots: list[Path]) -> str:
    """Soubor `rel` v prvním z `roots`, kde existuje (case přebíjí suite). Mimo kořen = chyba."""
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts or not rel:
        raise MockError(f"fixture path {rel!r} must be relative to the mock directory")
    for root in roots:
        cand = (root / p).resolve()
        if not cand.is_relative_to(root.resolve()):
            raise MockError(f"fixture path {rel!r} escapes the mock directory")
        if cand.is_file():
            return cand.read_text(encoding="utf-8", errors="replace")
    raise MockError(f"fixture file {rel} not found")


# ---------------------------------------------------------------------------------------------------------------------
# MockSet
# ---------------------------------------------------------------------------------------------------------------------


class MockSet:
    """Mocky case: `mock_dirs` v pořadí suite → case (pozdější přebíjí soubor po souboru)."""

    def __init__(self, mock_dirs: list[Path]):
        self.mock_dirs = [Path(d) for d in mock_dirs if Path(d).is_dir()]
        self.errors: list[str] = []
        self._specs: dict[str, dict[str, MockSpec]] = {}  # server → tool → spec (jen <tool>.md)
        self._server_specs: dict[str, MockSpec] = {}
        self._tools_json: dict[str, dict[str, dict[str, Any]]] = {}  # server → tool → definice z _tools.json
        self._replay: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._load()

    # --- načtení ------------------------------------------------------------------------------------------------------

    def _load(self) -> None:
        files: dict[str, dict[str, tuple[Path, Path]]] = {}  # server → název souboru → (cesta, mock_dir)
        for mdir in self.mock_dirs:
            for sdir in sorted(mdir.iterdir()):
                if not sdir.is_dir() or sdir.name.startswith("."):
                    continue
                for f in sorted(sdir.iterdir()):
                    if f.is_file() and (f.suffix == ".md" or f.name == "_tools.json"):
                        files.setdefault(sdir.name, {})[f.name] = (f, mdir)
            self._load_replay(mdir)

        for server, entries in files.items():
            for fname, (path, mdir) in entries.items():
                if fname == "_tools.json":
                    self._load_tools_json(server, path)
                elif fname == "_server.md":
                    spec = self._parse(server, None, path, mdir)
                    if spec:
                        self._server_specs[server] = spec
                else:
                    tool = fname[:-3]
                    spec = self._parse(server, tool, path, mdir)
                    if spec:
                        self._specs.setdefault(server, {})[tool] = spec

    def _err(self, path: Path, msg: str) -> None:
        self.errors.append(f"{path}: {msg}")

    def _parse(self, server: str, tool: str | None, path: Path, mdir: Path) -> MockSpec | None:
        try:
            meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
        except (ValueError, yaml.YAMLError, OSError) as ex:
            self._err(path, f"invalid mock file: {ex}")
            return None
        allowed = SERVER_KEYS if tool is None else TOOL_KEYS
        unknown = sorted(set(meta) - allowed)
        if unknown:
            self._err(path, f"unknown key(s) {', '.join(unknown)} (allowed: {', '.join(sorted(allowed))})")
            return None
        mtype = meta.get("type", "agent" if tool is None else "fixed")
        if mtype not in ("fixed", "agent"):
            self._err(path, f"invalid type {mtype!r} (fixed | agent)")
            return None
        if tool is None and mtype != "agent":
            self._err(path, "_server.md must be a `type: agent` mock")
            return None
        expect = meta.get("expect")
        errs = validate_expect(expect)
        for e in errs:
            self._err(path, e)
        if errs:
            return None
        error = meta.get("error", False)
        if not isinstance(error, bool):
            self._err(path, "error must be true or false")
            return None
        if error and mtype == "agent":
            self._err(path, "error: true is supported only for type: fixed")
            return None
        abort_when = meta.get("abort_when")
        if abort_when is not None and (mtype != "agent" or not isinstance(abort_when, str)):
            self._err(path, "abort_when must be a string and is supported only for type: agent")
            return None
        tools: list[str] = []
        if tool is None:
            raw_tools = meta.get("tools")
            if not isinstance(raw_tools, list) or not raw_tools or not all(isinstance(t, str) for t in raw_tools):
                self._err(path, "_server.md needs `tools:` with a list of tool names")
                return None
            tools = list(raw_tools)
            if expect and len(tools) != 1:
                self._err(path, "expect: in _server.md is allowed only when tools: lists a single tool")
                return None
        return MockSpec(server=server, tool=tool, path=path, mock_dir=mdir, type=mtype, body=body,
                        expect=expect or None, error=error, abort_when=abort_when, tools=tools)

    def _load_tools_json(self, server: str, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as ex:
            self._err(path, f"invalid JSON: {ex}")
            return
        tools = data.get("tools") if isinstance(data, dict) else data
        if not isinstance(tools, list):
            self._err(path, "expected {\"tools\": [...]} (saved tools/list response)")
            return
        self._tools_json[server] = {t["name"]: t for t in tools if isinstance(t, dict) and t.get("name")}

    def _load_replay(self, mdir: Path) -> None:
        rdir = mdir / REPLAY_DIR
        if not rdir.is_dir():
            return
        for sdir in sorted(rdir.iterdir()):
            if not sdir.is_dir():
                continue
            for f in sorted(sdir.glob("*.json")):
                try:
                    rec = json.loads(f.read_text(encoding="utf-8"))
                    key = (sdir.name, str(rec["tool"]), canonical_input(rec.get("input") or {}))
                    rec["_path"] = str(f)
                    self._replay[key] = rec  # case mocks (později) přebíjí suite
                except (OSError, json.JSONDecodeError, KeyError, TypeError) as ex:
                    self._err(f, f"invalid replay recording: {ex}")

    # --- dotazy -------------------------------------------------------------------------------------------------------

    def servers(self) -> set[str]:
        """Servery, které mají aspoň jeden mockovaný tool."""
        return {s for s in set(self._specs) | set(self._server_specs) if self.tool_names(s)}

    def tool_names(self, server: str) -> list[str]:
        names = list(self._specs.get(server, {}))
        if server in self._server_specs:
            names += [t for t in self._server_specs[server].tools if t not in names]
        return names

    def has_tool(self, server: str, tool: str) -> bool:
        return self.get(server, tool) is not None

    def get(self, server: str, tool: str) -> MockSpec | None:
        spec = self._specs.get(server, {}).get(tool)
        if spec:
            return spec
        sspec = self._server_specs.get(server)
        if sspec and tool in sspec.tools:
            return sspec
        return None

    def listed_tool(self, server: str, tool: str) -> dict[str, Any] | None:
        return self._tools_json.get(server, {}).get(tool)

    def tools_for(self, server: str) -> list[MockTool]:
        out = []
        for name in self.tool_names(server):
            t = self.listed_tool(server, name)
            if t:
                out.append(MockTool(name=name, description=t.get("description") or f"Mocked tool {name}",
                                    input_schema=t.get("inputSchema") or dict(PERMISSIVE_SCHEMA), title=t.get("title")))
            else:
                out.append(MockTool(name=name, description=f"Mocked tool {name}", input_schema=dict(PERMISSIVE_SCHEMA)))
        return out

    def fixture_roots(self, server: str) -> list[Path]:
        """Adresáře serveru pro `{{file:…}}` – case před suite."""
        return [d / server for d in reversed(self.mock_dirs) if (d / server).is_dir()]

    def replay(self, server: str, tool: str, input: dict[str, Any]) -> dict[str, Any] | None:
        return self._replay.get((server, tool, canonical_input(input)))


def validate_mock_dir(path: Path) -> list[str]:
    """Chyby mock souborů v jednom adresáři `mocks/` (pro `skilleval validate`)."""
    return MockSet([path]).errors


# ---------------------------------------------------------------------------------------------------------------------
# Nahrávky agent mocků
# ---------------------------------------------------------------------------------------------------------------------


def write_recording(recordings_dir: Path, *, spec: MockSpec, server: str, tool: str, input: dict[str, Any],
                    output: str, is_error: bool) -> Path:
    """`<recordings_dir>/<server>/<tool>-<sha8>.json` (formát čte i `.replay/`)."""
    target = recordings_dir / server / f"{tool}-{input_hash(input)}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    rec = {"server": server, "tool": tool, "input": input, "output": output, "is_error": is_error,
           "mock_file": str(spec.path), "mock_dir": str(spec.mock_dir),
           "recorded_at": datetime.now(UTC).isoformat(timespec="seconds")}
    target.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def write_adopt(results_dir: Path) -> Path | None:
    """`ADOPT.txt` v `<results_dir>/mock-recordings/`: každá nahrávka a `.replay/<server>/`, kam ji zkopírovat.

    Vrací cestu k ADOPT.txt, nebo None, když žádné nahrávky nejsou.
    """
    rdir = results_dir if results_dir.name == RECORDINGS_DIRNAME else results_dir / RECORDINGS_DIRNAME
    if not rdir.is_dir():
        return None
    recs = sorted(rdir.glob("*/*.json"))
    if not recs:
        return None
    lines = ["Agent mock recordings from clean runs. Copy a recording into the .replay/<server>/ directory",
             "beside the mock that produced it; later runs answer the identical call from it without a model call.",
             "Commit mocks/.replay/ with the rest of mocks/ so CI runs are repeatable.", ""]
    cmds = []
    for f in recs:
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            target = Path(rec["mock_dir"]) / REPLAY_DIR / rec.get("server", f.parent.name)
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        lines.append(f"{f.relative_to(rdir)}  ->  {target}/")
        cmds.append(f"mkdir -p '{target}' && cp '{f}' '{target}/'")
    if cmds:
        lines += ["", "# shell:"] + cmds
    out = rdir / "ADOPT.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
