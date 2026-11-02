"""Vestavěné tooly Claude Code (Skill, Read, Glob, Grep, Write, Edit, TodoWrite, AskUserQuestion).

Názvy, vstupy a texty výsledků odpovídají Claude Code (kvůli graderům `tool_used`/`tool_order` a regexům nad trace).
Čtení jen v `readable` (workspace + pluginy + add_dirs), nikdy ve `forbidden` (eval dir) – i přes symlinky; zápis jen
ve workspace. Bash/WebFetch/WebSearch/Agent/NotebookRead/Task* v1 nepodporujeme → `unsupported`, nenabízí se.
"""

from __future__ import annotations

import base64
import glob as globmod
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from skilleval.models import SkillInfo, ToolDef, ToolOutcome

SUPPORTED = ("Skill", "Read", "Glob", "Grep", "Write", "Edit", "TodoWrite", "AskUserQuestion")
ASK_USER_TEXT = ("User is not available in this non-interactive session; proceed with your best judgement "
                 "or stop and explain.")
TODO_TEXT = ("Todos have been modified successfully. Ensure that you continue to use the todo list to track your "
             "progress. Please proceed with the current tasks if applicable")
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
               ".webp": "image/webp"}
READ_DEFAULT_LIMIT = 2000
READ_MAX_LINE = 2000
GLOB_LIMIT = 100
GREP_MAX_FILE = 10 * 1024 * 1024
SKIP_DIRS = {".git", ".hg", ".svn"}
RG_TYPES = {"py": ["*.py", "*.pyi"], "js": ["*.js", "*.mjs", "*.cjs", "*.jsx"], "ts": ["*.ts", "*.tsx", "*.mts"],
            "md": ["*.md", "*.markdown"], "json": ["*.json", "*.jsonl"], "yaml": ["*.yaml", "*.yml"],
            "html": ["*.html", "*.htm"], "css": ["*.css", "*.scss"], "php": ["*.php"], "sh": ["*.sh", "*.bash"],
            "go": ["*.go"], "rust": ["*.rs"], "java": ["*.java"], "toml": ["*.toml"], "txt": ["*.txt"],
            "sql": ["*.sql"], "xml": ["*.xml"], "csv": ["*.csv"]}

AskUser = Callable[[dict[str, Any]], Awaitable[str]]


class ToolInputError(ValueError):
    pass


def _schema(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False,
            "$schema": "http://json-schema.org/draft-07/schema#"}


TOOL_SPECS: dict[str, tuple[str, dict[str, Any]]] = {
    "Skill": (
        "Execute a skill within the main conversation. When the user's task matches one of the available skills, "
        "invoke this tool with the skill name (e.g. \"pdf\" or \"plugin:skill\") BEFORE doing anything else; the "
        "skill's instructions are returned and must be followed. Do not invoke a skill that is already loaded.",
        _schema({"skill": {"type": "string", "description": "The skill name, e.g. \"commit\" or \"plugin:skill\""},
                 "args": {"type": "string", "description": "Optional arguments for the skill"}}, ["skill"])),
    "Read": (
        "Reads a file from the local filesystem. The file_path parameter must be an absolute path. By default reads "
        "up to 2000 lines from the beginning of the file; you can optionally specify a line offset and limit. Lines "
        "longer than 2000 characters are truncated. Results are returned using cat -n format, with line numbers "
        "starting at 1. Images (PNG, JPG, GIF, WebP) are presented visually. Directories cannot be read (use Glob).",
        _schema({"file_path": {"type": "string", "description": "The absolute path to the file to read"},
                 "offset": {"type": "number", "description": "The line number to start reading from"},
                 "limit": {"type": "number", "description": "The number of lines to read"}}, ["file_path"])),
    "Glob": (
        "Fast file pattern matching tool. Supports glob patterns like \"**/*.js\" or \"src/**/*.{ts,tsx}\". Returns "
        "matching file paths sorted by modification time.",
        _schema({"pattern": {"type": "string", "description": "The glob pattern to match files against"},
                 "path": {"type": "string", "description": "The directory to search in. Defaults to the current "
                                                           "working directory."}}, ["pattern"])),
    "Grep": (
        "A powerful search tool for file contents. Supports full regular expression syntax (e.g. \"log.*Error\"). "
        "Filter files with the glob parameter (e.g. \"*.js\") or type parameter (e.g. \"py\"). Output modes: "
        "\"content\" shows matching lines, \"files_with_matches\" shows only file paths (default), \"count\" shows "
        "match counts.",
        _schema({"pattern": {"type": "string", "description": "The regular expression pattern to search for"},
                 "path": {"type": "string", "description": "File or directory to search in. Defaults to the current "
                                                           "working directory."},
                 "glob": {"type": "string", "description": "Glob pattern to filter files (e.g. \"*.js\")"},
                 "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"],
                                 "description": "Output mode, defaults to files_with_matches"},
                 "-B": {"type": "number", "description": "Lines before each match (content mode)"},
                 "-A": {"type": "number", "description": "Lines after each match (content mode)"},
                 "-C": {"type": "number", "description": "Lines before and after each match (content mode)"},
                 "-n": {"type": "boolean", "description": "Show line numbers (content mode). Defaults to true."},
                 "-i": {"type": "boolean", "description": "Case insensitive search"},
                 "type": {"type": "string", "description": "File type to search (e.g. py, js, md)"},
                 "head_limit": {"type": "number", "description": "Limit output to the first N lines/entries"},
                 "multiline": {"type": "boolean", "description": "Enable multiline mode where . matches newlines"}},
                ["pattern"])),
    "Write": (
        "Writes a file to the local filesystem, overwriting an existing file. If the file exists you must Read it "
        "first. Paths must be absolute or relative to the working directory.",
        _schema({"file_path": {"type": "string", "description": "The absolute path to the file to write"},
                 "content": {"type": "string", "description": "The content to write to the file"}},
                ["file_path", "content"])),
    "Edit": (
        "Performs exact string replacements in files. You must Read the file first. The edit fails if old_string "
        "is not unique in the file unless replace_all is true.",
        _schema({"file_path": {"type": "string", "description": "The absolute path to the file to modify"},
                 "old_string": {"type": "string", "description": "The text to replace"},
                 "new_string": {"type": "string", "description": "The text to replace it with"},
                 "replace_all": {"type": "boolean", "default": False,
                                 "description": "Replace all occurrences of old_string"}},
                ["file_path", "old_string", "new_string"])),
    "TodoWrite": (
        "Create and manage a structured task list for the current session to track progress on multi-step tasks.",
        _schema({"todos": {"type": "array", "description": "The updated todo list", "items": {
            "type": "object", "properties": {
                "content": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                "activeForm": {"type": "string"}},
            "required": ["content", "status"]}}}, ["todos"])),
    "AskUserQuestion": (
        "Ask the user multiple-choice questions to gather preferences, clarify ambiguous instructions or get "
        "decisions on implementation choices.",
        _schema({"questions": {"type": "array", "description": "Questions to ask the user (1-4)", "items": {
            "type": "object", "properties": {
                "question": {"type": "string"}, "header": {"type": "string"},
                "options": {"type": "array", "items": {"type": "object", "properties": {
                    "label": {"type": "string"}, "description": {"type": "string"}}, "required": ["label"]}},
                "multiSelect": {"type": "boolean"}},
            "required": ["question"]}}}, ["questions"])),
}


def base_tool_name(entry: str) -> str:
    """`Bash(npm test *)` → `Bash`."""
    entry = entry.strip()
    return entry.split("(", 1)[0].strip() if "(" in entry else entry


def expand_braces(pattern: str) -> list[str]:
    """`*.{ts,tsx}` → [`*.ts`, `*.tsx`] (vnořené i vícenásobné složené závorky)."""
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m or "," not in m.group(1):
        return [pattern]
    out: list[str] = []
    for alt in m.group(1).split(","):
        out.extend(expand_braces(pattern[: m.start()] + alt + pattern[m.end():]))
    return out


def _glob_regex(pattern: str) -> re.Pattern[str]:
    parts = [globmod.translate(p, recursive=True, include_hidden=True) for p in expand_braces(pattern)]
    return re.compile("|".join(f"(?:{p})" for p in parts))


class BuiltinTools:
    """Implementace `ToolProvider` pro vestavěné tooly."""

    def __init__(self, *, workspace: Path, readable: list[Path], forbidden: list[Path], skills: list[SkillInfo],
                 tools: list[str], max_result_chars: int = 100_000, ask_user: AskUser | None = None):
        self.workspace = Path(workspace).resolve()
        self.readable = [self.workspace] + [Path(p).resolve() for p in readable]
        self.forbidden = [Path(p).resolve() for p in forbidden]
        self.skills = list(skills)
        self.max_result_chars = max_result_chars
        self.ask_user = ask_user
        self.unsupported: list[str] = []
        enabled: list[str] = []
        for entry in tools:
            name = base_tool_name(entry)
            if name in SUPPORTED:
                if name not in enabled:
                    enabled.append(name)
            elif entry not in self.unsupported:
                self.unsupported.append(entry)
        self.enabled = [n for n in SUPPORTED if n in enabled]
        self._read_files: set[Path] = set()
        self._snapshot: set[str] = set()
        self.snapshot()

    # --- rozhraní ToolProvider ----------------------------------------------------------------------------------------

    @property
    def tool_defs(self) -> list[ToolDef]:
        return [ToolDef(name=n, description=TOOL_SPECS[n][0], input_schema=TOOL_SPECS[n][1]) for n in self.enabled]

    async def call(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        if name not in self.enabled:
            return ToolOutcome(f"Error: No such tool available: {name}", is_error=True, source="denied")
        handler = getattr(self, f"_t_{name.lower()}")
        try:
            out: ToolOutcome = await handler(args or {})
        except ToolInputError as ex:
            out = ToolOutcome(f"InputValidationError: {ex}", is_error=True)
        except (OSError, UnicodeError, re.error) as ex:
            out = ToolOutcome(f"Error: {ex}", is_error=True)
        if len(out.content) > self.max_result_chars:
            cut = len(out.content) - self.max_result_chars
            out.content = out.content[: self.max_result_chars] + f"\n… [truncated {cut} chars]"
        return out

    # --- soubory vytvořené během runu --------------------------------------------------------------------------------

    def _workspace_files(self) -> set[str]:
        out: set[str] = set()
        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if d != ".git"]
            for f in files:
                out.add(Path(root, f).relative_to(self.workspace).as_posix())
        return out

    def snapshot(self) -> None:
        """Stav workspace po scaffoldu – soubory, které tu už byly, se nepočítají jako vytvořené."""
        self._snapshot = self._workspace_files()

    def files_created(self) -> list[str]:
        return sorted(self._workspace_files() - self._snapshot)

    # --- cesty --------------------------------------------------------------------------------------------------------

    def _resolve(self, raw: Any, key: str = "file_path") -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ToolInputError(f"{key} is required and must be a string")
        p = Path(raw)
        if not p.is_absolute():
            p = self.workspace / p
        return p.resolve()

    def _forbidden(self, p: Path) -> bool:
        return any(p == f or p.is_relative_to(f) for f in self.forbidden)

    def can_read(self, p: Path) -> bool:
        p = p.resolve()
        return not self._forbidden(p) and any(p == r or p.is_relative_to(r) for r in self.readable)

    def can_write(self, p: Path) -> bool:
        p = p.resolve()
        return not self._forbidden(p) and p.is_relative_to(self.workspace) and p != self.workspace

    @staticmethod
    def _denied(p: Path) -> ToolOutcome:
        return ToolOutcome(f"Permission denied: {p} is outside the directories this session can read.",
                           is_error=True)

    def _walk(self, base: Path) -> list[Path]:
        """Čitelné soubory pod base (symlinky resolvované, zakázané vynechané)."""
        out: list[Path] = []
        if base.is_file():
            return [base] if self.can_read(base) else []
        for root, dirs, files in os.walk(base):
            rp = Path(root)
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and self.can_read(rp / d))
            for f in sorted(files):
                fp = rp / f
                if self.can_read(fp) and fp.resolve().is_file():
                    out.append(fp)
        return out

    @staticmethod
    def _int(args: dict[str, Any], key: str, default: int | None = None) -> int | None:
        v = args.get(key)
        if v is None:
            return default
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise ToolInputError(f"{key} must be a number")
        try:
            return int(v)
        except ValueError as ex:
            raise ToolInputError(f"{key} must be a number") from ex

    # --- Skill --------------------------------------------------------------------------------------------------------

    def find_skill(self, name: str) -> SkillInfo | None:
        name = name.strip().lstrip("/")
        for s in self.skills:
            if name == s.qualified_name:
                return s
        for s in self.skills:
            if name == s.name:
                return s
        short = name.rsplit(":", 1)[-1]
        for s in self.skills:
            if short == s.name:
                return s
        return None

    async def _t_skill(self, args: dict[str, Any]) -> ToolOutcome:
        name = args.get("skill")
        if not isinstance(name, str) or not name.strip():
            raise ToolInputError("skill is required and must be a string")
        s = self.find_skill(name)
        if s is None:
            avail = ", ".join(s.qualified_name for s in self.skills) or "(none)"
            return ToolOutcome(f"Unknown skill: {name}. Available skills: {avail}", is_error=True)
        text = f"Base directory for this skill: {s.path.resolve()}\n\n{s.body.strip()}\n"
        if isinstance(args.get("args"), str) and args["args"].strip():
            text += f"\nARGUMENTS: {args['args']}\n"
        return ToolOutcome(text)

    # --- Read ---------------------------------------------------------------------------------------------------------

    async def _t_read(self, args: dict[str, Any]) -> ToolOutcome:
        p = self._resolve(args.get("file_path"))
        if not self.can_read(p):
            return self._denied(p)
        if not p.exists():
            return ToolOutcome(f"File does not exist. Note: your current working directory is {self.workspace}.",
                               is_error=True)
        if p.is_dir():
            return ToolOutcome(f"EISDIR: illegal operation on a directory, read '{p}'", is_error=True)
        if p.suffix.lower() in IMAGE_TYPES:
            data = base64.b64encode(p.read_bytes()).decode()
            self._read_files.add(p)
            return ToolOutcome(f"[image] {p}", images=[{"media_type": IMAGE_TYPES[p.suffix.lower()], "data": data}])
        raw = p.read_bytes()
        if b"\x00" in raw[:8192]:
            return ToolOutcome("This tool cannot read binary files.", is_error=True)
        self._read_files.add(p)
        text = raw.decode("utf-8", errors="replace")
        if not text:
            return ToolOutcome("<system-reminder>Warning: the file exists but the contents are empty.</system-reminder>")
        lines = text.splitlines()
        offset = max(1, self._int(args, "offset", 1) or 1)
        limit = self._int(args, "limit", READ_DEFAULT_LIMIT) or READ_DEFAULT_LIMIT
        if offset > len(lines):
            return ToolOutcome(f"<system-reminder>Warning: the file exists but is shorter than the provided offset "
                               f"({offset}). The file has {len(lines)} lines.</system-reminder>")
        out = []
        for n, line in enumerate(lines[offset - 1: offset - 1 + limit], start=offset):
            if len(line) > READ_MAX_LINE:
                line = line[:READ_MAX_LINE] + "… [line truncated]"
            out.append(f"{n:>6}\t{line}")
        return ToolOutcome("\n".join(out))

    # --- Glob ---------------------------------------------------------------------------------------------------------

    def _split_abs_pattern(self, pattern: str, base: Path) -> tuple[Path, str]:
        if not pattern.startswith("/"):
            return base, pattern
        parts = Path(pattern).parts
        prefix: list[str] = []
        for i, part in enumerate(parts):
            if any(c in part for c in "*?[{"):
                return Path(*prefix), "/".join(parts[i:])
            prefix.append(part)
        return Path(*prefix[:-1]), parts[-1]

    async def _t_glob(self, args: dict[str, Any]) -> ToolOutcome:
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ToolInputError("pattern is required and must be a string")
        base = self._resolve(args["path"], "path") if args.get("path") else self.workspace
        base, pattern = self._split_abs_pattern(pattern, base)
        base = base.resolve()
        if not self.can_read(base):
            return self._denied(base)
        if not base.is_dir():
            return ToolOutcome(f"Path does not exist or is not a directory: {base}", is_error=True)
        rx = _glob_regex(pattern)
        found = [f for f in self._walk(base) if rx.fullmatch(f.relative_to(base).as_posix())]
        if not found:
            return ToolOutcome("No files found")
        found.sort(key=lambda f: (f.stat().st_mtime, str(f)))
        lines = [str(f) for f in found[:GLOB_LIMIT]]
        if len(found) > GLOB_LIMIT:
            lines.append("(Results are truncated. Consider using a more specific path or pattern.)")
        return ToolOutcome("\n".join(lines))

    # --- Grep ---------------------------------------------------------------------------------------------------------

    async def _t_grep(self, args: dict[str, Any]) -> ToolOutcome:
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ToolInputError("pattern is required and must be a string")
        mode = args.get("output_mode") or "files_with_matches"
        if mode not in ("content", "files_with_matches", "count"):
            raise ToolInputError("output_mode must be one of content, files_with_matches, count")
        flags = re.I if args.get("-i") else 0
        multiline = bool(args.get("multiline"))
        if multiline:
            flags |= re.S | re.M
        rx = re.compile(pattern, flags)
        base = self._resolve(args["path"], "path") if args.get("path") else self.workspace
        if not self.can_read(base):
            return self._denied(base)
        if not base.exists():
            return ToolOutcome(f"Path does not exist: {base}", is_error=True)
        file_filters: list[re.Pattern[str]] = []
        if isinstance(args.get("glob"), str) and args["glob"]:
            file_filters.append(_glob_regex(args["glob"]))
        if isinstance(args.get("type"), str) and args["type"]:
            pats = RG_TYPES.get(args["type"], [f"*.{args['type']}"])
            file_filters.append(re.compile("|".join(f"(?:{_glob_regex(x).pattern})" for x in pats)))
        head = self._int(args, "head_limit")
        ctx = self._int(args, "-C", 0) or 0
        before = self._int(args, "-B", ctx) or 0
        after = self._int(args, "-A", ctx) or 0
        show_n = args.get("-n", True) is not False

        def wanted(f: Path) -> bool:
            rel = f.relative_to(base).as_posix() if base.is_dir() else f.name
            return all(fl.fullmatch(rel) or fl.fullmatch(f.name) for fl in file_filters)

        results: list[tuple[Path, list[str], int]] = []
        for f in self._walk(base):
            if file_filters and not wanted(f):
                continue
            try:
                if f.stat().st_size > GREP_MAX_FILE:
                    continue
                raw = f.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:8192]:
                continue
            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines()
            if multiline:
                hit_lines: set[int] = set()
                count = 0
                for m in rx.finditer(text):
                    count += 1
                    first = text.count("\n", 0, m.start())
                    last = text.count("\n", 0, max(m.start(), m.end() - 1))
                    hit_lines.update(range(first, last + 1))
            else:
                hit_lines = {i for i, ln in enumerate(lines) if rx.search(ln)}
                count = sum(len(rx.findall(ln)) or 1 for i, ln in enumerate(lines) if i in hit_lines)
            if not hit_lines:
                continue
            out_lines: list[str] = []
            if mode == "content":
                show: list[int] = sorted({j for i in hit_lines for j in range(max(0, i - before),
                                                                              min(len(lines), i + after + 1))})
                prev = None
                for j in show:
                    if prev is not None and j != prev + 1:
                        out_lines.append("--")
                    sep = ":" if j in hit_lines else "-"
                    out_lines.append(f"{f}{sep}{j + 1}{sep}{lines[j]}" if show_n else f"{f}{sep}{lines[j]}")
                    prev = j
            results.append((f, out_lines, count))

        if not results:
            return ToolOutcome("No matches found" if mode == "content" else "No files found")
        if mode == "files_with_matches":
            results.sort(key=lambda r: (-r[0].stat().st_mtime, str(r[0])))
            paths = [str(r[0]) for r in results]
            if head:
                paths = paths[:head]
            return ToolOutcome(f"Found {len(results)} file{'s' if len(results) != 1 else ''}\n" + "\n".join(paths))
        if mode == "count":
            entries = [f"{r[0]}:{r[2]}" for r in results]
            if head:
                entries = entries[:head]
            total = sum(r[2] for r in results)
            return ToolOutcome("\n".join(entries) + f"\n\nFound {total} total occurrences across {len(results)} files.")
        all_lines = [ln for r in results for ln in r[1]]
        if head:
            all_lines = all_lines[:head]
        return ToolOutcome("\n".join(all_lines))

    # --- Write / Edit -------------------------------------------------------------------------------------------------

    def _write_denied(self, p: Path) -> ToolOutcome:
        return ToolOutcome(f"Permission denied: writes are allowed only inside the working directory "
                           f"({self.workspace}), not {p}.", is_error=True)

    async def _t_write(self, args: dict[str, Any]) -> ToolOutcome:
        p = self._resolve(args.get("file_path"))
        content = args.get("content")
        if not isinstance(content, str):
            raise ToolInputError("content is required and must be a string")
        if not self.can_write(p):
            return self._write_denied(p)
        if p.is_dir():
            return ToolOutcome(f"EISDIR: illegal operation on a directory, open '{p}'", is_error=True)
        existed = p.exists()
        if existed and p not in self._read_files:
            return ToolOutcome("File has not been read yet. Read it first before writing to it.", is_error=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        self._read_files.add(p)
        if existed:
            return ToolOutcome(f"The file {p} has been updated successfully.")
        return ToolOutcome(f"File created successfully at: {p}")

    async def _t_edit(self, args: dict[str, Any]) -> ToolOutcome:
        p = self._resolve(args.get("file_path"))
        old, new = args.get("old_string"), args.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str):
            raise ToolInputError("old_string and new_string are required strings")
        replace_all = bool(args.get("replace_all"))
        if not self.can_write(p):
            return self._write_denied(p)
        if old == new:
            return ToolOutcome("No changes to make: old_string and new_string are exactly the same.", is_error=True)
        if not p.exists():
            if old == "":
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(new, encoding="utf-8")
                self._read_files.add(p)
                return ToolOutcome(f"File created successfully at: {p}")
            return ToolOutcome(f"File does not exist. Note: your current working directory is {self.workspace}.",
                               is_error=True)
        if p.is_dir():
            return ToolOutcome(f"EISDIR: illegal operation on a directory, read '{p}'", is_error=True)
        if p not in self._read_files:
            return ToolOutcome("File has not been read yet. Read it first before writing to it.", is_error=True)
        text = p.read_text(encoding="utf-8", errors="replace")
        if old == "":
            if text.strip():
                return ToolOutcome("Cannot create new file - file already exists.", is_error=True)
            p.write_text(new, encoding="utf-8")
            return ToolOutcome(f"The file {p} has been updated successfully.")
        n = text.count(old)
        if n == 0:
            return ToolOutcome(f"String to replace not found in file.\nString: {old}", is_error=True)
        if n > 1 and not replace_all:
            return ToolOutcome(f"Found {n} matches of the string to replace, but replace_all is false. To replace all "
                               f"occurrences, set replace_all to true. To replace only one occurrence, please provide "
                               f"more context to uniquely identify the instance.\nString: {old}", is_error=True)
        p.write_text(text.replace(old, new) if replace_all else text.replace(old, new, 1), encoding="utf-8")
        if replace_all and n > 1:
            return ToolOutcome(f"The file {p} has been updated. All occurrences of '{old}' were successfully "
                               f"replaced with '{new}'.")
        return ToolOutcome(f"The file {p} has been updated successfully.")

    # --- TodoWrite / AskUserQuestion ----------------------------------------------------------------------------------

    async def _t_todowrite(self, args: dict[str, Any]) -> ToolOutcome:
        if not isinstance(args.get("todos"), list):
            raise ToolInputError("todos is required and must be an array")
        return ToolOutcome(TODO_TEXT)

    async def _t_askuserquestion(self, args: dict[str, Any]) -> ToolOutcome:
        if not isinstance(args.get("questions"), list) or not args["questions"]:
            raise ToolInputError("questions is required and must be a non-empty array")
        if self.ask_user is not None:
            return ToolOutcome(await self.ask_user(args))
        return ToolOutcome(ASK_USER_TEXT)
