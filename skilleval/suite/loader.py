"""Načtení eval sady ve formátu Claude Code plugin evals (prompt.md, case.yaml, graders/*.md, skilleval.yaml)."""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from skilleval.config import Settings
from skilleval.models import CaseSpec, ContextSpec, ExtensionSpec, GraderSpec, LoadError, Suite
from skilleval.plugin import find_plugin_root, is_plugin_dir, load_plugin, plugin_eval_dir
from skilleval.suite.discovery import CASE_FILES, discover_case_dirs, is_case_dir
from skilleval.suite.frontmatter import FrontmatterError, load_yaml_file, read_frontmatter, strip_blank_lines

SCHEMA_VERSION = "1.1"

# Pole case sdílená prompt.md frontmatterem a case.yaml (top-level / execution)
TOP_FIELDS = ("description", "tags", "plugins", "runs", "expected_outcome")
EXECUTION_FIELDS = ("model", "max_turns", "timeout_seconds", "allowed_tools", "append_system_prompt", "env")
PROMPT_KEYS = frozenset({"schema_version", "name", *TOP_FIELDS, *EXECUTION_FIELDS})
YAML_KEYS = frozenset({"schema_version", "name", *TOP_FIELDS, "context", "graders", "execution"})
YAML_EXECUTION_KEYS = frozenset({*EXECUTION_FIELDS, "prompt"})
CONTEXT_KEYS = frozenset({"scaffold_script", "history_file", "add_dirs"})

GRADER_COMMON = frozenset({"type", "weight", "arm"})
GRADER_OPTIONS: dict[str, frozenset[str]] = {
    "regex": frozenset({"pattern", "flags", "match", "target"}),
    "tool_used": frozenset({"tool", "input_match", "min", "max"}),
    "tool_order": frozenset({"before", "after"}),
    "file_exists": frozenset({"path", "exists"}),
    "llm": frozenset({"criteria", "focus"}),
    "baseline": frozenset({"baseline_file", "criteria"}),
}
GRADER_ARMS = ("with-only", "both")
TARGET_VALUES = ("last_message", "trace", "files", "mock_calls")
JS_FLAGS = frozenset("dgimsuvy")

ENV_KEY_RE = re.compile(r"^EVAL_[A-Z0-9_]*$")
_COUNT_RE = re.compile(r"^count:(\d+)$")
LIMITS = {"runs": (1, 50), "max_turns": (1, 200), "timeout_seconds": (1, 3600)}


class CaseLoadError(Exception):
    """Chyba načtení jednoho case (→ Suite.load_errors). `path` = soubor/adresář, kterého se týká."""

    def __init__(self, message: str, path: Path | None = None):
        super().__init__(message)
        self.message = message
        self.path = path


# ---------------------------------------------------------------------------------------------------------------------
# Normalizace hodnot
# ---------------------------------------------------------------------------------------------------------------------


def _type_name(v: Any) -> str:
    return {dict: "mapa", list: "seznam", bool: "boolean", str: "text"}.get(type(v), type(v).__name__)


def _scalar_str(key: str, v: Any, where: str) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, int | float) and not isinstance(v, bool):
        return str(v)
    raise CaseLoadError(f"{where}: {key} musí být text, ne {_type_name(v)}")


def _str_list(key: str, v: Any, where: str) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if not isinstance(v, list):
        raise CaseLoadError(f"{where}: {key} musí být seznam, ne {_type_name(v)}")
    return [_scalar_str(f"{key}[{i}]", x, where) for i, x in enumerate(v)]


def _split_tools(s: str) -> list[str]:
    """`"Read, Glob Bash(npm test *)"` → položky oddělené čárkou/mezerou mimo závorky."""
    out, buf, depth = [], [], 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if depth == 0 and (ch == "," or ch.isspace()):
            if buf:
                out.append("".join(buf))
                buf = []
            continue
        buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def _int_in(key: str, v: Any, where: str) -> int:
    lo, hi = LIMITS[key]
    if isinstance(v, bool) or not isinstance(v, int):
        raise CaseLoadError(f"{where}: {key} musí být celé číslo {lo}–{hi}, ne {v!r}")
    if not lo <= v <= hi:
        raise CaseLoadError(f"{where}: {key} musí být v rozsahu {lo}–{hi}, je {v}")
    return v


def _env(v: Any, where: str) -> dict[str, str]:
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise CaseLoadError(f"{where}: env musí být mapa, ne {_type_name(v)}")
    out: dict[str, str] = {}
    for k, val in v.items():
        if not isinstance(k, str) or not ENV_KEY_RE.match(k):
            raise CaseLoadError(f"{where}: env klíč {k!r} neodpovídá EVAL_[A-Z0-9_]*")
        if val is None:
            out[k] = ""
        elif isinstance(val, bool):
            out[k] = "true" if val else "false"
        else:
            out[k] = _scalar_str(f"env.{k}", val, where)
    return out


def _norm_field(key: str, v: Any, where: str) -> Any:
    if key in ("description", "expected_outcome", "model", "append_system_prompt"):
        return None if v is None else _scalar_str(key, v, where)
    if key in ("tags", "plugins"):
        return _str_list(key, v, where)
    if key == "allowed_tools":
        if isinstance(v, str):
            return _split_tools(v)
        return _str_list(key, v, where)
    if key in LIMITS:
        return None if v is None and key == "runs" else _int_in(key, v, where)
    if key == "env":
        return _env(v, where)
    raise AssertionError(key)  # pragma: no cover


def _check_schema_version(v: Any, where: str) -> None:
    if str(v) != SCHEMA_VERSION:
        raise CaseLoadError(f"{where}: nepodporované schema_version {v!r} (očekává se \"{SCHEMA_VERSION}\")")


def _check_name(v: Any, where: str) -> str:
    name = _scalar_str("name", v, where).strip() if v is not None else ""
    if not name:
        raise CaseLoadError(f"{where}: name nesmí být prázdné")
    return name


def _unknown(keys: Any, allowed: frozenset[str], where: str, label: str = "neznámý klíč") -> None:
    bad = [str(k) for k in keys if k not in allowed]
    if bad:
        raise CaseLoadError(f"{where}: {label} {', '.join(repr(b) for b in bad)} "
                            f"(povolené: {', '.join(sorted(allowed))})")


# ---------------------------------------------------------------------------------------------------------------------
# Gradery
# ---------------------------------------------------------------------------------------------------------------------


def _check_target(key: str, v: Any, where: str) -> None:
    if isinstance(v, str) and v in TARGET_VALUES:
        return
    if isinstance(v, dict) and set(v) == {"source", "path"} and v.get("source") == "file" \
            and isinstance(v.get("path"), str) and v["path"].strip():
        return
    raise CaseLoadError(f"{where}: {key} musí být {'|'.join(TARGET_VALUES)} nebo {{source: file, path: <cesta>}}, "
                        f"ne {v!r}")


def _req_str(cfg: dict[str, Any], key: str, where: str) -> None:
    v = cfg.get(key)
    if not isinstance(v, str) or not v.strip():
        if key not in cfg or v is None:
            raise CaseLoadError(f"{where}: chybí povinná volba {key}")
        raise CaseLoadError(f"{where}: {key} musí být neprázdný text")


def _opt_str(cfg: dict[str, Any], key: str, where: str) -> None:
    if key in cfg and not isinstance(cfg[key], str):
        raise CaseLoadError(f"{where}: {key} musí být text")


def _opt_count(cfg: dict[str, Any], key: str, where: str) -> int | None:
    if key not in cfg or cfg[key] is None:
        return None
    v = cfg[key]
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise CaseLoadError(f"{where}: {key} musí být nezáporné celé číslo, ne {v!r}")
    return v


def _check_tool_ref(key: str, v: Any, where: str) -> None:
    if isinstance(v, str) and v.strip():
        return
    if isinstance(v, dict):
        _unknown(v, frozenset({"tool", "input_match"}), f"{where}: {key}")
        _req_str(v, "tool", f"{where}: {key}")
        _opt_str(v, "input_match", f"{where}: {key}")
        return
    if v is None:
        raise CaseLoadError(f"{where}: chybí povinná volba {key}")
    raise CaseLoadError(f"{where}: {key} musí být název toolu nebo {{tool, input_match}}")


def _check_grader_options(typ: str, cfg: dict[str, Any], where: str) -> None:
    if typ == "regex":
        _req_str(cfg, "pattern", where)
        flags = cfg.get("flags")
        if flags is not None:
            if not isinstance(flags, str) or any(c not in JS_FLAGS for c in flags) or len(set(flags)) != len(flags):
                raise CaseLoadError(f"{where}: flags musí být JS flagy z '{''.join(sorted(JS_FLAGS))}', ne {flags!r}")
        match = cfg.get("match")
        if match is not None and not (match in ("contains", "not_contains")
                                      or (isinstance(match, str) and _COUNT_RE.match(match))):
            raise CaseLoadError(f"{where}: match musí být contains | not_contains | count:N, ne {match!r}")
        if cfg.get("target") is not None:
            _check_target("target", cfg["target"], where)
    elif typ == "tool_used":
        _req_str(cfg, "tool", where)
        _opt_str(cfg, "input_match", where)
        lo, hi = _opt_count(cfg, "min", where), _opt_count(cfg, "max", where)
        if hi is not None and (lo if lo is not None else 1) > hi:
            raise CaseLoadError(f"{where}: min ({lo if lo is not None else 1}) je větší než max ({hi})")
    elif typ == "tool_order":
        _check_tool_ref("before", cfg.get("before"), where)
        _check_tool_ref("after", cfg.get("after"), where)
    elif typ == "file_exists":
        _req_str(cfg, "path", where)
        if "exists" in cfg and not isinstance(cfg["exists"], bool):
            raise CaseLoadError(f"{where}: exists musí být true/false")
    elif typ == "llm":
        _req_str(cfg, "criteria", where)
        if cfg.get("focus") is not None:
            _check_target("focus", cfg["focus"], where)
    elif typ == "baseline":
        _req_str(cfg, "baseline_file", where)
        _req_str(cfg, "criteria", where)


def build_grader(name: str, data: dict[str, Any], *, body: str | None, where: str) -> GraderSpec:
    """Grader z frontmatteru + těla (`body` != None = soubor graders/<name>.md) nebo z položky `graders:` v case.yaml."""
    opts = dict(data)
    typ = opts.pop("type", None)
    if typ is None:
        raise CaseLoadError(f"{where}: chybí povinný klíč type ({' | '.join(GRADER_OPTIONS)})")
    if typ not in GRADER_OPTIONS:
        raise CaseLoadError(f"{where}: neznámý type {typ!r} ({' | '.join(GRADER_OPTIONS)})")
    weight = opts.pop("weight", 1)
    if isinstance(weight, bool) or not isinstance(weight, int | float) or not weight > 0:
        raise CaseLoadError(f"{where}: weight musí být kladné číslo, ne {weight!r}")
    arm = opts.pop("arm", None)
    if arm is not None and arm not in GRADER_ARMS:
        raise CaseLoadError(f"{where}: arm musí být with-only nebo both, ne {arm!r}")
    _unknown(opts, GRADER_OPTIONS[typ], where, f"neznámá volba pro type {typ}:")

    if body is not None:
        markdown = strip_blank_lines(body)
        if typ in ("llm", "baseline") and markdown:
            # pořadí jako v CC: criteria (z těla) první, pak ostatní volby
            config = {"criteria": markdown, **{k: v for k, v in opts.items() if k != "criteria"}}
        elif typ == "regex" and "pattern" not in opts and markdown:
            config = {"pattern": markdown, **opts}
        else:
            config = opts
    else:
        config = opts
        md = opts.get("criteria") if typ in ("llm", "baseline") else opts.get("pattern")
        markdown = md if isinstance(md, str) else ""
    _check_grader_options(typ, config, where)
    return GraderSpec(name=name, type=typ, weight=weight, arm=arm, config=config, markdown=markdown)


# ---------------------------------------------------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------------------------------------------------


def _load_case_yaml(path: Path) -> tuple[dict[str, Any], str | None, ContextSpec, list[GraderSpec], str]:
    """→ (pole case, execution.prompt, context, gradery, name)."""
    where = "invalid case.yaml"
    try:
        data = load_yaml_file(path)
    except FrontmatterError as e:
        raise CaseLoadError(f"{where}: {e}", path) from e
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise CaseLoadError(f"{where}: soubor musí být YAML mapa", path)
    try:
        _unknown(data, YAML_KEYS, where)
        if "schema_version" not in data:
            raise CaseLoadError(f"{where}: chybí povinné schema_version: \"{SCHEMA_VERSION}\"")
        _check_schema_version(data["schema_version"], where)
        if data.get("name") is None:
            raise CaseLoadError(f"{where}: chybí povinné name")
        name = _check_name(data["name"], where)

        fields = {k: _norm_field(k, data[k], where) for k in TOP_FIELDS if k in data}

        execution = data.get("execution") or {}
        if not isinstance(execution, dict):
            raise CaseLoadError(f"{where}: execution musí být mapa")
        _unknown(execution, YAML_EXECUTION_KEYS, f"{where}: execution")
        fields.update({k: _norm_field(k, execution[k], f"{where}: execution") for k in EXECUTION_FIELDS
                       if k in execution})
        prompt = execution.get("prompt")
        if prompt is not None:
            prompt = strip_blank_lines(_scalar_str("prompt", prompt, f"{where}: execution"))

        ctx = data.get("context") or {}
        if not isinstance(ctx, dict):
            raise CaseLoadError(f"{where}: context musí být mapa")
        _unknown(ctx, CONTEXT_KEYS, f"{where}: context")
        context = ContextSpec(
            scaffold_script=None if ctx.get("scaffold_script") is None
            else _scalar_str("scaffold_script", ctx["scaffold_script"], f"{where}: context"),
            history_file=None if ctx.get("history_file") is None
            else _scalar_str("history_file", ctx["history_file"], f"{where}: context"),
            add_dirs=_str_list("add_dirs", ctx.get("add_dirs"), f"{where}: context"),
        )

        graders: list[GraderSpec] = []
        raw_graders = data.get("graders")
        if raw_graders is not None and not isinstance(raw_graders, list):
            raise CaseLoadError(f"{where}: graders musí být seznam")
        for i, g in enumerate(raw_graders or []):
            gwhere = f"{where}: graders[{i}]"
            if not isinstance(g, dict):
                raise CaseLoadError(f"{gwhere} musí být mapa")
            g = dict(g)
            gname = g.pop("name", None)
            if gname is None or not str(gname).strip():
                raise CaseLoadError(f"{gwhere}: chybí povinné name")
            gname = _scalar_str("name", gname, gwhere).strip()
            graders.append(build_grader(gname, g, body=None, where=f"{where}: graders[{gname}]"))
    except CaseLoadError as e:
        e.path = e.path or path
        raise
    return fields, prompt, context, graders, name


def _load_grader_files(case_dir: Path) -> list[GraderSpec]:
    gdir = case_dir / "graders"
    if not gdir.is_dir():
        return []
    out = []
    for f in sorted(p for p in gdir.iterdir() if p.is_file() and p.suffix == ".md" and not p.name.startswith(".")):
        where = f"graders/{f.name}"
        try:
            fm, body = read_frontmatter(f)
        except FrontmatterError as e:
            raise CaseLoadError(f"{where}: {e}", f) from e
        try:
            out.append(build_grader(f.stem, fm or {}, body=body, where=where))
        except CaseLoadError as e:
            e.path = f
            raise
    return out


def _load_extension(case_dir: Path) -> ExtensionSpec | None:
    path = case_dir / "skilleval.yaml"
    if not path.is_file():
        return None
    try:
        data = load_yaml_file(path)
    except FrontmatterError as e:
        raise CaseLoadError(f"skilleval.yaml: {e}", path) from e
    if data is None or data == {}:
        return None
    if not isinstance(data, dict):
        raise CaseLoadError("skilleval.yaml: soubor musí být YAML mapa", path)
    try:
        return ExtensionSpec.model_validate(data)
    except ValidationError as e:
        msgs = "; ".join(f"{'.'.join(map(str, x['loc'])) or '<root>'}: {x['msg']}" for x in e.errors())
        raise CaseLoadError(f"skilleval.yaml: {msgs}", path) from e


def _rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def load_case(case_dir: Path, *, plugin_root: Path | None, eval_dir: Path, suite_root: Path | None = None) -> CaseSpec:
    """Načte jeden case (prompt.md a/nebo case.yaml). Chyba = CaseLoadError.

    `suite_root` = základ pro `rel_dir` (default kořen pluginu, jinak rodič eval adresáře)."""
    case_dir = Path(case_dir).resolve()
    eval_dir = Path(eval_dir).resolve()
    prompt_path, yaml_path = case_dir / "prompt.md", case_dir / "case.yaml"
    has_prompt, has_yaml = prompt_path.is_file(), yaml_path.is_file()
    if not has_prompt and not has_yaml:
        raise CaseLoadError("adresář neobsahuje prompt.md ani case.yaml", case_dir)

    fields: dict[str, Any] = {}
    name = case_dir.name
    prompt: str | None = None
    context = ContextSpec()
    graders: list[GraderSpec] = []
    if has_yaml:
        fields, prompt, context, graders, name = _load_case_yaml(yaml_path)

    if has_prompt:
        where = "prompt.md"
        try:
            fm, body = read_frontmatter(prompt_path)
            fm = fm or {}
            _unknown(fm, PROMPT_KEYS, where)
            if "schema_version" in fm:
                _check_schema_version(fm["schema_version"], where)
            if "name" in fm:
                name = _check_name(fm["name"], where)
            # frontmatter prompt.md přebíjí case.yaml
            fields.update({k: _norm_field(k, v, where) for k, v in fm.items() if k not in ("schema_version", "name")})
        except FrontmatterError as e:
            raise CaseLoadError(f"{where}: {e}", prompt_path) from e
        except CaseLoadError as e:
            e.path = prompt_path
            raise
        body_prompt = strip_blank_lines(body)
        if body_prompt.strip():
            prompt = body_prompt

    if not prompt or not prompt.strip():
        raise CaseLoadError("chybí prompt (tělo prompt.md nebo execution.prompt v case.yaml)",
                            prompt_path if has_prompt else yaml_path)

    graders += _load_grader_files(case_dir)  # graders/*.md za gradery z case.yaml
    if not graders:
        raise CaseLoadError("invalid case.yaml: graders – case nemá žádný grader "
                            "(přidej graders/<name>.md nebo seznam graders: v case.yaml)", case_dir)
    seen: set[str] = set()
    for g in graders:
        if g.name in seen:
            raise CaseLoadError(f"duplicitní název graderu {g.name!r}", case_dir)
        seen.add(g.name)

    plugin_paths: list[Path] = []
    for p in fields.pop("plugins", None) or []:
        pp = Path(p).expanduser()
        pp = (pp if pp.is_absolute() else case_dir / pp).resolve()
        if not pp.is_dir():
            raise CaseLoadError(f"plugins: adresář {p!r} neexistuje ({pp})", prompt_path if has_prompt else yaml_path)
        plugin_paths.append(pp)
    if not plugin_paths and plugin_root is not None:
        plugin_paths = [Path(plugin_root).resolve()]

    mock_dirs = [d for d in (eval_dir / "mocks", case_dir / "mocks") if d.is_dir()]
    base = Path(suite_root).resolve() if suite_root else (Path(plugin_root).resolve() if plugin_root else eval_dir.parent)
    if has_prompt and has_yaml:
        source = "mixed"
    else:
        source = "prompt" if has_prompt else "yaml"

    return CaseSpec(
        name=name,
        dir=case_dir,
        rel_dir=_rel(case_dir, base),
        source=source,
        prompt=prompt,
        plugins=plugin_paths,
        context=context,
        graders=graders,
        mock_dirs=mock_dirs,
        extension=_load_extension(case_dir),
        **{k: v for k, v in fields.items() if v is not None},
    )


# ---------------------------------------------------------------------------------------------------------------------
# Sada
# ---------------------------------------------------------------------------------------------------------------------


def _resolve_target(target: str | Path | None) -> Path:
    if target is None or str(target) == "":
        return Path.cwd().resolve()
    p = Path(target).expanduser()
    if p.exists():
        return p.resolve()
    s = str(target)
    if "@" in s or ("/" not in s and "\\" not in s):
        raise ValueError(f"Target {s!r} není existující cesta. Instalované pluginy (name nebo name@marketplace) "
                         f"skilleval nepodporuje – předej adresář pluginu, např. ./plugins/{s.split('@')[0]}, "
                         f"nebo jeden prompt.md / case.yaml.")
    raise ValueError(f"Target {s!r} neexistuje")


def load_suite(target: str | Path | None, settings: Settings, *, eval_dir: str | None = None,
               case_globs: list[str] | None = None, tags: list[str] | None = None,
               profile: str | None = None, resolve_mcp: bool = True, warnings: list[str] | None = None) -> Suite:
    """Načte sadu: adresář pluginu, jeden prompt.md/case.yaml (nebo case adresář), skupinu v eval adresáři, None = CWD.

    Chyby jednotlivých cases jdou do `suite.load_errors`; chyby targetu/--eval-dir = ValueError, pluginu = ConfigError.
    `warnings` (volitelné) = seznam, kam se připíší varování (např. nepoužitelné experimental.evals)."""
    tpath = _resolve_target(target)
    warn: list[str] = []

    single_case: Path | None = None
    scan_dir: Path | None = None
    root: Path | None = None
    if tpath.is_file():
        if tpath.name not in CASE_FILES:
            raise ValueError(f"Target {tpath} není adresář pluginu ani prompt.md / case.yaml")
        single_case = tpath.parent
        root = find_plugin_root(single_case)
    elif is_plugin_dir(tpath):
        root = tpath
    elif is_case_dir(tpath):
        single_case = tpath
        root = find_plugin_root(tpath)
    else:
        enclosing = find_plugin_root(tpath)
        if enclosing is not None:
            edir = plugin_eval_dir(enclosing, eval_dir)[0].resolve()
            if tpath.is_relative_to(edir):  # skupina uvnitř eval adresáře pluginu
                root, scan_dir = enclosing, tpath

    plugin = load_plugin(root, settings, profile=profile, resolve_mcp=resolve_mcp, eval_dir=eval_dir,
                         warnings=warn) if root else None
    if plugin is not None:
        suite_root = plugin.root
        edir = (plugin.eval_dir or plugin.root / "evals").resolve()
        if single_case is not None and not single_case.is_relative_to(edir):
            edir = single_case.parent
    elif single_case is not None:
        suite_root = edir = single_case.parent
    else:
        suite_root = tpath
        edir_p, w = plugin_eval_dir(tpath, eval_dir)
        warn += w
        edir = edir_p.resolve()

    dirs = [single_case] if single_case is not None else discover_case_dirs(scan_dir or edir)
    cases: list[CaseSpec] = []
    error_dirs: list[tuple[Path, LoadError]] = []
    names: dict[str, Path] = {}
    for d in dirs:
        try:
            case = load_case(d, plugin_root=plugin.root if plugin else None, eval_dir=edir, suite_root=suite_root)
        except CaseLoadError as e:
            err = LoadError(path=str(e.path or d), message=e.message)
            error_dirs.append((d, err))
            continue
        if case.name in names:
            err = LoadError(path=str(d), message=f"duplicitní název case {case.name!r} (už v {names[case.name]})")
            error_dirs.append((d, err))
            continue
        names[case.name] = d
        cases.append(case)

    if case_globs:
        cases = [c for c in cases if any(fnmatchcase(c.name, g) for g in case_globs)]
        error_dirs = [(d, e) for d, e in error_dirs if any(fnmatchcase(d.name, g) for g in case_globs)]
    if tags:
        cases = [c for c in cases if any(t in c.all_tags for t in tags)]
    if warnings is not None:
        warnings.extend(warn)
    return Suite(root=suite_root, eval_dir=edir, plugin=plugin, cases=cases,
                 load_errors=[e for _, e in error_dirs])
