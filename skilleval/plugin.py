"""Nalezení pluginu, katalog skillů a MCP konfigurace pluginu (`.mcp.json` / manifest + skilleval.toml + profil)."""

from __future__ import annotations

import json
import logging
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from skilleval.config import ConfigError, Settings, deep_merge, expand_vars
from skilleval.models import McpServerConfig, PluginInfo, SkillInfo
from skilleval.suite.frontmatter import FrontmatterError, read_frontmatter, strip_blank_lines

log = logging.getLogger("skilleval.plugin")

MANIFEST_PATHS = ("plugin.json", ".claude-plugin/plugin.json")
DEFAULT_EVAL_DIR = "evals"
PLUGIN_ROOT_VAR = "${CLAUDE_PLUGIN_ROOT}"
_TYPE_ALIASES = {"streamable-http": "http", "streamable_http": "http", "streamableHttp": "http"}


# ---------------------------------------------------------------------------------------------------------------------
# Nalezení pluginu
# ---------------------------------------------------------------------------------------------------------------------


def manifest_path(root: Path) -> Path | None:
    for rel in MANIFEST_PATHS:
        p = root / rel
        if p.is_file():
            return p
    return None


def read_manifest(root: Path) -> dict[str, Any] | None:
    """Manifest pluginu (plugin.json / .claude-plugin/plugin.json) nebo None. Neplatný JSON = ConfigError."""
    p = manifest_path(root)
    if p is None:
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ConfigError(f"{p}: neplatný JSON manifestu: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: manifest musí být JSON objekt")
    return data


def is_plugin_dir(path: Path) -> bool:
    """Plugin = manifest, nebo skills-directory plugin (SKILL.md v kořeni), nebo adresář se skills/<x>/SKILL.md."""
    if not path.is_dir():
        return False
    if manifest_path(path) is not None or (path / "SKILL.md").is_file():
        return True
    skills = path / "skills"
    return skills.is_dir() and any((d / "SKILL.md").is_file() for d in skills.iterdir() if d.is_dir())


def find_plugin_root(path: Path) -> Path | None:
    """Nejbližší nadřazený adresář (včetně `path`; u souboru jeho adresář), který je plugin."""
    p = Path(path).resolve()
    if not p.is_dir():
        p = p.parent
    for cand in (p, *p.parents):
        if is_plugin_dir(cand):
            return cand
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Eval adresář
# ---------------------------------------------------------------------------------------------------------------------


def _bad_rel_path(value: str) -> bool:
    if not value or not value.strip():
        return True
    pp = PurePosixPath(value.replace("\\", "/"))
    return pp.is_absolute() or Path(value).is_absolute() or ".." in pp.parts


def plugin_eval_dir(root: Path, eval_dir_flag: str | None) -> tuple[Path, list[str]]:
    """(eval adresář, varování). Přednost: --eval-dir > manifest experimental.evals > evals/.

    Flag absolutní / s `..` = ValueError; nepoužitelná hodnota v manifestu = varování a `evals/`."""
    warnings: list[str] = []
    if eval_dir_flag is not None:
        if _bad_rel_path(eval_dir_flag):
            raise ValueError(f"--eval-dir musí být relativní cesta bez '..' (např. 'quality/evals'), ne '{eval_dir_flag}'")
        return root / eval_dir_flag, warnings
    try:
        manifest = read_manifest(root) or {}
    except ConfigError as e:
        warnings.append(f"Warning: {e}")
        manifest = {}
    exp = manifest.get("experimental")
    value = exp.get("evals") if isinstance(exp, dict) else None
    if value is not None:
        if isinstance(value, str) and not _bad_rel_path(value):
            return root / value, warnings
        warnings.append(f"Warning: experimental.evals v manifestu pluginu ({value!r}) není relativní cesta bez '..' "
                        f"– použije se {DEFAULT_EVAL_DIR}/")
    return root / DEFAULT_EVAL_DIR, warnings


# ---------------------------------------------------------------------------------------------------------------------
# Skilly
# ---------------------------------------------------------------------------------------------------------------------


def _load_skill(skill_dir: Path, plugin: str) -> SkillInfo:
    path = skill_dir / "SKILL.md"
    try:
        fm, body = read_frontmatter(path)
    except FrontmatterError as e:
        raise ConfigError(f"{path}: {e}") from e
    fm = fm or {}
    name = fm.get("name")
    desc = fm.get("description")
    return SkillInfo(
        name=str(name).strip() if isinstance(name, str) and name.strip() else skill_dir.name,
        description=" ".join(str(desc).split()) if desc is not None else "",
        body=strip_blank_lines(body),
        path=skill_dir.resolve(),
        plugin=plugin,
    )


def _skill_dirs(root: Path, manifest: dict[str, Any]) -> list[Path]:
    """Adresáře skillů: SKILL.md v kořeni, skills/<x>/, + cesty z manifestu `skills` (adresář skillu nebo jejich rodič)."""
    found: list[Path] = []

    def add_container(d: Path) -> None:
        if (d / "SKILL.md").is_file():
            found.append(d)
            return
        if d.is_dir():
            found.extend(sorted(x for x in d.iterdir() if x.is_dir() and (x / "SKILL.md").is_file()))

    if (root / "SKILL.md").is_file():
        found.append(root)
    add_container(root / "skills")
    extra = manifest.get("skills")
    for rel in [extra] if isinstance(extra, str) else (extra if isinstance(extra, list) else []):
        if isinstance(rel, str):
            add_container((root / rel.replace(PLUGIN_ROOT_VAR, str(root))).resolve())

    out: list[Path] = []
    seen: set[Path] = set()
    for d in found:
        r = d.resolve()
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# MCP servery
# ---------------------------------------------------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ConfigError(f"{path}: neplatná MCP konfigurace: {e}") from e


def _servers_of(data: Any, where: str) -> dict[str, dict[str, Any]]:
    """`{"mcpServers": {...}}` nebo přímo mapa serverů."""
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: MCP konfigurace musí být JSON objekt")
    servers = data.get("mcpServers", data)
    if not isinstance(servers, dict):
        raise ConfigError(f"{where}: mcpServers musí být objekt")
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"{where}: server {name!r} musí být objekt")
    return servers


def _plugin_servers(root: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    servers: dict[str, dict[str, Any]] = {}
    mcp_json = root / ".mcp.json"
    if mcp_json.is_file():
        servers.update(_servers_of(_read_json(mcp_json), str(mcp_json)))
    decl = manifest.get("mcpServers")
    items = decl if isinstance(decl, list) else ([decl] if decl is not None else [])
    for item in items:
        if isinstance(item, str):
            p = Path(item.replace(PLUGIN_ROOT_VAR, str(root)))
            p = p if p.is_absolute() else root / p
            more = _servers_of(_read_json(p), str(p))
        elif isinstance(item, dict):
            more = _servers_of(item, "plugin.json mcpServers")
        else:
            raise ConfigError(f"{root}: mcpServers v manifestu musí být objekt nebo cesta k JSON")
        for name, cfg in more.items():
            servers[name] = deep_merge(servers.get(name, {}), cfg)
    return servers


def _replace_root(value: Any, root: str) -> Any:
    if isinstance(value, str):
        return value.replace(PLUGIN_ROOT_VAR, root)
    if isinstance(value, dict):
        return {k: _replace_root(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_root(v, root) for v in value]
    return value


def _apply_overrides(servers: dict[str, dict[str, Any]], overrides: dict[str, Any], plugin: str, what: str,
                     warnings: list[str]) -> dict[str, dict[str, Any]]:
    """Přepis jen serverů, které plugin deklaruje; neznámé se ignorují (varování, jen když plugin nějaké servery má)."""
    out = dict(servers)
    for name, cfg in (overrides or {}).items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"{what}.{name} musí být tabulka/objekt")
        if name in out:
            out[name] = deep_merge(out[name], cfg)
        elif servers:
            warnings.append(f"Warning: {what}.{name}: plugin {plugin} server {name!r} nedeklaruje – ignorováno")
    return out


def resolve_mcp_servers(root: Path, plugin: str, settings: Settings, *, manifest: dict[str, Any] | None = None,
                        profile: str | None = None, resolve: bool = True,
                        warnings: list[str] | None = None) -> dict[str, McpServerConfig]:
    """Servery pluginu → merge se settings.mcp_servers → profil → ${CLAUDE_PLUGIN_ROOT} → (resolve) expand_vars."""
    warnings = warnings if warnings is not None else []
    if manifest is None:
        manifest = read_manifest(root) or {}
    servers = _plugin_servers(root, manifest)
    servers = _apply_overrides(servers, settings.mcp_servers, plugin, "mcp.servers", warnings)
    if profile:
        if profile not in settings.mcp_profiles:
            known = ", ".join(sorted(settings.mcp_profiles)) or "žádné"
            raise ConfigError(f"MCP profil {profile!r} není v konfiguraci (známé: {known})")
        prof = settings.mcp_profiles[profile] or {}
        servers = _apply_overrides(servers, prof.get("servers", {}) if isinstance(prof, dict) else {}, plugin,
                                   f"mcp.profiles.{profile}.servers", warnings)

    root_s = str(root)
    env = {**settings.env, "CLAUDE_PLUGIN_ROOT": root_s}
    out: dict[str, McpServerConfig] = {}
    for name, cfg in servers.items():
        cfg = _replace_root(cfg, root_s)
        if resolve:
            cfg = expand_vars(cfg, env, where=f"MCP serveru {name!r} pluginu {plugin}")
        cfg = dict(cfg)
        typ = cfg.get("type") or ("http" if cfg.get("url") else "stdio")
        typ = _TYPE_ALIASES.get(typ, typ)
        if typ not in ("http", "sse", "stdio"):
            raise ConfigError(f"MCP server {name!r} pluginu {plugin}: nepodporovaný type {typ!r} (http|sse|stdio)")
        cfg["type"] = typ
        cfg.pop("name", None)
        cfg.pop("plugin", None)
        try:
            out[name] = McpServerConfig(name=name, plugin=plugin, **cfg)
        except ValidationError as e:
            raise ConfigError(f"MCP server {name!r} pluginu {plugin}: neplatná konfigurace: "
                              + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())) from e
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------------------------------------------------


def load_plugin(path: Path, settings: Settings, *, profile: str | None = None, resolve_mcp: bool = True,
                eval_dir: str | None = None, warnings: list[str] | None = None) -> PluginInfo:
    """Načte plugin z jeho kořene. Není-li `path` plugin = ConfigError.

    `warnings` (volitelné) = seznam, do kterého se připíší varování (jinak jen log)."""
    root = Path(path).resolve()
    if not is_plugin_dir(root):
        raise ConfigError(f"{root} není plugin (chybí plugin.json, .claude-plugin/plugin.json, SKILL.md "
                          f"nebo skills/<skill>/SKILL.md)")
    collected: list[str] = []
    manifest = read_manifest(root) or {}
    mname = manifest.get("name")
    name = mname.strip() if isinstance(mname, str) and mname.strip() else root.name
    skills = [_load_skill(d, name) for d in _skill_dirs(root, manifest)]
    servers = resolve_mcp_servers(root, name, settings, manifest=manifest, profile=profile, resolve=resolve_mcp,
                                  warnings=collected)
    edir, w = plugin_eval_dir(root, eval_dir)
    collected += w
    for msg in collected:
        log.warning(msg)
    if warnings is not None:
        warnings.extend(collected)
    return PluginInfo(name=name, root=root, skills=skills, mcp_servers=servers, eval_dir=edir)
