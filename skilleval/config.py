"""Konfigurace: .env soubory, skilleval.toml, aliasy modelů, ceny, expanze ${VAR}."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_ALIASES = {
    "haiku": "anthropic/claude-haiku-5.5",
    "sonnet": "anthropic/claude-sonnet-5.5",
    "opus": "anthropic/claude-opus-5.5",
}

SECRET_KEY_RE = re.compile(r"(authorization|token|key|password|secret|cookie)", re.I)


class ConfigError(ValueError):
    pass


_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_vars(value: Any, env: dict[str, str], where: str = "") -> Any:
    """Rekurzivně expanduje ${VAR} a ${VAR:-default}. Nedefinovaná proměnná bez defaultu = ConfigError."""
    if isinstance(value, str):

        def repl(m: re.Match) -> str:
            name, default = m.group(1), m.group(2)
            if name in env and env[name] != "":
                return env[name]
            if default is not None:
                return default
            raise ConfigError(f"Nedefinovaná proměnná ${{{name}}}" + (f" v {where}" if where else ""))

        return _VAR_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: expand_vars(v, env, where) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_vars(v, env, where) for v in value]
    return value


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Settings:
    env: dict[str, str] = field(default_factory=dict)  # sloučené env (soubory + proces)
    config_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)  # skilleval.toml (neexpandovaný)
    aliases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ALIASES))
    default_model: str = "haiku"
    judge_model: str = "haiku"
    model_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)  # neexpandované přepisy serverů
    mcp_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_tool_result_chars: int = 100_000
    max_tool_name_len: int = 64
    prefer_structured_content: bool = True  # jako CC: je-li structuredContent, model dostane jeho JSON
    defer_tools: str = "auto"  # auto | on | off – odložené MCP tooly načítané přes ToolSearch (jako Claude Code)
    defer_tools_min_tokens: int = 10_000  # auto: odkládat, když schémata MCP toolů mají odhadem víc tokenů

    @property
    def api_key(self) -> str:
        return (self.env.get("OPENROUTER_API_KEY") or "").strip()

    @property
    def base_url(self) -> str:
        return self.env.get("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL

    def resolve_model(self, name: str | None, *, judge: bool = False) -> str:
        name = name or (self.env.get("SKILLEVAL_JUDGE_MODEL") if judge else self.env.get("SKILLEVAL_MODEL"))
        name = name or (self.judge_model if judge else self.default_model)
        return self.aliases.get(name, name)

    def options_for(self, slug: str) -> dict[str, Any]:
        return dict(self.model_options.get(slug, {}))

    def secrets(self) -> list[str]:
        """Hodnoty, které se nesmí objevit ve výstupech (maskování)."""
        out = []
        for k, v in self.env.items():
            if v and len(v) >= 8 and SECRET_KEY_RE.search(k):
                out.append(v)
        return out

    def mask(self, text: str) -> str:
        for s in self.secrets():
            text = text.replace(s, "***")
        return text


def find_config(explicit: str | None, plugin_root: Path | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise ConfigError(f"Konfigurace {p} neexistuje")
        return p
    candidates = [Path.cwd() / "skilleval.toml"]
    if plugin_root:
        candidates.append(plugin_root / "skilleval.toml")
    candidates.append(APP_DIR / "skilleval.toml")
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_settings(config: str | None = None, env_files: list[str] | None = None,
                  plugin_root: Path | None = None) -> Settings:
    path = find_config(config, plugin_root)
    raw: dict[str, Any] = {}
    if path:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)

    env: dict[str, str] = {}
    files: list[str] = []
    if (APP_DIR / ".env").is_file():
        files.append(str(APP_DIR / ".env"))
    files += [str((path.parent / f) if path and not os.path.isabs(f) else f) for f in raw.get("env_files", [])]
    files += list(env_files or [])
    for f in files:
        if not os.path.isfile(f):
            raise ConfigError(f"Env soubor {f} neexistuje")
        env.update({k: v for k, v in dotenv_values(f).items() if v is not None})
    env.update(os.environ)

    s = Settings(env=env, config_path=path, raw=raw)
    models = raw.get("models", {})
    s.aliases.update(models.get("aliases", {}))
    s.default_model = models.get("default", s.default_model)
    s.judge_model = models.get("judge", s.judge_model)
    s.model_options = models.get("options", {})
    s.pricing = raw.get("pricing", {})
    mcp = raw.get("mcp", {})
    s.mcp_servers = mcp.get("servers", {})
    s.mcp_profiles = mcp.get("profiles", {})
    limits = raw.get("limits", {})
    s.max_tool_result_chars = int(limits.get("max_tool_result_chars", s.max_tool_result_chars))
    s.max_tool_name_len = int(limits.get("max_tool_name_len", s.max_tool_name_len))
    s.prefer_structured_content = bool(limits.get("prefer_structured_content", s.prefer_structured_content))
    defer = str(limits.get("defer_tools", s.defer_tools)).lower()
    if defer in ("true", "false"):  # TOML bool
        defer = "on" if defer == "true" else "off"
    if defer not in ("auto", "on", "off"):
        raise ConfigError(f"limits.defer_tools musí být auto|on|off, ne {defer!r}")
    s.defer_tools = defer
    s.defer_tools_min_tokens = int(limits.get("defer_tools_min_tokens", s.defer_tools_min_tokens))
    return s
