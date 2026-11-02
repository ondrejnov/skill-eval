"""skilleval.plugin – nalezení pluginu, skilly, eval adresář, MCP konfigurace (.mcp.json + manifest + toml + profil)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skilleval.config import ConfigError, Settings
from skilleval.plugin import find_plugin_root, is_plugin_dir, load_plugin, plugin_eval_dir

FIX = Path(__file__).parent / "fixtures" / "suite"
DEMO = FIX / "demo-plugin"
SKILLDIR = FIX / "skills" / "my-skill"


def write(base: Path, rel: str, text: str) -> Path:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# --- nalezení pluginu ------------------------------------------------------------------------------------------------


def test_find_plugin_root_manifest_from_case_file():
    case_file = DEMO / "quality/evals/first-case/prompt.md"
    assert find_plugin_root(case_file) == DEMO.resolve()
    assert find_plugin_root(DEMO / "quality/evals/group/nested-yaml") == DEMO.resolve()


def test_find_plugin_root_skills_directory_plugin():
    assert find_plugin_root(SKILLDIR / "evals/case-a") == SKILLDIR.resolve()
    assert find_plugin_root(SKILLDIR) == SKILLDIR.resolve()


def test_find_plugin_root_variants(tmp_path):
    write(tmp_path, "a/plugin.json", "{}")
    write(tmp_path, "b/skills/x/SKILL.md", "---\nname: x\n---\n")
    (tmp_path / "c/deep").mkdir(parents=True)
    assert find_plugin_root(tmp_path / "a") == (tmp_path / "a").resolve()
    assert find_plugin_root(tmp_path / "b") == (tmp_path / "b").resolve()
    assert find_plugin_root(tmp_path / "c/deep") is None
    assert not is_plugin_dir(tmp_path / "c")


# --- load_plugin -----------------------------------------------------------------------------------------------------


def test_load_plugin_manifest(tmp_path):
    p = load_plugin(DEMO, Settings(), resolve_mcp=False)
    assert p.name == "demo-plugin"
    assert p.root == DEMO.resolve()
    assert p.eval_dir == DEMO.resolve() / "quality/evals"
    assert [s.name for s in p.skills] == ["changelog", "commit-msg"]
    cm = next(s for s in p.skills if s.name == "commit-msg")
    assert cm.description == "Drafts commit messages from a change description."
    assert cm.body.startswith("# Commit messages")
    assert "---" not in cm.body
    assert cm.qualified_name == "demo-plugin:commit-msg"
    assert next(s for s in p.skills if s.name == "changelog").path == (DEMO / "skills/changelog").resolve()


def test_load_plugin_skills_directory():
    p = load_plugin(SKILLDIR, Settings(), resolve_mcp=False)
    assert p.name == "my-skill"
    assert [(s.name, s.plugin) for s in p.skills] == [("my-skill", "my-skill")]
    assert p.skills[0].body.startswith("# My skill")
    assert p.eval_dir == SKILLDIR.resolve() / "evals"
    assert list(p.mcp_servers) == ["shop"]


def test_load_plugin_not_a_plugin(tmp_path):
    with pytest.raises(ConfigError, match="není plugin"):
        load_plugin(tmp_path, Settings())


# --- eval adresář ----------------------------------------------------------------------------------------------------


def test_eval_dir_precedence(tmp_path):
    write(tmp_path, "plugin.json", json.dumps({"name": "x", "experimental": {"evals": "quality/evals"}}))
    assert plugin_eval_dir(tmp_path, None) == (tmp_path / "quality/evals", [])
    assert plugin_eval_dir(tmp_path, "qa") == (tmp_path / "qa", [])


@pytest.mark.parametrize("flag", ["/abs/evals", "../evals", "qa/../../x", ""])
def test_eval_dir_flag_invalid(tmp_path, flag):
    with pytest.raises(ValueError, match="--eval-dir"):
        plugin_eval_dir(tmp_path, flag)


@pytest.mark.parametrize("value", ["/abs", "../x", 5])
def test_eval_dir_manifest_invalid_warns(tmp_path, value):
    write(tmp_path, ".claude-plugin/plugin.json", json.dumps({"name": "x", "experimental": {"evals": value}}))
    path, warnings = plugin_eval_dir(tmp_path, None)
    assert path == tmp_path / "evals"
    assert len(warnings) == 1 and warnings[0].startswith("Warning:")


def test_eval_dir_default(tmp_path):
    write(tmp_path, "SKILL.md", "---\nname: s\n---\n")
    assert plugin_eval_dir(tmp_path, None) == (tmp_path / "evals", [])


# --- MCP -------------------------------------------------------------------------------------------------------------


def test_mcp_unresolved_keeps_vars_but_expands_plugin_root():
    p = load_plugin(DEMO, Settings(), resolve_mcp=False)
    assert set(p.mcp_servers) == {"tracker", "local"}
    tr = p.mcp_servers["tracker"]
    assert tr.type == "http" and tr.plugin == "demo-plugin" and tr.name == "tracker"
    assert tr.headers["Authorization"] == "Bearer ${DEMO_TRACKER_TOKEN}"
    local = p.mcp_servers["local"]
    assert local.type == "stdio"
    assert local.command == f"{DEMO.resolve()}/servers/run.sh"
    assert local.args == ["--root", str(DEMO.resolve())]


def test_mcp_resolve_vars_and_defaults():
    s = Settings(env={"DEMO_TRACKER_TOKEN": "tok-123456789"})
    tr = load_plugin(DEMO, s).mcp_servers["tracker"]
    assert tr.url == "http://127.0.0.1:9999/mcp"  # ${VAR:-default}
    assert tr.headers == {"Authorization": "Bearer tok-123456789"}
    s.env["DEMO_TRACKER_URL"] = "http://example/mcp"
    assert load_plugin(DEMO, s).mcp_servers["tracker"].url == "http://example/mcp"


def test_mcp_missing_var_is_config_error():
    with pytest.raises(ConfigError, match="DEMO_TRACKER_TOKEN"):
        load_plugin(DEMO, Settings(env={}))


def test_mcp_settings_override_and_profile():
    s = Settings(
        env={"T1": "first-token-xx", "T5": "rights5-token-xx"},
        mcp_servers={
            "tracker": {"headers": {"Authorization": "Bearer ${T1}", "X-Extra": "1"}},
            "unknown": {"url": "http://x"},
        },
        mcp_profiles={"limited": {"servers": {"tracker": {"headers": {"Authorization": "Bearer ${T5}"}}}}},
    )
    warnings: list[str] = []
    p = load_plugin(DEMO, s, warnings=warnings)
    assert "unknown" not in p.mcp_servers
    assert any("unknown" in w for w in warnings)
    tr = p.mcp_servers["tracker"]
    assert tr.headers == {"Authorization": "Bearer first-token-xx", "X-Extra": "1"}
    assert tr.url == "http://127.0.0.1:9999/mcp"  # z .mcp.json zůstává

    tr5 = load_plugin(DEMO, s, profile="limited").mcp_servers["tracker"]
    assert tr5.headers == {"Authorization": "Bearer rights5-token-xx", "X-Extra": "1"}

    with pytest.raises(ConfigError, match="profil"):
        load_plugin(DEMO, s, profile="nope")


def test_mcp_override_without_plugin_servers_no_warning(tmp_path):
    write(tmp_path, "SKILL.md", "---\nname: s\ndescription: d\n---\nbody\n")
    warnings: list[str] = []
    p = load_plugin(tmp_path, Settings(mcp_servers={"webstore": {"url": "http://x"}}), warnings=warnings)
    assert p.mcp_servers == {} and warnings == []


def test_mcp_manifest_path_and_flat_format(tmp_path):
    write(tmp_path, "plugin.json", json.dumps({"name": "p", "mcpServers": "./mcp/servers.json"}))
    write(tmp_path, "mcp/servers.json", json.dumps({"db": {"command": "db-server", "env": {"X": "${X:-1}"}}}))
    write(tmp_path, "skills/a/SKILL.md", "---\nname: a\ndescription: A\n---\n")
    p = load_plugin(tmp_path, Settings())
    assert p.mcp_servers["db"].type == "stdio"
    assert p.mcp_servers["db"].env == {"X": "1"}
    assert p.mcp_servers["db"].plugin == "p"


def test_mcp_invalid_type_and_json(tmp_path):
    write(tmp_path, "SKILL.md", "---\nname: s\n---\n")
    write(tmp_path, ".mcp.json", json.dumps({"mcpServers": {"x": {"type": "ws", "url": "ws://x"}}}))
    with pytest.raises(ConfigError, match="type"):
        load_plugin(tmp_path, Settings())
    write(tmp_path, ".mcp.json", "{not json")
    with pytest.raises(ConfigError, match="neplatná MCP"):
        load_plugin(tmp_path, Settings())
