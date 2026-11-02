"""skilleval.suite.validate – hlubší kontroly sady (context soubory, regexy, mocky, skilleval.yaml)."""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

from skilleval.config import Settings
from skilleval.suite.loader import load_suite
from skilleval.suite.validate import validate_suite

FIX = Path(__file__).parent / "fixtures" / "suite"
DEMO = FIX / "demo-plugin"
LLM = "---\ntype: llm\n---\nPASS if ok.\n"


def make_case(base: Path, files: dict[str, str], name: str = "c") -> Path:
    d = base / "evals" / name
    for rel, text in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return d


def errors(base: Path, settings: Settings | None = None) -> list[str]:
    s = load_suite(base, settings or Settings(), resolve_mcp=False)
    return [e.message for e in validate_suite(s, settings)]


@pytest.fixture
def fake_regexjs(monkeypatch):
    """Podvrh skilleval.grading.regexjs (píše jiný agent): inline (?i) = chyba, jinak Python re."""

    def compile_js_regex(pattern: str, flags: str = "") -> re.Pattern:
        if "(?i" in pattern:
            raise ValueError("inline flags are not supported")
        return re.compile(pattern.replace("(?<", "(?P<"))

    mod = types.ModuleType("skilleval.grading.regexjs")
    mod.compile_js_regex = compile_js_regex
    monkeypatch.setitem(sys.modules, "skilleval.grading.regexjs", mod)


@pytest.fixture
def fake_expect(monkeypatch):
    def validate_expect(expect: dict) -> list[str]:
        return [f"expect.{k}: unsupported pattern {v}" for k, v in expect.items() if isinstance(v, str) and "|" in v]

    mod = types.ModuleType("skilleval.mcp.expect")
    mod.validate_expect = validate_expect
    monkeypatch.setitem(sys.modules, "skilleval.mcp.expect", mod)


@pytest.fixture
def no_helpers(monkeypatch):
    """Moduly ostatních agentů nedostupné → kontroly se přeskočí."""
    monkeypatch.setitem(sys.modules, "skilleval.grading.regexjs", None)
    monkeypatch.setitem(sys.modules, "skilleval.mcp.expect", None)


def test_demo_plugin_valid(fake_regexjs, fake_expect):
    s = load_suite(DEMO, Settings(), resolve_mcp=False)
    assert validate_suite(s, Settings(mcp_profiles={"limited": {}})) == []


def test_demo_unknown_profile():
    s = load_suite(DEMO, Settings(), resolve_mcp=False)
    msgs = [e.message for e in validate_suite(s, Settings())]
    assert any("mcp_profile 'limited'" in m for m in msgs)
    assert validate_suite(s) == []  # bez settings se profil nekontroluje


def test_load_errors_included(tmp_path):
    make_case(tmp_path, {"prompt.md": "Hi"})
    assert any("graders" in m for m in errors(tmp_path))


def test_context_files(tmp_path):
    (tmp_path / "secret.sh").write_text("x")
    make_case(tmp_path, {
        "case.yaml": 'schema_version: "1.1"\nname: c\ncontext:\n  scaffold_script: missing.sh\n'
                     "  history_file: ../../secret.sh\n  add_dirs: [res, ok]\n",
        "prompt.md": "Hi", "graders/g.md": LLM, "ok/x.txt": "x"})
    msgs = errors(tmp_path)
    assert any("scaffold_script" in m and "neexistuje" in m for m in msgs)
    assert any("history_file" in m and "mimo adresář case" in m for m in msgs)
    assert any("add_dirs" in m and "'res'" in m for m in msgs)
    assert not any("'ok'" in m for m in msgs)
    assert len(msgs) == 3


def test_baseline_file(tmp_path):
    make_case(tmp_path, {"prompt.md": "Hi",
                         "graders/b.md": "---\ntype: baseline\nbaseline_file: ref.jsonl\n---\nPASS if good.\n"})
    assert any("baseline_file" in m for m in errors(tmp_path))
    (tmp_path / "evals/c/ref.jsonl").write_text("{}\n")
    assert errors(tmp_path) == []


def test_regex_dialect(tmp_path, fake_regexjs):
    make_case(tmp_path, {
        "prompt.md": "Hi",
        "graders/a.md": "---\ntype: regex\npattern: '(?i)done'\n---\n",
        "graders/b.md": "---\ntype: tool_used\ntool: Read\ninput_match: '(unclosed'\n---\n",
        "graders/c.md": "---\ntype: tool_order\nbefore: {tool: Read, input_match: '[bad'}\nafter: Write\n---\n",
        "graders/d.md": "---\ntype: regex\npattern: '(?<name>ok)\\d+'\nflags: i\n---\n",
    })
    msgs = errors(tmp_path)
    assert len(msgs) == 3
    assert any("grader a: pattern" in m for m in msgs)
    assert any("grader b: input_match" in m for m in msgs)
    assert any("grader c: before.input_match" in m for m in msgs)


def test_regex_check_skipped_without_module(tmp_path, no_helpers):
    make_case(tmp_path, {"prompt.md": "Hi", "graders/a.md": "---\ntype: regex\npattern: '(?i)done'\n---\n"})
    assert errors(tmp_path) == []


def test_plugins_field_not_plugin(tmp_path):
    (tmp_path / "notplugin").mkdir()
    make_case(tmp_path, {"prompt.md": "---\nplugins: [../../notplugin]\n---\nHi", "graders/g.md": LLM})
    assert any("není plugin" in m for m in errors(tmp_path))


# --- mocky -----------------------------------------------------------------------------------------------------------


def write(base: Path, rel: str, text: str) -> None:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_mock_files(tmp_path, fake_expect):
    make_case(tmp_path, {"prompt.md": "Hi", "graders/g.md": LLM,
                         "mocks/srv/case_tool.md": "---\nresponse: x\n---\nbody"})
    write(tmp_path, "evals/mocks/srv/ok.md", "---\nexpect:\n  title: string\n---\nCreated {{input.title}}")
    write(tmp_path, "evals/mocks/srv/plain.md", "no frontmatter body")
    write(tmp_path, "evals/mocks/srv/badtype.md", "---\ntype: llm\n---\nx")
    write(tmp_path, "evals/mocks/srv/baderror.md", "---\nerror: 'yes'\n---\nx")
    write(tmp_path, "evals/mocks/srv/badexpect.md", "---\nexpect: [a]\n---\nx")
    write(tmp_path, "evals/mocks/srv/badregex.md", "---\nexpect:\n  a: '/x|y/'\n---\nx")
    write(tmp_path, "evals/mocks/srv/_tools.json", "{bad json")
    write(tmp_path, "evals/mocks/srv/fixtures/x.json", "{bad json but fixture}")
    write(tmp_path, "evals/mocks/.replay/srv/abc.json", "whatever")
    write(tmp_path, "evals/mocks/srv/broken.md", "---\na: [\n---\nx")
    msgs = errors(tmp_path)
    joined = "\n".join(msgs)
    assert "neznámý klíč 'response'" in joined  # case-level mocks
    assert "type musí být fixed nebo agent" in joined
    assert "error musí být true/false" in joined
    assert "expect musí být mapa" in joined
    assert "expect.a: unsupported pattern" in joined
    assert "neplatný JSON" in joined
    assert "neplatný YAML" in joined
    assert len(msgs) == 7


def test_server_md(tmp_path):
    make_case(tmp_path, {"prompt.md": "Hi", "graders/g.md": LLM})
    write(tmp_path, "evals/mocks/a/_server.md", "---\ntype: agent\ntools: [x, y]\nexpect: {q: string}\n---\nAct.")
    write(tmp_path, "evals/mocks/b/_server.md", "---\ntype: agent\ntools: [x]\nexpect: {q: string}\n---\nAct.")
    write(tmp_path, "evals/mocks/c/_server.md", "---\ntype: fixed\ntools: [x]\n---\nAct.")
    write(tmp_path, "evals/mocks/d/_server.md", "---\ntype: agent\n---\nAct.")
    write(tmp_path, "evals/mocks/e/x.md", "---\ntools: [x]\n---\nAct.")  # tools jen v _server.md
    msgs = errors(tmp_path)
    assert sum("expect je povolené jen" in m for m in msgs) == 1
    assert any("_server.md musí být type: agent" in m for m in msgs)
    assert any("tools musí být neprázdný seznam" in m for m in msgs)
    assert any("neznámý klíč 'tools'" in m for m in msgs)
    assert len(msgs) == 4


def test_suite_mocks_validated_once(tmp_path):
    for n in ("a", "b"):
        make_case(tmp_path, {"prompt.md": "Hi", "graders/g.md": LLM}, name=n)
    write(tmp_path, "evals/mocks/srv/t.md", "---\nbogus: 1\n---\nx")
    assert len(errors(tmp_path)) == 1


# --- skilleval.yaml --------------------------------------------------------------------------------------------------


def test_extension_semantics(tmp_path):
    make_case(tmp_path, {"prompt.md": "Hi", "graders/g.md": LLM, "skilleval.yaml": """
mcp_profile: limited
faults:
  - {tool: a, error: true}
  - {tool: b, call: 1, every: true, body: x}
  - {tool: c, before_call: 1}
  - {tool: d, call: 1, body: x, run: [bash, x.sh]}
  - {tool: e, call: 0, body: x}
  - {tool: f, call: 2}
  - {tool: ok, call: 1, error: true, body: '{"error": "conflict"}', passthrough_after: true}
  - {tool: ok2, before_call: 1, run: [bash, fixtures/touch.sh]}
simulated_user: {max_replies: 0}
"""})
    msgs = errors(tmp_path, Settings(mcp_profiles={"limited": {"servers": {}}}))
    joined = "\n".join(msgs)
    assert "faults[0] (a): chybí call" in joined
    assert "faults[1] (b): call / calls / every se vylučují" in joined
    assert "faults[2] (c): before_call vyžaduje run" in joined
    assert "faults[3] (d): run je povolené jen s before_call" in joined
    assert "faults[4] (e): pořadí volání je 1-based" in joined
    assert "faults[5] (f): fault bez body i error" in joined
    assert "max_replies" in joined and "zadej replies nebo agent" in joined
    assert "mcp_profile" not in joined
    assert not any("(ok" in m for m in msgs)
    assert len(msgs) == 8
