"""Chybová pravidla formátu – každé chybné case se nenačte a skončí v suite.load_errors (ne výjimka)."""

from __future__ import annotations

from pathlib import Path

import pytest

from skilleval.config import Settings
from skilleval.suite.loader import CaseLoadError, load_case, load_suite

LLM = "---\ntype: llm\n---\nPASS if ok.\n"


def make_case(base: Path, files: dict[str, str], name: str = "c") -> Path:
    d = base / "evals" / name
    for rel, text in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return d


def load_error(tmp_path: Path, files: dict[str, str]) -> str:
    make_case(tmp_path, files)
    s = load_suite(tmp_path, Settings())
    assert s.cases == [], "case se neměl načíst"
    assert len(s.load_errors) == 1
    return s.load_errors[0].message


def prompt(fm: str, body: str = "Do it.") -> str:
    return f"---\n{fm}\n---\n{body}\n"


# --- prompt.md -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fm, needle", [
    ("source: x", "neznámý klíč 'source'"),
    ("allowed-tools: [Read]", "neznámý klíč 'allowed-tools'"),
    ("runs: 0", "runs"),
    ("runs: 51", "runs"),
    ("runs: three", "runs"),
    ("max_turns: 201", "max_turns"),
    ("max_turns: 0", "max_turns"),
    ("timeout_seconds: 3601", "timeout_seconds"),
    ("timeout_seconds: true", "timeout_seconds"),
    ("env: {PATH: /bin}", "EVAL_"),
    ("env: {eval_x: 1}", "EVAL_"),
    ("env: [EVAL_X]", "env musí být mapa"),
    ("schema_version: '2.0'", "schema_version"),
    ("tags: {a: 1}", "tags musí být seznam"),
    ("name: ''", "name"),
    ("model: [a]", "model musí být text"),
])
def test_prompt_frontmatter_errors(tmp_path, fm, needle):
    msg = load_error(tmp_path, {"prompt.md": prompt(fm), "graders/g.md": LLM})
    assert needle in msg


def test_prompt_frontmatter_limits_ok(tmp_path):
    make_case(tmp_path, {"prompt.md": prompt("runs: 50\nmax_turns: 200\ntimeout_seconds: 3600\nenv: {EVAL_A_1: x}"),
                         "graders/g.md": LLM})
    c = load_suite(tmp_path, Settings()).cases[0]
    assert (c.runs, c.max_turns, c.timeout_seconds, c.env) == (50, 200, 3600, {"EVAL_A_1": "x"})


def test_prompt_frontmatter_bad_yaml(tmp_path):
    assert "neplatný YAML" in load_error(tmp_path, {"prompt.md": "---\ntags: [a\n---\nx", "graders/g.md": LLM})


def test_prompt_frontmatter_unclosed(tmp_path):
    assert "neukončený" in load_error(tmp_path, {"prompt.md": "---\ntags: [a]\nx", "graders/g.md": LLM})


def test_prompt_frontmatter_not_mapping(tmp_path):
    assert "mapa" in load_error(tmp_path, {"prompt.md": "---\n- a\n---\nx", "graders/g.md": LLM})


def test_empty_prompt(tmp_path):
    assert "chybí prompt" in load_error(tmp_path, {"prompt.md": "---\nruns: 1\n---\n\n  \n", "graders/g.md": LLM})


# --- gradery ---------------------------------------------------------------------------------------------------------


def test_case_without_graders(tmp_path):
    msg = load_error(tmp_path, {"prompt.md": "Hi"})
    assert msg.startswith("invalid case.yaml") and "graders" in msg


def test_case_with_empty_graders_dir_and_list(tmp_path):
    make_case(tmp_path, {"case.yaml": 'schema_version: "1.1"\nname: c\ngraders: []\nexecution: {prompt: hi}\n',
                         "graders/readme.txt": "not a grader"})
    s = load_suite(tmp_path, Settings())
    assert "invalid case.yaml" in s.load_errors[0].message and "graders" in s.load_errors[0].message


@pytest.mark.parametrize("grader, needle", [
    ("---\nweight: 1\n---\nx", "chybí povinný klíč type"),
    ("---\ntype: custom\n---\nx", "neznámý type"),
    ("---\ntype: llm\nweight: 0\n---\nx", "weight"),
    ("---\ntype: llm\nweight: -1\n---\nx", "weight"),
    ("---\ntype: llm\nweight: heavy\n---\nx", "weight"),
    ("---\ntype: llm\narm: without\n---\nx", "arm"),
    ("---\ntype: llm\npattern: x\n---\nx", "neznámá volba pro type llm: 'pattern'"),
    ("---\ntype: regex\ncriteria: x\npattern: y\n---\n", "neznámá volba pro type regex: 'criteria'"),
    ("---\ntype: tool_used\nmin: 1\n---\n", "chybí povinná volba tool"),
    ("---\ntype: tool_used\ntool: Skill\nmin: 2\nmax: 1\n---\n", "min"),
    ("---\ntype: tool_used\ntool: Skill\nmax: -1\n---\n", "max"),
    ("---\ntype: tool_order\nbefore: Read\n---\n", "chybí povinná volba after"),
    ("---\ntype: tool_order\nbefore: Read\nafter: {name: Write}\n---\n", "neznámý klíč 'name'"),
    ("---\ntype: file_exists\nexists: true\n---\n", "chybí povinná volba path"),
    ("---\ntype: file_exists\npath: a\nexists: maybe\n---\n", "exists"),
    ("---\ntype: llm\n---\n\n", "chybí povinná volba criteria"),
    ("---\ntype: llm\nfocus: everything\n---\nx", "focus"),
    ("---\ntype: llm\nfocus: {source: file}\n---\nx", "focus"),
    ("---\ntype: regex\n---\n", "chybí povinná volba pattern"),
    ("---\ntype: regex\npattern: a\nmatch: some\n---\n", "match"),
    ("---\ntype: regex\npattern: a\nflags: ix\n---\n", "flags"),
    ("---\ntype: regex\npattern: a\ntarget: stdout\n---\n", "target"),
    ("---\ntype: baseline\n---\nPASS if as good.", "chybí povinná volba baseline_file"),
    ("---\ntype: baseline\nbaseline_file: ref.jsonl\n---\n", "chybí povinná volba criteria"),
])
def test_grader_errors(tmp_path, grader, needle):
    msg = load_error(tmp_path, {"prompt.md": "Hi", "graders/g.md": grader})
    assert msg.startswith("graders/g.md:")
    assert needle in msg


def test_grader_error_path_is_file(tmp_path):
    make_case(tmp_path, {"prompt.md": "Hi", "graders/bad.md": "---\ntype: nope\n---\n"})
    err = load_suite(tmp_path, Settings()).load_errors[0]
    assert err.path.endswith("graders/bad.md")


def test_grader_valid_options(tmp_path):
    make_case(tmp_path, {
        "prompt.md": "Hi",
        "graders/a.md": "---\ntype: tool_used\ntool: Skill\nmin: 0\nmax: 0\narm: both\n---\n",
        "graders/b.md": "---\ntype: regex\npattern: done\nmatch: 'count:2'\nflags: im\ntarget: trace\n---\n",
        "graders/c.md": "---\ntype: baseline\nbaseline_file: ref.jsonl\n---\nPASS if at least as good.\n",
        "graders/d.md": "---\ntype: regex\ntarget: files\n---\n\\.md$\n",
    })
    c = load_suite(tmp_path, Settings()).cases[0]
    g = {x.name: x for x in c.graders}
    assert g["a"].config == {"tool": "Skill", "min": 0, "max": 0}
    assert g["c"].config == {"criteria": "PASS if at least as good.", "baseline_file": "ref.jsonl"}
    assert g["d"].config == {"pattern": "\\.md$", "target": "files"}


def test_duplicate_grader_names(tmp_path):
    msg = load_error(tmp_path, {
        "case.yaml": 'schema_version: "1.1"\nname: c\ngraders:\n  - {name: g, type: llm, criteria: x}\n',
        "prompt.md": "Hi", "graders/g.md": LLM})
    assert "duplicitní název graderu" in msg


# --- case.yaml -------------------------------------------------------------------------------------------------------

BASE_YAML = 'schema_version: "1.1"\nname: c\n'


@pytest.mark.parametrize("yaml_text, needle", [
    ("name: c\n", "schema_version"),
    ('schema_version: "1.0"\nname: c\n', "schema_version"),
    ('schema_version: "1.1"\n', "chybí povinné name"),
    (BASE_YAML + "source: x\n", "neznámý klíč 'source'"),
    (BASE_YAML + "max_turns: 5\n", "neznámý klíč 'max_turns'"),  # patří pod execution
    (BASE_YAML + "execution: {turns: 5}\n", "execution: neznámý klíč 'turns'"),
    (BASE_YAML + "execution: {max_turns: 500}\n", "max_turns"),
    (BASE_YAML + "execution: {env: {HOME: x}}\n", "EVAL_"),
    (BASE_YAML + "context: {fixtures: x}\n", "context: neznámý klíč 'fixtures'"),
    (BASE_YAML + "context: [a]\n", "context musí být mapa"),
    (BASE_YAML + "runs: 100\n", "runs"),
    (BASE_YAML + "graders: {a: 1}\n", "graders musí být seznam"),
    (BASE_YAML + "graders:\n  - {type: llm, criteria: x}\n", "chybí povinné name"),
    (BASE_YAML + "graders:\n  - {name: g, type: llm}\n", "chybí povinná volba criteria"),
    (BASE_YAML + "graders:\n  - {name: g, type: llm, criteria: x, bogus: 1}\n", "bogus"),
    ("- a\n- b\n", "YAML mapa"),
    ("a: [\n", "neplatný YAML"),
])
def test_case_yaml_errors(tmp_path, yaml_text, needle):
    msg = load_error(tmp_path, {"case.yaml": yaml_text, "prompt.md": "Hi", "graders/g.md": LLM})
    assert msg.startswith("invalid case.yaml")
    assert needle in msg


def test_case_yaml_schema_version_float_ok(tmp_path):
    make_case(tmp_path, {"case.yaml": "schema_version: 1.1\nname: c\nexecution: {prompt: hi}\n", "graders/g.md": LLM})
    assert load_suite(tmp_path, Settings()).cases[0].source == "yaml"


def test_yaml_only_without_prompt(tmp_path):
    msg = load_error(tmp_path, {"case.yaml": BASE_YAML, "graders/g.md": LLM})
    assert "chybí prompt" in msg


def test_plugins_nonexistent(tmp_path):
    msg = load_error(tmp_path, {"prompt.md": prompt("plugins: [../../nope]"), "graders/g.md": LLM})
    assert "plugins" in msg and "neexistuje" in msg


# --- skilleval.yaml --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ext, needle", [
    ("mcp_profil: x\n", "mcp_profil"),
    ("faults:\n  - call: 1\n", "faults.0.tool"),
    ("faults:\n  - {tool: t, call: 1, colour: red}\n", "colour"),
    ("simulated_user: {replies: [a], extra: 1}\n", "extra"),
    ("- a\n", "YAML mapa"),
])
def test_extension_errors(tmp_path, ext, needle):
    msg = load_error(tmp_path, {"prompt.md": "Hi", "graders/g.md": LLM, "skilleval.yaml": ext})
    assert msg.startswith("skilleval.yaml") and needle in msg


def test_extension_empty_is_none(tmp_path):
    make_case(tmp_path, {"prompt.md": "Hi", "graders/g.md": LLM, "skilleval.yaml": "\n"})
    assert load_suite(tmp_path, Settings()).cases[0].extension is None


# --- celá sada -------------------------------------------------------------------------------------------------------


def test_errors_do_not_stop_other_cases(tmp_path):
    make_case(tmp_path, {"prompt.md": "Hi", "graders/g.md": LLM}, name="good")
    make_case(tmp_path, {"prompt.md": "Hi"}, name="bad")
    s = load_suite(tmp_path, Settings())
    assert [c.name for c in s.cases] == ["good"]
    assert len(s.load_errors) == 1 and s.load_errors[0].path.endswith("evals/bad")


def test_errors_filtered_by_case_glob(tmp_path):
    make_case(tmp_path, {"prompt.md": "Hi", "graders/g.md": LLM}, name="good")
    make_case(tmp_path, {"prompt.md": "Hi"}, name="bad")
    s = load_suite(tmp_path, Settings(), case_globs=["good"])
    assert [c.name for c in s.cases] == ["good"] and s.load_errors == []


def test_duplicate_case_names(tmp_path):
    make_case(tmp_path, {"prompt.md": prompt("name: same"), "graders/g.md": LLM}, name="a")
    make_case(tmp_path, {"prompt.md": prompt("name: same"), "graders/g.md": LLM}, name="b")
    s = load_suite(tmp_path, Settings())
    assert len(s.cases) == 1 and "duplicitní název case" in s.load_errors[0].message


def test_load_case_raises(tmp_path):
    d = make_case(tmp_path, {"prompt.md": "Hi"})
    with pytest.raises(CaseLoadError, match="graders"):
        load_case(d, plugin_root=None, eval_dir=tmp_path / "evals")
    with pytest.raises(CaseLoadError, match="neobsahuje"):
        load_case(tmp_path, plugin_root=None, eval_dir=tmp_path / "evals")
