"""skilleval.suite.loader – načtení sady: discovery, sloučení prompt.md + case.yaml, gradery, filtry, targety."""

from __future__ import annotations

from pathlib import Path

import pytest

from skilleval.config import Settings
from skilleval.suite.loader import load_case, load_suite

FIX = Path(__file__).parent / "fixtures" / "suite"
DEMO = (FIX / "demo-plugin").resolve()
EVALS = DEMO / "quality/evals"
SKILLDIR = (FIX / "skills" / "my-skill").resolve()


def suite(target=DEMO, **kw):
    kw.setdefault("resolve_mcp", False)
    return load_suite(target, Settings(), **kw)


def by_name(s):
    return {c.name: c for c in s.cases}


def test_discovery_groups_and_skips():
    s = suite()
    assert s.load_errors == []
    assert s.root == DEMO and s.eval_dir == EVALS
    assert s.plugin is not None and s.plugin.name == "demo-plugin"
    # results/, mocks/ a skryté adresáře nejsou case; vnořená skupina ano
    assert sorted(by_name(s)) == ["changelog-from-diff", "first-case", "mixed-case", "with-extension"]
    assert by_name(s)["changelog-from-diff"].rel_dir == "quality/evals/group/nested-yaml"


def test_prompt_md_case_from_docs():
    c = by_name(suite())["first-case"]
    assert c.source == "prompt"
    assert c.prompt == ("Write me a commit message for this change: I renamed getUser to fetchUser and updated "
                        "the three call sites.")
    assert c.max_turns == 10 and c.timeout_seconds == 300 and c.runs is None
    assert c.allowed_tools == ["Read", "Glob", "Grep", "Skill"]
    assert c.plugins == [DEMO]
    assert c.dir == EVALS / "first-case"
    assert c.rel_dir == "quality/evals/first-case"
    assert c.mock_dirs == [EVALS / "mocks"]
    assert c.extension is None
    g = {g.name: g for g in c.graders}
    assert list(g) == ["criteria", "skill-fired"]  # seřazené podle názvu souboru
    assert g["criteria"].type == "llm"
    assert g["criteria"].config == {"criteria": "PASS if the subject line mentions fetchUser.\n"
                                                "FAIL if there is no commit message."}
    assert g["criteria"].markdown == g["criteria"].config["criteria"]
    assert g["skill-fired"].config == {"tool": "Skill",
                                       "input_match": '"skill"\\s*:\\s*"(?:[\\w-]+:)?commit-msg"'}
    assert g["skill-fired"].weight == 1 and g["skill-fired"].arm is None


def test_yaml_only_case():
    c = by_name(suite())["changelog-from-diff"]
    assert c.source == "yaml"
    assert c.prompt == "Write a changelog entry into CHANGELOG.md."
    assert c.description == "Whole case in YAML" and c.expected_outcome == "A changelog entry is written."
    assert c.tags == ["smoke"] and c.runs == 2
    assert (c.model, c.max_turns, c.timeout_seconds) == ("haiku", 15, 600)
    assert c.allowed_tools == ["Read", "Write"] and c.append_system_prompt == "Be brief."
    assert c.env == {"EVAL_MODE": "strict", "EVAL_LEVEL": "3"}
    assert c.context.scaffold_script == "fixture.sh" and c.context.add_dirs == ["resources"]
    g = {g.name: g for g in c.graders}
    assert [x.name for x in c.graders] == ["file-written", "mentions-version", "judged"]
    assert g["mentions-version"].config == {"pattern": r"\d+\.\d+\.\d+", "flags": "i", "match": "count:1",
                                            "target": {"source": "file", "path": "CHANGELOG.md"}}
    assert g["mentions-version"].weight == 2 and g["mentions-version"].markdown == r"\d+\.\d+\.\d+"
    assert g["judged"].arm == "both" and g["judged"].markdown == "PASS if the entry is concise."
    assert g["file-written"].markdown == ""


def test_mixed_case_merge():
    c = by_name(suite())["mixed-case"]
    assert c.source == "mixed"
    assert c.prompt == "Prompt from prompt.md."  # tělo prompt.md přebíjí execution.prompt
    assert c.tags == ["smoke", "mixed"] and c.runs == 1  # frontmatter přebíjí case.yaml
    assert c.max_turns == 30 and c.expected_outcome == "yaml outcome"  # zbytek z case.yaml
    # gradery z yaml první, pak graders/*.md seřazené
    assert [g.name for g in c.graders] == ["a-from-yaml", "a-file", "b-regex-body"]
    g = {g.name: g for g in c.graders}
    assert g["a-from-yaml"].config == {"before": "Skill", "after": {"tool": "Write", "input_match": "CHANGELOG"}}
    assert g["b-regex-body"].config == {"pattern": "TODO", "target": "last_message", "match": "not_contains"}
    assert g["a-file"].config == {"path": "*.md", "exists": False}
    assert g["a-file"].weight == 0.5 and g["a-file"].arm == "with-only"


def test_extension_and_case_mocks():
    c = by_name(suite())["with-extension"]
    assert c.extension is not None and c.extension.mcp_profile == "limited"
    assert c.extension.faults[0].tool == "create_issue" and c.extension.faults[0].error
    assert c.all_tags == ["ext", "db-write"]
    assert c.mock_dirs == [EVALS / "mocks", EVALS / "with-extension/mocks"]


def test_filters_case_glob_and_tags():
    assert [c.name for c in suite(case_globs=["first-*"]).cases] == ["first-case"]
    assert sorted(c.name for c in suite(case_globs=["first-case", "mixed*"]).cases) == ["first-case", "mixed-case"]
    assert sorted(c.name for c in suite(tags=["smoke"]).cases) == ["changelog-from-diff", "mixed-case"]
    assert [c.name for c in suite(tags=["db-write"]).cases] == ["with-extension"]  # tags_extra
    assert suite(tags=["nope"]).cases == []


def test_eval_dir_flag(tmp_path):
    s = suite(eval_dir="nonexistent")
    assert s.eval_dir == DEMO / "nonexistent" and s.cases == []
    with pytest.raises(ValueError):
        suite(eval_dir="../x")


def test_target_single_case_file():
    s = suite(EVALS / "mixed-case/prompt.md")
    assert [c.name for c in s.cases] == ["mixed-case"]
    assert s.root == DEMO and s.eval_dir == EVALS and s.plugin.name == "demo-plugin"
    s2 = suite(EVALS / "group/nested-yaml/case.yaml")
    assert [c.name for c in s2.cases] == ["changelog-from-diff"]
    assert s2.cases[0].plugins == [DEMO]


def test_target_case_dir_and_group_dir():
    assert [c.name for c in suite(EVALS / "first-case").cases] == ["first-case"]
    s = suite(EVALS / "group")
    assert [c.name for c in s.cases] == ["changelog-from-diff"] and s.root == DEMO and s.eval_dir == EVALS


def test_target_cwd(monkeypatch):
    monkeypatch.chdir(SKILLDIR)
    s = load_suite(None, Settings(), resolve_mcp=False)
    assert s.root == SKILLDIR and [c.name for c in s.cases] == ["case-a"]


@pytest.mark.parametrize("target", ["formatter@my-marketplace", "formatter", "my-plugin@skills-dir"])
def test_target_installed_plugin_rejected(target):
    with pytest.raises(ValueError, match="Instalované pluginy"):
        suite(target)


def test_target_bad_file(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("x")
    with pytest.raises(ValueError):
        suite(f)
    with pytest.raises(ValueError, match="neexistuje"):
        suite(tmp_path / "missing" / "dir")


def test_skills_directory_plugin_suite():
    s = suite(SKILLDIR)
    assert s.load_errors == []
    assert [c.name for c in s.cases] == ["case-a"]  # evals/results/ se ignoruje
    c = s.cases[0]
    assert c.source == "mixed" and c.tags == ["shop", "core"]
    assert (c.max_turns, c.timeout_seconds) == (25, 900)
    assert c.expected_outcome == "Agent finds it.\n"
    assert c.plugins == [SKILLDIR] and c.rel_dir == "evals/case-a"
    assert c.graders[0].config == {"criteria": "Agent zavolá core_search.", "focus": "trace"}


def test_no_plugin_suite(tmp_path):
    (tmp_path / "evals/c1/graders").mkdir(parents=True)
    (tmp_path / "evals/c1/prompt.md").write_text("Hello")
    (tmp_path / "evals/c1/graders/g.md").write_text("---\ntype: llm\n---\nPASS always\n")
    s = load_suite(tmp_path, Settings())
    assert s.plugin is None and s.root == tmp_path.resolve()
    assert s.cases[0].plugins == [] and s.cases[0].prompt == "Hello"


def test_plugins_field_relative_to_case(tmp_path):
    (tmp_path / "plug/skills/a").mkdir(parents=True)
    (tmp_path / "plug/skills/a/SKILL.md").write_text("---\nname: a\n---\n")
    (tmp_path / "plug/evals/c/graders").mkdir(parents=True)
    (tmp_path / "other").mkdir()
    (tmp_path / "other/SKILL.md").write_text("---\nname: o\n---\n")
    (tmp_path / "plug/evals/c/prompt.md").write_text('---\nplugins: ["../..", "../../../other"]\n---\nHi')
    (tmp_path / "plug/evals/c/graders/g.md").write_text("---\ntype: tool_used\ntool: Skill\n---\n")
    c = load_case(tmp_path / "plug/evals/c", plugin_root=tmp_path / "plug", eval_dir=tmp_path / "plug/evals")
    assert c.plugins == [(tmp_path / "plug").resolve(), (tmp_path / "other").resolve()]


def test_crlf_and_bom_frontmatter(tmp_path):
    (tmp_path / "evals/c/graders").mkdir(parents=True)
    (tmp_path / "evals/c/prompt.md").write_bytes("﻿---\r\nmax_turns: 5\r\n---\r\n\r\n  Indented prompt\r\n\r\n"
                                                 .encode())
    (tmp_path / "evals/c/graders/g.md").write_text("---\ntype: llm\n---\nok\n")
    c = load_suite(tmp_path, Settings()).cases[0]
    assert c.max_turns == 5 and c.prompt == "  Indented prompt"


def test_prompt_without_frontmatter(tmp_path):
    (tmp_path / "evals/c/graders").mkdir(parents=True)
    (tmp_path / "evals/c/prompt.md").write_text("\n\nJust the prompt with @file.md\n---\nnot frontmatter\n")
    (tmp_path / "evals/c/graders/g.md").write_text("---\ntype: llm\n---\nok\n")
    c = load_suite(tmp_path, Settings()).cases[0]
    assert c.prompt == "Just the prompt with @file.md\n---\nnot frontmatter"
    assert c.name == "c" and c.source == "prompt"


def test_allowed_tools_string_form(tmp_path):
    (tmp_path / "evals/c/graders").mkdir(parents=True)
    (tmp_path / "evals/c/prompt.md").write_text('---\nallowed_tools: "Read, Glob Bash(npm test *)"\n---\nx')
    (tmp_path / "evals/c/graders/g.md").write_text("---\ntype: llm\n---\nok\n")
    assert load_suite(tmp_path, Settings()).cases[0].allowed_tools == ["Read", "Glob", "Bash(npm test *)"]
