import json
from pathlib import Path

import pytest

from skilleval.mcp.mocks import (
    MockError,
    MockSet,
    canonical_input,
    input_hash,
    render_body,
    split_frontmatter,
    validate_mock_dir,
    write_adopt,
    write_recording,
)


def w(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_split_frontmatter():
    assert split_frontmatter("hello") == ({}, "hello")
    meta, body = split_frontmatter("---\ntype: agent\n---\n\nBody {{input.x}}\n")
    assert meta == {"type": "agent"} and body == "Body {{input.x}}\n"
    with pytest.raises(ValueError):
        split_frontmatter("---\ntype: x\n")


def test_render_body(tmp_path):
    w(tmp_path / "fixtures" / "71781.json", '{"id": 71781}')
    inp = {"title": "Widget", "order": {"id": 71781}, "tags": ["a"], "n": 0}
    out = render_body("T={{input.title}} id={{input.order.id}} tags={{input.tags}} n={{input.n}} "
                      "miss=[{{input.nope}}] f={{file:fixtures/{input.order.id}.json}}", inp, [tmp_path])
    assert out == 'T=Widget id=71781 tags=["a"] n=0 miss=[] f={"id": 71781}'
    assert render_body("{{ input }}", {"b": 1, "a": 2}, []) == '{"a":2,"b":1}'
    with pytest.raises(MockError, match="not found"):
        render_body("{{file:fixtures/{input.id}.json}}", {"id": 1}, [tmp_path])
    with pytest.raises(MockError):
        render_body("{{file:../secret.json}}", {}, [tmp_path])
    with pytest.raises(MockError):
        render_body("{{file:fixtures/{input.id}.json}}", {"id": "../../x"}, [tmp_path])


def test_mockset_layout_and_override(tmp_path):
    suite, case = tmp_path / "suite", tmp_path / "case"
    w(suite / "tracker" / "create_issue.md", "---\nexpect:\n  title: string\n---\nSuite {{input.title}}\n")
    w(suite / "tracker" / "get_issue.md", "Suite get\n")
    w(suite / "tracker" / "_tools.json", json.dumps({"tools": [
        {"name": "create_issue", "description": "Create an issue", "inputSchema": {
            "type": "object", "properties": {"title": {"type": "string"}}}}]}))
    w(suite / "tracker" / "fixtures" / "a.json", "suite-a")
    w(suite / "tracker" / "fixtures" / "b.json", "suite-b")
    w(case / "tracker" / "get_issue.md", "Case get\n")
    w(case / "tracker" / "fixtures" / "a.json", "case-a")
    w(suite / "other" / "_server.md", "---\ntype: agent\ntools: [x, y]\n---\nYou are other.\n")
    w(suite / "other" / "x.md", "fixed x\n")
    w(suite / "nomock" / "_tools.json", '{"tools": []}')

    ms = MockSet([suite, case, tmp_path / "missing"])
    assert ms.errors == []
    assert ms.servers() == {"tracker", "other"}
    assert ms.get("tracker", "get_issue").body == "Case get\n"  # case přebíjí soubor po souboru
    assert ms.get("tracker", "create_issue").body.startswith("Suite")
    assert ms.get("other", "x").type == "fixed"  # <tool>.md má přednost před _server.md
    assert ms.get("other", "y").type == "agent" and ms.get("other", "y").tool is None
    assert not ms.has_tool("other", "z") and not ms.has_tool("nomock", "a")
    tools = {t.name: t for t in ms.tools_for("tracker")}
    assert tools["create_issue"].description == "Create an issue"
    assert tools["create_issue"].input_schema["properties"]["title"]["type"] == "string"
    assert tools["get_issue"].description == "Mocked tool get_issue"
    assert tools["get_issue"].input_schema == {"type": "object", "additionalProperties": True}
    roots = ms.fixture_roots("tracker")
    assert render_body("{{file:fixtures/a.json}}/{{file:fixtures/b.json}}", {}, roots) == "case-a/suite-b"


@pytest.mark.parametrize("name,text,msg", [
    ("t.md", "---\nfoo: 1\n---\nx", "unknown key"),
    ("t.md", "---\ntype: weird\n---\nx", "invalid type"),
    ("t.md", "---\nexpect:\n  a: /(x)/\n---\nx", "invalid pattern /(x)/"),
    ("t.md", "---\ntype: agent\nerror: true\n---\nx", "fixed"),
    ("t.md", "---\nabort_when: x\n---\nx", "abort_when"),
    ("_server.md", "---\ntype: agent\n---\nx", "tools:"),
    ("_server.md", "---\ntype: fixed\ntools: [a]\n---\nx", "type: agent"),
    ("_server.md", "---\ntools: [a, b]\nexpect:\n  x: string\n---\nx", "single tool"),
    ("_tools.json", "{bad", "invalid JSON"),
])
def test_mock_validation_errors(tmp_path, name, text, msg):
    w(tmp_path / "srv" / name, text)
    errs = validate_mock_dir(tmp_path)
    assert len(errs) == 1 and msg in errs[0], errs


def test_server_md_single_tool_expect_ok(tmp_path):
    w(tmp_path / "srv" / "_server.md", "---\ntools: [a]\nexpect:\n  x: string\n---\nx")
    assert validate_mock_dir(tmp_path) == []


def test_replay_and_recordings(tmp_path):
    mocks = tmp_path / "mocks"
    spec_file = w(mocks / "srv" / "ask.md", "---\ntype: agent\n---\nAnswer.\n")
    ms = MockSet([mocks])
    spec = ms.get("srv", "ask")
    results = tmp_path / "results"
    rec = write_recording(results / "mock-recordings", spec=spec, server="srv", tool="ask", input={"b": 1, "a": "x"},
                          output="answer", is_error=False)
    assert rec.name == f"ask-{input_hash({'a': 'x', 'b': 1})}.json"
    data = json.loads(rec.read_text())
    assert data["mock_file"] == str(spec_file) and data["mock_dir"] == str(mocks)

    adopt = write_adopt(results)
    assert adopt == results / "mock-recordings" / "ADOPT.txt"
    text = adopt.read_text()
    assert f"srv/{rec.name}" in text and str(mocks / ".replay" / "srv") in text

    # zkopírovaná nahrávka odpovídá na identické volání
    (mocks / ".replay" / "srv").mkdir(parents=True)
    (mocks / ".replay" / "srv" / rec.name).write_text(rec.read_text())
    ms2 = MockSet([mocks])
    assert ms2.replay("srv", "ask", {"a": "x", "b": 1})["output"] == "answer"
    assert ms2.replay("srv", "ask", {"a": "y", "b": 1}) is None
    assert canonical_input({"b": 1, "a": "č"}) == '{"a":"č","b":1}'


def test_write_adopt_empty(tmp_path):
    assert write_adopt(tmp_path) is None
