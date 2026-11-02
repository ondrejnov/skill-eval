import base64
import os
import time
from pathlib import Path

import pytest

from skilleval.harness.tools_builtin import ASK_USER_TEXT, BuiltinTools, expand_braces
from skilleval.models import SkillInfo


@pytest.fixture
def env(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    plugin = tmp_path / "plugin"
    (plugin / "skills" / "core" / "references").mkdir(parents=True)
    (plugin / "skills" / "core" / "SKILL.md").write_text("---\nname: core\n---\nCore body\n")
    (plugin / "skills" / "core" / "references" / "ref.md").write_text("reference text\nsecond line\n")
    evals = plugin / "evals"
    (evals / "case1").mkdir(parents=True)
    (evals / "case1" / "prompt.md").write_text("secret prompt reference")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.txt").write_text("outside reference")
    skills = [SkillInfo(name="core", description="d", body="Core body\n", path=plugin / "skills" / "core",
                        plugin="webstore-core"),
              SkillInfo(name="orders", description="d", body="Orders body", path=plugin / "skills" / "core",
                        plugin="webstore-orders")]
    return {"ws": ws, "plugin": plugin, "evals": evals, "outside": outside, "skills": skills, "tmp": tmp_path}


def tools(env, names=("Read", "Glob", "Grep", "Write", "Edit", "Skill", "TodoWrite", "AskUserQuestion"), **kw):
    return BuiltinTools(workspace=env["ws"], readable=[env["plugin"]], forbidden=[env["evals"]], skills=env["skills"],
                        tools=list(names), **kw)


def test_tool_defs_and_unsupported(env):
    t = tools(env, ["Read", "Bash", "Bash(npm test *)", "WebFetch", "Agent", "NotebookRead", "TaskCreate", "Write(*)",
                    "Read"])
    assert [d.name for d in t.tool_defs] == ["Read", "Write"]
    assert t.unsupported == ["Bash", "Bash(npm test *)", "WebFetch", "Agent", "NotebookRead", "TaskCreate"]
    for d in tools(env).tool_defs:
        assert d.input_schema["type"] == "object" and d.description and d.kind == "builtin"


async def test_not_enabled(env):
    o = await tools(env, ["Read"]).call("Write", {"file_path": "a", "content": "x"})
    assert o.is_error and "No such tool" in o.content


async def test_skill(env):
    t = tools(env)
    for nm in ("core", "webstore-core:core", "/core"):
        o = await t.call("Skill", {"skill": nm})
        assert not o.is_error
        assert o.content.startswith(f"Base directory for this skill: {(env['plugin'] / 'skills' / 'core').resolve()}")
        assert "Core body" in o.content
    o = await t.call("Skill", {"skill": "webstore-orders:orders", "args": "71781"})
    assert "Orders body" in o.content and "ARGUMENTS: 71781" in o.content
    o = await t.call("Skill", {"skill": "nope"})
    assert o.is_error and "webstore-core:core" in o.content and "webstore-orders:orders" in o.content
    o = await t.call("Skill", {})
    assert o.is_error and "InputValidationError" in o.content


async def test_read(env):
    t = tools(env)
    f = env["ws"] / "a.txt"
    f.write_text("one\ntwo\nthree\n" + "y" * 2100 + "\n")
    o = await t.call("Read", {"file_path": str(f)})
    lines = o.content.split("\n")
    assert lines[0] == "     1\tone" and lines[2] == "     3\tthree"
    assert lines[3].startswith("     4\t" + "y" * 2000) and "truncated" in lines[3]
    o = await t.call("Read", {"file_path": "a.txt", "offset": 2, "limit": 1})  # relativně k workspace
    assert o.content == "     2\ttwo"
    o = await t.call("Read", {"file_path": "a.txt", "offset": 99})
    assert "shorter than the provided offset" in o.content
    o = await t.call("Read", {"file_path": str(env["plugin"] / "skills" / "core" / "references" / "ref.md")})
    assert "reference text" in o.content
    assert "does not exist" in (await t.call("Read", {"file_path": "missing.txt"})).content
    assert "EISDIR" in (await t.call("Read", {"file_path": str(env["ws"])})).content
    (env["ws"] / "empty.txt").write_text("")
    assert "contents are empty" in (await t.call("Read", {"file_path": "empty.txt"})).content
    (env["ws"] / "bin.dat").write_bytes(b"\x00\x01\x02")
    assert (await t.call("Read", {"file_path": "bin.dat"})).is_error
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
    (env["ws"] / "img.png").write_bytes(png)
    o = await t.call("Read", {"file_path": "img.png"})
    assert o.images and o.images[0]["media_type"] == "image/png" and base64.b64decode(o.images[0]["data"]) == png


async def test_read_forbidden_and_outside(env):
    t = tools(env)
    for p in (env["evals"] / "case1" / "prompt.md", env["outside"] / "x.txt", "../outside/x.txt", "/etc/passwd"):
        o = await t.call("Read", {"file_path": str(p)})
        assert o.is_error and "Permission denied" in o.content, p
    # symlink z workspace do eval dir / ven
    (env["ws"] / "link").symlink_to(env["evals"])
    (env["ws"] / "out.txt").symlink_to(env["outside"] / "x.txt")
    assert (await t.call("Read", {"file_path": "link/case1/prompt.md"})).is_error
    assert (await t.call("Read", {"file_path": "out.txt"})).is_error


async def test_glob(env):
    t = tools(env)
    (env["ws"] / "src" / "sub").mkdir(parents=True)
    for i, rel in enumerate(["a.ts", "src/b.tsx", "src/sub/c.ts", "src/d.py"]):
        p = env["ws"] / rel
        p.write_text("x")
        os.utime(p, (time.time() - 100 + i, time.time() - 100 + i))
    o = await t.call("Glob", {"pattern": "**/*.{ts,tsx}"})
    assert o.content.split("\n") == [str(env["ws"] / r) for r in ("a.ts", "src/b.tsx", "src/sub/c.ts")]
    o = await t.call("Glob", {"pattern": "*.ts"})
    assert o.content == str(env["ws"] / "a.ts")
    o = await t.call("Glob", {"pattern": "*.py", "path": "src"})
    assert o.content == str(env["ws"] / "src" / "d.py")
    assert (await t.call("Glob", {"pattern": "*.none"})).content == "No files found"
    # plugin je čitelný, ale evals v něm ne
    o = await t.call("Glob", {"pattern": "**/*.md", "path": str(env["plugin"])})
    assert "SKILL.md" in o.content and "ref.md" in o.content and "prompt.md" not in o.content
    o = await t.call("Glob", {"pattern": f"{env['plugin']}/**/*.md"})
    assert "SKILL.md" in o.content and "prompt.md" not in o.content
    assert (await t.call("Glob", {"pattern": "*", "path": str(env["evals"])})).is_error
    (env["ws"] / "evlink").symlink_to(env["evals"] / "case1" / "prompt.md")
    assert "evlink" not in (await t.call("Glob", {"pattern": "*"})).content


async def test_grep(env):
    t = tools(env)
    (env["ws"] / "a.py").write_text("import os\nprint('Hello')\nx = 1\nhello again\n")
    (env["ws"] / "b.md").write_text("Hello md\n")
    o = await t.call("Grep", {"pattern": "hello", "-i": True})
    assert o.content.startswith("Found 2 files")
    o = await t.call("Grep", {"pattern": "Hello", "glob": "*.py", "output_mode": "content"})
    assert o.content == f"{env['ws'] / 'a.py'}:2:print('Hello')"
    o = await t.call("Grep", {"pattern": "Hello", "type": "py", "output_mode": "content", "-n": False, "-A": 1})
    assert o.content.split("\n") == [f"{env['ws'] / 'a.py'}:print('Hello')", f"{env['ws'] / 'a.py'}-x = 1"]
    o = await t.call("Grep", {"pattern": "(?i)hello", "output_mode": "count"})
    assert f"{env['ws'] / 'a.py'}:2" in o.content and "Found 3 total occurrences across 2 files." in o.content
    o = await t.call("Grep", {"pattern": "o", "output_mode": "content", "head_limit": 2})
    assert len(o.content.split("\n")) == 2
    assert (await t.call("Grep", {"pattern": "zzz"})).content == "No files found"
    assert (await t.call("Grep", {"pattern": "("})).is_error
    o = await t.call("Grep", {"pattern": "reference", "path": str(env["plugin"])})
    assert "ref.md" in o.content and "prompt.md" not in o.content
    assert (await t.call("Grep", {"pattern": "x", "path": str(env["evals"])})).is_error
    o = await t.call("Grep", {"pattern": "import os.print", "multiline": True, "output_mode": "content"})
    assert ":1:import os" in o.content and ":2:print" in o.content


async def test_write_edit_and_files_created(env):
    (env["ws"] / "scaffolded.txt").write_text("from scaffold\n")
    t = tools(env)
    t.snapshot()
    o = await t.call("Write", {"file_path": "out/new.md", "content": "hello"})
    assert o.content == f"File created successfully at: {env['ws'] / 'out' / 'new.md'}"
    assert (env["ws"] / "out" / "new.md").read_text() == "hello"
    # přepsání nepřečteného souboru
    o = await t.call("Write", {"file_path": "scaffolded.txt", "content": "x"})
    assert o.is_error and "has not been read" in o.content
    await t.call("Read", {"file_path": "scaffolded.txt"})
    assert not (await t.call("Write", {"file_path": "scaffolded.txt", "content": "a a b"})).is_error
    o = await t.call("Edit", {"file_path": "scaffolded.txt", "old_string": "a", "new_string": "c"})
    assert o.is_error and "Found 2 matches" in o.content
    o = await t.call("Edit", {"file_path": "scaffolded.txt", "old_string": "b", "new_string": "d"})
    assert not o.is_error and (env["ws"] / "scaffolded.txt").read_text() == "a a d"
    o = await t.call("Edit", {"file_path": "scaffolded.txt", "old_string": "a", "new_string": "c", "replace_all": True})
    assert not o.is_error and (env["ws"] / "scaffolded.txt").read_text() == "c c d"
    assert "not found" in (await t.call("Edit", {"file_path": "scaffolded.txt", "old_string": "zz",
                                                 "new_string": "y"})).content
    assert (await t.call("Edit", {"file_path": "scaffolded.txt", "old_string": "c", "new_string": "c"})).is_error
    o = await t.call("Edit", {"file_path": "created-by-edit.txt", "old_string": "", "new_string": "new"})
    assert not o.is_error
    # zápis mimo workspace / do pluginu / přes symlink ven
    for p in (str(env["plugin"] / "x.md"), "../escape.txt", str(env["evals"] / "x.md")):
        o = await t.call("Write", {"file_path": p, "content": "x"})
        assert o.is_error and "Permission denied" in o.content, p
    (env["ws"] / "outlink").symlink_to(env["outside"])
    assert (await t.call("Write", {"file_path": "outlink/y.txt", "content": "x"})).is_error
    assert not (env["outside"] / "y.txt").exists()
    assert t.files_created() == ["created-by-edit.txt", "out/new.md"]


async def test_files_created_without_explicit_snapshot(env):
    (env["ws"] / "pre.txt").write_text("x")
    t = tools(env)
    (env["ws"] / "post.txt").write_text("y")  # i soubory vytvořené jinak než Write (např. hook)
    assert t.files_created() == ["post.txt"]


async def test_todo_and_ask(env):
    t = tools(env)
    o = await t.call("TodoWrite", {"todos": [{"content": "a", "status": "pending", "activeForm": "A"}]})
    assert o.content.startswith("Todos have been modified successfully.")
    o = await t.call("AskUserQuestion", {"questions": [{"question": "Dobropis?", "options": [{"label": "ano"}]}]})
    assert o.content == ASK_USER_TEXT

    async def ask(args):
        return "Chci dobropis. (" + args["questions"][0]["question"] + ")"

    t2 = tools(env, ask_user=ask)
    o = await t2.call("AskUserQuestion", {"questions": [{"question": "Q?"}]})
    assert o.content == "Chci dobropis. (Q?)"


async def test_result_truncation(env):
    (env["ws"] / "big.txt").write_text("z" * 500)
    t = tools(env, max_result_chars=100)
    o = await t.call("Read", {"file_path": "big.txt"})
    assert len(o.content) < 200 and "truncated" in o.content


def test_expand_braces():
    assert expand_braces("*.{ts,tsx}") == ["*.ts", "*.tsx"]
    assert expand_braces("{a,b}/{c,d}") == ["a/c", "a/d", "b/c", "b/d"]
    assert expand_braces("{x}") == ["{x}"]


def test_paths_are_absolute(env):
    t = tools(env)
    assert t.workspace == Path(env["ws"]).resolve()
