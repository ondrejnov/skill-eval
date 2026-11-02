import asyncio
import os
import time
from pathlib import Path

from skilleval.harness.sandbox import allowlisted_env, cleanup, make_run_dirs, run_hook, run_scaffold
from skilleval.models import CaseSpec, ContextSpec, GraderSpec


def make_case(tmp_path: Path, script: str | None = None, body: str | None = None) -> CaseSpec:
    d = tmp_path / "case"
    d.mkdir(exist_ok=True)
    if script and body is not None:
        (d / script).write_text(body)
    return CaseSpec(name="c", dir=d, rel_dir="evals/c", source="yaml", prompt="x",
                    context=ContextSpec(scaffold_script=script),
                    graders=[GraderSpec(name="g", type="regex", config={"pattern": "x"})])


def test_make_and_cleanup_dirs():
    dirs = make_run_dirs(keep=False)
    assert dirs.root.name.startswith("skilleval-") and dirs.workspace.is_dir() and dirs.home.is_dir()
    assert dirs.workspace.parent == dirs.root
    cleanup(dirs)
    assert not dirs.root.exists()
    kept = make_run_dirs(keep=True)
    cleanup(kept)
    assert kept.root.exists()
    kept.keep = False
    cleanup(kept)


async def test_scaffold_ok_env_and_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_FROM_SHELL", "leak")
    monkeypatch.setenv("EVAL_X", "case-env")
    case = make_case(tmp_path, "fixture.sh", 'echo "$HOME|$TERM|$TMPDIR|${SECRET_FROM_SHELL:-none}|${EVAL_X:-none}" '
                                             '> env.txt\nmkdir -p src && touch src/a.py\n')
    dirs = make_run_dirs(keep=False)
    try:
        assert await run_scaffold(case, dirs) is None
        home, term, tmpdir, secret, evalx = (dirs.workspace / "env.txt").read_text().strip().split("|")
        assert (home, term, tmpdir, secret, evalx) == (str(dirs.home), "dumb", str(dirs.tmp), "none", "none")
        assert (dirs.workspace / "src" / "a.py").exists()
    finally:
        cleanup(dirs)


async def test_scaffold_none_missing_fail_timeout(tmp_path):
    dirs = make_run_dirs(keep=False)
    try:
        assert await run_scaffold(make_case(tmp_path), dirs) is None
        assert await run_scaffold(make_case(tmp_path, "nope.sh"), dirs) == "scaffold failed: script nope.sh not found"
        err = await run_scaffold(make_case(tmp_path, "bad.sh", "echo oops >&2\nexit 4\n"), dirs)
        assert err == "scaffold failed: exit code 4: oops"
        t0 = time.monotonic()
        err = await run_scaffold(make_case(tmp_path, "slow.sh", "sleep 30 &\nsleep 30\n"), dirs, timeout=0.5)
        assert err is not None and err.startswith("scaffold failed: timed out after")
        assert time.monotonic() - t0 < 10
    finally:
        cleanup(dirs)


async def test_run_hook(tmp_path):
    case = make_case(tmp_path)
    (case.dir / "fixtures").mkdir()
    (case.dir / "fixtures" / "reset.sh").write_text('echo "$SKILLEVAL_CASE" > out.txt\n')
    assert await run_hook(["bash", "fixtures/reset.sh"], case=case, env={"SKILLEVAL_CASE": "c"}) is None
    assert (case.dir / "out.txt").read_text() == "c\n"
    err = await run_hook(["bash", "-c", "echo bad; exit 2"], case=case, env={})
    assert err is not None and "exited with code 2" in err and "bad" in err
    err = await run_hook(["sleep", "10"], case=case, env={}, timeout=1)
    assert err is not None and "timed out after 1 s" in err
    assert "failed" in (await run_hook(["/nonexistent/x"], case=case, env={}))
    assert await run_hook([], case=case, env={}) is not None


async def test_run_hook_cancel_kills_process(tmp_path):
    case = make_case(tmp_path)
    task = asyncio.create_task(run_hook(["bash", "-c", "echo $$ > pid.txt; sleep 30"], case=case, env={}))
    for _ in range(50):
        await asyncio.sleep(0.05)
        if (case.dir / "pid.txt").exists() and (case.dir / "pid.txt").read_text().strip():
            break
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    pid = int((case.dir / "pid.txt").read_text())
    await asyncio.sleep(0.2)
    try:
        os.kill(pid, 0)
        alive = Path(f"/proc/{pid}/stat").exists() and Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except ProcessLookupError:
        alive = False
    assert not alive


def test_allowlisted_env():
    env = allowlisted_env({"PATH": "/bin", "LC_ALL": "C", "HTTPS_PROXY": "p", "SECRET_TOKEN": "x", "HOME": "/root"})
    assert env == {"PATH": "/bin", "LC_ALL": "C", "HTTPS_PROXY": "p"}
