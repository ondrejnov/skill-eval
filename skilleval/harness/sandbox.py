"""Izolace runu: dočasný workspace + home, scaffold skript a hooky (setup/teardown/fault run)."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import shutil
import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path

from skilleval.models import CaseSpec

SCAFFOLD_TIMEOUT = 120
HOOK_TIMEOUT = 120

# Proměnné procesu, které se smí předat stdio MCP serverům a hookům (PATH, locale, proxy, certifikáty).
ENV_ALLOWLIST = ("PATH", "LANG", "LANGUAGE", "TZ", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
                 "http_proxy", "https_proxy", "no_proxy", "all_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR",
                 "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS")


@dataclass
class RunDirs:
    root: Path
    workspace: Path
    home: Path
    tmp: Path | None = None
    keep: bool = False


def allowlisted_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Allowlist proměnných procesu (PATH, LC_*, proxy, certifikáty)."""
    environ = dict(os.environ if environ is None else environ)
    return {k: v for k, v in environ.items() if k in ENV_ALLOWLIST or k.startswith("LC_")}


def make_run_dirs(keep: bool) -> RunDirs:
    root = Path(tempfile.mkdtemp(prefix="skilleval-")).resolve()
    dirs = RunDirs(root=root, workspace=root / "workspace", home=root / "home", tmp=root / "tmp", keep=keep)
    for d in (dirs.workspace, dirs.home, dirs.tmp):
        assert d is not None
        d.mkdir()
    return dirs


def cleanup(dirs: RunDirs) -> None:
    """Smaže dočasné adresáře runu (s `keep` je ponechá)."""
    if not dirs.keep:
        shutil.rmtree(dirs.root, ignore_errors=True)


def _tail(data: bytes, n: int = 500) -> str:
    text = data.decode("utf-8", errors="replace").strip()
    return text[-n:]


async def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], timeout: float) -> tuple[int | None, bytes]:
    """Spustí příkaz ve vlastní process group → (exit kód | None při timeoutu, stdout+stderr)."""
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(cwd), env=env, stdin=asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                                start_new_session=True)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, out or b""
    except TimeoutError:
        _kill(proc)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=5)
        return None, b""
    except asyncio.CancelledError:  # zrušený run → nenechat viset procesy
        _kill(proc)
        raise


def _kill(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)


async def run_scaffold(case: CaseSpec, dirs: RunDirs, timeout: float = SCAFFOLD_TIMEOUT) -> str | None:
    """`bash <scaffold_script>` v prázdném workspace. None = OK, jinak `scaffold failed: …`."""
    script = case.context.scaffold_script
    if not script:
        return None
    path = (case.dir / script).resolve()
    if not path.is_file():
        return f"scaffold failed: script {script} not found"
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": str(dirs.home),
           "TMPDIR": str(dirs.tmp or dirs.root), "TERM": "dumb"}
    try:
        code, out = await _run(["bash", str(path)], cwd=dirs.workspace, env=env, timeout=timeout)
    except OSError as ex:
        return f"scaffold failed: {ex}"
    if code is None:
        return f"scaffold failed: timed out after {int(timeout)} s"
    if code != 0:
        tail = _tail(out)
        return f"scaffold failed: exit code {code}" + (f": {tail}" if tail else "")
    return None


async def run_hook(cmd: list[str], *, case: CaseSpec, env: dict[str, str], timeout: int = HOOK_TIMEOUT) -> str | None:
    """Hook (setup/teardown/fault `run`), cwd = adresář case. None = OK, jinak popis chyby."""
    if not cmd:
        return "hook has an empty command"
    shown = shlex.join(cmd)
    full_env = dict(env)
    full_env.setdefault("PATH", os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"))
    try:
        code, out = await _run(list(cmd), cwd=case.dir, env=full_env, timeout=timeout)
    except OSError as ex:
        return f"`{shown}` failed: {ex}"
    if code is None:
        return f"`{shown}` timed out after {timeout} s"
    if code != 0:
        tail = _tail(out)
        return f"`{shown}` exited with code {code}" + (f": {tail}" if tail else "")
    return None
