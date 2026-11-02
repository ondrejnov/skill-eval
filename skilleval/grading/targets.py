"""Co grader vidí: `target` (regex) / `focus` (llm) – last_message | trace | files | {source: file, path} | mock_calls."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from skilleval.models import RunArtifacts

# llm judge vidí z trace prvních a posledních N řádků (jako Claude Code)
TRACE_HEAD = 12
TRACE_TAIL = 12

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
               ".webp": "image/webp"}
_MAGIC = [(b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"), (b"GIF87a", "image/gif"),
          (b"GIF89a", "image/gif")]


class TargetError(ValueError):
    """Target nelze vyhodnotit (chybí soubor, binární obsah …) → grader fail s tímto vysvětlením."""


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def elide_lines(lines: list[str], head: int = TRACE_HEAD, tail: int = TRACE_TAIL) -> str:
    """Prvních `head` + posledních `tail` řádků; mezi nimi značka jako v Claude Code (`[…N messages elided…]`)."""
    if len(lines) <= head + tail:
        return "\n".join(lines)
    omitted = len(lines) - head - tail
    return "\n".join([*lines[:head], f"[…{omitted} messages elided…]", *lines[-tail:]])


def target_label(target: Any) -> str:
    """Lidsky čitelný název targetu (do promptu soudce a explanation)."""
    if target is None:
        return "last_message"
    if isinstance(target, dict):
        return f"file {target.get('path')}"
    return str(target)


def _image_type(path: Path, data: bytes) -> str | None:
    for magic, mt in _MAGIC:
        if data.startswith(magic):
            return mt
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return IMAGE_TYPES.get(path.suffix.lower())


def _resolve_in_workspace(workspace: Path, rel: str) -> Path:
    root = workspace.resolve()
    p = Path(rel)
    full = (p if p.is_absolute() else root / p).resolve()
    if full != root and root not in full.parents:
        raise TargetError(f"path {rel!r} is outside the workspace")
    return full


def _file_target(spec: dict[str, Any], art: RunArtifacts, for_llm: bool) -> tuple[str, list[dict]]:
    if spec.get("source") != "file":
        raise ValueError(f"unsupported target {spec!r} (expected {{source: file, path: …}})")
    rel = spec.get("path")
    if not rel or not isinstance(rel, str):
        raise ValueError("target {source: file} requires `path`")
    full = _resolve_in_workspace(art.workspace, rel)
    if not full.is_file():
        raise TargetError(f"file not found in workspace: {rel}")
    data = full.read_bytes()
    mt = _image_type(full, data)
    if mt:
        if not for_llm:
            raise TargetError(f"{rel} is an image ({mt}); a regex can't match binary content – grade it with an "
                              "llm grader (focus: {source: file, path}) instead")
        b64 = base64.b64encode(data).decode("ascii")
        return f"[image {rel} ({mt}, {len(data)} bytes) attached]", [{"media_type": mt, "data": b64}]
    if b"\x00" in data[:8192]:
        raise TargetError(f"{rel} is a binary file; render it to an image or write it out as text and grade that")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise TargetError(f"{rel} is not UTF-8 text (binary file); render it to an image or write it out as text "
                          "and grade that") from None
    return text, []


def mock_calls_text(art: RunArtifacts) -> str:
    return "\n".join(_dumps({"server": c.server, "tool": c.tool, "input": c.input, "output": c.output,
                             "isError": c.is_error}) for c in art.mock_calls)


def target_text(target: Any, art: RunArtifacts, *, for_llm: bool) -> tuple[str, list[dict]]:
    """(text, images) pro daný target/focus. Neplatná hodnota = ValueError, nevyhodnotitelný soubor = TargetError."""
    if target is None or target == "last_message":
        return art.final_message or "", []
    if isinstance(target, dict):
        return _file_target(target, art, for_llm)
    if target == "trace":
        return (elide_lines(art.trace_lines) if for_llm else "\n".join(art.trace_lines)), []
    if target == "files":
        return "\n".join(art.files_created), []
    if target == "mock_calls":
        return mock_calls_text(art), []
    raise ValueError(f"unsupported target {target!r} (last_message | trace | files | mock_calls | "
                     "{source: file, path})")
