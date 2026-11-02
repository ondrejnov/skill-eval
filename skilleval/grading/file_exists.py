"""Grader `file_exists`: soubor vytvořený během runu odpovídá globu `path` (nebo žádný při `exists: false`)."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from skilleval.models import GraderSpec, RunArtifacts


def _brace_expand(pattern: str) -> list[str]:
    """`a.{md,txt}` → [a.md, a.txt] (vnořené závorky podporované, bez čárky = literál)."""
    depth, start, i = 0, -1, 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                parts, d, cur = [], 0, ""
                for c in pattern[start + 1:i]:
                    if c == "," and d == 0:
                        parts.append(cur)
                        cur = ""
                        continue
                    d += (c == "{") - (c == "}")
                    cur += c
                parts.append(cur)
                if len(parts) >= 2:
                    pre, post = pattern[:start], pattern[i + 1:]
                    return [x for part in parts for x in _brace_expand(pre + part + post)]
        i += 1
    return [pattern]


def _glob_to_regex(glob: str) -> str:
    out: list[str] = []
    i, n = 0, len(glob)
    while i < n:
        c = glob[i]
        if c == "*":
            if glob.startswith("**", i):
                # `**/` = libovolný počet adresářů (i žádný), `**` jinde = cokoli
                at_seg_start = i == 0 or glob[i - 1] == "/"
                if at_seg_start and glob.startswith("**/", i):
                    out.append("(?:.*/)?")
                    i += 3
                    continue
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = glob.find("]", i + 2 if glob[i + 1:i + 2] in ("!", "^") else i + 1)
            if j < 0:
                out.append("\\[")
            else:
                body = glob[i + 1:j]
                if body[:1] in ("!", "^"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j
        elif c == "\\" and i + 1 < n:
            i += 1
            out.append(re.escape(glob[i]))
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


@lru_cache(maxsize=256)
def compile_glob(pattern: str) -> re.Pattern[str]:
    alts = [_glob_to_regex(_normalize(p)) for p in _brace_expand(pattern)]
    return re.compile("(?:" + "|".join(alts) + r")\Z", re.S)


def _normalize(path: str) -> str:
    while path.startswith("./"):
        path = path[2:]
    return path


def glob_match(pattern: str, path: str) -> bool:
    return compile_glob(pattern).match(_normalize(path)) is not None


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
        raise ValueError(f"`exists` must be true or false, got {value!r}")
    return bool(value)


def grade_file_exists(spec: GraderSpec, art: RunArtifacts) -> tuple[bool, str, str | None]:
    cfg = spec.config
    pattern = cfg.get("path")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("file_exists grader requires `path`")
    want = _as_bool(cfg.get("exists"), True)
    # absolutní cesta uvnitř workspace → relativní
    ws = str(art.workspace).rstrip("/") + "/"
    rel_pattern = pattern[len(ws):] if pattern.startswith(ws) else pattern
    found = [f for f in art.files_created if glob_match(rel_pattern, f)]
    evidence = "\n".join(found[:20]) or None
    if want:
        if found:
            return True, f"created file(s) matching {pattern}: {', '.join(found[:5])}", evidence
        return False, f"no file matching {pattern} was created", "\n".join(art.files_created[:20]) or None
    if found:
        return False, f"unexpected file(s) matching {pattern} created: {', '.join(found[:5])}", evidence
    return True, f"no file matching {pattern} was created", None
