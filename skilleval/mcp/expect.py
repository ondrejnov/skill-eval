"""`expect:` blok mock souborů – validace omezeného regex dialektu a kontrola vstupu volání.

Hodnota pro dot-path vstupu může být:
- název typu `string | number | boolean | array | object`,
- `/regex/flags` v dialektu Claude Code (literály, `.`, escapy `\\d`, třídy `[a-z]`, kvantifikátory `* + ? {m,n}` vždy
  na jednom znaku/escapu/třídě, volitelně `^` na začátku a `$` na konci, flagy jen `i`, `s`),
- literál (porovnání rovností), nebo seznam povolených literálů.

Regex se testuje jako JS `RegExp.test` (hledání kdekoli), hodnota delší než limit vzoru je porušení. Limit (CC ho
nedokumentuje číslem) aproximujeme: 1 000 znaků, s `^` 10 000, děleno počtem neomezených kvantifikátorů.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

TYPE_NAMES = ("string", "number", "boolean", "array", "object")
_REGEX_LITERAL_RE = re.compile(r"^/(.+)/([A-Za-z]*)$", re.S)
_MISSING = object()

_BASE_LIMIT = 1_000
_ANCHORED_LIMIT = 10_000
_MIN_LIMIT = 100


class PatternError(ValueError):
    pass


@dataclass
class CompiledPattern:
    source: str  # původní `/…/flags`
    regex: re.Pattern[str]
    max_len: int


# ---------------------------------------------------------------------------------------------------------------------
# Regex dialekt
# ---------------------------------------------------------------------------------------------------------------------

_CLASS_ESC = {"d": "0-9", "w": "A-Za-z0-9_", "s": r"\s", "D": r"\D", "W": r"\W", "S": r"\S"}
_ATOM_ESC = {"d": "[0-9]", "D": "[^0-9]", "w": "[A-Za-z0-9_]", "W": "[^A-Za-z0-9_]", "s": r"\s", "S": r"\S"}
_CTRL_ESC = {"n": r"\n", "t": r"\t", "r": r"\r", "f": r"\f", "v": r"\v", "0": r"\x00"}
_HEX_RE = re.compile(r"x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}")
_QUANT_RE = re.compile(r"\{(\d+)(,(\d*))?\}")


def is_regex_literal(value: Any) -> bool:
    return isinstance(value, str) and bool(_REGEX_LITERAL_RE.match(value))


def _escape(body: str, i: int, in_class: bool) -> tuple[str, int]:
    """Escape na pozici i (za zpětným lomítkem) → (python fragment, nová pozice)."""
    if i >= len(body):
        raise PatternError("trailing backslash")
    c = body[i]
    if c in _ATOM_ESC:
        return (_CLASS_ESC if in_class else _ATOM_ESC)[c], i + 1
    if c in _CTRL_ESC:
        if c == "0" and i + 1 < len(body) and body[i + 1].isdigit():
            raise PatternError("octal escapes are not supported")
        return _CTRL_ESC[c], i + 1
    m = _HEX_RE.match(body, i)
    if m:
        return "\\" + m.group(0), m.end()
    if c.isdigit():
        raise PatternError("backreferences are not supported")
    if c in "bB":
        raise PatternError(f"assertion \\{c} is not supported")
    if c.isalnum():
        raise PatternError(f"unsupported escape \\{c}")
    return re.escape(c), i + 1


def _parse_class(body: str, i: int) -> tuple[str, int]:
    """Znaková třída od `[` na pozici i → (python fragment, pozice za `]`)."""
    out = ["["]
    j = i + 1
    if j < len(body) and body[j] == "^":
        out.append("^")
        j += 1
    start = j
    while j < len(body):
        c = body[j]
        if c == "]":
            if j == start:  # `[]` / `[^]` – v JS prázdná třída, v Pythonu chyba
                raise PatternError("empty character class")
            out.append("]")
            return "".join(out), j + 1
        if c == "\\":
            frag, j = _escape(body, j + 1, in_class=True)
            out.append(frag)
            continue
        out.append("\\" + c if c in "[&~|" else c)
        j += 1
    raise PatternError("unterminated character class")


def parse_pattern(value: str) -> CompiledPattern:
    """`/regex/flags` → zkompilovaný Python regex. Mimo dialekt → PatternError."""
    m = _REGEX_LITERAL_RE.match(value)
    if not m:
        raise PatternError("not a /regex/ literal")
    body, flags = m.group(1), m.group(2)
    py_flags = 0
    for f in flags:
        if f not in "is":
            raise PatternError(f"flag '{f}' is not supported (only i, s)")
        if flags.count(f) > 1:
            raise PatternError(f"duplicate flag '{f}'")
        py_flags |= re.I if f == "i" else re.S

    out: list[str] = []
    i = 0
    anchored = False
    unbounded = 0
    can_quantify = False  # předchozí token je atom (znak/escape/třída), který jde kvantifikovat
    while i < len(body):
        c = body[i]
        if c == "^":
            if i != 0:
                raise PatternError("'^' is allowed only at the start")
            anchored = True
            out.append("^")
            i += 1
            can_quantify = False
            continue
        if c == "$":
            if i != len(body) - 1:
                raise PatternError("'$' is allowed only at the end")
            out.append(r"\Z")  # JS `$` bez flagu m = jen konec řetězce
            i += 1
            can_quantify = False
            continue
        if c in "()":
            raise PatternError("groups are not supported")
        if c == "|":
            raise PatternError("alternation is not supported")
        if c in "*+?" or (c == "{" and _QUANT_RE.match(body, i)):
            if not can_quantify:
                raise PatternError(f"quantifier '{c}' must follow a single character, escape or class")
            if c == "{":
                q = _QUANT_RE.match(body, i)
                assert q is not None
                lo, comma, hi = q.group(1), q.group(2), q.group(3)
                if comma and hi and int(hi) < int(lo):
                    raise PatternError(f"invalid quantifier {q.group(0)}")
                if comma and not hi:
                    unbounded += 1
                out.append(q.group(0))
                i = q.end()
            else:
                if c in "*+":
                    unbounded += 1
                out.append(c)
                i += 1
            if i < len(body) and (body[i] in "*+?" or (body[i] == "{" and _QUANT_RE.match(body, i))):
                raise PatternError("nested or lazy quantifiers are not supported")
            can_quantify = False
            continue
        if c == "[":
            frag, i = _parse_class(body, i)
            out.append(frag)
            can_quantify = True
            continue
        if c == "\\":
            frag, i = _escape(body, i + 1, in_class=False)
            out.append(frag)
            can_quantify = True
            continue
        out.append("." if c == "." else re.escape(c))
        i += 1
        can_quantify = True

    try:
        regex = re.compile("".join(out), py_flags)
    except re.error as ex:  # pragma: no cover - dialekt by neměl propustit nevalidní regex
        raise PatternError(str(ex)) from ex
    limit = (_ANCHORED_LIMIT if anchored else _BASE_LIMIT) // max(1, unbounded)
    return CompiledPattern(source=value, regex=regex, max_len=max(_MIN_LIMIT, limit))


# ---------------------------------------------------------------------------------------------------------------------
# Validace a kontrola
# ---------------------------------------------------------------------------------------------------------------------


def validate_expect(expect: Any) -> list[str]:
    """Chyby `expect:` bloku (prázdný seznam = OK). Chybový text u regexu obsahuje pattern."""
    if expect is None:
        return []
    if not isinstance(expect, dict):
        return ["expect: must be a map of input paths to expectations"]
    errors: list[str] = []
    for path, rule in expect.items():
        if not isinstance(path, str) or not path.strip():
            errors.append(f"expect: invalid input path {path!r}")
            continue
        if is_regex_literal(rule):
            try:
                parse_pattern(rule)
            except PatternError as ex:
                errors.append(f"expect.{path}: invalid pattern {rule}: {ex}")
        elif isinstance(rule, list):
            if not rule:
                errors.append(f"expect.{path}: empty list of allowed values")
            for v in rule:
                if isinstance(v, (dict, list)):
                    errors.append(f"expect.{path}: list values must be literals, got {_short(v)}")
                    break
    return errors


def get_path(data: Any, path: str) -> Any:
    """Hodnota dot-path (`a.b.0.c`) nebo _MISSING."""
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return _MISSING
            cur = cur[part]
        elif isinstance(cur, list) and part.lstrip("-").isdigit():
            idx = int(part)
            if not -len(cur) <= idx < len(cur):
                return _MISSING
            cur = cur[idx]
        else:
            return _MISSING
    return cur


def lookup(data: Any, path: str) -> tuple[bool, Any]:
    """(nalezeno, hodnota) pro dot-path."""
    v = get_path(data, path)
    return (v is not _MISSING), (None if v is _MISSING else v)


def type_of(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def _short(v: Any, n: int = 80) -> str:
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 1] + "…"


def _js_string(v: Any) -> str | None:
    """Hodnota pro regex test (JS String(v)); objekty/pole → None."""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    if v is None:
        return "null"
    return None


def literal_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return expected == actual
    return type(expected) is type(actual) and expected == actual


def check_value(rule: Any, value: Any, path: str) -> str | None:
    if isinstance(rule, str) and rule in TYPE_NAMES:
        if type_of(value) != rule:
            return f"{path}: expected {rule}, got {type_of(value)} {_short(value)}"
        return None
    if is_regex_literal(rule):
        try:
            pat = parse_pattern(rule)
        except PatternError as ex:
            return f"{path}: invalid pattern {rule}: {ex}"
        s = _js_string(value)
        if s is None:
            return f"{path}: expected a value matching {rule}, got {type_of(value)}"
        if len(s) > pat.max_len:
            return f"{path}: value longer than {pat.max_len} chars cannot be checked by {rule}"
        if not pat.regex.search(s):
            return f"{path}: {_short(value)} does not match {rule}"
        return None
    if isinstance(rule, list):
        if not any(literal_equal(r, value) for r in rule):
            return f"{path}: {_short(value)} is not one of {_short(rule)}"
        return None
    if not literal_equal(rule, value):
        return f"{path}: expected {_short(rule)}, got {_short(value)}"
    return None


def check_expect(expect: dict[str, Any] | None, input: dict[str, Any]) -> str | None:
    """None = vstup splňuje `expect`, jinak důvod prvního porušení."""
    if not expect:
        return None
    for path, rule in expect.items():
        found, value = lookup(input, str(path))
        if not found:
            return f"{path}: missing"
        reason = check_value(rule, value, str(path))
        if reason:
            return reason
    return None


def subset_match(when: dict[str, Any] | None, input: dict[str, Any]) -> bool:
    """`when` faultu: dot-path → hodnota (nebo pravidlo jako v expect) – vše musí sedět."""
    if not when:
        return True
    for path, rule in when.items():
        found, value = lookup(input, str(path))
        if not found:
            return False
        if isinstance(rule, dict):
            if not isinstance(value, dict) or not subset_match(rule, value):
                return False
            continue
        if check_value(rule, value, str(path)) is not None:
            return False
    return True
