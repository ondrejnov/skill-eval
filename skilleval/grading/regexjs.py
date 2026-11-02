"""Převod JavaScript regexu (jak ho používá Claude Code v graderech) na Python `re`.

Emuluje sémantiku JS (bez flagu `u`/`v` se liší jen v okrajových případech):
- `\\d \\w \\b` jsou v JS ASCII (Python je má unicode) → explicitní třídy / lookaroundy,
- `\\s` = JS množina bílých znaků, `.` bez `s` nematchuje `\\n \\r \\u2028 \\u2029`,
- `^`/`$` bez `m` = začátek/konec textu (`$` NEmatchuje před koncovým `\\n` jako v Pythonu), s `m` na všech
  JS koncích řádků,
- `(?<name>…)` → `(?P<name>…)`, `\\k<name>` → `(?P=name)`, `[^]` → libovolný znak, `[]` → nic,
- inline flagy `(?i)` / `(?i:…)` a Python-only syntaxe (`(?P…)`, `(?#…)`, `(?>…)`, possessive `a++`) = ValueError.
"""

from __future__ import annotations

import re

_ALLOWED_FLAGS = set("gimsuyvd")

_WORD = "A-Za-z0-9_"
_DIGIT = "0-9"
# JS \s (WhiteSpace + LineTerminator)
_SPACE = "\\t\\n\\x0b\\x0c\\r \\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff"
_LINE_TERM = "\\n\\r\\u2028\\u2029"

_SHORTHAND = {"d": _DIGIT, "w": _WORD, "s": _SPACE}

_WORD_BOUNDARY = f"(?:(?<=[{_WORD}])(?![{_WORD}])|(?<![{_WORD}])(?=[{_WORD}]))"
_NON_WORD_BOUNDARY = f"(?:(?<=[{_WORD}])(?=[{_WORD}])|(?<![{_WORD}])(?![{_WORD}]))"

_QUANT_BRACE_RE = re.compile(r"\{(\d+)(,(\d*))?\}")
_INLINE_FLAGS_RE = re.compile(r"\(\?[a-zA-Z]*-?[a-zA-Z]*[):]")
_GROUP_NAME_RE = re.compile(r"[A-Za-z_$][\w$]*")


def _err(pattern: str, msg: str) -> ValueError:
    return ValueError(f"invalid regex /{pattern}/: {msg}")


def _escape_char(ch: str) -> str:
    """Literál znaku pro Python regex (mimo i uvnitř třídy)."""
    return re.escape(ch)


class _Converter:
    def __init__(self, pattern: str, flags: str):
        self.p = pattern
        self.i = 0
        self.unicode = "u" in flags or "v" in flags
        self.dotall = "s" in flags
        self.multiline = "m" in flags
        self.out: list[str] = []
        # lze kvantifikovat poslední prvek? (ne za kvantifikátorem `a++`, otevřením skupiny, `|`, kotvou)
        self.can_quantify = False

    # --- pomocné ---------------------------------------------------------------------------------------------------

    def peek(self, n: int = 0) -> str:
        j = self.i + n
        return self.p[j] if j < len(self.p) else ""

    def emit(self, s: str, *, quantifiable: bool = True) -> None:
        self.out.append(s)
        self.can_quantify = quantifiable

    # --- escape sekvence ---------------------------------------------------------------------------------------------

    def read_escape(self, in_class: bool) -> tuple[str, str | None]:
        """Přečte escape za `\\` (self.i ukazuje na znak za lomítkem).

        Vrací (python_fragment, negated_set): negated_set je obsah množiny pro negovanou zkratku (\\D \\W \\S) uvnitř
        třídy – tu nelze v Python třídě vyjádřit přímo, třída se pak přepíše na alternaci."""
        p = self.p
        if self.i >= len(p):
            raise _err(p, "\\ at end of pattern")
        c = p[self.i]
        self.i += 1
        if c in _SHORTHAND:
            return (_SHORTHAND[c] if in_class else f"[{_SHORTHAND[c]}]"), None
        if c in ("D", "W", "S"):
            body = _SHORTHAND[c.lower()]
            return ("" if in_class else f"[^{body}]"), (body if in_class else None)
        if c == "b":
            return ("\\x08" if in_class else _WORD_BOUNDARY), None
        if c == "B":
            return ("B" if in_class else _NON_WORD_BOUNDARY), None
        if c in "fnrtv":
            return {"f": "\\x0c", "n": "\\n", "r": "\\r", "t": "\\t", "v": "\\x0b"}[c], None
        if c == "0" and not self.peek().isdigit():
            return "\\x00", None
        if c.isdigit():
            if in_class:  # JS legacy octal ve třídě – vzácné
                raise _err(p, "numeric escape inside character class is not supported")
            num = c
            while self.peek().isdigit():
                num += self.peek()
                self.i += 1
            return f"(?:\\{num})", None  # backreference (skupina kvůli následujícím číslicím)
        if c == "k":
            if self.peek() == "<":
                end = p.find(">", self.i)
                if end < 0:
                    raise _err(p, "unterminated \\k<name>")
                name = p[self.i + 1:end]
                self.i = end + 1
                if in_class:
                    raise _err(p, "\\k<name> inside character class")
                return f"(?P={name})", None
            return "k", None
        if c == "c":
            letter = self.peek()
            if letter.isascii() and letter.isalpha():
                self.i += 1
                return f"\\x{ord(letter) % 32:02x}", None
            return "\\\\c", None
        if c == "x":
            hh = p[self.i:self.i + 2]
            if len(hh) == 2 and all(ch in "0123456789abcdefABCDEF" for ch in hh):
                self.i += 2
                return f"\\x{hh}", None
            return "x", None
        if c == "u":
            if self.peek() == "{" and self.unicode:
                end = p.find("}", self.i)
                hexs = p[self.i + 1:end] if end > 0 else ""
                if not hexs or any(ch not in "0123456789abcdefABCDEF" for ch in hexs):
                    raise _err(p, "invalid \\u{…} escape")
                self.i = end + 1
                return _escape_char(chr(int(hexs, 16))), None
            hhhh = p[self.i:self.i + 4]
            if len(hhhh) == 4 and all(ch in "0123456789abcdefABCDEF" for ch in hhhh):
                self.i += 4
                return f"\\u{hhhh}", None
            return "u", None
        if c in "pP" and self.unicode:
            raise _err(p, "Unicode property escapes \\p{…} are not supported")
        if c.isascii() and c.isalpha():
            return c, None  # JS identity escape (\e = e)
        return _escape_char(c), None  # \/ → /, \. → \. …

    # --- třída znaků -------------------------------------------------------------------------------------------------

    def read_class(self) -> str:
        p = self.p
        negated = False
        if self.peek() == "^":
            negated = True
            self.i += 1
        if self.peek() == "]":  # JS: [] = nic, [^] = cokoli
            self.i += 1
            return "[\\s\\S]" if negated else "(?!)"
        items: list[str] = []
        neg_sets: list[str] = []
        while True:
            if self.i >= len(p):
                raise _err(p, "unterminated character class")
            c = p[self.i]
            self.i += 1
            if c == "]":
                break
            if c == "\\":
                frag, neg = self.read_escape(in_class=True)
                items.append(frag)
                if neg is not None:
                    neg_sets.append(neg)
            elif c in "[&~|":  # v JS literály, Python by je bral jako (budoucí) množinové operace
                items.append("\\" + c)
            else:
                items.append(c)
        body = "".join(items)
        if not neg_sets:
            return f"[{'^' if negated else ''}{body}]"
        # \D \W \S uvnitř třídy → alternace množin
        alts = ([f"[{body}]"] if body else []) + [f"[^{s}]" for s in neg_sets]
        alt = "|".join(alts)
        return f"(?:(?!{alt})[\\s\\S])" if negated else f"(?:{alt})"

    # --- skupiny -------------------------------------------------------------------------------------------------------

    def read_group_open(self) -> str:
        p = self.p
        rest = p[self.i:]
        if not rest.startswith("?"):
            return "("
        if rest.startswith(("?:", "?=", "?!", "?<=", "?<!")):
            for tok in ("?<=", "?<!", "?:", "?=", "?!"):
                if rest.startswith(tok):
                    self.i += len(tok)
                    return "(" + tok
        if rest.startswith("?<"):
            end = p.find(">", self.i)
            name = p[self.i + 2:end] if end > 0 else ""
            if not _GROUP_NAME_RE.fullmatch(name):
                raise _err(p, f"invalid group name {name!r}")
            if "$" in name:
                raise _err(p, f"group name {name!r} with '$' is not supported")
            self.i = end + 1
            return f"(?P<{name}>"
        if _INLINE_FLAGS_RE.match(p, self.i - 1):
            raise _err(p, "inline flags not supported – use the `flags` option (e.g. flags: i) instead of (?i)")
        raise _err(p, f"unsupported group syntax '({rest[:3]}'")

    # --- hlavní smyčka -----------------------------------------------------------------------------------------------

    def convert(self) -> str:
        p = self.p
        while self.i < len(p):
            c = p[self.i]
            self.i += 1
            if c == "\\":
                frag, _ = self.read_escape(in_class=False)
                self.emit(frag, quantifiable=frag not in (_WORD_BOUNDARY, _NON_WORD_BOUNDARY))
            elif c == "[":
                self.emit(self.read_class())
            elif c == "(":
                self.emit(self.read_group_open(), quantifiable=False)
            elif c == ")":
                self.emit(")")
            elif c == "|":
                self.emit("|", quantifiable=False)
            elif c == ".":
                self.emit("[\\s\\S]" if self.dotall else f"[^{_LINE_TERM}]")
            elif c == "^":
                self.emit(f"(?:(?<![\\s\\S])|(?<=[{_LINE_TERM}]))" if self.multiline else "(?<![\\s\\S])",
                          quantifiable=False)
            elif c == "$":
                self.emit(f"(?=[{_LINE_TERM}]|(?![\\s\\S]))" if self.multiline else "(?![\\s\\S])",
                          quantifiable=False)
            elif c in "*+?":
                self.quantifier(c)
            elif c == "{":
                m = _QUANT_BRACE_RE.match(p, self.i - 1)
                if m:
                    if m.group(3) and int(m.group(3)) < int(m.group(1)):
                        raise _err(p, "numbers out of order in {} quantifier")
                    self.i = m.end()
                    self.quantifier(m.group(0))
                else:
                    self.emit("\\{")  # JS (bez u) literál; Python by `{,5}` bral jako kvantifikátor
            elif c == "}":
                self.emit("\\}")
            else:
                self.emit(c)
        return "".join(self.out)

    def quantifier(self, q: str) -> None:
        if not self.can_quantify:
            raise _err(self.p, "nothing to repeat")
        lazy = ""
        if self.peek() == "?":
            lazy = "?"
            self.i += 1
        self.emit(q + lazy, quantifiable=False)


def convert_js_regex(pattern: str, flags: str = "") -> tuple[str, int]:
    """JS pattern + flagy → (python pattern, python flags)."""
    flags = flags or ""
    seen: set[str] = set()
    for f in flags:
        if f not in _ALLOWED_FLAGS:
            raise _err(pattern, f"unsupported flag {f!r} (allowed JS flags: i, m, s, u, g, y)")
        if f in seen:
            raise _err(pattern, f"duplicate flag {f!r}")
        seen.add(f)
    py = _Converter(pattern, flags).convert()
    return py, (re.IGNORECASE if "i" in flags else 0)


def compile_js_regex(pattern: str, flags: str = "") -> re.Pattern[str]:
    """JS regex → zkompilovaný Python `re.Pattern`. Chyba syntaxe / nepodporovaná konstrukce = ValueError."""
    py, pyflags = convert_js_regex(pattern, flags)
    try:
        return re.compile(py, pyflags)
    except re.error as ex:
        raise _err(pattern, str(ex)) from None


def js_finditer(rx: re.Pattern[str], text: str):
    """Nepřekrývající se shody jako JS `matchAll` (po prázdné shodě posun o 1 znak)."""
    pos = 0
    n = len(text)
    while pos <= n:
        m = rx.search(text, pos)
        if not m:
            return
        yield m
        pos = m.end() + 1 if m.end() == m.start() else m.end()
