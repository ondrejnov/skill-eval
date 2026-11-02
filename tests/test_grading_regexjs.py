"""Převod JS regexu na Python re."""

from __future__ import annotations

import pytest

from skilleval.grading.regexjs import compile_js_regex, convert_js_regex, js_finditer


def found(pattern: str, text: str, flags: str = "") -> bool:
    return compile_js_regex(pattern, flags).search(text) is not None


def test_named_group_and_backreference():
    rx = compile_js_regex(r"(?<y>\d{4})-\k<y>")
    m = rx.search("rok 2026-2026")
    assert m and m.group("y") == "2026"
    assert not found(r"(?<y>\d{4})-\k<y>", "2026-2025")


def test_lookarounds_kept():
    assert found(r"(?<=id:)\d+", "id:71781")
    assert found(r"foo(?!bar)", "foobaz") and not found(r"foo(?!bar)", "foobar")
    assert found(r"(?<!x)y", "ay") and not found(r"(?<!x)y", "xy")


def test_escaped_slash_and_quotes():
    assert found(r"a\/b", "a/b")
    # JSON-escapované uvozovky v trace: pattern \\"status\\":2 = literál backslash + uvozovka
    assert found(r'\\"status\\":2', '{"input":"{\\"status\\":2}"}')
    assert found(r'\"x\"', '"x"')  # \" v JS = "


def test_any_char_class_and_empty_class():
    assert found(r"a[^]b", "a\nb")
    assert not found(r"a[]b", "ab")
    assert found(r"[\s\S]+", "x")


def test_dollar_without_m_is_end_of_text():
    assert not found(r"foo$", "foo\n")  # Python by matchnul (před koncovým \n)
    assert found(r"foo$", "foo")
    assert found(r"foo$", "foo\nbar", "m")
    assert found(r"^bar", "foo\nbar", "m")
    assert not found(r"^bar", "foo\nbar")
    assert found(r"foo$", "foo\r\nbar", "m")  # JS: $ před \r


def test_dot_and_s_flag():
    assert not found(r"a.b", "a\nb")
    assert not found(r"a.b", "a\rb")
    assert found(r"a.b", "a\nb", "s")
    assert found(r"a.b", "aXb")


def test_ascii_shorthands_like_js():
    assert not found(r"^\w+$", "číslo")  # JS \w je ASCII
    assert found(r"^\w+$", "cislo_1")
    assert not found(r"\d", "٣")  # arabská číslice
    assert found(r"\s", "a b") and found(r"\s", "﻿")
    assert not found(r"\s", "\x1c")  # Python \s by matchnul
    assert found(r"[\w\W]", "č")  # „cokoli“ přes negovanou zkratku ve třídě
    assert found(r"[^\D]", "a5") and not found(r"[^\D]", "abc")
    assert found(r"[\d.]+", "3.14")


def test_word_boundary_ascii():
    assert found(r"\bcislo\b", "je cislo 5")
    assert not found(r"\bčíslo", " číslo")  # JS: mezi mezerou a „č“ hranice slova není
    assert found(r"\Bb", "ab")


def test_case_insensitive_unicode():
    assert found("ČÍSLO", "číslo", "i")
    assert not found("ČÍSLO", "číslo")


def test_ignored_flags_and_literal_braces():
    assert found("a", "A", "giuy")
    assert found(r"x{,5}", "x{,5}")  # JS literál, Python by bral jako kvantifikátor
    assert found(r"a{2}", "aa") and not found(r"^a{2}$", "a")
    assert found(r"[[]", "[") and found(r"[a&&b]", "&")
    assert found(r"\e\q", "eq")  # identity escape


def test_control_hex_unicode_escapes():
    assert found(r"\x41B", "AB")
    assert found(r"\u{1F600}", "😀", "u")
    assert found(r"\cJ", "\n")
    assert found(r"\0", "\x00")


@pytest.mark.parametrize("pattern,flags,msg", [
    ("(?i)abc", "", "inline flags not supported"),
    ("(?i:abc)", "", "inline flags not supported"),
    ("(?s)a.b", "", "inline flags not supported"),
    ("(?P<n>a)", "", "unsupported group syntax"),
    ("(?#comment)", "", "unsupported group syntax"),
    ("(?>a)", "", "unsupported group syntax"),
    ("a++", "", "nothing to repeat"),
    ("a{2}*", "", "nothing to repeat"),
    ("*a", "", "nothing to repeat"),
    ("a|*", "", "nothing to repeat"),
    ("a", "x", "unsupported flag"),
    ("a", "ii", "duplicate flag"),
    ("[a", "", "unterminated character class"),
    ("a\\", "", "end of pattern"),
    ("(a", "", "invalid regex"),
    (r"\p{L}", "u", "property escapes"),
    ("a{3,1}", "", "out of order"),
])
def test_errors(pattern: str, flags: str, msg: str):
    with pytest.raises(ValueError, match=msg.replace("(", r"\(")):
        compile_js_regex(pattern, flags)


def test_lazy_quantifier_ok():
    m = compile_js_regex(r"<.+?>").search("<a><b>")
    assert m and m.group(0) == "<a>"


def test_js_finditer_non_overlapping_and_empty():
    rx = compile_js_regex("aa")
    assert len(list(js_finditer(rx, "aaaaa"))) == 2
    # JS 'baa'.match(/a*/g) → ['', 'aa', ''] (Python findall by dal 4)
    assert len(list(js_finditer(compile_js_regex("a*"), "baa"))) == 3


def test_convert_returns_python_flags():
    import re

    _, f = convert_js_regex("a", "i")
    assert f == re.IGNORECASE
    _, f = convert_js_regex("a", "ms")
    assert f == 0  # m/s se emulují přímo v patternu
