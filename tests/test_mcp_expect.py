import pytest

from skilleval.mcp.expect import PatternError, check_expect, parse_pattern, subset_match, validate_expect


@pytest.mark.parametrize("pattern", [
    "/abc/", r"/^ord-\d+$/", "/^[a-z]+$/i", "/a.b/s", r"/x{2,3}/", r"/x{2}/", r"/x{2,}/", r"/\w+@\w+\.cz/",
    r"/[^0-9]?/", r"/\//", r"/a\.b/", "/\\" + "u010d/", "/\\" + "x41/", "/}/", "/a{/", "/[a]]/",
])
def test_valid_patterns(pattern):
    assert validate_expect({"x": pattern}) == []


@pytest.mark.parametrize("pattern,why", [
    ("/(a)/", "groups"), ("/a|b/", "alternation"), (r"/(a)\1/", "groups"), (r"/a\1/", "backreferences"),
    ("/(?=a)/", "groups"), ("/a/g", "flag"), ("/a/m", "flag"), ("/a/ii", "duplicate"), ("/a**/", "nested"),
    ("/a+?/", "lazy"), ("/*a/", "quantifier"), ("/a^b/", "start"), ("/a$b/", "end"), ("/[a/", "unterminated"),
    (r"/\bx/", "assertion"), ("/[]/", "empty"), (r"/x{3,1}/", "invalid quantifier"), (r"/\q/", "unsupported"),
])
def test_invalid_patterns(pattern, why):
    errs = validate_expect({"title": pattern})
    assert len(errs) == 1
    assert pattern in errs[0] and "title" in errs[0]
    assert why in errs[0]


def test_validate_shapes():
    assert validate_expect(None) == []
    assert validate_expect(["a"]) != []
    assert validate_expect({"a": []}) != []
    assert validate_expect({"a": [{"x": 1}]}) != []
    assert validate_expect({"a": "string", "b": [1, "x", None, True], "c": 5, "d": "literal"}) == []


def test_types():
    exp = {"s": "string", "n": "number", "b": "boolean", "a": "array", "o": "object"}
    assert check_expect(exp, {"s": "x", "n": 1.5, "b": False, "a": [], "o": {}}) is None
    assert "expected number" in check_expect(exp, {"s": "x", "n": True, "b": False, "a": [], "o": {}})
    assert "expected string" in check_expect({"s": "string"}, {"s": 1})
    assert check_expect({"s": "string"}, {}) == "s: missing"


def test_literals_and_lists():
    assert check_expect({"priority": ["low", "medium", "high"]}, {"priority": "low"}) is None
    assert "not one of" in check_expect({"priority": ["low", "medium"]}, {"priority": "urgent"})
    assert check_expect({"dry_run": False}, {"dry_run": False}) is None
    assert check_expect({"dry_run": False}, {"dry_run": 0}) is not None  # bool ≠ number
    assert check_expect({"id": 71781}, {"id": 71781.0}) is None
    assert check_expect({"id": 71781}, {"id": "71781"}) is not None
    assert check_expect({"name": "literal"}, {"name": "literal"}) is None


def test_regex_and_dot_paths():
    exp = {"order.id": r"/^\d+$/", "items.0.code": "/^ab/i"}
    assert check_expect(exp, {"order": {"id": 71781}, "items": [{"code": "ABc"}]}) is None
    assert "does not match" in check_expect(exp, {"order": {"id": "x1"}, "items": [{"code": "ab"}]})
    assert check_expect({"a.b": "string"}, {"a": {}}) == "a.b: missing"
    assert check_expect({"x": "/^a$/"}, {"x": "a\n"}) is not None  # JS $ bez m = konec řetězce
    assert check_expect({"x": "/č/i"}, {"x": "Č"}) is None
    assert check_expect({"x": r"/\d/"}, {"x": "٣"}) is not None  # \d jen ASCII jako JS
    assert "expected a value" in check_expect({"x": "/a/"}, {"x": {"a": 1}})


def test_max_length():
    pat = parse_pattern("/a/")
    assert pat.max_len == 1000
    assert parse_pattern("/^a/").max_len == 10_000
    assert parse_pattern("/^a+b*c/").max_len == 5_000
    assert "longer than" in check_expect({"x": "/a/"}, {"x": "a" * 1001})
    assert check_expect({"x": "/^a/"}, {"x": "a" * 5000}) is None


def test_parse_pattern_errors():
    with pytest.raises(PatternError):
        parse_pattern("abc")


def test_subset_match():
    assert subset_match(None, {"a": 1})
    assert subset_match({"dry_run": False}, {"dry_run": False, "id": 1})
    assert not subset_match({"dry_run": False}, {"dry_run": True})
    assert not subset_match({"dry_run": False}, {})
    assert subset_match({"order": {"id": 1}}, {"order": {"id": 1, "x": 2}})
    assert subset_match({"order.id": 1}, {"order": {"id": 1}})
    assert subset_match({"id": "/^7/"}, {"id": 71781})
