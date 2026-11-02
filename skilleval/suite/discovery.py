"""Hledání case adresářů v eval adresáři (case = adresář s prompt.md a/nebo case.yaml; skupiny = vnoření)."""

from __future__ import annotations

from pathlib import Path

CASE_FILES = ("prompt.md", "case.yaml")
SKIP_DIRS = frozenset({"results", "mocks"})


def is_case_dir(path: Path) -> bool:
    return path.is_dir() and any((path / f).is_file() for f in CASE_FILES)


def discover_case_dirs(root: Path) -> list[Path]:
    """Case adresáře pod `root` (sám `root` se jako case nebere), seřazené podle cesty.

    Do case adresáře se nesestupuje (graders/, fixtures … patří case); přeskakuje results/, mocks/ a skryté adresáře."""
    out: list[Path] = []
    seen: set[Path] = set()

    def walk(d: Path) -> None:
        real = d.resolve()
        if real in seen:  # ochrana proti cyklům symlinků
            return
        seen.add(real)
        try:
            children = sorted(c for c in d.iterdir() if c.is_dir())
        except OSError:
            return
        for c in children:
            if c.name.startswith(".") or c.name in SKIP_DIRS:
                continue
            if is_case_dir(c):
                out.append(c)
            else:
                walk(c)

    if root.is_dir():
        walk(root)
    return out
