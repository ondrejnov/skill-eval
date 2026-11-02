"""Robustní parsování YAML frontmatteru (`---\\n…\\n---`) v Markdown souborech (prompt.md, gradery, mocky, SKILL.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class FrontmatterError(ValueError):
    pass


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """Rozdělí text na (surový frontmatter | None, tělo). Frontmatter = první řádek `---`, konec dalším `---` (nebo `...`)."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if not lines or lines[0].rstrip() != "---":
        return None, text
    for i in range(1, len(lines)):
        if lines[i].rstrip() in ("---", "..."):
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1:])
    raise FrontmatterError("neukončený frontmatter (chybí uzavírací řádek ---)")


def parse_frontmatter(text: str) -> tuple[dict[str, Any] | None, str]:
    """→ (data frontmatteru | None když chybí, tělo). Neplatný YAML nebo ne-mapa = FrontmatterError."""
    raw, body = split_frontmatter(text)
    if raw is None:
        return None, body
    try:
        data = yaml.safe_load(raw) if raw.strip() else {}
    except yaml.YAMLError as e:
        raise FrontmatterError(f"neplatný YAML ve frontmatteru: {_yaml_msg(e)}") from e
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise FrontmatterError("frontmatter musí být YAML mapa (klíč: hodnota)")
    return data, body


def read_frontmatter(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise FrontmatterError("soubor není v UTF-8") from e
    return parse_frontmatter(text)


def strip_blank_lines(text: str) -> str:
    """Odstraní prázdné (whitespace) řádky na začátku a konci; odsazení prvního řádku i vnitřek ponechá doslova."""
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def load_yaml_file(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise FrontmatterError(f"neplatný YAML: {_yaml_msg(e)}") from e
    except UnicodeDecodeError as e:
        raise FrontmatterError("soubor není v UTF-8") from e


def _yaml_msg(e: yaml.YAMLError) -> str:
    return " ".join(str(e).split())
