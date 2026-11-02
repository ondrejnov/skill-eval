"""target_text: last_message | trace | files | {source: file, path} | mock_calls."""

from __future__ import annotations

import base64
import json

import pytest
from test_grading_helpers import make_art

from skilleval.grading.targets import TargetError, elide_lines, target_text
from skilleval.models import McpCallRecord

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20


def test_last_message_default(tmp_path):
    art = make_art(tmp_path, final="Hotovo.")
    assert target_text(None, art, for_llm=False) == ("Hotovo.", [])
    assert target_text("last_message", art, for_llm=True) == ("Hotovo.", [])


def test_trace_regex_sees_all_lines_json_escaped(tmp_path):
    art = make_art(tmp_path)
    text, _ = target_text("trace", art, for_llm=False)
    assert text == "\n".join(art.trace_lines)
    assert '\\"vyřizuje se\\"' in text  # uvozovky v řetězcích escapované, ne-ASCII ponechané
    assert '"dry_run":false' in text


def test_trace_llm_head_tail(tmp_path):
    lines = [json.dumps({"n": i}) for i in range(30)]
    art = make_art(tmp_path, trace_lines=lines)
    text, _ = target_text("trace", art, for_llm=True)
    out = text.split("\n")
    assert len(out) == 25
    assert out[:12] == lines[:12] and out[-12:] == lines[-12:]
    assert out[12] == "[…6 messages elided…]"
    # krátký trace beze změny
    assert elide_lines(lines[:24]) == "\n".join(lines[:24])


def test_files_list(tmp_path):
    art = make_art(tmp_path, created=["out/a.md", "b.txt"])
    assert target_text("files", art, for_llm=False)[0] == "out/a.md\nb.txt"


def test_file_contents_text(tmp_path):
    art = make_art(tmp_path, files={"out/report.md": "# Report\nObrat 1 000 Kč"})
    text, imgs = target_text({"source": "file", "path": "out/report.md"}, art, for_llm=False)
    assert text.startswith("# Report") and imgs == []
    # absolutní cesta uvnitř workspace je OK
    text2, _ = target_text({"source": "file", "path": str(art.workspace / "out/report.md")}, art, for_llm=True)
    assert text2 == text


def test_file_image_for_llm_only(tmp_path):
    art = make_art(tmp_path, files={"chart.png": PNG})
    text, imgs = target_text({"source": "file", "path": "chart.png"}, art, for_llm=True)
    assert "image" in text and imgs[0]["media_type"] == "image/png"
    assert base64.b64decode(imgs[0]["data"]) == PNG
    with pytest.raises(TargetError, match="image"):
        target_text({"source": "file", "path": "chart.png"}, art, for_llm=False)


def test_file_webp_by_magic(tmp_path):
    art = make_art(tmp_path, files={"x.bin": b"RIFF\x00\x00\x00\x00WEBPVP8 "})
    _, imgs = target_text({"source": "file", "path": "x.bin"}, art, for_llm=True)
    assert imgs[0]["media_type"] == "image/webp"


def test_file_other_binary_refused(tmp_path):
    art = make_art(tmp_path, files={"deck.pptx": b"PK\x03\x04\x00\x00binary", "x.pdf": b"%PDF-1.4\n\xff\xfe\xfa"})
    with pytest.raises(TargetError, match="binary"):
        target_text({"source": "file", "path": "deck.pptx"}, art, for_llm=True)
    with pytest.raises(TargetError, match="not UTF-8"):
        target_text({"source": "file", "path": "x.pdf"}, art, for_llm=True)


def test_file_missing_and_outside(tmp_path):
    art = make_art(tmp_path)
    with pytest.raises(TargetError, match="not found"):
        target_text({"source": "file", "path": "nope.md"}, art, for_llm=False)
    with pytest.raises(TargetError, match="outside"):
        target_text({"source": "file", "path": "../secret.txt"}, art, for_llm=False)
    with pytest.raises(ValueError, match="path"):
        target_text({"source": "file"}, art, for_llm=False)


def test_mock_calls(tmp_path):
    art = make_art(tmp_path, mock_calls=[
        McpCallRecord(server="tracker", tool="create_issue", input={"title": "Chyba „X“"}, output="Created #1",
                      is_error=False, source="mock"),
        McpCallRecord(server="tracker", tool="close", input={}, output="nope", is_error=True, source="mock"),
    ])
    text, _ = target_text("mock_calls", art, for_llm=False)
    rows = [json.loads(x) for x in text.split("\n")]
    assert rows[0] == {"server": "tracker", "tool": "create_issue", "input": {"title": "Chyba „X“"},
                       "output": "Created #1", "isError": False}
    assert rows[1]["isError"] is True
    assert "„X“" in text  # ensure_ascii=False


def test_unknown_target(tmp_path):
    with pytest.raises(ValueError, match="unsupported target"):
        target_text("transcript", make_art(tmp_path), for_llm=False)
