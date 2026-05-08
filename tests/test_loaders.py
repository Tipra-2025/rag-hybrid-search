from __future__ import annotations

from pathlib import Path

import pytest

from rag.ingestion.loaders import load_document, load_path
from rag.models import DocumentFormat

from .conftest import write_file


def test_load_markdown_extracts_title(tmp_path: Path) -> None:
    p = write_file(tmp_path / "notes.md", "# Hello\n\nbody body body.\n")
    doc = load_document(p)
    assert doc.title == "Hello"
    assert doc.format is DocumentFormat.MARKDOWN
    assert "body" in doc.text


def test_load_text_falls_back_to_filename(tmp_path: Path) -> None:
    p = write_file(tmp_path / "raw.txt", "plain content\nsecond line")
    doc = load_document(p)
    assert doc.title == "raw"
    assert doc.format is DocumentFormat.TEXT


def test_load_html_strips_tags(tmp_path: Path) -> None:
    html = (
        "<html><head><title>Cool</title></head>"
        "<body><script>danger()</script><p>visible <b>text</b></p></body></html>"
    )
    p = write_file(tmp_path / "page.html", html)
    doc = load_document(p)
    assert doc.format is DocumentFormat.HTML
    assert "danger" not in doc.text
    # BeautifulSoup separates inline tags with our chosen newline separator,
    # so we expect "visible" and "text" both in the body but not necessarily
    # adjacent in the concatenated string.
    assert "visible" in doc.text
    assert "text" in doc.text
    assert doc.title == "Cool"


def test_load_document_rejects_empty(tmp_path: Path) -> None:
    p = write_file(tmp_path / "x.txt", "")
    with pytest.raises(ValueError):
        load_document(p)


def test_load_document_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "nope.txt")


def test_load_path_walks_directory(tmp_path: Path) -> None:
    write_file(tmp_path / "a.md", "# A\nbody")
    write_file(tmp_path / "subdir" / "b.txt", "body two")
    write_file(tmp_path / "skip.bin", "binary noise")
    docs = list(load_path(tmp_path))
    titles = {d.title for d in docs}
    assert "A" in titles
    assert "b" in titles
    # Binary extension was skipped.
    assert all(d.source != str(tmp_path / "skip.bin") for d in docs)


def test_load_path_with_single_file(tmp_path: Path) -> None:
    p = write_file(tmp_path / "x.md", "# Solo\nbody")
    docs = list(load_path(p))
    assert len(docs) == 1
