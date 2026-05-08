"""Multi-format loaders. Each returns a `Document` (raw text + metadata).

Format detection is by file extension; callers may override `format=`. We keep
the loader surface narrow on purpose — anything fancier (tables, layout,
images) belongs behind the same `Document` contract in a custom loader.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from bs4 import BeautifulSoup

from ..models import Document, DocumentFormat


def _format_for(path: Path, override: DocumentFormat | None) -> DocumentFormat:
    if override is not None:
        return override
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return DocumentFormat.MARKDOWN
    if suffix in {".html", ".htm"}:
        return DocumentFormat.HTML
    if suffix == ".pdf":
        return DocumentFormat.PDF
    return DocumentFormat.TEXT


def _read_pdf(path: Path) -> str:
    import pypdf

    reader = pypdf.PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)


def _read_html(text: str) -> tuple[str, str]:
    """Strip tags. Returns (extracted_title, plain_text)."""
    soup = BeautifulSoup(text, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    body = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in body.splitlines()]
    plain = "\n".join(ln for ln in lines if ln)
    return title, plain


def _title_from_markdown(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def load_document(path: Path, *, format: DocumentFormat | None = None) -> Document:
    """Load a single file into a `Document`."""
    if not path.exists():
        raise FileNotFoundError(path)
    fmt = _format_for(path, format)
    if fmt is DocumentFormat.PDF:
        text = _read_pdf(path)
        title = path.stem
    elif fmt is DocumentFormat.HTML:
        raw = path.read_text(encoding="utf-8", errors="replace")
        title, text = _read_html(raw)
    elif fmt is DocumentFormat.MARKDOWN:
        text = path.read_text(encoding="utf-8")
        title = _title_from_markdown(text) or path.stem
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        title = path.stem

    text = text.strip()
    if not text:
        raise ValueError(f"empty document: {path}")
    return Document(
        source=str(path),
        format=fmt,
        title=title,
        text=text,
        metadata={"path": str(path), "size_chars": str(len(text))},
    )


def load_path(
    path: Path,
    *,
    extensions: Iterable[str] = (".md", ".markdown", ".txt", ".html", ".htm", ".pdf"),
) -> Iterator[Document]:
    """Walk a directory (or yield a single file) into `Document` instances."""
    if path.is_file():
        yield load_document(path)
        return
    exts = {e.lower() for e in extensions}
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        if child.suffix.lower() not in exts:
            continue
        try:
            yield load_document(child)
        except (ValueError, FileNotFoundError):
            continue
