"""IngestionService: wraps the existing IngestionPipeline with KB isolation.

Responsibilities:
- Accept a file path + knowledge_base_id + chunking config
- Detect file type (md, txt, html, pdf, docx, pptx, xlsx, csv, json)
- Parse document using existing loaders + extended loaders
- Chunk using existing chunker
- Dedup + embed + index into the KB-specific stores
- Store DocumentRecord + ChunkMetaRecord metadata
- Update KB document/chunk counts
- Report success or failure

We intentionally reuse the existing IngestionPipeline and loaders.
We DO NOT rewrite the chunking or indexing engine.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

from ..ingestion.loaders import load_document
from ..ingestion.pipeline import IngestionPipeline
from ..llm_client import LLMClient
from ..logging import get_logger
from ..models import ChunkingStrategy, Document, DocumentFormat
from ..storage import ChunkMetaStore, DocumentStore, KnowledgeBaseStore
from ..storage.models import ChunkMetaRecord, DocumentStatus
from .kb_service import KnowledgeBaseService

log = get_logger(__name__)


def _compute_content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _detect_file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    mapping = {
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
        ".html": "html",
        ".htm": "html",
        ".pdf": "pdf",
        ".docx": "docx",
        ".pptx": "pptx",
        ".xlsx": "xlsx",
        ".csv": "csv",
        ".json": "json",
    }
    return mapping.get(suffix, "text")


def _load_extended_document(path: Path) -> Document:
    """Load DOCX, PPTX, XLSX, CSV, JSON in addition to the original formats."""
    suffix = path.suffix.lower()

    if suffix == ".docx":
        return _load_docx(path)
    if suffix == ".pptx":
        return _load_pptx(path)
    if suffix in (".xlsx", ".xls"):
        return _load_xlsx(path)
    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".json":
        return _load_json(path)

    # Fall back to original loaders for md/txt/html/pdf
    return load_document(path)


def _load_docx(path: Path) -> Document:
    try:
        import docx  # python-docx
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for .docx files. "
            "Install with: pip install python-docx"
        ) from exc

    doc = docx.Document(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    if not text:
        raise ValueError(f"empty DOCX: {path}")
    title = paragraphs[0][:100] if paragraphs else path.stem
    return Document(
        source=str(path),
        format=DocumentFormat.TEXT,
        title=title,
        text=text,
        metadata={"path": str(path), "size_chars": str(len(text)), "type": "docx"},
    )


def _load_pptx(path: Path) -> Document:
    try:
        from pptx import Presentation  # python-pptx
    except ImportError as exc:
        raise ImportError(
            "python-pptx is required for .pptx files. "
            "Install with: pip install python-pptx"
        ) from exc

    prs = Presentation(str(path))
    slides: list[str] = []
    for i, slide in enumerate(prs.slides, 1):
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_text.append(shape.text.strip())
        if slide_text:
            slides.append(f"[Slide {i}]\n" + "\n".join(slide_text))
    text = "\n\n".join(slides)
    if not text:
        raise ValueError(f"empty PPTX: {path}")
    return Document(
        source=str(path),
        format=DocumentFormat.TEXT,
        title=path.stem,
        text=text,
        metadata={"path": str(path), "size_chars": str(len(text)), "type": "pptx",
                  "slide_count": str(len(slides))},
    )


def _load_xlsx(path: Path) -> Document:
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError(
            "openpyxl is required for .xlsx files. "
            "Install with: pip install openpyxl"
        ) from exc

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    sheets: list[str] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows: list[str] = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            row_text = "\t".join(cells).strip()
            if row_text:
                rows.append(row_text)
        if rows:
            sheets.append(f"[Sheet: {sheet_name}]\n" + "\n".join(rows))
    text = "\n\n".join(sheets)
    if not text:
        raise ValueError(f"empty XLSX: {path}")
    return Document(
        source=str(path),
        format=DocumentFormat.TEXT,
        title=path.stem,
        text=text,
        metadata={"path": str(path), "size_chars": str(len(text)), "type": "xlsx",
                  "sheet_count": str(len(sheets))},
    )


def _load_csv(path: Path) -> Document:
    import csv

    lines: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            lines.append("\t".join(row))
    text = "\n".join(lines).strip()
    if not text:
        raise ValueError(f"empty CSV: {path}")
    return Document(
        source=str(path),
        format=DocumentFormat.TEXT,
        title=path.stem,
        text=text,
        metadata={"path": str(path), "size_chars": str(len(text)), "type": "csv"},
    )


def _load_json(path: Path) -> Document:
    import json

    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        obj = json.loads(raw)
        # Pretty-print for better chunking
        text = json.dumps(obj, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        text = raw
    text = text.strip()
    if not text:
        raise ValueError(f"empty JSON: {path}")
    return Document(
        source=str(path),
        format=DocumentFormat.TEXT,
        title=path.stem,
        text=text,
        metadata={"path": str(path), "size_chars": str(len(text)), "type": "json"},
    )


class IngestionService:
    """Ties the existing ingestion pipeline to the KB metadata layer."""

    def __init__(
        self,
        kb_service: KnowledgeBaseService,
        doc_store: DocumentStore,
        chunk_meta_store: ChunkMetaStore,
        llm_client: LLMClient,
        embedding_model: str,
    ) -> None:
        self._kb_service = kb_service
        self._doc_store = doc_store
        self._chunk_meta = chunk_meta_store
        self._client = llm_client
        self._embedding_model = embedding_model

    async def ingest_file(
        self,
        *,
        kb_id: str,
        file_path: Path,
        chunking_strategy: str = "recursive",
        chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 64,
        dedup_cosine_threshold: float = 0.95,
    ) -> dict[str, Any]:
        """Ingest a single file into a knowledge base.

        Returns a result dict with status, counts, and error if any.
        """
        kb = self._kb_service.get(kb_id)
        if kb is None:
            return {"success": False, "error": f"Knowledge base {kb_id} not found"}

        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        # Read file bytes for content hash
        file_bytes = file_path.read_bytes()
        content_hash = _compute_content_hash(file_bytes)
        file_type = _detect_file_type(file_path)
        file_size = len(file_bytes)

        # Check for duplicate
        existing = self._doc_store.find_by_hash(kb_id, content_hash)
        if existing and existing.status.value == "indexed":
            log.info(
                "ingestion.duplicate_skipped",
                filename=file_path.name,
                doc_id=existing.id,
            )
            return {
                "success": True,
                "skipped": True,
                "doc_id": existing.id,
                "message": "Duplicate document — already indexed",
            }

        # Create document record
        doc_record = self._doc_store.create(
            knowledge_base_id=kb_id,
            filename=file_path.name,
            file_type=file_type,
            file_size=file_size,
            original_path=str(file_path),
            content_hash=content_hash,
        )
        doc_id = doc_record.id

        # Mark as processing
        self._doc_store.set_status(kb_id, doc_id, DocumentStatus.PROCESSING)

        try:
            # Load document
            document = _load_extended_document(file_path)

            # Estimate page count
            page_count = 0
            if file_type == "pdf":
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                page_count = len(reader.pages)

            self._doc_store.update(kb_id, doc_id, page_count=page_count)

            # Get per-KB stores
            strategy_enum = ChunkingStrategy(chunking_strategy)
            dense = self._kb_service.get_dense_store(kb_id, chunking_strategy)
            sparse = self._kb_service.get_sparse_store(kb_id, chunking_strategy)

            # Build and run the ingestion pipeline
            pipeline = IngestionPipeline(
                client=self._client,
                embedding_model=self._embedding_model,
                dense=dense,
                sparse=sparse,
                strategy=strategy_enum,
                chunk_size_tokens=chunk_size_tokens,
                overlap_tokens=chunk_overlap_tokens,
                dedup_cosine_threshold=dedup_cosine_threshold,
            )
            report = await pipeline.ingest([document])

            # Store chunk metadata for UI
            # Pull chunks from the dense store (they were just inserted)
            all_chunks = dense.all_chunks()
            # Filter to only chunks from this document
            doc_chunks = [c for c in all_chunks if c.source == str(file_path)]
            chunk_records = [
                ChunkMetaRecord(
                    chunk_id=c.chunk_id,
                    document_id=doc_id,
                    knowledge_base_id=kb_id,
                    text=c.text,
                    page=int(c.metadata.get("page", "0") if c.metadata else 0),
                    position=c.position,
                    char_start=c.char_start,
                    char_end=c.char_end,
                    token_count=max(1, len(c.text) // 4),
                    char_count=len(c.text),
                    source=c.source,
                    title=c.title,
                    strategy=c.strategy.value,
                    metadata=dict(c.metadata) if c.metadata else {},
                )
                for c in doc_chunks
            ]
            if chunk_records:
                self._chunk_meta.upsert_many(kb_id, chunk_records)

            chunk_count = len(chunk_records)

            # Update document record
            self._doc_store.update(
                kb_id, doc_id, chunk_count=chunk_count
            )
            self._doc_store.set_status(kb_id, doc_id, DocumentStatus.INDEXED)

            # Update KB counters
            self._kb_service._kb_store.increment_counts(
                kb_id, doc_delta=1, chunk_delta=chunk_count
            )

            log.info(
                "ingestion.success",
                kb_id=kb_id,
                doc_id=doc_id,
                filename=file_path.name,
                chunks_added=report.n_chunks_added,
                chunks_deduped=report.n_chunks_deduplicated,
            )
            return {
                "success": True,
                "doc_id": doc_id,
                "chunks_added": report.n_chunks_added,
                "chunks_deduplicated": report.n_chunks_deduplicated,
                "chunk_count": chunk_count,
                "strategy": chunking_strategy,
            }

        except Exception as exc:
            error_msg = str(exc)
            log.warning(
                "ingestion.failed",
                kb_id=kb_id,
                doc_id=doc_id,
                filename=file_path.name,
                error=error_msg,
            )
            self._doc_store.set_status(
                kb_id, doc_id, DocumentStatus.FAILED, error_message=error_msg
            )
            return {
                "success": False,
                "doc_id": doc_id,
                "error": error_msg,
            }

    async def delete_document(self, kb_id: str, doc_id: str) -> dict[str, Any]:
        """Delete a document, its chunks from indexes, and its metadata."""
        doc = self._doc_store.get(kb_id, doc_id)
        if doc is None:
            return {"success": False, "error": "Document not found"}

        # Delete chunk metadata
        chunks_deleted = self._chunk_meta.delete_for_document(kb_id, doc_id)

        # Note: we cannot easily remove individual chunks from the flat JSON
        # dense/sparse stores without a full rebuild. We track that documents
        # are "deleted" via metadata. A full re-index of the KB would clean
        # the vector store.
        # For now we delete the metadata records.
        self._doc_store.delete(kb_id, doc_id)

        # Update KB counters
        self._kb_service._kb_store.increment_counts(
            kb_id, doc_delta=-1, chunk_delta=-chunks_deleted
        )

        return {
            "success": True,
            "chunks_removed_from_meta": chunks_deleted,
        }
