"""Document metadata store: per-knowledge-base JSON files."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .models import DocumentRecord, DocumentStatus


class DocumentStore:
    """Tracks document metadata per knowledge base. One JSON file per KB."""

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir
        self._lock = threading.Lock()
        # kb_id -> {doc_id -> DocumentRecord}
        self._records: dict[str, dict[str, DocumentRecord]] = {}

    def _kb_path(self, kb_id: str) -> Path:
        return self._dir / f"docs_{kb_id}.json"

    def _load_kb(self, kb_id: str) -> dict[str, DocumentRecord]:
        if kb_id in self._records:
            return self._records[kb_id]
        path = self._kb_path(kb_id)
        result: dict[str, DocumentRecord] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                for item in raw.get("documents", []):
                    rec = DocumentRecord.model_validate(item)
                    result[rec.id] = rec
            except (json.JSONDecodeError, Exception):
                pass
        self._records[kb_id] = result
        return result

    def _save_kb(self, kb_id: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        docs = self._records.get(kb_id, {})
        payload = {
            "version": 1,
            "documents": [r.model_dump(mode="json") for r in docs.values()],
        }
        self._kb_path(kb_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        knowledge_base_id: str,
        filename: str,
        file_type: str = "",
        file_size: int = 0,
        original_path: str = "",
        content_hash: str = "",
        page_count: int = 0,
        metadata: dict | None = None,
    ) -> DocumentRecord:
        doc_id = uuid.uuid4().hex
        record = DocumentRecord(
            id=doc_id,
            knowledge_base_id=knowledge_base_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            original_path=original_path,
            content_hash=content_hash,
            page_count=page_count,
            metadata=metadata or {},
            status=DocumentStatus.UPLOADED,
        )
        with self._lock:
            kb_docs = self._load_kb(knowledge_base_id)
            kb_docs[doc_id] = record
            self._save_kb(knowledge_base_id)
        return record

    def get(self, kb_id: str, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._load_kb(kb_id).get(doc_id)

    def find_by_hash(self, kb_id: str, content_hash: str) -> DocumentRecord | None:
        """Return the first doc with matching content_hash (for dedup)."""
        with self._lock:
            for rec in self._load_kb(kb_id).values():
                if rec.content_hash == content_hash:
                    return rec
        return None

    def list_for_kb(self, kb_id: str) -> list[DocumentRecord]:
        with self._lock:
            return list(self._load_kb(kb_id).values())

    def update(self, kb_id: str, doc_id: str, **fields: object) -> DocumentRecord | None:
        with self._lock:
            kb_docs = self._load_kb(kb_id)
            record = kb_docs.get(doc_id)
            if record is None:
                return None
            data = record.model_dump()
            data.update(fields)
            data["updated_at"] = datetime.now(UTC).isoformat()
            updated = DocumentRecord.model_validate(data)
            kb_docs[doc_id] = updated
            self._save_kb(kb_id)
        return updated

    def set_status(
        self,
        kb_id: str,
        doc_id: str,
        status: DocumentStatus,
        error_message: str = "",
    ) -> None:
        self.update(kb_id, doc_id, status=status.value, error_message=error_message)

    def delete(self, kb_id: str, doc_id: str) -> bool:
        with self._lock:
            kb_docs = self._load_kb(kb_id)
            if doc_id not in kb_docs:
                return False
            del kb_docs[doc_id]
            self._save_kb(kb_id)
        return True

    def delete_all_for_kb(self, kb_id: str) -> int:
        with self._lock:
            kb_docs = self._load_kb(kb_id)
            count = len(kb_docs)
            self._records[kb_id] = {}
            self._save_kb(kb_id)
        return count

    def count_for_kb(self, kb_id: str) -> int:
        with self._lock:
            return len(self._load_kb(kb_id))


def compute_file_hash(path: Path) -> str:
    """SHA-256 hash of file contents for content-addressed dedup."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()
