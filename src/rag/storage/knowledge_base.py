"""Knowledge base store: CRUD for KnowledgeBaseRecord backed by JSON."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .models import KnowledgeBaseRecord, KBStatus


class KnowledgeBaseStore:
    """Thread-safe JSON-backed store for knowledge base records."""

    def __init__(self, data_file: Path) -> None:
        self._path = data_file
        self._lock = threading.Lock()
        self._records: dict[str, KnowledgeBaseRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for item in raw.get("knowledge_bases", []):
                record = KnowledgeBaseRecord.model_validate(item)
                self._records[record.id] = record
        except (json.JSONDecodeError, Exception):
            pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "knowledge_bases": [r.model_dump(mode="json") for r in self._records.values()],
        }
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, name: str, description: str = "") -> KnowledgeBaseRecord:
        kb_id = uuid.uuid4().hex
        record = KnowledgeBaseRecord(
            id=kb_id,
            name=name,
            description=description,
            index_dir=kb_id,
        )
        with self._lock:
            self._records[kb_id] = record
            self._save()
        return record

    def get(self, kb_id: str) -> KnowledgeBaseRecord | None:
        return self._records.get(kb_id)

    def list_all(self) -> list[KnowledgeBaseRecord]:
        return list(self._records.values())

    def update(self, kb_id: str, **fields: object) -> KnowledgeBaseRecord | None:
        with self._lock:
            record = self._records.get(kb_id)
            if record is None:
                return None
            data = record.model_dump()
            data.update(fields)
            data["updated_at"] = datetime.now(UTC).isoformat()
            updated = KnowledgeBaseRecord.model_validate(data)
            self._records[kb_id] = updated
            self._save()
        return updated

    def delete(self, kb_id: str) -> bool:
        with self._lock:
            if kb_id not in self._records:
                return False
            del self._records[kb_id]
            self._save()
        return True

    def set_status(self, kb_id: str, status: KBStatus) -> None:
        self.update(kb_id, status=status.value)

    def increment_counts(
        self,
        kb_id: str,
        *,
        doc_delta: int = 0,
        chunk_delta: int = 0,
    ) -> None:
        with self._lock:
            record = self._records.get(kb_id)
            if record is None:
                return
            data = record.model_dump()
            data["document_count"] = max(0, data["document_count"] + doc_delta)
            data["chunk_count"] = max(0, data["chunk_count"] + chunk_delta)
            data["updated_at"] = datetime.now(UTC).isoformat()
            self._records[kb_id] = KnowledgeBaseRecord.model_validate(data)
            self._save()
