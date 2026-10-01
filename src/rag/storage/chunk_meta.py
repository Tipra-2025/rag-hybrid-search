"""Chunk metadata store: per-KB JSON files for chunk inspection and search."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .models import ChunkMetaRecord


class ChunkMetaStore:
    """Stores chunk metadata for the UI chunk inspector. Keyed per KB."""

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir
        self._lock = threading.Lock()
        # kb_id -> {chunk_id -> ChunkMetaRecord}
        self._records: dict[str, dict[str, ChunkMetaRecord]] = {}

    def _kb_path(self, kb_id: str) -> Path:
        return self._dir / f"chunks_{kb_id}.json"

    def _load_kb(self, kb_id: str) -> dict[str, ChunkMetaRecord]:
        if kb_id in self._records:
            return self._records[kb_id]
        path = self._kb_path(kb_id)
        result: dict[str, ChunkMetaRecord] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                for item in raw.get("chunks", []):
                    rec = ChunkMetaRecord.model_validate(item)
                    result[rec.chunk_id] = rec
            except (json.JSONDecodeError, Exception):
                pass
        self._records[kb_id] = result
        return result

    def _save_kb(self, kb_id: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        chunks = self._records.get(kb_id, {})
        payload = {
            "version": 1,
            "chunks": [r.model_dump(mode="json") for r in chunks.values()],
        }
        self._kb_path(kb_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def upsert_many(self, kb_id: str, records: list[ChunkMetaRecord]) -> None:
        with self._lock:
            kb_chunks = self._load_kb(kb_id)
            for rec in records:
                kb_chunks[rec.chunk_id] = rec
            self._save_kb(kb_id)

    def list_for_kb(
        self,
        kb_id: str,
        *,
        document_id: str | None = None,
        page: int | None = None,
        limit: int = 200,
        offset: int = 0,
        search: str | None = None,
    ) -> list[ChunkMetaRecord]:
        with self._lock:
            records = list(self._load_kb(kb_id).values())

        if document_id:
            records = [r for r in records if r.document_id == document_id]
        if page is not None:
            records = [r for r in records if r.page == page]
        if search:
            term = search.lower()
            records = [r for r in records if term in r.text.lower()]

        records.sort(key=lambda r: (r.document_id, r.position))
        return records[offset : offset + limit]

    def count_for_kb(self, kb_id: str, document_id: str | None = None) -> int:
        with self._lock:
            records = list(self._load_kb(kb_id).values())
        if document_id:
            return sum(1 for r in records if r.document_id == document_id)
        return len(records)

    def delete_for_document(self, kb_id: str, document_id: str) -> int:
        with self._lock:
            kb_chunks = self._load_kb(kb_id)
            to_delete = [cid for cid, r in kb_chunks.items() if r.document_id == document_id]
            for cid in to_delete:
                del kb_chunks[cid]
            if to_delete:
                self._save_kb(kb_id)
        return len(to_delete)

    def delete_all_for_kb(self, kb_id: str) -> int:
        with self._lock:
            kb_chunks = self._load_kb(kb_id)
            count = len(kb_chunks)
            self._records[kb_id] = {}
            self._save_kb(kb_id)
        return count
