"""Tests for the storage layer: KnowledgeBaseStore, DocumentStore, ChunkMetaStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.storage import (
    ChunkMetaStore,
    DocumentStore,
    KnowledgeBaseRecord,
    KnowledgeBaseStore,
)
from rag.storage.models import (
    ChunkMetaRecord,
    DocumentStatus,
    KBStatus,
    RetrievalSettings,
)


# ---------------------------------------------------------------------------
# KnowledgeBaseStore
# ---------------------------------------------------------------------------


def test_create_knowledge_base(tmp_path: Path) -> None:
    store = KnowledgeBaseStore(tmp_path / "kbs.json")
    kb = store.create("Test KB", description="A test knowledge base")
    assert kb.name == "Test KB"
    assert kb.description == "A test knowledge base"
    assert kb.id
    assert kb.document_count == 0
    assert kb.chunk_count == 0
    assert kb.status == KBStatus.ACTIVE


def test_knowledge_base_persists(tmp_path: Path) -> None:
    """KBs survive a store reload from disk."""
    path = tmp_path / "kbs.json"
    store = KnowledgeBaseStore(path)
    kb = store.create("Persisted KB")
    kb_id = kb.id

    # Reload from disk
    store2 = KnowledgeBaseStore(path)
    loaded = store2.get(kb_id)
    assert loaded is not None
    assert loaded.name == "Persisted KB"


def test_delete_knowledge_base(tmp_path: Path) -> None:
    store = KnowledgeBaseStore(tmp_path / "kbs.json")
    kb = store.create("To Delete")
    assert store.delete(kb.id) is True
    assert store.get(kb.id) is None


def test_delete_nonexistent_kb_returns_false(tmp_path: Path) -> None:
    store = KnowledgeBaseStore(tmp_path / "kbs.json")
    assert store.delete("nonexistent-id") is False


def test_kb_increment_counts(tmp_path: Path) -> None:
    store = KnowledgeBaseStore(tmp_path / "kbs.json")
    kb = store.create("Counter Test")
    store.increment_counts(kb.id, doc_delta=3, chunk_delta=42)
    updated = store.get(kb.id)
    assert updated is not None
    assert updated.document_count == 3
    assert updated.chunk_count == 42


def test_kb_increment_counts_does_not_go_negative(tmp_path: Path) -> None:
    store = KnowledgeBaseStore(tmp_path / "kbs.json")
    kb = store.create("Non-negative")
    store.increment_counts(kb.id, doc_delta=-100, chunk_delta=-100)
    updated = store.get(kb.id)
    assert updated is not None
    assert updated.document_count == 0
    assert updated.chunk_count == 0


def test_kb_list_all(tmp_path: Path) -> None:
    store = KnowledgeBaseStore(tmp_path / "kbs.json")
    store.create("KB 1")
    store.create("KB 2")
    store.create("KB 3")
    kbs = store.list_all()
    assert len(kbs) == 3


def test_kb_update_retrieval_settings(tmp_path: Path) -> None:
    store = KnowledgeBaseStore(tmp_path / "kbs.json")
    kb = store.create("Settings Test")
    settings = RetrievalSettings(dense_top_k=50, final_top_k=10)
    updated = store.update(kb.id, retrieval_settings=settings.model_dump())
    assert updated is not None
    # Reload
    store2 = KnowledgeBaseStore(tmp_path / "kbs.json")
    reloaded = store2.get(kb.id)
    assert reloaded is not None
    assert reloaded.retrieval_settings.dense_top_k == 50
    assert reloaded.retrieval_settings.final_top_k == 10


# ---------------------------------------------------------------------------
# DocumentStore
# ---------------------------------------------------------------------------


def test_create_document(tmp_path: Path) -> None:
    doc_store = DocumentStore(tmp_path)
    doc = doc_store.create(
        knowledge_base_id="kb1",
        filename="test.pdf",
        file_type="pdf",
        file_size=1024,
    )
    assert doc.filename == "test.pdf"
    assert doc.file_type == "pdf"
    assert doc.status == DocumentStatus.UPLOADED
    assert doc.id


def test_document_persists(tmp_path: Path) -> None:
    doc_store = DocumentStore(tmp_path)
    doc = doc_store.create(knowledge_base_id="kb1", filename="persist.md")
    doc_id = doc.id

    doc_store2 = DocumentStore(tmp_path)
    loaded = doc_store2.get("kb1", doc_id)
    assert loaded is not None
    assert loaded.filename == "persist.md"


def test_document_status_update(tmp_path: Path) -> None:
    doc_store = DocumentStore(tmp_path)
    doc = doc_store.create(knowledge_base_id="kb1", filename="status.txt")
    doc_store.set_status("kb1", doc.id, DocumentStatus.INDEXED)
    updated = doc_store.get("kb1", doc.id)
    assert updated is not None
    assert updated.status == DocumentStatus.INDEXED


def test_document_failure_records_error(tmp_path: Path) -> None:
    doc_store = DocumentStore(tmp_path)
    doc = doc_store.create(knowledge_base_id="kb1", filename="fail.pdf")
    doc_store.set_status(
        "kb1", doc.id, DocumentStatus.FAILED, error_message="Parse error"
    )
    updated = doc_store.get("kb1", doc.id)
    assert updated is not None
    assert updated.status == DocumentStatus.FAILED
    assert updated.error_message == "Parse error"


def test_duplicate_document_detection(tmp_path: Path) -> None:
    """find_by_hash enables content-hash dedup."""
    doc_store = DocumentStore(tmp_path)
    doc = doc_store.create(
        knowledge_base_id="kb1",
        filename="dup.md",
        content_hash="abc123",
    )
    doc_store.set_status("kb1", doc.id, DocumentStatus.INDEXED)

    found = doc_store.find_by_hash("kb1", "abc123")
    assert found is not None
    assert found.id == doc.id

    not_found = doc_store.find_by_hash("kb1", "xyz999")
    assert not_found is None


def test_document_isolation_across_kbs(tmp_path: Path) -> None:
    """Documents in KB-A must not appear in KB-B."""
    doc_store = DocumentStore(tmp_path)
    doc_store.create(knowledge_base_id="kb_a", filename="doc_a.md")
    doc_store.create(knowledge_base_id="kb_b", filename="doc_b.md")

    docs_a = doc_store.list_for_kb("kb_a")
    docs_b = doc_store.list_for_kb("kb_b")

    assert len(docs_a) == 1
    assert docs_a[0].filename == "doc_a.md"
    assert len(docs_b) == 1
    assert docs_b[0].filename == "doc_b.md"

    # Cross-check: KB-A doc not visible from KB-B
    assert not any(d.filename == "doc_a.md" for d in docs_b)


def test_document_delete(tmp_path: Path) -> None:
    doc_store = DocumentStore(tmp_path)
    doc = doc_store.create(knowledge_base_id="kb1", filename="todelete.md")
    assert doc_store.delete("kb1", doc.id) is True
    assert doc_store.get("kb1", doc.id) is None


def test_delete_all_for_kb(tmp_path: Path) -> None:
    doc_store = DocumentStore(tmp_path)
    for i in range(5):
        doc_store.create(knowledge_base_id="kb1", filename=f"doc{i}.md")
    count = doc_store.delete_all_for_kb("kb1")
    assert count == 5
    assert doc_store.list_for_kb("kb1") == []


# ---------------------------------------------------------------------------
# ChunkMetaStore
# ---------------------------------------------------------------------------


def test_chunk_meta_upsert_and_list(tmp_path: Path) -> None:
    store = ChunkMetaStore(tmp_path)
    records = [
        ChunkMetaRecord(
            chunk_id=f"chunk_{i}",
            document_id="doc1",
            knowledge_base_id="kb1",
            text=f"This is chunk number {i} with some content.",
            position=i,
            char_count=40,
            token_count=10,
        )
        for i in range(5)
    ]
    store.upsert_many("kb1", records)
    listed = store.list_for_kb("kb1")
    assert len(listed) == 5


def test_chunk_meta_filter_by_document(tmp_path: Path) -> None:
    store = ChunkMetaStore(tmp_path)
    for i in range(3):
        store.upsert_many(
            "kb1",
            [ChunkMetaRecord(
                chunk_id=f"doc1_chunk_{i}",
                document_id="doc1",
                knowledge_base_id="kb1",
                text="Doc 1 content.",
                position=i,
            )],
        )
    for i in range(2):
        store.upsert_many(
            "kb1",
            [ChunkMetaRecord(
                chunk_id=f"doc2_chunk_{i}",
                document_id="doc2",
                knowledge_base_id="kb1",
                text="Doc 2 content.",
                position=i,
            )],
        )
    doc1_chunks = store.list_for_kb("kb1", document_id="doc1")
    assert len(doc1_chunks) == 3
    assert all(c.document_id == "doc1" for c in doc1_chunks)


def test_chunk_meta_search(tmp_path: Path) -> None:
    store = ChunkMetaStore(tmp_path)
    store.upsert_many("kb1", [
        ChunkMetaRecord(
            chunk_id="c1", document_id="d1", knowledge_base_id="kb1",
            text="The password must be 14 characters long.", position=0
        ),
        ChunkMetaRecord(
            chunk_id="c2", document_id="d1", knowledge_base_id="kb1",
            text="Annual leave accrues at 15 days per year.", position=1
        ),
    ])
    results = store.list_for_kb("kb1", search="password")
    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_chunk_meta_delete_for_document(tmp_path: Path) -> None:
    store = ChunkMetaStore(tmp_path)
    store.upsert_many("kb1", [
        ChunkMetaRecord(
            chunk_id="c1", document_id="doc_to_delete", knowledge_base_id="kb1",
            text="Will be deleted.", position=0
        ),
        ChunkMetaRecord(
            chunk_id="c2", document_id="doc_to_keep", knowledge_base_id="kb1",
            text="Will remain.", position=0
        ),
    ])
    deleted = store.delete_for_document("kb1", "doc_to_delete")
    assert deleted == 1
    remaining = store.list_for_kb("kb1")
    assert len(remaining) == 1
    assert remaining[0].document_id == "doc_to_keep"


def test_chunk_meta_isolation_across_kbs(tmp_path: Path) -> None:
    """Chunks from KB-A must not appear in KB-B."""
    store = ChunkMetaStore(tmp_path)
    store.upsert_many("kb_a", [
        ChunkMetaRecord(
            chunk_id="a1", document_id="da", knowledge_base_id="kb_a",
            text="KB A content.", position=0
        )
    ])
    store.upsert_many("kb_b", [
        ChunkMetaRecord(
            chunk_id="b1", document_id="db", knowledge_base_id="kb_b",
            text="KB B content.", position=0
        )
    ])
    chunks_a = store.list_for_kb("kb_a")
    chunks_b = store.list_for_kb("kb_b")
    assert len(chunks_a) == 1
    assert len(chunks_b) == 1
    assert chunks_a[0].chunk_id == "a1"
    assert chunks_b[0].chunk_id == "b1"


def test_chunk_metadata_fields(tmp_path: Path) -> None:
    """Chunk records must retain all source information for citations."""
    store = ChunkMetaStore(tmp_path)
    record = ChunkMetaRecord(
        chunk_id="full_meta",
        document_id="doc1",
        knowledge_base_id="kb1",
        text="Full text content here.",
        page=3,
        position=7,
        char_start=500,
        char_end=523,
        token_count=6,
        char_count=23,
        source="docs/policy.md",
        title="Policy Document",
        strategy="recursive",
        metadata={"section": "Introduction"},
    )
    store.upsert_many("kb1", [record])
    loaded = store.list_for_kb("kb1")
    assert len(loaded) == 1
    c = loaded[0]
    assert c.page == 3
    assert c.position == 7
    assert c.char_start == 500
    assert c.char_end == 523
    assert c.source == "docs/policy.md"
    assert c.title == "Policy Document"
    assert c.metadata == {"section": "Introduction"}
