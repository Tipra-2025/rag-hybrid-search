from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.config import load_settings
from rag.models import Chunk, ChunkingStrategy, Document, DocumentFormat


def test_chunk_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Chunk.model_validate(
            {
                "chunk_id": "x",
                "source": "s",
                "title": "",
                "text": "t",
                "strategy": "fixed",
                "position": 0,
                "char_start": 0,
                "char_end": 1,
                "metadata": {},
                "extra": "nope",
            }
        )


def test_document_requires_text() -> None:
    with pytest.raises(ValidationError):
        Document(source="s", format=DocumentFormat.TEXT, text="")


def test_settings_defaults_are_consistent() -> None:
    s = load_settings()
    assert s.chunk_overlap_tokens < s.chunk_size_tokens
    assert s.semantic_chunk_min_tokens <= s.semantic_chunk_max_tokens


def test_settings_rejects_overlap_ge_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap_tokens"):
        load_settings(chunk_size_tokens=128, chunk_overlap_tokens=128)


def test_settings_rejects_semantic_inversion() -> None:
    with pytest.raises(ValueError, match="semantic"):
        load_settings(semantic_chunk_min_tokens=900, semantic_chunk_max_tokens=200)


def test_chunking_strategy_string_round_trip() -> None:
    assert ChunkingStrategy("fixed") is ChunkingStrategy.FIXED
    assert ChunkingStrategy.SEMANTIC.value == "semantic"
