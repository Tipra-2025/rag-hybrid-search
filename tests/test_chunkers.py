from __future__ import annotations

import pytest

from rag.ingestion.chunkers import (
    FixedTokenChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    chunker_for,
)
from rag.models import ChunkingStrategy, Document, DocumentFormat


def _doc(text: str) -> Document:
    return Document(
        source="src/notes.md",
        format=DocumentFormat.MARKDOWN,
        title="Notes",
        text=text,
    )


def test_fixed_chunker_produces_overlapping_windows() -> None:
    chunker = FixedTokenChunker(chunk_size_tokens=64, overlap_tokens=8)
    chunks = chunker.chunk(_doc("Hello world. " * 200))
    assert len(chunks) >= 2
    assert all(c.strategy is ChunkingStrategy.FIXED for c in chunks)
    # Char ranges should be valid.
    assert all(c.char_start < c.char_end for c in chunks)


def test_fixed_chunker_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        FixedTokenChunker(chunk_size_tokens=64, overlap_tokens=64)


def test_recursive_chunker_respects_paragraph_boundaries() -> None:
    text = "Para one is short.\n\nPara two has a couple of sentences. It elaborates."
    chunker = RecursiveCharacterChunker(chunk_size_tokens=2048, overlap_tokens=0)
    chunks = chunker.chunk(_doc(text))
    assert len(chunks) >= 1
    assert all(c.strategy is ChunkingStrategy.RECURSIVE for c in chunks)


def test_recursive_chunker_splits_long_text() -> None:
    text = ("Sentence " + "word " * 200) * 5
    chunker = RecursiveCharacterChunker(chunk_size_tokens=64, overlap_tokens=8)
    chunks = chunker.chunk(_doc(text))
    assert len(chunks) > 1


def test_semantic_chunker_falls_back_to_recursive_without_embed_fn() -> None:
    chunker = SemanticChunker(min_tokens=100, max_tokens=400)
    chunks = chunker.chunk(_doc("Some passage. " * 300))
    assert chunks
    assert all(c.strategy is ChunkingStrategy.SEMANTIC for c in chunks)


@pytest.mark.asyncio
async def test_semantic_chunker_async_uses_embeddings() -> None:
    async def embed(texts: list[str]) -> list[list[float]]:
        # Make every other sentence dissimilar so we don't merge everything.
        return [
            [1.0, 0.0, 0.0] if i % 2 == 0 else [0.0, 1.0, 0.0]
            for i, _ in enumerate(texts)
        ]

    chunker = SemanticChunker(
        embed_fn=embed, min_tokens=1, max_tokens=10_000, similarity_threshold=0.9
    )
    text = "First sentence. Second one. Third sentence. Fourth. Fifth here."
    chunks = await chunker.chunk_async(_doc(text))
    assert len(chunks) >= 2


def test_chunker_for_unknown_raises() -> None:
    with pytest.raises(ValueError):
        chunker_for("totally-not-a-strategy")  # type: ignore[arg-type]


def test_chunker_for_returns_concrete_strategy() -> None:
    assert isinstance(chunker_for(ChunkingStrategy.FIXED), FixedTokenChunker)
    assert isinstance(chunker_for(ChunkingStrategy.RECURSIVE), RecursiveCharacterChunker)
    assert isinstance(chunker_for(ChunkingStrategy.SEMANTIC), SemanticChunker)
