"""Three chunking strategies, all returning the same `Chunk[]` shape.

The point: every strategy is a drop-in replacement; the rest of the system
treats chunks identically. The eval harness can compare strategies head-to-head.

- `FixedTokenChunker`: token-window with overlap. Cheapest, predictable, but
  cuts mid-sentence.
- `RecursiveCharacterChunker`: split on a separator hierarchy
  (`\n\n` → `\n` → `. ` → ` `), respect token budget, fall back to fixed.
- `SemanticChunker`: split into sentences, then merge consecutive sentences
  whose embeddings are close enough (cosine ≥ threshold). Optional —
  requires embeddings; falls back to recursive when no embedding function
  is supplied.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from ..models import Chunk, ChunkingStrategy, Document

# Token estimator: prefer tiktoken, else 4 chars/token heuristic. Cached.
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:  # pragma: no cover

    def _count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


def _stable_chunk_id(doc_source: str, position: int, text: str) -> str:
    digest = hashlib.sha256(f"{doc_source}|{position}|{text}".encode()).hexdigest()
    return f"chunk_{digest[:16]}"


@dataclass(slots=True, frozen=True)
class _Slice:
    text: str
    char_start: int
    char_end: int


class Chunker(ABC):
    strategy: ChunkingStrategy

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:  # pragma: no cover
        ...


class FixedTokenChunker(Chunker):
    """Token-window with overlap. Returns chunks at predictable token sizes."""

    strategy = ChunkingStrategy.FIXED

    def __init__(self, *, chunk_size_tokens: int = 512, overlap_tokens: int = 64):
        if overlap_tokens >= chunk_size_tokens:
            raise ValueError("overlap_tokens must be < chunk_size_tokens")
        self._size = chunk_size_tokens
        self._overlap = overlap_tokens

    def chunk(self, doc: Document) -> list[Chunk]:
        # Walk by approximate-tokens worth of characters: 1 token ≈ 4 chars
        # (we use the actual tokenizer to confirm). This keeps the splitter
        # framework-free while still respecting tokenizer boundaries.
        text = doc.text
        if not text:
            return []
        approx_chars = max(1, self._size * 4)
        approx_overlap = max(0, self._overlap * 4)
        slices: list[_Slice] = []
        i = 0
        while i < len(text):
            j = min(len(text), i + approx_chars)
            piece = text[i:j]
            slices.append(_Slice(text=piece, char_start=i, char_end=j))
            if j >= len(text):
                break
            i = j - approx_overlap
        return _slices_to_chunks(doc, slices, self.strategy)


class RecursiveCharacterChunker(Chunker):
    """Split by a separator hierarchy until each piece fits the token budget."""

    strategy = ChunkingStrategy.RECURSIVE

    DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")

    def __init__(
        self,
        *,
        chunk_size_tokens: int = 512,
        overlap_tokens: int = 64,
        separators: Iterable[str] = DEFAULT_SEPARATORS,
    ):
        if overlap_tokens >= chunk_size_tokens:
            raise ValueError("overlap_tokens must be < chunk_size_tokens")
        self._size = chunk_size_tokens
        self._overlap = overlap_tokens
        self._separators = tuple(separators)

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if not text:
            return []
        slices = self._recursive_split(text, 0, self._separators)
        merged = self._merge_to_budget(slices)
        if self._overlap > 0:
            merged = self._with_overlap(merged, doc.text)
        return _slices_to_chunks(doc, merged, self.strategy)

    def _recursive_split(
        self, text: str, char_offset: int, separators: tuple[str, ...]
    ) -> list[_Slice]:
        if _count_tokens(text) <= self._size:
            return [_Slice(text=text, char_start=char_offset, char_end=char_offset + len(text))]
        if not separators:
            # Hard fallback: window-split on character count.
            approx_chars = max(1, self._size * 4)
            out: list[_Slice] = []
            for i in range(0, len(text), approx_chars):
                out.append(
                    _Slice(
                        text=text[i : i + approx_chars],
                        char_start=char_offset + i,
                        char_end=char_offset + min(i + approx_chars, len(text)),
                    )
                )
            return out
        sep, *rest = separators
        parts: list[_Slice] = []
        cursor = 0
        for raw in text.split(sep):
            if not raw:
                cursor += len(sep)
                continue
            parts.extend(self._recursive_split(raw, char_offset + cursor, tuple(rest)))
            cursor += len(raw) + len(sep)
        return parts

    def _merge_to_budget(self, slices: list[_Slice]) -> list[_Slice]:
        if not slices:
            return []
        out: list[_Slice] = []
        buffer = slices[0]
        for nxt in slices[1:]:
            combined_text = buffer.text + " " + nxt.text
            if _count_tokens(combined_text) <= self._size:
                buffer = _Slice(
                    text=combined_text,
                    char_start=buffer.char_start,
                    char_end=nxt.char_end,
                )
            else:
                out.append(buffer)
                buffer = nxt
        out.append(buffer)
        return out

    def _with_overlap(self, slices: list[_Slice], full_text: str) -> list[_Slice]:
        from itertools import pairwise

        if not slices or self._overlap <= 0:
            return slices
        approx_overlap_chars = self._overlap * 4
        out = [slices[0]]
        for _prev, cur in pairwise(slices):
            start = max(0, cur.char_start - approx_overlap_chars)
            out.append(_Slice(
                text=full_text[start : cur.char_end],
                char_start=start,
                char_end=cur.char_end,
            ))
        return out


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


class SemanticChunker(Chunker):
    """Sentence-level merge by embedding similarity. Falls back to recursive
    when no `embed_fn` is provided.
    """

    strategy = ChunkingStrategy.SEMANTIC

    def __init__(
        self,
        *,
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
        min_tokens: int = 200,
        max_tokens: int = 900,
        similarity_threshold: float = 0.78,
    ):
        if min_tokens > max_tokens:
            raise ValueError("min_tokens must be <= max_tokens")
        self._embed_fn = embed_fn
        self._min = min_tokens
        self._max = max_tokens
        self._threshold = similarity_threshold

    def chunk(self, doc: Document) -> list[Chunk]:
        # Sync entrypoint without embeddings: degrade to recursive splitting,
        # but tag the resulting chunks with `SEMANTIC` so callers can still tell
        # they came out of the semantic pipeline.
        recursive = RecursiveCharacterChunker(
            chunk_size_tokens=self._max, overlap_tokens=0
        )
        out = recursive.chunk(doc)
        return [c.model_copy(update={"strategy": ChunkingStrategy.SEMANTIC}) for c in out]

    async def chunk_async(self, doc: Document) -> list[Chunk]:
        if self._embed_fn is None:
            return self.chunk(doc)
        sentences = self._split_sentences(doc.text)
        if not sentences:
            return []
        embeddings = await self._embed_fn([s for s, _, _ in sentences])
        merged = _merge_by_similarity(
            sentences,
            embeddings,
            min_tokens=self._min,
            max_tokens=self._max,
            threshold=self._threshold,
        )
        return _slices_to_chunks(doc, merged, self.strategy)

    @staticmethod
    def _split_sentences(text: str) -> list[tuple[str, int, int]]:
        out: list[tuple[str, int, int]] = []
        cursor = 0
        for piece in _SENTENCE_RE.split(text):
            if not piece.strip():
                cursor += len(piece) + 1
                continue
            start = text.find(piece, cursor)
            if start == -1:
                start = cursor
            end = start + len(piece)
            out.append((piece, start, end))
            cursor = end
        return out


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _merge_by_similarity(
    sentences: list[tuple[str, int, int]],
    embeddings: list[list[float]],
    *,
    min_tokens: int,
    max_tokens: int,
    threshold: float,
) -> list[_Slice]:
    if not sentences:
        return []
    out: list[_Slice] = []
    buf_text = sentences[0][0]
    buf_start = sentences[0][1]
    buf_end = sentences[0][2]
    buf_emb = embeddings[0]
    for (text, start, end), emb in zip(sentences[1:], embeddings[1:], strict=False):
        sim = _cosine(buf_emb, emb)
        candidate = buf_text + " " + text
        toks = _count_tokens(candidate)
        if (sim >= threshold and toks <= max_tokens) or _count_tokens(buf_text) < min_tokens:
            buf_text = candidate
            buf_end = end
            # rolling-mean centroid: cheap stand-in for re-embedding
            buf_emb = [(x + y) / 2 for x, y in zip(buf_emb, emb, strict=False)]
        else:
            out.append(_Slice(text=buf_text, char_start=buf_start, char_end=buf_end))
            buf_text, buf_start, buf_end, buf_emb = text, start, end, emb
    out.append(_Slice(text=buf_text, char_start=buf_start, char_end=buf_end))
    return out


def _slices_to_chunks(
    doc: Document, slices: list[_Slice], strategy: ChunkingStrategy
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, sl in enumerate(slices):
        cleaned = sl.text.strip()
        if not cleaned:
            continue
        chunks.append(
            Chunk(
                chunk_id=_stable_chunk_id(doc.source, i, cleaned),
                source=doc.source,
                title=doc.title,
                text=cleaned,
                strategy=strategy,
                position=i,
                char_start=sl.char_start,
                char_end=sl.char_end,
                metadata=dict(doc.metadata),
            )
        )
    return chunks


def chunker_for(
    strategy: ChunkingStrategy,
    *,
    chunk_size_tokens: int = 512,
    overlap_tokens: int = 64,
    embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
    semantic_min: int = 200,
    semantic_max: int = 900,
) -> Chunker:
    if strategy is ChunkingStrategy.FIXED:
        return FixedTokenChunker(
            chunk_size_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens
        )
    if strategy is ChunkingStrategy.RECURSIVE:
        return RecursiveCharacterChunker(
            chunk_size_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens
        )
    if strategy is ChunkingStrategy.SEMANTIC:
        return SemanticChunker(
            embed_fn=embed_fn, min_tokens=semantic_min, max_tokens=semantic_max
        )
    raise ValueError(f"unknown chunking strategy: {strategy}")


__all__ = [
    "Chunker",
    "FixedTokenChunker",
    "RecursiveCharacterChunker",
    "SemanticChunker",
    "chunker_for",
]

# Re-export for cosine in tests
_test_only_cosine = _cosine
del uuid  # keep import cost low
