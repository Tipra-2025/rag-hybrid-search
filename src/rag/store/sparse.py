"""Sparse (BM25) store.

`rank_bm25.BM25Okapi` is small, pure-Python, and matches the spec. We rebuild
the corpus index on every `save()` because BM25 statistics are global —
incremental updates would require recomputing IDF anyway.

Tokenization is intentionally simple (lowercase + alphanumeric splits). RAG
quality on English text is dominated by the dense side; BM25's job here is
to recover keyword-exact phrases the embedding model glosses over.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..models import Chunk, SparseHit

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25Store:
    """Rebuilds BM25Okapi from `chunks` whenever the corpus changes."""

    def __init__(self, path: Path):
        self._path = path
        self._chunks: dict[str, Chunk] = {}
        self._index: BM25Okapi | None = None
        self._tokens: list[list[str]] = []
        self._ids: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        for raw in data.get("chunks", []):
            chunk = Chunk.model_validate(raw)
            self._chunks[chunk.chunk_id] = chunk
        self._rebuild_index()

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "chunks": [c.model_dump() for c in self._chunks.values()],
        }
        self._path.write_text(json.dumps(payload), encoding="utf-8")

    def upsert(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        self._rebuild_index()

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, *, top_k: int = 20) -> list[SparseHit]:
        if self._index is None or not self._ids:
            return []
        toks = _tokenize(query)
        if not toks:
            return []
        scores = self._index.get_scores(toks)
        # Use list comprehension since scores is small enough.
        ranked = sorted(
            range(len(scores)), key=lambda i: float(scores[i]), reverse=True
        )[:top_k]
        # BM25Okapi can return zero or negative scores on tiny corpora (the IDF
        # formula reduces to log(1)=0 when N=2 and df=1, for instance). We do
        # NOT filter on score here — the rank order is the signal that matters
        # for downstream RRF fusion. Filtering zero rows would hide most of
        # the test corpora and lose information at production scale too.
        return [
            SparseHit(chunk_id=self._ids[idx], score=float(scores[idx]), rank=rank)
            for rank, idx in enumerate(ranked, start=1)
        ]

    def _rebuild_index(self) -> None:
        if not self._chunks:
            self._index = None
            self._ids = []
            self._tokens = []
            return
        self._ids = list(self._chunks.keys())
        self._tokens = [_tokenize(self._chunks[cid].text) for cid in self._ids]
        # BM25Okapi requires a non-empty corpus and at least one non-empty doc.
        if not any(self._tokens):
            self._index = None
            return
        self._index = BM25Okapi(self._tokens)
