"""Pluggable reranker.

Three implementations:

- `NoOpReranker`: keeps RRF order. Used when speed > quality.
- `LLMReranker` (default): asks a small LLM to grade query/chunk on a 0-1
  scale; cheap, no extra ML deps.
- `CrossEncoderReranker`: lazy-imports `sentence-transformers` for an MS-MARCO
  cross-encoder. Optional via the `[reranker]` extra.

All implementations honor the same `Reranker.rerank()` async interface.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Final

from ..llm_client import ChatMessage, LLMClient
from ..models import Chunk, RankedHit

_JSON_OBJ: Final[re.Pattern[str]] = re.compile(r"\{.*\}", re.DOTALL)


class Reranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        *,
        query: str,
        candidates: list[RankedHit],
        top_k: int,
    ) -> list[RankedHit]:  # pragma: no cover
        ...


class NoOpReranker(Reranker):
    async def rerank(
        self,
        *,
        query: str,
        candidates: list[RankedHit],
        top_k: int,
    ) -> list[RankedHit]:
        return candidates[:top_k]


_LLM_RERANKER_SYSTEM = (
    "You score the relevance of a passage to a user's QUERY. Return STRICT "
    "JSON with one field: {\"score\": <float 0-1; 1 = directly answers, "
    "0 = unrelated>}. Output ONLY the JSON."
)


class LLMReranker(Reranker):
    """Cheap LLM-as-reranker. One call per candidate; trivially parallelizable."""

    def __init__(self, *, client: LLMClient, model: str, max_tokens: int = 60):
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[RankedHit],
        top_k: int,
    ) -> list[RankedHit]:
        if not candidates:
            return []
        import asyncio

        scores = await asyncio.gather(
            *(self._score_one(query, c.chunk) for c in candidates),
            return_exceptions=False,
        )
        rescored = [
            c.model_copy(update={"score": s})
            for c, s in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]

    async def _score_one(self, query: str, chunk: Chunk) -> float:
        try:
            resp = await self._client.chat(
                model=self._model,
                messages=[
                    ChatMessage(role="system", content=_LLM_RERANKER_SYSTEM),
                    ChatMessage(
                        role="user",
                        content=(
                            f"QUERY:\n{query}\n\n"
                            f"PASSAGE:\n{chunk.text[:1500]}\n\n"
                            f"Return JSON now."
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            return 0.0
        return _parse_score(resp.content)


def _parse_score(raw: str) -> float:
    match = _JSON_OBJ.search(raw or "")
    if not match:
        return 0.0
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 0.0
    try:
        return max(0.0, min(1.0, float(data.get("score", 0.0))))
    except (TypeError, ValueError):
        return 0.0


class CrossEncoderReranker(Reranker):  # pragma: no cover — optional dep
    """Cross-encoder backed reranker. Requires the `[reranker]` extra."""

    def __init__(self, *, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[RankedHit],
        top_k: int,
    ) -> list[RankedHit]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)
        rescored = [
            c.model_copy(update={"score": float(s)})
            for c, s in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]


def reranker_for(kind: str, *, client: LLMClient | None = None, model: str = "gpt-4o-mini") -> Reranker:
    if kind == "none":
        return NoOpReranker()
    if kind == "llm":
        if client is None:
            raise ValueError("kind='llm' requires a client")
        return LLMReranker(client=client, model=model)
    if kind == "cross-encoder":  # pragma: no cover
        return CrossEncoderReranker()
    raise ValueError(f"unknown reranker kind: {kind}")
