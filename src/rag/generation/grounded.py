"""Grounded answerer: writes an answer using bracketed citations [1], [2], …

System prompt forces:
- Answers come only from the supplied passages.
- Every factual claim is followed by `[N]` where N matches the passage order.
- If the passages don't contain the answer, return the literal "I don't know."
  This is the project's "low retrieval confidence → IDK" surface.
"""

from __future__ import annotations

import time
import uuid

from ..llm_client import ChatMessage, LLMClient
from ..models import Answer, RankedHit

_GROUNDED_SYSTEM = (
    "You are a careful research assistant. Answer the USER QUESTION using ONLY "
    "the numbered PASSAGES below. Rules:\n"
    "1. Every factual claim MUST end with one or more bracketed citations like "
    "[1] or [2][3] referencing the passage you used.\n"
    "2. Do NOT invent facts that are not present in the passages.\n"
    "3. If the passages do not contain the answer, reply EXACTLY: \"I don't know.\" "
    "with no citations.\n"
    "4. Be concise. Plain prose, no markdown headings."
)


class GroundedAnswerer:
    """Render passages → grounded prompt → LLM → `Answer` (without citation verify)."""

    def __init__(
        self,
        *,
        client: LLMClient,
        model: str,
        max_tokens: int = 800,
        temperature: float = 0.0,
        idk_retrieval_threshold: float = 0.35,
    ):
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._idk_threshold = idk_retrieval_threshold

    async def answer(
        self,
        *,
        question: str,
        hits: list[RankedHit],
        retrieval_confidence: float,
    ) -> Answer:
        request_id = uuid.uuid4().hex
        start = time.perf_counter()

        # Hard gate: if retrieval confidence is very low, short-circuit to IDK
        # and skip the LLM call entirely. The spec calls "I don't know" an
        # acceptable output and wants to avoid hallucinated answers when there
        # is nothing to ground against.
        if retrieval_confidence < self._idk_threshold or not hits:
            return Answer(
                request_id=request_id,
                question=question,
                text="I don't know.",
                citations=[],
                used_chunks=hits,
                retrieval_confidence=retrieval_confidence,
                citation_accuracy=1.0 if not hits else 0.0,
                composite_confidence=0.0,
                is_idk=True,
                model=self._model,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )

        passages = _render_passages(hits)
        user = (
            f"USER QUESTION:\n{question}\n\n"
            f"PASSAGES:\n{passages}\n\n"
            f"Answer the question now. Remember: bracketed citations on every "
            f"factual claim, or reply \"I don't know.\""
        )
        resp = await self._client.chat(
            model=self._model,
            messages=[
                ChatMessage(role="system", content=_GROUNDED_SYSTEM),
                ChatMessage(role="user", content=user),
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        text = resp.content.strip()
        is_idk = text.lower().startswith("i don't know") or text.lower() == "i dont know."
        return Answer(
            request_id=request_id,
            question=question,
            text=text,
            citations=[],  # filled in by parse_citations later
            used_chunks=hits,
            retrieval_confidence=retrieval_confidence,
            citation_accuracy=0.0,
            composite_confidence=0.0,
            is_idk=is_idk,
            model=resp.model,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )


def _render_passages(hits: list[RankedHit]) -> str:
    out: list[str] = []
    for i, h in enumerate(hits, start=1):
        title = h.chunk.title or h.chunk.source
        out.append(f"[{i}] (source: {title})\n{h.chunk.text}")
    return "\n\n".join(out)
