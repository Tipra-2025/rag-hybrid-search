"""End-to-end engine: question → retrieve → generate → verify → composite confidence."""

from __future__ import annotations

from .generation import (
    GroundedAnswerer,
    composite_confidence,
    parse_citations,
    verify_citations_async,
)
from .llm_client import LLMClient
from .logging import get_logger
from .models import Answer
from .retrieval import HybridRetriever

log = get_logger(__name__)


class RagEngine:
    """Single-call entrypoint. Stateless across requests."""

    def __init__(
        self,
        *,
        client: LLMClient,
        retriever: HybridRetriever,
        answerer: GroundedAnswerer,
        judge_model: str,
        judge_weight: float = 0.5,
    ):
        self._client = client
        self._retriever = retriever
        self._answerer = answerer
        self._judge_model = judge_model
        self._judge_weight = judge_weight

    async def answer(self, question: str) -> Answer:
        hits = await self._retriever.retrieve(question)
        retrieval_conf = HybridRetriever.aggregate_confidence(hits)

        draft = await self._answerer.answer(
            question=question,
            hits=hits,
            retrieval_confidence=retrieval_conf,
        )
        if draft.is_idk:
            composite = composite_confidence(
                retrieval_confidence=retrieval_conf,
                citation_accuracy=draft.citation_accuracy,
                judge_weight=self._judge_weight,
            )
            return draft.model_copy(update={"composite_confidence": composite})

        citations = parse_citations(draft.text, hits)
        accuracy, _supported = await verify_citations_async(
            client=self._client,
            judge_model=self._judge_model,
            question=question,
            answer_text=draft.text,
            citations=citations,
            hits=hits,
        )
        composite = composite_confidence(
            retrieval_confidence=retrieval_conf,
            citation_accuracy=accuracy,
            judge_weight=self._judge_weight,
        )
        return draft.model_copy(
            update={
                "citations": citations,
                "citation_accuracy": accuracy,
                "composite_confidence": composite,
            }
        )
