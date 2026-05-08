"""Parse `[N]` markers, then verify each citation with an LLM-as-judge.

A "verified" citation is one where the cited passage actually supports the
factual claim it's attached to. The verifier is JSON-only so callers never
have to parse free-form text.
"""

from __future__ import annotations

import json
import re
from typing import Final

from ..llm_client import ChatMessage, LLMClient
from ..models import Citation, RankedHit

_BRACKET_RE: Final[re.Pattern[str]] = re.compile(r"\[(\d+)\]")
_JSON_OBJ: Final[re.Pattern[str]] = re.compile(r"\{.*\}", re.DOTALL)

_VERIFIER_SYSTEM = (
    "You audit a single CITATION in a generated answer. Given the QUESTION, "
    "the ANSWER, the cited PASSAGE, and the marker (e.g. [2]), decide whether "
    "the cited passage actually supports the surrounding factual claim in the "
    "answer. Return STRICT JSON: {\"supported\": <bool>, "
    "\"reason\": \"<short>\"}. Output ONLY the JSON object."
)


def parse_citations(text: str, hits: list[RankedHit]) -> list[Citation]:
    """Pull bracketed markers out of the answer text. Markers that point at
    non-existent passage indices are silently dropped.
    """
    out: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for match in _BRACKET_RE.finditer(text):
        idx = int(match.group(1))
        if 1 <= idx <= len(hits):
            hit = hits[idx - 1]
            key = (match.group(0), hit.chunk.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                Citation(
                    marker=match.group(0),
                    chunk_id=hit.chunk.chunk_id,
                    source=hit.chunk.source,
                    title=hit.chunk.title,
                    quote=hit.chunk.text[:240],
                )
            )
    return out


async def verify_citations_async(
    *,
    client: LLMClient,
    judge_model: str,
    question: str,
    answer_text: str,
    citations: list[Citation],
    hits: list[RankedHit],
) -> tuple[float, list[bool]]:
    """Returns (accuracy, per_citation_supported). Accuracy is 1.0 when there
    are zero citations *and* the answer is "I don't know" — otherwise it's
    the fraction supported.
    """
    if not citations:
        return (1.0 if answer_text.lower().startswith("i don't know") else 0.0, [])

    chunk_by_id: dict[str, str] = {h.chunk.chunk_id: h.chunk.text for h in hits}
    import asyncio

    results = await asyncio.gather(
        *(
            _verify_one(
                client=client,
                judge_model=judge_model,
                question=question,
                answer_text=answer_text,
                citation=c,
                passage_text=chunk_by_id.get(c.chunk_id, ""),
            )
            for c in citations
        ),
        return_exceptions=False,
    )
    supported = list(results)
    accuracy = sum(1 for s in supported if s) / len(supported)
    return accuracy, supported


async def _verify_one(
    *,
    client: LLMClient,
    judge_model: str,
    question: str,
    answer_text: str,
    citation: Citation,
    passage_text: str,
) -> bool:
    if not passage_text:
        return False
    user = (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer_text}\n\n"
        f"CITATION MARKER: {citation.marker}\n"
        f"CITED PASSAGE:\n{passage_text[:2000]}\n\n"
        f"Return JSON now."
    )
    try:
        resp = await client.chat(
            model=judge_model,
            messages=[
                ChatMessage(role="system", content=_VERIFIER_SYSTEM),
                ChatMessage(role="user", content=user),
            ],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
    except Exception:
        return False

    match = _JSON_OBJ.search(resp.content or "")
    if not match:
        return False
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False
    return bool(data.get("supported", False))
