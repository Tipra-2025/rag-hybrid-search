"""Composite confidence: weighted blend of retrieval and citation accuracy.

The composite is a single number callers can threshold against. The default
weight (0.5) puts retrieval and citation accuracy on equal footing — both
catch different failure modes:

- Retrieval confidence catches "we couldn't find anything to ground against"
  (cosine top-K is uniformly low).
- Citation accuracy catches "we found something but the LLM cited it wrong"
  (the answer references a passage that doesn't actually support it).

A composite below 0.4 is a strong signal that the answer should not be shown
to the user without review.
"""

from __future__ import annotations


def composite_confidence(
    *,
    retrieval_confidence: float,
    citation_accuracy: float,
    judge_weight: float = 0.5,
) -> float:
    if not 0.0 <= judge_weight <= 1.0:
        raise ValueError("judge_weight must be in [0, 1]")
    blended = judge_weight * citation_accuracy + (1.0 - judge_weight) * retrieval_confidence
    return max(0.0, min(1.0, blended))
