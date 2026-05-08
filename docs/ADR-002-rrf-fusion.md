# ADR-002: Reciprocal Rank Fusion, not score-normalized linear blending

**Status:** Accepted

## Context

We have two retrievers that produce different signals: a dense (embedding) store with cosine in `[-1, 1]` and a sparse BM25 store with unbounded positive scores whose distribution shifts per query. Fusing them into one ranking is the central RAG decision.

Two families of approaches:

1. **Score-normalized blending.** Min-max or z-score the two score vectors, then take a weighted sum.
2. **Rank-based fusion.** Use the *positions* of items in each list, not their raw scores. Reciprocal Rank Fusion (RRF) is the canonical version: `Σ 1 / (k + rank_i)`.

## Decision

RRF with a default `k = 60` (the Cormack-Clarke choice the literature has converged on). Configurable via `RAG_RRF_K`.

## Consequences

**Why this is right:**

- **Per-query calibration is unnecessary.** BM25 absolute scores depend on query length and corpus IDF; embedding cosines depend on the embedding model. Normalization that works for one query frequently misranks another. Rank fusion sidesteps this entirely.
- **Hyperparameter-light.** `k` controls how aggressively low ranks decay. The Cormack-Clarke recommendation of 60 has held up across many corpora; we expose the knob but expect almost no team to need to tune it.
- **Robust to one retriever falling over.** If sparse returns an empty result (e.g. an out-of-vocabulary query), RRF reduces gracefully to the dense ranking; same the other way. Score-blending in that situation has to special-case missing-side handling.
- **Cheap.** RRF is O(N) over the union of both retrievers' top-K. There's no normalization pass.

**Trade-offs:**

- RRF discards score magnitude. A passage that scores 0.99 cosine and a passage that scores 0.51 cosine are both "rank 1" if they top their respective lists — but the difference matters. The cross-encoder rerank stage gets the actual scores back into the picture, so the lossy step is not the final word.
- The choice of `k` is somewhat arbitrary. We document this and note that 60 is a sane default; teams that want to expose their own knob already have one.

## Alternatives considered

- **Weighted score-blending with min-max normalization.** Tested informally; per-query variance blew it up. Rejected.
- **Learned-to-rank fusion.** Best-in-class but requires labeled data the project doesn't ship with. The eval harness is the foundation we'd build that on; deferred.
- **Rank-biased precision.** Interesting alternative; not enough public deployment evidence to justify departing from RRF.

## Revisit if

- The eval harness shows large gains from learned fusion on a specific corpus.
- A team runs into a workload where rank-discarding hurts and switches to a hybrid score-then-rank scheme. The retriever interface accepts a custom fusion function as a one-line override.
