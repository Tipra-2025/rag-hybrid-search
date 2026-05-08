# ADR-001: Ship three chunking strategies side-by-side, swap via config

**Status:** Accepted

## Context

Chunking is the single biggest knob in a RAG pipeline. The "right" strategy depends on the corpus: long-form prose wants paragraph-respecting splits, code/CSV wants different separators, FAQ-style content wants semantic merging. Picking one strategy and shipping it makes the project look opinionated when the truth is "it depends."

## Decision

Ship `FixedTokenChunker`, `RecursiveCharacterChunker`, and `SemanticChunker` behind a single `Chunker` interface. The active strategy is chosen per ingestion job via `ChunkingStrategy` (env: `RAG_CHUNKING_STRATEGY`). The index directory is namespaced by strategy so two strategies can coexist on disk and be compared head-to-head.

## Consequences

**Why this is right:**

- **Apples-to-apples eval.** The golden Q&A harness can run against any strategy by flipping `RAG_CHUNKING_STRATEGY` and re-pointing at the matching index. The retrieval and generation layers are strategy-agnostic, so we measure the chunking change in isolation.
- **Recursive is a sensible default**, but anchoring users to it is a mistake at scale: technical docs with dense tables benefit from `fixed`, conversational FAQs benefit from `semantic`. We let teams compare without code changes.
- **Stable chunk IDs.** All three strategies derive `chunk_id` from `sha256(source|position|text)`. Re-running ingestion is idempotent: unchanged chunks skip embedding entirely.

**Trade-offs:**

- Three strategies means three code paths to maintain. Mitigated by the shared `Chunker` ABC and a thin slice → chunks helper that all three reuse.
- The semantic chunker requires an embedding callback to actually be semantic. Without one (sync mode), it falls back to recursive — documented and tested.
- Storage doubles or triples if a team keeps multiple strategies indexed. The namespacing makes this explicit; teams either pick a single strategy in production or pay the disk cost knowingly.

## Alternatives considered

- **One strategy, stand by it.** Fragile in practice — recurring incidents with "this single PDF type doesn't chunk well" force ad-hoc fixes.
- **Strategy-per-document via a router LLM.** Tempting; rejected because it introduces a dependency on a model call at ingestion time and obscures the eval signal. We may revisit if the eval harness shows strategy choice should be content-type-conditional.

## Revisit if

- A team consistently runs only one strategy. We can keep the others behind an extra and lighten the default install.
- A new strategy (e.g. layout-aware PDF chunking via `pdfplumber`) becomes a strong default for a corpus type we serve.
