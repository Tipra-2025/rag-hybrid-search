"""CLI: ingest, ask, eval, serve."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import uvicorn

from .api.app import AppState, create_app
from .config import Settings, load_settings
from .engine import RagEngine
from .generation import GroundedAnswerer
from .ingestion import IngestionPipeline, load_path
from .llm_client import LLMClient
from .logging import configure as configure_logging
from .models import EvalCase
from .retrieval import HybridRetriever, reranker_for
from .store import BM25Store, DenseVectorStore


def _build_state(settings: Settings | None = None) -> AppState:
    settings = settings or load_settings()
    configure_logging(settings.log_level)
    if not settings.openai_api_key:
        click.echo("RAG_OPENAI_API_KEY is required", err=True)
        sys.exit(1)
    client = LLMClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout_s=settings.request_timeout_s,
        max_retries=settings.max_retries,
    )
    dense_path, sparse_path = IngestionPipeline.index_paths(
        settings.index_dir, settings.chunking_strategy
    )
    dense = DenseVectorStore(dense_path)
    sparse = BM25Store(sparse_path)

    ingestion = IngestionPipeline(
        client=client,
        embedding_model=settings.embedding_model,
        dense=dense,
        sparse=sparse,
        strategy=settings.chunking_strategy,
        chunk_size_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        semantic_min_tokens=settings.semantic_chunk_min_tokens,
        semantic_max_tokens=settings.semantic_chunk_max_tokens,
        dedup_cosine_threshold=settings.dedup_cosine_threshold,
    )
    reranker = reranker_for(
        settings.rerank_kind, client=client, model=settings.rerank_model
    )
    retriever = HybridRetriever(
        client=client,
        embedding_model=settings.embedding_model,
        dense=dense,
        sparse=sparse,
        reranker=reranker,
        dense_top_k=settings.dense_top_k,
        sparse_top_k=settings.sparse_top_k,
        rrf_k=settings.rrf_k,
        final_top_k=settings.final_top_k,
    )
    answerer = GroundedAnswerer(
        client=client,
        model=settings.generation_model,
        idk_retrieval_threshold=settings.idk_retrieval_threshold,
    )
    engine = RagEngine(
        client=client,
        retriever=retriever,
        answerer=answerer,
        judge_model=settings.judge_model,
        judge_weight=settings.judge_weight,
    )
    return AppState(
        client=client,
        engine=engine,
        ingestion=ingestion,
        dense=dense,
        sparse=sparse,
    )


@click.group()
def main() -> None:
    """rag-hybrid-search CLI."""


@main.command("ingest")
@click.option("--path", required=True, type=click.Path(path_type=Path, exists=True))
def ingest_cmd(path: Path) -> None:
    state = _build_state()

    async def _run() -> None:
        try:
            docs = list(load_path(path))
            report = await state.ingestion.ingest(docs)
            click.echo(
                f"docs={report.n_documents} added={report.n_chunks_added} "
                f"deduped={report.n_chunks_deduplicated} "
                f"failures={report.n_failures} strategy={report.strategy.value}"
            )
        finally:
            await state.client.aclose()

    asyncio.run(_run())


@main.command("ask")
@click.option("--question", required=True)
def ask_cmd(question: str) -> None:
    state = _build_state()

    async def _run() -> None:
        try:
            answer = await state.engine.answer(question)
            click.echo(answer.text)
            click.echo(
                f"\n[retrieval={answer.retrieval_confidence:.2f} "
                f"citations={answer.citation_accuracy:.2f} "
                f"composite={answer.composite_confidence:.2f} "
                f"idk={answer.is_idk}]"
            )
            for c in answer.citations:
                click.echo(f"  {c.marker} {c.title} ({c.source})")
        finally:
            await state.client.aclose()

    asyncio.run(_run())


@main.command("eval")
@click.option("--cases", "cases_path", type=click.Path(path_type=Path, exists=True),
              required=True, help="Golden Q&A JSONL.")
def eval_cmd(cases_path: Path) -> None:
    state = _build_state()

    async def _run() -> None:
        try:
            cases: list[EvalCase] = []
            with cases_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cases.append(EvalCase.model_validate_json(line))
            n = 0
            cited_ok = 0
            confident = 0
            for case in cases:
                ans = await state.engine.answer(case.question)
                n += 1
                cited_sources = {c.source for c in ans.citations}
                if all(src in cited_sources for src in case.must_cite_sources):
                    cited_ok += 1
                if ans.composite_confidence >= 0.6:
                    confident += 1
                click.echo(
                    f"{case.case_id}: composite={ans.composite_confidence:.2f} "
                    f"citations={len(ans.citations)} idk={ans.is_idk}"
                )
            click.echo("---")
            click.echo(f"total={n} confident={confident} citation_pass={cited_ok}")
        finally:
            await state.client.aclose()

    asyncio.run(_run())


@main.command("serve")
def serve_cmd() -> None:
    settings = load_settings()
    state = _build_state(settings)
    app = create_app(state)
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


@main.command("config")
def config_cmd() -> None:
    s = load_settings()
    click.echo(json.dumps(s.model_dump(mode="json"), indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
