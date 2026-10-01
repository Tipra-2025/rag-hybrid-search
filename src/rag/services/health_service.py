"""Health service: real-valued system status checks."""

from __future__ import annotations

from ..config import Settings
from ..llm_client import LLMClient


class HealthService:
    """Returns real health status for each system component."""

    def __init__(self, settings: Settings, llm_client: LLMClient | None = None) -> None:
        self._settings = settings
        self._client = llm_client

    def get_health(self, n_chunks_dense: int, n_chunks_sparse: int) -> dict[str, object]:
        s = self._settings

        # API is running if we can return this
        api_status = "ok"

        # Embedding service: check if key is configured
        embedding_status = (
            "ok" if s.openai_api_key else "not_configured"
        )
        embedding_model = s.embedding_model if s.openai_api_key else "—"

        # Reranker
        if s.rerank_kind == "none":
            reranker_status = "disabled"
            reranker_model = "none"
        elif s.rerank_kind == "llm":
            reranker_status = "ok" if s.openai_api_key else "not_configured"
            reranker_model = s.rerank_model
        else:
            # cross-encoder: check if sentence-transformers installed
            try:
                import sentence_transformers  # noqa: F401
                reranker_status = "ok"
            except ImportError:
                reranker_status = "not_configured"
            reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        # LLM generation
        llm_status = "ok" if s.openai_api_key else "not_configured"
        llm_model = s.generation_model if s.openai_api_key else "—"

        # Index status
        index_status = "ok" if n_chunks_dense > 0 else "empty"

        return {
            "status": "ok",
            "api": {"status": api_status},
            "indexes": {
                "status": index_status,
                "n_chunks_dense": n_chunks_dense,
                "n_chunks_sparse": n_chunks_sparse,
            },
            "embedding_service": {
                "status": embedding_status,
                "model": embedding_model,
            },
            "reranker": {
                "status": reranker_status,
                "kind": s.rerank_kind,
                "model": reranker_model,
            },
            "llm": {
                "status": llm_status,
                "model": llm_model,
            },
        }
