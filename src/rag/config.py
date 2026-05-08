"""Runtime configuration. Env-driven, with a per-strategy chunk-size knob."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import ChunkingStrategy


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o"
    judge_model: str = "gpt-4o-mini"
    rerank_model: str = "gpt-4o-mini"  # used when reranker_kind == "llm"

    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size_tokens: int = Field(default=512, ge=64, le=2048)
    chunk_overlap_tokens: int = Field(default=64, ge=0, le=512)
    semantic_chunk_min_tokens: int = Field(default=200, ge=64)
    semantic_chunk_max_tokens: int = Field(default=900, ge=64)

    dedup_cosine_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    dense_top_k: int = Field(default=20, ge=1, le=200)
    sparse_top_k: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    final_top_k: int = Field(default=5, ge=1, le=50)
    rerank_kind: str = "llm"  # "llm" | "cross-encoder" | "none"

    idk_retrieval_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    judge_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    index_dir: Path = Path("./.rag-index")
    eval_set_path: Path = Path("./eval/golden_qa.jsonl")

    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8100, ge=1, le=65535)

    request_timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_chunk(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be < chunk_size_tokens")
        if self.semantic_chunk_min_tokens > self.semantic_chunk_max_tokens:
            raise ValueError(
                "semantic_chunk_min_tokens must be <= semantic_chunk_max_tokens"
            )
        return self


def load_settings(**overrides: Any) -> Settings:
    return Settings(**overrides)
