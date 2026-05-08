"""Shared fixtures: deterministic embedding model + httpx-mocked LLMClient."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from rag.llm_client import LLMClient


def deterministic_embedding(text: str, dim: int = 32) -> list[float]:
    """Hash-derived pseudo-embedding. Stable across runs, no model needed."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seeds = [b for b in digest]
    out: list[float] = []
    for i in range(dim):
        # Use sin of a hashed seed so values are smooth and reproducible.
        out.append(math.sin(seeds[i % len(seeds)] * (i + 1) / 11.0))
    return out


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        },
    )


def chat_json(payload: dict[str, object]) -> httpx.Response:
    return chat_response(json.dumps(payload))


def embed_response(texts: list[str]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"data": [{"embedding": deterministic_embedding(t)} for t in texts]},
    )


@pytest.fixture
def make_llm_client() -> Callable[[Callable[[httpx.Request], httpx.Response]], LLMClient]:
    """Factory yields LLMClients that own their underlying httpx transport.

    `LLMClient.__init__` defaults `_owns_client=False` when a transport is
    passed in; flipping it to True here means `await client.aclose()` actually
    frees the AsyncClient. Without this, pytest's strict-warning mode raises
    PytestUnraisableExceptionWarning when the loop is GC'd and tests fail
    non-deterministically depending on ordering.
    """

    def make(handler: Callable[[httpx.Request], httpx.Response]) -> LLMClient:
        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = LLMClient(api_key="test", client=transport, max_retries=0)
        client._owns_client = True  # type: ignore[attr-defined]
        return client

    return make


def write_file(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def parse_json_input(body: bytes) -> dict[str, object]:
    return json.loads(body)
