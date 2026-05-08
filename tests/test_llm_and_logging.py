from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from rag.llm_client import ChatMessage, LLMClient, LLMError
from rag.logging import configure, get_logger


@pytest.mark.asyncio
async def test_llm_client_returns_response(
    make_llm_client: Callable[[Callable[[httpx.Request], httpx.Response]], LLMClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = make_llm_client(handler)
    try:
        resp = await client.chat(
            model="gpt-4o-mini", messages=[ChatMessage(role="user", content="x")]
        )
    finally:
        await client.aclose()
    assert resp.content == "ok"


@pytest.mark.asyncio
async def test_llm_client_raises_on_4xx(
    make_llm_client: Callable[[Callable[[httpx.Request], httpx.Response]], LLMClient],
) -> None:
    client = make_llm_client(lambda r: httpx.Response(400, text="bad"))
    try:
        with pytest.raises(LLMError) as ei:
            await client.chat(
                model="x", messages=[ChatMessage(role="user", content="y")]
            )
        assert ei.value.status == 400
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_llm_client_embed_returns_vectors(
    make_llm_client: Callable[[Callable[[httpx.Request], httpx.Response]], LLMClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]}
        )

    client = make_llm_client(handler)
    try:
        out = await client.embed_batch("model", ["a", "b"])
    finally:
        await client.aclose()
    assert out == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.asyncio
async def test_llm_client_embed_empty_input(
    make_llm_client: Callable[[Callable[[httpx.Request], httpx.Response]], LLMClient],
) -> None:
    client = make_llm_client(lambda r: httpx.Response(500))  # should not be called
    try:
        out = await client.embed_batch("m", [])
    finally:
        await client.aclose()
    assert out == []


def test_logging_idempotent() -> None:
    configure("INFO")
    configure("WARNING")
    log = get_logger("t")
    log.info("ok")
