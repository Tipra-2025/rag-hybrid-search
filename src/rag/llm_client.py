"""Async OpenAI-compatible chat + embeddings client."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class LLMError(Exception):
    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class _RetryableError(LLMError):
    pass


@dataclass(slots=True, frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(slots=True, frozen=True)
class ChatResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class LLMClient:
    """Tiny POST /chat/completions + /embeddings wrapper. Mockable via httpx.MockTransport."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 60.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, str] | None = None,
    ) -> ChatResponse:
        payload: dict[str, object] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type(_RetryableError),
            reraise=True,
        ):
            with attempt:
                return await self._post_chat(payload, model)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def embed_batch(self, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, object] = {"model": model, "input": texts}
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type(_RetryableError),
            reraise=True,
        ):
            with attempt:
                return await self._post_embed(payload)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _post_chat(self, payload: dict[str, object], model: str) -> ChatResponse:
        start = time.perf_counter()
        try:
            resp = await self._client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise _RetryableError(str(exc), retryable=True) from exc

        if resp.status_code in (429, 500, 502, 503, 504):
            raise _RetryableError(
                f"http {resp.status_code}: {resp.text[:200]}",
                retryable=True,
                status=resp.status_code,
            )
        if resp.status_code >= 400:
            raise LLMError(
                f"http {resp.status_code}: {resp.text[:200]}",
                retryable=False,
                status=resp.status_code,
            )
        data = resp.json()
        usage = data.get("usage", {})
        return ChatResponse(
            content=data["choices"][0]["message"]["content"],
            model=model,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    async def _post_embed(self, payload: dict[str, object]) -> list[list[float]]:
        try:
            resp = await self._client.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise _RetryableError(str(exc), retryable=True) from exc

        if resp.status_code in (429, 500, 502, 503, 504):
            raise _RetryableError(
                f"http {resp.status_code}: {resp.text[:200]}",
                retryable=True,
                status=resp.status_code,
            )
        if resp.status_code >= 400:
            raise LLMError(
                f"http {resp.status_code}: {resp.text[:200]}",
                retryable=False,
                status=resp.status_code,
            )
        data = resp.json()
        return [list(item["embedding"]) for item in data.get("data", [])]
