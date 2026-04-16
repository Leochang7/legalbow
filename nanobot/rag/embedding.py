"""Embedding client for RAG — OpenAI-compatible API only (MVP)."""

from __future__ import annotations

from typing import Any

from loguru import logger


class EmbeddingClient:
    """Remote embedding client using any OpenAI-compatible API."""

    # Known model dimensions
    _MODEL_DIMS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "text-embedding-v3": 1024,  # DashScope
        "embedding-3": 2048,  # Zhipu
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        api_base: str = "",
        extra_headers: dict[str, str] | None = None,
        dim: int | None = None,
    ):
        self._model = model
        self._dim = dim or self._MODEL_DIMS.get(model, 1536)
        self._client_kwargs: dict[str, Any] = {}
        if api_key:
            self._client_kwargs["api_key"] = api_key
        if api_base:
            self._client_kwargs["base_url"] = api_base
        if extra_headers:
            self._client_kwargs["default_headers"] = extra_headers
        self._client: Any = None  # lazy init

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(**self._client_kwargs)
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning vectors."""
        if not texts:
            return []
        client = self._get_client()
        # OpenAI API has a max batch size; chunk if needed
        batch_size = 100
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = await client.embeddings.create(model=self._model, input=batch)
            for item in resp.data:
                all_vectors.append(item.embedding)
        logger.debug("Embedded {} texts (model={})", len(texts), self._model)
        return all_vectors

    def dim(self) -> int:
        """Return the embedding dimension for the configured model."""
        return self._dim
