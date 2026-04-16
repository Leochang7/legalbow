"""Reranker module — DashScope qwen3-rerank / qwen3-vl-rerank."""

from __future__ import annotations

from typing import Any

from loguru import logger

from nanobot.rag.chunker import Chunk
from nanobot.rag.retriever import RetrievalResult


class Reranker:
    """Base reranker interface."""

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        ...


class DashScopeReranker(Reranker):
    """DashScope reranker using qwen3-rerank or qwen3-vl-rerank.

    API docs: https://help.aliyun.com/zh/model-studio/text-rerank-api
    """

    # Endpoint for qwen3-rerank (OpenAI-compatible)
    _COMPAT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    # Endpoint for qwen3-vl-rerank / gte-rerank-v2 (native)
    _NATIVE_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-rerank",
        top_n: int | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._top_n = top_n

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank candidates using DashScope rerank API."""
        if not candidates:
            return []

        import httpx

        documents = [c.chunk.text for c in candidates]

        # Use compatible API for qwen3-rerank, native API for others
        if self._model == "qwen3-rerank":
            return await self._rerank_compatible(query, candidates, documents, top_k)
        else:
            return await self._rerank_native(query, candidates, documents, top_k)

    async def _rerank_compatible(
        self,
        query: str,
        candidates: list[RetrievalResult],
        documents: list[str],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rerank using OpenAI-compatible endpoint (qwen3-rerank)."""
        import httpx

        body: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": self._top_n or top_k,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self._COMPAT_ENDPOINT, json=body, headers=headers)

        if resp.status_code != 200:
            logger.warning("DashScope rerank API error: {} {}", resp.status_code, resp.text[:200])
            return candidates[:top_k]

        data = resp.json()
        results = data.get("results", data.get("data", []))

        reranked: list[RetrievalResult] = []
        for item in results[:top_k]:
            idx = item.get("index", 0)
            if 0 <= idx < len(candidates):
                score = item.get("relevance_score", candidates[idx].score)
                reranked.append(RetrievalResult(
                    chunk=candidates[idx].chunk,
                    score=score,
                ))

        logger.debug("DashScope rerank: {} candidates → {} results", len(candidates), len(reranked))
        return reranked if reranked else candidates[:top_k]

    async def _rerank_native(
        self,
        query: str,
        candidates: list[RetrievalResult],
        documents: list[str],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rerank using DashScope native endpoint (qwen3-vl-rerank, gte-rerank-v2)."""
        import httpx

        body: dict[str, Any] = {
            "model": self._model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "return_documents": False,
                "top_n": self._top_n or top_k,
            },
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self._NATIVE_ENDPOINT, json=body, headers=headers)

        if resp.status_code != 200:
            logger.warning("DashScope rerank API error: {} {}", resp.status_code, resp.text[:200])
            return candidates[:top_k]

        data = resp.json()
        output = data.get("output", {})
        results = output.get("results", [])

        reranked: list[RetrievalResult] = []
        for item in results[:top_k]:
            idx = item.get("index", 0)
            if 0 <= idx < len(candidates):
                score = item.get("relevance_score", candidates[idx].score)
                reranked.append(RetrievalResult(
                    chunk=candidates[idx].chunk,
                    score=score,
                ))

        logger.debug("DashScope rerank: {} candidates → {} results", len(candidates), len(reranked))
        return reranked if reranked else candidates[:top_k]
