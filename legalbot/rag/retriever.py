"""Hybrid retriever for legal RAG — vector search + BM25 with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from legalbot.rag.chunker import Chunk
from legalbot.rag.embedding import EmbeddingClient
from legalbot.rag.vectorstore import ChromaVectorStore, SearchResult, VectorStore


@dataclass
class RetrievalResult:
    """A single retrieval result with scored chunk."""

    chunk: Chunk
    score: float


@dataclass
class RetrievalPipelineResult:
    """Result of a full retrieval pipeline call.

    Attributes:
        rrf_candidates: All RRF-merged candidates (before rerank), up to 60.
        top_k: Final reranked top-k results.
    """

    rrf_candidates: list[RetrievalResult]
    top_k: list[RetrievalResult]


class BM25Store:
    """BM25 keyword search store using jieba tokenization + rank_bm25.

    Uses lazy initialization: the BM25 index is only built on first search,
    and rebuilt only when new chunks are added since the last search.
    """

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: Any = None
        self._dirty: bool = False  # True when index needs rebuild

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the BM25 store. Index is rebuilt lazily on next search."""
        import jieba

        self._chunks.extend(chunks)
        new_tokenized = [list(jieba.cut(c.text)) for c in chunks]
        self._tokenized_corpus.extend(new_tokenized)
        self._dirty = True  # Mark index as needing rebuild

    def _ensure_index(self) -> None:
        """Build or rebuild the BM25 index if dirty."""
        if not self._dirty or not self._tokenized_corpus:
            return
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._dirty = False

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Search BM25 index, returning (chunk, score) pairs."""
        self._ensure_index()
        if not self._bm25 or not self._chunks:
            return []
        import jieba

        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)
        # Get top_k indices
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in indexed[:top_k]:
            if score > 0:
                results.append((self._chunks[idx], float(score)))
        return results


class LegalRetriever:
    """Hybrid retriever: vector search + BM25 with Reciprocal Rank Fusion + optional reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
        bm25_store: BM25Store | None = None,
        reranker: Any = None,
        top_k: int = 5,
    ):
        self._vector_store = vector_store
        self._embedding_client = embedding_client
        self._bm25_store = bm25_store
        self._reranker = reranker
        self._top_k = top_k

    async def index(self, chunks: list[Chunk]) -> None:
        """Build index: embed chunks and add to vector store + BM25 store."""
        if not chunks:
            return

        # Embed all chunk texts
        texts = [c.text for c in chunks]
        vectors = await self._embedding_client.embed(texts)

        # Add to vector store
        ids = [c.id for c in chunks]
        metadatas = [dict(c.metadata) for c in chunks]
        await self._vector_store.add(ids, vectors, metadatas, texts)

        # Add to BM25 store
        if self._bm25_store is not None:
            self._bm25_store.add(chunks)

        logger.info("Indexed {} chunks (vector + BM25)", len(chunks))

    async def retrieve(
        self,
        query: str,
        law_area: str | None = None,
        doc_type: str | None = None,
        top_k: int | None = None,
    ) -> RetrievalPipelineResult:
        """Retrieve relevant chunks using hybrid search.

        Flow:
        1. Vector search: top_k * 3 candidates
        2. BM25 search: top_k * 3 candidates
        3. Merge with Reciprocal Rank Fusion (RRF) → rrf_candidates (up to 60)
        4. Rerank if reranker is available → top_k

        Returns:
            RetrievalPipelineResult with both rrf_candidates (pre-rerank) and top_k (post-rerank).
        """
        k = top_k or self._top_k
        expand = k * 3

        # Build filter for vector store
        vfilter: dict[str, Any] | None = None
        if law_area or doc_type:
            conditions = []
            if law_area:
                conditions.append({"law_area": law_area})
            if doc_type:
                conditions.append({"doc_type": doc_type})
            if len(conditions) == 1:
                vfilter = conditions[0]
            else:
                vfilter = {"$and": conditions}

        # Step 1: Vector search
        query_vecs = await self._embedding_client.embed([query])
        vector_results = await self._vector_store.search(
            query_vecs[0], top_k=expand, filter=vfilter
        )

        # Step 2: BM25 search
        bm25_results: list[tuple[Chunk, float]] = []
        if self._bm25_store is not None:
            bm25_results = self._bm25_store.search(query, top_k=expand)
            # Apply filter to BM25 results
            if law_area or doc_type:
                bm25_results = [
                    (c, s) for c, s in bm25_results
                    if (not law_area or c.metadata.get("law_area") == law_area)
                    and (not doc_type or c.metadata.get("doc_type") == doc_type)
                ]

        # Step 3: Merge with RRF
        rrf_candidates = self._rrf_merge(vector_results, bm25_results)

        # Step 4: Rerank if reranker is available
        final = rrf_candidates
        if self._reranker is not None:
            final = await self._reranker.rerank(query, final, top_k=k)

        # Step 5: Return both rrf_candidates and top_k
        return RetrievalPipelineResult(
            rrf_candidates=rrf_candidates,
            top_k=final[:k],
        )

    @staticmethod
    def _rrf_merge(
        vector_results: list[SearchResult],
        bm25_results: list[tuple[Chunk, float]],
        rrf_k: int = 60,
    ) -> list[RetrievalResult]:
        """Merge vector and BM25 results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each result list.
        """
        scores: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}

        # Vector results
        for rank, sr in enumerate(vector_results):
            doc_id = sr.id
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if doc_id not in chunks_by_id:
                chunks_by_id[doc_id] = Chunk(
                    id=sr.id, text=sr.text, metadata=sr.metadata
                )

        # BM25 results
        for rank, (chunk, _score) in enumerate(bm25_results):
            doc_id = chunk.id
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if doc_id not in chunks_by_id:
                chunks_by_id[doc_id] = chunk

        # Sort by RRF score descending
        sorted_ids = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
        return [
            RetrievalResult(chunk=chunks_by_id[doc_id], score=scores[doc_id])
            for doc_id in sorted_ids
        ]
