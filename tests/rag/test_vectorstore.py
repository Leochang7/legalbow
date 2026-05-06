"""Unit tests for ChromaVectorStore — using EphemeralClient (in-memory)."""

from __future__ import annotations

import uuid

import pytest

from legalbot.rag.vectorstore import ChromaVectorStore, SearchResult


def _unique_name() -> str:
    return f"test_{uuid.uuid4().hex[:8]}"


def _make_vectors(n: int, dim: int = 8) -> list[list[float]]:
    """Create n deterministic vectors of given dimension."""
    import hashlib

    vectors = []
    for i in range(n):
        h = hashlib.sha256(f"vec-{i}".encode()).digest()
        vec = [(h[j % len(h)] / 255.0 - 0.5) * 2 for j in range(dim)]
        vectors.append(vec)
    return vectors


class TestChromaVectorStore:

    async def test_add_and_search(self):
        store = ChromaVectorStore(persist_dir=None, collection_name=_unique_name())
        ids = ["c1", "c2", "c3"]
        vectors = _make_vectors(3)
        metadatas = [
            {"law_name": "民法典", "article_no": "第一条"},
            {"law_name": "民法典", "article_no": "第二条"},
            {"law_name": "刑法", "article_no": "第一条"},
        ]
        documents = ["第一条内容", "第二条内容", "刑法第一条内容"]

        await store.add(ids, vectors, metadatas, documents)
        results = await store.search(vectors[0], top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        # First result should be the query vector itself (most similar)
        assert results[0].id == "c1"

    async def test_search_with_filter(self):
        store = ChromaVectorStore(persist_dir=None, collection_name=_unique_name())
        ids = ["c1", "c2"]
        vectors = _make_vectors(2)
        metadatas = [
            {"law_name": "民法典", "law_area": "民法"},
            {"law_name": "刑法", "law_area": "刑法"},
        ]
        documents = ["民法内容", "刑法内容"]

        await store.add(ids, vectors, metadatas, documents)

        # Filter by law_area
        results = await store.search(
            vectors[0], top_k=2, filter={"law_area": "刑法"}
        )
        # Should only get the 刑法 result
        assert len(results) == 1
        assert results[0].id == "c2"

    async def test_delete(self):
        store = ChromaVectorStore(persist_dir=None, collection_name=_unique_name())
        ids = ["c1", "c2"]
        vectors = _make_vectors(2)
        metadatas = [{"law_name": "民法典"}, {"law_name": "刑法"}]
        documents = ["内容1", "内容2"]

        await store.add(ids, vectors, metadatas, documents)
        await store.delete(["c1"])

        results = await store.search(vectors[0], top_k=5)
        remaining_ids = [r.id for r in results]
        assert "c1" not in remaining_ids
        assert "c2" in remaining_ids

    async def test_empty_search(self):
        store = ChromaVectorStore(persist_dir=None, collection_name=_unique_name())
        query_vec = _make_vectors(1)[0]
        results = await store.search(query_vec, top_k=5)
        assert results == []

    async def test_none_metadata_values_filtered(self):
        """ChromaDB doesn't accept None values in metadata."""
        store = ChromaVectorStore(persist_dir=None, collection_name=_unique_name())
        ids = ["c1"]
        vectors = _make_vectors(1)
        metadatas = [{"law_name": "民法典", "chapter": None, "section": None}]
        documents = ["内容"]

        # Should not raise
        await store.add(ids, vectors, metadatas, documents)
        results = await store.search(vectors[0], top_k=1)
        assert len(results) == 1
        assert "chapter" not in results[0].metadata
