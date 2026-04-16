"""Unit tests for LegalRetriever — hybrid vector + BM25 retrieval."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from nanobot.rag.chunker import Chunk, ChunkMeta
from nanobot.rag.retriever import BM25Store, LegalRetriever, RetrievalResult
from nanobot.rag.vectorstore import ChromaVectorStore, SearchResult


def _make_chunk(id: str, text: str, **meta_kwargs) -> Chunk:
    return Chunk(id=id, text=text, metadata=ChunkMeta(**meta_kwargs))


class TestBM25Store:

    def test_add_and_search(self):
        store = BM25Store()
        chunks = [
            _make_chunk("c1", "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担违约责任", law_name="民法典", law_area="民法"),
            _make_chunk("c2", "抢劫公私财物的，处三年以上十年以下有期徒刑", law_name="刑法", law_area="刑法"),
            _make_chunk("c3", "用人单位应当按照劳动合同约定和国家规定支付劳动报酬", law_name="劳动合同法", law_area="劳动法"),
        ]
        store.add(chunks)
        results = store.search("违约责任 合同", top_k=2)

        assert len(results) <= 2
        # First result should be about 合同违约
        if results:
            assert results[0][0].id == "c1"

    def test_empty_search(self):
        store = BM25Store()
        results = store.search("查询", top_k=5)
        assert results == []

    def test_no_relevant_results(self):
        store = BM25Store()
        chunks = [_make_chunk("c1", "这是一条关于环境保护的规定", law_name="环保法")]
        store.add(chunks)
        results = store.search("金融证券投资", top_k=5)
        # Scores should be 0 or results should be empty for irrelevant query
        for chunk, score in results:
            assert score > 0  # BM25 returns only positive scores


class TestLegalRetriever:

    @pytest.fixture
    def mock_embedding(self):
        """Mock EmbeddingClient that returns deterministic vectors."""
        client = AsyncMock()
        # Return unit-like vectors; each text gets a unique but consistent vector
        async def mock_embed(texts):
            import hashlib
            vectors = []
            for t in texts:
                h = hashlib.sha256(t.encode()).digest()
                vec = [h[i % len(h)] / 255.0 for i in range(8)]
                vectors.append(vec)
            return vectors

        client.embed = mock_embed
        client.dim = AsyncMock(return_value=8)
        return client

    @pytest.fixture
    def vector_store(self):
        return ChromaVectorStore(persist_dir=None, collection_name=f"test_ret_{uuid.uuid4().hex[:8]}")

    async def test_vector_only_retrieve(self, mock_embedding, vector_store):
        bm25 = None  # No BM25
        retriever = LegalRetriever(vector_store, mock_embedding, bm25_store=bm25, top_k=3)

        chunks = [
            _make_chunk("c1", "合同违约金的规定", law_name="民法典", article_no="第五百八十五条"),
            _make_chunk("c2", "抢劫罪的刑罚规定", law_name="刑法", article_no="第二百六十三条"),
        ]
        await retriever.index(chunks)
        results = await retriever.retrieve("违约金 合同", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, RetrievalResult) for r in results)

    async def test_hybrid_retrieve(self, mock_embedding, vector_store):
        bm25 = BM25Store()
        retriever = LegalRetriever(vector_store, mock_embedding, bm25_store=bm25, top_k=3)

        chunks = [
            _make_chunk("c1", "当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金", law_name="民法典", article_no="第五百八十五条", law_area="民法"),
            _make_chunk("c2", "抢劫公私财物的处三年以上十年以下有期徒刑", law_name="刑法", article_no="第二百六十三条", law_area="刑法"),
            _make_chunk("c3", "用人单位应当自用工之日起一个月内订立书面劳动合同", law_name="劳动合同法", article_no="第十条", law_area="劳动法"),
        ]
        await retriever.index(chunks)
        results = await retriever.retrieve("违约金", top_k=2)

        # Should return results, ideally the 民法典 one first
        assert len(results) >= 1
        assert isinstance(results[0], RetrievalResult)

    async def test_filter_by_law_area(self, mock_embedding, vector_store):
        bm25 = BM25Store()
        retriever = LegalRetriever(vector_store, mock_embedding, bm25_store=bm25, top_k=5)

        chunks = [
            _make_chunk("c1", "合同违约金规定", law_name="民法典", law_area="民法"),
            _make_chunk("c2", "抢劫罪量刑标准", law_name="刑法", law_area="刑法"),
        ]
        await retriever.index(chunks)
        results = await retriever.retrieve("法律规定", law_area="民法", top_k=5)

        # All results should have law_area=民法
        for r in results:
            assert r.chunk.metadata.get("law_area") == "民法"


class TestRRFMerge:

    def test_rrf_merge_basic(self):
        vector_results = [
            SearchResult(id="c1", text="text1", metadata={"law_name": "法1"}, score=0.9),
            SearchResult(id="c2", text="text2", metadata={"law_name": "法2"}, score=0.8),
        ]
        bm25_results = [
            (_make_chunk("c2", "text2", law_name="法2"), 2.0),
            (_make_chunk("c3", "text3", law_name="法3"), 1.5),
        ]
        results = LegalRetriever._rrf_merge(vector_results, bm25_results)

        # c2 appears in both lists, should rank highest
        assert len(results) == 3
        assert results[0].chunk.id == "c2"

    def test_rrf_merge_empty_bm25(self):
        vector_results = [
            SearchResult(id="c1", text="text1", metadata={}, score=0.9),
        ]
        results = LegalRetriever._rrf_merge(vector_results, [])
        assert len(results) == 1
        assert results[0].chunk.id == "c1"
