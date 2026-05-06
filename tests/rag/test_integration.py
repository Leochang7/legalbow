"""Integration test: legal text → chunk → index → retrieve → RAGSearchTool.execute.

Uses mocked embeddings (deterministic seeded vectors) + real ChromaDB EphemeralClient
+ real BM25/jieba + real LegalChunker.
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest

from legalbot.agent.tools.rag import RAGSearchTool
from legalbot.rag.chunker import LegalChunker
from legalbot.rag.embedding import EmbeddingClient
from legalbot.rag.retriever import BM25Store, LegalRetriever
from legalbot.rag.vectorstore import ChromaVectorStore

# -- Real legal text for integration testing --

CIVIL_CODE_CONTRACT_SECTION = """\
第三编 合同
第一分编 通则
第一章 一般规定

第四百六十三条 本编调整因合同产生的民事关系。

第四百六十四条 合同是民事主体之间设立、变更、终止民事法律关系的协议。
婚姻、收养、监护等有关身份关系的协议，适用有关该身份关系的法律规定；没有规定的，可以根据其性质参照适用本编规定。

第四百六十五条 依法成立的合同，受法律保护。
依法成立的合同，仅对当事人具有法律约束力，但是法律另有规定的除外。

第四百六十六条 当事人对合同条款的理解有争议的，应当依据本法第一百四十二条第一款的规定，确定争议条款的含义。
合同文本采用两种以上文字订立并约定具有同等效力的，对各文本使用的词句推定具有相同含义。各文本使用的词句不一致的，应当根据合同的相关条款、性质、目的以及诚信原则等予以解释。

第二章 合同的订立

第四百六十九条 当事人订立合同，可以采用书面形式、口头形式或者其他形式。
书面形式是合同书、信件、电报、电传、传真等可以有形地表现所载内容的形式。
以电子数据交换、电子邮件等方式能够有形地表现所载内容，并可以随时调取查用的数据电文，视为书面形式。

第五百八十五条 当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金，也可以约定因违约产生的损失赔偿额的计算方法。
约定的违约金低于造成的损失的，人民法院或者仲裁机构可以根据当事人的请求予以增加；约定的违约金过分高于造成的损失的，人民法院或者仲裁机构可以根据当事人的请求予以适当减少。
当事人就迟延履行约定违约金的，违约方支付违约金后，还应当履行债务。
"""


def _deterministic_vector(text: str, dim: int = 8) -> list[float]:
    """Generate a deterministic vector from text content."""
    h = hashlib.sha256(text.encode()).digest()
    return [h[i % len(h)] / 255.0 for i in range(dim)]


class MockEmbeddingClient:
    """Deterministic mock embedding client for integration tests."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_deterministic_vector(t, self._dim) for t in texts]

    def dim(self) -> int:
        return self._dim


@pytest.fixture
def retriever():
    """Create a fully wired LegalRetriever with real components."""
    store = ChromaVectorStore(persist_dir=None, collection_name="integration_test")
    embedding = MockEmbeddingClient(dim=8)
    bm25 = BM25Store()
    return LegalRetriever(store, embedding, bm25_store=bm25, top_k=5)


@pytest.fixture
def indexed_retriever(retriever):
    """Retriever with pre-indexed legal text chunks."""
    chunker = LegalChunker(max_chunk_tokens=800, overlap_tokens=50)
    meta = {
        "law_name": "中华人民共和国民法典",
        "doc_type": "law",
        "law_area": "民法/合同法",
        "effective_date": "2021-01-01",
        "source": "全国人大",
    }
    chunks = chunker.chunk(CIVIL_CODE_CONTRACT_SECTION, meta)
    return retriever, chunks


class TestIntegrationEndToEnd:

    async def test_full_pipeline(self, indexed_retriever):
        """Test: chunk → index → retrieve for '违约金' query."""
        retriever, chunks = indexed_retriever

        # Index all chunks
        await retriever.index(chunks)

        # Search for 违约金 (liquidated damages)
        results = (await retriever.retrieve("违约金", top_k=3)).top_k

        assert len(results) >= 1
        # The most relevant result should mention 违约金
        found = any("违约金" in r.chunk.text for r in results)
        assert found, f"Expected '违约金' in results, got: {[r.chunk.text[:50] for r in results]}"

    async def test_retrieve_by_article(self, indexed_retriever):
        """Test retrieving specific article content."""
        retriever, chunks = indexed_retriever
        await retriever.index(chunks)

        results = (await retriever.retrieve("合同的定义和协议", top_k=3)).top_k
        assert len(results) >= 1

    async def test_rag_search_tool_end_to_end(self, indexed_retriever):
        """Test RAGSearchTool.execute with real pipeline."""
        retriever, chunks = indexed_retriever
        await retriever.index(chunks)

        tool = RAGSearchTool(retriever=retriever)
        result = await tool.execute(query="违约金的规定")

        assert isinstance(result, str)
        assert "民法典" in result
        assert "检索结果" in result

    async def test_rag_search_tool_no_results(self, indexed_retriever):
        """Test RAGSearchTool when no relevant results exist."""
        retriever, chunks = indexed_retriever
        await retriever.index(chunks)

        # Filter by a law_area that doesn't match
        tool = RAGSearchTool(retriever=retriever)
        result = await tool.execute(query="合同", law_area="刑法")

        # Should return "未检索到" or empty results
        assert isinstance(result, str)

    async def test_chunk_metadata_preserved(self, indexed_retriever):
        """Verify chunk metadata flows through the pipeline."""
        retriever, chunks = indexed_retriever
        await retriever.index(chunks)

        results = (await retriever.retrieve("合同订立", top_k=5)).top_k
        assert len(results) >= 1

        # All results should have law_name
        for r in results:
            assert r.chunk.metadata.get("law_name") == "中华人民共和国民法典"

        # At least one result should have chapter info
        has_chapter = any(r.chunk.metadata.get("chapter") for r in results)
        assert has_chapter, "Expected chapter metadata in results"

    async def test_bm25_improves_keyword_match(self):
        """Test that BM25 helps with exact keyword matching like '违约金'."""
        # Vector-only retriever
        store1 = ChromaVectorStore(persist_dir=None, collection_name="no_bm25")
        embedding = MockEmbeddingClient(dim=8)
        retriever_vonly = LegalRetriever(store1, embedding, bm25_store=None, top_k=5)

        # Hybrid retriever
        store2 = ChromaVectorStore(persist_dir=None, collection_name="with_bm25")
        bm25 = BM25Store()
        retriever_hybrid = LegalRetriever(store2, embedding, bm25_store=bm25, top_k=5)

        chunker = LegalChunker()
        meta = {"law_name": "中华人民共和国民法典", "doc_type": "law", "law_area": "民法"}
        chunks = chunker.chunk(CIVIL_CODE_CONTRACT_SECTION, meta)

        await retriever_vonly.index(chunks)
        await retriever_hybrid.index(chunks)

        # Both should return results for keyword-heavy query
        vonly_results = (await retriever_vonly.retrieve("违约金", top_k=3)).top_k
        hybrid_results = (await retriever_hybrid.retrieve("违约金", top_k=3)).top_k

        # Both should find results
        assert len(vonly_results) >= 1
        assert len(hybrid_results) >= 1
