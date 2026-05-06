"""End-to-end integration tests using DashScope embedding + reranker + ChromaDB.

These tests use DashScope (阿里云百炼) text-embedding-v4 and qwen3-vl-rerank.
Tests are skipped automatically if DashScope API is unavailable.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from legalbot.rag.chunker import Chunk, LegalChunker
from legalbot.rag.embedding import EmbeddingClient
from legalbot.rag.loader import LegalDocumentLoader
from legalbot.rag.reranker import DashScopeReranker
from legalbot.rag.retriever import BM25Store, LegalRetriever, RetrievalResult
from legalbot.rag.vectorstore import ChromaVectorStore

# ---------------------------------------------------------------------------
# DashScope availability check
# ---------------------------------------------------------------------------

_DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-67e11d47e65b41fc86a546c6446c9c20")
_DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

def _embedding_available() -> bool:
    """Check if DashScope embedding API is reachable."""
    try:
        import httpx
        r = httpx.post(
            _DASHSCOPE_BASE + "/embeddings",
            headers={"Authorization": f"Bearer {_DASHSCOPE_KEY}"},
            json={"model": "text-embedding-v4", "input": ["test"]},
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


requires_embedding = pytest.mark.skipif(
    not _embedding_available(),
    reason="DashScope embedding API not available",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def embedding_client():
    return EmbeddingClient(
        model=EMBEDDING_MODEL,
        api_key=_DASHSCOPE_KEY,
        api_base=_DASHSCOPE_BASE,
        dim=EMBEDDING_DIM,
    )


@pytest.fixture
def reranker():
    return DashScopeReranker(
        api_key=_DASHSCOPE_KEY,
        model="qwen3-vl-rerank",
    )


@pytest.fixture
def vector_store():
    """In-memory ChromaDB for testing."""
    return ChromaVectorStore(
        persist_dir=None,
        collection_name=f"test_e2e_{uuid.uuid4().hex[:8]}",
    )


@pytest.fixture
def chunker():
    return LegalChunker(max_chunk_tokens=800, overlap_tokens=100)


@pytest.fixture
def retriever(embedding_client, vector_store):
    return LegalRetriever(
        vector_store=vector_store,
        embedding_client=embedding_client,
        bm25_store=BM25Store(),
        top_k=5,
    )


@pytest.fixture
def retriever_with_reranker(embedding_client, vector_store, reranker):
    return LegalRetriever(
        vector_store=vector_store,
        embedding_client=embedding_client,
        bm25_store=BM25Store(),
        reranker=reranker,
        top_k=5,
    )


# ---------------------------------------------------------------------------
# Test: Embedding connectivity
# ---------------------------------------------------------------------------

@requires_embedding
@pytest.mark.asyncio
async def test_dashscope_embedding_basic(embedding_client):
    """Verify DashScope embedding works with Chinese text."""
    vectors = await embedding_client.embed(["劳动合同", "用人单位应当签订书面劳动合同"])
    assert len(vectors) == 2
    assert len(vectors[0]) == EMBEDDING_DIM
    assert any(v != 0.0 for v in vectors[0])


@requires_embedding
@pytest.mark.asyncio
async def test_dashscope_embedding_batch(embedding_client):
    """Verify batch embedding works."""
    texts = [f"测试文本{i}" for i in range(5)]
    vectors = await embedding_client.embed(texts)
    assert len(vectors) == 5
    for v in vectors:
        assert len(v) == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Test: Reranker
# ---------------------------------------------------------------------------

@requires_embedding
@pytest.mark.asyncio
async def test_reranker_basic(reranker):
    """DashScope reranker should reorder results by relevance."""
    candidates = [
        RetrievalResult(chunk=Chunk(id='1', text='用人单位应当签订书面劳动合同', metadata={'law_name': '劳动合同法'}), score=0.3),
        RetrievalResult(chunk=Chunk(id='2', text='民法典关于合同违约金的规定', metadata={'law_name': '民法典'}), score=0.5),
        RetrievalResult(chunk=Chunk(id='3', text='天气预报明天晴天', metadata={'law_name': '无关'}), score=0.4),
    ]

    results = await reranker.rerank('劳动合同签订', candidates, top_k=2)
    assert len(results) == 2
    # Most relevant should be 劳动合同法
    assert "劳动合同" in results[0].chunk.text
    assert results[0].score > results[1].score


@requires_embedding
@pytest.mark.asyncio
async def test_reranker_filters_irrelevant(reranker):
    """Reranker should push irrelevant documents down."""
    candidates = [
        RetrievalResult(chunk=Chunk(id='1', text='中华人民共和国劳动合同法第八十二条', metadata={'law_name': '劳动合同法'}), score=0.3),
        RetrievalResult(chunk=Chunk(id='2', text='今日股市行情分析报告', metadata={'law_name': '无关'}), score=0.6),
    ]

    results = await reranker.rerank('劳动合同法赔偿标准', candidates, top_k=2)
    # Legal text should rank higher than stock report
    assert "劳动合同" in results[0].chunk.text


# ---------------------------------------------------------------------------
# Test: Index + Retrieve with real embedding
# ---------------------------------------------------------------------------

@requires_embedding
@pytest.mark.asyncio
async def test_index_and_retrieve_labor_law(retriever, chunker):
    """Index labor law chunks and retrieve by query."""
    labor_law_text = """
中华人民共和国劳动合同法

第一章 总则

第一条 为了完善劳动合同制度，明确劳动合同双方当事人的权利和义务，保护劳动者的合法权益，构建和发展和谐稳定的劳动关系，制定本法。

第十条 建立劳动关系，应当订立书面劳动合同。已建立劳动关系，未同时订立书面劳动合同的，应当自用工之日起一个月内订立书面劳动合同。

第八十二条 用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。

第八十五条 用人单位未按照劳动合同的约定或者国家规定及时足额支付劳动者劳动报酬的，由劳动行政部门责令限期支付劳动报酬。
"""
    chunks = chunker.chunk(labor_law_text, {
        "law_name": "中华人民共和国劳动合同法",
        "law_area": "劳动法",
        "doc_type": "law",
        "source": "test",
    })
    assert len(chunks) > 0

    await retriever.index(chunks)

    results = (await retriever.retrieve("用人单位不签劳动合同怎么办")).top_k
    assert len(results) > 0
    top_texts = " ".join(r.chunk.text for r in results[:3])
    assert "劳动合同" in top_texts


@requires_embedding
@pytest.mark.asyncio
async def test_index_and_retrieve_with_reranker(retriever_with_reranker, chunker):
    """Retrieve with reranker should produce better-ordered results."""
    labor_text = """
中华人民共和国劳动合同法
第十条 建立劳动关系，应当订立书面劳动合同。
第八十二条 用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。
"""
    civil_text = """
中华人民共和国民法典
第五百八十五条 违约金条款的规定。
第一百一十一条 自然人的个人信息受法律保护。
"""
    all_chunks = []
    for text, law_name, law_area in [
        (labor_text, "劳动合同法", "劳动法"),
        (civil_text, "民法典", "民法"),
    ]:
        chunks = chunker.chunk(text, {
            "law_name": law_name,
            "law_area": law_area,
            "doc_type": "law",
            "source": "test",
        })
        all_chunks.extend(chunks)

    await retriever_with_reranker.index(all_chunks)

    results = (await retriever_with_reranker.retrieve("用人单位不签劳动合同")).top_k
    assert len(results) > 0
    # Top result should be about 劳动合同
    assert "劳动合同" in results[0].chunk.text or "用人单位" in results[0].chunk.text


@requires_embedding
@pytest.mark.asyncio
async def test_hybrid_search_improves_keyword_match(retriever, chunker):
    """BM25 should boost exact keyword matches like legal terms."""
    texts = {
        "民法典": "中华人民共和国民法典\n第一百一十一条 自然人的个人信息受法律保护。任何组织或者个人需要获取他人个人信息的，应当依法取得并确保信息安全。",
        "刑法": "中华人民共和国刑法\n第二百五十三条之一 违反国家有关规定，向他人出售或者提供公民个人信息，情节严重的，处三年以下有期徒刑或者拘役，并处或者单处罚金。",
        "劳动合同法": "中华人民共和国劳动合同法\n第八十二条 用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。",
    }

    all_chunks = []
    for law_name, text in texts.items():
        law_area = "民法" if "民法典" in law_name else ("刑法" if "刑法" in law_name else "劳动法")
        chunks = chunker.chunk(text, {
            "law_name": law_name,
            "law_area": law_area,
            "doc_type": "law",
            "source": "test",
        })
        all_chunks.extend(chunks)

    await retriever.index(all_chunks)

    results = (await retriever.retrieve("个人信息保护")).top_k
    assert len(results) > 0
    top_law = results[0].chunk.metadata.get("law_name", "")
    assert "民法典" in top_law or "刑法" in top_law


@requires_embedding
@pytest.mark.asyncio
async def test_filter_by_law_area(retriever, chunker):
    """Filter search results by law_area metadata."""
    labor_chunks = chunker.chunk("中华人民共和国劳动合同法\n第十条 建立劳动关系，应当订立书面劳动合同。", {
        "law_name": "劳动合同法", "law_area": "劳动法", "doc_type": "law", "source": "test",
    })
    await retriever.index(labor_chunks)

    civil_chunks = chunker.chunk("中华人民共和国民法典\n第五百八十五条 违约金条款的规定。", {
        "law_name": "民法典", "law_area": "民法", "doc_type": "law", "source": "test",
    })
    await retriever.index(civil_chunks)

    results = (await retriever.retrieve("劳动合同", law_area="劳动法")).top_k
    assert len(results) > 0
    for r in results:
        assert r.chunk.metadata.get("law_area") == "劳动法"


# ---------------------------------------------------------------------------
# Test: RAGSearchTool with real embedding
# ---------------------------------------------------------------------------

@requires_embedding
@pytest.mark.asyncio
async def test_rag_search_tool_with_real_embedding(retriever, chunker):
    """RAGSearchTool should return formatted results with real embedding."""
    from legalbot.agent.tools.rag import RAGSearchTool

    chunks = chunker.chunk("中华人民共和国劳动合同法\n第八十二条 用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。", {
        "law_name": "劳动合同法", "law_area": "劳动法", "doc_type": "law", "source": "test",
    })
    await retriever.index(chunks)

    tool = RAGSearchTool(retriever=retriever)
    result = await tool.execute(query="用人单位不签劳动合同怎么办")
    assert "劳动合同法" in result
    assert "检索结果" in result


# ---------------------------------------------------------------------------
# Test: Real law file pipeline
# ---------------------------------------------------------------------------

@requires_embedding
@pytest.mark.asyncio
async def test_real_law_file_pipeline(embedding_client):
    """Load, chunk, index, and retrieve from real law files."""
    laws_dir = Path("legal_data/laws")
    if not laws_dir.exists():
        pytest.skip("No legal_data/laws directory")

    vector_store = ChromaVectorStore(
        persist_dir=None,
        collection_name=f"test_real_{uuid.uuid4().hex[:8]}",
    )
    retriever = LegalRetriever(
        vector_store=vector_store,
        embedding_client=embedding_client,
        bm25_store=BM25Store(),
        top_k=5,
    )
    loader = LegalDocumentLoader()
    chunker = LegalChunker()

    docs = loader.load_directory(laws_dir)
    if not docs:
        pytest.skip("No documents loaded")

    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc.text, {
            "law_name": doc.title,
            "law_area": doc.law_area,
            "doc_type": doc.doc_type,
            "source": doc.source_path,
        })
        all_chunks.extend(chunks)

    assert len(all_chunks) > 0
    await retriever.index(all_chunks)

    results = (await retriever.retrieve("用人单位不签劳动合同怎么办")).top_k
    assert len(results) > 0
    top_text = results[0].chunk.text
    assert "劳动合同" in top_text or "用人单位" in top_text

    for r in results:
        assert r.chunk.metadata.get("law_name", "") != ""
        assert r.score > 0


# ---------------------------------------------------------------------------
# Test: Orchestrator with real RAG
# ---------------------------------------------------------------------------

@requires_embedding
@pytest.mark.asyncio
async def test_orchestrator_with_real_rag(retriever, chunker):
    """LegalOrchestrator should use real RAG results."""
    from legalbot.agent.orchestrator import LegalOrchestrator, INTENT_LEGAL_QUERY
    from legalbot.config.schema import AgentDefConfig, OrchestrateConfig

    chunks = chunker.chunk("中华人民共和国劳动合同法\n第八十二条 用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。", {
        "law_name": "劳动合同法", "law_area": "劳动法", "doc_type": "law", "source": "test",
    })
    await retriever.index(chunks)

    from dataclasses import dataclass as _dc
    provider = MagicMock()

    @_dc
    class FakeResponse:
        content: str | None = "legal_query"
        tool_calls: list = None
        finish_reason: str = "stop"
        usage: dict = None

        def __post_init__(self):
            if self.tool_calls is None: self.tool_calls = []
            if self.usage is None: self.usage = {}

    provider.chat = AsyncMock(return_value=FakeResponse(content="legal_query"))

    config = OrchestrateConfig(
        enable=True,
        agents={"legal_research": AgentDefConfig(system_prompt="法律检索专家", tools=["legal_rag_search"])},
    )

    from legalbot.agent.subagent import SubagentManager
    from legalbot.bus.queue import MessageBus
    bus = MessageBus()
    subagent_mgr = SubagentManager(
        provider=provider,
        workspace=Path(tempfile.gettempdir()) / "test_workspace",
        bus=bus,
        max_tool_result_chars=16000,
    )

    from legalbot.agent.tools.rag import RAGSearchTool
    rag_tool = RAGSearchTool(retriever=retriever)

    orch = LegalOrchestrator(provider, subagent_mgr, config, main_tools={"legal_rag_search": rag_tool})

    intent = await orch.classify_intent("用人单位不签劳动合同怎么办")
    assert intent == INTENT_LEGAL_QUERY


# ---------------------------------------------------------------------------
# Test: Dimension consistency
# ---------------------------------------------------------------------------

@requires_embedding
@pytest.mark.asyncio
async def test_embedding_dimension_consistency(embedding_client, vector_store):
    """Verify embedding dimension matches between client and vector store."""
    dim = embedding_client.dim()
    assert dim == EMBEDDING_DIM

    vectors = await embedding_client.embed(["测试文本"])
    assert len(vectors[0]) == dim

    await vector_store.add(ids=["test_1"], vectors=vectors, metadatas=[{"law_name": "测试"}], documents=["测试文本"])

    results = await vector_store.search(vectors[0], top_k=1)
    assert len(results) == 1
    assert results[0].id == "test_1"
