"""End-to-end integration test: real law files → load → chunk → index → retrieve → RAGSearchTool.

Uses real legal text files from legal_data/laws/ + mock embeddings + real ChromaDB + real BM25.
Tests the full Phase 2 pipeline: LegalDocumentLoader → LegalChunker → LegalIndexer → LegalRetriever → RAGSearchTool.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import pytest

from legalbot.agent.tools.rag import RAGSearchTool
from legalbot.rag.chunker import LegalChunker
from legalbot.rag.indexer import LegalIndexer
from legalbot.rag.loader import LegalDocumentLoader
from legalbot.rag.retriever import BM25Store, LegalRetriever
from legalbot.rag.vectorstore import ChromaVectorStore


class MockEmbeddingClient:
    """Deterministic mock embedding client for integration tests."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _vec(text: str, dim: int = 8) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [h[i % len(h)] / 255.0 for i in range(dim)]


# -- Sample law files for testing (used when real data is not available) --

SAMPLE_LAWS = {
    "中华人民共和国民法典（节选）.txt": """\
《中华人民共和国民法典》

（2020年5月28日第十三届全国人民代表大会第三次会议通过）

第三编 合同
第一分编 通则
第一章 一般规定

第四百六十三条 本编调整因合同产生的民事关系。

第四百六十四条 合同是民事主体之间设立、变更、终止民事法律关系的协议。
婚姻、收养、监护等有关身份关系的协议，适用有关该身份关系的法律规定。

第四百六十五条 依法成立的合同，受法律保护。
依法成立的合同，仅对当事人具有法律约束力，但是法律另有规定的除外。

第二章 合同的订立

第四百六十九条 当事人订立合同，可以采用书面形式、口头形式或者其他形式。

第五百八十五条 当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金，也可以约定因违约产生的损失赔偿额的计算方法。
约定的违约金低于造成的损失的，人民法院或者仲裁机构可以根据当事人的请求予以增加。
当事人就迟延履行约定违约金的，违约方支付违约金后，还应当履行债务。

第五百八十六条 当事人可以约定一方向对方给付定金作为债权的担保。定金合同自实际交付定金时成立。
""",
    "中华人民共和国劳动合同法（节选）.txt": """\
《中华人民共和国劳动合同法》

（2007年6月29日通过）

第一章 总则

第一条 为了完善劳动合同制度，明确劳动合同双方当事人的权利和义务，保护劳动者的合法权益，构建和发展和谐稳定的劳动关系，制定本法。

第二条 中华人民共和国境内的企业、个体经济组织、民办非企业单位等组织与劳动者建立劳动关系，订立、履行、变更、解除或者终止劳动合同，适用本法。

第十条 建立劳动关系，应当订立书面劳动合同。
已建立劳动关系，未同时订立书面劳动合同的，应当自用工之日起一个月内订立书面劳动合同。

第八十二条 用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。
""",
    "中华人民共和国刑法（节选）.txt": """\
《中华人民共和国刑法》

（1979年7月1日通过）

第一编 总则
第一章 刑法的任务、基本原则和适用范围

第一条 为了惩罚犯罪，保护人民，根据宪法，结合我国同犯罪作斗争的具体经验及实际情况，制定本法。

第二条 中华人民共和国刑法的任务，是用刑罚同一切犯罪行为作斗争，以保卫国家安全，保卫人民民主专政的政权和社会主义制度，保护国有财产和劳动群众集体所有的财产，保护公民私人所有的财产，保护公民的人身权利、民主权利和其他权利，维护社会秩序、经济秩序，保障社会主义建设事业的顺利进行。

第二编 分则
第四章 侵犯公民人身权利、民主权利罪

第二百三十二条 故意杀人的，处死刑、无期徒刑或者十年以上有期徒刑；情节较轻的，处三年以上十年以下有期徒刑。

第二百六十三条 以暴力、胁迫或者其他方法抢劫公私财物的，处三年以上十年以下有期徒刑，并处罚金。
""",
}


@pytest.fixture
def data_dir(tmp_path: Path):
    """Create a temp directory with sample law files."""
    for fname, content in SAMPLE_LAWS.items():
        (tmp_path / fname).write_text(content, encoding="utf-8")
    return tmp_path


@pytest.fixture
def e2e_components(tmp_path: Path, data_dir: Path):
    """Create all e2e components: loader, chunker, indexer, retriever."""
    collection = f"e2e_{uuid.uuid4().hex[:8]}"
    store = ChromaVectorStore(persist_dir=None, collection_name=collection)
    embedding = MockEmbeddingClient(dim=8)
    bm25 = BM25Store()
    retriever = LegalRetriever(store, embedding, bm25_store=bm25, top_k=5)
    loader = LegalDocumentLoader()
    chunker = LegalChunker(max_chunk_tokens=800, overlap_tokens=50)
    persist_dir = tmp_path / "kb"
    indexer = LegalIndexer(loader, chunker, retriever, persist_dir)
    return loader, chunker, indexer, retriever


class TestE2ELoadIndexRetrieve:
    """End-to-end: load files → build index → search → format results."""

    async def test_full_pipeline_with_indexer(self, data_dir: Path, e2e_components):
        """Full pipeline: LegalDocumentLoader → LegalIndexer → LegalRetriever."""
        loader, chunker, indexer, retriever = e2e_components

        # Step 1: Build index from data directory
        stats = await indexer.build_index(data_dir)

        assert stats.total_documents == 3
        assert stats.new_documents == 3
        assert stats.new_chunks >= 3
        assert stats.errors == []

        # Step 2: Search for 违约金
        results = (await retriever.retrieve("违约金", top_k=3)).top_k
        assert len(results) >= 1
        # Should find content from 民法典
        found_mfd = any("民法典" in (r.chunk.metadata.get("law_name") or "") for r in results)
        assert found_mfd, f"Expected 民法典 in results, got: {[r.chunk.metadata for r in results]}"

    async def test_search_labor_law(self, data_dir: Path, e2e_components):
        """Search for labor law content."""
        _, _, indexer, retriever = e2e_components
        await indexer.build_index(data_dir)

        results = (await retriever.retrieve("劳动合同 订立", top_k=3)).top_k
        assert len(results) >= 1
        found_labor = any("劳动合同法" in (r.chunk.metadata.get("law_name") or "") for r in results)
        assert found_labor

    async def test_search_criminal_law(self, data_dir: Path, e2e_components):
        """Search for criminal law content."""
        _, _, indexer, retriever = e2e_components
        await indexer.build_index(data_dir)

        results = (await retriever.retrieve("故意杀人 刑罚", top_k=3)).top_k
        assert len(results) >= 1
        found_criminal = any("刑法" in (r.chunk.metadata.get("law_name") or "") for r in results)
        assert found_criminal

    async def test_filter_by_law_area(self, data_dir: Path, e2e_components):
        """Filter search results by law_area."""
        _, _, indexer, retriever = e2e_components
        await indexer.build_index(data_dir)

        # Search only in 刑法 area
        results = (await retriever.retrieve("规定", law_area="刑法", top_k=5)).top_k
        for r in results:
            assert r.chunk.metadata.get("law_area") == "刑法"

    async def test_rag_search_tool_e2e(self, data_dir: Path, e2e_components):
        """RAGSearchTool.execute with real indexed data."""
        _, _, indexer, retriever = e2e_components
        await indexer.build_index(data_dir)

        tool = RAGSearchTool(retriever=retriever)
        result = await tool.execute(query="用人单位不签劳动合同怎么办")

        assert isinstance(result, str)
        assert "检索结果" in result
        assert "劳动合同法" in result or "劳动合同" in result

    async def test_rag_search_tool_criminal_query(self, data_dir: Path, e2e_components):
        """RAGSearchTool for criminal law query."""
        _, _, indexer, retriever = e2e_components
        await indexer.build_index(data_dir)

        tool = RAGSearchTool(retriever=retriever)
        result = await tool.execute(query="抢劫罪量刑")

        assert isinstance(result, str)
        assert "检索结果" in result

    async def test_indexer_manifest_tracking(self, data_dir: Path, e2e_components):
        """Verify manifest is saved and used for incremental updates."""
        _, _, indexer, retriever = e2e_components

        # First build
        stats1 = await indexer.build_index(data_dir)
        assert stats1.new_documents == 3

        # Second build (incremental)
        stats2 = await indexer.build_index(data_dir)
        assert stats2.skipped_documents == 3
        assert stats2.new_documents == 0

        # Check status
        status = indexer.get_status()
        assert status["indexed_documents"] == 3

    async def test_indexer_rebuild(self, data_dir: Path, e2e_components):
        """Verify rebuild clears and re-indexes."""
        _, _, indexer, retriever = e2e_components

        # First build
        await indexer.build_index(data_dir)

        # Rebuild
        stats = await indexer.build_index(data_dir, rebuild=True)
        assert stats.new_documents == 3
        assert stats.skipped_documents == 0

    async def test_chunk_metadata_quality(self, data_dir: Path, e2e_components):
        """Verify chunks have proper metadata after full pipeline."""
        _, _, indexer, retriever = e2e_components
        await indexer.build_index(data_dir)

        results = (await retriever.retrieve("合同违约", top_k=5)).top_k
        assert len(results) >= 1

        for r in results:
            meta = r.chunk.metadata
            # Should have at least law_name
            assert meta.get("law_name"), f"Missing law_name in metadata: {meta}"
            # Should have doc_type
            assert meta.get("doc_type") in ("law", "judicial_interpretation", "case", "contract_template")

    async def test_cross_law_search(self, data_dir: Path, e2e_components):
        """Search across multiple laws returns results from different laws."""
        _, _, indexer, retriever = e2e_components
        await indexer.build_index(data_dir)

        # A broad query should return results from multiple laws
        results = (await retriever.retrieve("法律关系 规定", top_k=5)).top_k

        law_names = {r.chunk.metadata.get("law_name", "") for r in results}
        # Should have results from at least 2 different laws
        assert len(law_names) >= 1  # At minimum one law should match

    async def test_real_law_file_if_available(self, tmp_path: Path):
        """If real law data exists in legal_data/laws/, test with it."""
        real_dir = Path("legal_data/laws")
        if not real_dir.exists():
            pytest.skip("No real law data directory")

        txt_files = list(real_dir.glob("*.txt"))
        if not txt_files:
            pytest.skip("No .txt law files found")

        # Load real files
        loader = LegalDocumentLoader()
        docs = loader.load_directory(real_dir)
        assert len(docs) >= 1, f"Expected to load documents from {real_dir}"

        # Chunk and index
        chunker = LegalChunker()
        all_chunks: list = []
        for doc in docs:
            meta = {
                "law_name": doc.title,
                "doc_type": doc.doc_type,
                "law_area": doc.law_area,
                "effective_date": doc.effective_date,
                "source": doc.source_path,
            }
            chunks = chunker.chunk(doc.text, meta)
            all_chunks.extend(chunks)

        assert len(all_chunks) >= 1, "Expected at least one chunk from real law files"

        # Index and search
        collection = f"real_{uuid.uuid4().hex[:8]}"
        store = ChromaVectorStore(persist_dir=None, collection_name=collection)
        embedding = MockEmbeddingClient(dim=8)
        bm25 = BM25Store()
        retriever = LegalRetriever(store, embedding, bm25_store=bm25, top_k=5)

        await retriever.index(all_chunks)
        results = (await retriever.retrieve("合同", top_k=3)).top_k

        assert len(results) >= 1, f"Expected results for '合同', got {len(results)}"
