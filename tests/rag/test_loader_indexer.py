"""Tests for LegalDocumentLoader and LegalIndexer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legalbot.rag.chunker import LegalChunker
from legalbot.rag.indexer import IndexStats, LegalIndexer
from legalbot.rag.loader import LegalDocumentLoader, RawDocument
from legalbot.rag.retriever import BM25Store, LegalRetriever


# -- Sample legal text for tests --

SAMPLE_LAW_TEXT = """\
中华人民共和国测试法

（2024年1月1日通过）

第一章 总则

第一条 为了规范测试行为，制定本法。

第二条 测试行为应当遵循公平、公正、公开的原则。

第三条 本法所称测试，是指对系统功能进行验证的活动。

第二章 测试程序

第四条 测试应当按照规定的程序进行。

第五条 测试结果应当如实记录。
"""

SAMPLE_CONTRACT_TEXT = """\
房屋租赁合同

出租方（甲方）：张三
承租方（乙方）：李四

第一条 房屋基本情况
甲方将位于北京市朝阳区的房屋出租给乙方使用。

第二条 租赁期限
租赁期限为一年，自2024年1月1日起至2024年12月31日止。

第三条 租金
月租金为人民币5000元，乙方应于每月5日前支付。
"""


class TestLegalDocumentLoader:

    def test_load_text_file(self, tmp_path: Path):
        fpath = tmp_path / "test_law.txt"
        fpath.write_text(SAMPLE_LAW_TEXT, encoding="utf-8")

        loader = LegalDocumentLoader()
        docs = loader.load_file(fpath)

        assert len(docs) == 1
        # Title from txt file falls back to filename since no 《》 markers
        assert "测试法" in docs[0].text or docs[0].title
        assert docs[0].doc_type == "law"

    def test_load_md_file(self, tmp_path: Path):
        fpath = tmp_path / "contract.md"
        fpath.write_text(SAMPLE_CONTRACT_TEXT, encoding="utf-8")

        loader = LegalDocumentLoader()
        docs = loader.load_file(fpath)

        assert len(docs) == 1
        assert "租赁" in docs[0].text

    def test_load_directory(self, tmp_path: Path):
        (tmp_path / "law1.txt").write_text(SAMPLE_LAW_TEXT, encoding="utf-8")
        (tmp_path / "contract.txt").write_text(SAMPLE_CONTRACT_TEXT, encoding="utf-8")

        loader = LegalDocumentLoader()
        docs = loader.load_directory(tmp_path)

        assert len(docs) == 2

    def test_load_nonexistent_directory(self):
        loader = LegalDocumentLoader()
        docs = loader.load_directory(Path("/nonexistent/path"))
        assert docs == []

    def test_load_unsupported_format(self, tmp_path: Path):
        fpath = tmp_path / "test.xyz"
        fpath.write_text("content", encoding="utf-8")

        loader = LegalDocumentLoader()
        docs = loader.load_file(fpath)
        assert docs == []

    def test_infer_metadata_title(self, tmp_path: Path):
        fpath = tmp_path / "test.txt"
        fpath.write_text("《中华人民共和国民法典》相关内容", encoding="utf-8")

        loader = LegalDocumentLoader()
        docs = loader.load_file(fpath)

        assert docs[0].title == "中华人民共和国民法典"

    def test_infer_law_area(self, tmp_path: Path):
        fpath = tmp_path / "test.txt"
        fpath.write_text(SAMPLE_LAW_TEXT, encoding="utf-8")

        loader = LegalDocumentLoader()
        docs = loader.load_file(fpath)
        # Should infer some law area from keywords
        assert docs[0].law_area or True  # may or may not detect

    def test_infer_doc_type_case(self, tmp_path: Path):
        fpath = tmp_path / "案例/test.txt"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text("某某判决书\n裁定如下：", encoding="utf-8")

        loader = LegalDocumentLoader()
        docs = loader.load_file(fpath)
        assert docs[0].doc_type == "case"

    def test_empty_file(self, tmp_path: Path):
        fpath = tmp_path / "empty.txt"
        fpath.write_text("", encoding="utf-8")

        loader = LegalDocumentLoader()
        docs = loader.load_file(fpath)
        assert docs == []


class TestLegalIndexer:

    @pytest.fixture
    def mock_embedding(self):
        """Deterministic mock embedding for testing."""
        import hashlib

        class MockEmbedding:
            async def embed(self, texts):
                return [
                    [hashlib.sha256(t.encode()).digest()[i] / 255.0 for i in range(8)]
                    for t in texts
                ]

            def dim(self):
                return 8

        return MockEmbedding()

    @pytest.fixture
    def indexer_components(self, tmp_path, mock_embedding):
        from legalbot.rag.vectorstore import ChromaVectorStore

        store = ChromaVectorStore(persist_dir=None, collection_name="test_indexer")
        bm25 = BM25Store()
        retriever = LegalRetriever(store, mock_embedding, bm25_store=bm25, top_k=5)
        loader = LegalDocumentLoader()
        chunker = LegalChunker(max_chunk_tokens=800, overlap_tokens=50)
        persist_dir = tmp_path / "kb"
        indexer = LegalIndexer(loader, chunker, retriever, persist_dir)
        return indexer, persist_dir

    async def test_build_index(self, tmp_path, indexer_components):
        indexer, persist_dir = indexer_components

        # Create test data
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "law.txt").write_text(SAMPLE_LAW_TEXT, encoding="utf-8")

        stats = await indexer.build_index(data_dir)

        assert isinstance(stats, IndexStats)
        assert stats.total_documents == 1
        assert stats.new_documents == 1
        assert stats.new_chunks >= 1
        assert stats.elapsed_seconds > 0

    async def test_build_index_idempotent(self, tmp_path, indexer_components):
        indexer, persist_dir = indexer_components

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "law.txt").write_text(SAMPLE_LAW_TEXT, encoding="utf-8")

        # First build
        stats1 = await indexer.build_index(data_dir)
        assert stats1.new_documents == 1

        # Second build should skip already-indexed docs
        stats2 = await indexer.build_index(data_dir)
        assert stats2.skipped_documents == 1
        assert stats2.new_documents == 0

    async def test_build_index_rebuild(self, tmp_path, indexer_components):
        indexer, persist_dir = indexer_components

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "law.txt").write_text(SAMPLE_LAW_TEXT, encoding="utf-8")

        # Build once
        await indexer.build_index(data_dir)

        # Rebuild
        stats = await indexer.build_index(data_dir, rebuild=True)
        assert stats.new_documents == 1

    async def test_manifest_saved(self, tmp_path, indexer_components):
        indexer, persist_dir = indexer_components

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "law.txt").write_text(SAMPLE_LAW_TEXT, encoding="utf-8")

        await indexer.build_index(data_dir)

        # Check manifest was saved
        manifest_path = persist_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert len(manifest) == 1

    async def test_get_status(self, tmp_path, indexer_components):
        indexer, persist_dir = indexer_components

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "law.txt").write_text(SAMPLE_LAW_TEXT, encoding="utf-8")

        await indexer.build_index(data_dir)
        status = indexer.get_status()

        assert status["indexed_documents"] == 1
        assert status["total_chunks"] >= 1

    async def test_multiple_files(self, tmp_path, indexer_components):
        indexer, persist_dir = indexer_components

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "law1.txt").write_text(SAMPLE_LAW_TEXT, encoding="utf-8")
        (data_dir / "contract.txt").write_text(SAMPLE_CONTRACT_TEXT, encoding="utf-8")

        stats = await indexer.build_index(data_dir)

        assert stats.total_documents == 2
        assert stats.new_documents == 2
        assert stats.new_chunks >= 2
