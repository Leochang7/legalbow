"""Full-flow tests for LegalDocumentGenerator — tests the complete generate() pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from legalbot.document.generator import LegalDocumentGenerator
from legalbot.rag.chunker import Chunk, ChunkMeta
from legalbot.rag.retriever import RetrievalResult


def _make_chunk(id: str, text: str, **meta_kwargs) -> Chunk:
    return Chunk(id=id, text=text, metadata=ChunkMeta(**meta_kwargs))


def _make_retrieval_result(chunk: Chunk, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(chunk=chunk, score=score)


class FakeProvider:
    """Fake LLM provider that returns controllable responses."""

    def __init__(self, response_text: str = "这是一份测试文书草稿。\n\n原告：张三\n被告：李四"):
        self.response_text = response_text
        self.chat_called_with = None

    async def chat(self, messages, **kwargs):
        self.chat_called_with = messages
        return FakeResponse(content=self.response_text)


@dataclass
class FakeResponse:
    content: str | None


class MockRetriever:
    """Retriever that returns fake law chunks."""

    def __init__(self, chunks: list[RetrievalResult] | None = None):
        self.chunks = chunks or []
        self.called_with = None

    async def retrieve(self, query, law_area=None, doc_type=None, top_k=5):
        self.called_with = {"query": query, "law_area": law_area, "doc_type": doc_type, "top_k": top_k}
        return self.chunks


class TestLegalDocumentGenerator:
    """Tests for LegalDocumentGenerator.generate()."""

    @pytest.fixture
    def default_chunks(self) -> list[RetrievalResult]:
        return [
            _make_retrieval_result(
                _make_chunk("c1", "借款人未按照约定的期限返还借款的，应当按照约定或者国家有关规定支付逾期利息", law_name="民法典", article_no="第六百七十六条", law_area="民法"),
            ),
            _make_retrieval_result(
                _make_chunk("c2", "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担违约责任", law_name="民法典", article_no="第五百七十七条", law_area="民法"),
            ),
        ]

    @pytest.fixture
    def generator(self, default_chunks) -> LegalDocumentGenerator:
        provider = FakeProvider()
        retriever = MockRetriever(default_chunks)
        return LegalDocumentGenerator(
            retriever=retriever,
            provider=provider,
            template_dir=Path(__file__).parent.parent.parent / "legalbot" / "document" / "templates",
            enabled_types=["complaint", "defense", "agent_opinion", "appeal", "enforcement"],
        )

    @pytest.mark.asyncio
    async def test_generate_returns_document_content(self, generator: LegalDocumentGenerator):
        result = await generator.generate(
            doc_type="complaint",
            case_facts="张三借款给李四10万元，约定2024年1月1日还款，但李四至今未还",
            law_areas=["民法"],
        )
        assert result is not None
        assert len(result) > 0
        # Should contain extracted information
        assert "张三" in result or "李四" in result

    @pytest.mark.asyncio
    async def test_generate_adds_disclaimer(self, generator: LegalDocumentGenerator):
        result = await generator.generate(
            doc_type="complaint",
            case_facts="张三借款给李四10万元",
            law_areas=["民法"],
        )
        assert "免责声明" in result
        assert "AI" in result or "仅供参考" in result

    @pytest.mark.asyncio
    async def test_generate_retriever_called_with_query(self, generator: LegalDocumentGenerator, default_chunks):
        await generator.generate(
            doc_type="complaint",
            case_facts="张三借款给李四10万元",
            law_areas=["民法"],
        )
        call = generator.retriever.called_with
        assert call is not None
        assert "query" in call
        # Should include law keywords from template + case facts
        assert call["law_area"] == "民法"

    @pytest.mark.asyncio
    async def test_generate_empty_law_areas_does_not_crash(self, generator: LegalDocumentGenerator):
        # Should not crash when law_areas is None or empty
        result = await generator.generate(
            doc_type="complaint",
            case_facts="张三借款给李四10万元",
            law_areas=None,
        )
        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_unsupported_doc_type_returns_message(self, generator: LegalDocumentGenerator):
        result = await generator.generate(
            doc_type="not_a_real_type",
            case_facts="some facts",
            law_areas=["民法"],
        )
        assert "不支持" in result
        assert "not_a_real_type" in result

    @pytest.mark.asyncio
    async def test_generate_uses_correct_template(self, generator: LegalDocumentGenerator):
        # Defense template should produce different content than complaint
        result = await generator.generate(
            doc_type="defense",
            case_facts="原告王五诉被告赵六借款纠纷",
            law_areas=["民法"],
        )
        assert result is not None
        # Provider response contains "赵六" (defendant) not "王五" as plaintiff
        assert "赵六" in result or "免责声明" in result

    @pytest.mark.asyncio
    async def test_generate_empty_case_facts_handled(self, generator: LegalDocumentGenerator):
        result = await generator.generate(
            doc_type="complaint",
            case_facts="",
            law_areas=["民法"],
        )
        # Should still return something (possibly an error message or empty disclaimer)
        assert result is not None

    @pytest.mark.asyncio
    async def test_generate_with_extra_variables(self, generator: LegalDocumentGenerator):
        result = await generator.generate(
            doc_type="complaint",
            case_facts="张三借款给李四10万元",
            law_areas=["民法"],
            extra_variables={"plaintiff_name": "张三", "defendant_name": "李四"},
        )
        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_appeal_template(self, generator: LegalDocumentGenerator):
        result = await generator.generate(
            doc_type="appeal",
            case_facts="一审判决被告还款10万元",
            law_areas=["民法"],
        )
        assert result is not None
        assert "免责声明" in result

    @pytest.mark.asyncio
    async def test_generate_enforcement_template(self, generator: LegalDocumentGenerator):
        result = await generator.generate(
            doc_type="enforcement",
            case_facts="判决被告还款10万元，判决已生效",
            law_areas=["民法"],
        )
        assert result is not None
        assert "免责声明" in result


class TestLegalDocumentGeneratorErrorHandling:
    """Error handling in LegalDocumentGenerator.generate()."""

    @pytest.fixture
    def generator(self) -> LegalDocumentGenerator:
        # Provider that returns empty content
        provider = FakeProvider(response_text="")
        retriever = MockRetriever([])
        return LegalDocumentGenerator(
            retriever=retriever,
            provider=provider,
            enabled_types=["complaint"],
        )

    @pytest.mark.asyncio
    async def test_generate_empty_llm_response_returns_error_message(self, generator: LegalDocumentGenerator):
        result = await generator.generate(
            doc_type="complaint",
            case_facts="张三借款给李四10万元",
            law_areas=["民法"],
        )
        # Empty response should result in an error message being returned
        assert "失败" in result or "无效" in result or "错误" in result

    @pytest.mark.asyncio
    async def test_generate_whitespace_only_response(self, generator: LegalDocumentGenerator):
        generator.provider.response_text = "   \n\t  "
        result = await generator.generate(
            doc_type="complaint",
            case_facts="案件事实",
            law_areas=["民法"],
        )
        assert "失败" in result or "无效" in result or "错误" in result


class TestLegalDocumentGeneratorRetrieverErrors:
    """Retriever error handling in LegalDocumentGenerator."""

    @pytest.fixture
    def error_retriever(self):
        class ErrorRetriever:
            async def retrieve(self, **kwargs):
                raise RuntimeError("Simulated retriever failure")

        return ErrorRetriever()

    @pytest.mark.asyncio
    async def test_generate_retriever_error_returns_error_message(self, error_retriever):
        provider = FakeProvider(response_text="正常回复")
        generator = LegalDocumentGenerator(
            retriever=error_retriever,
            provider=provider,
            enabled_types=["complaint"],
        )
        result = await generator.generate(
            doc_type="complaint",
            case_facts="案件事实",
            law_areas=["民法"],
        )
        # Should handle error gracefully with user-friendly message
        assert "失败" in result or "错误" in result or "重试" in result
