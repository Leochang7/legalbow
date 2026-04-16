"""Unit tests for RAGSearchTool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.tools.rag import RAGSearchTool
from nanobot.rag.chunker import Chunk, ChunkMeta
from nanobot.rag.retriever import RetrievalResult


def _make_result(law_name: str, article_no: str, text: str) -> RetrievalResult:
    chunk = Chunk(
        id="test-id",
        text=text,
        metadata=ChunkMeta(law_name=law_name, article_no=article_no, law_area="民法", doc_type="law"),
    )
    return RetrievalResult(chunk=chunk, score=0.9)


class TestRAGSearchTool:

    def test_tool_schema(self):
        retriever = MagicMock()
        tool = RAGSearchTool(retriever=retriever)
        schema = tool.parameters

        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "query" in schema["required"]

    def test_name(self):
        tool = RAGSearchTool(retriever=MagicMock())
        assert tool.name == "legal_rag_search"

    def test_read_only(self):
        tool = RAGSearchTool(retriever=MagicMock())
        assert tool.read_only is True

    async def test_execute_returns_formatted_results(self):
        retriever = AsyncMock()
        retriever.retrieve.return_value = [
            _make_result("中华人民共和国民法典", "第五百八十五条", "当事人可以约定一方违约时应当支付违约金。"),
        ]

        tool = RAGSearchTool(retriever=retriever)
        result = await tool.execute(query="违约金")

        assert "民法典" in result
        assert "第五百八十五条" in result
        assert "违约金" in result

    async def test_execute_no_results(self):
        retriever = AsyncMock()
        retriever.retrieve.return_value = []

        tool = RAGSearchTool(retriever=retriever)
        result = await tool.execute(query="不存在的法律问题")

        assert "未检索到" in result
        assert "不存在的法律问题" in result

    async def test_execute_passes_filters(self):
        retriever = AsyncMock()
        retriever.retrieve.return_value = []

        tool = RAGSearchTool(retriever=retriever)
        await tool.execute(query="合同", law_area="民法", doc_type="law", top_k=3)

        retriever.retrieve.assert_called_once()
        call_kwargs = retriever.retrieve.call_args
        assert call_kwargs.kwargs.get("query") == "合同" or call_kwargs.args[0] == "合同"
        assert call_kwargs.kwargs.get("law_area") == "民法"
        assert call_kwargs.kwargs.get("doc_type") == "law"
        assert call_kwargs.kwargs.get("top_k") == 3

    async def test_long_text_truncated(self):
        long_text = "这是一段很长的法律条文。" * 50
        retriever = AsyncMock()
        retriever.retrieve.return_value = [
            _make_result("民法典", "第一条", long_text),
        ]

        tool = RAGSearchTool(retriever=retriever)
        result = await tool.execute(query="法律")

        # Should contain "..." for truncated text
        assert "..." in result
