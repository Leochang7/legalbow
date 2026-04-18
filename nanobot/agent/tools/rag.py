"""RAG search tool for legal knowledge base retrieval."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema


@tool_parameters(
    tool_parameters_schema(
        query=StringSchema("法律问题或关键词"),
        law_area=StringSchema("法律领域过滤：民法/刑法/商法/劳动法/行政法等", nullable=True),
        doc_type=StringSchema("文档类型过滤：law/judicial_interpretation/case/contract_template", nullable=True),
        top_k=IntegerSchema(5, description="返回结果数", minimum=1, maximum=10),
        required=["query"],
    )
)
class RAGSearchTool(Tool):
    """法律知识库检索工具 — 搜索法规、司法解释和案例."""

    def __init__(self, retriever: Any):
        self._retriever = retriever

    @property
    def name(self) -> str:
        return "legal_rag_search"

    @property
    def description(self) -> str:
        return (
            "搜索法律知识库，检索相关法规、司法解释和案例。"
            "输入法律问题或关键词，返回最相关的法律条文和案例。"
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self,
        query: str,
        law_area: str | None = None,
        doc_type: str | None = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> str:
        results = await self._retriever.retrieve(
            query, law_area=law_area, doc_type=doc_type, top_k=top_k
        )
        return self._format_results(query, results.top_k)

    @staticmethod
    def _format_results(query: str, results: list) -> str:
        if not results:
            return f"未检索到与「{query}」相关的法律条文。"
        lines = [f"检索结果（{query}）：\n"]
        for i, r in enumerate(results, 1):
            meta = r.chunk.metadata if hasattr(r, "chunk") else {}
            lines.append(f"{i}. 【{meta.get('law_name', '未知')}】{meta.get('article_no', '')}")
            lines.append(f"   领域：{meta.get('law_area', '')} | 类型：{meta.get('doc_type', '')}")
            text = r.chunk.text if hasattr(r, "chunk") else str(r)
            lines.append(f"   {text[:300]}")
            if len(text) > 300:
                lines.append("   ...")
            lines.append("")
        return "\n".join(lines)
