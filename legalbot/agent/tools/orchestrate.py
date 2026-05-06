"""Orchestrate tool for MultiAgent legal dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from legalbot.agent.tools.base import Tool, tool_parameters
from legalbot.agent.tools.schema import StringSchema, tool_parameters_schema

if TYPE_CHECKING:
    from legalbot.agent.orchestrator import LegalOrchestrator


@tool_parameters(
    tool_parameters_schema(
        query=StringSchema("用户的法律问题或请求"),
        intent=StringSchema(
            "意图类别覆盖：legal_query/contract_review/case_search，"
            "留空则由编排器自动分类",
            nullable=True,
        ),
        required=["query"],
    )
)
class OrchestrateTool(Tool):
    """MultiAgent orchestration tool — dispatch to specialized legal agents.

    When orchestration is enabled, this is the ONLY legal tool available to the
    main agent. It handles both simple queries (via direct RAG) and complex
    queries (via multi-step legal reasoner).
    """

    def __init__(self, orchestrator: LegalOrchestrator, retriever: Any = None, get_tools: Any = None):
        self._orchestrator = orchestrator
        self._retriever = retriever  # for direct RAG on simple queries
        self._get_tools = get_tools  # callable returning current tools dict

    @property
    def _main_tools(self) -> dict[str, Any]:
        if self._get_tools:
            return self._get_tools()
        return self._orchestrator._main_tools

    @property
    def name(self) -> str:
        return "legal_orchestrate"

    @property
    def description(self) -> str:
        return (
            "调度法律专业 Agent 团队处理所有法律任务。"
            "可自动路由到：法律检索 Agent、合同审查 Agent、法律辩论 Agent、"
            "案例对比 Agent、法律文书起草 Agent（生成 .docx 格式起诉状/答辩状/代理词/上诉状/执行申请书）。"
            "所有法律请求（法律咨询、合同审查、案例检索、辩论分析、文书起草）都应先调用此工具。"
        )

    @property
    def exclusive(self) -> bool:
        return True  # Orchestration tasks run exclusively

    async def execute(
        self,
        query: str,
        intent: str | None = None,
        **kwargs: Any,
    ) -> str:
        # Detect debate requests
        debate_keywords = (
            "辩论", "debate", "争议焦点", "原告", "被告",
            "诉讼请求", "答辩", "抗辩", "法律辩论",
        )
        # Detect case comparison requests
        case_compare_keywords = (
            "案例对比", "对比案例", "类似案例", "相似案例",
            "case compare", "类案", "案例检索",
        )

        if any(kw in query for kw in debate_keywords):
            if self._orchestrator.config.debate and self._orchestrator.config.debate.enable:
                return await self._orchestrator.run_debate_sync(
                    case_description=query,
                    plaintiff_claims=kwargs.get("plaintiff_claims"),
                    defendant_response=kwargs.get("defendant_response"),
                    debate_rounds=kwargs.get("debate_rounds", 1),
                )
            return "辩论模式未启用。请在配置中启用 orchestrate.debate.enable=true"

        if any(kw in query for kw in case_compare_keywords):
            tool = self._main_tools.get("legal_case_compare")
            if tool:
                return await tool.execute(dispute_facts=query)
            return "案例对比工具未配置，请使用 legal_rag_search 检索相关案例。"

        # Detect document draft requests
        doc_draft_keywords = (
            "起诉状", "答辩状", "代理词", "上诉状", "执行申请书",
            "写一份起诉书", "起草", "帮我写诉状", "写诉状",
            "complaint", "defense", "appeal", "enforcement",
        )
        if any(kw in query for kw in doc_draft_keywords):
            tool = self._main_tools.get("legal_document_generate")
            if tool:
                return await tool.execute(case_facts=query, doc_type="complaint")
            return "法律文书生成工具未配置，请使用 legal_rag_search 检索相关法律信息。"

        # If intent is explicitly provided, skip classification
        if intent and intent in ("legal_query", "contract_review", "case_search"):
            if intent in ("legal_query", "case_search"):
                # Simple legal query: use RAG directly if retriever is available
                if self._retriever is not None:
                    results = await self._retriever.retrieve(query=query, top_k=5)
                    from legalbot.agent.tools.rag import RAGSearchTool
                    return RAGSearchTool._format_results(query, results.top_k)
                result = await self._orchestrator._run_agent_sync("legal_research", query)  # noqa: SLF001
            elif intent == "contract_review":
                result = await self._orchestrator._contract_review_flow_sync(query)
            else:
                result = None
        else:
            # Automatic classification: dispatch through orchestrator
            result = await self._orchestrator.dispatch_sync(query)

        return result or "未能调度专业 Agent，请直接使用 legal_rag_search 工具检索。"
