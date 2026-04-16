"""Orchestrate tool for MultiAgent legal dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import StringSchema, tool_parameters_schema

if TYPE_CHECKING:
    from nanobot.agent.orchestrator import LegalOrchestrator


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
    """MultiAgent orchestration tool — dispatch to specialized legal agents."""

    def __init__(self, orchestrator: LegalOrchestrator):
        self._orchestrator = orchestrator

    @property
    def name(self) -> str:
        return "legal_orchestrate"

    @property
    def description(self) -> str:
        return (
            "调度法律专业 Agent 团队处理复杂法律任务。"
            "可根据任务类型自动路由到法律检索 Agent 或合同审查 Agent。"
            "适用于需要专业法律分析、合同审查或多步骤法律推理的场景。"
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
        # If intent is explicitly provided, skip classification
        if intent and intent in ("legal_query", "contract_review", "case_search"):
            if intent in ("legal_query", "case_search"):
                result = await self._orchestrator._run_agent_sync("legal_research", query)  # noqa: SLF001
            elif intent == "contract_review":
                result = await self._orchestrator._contract_review_flow_sync(query)
            else:
                result = None
        else:
            result = await self._orchestrator.dispatch_sync(query)

        return result or "未能调度专业 Agent，请直接使用 legal_rag_search 工具检索。"
