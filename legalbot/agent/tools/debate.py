"""Debate tool for legal dispute analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from legalbot.agent.tools.base import Tool, tool_parameters
from legalbot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema

if TYPE_CHECKING:
    from legalbot.agent.orchestrator import LegalOrchestrator


@tool_parameters(
    tool_parameters_schema(
        case_description=StringSchema("案件事实描述"),
        plaintiff_claims=StringSchema("原告诉讼请求（可选）", nullable=True),
        defendant_response=StringSchema("被告答辩意见（可选）", nullable=True),
        debate_rounds=IntegerSchema(
            1,
            description="辩论轮次：1=单轮快速辩论，2=双轮深度辩论",
            minimum=1,
            maximum=3,
        ),
        required=["case_description"],
    )
)
class DebateTool(Tool):
    """法律辩论工具 — 启动原告 vs 被告 Agent 辩论并生成分析报告."""

    name = "legal_debate"
    description = (
        "启动法律辩论模式：原告代理 Agent 与被告代理 Agent 针对法律纠纷展开并行论证，"
        "由审判 Agent 综合双方论点生成《争议焦点分析报告》。"
    )

    def __init__(self, orchestrator: LegalOrchestrator):
        self._orchestrator = orchestrator

    @property
    def exclusive(self) -> bool:
        return True

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self,
        case_description: str,
        plaintiff_claims: str | None = None,
        defendant_response: str | None = None,
        debate_rounds: int = 1,
        **kwargs: Any,
    ) -> str:
        return await self._orchestrator.run_debate_sync(
            case_description=case_description,
            plaintiff_claims=plaintiff_claims,
            defendant_response=defendant_response,
            debate_rounds=debate_rounds,
        )
