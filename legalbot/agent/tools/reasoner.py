"""Multi-step legal reasoning tool — exposes MultiStepLegalReasoner to the agent loop."""

from __future__ import annotations

from typing import Any

from legalbot.agent.tools.base import Tool, tool_parameters
from legalbot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema


@tool_parameters(
    tool_parameters_schema(
        query=StringSchema("复杂法律问题"),
        max_steps=IntegerSchema(
            5, description="最大推理步数（不含最终合成）", minimum=1, maximum=10
        ),
        law_area=StringSchema(
            "法律领域过滤：民法/刑法/商法/劳动法/行政法等",
            nullable=True,
        ),
    )
)
class MultiStepReasoningTool(Tool):
    """多轮法律链式推理工具 — 对复杂法律问题进行链式推理分析."""

    def __init__(self, reasoner: Any, retriever: Any):
        self._reasoner = reasoner
        self._retriever = retriever

    @property
    def name(self) -> str:
        return "legal_multi_step_reasoning"

    @property
    def description(self) -> str:
        return (
            "对复杂法律问题进行多轮链式推理分析。"
            "每一步会检索相关法条、分析法律关系、识别遗漏点，"
            "最终生成包含完整推理链和法律引用的分析报告。"
        )

    @property
    def exclusive(self) -> bool:
        return True

    async def execute(
        self,
        query: str,
        max_steps: int = 5,
        law_area: str | None = None,
        **kwargs: Any,
    ) -> str:
        chain = await self._reasoner.reason(question=query, law_area=law_area)
        return chain.to_display_string()
