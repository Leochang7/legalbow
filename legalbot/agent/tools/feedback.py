"""Feedback tool for RAG retrieval quality rating."""

from __future__ import annotations

from legalbot.agent.tools.base import Tool, tool_parameters
from legalbot.agent.tools.schema import BooleanSchema, StringSchema, tool_parameters_schema


@tool_parameters(
    tool_parameters_schema(
        result_id=StringSchema("被评价的检索结果ID（chunk_id）"),
        helpful=BooleanSchema(description="该结果是否有帮助", nullable=True),
        correction=StringSchema("纠正内容（当结果不准确时填写）", nullable=True),
        reason=StringSchema("评价原因：inaccurate/outdated/incomplete/other", nullable=True),
        required=["result_id"],
    )
)
class FeedbackTool(Tool):
    """提交检索结果反馈 — 有帮助(+) / 不准确(-) / 纠正."""

    name = "legal_feedback"
    description = (
        "提交法律检索结果的反馈。用于评价检索结果是否有用，"
        "或提供纠正意见。result_id为必填，helpful和correction二选一。"
    )

    @property
    def read_only(self) -> bool:
        return True

    def __init__(self, collector: "FeedbackCollector", retriever: Any | None = None):
        # Avoid circular import at module level
        from legalbot.feedback.collector import FeedbackCollector
        self._collector: FeedbackCollector = collector
        self._retriever = retriever

    async def execute(
        self,
        result_id: str,
        helpful: bool | None = None,
        correction: str | None = None,
        reason: str | None = None,
    ) -> str:
        if correction:
            feedback_id = await self._collector.submit_correction(
                result_id=result_id,
                corrected_text=correction,
                reason=reason or "inaccurate",
            )
            return f"已收到纠正反馈 (ID: {feedback_id})，感谢您的修正！"
        elif helpful is not None:
            if helpful:
                feedback_id = await self._collector.submit_helpful(result_id=result_id)
            else:
                feedback_id = await self._collector.submit_unhelpful(
                    result_id=result_id, reason=reason
                )
            return f"感谢您的反馈 (ID: {feedback_id})！"
        else:
            return "请提供 helpful=true/false 或 correction 内容。"
