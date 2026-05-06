"""Unit tests for FeedbackTool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from legalbot.agent.tools.feedback import FeedbackTool


class TestFeedbackTool:
    @pytest.mark.asyncio
    async def test_submit_helpful(self):
        collector = MagicMock()
        collector.submit_helpful = AsyncMock(return_value="fb-20260419-abc123")

        tool = FeedbackTool(collector=collector)
        result = await tool.execute(result_id="chunk-001", helpful=True)

        collector.submit_helpful.assert_called_once_with(result_id="chunk-001")
        assert "感谢您的反馈" in result
        assert "fb-20260419-abc123" in result

    @pytest.mark.asyncio
    async def test_submit_unhelpful(self):
        collector = MagicMock()
        collector.submit_unhelpful = AsyncMock(return_value="fb-20260419-def456")

        tool = FeedbackTool(collector=collector)
        result = await tool.execute(result_id="chunk-002", helpful=False, reason="inaccurate")

        collector.submit_unhelpful.assert_called_once_with(result_id="chunk-002", reason="inaccurate")
        assert "感谢您的反馈" in result

    @pytest.mark.asyncio
    async def test_submit_correction(self):
        collector = MagicMock()
        collector.submit_correction = AsyncMock(return_value="fb-20260419-ghi789")

        tool = FeedbackTool(collector=collector)
        result = await tool.execute(
            result_id="chunk-003",
            correction="正确的法条内容是...",
            reason="outdated",
        )

        collector.submit_correction.assert_called_once_with(
            result_id="chunk-003",
            corrected_text="正确的法条内容是...",
            reason="outdated",
        )
        assert "已收到纠正反馈" in result

    @pytest.mark.asyncio
    async def test_missing_feedback_input(self):
        collector = MagicMock()
        tool = FeedbackTool(collector=collector)
        result = await tool.execute(result_id="chunk-004")

        assert "请提供" in result
        collector.submit_helpful.assert_not_called()
