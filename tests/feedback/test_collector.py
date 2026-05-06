"""Unit tests for feedback collector."""

import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from legalbot.feedback.collector import FeedbackCollector
from legalbot.feedback.storage import FeedbackStorage


@pytest.fixture
def collector():
    with tempfile.TemporaryDirectory() as td:
        storage = FeedbackStorage(feedback_dir=td)
        yield FeedbackCollector(storage)


class TestFeedbackCollector:
    @pytest.mark.asyncio
    async def test_submit_helpful(self, collector):
        feedback_id = await collector.submit_helpful(
            result_id="chunk-001",
            query_info={"text": "劳动合同不续约", "law_area": "劳动法"},
            rank=1,
            law_name="劳动合同法",
            article_no="第四十六条",
            score=0.95,
        )

        assert feedback_id.startswith("fb-")
        records = collector._storage.query()
        assert len(records) == 1
        assert records[0].type == "helpful"
        assert records[0].results[0].chunk_id == "chunk-001"
        assert records[0].results[0].helpful is True

    @pytest.mark.asyncio
    async def test_submit_unhelpful(self, collector):
        feedback_id = await collector.submit_unhelpful(
            result_id="chunk-002",
            query_info={"text": "加班费计算"},
            reason="inaccurate",
        )

        records = collector._storage.query()
        assert len(records) == 1
        assert records[0].type == "unhelpful"
        assert records[0].results[0].helpful is False

    @pytest.mark.asyncio
    async def test_submit_correction(self, collector):
        feedback_id = await collector.submit_correction(
            result_id="chunk-003",
            corrected_text="正确的法条内容应该是...",
            query_info={"text": "违约金条款"},
            reason="outdated",
        )

        records = collector._storage.query()
        assert len(records) == 1
        assert records[0].type == "correction"
        assert records[0].correction is not None
        assert records[0].correction.chunk_id == "chunk-003"
        assert records[0].correction.reason == "outdated"

    @pytest.mark.asyncio
    async def test_submit_generic(self, collector):
        feedback_id = await collector.submit(
            feedback_type="helpful",
            query_info={"text": "测试查询"},
            result_id="chunk-010",
            helpful=True,
            rank=2,
            law_name="民法典",
            article_no="第五百条",
            score=0.88,
        )

        assert feedback_id.startswith("fb-")
        records = collector._storage.query()
        assert len(records) == 1
        assert records[0].type == "helpful"
