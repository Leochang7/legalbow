"""Unit tests for feedback storage."""

import json
import tempfile
from datetime import datetime

import pytest

from legalbot.feedback.models import ChunkResult, FeedbackRecord, QueryInfo
from legalbot.feedback.storage import FeedbackStorage


@pytest.fixture
def temp_storage():
    with tempfile.TemporaryDirectory() as td:
        yield FeedbackStorage(feedback_dir=td)


@pytest.fixture
def sample_record():
    return FeedbackRecord(
        id="fb-20260419-test001",
        timestamp=datetime(2026, 4, 19, 10, 30, 0),
        type="helpful",
        query=QueryInfo(text="劳动合同到期不续约", law_area="劳动法", doc_type="law"),
        results=[
            ChunkResult(
                rank=1,
                chunk_id="chunk-001",
                law_name="劳动合同法",
                article_no="第四十六条",
                score=0.95,
                helpful=True,
            )
        ],
        channel="cli",
    )


class TestFeedbackStorage:
    @pytest.mark.asyncio
    async def test_append_creates_daily_file(self, temp_storage, sample_record):
        await temp_storage.append(sample_record)

        records = temp_storage.query(
            since=datetime(2026, 4, 1),
            until=datetime(2026, 4, 30),
        )

        assert len(records) == 1
        assert records[0].id == "fb-20260419-test001"
        assert records[0].type == "helpful"

    @pytest.mark.asyncio
    async def test_query_by_type_filter(self, temp_storage, sample_record):
        await temp_storage.append(sample_record)

        unhelpful = FeedbackRecord(
            id="fb-20260419-test002",
            timestamp=datetime(2026, 4, 19, 11, 0, 0),
            type="unhelpful",
            query=QueryInfo(text="加班费"),
            results=[],
        )
        await temp_storage.append(unhelpful)

        helpful_records = temp_storage.query(feedback_type="helpful")
        assert len(helpful_records) == 1
        assert helpful_records[0].type == "helpful"

        unhelpful_records = temp_storage.query(feedback_type="unhelpful")
        assert len(unhelpful_records) == 1
        assert unhelpful_records[0].type == "unhelpful"

    @pytest.mark.asyncio
    async def test_query_by_date_range(self, temp_storage, sample_record):
        await temp_storage.append(sample_record)

        older = FeedbackRecord(
            id="fb-20260401-test003",
            timestamp=datetime(2026, 4, 1, 9, 0, 0),
            type="helpful",
            query=QueryInfo(text=" older query"),
            results=[],
        )
        await temp_storage.append(older)

        # Query only April 19+
        recent = temp_storage.query(since=datetime(2026, 4, 19))
        assert len(recent) == 1
        assert recent[0].id == "fb-20260419-test001"

        # Query only April 1-10
        early = temp_storage.query(since=datetime(2026, 4, 1), until=datetime(2026, 4, 10))
        assert len(early) == 1
        assert early[0].id == "fb-20260401-test003"

    @pytest.mark.asyncio
    async def test_list_records_limit(self, temp_storage):
        for i in range(5):
            record = FeedbackRecord(
                id=f"fb-20260419-test{i:03d}",
                timestamp=datetime(2026, 4, 19, 10, i, 0),
                type="helpful",
                query=QueryInfo(text=f"query {i}"),
                results=[],
            )
            await temp_storage.append(record)

        listed = temp_storage.list_records(limit=3)
        assert len(listed) == 3

    def test_list_records_filtered_by_type(self, temp_storage):
        import asyncio

        record1 = FeedbackRecord(
            id="fb-20260419-type1",
            timestamp=datetime(2026, 4, 19, 10, 0, 0),
            type="helpful",
            query=QueryInfo(text="helpful query"),
            results=[],
        )
        record2 = FeedbackRecord(
            id="fb-20260419-type2",
            timestamp=datetime(2026, 4, 19, 11, 0, 0),
            type="correction",
            query=QueryInfo(text="correction query"),
            results=[],
        )

        asyncio.run(temp_storage.append(record1))
        asyncio.run(temp_storage.append(record2))

        correction_only = temp_storage.list_records(feedback_type="correction")
        assert len(correction_only) == 1
        assert correction_only[0].type == "correction"
