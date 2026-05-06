"""Unit tests for feedback analyzer."""

import tempfile
from datetime import datetime, timedelta

import pytest

from legalbot.feedback.analyzer import FeedbackAnalyzer
from legalbot.feedback.models import ChunkResult, CorrectionInfo, FeedbackRecord, QueryInfo
from legalbot.feedback.storage import FeedbackStorage


@pytest.fixture
def analyzer():
    with tempfile.TemporaryDirectory() as td:
        storage = FeedbackStorage(feedback_dir=td)
        yield FeedbackAnalyzer(storage)


@pytest.fixture
def populated_storage(analyzer):
    now = datetime.now()
    records = [
        FeedbackRecord(
            id="fb-h1",
            timestamp=now - timedelta(hours=1),
            type="helpful",
            query=QueryInfo(text="劳动合同不续约有赔偿吗", law_area="劳动法"),
            results=[ChunkResult(rank=1, chunk_id="c1", law_name="劳动合同法", article_no="46", score=0.9, helpful=True)],
        ),
        FeedbackRecord(
            id="fb-u1",
            timestamp=now - timedelta(hours=2),
            type="unhelpful",
            query=QueryInfo(text="加班费如何计算", law_area="劳动法"),
            results=[ChunkResult(rank=1, chunk_id="c2", law_name="劳动合同法", article_no="44", score=0.8, helpful=False)],
        ),
        FeedbackRecord(
            id="fb-u2",
            timestamp=now - timedelta(hours=3),
            type="unhelpful",
            query=QueryInfo(text="加班费如何计算", law_area="劳动法"),
            results=[ChunkResult(rank=1, chunk_id="c2", law_name="劳动合同法", article_no="44", score=0.7, helpful=False)],
        ),
        FeedbackRecord(
            id="fb-c1",
            timestamp=now - timedelta(hours=4),
            type="correction",
            query=QueryInfo(text="违约金规定"),
            correction=CorrectionInfo(chunk_id="c3", corrected_text="正确内容", reason="outdated"),
        ),
        FeedbackRecord(
            id="fb-c2",
            timestamp=now - timedelta(hours=5),
            type="correction",
            query=QueryInfo(text="违约金规定"),
            correction=CorrectionInfo(chunk_id="c3", corrected_text="另一正确内容", reason="outdated"),
        ),
    ]
    import asyncio
    for r in records:
        asyncio.run(analyzer._storage.append(r))
    yield analyzer


class TestFeedbackAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_period_basic(self, populated_storage):
        now = datetime.now()
        report = await populated_storage.analyze_period(
            since=now - timedelta(days=1),
            until=now,
        )

        assert report.total_feedback == 5
        assert report.helpful_rate == pytest.approx(1 / 5)
        assert len(report.top_problem_queries) >= 1
        # "加班费如何计算" should be a top problem query (2 unhelpful)
        problem_texts = [q.query_text for q in report.top_problem_queries]
        assert "加班费如何计算" in problem_texts

    @pytest.mark.asyncio
    async def test_identify_outdated_chunks(self, populated_storage):
        outdated = await populated_storage.identify_outdated_chunks(min_corrections=2)
        assert len(outdated) == 1
        assert outdated[0].chunk_id == "c3"
        assert outdated[0].correction_count == 2

    @pytest.mark.asyncio
    async def test_identify_low_score_chunks(self, populated_storage):
        # c2 has 2 unhelpful ratings (0 helpful), should be low score
        low = await populated_storage.identify_low_score_chunks(threshold=0.5, min_reports=2)
        c2_chunks = [c for c in low if c.chunk_id == "c2"]
        assert len(c2_chunks) == 1
        assert c2_chunks[0].avg_helpful_score == 0.0

    def test_generate_markdown_report(self, populated_storage):
        import asyncio
        now = datetime.now()
        report = asyncio.run(populated_storage.analyze_period(
            since=now - timedelta(days=1),
            until=now,
        ))
        markdown = populated_storage.generate_markdown_report(report)

        assert "# 反馈分析报告" in markdown
        assert "总体统计" in markdown
        assert "总反馈数: 5" in markdown
