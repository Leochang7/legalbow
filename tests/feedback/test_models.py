"""Unit tests for feedback models."""

from datetime import datetime

from legalbot.feedback.models import ChunkResult, CorrectionInfo, FeedbackRecord, QueryInfo


class TestFeedbackRecord:
    def test_to_dict_and_from_dict_roundtrip(self):
        record = FeedbackRecord(
            id="fb-20260419-abc123",
            timestamp=datetime(2026, 4, 19, 10, 30, 0),
            type="helpful",
            query=QueryInfo(text="劳动合同不续约有赔偿吗", law_area="劳动法", doc_type="law"),
            results=[
                ChunkResult(
                    rank=1,
                    chunk_id="chunk-001",
                    law_name="中华人民共和国劳动合同法",
                    article_no="第四十六条",
                    score=0.95,
                    helpful=True,
                )
            ],
            session_id="cli:user1",
            channel="cli",
            model_version="text-embedding-3-small",
            reranker_enabled=True,
            latency_ms=234,
        )

        d = record.to_dict()
        restored = FeedbackRecord.from_dict(d)

        assert restored.id == record.id
        assert restored.type == record.type
        assert restored.query.text == record.query.text
        assert restored.query.law_area == record.query.law_area
        assert len(restored.results) == 1
        assert restored.results[0].chunk_id == "chunk-001"
        assert restored.session_id == "cli:user1"
        assert restored.reranker_enabled is True

    def test_correction_roundtrip(self):
        record = FeedbackRecord(
            id="fb-20260419-def456",
            timestamp=datetime(2026, 4, 19, 11, 0, 0),
            type="correction",
            query=QueryInfo(text="加班费如何计算"),
            correction=CorrectionInfo(
                chunk_id="chunk-002",
                corrected_text="正确的内容应该是...",
                reason="outdated",
            ),
        )

        d = record.to_dict()
        restored = FeedbackRecord.from_dict(d)

        assert restored.type == "correction"
        assert restored.correction is not None
        assert restored.correction.chunk_id == "chunk-002"
        assert restored.correction.reason == "outdated"
