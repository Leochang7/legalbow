"""Feedback collection interface."""

import uuid
from datetime import datetime

from legalbot.feedback.models import ChunkResult, CorrectionInfo, FeedbackRecord, QueryInfo
from legalbot.feedback.storage import FeedbackStorage


class FeedbackCollector:
    """Collects and submits user feedback on RAG retrieval results."""

    def __init__(self, storage: FeedbackStorage):
        self._storage = storage

    async def submit(
        self,
        feedback_type: str,
        query_info: dict | None = None,
        result_id: str | None = None,
        helpful: bool | None = None,
        correction: str | None = None,
        reason: str | None = None,
        rank: int = 0,
        law_name: str = "",
        article_no: str = "",
        score: float = 0.0,
        session_id: str | None = None,
        channel: str = "cli",
        model_version: str = "",
        reranker_enabled: bool = False,
        latency_ms: int | None = None,
    ) -> str:
        """Submit a feedback record.

        Returns the feedback ID.
        """
        feedback_id = f"fb-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

        query = QueryInfo(**query_info) if query_info else QueryInfo(text="")

        results: list[ChunkResult] = []
        correction_info: CorrectionInfo | None = None

        if result_id:
            if correction:
                correction_info = CorrectionInfo(
                    chunk_id=result_id,
                    corrected_text=correction,
                    reason=reason,
                )
            else:
                results = [
                    ChunkResult(
                        rank=rank,
                        chunk_id=result_id,
                        law_name=law_name,
                        article_no=article_no,
                        score=score,
                        helpful=helpful,
                    )
                ]

        record = FeedbackRecord(
            id=feedback_id,
            timestamp=datetime.now(),
            type=feedback_type,
            query=query,
            results=results,
            correction=correction_info,
            session_id=session_id,
            channel=channel,
            model_version=model_version,
            reranker_enabled=reranker_enabled,
            latency_ms=latency_ms,
        )

        await self._storage.append(record)
        return feedback_id

    async def submit_helpful(
        self,
        result_id: str,
        query_info: dict | None = None,
        rank: int = 0,
        law_name: str = "",
        article_no: str = "",
        score: float = 0.0,
        **kwargs,
    ) -> str:
        """Submit a helpful feedback."""
        return await self.submit(
            feedback_type="helpful",
            query_info=query_info,
            result_id=result_id,
            helpful=True,
            rank=rank,
            law_name=law_name,
            article_no=article_no,
            score=score,
            **kwargs,
        )

    async def submit_unhelpful(
        self,
        result_id: str,
        query_info: dict | None = None,
        reason: str | None = None,
        rank: int = 0,
        law_name: str = "",
        article_no: str = "",
        score: float = 0.0,
        **kwargs,
    ) -> str:
        """Submit an unhelpful feedback."""
        return await self.submit(
            feedback_type="unhelpful",
            query_info=query_info,
            result_id=result_id,
            helpful=False,
            reason=reason,
            rank=rank,
            law_name=law_name,
            article_no=article_no,
            score=score,
            **kwargs,
        )

    async def submit_correction(
        self,
        result_id: str,
        corrected_text: str,
        query_info: dict | None = None,
        reason: str | None = None,
        **kwargs,
    ) -> str:
        """Submit a correction for a retrieval result."""
        return await self.submit(
            feedback_type="correction",
            query_info=query_info,
            result_id=result_id,
            correction=corrected_text,
            reason=reason,
            **kwargs,
        )
