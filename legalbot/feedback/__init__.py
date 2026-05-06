"""User feedback collection and analysis for LegalBot RAG."""

from legalbot.feedback.analyzer import FeedbackAnalyzer
from legalbot.feedback.collector import FeedbackCollector
from legalbot.feedback.models import ChunkResult, CorrectionInfo, FeedbackRecord, QueryInfo
from legalbot.feedback.storage import FeedbackStorage

__all__ = [
    "FeedbackAnalyzer",
    "FeedbackCollector",
    "FeedbackRecord",
    "QueryInfo",
    "ChunkResult",
    "CorrectionInfo",
    "FeedbackStorage",
]
