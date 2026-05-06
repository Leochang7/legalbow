"""Feedback data models."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class QueryInfo:
    text: str
    law_area: str | None = None
    doc_type: str | None = None


@dataclass
class ChunkResult:
    rank: int
    chunk_id: str
    law_name: str
    article_no: str
    score: float
    helpful: bool | None = None
    comment: str | None = None


@dataclass
class CorrectionInfo:
    chunk_id: str
    corrected_text: str
    reason: str | None = None


@dataclass
class FeedbackRecord:
    id: str
    timestamp: datetime
    type: Literal["helpful", "unhelpful", "correction"]
    query: QueryInfo
    results: list[ChunkResult] = field(default_factory=list)
    correction: CorrectionInfo | None = None
    session_id: str | None = None
    channel: str = "cli"
    model_version: str = ""
    reranker_enabled: bool = False
    latency_ms: int | None = None

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "type": self.type,
            "query": asdict(self.query),
            "results": [asdict(r) for r in self.results],
            "correction": asdict(self.correction) if self.correction else None,
            "session_id": self.session_id,
            "channel": self.channel,
            "model_version": self.model_version,
            "reranker_enabled": self.reranker_enabled,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeedbackRecord":
        """Create from dict."""
        query = QueryInfo(**data["query"])
        results = [ChunkResult(**r) for r in data.get("results", [])]
        correction = None
        if data.get("correction"):
            correction = CorrectionInfo(**data["correction"])
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            type=data["type"],
            query=query,
            results=results,
            correction=correction,
            session_id=data.get("session_id"),
            channel=data.get("channel", "cli"),
            model_version=data.get("model_version", ""),
            reranker_enabled=data.get("reranker_enabled", False),
            latency_ms=data.get("latency_ms"),
        )
