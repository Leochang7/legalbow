"""Feedback analysis pipeline."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from legalbot.feedback.storage import FeedbackStorage


@dataclass
class QueryStats:
    query_text: str
    total_reports: int
    helpful_count: int
    unhelpful_count: int
    law_area: str | None
    sample_unhelpful: list[str]


@dataclass
class ChunkStats:
    chunk_id: str
    law_name: str
    article_no: str
    total_reports: int
    correction_count: int
    outdated_count: int
    avg_helpful_score: float


@dataclass
class AnalysisReport:
    period_start: datetime
    period_end: datetime
    total_feedback: int
    helpful_rate: float
    top_problem_queries: list[QueryStats] = field(default_factory=list)
    low_score_chunks: list[ChunkStats] = field(default_factory=list)
    outdated_chunks: list[ChunkStats] = field(default_factory=list)


class FeedbackAnalyzer:
    """Analyzes feedback data and generates improvement suggestions."""

    def __init__(self, storage: FeedbackStorage):
        self._storage = storage

    async def analyze_period(
        self, since: datetime, until: datetime
    ) -> AnalysisReport:
        """Analyze feedback for a given time period."""
        records = self._storage.query(since=since, until=until)

        total = len(records)
        helpful_count = sum(1 for r in records if r.type == "helpful")
        unhelpful_count = sum(1 for r in records if r.type == "unhelpful")

        # Build top problem queries
        query_map: dict[str, dict] = {}
        for r in records:
            key = r.query.text
            if key not in query_map:
                query_map[key] = {
                    "text": key,
                    "law_area": r.query.law_area,
                    "helpful": 0,
                    "unhelpful": 0,
                    "unhelpful_samples": [],
                }
            if r.type == "helpful":
                query_map[key]["helpful"] += 1
            elif r.type == "unhelpful":
                query_map[key]["unhelpful"] += 1
                # Collect sample unhelpful reasons from corrections
                if r.correction:
                    sample = r.correction.corrected_text[:100]
                    if len(query_map[key]["unhelpful_samples"]) < 3:
                        query_map[key]["unhelpful_samples"].append(sample)

        # Sort by unhelpful ratio
        problem_queries = sorted(
            [
                QueryStats(
                    query_text=v["text"],
                    total_reports=v["helpful"] + v["unhelpful"],
                    helpful_count=v["helpful"],
                    unhelpful_count=v["unhelpful"],
                    law_area=v["law_area"],
                    sample_unhelpful=v["unhelpful_samples"],
                )
                for v in query_map.values()
                if v["unhelpful"] >= 1
            ],
            key=lambda x: x.unhelpful_count / max(x.total_reports, 1),
            reverse=True,
        )[:10]

        # Low score chunks
        low_score = await self.identify_low_score_chunks(threshold=0.5, min_reports=2)

        # Outdated chunks
        outdated = await self.identify_outdated_chunks(min_corrections=2)

        return AnalysisReport(
            period_start=since,
            period_end=until,
            total_feedback=total,
            helpful_rate=helpful_count / total if total > 0 else 0.0,
            top_problem_queries=problem_queries,
            low_score_chunks=low_score,
            outdated_chunks=outdated,
        )

    async def identify_low_score_chunks(
        self, threshold: float = 0.5, min_reports: int = 3
    ) -> list[ChunkStats]:
        """Identify chunks with low helpful scores."""
        chunk_map: dict[str, dict] = {}

        for record in self._storage.query(feedback_type="unhelpful"):
            for result in record.results:
                key = result.chunk_id
                if key not in chunk_map:
                    chunk_map[key] = {
                        "chunk_id": key,
                        "law_name": result.law_name,
                        "article_no": result.article_no,
                        "reports": [],
                    }
                chunk_map[key]["reports"].append(result)

        low_score_chunks = []
        for data in chunk_map.values():
            reports = data["reports"]
            if len(reports) < min_reports:
                continue
            helpful_count = sum(1 for r in reports if r.helpful is True)
            avg_score = helpful_count / len(reports)
            if avg_score < threshold:
                low_score_chunks.append(
                    ChunkStats(
                        chunk_id=data["chunk_id"],
                        law_name=data["law_name"],
                        article_no=data["article_no"],
                        total_reports=len(reports),
                        correction_count=0,
                        outdated_count=sum(1 for r in reports if r.comment == "outdated"),
                        avg_helpful_score=avg_score,
                    )
                )

        return sorted(low_score_chunks, key=lambda x: x.avg_helpful_score)[:20]

    async def identify_outdated_chunks(self, min_corrections: int = 2) -> list[ChunkStats]:
        """Identify chunks that have been corrected multiple times."""
        chunk_map: dict[str, dict] = {}

        for record in self._storage.query(feedback_type="correction"):
            if not record.correction:
                continue
            key = record.correction.chunk_id
            if key not in chunk_map:
                chunk_map[key] = {
                    "chunk_id": key,
                    "law_name": record.query.text.split("|")[0] if "|" in record.query.text else "",
                    "article_no": "",
                    "corrections": [],
                }
            chunk_map[key]["corrections"].append(record.correction)

        outdated = [
            ChunkStats(
                chunk_id=data["chunk_id"],
                law_name=data["law_name"],
                article_no=data["article_no"],
                total_reports=len(data["corrections"]),
                correction_count=len(data["corrections"]),
                outdated_count=sum(
                    1 for c in data["corrections"] if c.reason == "outdated"
                ),
                avg_helpful_score=0.0,
            )
            for data in chunk_map.values()
            if len(data["corrections"]) >= min_corrections
        ]

        return sorted(outdated, key=lambda x: x.correction_count, reverse=True)[:20]

    def generate_markdown_report(self, report: AnalysisReport) -> str:
        """Generate a markdown-formatted analysis report."""
        lines = [
            f"# 反馈分析报告",
            f"",
            f"**统计周期**: {report.period_start.strftime('%Y-%m-%d')} ~ {report.period_end.strftime('%Y-%m-%d')}",
            f"",
            f"## 总体统计",
            f"",
            f"- 总反馈数: {report.total_feedback}",
            f"- 有帮助率: {report.helpful_rate:.1%}",
            f"- 问题查询数: {len(report.top_problem_queries)}",
            f"- 低分Chunk数: {len(report.low_score_chunks)}",
            f"- 过期Chunk数: {len(report.outdated_chunks)}",
            f"",
        ]

        if report.top_problem_queries:
            lines.append("## 高频问题查询")
            lines.append("")
            for i, qs in enumerate(report.top_problem_queries[:5], 1):
                lines.append(f"{i}. **{qs.query_text[:60]}**")
                lines.append(f"   - 总反馈: {qs.total_reports} | 有帮助: {qs.helpful_count} | 无帮助: {qs.unhelpful_count}")
                if qs.sample_unhelpful:
                    lines.append(f"   - 示例问题: {qs.sample_unhelpful[0][:80]}")
                lines.append("")

        if report.low_score_chunks:
            lines.append("## 低分法律条文")
            lines.append("")
            for cs in report.low_score_chunks[:10]:
                lines.append(f"- [{cs.law_name} {cs.article_no}]({cs.chunk_id}) — 准确率: {cs.avg_helpful_score:.1%} ({cs.total_reports}次反馈)")
            lines.append("")

        if report.outdated_chunks:
            lines.append("## 可能过期的法律条文")
            lines.append("")
            for cs in report.outdated_chunks[:10]:
                lines.append(f"- [{cs.law_name} {cs.article_no}]({cs.chunk_id}) — 纠错次数: {cs.correction_count}")
            lines.append("")

        lines.append("---")
        lines.append(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(lines)
