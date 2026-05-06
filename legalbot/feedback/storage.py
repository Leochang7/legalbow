"""JSONL-based feedback storage."""

import json
from datetime import datetime
from pathlib import Path

from legalbot.feedback.models import FeedbackRecord


class FeedbackStorage:
    """JSONL-based feedback storage.

    Storage structure:
        ~/.legalbot/feedback/
        ├── metadata.json
        └── YYYY/
            └── MM/
                └── feedback-YYYY-MM-DD.jsonl
    """

    def __init__(self, feedback_dir: Path | str | None = None):
        self._feedback_dir = Path(feedback_dir) if feedback_dir else self._default_dir()
        self._ensure_dir()

    def _default_dir(self) -> Path:
        return Path.home() / ".legalbot" / "feedback"

    def _ensure_dir(self) -> None:
        self._feedback_dir.mkdir(parents=True, exist_ok=True)

    def _daily_file(self, date: datetime) -> Path:
        return (
            self._feedback_dir
            / date.strftime("%Y")
            / date.strftime("%m")
            / f"feedback-{date.strftime('%Y-%m-%d')}.jsonl"
        )

    async def append(self, record: FeedbackRecord) -> None:
        """Append a feedback record to the daily JSONL file."""
        file_path = self._daily_file(record.timestamp)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def query(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        feedback_type: str | None = None,
    ) -> list[FeedbackRecord]:
        """Query feedback records by date range and optional type filter."""
        records = []
        for file_path in self._feedback_dir.rglob("feedback-*.jsonl"):
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = FeedbackRecord.from_dict(json.loads(line))
                    if since and record.timestamp < since:
                        continue
                    if until and record.timestamp > until:
                        continue
                    if feedback_type and record.type != feedback_type:
                        continue
                    records.append(record)
        return records

    def list_records(
        self,
        limit: int = 20,
        feedback_type: str | None = None,
    ) -> list[FeedbackRecord]:
        """List the most recent feedback records."""
        all_records: list[FeedbackRecord] = []
        for file_path in sorted(
            self._feedback_dir.rglob("feedback-*.jsonl"), reverse=True
        ):
            with open(file_path, encoding="utf-8") as f:
                for line in reversed(f.readlines()):
                    if not line.strip():
                        continue
                    record = FeedbackRecord.from_dict(json.loads(line))
                    if feedback_type and record.type != feedback_type:
                        continue
                    all_records.append(record)
                    if len(all_records) >= limit:
                        return all_records
        return all_records
