"""Legal audit logger — async JSONL-based audit trail for legal AI events."""

from __future__ import annotations

import asyncio
import json
import re
import hashlib
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger


class LegalEventType(str, Enum):
    """Legal event types tracked by the audit logger."""

    LEGAL_QUERY = "legal_query"
    DOCUMENT_DRAFT = "document_draft"
    CASE_COMPARE = "case_compare"
    DEBATE = "debate"
    CONTRACT_REVIEW = "contract_review"
    SYSTEM_MESSAGE = "system_message"


# Patterns for PII detection (minimal set)
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("id_card", re.compile(r"\b\d{17}[\dXx]\b")),
    ("phone_cn", re.compile(r"\b1[3-9]\d{9}\b")),
    ("email", re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")),
]


def _mask_pii(text: str) -> str:
    """Mask potential PII in text."""
    masked = text
    for label, pattern in _PII_PATTERNS:
        if label == "id_card":
            masked = pattern.sub(lambda m: f"{m.group()[:6]}****{m.group()[-4:]}", masked)
        elif label == "phone_cn":
            masked = pattern.sub(lambda m: f"{m.group()[:3]}****{m.group()[-4:]}", masked)
        elif label == "email":
            masked = pattern.sub(
                lambda m: m.group()[0] + "***@" + m.group().split("@")[1], masked
            )
    return masked


def _compute_hash(record: dict[str, Any]) -> str:
    """Compute a truncated SHA256 hash for tamper detection."""
    content = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _write_line(path: Path, line: str) -> None:
    """Synchronous file write (runs in thread pool)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def _read_all_lines(path: Path) -> list[str]:
    """Synchronous file read (runs in thread pool)."""
    with open(path, encoding="utf-8") as f:
        return f.readlines()


class LegalAuditLogger:
    """Async audit logger for legal AI events.

    Writes one JSON record per line to daily JSONL files under ``audit_dir``.
    Each record includes a truncated SHA256 hash for integrity verification.
    Uses asyncio.to_thread() for non-blocking file I/O.
    """

    def __init__(
        self,
        audit_dir: str = "~/.legalbot/audit",
        retention_days: int = 90,
        pii_masking: bool = True,
    ):
        self._audit_dir = Path(audit_dir).expanduser()
        self._retention_days = retention_days
        self._pii_masking = pii_masking
        self._lock: asyncio.Lock | None = None
        self._initialized = False

    def _init(self) -> None:
        if self._initialized:
            return
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialized = True

    def _today_file(self) -> Path:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._audit_dir / f"{today}.jsonl"

    def _parse_date_from_filename(self, path: Path) -> datetime | None:
        try:
            return datetime.strptime(path.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    async def log(
        self,
        event_type: str,
        session_id: str,
        channel: str,
        query: dict[str, Any],
        response: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        user_id: str = "anonymous",
    ) -> str:
        """Write a legal event to the audit log. Returns the event_id."""
        self._init()
        event_id = str(uuid4())
        ts = datetime.now(timezone.utc).isoformat()

        # Mask PII in query and response content
        def mask_content(obj: Any) -> Any:
            if isinstance(obj, str) and self._pii_masking:
                return _mask_pii(obj)
            if isinstance(obj, dict):
                return {k: mask_content(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [mask_content(i) for i in obj]
            return obj

        masked_query = mask_content(query)
        masked_response = mask_content(response)

        record: dict[str, Any] = {
            "event_id": event_id,
            "timestamp": ts,
            "event_type": event_type,
            "session_id": session_id,
            "channel": channel,
            "user_id": user_id,
            "query": masked_query,
            "response": masked_response,
            "metadata": mask_content(metadata) if metadata else {},
        }
        record["_hash"] = _compute_hash(record)

        line = json.dumps(record, ensure_ascii=False) + "\n"
        await asyncio.to_thread(_write_line, self._today_file(), line)

        logger.debug("Audit log written: {} event={}", event_id, event_type)
        return event_id

    async def query(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        event_type: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit logs with optional filters.

        Dates are in ISO format (YYYY-MM-DD). Returns newest first.
        """
        self._init()
        start = (
            datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
            if start_date
            else datetime.min.replace(tzinfo=timezone.utc)
        )
        end = (
            datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc) + timedelta(days=1)
            if end_date
            else datetime.max.replace(tzinfo=timezone.utc)
        )

        results: list[tuple[datetime, dict[str, Any]]] = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)

        for file_path in sorted(self._audit_dir.glob("*.jsonl"), reverse=True):
            file_date = self._parse_date_from_filename(file_path)
            if file_date is None or file_date < cutoff:
                continue
            if file_date > end:
                continue

            lines = await asyncio.to_thread(_read_all_lines, file_path)
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                record_date = datetime.fromisoformat(record["timestamp"]).replace(
                    tzinfo=timezone.utc
                )
                if record_date < start or record_date > end:
                    continue
                if event_type and record.get("event_type") != event_type:
                    continue
                if session_id and record.get("session_id") != session_id:
                    continue

                results.append((record_date, record))
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break

        results.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in results]

    async def cleanup_old_logs(self) -> int:
        """Delete log files older than retention_days. Returns count of deleted files."""
        self._init()
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        deleted = 0

        for file_path in self._audit_dir.glob("*.jsonl"):
            file_date = self._parse_date_from_filename(file_path)
            if file_date is not None and file_date < cutoff:
                await asyncio.to_thread(file_path.unlink)
                deleted += 1
                logger.info("Deleted old audit log: {}", file_path.name)

        return deleted

    async def verify_integrity(self, event_id: str | None = None) -> dict[str, Any]:
        """Verify hash integrity of audit records.

        If event_id is given, verify that specific record.
        Otherwise, spot-check 10 random recent records.
        Returns dict with 'checked', 'valid', and 'corrupted' keys.
        """
        self._init()
        import random

        records_to_check: list[tuple[Path, int]] = []

        for file_path in sorted(self._audit_dir.glob("*.jsonl"), reverse=True):
            lines = await asyncio.to_thread(_read_all_lines, file_path)
            for i, line in enumerate(lines, 1):
                if line.strip():
                    records_to_check.append((file_path, i))
                if len(records_to_check) >= 500:
                    break
            if len(records_to_check) >= 500:
                break

        to_check: list[tuple[Path, int]]
        if event_id:
            to_check = [(r[0], r[1]) for r in records_to_check if r[0].name == event_id]
            if not to_check:
                return {"checked": 0, "valid": 0, "corrupted": [], "note": "event_id not found"}
        else:
            to_check = random.sample(records_to_check, min(10, len(records_to_check)))

        valid = 0
        corrupted: list[str] = []

        for file_path, line_no in to_check:
            lines = await asyncio.to_thread(_read_all_lines, file_path)
            if line_no <= len(lines):
                line = lines[line_no - 1].strip()
                try:
                    record = json.loads(line)
                    stored_hash = record.pop("_hash", None)
                    expected = _compute_hash(record)
                    if stored_hash == expected:
                        valid += 1
                    else:
                        corrupted.append(record.get("event_id", "unknown"))
                except Exception as e:
                    corrupted.append(f"parse_error:{file_path.name}:{line_no}:{e}")

        return {"checked": len(to_check), "valid": valid, "corrupted": corrupted}
