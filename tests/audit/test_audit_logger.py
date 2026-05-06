"""Tests for LegalAuditLogger and LegalEventType."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from legalbot.audit.logger import (
    LegalAuditLogger,
    LegalEventType,
    _compute_hash,
    _mask_pii,
)


class TestMaskPII:
    """Tests for PII masking utility."""

    def test_id_card_masking(self):
        # 18-digit Chinese ID (17 digits + checksum digit X)
        text = "身份证号11010119900101123X"
        masked = _mask_pii(text)
        assert "110101" in masked
        assert "****" in masked
        assert "19900101" not in masked
        assert "123X" in masked

    def test_phone_masking(self):
        text = "手机13812345678"
        masked = _mask_pii(text)
        assert "138" in masked
        assert "****" in masked
        assert "5678" in masked  # last 4 digits preserved

    def test_email_masking(self):
        text = "邮箱zhangsan@example.com"
        masked = _mask_pii(text)
        assert "z" in masked
        assert "***@" in masked
        assert "example.com" in masked

    def test_no_pii(self):
        text = "这是一段普通法律文本，不包含个人信息"
        masked = _mask_pii(text)
        assert masked == text

    def test_mixed_pii(self):
        text = "张三，身份证11010119900101123X，电话13812345678，邮箱test@example.com"
        masked = _mask_pii(text)
        assert "张三" in masked  # name preserved
        assert "110101" in masked  # first 6 of ID preserved
        assert "123X" in masked  # last 3 of ID + checksum preserved
        assert "138" in masked  # first 3 of phone preserved
        assert "5678" in masked  # last 4 of phone preserved
        assert "t***@" in masked  # email masked: first char + ***@


class TestComputeHash:
    """Tests for hash computation."""

    def test_hash_deterministic(self):
        record = {"event_id": "abc123", "timestamp": "2026-04-19T12:00:00Z"}
        h1 = _compute_hash(record)
        h2 = _compute_hash(record)
        assert h1 == h2

    def test_hash_changes_with_content(self):
        r1 = {"event_id": "abc123", "data": "value1"}
        r2 = {"event_id": "abc123", "data": "value2"}
        assert _compute_hash(r1) != _compute_hash(r2)


class TestLegalAuditLogger:
    """Tests for LegalAuditLogger."""

    @pytest.fixture
    def audit_dir(self, tmp_path: Path) -> str:
        return str(tmp_path / "audit")

    @pytest.fixture
    def logger(self, audit_dir: str) -> LegalAuditLogger:
        return LegalAuditLogger(audit_dir=audit_dir, retention_days=90, pii_masking=True)

    @pytest.mark.asyncio
    async def test_log_writes_jsonl(self, logger: LegalAuditLogger):
        event_id = await logger.log(
            event_type=LegalEventType.LEGAL_QUERY,
            session_id="cli:test",
            channel="cli",
            query={"original_text": "民间借贷纠纷如何起诉？"},
            response={"final_content": "根据《民法典》第675条...", "tools_called": [], "citations": [], "disclaimer_shown": True},
            metadata={"model": "test-model"},
        )
        assert event_id is not None
        assert len(event_id) == 36  # UUID format

        files = list(Path(logger._audit_dir).glob("*.jsonl"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert event_id in content
        assert "legal_query" in content

    @pytest.mark.asyncio
    async def test_log_masks_pii(self, logger: LegalAuditLogger):
        await logger.log(
            event_type=LegalEventType.LEGAL_QUERY,
            session_id="cli:test",
            channel="cli",
            query={"original_text": "身份证号110101199001011234，电话13812345678"},
            response={"final_content": "测试回复"},
            metadata={},
        )
        files = list(Path(logger._audit_dir).glob("*.jsonl"))
        content = files[0].read_text(encoding="utf-8")
        assert "110101199" not in content
        assert "1381234" not in content
        assert "****" in content

    @pytest.mark.asyncio
    async def test_query_returns_results(self, logger: LegalAuditLogger):
        await logger.log(
            event_type=LegalEventType.DOCUMENT_DRAFT,
            session_id="cli:test",
            channel="cli",
            query={"original_text": "帮我写起诉状"},
            response={"final_content": "起诉状草稿..."},
            metadata={},
        )
        results = await logger.query(limit=10)
        assert len(results) >= 1
        assert results[0]["event_type"] == "document_draft"

    @pytest.mark.asyncio
    async def test_query_filter_by_event_type(self, logger: LegalAuditLogger):
        await logger.log(
            event_type=LegalEventType.DEBATE,
            session_id="cli:test",
            channel="cli",
            query={"original_text": "辩论：借贷纠纷"},
            response={"final_content": "辩论结果"},
            metadata={},
        )
        await logger.log(
            event_type=LegalEventType.LEGAL_QUERY,
            session_id="cli:test",
            channel="cli",
            query={"original_text": "法律问题"},
            response={"final_content": "回复"},
            metadata={},
        )
        results = await logger.query(event_type="debate", limit=10)
        assert all(r["event_type"] == "debate" for r in results)

    @pytest.mark.asyncio
    async def test_query_filter_by_session_id(self, logger: LegalAuditLogger):
        await logger.log(
            event_type=LegalEventType.LEGAL_QUERY,
            session_id="cli:session_a",
            channel="cli",
            query={"original_text": "A"},
            response={"final_content": "A回复"},
            metadata={},
        )
        await logger.log(
            event_type=LegalEventType.LEGAL_QUERY,
            session_id="cli:session_b",
            channel="cli",
            query={"original_text": "B"},
            response={"final_content": "B回复"},
            metadata={},
        )
        results = await logger.query(session_id="cli:session_a", limit=10)
        assert all(r["session_id"] == "cli:session_a" for r in results)

    @pytest.mark.asyncio
    async def test_query_empty_when_no_match(self, logger: LegalAuditLogger):
        await logger.log(
            event_type=LegalEventType.LEGAL_QUERY,
            session_id="cli:test",
            channel="cli",
            query={"original_text": "测试"},
            response={"final_content": "回复"},
            metadata={},
        )
        results = await logger.query(event_type="nonexistent_type", limit=10)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_cleanup_old_logs(self, audit_dir: str, tmp_path: Path):
        # Create a logger with 1-day retention
        logger = LegalAuditLogger(audit_dir=audit_dir, retention_days=1)

        # Create the audit directory (logger doesn't create it until first log call)
        Path(audit_dir).mkdir(parents=True, exist_ok=True)

        # Create a fake old file
        old_file = Path(audit_dir) / "2020-01-01.jsonl"
        old_file.write_text('{"event_id":"test","timestamp":"2020-01-01T00:00:00Z"}\n', encoding="utf-8")

        # Create a fake today file
        today_file = Path(audit_dir) / "2026-04-19.jsonl"
        today_file.write_text('{"event_id":"test2","timestamp":"2026-04-19T00:00:00Z"}\n', encoding="utf-8")

        deleted = await logger.cleanup_old_logs()
        assert deleted == 1
        assert old_file.exists() is False
        assert today_file.exists() is True

    @pytest.mark.asyncio
    async def test_verify_integrity_valid(self, logger: LegalAuditLogger):
        await logger.log(
            event_type=LegalEventType.LEGAL_QUERY,
            session_id="cli:test",
            channel="cli",
            query={"original_text": "测试"},
            response={"final_content": "回复"},
            metadata={},
        )
        result = await logger.verify_integrity()
        assert result["valid"] >= 1
        assert result["corrupted"] == []

    @pytest.mark.asyncio
    async def test_verify_integrity_tampered_record(self, audit_dir: str, logger: LegalAuditLogger):
        await logger.log(
            event_type=LegalEventType.LEGAL_QUERY,
            session_id="cli:test",
            channel="cli",
            query={"original_text": "测试"},
            response={"final_content": "回复"},
            metadata={},
        )
        files = list(Path(audit_dir).glob("*.jsonl"))
        content = files[0].read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        # Tamper with the record
        tampered = lines[0].replace('"original_text": "测试"', '"original_text": "篡改"')
        files[0].write_text(tampered + "\n", encoding="utf-8")

        result = await logger.verify_integrity()
        assert result["corrupted"] != []


class TestLegalEventType:
    """Tests for LegalEventType enum."""

    def test_all_event_types_exist(self):
        assert LegalEventType.LEGAL_QUERY.value == "legal_query"
        assert LegalEventType.DOCUMENT_DRAFT.value == "document_draft"
        assert LegalEventType.CASE_COMPARE.value == "case_compare"
        assert LegalEventType.DEBATE.value == "debate"
        assert LegalEventType.CONTRACT_REVIEW.value == "contract_review"
        assert LegalEventType.SYSTEM_MESSAGE.value == "system_message"
