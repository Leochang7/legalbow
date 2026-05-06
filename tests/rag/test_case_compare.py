"""Unit tests for case comparison module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from legalbot.rag.case_analyzer import CaseAnalyzer
from legalbot.rag.case_types import CaseCoreData, CaseCompareConfig


class TestCaseCoreData:
    def test_to_dict(self):
        case = CaseCoreData(
            case_no="(2021)沪01民终1234号",
            dispute_focus="一房二卖中买受人权利顺位认定",
            ruling_rule="出卖人将房屋所有权转移给后买受人时，前买受人可主张违约责任",
            applicable_laws=["《民法典》第209条", "《民法典》第224条"],
            source_chunk_id="chunk-001",
        )
        d = case.to_dict()
        assert d["case_no"] == "(2021)沪01民终1234号"
        assert d["dispute_focus"] == "一房二卖中买受人权利顺位认定"
        assert len(d["applicable_laws"]) == 2

    def test_from_dict(self):
        data = {
            "case_no": "(2020)京02民终5678号",
            "dispute_focus": "卖家欺诈认定",
            "ruling_rule": "卖家故意隐瞒房屋权利瑕疵应退还房款并赔偿",
            "applicable_laws": ["《民法典》第500条"],
            "source_chunk_id": "chunk-002",
        }
        case = CaseCoreData.from_dict(data)
        assert case.case_no == "(2020)京02民终5678号"
        assert case.ruling_rule == "卖家故意隐瞒房屋权利瑕疵应退还房款并赔偿"


class TestCaseAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_single_extracts_fields(self):
        provider = MagicMock()
        provider.chat = AsyncMock(return_value=MagicMock(
            content='{"case_no": "(2021)沪01民终1234号", "dispute_focus": "一房二卖", "ruling_rule": "前买受人可主张违约责任", "applicable_laws": ["《民法典》第209条"]}'
        ))

        chunk = MagicMock()
        chunk.chunk.id = "chunk-abc"
        chunk.chunk.text = "某案例文本..."

        analyzer = CaseAnalyzer(provider)
        result = await analyzer.analyze_single(chunk)

        assert result.case_no == "(2021)沪01民终1234号"
        assert result.dispute_focus == "一房二卖"
        assert result.source_chunk_id == "chunk-abc"

    @pytest.mark.asyncio
    async def test_analyze_single_handles_json_code_block(self):
        provider = MagicMock()
        provider.chat = AsyncMock(return_value=MagicMock(
            content='```json\n{"case_no": "(2021)沪01民终1234号", "dispute_focus": "一房二卖", "ruling_rule": "前买受人可主张违约责任", "applicable_laws": []}\n```'
        ))

        chunk = MagicMock()
        chunk.chunk.id = "chunk-abc"
        chunk.chunk.text = "某案例文本..."

        analyzer = CaseAnalyzer(provider)
        result = await analyzer.analyze_single(chunk)

        assert result.case_no == "(2021)沪01民终1234号"

    @pytest.mark.asyncio
    async def test_analyze_batch(self):
        provider = MagicMock()
        provider.chat = AsyncMock(return_value=MagicMock(
            content='{"case_no": "CASE-001", "dispute_focus": "测试焦点", "ruling_rule": "测试规则", "applicable_laws": []}'
        ))

        chunk1 = MagicMock()
        chunk1.chunk.id = "c1"
        chunk1.chunk.text = "文本1"

        chunk2 = MagicMock()
        chunk2.chunk.id = "c2"
        chunk2.chunk.text = "文本2"

        analyzer = CaseAnalyzer(provider)
        results = await analyzer.analyze_batch([chunk1, chunk2])

        assert len(results) == 2
        assert results[0].case_no == "CASE-001"


class TestCaseCompareConfig:
    def test_defaults(self):
        cfg = CaseCompareConfig()
        assert cfg.enable is True
        assert cfg.max_cases == 10
        assert cfg.top_k_default == 5
