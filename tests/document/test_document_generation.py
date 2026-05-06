"""Tests for legal document generation."""

import pytest

from legalbot.document import CaseFacts, CaseFactsExtractor, DocumentDraftConfig
from legalbot.document.templates.agent_opinion import AgentOpinionTemplate
from legalbot.document.templates.appeal import AppealTemplate
from legalbot.document.templates.complaint import ComplaintTemplate
from legalbot.document.templates.defense import DefenseTemplate
from legalbot.document.templates.enforcement import EnforcementTemplate


class TestDocumentDraftConfig:
    def test_defaults(self):
        cfg = DocumentDraftConfig()
        assert cfg.enable is False
        assert cfg.max_laws_retrieved == 8
        assert "complaint" in cfg.enabled_types
        assert "defense" in cfg.enabled_types
        assert "agent_opinion" in cfg.enabled_types
        assert "appeal" in cfg.enabled_types
        assert "enforcement" in cfg.enabled_types

    def test_enable(self):
        cfg = DocumentDraftConfig(enable=True)
        assert cfg.enable is True


class TestCaseFacts:
    def test_to_dict(self):
        facts = CaseFacts(
            raw_text="张三借款给李四10万元",
            doc_type="complaint",
            parties={
                "plaintiff": {"name": "张三", "address": "北京"},
                "defendant": {"name": "李四", "address": "上海"},
            },
            monetary_amount=100000.0,
            case_type="民间借贷",
        )
        d = facts.to_dict()
        assert d["原告姓名"] == "张三"
        assert d["被告姓名"] == "李四"
        assert d["争议金额"] == "100000.0"
        assert d["案由"] == "民间借贷"

    def test_to_dict_empty(self):
        facts = CaseFacts(raw_text="test", doc_type="complaint")
        d = facts.to_dict()
        assert d["原告姓名"] == ""
        assert d["被告姓名"] == ""


class TestComplaintTemplate:
    def test_doc_type(self):
        t = ComplaintTemplate()
        assert t.doc_type == "complaint"
        assert t.display_name == "起诉状"

    def test_required_variables(self):
        t = ComplaintTemplate()
        keys = {v.key for v in t.required_variables}
        assert "plaintiff_name" in keys
        assert "defendant_name" in keys
        assert "litigation_requests" in keys
        assert "facts_and_reasons" in keys

    def test_law_keywords(self):
        t = ComplaintTemplate()
        assert "中华人民共和国民事诉讼法" in t.law_keywords

    def test_build_prompt(self):
        t = ComplaintTemplate()
        prompt = t.build_prompt(
            case_facts="张三借给李四10万元",
            relevant_laws=["《民法典》第188条"],
            variable_set={"plaintiff_name": "张三", "defendant_name": "李四"},
        )
        assert "起诉状" in prompt
        assert "张三" in prompt
        assert "李四" in prompt
        assert "《民法典》第188条" in prompt


class TestDefenseTemplate:
    def test_doc_type(self):
        t = DefenseTemplate()
        assert t.doc_type == "defense"
        assert t.display_name == "答辩状"

    def test_build_prompt(self):
        t = DefenseTemplate()
        prompt = t.build_prompt(
            case_facts="原告称被告借款不还",
            relevant_laws=["《民法典》第675条"],
            variable_set={"defendant_name": "李四", "plaintiff_name": "张三"},
        )
        assert "答辩状" in prompt
        assert "李四" in prompt


class TestAgentOpinionTemplate:
    def test_doc_type(self):
        t = AgentOpinionTemplate()
        assert t.doc_type == "agent_opinion"
        assert t.display_name == "代理词"


class TestAppealTemplate:
    def test_doc_type(self):
        t = AppealTemplate()
        assert t.doc_type == "appeal"
        assert t.display_name == "上诉状"

    def test_build_prompt(self):
        t = AppealTemplate()
        prompt = t.build_prompt(
            case_facts="一审判决被告还款",
            relevant_laws=[],
            variable_set={
                "appellant_name": "李四",
                "appellee_name": "张三",
                "first_instance_case_no": "（2024）京01民初123号",
                "first_instance_judgment": "判令李四还款10万",
                "appeal_requests": "撤销原判",
                "appeal_reasons": "事实认定错误",
            },
        )
        assert "上诉状" in prompt
        assert "李四" in prompt
        assert "撤销原判" in prompt


class TestEnforcementTemplate:
    def test_doc_type(self):
        t = EnforcementTemplate()
        assert t.doc_type == "enforcement"
        assert t.display_name == "执行申请书"


class TestTemplateFormatFacts:
    def test_complaint_format_dict(self):
        t = ComplaintTemplate()
        result = t._format_case_facts({"原告": "张三", "被告": "李四", "金额": "10万"})
        assert "原告" in result
        assert "张三" in result

    def test_complaint_format_str(self):
        t = ComplaintTemplate()
        result = t._format_case_facts("张三借给李四10万元")
        assert result == "张三借给李四10万元"

    def test_complaint_format_none(self):
        t = ComplaintTemplate()
        result = t._format_case_facts(None)
        assert "未提供" in result
