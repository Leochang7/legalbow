"""执行申请书 (Enforcement Application) template."""

from typing import Any

from legalbot.document.templates.base import (
    DocumentSection,
    DocumentVariable,
    LegalDocumentTemplate,
)


class EnforcementTemplate(LegalDocumentTemplate):
    """执行申请书 — Enforcement Application"""

    @property
    def doc_type(self) -> str:
        return "enforcement"

    @property
    def display_name(self) -> str:
        return "执行申请书"

    @property
    def law_keywords(self) -> list[str]:
        return [
            "执行申请", "强制执行", "执行依据", "执行标的",
            "民事执行", "中华人民共和国民事诉讼法",
            "最高人民法院关于人民法院执行工作若干问题的规定",
        ]

    @property
    def required_variables(self) -> list[DocumentVariable]:
        return [
            DocumentVariable(
                key="enforcement_applicant", label="申请执行人姓名",
                description="已胜诉的一方当事人",
                example="张三",
            ),
            DocumentVariable(
                key="respondent", label="被执行人姓名",
                description="需要履行判决义务的一方",
                example="李四",
            ),
            DocumentVariable(
                key="original_case_no", label="原案案号",
                description="已生效判决的案件编号",
                example="（2024）京01民初1234号",
            ),
            DocumentVariable(
                key="judgment_summary", label="判决内容摘要",
                description="生效判决的主要内容",
                example="判令被告返还借款本金10万元及利息",
            ),
            DocumentVariable(
                key="enforcement_requests", label="执行请求",
                description="申请执行的具体内容",
                example="1. 冻结被执行人银行账户；2. 查封被执行人房产...",
            ),
        ]

    @property
    def optional_variables(self) -> list[DocumentVariable]:
        return [
            DocumentVariable(key="respondent_address", label="被执行人地址", required=False),
            DocumentVariable(key="respondent_assets", label="被执行人财产线索", required=False),
            DocumentVariable(key="enforcement_court", label="执行法院", required=False),
        ]

    def build_prompt(
        self,
        case_facts: dict[str, Any],
        relevant_laws: list[str],
        variable_set: dict[str, Any],
    ) -> str:
        relevant_laws_text = "\n".join(
            " - " + law for law in relevant_laws
        ) if relevant_laws else "（未检索到相关法条）"

        enforcement_applicant = variable_set.get("enforcement_applicant", "（姓名）")
        respondent = variable_set.get("respondent", "（姓名）")
        respondent_address = variable_set.get("respondent_address", "（地址）")
        respondent_assets = variable_set.get("respondent_assets", "（如有）")
        original_case_no = variable_set.get("original_case_no", "（案号）")
        judgment_summary = variable_set.get("judgment_summary", "（判决内容摘要）")
        enforcement_requests = variable_set.get("enforcement_requests", "（执行请求）")
        enforcement_court = variable_set.get("enforcement_court", "（执行法院）")

        missing_vars = [
            v.label for v in self.required_variables
            if v.key not in variable_set or not variable_set[v.key]
        ]
        missing_note = (
            "\n注意：以下必填变量未提供，请基于常识推断填写：" + ", ".join(missing_vars)
            if missing_vars
            else ""
        )

        return (
            "你是一名中国执业律师。请根据以下案件事实，起草一份完整的执行申请书。\n\n"
            "## 案件事实\n"
            + self._format_case_facts(case_facts)
            + "\n\n"
            "## 相关法律条文（来自法律知识库检索）\n"
            + relevant_laws_text
            + missing_note
            + "\n\n"
            "## 执行申请书要求\n"
            "请按以下格式生成完整的执行申请书：\n\n"
            "### 申请执行人信息\n"
            "申请执行人：" + enforcement_applicant + "\n\n"
            "### 被执行人信息\n"
            "被执行人：" + respondent + "\n"
            "住所：" + respondent_address + "\n"
            "财产线索：" + respondent_assets + "\n\n"
            "### 执行依据\n"
            "原案案号：" + original_case_no + "\n\n"
            "判决/裁定内容：\n"
            + judgment_summary
            + "\n\n"
            "### 执行请求\n"
            + enforcement_requests
            + "\n\n"
            "### 此致\n"
            "此致\n"
            + enforcement_court + "\n"
            "申请执行人（签名）： -----------\n"
            "____年____月____日\n\n"
            "---\n"
            "要求：\n"
            "1. 执行请求具体、明确、可操作\n"
            "2. 提供被执行人准确的财产线索（如有）\n"
            "3. 写明执行法院（一般为作出生效判决的法院或被执行人住所地法院）\n"
            "4. 语言严谨规范，符合中国法律文书标准\n"
            "5. 根据《民事诉讼法》规定，生效法律文书申请执行的时效为两年"
        )

    @staticmethod
    def _format_case_facts(facts: Any) -> str:
        if not facts:
            return "（未提供具体事实）"
        if isinstance(facts, str):
            return facts
        lines = []
        if isinstance(facts, dict):
            for key, value in facts.items():
                if value:
                    lines.append(" - " + str(key) + "：" + str(value))
        return "\n".join(lines) if lines else "（未提供具体事实）"
