"""上诉状 (Appeal) template."""

from typing import Any

from legalbot.document.templates.base import (
    DocumentSection,
    DocumentVariable,
    LegalDocumentTemplate,
)


class AppealTemplate(LegalDocumentTemplate):
    """上诉状 — Appeal"""

    @property
    def doc_type(self) -> str:
        return "appeal"

    @property
    def display_name(self) -> str:
        return "上诉状"

    @property
    def law_keywords(self) -> list[str]:
        return [
            "上诉状", "上诉人", "被上诉人", "上诉请求",
            "一审判决", "不服判决", "民事上诉", "中华人民共和国民事诉讼法",
        ]

    @property
    def required_variables(self) -> list[DocumentVariable]:
        return [
            DocumentVariable(
                key="appellant_name", label="上诉人姓名",
                description="提出上诉的一方当事人",
                example="张三",
            ),
            DocumentVariable(
                key="appellee_name", label="被上诉人姓名",
                description="一审中的对方当事人",
                example="李四",
            ),
            DocumentVariable(
                key="first_instance_case_no", label="一审案号",
                description="一审法院的案件编号",
                example="（2024）京01民初1234号",
            ),
            DocumentVariable(
                key="first_instance_judgment", label="一审判决内容",
                description="一审法院的判决结果",
                example="判令被告返还借款10万元",
            ),
            DocumentVariable(
                key="appeal_requests", label="上诉请求",
                description="上诉人希望二审法院支持什么",
                example="1. 撤销一审判决；2. 改判驳回原告全部诉讼请求...",
            ),
            DocumentVariable(
                key="appeal_reasons", label="上诉理由",
                description="不服一审判决的具体理由",
                example="一审认定事实不清，适用法律错误...",
            ),
        ]

    @property
    def optional_variables(self) -> list[DocumentVariable]:
        return [
            DocumentVariable(key="first_instance_court", label="一审法院", required=False),
            DocumentVariable(key="judgment_date", label="判决日期", required=False),
            DocumentVariable(key="new_evidence", label="新证据", required=False),
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

        appellant_name = variable_set.get("appellant_name", "（姓名）")
        appellee_name = variable_set.get("appellee_name", "（姓名）")
        first_instance_case_no = variable_set.get("first_instance_case_no", "（案号）")
        first_instance_court = variable_set.get("first_instance_court", "（法院）")
        judgment_date = variable_set.get("judgment_date", "（日期）")
        first_instance_judgment = variable_set.get("first_instance_judgment", "（一审判决内容）")
        appeal_requests = variable_set.get("appeal_requests", "（上诉请求）")
        appeal_reasons = variable_set.get("appeal_reasons", "（上诉理由）")
        new_evidence = variable_set.get("new_evidence", "（新证据说明）")

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
            "你是一名中国执业律师。请根据以下案件事实，起草一份完整的民事上诉状。\n\n"
            "## 案件事实\n"
            + self._format_case_facts(case_facts)
            + "\n\n"
            "## 相关法律条文（来自法律知识库检索）\n"
            + relevant_laws_text
            + missing_note
            + "\n\n"
            "## 上诉状要求\n"
            "请按以下格式生成完整的上诉状：\n\n"
            "### 上诉人信息\n"
            "上诉人：" + appellant_name + "\n"
            "被上诉人：" + appellee_name + "\n\n"
            "### 一审信息\n"
            "一审案号：" + first_instance_case_no + "\n"
            "一审法院：" + first_instance_court + "\n"
            "判决日期：" + judgment_date + "\n\n"
            "### 一审判决内容\n"
            + first_instance_judgment
            + "\n\n"
            "### 上诉请求\n"
            + appeal_requests
            + "\n\n"
            "### 上诉理由\n"
            + appeal_reasons
            + "\n\n"
            "### 新证据（如有）\n"
            + new_evidence
            + "\n\n"
            "### 此致\n"
            "此致\n"
            "{第二审法院名称}\n"
            "上诉人（签名）： -----------\n"
            "____年____月____日\n\n"
            "---\n"
            "要求：\n"
            "1. 上诉请求明确、具体\n"
            "2. 上诉理由论述充分，针对一审错误逐一反驳\n"
            "3. 法律依据引用准确\n"
            "4. 语言严谨规范，符合中国法律文书标准"
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
