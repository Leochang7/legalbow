"""起诉状 (Complaint) template."""

from typing import Any

from legalbot.document.templates.base import (
    DocumentSection,
    DocumentVariable,
    LegalDocumentTemplate,
)


class ComplaintTemplate(LegalDocumentTemplate):
    """民事起诉状 — Complaint / Civil Action"""

    @property
    def doc_type(self) -> str:
        return "complaint"

    @property
    def display_name(self) -> str:
        return "起诉状"

    @property
    def law_keywords(self) -> list[str]:
        return [
            "民事起诉", "起诉状", "诉讼请求", "事实与理由",
            "民间借贷", "买卖合同", "合同纠纷", "给付之诉",
            "中华人民共和国民事诉讼法",
        ]

    @property
    def required_variables(self) -> list[DocumentVariable]:
        return [
            DocumentVariable(
                key="plaintiff_name", label="原告姓名",
                description="原告的全名（自然人）或单位名称（法人）",
                example="张三",
            ),
            DocumentVariable(
                key="defendant_name", label="被告姓名",
                description="被告的全名（自然人）或单位名称（法人）",
                example="李四",
            ),
            DocumentVariable(
                key="defendant_address", label="被告地址",
                description="被告的户籍地址或注册地址",
                example="北京市朝阳区某街道某号",
            ),
            DocumentVariable(
                key="case_type", label="纠纷类型",
                description="案件类型：民间借贷/买卖合同/租赁合同/侵权/其他",
                example="民间借贷",
            ),
            DocumentVariable(
                key="disputed_amount", label="争议金额（元）",
                description="涉及金额（仅数字，单位为人民币元）",
                example="100000",
            ),
            DocumentVariable(
                key="litigation_requests", label="诉讼请求",
                description="原告向法院提出的具体请求（分条列出）",
                example="1. 判令被告返还借款本金10万元；2. 判令被告支付利息...",
            ),
            DocumentVariable(
                key="facts_and_reasons", label="事实与理由",
                description="简述纠纷经过和原告主张的法律依据",
                example="2024年1月，被告因资金周转需要向原告借款10万元...",
            ),
        ]

    @property
    def optional_variables(self) -> list[DocumentVariable]:
        return [
            DocumentVariable(key="plaintiff_address", label="原告地址", required=False),
            DocumentVariable(key="plaintiff_phone", label="原告电话", required=False),
            DocumentVariable(key="defendant_phone", label="被告电话", required=False),
            DocumentVariable(key="contract_date", label="合同/借款日期", required=False),
            DocumentVariable(key="evidence_list", label="证据清单", required=False),
            DocumentVariable(key="plaintiff_id_number", label="原告身份证号", required=False),
            DocumentVariable(key="defendant_id_number", label="被告身份证号", required=False),
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

        plaintiff_name = variable_set.get("plaintiff_name", "（姓名）")
        plaintiff_address = variable_set.get("plaintiff_address", "（地址）")
        plaintiff_phone = variable_set.get("plaintiff_phone", "（电话）")
        defendant_name = variable_set.get("defendant_name", "（姓名）")
        defendant_address = variable_set.get("defendant_address", "（地址）")
        defendant_phone = variable_set.get("defendant_phone", "（电话）")
        litigation_requests = variable_set.get("litigation_requests", "（诉讼请求内容）")
        facts_and_reasons = variable_set.get("facts_and_reasons", "（事实与理由）")
        evidence_list = variable_set.get("evidence_list", "（证据清单）")

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
            "你是一名中国执业律师。请根据以下案件事实，起草一份完整的民事起诉状。\n\n"
            "## 案件事实\n"
            + self._format_case_facts(case_facts)
            + "\n\n"
            "## 相关法律条文（来自法律知识库检索）\n"
            + relevant_laws_text
            + missing_note
            + "\n\n"
            "## 起诉状要求\n"
            "请按以下格式生成完整的起诉状，包含所有必要条款：\n\n"
            "### 一、当事人信息\n"
            "原告：" + plaintiff_name + "\n"
            "住所：" + plaintiff_address + "\n"
            "电话：" + plaintiff_phone + "\n\n"
            "被告：" + defendant_name + "\n"
            "住所：" + defendant_address + "\n"
            "电话：" + defendant_phone + "\n\n"
            "### 二、诉讼请求\n"
            + litigation_requests
            + "\n\n"
            "### 三、事实与理由\n"
            + facts_and_reasons
            + "\n\n"
            "### 四、证据和证据来源\n"
            + evidence_list
            + "\n\n"
            "### 五、此致\n"
            "此致\n"
            "{法院名称}\n"
            "具状人（签名）： -----------\n"
            "原告：" + plaintiff_name + "（签名）\n"
            "____年____月____日\n\n"
            "---\n"
            "要求：\n"
            "1. 法律条文引用准确，使用《法律全称》第X条格式\n"
            "2. 诉讼请求明确、具体、可执行\n"
            "3. 事实叙述客观、连贯、有证据支撑\n"
            "4. 语言严谨规范，符合中国法律文书标准\n"
            "5. 事实与理由部分应充分论述法律关系和被告过错"
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
