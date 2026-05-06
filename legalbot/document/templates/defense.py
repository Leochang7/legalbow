"""答辩状 (Defense Statement) template."""

from typing import Any

from legalbot.document.templates.base import (
    DocumentSection,
    DocumentVariable,
    LegalDocumentTemplate,
)


class DefenseTemplate(LegalDocumentTemplate):
    """答辩状 — Defense Statement"""

    @property
    def doc_type(self) -> str:
        return "defense"

    @property
    def display_name(self) -> str:
        return "答辩状"

    @property
    def law_keywords(self) -> list[str]:
        return [
            "答辩状", "答辩人", "被答辩人", "答辩意见",
            "民事答辩", "抗辩", "反诉", "中华人民共和国民事诉讼法",
        ]

    @property
    def required_variables(self) -> list[DocumentVariable]:
        return [
            DocumentVariable(
                key="defendant_name", label="答辩人姓名",
                description="答辩人的全名（自然人）或单位名称（法人）",
                example="李四",
            ),
            DocumentVariable(
                key="plaintiff_name", label="被答辩人姓名",
                description="原告的全名（自然人）或单位名称（法人）",
                example="张三",
            ),
            DocumentVariable(
                key="case_no", label="案号",
                description="法院传票或起诉状上标注的案号",
                example="（2024）京01民初1234号",
            ),
            DocumentVariable(
                key="defense_points", label="答辩要点",
                description="针对原告诉讼请求的具体答辩意见（分条列出）",
                example="1. 原告主张的借款事实不存在；2. 即使借款存在，已过诉讼时效...",
            ),
            DocumentVariable(
                key="facts_and_reasons", label="事实与理由",
                description="阐述答辩的事实依据和法律依据",
                example="答辩人从未向原告借过任何款项，原告提交的借条系伪造...",
            ),
        ]

    @property
    def optional_variables(self) -> list[DocumentVariable]:
        return [
            DocumentVariable(key="defendant_address", label="答辩人地址", required=False),
            DocumentVariable(key="plaintiff_address", label="被答辩人地址", required=False),
            DocumentVariable(key="evidence_list", label="证据清单", required=False),
            DocumentVariable(key="counter_claims", label="反诉请求", required=False),
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

        defendant_name = variable_set.get("defendant_name", "（姓名）")
        defendant_address = variable_set.get("defendant_address", "（地址）")
        plaintiff_name = variable_set.get("plaintiff_name", "（姓名）")
        plaintiff_address = variable_set.get("plaintiff_address", "（地址）")
        case_no = variable_set.get("case_no", "（案号）")
        defense_points = variable_set.get("defense_points", "（答辩要点）")
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
            "你是一名中国执业律师。请根据以下案件事实，起草一份完整的民事答辩状。\n\n"
            "## 案件事实\n"
            + self._format_case_facts(case_facts)
            + "\n\n"
            "## 相关法律条文（来自法律知识库检索）\n"
            + relevant_laws_text
            + missing_note
            + "\n\n"
            "## 答辩状要求\n"
            "请按以下格式生成完整的答辩状：\n\n"
            "### 一、答辩人信息\n"
            "答辩人：" + defendant_name + "\n"
            "住所：" + defendant_address + "\n\n"
            "被答辩人：" + plaintiff_name + "\n"
            "住所：" + plaintiff_address + "\n\n"
            "### 二、案号\n"
            + case_no
            + "\n\n"
            "### 三、答辩意见\n"
            + defense_points
            + "\n\n"
            "### 四、事实与理由\n"
            + facts_and_reasons
            + "\n\n"
            "### 五、证据和证据来源\n"
            + evidence_list
            + "\n\n"
            "### 六、此致\n"
            "此致\n"
            "{法院名称}\n"
            "答辩人（签名）： -----------\n"
            "____年____月____日\n\n"
            "---\n"
            "要求：\n"
            "1. 针对原告的每项诉讼请求逐一答辩\n"
            "2. 法律依据引用准确\n"
            "3. 事实叙述客观、有证据支撑\n"
            "4. 语言严谨规范，符合中国法律文书标准\n"
            "5. 如有反诉请求，应另行列明"
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
