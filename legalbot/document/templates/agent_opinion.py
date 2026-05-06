"""代理词 (Agent Opinion / Closing Argument) template."""

from typing import Any

from legalbot.document.templates.base import (
    DocumentSection,
    DocumentVariable,
    LegalDocumentTemplate,
)


class AgentOpinionTemplate(LegalDocumentTemplate):
    """代理词 — Agent Opinion / Closing Argument"""

    @property
    def doc_type(self) -> str:
        return "agent_opinion"

    @property
    def display_name(self) -> str:
        return "代理词"

    @property
    def law_keywords(self) -> list[str]:
        return [
            "代理词", "诉讼代理人", "代理意见", "庭审意见",
            "民事代理", "辩论意见", "中华人民共和国民事诉讼法",
        ]

    @property
    def required_variables(self) -> list[DocumentVariable]:
        return [
            DocumentVariable(
                key="agent_name", label="代理人姓名",
                description="诉讼代理人的姓名",
                example="王律师",
            ),
            DocumentVariable(
                key="case_no", label="案号",
                description="法院案件编号",
                example="（2024）京01民初1234号",
            ),
            DocumentVariable(
                key="client_name", label="委托人姓名",
                description="代理一方的当事人姓名",
                example="张三",
            ),
            DocumentVariable(
                key="opponent_name", label="对方当事人",
                description="对方当事人的姓名",
                example="李四",
            ),
            DocumentVariable(
                key="case_type", label="案件类型",
                description="案件类型",
                example="民间借贷纠纷",
            ),
            DocumentVariable(
                key="opinion_content", label="代理意见",
                description="代理人的核心观点和论证",
                example="一、关于借贷关系是否成立...\n二、关于诉讼时效...",
            ),
        ]

    @property
    def optional_variables(self) -> list[DocumentVariable]:
        return [
            DocumentVariable(key="court_name", label="法院名称", required=False),
            DocumentVariable(key="hearing_date", label="开庭日期", required=False),
            DocumentVariable(key="key_evidence", label="关键证据", required=False),
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

        agent_name = variable_set.get("agent_name", "（代理人）")
        client_name = variable_set.get("client_name", "（委托人）")
        case_no = variable_set.get("case_no", "（案号）")
        court_name = variable_set.get("court_name", "（法院名称）")
        hearing_date = variable_set.get("hearing_date", "（日期）")
        opinion_content = variable_set.get("opinion_content", "（代理意见内容）")
        key_evidence = variable_set.get("key_evidence", "（关键证据说明）")

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
            "你是一名中国执业律师。请根据以下案件事实，起草一份完整的代理词（庭审代理意见）。\n\n"
            "## 案件事实\n"
            + self._format_case_facts(case_facts)
            + "\n\n"
            "## 相关法律条文（来自法律知识库检索）\n"
            + relevant_laws_text
            + missing_note
            + "\n\n"
            "## 代理词要求\n"
            "请按以下格式生成完整的代理词：\n\n"
            "### 代理词\n"
            "审判长、审判员：\n\n"
            + agent_name + "作为" + client_name + "的诉讼代理人，就本案发表如下代理意见：\n\n"
            "### 一、案件基本情况\n"
            "案号：" + case_no + "\n"
            "法院：" + court_name + "\n"
            "开庭日期：" + hearing_date + "\n\n"
            "### 二、代理意见\n"
            + opinion_content
            + "\n\n"
            "### 三、关键证据\n"
            + key_evidence
            + "\n\n"
            "### 四、结论\n"
            "综上所述，恳请法院依法支持委托人的诉讼请求/答辩意见。\n\n"
            "此致\n"
            + court_name + "（法院名称）\n"
            "代理人：" + agent_name + "\n"
            "____年____月____日\n\n"
            "---\n"
            "要求：\n"
            "1. 逻辑清晰，层层递进\n"
            "2. 事实陈述与法律论证相结合\n"
            "3. 引用法条准确有力\n"
            "4. 语言严谨、专业、富有说服力\n"
            "5. 符合中国法庭代理词的规范格式"
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
