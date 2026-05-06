"""Case comparison tool for legal dispute analysis."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from legalbot.agent.tools.base import Tool, tool_parameters
from legalbot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema

if TYPE_CHECKING:
    from legalbot.rag.case_analyzer import CaseAnalyzer


CASE_COMPARE_PROMPT = """\
你是一个法律案例分析助手。用户输入了一个纠纷事实，请与已分析的相似案例进行对比，生成结构化对比表和适用性预测。

## 用户纠纷事实
---
{user_dispute}
---

## 已分析的相似案例
```json
{cases_json}
```

## 输出格式

### 1. 案例对比表（Markdown表格）
| 案号 | 争议焦点 | 裁判规则 | 适用法条 | 相似度 |
|------|---------|---------|---------|-------|

### 2. 适用性预测
- 最相似案例：[案号]
- 相似度：[High/Medium/Low]
- 关键相似点：[列出2-3个关键相似事实]
- 处理建议：[2-3条建议策略]
- 风险提示：[1-2个需要特别注意的风险]

## 相似度判断标准
- **High**：当事人关系相同、核心行为相同、损害结果类似
- **Medium**：纠纷类型相同、部分事实要素相同
- **Low**：仅属于同一法律领域但事实模式不同

## 要求
1. 对比表必须包含所有输入案例
2. 相似度判断需说明理由
3. 法条引用必须准确
4. 末尾必须附加免责声明：以上类案分析仅供参考，具体案件结果需结合全部事实和证据判断。
"""


@tool_parameters(
    tool_parameters_schema(
        dispute_facts=StringSchema(
            "用户描述的纠纷事实，应包含当事人、行为、结果等关键信息",
            min_length=10,
        ),
        law_area=StringSchema(
            "法律领域过滤：民法/刑法/商法/劳动法/行政法等",
            nullable=True,
        ),
        top_k=IntegerSchema(
            5,
            description="对比案例数量",
            minimum=3,
            maximum=10,
        ),
        required=["dispute_facts"],
    )
)
class CaseCompareTool(Tool):
    """案例对比分析工具 — 检索相似案例并生成结构化对比表."""

    name = "legal_case_compare"
    description = (
        "输入纠纷事实，检索知识库中的相似案例，"
        "生成结构化对比表（含案号、争议焦点、裁判规则、适用法条），"
        "并输出适用性预测和建议策略。"
    )

    def __init__(
        self,
        retriever: Any,
        case_analyzer: "CaseAnalyzer",
        llm_provider: Any,
        config: Any | None = None,
    ):
        self._retriever = retriever
        self._case_analyzer = case_analyzer
        self._llm_provider = llm_provider
        self._config = config or {}

    @property
    def read_only(self) -> bool:
        return True

    @property
    def exclusive(self) -> bool:
        return False

    async def execute(
        self,
        dispute_facts: str,
        law_area: str | None = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> str:
        # Step 1: Retrieve similar cases (filter by doc_type=case)
        results = await self._retriever.retrieve(
            query=dispute_facts,
            law_area=law_area,
            doc_type="case",
            top_k=top_k,
        )

        if not results or not results.top_k:
            return f"未检索到与「{dispute_facts[:30]}...」相关的案例。"

        # Step 2: Analyze each case
        analyzed_cases = await self._case_analyzer.analyze_batch(results.top_k)

        # Step 3: Generate comparison table
        comparison_output = await self._generate_comparison(dispute_facts, analyzed_cases)
        return comparison_output

    async def _generate_comparison(self, user_dispute: str, cases: list) -> str:
        cases_data = [c.to_dict() for c in cases]
        prompt = CASE_COMPARE_PROMPT.format(
            user_dispute=user_dispute,
            cases_json=json.dumps(cases_data, ensure_ascii=False, indent=2),
        )
        messages = [{"role": "user", "content": prompt}]
        response = await self._llm_provider.chat(messages=messages, temperature=0.1)
        return response.content or "对比分析生成失败。"
