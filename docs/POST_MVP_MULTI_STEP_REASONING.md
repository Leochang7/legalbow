# 多轮复杂推理 (Multi-Step Complex Legal Reasoning)

> LegalBot MVP 扩展功能实施计划

---

## 一、功能概述与目标

### 1.1 问题陈述

当前 LegalBot MVP 以单轮 RAG + 回答模式运行。复杂法律问题（如"用人单位拖欠工资且不签劳动合同如何维权"）需要**多轮 检索 → 分析 → 检索 循环**。

### 1.2 目标

1. **推理透明度**：每条结论均需引用具体法条
2. **多轮检索**：自动发起后续检索
3. **推理链展示**：最终答案格式化为显式逐步推理过程
4. **引用验证**：每个引用法条均需在检索结果中确认存在
5. **优雅降级**：达到限制时回退到单轮 RAG

### 1.3 非目标

- 不是通用思维链推理器（专为法律领域定制）
- 不替代现有 `LegalOrchestrator` 意图分类
- 不新增向量索引（复用 `LegalRetriever`）

---

## 二、架构设计

### 2.1 高层架构

```
用户查询
    │
    ▼
LegalOrchestrator.classify_intent()
    │
    ├─── INTENT_LEGAL_QUERY ──────► MultiStepLegalReasoner.reason()
    │                                      │
    │                                      ▼
    │                              第 1 步：初始检索
    │                                      │
    │                                      ▼
    │                              第 2 步：推理分析
    │                                      │
    │                                      ▼
    │                              第 3 步：[如有遗漏] 补充检索
    │                                      │
    │                                      ▼
    │                              第 4 步：综合 → 最终答案
    │
    └─── INTENT_GENERAL ──► 现有单轮流程
```

### 2.2 新增组件

| 组件                       | 位置                                        | 职责                              |
| ------------------------ | ----------------------------------------- | ------------------------------- |
| `MultiStepLegalReasoner` | `legalbot/agent/reasoner.py`               | 编排多步推理循环                        |
| `ReasoningChain`         | `legalbot/agent/reasoner.py`（dataclass）    | 每条推理步骤的不可变记录                    |
| `ReasoningTool`          | `legalbot/agent/tools/reasoner.py`         | 将 reasoner 暴露给 agent loop 的工具封装 |
| `legal-reasoning` 技能     | `legalbot/skills/legal-reasoning/SKILL.md` | LLM 驱动推理的提示指南                   |

---

## 三、关键数据结构

### 3.1 reasoner.py

**文件**：`legalbot/agent/reasoner.py`

```python
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ReasoningStep:
    """推理链中的单一步骤。"""
    step_id: int
    reasoning_type: Literal["retrieval", "analysis", "synthesis", "conclusion"]
    prompt: str
    law_retrieved: list  # 原始检索结果
    llm_reasoning: str
    citations: list[str]  # 例如：["《劳动合同法》第10条", "《民法典》第576条"]
    next_action: Literal["continue", "stop", "retrieve_more"]
    follow_up_query: str | None = None


@dataclass
class ReasoningChain:
    """法律问题的完整推理链。"""
    question: str
    steps: list[ReasoningStep] = field(default_factory=list)
    max_steps: int = 5
    iteration_count: int = 0

    def add_step(self, step: ReasoningStep) -> None:
        self.steps.append(step)
        self.iteration_count += 1

    def is_complete(self) -> bool:
        last = self.steps[-1] if self.steps else None
        return (
            len(self.steps) >= self.max_steps
            or (last is not None and last.next_action == "stop")
        )

    def to_display_string(self) -> str:
        lines = ["## 法律推理过程\n"]
        for step in self.steps:
            lines.append(f"### 第 {step.step_id} 步：{step.reasoning_type}")
            if step.llm_reasoning:
                lines.append(f"**推理**：{step.llm_reasoning}")
            if step.citations:
                lines.append(f"**引用**：{'；'.join(step.citations)}")
            if step.law_retrieved:
                lines.append(f"**检索到 {len(step.law_retrieved)} 条相关法规**")
            lines.append("")
        return "\n".join(lines)


INTENT_COMPLEX_LEGAL_QUERY = "complex_legal_query"
VALID_INTENTS = {INTENT_LEGAL_QUERY, INTENT_CONTRACT_REVIEW, INTENT_CASE_SEARCH,
                  INTENT_GENERAL, INTENT_COMPLEX_LEGAL_QUERY}

COMPLEXITY_CLASSIFICATION_PROMPT = """\
分析以下法律问题的复杂程度。

复杂度判断标准：
- simple: 仅需单条法律条文即可回答（如：法定婚龄是多少？）
- complex: 需要引用多条法律条文、进行逻辑推导、或涉及多个法律领域
  （如：用人单位拖欠工资且不签劳动合同如何维权？）

用户输入：{query}

返回类别：simple 或 complex

只返回类别名，不要解释。"""

REASONING_PROMPTS = {
    "initial_retrieval": """\
根据用户提出的法律问题，构造精确的检索 query。

用户问题：{question}

要求：
1. 识别问题涉及的法律领域
2. 提取核心法律概念，去除口语化表述
3. 识别是否需要同时检索多个相关法律

直接返回检索 query 列表，每行一个 query。用空格分隔关键词。""",

    "retrieval_response_analysis": """\
基于以下法律检索结果，分析法律问题。

检索结果：
{retrieved_laws}

用户问题：{question}

分析要求：
1. 列出每条检索结果与问题的相关性
2. 识别已找到的直接适用法条
3. 识别可能需要补充检索的相关法条

如发现需要补充检索的内容，在最后一行写：NEED_MORE_RETRIEVAL: [具体补充检索问题]""",

    "synthesis": """\
综合以下所有法律检索和分析，形成完整的法律分析结论。

用户问题：{question}

已确认的法律依据：
{confirmed_laws}

分析结论要求：
1. 清晰陈述法律结论
2. 每条结论必须附带具体法条引用（格式：《法律名》第X条）
3. 如涉及程序性权利（仲裁、诉讼）也要说明
4. 结论末尾附加风险提示和法律免责

用以下格式回答：

## 法律分析结论
[结论陈述]

## 法律依据
- [法条1]
- [法条2]
...

## 维权步骤
1. [步骤1]
2. [步骤2]
...

## 风险提示
[提示内容]

---
*以上分析基于当前知识库中的法律法规，仅供参考，不构成正式法律意见。*""",
}


class MultiStepLegalReasoner:
    """带思维链的多步法律推理引擎。"""

    def __init__(
        self,
        provider: Any,
        retriever: Any,
        max_steps: int = 5,
    ):
        self._provider = provider
        self._retriever = retriever
        self._max_steps = max_steps

    async def reason(
        self,
        question: str,
        law_area: str | None = None,
    ) -> ReasoningChain:
        chain = ReasoningChain(question=question, max_steps=self._max_steps)

        # 第 1 步：初始检索
        retrieval_results = await self._initial_retrieval(question, law_area)
        chain.add_step(ReasoningStep(
            step_id=1,
            reasoning_type="retrieval",
            prompt=question,
            law_retrieved=retrieval_results,
            llm_reasoning="已完成初始检索",
            citations=[],
            next_action="continue",
        ))

        # 第 2-N 步：分析 + 可能的后续检索
        current_results = retrieval_results
        step_num = 2

        while not chain.is_complete() and step_num <= self._max_steps:
            # 分析当前结果
            analysis_result = await self._analyze_retrieval(
                question, current_results
            )
            citations = analysis_result["citations"]
            next_action = analysis_result["next_action"]

            chain.add_step(ReasoningStep(
                step_id=step_num,
                reasoning_type="analysis",
                prompt="",
                law_retrieved=[],
                llm_reasoning=analysis_result["reasoning"],
                citations=citations,
                next_action=next_action,
            ))

            if next_action == "stop":
                break

            # 需要时进行补充检索
            follow_up_query = analysis_result.get("follow_up_query")
            if follow_up_query and step_num < self._max_steps:
                current_results = await self._retriever.retrieve(
                    query=follow_up_query,
                    law_area=law_area,
                    top_k=5,
                )
                chain.add_step(ReasoningStep(
                    step_id=step_num + 1,
                    reasoning_type="retrieval",
                    prompt=follow_up_query,
                    law_retrieved=current_results,
                    llm_reasoning=f"补充检索：{follow_up_query}",
                    citations=[],
                    next_action="continue",
                ))
                step_num += 2
            else:
                step_num += 1

        # 最终综合步骤
        if not chain.is_complete():
            synthesis_result = await self._synthesize(question, chain)
            chain.add_step(ReasoningStep(
                step_id=len(chain.steps) + 1,
                reasoning_type="synthesis",
                prompt="",
                law_retrieved=[],
                llm_reasoning=synthesis_result,
                citations=self._collect_all_citations(chain),
                next_action="stop",
            ))

        return chain

    async def _initial_retrieval(self, question: str, law_area: str | None):
        prompt = REASONING_PROMPTS["initial_retrieval"].format(question=question)
        response = await self._provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        queries = (response.content or question).strip().split("\n")
        results = []
        for q in queries[:3]:
            r = await self._retriever.retrieve(query=q.strip(), law_area=law_area, top_k=3)
            results.extend(r)
        return results[:5]

    async def _analyze_retrieval(self, question: str, retrieved: list) -> dict:
        laws_text = "\n".join(
            f"- {r.chunk.metadata.get('law_name', '')}{r.chunk.metadata.get('article_no', '')}: {r.chunk.text[:200]}"
            for r in retrieved
        )
        prompt = REASONING_PROMPTS["retrieval_response_analysis"].format(
            retrieved_laws=laws_text or "（无检索结果）",
            question=question,
        )
        response = await self._provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.content or ""
        citations = self._extract_citations(content)
        needs_more = "NEED_MORE_RETRIEVAL:" in content
        follow_up = None
        if needs_more:
            parts = content.split("NEED_MORE_RETRIEVAL:", 1)
            if len(parts) > 1:
                follow_up = parts[1].strip().split("\n")[0].strip()
        return {
            "reasoning": content[:1000],
            "citations": citations,
            "next_action": "retrieve_more" if needs_more else "stop",
            "follow_up_query": follow_up,
        }

    async def _synthesize(self, question: str, chain: ReasoningChain) -> str:
        all_citations = self._collect_all_citations(chain)
        confirmed_laws = "\n".join(f"- {c}" for c in all_citations) if all_citations else "（无确认法条）"
        prompt = REASONING_PROMPTS["synthesis"].format(
            question=question,
            confirmed_laws=confirmed_laws,
        )
        response = await self._provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.content or "综合分析完成，但未生成有效结论。"

    def _extract_citations(self, text: str) -> list[str]:
        import re
        pattern = r"《[^》]+》[^。，,\n]*?第[零一二三四五六七八九十百千\d]+条"
        matches = re.findall(pattern, text)
        return list(dict.fromkeys(matches))[:10]

    def _collect_all_citations(self, chain: ReasoningChain) -> list[str]:
        all_citations = []
        for step in chain.steps:
            all_citations.extend(step.citations)
        # 去重同时保持顺序
        seen = set()
        result = []
        for c in all_citations:
            if c not in seen:
                seen.add(c)
                result.append(c)
        return result
```

---

## 四、工具设计

### 4.1 MultiStepReasoningTool

**文件**：`legalbot/agent/tools/reasoner.py`

```python
from __future__ import annotations

from typing import Any

from legalbot.agent.tools.base import Tool, tool_parameters
from legalbot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema


@tool_parameters(
    tool_parameters_schema(
        query=StringSchema("法律问题"),
        max_steps=IntegerSchema(
            default=5, description="最大推理步数", minimum=1, maximum=10
        ),
        law_area=StringSchema(
            "法律领域过滤：民法/刑法/商法/劳动法/行政法等",
            nullable=True,
        ),
    )
)
class MultiStepReasoningTool(Tool):
    """多轮法律推理工具 — 对复杂法律问题进行链式推理"""

    def __init__(self, reasoner: Any, retriever: Any):
        self._reasoner = reasoner
        self._retriever = retriever

    @property
    def name(self) -> str:
        return "legal_multi_step_reasoning"

    @property
    def description(self) -> str:
        return (
            "对复杂法律问题进行多轮链式推理分析。"
            "每一步会检索相关法条、分析法律关系、识别遗漏点，"
            "最终生成包含完整推理链和法律引用的分析报告。"
        )

    @property
    def exclusive(self) -> bool:
        return True

    async def execute(
        self,
        query: str,
        max_steps: int = 5,
        law_area: str | None = None,
        **kwargs: Any,
    ) -> str:
        chain = await self._reasoner.reason(question=query, law_area=law_area)
        return chain.to_display_string()
```

---

## 五、集成到 LegalOrchestrator

### 5.1 新增意图分类

**文件**：`legalbot/agent/orchestrator.py`

```python
INTENT_COMPLEX_LEGAL_QUERY = "complex_legal_query"

# 更新 classify_intent() — 两阶段：
# 1. 分类为 INTENT_LEGAL_QUERY 或其他
# 2. 如果是 INTENT_LEGAL_QUERY，进一步分类为简单或复杂

async def classify_intent(self, query: str) -> str:
    # 第一阶段：基本意图分类
    prompt = INTENT_PROMPT.format(query=query)
    response = await self.provider.chat(messages=[{"role": "user", "content": prompt}])
    intent = self._parse_intent(response.content)

    # 第二阶段：法律查询的复杂度检查
    if intent == INTENT_LEGAL_QUERY:
        complex_prompt = COMPLEXITY_CLASSIFICATION_PROMPT.format(query=query)
        complex_response = await self.provider.chat(messages=[{"role": "user", "content": complex_prompt}])
        if "complex" in (complex_response.content or "").lower():
            return INTENT_COMPLEX_LEGAL_QUERY
    return intent
```

### 5.2 调度路由

```python
# 在 dispatch_sync() 中：
if intent == INTENT_COMPLEX_LEGAL_QUERY:
    return await self.reason_multi_step(query, context)

# 新方法：
async def reason_multi_step(self, query: str, context: dict | None = None) -> str:
    from legalbot.agent.reasoner import MultiStepLegalReasoner
    reasoner = MultiStepLegalReasoner(
        provider=self.provider,
        retriever=self._get_rag_retriever(),
    )
    chain = await reasoner.reason(question=query)
    return chain.to_display_string()

def _get_rag_retriever(self):
    if self._main_tools:
        rag_tool = self._main_tools.get("legal_rag_search")
        if rag_tool and hasattr(rag_tool, "_retriever"):
            return rag_tool._retriever
    available = self.subagents._build_available_tools()
    rag_tool = available.get("legal_rag_search")
    if rag_tool and hasattr(rag_tool, "_retriever"):
        return rag_tool._retriever
    raise RuntimeError("LegalRetriever not available for multi-step reasoning")
```

---

## 六、技能设计

### 6.1 legal-reasoning 技能

**文件**：`legalbot/skills/legal-reasoning/SKILL.md`

```markdown
---
name: legal-reasoning
description: 复杂法律问题的多步链式推理技能
always: false
---

## 复杂法律问题推理原则

当遇到需要引用多条法律条文、进行法律逻辑推导的复杂法律问题时，使用 `legal_multi_step_reasoning` 工具。

### 触发条件
复杂法律问题通常具有以下特征：
- 涉及多个法律主体（如用人单位+劳动者）
- 涉及多个法律领域（如劳动报酬 + 合同签订 + 社会保险）
- 需要进行法律逻辑推导（如：A条款 + B条款 = C权利）
- 问题包含"如何维权"、"能否主张"、"是否违法"等表述

### 推理步骤

**第一步：法律关系识别** — 明确法律关系类型
**第二步：法律依据检索** — 以法律概念为关键词检索
**第三步：法律适用分析** — 逐一比对事实构成与法律条文要件
**第四步：综合结论形成** — 整合各法律依据形成完整结论

### 工具使用

```python
legal_multi_step_reasoning(
    query="具体法律问题",
    max_steps=5,
    law_area="劳动法"
)
```

### 注意事项

1. 不要在一次回答中堆砌过多法条 — 按推理步骤逐步引用
2. 必须验证检索结果中的法条有效性
3. 必须结论末尾附加法律免责声明
```

---

## 七、实施步骤

### 第一阶段：核心基础设施（第 1 周）
- 1.1 创建 `legalbot/agent/reasoner.py`
- 1.2 实现 `ReasoningChain`、`ReasoningStep`
- 1.3 实现 `MultiStepLegalReasoner`
- 1.4 独立单元测试

### 第二阶段：工具集成（第 2 周）
- 2.1 创建 `legalbot/agent/tools/reasoner.py`
- 2.2 添加 `INTENT_COMPLEX_LEGAL_QUERY` 到编排器
- 2.3 添加 `reason_multi_step()` 方法
- 2.4 注册工具

### 第三阶段：技能（第 2-3 周）
- 3.1 创建 `legalbot/skills/legal-reasoning/SKILL.md`
- 3.2 根据测试优化提示词

### 第四阶段：安全护栏（第 3 周）
- 4.1 添加 `max_steps` 保护
- 4.2 添加引用验证
- 4.3 添加上下文溢出保护

### 第五阶段：测试（第 4 周）
- 4.1 集成测试
- 4.2 性能测试

---

## 八、文件变更摘要

### 新增文件

| 文件 | 用途 |
|------|---------|
| `legalbot/agent/reasoner.py` | `MultiStepLegalReasoner`、`ReasoningChain`、`ReasoningStep` |
| `legalbot/agent/tools/reasoner.py` | `MultiStepReasoningTool` |
| `legalbot/skills/legal-reasoning/SKILL.md` | 技能文档 |
| `docs/POST_MVP_MULTI_STEP_REASONING.md` | 本文档 |

### 修改文件

| 文件 | 变更 |
|------|---------|
| `legalbot/agent/orchestrator.py` | 添加 `INTENT_COMPLEX_LEGAL_QUERY`、`reason_multi_step()`，更新 `dispatch_sync()` |
| `legalbot/agent/tools/orchestrate.py` | 处理 `complex_legal_query` 意图 |
| `legalbot/agent/loop.py` | 注册 `MultiStepReasoningTool` |

---

## 九、风险与缓解

| 风险 | 缓解措施 |
|------|------------|
| 无限推理循环 | 硬性上限 `max_steps=5` |
| 上下文溢出 | 如果链超过限制则截断最旧步骤 |
| 引用幻觉 | 验证引用的文章是否出现在 `law_retrieved` 中 |
| 简单查询降级 | 两阶段复杂度分类 |
| LLM 非确定性 | 保守解析回退 |

---

## 十、测试计划

| 测试 | 预期结果 |
|------|----------------|
| `test_reasoning_chain_add_step` | iteration_count 递增 |
| `test_reasoning_chain_is_complete_at_max_steps` | 达到 max_steps 时为 True |
| `test_unverified_citation_is_filtered` | 引用不在检索结果中则被移除 |
| `test_simple_query_routes_to_single_turn` | 进入单轮 RAG |
| `test_complex_query_routes_to_multi_step` | 包含 "## 法律推理过程" |
| `test_final_answer_has_citations` | 最终步骤有引用 |
| `test_disclaimer_included` | 输出中包含"仅供参考" |