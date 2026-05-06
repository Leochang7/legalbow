# 案例对比分析 (Case Comparison Analysis)

> LegalBot MVP 扩展功能实施计划

---

## 一、功能概述与目标

### 1.1 功能描述

案例对比分析功能允许用户输入一个法律纠纷或事实描述，从 RAG 知识库中检索相似案例，并生成一份结构化对比表，提取关键司法裁判规则。

### 1.2 用户流程

```
用户输入（纠纷事实）
       │
       ▼
┌─────────────────────────────────────┐
│  legal_case_compare 工具             │
│                                     │
│  1. 调用 LegalRetriever             │
│     参数 doc_type="case"             │
│                                     │
│  2. 检索 top_k 个相似案例           │
│                                     │
│  3. LLM 生成对比表                  │
│                                     │
│  4. 返回：                          │
│     - 对比表                        │
│     - 适用性预测                     │
└─────────────────────────────────────┘
```

### 1.3 目标

- 从半结构化案例文档中提取结构化对比数据
- 为用户的纠纷生成可执行的"适用性预测"
- 保持可追溯性：每个表格单元格都能映射回源案例文本

---

## 二、架构设计

### 2.1 新增模块结构

```
D:\workspace\legalbot\
├── legalbot/
│   ├── agent/
│   │   └── tools/
│   │       └── case_compare.py     # CaseCompareTool
│   └── rag/
│       ├── case_analyzer.py        # CaseAnalyzer — LLM 提取
│       └── case_types.py           # 案例数据模型
└── docs/
    └── POST_MVP_CASE_COMPARISON.md
```

### 2.2 模块职责

| 文件 | 职责 |
|------|------|
| `legalbot/rag/case_types.py` | `CaseCoreData`、`ComparisonTable`、`ApplicabilityPrediction`、`CaseCompareConfig` |
| `legalbot/rag/case_analyzer.py` | `CaseAnalyzer` — 从原始案例文本块中提取结构化字段 |
| `legalbot/agent/tools/case_compare.py` | `CaseCompareTool` — 编排检索 + 分析 + 对比表生成 |

---

## 三、对比输出格式

### 3.1 结构化表格 Schema

| 列 | 描述 |
|--------|-------------|
| `案号` | 案例编号（如："（2021）沪01民终1234号"） |
| `争议焦点` | 核心法律争议（1-2句话） |
| `裁判规则` | 提取的关键司法规则/理由 |
| `适用法条` | 适用的法律条文 |
| `相似度` | High / Medium / Low |

### 3.2 输出示例

```
## 案例对比分析结果

**用户纠纷概要**：买家付款后卖家拒绝交房，且房屋已被二次出售给第三方

### 相似案例对比表

| 案号 | 争议焦点 | 裁判规则 | 适用法条 | 相似度 |
|------|---------|---------|---------|-------|
| (2021)沪01民终1234号 | "一房二卖"中前后买受人权利顺位认定 | 出卖人将房屋所有权转移给后买受人时，前买受人可主张违约责任但不能对抗善意取得 | 《民法典》第209条、第224条 | High |
| (2020)京02民终5678号 | 二手房交易中卖家欺诈的认定与赔偿责任 | 卖家故意隐瞒房屋权利瑕疵导致交易失败，应退还房款并赔偿买家直接损失 | 《民法典》第500条第2项、第584条 | High |
| (2019)粤03民终9012号 | 借名买房纠纷中实际出资人的权利保护 | 借名买房合同有效，实际出资人可主张过户，但不能对抗登记簿公示效力 | 《民法典》第209条 | Medium |

### 适用性预测

**本案与 `（2021）沪01民终1234号` 最为相似**，争议焦点均为"一房二卖"。

**关键相似点**：买家已付款但未取得房屋所有权；卖家将房屋再次出售给第三方

**本案处理建议**：
1. 主张卖家违约，要求退还房款并赔偿损失（基于《民法典》第584条）
2. 如房屋尚未过户给后买受人，可主张继续履行合同
3. 风险提示：如第三方已善意取得，则原合同目的无法实现

> 以上分析基于类案检索结果，仅供参考。
```

---

## 四、工具设计

### 4.1 CaseCompareTool

**文件**：`legalbot/agent/tools/case_compare.py`

```python
from __future__ import annotations

from typing import Any

from legalbot.agent.tools.base import Tool, tool_parameters
from legalbot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema


@tool_parameters(
    tool_parameters_schema(
        dispute_facts=StringSchema(
            "用户描述的纠纷事实，应包含当事人、行为、结果等关键信息",
            min_length=10,
            max_length=2000,
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
    """案例对比分析工具 — 检索相似案例并生成结构化对比表"""

    name = "legal_case_compare"
    description = (
        "输入纠纷事实，检索知识库中的相似案例，"
        "生成结构化对比表（含案号、争议焦点、裁判规则、适用法条），"
        "并输出适用性预测和建议策略。"
    )

    def __init__(
        self,
        retriever: Any,
        case_analyzer: Any,
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
        # Step 1: 检索相似案例
        results = await self._retriever.retrieve(
            query=dispute_facts,
            law_area=law_area,
            doc_type="case",
            top_k=top_k,
        )

        if not results:
            return f"未检索到与「{dispute_facts[:50]}...」相关的案例。"

        # Step 2: 分析每个案例
        analyzed_cases = await self._case_analyzer.analyze_batch(results)

        # Step 3: 生成对比表
        comparison_output = await self._generate_comparison(dispute_facts, analyzed_cases)
        return comparison_output

    async def _generate_comparison(self, user_dispute: str, cases: list) -> str:
        prompt = CASE_COMPARE_PROMPT.format(
            user_dispute=user_dispute,
            cases_json=json.dumps([asdict(c) for c in cases], ensure_ascii=False, indent=2),
        )
        messages = [{"role": "user", "content": prompt}]
        response = await self._llm_provider.chat(messages=messages, temperature=0.1)
        return response.content or ""


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
4. 末尾必须附加免责声明
"""
```

---

## 五、案例数据模型

### 5.1 case_types.py

**文件**：`legalbot/rag/case_types.py`

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaseCoreData:
    """从单个案例中提取的结构化数据"""
    case_no: str | None = None          # 案号
    case_name: str | None = None        # 案件名称
    court: str | None = None             # 审理法院
    judge_date: str | None = None        # 裁判日期
    dispute_type: str | None = None     # 纠纷类型
    dispute_focus: str | None = None    # 争议焦点
    ruling_rule: str | None = None       # 裁判规则
    applicable_laws: list[str] = field(default_factory=list)  # 适用法条
    source_chunk_id: str | None = None  # 来源 chunk ID


@dataclass
class ApplicabilityPrediction:
    most_similar_case: str           # 案号
    similarity_score: str             # "High" | "Medium" | "Low"
    key_similarities: list[str]      # 共享事实模式
    suggested_strategies: list[str]   # 建议策略
    risk_warnings: list[str]         # 关键风险
    applicable_laws: list[str]        # 最相关法律


class CaseCompareConfig:
    comparison_model: str = ""
    max_cases_for_comparison: int = 10
    similarity_threshold_high: float = 0.85
    similarity_threshold_medium: float = 0.65
```

---

## 六、案例分析器

### 6.1 case_analyzer.py

**文件**：`legalbot/rag/case_analyzer.py`

```python
"""基于 LLM 的案例结构提取"""

from __future__ import annotations

import json
from typing import Any

from legalbot.rag.case_types import CaseCoreData


CASE_EXTRACTION_PROMPT = """\
你是一个法律案例分析助手。请从以下案例文本中提取结构化信息。

## 需要提取的字段
- 案号：法院判决书的编号
- 争议焦点：法院认定的核心法律争议（1-2句话）
- 裁判规则：法院裁判该争议的具体规则或逻辑
- 适用法条：判决适用的具体法律条文（列出条文号和法律全称）

## 输出格式
请用JSON格式输出，字段名为：case_no, dispute_focus, ruling_rule, applicable_laws

## 案例文本
---
{case_text}
---

## 要求
1. 只输出JSON，不要有其他文字
2. 如果某个字段无法从文本确定，输出null
3. 适用法条用数组格式，每个元素为"《法律全称》第X条"格式
"""


class CaseAnalyzer:
    """使用 LLM 从原始案例文本块中提取结构化数据"""

    def __init__(self, provider: Any):
        self._provider = provider

    async def analyze_single(self, chunk: Any) -> CaseCoreData:
        """分析单个案例文本块"""
        prompt = CASE_EXTRACTION_PROMPT.format(case_text=chunk.chunk.text[:3000])
        messages = [{"role": "user", "content": prompt}]
        response = await self._provider.chat(messages=messages, temperature=0.1)
        content = response.content or "{}"

        # 解析 JSON
        try:
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.rfind("```")
                content = content[start:end]
            data = json.loads(content.strip())
        except json.JSONDecodeError:
            data = {}

        return CaseCoreData(
            case_no=data.get("case_no"),
            dispute_focus=data.get("dispute_focus"),
            ruling_rule=data.get("ruling_rule"),
            applicable_laws=data.get("applicable_laws", []),
            source_chunk_id=chunk.chunk.id if hasattr(chunk, "chunk") else None,
        )

    async def analyze_batch(self, retrieval_results: list) -> list[CaseCoreData]:
        """顺序分析多个案例文本块"""
        results = []
        for r in retrieval_results:
            try:
                case_data = await self.analyze_single(r)
                results.append(case_data)
            except Exception:
                # 降级：创建最小案例数据
                results.append(CaseCoreData(
                    dispute_focus=str(r)[:200],
                    applicable_laws=[],
                ))
        return results
```

---

## 七、集成点

### 7.1 编排器集成

**文件**：`legalbot/agent/orchestrator.py`

添加新意图：

```python
INTENT_CASE_COMPARE = "case_compare"

# 更新 INTENT_PROMPT：
- case_compare: 案例对比分析（需要生成结构化对比表）

# LegalOrchestrator 中的新方法：
async def _run_case_compare_sync(self, query: str) -> str:
    tool = self._main_tools.get("legal_case_compare")
    if not tool:
        return "案例对比工具未配置，请使用 legal_rag_search 检索相关案例。"
    return await tool.execute(dispute_facts=query)
```

### 7.2 工具注册

**文件**：`legalbot/agent/loop.py`

```python
# 在 _register_default_tools() 中，RAG 工具注册之后：
if self.rag_config and self.rag_config.enable:
    from legalbot.agent.tools.case_compare import CaseCompareTool
    from legalbot.rag.case_analyzer import CaseAnalyzer

    retriever = create_retriever(self.rag_config)
    case_analyzer = CaseAnalyzer(self.provider)
    case_compare_tool = CaseCompareTool(
        retriever=retriever,
        case_analyzer=case_analyzer,
        llm_provider=self.provider,
    )
    self.tools.register(case_compare_tool)
```

---

## 八、配置

### 8.1 CaseCompareConfig

**文件**：`legalbot/config/schema.py`

```python
class CaseCompareConfig(Base):
    enable: bool = True
    comparison_model: str = ""
    max_cases: int = 10
    top_k_default: int = 5


class OrchestrateConfig(Base):
    enable: bool = False
    intent_model: str = ""
    agents: dict[str, AgentDefConfig] = Field(default_factory=dict)
    case_compare: CaseCompareConfig = Field(default_factory=CaseCompareConfig)
```

---

## 九、实施步骤

### 第一阶段：核心数据模型和案例分析（第 1 周）
- 1.1 创建 `legalbot/rag/case_types.py`
- 1.2 创建 `legalbot/rag/case_analyzer.py`
- 1.3 为 CaseAnalyzer 编写单元测试

### 第二阶段：工具实现（第 1-2 周）
- 2.1 创建 `legalbot/agent/tools/case_compare.py`
- 2.2 创建 `CASE_COMPARE_PROMPT` 模板
- 2.3 编写集成测试

### 第三阶段：编排器集成（第 2 周）
- 3.1 添加 `INTENT_CASE_COMPARE` 到编排器
- 3.2 在 AgentLoop 中注册工具
- 3.3 创建 `legal-case-compare` 技能

### 第四阶段：端到端测试（第 2-3 周）
- 4.1 使用示例案例数据创建测试 fixtures
- 4.2 人工评估

---

## 十、文件变更摘要

### 新增文件

| 文件 | 用途 |
|------|------|
| `legalbot/rag/case_types.py` | 案例数据模型和配置 |
| `legalbot/rag/case_analyzer.py` | 基于 LLM 的案例结构提取器 |
| `legalbot/agent/tools/case_compare.py` | CaseCompareTool |
| `legalbot/skills/legal-case-compare/SKILL.md` | 案例对比技能 |
| `docs/POST_MVP_CASE_COMPARISON.md` | 本文档 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `legalbot/config/schema.py` | 添加 `CaseCompareConfig`，扩展 `OrchestrateConfig` |
| `legalbot/agent/orchestrator.py` | 添加 `INTENT_CASE_COMPARE`、`_run_case_compare_sync()` |
| `legalbot/agent/loop.py` | 注册 `CaseCompareTool` |

---

## 十一、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|--------|------------|
| 案例文本质量参差不齐 — 元数据可能不完整 | 中 | 添加验证；降级到原始文本 |
| LLM 提取幻觉 | 中 | 与 RAG 交叉验证提取的法律条文 |
| 案例相似度判断过于表面 | 中 | 包含 High/Medium/Low 评分管理预期 |
| 长案例被分割到多个 chunks | 中 | 分析前获取完整案例（多个相关 chunks） |

---

## 十二、测试计划

| 测试 | 预期结果 |
|------|----------------|
| `test_case_analyzer_extracts_case_number` | 正确提取 case_no |
| `test_case_analyzer_handles_missing_fields` | 优雅降级 |
| `test_case_compare_tool_no_results` | 返回"未检索到相关案例" |
| `test_case_compare_tool_formats_table` | 输出包含 markdown 表格 |
| `test_full_case_compare_flow` | 输入纠纷 → 表格 → 适用性预测 |
