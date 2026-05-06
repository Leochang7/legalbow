# 法律文书起草 (Legal Document Generation)

> LegalBot MVP 扩展功能实施计划

---

## 一、功能概述与目标

### 1.1 功能概述

法律文书起草功能允许用户输入案件事实并接收 AI 生成的法律文书：起诉状、答辩状、代理词、上诉状和执行申请书。

### 1.2 核心机制

**基于模板的生成 + LLM 填充 + RAG 检索法律依据**

```
用户输入案件事实
    │
    ├── 1. RAG 检索相关法条 (legal_rag_search / 复用已有 retriever)
    ├── 2. 选定文书模板 (template selector)
    ├── 3. LLM 填充模板变量 (case facts → document sections)
    └── 4. 返回结构化法律文书
```

### 1.3 非目标

- 不替代律师的正式法律意见
- MVP 知识库为静态（无实时法律更新）
- 无诉讼程序性指导

### 1.4 覆盖的文书类型

| 文书类型 | 中文 | 使用场景 |
|---|---|---|
| 起诉状 | 起诉状 | 原告发起诉讼 |
| 答辩状 | 答辩状 | 被告回应起诉 |
| 代理词 | 代理词 | 诉讼代理人发表意见 |
| 上诉状 | 上诉状 | 当事人不服一审判决提起上诉 |
| 执行申请书 | 执行申请书 | 债权人申请强制执行 |

---

## 二、架构设计

### 2.1 新增模块结构

```
legalbot/document/
├── __init__.py
├── config.py              # DocumentConfig schema
├── templates/
│   ├── __init__.py
│   ├── base.py            # LegalDocumentTemplate 抽象基类
│   ├── complaint.py       # 起诉状 template
│   ├── defense.py         # 答辩状 template
│   ├── agent_opinion.py   # 代理词 template
│   ├── appeal.py          # 上诉状 template
│   └── enforcement.py     # 执行申请书 template
├── generator.py           # LegalDocumentGenerator
└── variables.py           # Document variable extraction

legalbot/agent/tools/
└── document.py            # legal_document_generate Tool（新增）
```

### 2.2 模块职责

| 文件 | 职责 |
|---|---|
| `document/config.py` | `DocumentDraftConfig`：template_dir、enabled_types、default_model |
| `document/templates/base.py` | `LegalDocumentTemplate` 抽象基类：name、doc_type、required_variables、render() |
| `document/templates/*.py` | 各文书类型的具体模板实现 |
| `document/generator.py` | `LegalDocumentGenerator`：编排 RAG 检索 → 模板选择 → LLM 填充 |
| `document/variables.py` | `CaseFacts` dataclass、变量提取和验证 |
| `document.py` 工具 | `LegalDocumentGenerateTool`：将生成器暴露给 agent 工具注册表 |
| `legal-document-draft/SKILL.md` | 教 agent 何时使用此工具 |

---

## 三、模板系统设计

### 3.1 模板基类

**文件**：`legalbot/document/templates/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentVariable:
    """法律文书模板中的单个变量。"""
    key: str                          # 例如："plaintiff_name"
    label: str                        # 例如："原告姓名"
    description: str                  # 人类可读描述
    required: bool = True
    example: str | None = None        # LLM 指导的示例值


@dataclass
class DocumentSection:
    """法律文书中的一个章节。"""
    key: str                          # 例如："litigation_requests"
    heading: str                      # 例如："诉讼请求"
    instructions: str                # LLM 填充此章节的指导
    variables: list[DocumentVariable] = field(default_factory=list)
    min_tokens: int = 50              # 最小预期输出 tokens


class LegalDocumentTemplate(ABC):
    """所有法律文书模板的抽象基类。"""

    @property
    @abstractmethod
    def doc_type(self) -> str:
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        ...

    @property
    def required_variables(self) -> list[DocumentVariable]:
        return []

    @property
    def optional_variables(self) -> list[DocumentVariable]:
        return []

    @property
    def sections(self) -> list[DocumentSection]:
        return []

    @property
    def law_keywords(self) -> list[str]:
        return []

    @abstractmethod
    def build_prompt(
        self,
        case_facts: dict[str, Any],
        relevant_laws: list[str],
        variable_set: dict[str, Any],
    ) -> str:
        ...

    def format_document(self, filled: dict[str, Any]) -> str:
        ...
```

### 3.2 模板示例：起诉状

**文件**：`legalbot/document/templates/complaint.py`

```python
from dataclasses import dataclass
from typing import Any

from legalbot.document.templates.base import (
    DocumentSection,
    DocumentVariable,
    LegalDocumentTemplate,
)


class ComplaintTemplate(LegalDocumentTemplate):
    """起诉状 — Complaint / Civil Action"""

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
            DocumentVariable(key="plaintiff_name", label="原告姓名",
                             description="原告的全名（自然人）或单位名称（法人）", example="张三"),
            DocumentVariable(key="defendant_name", label="被告姓名",
                             description="被告的全名（自然人）或单位名称（法人）", example="李四"),
            DocumentVariable(key="defendant_address", label="被告地址",
                             description="被告的户籍地址或注册地址", example="北京市朝阳区某街道某号"),
            DocumentVariable(key="case_type", label="纠纷类型",
                             description="案件类型：民间借贷/买卖合同/租赁合同/侵权/其他", example="民间借贷"),
            DocumentVariable(key="disputed_amount", label="争议金额（元）",
                             description="涉及金额（仅数字，单位为人民币元）", example="100000"),
            DocumentVariable(key="litigation_requests", label="诉讼请求",
                             description="原告向法院提出的具体请求（分条列出）",
                             example="1. 判令被告返还借款本金10万元；2. 判令被告支付利息..."),
            DocumentVariable(key="facts_and_reasons", label="事实与理由",
                             description="简述纠纷经过和原告主张的法律依据",
                             example="2024年1月，被告因资金周转需要向原告借款10万元..."),
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

    @property
    def sections(self) -> list[DocumentSection]:
        return [
            DocumentSection(key="header", heading="民事起诉状",
                            instructions="起诉状标准抬头，居中加粗", variables=[]),
            DocumentSection(key="parties", heading="当事人信息",
                            instructions="列出原告和被告的基本信息", variables=[
                                DocumentVariable(key="plaintiff_name", label="原告"),
                                DocumentVariable(key="plaintiff_address", label="原告地址", required=False),
                                DocumentVariable(key="defendant_name", label="被告"),
                                DocumentVariable(key="defendant_address", label="被告地址"),
                            ]),
            DocumentSection(key="litigation_requests", heading="诉讼请求",
                            instructions="用编号列表写出所有诉讼请求",
                            variables=[
                                DocumentVariable(key="disputed_amount", label="争议金额"),
                                DocumentVariable(key="litigation_requests", label="诉讼请求内容"),
                            ]),
            DocumentSection(key="facts_and_reasons", heading="事实与理由",
                            instructions="叙述纠纷的时间、地点、经过、因果关系，引用相关法律条文",
                            variables=[
                                DocumentVariable(key="case_type", label="案由"),
                                DocumentVariable(key="contract_date", label="时间", required=False),
                                DocumentVariable(key="facts_and_reasons", label="事实与理由"),
                            ]),
            DocumentSection(key="evidence", heading="证据和证据来源",
                            instructions="列出所有证据及来源",
                            variables=[DocumentVariable(key="evidence_list", label="证据清单", required=False)]),
            DocumentSection(key="footer", heading="此致",
                            instructions="固定格式：此致 + 法院名称 + 具状人签名 + 日期", variables=[]),
        ]

    def build_prompt(
        self,
        case_facts: dict[str, Any],
        relevant_laws: list[str],
        variable_set: dict[str, Any],
    ) -> str:
        relevant_laws_text = "\n".join(
            f"- {law}" for law in relevant_laws
        ) if relevant_laws else "（未检索到相关法条）"

        missing_vars = [
            v.label for v in self.required_variables
            if v.key not in variable_set or not variable_set[v.key]
        ]
        missing_note = f"\n注意：以下必填变量未提供，请基于常识推断填写：{', '.join(missing_vars)}" if missing_vars else ""

        prompt = f"""\
你是一名中国执业律师。请根据以下案件事实，起草一份完整的民事起诉状。

## 案件事实
{self._format_case_facts(case_facts)}

## 相关法律条文（来自法律知识库检索）
{relevant_laws_text}
{missing_note}

## 起诉状要求
请按以下格式生成完整的起诉状，包含所有必要条款：

### 一、当事人信息
【格式】
原告：{variable_set.get('plaintiff_name', '（姓名）')}
住所：{variable_set.get('plaintiff_address', '（地址）')}
电话：{variable_set.get('plaintiff_phone', '（电话）')}

被告：{variable_set.get('defendant_name', '（姓名）')}
住所：{variable_set.get('defendant_address', '（地址）')}
电话：{variable_set.get('defendant_phone', '（电话）')}

### 二、诉讼请求
{variable_set.get('litigation_requests', '（诉讼请求内容）')}

### 三、事实与理由
{variable_set.get('facts_and_reasons', '（事实与理由）')}

### 四、证据和证据来源
{variable_set.get('evidence_list', '（证据清单）')}

### 五、此致
此致
{{法院名称}}
具状人（签名）：___________
原告：{variable_set.get('plaintiff_name', '')}（签名）
____年____月____日

---
要求：
1. 法律条文引用准确，使用《法律全称》第X条格式
2. 诉讼请求明确、具体、可执行
3. 事实叙述客观、连贯、有证据支撑
4. 语言严谨规范，符合中国法律文书标准
5. 事实与理由部分应充分论述法律关系和被告过错
"""
        return prompt

    @staticmethod
    def _format_case_facts(facts: dict[str, Any]) -> str:
        lines = []
        for key, value in facts.items():
            if value:
                lines.append(f"- {key}：{value}")
        return "\n".join(lines) if lines else "（未提供具体事实）"
```

### 3.3 所有模板

| 模板文件 | doc_type | 显示名称 |
|---|---|---|
| `complaint.py` | `complaint` | 起诉状 |
| `defense.py` | `defense` | 答辩状 |
| `agent_opinion.py` | `agent_opinion` | 代理词 |
| `appeal.py` | `appeal` | 上诉状 |
| `enforcement.py` | `enforcement` | 执行申请书 |

### 3.4 变量提取

**文件**：`legalbot/document/variables.py`

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaseFacts:
    """从用户输入中提取的结构化案件事实。"""
    raw_text: str
    doc_type: str
    parties: dict[str, dict[str, str]] = field(default_factory=dict)
    monetary_amount: float | None = None
    case_type: str | None = None
    date_of_dispute: str | None = None
    contract_date: str | None = None
    additional_facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "原告姓名": self.parties.get("plaintiff", {}).get("name", ""),
            "被告姓名": self.parties.get("defendant", {}).get("name", ""),
            "争议金额": str(self.monetary_amount) if self.monetary_amount else "",
            "案由": self.case_type or "",
            "事实经过": self.additional_facts.get("facts", ""),
        }
        for party_type, fields in self.parties.items():
            for field_name, value in fields.items():
                if value:
                    result[f"{party_type}_{field_name}"] = value
        for k, v in self.additional_facts.items():
            if k not in ("facts",) and v:
                result[k] = v
        return result


class CaseFactsExtractor:
    """使用 LLM 从自由文本案件描述中提取结构化 CaseFacts。"""

    EXTRACTION_PROMPT = """\
从以下案件描述中提取结构化信息，用于生成法律文书。

案件描述：
{case_text}

文档类型：{doc_type}

请以JSON格式返回，字段包括：
- parties: {{plaintiff: {{name, address, phone, id_number}}, defendant: {{name, address, phone, id_number}}}}
- monetary_amount: 涉及金额（数字，无金额则填null）
- case_type: 案件类型（民间借贷/买卖合同/租赁合同/侵权/劳动争议/婚姻家庭/其他）
- date_of_dispute: 纠纷发生日期（无法确定则填null）
- contract_date: 合同签订日期或借款日期（无法确定则填null）
- additional_facts: 其他重要事实（键值对格式）

只返回JSON，不要有任何其他内容。"""

    def __init__(self, provider: Any):
        self.provider = provider

    async def extract(self, case_text: str, doc_type: str) -> CaseFacts:
        import json
        prompt = self.EXTRACTION_PROMPT.format(case_text=case_text, doc_type=doc_type)
        response = await self.provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.content or "{}"
        try:
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.rfind("```")
                content = content[start:end]
            data = json.loads(content.strip())
        except json.JSONDecodeError:
            data = {"additional_facts": {"raw": case_text}}

        return CaseFacts(
            raw_text=case_text,
            doc_type=doc_type,
            parties=data.get("parties", {}),
            monetary_amount=data.get("monetary_amount"),
            case_type=data.get("case_type"),
            date_of_dispute=data.get("date_of_dispute"),
            contract_date=data.get("contract_date"),
            additional_facts=data.get("additional_facts", {}),
        )
```

---

## 四、工具设计

### 4.1 LegalDocumentGenerateTool

**文件**：`legalbot/agent/tools/document.py`

```python
"""legal_document_generate — 根据案件事实生成法律文书的工具。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from legalbot.agent.tools.base import Tool, tool_parameters
from legalbot.agent.tools.schema import StringSchema, tool_parameters_schema

if TYPE_CHECKING:
    from legalbot.document.generator import LegalDocumentGenerator


@tool_parameters(
    tool_parameters_schema(
        doc_type=StringSchema(
            "文书类型：complaint(起诉状)/defense(答辩状)/agent_opinion(代理词)/appeal(上诉状)/enforcement(执行申请书)",
            enum=["complaint", "defense", "agent_opinion", "appeal", "enforcement"],
        ),
        case_facts=StringSchema(
            "案件事实描述",
            min_length=10,
            max_length=10000,
        ),
        extra_variables=StringSchema("额外变量（JSON格式），用于覆盖或补充从案件事实中提取的信息", nullable=True),
        law_areas=StringSchema("法律领域过滤（可选）：民法/刑法/商法/劳动法/行政法等", nullable=True),
        required=["doc_type", "case_facts"],
    )
)
class LegalDocumentGenerateTool(Tool):
    """法律文书起草工具 — 根据案件事实生成法律文书（起诉状、答辩状等）。"""

    def __init__(self, generator: LegalDocumentGenerator):
        self._generator = generator

    @property
    def name(self) -> str:
        return "legal_document_generate"

    @property
    def description(self) -> str:
        return (
            "根据案件事实生成法律文书。支持生成起诉状、答辩状、代理词、上诉状和执行申请书。"
            "系统会自动检索相关法律法规条文作为依据，并按中国法律文书规范格式输出。"
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def exclusive(self) -> bool:
        return True

    async def execute(
        self,
        doc_type: str,
        case_facts: str,
        extra_variables: str | None = None,
        law_areas: str | None = None,
        **kwargs: Any,
    ) -> str:
        import json
        extra_vars = None
        if extra_variables:
            try:
                extra_vars = json.loads(extra_variables)
            except json.JSONDecodeError:
                pass
        areas = [law_areas] if law_areas else None
        return await self._generator.generate(
            doc_type=doc_type,
            case_facts=case_facts,
            extra_variables=extra_vars or {},
            law_areas=areas,
        )
```

### 4.2 文书生成器核心

**文件**：`legalbot/document/generator.py`

```python
"""法律文书生成器 — 编排检索 + 模板填充。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from legalbot.document.templates.base import LegalDocumentTemplate
from legalbot.document.variables import CaseFactsExtractor
from legalbot.rag.retriever import LegalRetriever


class LegalDocumentGenerator:
    """使用 RAG + 模板 + LLM 从案件事实生成法律文书。"""

    def __init__(
        self,
        retriever: LegalRetriever,
        provider: Any,
        template_dir: Path | None = None,
        enabled_types: list[str] | None = None,
    ):
        self.retriever = retriever
        self.provider = provider
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        self.enabled_types = enabled_types or [
            "complaint", "defense", "agent_opinion", "appeal", "enforcement"
        ]
        self._templates: dict[str, LegalDocumentTemplate] = {}
        self._extractor = CaseFactsExtractor(provider)
        self._load_templates()

    def _load_templates(self) -> None:
        from legalbot.document.templates.complaint import ComplaintTemplate
        from legalbot.document.templates.defense import DefenseTemplate
        from legalbot.document.templates.agent_opinion import AgentOpinionTemplate
        from legalbot.document.templates.appeal import AppealTemplate
        from legalbot.document.templates.enforcement import EnforcementTemplate

        template_classes = [
            ("complaint", ComplaintTemplate),
            ("defense", DefenseTemplate),
            ("agent_opinion", AgentOpinionTemplate),
            ("appeal", AppealTemplate),
            ("enforcement", EnforcementTemplate),
        ]

        for doc_type, cls in template_classes:
            if doc_type in self.enabled_types:
                self._templates[doc_type] = cls()

        logger.info("Loaded {} document templates: {}", len(self._templates), list(self._templates.keys()))

    def get_template(self, doc_type: str) -> LegalDocumentTemplate | None:
        return self._templates.get(doc_type)

    async def generate(
        self,
        doc_type: str,
        case_facts: str,
        extra_variables: dict[str, Any] | None = None,
        law_areas: list[str] | None = None,
    ) -> str:
        template = self._templates.get(doc_type)
        if not template:
            available = ", ".join(self._templates.keys())
            return f"不支持的文书类型：{doc_type}。支持的类型：{available}。"

        try:
            # 第 1 步：提取结构化事实
            extracted = await self._extractor.extract(case_facts, doc_type)
            variable_set = extracted.to_dict()
            if extra_variables:
                variable_set.update(extra_variables)

            # 第 2 步：RAG 检索相关法律
            law_query = " ".join(template.law_keywords) + " " + case_facts[:500]
            rag_results = await self.retriever.retrieve(
                query=law_query,
                law_area=law_areas[0] if law_areas else None,
                doc_type=None,
                top_k=8,
            )
            relevant_laws = [
                f"《{r.chunk.metadata.get('law_name', '未知')}"
                f》第{r.chunk.metadata.get('article_no', '')}条：{r.chunk.text[:200]}"
                for r in rag_results
            ]

            # 第 3 步：构建 prompt 并调用 LLM
            prompt = template.build_prompt(
                case_facts=case_facts,
                relevant_laws=relevant_laws,
                variable_set=variable_set,
            )

            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            filled_document = response.content or ""
            if not filled_document.strip():
                return "文书生成失败：LLM未返回有效内容。请尝试提供更详细的案件事实。"

            disclaimer = (
                "\n\n---\n"
                "【免责声明】\n"
                "本文书由 AI 自动生成，仅供参考，不构成正式法律意见。\n"
                "诉讼材料的正式提交应由执业律师审核确认。\n"
                "如需正式法律意见，请咨询执业律师。"
            )

            return filled_document.strip() + disclaimer

        except Exception as e:
            logger.exception("Document generation failed for doc_type={}", doc_type)
            return f"文书生成失败：{type(e).__name__}: {e}"
```

---

## 五、技能设计

### 5.1 legal-document-draft 技能

**新建目录**：`legalbot/skills/legal-document-draft/`

**文件**：`legalbot/skills/legal-document-draft/SKILL.md`

```markdown
---
name: legal-document-draft
description: 法律文书起草技能 — 根据案件事实生成起诉状、答辩状等法律文书
always: false
---

## 功能说明

legal-document-draft 技能使 AI 能够根据用户提供的案件事实，生成结构化的中国法律文书。

支持的文书类型：
- **起诉状** (complaint): 原告向法院提起民事诉讼
- **答辩状** (defense): 被告针对原告起诉进行答辩
- **代理词** (agent_opinion): 诉讼代理人在庭审中发表的意见
- **上诉状** (appeal): 当事人不服一审判决提起上诉
- **执行申请书** (enforcement): 胜诉后申请强制执行

## 使用场景

当用户提出以下请求时，应使用 legal_document_generate 工具：
- "帮我写一份起诉状"
- "起草一份答辩状"
- "我要起诉对方"
- "帮我写上诉状"
- "生成一份执行申请书"
- 其他需要生成法律文书的情形

## 使用方法

### 步骤一：识别文书类型

根据用户意图判断文书类型：
- 原告发起诉讼 → complaint
- 被告回应起诉 → defense
- 不服一审判决 → appeal
- 胜诉后申请执行 → enforcement
- 诉讼代理发表意见 → agent_opinion

### 步骤二：引导用户提供案件事实

调用 `legal_document_generate` 工具前，确保收集以下信息：
- **当事人信息**：原告/被告的姓名（名称）、地址、联系方式
- **纠纷经过**：时间、地点、经过、因果关系
- **争议金额**：涉及的具体金额（如有）
- **诉讼请求**：原告希望法院支持的请求
- **证据情况**：有哪些证据支持

### 步骤三：调用工具

```
legal_document_generate(
    doc_type="complaint",
    case_facts="原告张三（身份证号110101199001011234）借给被告李四（北京市朝阳区某街道某号）10万元，2024年1月借出，约定一年后还款，有借条，被告至今未还",
    law_areas=["民法"]
)
```

### 步骤四：返回结果

向用户说明：
1. 这是草稿，需要由执业律师审核才能正式使用
2. 如有需要修改的地方，可以进一步调整
3. 诉讼时效、管辖法院等程序性问题需要咨询律师

## 局限性

1. AI 生成的文书仅供参照，不能替代律师起草的正式文书
2. AI 无法核实用户提供的事实是否准确
3. 涉及重大利益的案件，建议委托律师全程代理
```

---

## 六、集成点

### 6.1 编排器集成

**文件**：`legalbot/agent/orchestrator.py`

添加新意图：

```python
INTENT_DOCUMENT_DRAFT = "document_draft"

# 更新 INTENT_PROMPT 以包含：
- document_draft: 法律文书起草（起诉状、答辩状、代理词、上诉状等）

# LegalOrchestrator 中的新方法：
async def _document_draft_flow(self, query: str, context: dict | None = None) -> str:
    doc_type_hint = await self._classify_doc_type(query)
    agent_def = self.config.agents.get("document_draft")
    if not agent_def:
        return "文书起草功能未配置。"
    return await self.subagents.spawn_with_config(
        task=f"请根据以下信息起草法律文书：\n{query}\n\n文书类型：{doc_type_hint}",
        system_prompt=agent_def.system_prompt,
        allowed_tools=agent_def.tools or None,
        model=agent_def.model or None,
        label="document_draft",
    )

async def _classify_doc_type(self, query: str) -> str:
    doc_type_prompt = f"""分析以下用户需求，确定要生成的法律文书类型。
类型选项：complaint/defense/appeal/enforcement/agent_opinion
用户需求：{query}
只返回一个类型名称（小写），不要解释。"""
    messages = [{"role": "user", "content": doc_type_prompt}]
    try:
        response = await self.provider.chat(messages=messages, temperature=0.0)
        content = (response.content or "").strip().lower()
        valid_types = {"complaint", "defense", "appeal", "enforcement", "agent_opinion"}
        for t in valid_types:
            if t in content:
                return t
        return "complaint"
    except Exception:
        return "complaint"
```

### 6.2 AgentLoop 中的工具注册

**文件**：`legalbot/agent/loop.py`

```python
# 在 _register_default_tools() 中 — 编排工具注册之后添加：
if self.document_draft_config and self.document_draft_config.enable:
    try:
        from legalbot.agent.tools.document import LegalDocumentGenerateTool
        from legalbot.document.generator import LegalDocumentGenerator
        from legalbot.rag import create_retriever

        retriever = create_retriever(self.rag_config)
        generator = LegalDocumentGenerator(retriever=retriever, provider=self.provider)
        self.tools.register(LegalDocumentGenerateTool(generator=generator))
    except ImportError:
        logger.warning("Legal document generation dependencies not installed.")
```

---

## 七、配置 schema 扩展

### 7.1 DocumentDraftConfig

**文件**：`legalbot/config/schema.py`

```python
class DocumentDraftConfig(Base):
    """法律文书起草配置"""
    enable: bool = False
    template_dir: str = ""
    enabled_types: list[str] = Field(
        default_factory=lambda: ["complaint", "defense", "agent_opinion", "appeal", "enforcement"]
    )
    default_model: str = ""
    max_laws_retrieved: int = 8


# 在 ToolsConfig 中，添加：
class ToolsConfig(Base):
    # ... 现有字段 ...
    rag: RAGConfig = Field(default_factory=RAGConfig)
    orchestrate: OrchestrateConfig = Field(default_factory=OrchestrateConfig)
    document_draft: DocumentDraftConfig = Field(default_factory=DocumentDraftConfig)
```

### 7.2 config.json 示例

```json
{
  "tools": {
    "rag": { "enable": true },
    "orchestrate": {
      "enable": true,
      "agents": {
        "document_draft": {
          "system_prompt": "你是法律文书起草专家。根据用户提供的案件事实，起草符合中国法律规范的法律文书草稿。",
          "tools": ["legal_document_generate", "legal_rag_search"]
        }
      }
    },
    "document_draft": {
      "enable": true,
      "enabled_types": ["complaint", "defense", "agent_opinion", "appeal", "enforcement"]
    }
  }
}
```

---

## 八、实施步骤（分阶段）

### 第一阶段：核心基础设施（第 1 周）
- 1.1 创建 `legalbot/document/` 包结构
- 1.2 实现 `document/templates/base.py` — `LegalDocumentTemplate` 抽象基类
- 1.3 实现 `document/variables.py` — `CaseFacts`、`CaseFactsExtractor`
- 1.4 在 `config/schema.py` 添加 `DocumentDraftConfig`

### 第二阶段：模板实现（第 1-2 周）
- 2.1 实现 `document/templates/complaint.py` — 起诉状
- 2.2 实现 `document/templates/defense.py` — 答辩状
- 2.3 实现 `document/templates/agent_opinion.py` — 代理词
- 2.4 实现 `document/templates/appeal.py` — 上诉状
- 2.5 实现 `document/templates/enforcement.py` — 执行申请书

### 第三阶段：生成器和工具（第 2 周）
- 3.1 实现 `document/generator.py` — `LegalDocumentGenerator`
- 3.2 实现 `legalbot/agent/tools/document.py` — `LegalDocumentGenerateTool`
- 3.3 在 `AgentLoop` 中注册工具

### 第四阶段：编排器集成（第 2-3 周）
- 4.1 添加 `INTENT_DOCUMENT_DRAFT` 到编排器
- 4.2 添加 `_document_draft_flow()` 方法
- 4.3 创建 `legalbot/skills/legal-document-draft/SKILL.md`

### 第五阶段：测试（第 3-4 周）
- 为模板、生成器、工具编写单元测试
- 完整流程集成测试

---

## 九、文件变更清单

### 新增文件

| 文件 | 描述 |
|---|---|
| `legalbot/document/__init__.py` | 包初始化 + 导出 |
| `legalbot/document/config.py` | `DocumentDraftConfig` |
| `legalbot/document/variables.py` | `CaseFacts`、`CaseFactsExtractor` |
| `legalbot/document/templates/__init__.py` | 模板注册表 |
| `legalbot/document/templates/base.py` | `LegalDocumentTemplate` 抽象基类 |
| `legalbot/document/templates/complaint.py` | `ComplaintTemplate` |
| `legalbot/document/templates/defense.py` | `DefenseTemplate` |
| `legalbot/document/templates/agent_opinion.py` | `AgentOpinionTemplate` |
| `legalbot/document/templates/appeal.py` | `AppealTemplate` |
| `legalbot/document/templates/enforcement.py` | `EnforcementTemplate` |
| `legalbot/document/generator.py` | `LegalDocumentGenerator` |
| `legalbot/agent/tools/document.py` | `LegalDocumentGenerateTool` |
| `legalbot/skills/legal-document-draft/SKILL.md` | 技能定义 |
| `docs/POST_MVP_LEGAL_DOCUMENT_GENERATION.md` | 本文档 |

### 修改文件

| 文件 | 变更 |
|---|---|
| `legalbot/config/schema.py` | 添加 `DocumentDraftConfig`；在 `ToolsConfig` 中添加 `document_draft` 字段 |
| `legalbot/agent/loop.py` | 添加 `document_draft_config` 参数；注册工具 |
| `legalbot/agent/orchestrator.py` | 添加 `INTENT_DOCUMENT_DRAFT`；添加 `_document_draft_flow()` |

---

## 十、风险与缓解

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| LLM 生成事实错误的法律文书 | 高 | RAG 检索作为依据；每份文书附加免责声明 |
| 法律文书格式错误导致被退回 | 高 | 模板强制严格格式；建议律师审核 |
| 从自由文本提取变量失败 | 中 | 回退到将原始文本作为 additional_facts |
| 幻觉法律引用 | 高 | RAG 结果提供真实依据；必须附加免责声明 |

---

## 十一、测试计划

| 测试 | 预期结果 |
|---|---|
| 使用完整事实生成起诉状 | 文书以"民事起诉状"开头，包含当事人、请求、引用 |
| 生成答辩状 | 文书包含"答辩状"标题和答辩要点 |
| 未知 doc_type | 返回错误消息列出有效类型 |
| 最少事实 | 文书生成并带有占位符推断 + 免责声明 |
| RAG 依据 | 相关法律条文出现在文书正文中 |