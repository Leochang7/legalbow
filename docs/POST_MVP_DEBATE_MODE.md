# Agent 辩论模式 (Agent Debate Mode)

> LegalBot MVP 扩展功能实施计划

---

## 一、功能概述与目标

### 1.1 功能描述

Agent 辩论模式是一个法律辩论 MultiAgent 系统，让两个专业法律 Agent（原告代理 Agent vs 被告代理 Agent）围绕一个法律纠纷进行并行论证，由 Judge Agent 综合双方论点，最终生成《争议焦点分析报告》。

### 1.2 核心目标

- **并行论证**：原告 Agent 和被告 Agent 同时启动、独立论证
- **结构化输出**：Judge Agent 生成《争议焦点分析报告》
- **可配置辩论轮次**：支持单轮和多轮深度辩论
- **工具链复用**：复用 legal_rag_search 工具进行法律检索

### 1.3 输入格式

```json
{
  "case_description": "案情描述",
  "plaintiff_claims": "原告诉求（可选）",
  "defendant_response": "被告答辩（可选）"
}
```

---

## 二、架构设计

### 2.1 高层架构

```
用户输入
     │
     ▼
LegalOrchestrator
  ┌───────────────────────────────────────┐
  │  run_debate_sync()                    │
  │  - 解析输入                           │
  │  - 并行启动原告/被告 Agent             │
  │  - 等待双方完成                        │
  │  - Judge Agent 综合                   │
  └───────────────────────────────────────┘
     │
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  ▼
plaintiff_agent   defendant_agent      judge_agent
 (原告代理律师)     (被告代理律师)      (审判 Agent)
```

### 2.2 并行执行

使用 `asyncio.gather()` 同时启动原告 Agent 和被告 Agent，双方独立执行后汇总给 Judge Agent。

---

## 三、Agent 定义

### 3.1 原告 Agent (plaintiff_agent)

**角色**：作为原告代理律师，构建最具说服力的法律论证。

**System Prompt**：
```
你是原告代理律师，专注于为委托人（原告）构建最具说服力的法律论证。

## 你的职责
1. 深入分析案情，站在原告立场构建论点
2. 检索相关法律法规、司法解释和指导性案例
3. 引用具体法条支持每一个法律主张
4. 预判对方（被告）可能的反驳点，并准备反驳论据

## 论证框架
### 一、案件事实梳理
- 列出对原告有利的关键事实

### 二、法律依据
- 引用最直接相关的法律条文（必须包含法律全称和条文号）

### 三、原告主张
- 逐项列明原告的诉讼请求及法律基础

### 四、法律论证
- 对每个争议焦点给出有利于原告的法律分析

### 五、对方可能反驳及应对

### 六、结论

## 重要原则
- 只引用真实存在的法律条文
- 始终以维护委托人合法权益为目标
```

**允许使用的工具**：`["legal_rag_search", "web_search", "web_fetch", "read_file"]`

### 3.2 被告 Agent (defendant_agent)

**角色**：作为被告代理律师，构建最具说服力的法律防御。

**System Prompt**：
```
你是被告代理律师，专注于为委托人（被告）构建最具说服力的法律防御和反驳论证。

## 你的职责
1. 深入分析案情，站在被告立场构建防御论点
2. 检索相关法律法规、司法解释和指导性案例
3. 引用具体法条支持每一个法律主张
4. 攻击原告论点中的薄弱环节

## 论证框架
### 一、案件事实梳理
- 列出对被告有利的事实

### 二、法律依据

### 三、被告答辩
- 逐项回应原告的诉讼请求

### 四、法律防御
- 对每个争议焦点给出有利于被告的法律分析

### 五、反驳原告核心论点

### 六、结论
```

**允许使用的工具**：`["legal_rag_search", "web_search", "web_fetch", "read_file"]`

### 3.3 审判 Agent (judge_agent)

**角色**：中立的审判 Agent，综合双方论点，生成《争议焦点分析报告》。

**System Prompt**：
```
你是审判法官（Judge），负责综合双方律师的论证，生成一份专业、客观的《争议焦点分析报告》。

## 报告结构

# 《争议焦点分析报告》

## 一、案件基本信息

## 二、争议焦点梳理
### 焦点一：[焦点描述]
- 原告立场摘要
- 被告立场摘要

## 三、原告方论点分析
### 3.1 核心论点
| 论点 | 支持法条 | Strength (1-10) | 评价 |

### 3.2 弱点提示

## 四、被告方论点分析
（同结构）

## 五、论点评分对比
| 争议焦点 | 原告得分 | 被告得分 | 更强方 | 理由 |

## 六、法律建议

## 七、风险提示
### 7.1 原告风险
### 7.2 被告风险

## 八、结论

## 重要原则
- 保持中立，不偏袒任何一方
- 只评价有法律依据的论点
- 评分标准：论点有充分法律依据+论证逻辑严密=高分
```

---

## 四、工具设计

### 4.1 DebateTool

**文件**：`legalbot/agent/tools/debate.py`

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from legalbot.agent.tools.base import Tool, tool_parameters
from legalbot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema

if TYPE_CHECKING:
    from legalbot.agent.orchestrator import LegalOrchestrator


@tool_parameters(
    tool_parameters_schema(
        case_description=StringSchema("案件事实描述"),
        plaintiff_claims=StringSchema("原告诉讼请求", nullable=True),
        defendant_response=StringSchema("被告答辩意见", nullable=True),
        debate_rounds=IntegerSchema(
            1,
            description="辩论轮次：1=单轮快速辩论，2=双轮深度辩论",
            minimum=1,
            maximum=3,
        ),
        required=["case_description"],
    )
)
class DebateTool(Tool):
    """法律辩论工具 — 启动原告 vs 被告 Agent 辩论并生成分析报告."""

    name = "legal_debate"
    description = (
        "启动法律辩论模式：原告代理 Agent 与被告代理 Agent 针对法律纠纷展开并行论证，"
        "由审判 Agent 综合双方论点生成《争议焦点分析报告》。"
    )

    def __init__(self, orchestrator: LegalOrchestrator):
        self._orchestrator = orchestrator

    @property
    def exclusive(self) -> bool:
        return True

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self,
        case_description: str,
        plaintiff_claims: str | None = None,
        defendant_response: str | None = None,
        debate_rounds: int = 1,
        **kwargs: Any,
    ) -> str:
        return await self._orchestrator.run_debate_sync(
            case_description=case_description,
            plaintiff_claims=plaintiff_claims,
            defendant_response=defendant_response,
            debate_rounds=debate_rounds,
        )
```

---

## 五、集成到 LegalOrchestrator

### 5.1 DebateConfig

**文件**：`legalbot/config/schema.py`

```python
class DebateConfig(Base):
    """辩论模式配置"""
    enable: bool = False
    rounds: int = 1
    timeout_per_agent: int = 120
    timeout_total: int = 300
    max_retries: int = 2
    judge_model: str = ""
    plaintiff_model: str = ""
    defendant_model: str = ""


class OrchestrateConfig(Base):
    enable: bool = False
    intent_model: str = ""
    agents: dict[str, AgentDefConfig] = Field(default_factory=dict)
    debate: DebateConfig = Field(default_factory=DebateConfig)
```

### 5.2 编排器方法

**文件**：`legalbot/agent/orchestrator.py`

```python
import asyncio
from dataclasses import dataclass


@dataclass
class DebateInput:
    case_description: str
    plaintiff_claims: str | None = None
    defendant_response: str | None = None


@dataclass
class DebateResult:
    plaintiff_arguments: str
    defendant_arguments: str
    judge_report: str
    metadata: dict


class LegalOrchestrator:
    # ... 现有代码 ...

    async def run_debate_sync(
        self,
        case_description: str,
        plaintiff_claims: str | None = None,
        defendant_response: str | None = None,
        debate_rounds: int = 1,
    ) -> str:
        if not self.config.debate.enable:
            return "辩论模式未启用，请在配置中启用 debate.enable=true"

        debate_input = DebateInput(
            case_description=case_description,
            plaintiff_claims=plaintiff_claims,
            defendant_response=defendant_response,
        )

        result = await self._run_debate(
            debate_input,
            debate_rounds,
        )
        return self._format_debate_result(result)

    async def _run_debate(
        self,
        debate_input: DebateInput,
        debate_rounds: int,
    ) -> DebateResult:
        timeout = self.config.debate.timeout_per_agent

        # 构建任务
        plaintiff_task = self._build_plaintiff_task(debate_input)
        defendant_task = self._build_defendant_task(debate_input)

        # 并行执行
        try:
            plaintiff_result, defendant_result = await asyncio.wait_for(
                asyncio.gather(
                    self._run_debate_agent("plaintiff", plaintiff_task, timeout),
                    self._run_debate_agent("defendant", defendant_task, timeout),
                ),
                timeout=self.config.debate.timeout_total,
            )
        except asyncio.TimeoutError:
            plaintiff_result = "（执行超时，未获得原告方论证）"
            defendant_result = "（执行超时，未获得被告方论证）"

        # 审判综合
        judge_result = await self._run_judge_agent(
            debate_input,
            plaintiff_result,
            defendant_result,
        )

        return DebateResult(
            plaintiff_arguments=plaintiff_result,
            defendant_arguments=defendant_result,
            judge_report=judge_result,
            metadata={"rounds": debate_rounds},
        )

    async def _run_debate_agent(
        self,
        role: str,
        task: str,
        timeout: int,
    ) -> str:
        agent_config = self._get_debate_agent_config(role)
        try:
            result = await asyncio.wait_for(
                self._execute_debate_agent(role, task, agent_config),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            return f"（{role} Agent 执行超时）"
        except Exception as e:
            return f"（{role} Agent 执行错误：{str(e)}）"

    async def _execute_debate_agent(
        self,
        role: str,
        task: str,
        agent_config: AgentDefConfig,
    ) -> str:
        available = self.subagents._build_available_tools()
        if self._main_tools and "legal_rag_search" in self._main_tools:
            available["legal_rag_search"] = self._main_tools["legal_rag_search"]

        from legalbot.agent.tools.registry import ToolRegistry
        tools = ToolRegistry()
        if agent_config.tools:
            for name in agent_config.tools:
                if tool := available.get(name):
                    tools.register(tool)
        else:
            for tool in available.values():
                tools.register(tool)

        messages = [
            {"role": "system", "content": agent_config.system_prompt or ""},
            {"role": "user", "content": task},
        ]

        from legalbot.agent.runner import AgentRunner, AgentRunSpec
        runner = AgentRunner(self.provider)
        model = agent_config.model or self.subagents.model

        result = await runner.run(AgentRunSpec(
            initial_messages=messages,
            tools=tools,
            model=model,
            max_iterations=15,
            max_tool_result_chars=self.subagents.max_tool_result_chars,
            error_message=None,
            fail_on_tool_error=True,
            max_iterations_message="论证任务完成但未生成最终回复。",
        ))

        if result.stop_reason == "error":
            return f"Agent 执行出错：{result.error or '未知错误'}"
        return result.final_content or "（无内容返回）"

    async def _run_judge_agent(
        self,
        debate_input: DebateInput,
        plaintiff_arguments: str,
        defendant_arguments: str,
    ) -> str:
        judge_config = self._get_debate_agent_config("judge")
        judge_task = self._build_judge_task(
            debate_input, plaintiff_arguments, defendant_arguments
        )
        timeout = self.config.debate.timeout_per_agent
        try:
            result = await asyncio.wait_for(
                self._execute_debate_agent("judge", judge_task, judge_config),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            return "（Judge Agent 执行超时）"

    def _get_debate_agent_config(self, role: str) -> AgentDefConfig:
        if role == "plaintiff":
            return self.config.debate.agents.get("plaintiff_agent", AgentDefConfig())
        elif role == "defendant":
            return self.config.debate.agents.get("defendant_agent", AgentDefConfig())
        elif role == "judge":
            judge_cfg = self.config.debate.agents.get("judge_agent", AgentDefConfig())
            if not judge_cfg.system_prompt:
                judge_cfg.system_prompt = self._get_default_judge_prompt()
            return judge_cfg
        return AgentDefConfig()

    def _get_default_judge_prompt(self) -> str:
        return """\
你是审判法官，负责综合双方律师的论证，生成《争议焦点分析报告》。

报告必须包含：争议焦点梳理、双方论点评分、法律建议、风险提示。
保持中立，只评价有法律依据的论点。"""

    def _build_plaintiff_task(self, debate_input: DebateInput) -> str:
        task = f"## 案情描述\n{debate_input.case_description}\n\n"
        if debate_input.plaintiff_claims:
            task += f"## 原告诉讼请求\n{debate_input.plaintiff_claims}\n\n"
        if debate_input.defendant_response:
            task += f"## 被告答辩（供参考）\n{debate_input.defendant_response}\n\n"
        task += "请作为原告代理律师，基于以上案情构建完整的法律论证。"
        return task

    def _build_defendant_task(self, debate_input: DebateInput) -> str:
        task = f"## 案情描述\n{debate_input.case_description}\n\n"
        if debate_input.defendant_response:
            task += f"## 被告答辩意见\n{debate_input.defendant_response}\n\n"
        if debate_input.plaintiff_claims:
            task += f"## 原告诉讼请求（供参考）\n{debate_input.plaintiff_claims}\n\n"
        task += "请作为被告代理律师，基于以上案情构建完整的法律论证。"
        return task

    def _build_judge_task(
        self,
        debate_input: DebateInput,
        plaintiff_arguments: str,
        defendant_arguments: str,
    ) -> str:
        return f"""## 辩论结束，请生成《争议焦点分析报告》

## 案情描述
{debate_input.case_description}

{## 原告诉讼请求" if debate_input.plaintiff_claims else ""}
{debate_input.plaintiff_claims or ""}

{## 被告答辩意见" if debate_input.defendant_response else ""}
{debate_input.defendant_response or ""}

## 原告方完整论证
{plaintiff_arguments}

## 被告方完整论证
{defendant_arguments}

请作为中立审判法官，综合以上双方论证，生成完整的《争议焦点分析报告》。
"""

    def _format_debate_result(self, result: DebateResult) -> str:
        output = [
            "=" * 60,
            "                    法律辩论分析报告",
            "=" * 60,
            "",
            result.judge_report,
            "",
            "=" * 60,
            "                    附录：双方原始论证",
            "=" * 60,
            "",
            "--- 原告方论证 ---",
            result.plaintiff_arguments[:3000] + ("..." if len(result.plaintiff_arguments) > 3000 else ""),
            "",
            "--- 被告方论证 ---",
            result.defendant_arguments[:3000] + ("..." if len(result.defendant_arguments) > 3000 else ""),
            "",
            "=" * 60,
            "注：本报告由 AI 生成，仅供参考，不构成正式法律意见。",
            "=" * 60,
        ]
        return "\n".join(output)
```

---

## 六、配置示例

```json
{
  "tools": {
    "orchestrate": {
      "enable": true,
      "debate": {
        "enable": true,
        "rounds": 1,
        "timeout_per_agent": 120,
        "timeout_total": 300
      },
      "agents": {
        "plaintiff_agent": {
          "system_prompt": "你是原告代理律师...",
          "tools": ["legal_rag_search", "web_search"]
        },
        "defendant_agent": {
          "system_prompt": "你是被告代理律师...",
          "tools": ["legal_rag_search", "web_search"]
        },
        "judge_agent": {
          "system_prompt": "你是审判法官...",
          "tools": ["legal_rag_search"]
        }
      }
    }
  }
}
```

---

## 七、技能设计

### 7.1 legal-debate 技能

**文件**：`legalbot/skills/legal-debate/SKILL.md`

```markdown
---
name: legal-debate
description: 法律辩论技能 — 启动 Agent 辩论模式分析复杂法律纠纷
always: false
---

## 功能说明

法律辩论技能允许启动 Agent 辩论模式，让两个专业法律 Agent（原告代理 vs 被告代理）围绕法律纠纷展开并行论证，由 Judge Agent 生成《争议焦点分析报告》。

## 使用场景

1. **复杂纠纷分析**：涉及多个争议焦点的大型纠纷
2. **诉讼策略评估**：在提起诉讼前评估双方论点
3. **法律培训**：模拟法庭辩论场景

## 使用方法

使用 `legal_debate` 工具启动辩论：

```
legal_debate(
    case_description="案件事实描述",
    plaintiff_claims="原告诉讼请求（可选）",
    defendant_response="被告答辩意见（可选）",
    debate_rounds=1
)
```

## 辩论轮次说明

| 轮次 | 适用场景 | 耗时 |
|------|---------|------|
| 1轮  | 快速分析 | ~1分钟 |
| 2轮  | 深度辩论 | ~2分钟 |

## 局限性

1. 辩论结果受限于 Agent 的法律知识水平
2. 不替代专业律师的人工分析
3. 多轮辩论成本较高
```

---

## 八、实施步骤

### 第一阶段：核心基础设施（第 1-2 周）
- 1.1 扩展配置：`DebateConfig` in `schema.py`
- 1.2 创建 `legalbot/agent/tools/debate.py` — `DebateTool`
- 1.3 在编排器中实现辩论方法

### 第二阶段：测试（第 3 周）
- 2.1 辩论流程单元测试
- 2.2 集成测试
- 2.3 超时和错误处理

### 第三阶段：技能与文档（第 4 周）
- 3.1 创建 `legal-debate` 技能
- 3.2 用户文档

---

## 九、文件变更摘要

### 修改的文件

| 文件 | 变更 |
|------|---------|
| `legalbot/config/schema.py` | 添加 `DebateConfig`，扩展 `OrchestrateConfig` |
| `legalbot/agent/orchestrator.py` | 添加辩论方法和 dataclass |
| `legalbot/agent/loop.py` | 注册 `DebateTool` |

### 新增文件

| 文件 | 描述 |
|------|-------------|
| `legalbot/agent/tools/debate.py` | DebateTool |
| `legalbot/skills/legal-debate/SKILL.md` | 技能 |
| `docs/POST_MVP_DEBATE_MODE.md` | 本文档 |

---

## 十、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|--------|------------|
| 成本翻倍（3个 Agent） | 成本 3-5 倍 | `max_tokens` 限制；默认 `debate_rounds=1` |
| 无限循环 | 资源浪费 | `max_iterations=15`；每个 Agent 超时 |
| 审判者偏见 | 报告不公平 | 提示词强调中立；评分平衡检查 |
| 法律准确性 | 误导性 | 需要 RAG 检索；只引用检索到的内容规则 |