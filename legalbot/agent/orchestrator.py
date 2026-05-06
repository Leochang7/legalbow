"""Legal MultiAgent orchestrator: intent classification + agent routing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from legalbot.agent.subagent import SubagentManager
    from legalbot.config.schema import AgentDefConfig, OrchestrateConfig
    from legalbot.providers.base import LLMProvider


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

# Legal intent categories
INTENT_LEGAL_QUERY = "legal_query"
INTENT_CONTRACT_REVIEW = "contract_review"
INTENT_CASE_SEARCH = "case_search"
INTENT_COMPLEX_LEGAL_QUERY = "complex_legal_query"
INTENT_DEBATE = "debate"
INTENT_CASE_COMPARE = "case_compare"
INTENT_DOCUMENT_DRAFT = "document_draft"
INTENT_GENERAL = "general"

VALID_INTENTS = {INTENT_LEGAL_QUERY, INTENT_CONTRACT_REVIEW, INTENT_CASE_SEARCH,
                 INTENT_COMPLEX_LEGAL_QUERY, INTENT_DEBATE, INTENT_CASE_COMPARE,
                 INTENT_DOCUMENT_DRAFT, INTENT_GENERAL}

COMPLEXITY_CLASSIFICATION_PROMPT = """\
分析以下法律问题的复杂程度。

复杂度判断标准：
- simple: 仅需单条法律条文即可回答（如：法定婚龄是多少？）
- complex: 需要引用多条法律条文、进行逻辑推导、或涉及多个法律领域
  （如：用人单位拖欠工资且不签劳动合同如何维权？）

用户输入：{query}

返回类别：simple 或 complex

只返回类别名，不要解释。"""

INTENT_PROMPT = """\
分析以下用户输入的意图，返回最匹配的类别。

类别：
- legal_query: 法律问题咨询（需检索法条后回答）
- contract_review: 合同/协议审查
- case_search: 案例检索
- debate: 法律辩论分析（双方论证、争议焦点分析报告）
- case_compare: 案例对比分析（需要生成结构化对比表）
- document_draft: 法律文书起草（起诉状、答辩状、代理词、上诉状、执行申请书）
- general: 通用对话（不需要专业 Agent）

用户输入：{query}

只返回类别名，不要解释。"""


class LegalOrchestrator:
    """Legal MultiAgent orchestrator: classifies intent and dispatches to specialized agents."""

    def __init__(
        self,
        provider: LLMProvider,
        subagent_mgr: SubagentManager,
        config: OrchestrateConfig,
        *,
        main_tools: dict[str, Any] | None = None,
        retriever: Any = None,
    ) -> None:
        self.provider = provider
        self.subagents = subagent_mgr
        self.config = config
        # Reference to the main agent's tool registry for tool lookup
        self._main_tools = main_tools or {}
        # Direct retriever reference (used when legal_rag_search is not in main_tools)
        self._retriever = retriever

    async def classify_intent(self, query: str) -> str:
        """Classify user intent via LLM — two-stage for legal queries."""
        prompt = INTENT_PROMPT.format(query=query)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.provider.chat(
                messages=messages,
                model=self.config.intent_model or None,
                temperature=0.0,
            )
            content = (response.content or "").strip().lower()

            # Extract intent from response — handle various LLM output formats
            for intent in VALID_INTENTS:
                if intent in content:
                    # Second stage: check complexity for legal queries
                    if intent == INTENT_LEGAL_QUERY:
                        complex_intent = await self._classify_complexity(query)
                        if complex_intent == "complex":
                            return INTENT_COMPLEX_LEGAL_QUERY
                    return intent

            # Default to general if no match
            logger.warning("Intent classification returned unrecognized result: {}", content)
            return INTENT_GENERAL

        except Exception as e:
            logger.error("Intent classification failed: {}", e)
            return INTENT_GENERAL

    async def _classify_complexity(self, query: str) -> str:
        """判断法律问题是 simple 还是 complex。"""
        prompt = COMPLEXITY_CLASSIFICATION_PROMPT.format(query=query)
        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = (response.content or "").strip().lower()
            if "complex" in content:
                return "complex"
            return "simple"
        except Exception as e:
            logger.error("Complexity classification failed: {}", e)
            return "simple"

    async def dispatch(self, query: str, context: dict | None = None) -> str | None:
        """Classify intent and route to the appropriate agent.

        Returns the subagent task ID, or None if the main agent should handle it.
        """
        if not self.config.enable:
            return None

        intent = await self.classify_intent(query)
        logger.info("LegalOrchestrator: intent={} for query={}", intent, query[:80])

        if intent == INTENT_GENERAL:
            return None

        if intent == INTENT_COMPLEX_LEGAL_QUERY:
            return await self._multi_step_reasoning(query, context)

        if intent in (INTENT_LEGAL_QUERY, INTENT_CASE_SEARCH):
            return await self._single_agent("legal_research", query, context)

        if intent == INTENT_CONTRACT_REVIEW:
            return await self._contract_review_flow(query, context)

        if intent == INTENT_DOCUMENT_DRAFT:
            return await self._document_draft_flow(query, context)

        return None

    async def dispatch_sync(self, query: str, context: dict | None = None) -> str:
        """Dispatch and wait for the result synchronously (for tool execution).

        Returns the subagent's final result text.
        """
        if not self.config.enable:
            return "编排功能未启用。请直接使用 legal_rag_search 工具检索。"

        intent = await self.classify_intent(query)
        logger.info("LegalOrchestrator dispatch_sync: intent={} for query={}", intent, query[:80])

        if intent == INTENT_GENERAL:
            return None  # type: ignore[return-value]

        if intent == INTENT_COMPLEX_LEGAL_QUERY:
            return await self._multi_step_reasoning_sync(query)

        if intent in (INTENT_LEGAL_QUERY, INTENT_CASE_SEARCH):
            return await self._run_agent_sync("legal_research", query)

        if intent == INTENT_CONTRACT_REVIEW:
            return await self._contract_review_flow_sync(query)

        if intent == INTENT_DEBATE:
            return "请使用 legal_debate 工具启动辩论分析。"

        if intent == INTENT_CASE_COMPARE:
            return "请使用 legal_case_compare 工具进行案例对比分析。"

        if intent == INTENT_DOCUMENT_DRAFT:
            return "请使用 legal_document_generate 工具生成法律文书。"

        return None  # type: ignore[return-value]

    def _get_agent_def(self, agent_name: str) -> "AgentDefConfig | None":
        """Get agent definition, falling back to defaults."""
        agents = self.config.agents if self.config.agents else self.config.get_default_agents()
        return agents.get(agent_name)

    async def _single_agent(
        self,
        agent_name: str,
        task: str,
        context: dict | None = None,
    ) -> str:
        """Dispatch to a single specialized agent (fire-and-forget)."""
        agent_def = self._get_agent_def(agent_name)
        if not agent_def:
            logger.warning("Agent '{}' not found in config", agent_name)
            return None  # type: ignore[return-value]

        origin = context or {}
        return await self.subagents.spawn_with_config(
            task=task,
            system_prompt=agent_def.system_prompt or None,
            allowed_tools=agent_def.tools or None,
            model=agent_def.model or None,
            label=agent_name,
            origin_channel=origin.get("channel", "cli"),
            origin_chat_id=origin.get("chat_id", "direct"),
        )

    async def _run_agent_sync(self, agent_name: str, task: str) -> str:
        """Run a specialized agent synchronously and return the result."""
        agent_def = self._get_agent_def(agent_name)
        if not agent_def:
            return f"Agent '{agent_name}' 未在配置中定义。"

        # Build tools for the subagent
        available = self.subagents._build_available_tools()  # noqa: SLF001
        # Also include RAG search tool if available in main tools
        if self._main_tools and "legal_rag_search" in self._main_tools:
            available["legal_rag_search"] = self._main_tools["legal_rag_search"]

        from legalbot.agent.runner import AgentRunSpec, AgentRunner
        from legalbot.agent.tools.registry import ToolRegistry

        tools = ToolRegistry()
        allowed = agent_def.tools
        if allowed:
            for name in allowed:
                if tool := available.get(name):
                    tools.register(tool)
        else:
            for tool in available.values():
                tools.register(tool)

        prompt = agent_def.system_prompt or self.subagents._build_subagent_prompt()  # noqa: SLF001
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": task},
        ]

        runner = AgentRunner(self.provider)
        model = agent_def.model or self.subagents.model
        result = await runner.run(AgentRunSpec(
            initial_messages=messages,
            tools=tools,
            model=model,
            max_iterations=15,
            max_tool_result_chars=self.subagents.max_tool_result_chars,
            error_message=None,
            fail_on_tool_error=True,
            max_iterations_message="任务完成但未生成最终回复。",
        ))

        if result.stop_reason == "error":
            return f"Agent 执行出错：{result.error or '未知错误'}"
        return result.final_content or "任务完成但未生成最终回复。"

    async def _contract_review_flow(self, query: str, context: dict | None = None) -> str:
        """Contract review flow: research then review (fire-and-forget)."""
        # Step 1: Dispatch legal research first
        research_task = f"检索与以下合同内容相关的法律法规：\n{query}"
        research_id = await self._single_agent("legal_research", research_task, context)
        # For fire-and-forget, we just dispatch and let results come back
        # The contract_review agent will be dispatched by the main agent after research completes
        return research_id

    async def _contract_review_flow_sync(self, query: str) -> str:
        """Contract review flow: research then review (synchronous)."""
        # Step 1: Legal research
        research_task = f"检索与以下合同内容相关的法律法规：\n{query}"
        research_result = await self._run_agent_sync("legal_research", research_task)

        # Step 2: Contract review based on research
        review_task = (
            f"基于以下法律检索结果：\n{research_result}\n\n"
            f"审查合同内容并识别法律风险：\n{query}"
        )
        review_result = await self._run_agent_sync("contract_review", review_task)

        return review_result

    async def _multi_step_reasoning(self, query: str, context: dict | None = None) -> str | None:
        """调度多步推理（fire-and-forget）。"""
        reasoner = self._build_reasoner()
        if reasoner is None:
            return None
        task = f"请对以下法律问题进行多步链式推理分析：{query}"
        return await self.subagents.spawn_with_config(
            task=task,
            system_prompt="你是一个专业的法律推理专家。遇到复杂法律问题时应使用 legal_multi_step_reasoning 工具进行多步链式推理。",
            allowed_tools=["legal_multi_step_reasoning"],
            label="multi_step_reasoning",
            origin_channel=(context or {}).get("channel", "cli"),
            origin_chat_id=(context or {}).get("chat_id", "direct"),
        )

    async def _multi_step_reasoning_sync(self, query: str) -> str:
        """同步执行多步推理并返回结果。"""
        reasoner = self._build_reasoner()
        if reasoner is None:
            return "无法获取 RAG 检索器，多步推理功能不可用。"
        retriever = self._get_rag_retriever()
        if retriever is None:
            return "无法获取 RAG 检索器，多步推理功能不可用。"
        from legalbot.agent.reasoner import MultiStepLegalReasoner
        reasoner = MultiStepLegalReasoner(provider=self.provider, retriever=retriever)
        chain = await reasoner.reason(question=query)
        return chain.to_display_string()

    def _build_reasoner(self):
        """构建多步推理引擎（如果 retriever 可用）。"""
        try:
            retriever = self._get_rag_retriever()
            if retriever is None:
                return None
            from legalbot.agent.reasoner import MultiStepLegalReasoner
            return MultiStepLegalReasoner(provider=self.provider, retriever=retriever)
        except Exception as e:
            logger.error("Failed to build MultiStepLegalReasoner: {}", e)
            return None

    def _get_rag_retriever(self):
        """从 main_tools 或 subagent 可用工具中获取 LegalRetriever。"""
        # Direct retriever reference (preferred when legal_rag_search not in main_tools)
        if self._retriever is not None:
            return self._retriever
        if self._main_tools:
            rag_tool = self._main_tools.get("legal_rag_search")
            if rag_tool and hasattr(rag_tool, "_retriever"):
                return rag_tool._retriever
        available = self.subagents._build_available_tools()  # noqa: SLF001
        rag_tool = available.get("legal_rag_search")
        if rag_tool and hasattr(rag_tool, "_retriever"):
            return rag_tool._retriever
        return None

    # ========================================================================
    # Debate Mode
    # ========================================================================

    async def run_debate_sync(
        self,
        case_description: str,
        plaintiff_claims: str | None = None,
        defendant_response: str | None = None,
        debate_rounds: int = 1,
    ) -> str:
        """Run a legal debate and return the analysis report."""
        debate_input = DebateInput(
            case_description=case_description,
            plaintiff_claims=plaintiff_claims,
            defendant_response=defendant_response,
        )

        result = await self._run_debate(debate_input, debate_rounds)
        return self._format_debate_result(result)

    async def _run_debate(
        self,
        debate_input: DebateInput,
        debate_rounds: int,
    ) -> DebateResult:
        timeout = self.config.debate.timeout_per_agent

        plaintiff_task = self._build_plaintiff_task(debate_input)
        defendant_task = self._build_defendant_task(debate_input)

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
            return await asyncio.wait_for(
                self._execute_debate_agent(role, task, agent_config),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return f"（{role} Agent 执行超时）"
        except Exception as e:
            logger.exception("Debate agent {} failed: {}", role, str(e))
            return f"（{role} Agent 执行错误，请稍后重试）"

    async def _execute_debate_agent(
        self,
        role: str,
        task: str,
        agent_config: "AgentDefConfig",
    ) -> str:
        available = self.subagents._build_available_tools()  # noqa: SLF001
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

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": agent_config.system_prompt or ""},
            {"role": "user", "content": task},
        ]

        from legalbot.agent.runner import AgentRunSpec, AgentRunner
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
            return await asyncio.wait_for(
                self._execute_debate_agent("judge", judge_task, judge_config),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return "（Judge Agent 执行超时）"

    def _get_debate_agent_config(self, role: str) -> "AgentDefConfig":
        agents = self.config.debate.agents if self.config.debate.agents else {}
        if not agents:
            agents = self.config.debate.get_default_debate_agents()

        if role == "plaintiff":
            return agents.get("plaintiff_agent", agents.get("plaintiff", agents.get("plaintiff_agent",  # type: ignore[return-value]
                self._make_default_agent_def())))
        elif role == "defendant":
            return agents.get("defendant_agent", agents.get("defendant",  # type: ignore[return-value]
                self._make_default_agent_def()))
        elif role == "judge":
            cfg = agents.get("judge_agent", agents.get("judge",  # type: ignore[return-value]
                self._make_default_agent_def()))
            return cfg
        return self._make_default_agent_def()

    def _make_default_agent_def(self) -> "AgentDefConfig":
        from legalbot.config.schema import AgentDefConfig
        return AgentDefConfig()

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
        plaintiff_section = f"## 原告诉讼请求\n{debate_input.plaintiff_claims}\n\n" if debate_input.plaintiff_claims else ""
        defendant_section = f"## 被告答辩意见\n{debate_input.defendant_response}\n\n" if debate_input.defendant_response else ""
        return f"""## 辩论结束，请生成《争议焦点分析报告》

## 案情描述
{debate_input.case_description}

{plaintiff_section}{defendant_section}## 原告方完整论证
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

    # ========================================================================
    # Document Draft
    # ========================================================================

    async def _document_draft_flow(self, query: str, context: dict | None = None) -> str | None:
        """Dispatch document draft to subagent (fire-and-forget)."""
        doc_type = await self._classify_doc_type(query)
        agent_def = self._get_agent_def("document_draft")
        if not agent_def:
            return None
        return await self.subagents.spawn_with_config(
            task=f"请根据以下信息起草法律文书：\n{query}\n\n文书类型：{doc_type}",
            system_prompt=agent_def.system_prompt or None,
            allowed_tools=agent_def.tools or None,
            model=agent_def.model or None,
            label="document_draft",
            origin_channel=(context or {}).get("channel", "cli"),
            origin_chat_id=(context or {}).get("chat_id", "direct"),
        )

    async def _classify_doc_type(self, query: str) -> str:
        """Classify which document type to generate."""
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
