"""Legal MultiAgent orchestrator: intent classification + agent routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.config.schema import OrchestrateConfig
    from nanobot.providers.base import LLMProvider

# Legal intent categories
INTENT_LEGAL_QUERY = "legal_query"
INTENT_CONTRACT_REVIEW = "contract_review"
INTENT_CASE_SEARCH = "case_search"
INTENT_GENERAL = "general"

VALID_INTENTS = {INTENT_LEGAL_QUERY, INTENT_CONTRACT_REVIEW, INTENT_CASE_SEARCH, INTENT_GENERAL}

INTENT_PROMPT = """\
分析以下用户输入的意图，返回最匹配的类别。

类别：
- legal_query: 法律问题咨询（需检索法条后回答）
- contract_review: 合同/协议审查
- case_search: 案例检索
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
    ) -> None:
        self.provider = provider
        self.subagents = subagent_mgr
        self.config = config
        # Reference to the main agent's tool registry for tool lookup
        self._main_tools = main_tools or {}

    async def classify_intent(self, query: str) -> str:
        """Classify user intent via LLM."""
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
                    return intent

            # Default to general if no match
            logger.warning("Intent classification returned unrecognized result: {}", content)
            return INTENT_GENERAL

        except Exception as e:
            logger.error("Intent classification failed: {}", e)
            return INTENT_GENERAL

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

        if intent in (INTENT_LEGAL_QUERY, INTENT_CASE_SEARCH):
            return await self._single_agent("legal_research", query, context)

        if intent == INTENT_CONTRACT_REVIEW:
            return await self._contract_review_flow(query, context)

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

        if intent in (INTENT_LEGAL_QUERY, INTENT_CASE_SEARCH):
            return await self._run_agent_sync("legal_research", query)

        if intent == INTENT_CONTRACT_REVIEW:
            return await self._contract_review_flow_sync(query)

        return None  # type: ignore[return-value]

    async def _single_agent(
        self,
        agent_name: str,
        task: str,
        context: dict | None = None,
    ) -> str:
        """Dispatch to a single specialized agent (fire-and-forget)."""
        agent_def = self.config.agents.get(agent_name)
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
        agent_def = self.config.agents.get(agent_name)
        if not agent_def:
            return f"Agent '{agent_name}' 未在配置中定义。"

        # Build tools for the subagent
        available = self.subagents._build_available_tools()  # noqa: SLF001
        # Also include RAG search tool if available in main tools
        if self._main_tools and "legal_rag_search" in self._main_tools:
            available["legal_rag_search"] = self._main_tools["legal_rag_search"]

        from nanobot.agent.runner import AgentRunSpec, AgentRunner
        from nanobot.agent.tools.registry import ToolRegistry

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
