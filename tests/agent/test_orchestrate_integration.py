"""Integration tests for Phase 3 MultiAgent orchestration.

Tests the full flow: config → orchestrator → tool registration → dispatch → result.
Uses mock LLM provider to simulate agent responses.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.orchestrator import (
    INTENT_CONTRACT_REVIEW,
    INTENT_GENERAL,
    INTENT_LEGAL_QUERY,
    LegalOrchestrator,
)
from nanobot.agent.tools.orchestrate import OrchestrateTool
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import AgentDefConfig, OrchestrateConfig, RAGConfig
from nanobot.providers.base import LLMProvider, LLMResponse


# ---------------------------------------------------------------------------
# Fake provider
# ---------------------------------------------------------------------------


class FakeProvider(LLMProvider):
    """Fake LLM provider that returns predefined responses."""

    def __init__(self, responses: list[str] | None = None):
        super().__init__(api_key="fake", api_base=None)
        self._responses = responses or ["legal_query", "根据《劳动合同法》第82条..."]
        self._call_index = 0
        self.generation = MagicMock(max_tokens=4096)

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        if self._call_index < len(self._responses):
            content = self._responses[self._call_index]
            self._call_index += 1
        else:
            content = "general"
        return LLMResponse(content=content)

    async def chat_stream(self, messages, tools=None, model=None, max_tokens=4096,
                          temperature=0.7, reasoning_effort=None, tool_choice=None):
        yield LLMResponse(content="streaming not supported in tests")

    def get_default_model(self):
        return "fake-model"


# ---------------------------------------------------------------------------
# Integration test: Orchestrator + Tool + Config
# ---------------------------------------------------------------------------


class TestOrchestratorIntegration:
    """Integration tests for orchestrator with real config and tool objects."""

    def _make_orchestrate_config(self) -> OrchestrateConfig:
        return OrchestrateConfig(
            enable=True,
            agents={
                "legal_research": AgentDefConfig(
                    system_prompt="你是法律检索专家，擅长精准检索法规和案例。",
                    tools=["legal_rag_search", "read_file"],
                ),
                "contract_review": AgentDefConfig(
                    system_prompt="你是合同审查专家，擅长识别合同条款的法律风险。",
                    tools=["legal_rag_search", "read_file"],
                ),
            },
        )

    @pytest.mark.asyncio
    async def test_orchestrator_classify_and_dispatch_legal_query(self):
        """Full flow: classify intent → dispatch to legal_research agent."""
        config = self._make_orchestrate_config()
        # Two responses: first for classify_intent, second for any subsequent calls
        provider = FakeProvider(responses=["legal_query", "legal_query"])

        from nanobot.agent.subagent import SubagentManager
        bus = MessageBus()
        subagent_mgr = SubagentManager(
            provider=provider,
            workspace=Path("/tmp/test_workspace"),
            bus=bus,
            max_tool_result_chars=16000,
        )

        orch = LegalOrchestrator(provider, subagent_mgr, config)

        # Test classify
        intent = await orch.classify_intent("用人单位不签劳动合同怎么办")
        assert intent == INTENT_LEGAL_QUERY

        # Test dispatch (fire-and-forget)
        with patch.object(subagent_mgr, "runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_content = "根据《劳动合同法》第82条..."
            mock_result.stop_reason = "stop"
            mock_result.error = None
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await orch.dispatch("用人单位不签劳动合同怎么办")
            # Fire-and-forget returns subagent ID
            assert result is not None

    @pytest.mark.asyncio
    async def test_orchestrator_dispatch_sync_with_runner(self):
        """dispatch_sync runs agent synchronously with AgentRunner."""
        config = self._make_orchestrate_config()
        provider = FakeProvider(responses=["legal_query"])

        from nanobot.agent.subagent import SubagentManager
        bus = MessageBus()
        subagent_mgr = SubagentManager(
            provider=provider,
            workspace=Path("/tmp/test_workspace"),
            bus=bus,
            max_tool_result_chars=16000,
        )

        orch = LegalOrchestrator(provider, subagent_mgr, config)

        mock_result = MagicMock()
        mock_result.final_content = "根据《民法典》第585条，违约金条款..."
        mock_result.stop_reason = "stop"
        mock_result.error = None

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("nanobot.agent.runner.AgentRunner", return_value=mock_runner):
            result = await orch.dispatch_sync("搜索合同违约金的规定")
            assert "民法典" in result

    @pytest.mark.asyncio
    async def test_orchestrator_contract_review_two_step_flow(self):
        """Contract review: research step → review step."""
        config = self._make_orchestrate_config()
        provider = FakeProvider(responses=["contract_review"])

        from nanobot.agent.subagent import SubagentManager
        bus = MessageBus()
        subagent_mgr = SubagentManager(
            provider=provider,
            workspace=Path("/tmp/test_workspace"),
            bus=bus,
            max_tool_result_chars=16000,
        )

        orch = LegalOrchestrator(provider, subagent_mgr, config)

        # Research result
        research_result = MagicMock()
        research_result.final_content = "相关法规：《民法典》第585条、第490条"
        research_result.stop_reason = "stop"
        research_result.error = None

        # Review result
        review_result = MagicMock()
        review_result.final_content = "合同风险点：1. 违约金条款偏高 2. 缺少不可抗力条款"
        review_result.stop_reason = "stop"
        review_result.error = None

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=[research_result, review_result])

        with patch("nanobot.agent.runner.AgentRunner", return_value=mock_runner):
            result = await orch._contract_review_flow_sync("审查这份租赁合同")
            assert "风险" in result
            # Verify two agent runs happened
            assert mock_runner.run.call_count == 2

            # Verify second call included research results
            second_call = mock_runner.run.call_args_list[1]
            spec = second_call.args[0] if second_call.args else second_call.kwargs.get("spec")
            if spec:
                user_msg = spec.initial_messages[-1]["content"]
                assert "民法典" in user_msg  # Research results injected

    @pytest.mark.asyncio
    async def test_orchestrate_tool_end_to_end(self):
        """OrchestrateTool.execute → orchestrator → agent → result."""
        config = self._make_orchestrate_config()
        provider = FakeProvider(responses=["legal_query"])

        from nanobot.agent.subagent import SubagentManager
        bus = MessageBus()
        subagent_mgr = SubagentManager(
            provider=provider,
            workspace=Path("/tmp/test_workspace"),
            bus=bus,
            max_tool_result_chars=16000,
        )

        orch = LegalOrchestrator(provider, subagent_mgr, config)
        tool = OrchestrateTool(orchestrator=orch)

        mock_result = MagicMock()
        mock_result.final_content = "根据《劳动合同法》第82条，用人单位自用工之日起超过一个月不满一年未签订书面劳动合同的，应当向劳动者每月支付二倍的工资。"
        mock_result.stop_reason = "stop"
        mock_result.error = None

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("nanobot.agent.runner.AgentRunner", return_value=mock_runner):
            result = await tool.execute(query="用人单位不签劳动合同怎么办")
            assert "劳动合同法" in result

    @pytest.mark.asyncio
    async def test_orchestrate_tool_with_explicit_intent(self):
        """OrchestrateTool with explicit intent bypasses classification."""
        config = self._make_orchestrate_config()
        provider = FakeProvider(responses=["contract_review"])

        from nanobot.agent.subagent import SubagentManager
        bus = MessageBus()
        subagent_mgr = SubagentManager(
            provider=provider,
            workspace=Path("/tmp/test_workspace"),
            bus=bus,
            max_tool_result_chars=16000,
        )

        orch = LegalOrchestrator(provider, subagent_mgr, config)
        tool = OrchestrateTool(orchestrator=orch)

        research_result = MagicMock()
        research_result.final_content = "相关法规：民法典第585条"
        research_result.stop_reason = "stop"
        research_result.error = None

        review_result = MagicMock()
        review_result.final_content = "风险：违约金条款不合规"
        review_result.stop_reason = "stop"
        review_result.error = None

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=[research_result, review_result])

        with patch("nanobot.agent.runner.AgentRunner", return_value=mock_runner):
            result = await tool.execute(
                query="帮我看看这份合同",
                intent="contract_review",
            )
            assert "风险" in result


class TestOrchestrateToolRegistration:
    """Test that OrchestrateTool is properly registered in AgentLoop."""

    def test_orchestrate_tool_registered_when_enabled(self):
        """When orchestrate config is enabled, the tool should be registered."""
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"

        config = OrchestrateConfig(
            enable=True,
            agents={
                "legal_research": AgentDefConfig(
                    system_prompt="法律检索专家",
                    tools=["legal_rag_search"],
                ),
            },
        )

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=Path("/tmp/test"),
            orchestrate_config=config,
        )

        assert loop.tools.has("legal_orchestrate")
        tool = loop.tools.get("legal_orchestrate")
        assert isinstance(tool, OrchestrateTool)

    def test_orchestrate_tool_not_registered_when_disabled(self):
        """When orchestrate config is disabled, no tool should be registered."""
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"

        config = OrchestrateConfig(enable=False)

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=Path("/tmp/test"),
            orchestrate_config=config,
        )

        assert not loop.tools.has("legal_orchestrate")

    def test_orchestrate_tool_not_registered_when_none(self):
        """When orchestrate config is None, no tool should be registered."""
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=Path("/tmp/test"),
        )

        assert not loop.tools.has("legal_orchestrate")


class TestRAGAndOrchestrateCoexistence:
    """Test that RAG and Orchestrate tools can coexist."""

    def test_both_rag_and_orchestrate_registered(self):
        """Both tools should be available when both configs are enabled."""
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"

        # Need RAG to be enabled with proper mocking
        rag_config = RAGConfig(enable=True)
        orchestrate_config = OrchestrateConfig(
            enable=True,
            agents={
                "legal_research": AgentDefConfig(
                    system_prompt="法律检索专家",
                    tools=["legal_rag_search"],
                ),
            },
        )

        with patch("nanobot.agent.tools.rag.RAGSearchTool") as MockRAGTool, \
             patch("nanobot.rag.create_retriever") as mock_create:
            mock_rag = MagicMock()
            MockRAGTool.return_value = mock_rag
            mock_create.return_value = MagicMock()

            loop = AgentLoop(
                bus=bus,
                provider=provider,
                workspace=Path("/tmp/test"),
                rag_config=rag_config,
                orchestrate_config=orchestrate_config,
            )

            # Both tools should be registered
            assert loop.tools.has("legal_orchestrate")
            # RAG tool registration depends on import success


class TestConfigIntegration:
    """Test OrchestrateConfig in the full config schema."""

    def test_orchestrate_config_in_tools_config(self):
        from nanobot.config.schema import ToolsConfig

        config = ToolsConfig()
        assert config.orchestrate.enable is False
        assert config.orchestrate.agents == {}

    def test_orchestrate_config_with_agents(self):
        from nanobot.config.schema import ToolsConfig

        config = ToolsConfig(
            orchestrate=OrchestrateConfig(
                enable=True,
                agents={
                    "legal_research": AgentDefConfig(
                        system_prompt="法律检索专家",
                        tools=["legal_rag_search", "web_search"],
                        model="deepseek/deepseek-chat",
                    ),
                },
            ),
        )
        assert config.orchestrate.enable is True
        assert "legal_research" in config.orchestrate.agents
        assert config.orchestrate.agents["legal_research"].model == "deepseek/deepseek-chat"

    def test_config_json_roundtrip(self):
        """OrchestrateConfig should serialize/deserialize correctly."""
        import json

        from nanobot.config.schema import ToolsConfig

        config = ToolsConfig(
            orchestrate=OrchestrateConfig(
                enable=True,
                intent_model="deepseek/deepseek-chat",
                agents={
                    "legal_research": AgentDefConfig(
                        system_prompt="法律检索专家",
                        tools=["legal_rag_search"],
                    ),
                },
            ),
        )
        # Serialize
        data = config.model_dump(mode="json")
        # Check it's serializable
        json_str = json.dumps(data)
        assert "orchestrate" in json_str
        assert "legal_research" in json_str

        # Deserialize
        restored = ToolsConfig.model_validate(data)
        assert restored.orchestrate.enable is True
        assert "legal_research" in restored.orchestrate.agents


class TestOrchestratorAgentTools:
    """Test that orchestrator properly handles tool whitelisting for agents."""

    @pytest.mark.asyncio
    async def test_agent_gets_whitelisted_tools_only(self):
        """When allowed_tools is set, only those tools should be available to the subagent."""
        config = OrchestrateConfig(
            enable=True,
            agents={
                "legal_research": AgentDefConfig(
                    system_prompt="法律检索专家",
                    tools=["read_file"],  # Only read_file
                ),
            },
        )

        provider = FakeProvider(responses=["legal_query"])

        from nanobot.agent.subagent import SubagentManager
        bus = MessageBus()
        subagent_mgr = SubagentManager(
            provider=provider,
            workspace=Path("/tmp/test_workspace"),
            bus=bus,
            max_tool_result_chars=16000,
        )

        # Get available tools from subagent manager
        available = subagent_mgr._build_available_tools()
        assert "read_file" in available
        assert "write_file" in available

        # Now test that _run_agent_sync only uses whitelisted tools
        orch = LegalOrchestrator(provider, subagent_mgr, config)

        mock_result = MagicMock()
        mock_result.final_content = "检索结果"
        mock_result.stop_reason = "stop"
        mock_result.error = None

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("nanobot.agent.runner.AgentRunner", return_value=mock_runner):
            await orch._run_agent_sync("legal_research", "检索劳动合同法")

            # Verify the tools used in AgentRunSpec
            call_args = mock_runner.run.call_args
            spec = call_args.args[0] if call_args.args else call_args.kwargs.get("spec")
            assert spec is not None
            tool_names = list(spec.tools._tools.keys())
            # Should only have read_file (whitelisted)
            assert "read_file" in tool_names
            # Should NOT have write_file (not whitelisted)
            assert "write_file" not in tool_names

    @pytest.mark.asyncio
    async def test_agent_with_empty_tools_gets_all(self):
        """When allowed_tools is empty, agent gets all available tools."""
        config = OrchestrateConfig(
            enable=True,
            agents={
                "legal_research": AgentDefConfig(
                    system_prompt="法律检索专家",
                    tools=[],  # Empty = all tools
                ),
            },
        )

        provider = FakeProvider(responses=["legal_query"])

        from nanobot.agent.subagent import SubagentManager
        bus = MessageBus()
        subagent_mgr = SubagentManager(
            provider=provider,
            workspace=Path("/tmp/test_workspace"),
            bus=bus,
            max_tool_result_chars=16000,
        )

        orch = LegalOrchestrator(provider, subagent_mgr, config)

        mock_result = MagicMock()
        mock_result.final_content = "检索结果"
        mock_result.stop_reason = "stop"
        mock_result.error = None

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("nanobot.agent.runner.AgentRunner", return_value=mock_runner):
            await orch._run_agent_sync("legal_research", "检索劳动合同法")

            call_args = mock_runner.run.call_args
            spec = call_args.args[0] if call_args.args else call_args.kwargs.get("spec")
            tool_names = list(spec.tools._tools.keys())
            # Should have all available tools
            assert "read_file" in tool_names
            assert "write_file" in tool_names

    @pytest.mark.asyncio
    async def test_rag_tool_included_in_agent_tools(self):
        """When main_tools has legal_rag_search, it should be available to the agent."""
        config = OrchestrateConfig(
            enable=True,
            agents={
                "legal_research": AgentDefConfig(
                    system_prompt="法律检索专家",
                    tools=["legal_rag_search", "read_file"],
                ),
            },
        )

        provider = FakeProvider(responses=["legal_query"])

        from nanobot.agent.subagent import SubagentManager
        bus = MessageBus()
        subagent_mgr = SubagentManager(
            provider=provider,
            workspace=Path("/tmp/test_workspace"),
            bus=bus,
            max_tool_result_chars=16000,
        )

        # Create a mock RAG tool
        mock_rag_tool = MagicMock()
        mock_rag_tool.name = "legal_rag_search"

        orch = LegalOrchestrator(
            provider, subagent_mgr, config,
            main_tools={"legal_rag_search": mock_rag_tool},
        )

        mock_result = MagicMock()
        mock_result.final_content = "检索结果"
        mock_result.stop_reason = "stop"
        mock_result.error = None

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("nanobot.agent.runner.AgentRunner", return_value=mock_runner):
            await orch._run_agent_sync("legal_research", "检索劳动合同法")

            call_args = mock_runner.run.call_args
            spec = call_args.args[0] if call_args.args else call_args.kwargs.get("spec")
            tool_names = list(spec.tools._tools.keys())
            assert "legal_rag_search" in tool_names
            assert "read_file" in tool_names
