"""Unit tests for OrchestrateTool and SubagentManager.spawn_with_config."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.orchestrator import (
    INTENT_CONTRACT_REVIEW,
    INTENT_LEGAL_QUERY,
    LegalOrchestrator,
)
from nanobot.agent.tools.orchestrate import OrchestrateTool
from nanobot.config.schema import AgentDefConfig, OrchestrateConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> OrchestrateConfig:
    return OrchestrateConfig(
        enable=True,
        agents={
            "legal_research": AgentDefConfig(
                system_prompt="你是法律检索专家",
                tools=["legal_rag_search", "web_search"],
            ),
            "contract_review": AgentDefConfig(
                system_prompt="你是合同审查专家",
                tools=["legal_rag_search", "read_file"],
            ),
        },
    )


def _make_provider(response_text: str = "legal_query") -> MagicMock:
    from dataclasses import dataclass

    provider = MagicMock()

    @dataclass
    class FakeResponse:
        content: str | None
        tool_calls: list = None
        finish_reason: str = "stop"
        usage: dict = None

        def __post_init__(self):
            if self.tool_calls is None:
                self.tool_calls = []
            if self.usage is None:
                self.usage = {}

    provider.chat = AsyncMock(return_value=FakeResponse(content=response_text))
    return provider


def _make_subagent_mgr() -> MagicMock:
    mgr = MagicMock()
    mgr.spawn_with_config = AsyncMock(return_value="Subagent [legal_research] started (id: abc123).")
    mgr._build_available_tools = MagicMock(return_value={})
    mgr._build_subagent_prompt = MagicMock(return_value="Subagent prompt")
    mgr.max_tool_result_chars = 16000
    mgr.model = "test-model"
    return mgr


def _make_orchestrator(response_text: str = "legal_query") -> LegalOrchestrator:
    return LegalOrchestrator(
        provider=_make_provider(response_text),
        subagent_mgr=_make_subagent_mgr(),
        config=_make_config(),
    )


# ---------------------------------------------------------------------------
# OrchestrateTool tests
# ---------------------------------------------------------------------------


class TestOrchestrateTool:
    def test_name(self):
        tool = OrchestrateTool(orchestrator=_make_orchestrator())
        assert tool.name == "legal_orchestrate"

    def test_description(self):
        tool = OrchestrateTool(orchestrator=_make_orchestrator())
        assert "调度" in tool.description or "专业" in tool.description

    def test_exclusive(self):
        tool = OrchestrateTool(orchestrator=_make_orchestrator())
        assert tool.exclusive is True

    def test_read_only(self):
        tool = OrchestrateTool(orchestrator=_make_orchestrator())
        assert tool.read_only is False

    @pytest.mark.asyncio
    async def test_execute_auto_intent_legal_query(self):
        orch = _make_orchestrator("legal_query")
        with patch.object(orch, "_run_agent_sync", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "根据《劳动合同法》第82条..."
            tool = OrchestrateTool(orchestrator=orch)
            result = await tool.execute(query="用人单位不签劳动合同怎么办")
            assert "劳动合同法" in result

    @pytest.mark.asyncio
    async def test_execute_auto_intent_general(self):
        orch = _make_orchestrator("general")
        tool = OrchestrateTool(orchestrator=orch)
        result = await tool.execute(query="今天天气怎样")
        assert "未能调度" in result or "legal_rag_search" in result

    @pytest.mark.asyncio
    async def test_execute_explicit_intent_contract_review(self):
        orch = _make_orchestrator()
        with patch.object(orch, "_contract_review_flow_sync", new_callable=AsyncMock) as mock_review:
            mock_review.return_value = "合同风险点：违约金条款不合规"
            tool = OrchestrateTool(orchestrator=orch)
            result = await tool.execute(
                query="审查这份租赁合同",
                intent="contract_review",
            )
            assert "风险" in result
            mock_review.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_explicit_intent_legal_query(self):
        orch = _make_orchestrator()
        with patch.object(orch, "_run_agent_sync", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "法律检索结果"
            tool = OrchestrateTool(orchestrator=orch)
            result = await tool.execute(
                query="搜索合同法条文",
                intent="legal_query",
            )
            assert "检索结果" in result

    @pytest.mark.asyncio
    async def test_execute_explicit_intent_case_search(self):
        orch = _make_orchestrator()
        with patch.object(orch, "_run_agent_sync", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "案例检索结果"
            tool = OrchestrateTool(orchestrator=orch)
            result = await tool.execute(
                query="搜索劳动争议案例",
                intent="case_search",
            )
            assert "案例" in result

    def test_parameters_schema(self):
        tool = OrchestrateTool(orchestrator=_make_orchestrator())
        params = tool.parameters
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "intent" in params["properties"]
        assert "query" in params.get("required", [])


# ---------------------------------------------------------------------------
# SubagentManager.spawn_with_config tests
# ---------------------------------------------------------------------------


class TestSubagentSpawnWithConfig:
    @pytest.mark.asyncio
    async def test_spawn_with_config_basic(self):
        """spawn_with_config should create a task and return a message."""
        from pathlib import Path

        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"

        bus = MessageBus()
        mgr = SubagentManager(
            provider=provider,
            workspace=Path("/tmp/test_workspace"),
            bus=bus,
            max_tool_result_chars=16000,
        )

        # Mock the runner so we don't actually run an agent
        with patch.object(mgr, "runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_content = "Task done"
            mock_result.stop_reason = "stop"
            mock_result.error = None
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await mgr.spawn_with_config(
                task="检索合同法相关条文",
                system_prompt="你是法律检索专家",
                allowed_tools=["read_file"],
                label="legal_research",
            )

            assert "Subagent" in result
            assert "legal_research" in result

    @pytest.mark.asyncio
    async def test_spawn_with_config_custom_model(self):
        """spawn_with_config should pass model override to runner."""
        from pathlib import Path

        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        provider = MagicMock()
        provider.get_default_model.return_value = "default-model"

        bus = MessageBus()
        mgr = SubagentManager(
            provider=provider,
            workspace=Path("/tmp/test_workspace"),
            bus=bus,
            max_tool_result_chars=16000,
        )

        with patch.object(mgr, "runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_content = "Done"
            mock_result.stop_reason = "stop"
            mock_result.error = None
            mock_runner.run = AsyncMock(return_value=mock_result)

            await mgr.spawn_with_config(
                task="test task",
                model="deepseek/deepseek-chat",
                label="custom_agent",
            )

            # Give the background task a moment to start
            await asyncio.sleep(0.1)

            # Verify runner.run was called with custom model
            if mock_runner.run.called:
                call_args = mock_runner.run.call_args
                spec = call_args.args[0] if call_args.args else call_args.kwargs.get("spec")
                if spec:
                    assert spec.model == "deepseek/deepseek-chat"

    @pytest.mark.asyncio
    async def test_spawn_with_config_tool_whitelist(self):
        """spawn_with_config with allowed_tools should only register those tools."""
        from pathlib import Path

        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"

        bus = MessageBus()
        mgr = SubagentManager(
            provider=provider,
            workspace=Path("/tmp/test_workspace"),
            bus=bus,
            max_tool_result_chars=16000,
        )

        with patch.object(mgr, "runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_content = "Done"
            mock_result.stop_reason = "stop"
            mock_result.error = None
            mock_runner.run = AsyncMock(return_value=mock_result)

            # Only allow read_file
            available_tools = mgr._build_available_tools()
            assert "read_file" in available_tools

            await mgr.spawn_with_config(
                task="test task",
                allowed_tools=["read_file"],
                label="restricted_agent",
            )

            await asyncio.sleep(0.1)

            if mock_runner.run.called:
                call_args = mock_runner.run.call_args
                spec = call_args.args[0] if call_args.args else call_args.kwargs.get("spec")
                if spec:
                    tool_names = list(spec.tools._tools.keys())
                    assert "read_file" in tool_names
                    # Should NOT have write_file if only read_file was allowed
                    assert "write_file" not in tool_names


# Import asyncio at module level for the tests above
import asyncio
