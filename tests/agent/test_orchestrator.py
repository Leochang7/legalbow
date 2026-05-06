"""Unit tests for LegalOrchestrator: intent classification, dispatch, and contract review flow."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from legalbot.agent.orchestrator import (
    INTENT_CASE_SEARCH,
    INTENT_CONTRACT_REVIEW,
    INTENT_GENERAL,
    INTENT_LEGAL_QUERY,
    INTENT_PROMPT,
    VALID_INTENTS,
    LegalOrchestrator,
)
from legalbot.config.schema import AgentDefConfig, OrchestrateConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(
    enable: bool = True,
    agents: dict[str, AgentDefConfig] | None = None,
) -> OrchestrateConfig:
    if agents is None:
        agents = {
            "legal_research": AgentDefConfig(
                system_prompt="你是法律检索专家",
                tools=["legal_rag_search", "web_search", "read_file"],
            ),
            "contract_review": AgentDefConfig(
                system_prompt="你是合同审查专家",
                tools=["legal_rag_search", "read_file"],
            ),
        }
    return OrchestrateConfig(enable=enable, agents=agents)


def _make_provider(response_text: str = "legal_query") -> MagicMock:
    """Create a mock LLMProvider that returns the given text."""
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


# ---------------------------------------------------------------------------
# classify_intent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_intent_legal_query():
    provider = _make_provider("legal_query")
    config = _make_config()
    orch = LegalOrchestrator(provider, _make_subagent_mgr(), config)

    result = await orch.classify_intent("用人单位不签劳动合同怎么办")
    assert result == INTENT_LEGAL_QUERY


@pytest.mark.asyncio
async def test_classify_intent_contract_review():
    provider = _make_provider("contract_review")
    config = _make_config()
    orch = LegalOrchestrator(provider, _make_subagent_mgr(), config)

    result = await orch.classify_intent("帮我审查这份租赁合同")
    assert result == INTENT_CONTRACT_REVIEW


@pytest.mark.asyncio
async def test_classify_intent_case_search():
    provider = _make_provider("case_search")
    config = _make_config()
    orch = LegalOrchestrator(provider, _make_subagent_mgr(), config)

    result = await orch.classify_intent("搜索劳动争议相关案例")
    assert result == INTENT_CASE_SEARCH


@pytest.mark.asyncio
async def test_classify_intent_general():
    provider = _make_provider("general")
    config = _make_config()
    orch = LegalOrchestrator(provider, _make_subagent_mgr(), config)

    result = await orch.classify_intent("今天天气怎样")
    assert result == INTENT_GENERAL


@pytest.mark.asyncio
async def test_classify_intent_unrecognized_falls_to_general():
    """If LLM returns something not in VALID_INTENTS, fall back to general."""
    provider = _make_provider("some_random_output")
    config = _make_config()
    orch = LegalOrchestrator(provider, _make_subagent_mgr(), config)

    result = await orch.classify_intent("测试问题")
    assert result == INTENT_GENERAL


@pytest.mark.asyncio
async def test_classify_intent_exception_falls_to_general():
    """If LLM call fails, fall back to general."""
    provider = _make_provider()
    provider.chat = AsyncMock(side_effect=Exception("API error"))
    config = _make_config()
    orch = LegalOrchestrator(provider, _make_subagent_mgr(), config)

    result = await orch.classify_intent("用人单位不签劳动合同怎么办")
    assert result == INTENT_GENERAL


@pytest.mark.asyncio
async def test_classify_intent_prompt_format():
    """Verify the prompt template is correctly formatted."""
    query = "测试问题"
    formatted = INTENT_PROMPT.format(query=query)
    assert query in formatted
    assert "legal_query" in formatted
    assert "contract_review" in formatted


# ---------------------------------------------------------------------------
# dispatch tests (fire-and-forget)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_disabled_returns_none():
    config = _make_config(enable=False)
    orch = LegalOrchestrator(_make_provider(), _make_subagent_mgr(), config)

    result = await orch.dispatch("用人单位不签劳动合同怎么办")
    assert result is None


@pytest.mark.asyncio
async def test_dispatch_general_returns_none():
    provider = _make_provider("general")
    config = _make_config()
    subagent_mgr = _make_subagent_mgr()
    orch = LegalOrchestrator(provider, subagent_mgr, config)

    result = await orch.dispatch("今天天气怎样")
    assert result is None


@pytest.mark.asyncio
async def test_dispatch_legal_query_routes_to_legal_research():
    provider = _make_provider("legal_query")
    config = _make_config()
    subagent_mgr = _make_subagent_mgr()
    orch = LegalOrchestrator(provider, subagent_mgr, config)

    result = await orch.dispatch("用人单位不签劳动合同怎么办")
    # Should have called spawn_with_config with legal_research agent config
    subagent_mgr.spawn_with_config.assert_called_once()
    call_kwargs = subagent_mgr.spawn_with_config.call_args.kwargs
    assert call_kwargs["label"] == "legal_research"
    assert "劳动合同" in call_kwargs["task"]


@pytest.mark.asyncio
async def test_dispatch_case_search_routes_to_legal_research():
    provider = _make_provider("case_search")
    config = _make_config()
    subagent_mgr = _make_subagent_mgr()
    orch = LegalOrchestrator(provider, subagent_mgr, config)

    result = await orch.dispatch("搜索劳动争议案例")
    subagent_mgr.spawn_with_config.assert_called_once()
    call_kwargs = subagent_mgr.spawn_with_config.call_args.kwargs
    assert call_kwargs["label"] == "legal_research"


@pytest.mark.asyncio
async def test_dispatch_contract_review_routes_to_legal_research():
    """Contract review dispatch first step is legal research (fire-and-forget)."""
    provider = _make_provider("contract_review")
    config = _make_config()
    subagent_mgr = _make_subagent_mgr()
    orch = LegalOrchestrator(provider, subagent_mgr, config)

    result = await orch.dispatch("帮我审查这份租赁合同")
    subagent_mgr.spawn_with_config.assert_called_once()
    call_kwargs = subagent_mgr.spawn_with_config.call_args.kwargs
    assert call_kwargs["label"] == "legal_research"


@pytest.mark.asyncio
async def test_dispatch_agent_not_found():
    """If agent name not in config, returns None."""
    provider = _make_provider("legal_query")
    config = _make_config(agents={})  # No agents defined
    subagent_mgr = _make_subagent_mgr()
    orch = LegalOrchestrator(provider, subagent_mgr, config)

    result = await orch.dispatch("用人单位不签劳动合同怎么办")
    assert result is None


# ---------------------------------------------------------------------------
# dispatch_sync tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_sync_disabled():
    config = _make_config(enable=False)
    orch = LegalOrchestrator(_make_provider(), _make_subagent_mgr(), config)

    result = await orch.dispatch_sync("用人单位不签劳动合同怎么办")
    assert "未启用" in result


@pytest.mark.asyncio
async def test_dispatch_sync_general_returns_none():
    provider = _make_provider("general")
    config = _make_config()
    orch = LegalOrchestrator(provider, _make_subagent_mgr(), config)

    result = await orch.dispatch_sync("今天天气怎样")
    assert result is None


@pytest.mark.asyncio
async def test_dispatch_sync_legal_query_runs_agent():
    """dispatch_sync for legal_query should run the agent synchronously."""
    provider = _make_provider("legal_query")
    config = _make_config()
    subagent_mgr = _make_subagent_mgr()

    mock_result = MagicMock()
    mock_result.final_content = "根据《劳动合同法》第82条..."
    mock_result.stop_reason = "stop"
    mock_result.error = None

    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(return_value=mock_result)

    orch = LegalOrchestrator(provider, subagent_mgr, config)
    with patch("legalbot.agent.runner.AgentRunner", return_value=mock_runner):
        result = await orch.dispatch_sync("用人单位不签劳动合同怎么办")
        assert "劳动合同法" in result


# ---------------------------------------------------------------------------
# contract review flow tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contract_review_flow_sync():
    """Contract review flow: research -> review."""
    provider = _make_provider("contract_review")
    config = _make_config()
    subagent_mgr = _make_subagent_mgr()

    research_result = MagicMock()
    research_result.final_content = "相关法律条文：民法典第585条..."
    research_result.stop_reason = "stop"
    research_result.error = None

    review_result = MagicMock()
    review_result.final_content = "合同风险点：1. 违约金条款不合规..."
    review_result.stop_reason = "stop"
    review_result.error = None

    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(side_effect=[research_result, review_result])

    orch = LegalOrchestrator(provider, subagent_mgr, config)
    with patch("legalbot.agent.runner.AgentRunner", return_value=mock_runner):
        result = await orch._contract_review_flow_sync("审查这份租赁合同")
        assert "风险" in result
        assert mock_runner.run.call_count == 2


# ---------------------------------------------------------------------------
# config tests
# ---------------------------------------------------------------------------


def test_orchestrate_config_defaults():
    config = OrchestrateConfig()
    assert config.enable is False
    assert config.intent_model == ""
    assert config.agents == {}


def test_agent_def_config_defaults():
    agent = AgentDefConfig()
    assert agent.system_prompt == ""
    assert agent.tools == []
    assert agent.model == ""


def test_orchestrate_config_with_agents():
    config = _make_config()
    assert config.enable is True
    assert "legal_research" in config.agents
    assert "contract_review" in config.agents
    assert config.agents["legal_research"].system_prompt == "你是法律检索专家"
    assert "legal_rag_search" in config.agents["legal_research"].tools
