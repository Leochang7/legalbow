"""Unit tests for debate module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from legalbot.agent.orchestrator import DebateInput, DebateResult, LegalOrchestrator
from legalbot.config.schema import AgentDefConfig, DebateConfig, OrchestrateConfig


@pytest.fixture
def debate_config():
    cfg = DebateConfig(enable=True, timeout_per_agent=30, timeout_total=60)
    return cfg


@pytest.fixture
def orchestrator_config(debate_config):
    return OrchestrateConfig(enable=True, debate=debate_config)


@pytest.fixture
def mock_orchestrator(orchestrator_config):
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=MagicMock(content="general"))
    subagent_mgr = MagicMock()
    subagent_mgr._build_available_tools = MagicMock(return_value={})
    subagent_mgr.model = "deepseek-chat"
    subagent_mgr.max_tool_result_chars = 16000
    orch = LegalOrchestrator(
        provider=provider,
        subagent_mgr=subagent_mgr,
        config=orchestrator_config,
        main_tools={},
    )
    return orch


class TestDebateInput:
    def test_debate_input_basic(self):
        inp = DebateInput(
            case_description="甲向乙借款10万元，约定一年后还本付息",
            plaintiff_claims="要求甲返还借款10万元及利息",
            defendant_response="已过诉讼时效",
        )
        assert inp.case_description == "甲向乙借款10万元，约定一年后还本付息"
        assert inp.plaintiff_claims == "要求甲返还借款10万元及利息"
        assert inp.defendant_response == "已过诉讼时效"

    def test_debate_input_optional(self):
        inp = DebateInput(case_description="简单案情")
        assert inp.case_description == "简单案情"
        assert inp.plaintiff_claims is None
        assert inp.defendant_response is None


class TestDebateConfig:
    def test_debate_config_defaults(self):
        cfg = DebateConfig()
        assert cfg.enable is False
        assert cfg.rounds == 1
        assert cfg.timeout_per_agent == 120
        assert cfg.timeout_total == 300

    def test_get_default_debate_agents(self):
        cfg = DebateConfig(enable=True)
        agents = cfg.get_default_debate_agents()
        assert "plaintiff_agent" in agents
        assert "defendant_agent" in agents
        assert "judge_agent" in agents
        assert "legal_rag_search" in agents["plaintiff_agent"].tools


class TestLegalOrchestratorDebate:
    @pytest.mark.asyncio
    async def test_build_plaintiff_task(self, mock_orchestrator):
        inp = DebateInput(
            case_description="甲向乙借款10万元",
            plaintiff_claims="要求返还本金",
            defendant_response="已过时效",
        )
        task = mock_orchestrator._build_plaintiff_task(inp)
        assert "甲向乙借款10万元" in task
        assert "要求返还本金" in task
        assert "原告代理律师" in task

    @pytest.mark.asyncio
    async def test_build_defendant_task(self, mock_orchestrator):
        inp = DebateInput(
            case_description="甲向乙借款10万元",
            plaintiff_claims="要求返还本金",
            defendant_response="已过时效",
        )
        task = mock_orchestrator._build_defendant_task(inp)
        assert "甲向乙借款10万元" in task
        assert "已过时效" in task
        assert "被告代理律师" in task

    @pytest.mark.asyncio
    async def test_build_judge_task(self, mock_orchestrator):
        inp = DebateInput(
            case_description="甲向乙借款10万元",
            plaintiff_claims="要求返还本金",
            defendant_response="已过时效",
        )
        task = mock_orchestrator._build_judge_task(
            inp,
            plaintiff_arguments="原告论证内容",
            defendant_arguments="被告论证内容",
        )
        assert "争议焦点分析报告" in task
        assert "原告论证内容" in task
        assert "被告论证内容" in task

    def test_get_debate_agent_config(self, mock_orchestrator):
        # Should return default debate agents
        cfg = mock_orchestrator._get_debate_agent_config("plaintiff")
        assert cfg is not None
        assert "原告代理律师" in cfg.system_prompt

    def test_get_debate_agent_config_judge(self, mock_orchestrator):
        cfg = mock_orchestrator._get_debate_agent_config("judge")
        assert cfg is not None
        assert "审判法官" in cfg.system_prompt

    def test_format_debate_result(self, mock_orchestrator):
        result = DebateResult(
            plaintiff_arguments="原告认为应该还钱",
            defendant_arguments="被告说已过时效",
            judge_report="争议焦点：诉讼时效问题",
            metadata={"rounds": 1},
        )
        output = mock_orchestrator._format_debate_result(result)
        assert "法律辩论分析报告" in output
        assert "争议焦点：诉讼时效问题" in output
        assert "原告认为应该还钱" in output
        assert "被告说已过时效" in output
        assert "附录" in output


class TestDebateTool:
    @pytest.mark.asyncio
    async def test_debate_tool_calls_orchestrator(self):
        orchestrator = MagicMock()
        orchestrator.run_debate_sync = AsyncMock(return_value="辩论报告内容")

        from legalbot.agent.tools.debate import DebateTool
        tool = DebateTool(orchestrator=orchestrator)

        result = await tool.execute(
            case_description="甲向乙借款10万元",
            plaintiff_claims="要求返还",
            debate_rounds=1,
        )

        orchestrator.run_debate_sync.assert_called_once_with(
            case_description="甲向乙借款10万元",
            plaintiff_claims="要求返还",
            defendant_response=None,
            debate_rounds=1,
        )
        assert result == "辩论报告内容"
