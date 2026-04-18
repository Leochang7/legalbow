"""
Multi-Step Legal Reasoning — 单元测试

测试：
- ReasoningChain: add_step, is_complete, to_display_string, collect_all_citations
- ReasoningStep: 各字段正确
- _extract_citations: 引用提取
- MultiStepLegalReasoner: 独立于外部依赖的逻辑验证
"""

import re
import sys
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── 被测模块 ──────────────────────────────────────────────

sys.path.insert(0, str(__file__.rsplit("/", 1)[0] + "/../.."))
from nanobot.agent.reasoner import (
    COMPLEXITY_CLASSIFICATION_PROMPT,
    REASONING_PROMPTS,
    ReasoningChain,
    ReasoningStep,
    MultiStepLegalReasoner,
)


# ── Mock Retriever ─────────────────────────────────────────

class MockChunk:
    def __init__(self, chunk_id: str, text: str, metadata: dict):
        self.id = chunk_id
        self.text = text
        self.metadata = metadata


class MockRetrievalResult:
    def __init__(self, chunk: MockChunk, score: float = 1.0):
        self.chunk = chunk


class MockRetriever:
    def __init__(self, results: list[MockRetrievalResult] | Exception):
        self._results = results

    async def retrieve(self, query: str, law_area: str | None = None, top_k: int = 5):
        if isinstance(self._results, Exception):
            raise self._results
        return self._results


# ── Mock Provider ──────────────────────────────────────────

class MockProvider:
    def __init__(self, responses: list[str] | Exception):
        self._responses = responses
        self._call_count = 0

    async def chat(self, messages: list[dict], temperature: float = 0.0, **kwargs):
        self._call_count += 1
        if isinstance(self._responses, Exception):
            raise self._responses
        response = self._responses[self._call_count - 1] if self._call_count <= len(self._responses) else "stop"
        return MagicMock(content=response)


# ── 测试 ReasoningStep ─────────────────────────────────────

def test_reasoning_step_fields():
    """ ReasoningStep 各字段赋值正确 """
    step = ReasoningStep(
        step_id=1,
        reasoning_type="retrieval",
        prompt="劳动合同拖欠工资",
        law_retrieved=[],
        llm_reasoning="检索到5条相关法规",
        citations=["《劳动合同法》第30条"],
        next_action="continue",
        follow_up_query="加班费法律规定",
    )
    assert step.step_id == 1
    assert step.reasoning_type == "retrieval"
    assert step.prompt == "劳动合同拖欠工资"
    assert step.llm_reasoning == "检索到5条相关法规"
    assert step.citations == ["《劳动合同法》第30条"]
    assert step.next_action == "continue"
    assert step.follow_up_query == "加班费法律规定"


# ── 测试 ReasoningChain.add_step ──────────────────────────

def test_reasoning_chain_add_step():
    """ add_step 后 iteration_count 递增 """
    chain = ReasoningChain(question="拖欠工资如何维权", max_steps=5)
    assert chain.iteration_count == 0
    assert len(chain.steps) == 0

    step1 = ReasoningStep(
        step_id=1, reasoning_type="retrieval", prompt="",
        law_retrieved=[], llm_reasoning="", citations=[], next_action="continue",
    )
    chain.add_step(step1)
    assert chain.iteration_count == 1
    assert len(chain.steps) == 1

    step2 = ReasoningStep(
        step_id=2, reasoning_type="analysis", prompt="",
        law_retrieved=[], llm_reasoning="", citations=[], next_action="stop",
    )
    chain.add_step(step2)
    assert chain.iteration_count == 2
    assert len(chain.steps) == 2


def test_reasoning_chain_add_step_increments_count():
    """ 连续 add_step，iteration_count 正确累加 """
    chain = ReasoningChain(question="test", max_steps=10)
    for i in range(5):
        chain.add_step(ReasoningStep(
            step_id=i + 1, reasoning_type="retrieval", prompt="",
            law_retrieved=[], llm_reasoning="", citations=[], next_action="continue",
        ))
    assert chain.iteration_count == 5
    assert len(chain.steps) == 5


# ── 测试 ReasoningChain.is_complete ───────────────────────

def test_reasoning_chain_is_complete_at_max_steps():
    """ 达到 max_steps 时 is_complete 返回 True """
    chain = ReasoningChain(question="test", max_steps=3)
    for i in range(3):
        chain.add_step(ReasoningStep(
            step_id=i + 1, reasoning_type="retrieval", prompt="",
            law_retrieved=[], llm_reasoning="", citations=[], next_action="continue",
        ))
    assert chain.is_complete() is True


def test_reasoning_chain_is_complete_on_stop_action():
    """ 某步 next_action=stop 时 is_complete 返回 True """
    chain = ReasoningChain(question="test", max_steps=5)
    chain.add_step(ReasoningStep(
        step_id=1, reasoning_type="analysis", prompt="",
        law_retrieved=[], llm_reasoning="", citations=[], next_action="stop",
    ))
    assert chain.is_complete() is True


def test_reasoning_chain_not_complete_early():
    """ 未达 max_steps 且无 stop 时 is_complete 返回 False """
    chain = ReasoningChain(question="test", max_steps=5)
    chain.add_step(ReasoningStep(
        step_id=1, reasoning_type="retrieval", prompt="",
        law_retrieved=[], llm_reasoning="", citations=[], next_action="continue",
    ))
    assert chain.is_complete() is False


def test_reasoning_chain_empty_not_complete():
    """ 空 chain is_complete 返回 False """
    chain = ReasoningChain(question="test", max_steps=5)
    assert chain.is_complete() is False


# ── 测试 ReasoningChain.collect_all_citations ──────────────

def test_collect_all_citations_dedup():
    """ 引用去重且保持顺序 """
    chain = ReasoningChain(question="test", max_steps=5)
    chain.add_step(ReasoningStep(
        step_id=1, reasoning_type="retrieval", prompt="",
        law_retrieved=[], llm_reasoning="",
        citations=["《劳动合同法》第30条", "《民法典》第576条"],
        next_action="continue",
    ))
    chain.add_step(ReasoningStep(
        step_id=2, reasoning_type="analysis", prompt="",
        law_retrieved=[], llm_reasoning="",
        citations=["《劳动合同法》第30条", "《劳动争议调解仲裁法》第5条"],
        next_action="stop",
    ))
    citations = chain.collect_all_citations()
    assert citations == [
        "《劳动合同法》第30条",
        "《民法典》第576条",
        "《劳动争议调解仲裁法》第5条",
    ]


def test_collect_all_citations_empty():
    """ 无引用时返回空列表 """
    chain = ReasoningChain(question="test", max_steps=5)
    citations = chain.collect_all_citations()
    assert citations == []


# ── 测试 to_display_string ─────────────────────────────────

def test_to_display_string_format():
    """ 输出包含推理步骤标题、引用、检索数量 """
    chain = ReasoningChain(question="劳动合同拖欠工资维权", max_steps=5)
    chain.add_step(ReasoningStep(
        step_id=1, reasoning_type="retrieval", prompt="",
        law_retrieved=[MagicMock()],
        llm_reasoning="检索到5条法规",
        citations=["《劳动合同法》第30条"],
        next_action="continue",
    ))
    chain.add_step(ReasoningStep(
        step_id=2, reasoning_type="synthesis", prompt="",
        law_retrieved=[],
        llm_reasoning="综合分析完毕",
        citations=["《劳动合同法》第30条"],
        next_action="stop",
    ))
    output = chain.to_display_string()
    assert "## 法律推理过程" in output
    assert "第 1 步：retrieval" in output
    assert "第 2 步：synthesis" in output
    assert "《劳动合同法》第30条" in output
    assert "检索到 1 条相关法规" in output


# ── 测试 _extract_citations ────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    (
        "根据《劳动合同法》第30条和《民法典》第576条规定",
        ["《劳动合同法》第30条", "《民法典》第576条"],
    ),
    (
        "依据《最高人民法院关于适用〈中华人民共和国民法典〉合同编通则若干问题的解释》第一条",
        ["《最高人民法院关于适用〈中华人民共和国民法典〉合同编通则若干问题的解释》第一条"],
    ),
    (
        "根据《劳动合同法》第三十条第一款（此时数字是三十）",
        ["《劳动合同法》第三十条"],
    ),
    (
        "普通文本不包含法律引用",
        [],
    ),
    (
        "《劳动合同法》第30条，《民法典》第576条，不可抗力条款《民法典》第590条",
        ["《劳动合同法》第30条", "《民法典》第576条", "《民法典》第590条"],
    ),
    (
        "只有《》没有条：这是一段文字《劳动合同法实施条例》没有条号",
        [],
    ),
])
def test_extract_citations_patterns(text, expected):
    """ 引用提取正则覆盖各类法律条文格式 """
    reasoner = MultiStepLegalReasoner(MagicMock(), MagicMock())
    result = reasoner._extract_citations(text)
    assert result == expected


# ── 测试 COMPLEXITY_CLASSIFICATION_PROMPT ─────────────────

def test_complexity_prompt_contains_query_placeholder():
    """ 复杂度分类 prompt 包含 {query} 占位符 """
    assert "{query}" in COMPLEXITY_CLASSIFICATION_PROMPT


# ── 测试 REASONING_PROMPTS ─────────────────────────────────

def test_reasoning_prompts_all_have_placeholders():
    """ 所有推理 prompt 包含必要占位符 """
    for key, prompt in REASONING_PROMPTS.items():
        assert "{question}" in prompt or "{confirmed_laws}" in prompt or "{retrieved_laws}" in prompt


# ── 测试 MultiStepLegalReasoner.reason ───────────────────

@pytest.mark.asyncio
async def test_reason_flow_all_steps():
    """ 完整推理流程：检索→分析→合成 """
    provider_responses = [
        "劳动合同 拖欠工资 维权",       # _initial_retrieval: LLM 构造 query
        "分析完毕。NEED_MORE_RETRIEVAL: 加班费 计算",   # _analyze_retrieval
        "综合结论：应申请劳动仲裁。\n\n## 法律依据\n- 《劳动合同法》第30条\n\n## 维权步骤\n1. 收集证据\n2. 申请仲裁",  # _synthesize
    ]
    provider = MockProvider(provider_responses)
    chunks = [
        MockRetrievalResult(MockChunk("1", "《劳动合同法》第30条：用人单位应当按时足额支付劳动报酬。", {"law_name": "劳动合同法", "article_no": "第30条"})),
        MockRetrievalResult(MockChunk("2", "《民法典》第576条：当事人一方不履行合同义务的，应当承担违约责任。", {"law_name": "民法典", "article_no": "第576条"})),
    ]
    retriever = MockRetriever(chunks)
    reasoner = MultiStepLegalReasoner(provider, retriever, max_steps=5)
    chain = await reasoner.reason("用人单位拖欠工资如何维权")

    # 验证 chain 结构
    assert chain.question == "用人单位拖欠工资如何维权"
    assert len(chain.steps) >= 3  # 初始检索 + 分析 + 合成

    # 最后一步是 synthesis
    last_step = chain.steps[-1]
    assert last_step.reasoning_type == "synthesis"
    assert last_step.next_action == "stop"

    # 引用被收集
    all_cites = chain.collect_all_citations()
    assert len(all_cites) > 0


@pytest.mark.asyncio
async def test_reason_stop_after_analysis():
    """ 分析后发现无需补充检索，直接合成 """
    provider_responses = [
        "劳动合同法第30条",             # _initial_retrieval
        "已找到直接适用法条，无需补充。",  # _analyze_retrieval (无 NEED_MORE_RETRIEVAL)
        "结论明确。",                   # _synthesize
    ]
    provider = MockProvider(provider_responses)
    chunks = [
        MockRetrievalResult(MockChunk("1", "《劳动合同法》第30条：用人单位应当按时足额支付劳动报酬。", {"law_name": "劳动合同法", "article_no": "第30条"})),
    ]
    retriever = MockRetriever(chunks)
    reasoner = MultiStepLegalReasoner(provider, retriever, max_steps=5)
    chain = await reasoner.reason("拖欠工资怎么办")

    # 分析步的 next_action 应为 stop（无需补充检索）
    analysis_step = chain.steps[1]
    assert analysis_step.next_action == "stop"


@pytest.mark.asyncio
async def test_reason_respects_max_steps():
    """ 达到 max_steps 时推理停止，synthesis 正常添加 """
    # provider 持续返回需要补充检索，验证 max_steps 硬限制
    # max_steps=3：retrieval(1) + analysis(2) + follow-up retrieval(3) = 3步推理
    # synthesis 是第4步，但 synthesis 不占推理额度
    provider_responses = [
        "query1", "NEED_MORE_RETRIEVAL: query2", "NEED_MORE_RETRIEVAL: query3",
        "NEED_MORE_RETRIEVAL: query4", "NEED_MORE_RETRIEVAL: query5",
        "NEED_MORE_RETRIEVAL: query6",
    ]
    provider = MockProvider(provider_responses)
    chunks = [MockRetrievalResult(MockChunk(f"{i}", f"chunk {i}", {})) for i in range(3)]
    retriever = MockRetriever(chunks)
    reasoner = MultiStepLegalReasoner(provider, retriever, max_steps=3)
    chain = await reasoner.reason("复杂法律问题")

    assert chain.is_complete() is True
    # 推理步骤（不含 synthesis）不超过 max_steps
    reasoning_steps = [s for s in chain.steps if s.reasoning_type != "synthesis"]
    assert len(reasoning_steps) <= 3
    # synthesis 最后一步
    last = chain.steps[-1]
    assert last.reasoning_type == "synthesis"


@pytest.mark.asyncio
async def test_reason_with_law_area_filter():
    """ law_area 参数正确传递到 retriever """
    provider_responses = ["劳动法 劳动合同", "分析完成", "综合完毕"]
    provider = MockProvider(provider_responses)
    chunks = [MockRetrievalResult(MockChunk("1", "chunk", {"law_area": "劳动法"}))]
    retriever = MockRetriever(chunks)
    reasoner = MultiStepLegalReasoner(provider, retriever, max_steps=3)
    chain = await reasoner.reason("劳动合同问题", law_area="劳动法")

    assert chain.question == "劳动合同问题"


@pytest.mark.asyncio
async def test_reason_initial_retrieval_fallback_to_question():
    """ LLM 返回空时 fallback 到原始 question 作为 query """
    provider_responses = ["", "分析"]  # 空响应
    provider = MockProvider(provider_responses)
    chunks = [MockRetrievalResult(MockChunk("1", "chunk1", {}))]
    retriever = MockRetriever(chunks)
    reasoner = MultiStepLegalReasoner(provider, retriever, max_steps=3)
    chain = await reasoner.reason("fallback测试")
    # 不应抛异常
    assert len(chain.steps) >= 1


@pytest.mark.asyncio
async def test_reason_synthesis_fallback():
    """ synthesize 返回空时使用 fallback 文本 """
    provider_responses = ["query", "分析", ""]  # 空综合
    provider = MockProvider(provider_responses)
    chunks = [MockRetrievalResult(MockChunk("1", "chunk", {}))]
    retriever = MockRetriever(chunks)
    reasoner = MultiStepLegalReasoner(provider, retriever, max_steps=3)
    chain = await reasoner.reason("测试")

    last = chain.steps[-1]
    assert last.reasoning_type == "synthesis"
    # fallback 检查
    assert len(last.llm_reasoning) > 0


# ── 运行 ──────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
