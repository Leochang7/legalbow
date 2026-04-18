"""多轮法律链式推理引擎 — Multi-Step Legal Reasoning Engine."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal


# Intent 常量
INTENT_COMPLEX_LEGAL_QUERY = "complex_legal_query"

# 复杂度分类 prompt
COMPLEXITY_CLASSIFICATION_PROMPT = """\
分析以下法律问题的复杂程度。

复杂度判断标准：
- simple: 仅需单条法律条文即可回答（如：法定婚龄是多少？）
- complex: 需要引用多条法律条文、进行逻辑推导、或涉及多个法律领域
  （如：用人单位拖欠工资且不签劳动合同如何维权？）

用户输入：{query}

返回类别：simple 或 complex

只返回类别名，不要解释。"""


REASONING_PROMPTS = {
    "initial_retrieval": """\
根据用户提出的法律问题，构造精确的检索 query。

用户问题：{question}

要求：
1. 识别问题涉及的法律领域
2. 提取核心法律概念，去除口语化表述
3. 识别是否需要同时检索多个相关法律

直接返回检索 query 列表，每行一个 query。用空格分隔关键词。""",

    "retrieval_response_analysis": """\
基于以下法律检索结果，分析法律问题。

检索结果：
{retrieved_laws}

用户问题：{question}

分析要求：
1. 列出每条检索结果与问题的相关性
2. 识别已找到的直接适用法条
3. 识别可能需要补充检索的相关法条

如发现需要补充检索的内容，在最后一行写：NEED_MORE_RETRIEVAL: [具体补充检索问题]""",

    "synthesis": """\
综合以下所有法律检索和分析，形成完整的法律分析结论。

用户问题：{question}

已确认的法律依据：
{confirmed_laws}

分析结论要求：
1. 清晰陈述法律结论
2. 每条结论必须附带具体法条引用（格式：《法律名》第X条）
3. 如涉及程序性权利（仲裁、诉讼）也要说明
4. 结论末尾附加风险提示和法律免责

用以下格式回答：

## 法律分析结论
[结论陈述]

## 法律依据
- [法条1]
- [法条2]
...

## 维权步骤
1. [步骤1]
2. [步骤2]
...

## 风险提示
[提示内容]

---
*以上分析基于当前知识库中的法律法规，仅供参考，不构成正式法律意见。*""",
}


@dataclass(frozen=True)
class ReasoningStep:
    """推理链中的单一步骤。"""
    step_id: int
    reasoning_type: Literal["retrieval", "analysis", "synthesis", "conclusion"]
    prompt: str
    law_retrieved: list[Any]  # 原始检索结果
    llm_reasoning: str
    citations: list[str]  # e.g. ["《劳动合同法》第10条", "《民法典》第576条"]
    next_action: Literal["continue", "stop", "retrieve_more"]
    follow_up_query: str | None = None


@dataclass
class ReasoningChain:
    """完整推理链。"""
    question: str
    steps: list[ReasoningStep] = field(default_factory=list)
    max_steps: int = 5
    iteration_count: int = 0

    def add_step(self, step: ReasoningStep) -> None:
        self.steps.append(step)
        self.iteration_count += 1

    def is_complete(self) -> bool:
        last = self.steps[-1] if self.steps else None
        return (
            len(self.steps) >= self.max_steps
            or (last is not None and last.next_action == "stop")
        )

    def to_display_string(self) -> str:
        lines = ["## 法律推理过程\n"]
        for step in self.steps:
            lines.append(f"### 第 {step.step_id} 步：{step.reasoning_type}")
            if step.llm_reasoning:
                lines.append(f"**推理**：{step.llm_reasoning}")
            if step.citations:
                lines.append(f"**引用**：{'；'.join(step.citations)}")
            if step.law_retrieved:
                lines.append(f"**检索到 {len(step.law_retrieved)} 条相关法规**")
            lines.append("")
        return "\n".join(lines)

    def collect_all_citations(self) -> list[str]:
        """收集所有步骤中的引用，去重并保持顺序。"""
        all_citations = []
        seen: set[str] = set()
        for step in self.steps:
            for c in step.citations:
                if c not in seen:
                    seen.add(c)
                    all_citations.append(c)
        return all_citations

    def truncate(self, max_steps: int) -> None:
        """截断最早的推理步骤，保留最后一个 synthesis。

        用于防止 context overflow。Synthesis 结论始终保留。
        """
        if len(self.steps) <= max_steps:
            return
        # 只保留最后一个 synthesis
        synthesis_steps = [s for s in self.steps if s.reasoning_type == "synthesis"]
        last_synthesis = synthesis_steps[-1] if synthesis_steps else None
        non_synthesis = [s for s in self.steps if s.reasoning_type != "synthesis"]
        slots_for_non_synthesis = max_steps - (1 if last_synthesis else 0)
        keep_non_synth = non_synthesis[-slots_for_non_synthesis:] if slots_for_non_synthesis > 0 else []
        self.steps = keep_non_synth + ([last_synthesis] if last_synthesis else [])

    def verify_citations(self) -> list[str]:
        """验证引用是否出现在检索结果中，返回通过验证的引用。

        引用 hallucination 检测：只保留确实出现在 law_retrieved 中的引用。
        """
        # 收集所有检索结果中出现过的法律名称+条号组合
        retrieved_texts: set[str] = set()
        for step in self.steps:
            law_results = step.law_retrieved
            # Handle both list and RetrievalPipelineResult
            if hasattr(law_results, "top_k"):
                law_results = law_results.top_k
            for r in law_results:
                chunk = getattr(r, "chunk", None)
                if chunk is None:
                    continue
                text = chunk.text if hasattr(chunk, "text") else str(chunk)
                retrieved_texts.add(text)

        verified: list[str] = []
        for citation in self.collect_all_citations():
            # 检查引用是否出现在任何检索文本中
            if any(citation in rt for rt in retrieved_texts):
                verified.append(citation)
        return verified


class MultiStepLegalReasoner:
    """多轮法律链式推理引擎。"""

    def __init__(
        self,
        provider: Any,
        retriever: Any,
        max_steps: int = 5,
    ):
        self._provider = provider
        self._retriever = retriever
        self._max_steps = max_steps

    async def reason(
        self,
        question: str,
        law_area: str | None = None,
    ) -> ReasoningChain:
        chain = ReasoningChain(question=question, max_steps=self._max_steps)

        # Step 1: Initial retrieval
        retrieval_results = await self._initial_retrieval(question, law_area)
        chain.add_step(ReasoningStep(
            step_id=1,
            reasoning_type="retrieval",
            prompt=question,
            law_retrieved=retrieval_results,
            llm_reasoning="已完成初始检索",
            citations=[],
            next_action="continue",
        ))

        # Step 2-N: Analysis + possible follow-up retrieval
        current_results = retrieval_results
        step_num = 2

        while not chain.is_complete() and step_num <= self._max_steps:
            # Analyze current results
            analysis_result = await self._analyze_retrieval(question, current_results)
            citations = analysis_result["citations"]
            next_action = analysis_result["next_action"]

            chain.add_step(ReasoningStep(
                step_id=step_num,
                reasoning_type="analysis",
                prompt="",
                law_retrieved=[],
                llm_reasoning=analysis_result["reasoning"],
                citations=citations,
                next_action=next_action,
            ))

            # Determine next step
            follow_up_query = analysis_result.get("follow_up_query")

            if next_action == "stop" or not follow_up_query:
                # No more retrieval needed — exit loop to add synthesis
                break

            if step_num >= self._max_steps:
                # Cannot do follow-up, would exceed max_steps
                break

            # Perform follow-up retrieval
            current_results = await self._retriever.retrieve(
                query=follow_up_query,
                law_area=law_area,
                top_k=5,
            )
            chain.add_step(ReasoningStep(
                step_id=step_num + 1,
                reasoning_type="retrieval",
                prompt=follow_up_query,
                law_retrieved=current_results,
                llm_reasoning=f"补充检索：{follow_up_query}",
                citations=[],
                next_action="continue",
            ))
            step_num += 2

        # Final synthesis step — use verified citations only
        synthesis_result = await self._synthesize(question, chain)
        verified_citations = chain.verify_citations()
        chain.add_step(ReasoningStep(
            step_id=len(chain.steps) + 1,
            reasoning_type="synthesis",
            prompt="",
            law_retrieved=[],
            llm_reasoning=synthesis_result,
            citations=verified_citations,
            next_action="stop",
        ))

        return chain

    async def _initial_retrieval(self, question: str, law_area: str | None):
        prompt = REASONING_PROMPTS["initial_retrieval"].format(question=question)
        response = await self._provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        queries = (response.content or question).strip().split("\n")
        results = []
        for q in queries[:3]:
            r = await self._retriever.retrieve(query=q.strip(), law_area=law_area, top_k=3)
            # Handle both RetrievalPipelineResult and plain list (mock retriever in unit tests)
            if hasattr(r, "top_k"):
                results.extend(r.top_k)
            else:
                results.extend(r)
        # Deduplicate by id
        seen: set[str] = set()
        deduped = []
        for r in results:
            if r.chunk.id not in seen:
                seen.add(r.chunk.id)
                deduped.append(r)
        return deduped[:5]

    async def _analyze_retrieval(self, question: str, retrieved: list) -> dict:
        laws_text = "\n".join(
            f"- {r.chunk.metadata.get('law_name', '')}{r.chunk.metadata.get('article_no', '')}: {r.chunk.text[:200]}"
            for r in retrieved
        )
        prompt = REASONING_PROMPTS["retrieval_response_analysis"].format(
            retrieved_laws=laws_text or "（无检索结果）",
            question=question,
        )
        response = await self._provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.content or ""
        citations = self._extract_citations(content)
        needs_more = "NEED_MORE_RETRIEVAL:" in content
        follow_up = None
        if needs_more:
            parts = content.split("NEED_MORE_RETRIEVAL:", 1)
            if len(parts) > 1:
                follow_up = parts[1].strip().split("\n")[0].strip()
        return {
            "reasoning": content[:1000],
            "citations": citations,
            "next_action": "retrieve_more" if needs_more else "stop",
            "follow_up_query": follow_up,
        }

    async def _synthesize(self, question: str, chain: ReasoningChain) -> str:
        all_citations = chain.collect_all_citations()
        confirmed_laws = "\n".join(f"- {c}" for c in all_citations) if all_citations else "（无确认法条）"
        prompt = REASONING_PROMPTS["synthesis"].format(
            question=question,
            confirmed_laws=confirmed_laws,
        )
        response = await self._provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.content or "综合分析完成，但未生成有效结论。"

    def _extract_citations(self, text: str) -> list[str]:
        """从文本中提取法律引用。"""
        pattern = r"《[^》]+》[^。，,\n]*?第[零一二三四五六七八九十百千\d]+条"
        matches = re.findall(pattern, text)
        # 去重保持顺序
        seen: set[str] = set()
        result = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result[:10]
