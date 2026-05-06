"""LLM-based case structure extraction from raw case chunks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from legalbot.rag.case_types import CaseCoreData


CASE_EXTRACTION_PROMPT = """\
你是一个法律案例分析助手。请从以下案例文本中提取结构化信息。

## 需要提取的字段
- case_no：法院判决书的编号（如：(2021)沪01民终1234号）
- dispute_focus：法院认定的核心法律争议（1-2句话）
- ruling_rule：法院裁判该争议的具体规则或逻辑
- applicable_laws：判决适用的具体法律条文（数组格式，每个元素为"《法律全称》第X条"）

## 输出格式
请用JSON格式输出，字段名为：case_no, dispute_focus, ruling_rule, applicable_laws

## 案例文本
---
{case_text}
---

## 要求
1. 只输出JSON，不要有其他文字
2. 如果某个字段无法从文本确定，输出null
3. 适用法条用数组格式，每个元素为"《法律全称》第X条"格式
"""


class CaseAnalyzer:
    """Extracts structured data from raw case chunk texts using LLM."""

    def __init__(self, provider: Any):
        self._provider = provider

    async def analyze_single(self, chunk: Any) -> "CaseCoreData":
        """Analyze a single case chunk and extract structured data."""
        from legalbot.rag.case_types import CaseCoreData

        case_text = chunk.chunk.text if hasattr(chunk, "chunk") else str(chunk)
        prompt = CASE_EXTRACTION_PROMPT.format(case_text=case_text[:3000])
        messages = [{"role": "user", "content": prompt}]
        response = await self._provider.chat(messages=messages, temperature=0.1)
        content = (response.content or "").strip()

        data = self._parse_json(content)
        source_id = chunk.chunk.id if hasattr(chunk, "chunk") else None

        return CaseCoreData(
            case_no=data.get("case_no"),
            dispute_focus=data.get("dispute_focus"),
            ruling_rule=data.get("ruling_rule"),
            applicable_laws=data.get("applicable_laws", []),
            source_chunk_id=source_id,
        )

    def _parse_json(self, content: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = content
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.rfind("```")
            if end > start:
                text = text[start:end]
        elif "```" in text:
            start = text.find("```") + 3
            end = text.rfind("```")
            if end > start:
                text = text[start:end]

        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    async def analyze_batch(self, retrieval_results: list) -> list["CaseCoreData"]:
        """Analyze multiple case chunks in sequence."""
        from legalbot.rag.case_types import CaseCoreData

        results = []
        for r in retrieval_results:
            try:
                case_data = await self.analyze_single(r)
                results.append(case_data)
            except Exception:
                # Fallback: create minimal case data from raw text
                text = r.chunk.text if hasattr(r, "chunk") else str(r)
                results.append(CaseCoreData(
                    dispute_focus=text[:200],
                    applicable_laws=[],
                ))
        return results
