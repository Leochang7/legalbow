"""Case fact extraction for legal document generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaseFacts:
    """Structured case facts extracted from user input."""
    raw_text: str
    doc_type: str
    parties: dict[str, dict[str, str]] = field(default_factory=dict)
    monetary_amount: float | None = None
    case_type: str | None = None
    date_of_dispute: str | None = None
    contract_date: str | None = None
    additional_facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "原告姓名": self.parties.get("plaintiff", {}).get("name", ""),
            "被告姓名": self.parties.get("defendant", {}).get("name", ""),
            "争议金额": str(self.monetary_amount) if self.monetary_amount else "",
            "案由": self.case_type or "",
            "事实经过": self.additional_facts.get("facts", ""),
        }
        for party_type, fields in self.parties.items():
            for field_name, value in fields.items():
                if value:
                    result[f"{party_type}_{field_name}"] = value
        for k, v in self.additional_facts.items():
            if k not in ("facts",) and v:
                result[k] = v
        return result


class CaseFactsExtractor:
    """Extract structured CaseFacts from free-text case description using LLM."""

    EXTRACTION_PROMPT = """\
从以下案件描述中提取结构化信息，用于生成法律文书。

案件描述：
{case_text}

文档类型：{doc_type}

请以JSON格式返回，字段包括：
- parties: {{plaintiff: {{name, address, phone, id_number}}, defendant: {{name, address, phone, id_number}}}}
- monetary_amount: 涉及金额（数字，无金额则填null）
- case_type: 案件类型（民间借贷/买卖合同/租赁合同/侵权/劳动争议/婚姻家庭/其他）
- date_of_dispute: 纠纷发生日期（无法确定则填null）
- contract_date: 合同签订日期或借款日期（无法确定则填null）
- additional_facts: 其他重要事实（键值对格式）

只返回JSON，不要有任何其他内容。"""

    def __init__(self, provider: Any):
        self.provider = provider

    async def extract(self, case_text: str, doc_type: str) -> CaseFacts:
        prompt = self.EXTRACTION_PROMPT.format(case_text=case_text, doc_type=doc_type)
        response = await self.provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.content or "{}"
        try:
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.rfind("```")
                content = content[start:end]
            data = json.loads(content.strip())
        except json.JSONDecodeError:
            data = {"additional_facts": {"raw": case_text}}

        return CaseFacts(
            raw_text=case_text,
            doc_type=doc_type,
            parties=data.get("parties", {}),
            monetary_amount=data.get("monetary_amount"),
            case_type=data.get("case_type"),
            date_of_dispute=data.get("date_of_dispute"),
            contract_date=data.get("contract_date"),
            additional_facts=data.get("additional_facts", {}),
        )
