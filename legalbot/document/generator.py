"""Legal document generator — orchestrates retrieval + template filling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from legalbot.document.templates.base import LegalDocumentTemplate
from legalbot.document.variables import CaseFactsExtractor


class LegalDocumentGenerator:
    """Generates legal documents from case facts using RAG + templates + LLM."""

    def __init__(
        self,
        retriever: Any,
        provider: Any,
        template_dir: Path | None = None,
        enabled_types: list[str] | None = None,
    ):
        self.retriever = retriever
        self.provider = provider
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        self.enabled_types = enabled_types or [
            "complaint", "defense", "agent_opinion", "appeal", "enforcement",
        ]
        self._templates: dict[str, LegalDocumentTemplate] = {}
        self._extractor = CaseFactsExtractor(provider)
        self._load_templates()

    def _load_templates(self) -> None:
        from legalbot.document.templates.complaint import ComplaintTemplate
        from legalbot.document.templates.defense import DefenseTemplate
        from legalbot.document.templates.agent_opinion import AgentOpinionTemplate
        from legalbot.document.templates.appeal import AppealTemplate
        from legalbot.document.templates.enforcement import EnforcementTemplate

        template_classes: list[tuple[str, type[LegalDocumentTemplate]]] = [
            ("complaint", ComplaintTemplate),
            ("defense", DefenseTemplate),
            ("agent_opinion", AgentOpinionTemplate),
            ("appeal", AppealTemplate),
            ("enforcement", EnforcementTemplate),
        ]

        for doc_type, cls in template_classes:
            if doc_type in self.enabled_types:
                self._templates[doc_type] = cls()

        logger.info(
            "Loaded {} document templates: {}",
            len(self._templates),
            list(self._templates.keys()),
        )

    def get_template(self, doc_type: str) -> LegalDocumentTemplate | None:
        return self._templates.get(doc_type)

    async def generate(
        self,
        doc_type: str,
        case_facts: str,
        extra_variables: dict[str, Any] | None = None,
        law_areas: list[str] | None = None,
    ) -> str:
        template = self._templates.get(doc_type)
        if not template:
            available = ", ".join(self._templates.keys())
            return f"不支持的文书类型：{doc_type}。支持的类型：{available}。"

        try:
            # Step 1: Extract structured facts
            extracted = await self._extractor.extract(case_facts, doc_type)
            variable_set = extracted.to_dict()
            if extra_variables:
                variable_set.update(extra_variables)

            # Step 2: RAG retrieval of relevant laws
            law_query = " ".join(template.law_keywords) + " " + case_facts[:500]
            rag_results = await self.retriever.retrieve(
                query=law_query,
                law_area=law_areas[0] if law_areas and law_areas else None,
                doc_type=None,
                top_k=8,
            )
            relevant_laws = [
                "[%s] 第%s条：%s" % (
                    r.chunk.metadata.get("law_name", "未知"),
                    r.chunk.metadata.get("article_no", ""),
                    r.chunk.text[:200],
                )
                for r in rag_results
            ]

            # Step 3: Build prompt and call LLM
            prompt = template.build_prompt(
                case_facts=case_facts,
                relevant_laws=relevant_laws,
                variable_set=variable_set,
            )

            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            filled_document = response.content or ""
            if not filled_document.strip():
                return "文书生成失败：LLM未返回有效内容。请尝试提供更详细的案件事实。"

            disclaimer = (
                "\n\n---\n"
                "【免责声明】\n"
                "本文书由 AI 自动生成，仅供参考，不构成正式法律意见。\n"
                "诉讼材料的正式提交应由执业律师审核确认。\n"
                "如需正式法律意见，请咨询执业律师。"
            )

            return filled_document.strip() + disclaimer

        except Exception as e:
            logger.exception("Document generation failed for doc_type={}", doc_type)
            return "文书生成失败：模板渲染时发生错误，请稍后重试或联系管理员。"
