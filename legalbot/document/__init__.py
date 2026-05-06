"""Legal document generation package."""

from legalbot.document.config import DocumentDraftConfig
from legalbot.document.generator import LegalDocumentGenerator
from legalbot.document.variables import CaseFacts, CaseFactsExtractor

__all__ = [
    "DocumentDraftConfig",
    "LegalDocumentGenerator",
    "CaseFacts",
    "CaseFactsExtractor",
]
