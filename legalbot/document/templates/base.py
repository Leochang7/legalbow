"""Base class for legal document templates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentVariable:
    """A single variable in a legal document template."""
    key: str
    label: str
    description: str
    required: bool = True
    example: str | None = None


@dataclass
class DocumentSection:
    """A section within a legal document."""
    key: str
    heading: str
    instructions: str
    variables: list[DocumentVariable] = field(default_factory=list)
    min_tokens: int = 50


class LegalDocumentTemplate(ABC):
    """Abstract base for all legal document templates."""

    @property
    @abstractmethod
    def doc_type(self) -> str:
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        ...

    @property
    def required_variables(self) -> list[DocumentVariable]:
        return []

    @property
    def optional_variables(self) -> list[DocumentVariable]:
        return []

    @property
    def sections(self) -> list[DocumentSection]:
        return []

    @property
    def law_keywords(self) -> list[str]:
        return []

    @abstractmethod
    def build_prompt(
        self,
        case_facts: dict[str, Any],
        relevant_laws: list[str],
        variable_set: dict[str, Any],
    ) -> str:
        ...

    def format_document(self, filled: dict[str, Any]) -> str:
        return filled.get("content", "")
