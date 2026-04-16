"""Legal document chunker — splits by article structure (条/款/项)."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import TypedDict

import tiktoken


class ChunkMeta(TypedDict, total=False):
    law_name: str          # 法规名称，如"中华人民共和国民法典"
    article_no: str        # 条号，如"第五百八十三条"
    chapter: str           # 篇章，如"第三编 合同"
    section: str           # 节，如"第二节 合同的效力"
    law_area: str          # 法律领域，如"民法/合同法"
    doc_type: str          # 文档类型：law/judicial_interpretation/case/contract_template
    effective_date: str    # 生效日期
    source: str            # 来源


@dataclass
class Chunk:
    id: str
    text: str
    metadata: ChunkMeta


# ---- Regex patterns for Chinese legal document structure ----

# 篇/编：第一编 总则
_PART_PATTERN = re.compile(r"第[一二三四五六七八九十百千\d]+编\s+\S+")
# 章：第一章 一般规定
_CHAPTER_PATTERN = re.compile(r"第[一二三四五六七八九十百千\d]+章\s+\S+")
# 节：第一节 合同的订立
_SECTION_PATTERN = re.compile(r"第[一二三四五六七八九十百千\d]+节\s+\S+")
# 条：第一条
_ARTICLE_PATTERN = re.compile(r"第[一二三四五六七八九十百千\d]+条")
# Full article header line: 第一条  ...
_ARTICLE_HEADER_PATTERN = re.compile(r"^(第[一二三四五六七八九十百千\d]+条)\s*")


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base (same as nanobot core)."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


class LegalChunker:
    """法律文档语义分块器 — 按条文/款/项结构切分"""

    def __init__(self, max_chunk_tokens: int = 800, overlap_tokens: int = 100):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        """分块策略：
        1. 优先按「第X条」切分
        2. 超长条文按 token 上限二次切分
        3. 保留上下文重叠（overlap）
        4. 每个 chunk 携带元数据：law_name, article_no, chapter, law_area
        """
        if not text or not text.strip():
            return []

        # Step 1: Split by structural headers to track chapter/section
        sections = self._split_by_structure(text)

        # Step 2: Split each section by articles
        chunks: list[Chunk] = []
        for section_text, current_chapter, current_section in sections:
            article_chunks = self._split_by_articles(
                section_text, metadata, current_chapter, current_section
            )
            chunks.extend(article_chunks)

        # Step 3: Apply overlap to consecutive chunks in same article range
        if self.overlap_tokens > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _split_by_structure(self, text: str) -> list[tuple[str, str, str]]:
        """Split text by structural headers (编/章/节), returning (text, chapter, section)."""
        lines = text.split("\n")
        sections: list[tuple[str, str, str]] = []
        current_chapter = ""
        current_section = ""
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            # Check for chapter header
            ch_match = _CHAPTER_PATTERN.match(stripped)
            part_match = _PART_PATTERN.match(stripped)
            sec_match = _SECTION_PATTERN.match(stripped)

            if part_match or ch_match:
                # Save previous section
                if current_lines:
                    sections.append(("\n".join(current_lines), current_chapter, current_section))
                    current_lines = []
                if ch_match:
                    current_chapter = stripped
                    current_section = ""
                elif part_match:
                    # 编/篇 resets both chapter and section
                    current_chapter = ""
                    current_section = ""
            elif sec_match:
                if current_lines:
                    sections.append(("\n".join(current_lines), current_chapter, current_section))
                    current_lines = []
                current_section = stripped
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(("\n".join(current_lines), current_chapter, current_section))

        return sections

    def _split_by_articles(
        self,
        text: str,
        base_metadata: dict,
        chapter: str,
        section: str,
    ) -> list[Chunk]:
        """Split a section by article headers (第X条)."""
        lines = text.split("\n")
        articles: list[tuple[str, str]] = []  # (article_no, content_lines)
        current_article_no = ""
        current_lines: list[str] = []

        for line in lines:
            header_match = _ARTICLE_HEADER_PATTERN.match(line.strip())
            if header_match:
                # Save previous article
                if current_lines:
                    articles.append((current_article_no, "\n".join(current_lines)))
                    current_lines = []
                current_article_no = header_match.group(1)
                # Rest of the line after the article number
                rest = line.strip()[header_match.end():].strip()
                if rest:
                    current_lines.append(rest)
            else:
                current_lines.append(line)

        if current_lines:
            articles.append((current_article_no, "\n".join(current_lines)))

        # If no articles found, treat entire text as one chunk
        if not articles:
            return self._make_chunks("", text.strip(), base_metadata, chapter, section)

        chunks: list[Chunk] = []
        for article_no, content in articles:
            content = content.strip()
            if not content:
                continue
            chunks.extend(
                self._make_chunks(article_no, content, base_metadata, chapter, section)
            )

        return chunks

    def _make_chunks(
        self,
        article_no: str,
        text: str,
        base_metadata: dict,
        chapter: str,
        section: str,
    ) -> list[Chunk]:
        """Create Chunk(s) from text, splitting if it exceeds max_chunk_tokens."""
        token_count = _count_tokens(text)
        if token_count <= self.max_chunk_tokens:
            meta = ChunkMeta(
                law_name=base_metadata.get("law_name", ""),
                article_no=article_no,
                chapter=chapter,
                section=section,
                law_area=base_metadata.get("law_area", ""),
                doc_type=base_metadata.get("doc_type", ""),
                effective_date=base_metadata.get("effective_date", ""),
                source=base_metadata.get("source", ""),
            )
            return [Chunk(id=uuid.uuid4().hex[:12], text=text, metadata=meta)]

        # Split long article by token limit
        chunks: list[Chunk] = []
        sentences = self._split_sentences(text)
        current_text = ""
        current_tokens = 0
        part_idx = 1

        for sentence in sentences:
            sent_tokens = _count_tokens(sentence)
            if current_tokens + sent_tokens > self.max_chunk_tokens and current_text:
                # Save current chunk
                suffix = f" (续{part_idx})" if part_idx > 1 else ""
                meta = ChunkMeta(
                    law_name=base_metadata.get("law_name", ""),
                    article_no=f"{article_no}{suffix}",
                    chapter=chapter,
                    section=section,
                    law_area=base_metadata.get("law_area", ""),
                    doc_type=base_metadata.get("doc_type", ""),
                    effective_date=base_metadata.get("effective_date", ""),
                    source=base_metadata.get("source", ""),
                )
                chunks.append(Chunk(id=uuid.uuid4().hex[:12], text=current_text.strip(), metadata=meta))
                current_text = sentence
                current_tokens = sent_tokens
                part_idx += 1
            else:
                current_text += sentence
                current_tokens += sent_tokens

        if current_text.strip():
            suffix = f" (续{part_idx})" if part_idx > 1 else ""
            meta = ChunkMeta(
                law_name=base_metadata.get("law_name", ""),
                article_no=f"{article_no}{suffix}",
                chapter=chapter,
                section=section,
                law_area=base_metadata.get("law_area", ""),
                doc_type=base_metadata.get("doc_type", ""),
                effective_date=base_metadata.get("effective_date", ""),
                source=base_metadata.get("source", ""),
            )
            chunks.append(Chunk(id=uuid.uuid4().hex[:12], text=current_text.strip(), metadata=meta))

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences for granular splitting of long articles."""
        # Split on Chinese sentence endings
        parts = re.split(r"(。|；|！|？)", text)
        sentences: list[str] = []
        for i in range(0, len(parts) - 1, 2):
            sentences.append(parts[i] + parts[i + 1])
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1])
        return [s for s in sentences if s.strip()] or [text]

    def _apply_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        """Prepend overlap from previous chunk's tail to current chunk's head."""
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            # Only overlap if same law and adjacent article
            if prev.metadata.get("law_name") == curr.metadata.get("law_name"):
                overlap_text = self._tail_text(prev.text, self.overlap_tokens)
                if overlap_text:
                    new_text = overlap_text + "\n" + curr.text
                    result.append(Chunk(id=curr.id, text=new_text, metadata=curr.metadata))
                    continue
            result.append(curr)
        return result

    @staticmethod
    def _tail_text(text: str, max_tokens: int) -> str:
        """Get the tail of text up to max_tokens."""
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            tail_tokens = tokens[-max_tokens:]
            return enc.decode(tail_tokens)
        except Exception:
            chars = max_tokens * 4
            return text[-chars:] if len(text) > chars else text
