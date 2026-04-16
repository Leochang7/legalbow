"""Unit tests for LegalChunker — article structure splitting."""

from __future__ import annotations

from nanobot.rag.chunker import Chunk, ChunkMeta, LegalChunker

# -- Sample legal text fixtures --

CIVIL_CODE_FRAGMENT = """\
第一编 总则
第一章 基本规定

第一条 为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，适应中国特色社会主义发展要求，弘扬社会主义核心价值观，根据宪法，制定本法。

第二条 民法调整平等主体的自然人、法人和非法人组织之间的人身关系和财产关系。

第三条 民事主体的人身权利、财产权利以及其他合法权益受法律保护，任何组织或者个人不得侵犯。
"""

CONTRACT_LAW_FRAGMENT = """\
第三编 合同
第一分编 通则
第一章 一般规定

第四百六十三条 本编调整因合同产生的民事关系。

第四百六十四条 合同是民事主体之间设立、变更、终止民事法律关系的协议。
婚姻、收养、监护等有关身份关系的协议，适用有关该身份关系的法律规定；没有规定的，可以根据其性质参照适用本编规定。

第二节 合同的订立

第四百六十九条 当事人订立合同，可以采用书面形式、口头形式或者其他形式。
"""


def _make_chunker(**kwargs) -> LegalChunker:
    return LegalChunker(**kwargs)


class TestSplitByArticles:
    """Test article-based splitting."""

    def test_basic_split(self):
        chunker = _make_chunker(max_chunk_tokens=2000)
        meta = {"law_name": "中华人民共和国民法典", "doc_type": "law"}
        chunks = chunker.chunk(CIVIL_CODE_FRAGMENT, meta)

        # Should produce at least 3 chunks (one per article)
        assert len(chunks) >= 3

        # Each chunk should have article_no
        article_nos = [c.metadata.get("article_no", "") for c in chunks]
        assert "第一条" in article_nos
        assert "第二条" in article_nos
        assert "第三条" in article_nos

    def test_chunk_has_metadata(self):
        chunker = _make_chunker()
        meta = {"law_name": "中华人民共和国民法典", "doc_type": "law", "law_area": "民法"}
        chunks = chunker.chunk(CIVIL_CODE_FRAGMENT, meta)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.id
            assert chunk.text
            assert chunk.metadata.get("law_name") == "中华人民共和国民法典"
            assert chunk.metadata.get("doc_type") == "law"


class TestChapterMetadata:
    """Test chapter/section metadata extraction."""

    def test_chapter_extracted(self):
        chunker = _make_chunker()
        meta = {"law_name": "中华人民共和国民法典"}
        chunks = chunker.chunk(CIVIL_CODE_FRAGMENT, meta)

        # Articles in 第一章 should have chapter set
        chapter_chunks = [c for c in chunks if "第一章" in (c.metadata.get("chapter") or "")]
        assert len(chapter_chunks) > 0

    def test_section_extracted(self):
        chunker = _make_chunker()
        meta = {"law_name": "中华人民共和国民法典"}
        chunks = chunker.chunk(CONTRACT_LAW_FRAGMENT, meta)

        # Articles in 第二节 should have section set
        section_chunks = [c for c in chunks if "第二节" in (c.metadata.get("section") or "")]
        assert len(section_chunks) > 0
        # 第四百六十九条 is in 第二节
        sec_article = [c for c in section_chunks if c.metadata.get("article_no") == "第四百六十九条"]
        assert len(sec_article) == 1

    def test_chapter_from_different_section(self):
        chunker = _make_chunker()
        meta = {"law_name": "中华人民共和国民法典"}
        chunks = chunker.chunk(CONTRACT_LAW_FRAGMENT, meta)

        # 第四百六十三条 and 第四百六十四条 are in 第一章 一般规定
        ch1_chunks = [c for c in chunks if "第一章" in (c.metadata.get("chapter") or "")]
        assert len(ch1_chunks) >= 1


class TestLongArticleSplit:
    """Test splitting of articles that exceed max_chunk_tokens."""

    def test_long_article_split(self):
        # Create a very long article
        long_text = "第一条 " + "这是一段很长的法律条文内容。" * 200
        chunker = _make_chunker(max_chunk_tokens=100, overlap_tokens=0)
        meta = {"law_name": "测试法律"}
        chunks = chunker.chunk(long_text, meta)

        assert len(chunks) > 1
        # First chunk should have article_no = 第一条
        assert chunks[0].metadata.get("article_no") == "第一条"
        # Subsequent chunks should have (续N) suffix
        assert any("续" in (c.metadata.get("article_no") or "") for c in chunks[1:])


class TestOverlap:
    """Test overlap between consecutive chunks."""

    def test_overlap_applied(self):
        # Create two articles where overlap is meaningful
        text = "第一条 第一条的内容部分。\n第二条 第二条的内容部分。"
        chunker = _make_chunker(max_chunk_tokens=2000, overlap_tokens=50)
        meta = {"law_name": "测试法律"}
        chunks = chunker.chunk(text, meta)

        if len(chunks) >= 2:
            # Second chunk should contain some text from first chunk's tail
            assert len(chunks[1].text) > len("第二条 第二条的内容部分。")


class TestEmptyInput:
    """Test edge cases with empty input."""

    def test_empty_text(self):
        chunker = _make_chunker()
        chunks = chunker.chunk("", {})
        assert chunks == []

    def test_whitespace_only(self):
        chunker = _make_chunker()
        chunks = chunker.chunk("   \n  \n  ", {})
        assert chunks == []

    def test_no_articles(self):
        chunker = _make_chunker()
        text = "这是一段没有条文结构的普通文本内容。"
        chunks = chunker.chunk(text, {"law_name": "测试"})
        # Should produce at least one chunk
        assert len(chunks) >= 1
