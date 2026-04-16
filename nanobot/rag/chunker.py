from dataclasses import dataclass
from typing import TypedDict
import re


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

class LegalChunker:
    """法律文档语义分块器 — 按条文/款/项结构切分"""

    # 法律文本结构模式
    ARTICLE_PATTERN = re.compile(r"第[一二三四五六七八九十百千\d]+条")
    PARAGRAPH_PATTERN = re.compile(r"^[一二三四五六七八九十]+[、.]")  # 款
    ITEM_PATTERN = re.compile(r"^[（(]\s*[一二三四五六七八九十\d]+\s*[)）]")  # 项

    def __init__(self, max_chunk_tokens: int = 800, overlap_tokens: int = 100):
        ...

    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        """分块策略：
        1. 优先按「第X条」切分
        2. 超长条文按「款」或 token 上限二次切分
        3. 保留上下文重叠（overlap）
        4. 每个 chunk 携带元数据：law_name, article_no, chapter, law_area
        """

