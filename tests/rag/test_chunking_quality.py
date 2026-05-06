"""
Phase 1 验收测试 — 数据清洗 + 分块质量验证

验收标准（POST_MVP_CHUNKING_OPTIMIZATION.md Phase 1）：
  ✅ HTML/Markdown 噪声 chunks < 1%（当前 9.9%）
  ✅ 无 article_no chunks < 1%（当前 4.2%）
  ✅ 文本长度 p50 > 300 字符（当前 175）
  ✅ 超长 chunk（>7800字符）全部处理

运行方式：
  uv run python tests/rag/test_chunking_quality.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

# ── Patterns ────────────────────────────────────────────────────────────────

_HTML_INFO_TAG_RE = re.compile(r"<!--\s*INFO\s*END\s*-->", re.IGNORECASE)
_MD_TITLE_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+")

# ── Loader (clean version) ─────────────────────────────────────────────────

from legalbot.rag.chunker import LegalChunker
from legalbot.rag.loader import LegalDocumentLoader


def _clean_legal_text(text: str) -> str:
    """清洗函数 — 与 loader.py._clean_text 保持同步"""
    text = _HTML_INFO_TAG_RE.sub("\n", text, flags=re.IGNORECASE)
    lines = text.split("\n")
    lines = [l for l in lines if not _MD_TITLE_RE.match(l.strip())]
    text = "\n".join(lines)
    text = _HTML_TAG_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"-\n", "", text)
    return text.strip()


# ── Load all chunks ─────────────────────────────────────────────────────────

DATA_DIR = Path("legal_data")
CHUNKER = LegalChunker(
    max_chunk_tokens=600,
    overlap_tokens=200,
    min_chunk_chars=50,
    max_chunk_chars=7800,
)

loader = LegalDocumentLoader()
docs = loader.load_directory(DATA_DIR)

all_chunks = []
for doc in docs:
    meta = {
        "law_name": doc.title or "",
        "doc_type": doc.doc_type,
        "law_area": doc.law_area,
        "source": doc.source_path,
    }
    chunks = CHUNKER.chunk(doc.text, meta)
    for c in chunks:
        all_chunks.append({
            "id": c.id,
            "text": c.text,
            "metadata": dict(c.metadata),
        })

print(f"[+] Total chunks after cleaning + chunking: {len(all_chunks)}")


# ── Criterion 1: HTML/Markdown noise chunks ────────────────────────────────

def has_html_noise(text: str) -> bool:
    return bool(_HTML_INFO_TAG_RE.search(text)) or bool(_MD_TITLE_RE.match(text.strip()))

noise_chunks = [c for c in all_chunks if has_html_noise(c["text"])]
noise_pct = len(noise_chunks) / len(all_chunks) * 100
criterion_1 = noise_pct < 1.0

print(f"\n{'='*60}")
print(f"验收 1: HTML/Markdown 噪声 chunks < 1%")
print(f"{'='*60}")
print(f"  噪声 chunks : {len(noise_chunks)} / {len(all_chunks)}")
print(f"  占比         : {noise_pct:.2f}%  (PASS if criterion_1 else FAIL)")
print(f"  目标         : < 1%")
if noise_chunks:
    print(f"  示例（前100字）: {noise_chunks[0]['text'][:100]}")


# ── Criterion 2: chunks without article_no ──────────────────────────────────

no_article = [c for c in all_chunks if not c["metadata"].get("article_no", "")]
no_art_pct = len(no_article) / len(all_chunks) * 100
criterion_2 = no_art_pct < 1.0

print(f"\n{'='*60}")
print(f"验收 2: 无 article_no chunks < 1%")
print(f"{'='*60}")
print(f"  无 article_no: {len(no_article)} / {len(all_chunks)}")
print(f"  占比          : {no_art_pct:.2f}%  (PASS if criterion_2 else FAIL)")
print(f"  目标          : < 1%")


# ── Criterion 3: text length p50 > 300 chars ───────────────────────────────

lens = sorted([len(c["text"]) for c in all_chunks])
p50_idx = int(len(lens) * 0.50) - 1
p50 = lens[p50_idx]
p75 = lens[int(len(lens) * 0.75) - 1]
p90 = lens[int(len(lens) * 0.90) - 1]
p95 = lens[int(len(lens) * 0.95) - 1]
p99 = lens[int(len(lens) * 0.99) - 1]
min_len = min(lens)
max_len = max(lens)
avg_len = sum(lens) / len(lens)

criterion_3 = p50 > 300

print(f"\n{'='*60}")
print(f"验收 3: 文本长度 p50 > 300 字符")
print(f"{'='*60}")
print(f"  字符数分布:")
print(f"    min   : {min_len:>6} 字符")
print(f"    p50   : {p50:>6} 字符  ({'PASS' if criterion_3 else 'FAIL'}）")
print(f"    p75   : {p75:>6} 字符")
print(f"    p90   : {p90:>6} 字符")
print(f"    p95   : {p95:>6} 字符")
print(f"    p99   : {p99:>6} 字符")
print(f"    max   : {max_len:>6} 字符")
print(f"    avg   : {avg_len:>6.0f} 字符")


# ── Criterion 4: no chunks > 7800 chars ─────────────────────────────────────

MAX_ALLOWED = 7800
oversized = [c for c in all_chunks if len(c["text"]) > MAX_ALLOWED]
criterion_4 = len(oversized) == 0

print(f"\n{'='*60}")
print(f"验收 4: 无超长 chunk（> {MAX_ALLOWED} 字符）")
print(f"{'='*60}")
print(f"  超长 chunks: {len(oversized)}  {'PASS' if criterion_4 else 'FAIL'}）")


# ── Criterion 5: law_area distribution ────────────────────────────────────

areas = Counter(c["metadata"].get("law_area", "未知") for c in all_chunks)
print(f"\n{'='*60}")
print(f"验收 5: law_area 分布（非零）")
print(f"{'='*60}")
for area, cnt in sorted(areas.items(), key=lambda x: -x[1]):
    pct = cnt / len(all_chunks) * 100
    print(f"  {area:<14} {cnt:>6} ({pct:5.1f}%)")


# ── Summary ─────────────────────────────────────────────────────────────────

all_pass = criterion_1 and criterion_2 and criterion_3 and criterion_4

print(f"\n{'='*60}")
print(f"                    验收结果汇总")
print(f"{'='*60}")
print(f"  1. HTML/Markdown 噪声 < 1%  : {'PASS' if criterion_1 else 'FAIL'} ({noise_pct:.2f}%)")
print(f"  2. 无 article_no < 1%        : {'PASS' if criterion_2 else 'FAIL'} ({no_art_pct:.2f}%)")
print(f"  3. p50 文本长度 > 300 字符   : {'PASS' if criterion_3 else 'FAIL'} ({p50} 字符)")
print(f"  4. 无超长 chunk > 7800 字符  : {'PASS' if criterion_4 else 'FAIL'} ({len(oversized)} 个)")
print(f"\n  总计            : {'ALL PASS' if all_pass else 'FAIL (部分指标未达标)'}")
print(f"  总 chunks       : {len(all_chunks)}")
print(f"  原始文档数     : {len(docs)}")


# ── Save cleaned chunks to JSONL ────────────────────────────────────────────

output_path = Path("data/chunks_cleaned.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    for c in all_chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"\n[+] 清洗后的 chunks 已保存: {output_path}")
print(f"    （后续评测可直接从该文件加载，跳过清洗步骤）")
