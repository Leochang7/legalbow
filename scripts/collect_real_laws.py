"""Script to collect real Chinese law texts from publicly accessible sources.

Primary source: 法律快车 (lawtime.cn) — has full text of most Chinese laws
Fallback: Generates sample texts for laws that can't be fetched

Usage:
    python scripts/collect_real_laws.py [--output-dir ./legal_data/laws]
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx is required: pip install httpx")
    sys.exit(1)

from loguru import logger


# Map law names to their URLs on lawtime.cn (manually curated for reliability)
LAWTIME_URLS = {
    "中华人民共和国民法典": "https://www.lawtime.cn/info/minfa/mfdian/20210322/3585426.html",
    "中华人民共和国刑法": "https://www.lawtime.cn/info/xingfa/xingfadian/20200902/3368867.html",
    "中华人民共和国公司法": "https://www.lawtime.cn/info/gongsi/gsf/20240102/4187663.html",
    "中华人民共和国劳动法": "https://www.lawtime.cn/info/laodong/ldf/201101085855.html",
    "中华人民共和国劳动合同法": "https://www.lawtime.cn/info/laodong/ldhtf/20240103/4188085.html",
    "中华人民共和国行政处罚法": "https://www.lawtime.cn/info/xingzheng/xzcfgf/20210715/3618843.html",
    "中华人民共和国民事诉讼法": "https://www.lawtime.cn/info/minsu/msssf/20230901/4026801.html",
    "中华人民共和国刑事诉讼法": "https://www.lawtime.cn/info/xingshi/xsssf/20230901/4026804.html",
    "中华人民共和国专利法": "https://www.lawtime.cn/info/zscq/zlf/20210611/3598263.html",
    "中华人民共和国商标法": "https://www.lawtime.cn/info/zscq/sbf/20231229/4169484.html",
    "中华人民共和国著作权法": "https://www.lawtime.cn/info/zscq/zqq/20210611/3598267.html",
    "中华人民共和国宪法": "https://www.lawtime.cn/info/zhongyang/xf/2011010815418.html",
    "中华人民共和国消费者权益保护法": "https://www.lawtime.cn/info/minfa/xfz/201101085860.html",
    "中华人民共和国反不正当竞争法": "https://www.lawtime.cn/info/jingji/fbzd/20110108330157.html",
    "中华人民共和国反垄断法": "https://www.lawtime.cn/info/jingji/fldf/20110108330247.html",
    "中华人民共和国道路交通安全法": "https://www.lawtime.cn/info/jiaotong/jtaqf/20210514/3579661.html",
    "中华人民共和国社会保险法": "https://www.lawtime.cn/info/laodong/shebaof/20110108335158.html",
    "中华人民共和国仲裁法": "https://www.lawtime.cn/info/minsu/zcf/201101085848.html",
    "中华人民共和国立法法": "https://www.lawtime.cn/info/zhongyang/lff/20230313/3933621.html",
    "中华人民共和国行政许可法": "https://www.lawtime.cn/info/xingzheng/xzxkf/20110108335201.html",
    "中华人民共和国个人所得税法": "https://www.lawtime.cn/info/caishui/grsds/20240103/4188099.html",
    "中华人民共和国企业所得税法": "https://www.lawtime.cn/info/caishui/qysds/20110108332749.html",
    "中华人民共和国网络安全法": "https://www.lawtime.cn/info/anquan/wlaqf/20210622/3601071.html",
    "中华人民共和国数据安全法": "https://www.lawtime.cn/info/anquan/sjaqf/20210901/3623226.html",
    "中华人民共和国个人信息保护法": "https://www.lawtime.cn/info/anquan/grxxbhf/20211101/3664951.html",
    "中华人民共和国未成年人保护法": "https://www.lawtime.cn/info/minfa/wrbrbhf/20210611/3598283.html",
    "中华人民共和国环境保护法": "https://www.lawtime.cn/info/huanbao/hjbhf/20110108330150.html",
    "中华人民共和国招标投标法": "https://www.lawtime.cn/info/jingji/zbtbf/20110108330227.html",
    "中华人民共和国外商投资法": "https://www.lawtime.cn/info/jingji/wstzf/20110108335206.html",
    "中华人民共和国证券法": "https://www.lawtime.cn/info/jingji/zqf/20231229/4169498.html",
}

CORE_LAWS = [
    ("中华人民共和国民法典", "民法", "law"),
    ("中华人民共和国刑法", "刑法", "law"),
    ("中华人民共和国公司法", "商法", "law"),
    ("中华人民共和国劳动法", "劳动法", "law"),
    ("中华人民共和国劳动合同法", "劳动法", "law"),
    ("中华人民共和国行政处罚法", "行政法", "law"),
    ("中华人民共和国民事诉讼法", "诉讼法", "law"),
    ("中华人民共和国刑事诉讼法", "诉讼法", "law"),
    ("中华人民共和国专利法", "知识产权", "law"),
    ("中华人民共和国商标法", "知识产权", "law"),
    ("中华人民共和国著作权法", "知识产权", "law"),
    ("中华人民共和国宪法", "宪法", "law"),
    ("中华人民共和国消费者权益保护法", "民法", "law"),
    ("中华人民共和国反不正当竞争法", "知识产权", "law"),
    ("中华人民共和国反垄断法", "商法", "law"),
    ("中华人民共和国道路交通安全法", "行政法", "law"),
    ("中华人民共和国社会保险法", "劳动法", "law"),
    ("中华人民共和国仲裁法", "诉讼法", "law"),
    ("中华人民共和国立法法", "宪法", "law"),
    ("中华人民共和国行政许可法", "行政法", "law"),
    ("中华人民共和国个人所得税法", "商法", "law"),
    ("中华人民共和国企业所得税法", "商法", "law"),
    ("中华人民共和国网络安全法", "行政法", "law"),
    ("中华人民共和国数据安全法", "行政法", "law"),
    ("中华人民共和国个人信息保护法", "行政法", "law"),
    ("中华人民共和国未成年人保护法", "民法", "law"),
    ("中华人民共和国环境保护法", "行政法", "law"),
    ("中华人民共和国招标投标法", "商法", "law"),
    ("中华人民共和国外商投资法", "商法", "law"),
    ("中华人民共和国证券法", "商法", "law"),
]


async def fetch_law_page(client: httpx.AsyncClient, url: str) -> str | None:
    """Fetch and extract law text from a web page."""
    try:
        resp = await client.get(url, timeout=20, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text

        # Try to extract the article content using common patterns
        # Most law pages have the text in a div with class like "article-content" or "law-content"
        
        # Method 1: Look for content in common div patterns
        content_match = re.search(
            r'<div[^>]*class="[^"]*(?:article|content|law|text|detail)[^"]*"[^>]*>(.*?)</div>',
            html,
            re.DOTALL | re.IGNORECASE,
        )
        if content_match:
            text = _strip_html(content_match.group(1))
            if len(text) > 200:  # Must have substantial content
                return _format_law_text(text)

        # Method 2: Strip all HTML and take the longest text block
        text = _strip_html(html)
        if len(text) > 200:
            return _format_law_text(text)

    except Exception as e:
        logger.warning("Failed to fetch {}: {}", url, e)
    return None


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode entities."""
    import html as html_mod
    # Remove scripts and styles
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", html)
    # Decode entities
    text = html_mod.unescape(text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def _format_law_text(text: str) -> str:
    """Format extracted text into proper legal document structure."""
    # Add newlines before structural headers
    text = re.sub(r"(第[一二三四五六七八九十百千零\d]+编\s*\S*)", r"\n\1", text)
    text = re.sub(r"(第[一二三四五六七八九十百千零\d]+章\s*\S*)", r"\n\1", text)
    text = re.sub(r"(第[一二三四五六七八九十百千零\d]+节\s*\S*)", r"\n\1", text)
    text = re.sub(r"(第[一二三四五六七八九十百千零\d]+条)", r"\n\1", text)
    # Clean up excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _generate_sample_law(title: str, area: str) -> str:
    """Generate a minimal sample law text for development/testing."""
    return f"""{title}

（2020年5月28日第十三届全国人民代表大会第三次会议通过）

第一编 总则
第一章 基本规定

第一条 为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，根据宪法，制定本法。

第二条 本法调整平等主体的自然人、法人和非法人组织之间的人身关系和财产关系。

第三条 民事主体的人身权利、财产权利以及其他合法权益受法律保护，任何组织或者个人不得侵犯。

第四条 民事主体在民事活动中的法律地位一律平等。

第五条 民事主体从事民事活动，应当遵循自愿原则，按照自己的意思设立、变更、终止民事法律关系。

第二章 自然人

第六条 自然人的民事权利能力一律平等。

第七条 自然人从出生时起到死亡时止，具有民事权利能力，依法享有民事权利，承担民事义务。

第八条 自然人的民事权利能力不得放弃。

第九条 自然人享有生命权、身体权、健康权、姓名权、肖像权、名誉权、荣誉权、隐私权、婚姻自主权等权利。

第十条 自然人的民事权利能力始于出生，终于死亡。
"""


async def collect_laws(output_dir: Path):
    """Collect laws from online sources, generate samples as fallback."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {len(CORE_LAWS)} laws to {output_dir}")

    fetched = 0
    generated = 0

    async with httpx.AsyncClient(
        timeout=20,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
    ) as client:
        for i, (title, area, doc_type) in enumerate(CORE_LAWS, 1):
            print(f"[{i}/{len(CORE_LAWS)}] {title} ({area})")

            safe_name = re.sub(r"[\\/:*?\"<>|]", "_", title)
            out_path = output_dir / f"{safe_name}.txt"
            if out_path.exists():
                print(f"  Already exists, skipping")
                fetched += 1
                continue

            text = None

            # Try lawtime.cn URL if available
            url = LAWTIME_URLS.get(title)
            if url:
                text = await fetch_law_page(client, url)
                if text and len(text) > 100:
                    out_path.write_text(text, encoding="utf-8")
                    print(f"  Fetched from lawtime.cn ({len(text)} chars)")
                    fetched += 1
                    await asyncio.sleep(1.0)  # Be polite
                    continue

            # Fallback: generate sample
            sample = _generate_sample_law(title, area)
            out_path.write_text(sample, encoding="utf-8")
            print(f"  Generated sample ({len(sample)} chars)")
            generated += 1

    print(f"\nDone: {fetched} fetched, {generated} generated")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect real Chinese law texts")
    parser.add_argument("--output-dir", default="./legal_data/laws", help="Output directory")
    args = parser.parse_args()

    output = Path(args.output_dir)
    asyncio.run(collect_laws(output))
