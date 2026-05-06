"""
RAG 检索指标评测 — 使用项目实际的 LegalRetriever（向量+BM25+RRF+Rerank）

评测流程：
  legal_data → 分块 → [向量+BM25混合索引] → 查询 → RRF合并 → Rerank → LLM评判 → 指标

使用项目的 legalbot.rag.Retriever 组件，确保评测与实际使用一致。
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────
DATA_DIR = Path("legal_data")
JUDGE_MODEL = "deepseek-chat"
SAMPLE_SIZE = 80            # 抽样 chunk 数量
QUERIES_PER_CHUNK = 2      # 每个 chunk 生成多少个 query
TOP_K = 20                  # 检索结果数（Rerank 候选数，最终取 top5 评分）
RANDOM_SEED = 42
SAVE_DETAILS = True
CHUNKS_JSONL = Path("data/chunks.jsonl")   # chunk 持久化文件

random.seed(RANDOM_SEED)


# ── LLM Provider（仅用于 chat/Judge）───────────────────────
class JudgeProvider:
    """仅封装 chat 接口（LLM-as-Judge + Query 生成）。"""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    async def chat(self, messages: list[dict], model: str = JUDGE_MODEL,
                   temperature: float = 0.0) -> str:
        import aiohttp
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {"model": model, "messages": messages, "temperature": temperature}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=body, timeout=120) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Chat API error {resp.status}: {text}")
                result = await resp.json()
                return result["choices"][0]["message"]["content"]


# ── 数据模型 ──────────────────────────────────────────────
@dataclass
class MetricResult:
    query: str
    law_area: str
    recall_at_5: float      # 二进制：source chunk 在 rerank top5 中 → 1 else 0
    rrf_recall_at_60: float  # 二进制：source chunk 在 RRF top60 候选中 → 1 else 0
    mrr: float
    ndcg_at_5: float
    judgments: list[int]


QUERY_GENERATION_PROMPT = """\
基于以下法律条文，生成 {n} 个用户可能会搜索的法律问题。

法律条文：
《{law_name}》{article_no}
{text}

要求：
1. 每个问题 10-30 字
2. 用口语化表述，符合普通用户搜索习惯
3. 只输出问题，每行一个，不要编号，不要加引号
"""

JUDGMENT_PROMPT = """\
你是一个法律检索评测员。判断以下检索结果对用户查询的相关性。

用户查询：{query}

检索结果：
《{law_name}》{article_no}
{text}

评分标准：
3 = 高度相关：直接回答了用户的法律问题
2 = 相关：提供了用户查询所需的法律依据
1 = 弱相关：涉及相关法律领域但不够直接
0 = 不相关：与用户查询无关

只返回一个数字（0/1/2/3），不要解释。
"""


# ── Chunk 持久化 ──────────────────────────────────────────

def _save_chunks_jsonl(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[*] 已保存 {len(chunks)} 个 chunk → {path}")


def _load_chunks_jsonl(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"[*] 从 {path} 加载了 {len(chunks)} 个 chunk")
    return chunks


# ── 索引构建（使用 LegalRetriever）────────────────────────

async def build_index(
    data_dir: Path,
    embed_client,       # EmbeddingClient from legalbot.rag.embedding
    reranker_api_key: str,
) -> tuple[list[dict], list[dict], "LegalRetriever"]:
    """
    构建 LegalRetriever（向量 + BM25 + RRF + Rerank 完整混合检索）。
    返回 (all_chunks, sampled_chunks, legal_retriever)。

    优化：从 data/chunks.jsonl 加载已缓存的向量，避免重复 embedding。
    """
    from legalbot.rag.chunker import LegalChunker
    from legalbot.rag.loader import LegalDocumentLoader
    from legalbot.rag.retriever import BM25Store, LegalRetriever
    from legalbot.rag.reranker import DashScopeReranker
    from legalbot.rag.vectorstore import ChromaVectorStore

    MAX_CHARS = 8000  # DashScope limit is 8192 chars

    # 尝试从 JSONL 加载已缓存的 chunks（含向量）
    cached = _load_chunks_jsonl(CHUNKS_JSONL)
    if cached:
        all_chunks = cached
        print(f"[*] 从缓存加载 {len(all_chunks)} 个 chunks（含向量）")

        # 分层抽样（使用缓存的 chunks）
        sampled = _stratified_sample(all_chunks, SAMPLE_SIZE)
        print(f"[*] 分层抽样 {len(sampled)} 个 chunks")

        # 构建 LegalRetriever，直接用缓存向量写入 store，跳过 embed()
        print("[*] 构建 LegalRetriever（向量 + BM25 + RRF + Rerank）...")
        chroma_store = ChromaVectorStore(collection_name="eval_kb")
        bm25_store = BM25Store()
        reranker = DashScopeReranker(api_key=reranker_api_key, model="qwen3-vl-rerank")

        retriever = LegalRetriever(
            vector_store=chroma_store,
            embedding_client=embed_client,
            bm25_store=bm25_store,
            reranker=reranker,
            top_k=TOP_K,
        )

        # 直接写入向量存储（使用缓存的预计算向量，跳过 embed 调用）
        ids = [c["id"] for c in all_chunks]
        vectors = [c["embedding"] for c in all_chunks]
        metadatas = [c["metadata"] for c in all_chunks]
        texts = [c["text"][:MAX_CHARS] for c in all_chunks]
        await chroma_store.add(ids, vectors, metadatas, texts)

        # 构建 BM25 索引
        from legalbot.rag.chunker import Chunk as ProjectChunk
        project_chunks = [
            ProjectChunk(id=c["id"], text=c["text"][:MAX_CHARS], metadata=c["metadata"])
            for c in all_chunks
        ]
        bm25_store.add(project_chunks)
        print(f"[*] 索引构建完成（向量 {len(ids)} + BM25）")

        return all_chunks, sampled, retriever

    # 无缓存，从头构建
    print(f"[*] 加载法律文件: {data_dir}")
    loader = LegalDocumentLoader()
    chunker = LegalChunker(max_chunk_tokens=800, overlap_tokens=50)

    all_docs = loader.load_directory(data_dir)
    print(f"[*] 加载完成: {len(all_docs)} 个文档")

    # 分块
    all_chunks: list[dict] = []
    for doc in all_docs:
        meta = {
            "law_name": doc.title or "",
            "doc_type": doc.doc_type,
            "law_area": doc.law_area,
            "source": doc.source_path,
        }
        chunks = chunker.chunk(doc.text, meta)
        for c in chunks:
            all_chunks.append({
                "id": c.id,
                "text": c.text,
                "metadata": dict(c.metadata),
            })
    print(f"[*] 分块完成: {len(all_docs)} 个文档, {len(all_chunks)} 个 chunks")

    # Embed all chunks via EmbeddingClient
    print(f"[*] 开始 embedding {len(all_chunks)} 个 chunks...")
    texts = [c["text"][:MAX_CHARS] for c in all_chunks]
    vectors = await embed_client.embed(texts)
    print(f"[*] Embedding 完成，共 {len(vectors)} 个向量")

    # 合并 embedding 到 chunk
    for chunk, vec in zip(all_chunks, vectors):
        chunk["embedding"] = vec

    # 保存到 JSONL
    _save_chunks_jsonl(all_chunks, CHUNKS_JSONL)

    # 构建 LegalRetriever
    print("[*] 构建 LegalRetriever（向量 + BM25 + RRF + Rerank）...")
    chroma_store = ChromaVectorStore(collection_name="eval_kb")
    bm25_store = BM25Store()
    reranker = DashScopeReranker(api_key=reranker_api_key, model="qwen3-vl-rerank")

    retriever = LegalRetriever(
        vector_store=chroma_store,
        embedding_client=embed_client,
        bm25_store=bm25_store,
        reranker=reranker,
        top_k=TOP_K,
    )

    # 直接写入向量存储（跳过 embed）
    ids = [c["id"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    await chroma_store.add(ids, vectors, metadatas, texts)

    # 构建 BM25 索引
    from legalbot.rag.chunker import Chunk as ProjectChunk
    project_chunks = [
        ProjectChunk(id=c["id"], text=c["text"][:MAX_CHARS], metadata=c["metadata"])
        for c in all_chunks
    ]
    bm25_store.add(project_chunks)
    print(f"[*] 索引构建完成（向量 {len(ids)} + BM25）")

    # 分层抽样
    sampled = _stratified_sample(all_chunks, SAMPLE_SIZE)
    print(f"[*] 分层抽样 {len(sampled)} 个 chunks")

    return all_chunks, sampled, retriever


# ── 评测 ─────────────────────────────────────────────────

async def run_evaluation(
    sampled_chunks: list[dict],
    judge: JudgeProvider,
    retriever,  # LegalRetriever
) -> dict:
    from legalbot.rag.retriever import RetrievalResult

    results: list[MetricResult] = []
    areas_stats: dict[str, dict] = {}
    processed = 0

    for chunk in sampled_chunks:
        area = chunk["metadata"].get("law_area", "未知")
        if area not in areas_stats:
            areas_stats[area] = {"n": 0, "recall": 0.0, "rrf_recall": 0.0, "mrr": 0.0, "ndcg": 0.0}

        queries = await _generate_queries(chunk, judge)
        if not queries:
            continue

        for query_text in queries[:QUERIES_PER_CHUNK]:
            try:
                # 使用 LegalRetriever.retrieve（完整混合检索 + RRF + Rerank）
                pipeline = await retriever.retrieve(
                    query=query_text,
                    law_area=area,
                    top_k=TOP_K,
                )

                # RRF top60 候选（rerank 前）
                rrf_ids = {r.chunk.id for r in pipeline.rrf_candidates}

                # Rerank 后 top5 作为最终评测结果
                top5 = pipeline.top_k[:5]

                # 构建 retrieved 列表
                retrieved = []
                for r in top5:
                    retrieved.append({
                        "id": r.chunk.id,
                        "text": r.chunk.text,
                        "metadata": dict(r.chunk.metadata),
                    })

                # LLM-as-Judge 评分
                judgments = []
                for r in top5:
                    score = await _judge(query_text, r.chunk.text[:500], r.chunk.metadata, judge)
                    judgments.append(score)

                # 计算指标（以 source chunk id 为 ground truth）
                source_id = chunk["id"]

                # Recall@5: 二进制（source chunk 在 rerank top5 → 1 else 0）
                recall_at_5 = 1.0 if source_id in {r["id"] for r in retrieved} else 0.0

                # RRFRecall@60: 二进制（source chunk 在 RRF top60 → 1 else 0）
                rrf_recall_at_60 = 1.0 if source_id in rrf_ids else 0.0

                # MRR
                mrr = 0.0
                for i, r in enumerate(retrieved, 1):
                    if r["id"] == source_id:
                        mrr = 1.0 / i
                        break

                # NDCG@5
                dcg = sum((2**j - 1) / (i + 1) for i, j in enumerate(judgments))
                ideal = sum((2**3 - 1) / (i + 1) for i in range(min(len(judgments), TOP_K)))
                ndcg = dcg / ideal if ideal > 0 else 0.0

                results.append(MetricResult(
                    query=query_text,
                    law_area=area,
                    recall_at_5=recall_at_5,
                    rrf_recall_at_60=rrf_recall_at_60,
                    mrr=mrr,
                    ndcg_at_5=ndcg,
                    judgments=judgments,
                ))
                areas_stats[area]["n"] += 1
                areas_stats[area]["recall"] += recall_at_5
                areas_stats[area]["rrf_recall"] += rrf_recall_at_60
                areas_stats[area]["mrr"] += mrr
                areas_stats[area]["ndcg"] += ndcg

                processed += 1
                if processed % 20 == 0:
                    print(f"  [{processed}] queries evaluated", flush=True)

            except Exception as e:
                print(f"  [!] Error: {e}")
                continue

    total = len(results)
    if total == 0:
        return {"error": "No results"}

    overall = {
        "recall@5": sum(r.recall_at_5 for r in results) / total,
        "rrf_recall@60": sum(r.rrf_recall_at_60 for r in results) / total,
        "mrr": sum(r.mrr for r in results) / total,
        "ndcg@5": sum(r.ndcg_at_5 for r in results) / total,
    }

    by_law_area = {}
    for area, stats in areas_stats.items():
        n = stats["n"]
        if n > 0:
            by_law_area[area] = {
                "n_queries": n,
                "recall@5": stats["recall"] / n,
                "rrf_recall@60": stats["rrf_recall"] / n,
                "mrr": stats["mrr"] / n,
                "ndcg@5": stats["ndcg"] / n,
            }

    return {
        "overall": overall,
        "by_law_area": by_law_area,
        "total_queries": total,
        "sample_chunks": len(sampled_chunks),
    }


# ── 辅助 ─────────────────────────────────────────────────

def _stratified_sample(chunks: list[dict], sample_size: int) -> list[dict]:
    by_area: dict[str, list[dict]] = {}
    for c in chunks:
        area = c["metadata"].get("law_area", "未知")
        by_area.setdefault(area, []).append(c)
    areas = list(by_area.keys())
    quota = sample_size // len(areas) if areas else 0
    remainder = sample_size % len(areas)
    sampled = []
    for i, area in enumerate(areas):
        n = quota + (1 if i < remainder else 0)
        sampled.extend(random.sample(by_area[area], k=min(n, len(by_area[area]))))
    random.shuffle(sampled)
    return sampled


async def _generate_queries(chunk: dict, judge: JudgeProvider) -> list[str]:
    from legalbot.rag.chunker import ChunkMeta

    meta = chunk["metadata"]
    law_name = meta.get("law_name", "")
    article_no = meta.get("article_no", "")
    text = chunk["text"][:800]  # 截断避免 token 过多

    prompt = QUERY_GENERATION_PROMPT.format(
        n=QUERIES_PER_CHUNK,
        law_name=law_name,
        article_no=article_no,
        text=text,
    )

    try:
        response = await judge.chat([{"role": "user", "content": prompt}], temperature=0.3)
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        # 过滤掉可能夹杂的编号和引号
        clean = []
        for l in lines:
            l = l.lstrip("0123456789.、)").strip('"\'')
            if l and len(l) >= 5:
                clean.append(l)
        return clean[:QUERIES_PER_CHUNK]
    except Exception as e:
        print(f"  [!] Query generation error: {e}")
        return []


async def _judge(query: str, text: str, metadata: dict, judge: JudgeProvider) -> int:
    prompt = JUDGMENT_PROMPT.format(
        query=query,
        law_name=metadata.get("law_name", ""),
        article_no=metadata.get("article_no", ""),
        text=text[:500],
    )
    try:
        response = await judge.chat([{"role": "user", "content": prompt}], temperature=0)
        response = response.strip()
        if response in ("0", "1", "2", "3"):
            return int(response)
        # 尝试提取数字
        for c in response:
            if c in "0123":
                return int(c)
        return 0
    except Exception:
        return 0


# ── 主流程 ────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("RAG 检索指标评测 (LLM-as-Judge + LegalRetriever)")
    print("=" * 60)

    # 读取配置
    from legalbot.config.loader import load_config
    from legalbot.rag.embedding import EmbeddingClient

    cfg = load_config()

    # Chat — DeepSeek 官方
    chat_cfg = cfg.providers.deepseek
    chat_key = chat_cfg.api_key
    chat_base = chat_cfg.api_base or "https://api.deepseek.com"
    if not chat_key:
        print("[ERROR] config.providers.deepseek.api_key 未设置")
        return

    # Embedding — DashScope
    embed_cfg = cfg.tools.rag
    embed_key = embed_cfg.embedding_api_key
    embed_base = embed_cfg.embedding_api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embed_model = embed_cfg.embedding_model or "text-embedding-v4"
    embed_dim = embed_cfg.embedding_dim or 1536
    if not embed_key:
        print("[ERROR] config.tools.rag.embedding_api_key 未设置")
        return

    # Rerank — DashScope（同 embed_key）
    rerank_key = embed_key

    # EmbeddingClient（项目实际使用的封装）
    embed_client = EmbeddingClient(
        model=embed_model,
        api_key=embed_key,
        api_base=embed_base,
        dim=embed_dim,
    )

    judge = JudgeProvider(api_key=chat_key, base_url=chat_base)

    # Step 1: 构建 LegalRetriever 索引
    print(f"\n[Step 1] 构建 LegalRetriever 索引: {DATA_DIR}")
    t0 = time.time()
    all_chunks, sampled, retriever = await build_index(
        DATA_DIR, embed_client, rerank_key
    )
    print(f"    耗时: {time.time() - t0:.1f}s")

    # Step 2: 执行评测
    print(f"\n[Step 2] 执行评测 (sample={len(sampled)}, queries_per_chunk={QUERIES_PER_CHUNK})")
    t1 = time.time()
    results = await run_evaluation(sampled, judge, retriever)
    print(f"    耗时: {time.time() - t1:.1f}s")

    if "error" in results:
        print(f"[ERROR] {results['error']}")
        return

    # Step 3: 打印结果
    overall = results["overall"]
    print("\n" + "=" * 60)
    print("                    评测结果")
    print("=" * 60)
    print(f"\n整体指标:")
    print(f"  RRFRecall@60: {overall['rrf_recall@60']:.1%}  (RRF top60 召回)")
    print(f"  Recall@5    : {overall['recall@5']:.1%}  (rerank 后 top5, 目标 ≥80%)")
    print(f"  MRR         : {overall['mrr']:.3f}  (目标 ≥0.70)")
    print(f"  NDCG@5      : {overall['ndcg@5']:.3f}")
    print(f"\n评测规模: {results['total_queries']} 个 query, {results['sample_chunks']} 个 chunks")

    print(f"\n按 law_area 分项:")
    print(f"  {'law_area':<14} {'n_queries':>10} {'RRFRecall@60':>14} {'Recall@5':>10} {'MRR':>8} {'NDCG@5':>8}")
    print(f"  {'-'*14:14} {'-'*10:10} {'-'*14:14} {'-'*10:10} {'-'*8:8} {'-'*8:8}")
    for area, stats in sorted(results["by_law_area"].items()):
        print(f"  {area:<14} {stats['n_queries']:>10} "
              f"{stats['rrf_recall@60']:>14.1%} {stats['recall@5']:>10.1%} {stats['mrr']:>8.3f} {stats['ndcg@5']:>8.3f}")

    print(f"\n目标达成情况:")
    rrf_ok = overall["rrf_recall@60"] >= 0.80
    recall_ok = overall["recall@5"] >= 0.80
    mrr_ok = overall["mrr"] >= 0.70
    print(f"  RRFRecall@60 ≥80%: {'PASS' if rrf_ok else 'FAIL'}  {overall['rrf_recall@60']:.1%}")
    print(f"  Recall@5    ≥80%: {'PASS' if recall_ok else 'FAIL'}  {overall['recall@5']:.1%}")
    print(f"  MRR         ≥0.70: {'PASS' if mrr_ok else 'FAIL'}  {overall['mrr']:.3f}")

    # 保存详细结果
    if SAVE_DETAILS:
        output_path = Path("rag_metrics_results.json")
        serializable = {
            "overall": results["overall"],
            "by_law_area": results["by_law_area"],
            "metadata": {
                "sample_size": len(sampled),
                "queries_per_chunk": QUERIES_PER_CHUNK,
                "top_k": TOP_K,
                "data_dir": str(DATA_DIR),
                "retrieval": "LegalRetriever (vector+BM25+RRF+rerank)",
                "embedding_model": embed_model,
                "embedding_dim": embed_dim,
            }
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
