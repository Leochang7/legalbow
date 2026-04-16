from abc import ABC, abstractmethod
class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, candidates: list, top_k: int) -> list:
        ...

class BGEReranker(Reranker):
    """BGE-Reranker-v2-m3 — 中文法律文本重排效果好"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        ...

class CohereReranker(Reranker):
    """Cohere Rerank API — 无需本地 GPU"""

    def __init__(self, api_key: str):