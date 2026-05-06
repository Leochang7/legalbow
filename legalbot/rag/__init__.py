"""RAG module for legal knowledge base retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from legalbot.rag.chunker import Chunk, ChunkMeta, LegalChunker
from legalbot.rag.embedding import EmbeddingClient
from legalbot.rag.loader import LegalDocumentLoader, RawDocument
from legalbot.rag.reranker import DashScopeReranker, Reranker
from legalbot.rag.retriever import BM25Store, LegalRetriever, RetrievalResult
from legalbot.rag.vectorstore import ChromaVectorStore, SearchResult, VectorStore

if TYPE_CHECKING:
    from legalbot.config.schema import RAGConfig

__all__ = [
    "Chunk",
    "ChunkMeta",
    "LegalChunker",
    "EmbeddingClient",
    "ChromaVectorStore",
    "VectorStore",
    "SearchResult",
    "BM25Store",
    "LegalRetriever",
    "RetrievalResult",
    "Reranker",
    "DashScopeReranker",
    "RawDocument",
    "LegalDocumentLoader",
    "create_retriever",
]


def create_retriever(config: RAGConfig) -> LegalRetriever:
    """Create a LegalRetriever from RAGConfig."""
    persist_dir = Path(config.persist_dir).expanduser() if config.persist_dir else None

    embedding_client = EmbeddingClient(
        model=config.embedding_model,
        api_key=config.embedding_api_key,
        api_base=config.embedding_api_base,
        dim=config.embedding_dim,
    )

    vector_store = ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name="legal_kb",
    )

    bm25_store = BM25Store() if config.bm25_enable else None

    # Build reranker if configured
    reranker: Reranker | None = None
    if config.reranker:
        reranker_api_key = config.reranker_api_key or config.embedding_api_key
        if reranker_api_key:
            reranker = DashScopeReranker(
                api_key=reranker_api_key,
                model=config.reranker,
            )

    return LegalRetriever(
        vector_store=vector_store,
        embedding_client=embedding_client,
        bm25_store=bm25_store,
        reranker=reranker,
        top_k=config.top_k,
    )
