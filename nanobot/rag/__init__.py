"""RAG module for legal knowledge base retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nanobot.rag.chunker import Chunk, ChunkMeta, LegalChunker
from nanobot.rag.embedding import EmbeddingClient
from nanobot.rag.retriever import BM25Store, LegalRetriever, RetrievalResult
from nanobot.rag.vectorstore import ChromaVectorStore, SearchResult, VectorStore

if TYPE_CHECKING:
    from nanobot.config.schema import RAGConfig

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

    return LegalRetriever(
        vector_store=vector_store,
        embedding_client=embedding_client,
        bm25_store=bm25_store,
        top_k=config.top_k,
    )
