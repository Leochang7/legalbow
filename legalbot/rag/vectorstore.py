"""Vector store for RAG — ChromaDB implementation (MVP)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    id: str
    text: str
    metadata: dict[str, Any]
    score: float


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    async def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        ...


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store — persistent or ephemeral (in-memory for tests)."""

    def __init__(
        self,
        persist_dir: Path | None = None,
        collection_name: str = "legal_kb",
    ):
        import chromadb

        if persist_dir is not None:
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            "ChromaVectorStore initialized (persist_dir={}, collection={}, count={})",
            persist_dir,
            collection_name,
            self._collection.count(),
        )

    async def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        # ChromaDB max batch size is 5461; split into batches
        CHROMA_BATCH_SIZE = 5000
        # Ensure metadata values are all str/int/float/bool (ChromaDB requirement)
        clean_metas = []
        for meta in metadatas:
            clean = {k: v for k, v in meta.items() if v is not None}
            clean_metas.append(clean)

        for i in range(0, len(ids), CHROMA_BATCH_SIZE):
            batch_ids = ids[i:i + CHROMA_BATCH_SIZE]
            batch_vectors = vectors[i:i + CHROMA_BATCH_SIZE]
            batch_metas = clean_metas[i:i + CHROMA_BATCH_SIZE]
            batch_docs = documents[i:i + CHROMA_BATCH_SIZE]
            self._collection.add(
                ids=batch_ids,
                embeddings=batch_vectors,
                metadatas=batch_metas,  # type: ignore[arg-type]
                documents=batch_docs,
            )
        # Ensure data is flushed to disk on Windows / Chroma 1.5.x
        if hasattr(self._client, "persist"):
            self._client.persist()

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
        }
        if filter:
            kwargs["where"] = filter
        results = self._collection.query(**kwargs)
        # results is dict with lists: ids, documents, metadatas, distances
        if not results["ids"] or not results["ids"][0]:
            return []
        search_results = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]  # type: ignore[index]
            # cosine distance → similarity: 1 - distance
            score = 1.0 - distance
            search_results.append(
                SearchResult(
                    id=doc_id,
                    text=results["documents"][0][i],  # type: ignore[index]
                    metadata=results["metadatas"][0][i],  # type: ignore[index]
                    score=score,
                )
            )
        return search_results

    async def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)
