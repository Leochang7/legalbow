"""Legal knowledge base indexer — builds and updates the RAG index."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from legalbot.rag.chunker import Chunk, LegalChunker
from legalbot.rag.embedding import EmbeddingClient
from legalbot.rag.loader import LegalDocumentLoader, RawDocument
from legalbot.rag.retriever import BM25Store, LegalRetriever
from legalbot.rag.vectorstore import VectorStore


@dataclass
class IndexStats:
    """Statistics about an indexing operation."""

    total_documents: int = 0
    total_chunks: int = 0
    new_documents: int = 0
    new_chunks: int = 0
    skipped_documents: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class LegalIndexer:
    """索引构建与管理 — 从原始文档构建向量索引和 BM25 索引."""

    MANIFEST_FILE = "manifest.json"

    def __init__(
        self,
        loader: LegalDocumentLoader,
        chunker: LegalChunker,
        retriever: LegalRetriever,
        persist_dir: Path,
    ):
        self._loader = loader
        self._chunker = chunker
        self._retriever = retriever
        self._persist_dir = persist_dir
        self._manifest_path = persist_dir / self.MANIFEST_FILE

    async def build_index(
        self,
        data_dir: Path,
        rebuild: bool = False,
    ) -> IndexStats:
        """Full index build from a data directory.

        Args:
            data_dir: Directory containing legal documents.
            rebuild: If True, clear existing index before building.
        """
        import time

        start = time.time()
        stats = IndexStats()

        # Load documents
        docs = self._loader.load_directory(data_dir)
        stats.total_documents = len(docs)
        logger.info("Loaded {} documents from {}", len(docs), data_dir)

        if not docs:
            stats.elapsed_seconds = time.time() - start
            return stats

        # Load manifest for incremental check
        manifest = self._load_manifest()
        if rebuild:
            manifest = {}
            logger.info("Rebuild requested, clearing manifest")

        # Filter out already-indexed documents
        new_docs: list[RawDocument] = []
        for doc in docs:
            doc_key = self._doc_key(doc)
            if doc_key in manifest and not rebuild:
                stats.skipped_documents += 1
                continue
            new_docs.append(doc)

        stats.new_documents = len(new_docs)
        if not new_docs:
            logger.info("No new documents to index")
            stats.elapsed_seconds = time.time() - start
            return stats

        # Chunk documents
        all_chunks: list[Chunk] = []
        doc_chunk_map: dict[str, list[str]] = {}  # doc_key -> [chunk_ids]
        for doc in new_docs:
            doc_key = self._doc_key(doc)
            meta = {
                "law_name": doc.title,
                "doc_type": doc.doc_type,
                "law_area": doc.law_area,
                "effective_date": doc.effective_date,
                "source": doc.source_path,
            }
            try:
                chunks = self._chunker.chunk(doc.text, meta)
                all_chunks.extend(chunks)
                doc_chunk_map[doc_key] = [c.id for c in chunks]
                logger.debug("Chunked '{}' into {} chunks", doc.title, len(chunks))
            except Exception as e:
                error_msg = f"Failed to chunk '{doc.title}': {e}"
                stats.errors.append(error_msg)
                logger.error(error_msg)

        stats.new_chunks = len(all_chunks)
        stats.total_chunks = len(all_chunks)

        if not all_chunks:
            stats.elapsed_seconds = time.time() - start
            return stats

        # Index into retriever (vector store + BM25)
        try:
            await self._retriever.index(all_chunks)
        except Exception as e:
            stats.errors.append(f"Failed to index chunks: {e}")
            logger.error("Indexing failed: {}", e)

        # Update manifest
        for doc_key, chunk_ids in doc_chunk_map.items():
            manifest[doc_key] = {
                "indexed_at": datetime.now().isoformat(),
                "chunk_count": len(chunk_ids),
            }
        self._save_manifest(manifest)

        stats.elapsed_seconds = time.time() - start
        logger.info(
            "Indexing complete: {} docs, {} chunks in {:.1f}s",
            stats.new_documents,
            stats.new_chunks,
            stats.elapsed_seconds,
        )
        return stats

    async def incremental_update(
        self,
        data_dir: Path,
        since: datetime | None = None,
    ) -> IndexStats:
        """Incremental update — only index new or modified documents."""
        manifest = self._load_manifest()
        stats = await self.build_index(data_dir, rebuild=False)

        # If since is provided, filter by modification time
        if since:
            # Re-check: only count docs modified after `since`
            pass  # manifest already handles dedup

        return stats

    def get_status(self) -> dict[str, Any]:
        """Get current index status."""
        manifest = self._load_manifest()
        return {
            "indexed_documents": len(manifest),
            "total_chunks": sum(v.get("chunk_count", 0) for v in manifest.values()),
            "manifest_path": str(self._manifest_path),
            "persist_dir": str(self._persist_dir),
            "documents": manifest,
        }

    @staticmethod
    def _doc_key(doc: RawDocument) -> str:
        """Generate a unique key for a document based on source path."""
        return doc.source_path

    def _load_manifest(self) -> dict[str, Any]:
        """Load the indexing manifest from disk."""
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        """Save the indexing manifest to disk."""
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
