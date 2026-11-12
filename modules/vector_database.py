"""Persistent ChromaDB vector store for document chunks and metadata management."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from modules.document_processor import DocumentChunk


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with its similarity score and source metadata."""

    chunk_id: str
    text: str
    filename: str
    page_number: int
    chunk_index: int
    category: str
    similarity_score: float
    metadata: dict[str, Any]


class VectorDatabase:
    """Manages persistent ChromaDB storage, duplicate checks, querying, and deletion."""

    COLLECTION_NAME = "renewable_energy_kb"

    def __init__(self, persist_dir: Path | str) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize ChromaDB at '{self.persist_dir}': {exc}") from exc

    def document_exists(self, filename: str, file_hash: str) -> tuple[bool, str]:
        """Check whether a document with the same content hash or filename is already indexed."""
        by_hash = self._collection.get(where={"file_hash": file_hash}, limit=1, include=["metadatas"])
        if by_hash and by_hash.get("ids"):
            existing_name = (by_hash["metadatas"][0] or {}).get("filename", filename)
            return True, f"Identical content already indexed under '{existing_name}'."

        by_name = self._collection.get(where={"filename": filename}, limit=1, include=["metadatas"])
        if by_name and by_name.get("ids"):
            return True, f"A document named '{filename}' is already indexed in the knowledge base."

        return False, ""

    def add_document_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Insert document chunks and embeddings into ChromaDB without deleting existing entries."""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings.")

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [dict(c.metadata) for c in chunks]

        # Batch inserts in slices of 200 to stay safely below SQLite/Chroma parameter limits
        batch_size = 200
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
        return len(chunks)

    def query_similar(
        self,
        query_embedding: list[float],
        top_k: int = 4,
        category_filter: str | None = None,
        min_similarity: float = 0.32,
    ) -> list[RetrievedChunk]:
        """Search ChromaDB for the top-k most similar chunks above min_similarity."""
        total_chunks = self._collection.count()
        if total_chunks == 0:
            return []

        where_clause: dict[str, Any] | None = None
        if category_filter and category_filter != "All Categories":
            where_clause = {"category": category_filter}

        available_count = total_chunks
        if where_clause is not None:
            matching = self._collection.get(where=where_clause, include=[])
            available_count = len(matching.get("ids") or [])
            if available_count == 0:
                return []

        n_results = min(max(top_k, 1), available_count)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_clause is not None:
            query_kwargs["where"] = where_clause

        try:
            results = self._collection.query(**query_kwargs)
        except Exception:
            return []

        ids_list = (results.get("ids") or [[]])[0]
        docs_list = (results.get("documents") or [[]])[0]
        metas_list = (results.get("metadatas") or [[]])[0]
        dists_list = (results.get("distances") or [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, doc_text, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
            meta = meta or {}
            # Cosine distance in ChromaDB is 1 - cosine_similarity
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            if similarity < min_similarity:
                continue
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=str(doc_text),
                    filename=str(meta.get("filename", "Unknown")),
                    page_number=int(meta.get("page_number", 1)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    category=str(meta.get("category", "General Renewable Energy")),
                    similarity_score=round(similarity, 4),
                    metadata=meta,
                )
            )

        retrieved.sort(key=lambda item: item.similarity_score, reverse=True)
        return retrieved

    def list_documents(self) -> list[dict[str, Any]]:
        """Aggregate indexed document metadata for the Knowledge Repository view."""
        total = self._collection.count()
        if total == 0:
            return []

        records = self._collection.get(include=["metadatas"])
        metadatas = records.get("metadatas") or []

        grouped: dict[str, dict[str, Any]] = {}
        for meta in metadatas:
            if not meta:
                continue
            filename = str(meta.get("filename", "Unknown"))
            chunk_category = str(meta.get("category", "General Renewable Energy"))
            if filename not in grouped:
                grouped[filename] = {
                    "filename": filename,
                    "file_hash": str(meta.get("file_hash", "")),
                    "file_type": str(meta.get("file_type", "pdf")).upper(),
                    "categories": [],
                    "category_chunks": defaultdict(int),
                    "category": chunk_category,
                    "page_count": int(meta.get("total_pages", meta.get("page_number", 1))),
                    "chunk_count": 0,
                    "uploaded_at": str(meta.get("uploaded_at", "N/A")),
                }
            if chunk_category not in grouped[filename]["categories"]:
                grouped[filename]["categories"].append(chunk_category)
            grouped[filename]["category_chunks"][chunk_category] += 1
            grouped[filename]["category"] = ", ".join(grouped[filename]["categories"])
            grouped[filename]["chunk_count"] += 1

        for doc in grouped.values():
            doc["category_chunks"] = dict(doc["category_chunks"])

        return sorted(grouped.values(), key=lambda d: d["filename"].lower())

    def delete_document(self, filename: str) -> int:
        """Delete all indexed chunks belonging to filename and return deleted chunk count."""
        matches = self._collection.get(where={"filename": filename}, include=[])
        ids = matches.get("ids") or []
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def clear_all(self) -> None:
        """Remove all documents and reset the ChromaDB collection."""
        self._client.delete_collection(name=self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics of the vector knowledge base."""
        docs = self.list_documents()
        category_counts: dict[str, int] = defaultdict(int)
        for doc in docs:
            for category in doc.get("categories") or [doc["category"]]:
                category_counts[category] += 1

        return {
            "total_documents": len(docs),
            "total_chunks": self._collection.count(),
            "categories_breakdown": dict(sorted(category_counts.items())),
            "storage_path": str(self.persist_dir),
        }
