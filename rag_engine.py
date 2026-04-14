"""
Core RAG Engine - Handles document ingestion, embedding, and retrieval
"""

import os
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class RetrievedChunk:
    document: Document
    score: float
    rank: int


class SimpleVectorStore:
    """
    In-memory vector store with cosine similarity search.
    Replace with Chroma / Qdrant / Pinecone in production.
    """

    def __init__(self):
        self.documents: List[Document] = []

    def add(self, doc: Document):
        self.documents.append(doc)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievedChunk]:
        if not self.documents:
            return []

        q = np.array(query_embedding)
        scored = []
        for doc in self.documents:
            if doc.embedding:
                d = np.array(doc.embedding)
                norm = np.linalg.norm(q) * np.linalg.norm(d)
                score = float(np.dot(q, d) / norm) if norm > 0 else 0.0
                scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievedChunk(document=doc, score=score, rank=i + 1)
            for i, (doc, score) in enumerate(scored[:top_k])
        ]

    def __len__(self):
        return len(self.documents)


class EmbeddingModel:
    """
    Wrapper for embedding model.
    Uses Anthropic's API for text embeddings via a hash-based mock for portability.
    Swap embed() with OpenAI / Cohere / HuggingFace in production.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        """
        Deterministic mock embedding based on text hash.
        Replace with: openai.embeddings.create() or sentence-transformers
        """
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim)
        return (vec / np.linalg.norm(vec)).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class TextSplitter:
    """Recursive character text splitter."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, metadata: Dict = None) -> List[Document]:
        metadata = metadata or {}
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            # Try to split at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind(". ")
                if last_period > self.chunk_size // 2:
                    chunk_text = chunk_text[: last_period + 1]
                    end = start + last_period + 1
            chunks.append(
                Document(
                    content=chunk_text.strip(),
                    metadata={**metadata, "chunk_index": len(chunks), "char_start": start},
                )
            )
            start = end - self.chunk_overlap
        return chunks


class RAGEngine:
    """
    Core retrieval engine: ingest → embed → store → retrieve
    """

    def __init__(self, embedding_dim: int = 384, chunk_size: int = 512, top_k: int = 5):
        self.embedder = EmbeddingModel(dim=embedding_dim)
        self.splitter = TextSplitter(chunk_size=chunk_size)
        self.vector_store = SimpleVectorStore()
        self.top_k = top_k

    def ingest(self, text: str, metadata: Dict = None) -> int:
        """Ingest raw text: split → embed → store. Returns chunk count."""
        chunks = self.splitter.split(text, metadata)
        for chunk in chunks:
            chunk.embedding = self.embedder.embed(chunk.content)
            self.vector_store.add(chunk)
        return len(chunks)

    def ingest_file(self, filepath: str) -> int:
        path = Path(filepath)
        text = path.read_text(encoding="utf-8")
        return self.ingest(text, metadata={"source": path.name})

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        k = top_k or self.top_k
        query_emb = self.embedder.embed(query)
        return self.vector_store.search(query_emb, top_k=k)

    def retrieve_as_context(self, query: str, top_k: Optional[int] = None) -> str:
        chunks = self.retrieve(query, top_k)
        if not chunks:
            return "No relevant documents found."
        parts = []
        for c in chunks:
            src = c.document.metadata.get("source", "unknown")
            parts.append(f"[Source: {src} | Score: {c.score:.3f}]\n{c.document.content}")
        return "\n\n---\n\n".join(parts)

    @property
    def doc_count(self):
        return len(self.vector_store)
