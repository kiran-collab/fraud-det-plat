"""Case vector store: Pinecone in production, local index in dev and CI.

Both backends implement the same small interface, so ``rag.py`` never branches
on which one is live. The local backend is a plain numpy cosine search over a
persisted matrix - not a toy, just unnecessary above a few hundred thousand
cases, at which point Pinecone's serverless index takes over.

Embeddings are deterministic hashed n-grams rather than a neural encoder. That
is a deliberate default, not an oversight: it needs no model download, no GPU
and no per-embedding API call, so the whole RAG path is runnable and testable
offline. ``EmbeddingModel`` is the seam - point it at a sentence-transformer or
a hosted embedding endpoint and nothing else changes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from fraudplat.config import SETTINGS


@dataclass
class CaseDocument:
    case_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedCase:
    case_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingModel(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedding:
    """Deterministic character n-gram hashing into a fixed vector.

    Character n-grams rather than words: fraud case text is dense with
    identifiers and codes (``mch_000123``, ``NG``, ``card_testing``) where
    substring overlap is exactly the similarity signal, and word tokenisation
    would fragment them.
    """

    def __init__(self, dim: int = 512, ngram: int = 4) -> None:
        self.dim = dim
        self.ngram = ngram

    def _vec(self, text: str) -> np.ndarray:
        text = re.sub(r"\s+", " ", text.lower().strip())
        v = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return v
        for i in range(max(1, len(text) - self.ngram + 1)):
            gram = text[i:i + self.ngram]
            # Signed hashing keeps the projection roughly unbiased.
            h = hash(gram)
            v[abs(h) % self.dim] += 1.0 if h >= 0 else -1.0
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class LocalVectorIndex:
    """Cosine-similarity index persisted to disk."""

    def __init__(self, path: Path | None = None, embedding: EmbeddingModel | None = None) -> None:
        self.path = Path(path or SETTINGS.paths.index) / "cases"
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedding = embedding or HashingEmbedding(SETTINGS.genai.embedding_dim)
        self._vectors: np.ndarray = np.zeros((0, self.embedding.dim), dtype=np.float32)
        self._docs: list[CaseDocument] = []
        self._load()

    @property
    def backend(self) -> str:
        return "local"

    def __len__(self) -> int:
        return len(self._docs)

    def _load(self) -> None:
        vec_file, doc_file = self.path / "vectors.npy", self.path / "docs.jsonl"
        if vec_file.exists() and doc_file.exists():
            self._vectors = np.load(vec_file)
            self._docs = [
                CaseDocument(**json.loads(line))
                for line in doc_file.read_text().splitlines() if line.strip()
            ]

    def _persist(self) -> None:
        np.save(self.path / "vectors.npy", self._vectors)
        (self.path / "docs.jsonl").write_text(
            "\n".join(json.dumps(d.__dict__) for d in self._docs)
        )

    def upsert(self, docs: list[CaseDocument]) -> int:
        if not docs:
            return 0
        known = {d.case_id: i for i, d in enumerate(self._docs)}
        new_docs = [d for d in docs if d.case_id not in known]
        for d in docs:
            if d.case_id in known:  # replace in place; embeddings recomputed below
                self._docs[known[d.case_id]] = d
        if new_docs:
            self._docs.extend(new_docs)
        self._vectors = self.embedding.encode([d.text for d in self._docs])
        self._persist()
        return len(docs)

    def search(self, query: str, top_k: int = 6, filters: dict[str, Any] | None = None) -> list[RetrievedCase]:
        if not self._docs:
            return []
        q = self.embedding.encode([query])[0]
        sims = self._vectors @ q
        order = np.argsort(-sims)
        results: list[RetrievedCase] = []
        for i in order:
            doc = self._docs[int(i)]
            if filters and any(doc.metadata.get(k) != v for k, v in filters.items()):
                continue
            results.append(
                RetrievedCase(doc.case_id, doc.text, float(sims[int(i)]), doc.metadata)
            )
            if len(results) >= top_k:
                break
        return results


class PineconeVectorIndex:
    """Pinecone-backed index for production volumes."""

    def __init__(self, index_name: str | None = None, embedding: EmbeddingModel | None = None) -> None:
        from pinecone import Pinecone

        self.embedding = embedding or HashingEmbedding(SETTINGS.genai.embedding_dim)
        self.index_name = index_name or SETTINGS.genai.pinecone_index
        self._client = Pinecone()
        self._index = self._client.Index(self.index_name)

    @property
    def backend(self) -> str:
        return "pinecone"

    def upsert(self, docs: list[CaseDocument], batch_size: int = 100) -> int:
        vectors = self.embedding.encode([d.text for d in docs])
        payload = [
            {
                "id": d.case_id,
                "values": vec.tolist(),
                # Text lives in metadata so retrieval is a single round trip;
                # it is already redacted by the ingestion job.
                "metadata": {**d.metadata, "text": d.text},
            }
            for d, vec in zip(docs, vectors, strict=True)
        ]
        for i in range(0, len(payload), batch_size):
            self._index.upsert(vectors=payload[i:i + batch_size])
        return len(payload)

    def search(self, query: str, top_k: int = 6, filters: dict[str, Any] | None = None) -> list[RetrievedCase]:
        q = self.embedding.encode([query])[0].tolist()
        res = self._index.query(
            vector=q, top_k=top_k, include_metadata=True, filter=filters or None
        )
        out = []
        for match in res.get("matches", []):
            md = dict(match.get("metadata") or {})
            out.append(RetrievedCase(match["id"], md.pop("text", ""), float(match["score"]), md))
        return out


def build_index(backend: str | None = None) -> LocalVectorIndex | PineconeVectorIndex:
    """Select a backend, falling back to local if Pinecone is unreachable.

    Degrading is the right call here: a broken vector index should make the
    assistant less useful, not take it offline. The active backend is reported
    in every answer so an analyst knows what they are looking at.
    """
    backend = backend or SETTINGS.genai.vector_backend
    if backend == "pinecone":
        try:
            return PineconeVectorIndex()
        except Exception:
            return LocalVectorIndex()
    return LocalVectorIndex()
