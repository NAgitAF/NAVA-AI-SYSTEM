"""Retrieval System — lexical, semantic and hybrid retrieval."""

from __future__ import annotations

import re
import json
import os
import sqlite3
import importlib
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = re.sub(r"[^\w\s\u0621-\u064A]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_retrieval(results: Iterable[str], relevant: Iterable[str], *, k: int = 5) -> Dict[str, float]:
    """Compute Recall@K, MRR and NDCG@K for a ranked result list."""
    ranked = list(results)[:max(0, k)]
    expected = set(relevant)
    if not expected:
        return {"recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}
    hits = [index for index, item in enumerate(ranked) if item in expected]
    dcg = sum(1.0 / math.log2(index + 2) for index in hits)
    ideal_hits = min(len(expected), len(ranked))
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return {
        "recall_at_k": len(hits) / len(expected),
        "mrr": 1.0 / (hits[0] + 1) if hits else 0.0,
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
    }


class LexicalRetriever:
    """Performs keyword-based matching."""

    def __init__(self):
        self.index: List[str] = []

    def add(self, text: str) -> None:
        if text:
            self.index.append(text)

    def retrieve(self, query: str, corpus: Optional[Iterable[str]] = None, limit: int = 5) -> List[Dict[str, Any]]:
        items = list(corpus) if corpus is not None else list(self.index)
        scores = []
        q_tokens = set(normalize_text(query).split())
        if not q_tokens:
            return []

        for item in items:
            t_tokens = set(normalize_text(item).split())
            if not t_tokens:
                continue
            overlap = len(q_tokens & t_tokens)
            score = overlap / max(1, len(q_tokens | t_tokens))
            if score > 0:
                scores.append({"text": item, "score": score})

        scores.sort(key=lambda entry: entry["score"], reverse=True)
        return scores[:limit]


class SemanticRetriever:
    """Lightweight semantic matching using token frequency vectors."""

    def __init__(self):
        self.index: List[str] = []

    def add(self, text: str) -> None:
        if text:
            self.index.append(text)

    def _vectorize(self, text: str) -> Dict[str, float]:
        return dict(Counter(normalize_text(text).split()))

    def _cosine_similarity(self, vector_a: Dict[str, float], vector_b: Dict[str, float]) -> float:
        if not vector_a or not vector_b:
            return 0.0

        keys = set(vector_a) | set(vector_b)
        numerator = sum(vector_a.get(k, 0.0) * vector_b.get(k, 0.0) for k in keys)
        a_norm = sum(value * value for value in vector_a.values()) ** 0.5
        b_norm = sum(value * value for value in vector_b.values()) ** 0.5
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return numerator / (a_norm * b_norm)

    def retrieve(self, query: str, corpus: Optional[Iterable[str]] = None, limit: int = 5) -> List[Dict[str, Any]]:
        items = list(corpus) if corpus is not None else list(self.index)
        query_vector = self._vectorize(query)
        scored = []
        for item in items:
            similarity = self._cosine_similarity(query_vector, self._vectorize(item))
            if similarity > 0:
                scored.append({"text": item, "score": similarity})
        scored.sort(key=lambda entry: entry["score"], reverse=True)
        return scored[:limit]


class EmbeddingRetriever:
    """Optional local embedding index with a deterministic lexical fallback."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        index_path: Optional[str] = None,
        *,
        allow_download: bool = False,
    ):
        self.model_name = model_name
        self.index_path = index_path
        self.index: List[Dict[str, Any]] = []
        self._model = None
        self.available = False
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
            SentenceTransformer = sentence_transformers.SentenceTransformer
            self._model = SentenceTransformer(
                model_name,
                local_files_only=not allow_download,
            )
            self.available = True
        except (ImportError, OSError, RuntimeError):
            self.available = False
        self._load()

    def _load(self):
        if not self.index_path or not os.path.exists(self.index_path):
            return
        try:
            with open(self.index_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("model_name") == self.model_name:
                self.index = data.get("items", [])
        except (OSError, json.JSONDecodeError):
            self.index = []

    def _save(self):
        if not self.index_path:
            return
        directory = os.path.dirname(self.index_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = f"{self.index_path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump({"model_name": self.model_name, "items": self.index}, handle, ensure_ascii=False)
        os.replace(temporary, self.index_path)

    def add(self, text: str) -> None:
        if not text:
            return
        if any(item["text"] == text for item in self.index):
            return
        if not self.available:
            self.index.append({"text": text, "vector": None})
        else:
            vector = self._model.encode(text, normalize_embeddings=True).tolist()
            self.index.append({"text": text, "vector": vector})
        self._save()

    def retrieve(self, query: str, corpus: Optional[Iterable[str]] = None, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.available:
            return SemanticRetriever().retrieve(query, corpus=corpus or [item["text"] for item in self.index], limit=limit)
        items = [{"text": item, "vector": self._model.encode(item, normalize_embeddings=True).tolist()} for item in corpus] if corpus is not None else self.index
        query_vector = self._model.encode(query, normalize_embeddings=True).tolist()
        scored = []
        for item in items:
            score = sum(left * right for left, right in zip(query_vector, item["vector"]))
            if score > 0:
                scored.append({"text": item["text"], "score": float(score), "retrieval": "embedding"})
        scored.sort(key=lambda entry: entry["score"], reverse=True)
        return scored[:limit]


class SQLiteVectorStore:
    """Dependency-free persistent vector store for local production deployments."""

    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT UNIQUE NOT NULL,
                    vector TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents (metadata)")
            connection.commit()

    @staticmethod
    def _utc_timestamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _normalize_vector(vector: Optional[List[float]]) -> Optional[List[float]]:
        if vector is None:
            return None
        if not isinstance(vector, list):
            vector = list(vector)
        normalized = []
        for value in vector:
            try:
                normalized.append(float(value))
            except (TypeError, ValueError):
                continue
        if not normalized:
            return None
        magnitude = max((sum(v * v for v in normalized) ** 0.5), 1.0)
        return [value / magnitude for value in normalized]

    @staticmethod
    def _token_vector(text: str) -> List[float]:
        counts = Counter(normalize_text(text).split())
        if not counts:
            return []
        tokens = sorted(counts)
        values = [float(counts[token]) for token in tokens]
        magnitude = max((sum(value * value for value in values) ** 0.5), 1.0)
        return [value / magnitude for value in values]

    @staticmethod
    def _cosine_similarity(vector_a: Optional[List[float]], vector_b: Optional[List[float]]) -> float:
        if not vector_a or not vector_b:
            return 0.0
        dim = min(len(vector_a), len(vector_b))
        if dim == 0:
            return 0.0
        numerator = sum(vector_a[index] * vector_b[index] for index in range(dim))
        a_norm = sum(value * value for value in vector_a) ** 0.5
        b_norm = sum(value * value for value in vector_b) ** 0.5
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return numerator / (a_norm * b_norm)

    def upsert(self, text: str, vector: Optional[List[float]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Document text must be a non-empty string.")
        metadata_payload = dict(metadata or {})
        now = self._utc_timestamp()
        final_vector = self._normalize_vector(vector) if vector is not None else self._token_vector(text)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO documents(text, vector, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(text) DO UPDATE SET
                    vector = excluded.vector,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (text, json.dumps(final_vector, ensure_ascii=False), json.dumps(metadata_payload, ensure_ascii=False), now, now),
            )
            connection.commit()

    def upsert_many(self, documents: Iterable[Dict[str, Any]]) -> None:
        for document in documents:
            self.upsert(document.get("text", ""), document.get("vector"), document.get("metadata"))

    def rebuild(self, documents: Iterable[Dict[str, Any]]) -> None:
        items = list(documents)
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM documents")
            connection.commit()
        for document in items:
            self.upsert(document.get("text", ""), document.get("vector"), document.get("metadata"))

    def all(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute("SELECT text, vector, metadata FROM documents ORDER BY id").fetchall()
        return [
            {"text": text, "vector": json.loads(vector) if vector else None, "metadata": json.loads(metadata or "{}")}
            for text, vector, metadata in rows
        ]

    def count(self) -> int:
        with sqlite3.connect(self.path) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def search(self, query: str, *, limit: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            return []
        rows = self.all()
        if metadata_filter:
            rows = [row for row in rows if all(row.get("metadata", {}).get(key) == value for key, value in metadata_filter.items())]
        if not rows:
            return []
        query_tokens = set(normalize_text(query).split())
        query_vector = self._token_vector(query)
        scored = []
        for row in rows:
            candidate_vector = row.get("vector") or self._token_vector(row["text"])
            candidate_tokens = set(normalize_text(row["text"]).split())
            lexical_score = len(query_tokens & candidate_tokens) / max(1, len(query_tokens | candidate_tokens))
            score = 0.7 * lexical_score + 0.3 * self._cosine_similarity(query_vector, candidate_vector)
            if score > 0.0:
                scored.append({"text": row["text"], "score": float(score), "metadata": row.get("metadata", {})})
        scored.sort(key=lambda entry: entry["score"], reverse=True)
        return scored[: max(1, int(limit))]

    def delete(self, text: str) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM documents WHERE text = ?", (text,))
            connection.commit()


class HybridRetriever:
    """Combines lexical and semantic lookup strategies with a persistent vector store backend."""

    def __init__(self, *, embedding_model: Optional[str] = None, index_path: Optional[str] = None, vector_store: Optional[SQLiteVectorStore] = None):
        self.lexical_index: List[str] = []
        self.semantic_index: List[str] = []
        self.embedding = EmbeddingRetriever(embedding_model, index_path) if embedding_model else None
        self.vector_store = vector_store or SQLiteVectorStore(index_path or os.path.join(os.getcwd(), "data", "vector_store.db"))

    def add(self, text: str, *, metadata: Optional[Dict[str, Any]] = None, vector: Optional[List[float]] = None) -> None:
        if text:
            self.lexical_index.append(text)
            self.semantic_index.append(text)
            if self.vector_store is not None:
                self.vector_store.upsert(text, vector=vector, metadata=metadata)
            if self.embedding:
                self.embedding.add(text)

    def retrieve(self, query: str, corpus: Optional[Iterable[str]] = None, limit: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if self.vector_store is not None and corpus is None:
            return self.vector_store.search(query, limit=limit, metadata_filter=metadata_filter)

        items = list(corpus) if corpus is not None else list(self.lexical_index)
        lexical = LexicalRetriever().retrieve(query, corpus=items, limit=limit)
        semantic = (self.embedding.retrieve(query, corpus=items, limit=limit)
                    if self.embedding else SemanticRetriever().retrieve(query, corpus=items, limit=limit))

        merged: Dict[str, Dict[str, Any]] = {}
        for entry in lexical:
            merged.setdefault(entry["text"], {"text": entry["text"], "score": 0.0, "retrieval": "hybrid"})
            merged[entry["text"]]["score"] += entry["score"] * 0.65
        for entry in semantic:
            merged.setdefault(entry["text"], {"text": entry["text"], "score": 0.0, "retrieval": "hybrid"})
            merged[entry["text"]]["score"] += entry["score"] * 0.35

        ordered = sorted(merged.values(), key=lambda entry: entry["score"], reverse=True)
        return ordered[:limit]

    def evaluate(self, query: str, relevant: Iterable[str], *, limit: int = 5, corpus: Optional[Iterable[str]] = None) -> Dict[str, float]:
        results = self.retrieve(query, corpus=corpus, limit=limit)
        return evaluate_retrieval([entry["text"] for entry in results], relevant, k=limit)


__all__ = ["LexicalRetriever", "SemanticRetriever", "EmbeddingRetriever", "SQLiteVectorStore", "HybridRetriever", "evaluate_retrieval", "normalize_text"]
