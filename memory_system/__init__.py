"""Persistent memory with SQLite storage, expiry and bounded forgetting."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = re.sub(r"[^\w\s\u0621-\u064A]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class MemoryManager:
    """Memory manager with durable SQLite records and explicit forgetting policy."""

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        max_entries: int = 10_000,
        default_ttl_seconds: Optional[int] = None,
        forget_below_importance: float = 0.0,
    ):
        self.path = path or os.path.join(os.getcwd(), "data", "memory_store.db")
        self.max_entries = max_entries
        self.default_ttl_seconds = default_ttl_seconds
        self.forget_below_importance = forget_below_importance
        self._sqlite = self.path.endswith(".db")
        self.short_term: List[Dict[str, Any]] = []
        self.long_term: List[Dict[str, Any]] = []
        self.entries: List[Dict[str, Any]] = []
        self._ensure_parent_dir()
        if self._sqlite:
            self._init_db()
        else:
            self._load_json()
        self.purge_expired()

    def _ensure_parent_dir(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL, fact TEXT NOT NULL, source TEXT NOT NULL,
                    scope TEXT NOT NULL, created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL, expires_at TEXT,
                    importance REAL NOT NULL, confidence REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0, metadata TEXT NOT NULL,
                    UNIQUE(topic, fact)
                )"""
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_memories_expiry ON memories(expires_at)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)")

    def _load_json(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            data = []
        self.entries = data if isinstance(data, list) else []
        self._refresh_lists()

    def _refresh_lists(self) -> None:
        self.short_term = [item for item in self.entries if item.get("scope") == "short_term"]
        self.long_term = [item for item in self.entries if item.get("scope") != "short_term"]

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _iso(value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat()

    def _token_overlap(self, query: str, text: str) -> float:
        q_tokens, t_tokens = set(normalize_text(query).split()), set(normalize_text(text).split())
        return len(q_tokens & t_tokens) / max(1, len(q_tokens | t_tokens)) if q_tokens and t_tokens else 0.0

    def add(
        self, topic: str, fact: str, source: str = "system", *, scope: str = "long_term",
        metadata: Optional[Dict[str, Any]] = None, importance: float = 0.5,
        confidence: float = 0.5, ttl_seconds: Optional[int] = None,
    ):
        if not topic or not fact:
            raise ValueError("Memory topic and fact are required.")
        now = self._now()
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        expires_at = self._iso(now + timedelta(seconds=ttl)) if ttl is not None else None
        record = {
            "topic": str(topic).strip(), "fact": str(fact).strip(), "source": str(source),
            "scope": scope, "created_at": self._iso(now), "last_accessed_at": self._iso(now),
            "expires_at": expires_at, "importance": max(0.0, min(1.0, float(importance))),
            "confidence": max(0.0, min(1.0, float(confidence))), "access_count": 0,
            "metadata": metadata or {},
        }
        if self._sqlite:
            with self._connect() as db:
                db.execute(
                    """INSERT INTO memories
                    (topic,fact,source,scope,created_at,last_accessed_at,expires_at,importance,confidence,metadata)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(topic,fact) DO UPDATE SET
                    last_accessed_at=excluded.last_accessed_at, expires_at=excluded.expires_at,
                    importance=excluded.importance, confidence=excluded.confidence""",
                    (*[record[key] for key in ("topic", "fact", "source", "scope", "created_at", "last_accessed_at", "expires_at", "importance", "confidence")], json.dumps(record["metadata"], ensure_ascii=False)),
                )
            self._enforce_capacity()
            return record
        for existing in self.entries:
            if normalize_text(existing.get("topic")) == normalize_text(topic) and normalize_text(existing.get("fact")) == normalize_text(fact):
                return existing
        self.entries.append(record)
        self._save_json()
        self._refresh_lists()
        self._enforce_capacity()
        return record

    def add_short_term(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self.add(item.get("topic", ""), item.get("fact", ""), item.get("source", "system"), scope="short_term", metadata=item.get("metadata", {}), ttl_seconds=item.get("ttl_seconds"))

    def add_long_term(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self.add(item.get("topic", ""), item.get("fact", ""), item.get("source", "system"), scope="long_term", metadata=item.get("metadata", {}), importance=item.get("importance", 0.5), confidence=item.get("confidence", 0.5), ttl_seconds=item.get("ttl_seconds"))

    def purge_expired(self) -> int:
        now = self._iso(self._now())
        if self._sqlite:
            with self._connect() as db:
                result = db.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,))
                return result.rowcount
        before = len(self.entries)
        self.entries = [item for item in self.entries if not item.get("expires_at") or item["expires_at"] > now]
        self._save_json()
        self._refresh_lists()
        return before - len(self.entries)

    def forget(self, *, below_importance: Optional[float] = None) -> int:
        threshold = self.forget_below_importance if below_importance is None else below_importance
        if self._sqlite:
            with self._connect() as db:
                result = db.execute("DELETE FROM memories WHERE importance < ?", (threshold,))
                return result.rowcount
        before = len(self.entries)
        self.entries = [item for item in self.entries if item.get("importance", 0.5) >= threshold]
        self._save_json()
        self._refresh_lists()
        return before - len(self.entries)

    def _enforce_capacity(self) -> None:
        self.purge_expired()
        if self._sqlite:
            with self._connect() as db:
                db.execute(
                    """DELETE FROM memories WHERE id IN (
                    SELECT id FROM memories ORDER BY importance ASC, access_count ASC, last_accessed_at ASC
                    LIMIT MAX(0, (SELECT COUNT(*) FROM memories) - ?))""",
                    (self.max_entries,),
                )
        elif len(self.entries) > self.max_entries:
            self.entries.sort(key=lambda item: (item.get("importance", 0.5), item.get("access_count", 0), item.get("last_accessed_at", "")))
            self.entries = self.entries[-self.max_entries:]
            self._save_json()
            self._refresh_lists()

    def retrieve(self, query: str, limit: int = 5, *, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        if not query:
            return []
        self.purge_expired()
        if self._sqlite:
            with self._connect() as db:
                rows = db.execute("SELECT * FROM memories" + (" WHERE scope = ?" if scope else ""), ((scope,) if scope else ())).fetchall()
            candidates = [dict(row) | {"metadata": json.loads(row["metadata"])} for row in rows]
        else:
            candidates = [item for item in self.entries if scope is None or item.get("scope") == scope]
        scored = []
        for item in candidates:
            score = self._token_overlap(query, f"{item.get('topic', '')} {item.get('fact', '')}")
            if score:
                item["score"] = score + (0.05 * item.get("importance", 0.5))
                scored.append(item)
        scored.sort(key=lambda entry: entry["score"], reverse=True)
        selected = scored[:limit]
        if self._sqlite and selected:
            with self._connect() as db:
                now = self._iso(self._now())
                for item in selected:
                    db.execute("UPDATE memories SET access_count=access_count+1,last_accessed_at=? WHERE id=?", (now, item["id"]))
        return selected

    def all(self) -> List[Dict[str, Any]]:
        if self._sqlite:
            with self._connect() as db:
                return [dict(row) for row in db.execute("SELECT * FROM memories ORDER BY id").fetchall()]
        return list(self.entries)

    def _save_json(self) -> None:
        if not self._sqlite:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self.entries, handle, ensure_ascii=False, indent=2)

    def snapshot(self) -> Dict[str, Any]:
        count = len(self.all())
        return {"short_term_count": len(self.short_term), "long_term_count": count - len(self.short_term), "total_count": count, "path": self.path, "storage": "sqlite" if self._sqlite else "json"}


__all__ = ["MemoryManager", "normalize_text"]
