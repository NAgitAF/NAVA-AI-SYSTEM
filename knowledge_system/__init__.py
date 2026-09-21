"""Knowledge System — persistent structured facts and relationship traversal."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import deque
from typing import Any, Dict, List, Optional


def normalize_term(value: Any) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def extract_facts(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Extract conservative entity and relation candidates from plain text.

    This is deliberately precision-first: only explicit copula/verb patterns
    are accepted, and ambiguous text is returned as entities without inventing
    relations.
    """
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    entities: set[str] = set()
    relations: list[dict[str, str]] = []
    def clean_component(value: str) -> str:
        return re.sub(r"^(?:and|the|a|an|و|ال)\s+", "", value.strip(), flags=re.IGNORECASE)
    pattern = re.compile(
        r"(?P<subject>[\w\u0621-\u064A][\w\u0621-\u064A -]{0,60}?)\s+"
        r"(?P<predicate>is|uses|has|contains|includes|هو|هي|يستخدم|تستخدم|يتضمن|يتضمنه|يحتوي على|يضم)\s+"
        r"(?P<object>[\w\u0621-\u064A][\w\u0621-\u064A -]{0,60}?)(?=[,.!?؛،]|$)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(clean):
        subject = clean_component(match.group("subject").strip(" ,.;:()[]"))
        predicate = match.group("predicate").strip()
        obj = clean_component(match.group("object").strip(" ,.;:()[]"))
        if not subject or not obj:
            continue
        entities.update((subject, obj))
        relations.append({"subject": subject, "predicate": predicate, "object": obj})
    for candidate in re.split(r"[,،؛.!?]|\band\b|و", clean, flags=re.IGNORECASE):
        candidate = candidate.strip(" ,.;:()[]")
        if 1 < len(candidate.split()) <= 6:
            entities.add(candidate)
    return {
        "entities": [{"name": item, "type": "entity"} for item in sorted(entities, key=normalize_term)],
        "relations": relations,
    }


class KnowledgeGraph:
    """A small persistent graph with deduplication and bounded traversal."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(os.getcwd(), "data", "knowledge_graph.json")
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict):
            self.nodes = data.get("nodes", {}) if isinstance(data.get("nodes", {}), dict) else {}
            self.edges = data.get("edges", []) if isinstance(data.get("edges", []), list) else []

    def _save(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"nodes": self.nodes, "edges": self.edges}, handle, ensure_ascii=False, indent=2)

    def add_node(self, name: str, *, node_type: str = "entity", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        key = normalize_term(name)
        if not key:
            raise ValueError("Knowledge graph node name cannot be empty.")
        node = self.nodes.setdefault(key, {"name": str(name).strip(), "type": node_type, "metadata": {}})
        node["type"] = node_type
        if metadata:
            node["metadata"].update(metadata)
        self._save()
        return node

    def add_fact(self, subject: str, predicate: str, obj: str, *, source: str = "system", confidence: float = 1.0) -> Dict[str, Any]:
        subject_key = normalize_term(subject)
        predicate_key = normalize_term(predicate)
        object_key = normalize_term(obj)
        if not subject_key or not predicate_key or not object_key:
            raise ValueError("Subject, predicate and object are required.")

        self.add_node(subject, node_type="entity")
        self.add_node(obj, node_type="entity")
        edge = {
            "subject": subject_key,
            "predicate": predicate_key,
            "object": object_key,
            "source": source,
            "confidence": max(0.0, min(1.0, float(confidence))),
        }
        if edge not in self.edges:
            self.edges.append(edge)
            self._save()
        return edge

    def ingest_text(self, text: str, *, source: str = "extraction", confidence: float = 0.7) -> Dict[str, Any]:
        extracted = extract_facts(text)
        for entity in extracted["entities"]:
            self.add_node(entity["name"], node_type=entity["type"], metadata={"extracted": True, "source": source})
        for relation in extracted["relations"]:
            self.add_fact(
                relation["subject"],
                relation["predicate"],
                relation["object"],
                source=source,
                confidence=confidence,
            )
        return extracted

    def query(self, term: str, *, predicate: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        key = normalize_term(term)
        predicate_key = normalize_term(predicate) if predicate else None
        matches = [
            edge for edge in self.edges
            if (edge["subject"] == key or edge["object"] == key)
            and (predicate_key is None or edge["predicate"] == predicate_key)
        ]
        return matches[:max(0, limit)]

    def neighbors(self, term: str, *, limit: int = 20) -> List[Dict[str, Any]]:
        return self.query(term, limit=limit)

    def traverse(self, start: str, *, depth: int = 2, limit: int = 20) -> List[Dict[str, Any]]:
        start_key = normalize_term(start)
        if not start_key or depth < 0:
            return []
        visited = {start_key}
        visited_edges = set()
        queue = deque([(start_key, 0)])
        results: List[Dict[str, Any]] = []
        while queue and len(results) < limit:
            current, current_depth = queue.popleft()
            if current_depth >= depth:
                continue
            for edge in self.query(current, limit=limit):
                edge_key = (edge["subject"], edge["predicate"], edge["object"])
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)
                results.append({**edge, "depth": current_depth + 1})
                next_term = edge["object"] if edge["subject"] == current else edge["subject"]
                if next_term not in visited:
                    visited.add(next_term)
                    queue.append((next_term, current_depth + 1))
                if len(results) >= limit:
                    break
        return results

    def snapshot(self) -> Dict[str, int]:
        return {"node_count": len(self.nodes), "edge_count": len(self.edges)}


class SQLiteKnowledgeGraph:
    """Scalable local graph backend using indexed SQLite tables."""

    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.executescript(
                "CREATE TABLE IF NOT EXISTS nodes (name TEXT PRIMARY KEY, type TEXT NOT NULL, metadata TEXT NOT NULL);"
                "CREATE TABLE IF NOT EXISTS edges (subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL, source TEXT, confidence REAL, PRIMARY KEY(subject,predicate,object));"
                "CREATE INDEX IF NOT EXISTS edges_subject ON edges(subject);"
                "CREATE INDEX IF NOT EXISTS edges_object ON edges(object);"
            )

    def add_fact(self, subject, predicate, obj, *, source="system", confidence=1.0):
        values = (normalize_term(subject), normalize_term(predicate), normalize_term(obj), source, max(0.0, min(1.0, float(confidence))))
        with sqlite3.connect(self.path) as connection:
            connection.execute("INSERT OR IGNORE INTO nodes(name,type,metadata) VALUES (?, 'entity', '{}')", (values[0],))
            connection.execute("INSERT OR IGNORE INTO nodes(name,type,metadata) VALUES (?, 'entity', '{}')", (values[2],))
            connection.execute("INSERT OR REPLACE INTO edges(subject,predicate,object,source,confidence) VALUES (?,?,?,?,?)", values)
            connection.commit()
        return {"subject": values[0], "predicate": values[1], "object": values[2], "source": source, "confidence": values[4]}

    def ingest_text(self, text, *, source="extraction", confidence=0.7):
        extracted = extract_facts(text)
        with sqlite3.connect(self.path) as connection:
            for entity in extracted["entities"]:
                connection.execute(
                    "INSERT OR IGNORE INTO nodes(name,type,metadata) VALUES (?, ?, ?)",
                    (normalize_term(entity["name"]), entity["type"], json.dumps({"extracted": True, "source": source})),
                )
        for relation in extracted["relations"]:
            self.add_fact(
                relation["subject"],
                relation["predicate"],
                relation["object"],
                source=source,
                confidence=confidence,
            )
        return extracted

    def query(self, term, *, predicate=None, limit=20):
        params = [normalize_term(term)]
        sql = "SELECT subject,predicate,object,source,confidence FROM edges WHERE (subject=? OR object=?)"
        params.append(params[0])
        if predicate:
            sql += " AND predicate=?"
            params.append(normalize_term(predicate))
        sql += " LIMIT ?"
        params.append(max(0, int(limit)))
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(sql, params).fetchall()
        return [{"subject": a, "predicate": b, "object": c, "source": d, "confidence": e} for a, b, c, d, e in rows]

    def snapshot(self):
        with sqlite3.connect(self.path) as connection:
            nodes = connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            edges = connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        return {"node_count": nodes, "edge_count": edges}


__all__ = ["KnowledgeGraph", "SQLiteKnowledgeGraph", "extract_facts", "normalize_term"]
