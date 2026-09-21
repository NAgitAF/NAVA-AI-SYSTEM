"""Durable event bus with bounded retries and idempotent delivery."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Protocol


class EventPublisher(Protocol):
    def publish(self, event: Dict[str, Any]) -> None: ...


class EventBus:
    """Dispatches events synchronously and persists an append-only event log."""

    def __init__(self, path: Optional[str] = None, max_retries: int = 2, publisher: Optional[EventPublisher] = None):
        self.path = path or os.path.join(os.getcwd(), "data", "events.jsonl")
        self.max_retries = max(0, int(max_retries))
        self.listeners: Dict[str, list[Callable[[Any], None]]] = {}
        self._seen: set[str] = set()
        self._lock = threading.Lock()
        self.publisher = publisher

    def subscribe(self, event_name: str, callback: Callable[[Any], None]) -> None:
        self.listeners.setdefault(event_name, []).append(callback)

    def emit(self, event_name: str, payload: Any = None, *, event_id: Optional[str] = None) -> Dict[str, Any]:
        event = {
            "id": event_id or uuid.uuid4().hex,
            "name": event_name,
            "payload": payload,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "attempts": 0,
            "status": "pending",
        }
        with self._lock:
            if event["id"] in self._seen:
                return {**event, "status": "duplicate"}
            self._seen.add(event["id"])
            self._persist(event)
            if self.publisher is not None:
                try:
                    self.publisher.publish(event)
                    event["published"] = True
                except Exception as exc:
                    event["published"] = False
                    event["publish_error"] = str(exc)
        callbacks = self.listeners.get(event_name, [])
        try:
            for callback in callbacks:
                attempts = 0
                while True:
                    try:
                        callback(payload)
                        break
                    except Exception:
                        attempts += 1
                        event["attempts"] = attempts
                        if attempts > self.max_retries:
                            raise
            event["status"] = "delivered"
        except Exception as exc:
            event["status"] = "failed"
            event["error"] = str(exc)
        self._persist(event)
        return event

    def _persist(self, event: Dict[str, Any]) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def snapshot(self) -> Dict[str, int]:
        return {"listeners": sum(len(items) for items in self.listeners.values()), "seen_events": len(self._seen), "distributed": self.publisher is not None}


__all__ = ["EventBus"]
