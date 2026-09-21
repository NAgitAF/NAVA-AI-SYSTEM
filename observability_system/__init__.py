"""Structured metrics, logs and lightweight tracing."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class ObservabilityCollector:
    def __init__(self, log_path: Optional[str] = None):
        self.metrics: Dict[str, Any] = {}
        self.events: list[Dict[str, Any]] = []
        self.log_path = log_path
        self._metric_types: Counter[str] = Counter()
        self.service_name = "nava"

    def record(self, name: str, value: Any) -> Any:
        self.metrics[name] = value
        return value

    def increment(self, name: str, amount: float = 1) -> float:
        value = float(self.metrics.get(name, 0)) + amount
        self.metrics[name] = value
        return value

    def prometheus(self) -> str:
        lines = []
        for name, value in self.metrics.items():
            metric = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            lines.append(f"# TYPE {metric} gauge")
            lines.append(f"{metric} {float(value)}")
        lines.append(f'nava_events_total {len(self.events)}')
        return "\n".join(lines) + "\n"

    def trace_context(self) -> Dict[str, str]:
        return {"trace_id": uuid.uuid4().hex, "service": self.service_name}

    def event(self, name: str, **payload: Any) -> Dict[str, Any]:
        event = {"name": name, "payload": payload, "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}
        self.events.append(event)
        if self.log_path:
            directory = os.path.dirname(self.log_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def start_trace(self, name: str) -> "TraceSpan":
        return TraceSpan(self, name)

    def snapshot(self) -> Dict[str, Any]:
        return {"metrics": dict(self.metrics), "event_count": len(self.events)}


class TraceSpan:
    def __init__(self, collector: ObservabilityCollector, name: str):
        self.collector = collector
        self.name = name
        self.trace_id = collector.trace_context()["trace_id"]
        self.started = time.monotonic()

    def finish(self, **payload: Any) -> Dict[str, Any]:
        duration = round(time.monotonic() - self.started, 6)
        payload.update({"trace_id": self.trace_id, "duration_seconds": duration})
        self.collector.increment(f"trace.{self.name}.count")
        self.collector.record(f"trace.{self.name}.last_duration_seconds", duration)
        return self.collector.event(self.name, **payload)


__all__ = ["ObservabilityCollector", "TraceSpan"]
