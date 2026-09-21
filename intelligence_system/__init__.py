"""Intelligence System — supervision, routing, planning and reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
import json
from pathlib import Path
import re
import inspect
from typing import Any, Callable


class SupervisorAgent:
    """Central orchestrator responsible for intent classification and strategy."""

    def __init__(self):
        self.last_intent = None

    def analyze(self, prompt):
        self.last_intent = {"prompt": prompt, "type": "general"}
        return self.last_intent


class DynamicExpertRouter:
    """Chooses the relevant domain expert for a task."""

    def __init__(self):
        self.experts = ["general", "coding", "security", "research", "mathematics"]

    def route(self, intent):
        return intent.get("type", "general")


class LearnedExpertRouter:
    """A dependency-free online classifier based on character/word centroids.

    It learns from labelled examples and feedback; rules are only used by the
    caller as a last-resort fallback when confidence is too low.
    """

    def __init__(self, experts: list[str] | None = None, min_confidence: float = 0.38):
        self.experts = experts or ["general", "coding", "security", "research", "mathematics"]
        self.min_confidence = min_confidence
        self._profiles: dict[str, Counter[str]] = {expert: Counter() for expert in self.experts}
        self._counts: Counter[str] = Counter()
        self._vocabulary: Counter[str] = Counter()

    @staticmethod
    def _features(text: str) -> Counter[str]:
        normalized = " ".join(str(text or "").lower().split())
        words = re.findall(r"[\w\u0621-\u064A]+", normalized, flags=re.UNICODE)
        features = Counter(f"w:{word}" for word in words)
        compact = re.sub(r"\s+", " ", normalized)
        features.update(f"c:{compact[index:index + 3]}" for index in range(max(0, len(compact) - 2)))
        return features

    def fit(self, examples: list[tuple[str, str]]) -> "LearnedExpertRouter":
        for text, expert in examples:
            self.learn(text, expert)
        return self

    def learn(self, text: str, expert: str, weight: float = 1.0) -> None:
        if expert not in self._profiles:
            self._profiles[expert] = Counter()
        features = self._features(text)
        if not features:
            return
        self._profiles[expert].update({key: value * max(0.1, weight) for key, value in features.items()})
        self._counts[expert] += weight
        self._vocabulary.update(features.keys())

    def _similarity(self, query: Counter[str], profile: Counter[str]) -> float:
        if not query or not profile:
            return 0.0
        keys = set(query) | set(profile)
        numerator = sum(query[key] * profile[key] for key in keys)
        q_norm = math.sqrt(sum(value * value for value in query.values()))
        p_norm = math.sqrt(sum(value * value for value in profile.values()))
        return numerator / (q_norm * p_norm) if q_norm and p_norm else 0.0

    def route_details(self, text: str) -> dict[str, Any]:
        query = self._features(text)
        scores = {expert: self._similarity(query, profile) for expert, profile in self._profiles.items()}
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        best, best_score = ranked[0] if ranked else ("general", 0.0)
        total = sum(max(score, 0.0) for _, score in ranked)
        confidence = best_score / total if total else 0.0
        return {
            "expert": best if confidence >= self.min_confidence else "general",
            "confidence": round(confidence, 3),
            "learned": bool(self._vocabulary),
            "candidates": [{"expert": expert, "score": round(score, 3)} for expert, score in ranked if score > 0],
        }

    def route(self, text: str) -> str:
        return self.route_details(text)["expert"]


@dataclass(frozen=True)
class ReasoningTrace:
    objective: str
    subgoals: list[str]
    assumptions: list[str]
    constraints: list[str]
    confidence: float
    next_action: str
    strategy: str
    evidence_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "subgoals": self.subgoals,
            "assumptions": self.assumptions,
            "constraints": self.constraints,
            "confidence": self.confidence,
            "next_action": self.next_action,
            "strategy": self.strategy,
            "evidence_count": self.evidence_count,
        }


class Reasoner:
    """Produces an explicit, bounded reasoning trace before execution."""

    def reason(self, prompt: str, intent: dict[str, Any], *, evidence: list[Any] | None = None) -> dict[str, Any]:
        objective = " ".join(str(prompt or "").split())
        if not objective:
            raise ValueError("A non-empty prompt is required for reasoning.")
        clauses = [
            part.strip(" ,.")
            for part in re.split(r"\b(?:then|and then|ثم|و)\b", objective, flags=re.IGNORECASE)
            if part.strip()
        ]
        subgoals = clauses[:4] or [objective]
        domain = intent.get("expert", intent.get("intent", "general"))
        usable_evidence = [item for item in (evidence or []) if item is not None and str(item).strip()]
        assumptions = [] if usable_evidence else ["No external evidence was supplied."]
        constraints = ["Use only permitted tools.", "Stop after the configured step budget."]
        if domain == "research":
            constraints.append("Separate retrieved evidence from generated conclusions.")
        if intent.get("needs_tools"):
            constraints.append("Validate tool results before using them.")
        routing_confidence = float(intent.get("routing", {}).get("confidence", 0.0) or 0.0)
        confidence = min(
            0.95,
            0.45
            + min(0.2, routing_confidence)
            + (0.15 if domain != "general" else 0)
            + (0.1 if usable_evidence else 0),
        )
        return ReasoningTrace(
            objective=objective,
            subgoals=subgoals,
            assumptions=assumptions,
            constraints=constraints,
            confidence=round(confidence, 2),
            next_action="retrieve_evidence" if intent.get("needs_retrieval") and not usable_evidence else "generate_response",
            strategy="evidence_first" if intent.get("needs_retrieval") else "direct_execution",
            evidence_count=len(usable_evidence),
        ).as_dict()


class AgentLoop:
    """Executes a plan through named handlers with retry and step limits."""

    def __init__(self, task_manager: "TaskManager"):
        self.task_manager = task_manager

    def run(self, task_id: str, handlers: dict[str, Callable[[], Any]]) -> dict[str, Any]:
        if task_id not in self.task_manager.tasks:
            raise KeyError(f"Unknown task '{task_id}'.")
        task = self.task_manager.tasks[task_id]
        results: dict[str, Any] = {}
        while True:
            step = self.task_manager.next_step(task_id)
            if step is None:
                break
            handler = handlers.get(step)
            if handler is None:
                error = f"No handler registered for '{step}'."
                self.task_manager.fail_step(task_id, step, error)
                continue
            try:
                if len(inspect.signature(handler).parameters) == 1:
                    results[step] = handler(dict(results))
                else:
                    results[step] = handler()
                self.task_manager.complete_step(task_id, step)
            except Exception as exc:
                self.task_manager.fail_step(task_id, step, exc)
        task["outputs"] = dict(results)
        return results


class ExpertModelRegistry:
    """Maps experts to independently deployable model/adapters.

    Registration is metadata-only; ``is_trained`` is true only when the
    referenced directory contains both a model configuration and weights.
    """

    def __init__(self):
        self.models = {}

    def register(self, expert, model_path, *, adapter_path=None, enabled=True):
        if not expert or not model_path:
            raise ValueError("Expert and model_path are required.")
        self.models[expert] = {"model_path": model_path, "adapter_path": adapter_path, "enabled": bool(enabled)}
        return self.models[expert]

    def register_manifest(self, manifest_path, *, enabled=True):
        path = Path(manifest_path)
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid expert manifest: {path}") from exc
        required = {"expert", "model_path", "adapter_path", "training_data"}
        if not required.issubset(manifest):
            raise ValueError(f"Expert manifest is missing fields: {sorted(required - set(manifest))}")
        def resolve_path(value):
            resolved = Path(value)
            return str(resolved if resolved.is_absolute() else (path.parent / resolved).resolve())
        entry = self.register(
            manifest["expert"],
            resolve_path(manifest["model_path"]),
            adapter_path=resolve_path(manifest["adapter_path"]),
            enabled=enabled,
        )
        entry.update({
            "expert": manifest["expert"],
            "manifest_path": str(path),
            "training_data": resolve_path(manifest["training_data"]),
            "base_model": manifest.get("base_model"),
            "status": manifest.get("status", "untrained"),
        })
        return entry

    def is_trained(self, expert):
        model = self.models.get(expert)
        if not model or not model["enabled"]:
            return False
        model_path = Path(model["model_path"])
        adapter_path = Path(model["adapter_path"]) if model.get("adapter_path") else None
        return (
            model_path.is_dir()
            and (model_path / "config.json").is_file()
            and any((model_path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
            and adapter_path is not None
            and adapter_path.is_dir()
            and (adapter_path / "adapter_config.json").is_file()
            and any((adapter_path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))
        )

    def resolve(self, expert):
        model = self.models.get(expert)
        return model if model and model["enabled"] else None

    def disable(self, expert):
        if expert in self.models:
            self.models[expert]["enabled"] = False
        return self.models.get(expert)


class TaskPlanner:
    """Creates bounded execution plans from an analyzed request."""

    def build_plan(self, prompt, intent=None):
        intent = intent or {"type": "general"}
        expert = intent.get("expert", intent.get("intent", intent.get("type", "general")))
        steps = ["analyze_request", "build_context"]
        if expert in {"research", "coding"}:
            steps.append("retrieve_evidence")
        if intent.get("needs_tools"):
            steps.append("execute_allowed_tools")
        steps.extend(["generate_response", "evaluate_response", "store_experience"])
        return {
            "prompt": prompt,
            "intent": intent,
            "steps": steps,
            "max_retries": 2,
            "retry_budget": 3,
            "step_budget": min(8, len(steps)),
        }


class TaskManager:
    """Tracks task lifecycle and prevents unbounded agent loops."""

    def __init__(self, max_steps=8):
        self.max_steps = max_steps
        self.tasks = {}

    def start(self, task_id, plan):
        if not task_id:
            raise ValueError("task_id is required.")
        if not isinstance(plan, dict) or not plan.get("steps"):
            raise ValueError("A task plan with at least one step is required.")
        self.tasks[task_id] = {
            "status": "running",
            "plan": plan,
            "completed_steps": [],
            "attempts": {},
            "errors": [],
            "retry_count": 0,
            "outputs": {},
        }
        return self.tasks[task_id]

    def complete_step(self, task_id, step):
        task = self.tasks[task_id]
        if task["status"] != "running":
            raise RuntimeError(f"Task '{task_id}' is not running.")
        if step in task["completed_steps"]:
            return task
        step_budget = min(self.max_steps, int(task["plan"].get("step_budget", self.max_steps)))
        if len(task["completed_steps"]) >= step_budget:
            raise RuntimeError("Task exceeded the maximum execution steps.")
        task["completed_steps"].append(step)
        return task

    def next_step(self, task_id):
        task = self.tasks[task_id]
        if task["status"] != "running":
            return None
        for step in task["plan"].get("steps", []):
            if step not in task["completed_steps"]:
                return step
        return None

    def fail_step(self, task_id, step, error):
        task = self.tasks[task_id]
        attempts = task["attempts"].get(step, 0) + 1
        task["attempts"][step] = attempts
        task["retry_count"] += 1
        task["errors"].append({"step": step, "error": str(error), "attempt": attempts})
        if attempts > task["plan"].get("max_retries", 0) or task["retry_count"] > task["plan"].get("retry_budget", float("inf")):
            task["status"] = "failed"
            raise RuntimeError(f"Step '{step}' exceeded retry budget.")
        return task

    def cancel(self, task_id):
        self.tasks[task_id]["status"] = "cancelled"
        return self.tasks[task_id]

    def finish(self, task_id, status="completed"):
        self.tasks[task_id]["status"] = status
        return self.tasks[task_id]
