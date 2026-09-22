from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from memory_system import MemoryManager
from retrieval_system import HybridRetriever
from tools_system import ToolRegistry, build_default_registry
from knowledge_system import KnowledgeGraph
from intelligence_system import AgentLoop, ExpertModelRegistry, LearnedExpertRouter, Reasoner, TaskManager, TaskPlanner
from evaluation_system import Evaluator
from learning_system import ExperienceArchive
from observability_system import ObservabilityCollector


@dataclass
class MemoryEntry:
    """A single memory unit stored in the system."""

    content: str
    source: str
    importance: float = 0.5
    confidence: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))
    verified: bool = False
    expires_at: Optional[str] = None


@dataclass
class KnowledgeEntry:
    """Structured knowledge created for retrieval and graph reasoning."""

    subject: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolPermission:
    """Restricts what an executed tool is allowed to do."""

    tool_name: str
    allowed: bool = False
    scope: str = "restricted"
    max_runtime_seconds: int = 30
    network_access: bool = False


@dataclass
class VersionRecord:
    """Tracks a candidate model/version before merge."""

    version: str
    timestamp: str
    benchmark_score: float
    status: str = "candidate"


@dataclass
class NAVAState:
    """Global runtime state for the active model and subsystems."""

    current_model: str = "seed_brain"
    active_experts: List[str] = field(default_factory=list)
    active_tasks: List[str] = field(default_factory=list)
    training_state: str = "idle"
    system_health: str = "healthy"
    current_version: str = "v0.1"


class EventBus:
    """Simple event bus for subsystem communication."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[Callable[[Any], None]]] = {}

    def subscribe(self, event_name: str, callback: Callable[[Any], None]) -> None:
        self._listeners.setdefault(event_name, []).append(callback)

    def emit(self, event_name: str, payload: Any = None) -> None:
        for callback in self._listeners.get(event_name, []):
            callback(payload)


class ExpertRouter:
    """Routes requests using learned expert profiles."""

    def __init__(self) -> None:
        self.learned = LearnedExpertRouter()
        self.learned.fit([
            ("write python code debug an api test", "coding"),
            ("حل مشكلة برمجية واكتب كود بايثون", "coding"),
            ("review jwt oauth vulnerability threat", "security"),
            ("تحليل هجوم وثغرة وأمن النظام", "security"),
            ("research papers compare sources and evidence", "research"),
            ("ابحث عن مصادر ومعلومات موثوقة", "research"),
            ("calculate equation algebra formula", "mathematics"),
            ("احسب المعادلة الرياضية والجبر", "mathematics"),
            ("explain a concept or help me decide", "general"),
        ])

    def route_details(self, intent: str) -> Dict[str, Any]:
        return self.learned.route_details(intent)

    def route(self, intent: str) -> str:
        return self.route_details(intent)["expert"]

    def learn(self, text: str, expert: str, weight: float = 1.0) -> Dict[str, Any]:
        """Update routing from a labelled human/evaluation outcome."""
        self.learned.learn(text, expert, weight)
        return self.learned.route_details(text)


class SupervisorAgent:
    """Analyzes the prompt and decides the execution strategy."""

    def __init__(self) -> None:
        self.router = ExpertRouter()

    def analyze(self, prompt: str) -> Dict[str, Any]:
        normalized = (prompt or "").strip()
        if not normalized:
            return {"intent": "general", "expert": "general", "needs_tools": False, "needs_retrieval": False}

        routing = self.router.route_details(normalized)
        intent = routing["expert"]

        return {
            "intent": intent,
            "expert": intent,
            "routing": routing,
            "needs_tools": intent in {"coding", "mathematics", "research"},
            "needs_retrieval": intent in {"research", "general"},
        }


class NAVAArchitecture:
    """High-level architectural foundation for the NAVA AI runtime."""

    def __init__(self) -> None:
        self.state = NAVAState()
        self.event_bus = EventBus()
        self.memories: List[MemoryEntry] = []
        self.knowledge: List[KnowledgeEntry] = []
        self.tools: Dict[str, ToolPermission] = {}
        self.versions: List[VersionRecord] = []
        self.memory_manager = MemoryManager()
        self.retriever = HybridRetriever()
        self.tool_registry = build_default_registry()
        self.knowledge_graph = KnowledgeGraph()
        self.task_planner = TaskPlanner()
        self.task_manager = TaskManager()
        self.agent_loop = AgentLoop(self.task_manager)
        self.evaluator = Evaluator()
        self.experience_archive = ExperienceArchive()
        self.observability = ObservabilityCollector()
        self.expert_models = ExpertModelRegistry()
        self._register_expert_models()

    def _register_expert_models(self) -> None:
        """Register expert model manifests for any trained or pending expert adapters."""
        root = Path(__file__).resolve().parent
        experts_dir = root / "brain_system" / "experts"
        if not experts_dir.exists():
            return
        for manifest_path in sorted(experts_dir.glob("*/manifest.json")):
            try:
                self.expert_models.register_manifest(str(manifest_path), enabled=True)
            except ValueError:
                continue

    def register_tool(self, tool_name: str, allowed: bool = False, scope: str = "restricted") -> ToolPermission:
        permission = ToolPermission(tool_name=tool_name, allowed=allowed, scope=scope)
        self.tools[tool_name] = permission
        self.tool_registry.register(tool_name, lambda *args, **kwargs: {"tool": tool_name, "args": args, "kwargs": kwargs}, allowed=allowed, scope=scope)
        self.event_bus.emit("TOOL_REGISTERED", {"tool_name": tool_name, "allowed": allowed})
        return permission

    def grant_permission(self, tool_name: str, *, allowed: bool, scope: str = "restricted", max_runtime_seconds: int = 30, network_access: bool = False) -> ToolPermission:
        permission = self.tools.get(tool_name)
        if permission is None:
            permission = ToolPermission(tool_name=tool_name)
        permission.allowed = allowed
        permission.scope = scope
        permission.max_runtime_seconds = max_runtime_seconds
        permission.network_access = network_access
        self.tools[tool_name] = permission
        self.tool_registry.permission_engine.allow(tool_name, allowed, scope=scope, max_runtime_seconds=max_runtime_seconds, network_access=network_access)
        return permission

    def execute_tool(self, tool_name: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        if tool_name not in self.tool_registry.tools:
            return {"ok": False, "error": f"Tool '{tool_name}' is not registered."}
        return self.tool_registry.execute(tool_name, *args, **kwargs)

    def record_memory(self, content: str, source: str = "user", *, importance: float = 0.5, confidence: float = 0.5, verified: bool = False, ttl_seconds: Optional[int] = None) -> MemoryEntry:
        entry = MemoryEntry(content=content, source=source, importance=importance, confidence=confidence, verified=verified)
        self.memories.append(entry)
        self.memory_manager.add(content[:80], content, source=source, scope="long_term", importance=importance, confidence=confidence, ttl_seconds=ttl_seconds)
        self.retriever.add(content)
        self.event_bus.emit("MEMORY_CREATED", {"content": content, "source": source})
        return entry

    def store_knowledge(self, subject: str, description: str, **metadata: Any) -> KnowledgeEntry:
        entry = KnowledgeEntry(subject=subject, description=description, metadata=metadata)
        self.knowledge.append(entry)
        self.memory_manager.add(subject, description, source="knowledge", scope="long_term", metadata=metadata)
        self.retriever.add(f"{subject}: {description}")
        self.knowledge_graph.ingest_text(
            f"{subject} {description}",
            source=metadata.get("source", "knowledge"),
            confidence=float(metadata.get("confidence", 0.7)),
        )
        for relation in metadata.get("relations", []):
            if isinstance(relation, dict) and {"predicate", "object"} <= relation.keys():
                self.knowledge_graph.add_fact(
                    subject,
                    relation["predicate"],
                    relation["object"],
                    source=metadata.get("source", "knowledge"),
                    confidence=metadata.get("confidence", 1.0),
                )
        self.event_bus.emit("KNOWLEDGE_ADDED", {"subject": subject})
        return entry

    def add_knowledge_fact(self, subject: str, predicate: str, obj: str, **metadata: Any) -> Dict[str, Any]:
        edge = self.knowledge_graph.add_fact(subject, predicate, obj, **metadata)
        self.event_bus.emit("KNOWLEDGE_RELATION_ADDED", edge)
        return edge

    def mark_version(self, version: str, benchmark_score: float, status: str = "candidate") -> VersionRecord:
        record = VersionRecord(version=version, timestamp=datetime.utcnow().isoformat(timespec="seconds"), benchmark_score=benchmark_score, status=status)
        self.versions.append(record)
        self.event_bus.emit("VERSION_REGISTERED", {"version": version, "benchmark_score": benchmark_score})
        return record

    def snapshot(self) -> Dict[str, Any]:
        return {
            "current_model": self.state.current_model,
            "active_experts": self.state.active_experts,
            "active_tasks": self.state.active_tasks,
            "training_state": self.state.training_state,
            "system_health": self.state.system_health,
            "memory_count": len(self.memories),
            "knowledge_count": len(self.knowledge),
            "tool_count": len(self.tools),
            "version_count": len(self.versions),
            "current_version": self.state.current_version,
            "knowledge_graph": self.knowledge_graph.snapshot(),
            "observability": self.observability.snapshot(),
            "active_task_count": len(self.task_manager.tasks),
        }


class NAVAAgentRuntime:
    """Concrete runtime loop for intent analysis, routing, tool use and response generation."""

    def __init__(self, architecture: Optional[NAVAArchitecture] = None) -> None:
        self.architecture = architecture or NAVAArchitecture()
        self.supervisor = SupervisorAgent()
        self.reasoner = Reasoner()
        self.expert_models = ExpertModelRegistry()
        self.tool_plan: List[str] = []

    def execute(self, user_input: str, *, memory_store=None, tool_registry=None, safety_guard=None, generator: Optional[Callable[..., str]] = None) -> Dict[str, Any]:
        intent_info = self.supervisor.analyze(user_input)
        expert = intent_info.get("expert", "general")
        selected_model = self.architecture.expert_models.resolve(expert)
        if selected_model is None:
            selected_model = {"expert": expert, "enabled": True, "status": "fallback_to_general", "model_path": self.architecture.state.current_model}
        if not self.architecture.expert_models.is_trained(expert):
            expert = "general"
            intent_info["expert"] = expert
            intent_info["intent"] = expert
            selected_model = self.architecture.expert_models.resolve(expert) or {"expert": expert, "enabled": True, "status": "general_fallback"}
        task_id = f"task-{len(self.architecture.task_manager.tasks) + 1}"
        plan = self.architecture.task_planner.build_plan(user_input, intent_info)
        self.architecture.task_manager.start(task_id, plan)
        reasoning = self.reasoner.reason(user_input, intent_info)
        self.architecture.observability.event("task_started", task_id=task_id, expert=expert)

        self.architecture.state.active_experts = [expert]
        self.architecture.state.active_tasks = [user_input[:80]]
        self.architecture.state.training_state = "active"
        self.architecture.state.system_health = "healthy"
        self.architecture.event_bus.emit("TASK_STARTED", {"expert": expert, "prompt": user_input})

        memory_context = []
        self.architecture.task_manager.complete_step(task_id, "analyze_request")
        if memory_store is not None and hasattr(memory_store, "retrieve"):
            try:
                memory_context = memory_store.retrieve(user_input, limit=3)
            except Exception:
                memory_context = []
        elif hasattr(self.architecture, "memory_manager"):
            try:
                memory_context = self.architecture.memory_manager.retrieve(user_input, limit=3)
            except Exception:
                memory_context = []

        if hasattr(self.architecture, "retriever") and memory_context:
            retrieval_hits = self.architecture.retriever.retrieve(user_input, corpus=[f"{item['topic']} {item['fact']}" for item in memory_context], limit=3)
            if retrieval_hits:
                memory_context = [
                    {"topic": "retrieval", "fact": hit["text"], "source": "retrieval", "scope": "retrieved", "score": hit["score"]}
                    for hit in retrieval_hits
                ]

        graph_context = []
        if hasattr(self.architecture, "knowledge_graph"):
            graph_context = self.architecture.knowledge_graph.traverse(user_input, depth=1, limit=3)
        self.architecture.task_manager.complete_step(task_id, "build_context")

        tool_results = []
        if tool_registry is not None and hasattr(tool_registry, "execute"):
            try:
                tool_results = tool_registry.execute(user_input)
            except Exception:
                tool_results = []
        elif hasattr(self.architecture, "tool_registry"):
            try:
                tool_results = self.architecture.tool_registry.list_tools()
                tool_results = [{"tool": name, "result": self.architecture.execute_tool(name, user_input)} for name in tool_results if self.architecture.tool_registry.permission_engine.check(name)]
            except Exception:
                tool_results = []
        self.tool_plan = [item.get("tool") for item in tool_results if isinstance(item, dict)]
        if "retrieve_evidence" in plan["steps"]:
            self.architecture.task_manager.complete_step(task_id, "retrieve_evidence")
        if "execute_allowed_tools" in plan["steps"]:
            self.architecture.task_manager.complete_step(task_id, "execute_allowed_tools")

        if generator is not None:
            response = generator(user_input, intent_info, expert, memory_context, tool_results)
        else:
            response = (
                f"تم تحليل الطلب بصيغة {intent_info.get('intent', 'general')}، "
                f"والخبير المختار هو {expert}. "
                "جارٍ معالجة المهمة داخل NAVA Runtime."
            )
        self.architecture.task_manager.complete_step(task_id, "generate_response")

        if safety_guard is not None and hasattr(safety_guard, "review"):
            safety = safety_guard.review(response)
            if not safety.get("safe", True):
                refusal = getattr(safety_guard, "apply_refusal", lambda r: r)(response)
                response = refusal

        evaluation = self.architecture.evaluator.score(response, user_input, memory_context)
        self.architecture.task_manager.complete_step(task_id, "evaluate_response")
        self.architecture.experience_archive.add({"input": user_input, "output": response, "score": evaluation["score"], "expert": expert})
        self.architecture.task_manager.finish(task_id, "completed" if evaluation["passed"] else "needs_review")
        self.architecture.observability.record("last_quality_score", evaluation["score"])
        self.architecture.observability.event("task_completed", task_id=task_id, score=evaluation["score"])

        self.architecture.record_memory(user_input, source="user", importance=0.9, confidence=0.9)
        self.architecture.record_memory(response, source="assistant", importance=0.8, confidence=0.8)
        self.architecture.task_manager.complete_step(task_id, "store_experience")
        self.architecture.event_bus.emit("TASK_COMPLETED", {"expert": expert, "response_length": len(response)})

        return {
            "intent": intent_info,
            "expert": expert,
            "expert_model": selected_model,
            "memory_context": memory_context,
            "graph_context": graph_context,
            "tool_results": tool_results,
            "evaluation": evaluation,
            "plan": plan,
            "reasoning": reasoning,
            "response": response,
        }


if __name__ == "__main__":
    system = NAVAArchitecture()
    runtime = NAVAAgentRuntime(system)
    result = runtime.execute("احسب 12 * 7 ثم اشرح النتيجة.")
    system.record_memory("خطة NAVA المهندسة مبنية على طبقات متداخلة من الذاكرة والذكاء والأمان.", source="architecture")
    system.store_knowledge("NAVA", "نظام ذكاء اصطناعي متكامل يتضمن نموذجًا، ذاكرة، أدوات، أمان، وتعلم متكرر.", domain="ai")
    print(result)
    print(system.snapshot())
