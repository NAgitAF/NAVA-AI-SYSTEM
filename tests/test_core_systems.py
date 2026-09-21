import tempfile
from pathlib import Path

from event_system import EventBus
from intelligence_system import AgentLoop, ExpertModelRegistry, LearnedExpertRouter, Reasoner, TaskManager
from nava_architecture import ExpertRouter
from knowledge_system import SQLiteKnowledgeGraph, extract_facts
from retrieval_system import SQLiteVectorStore, HybridRetriever, evaluate_retrieval
from gateway_system import JWTAuth
from brain_system.validator import validate_brain_artifacts
from memory_system import MemoryManager
from safety_system import PolicyEngine, RiskLevel, SafetyGuard
from tools_system import ToolRegistry, PermissionEngine, ToolSandbox
from evaluation_system import EvaluationCase, Evaluator
from learning_system import ExperienceArchive
from training_system.pipeline import TrainingPipeline
from benchmark_system import BenchmarkRunner
from versioning_system import ModelVersionStore, VersionRegistry
from observability_system import ObservabilityCollector


def test_event_bus_is_idempotent():
    calls = []
    bus = EventBus(path=tempfile.mktemp())
    bus.subscribe("task", calls.append)
    first = bus.emit("task", {"id": 1}, event_id="same")
    second = bus.emit("task", {"id": 1}, event_id="same")
    assert first["status"] == "delivered"
    assert second["status"] == "duplicate"
    assert len(calls) == 1


def test_event_bus_supports_optional_distributed_publisher():
    published = []
    bus = EventBus(path=tempfile.mktemp(), publisher=type("Publisher", (), {"publish": published.append})())
    result = bus.emit("task", {"id": 1})
    assert result["published"] is True
    assert published[0]["name"] == "task"
    assert bus.snapshot()["distributed"] is True


def test_observability_exports_prometheus_metrics_and_trace_context():
    collector = ObservabilityCollector()
    collector.increment("task.completed")
    span = collector.start_trace("task")
    event = span.finish(status="ok")
    assert "task_completed" in collector.prometheus()
    assert event["payload"]["trace_id"]


def test_task_retry_budget():
    manager = TaskManager()
    manager.start("t", {"steps": ["x"], "max_retries": 1})
    manager.fail_step("t", "x", "temporary")
    try:
        manager.fail_step("t", "x", "permanent")
    except RuntimeError:
        pass
    else:
        raise AssertionError("retry budget was not enforced")


def test_sqlite_indexes():
    with tempfile.TemporaryDirectory() as directory:
        vector = SQLiteVectorStore(str(Path(directory) / "vectors.db"))
        vector.upsert("hello", [1.0], {"source": "test"})
        assert vector.all()[0]["text"] == "hello"
        graph = SQLiteKnowledgeGraph(str(Path(directory) / "graph.db"))
        graph.add_fact("NAVA", "uses", "Python")
        assert graph.snapshot() == {"node_count": 2, "edge_count": 1}


def test_knowledge_extraction_finds_explicit_entities_and_relations():
    extracted = extract_facts("NAVA uses Python, and Python is a language.")
    assert {"subject": "NAVA", "predicate": "uses", "object": "Python"} in extracted["relations"]
    assert any(entity["name"] == "language" for entity in extracted["entities"])


def test_sqlite_graph_ingests_extracted_facts():
    with tempfile.TemporaryDirectory() as directory:
        graph = SQLiteKnowledgeGraph(str(Path(directory) / "graph.db"))
        result = graph.ingest_text("NAVA uses Python.", source="document", confidence=0.8)
        assert result["relations"][0]["predicate"] == "uses"
        assert graph.query("nava", predicate="uses")[0]["confidence"] == 0.8


def test_vector_store_search_and_rebuild_support_metadata_filters():
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteVectorStore(str(Path(directory) / "vectors.db"))
        store.upsert("NAVA runs on Python and local memory", [1.0, 0.0, 0.0], {"domain": "core", "lang": "python"})
        store.upsert("Python is a language for scripting", [0.0, 1.0, 0.0], {"domain": "language", "lang": "python"})

        hits = store.search("NAVA Python", limit=5, metadata_filter={"domain": "core"})
        assert hits
        assert hits[0]["text"] == "NAVA runs on Python and local memory"

        store.rebuild([
            {"text": "rebuild document", "vector": [0.5, 0.5, 0.0], "metadata": {"domain": "rebuild"}},
        ])
        assert store.count() == 1
        assert store.search("document", metadata_filter={"domain": "rebuild"})[0]["text"] == "rebuild document"


def test_retrieval_metrics_measure_ranked_quality():
    metrics = evaluate_retrieval(["wrong", "answer", "other"], ["answer"], k=3)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["ndcg_at_k"] == 0.6309297535714575


def test_hybrid_retriever_reports_quality_against_relevance_labels():
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteVectorStore(str(Path(directory) / "vectors.db"))
        retriever = HybridRetriever(vector_store=store)
        retriever.add("Python API authentication guide")
        retriever.add("Cooking recipes and ingredients")
        metrics = retriever.evaluate("Python authentication", ["Python API authentication guide"])
        assert metrics["recall_at_k"] == 1.0
        assert metrics["mrr"] == 1.0


def test_jwt_auth_validates_signature_and_expiry():
    auth = JWTAuth("x" * 32, issuer="nava", audience="api")
    token = auth.issue("client", {"role": "user"})
    assert auth.is_allowed({"token": token})
    assert not auth.is_allowed({"token": token[:-1] + ("a" if token[-1] != "a" else "b")})


def test_jwt_auth_rejects_wrong_claims():
    auth = JWTAuth("x" * 32, issuer="nava", audience="api")
    token = JWTAuth("x" * 32, issuer="other", audience="api").issue("client")
    assert not auth.is_allowed({"token": token})


def test_expert_router_ranks_multiple_domains():
    routing = ExpertRouter().route_details("debug Python authentication security error")
    assert routing["expert"] == "coding"
    assert routing["candidates"][0]["score"] >= routing["candidates"][-1]["score"]


def test_reasoner_and_agent_loop_execute_bounded_plan():
    manager = TaskManager(max_steps=3)
    plan = {"steps": ["first", "second"], "max_retries": 0}
    manager.start("reasoning", plan)
    trace = Reasoner().reason("research then summarize", {"intent": "research", "expert": "research", "needs_retrieval": True})
    results = AgentLoop(manager).run("reasoning", {"first": lambda: 1, "second": lambda: 2})
    assert trace["subgoals"] == ["research", "summarize"]
    assert results == {"first": 1, "second": 2}
    assert manager.finish("reasoning")["status"] == "completed"


def test_reasoner_uses_evidence_and_routing_confidence():
    trace = Reasoner().reason(
        "research then summarize",
        {
            "expert": "research",
            "routing": {"confidence": 0.8},
            "needs_retrieval": True,
            "needs_tools": True,
        },
        evidence=["source result"],
    )
    assert trace["strategy"] == "evidence_first"
    assert trace["next_action"] == "generate_response"
    assert trace["evidence_count"] == 1
    assert "Validate tool results before using them." in trace["constraints"]


def test_agent_loop_passes_previous_outputs_to_context_handlers():
    manager = TaskManager()
    manager.start("context", {"steps": ["first", "second"], "max_retries": 0})
    results = AgentLoop(manager).run(
        "context",
        {
            "first": lambda: 3,
            "second": lambda context: context["first"] * 2,
        },
    )
    assert results == {"first": 3, "second": 6}
    assert manager.tasks["context"]["outputs"] == results


def test_agent_loop_exhausts_global_retry_budget():
    manager = TaskManager()
    manager.start("retry", {"steps": ["unstable"], "max_retries": 5, "retry_budget": 1})

    def fail():
        raise ValueError("temporary")

    try:
        AgentLoop(manager).run("retry", {"unstable": fail})
    except RuntimeError as exc:
        assert "retry budget" in str(exc)
    else:
        raise AssertionError("global retry budget was not enforced")
    assert manager.tasks["retry"]["status"] == "failed"


def test_learned_router_uses_examples_without_keyword_rules():
    router = LearnedExpertRouter(min_confidence=0.2).fit([
        ("repair a failing service implementation", "coding"),
        ("investigate an exploit in the login flow", "security"),
    ])
    result = router.route_details("fix the broken service implementation")
    assert result["expert"] == "coding"
    assert result["learned"] is True


def test_expert_model_registry_resolves_enabled_model():
    registry = ExpertModelRegistry()
    registry.register("coding", "brain_system/nava_tuned", adapter_path="adapters/coding")
    assert registry.resolve("coding")["adapter_path"] == "adapters/coding"
    registry.disable("coding")
    assert registry.resolve("coding") is None


def test_expert_manifest_tracks_training_artifact_status():
    registry = ExpertModelRegistry()
    manifest = Path("brain_system/experts/coding/manifest.json")
    model = registry.register_manifest(manifest)
    assert model["expert"] == "coding"
    assert model["status"] == "untrained"
    assert Path(model["training_data"]).is_file()
    assert registry.is_trained("coding") is False


def test_brain_artifacts_are_structurally_valid():
    report = validate_brain_artifacts("brain_system/seed_brain", "brain_system/nava_tuned")
    assert report["valid"] is True
    assert report["model_type"] == "qwen2"
    assert report["lora_rank"] == 16
    assert report["base_model_matches"] is True


def test_memory_sqlite_expiry_and_forgetting_policy():
    with tempfile.TemporaryDirectory() as directory:
        memory = MemoryManager(str(Path(directory) / "memory.db"), max_entries=2)
        memory.add("temporary", "expires immediately", ttl_seconds=-1)
        memory.add("durable", "keep this fact", importance=0.9)
        memory.add("discard", "low value fact", importance=0.1)
        assert all(item["topic"] != "temporary" for item in memory.all())
        assert memory.forget(below_importance=0.5) == 1
        assert memory.retrieve("keep fact")[0]["topic"] == "durable"
        assert memory.snapshot()["storage"] == "sqlite"


def test_policy_engine_blocks_high_risk_without_approval_or_isolation():
    policy = PolicyEngine()
    decision = policy.decide("execute command sudo rm -rf data", operation="tool")
    assert decision["allowed"] is False
    assert decision["level"] == "critical"
    assert decision["requires_isolation"] is True


def test_safety_guard_reviews_output_as_output():
    guard = SafetyGuard()
    assert guard.review("A normal answer")["safe"] is True
    assert guard.snapshot()["events"] >= 1


def test_tool_registry_applies_policy_before_handler():
    permissions = PermissionEngine()
    permissions.allow("runner", True)
    called = []
    registry = ToolRegistry(permission_engine=permissions, sandbox=ToolSandbox(), policy_engine=PolicyEngine())
    registry.register("runner", lambda value: called.append(value) or value, allowed=True)
    result = registry.execute("runner", "sudo delete credentials")
    assert result["ok"] is False
    assert called == []


def test_evaluator_uses_ground_truth_and_rejects_forbidden_content():
    evaluator = Evaluator()
    case = EvaluationCase("capital", "The capital is Paris", required_terms=("Paris",), forbidden_terms=("London",))
    passed = evaluator.score_ground_truth("Paris is the capital.", case)
    failed = evaluator.score_ground_truth("The capital is London.", case)
    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["ground_truth"] is True


def test_continuous_learning_exports_only_ground_truth_approved_examples():
    with tempfile.TemporaryDirectory() as directory:
        archive = ExperienceArchive(str(Path(directory) / "experiences.json"))
        archive.add({"input": "capital?", "output": "Paris", "expected": "Paris"})
        archive.add({"input": "capital?", "output": "London", "expected": "Paris"})
        queue = Path(directory) / "queue.json"
        result = archive.continuous_learning_cycle(Evaluator(), output_path=str(queue))
        assert result["reviewed"] == 2
        assert result["approved"] == 1
        assert result["export"]["exported"] == 1


def test_training_pipeline_never_merges_a_failed_candidate():
    merged = []
    pipeline = TrainingPipeline(threshold=0.8)
    result = pipeline.run(
        train=lambda: {"version": "candidate"},
        cases=[EvaluationCase("capital", "Paris", required_terms=("Paris",))],
        generate=lambda artifact, query: "London",
        merge=lambda artifact, report: merged.append(artifact),
    )
    assert result["status"] == "rejected"
    assert "quality_gate_failed" in result["stages"]
    assert merged == []


def test_training_pipeline_merges_only_after_quality_gate():
    merged = []
    pipeline = TrainingPipeline(threshold=0.8)
    result = pipeline.run(
        train=lambda: {"version": "candidate"},
        cases=[EvaluationCase("capital", "Paris", required_terms=("Paris",))],
        generate=lambda artifact, query: "Paris",
        merge=lambda artifact, report: merged.append(report["quality_gate"]) or "activated",
    )
    assert result["status"] == "merged"
    assert result["merge_result"] == "activated"
    assert merged[0]["passed"] is True


def test_default_benchmark_covers_required_capability_categories():
    cases = BenchmarkRunner.default_cases()
    categories = {case.category for case in cases}
    assert len(cases) >= 20
    assert {"safety", "retrieval", "knowledge", "versioning"} <= categories


def test_version_activation_rejects_tampered_artifact():
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "artifact"
        source.mkdir()
        (source / "config.json").write_text("{}", encoding="utf-8")
        store = ModelVersionStore(str(Path(directory) / "versions"))
        staged = store.stage(str(source), "v1")
        registry = VersionRegistry(str(Path(directory) / "registry.json"))
        registry.register("v1", 1.0, benchmark_report={"quality_gate": {"passed": True}}, artifact_path=staged, artifact_hash="tampered")
        try:
            registry.activate("v1", target_path=str(Path(directory) / "active"))
        except ValueError as exc:
            assert "hash" in str(exc)
        else:
            raise AssertionError("tampered artifact was activated")
