"""Automated training-to-promotion pipeline with an explicit quality gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from benchmark_system import BenchmarkCase, BenchmarkRunner
from evaluation_system import EvaluationCase, Evaluator


@dataclass
class PipelineResult:
    status: str
    stages: list[str] = field(default_factory=list)
    benchmark_report: dict[str, Any] = field(default_factory=dict)
    artifact: Any = None
    merge_result: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stages": list(self.stages),
            "benchmark_report": self.benchmark_report,
            "artifact": self.artifact,
            "merge_result": self.merge_result,
        }


class TrainingPipeline:
    """Runs training, independent evaluation, gate validation and promotion.

    The merge callback is never called unless every benchmark requirement passes.
    """

    def __init__(self, *, evaluator: Evaluator | None = None, threshold: float = 0.8, minimum_case_score: float = 0.0):
        self.evaluator = evaluator or Evaluator(threshold=threshold)
        self.threshold = threshold
        self.minimum_case_score = minimum_case_score

    def run(
        self,
        *,
        train: Callable[[], Any],
        cases: Iterable[EvaluationCase | Mapping[str, str]],
        generate: Callable[[Any, str], str],
        merge: Callable[[Any, dict[str, Any]], Any],
    ) -> dict[str, Any]:
        result = PipelineResult(status="started")
        artifact = train()
        result.artifact = artifact
        result.stages.append("trained")

        runner = BenchmarkRunner()
        for raw_case in cases:
            case = raw_case if isinstance(raw_case, EvaluationCase) else EvaluationCase(
                raw_case["query"], raw_case["expected"],
                tuple(raw_case.get("required_terms", ())),
                tuple(raw_case.get("forbidden_terms", ())),
            )
            answer = generate(artifact, case.query)
            evaluation = self.evaluator.score_ground_truth(answer, case)
            runner.run(case.query, evaluation, category="evaluation")
        report = runner.report(self.threshold)
        report["quality_gate"] = runner.quality_gate(
            self.threshold,
            minimum_case_score=self.minimum_case_score,
        )
        result.benchmark_report = report
        result.stages.append("evaluated")
        if not report["quality_gate"]["passed"]:
            result.status = "rejected"
            result.stages.append("quality_gate_failed")
            return result.as_dict()

        result.stages.append("quality_gate_passed")
        result.merge_result = merge(artifact, report)
        result.status = "merged"
        result.stages.append("merged")
        return result.as_dict()
