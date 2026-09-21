"""Independent, reproducible evaluation with optional ground truth."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[\w\u0621-\u064A]+", str(value or "").lower()))


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    expected: str
    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()


class Evaluator:
    """Scores responses against explicit ground truth, not only length."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def score_ground_truth(self, answer: str, case: EvaluationCase) -> dict[str, Any]:
        answer_tokens = _tokens(answer)
        expected_tokens = _tokens(case.expected)
        overlap = len(answer_tokens & expected_tokens) / max(1, len(expected_tokens))
        required = _tokens(" ".join(case.required_terms))
        required_score = len(answer_tokens & required) / len(required) if required else 1.0
        forbidden_hits = answer_tokens & _tokens(" ".join(case.forbidden_terms))
        score = round((overlap * 0.65) + (required_score * 0.35), 3)
        if forbidden_hits:
            score = 0.0
        return {
            "score": score,
            "threshold": self.threshold,
            "passed": score >= self.threshold and not forbidden_hits,
            "ground_truth": True,
            "overlap": round(overlap, 3),
            "required_terms": sorted(required),
            "forbidden_hits": sorted(forbidden_hits),
        }

    def score(self, answer: str, query: str = "", evidence: Iterable[Any] | None = None, *, expected: str | None = None, required_terms: Iterable[str] = (), forbidden_terms: Iterable[str] = ()) -> dict[str, Any]:
        if expected is not None:
            return self.score_ground_truth(
                answer,
                EvaluationCase(query, expected, tuple(required_terms), tuple(forbidden_terms)),
            )
        answer_tokens = _tokens(answer)
        query_tokens = _tokens(query)
        relevance = len(answer_tokens & query_tokens) / max(1, len(query_tokens)) if query_tokens else 0.0
        evidence_score = min(1.0, len(list(evidence or [])) / 2.0)
        score = round((min(1.0, len(str(answer or "").strip()) / 120.0) * 0.35) + (relevance * 0.4) + (evidence_score * 0.25), 3)
        return {"score": score, "threshold": self.threshold, "passed": score >= self.threshold, "ground_truth": False}

    def evaluate_dataset(self, cases: Iterable[EvaluationCase], answers: Iterable[str]) -> dict[str, Any]:
        results = [self.score_ground_truth(answer, case) for case, answer in zip(cases, answers)]
        return {
            "cases": len(results),
            "passed": sum(item["passed"] for item in results),
            "accuracy": sum(item["passed"] for item in results) / len(results) if results else 0.0,
            "mean_score": sum(item["score"] for item in results) / len(results) if results else 0.0,
            "results": results,
        }

    def critique(self, answer: str, query: str = "", evidence=None, **kwargs: Any):
        result = self.score(answer, query, evidence, **kwargs)
        return {**result, "reason": "passed" if result["passed"] else "needs_repair"}
