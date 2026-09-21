"""Learning System — experience capture and dataset preparation."""


import json
import os
import uuid
from datetime import datetime, timezone


class ExperienceArchive:
    """Stores successful and failed experiences for future training."""

    def __init__(self, path=None):
        self.experiences = []
        self.path = path or os.path.join(os.getcwd(), "data", "experience_archive.json")
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                self.experiences = data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            self.experiences = []

    def add(self, example):
        example = {**example, "id": example.get("id", uuid.uuid4().hex), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")} if isinstance(example, dict) else {"id": uuid.uuid4().hex, "value": example}
        self.experiences.append(example)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.experiences, handle, ensure_ascii=False, indent=2)
        return example

    def review(self, experience_id, *, approved: bool, reviewer: str, feedback: str = ""):
        for item in self.experiences:
            if item.get("id") == experience_id:
                item["review"] = {
                    "approved": bool(approved),
                    "reviewer": reviewer,
                    "feedback": feedback,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                self._save()
                return item
        raise KeyError(f"Experience '{experience_id}' was not found.")

    def build_training_candidates(self, minimum_score=0.7):
        return [
            item for item in self.experiences
            if item.get("score", 0) >= minimum_score
            and item.get("input")
            and item.get("output")
            and item.get("review", {}).get("approved") is True
        ]

    def continuous_learning_cycle(self, evaluator, *, minimum_score=0.8, output_path=None, reviewer="automated"):
        """Evaluate unprocessed experiences and export only approved examples."""
        reviewed = 0
        approved = 0
        for item in self.experiences:
            if item.get("evaluation_status") == "processed":
                continue
            expected = item.get("expected") or item.get("ground_truth")
            if not expected or not item.get("input") or not item.get("output"):
                item["evaluation_status"] = "skipped"
                continue
            result = evaluator.score(item["output"], item["input"], item.get("evidence"), expected=expected)
            item["score"] = result["score"]
            item["evaluation"] = result
            item["evaluation_status"] = "processed"
            self.review(item["id"], approved=result["score"] >= minimum_score, reviewer=reviewer, feedback="ground_truth_evaluation")
            reviewed += 1
            approved += int(result["score"] >= minimum_score)
        self._save()
        export = self.export_training_queue(output_path, minimum_score) if output_path else None
        return {"reviewed": reviewed, "approved": approved, "export": export}

    def analyze_errors(self):
        failed = [item for item in self.experiences if item.get("score", 0) < 0.7]
        return {
            "total": len(self.experiences),
            "failed": len(failed),
            "failure_rate": len(failed) / len(self.experiences) if self.experiences else 0.0,
            "common_experts": sorted({item.get("expert", "unknown") for item in failed}),
        }

    def export_training_queue(self, output_path, minimum_score=0.7):
        candidates = self.build_training_candidates(minimum_score)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(
                [{"instruction": item["input"], "output": item["output"], "source": "experience_archive"} for item in candidates],
                handle,
                ensure_ascii=False,
                indent=2,
            )
        return {"exported": len(candidates), "path": output_path}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        temporary = f"{self.path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.experiences, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, self.path)
