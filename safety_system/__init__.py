"""Safety System — policy validation, input guard and output guard."""


import re
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict


class RiskLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class PolicyEngine:
    """Classifies risk and makes explicit, auditable allow/deny decisions."""

    def __init__(self, *, require_approval_above: RiskLevel = RiskLevel.HIGH):
        self.require_approval_above = require_approval_above
        self.audit_log: list[dict[str, Any]] = []
        self.patterns = {
            RiskLevel.CRITICAL: re.compile(r"\b(?:rm\s+-rf|format\s+c:|credential|private key|exfiltrat|reverse shell)\b", re.I),
            RiskLevel.HIGH: re.compile(r"\b(?:sudo|chmod\s+777|delete|drop table|network|download|upload|execute command)\b", re.I),
            RiskLevel.MEDIUM: re.compile(r"\b(?:write file|read file|tool|script|code)\b", re.I),
        }

    def classify(self, text: str, *, operation: str = "input") -> Dict[str, Any]:
        value = str(text or "")
        risk = RiskLevel.LOW
        for level, pattern in self.patterns.items():
            if pattern.search(value):
                risk = max(risk, level)
        return {"operation": operation, "level": risk.name.lower(), "score": int(risk), "signals": [level.name.lower() for level, pattern in self.patterns.items() if pattern.search(value)]}

    def decide(self, text: str, *, operation: str = "input", approved: bool = False, isolated: bool = False) -> Dict[str, Any]:
        classification = self.classify(text, operation=operation)
        score = classification["score"]
        allowed = score < int(self.require_approval_above) or approved
        if score >= int(RiskLevel.HIGH) and not isolated and not approved:
            allowed = False
        decision = {**classification, "allowed": allowed, "requires_approval": score >= int(self.require_approval_above), "requires_isolation": score >= int(RiskLevel.HIGH)}
        self.audit_log.append({**decision, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        return decision

    def snapshot(self) -> Dict[str, int]:
        return {"events": len(self.audit_log), "denied": sum(1 for item in self.audit_log if not item["allowed"])}


class SafetyGuard:
    """Checks prompts and outputs and records security audit events."""

    def __init__(self, policy: PolicyEngine | None = None):
        self.blocked_tokens = ["ignore previous instructions", "system prompt leak"]
        self.audit_log = []
        self.policy = policy or PolicyEngine()

    def _audit(self, event, allowed, reason):
        record = {"event": event, "allowed": allowed, "reason": reason, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.audit_log.append(record)
        return record

    def validate_input(self, text):
        lowered = (text or "").lower()
        for token in self.blocked_tokens:
            if token in lowered:
                self._audit("input", False, "prompt_injection_pattern")
                return False
        decision = self.policy.decide(text, operation="input")
        self._audit("input", decision["allowed"], decision["level"])
        return decision["allowed"]

    def validate_output(self, text):
        valid = bool(text and text.strip()) and len(text) <= 100000
        decision = self.policy.decide(text, operation="output", approved=True, isolated=True)
        valid = valid and decision["allowed"]
        self._audit("output", valid, "accepted" if valid else "empty_or_oversized_or_risky")
        return valid

    def review(self, text):
        safe = self.validate_output(text)
        return {"safe": safe, "reason": "accepted" if safe else self.audit_log[-1]["reason"], "policy": self.policy.classify(text, operation="output")}

    def snapshot(self):
        return {**{"events": len(self.audit_log), "blocked": sum(1 for item in self.audit_log if not item["allowed"])}, "policy": self.policy.snapshot()}
