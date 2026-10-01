"""Expert-brain registry for specialized reasoning domains."""

from __future__ import annotations

from .coding import CodingExpert
from .general import GeneralExpert
from .mathematics import MathematicsExpert
from .research import ResearchExpert
from .security import SecurityExpert

EXPERTS = {
    "coding": CodingExpert,
    "general": GeneralExpert,
    "mathematics": MathematicsExpert,
    "research": ResearchExpert,
    "security": SecurityExpert,
}


def get_expert(domain: str, *args, **kwargs):
    """Instantiate a specialist expert by domain name."""
    key = domain.lower()
    try:
        expert_cls = EXPERTS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown expert domain: {domain}") from exc
    return expert_cls(*args, **kwargs)


__all__ = [
    "CodingExpert",
    "GeneralExpert",
    "MathematicsExpert",
    "ResearchExpert",
    "SecurityExpert",
    "EXPERTS",
    "get_expert",
]
