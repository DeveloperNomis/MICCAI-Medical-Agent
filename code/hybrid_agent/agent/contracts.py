# contracts.py

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RelationQuery:
    """
    Structured representation of a spatial relation query.
    """
    entity_a: str
    relation: str
    entity_b: str


@dataclass
class AgentResult:
    """
    Standard output object returned by the hybrid agent.
    """
    query: Optional[RelationQuery]
    result: bool
    explanation: Optional[str]
    detections: Any
    confidence: float
    audit_log: Optional[dict]
    warnings: Optional[list[str]] = None