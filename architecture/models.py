from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import uuid

from workflow import Status  # re-export as the canonical Status


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class CodeSpan:
    file: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    symbol: Optional[str] = None


@dataclass
class Evidence:
    id: str
    source: str
    summary: str
    locations: List[CodeSpan] = field(default_factory=list)
    strength: float = 0.5
    raw_refs: Dict[str, Any] = field(default_factory=dict)
    witness: Optional[str] = None


@dataclass
class SustainingCondition:
    id: str
    text: str
    status: Status = Status.UNKNOWN
    evidence: List[Evidence] = field(default_factory=list)
    children: List["SustainingCondition"] = field(default_factory=list)
    discovered_by_agent: Optional[str] = None
    parent_id: Optional[str] = None
    depth: int = 0
    plan_params: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class ValidationInput:
    condition: SustainingCondition
    context: Dict[str, Any]


@dataclass
class ValidationResult:
    status: Status
    children: List[SustainingCondition]
    evidence: List[Evidence]
    notes: List[str]


@dataclass
class Finding:
    id: str
    origin_file: str
    claim: str
    severity: Optional[str] = None
    evidence: List[Evidence] = field(default_factory=list)
    root_conditions: List[SustainingCondition] = field(default_factory=list)
    invalidated: bool = False
    invalidation_reason: Optional[str] = None
    agent_id: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class AgentReport:
    start_file: str
    findings: List[Finding]
    logs: List[str] = field(default_factory=list)
    agent_id: Optional[str] = None
    duration_s: float = 0.0


@dataclass
class AuditReport:
    findings: List[Finding]
    started_at: float
    finished_at: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "findings": [asdict(f) for f in self.findings],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "meta": self.meta,
        }


def render_report(rep: AuditReport) -> str:
    def badge(s: Status) -> str:
        return {Status.SATISFIED: "✅", Status.VIOLATED: "❌", Status.UNKNOWN: "❔"}[s]

    lines: List[str] = []
    for f in rep.findings:
        head = "INVALID" if f.invalidated else "VALID?"
        reason = f.invalidation_reason or ""
        lines.append(f"[{head}] {f.claim}  ({f.origin_file}) {reason}")

        for rc in f.root_conditions:
            def walk(c: SustainingCondition, indent: int = 2) -> None:
                lines.append(" " * indent + f"{badge(c.status)} {c.text}")
                for ev in c.evidence:
                    lines.append(" " * (indent + 2) + f"- {ev.summary}")
                for note in c.notes:
                    lines.append(" " * (indent + 2) + f"# {note}")
                for ch in c.children:
                    walk(ch, indent + 2)

            walk(rc)

    return "\n".join(lines)
