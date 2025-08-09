from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import List, Dict, Optional, Any, Tuple, Set
from collections import defaultdict
import asyncio
import time
import uuid


# =============== Domain Model ===============

class Status(Enum):
    UNKNOWN = auto()
    SATISFIED = auto()
    VIOLATED = auto()


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
    plan_kind: Optional[str] = None
    plan_params: Dict[str, Any] = field(default_factory=dict)


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
                for ch in c.children:
                    walk(ch, indent + 2)

            walk(rc)

    return "\n".join(lines)


# =============== Adapters (IO / Tools) ===============

class CodebaseAdapter:
    """Minimal stub used for simulation."""

    def __init__(self, root: str):
        self.root = root

    async def read_file(self, path: str) -> str:
        return ""

    async def search(self, pattern: str, *, limit: int = 100) -> List[CodeSpan]:
        return []

    async def imports_of(self, path: str, *, limit: int = 50) -> List[str]:
        return []

    async def call_sites(self, symbol: str, *, limit: int = 50) -> List[CodeSpan]:
        return []


class WebSearchAdapter:
    async def search(self, query: str, *, limit: int = 5) -> List[Dict[str, Any]]:
        return []


# =============== LLM Stub ===============

class LLMClient:
    async def complete(self, *, system: str, messages: List[Dict[str, str]],
                       json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {}


# =============== Agents ===============

class TinyShellAgent:
    """Simulated agent producing a hard-coded finding."""

    def __init__(self, agent_id: str, llm: LLMClient, code: CodebaseAdapter,
                 traversal_depth: int = 2, traversal_width: int = 20):
        self.agent_id = agent_id
        self.llm = llm
        self.code = code
        self.traversal_depth = traversal_depth
        self.traversal_width = traversal_width

    async def run(self, start_file: str, user_goal: Optional[str]) -> AgentReport:
        t0 = time.time()
        logs: List[str] = ["simulated scouting"]
        finding = Finding(
            id=_id("finding"),
            origin_file=start_file,
            claim=f"Simulated claim for {start_file}",
            severity="Low",
            evidence=[],
            root_conditions=[
                SustainingCondition(
                    id=_id("cond"),
                    text="input is sanitized",
                    plan_kind="IS_USER_CONTROLLED",
                    plan_params={"symbol": "request.args"},
                ),
                SustainingCondition(
                    id=_id("cond"),
                    text="path reaches sink",
                    plan_kind="PATH_EXISTS",
                    plan_params={"sink": "util.eval"},
                ),
            ],
            agent_id=self.agent_id,
        )
        return AgentReport(
            start_file=start_file,
            findings=[finding],
            logs=logs,
            agent_id=self.agent_id,
            duration_s=time.time() - t0,
        )


class ValidationAgent:
    """Simple validator that expands one condition into two sub-conditions."""

    def __init__(self, agent_id: str, llm: LLMClient, code: CodebaseAdapter,
                 web: Optional[WebSearchAdapter] = None):
        self.agent_id = agent_id
        self.llm = llm
        self.code = code
        self.web = web

    async def validate(self, cond: SustainingCondition, context: Dict[str, Any]) -> SustainingCondition:
        updated = SustainingCondition(
            id=cond.id,
            text=cond.text,
            plan_kind=cond.plan_kind,
            plan_params=cond.plan_params,
        )
        if cond.plan_kind == "IS_USER_CONTROLLED":
            updated.status = Status.SATISFIED
            updated.evidence.append(
                Evidence(
                    id=_id("ev"),
                    source="simulation",
                    summary=f"{cond.plan_params.get('symbol', 'input')} appears user-controlled",
                    witness="example: request.args['q']",
                )
            )
        elif cond.plan_kind == "PATH_EXISTS":
            if cond.text == "path reaches sink":
                updated.children = [
                    SustainingCondition(
                        id=_id("cond"),
                        text="source reachable",
                        plan_kind="PATH_EXISTS",
                        plan_params={"sink": cond.plan_params.get("sink")},
                    ),
                    SustainingCondition(
                        id=_id("cond"),
                        text="no filter",
                        plan_kind="IS_AUTH_GUARDED",
                        plan_params={"guard": "sanitize"},
                    ),
                ]
            else:
                updated.status = Status.SATISFIED
                updated.evidence.append(
                    Evidence(
                        id=_id("ev"),
                        source="simulation",
                        summary="path found",
                        witness="example path",
                    )
                )
        elif cond.plan_kind == "IS_AUTH_GUARDED":
            updated.status = Status.VIOLATED
            updated.evidence.append(
                Evidence(
                    id=_id("ev"),
                    source="simulation",
                    summary=f"{cond.plan_params.get('guard', 'guard')} missing",
                    witness=f"no {cond.plan_params.get('guard', 'guard')}",
                )
            )
        elif cond.text == "input is sanitized":
            updated.status = Status.SATISFIED
            updated.evidence.append(
                Evidence(id=_id("ev"), source="simulation", summary="inputs are sanitized")
            )
        elif cond.text == "path reaches sink":
            updated.children = [
                SustainingCondition(id=_id("cond"), text="source reachable"),
                SustainingCondition(id=_id("cond"), text="no filter"),
            ]
        elif cond.text == "source reachable":
            updated.status = Status.SATISFIED
            updated.evidence.append(
                Evidence(id=_id("ev"), source="simulation", summary="path found")
            )
        elif cond.text == "no filter":
            updated.status = Status.VIOLATED
            updated.evidence.append(
                Evidence(id=_id("ev"), source="simulation", summary="filter present")
            )
        else:
            updated.status = Status.SATISFIED
        return updated


# =============== Orchestrator ===============

@dataclass
class OrchestratorConfig:
    max_depth: int = 4
    max_tasks: int = 500
    per_finding_max_tasks: int = 100
    concurrent_validations: int = 16
    early_invalidate_on_first_violation: bool = True
    dedup_normalize_text: bool = True


@dataclass
class Task:
    id: str
    kind: str
    payload: Dict[str, Any]
    parent_id: Optional[str]
    depth: int
    priority: int = 0


class AuditOrchestrator:
    def __init__(self, code: CodebaseAdapter, llm: LLMClient,
                 web: Optional[WebSearchAdapter] = None, config: Optional[OrchestratorConfig] = None):
        self.code = code
        self.llm = llm
        self.web = web
        self.cfg = config or OrchestratorConfig()
        self.validation_agent = ValidationAgent(agent_id="validator", llm=llm, code=code, web=web)
        self._condition_index: Dict[str, str] = {}
        self._condition_store: Dict[str, SustainingCondition] = {}
        self._per_finding_tasks = defaultdict(int)

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(text.lower().split())

    def _index_condition(self, cond: SustainingCondition) -> SustainingCondition:
        key = self._norm(cond.text) if self.cfg.dedup_normalize_text else cond.text
        if key in self._condition_index:
            existing_id = self._condition_index[key]
            return self._condition_store[existing_id]
        self._condition_index[key] = cond.id
        self._condition_store[cond.id] = cond
        return cond

    async def _scout(self, start_file: str, user_goal: Optional[str]) -> AgentReport:
        agent = TinyShellAgent(agent_id=f"scout_{uuid.uuid4().hex[:6]}", llm=self.llm, code=self.code)
        return await agent.run(start_file=start_file, user_goal=user_goal)

    async def run(self, start_files: List[str], user_goal: Optional[str] = None) -> AuditReport:
        t0 = time.time()
        scout_reports = await asyncio.gather(*(self._scout(sf, user_goal) for sf in start_files))
        findings: List[Finding] = []
        for r in scout_reports:
            findings.extend(r.findings)
        self._per_finding_tasks.clear()
        task_queue: asyncio.PriorityQueue[Tuple[int, str, Task]] = asyncio.PriorityQueue()
        tasks_created = 0

        def enqueue(f: Finding, task: Task) -> None:
            nonlocal tasks_created
            if tasks_created >= self.cfg.max_tasks:
                return
            if self._per_finding_tasks[f.id] >= self.cfg.per_finding_max_tasks:
                f.invalidated = True
                f.invalidation_reason = "Budget exceeded"
                return
            self._per_finding_tasks[f.id] += 1
            task_queue.put_nowait((task.priority, task.id, task))
            tasks_created += 1

        for f in findings:
            for rc in f.root_conditions:
                rc.parent_id = None
                rc.depth = 0
                rc.discovered_by_agent = f.agent_id
                node = self._index_condition(rc)
                task = Task(
                    id=_id("task"),
                    kind="validate_condition",
                    payload={"finding_id": f.id, "condition_id": node.id},
                    parent_id=None,
                    depth=0,
                    priority=0,
                )
                enqueue(f, task)
        self._enqueue_task = enqueue
        in_flight: Set[str] = set()
        sem = asyncio.Semaphore(self.cfg.concurrent_validations)

        async def worker():
            nonlocal tasks_created
            while not task_queue.empty():
                _, _, task = await task_queue.get()
                if task.depth > self.cfg.max_depth or tasks_created > self.cfg.max_tasks:
                    task_queue.task_done()
                    continue
                async with sem:
                    in_flight.add(task.id)
                    try:
                        await self._handle_task(task, findings, task_queue)
                    finally:
                        in_flight.discard(task.id)
                        task_queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(self.cfg.concurrent_validations)]
        await task_queue.join()
        for w in workers:
            w.cancel()
        t1 = time.time()
        return AuditReport(findings=findings, started_at=t0, finished_at=t1,
                           meta={"tasks_created": tasks_created})

    async def _handle_task(self, task: Task, findings: List[Finding],
                           task_queue: asyncio.PriorityQueue):
        if task.kind == "validate_condition":
            await self._handle_validate(task, findings, task_queue)

    async def _handle_validate(self, task: Task, findings: List[Finding],
                               task_queue: asyncio.PriorityQueue):
        cond_id = task.payload["condition_id"]
        finding_id = task.payload["finding_id"]
        cond = self._condition_store[cond_id]
        finding = next((f for f in findings if f.id == finding_id), None)
        if finding is None:
            return
        if finding.invalidated and self.cfg.early_invalidate_on_first_violation:
            return
        context = {
            "finding_claim": finding.claim,
            "finding_evidence": [asdict(e) for e in finding.evidence],
            "siblings_status": [c.status.name for c in finding.root_conditions if c.id != cond.id],
        }
        updated = await self.validation_agent.validate(cond, context)
        cond.evidence.extend(updated.evidence)
        cond.status = updated.status
        if cond.status == Status.VIOLATED and self.cfg.early_invalidate_on_first_violation:
            finding.invalidated = True
            finding.invalidation_reason = f"Condition failed: {cond.text}"
            return
        for child in updated.children:
            child.parent_id = cond.id
            child.depth = cond.depth + 1
            node = self._index_condition(child)
            if node.id not in [c.id for c in cond.children]:
                cond.children.append(node)
            if node.status == Status.UNKNOWN:
                new_task = Task(
                    id=_id("task"),
                    kind="validate_condition",
                    payload={"finding_id": finding.id, "condition_id": node.id},
                    parent_id=task.id,
                    depth=task.depth + 1,
                    priority=task.priority + 1,
                )
                self._enqueue_task(finding, new_task)


# =============== Demo ===============

async def main() -> None:
    code = CodebaseAdapter(root=".")
    llm = LLMClient()
    orch = AuditOrchestrator(code=code, llm=llm)
    report = await orch.run(start_files=["start.py"])
    print(render_report(report))


if __name__ == "__main__":
    asyncio.run(main())
