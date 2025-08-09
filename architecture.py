from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import List, Dict, Optional, Any, Tuple, Set, Protocol
from collections import defaultdict
import asyncio
import time
import uuid
import json
import jsonschema
import re


# =============== Prompt Packet & Schemas ===============

PROMPT_PACKET = """\
SYSTEM:
You are {role}. Follow the RULES strictly.

RULES:
- Think privately; do not include analysis in output.
- Return JSON that matches OUTPUT_SCHEMA exactly.
- If information is missing, set fields to null and add an item to `notes`.
- Never invent file paths or symbols; use ones provided in CONTEXT.
- If you need sub-steps, propose them in `children`, do not execute them.

TASK_INTENT: {intent}

IDENTIFIERS:
run_id: {run_id}
finding_id: {finding_id}
condition_id: {condition_id}
parent_condition_id: {parent_condition_id}

CONTEXT:
{context_block}

OUTPUT_SCHEMA (JSON Schema):
{json_schema}

OUTPUT_EXAMPLE:
{few_shot_example}

RESPONSE:
Return ONLY the JSON object. No prose, no backticks.
"""


VALIDATION_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["status", "evidence", "children", "notes"],
    "properties": {
        "status": {"type": "string", "enum": ["SATISFIED", "VIOLATED", "UNKNOWN"]},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["summary", "source", "strength"],
                "properties": {
                    "summary": {"type": "string", "minLength": 1},
                    "source": {"type": "string"},
                    "locations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {
                                "file": {"type": "string"},
                                "start_line": {"type": ["integer", "null"]},
                                "end_line": {"type": ["integer", "null"]},
                                "symbol": {"type": ["string", "null"]}
                            }
                        }
                    },
                    "strength": {"type": "number", "minimum": 0, "maximum": 1},
                    "raw_refs": {"type": "object"},
                    "witness": {"type": ["string", "null"]}
                }
            }
        },
        "children": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "plan_kind", "plan_params"],
                "properties": {
                    "text": {"type": "string"},
                    "plan_kind": {"type": "string"},
                    "plan_params": {"type": "object"}
                }
            }
        },
        "notes": {"type": "array", "items": {"type": "string"}}
    }
}


SCOUT_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["findings", "notes"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["claim", "origin_file", "root_conditions"],
                "properties": {
                    "claim": {"type": "string"},
                    "origin_file": {"type": "string"},
                    "root_conditions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["text", "plan_kind", "plan_params"],
                            "properties": {
                                "text": {"type": "string"},
                                "plan_kind": {"type": "string"},
                                "plan_params": {"type": "object"}
                            }
                        }
                    }
                }
            }
        },
        "notes": {"type": "array", "items": {"type": "string"}}
    }
}


LLM_FAILURES: List[str] = []

async def call_llm_with_schema(llm, *, system, messages, schema, max_retries=2,
                               temperature=0, top_p=1, max_tokens=1024):
    for attempt in range(max_retries + 1):
        resp = await llm.complete(system=system, messages=messages, json_schema=schema,
                                  temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        text = resp.get("text") if isinstance(resp, dict) else resp
        m = re.search(r"\{.*\}\s*$", text or "", re.S)
        if not m:
            messages.append({"role": "system", "content": "Your previous reply did not contain a JSON object. Return ONLY JSON."})
            continue
        try:
            obj = json.loads(m.group(0))
            jsonschema.validate(obj, schema)
            return obj
        except Exception as e:
            messages.append({"role": "system", "content": f"Your JSON was invalid ({e}). Fix it to match the schema exactly."})
    LLM_FAILURES.append("exhausted retries")
    messages.append({"role": "system", "content": "LLM call failed after retries"})
    raise ValueError("LLM failed to produce valid JSON after retries")


# =============== Domain Model ===============

class Status(Enum):
    UNKNOWN = auto()
    SATISFIED = auto()
    VIOLATED = auto()


class PlanKind(Enum):
    PATH_EXISTS = "PATH_EXISTS"
    IS_USER_CONTROLLED = "IS_USER_CONTROLLED"


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
                       json_schema: Optional[Dict[str, Any]] = None,
                       temperature: float = 0, top_p: float = 1,
                       max_tokens: int = 1024) -> Dict[str, Any]:
        if json_schema and "findings" in json_schema.get("properties", {}):
            text = json.dumps({
                "findings": [{
                    "claim": "Simulated claim",
                    "origin_file": "start.py",
                    "root_conditions": [
                        {"text": "input is sanitized", "plan_kind": "IS_USER_CONTROLLED", "plan_params": {}},
                        {"text": "path reaches sink", "plan_kind": "PATH_EXISTS", "plan_params": {"sink": "util.eval"}},
                    ],
                }],
                "notes": []
            })
            return {"text": text}
        if json_schema and "status" in json_schema.get("properties", {}):
            text = json.dumps({"status": "UNKNOWN", "evidence": [], "children": [], "notes": ["demo note"]})
            return {"text": text}
        return {"text": "{}"}


# =============== Agents ===============

class ValidationStrategy(Protocol):
    kind: PlanKind
    async def validate(self, inp: ValidationInput) -> ValidationResult:
        ...


class StrategyRegistry:
    def __init__(self) -> None:
        self._by: Dict[PlanKind, ValidationStrategy] = {}

    def register(self, strat: ValidationStrategy) -> None:
        self._by[strat.kind] = strat

    def get(self, kind: PlanKind) -> ValidationStrategy:
        return self._by[kind]


class PathExistsStrategy:
    kind = PlanKind.PATH_EXISTS

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm

    async def validate(self, inp: ValidationInput) -> ValidationResult:
        if self.llm:
            try:
                resp = await self.llm.complete(
                    system="path_exists",
                    messages=[],
                    json_schema=VALIDATION_OUTPUT_SCHEMA,
                )
                data = json.loads(resp.get("text", "{}"))
                jsonschema.validate(data, VALIDATION_OUTPUT_SCHEMA)
            except asyncio.TimeoutError:
                raise
            except Exception as e:
                raise ValueError("LLM returned invalid output") from e
            status = Status[data.get("status", "UNKNOWN")]
            notes = data.get("notes", [])
            children = [
                SustainingCondition(
                    id=_id("cond"),
                    text=ch["text"],
                    plan_kind=ch.get("plan_kind"),
                    plan_params=ch.get("plan_params", {}),
                )
                for ch in data.get("children", [])
            ]
            return ValidationResult(status=status, children=children, evidence=[], notes=notes)
        child = SustainingCondition(id=_id("cond"), text="trace from A to sink X", plan_kind=None, plan_params={})
        return ValidationResult(status=Status.UNKNOWN, children=[child], evidence=[], notes=["needs call graph"])


class IsUserControlledStrategy:
    kind = PlanKind.IS_USER_CONTROLLED

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm

    async def validate(self, inp: ValidationInput) -> ValidationResult:
        if self.llm:
            try:
                resp = await self.llm.complete(
                    system="is_user_controlled",
                    messages=[],
                    json_schema=VALIDATION_OUTPUT_SCHEMA,
                )
                data = json.loads(resp.get("text", "{}"))
                jsonschema.validate(data, VALIDATION_OUTPUT_SCHEMA)
            except asyncio.TimeoutError:
                raise
            except Exception as e:
                raise ValueError("LLM returned invalid output") from e
            status = Status[data.get("status", "UNKNOWN")]
            notes = data.get("notes", [])
            evidence = [
                Evidence(
                    id=_id("ev"),
                    source=ev.get("source", ""),
                    summary=ev.get("summary", ""),
                    locations=[CodeSpan(**loc) for loc in ev.get("locations", [])],
                    strength=ev.get("strength", 0.5),
                    raw_refs=ev.get("raw_refs", {}),
                    witness=ev.get("witness"),
                )
                for ev in data.get("evidence", [])
            ]
            return ValidationResult(status=status, children=[], evidence=evidence, notes=notes)
        ev = Evidence(id=_id("ev"), source="analysis", summary="user input sanitized", locations=[], strength=0.5)
        return ValidationResult(status=Status.SATISFIED, children=[], evidence=[ev], notes=[])


class ValidationAgent:
    def __init__(self, registry: StrategyRegistry):
        self.r = registry

    async def validate(self, cond: SustainingCondition, ctx: Dict[str, Any]) -> ValidationResult:
        try:
            kind = PlanKind(cond.plan_kind) if cond.plan_kind else PlanKind.PATH_EXISTS
        except ValueError as e:
            raise KeyError(f"Unknown plan kind: {cond.plan_kind}") from e
        strat = self.r.get(kind)
        return await strat.validate(ValidationInput(cond, ctx))


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
        context = {"start_file": start_file, "user_goal": user_goal}
        few_shot = json.dumps({
            "findings": [
                {
                    "claim": "Simulated claim",
                    "origin_file": "a.py",
                    "root_conditions": [
                        {"text": "input is sanitized", "plan_kind": "IS_USER_CONTROLLED", "plan_params": {}}
                    ]
                }
            ],
            "notes": []
        }, indent=2)
        packet = PROMPT_PACKET.format(
            role="a codebase scouting agent",
            intent="SCOUT_FINDINGS",
            run_id=_id("run"),
            finding_id="",
            condition_id="",
            parent_condition_id="",
            context_block=json.dumps(context, indent=2),
            json_schema=json.dumps(SCOUT_OUTPUT_SCHEMA, indent=2),
            few_shot_example=few_shot,
        )
        messages = [{"role": "system", "content": packet}]
        out = await call_llm_with_schema(
            self.llm, system="Follow the packet.", messages=messages, schema=SCOUT_OUTPUT_SCHEMA,
            temperature=0, top_p=1, max_tokens=1024
        )
        findings: List[Finding] = []
        for f in out["findings"]:
            fin = Finding(id=_id("finding"), origin_file=f["origin_file"], claim=f["claim"], agent_id=self.agent_id)
            for rc in f.get("root_conditions", []):
                fin.root_conditions.append(
                    SustainingCondition(
                        id=_id("cond"),
                        text=rc["text"],
                        plan_kind=rc.get("plan_kind"),
                        plan_params=rc.get("plan_params", {}),
                    )
                )
            findings.append(fin)
        return AgentReport(
            start_file=start_file,
            findings=findings,
            logs=[],
            agent_id=self.agent_id,
            duration_s=time.time() - t0,
        )


# =============== Orchestrator ===============

@dataclass
class OrchestratorConfig:
    max_depth: int = 4
    max_tasks: int = 500
    per_finding_max_tasks: int = 100
    concurrent_validations: int = 16
    early_invalidate_on_first_violation: bool = True
    dedup_normalize_text: bool = True
    max_children_allowed: int = 3


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
        registry = StrategyRegistry()
        registry.register(PathExistsStrategy(llm=llm))
        registry.register(IsUserControlledStrategy(llm=llm))
        self.validation_agent = ValidationAgent(registry)
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
        self.run_id = _id("run")
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
                           meta={"tasks_created": tasks_created, "llm_failures": LLM_FAILURES})

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
            "run_id": self.run_id,
            "finding_id": finding.id,
            "finding_claim": finding.claim,
            "finding_evidence": [asdict(e) for e in finding.evidence],
            "siblings_status": [c.status.name for c in finding.root_conditions if c.id != cond.id],
            "max_children_allowed": self.cfg.max_children_allowed,
        }
        result = await self.validation_agent.validate(cond, context)
        cond.evidence.extend(result.evidence)
        cond.status = result.status
        cond.notes.extend(result.notes)
        if cond.status == Status.VIOLATED and self.cfg.early_invalidate_on_first_violation:
            finding.invalidated = True
            finding.invalidation_reason = f"Condition failed: {cond.text}"
            return
        for child in result.children[: self.cfg.max_children_allowed]:
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
