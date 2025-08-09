from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import asyncio
from asyncio import CancelledError
import time
import uuid
import sys
from pathlib import Path

# Ensure the project root is on the import path when running this module as a script.
sys.path.append(str(Path(__file__).resolve().parent.parent))
if __package__ in (None, ""):
    __package__ = "architecture"

from persistence import RunStore, BlobStore, RunEvent, EventType, DerivedState

from .adapters import CodebaseAdapter, WebSearchAdapter, LLMClient
from .agents import ShellLLMAgent, TinyShellAgent
from .models import (
    SustainingCondition,
    Finding,
    AgentReport,
    AuditReport,
    Evidence,
    CodeSpan,
    Status,
    _id,
    render_report,
)
from .llm import LLM_FAILURES, VALIDATION_OUTPUT_SCHEMA
from workflow import compute_digest


@dataclass
class OrchestratorConfig:
    max_depth: int = 4
    max_tasks: int = 500
    per_finding_max_tasks: int = 100
    concurrent_validations: int = 16
    early_invalidate_on_first_violation: bool = True
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
    def __init__(
        self,
        code: CodebaseAdapter,
        llm: LLMClient,
        web: Optional[WebSearchAdapter] = None,
        config: Optional[OrchestratorConfig] = None,
        run_store: Optional[RunStore] = None,
        blob_store: Optional[BlobStore] = None,
    ):
        self.code = code
        self.llm = llm
        self.web = web
        self.cfg = config or OrchestratorConfig()
        self.agent = ShellLLMAgent(llm)
        self._condition_index: Dict[str, str] = {}  # digest -> cond.id
        self._condition_store: Dict[str, SustainingCondition] = {}
        self._per_finding_tasks = defaultdict(int)
        self.run_store = run_store or RunStore()
        self.blob_store = blob_store or BlobStore()

    def _index_condition(self, cond: SustainingCondition) -> SustainingCondition:
        digest = compute_digest(cond.text, cond.plan_params)
        if digest in self._condition_index:
            existing_id = self._condition_index[digest]
            return self._condition_store[existing_id]
        self._condition_index[digest] = cond.id
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
                self.run_store.append(
                    RunEvent(
                        ts=time.time(),
                        run_id=self.run_id,
                        type=EventType.BUDGET_HIT,
                        data={"task_id": task.id, "finding_id": f.id, "kind": "total"},
                    )
                )
                return
            if self._per_finding_tasks[f.id] >= self.cfg.per_finding_max_tasks:
                self.run_store.append(
                    RunEvent(
                        ts=time.time(),
                        run_id=self.run_id,
                        type=EventType.BUDGET_HIT,
                        data={"task_id": task.id, "finding_id": f.id, "kind": "per_finding"},
                    )
                )
                f.invalidated = True
                f.invalidation_reason = "Budget exceeded"
                return
            self._per_finding_tasks[f.id] += 1
            task_queue.put_nowait((task.priority, task.id, task))
            tasks_created += 1
            cond_id = task.payload.get("condition_id")
            cond = self._condition_store.get(cond_id)
            cond_data = (
                {
                    "id": cond.id,
                    "text": cond.text,
                    "status": cond.status.name,
                    "parent_id": cond.parent_id,
                    "depth": cond.depth,
                    "plan_params": cond.plan_params,
                }
                if cond
                else {}
            )
            self.run_store.append(
                RunEvent(
                    ts=time.time(),
                    run_id=self.run_id,
                    type=EventType.TASK_ENQUEUED,
                    data={"task": asdict(task), "finding_id": f.id, "condition": cond_data},
                )
            )

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
                self.run_store.append(
                    RunEvent(
                        ts=time.time(),
                        run_id=self.run_id,
                        type=EventType.TASK_STARTED,
                        data={
                            "task_id": task.id,
                            "finding_id": task.payload.get("finding_id"),
                            "condition_id": task.payload.get("condition_id"),
                        },
                    )
                )
                try:
                    async with sem:
                        in_flight.add(task.id)
                        usage = await self._handle_task(task, findings, task_queue)
                except CancelledError:
                    raise
                except Exception as e:
                    usage = None
                    self.run_store.append(
                        RunEvent(
                            ts=time.time(),
                            run_id=self.run_id,
                            type=EventType.NODE_STATUS,
                            data={
                                "task_id": task.id,
                                "finding_id": task.payload.get("finding_id"),
                                "condition_id": task.payload.get("condition_id"),
                                "status": "ERROR",
                                "error": repr(e),
                            },
                        )
                    )
                finally:
                    in_flight.discard(task.id)
                    self.run_store.append(
                        RunEvent(
                            ts=time.time(),
                            run_id=self.run_id,
                            type=EventType.TASK_FINISHED,
                            data={
                                "task_id": task.id,
                                "finding_id": task.payload.get("finding_id"),
                                "condition_id": task.payload.get("condition_id"),
                                **(usage or {}),
                            },
                        )
                    )
                    task_queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(self.cfg.concurrent_validations)]
        await task_queue.join()
        for w in workers:
            w.cancel()
        t1 = time.time()
        return AuditReport(findings=findings, started_at=t0, finished_at=t1,
                           meta={"tasks_created": tasks_created, "llm_failures": LLM_FAILURES})

    def resume(self, run_id: str) -> DerivedState:
        """Load events for ``run_id`` and return derived state.

        This provides insight into pending tasks and counts to support
        resuming or auditing a prior run.
        """
        return self.run_store.latest_state(run_id)

    async def _handle_task(
        self, task: Task, findings: List[Finding], task_queue: asyncio.PriorityQueue
    ):
        if task.kind == "validate_condition":
            return await self._handle_validate(task, findings, task_queue)
        return None

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
        }
        out = await self.agent.run_task(
            objective=cond.text,
            context=context,
            constraints={"max_children": self.cfg.max_children_allowed},
            schema=VALIDATION_OUTPUT_SCHEMA,
        )
        status = Status[out.get("status", "UNKNOWN")]
        notes = out.get("notes", [])
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
            for ev in out.get("evidence", [])
        ]
        children = [
            SustainingCondition(
                id=_id("cond"),
                text=ch["text"],
                plan_params=ch.get("plan_params", {}),
            )
            for ch in out.get("children", [])
        ]
        cond.evidence.extend(evidence)
        cond.status = status
        cond.notes.extend(notes)
        self.run_store.append(
            RunEvent(
                ts=time.time(),
                run_id=self.run_id,
                type=EventType.NODE_STATUS,
                data={
                    "task_id": task.id,
                    "finding_id": finding.id,
                    "condition_id": cond.id,
                    "status": cond.status.name,
                },
            )
        )
        if cond.status == Status.VIOLATED and self.cfg.early_invalidate_on_first_violation:
            finding.invalidated = True
            finding.invalidation_reason = f"Condition failed: {cond.text}"
            return
        for child in children[: self.cfg.max_children_allowed]:
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
        return

# =============== Demo ===============

async def main() -> None:
    code = CodebaseAdapter(root=".")
    llm = LLMClient()
    orch = AuditOrchestrator(code=code, llm=llm)
    report = await orch.run(start_files=["start.py"])
    print(render_report(report))


if __name__ == "__main__":
    asyncio.run(main())
