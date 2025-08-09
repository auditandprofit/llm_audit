import asyncio
import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from architecture.orchestrator import AuditOrchestrator
from architecture.adapters import CodebaseAdapter, LLMClient
from architecture.models import SustainingCondition, Finding, AgentReport
from architecture.orchestrator import OrchestratorConfig
from architecture.llm import _last_json_obj, call_llm_with_schema
from persistence import RunStore, EventType


def test_last_json_obj_extracts_last_object():
    text = "prose {\"a\":1} noise {\"b\":2}"
    assert _last_json_obj(text) == {"b": 2}


def test_call_llm_with_schema_retries_and_stops():
    class BadLLM:
        async def complete(self, **kwargs):
            return {"text": "no json here"}

    run_store = RunStore(":memory:")
    schema = {"type": "object", "properties": {}, "required": []}
    with pytest.raises(ValueError):
        asyncio.run(
            call_llm_with_schema(
                BadLLM(),
                system="",
                messages=[],
                schema=schema,
                max_retries=1,
                run_store=run_store,
                run_id="r1",
                task_id="t1",
            )
        )
    events = run_store.load("r1")
    retries = [e for e in events if e.type is EventType.LLM_RETRY]
    assert len(retries) == 2 and retries[-1].data["attempt"] == 2


def test_worker_continues_after_task_error():
    code = CodebaseAdapter(root=".")
    llm = LLMClient()
    run_store = RunStore(":memory:")

    orch = AuditOrchestrator(code=code, llm=llm, run_store=run_store, config=OrchestratorConfig(concurrent_validations=1))

    async def fake_scout(start_file, user_goal):
        c1 = SustainingCondition(id="c1", text="a", plan_kind="PATH_EXISTS", plan_params={})
        c2 = SustainingCondition(id="c2", text="b", plan_kind="PATH_EXISTS", plan_params={})
        f = Finding(id="f1", origin_file="f.py", claim="claim", agent_id="agent", root_conditions=[c1, c2])
        return AgentReport(start_file=start_file, findings=[f], agent_id="agent", duration_s=0)

    async def fake_handle(task, findings, queue):
        if task.payload["condition_id"] == "c1":
            raise RuntimeError("boom")
        return {}

    orch._scout = fake_scout  # type: ignore
    orch._handle_task = fake_handle  # type: ignore

    asyncio.run(orch.run(start_files=["x.py"]))
    events = run_store.load(orch.run_id)
    errors = [e for e in events if e.type is EventType.NODE_STATUS and e.data.get("status") == "ERROR"]
    finished = [e for e in events if e.type is EventType.TASK_FINISHED]
    assert len(errors) == 1
    assert len(finished) == 2

