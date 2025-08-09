from __future__ import annotations

from typing import Any, Dict, List, Optional
import time
import json

from .strategies import StrategyRegistry
from .models import (
    PlanKind,
    ValidationInput,
    ValidationResult,
    SustainingCondition,
    Finding,
    AgentReport,
    _id,
)
from .adapters import CodebaseAdapter, LLMClient
from .llm import PROMPT_PACKET, SCOUT_OUTPUT_SCHEMA, call_llm_with_schema


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
        try:
            out, _usage = await call_llm_with_schema(
                self.llm,
                system="Follow the packet.",
                messages=messages,
                schema=SCOUT_OUTPUT_SCHEMA,
                temperature=0,
                top_p=1,
                max_tokens=1024,
            )
        except Exception:
            t1 = time.time()
            return AgentReport(start_file=start_file, findings=[], agent_id=self.agent_id, duration_s=t1 - t0)
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
        t1 = time.time()
        return AgentReport(start_file=start_file, findings=findings, agent_id=self.agent_id, duration_s=t1 - t0)
