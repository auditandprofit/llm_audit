from __future__ import annotations

from typing import Dict, Optional, Protocol
import json
import jsonschema
import asyncio

from .models import (
    PlanKind,
    ValidationInput,
    ValidationResult,
    SustainingCondition,
    Evidence,
    CodeSpan,
    Status,
    _id,
)
from .adapters import LLMClient
from .llm import VALIDATION_OUTPUT_SCHEMA


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
        child = SustainingCondition(
            id=_id("cond"),
            text="trace from A to sink X",
            plan_kind=None,
            plan_params={}
        )
        return ValidationResult(
            status=Status.UNKNOWN,
            children=[child],
            evidence=[],
            notes=["needs call graph"],
        )


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
        ev = Evidence(
            id=_id("ev"),
            source="analysis",
            summary="user input sanitized",
            locations=[],
            strength=0.5,
        )
        return ValidationResult(status=Status.SATISFIED, children=[], evidence=[ev], notes=[])
