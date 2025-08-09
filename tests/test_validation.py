import asyncio
import json
import pytest

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from architecture import (
    ValidationInput,
    PlanKind,
    StrategyRegistry,
    PathExistsStrategy,
    IsUserControlledStrategy,
    ValidationAgent,
    Status,
    SustainingCondition,
)


class FakeLLM:
    def __init__(self, mode, payload=None):
        self.mode = mode
        self.payload = payload or {}

    async def complete(self, **kwargs):
        if self.mode == "valid":
            return {"text": json.dumps(self.payload)}
        if self.mode == "malformed":
            return {"text": "not-json"}
        if self.mode == "timeout":
            raise asyncio.TimeoutError()
        return {"text": "{}"}


def test_path_exists_strategy_valid():
    async def run():
        payload = {
            "status": "UNKNOWN",
            "evidence": [],
            "children": [{"text": "c1", "plan_kind": "PATH_EXISTS", "plan_params": {}}],
            "notes": ["ok"],
        }
        strat = PathExistsStrategy(llm=FakeLLM("valid", payload))
        inp = ValidationInput(condition=SustainingCondition(id="c", text="t"), context={})
        res = await strat.validate(inp)
        assert res.status == Status.UNKNOWN
        assert res.children[0].text == "c1"
        assert res.notes == ["ok"]

    asyncio.run(run())


def test_path_exists_strategy_malformed():
    async def run():
        strat = PathExistsStrategy(llm=FakeLLM("malformed"))
        inp = ValidationInput(condition=SustainingCondition(id="c", text="t"), context={})
        with pytest.raises(ValueError):
            await strat.validate(inp)

    asyncio.run(run())


def test_path_exists_strategy_timeout():
    async def run():
        strat = PathExistsStrategy(llm=FakeLLM("timeout"))
        inp = ValidationInput(condition=SustainingCondition(id="c", text="t"), context={})
        with pytest.raises(asyncio.TimeoutError):
            await strat.validate(inp)

    asyncio.run(run())


def test_is_user_controlled_strategy_valid():
    async def run():
        payload = {
            "status": "SATISFIED",
            "evidence": [{"summary": "ok", "source": "src", "strength": 0.5, "locations": []}],
            "children": [],
            "notes": [],
        }
        strat = IsUserControlledStrategy(llm=FakeLLM("valid", payload))
        inp = ValidationInput(condition=SustainingCondition(id="c", text="t"), context={})
        res = await strat.validate(inp)
        assert res.status == Status.SATISFIED
        assert res.evidence[0].summary == "ok"

    asyncio.run(run())


def test_is_user_controlled_strategy_malformed():
    async def run():
        strat = IsUserControlledStrategy(llm=FakeLLM("malformed"))
        inp = ValidationInput(condition=SustainingCondition(id="c", text="t"), context={})
        with pytest.raises(ValueError):
            await strat.validate(inp)

    asyncio.run(run())


def test_is_user_controlled_strategy_timeout():
    async def run():
        strat = IsUserControlledStrategy(llm=FakeLLM("timeout"))
        inp = ValidationInput(condition=SustainingCondition(id="c", text="t"), context={})
        with pytest.raises(asyncio.TimeoutError):
            await strat.validate(inp)

    asyncio.run(run())


def test_validation_agent_routing_and_default():
    async def run():
        payload_path = {
            "status": "UNKNOWN",
            "evidence": [],
            "children": [{"text": "c1", "plan_kind": "PATH_EXISTS", "plan_params": {}}],
            "notes": [],
        }
        payload_user = {
            "status": "SATISFIED",
            "evidence": [{"summary": "ok", "source": "src", "strength": 0.5, "locations": []}],
            "children": [],
            "notes": [],
        }
        registry = StrategyRegistry()
        registry.register(PathExistsStrategy(FakeLLM("valid", payload_path)))
        registry.register(IsUserControlledStrategy(FakeLLM("valid", payload_user)))
        agent = ValidationAgent(registry)

        cond_default = SustainingCondition(id="1", text="t", plan_kind=None)
        res_default = await agent.validate(cond_default, {})
        assert res_default.children

        cond_user = SustainingCondition(id="2", text="t", plan_kind="IS_USER_CONTROLLED")
        res_user = await agent.validate(cond_user, {})
        assert res_user.status == Status.SATISFIED

    asyncio.run(run())


def test_validation_agent_unknown_plan_kind():
    async def run():
        payload_path = {
            "status": "UNKNOWN",
            "evidence": [],
            "children": [],
            "notes": [],
        }
        registry = StrategyRegistry()
        registry.register(PathExistsStrategy(FakeLLM("valid", payload_path)))
        agent = ValidationAgent(registry)
        cond = SustainingCondition(id="3", text="t", plan_kind="BOGUS")
        with pytest.raises(KeyError):
            await agent.validate(cond, {})

    asyncio.run(run())
