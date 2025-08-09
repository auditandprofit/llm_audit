import asyncio
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from workflow import (
    Budget,
    Graph,
    Node,
    NodeKey,
    NodeKind,
    Scheduler,
    Status,
    ValidationResult,
)


def make_node(text: str, *, kind: NodeKind = NodeKind.CONDITION) -> Node:
    digest = f"digest-{text}"
    return Node(key=NodeKey(kind, digest), data={"depth": 0})


def test_graph_deduplication():
    g = Graph()
    n1 = make_node("a")
    n2 = make_node("a")
    assert g.get_or_add(n1) is n1
    assert g.get_or_add(n2) is n1  # deduped


def test_scheduler_budget_and_cancellation():
    g = Graph()
    budget = Budget(tokens=0, seconds=10, tasks=10)
    sched = Scheduler(g, budget, early_cancel=True)
    root = make_node("root")
    child = make_node("child")
    g.get_or_add(root)
    g.get_or_add(child)
    g.add_edge(root.key, child.key)

    async def handler(key: NodeKey) -> ValidationResult:
        if key == root.key:
            return ValidationResult(status=Status.VIOLATED, children=[])
        raise AssertionError("child should not be validated")

    sched.enqueue(root.key, prio=0)
    sched.enqueue(child.key, prio=1)
    asyncio.run(sched.run(handler))
    assert g.node(root.key).status is Status.VIOLATED
    assert g.node(child.key).status is Status.CANCELLED


def test_scheduler_respects_task_budget():
    g = Graph()
    budget = Budget(tokens=0, seconds=10, tasks=1)
    sched = Scheduler(g, budget)
    n1 = make_node("n1")
    n2 = make_node("n2")
    g.get_or_add(n1)
    g.get_or_add(n2)

    async def handler(key: NodeKey) -> ValidationResult:
        return ValidationResult(status=Status.SATISFIED, children=[])

    sched.enqueue(n1.key, prio=0)
    sched.enqueue(n2.key, prio=0)
    asyncio.run(sched.run(handler))
    assert g.node(n1.key).status is Status.SATISFIED
    assert g.node(n2.key).status is Status.UNKNOWN

