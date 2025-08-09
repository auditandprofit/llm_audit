"""Workflow graph and budget-aware scheduler.

This module provides minimal building blocks for orchestrating audit tasks
in a deterministic and budget conscious manner. Nodes are content addressed
so identical work items are deduplicated automatically. A priority based
scheduler coordinates execution while tracking global budgets for tokens,
wall clock time and number of tasks. Descendants of violated conditions can
be cancelled eagerly according to policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Awaitable, Callable
import asyncio
import heapq
import hashlib
import json
import time


# ---------------------------------------------------------------------------
# Core domain objects
# ---------------------------------------------------------------------------


class Status(Enum):
    """Lifecycle status of a node."""

    UNKNOWN = auto()
    SATISFIED = auto()
    VIOLATED = auto()
    CANCELLED = auto()


class NodeKind(Enum):
    FINDING = 1
    CONDITION = 2


def _normalize(text: str) -> str:
    return " ".join(text.strip().split())


def compute_digest(text: str, params: Dict[str, Any]) -> str:
    """Create a deterministic digest for a node's content."""

    norm = _normalize(text)
    param_str = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
    base = f"{norm}|{param_str}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NodeKey:
    kind: NodeKind
    digest: str


@dataclass
class Node:
    key: NodeKey
    status: Status = Status.UNKNOWN
    data: Dict[str, Any] = field(default_factory=dict)


class Graph:
    """Content addressed DAG of findings and conditions."""

    def __init__(self) -> None:
        self._nodes: Dict[NodeKey, Node] = {}
        self._children: Dict[NodeKey, Set[NodeKey]] = {}
        self._parents: Dict[NodeKey, Set[NodeKey]] = {}

    def get_or_add(self, node: Node) -> Node:
        existing = self._nodes.get(node.key)
        if existing is not None:
            return existing
        self._nodes[node.key] = node
        return node

    def add_edge(self, parent: NodeKey, child: NodeKey) -> None:
        self._children.setdefault(parent, set()).add(child)
        self._parents.setdefault(child, set()).add(parent)

    def children(self, key: NodeKey) -> Set[NodeKey]:
        return self._children.get(key, set())

    def parents(self, key: NodeKey) -> Set[NodeKey]:
        return self._parents.get(key, set())

    def node(self, key: NodeKey) -> Node:
        return self._nodes[key]


@dataclass
class Budget:
    tokens: int
    seconds: float
    tasks: int

    def __post_init__(self) -> None:
        self._remaining_tokens = self.tokens
        self._remaining_tasks = self.tasks
        self._start = time.time()

    def allow(self, *, tks: int = 0, s: float = 0.0, n: int = 0) -> bool:
        now = time.time()
        elapsed = now - self._start
        if self._remaining_tokens < tks:
            return False
        if self._remaining_tasks < n:
            return False
        if elapsed + s > self.seconds:
            return False
        return True

    def consume(self, *, tks: int = 0, s: float = 0.0, n: int = 0) -> None:
        self._remaining_tokens -= tks
        self._remaining_tasks -= n


@dataclass
class ValidationResult:
    status: Status
    children: List[Node]
    cost: Budget = field(default_factory=lambda: Budget(0, 0.0, 0))


class Scheduler:
    """Budget aware scheduler operating on a graph of nodes."""

    def __init__(self, graph: Graph, budget: Budget, *, early_cancel: bool = True) -> None:
        self.graph = graph
        self.budget = budget
        self.early_cancel = early_cancel
        self._queue: List[Tuple[int, int, NodeKey]] = []
        self._counter = 0  # tie-breaker for heapq

    def enqueue(self, node: NodeKey, prio: int) -> None:
        heapq.heappush(self._queue, (prio, self._counter, node))
        self._counter += 1

    async def run(self, handler: Callable[[NodeKey], Awaitable[ValidationResult]]) -> None:
        while self._queue:
            _, _, key = heapq.heappop(self._queue)
            node = self.graph.node(key)
            if node.status == Status.CANCELLED:
                continue
            if self._violated_ancestor(key):
                continue
            if not self.budget.allow(n=1):
                break
            self.budget.consume(n=1)
            result = await handler(key)
            node.status = result.status
            if result.cost:
                self.budget.consume(tks=result.cost.tokens, s=result.cost.seconds, n=result.cost.tasks)
            if result.status == Status.VIOLATED and self.early_cancel:
                self._cancel_descendants(key)
                continue
            for child in result.children:
                ch = self.graph.get_or_add(child)
                self.graph.add_edge(key, ch.key)
                if ch.status == Status.UNKNOWN:
                    self.enqueue(ch.key, -self._score(ch))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score(self, node: Node) -> int:
        data = node.data
        depth = data.get("depth", 0)
        severity = data.get("severity", 0)
        confidence = data.get("confidence", 0)
        novelty = data.get("novelty", 0)
        return severity + confidence + novelty - depth

    def _violated_ancestor(self, key: NodeKey) -> bool:
        visited: Set[NodeKey] = set()

        def dfs(k: NodeKey) -> bool:
            if k in visited:
                return False
            visited.add(k)
            for parent in self.graph.parents(k):
                pnode = self.graph.node(parent)
                if pnode.status == Status.VIOLATED or dfs(parent):
                    return True
            return False

        return dfs(key)

    def _cancel_descendants(self, key: NodeKey) -> None:
        for child in self.graph.children(key):
            node = self.graph.node(child)
            if node.status != Status.CANCELLED:
                node.status = Status.CANCELLED
                self._cancel_descendants(child)


__all__ = [
    "Status",
    "NodeKind",
    "NodeKey",
    "Node",
    "Graph",
    "Budget",
    "ValidationResult",
    "Scheduler",
    "compute_digest",
]

