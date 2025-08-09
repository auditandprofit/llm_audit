from __future__ import annotations
import sqlite3
import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional


class EventType(Enum):
    TASK_ENQUEUED = 1
    TASK_STARTED = 2
    TASK_FINISHED = 3
    NODE_STATUS = 4
    LLM_RETRY = 5
    BUDGET_HIT = 6
    CANCEL = 7


@dataclass
class RunEvent:
    ts: float
    run_id: str
    type: EventType
    data: Dict[str, Any]


@dataclass
class DerivedState:
    tasks_enqueued: int = 0
    tasks_started: int = 0
    tasks_finished: int = 0
    queue: List[Dict[str, Any]] = field(default_factory=list)


class RunStore:
    """SQLite-backed event store."""

    def __init__(self, path: str = "runs.db") -> None:
        self.path = path
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events(
              id INTEGER PRIMARY KEY,
              run_id TEXT,
              ts REAL,
              type TEXT,
              data_json TEXT
            )
            """
        )
        self.conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_events_run_ts ON events(run_id, ts)"""
        )
        self.conn.commit()

    def append(self, ev: RunEvent) -> None:
        self.conn.execute(
            "INSERT INTO events(run_id, ts, type, data_json) VALUES(?,?,?,?)",
            (ev.run_id, ev.ts, ev.type.name, json.dumps(ev.data)),
        )
        self.conn.commit()

    def load(self, run_id: str) -> List[RunEvent]:
        cur = self.conn.execute(
            "SELECT ts, type, data_json FROM events WHERE run_id=? ORDER BY ts",
            (run_id,),
        )
        events: List[RunEvent] = []
        for ts, type_str, data_json in cur.fetchall():
            events.append(
                RunEvent(
                    ts=ts,
                    run_id=run_id,
                    type=EventType[type_str],
                    data=json.loads(data_json),
                )
            )
        return events

    def latest_state(self, run_id: str) -> DerivedState:
        state = DerivedState()
        tasks: Dict[str, Dict[str, Any]] = {}
        for ev in self.load(run_id):
            if ev.type == EventType.TASK_ENQUEUED:
                t = ev.data.get("task")
                if t:
                    tasks[t["id"]] = ev.data
                state.tasks_enqueued += 1
            elif ev.type == EventType.TASK_STARTED:
                state.tasks_started += 1
            elif ev.type == EventType.TASK_FINISHED:
                tid = ev.data.get("task_id")
                if tid and tid in tasks:
                    tasks.pop(tid)
                state.tasks_finished += 1
        state.queue = list(tasks.values())
        return state


class BlobStore:
    """Content-addressed blob store backed by SQLite."""

    def __init__(self, path: str = "runs.db") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blobs(
              sha TEXT PRIMARY KEY,
              bytes BLOB
            )
            """
        )
        self.conn.commit()

    def put(self, b: bytes) -> str:
        sha = hashlib.sha256(b).hexdigest()
        self.conn.execute(
            "INSERT OR IGNORE INTO blobs(sha, bytes) VALUES(?, ?)", (sha, b)
        )
        self.conn.commit()
        return sha

    def get(self, sha: str) -> bytes:
        cur = self.conn.execute("SELECT bytes FROM blobs WHERE sha=?", (sha,))
        row = cur.fetchone()
        if row is None:
            raise KeyError(sha)
        return row[0]
