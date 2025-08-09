from __future__ import annotations
import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


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
    """SQLite-backed event store (safe for concurrent async tasks)."""

    def __init__(self, path: str = "runs.db") -> None:
        self.path = path
        self._lock = threading.RLock()  # serialize writers
        self.conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None
        )
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute("PRAGMA busy_timeout=3000;")
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
            "CREATE INDEX IF NOT EXISTS idx_events_run_ts ON events(run_id, ts)"
        )

    @contextmanager
    def _txn(self):
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                yield
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def append(self, ev: RunEvent) -> None:
        with self._txn():
            self.conn.execute(
                "INSERT INTO events(run_id, ts, type, data_json) VALUES(?,?,?,?)",
                (ev.run_id, ev.ts, ev.type.name, json.dumps(ev.data)),
            )

    def append_many(self, events: List[RunEvent]) -> None:
        if not events:
            return
        with self._txn():
            self.conn.executemany(
                "INSERT INTO events(run_id, ts, type, data_json) VALUES(?,?,?,?)",
                [
                    (e.run_id, e.ts, e.type.name, json.dumps(e.data))
                    for e in events
                ],
            )

    def load(self, run_id: str) -> List[RunEvent]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT ts, type, data_json FROM events WHERE run_id=? ORDER BY ts",
                (run_id,),
            )
            rows = cur.fetchall()
        out: List[RunEvent] = []
        for ts, type_str, data_json in rows:
            out.append(
                RunEvent(
                    ts=ts,
                    run_id=run_id,
                    type=EventType[type_str],
                    data=json.loads(data_json),
                )
            )
        return out

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

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class BlobStore:
    """Content-addressed blob store backed by SQLite."""

    def __init__(self, path: str = "runs.db") -> None:
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(
            path, check_same_thread=False, isolation_level=None
        )
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute("PRAGMA busy_timeout=3000;")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blobs(
              sha TEXT PRIMARY KEY,
              bytes BLOB
            )
            """
        )

    def put(self, b: bytes) -> str:
        sha = hashlib.sha256(b).hexdigest()
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO blobs(sha, bytes) VALUES(?, ?)", (sha, b)
            )
        return sha

    def get(self, sha: str) -> bytes:
        with self._lock:
            cur = self.conn.execute("SELECT bytes FROM blobs WHERE sha=?", (sha,))
            row = cur.fetchone()
        if row is None:
            raise KeyError(sha)
        return row[0]

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
