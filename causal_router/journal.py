"""Crash-consistent, deterministic, research-only SQLite event journal.

The database records original events, frozen research results, a policy witness
and a SHA-256 chain. The chain catches accidental edits, not malicious rewriting
of both the database and its root of trust. Reopening replays every transition.
No broker or order execution is available from this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from pathlib import Path

from .engine import Engine, Policy


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _record_hash(previous, event_text, result_text):
    return hashlib.sha256(
        (previous + "\n" + event_text + "\n" + result_text).encode("utf-8")
    ).hexdigest()


class Session:
    """Append immutable events with exact retries and verified replay on open.

    SQLite transaction boundaries precede promotion of the in-memory engine.
    Another writer's append is detected rather than silently overwriting state.
    The policy is pinned to the original database, so a changed candidate review
    or risk constraint cannot reinterpret an existing history.
    """

    def __init__(self, db, policy):
        if not isinstance(policy, Policy):
            raise TypeError("Policy instance required")
        self.path = Path(db)
        self.policy = copy.deepcopy(policy)
        self.engine = Engine(self.policy)
        self._count = 0
        self._tail = "0" * 64
        self._events = {}
        self._closed = False
        self._connection = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        try:
            self._connection.execute("PRAGMA busy_timeout=10000")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata "
                "(name TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS events "
                "(sequence INTEGER PRIMARY KEY, event_id TEXT NOT NULL UNIQUE, "
                "event_json TEXT NOT NULL, result_json TEXT NOT NULL, "
                "previous_hash TEXT NOT NULL, entry_hash TEXT NOT NULL)"
            )
            self._validate_policy()
            self._replay()
        except BaseException:
            self._connection.close()
            self._closed = True
            raise

    def _validate_policy(self):
        witness = _fingerprint(self.policy.to_dict())
        current = self._connection.execute(
            "SELECT value FROM metadata WHERE name='policy_sha256'"
        ).fetchone()
        if current is None:
            self._connection.execute(
                "INSERT INTO metadata (name,value) VALUES ('policy_sha256',?)",
                (witness,),
            )
        elif current[0] != witness:
            raise ValueError("journal policy fingerprint mismatch")

    def _replay(self):
        previous = "0" * 64
        for sequence, event_id, event_text, result_text, parent, digest in self._connection.execute(
            "SELECT sequence,event_id,event_json,result_json,previous_hash,entry_hash "
            "FROM events ORDER BY sequence"
        ):
            if sequence != self._count + 1:
                raise ValueError("journal sequence discontinuity")
            if parent != previous or digest != _record_hash(parent, event_text, result_text):
                raise ValueError("journal hash chain mismatch")
            event = json.loads(event_text)
            result = json.loads(result_text)
            if not isinstance(event, dict) or event.get("id") != event_id:
                raise ValueError("journal event identity mismatch")
            if _canonical(event) != event_text or _canonical(result) != result_text:
                raise ValueError("journal has noncanonical or modified records")
            replayed = self.engine.apply(event)
            if _canonical(replayed) != result_text:
                raise ValueError("journal deterministic replay mismatch")
            self._events[event_id] = (event_text, result_text)
            self._count += 1
            previous = digest
        self._tail = previous

    def ingest(self, event):
        if self._closed:
            raise RuntimeError("journal is closed")
        if not isinstance(event, dict):
            raise ValueError("event object required")
        event_text = _canonical(event)
        identity = event.get("id")
        if not isinstance(identity, str):
            raise ValueError("event id required")
        previous = self._events.get(identity)
        if previous is not None:
            if previous[0] != event_text:
                raise ValueError("event id already committed with different contents")
            return copy.deepcopy(json.loads(previous[1]))

        trial = copy.deepcopy(self.engine)
        result = trial.apply(event)
        if result.get("execution_allowed") is not False or result.get("authority") != "research_only":
            raise ValueError("research-only authority invariant violated")
        result_text = _canonical(result)
        record_hash = _record_hash(self._tail, event_text, result_text)
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            count, tail = self._connection.execute(
                "SELECT COUNT(*), COALESCE((SELECT entry_hash FROM events "
                "ORDER BY sequence DESC LIMIT 1), ? ) FROM events",
                ("0" * 64,),
            ).fetchone()
            if count != self._count or tail != self._tail:
                raise ValueError("concurrent journal writer changed history; reopen before retry")
            self._connection.execute(
                "INSERT INTO events (sequence,event_id,event_json,result_json,"
                "previous_hash,entry_hash) VALUES (?,?,?,?,?,?)",
                (self._count + 1, identity, event_text, result_text, self._tail, record_hash),
            )
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

        self.engine = trial
        self._events[identity] = (event_text, result_text)
        self._count += 1
        self._tail = record_hash
        return copy.deepcopy(result)

    def summary(self):
        if self._closed:
            raise RuntimeError("journal is closed")
        return self.engine.summary()

    def close(self):
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("journal is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
