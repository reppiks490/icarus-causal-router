"""Isolated persistence acceptance; synthetic events are not market evidence."""

import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path

from causal_router.demo import make_demo
from causal_router.journal import Session, _record_hash


class JournalAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "research.db"
        self.policy, self.events, _ = make_demo(steps=2, seed=7)

    def seed(self):
        with Session(self.path, self.policy) as session:
            session.ingest(self.events[0])
            return session.summary()

    def test_missing_policy_witness_cannot_be_recreated(self):
        self.seed()
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM metadata WHERE name='policy_sha256'")
        before = self.path.read_bytes()
        changed = copy.deepcopy(self.policy)
        changed.max_pending += 1
        with self.assertRaisesRegex(ValueError, "policy witness missing"):
            Session(self.path, changed)
        self.assertEqual(self.path.read_bytes(), before)

    def test_missing_metadata_table_does_not_mutate_history(self):
        self.seed()
        with sqlite3.connect(self.path) as db:
            db.execute("DROP TABLE metadata")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "journal schema incomplete"):
            Session(self.path, self.policy)
        self.assertEqual(self.path.read_bytes(), before)

    def test_unrelated_database_is_preserved(self):
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE owner_data(value TEXT)")
            db.execute("INSERT INTO owner_data VALUES ('preserve')")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "unrecognized database"):
            Session(self.path, self.policy)
        self.assertEqual(self.path.read_bytes(), before)

    def test_concurrent_append_rejected_without_local_mutation(self):
        with Session(self.path, self.policy) as first, Session(self.path, self.policy) as stale:
            initial = stale.summary()
            first.ingest(self.events[0])
            with self.assertRaisesRegex(ValueError, "concurrent journal writer"):
                stale.ingest(self.events[1])
            self.assertEqual(stale.summary(), initial)
        with Session(self.path, self.policy) as fresh:
            self.assertEqual(fresh.summary()["events"], 1)
            fresh.ingest(self.events[1])
            self.assertEqual(fresh.summary()["events"], 2)

    def test_failed_insert_rolls_back_database_and_engine(self):
        with Session(self.path, self.policy) as session:
            before = session.summary()
            with sqlite3.connect(self.path) as db:
                db.execute("CREATE TRIGGER fail_insert BEFORE INSERT ON events "
                           "BEGIN SELECT RAISE(ABORT, 'injected write failure'); END")
            with self.assertRaisesRegex(sqlite3.IntegrityError, "injected write failure"):
                session.ingest(self.events[0])
            self.assertEqual(session.summary(), before)
            with sqlite3.connect(self.path) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM events").fetchone()[0], 0)
                db.execute("DROP TRIGGER fail_insert")
            session.ingest(self.events[0])
        with Session(self.path, self.policy) as fresh:
            self.assertEqual(fresh.summary()["events"], 1)

    def test_recomputed_hash_cannot_certify_false_result(self):
        self.seed()
        with sqlite3.connect(self.path) as db:
            parent, event = db.execute("SELECT previous_hash,event_json FROM events").fetchone()
            result = '{"authority":"research_only","execution_allowed":false,"forged":true}'
            db.execute("UPDATE events SET result_json=?,entry_hash=?", (result, _record_hash(parent, event, result)))
        with self.assertRaisesRegex(ValueError, "deterministic replay mismatch"):
            Session(self.path, self.policy)

    def test_protected_input_rejected_without_persistence(self):
        with Session(self.path, self.policy) as session:
            bad = copy.deepcopy(self.events[0])
            bad["split"] = "final_holdout"
            with self.assertRaises(ValueError):
                session.ingest(bad)
            self.assertEqual(session.summary()["events"], 0)
        with Session(self.path, self.policy) as fresh:
            self.assertEqual(fresh.summary()["events"], 0)

    def test_returned_values_cannot_modify_frozen_result(self):
        with Session(self.path, self.policy) as session:
            result = session.ingest(self.events[0])
            expected = copy.deepcopy(result)
            result["execution_allowed"] = True
            self.assertEqual(session.ingest(self.events[0]), expected)


if __name__ == "__main__":
    unittest.main()
