"""Journal regression checks: exact retries, deterministic restart and tamper rejection."""

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from causal_router.demo import make_demo
from causal_router.journal import Session


class JournalTests(unittest.TestCase):
    def test_identical_replay_and_retry(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "research.db"
            policy, events, _ = make_demo(steps=8, seed=7)
            with Session(path, policy) as session:
                for event in events:
                    session.ingest(event)
                expected = session.summary()
                self.assertEqual(session.ingest(events[0])["authority"], "research_only")
                self.assertFalse(session.ingest(events[0])["execution_allowed"])
                self.assertEqual(session.summary(), expected)
            with Session(path, policy) as recovered:
                self.assertEqual(recovered.summary(), expected)
                self.assertEqual(recovered.ingest(events[-1])["authority"], "research_only")
                self.assertEqual(recovered.summary(), expected)

    def test_reused_identity_with_different_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            policy, events, _ = make_demo(steps=1, seed=7)
            with Session(Path(root) / "research.db", policy) as session:
                session.ingest(events[0])
                altered = copy.deepcopy(events[0])
                altered["observed_at"] += 1
                with self.assertRaisesRegex(ValueError, "different contents"):
                    session.ingest(altered)

    def test_modification_detected_at_reopen(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "research.db"
            policy, events, _ = make_demo(steps=1, seed=7)
            with Session(path, policy) as session:
                session.ingest(events[0])
            with sqlite3.connect(path) as database:
                database.execute(
                    "UPDATE events SET result_json=? WHERE sequence=1",
                    (json.dumps({"forged": True}),),
                )
            with self.assertRaisesRegex(ValueError, "hash chain mismatch"):
                Session(path, policy)

    def test_policy_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "research.db"
            policy, events, _ = make_demo(steps=1, seed=7)
            with Session(path, policy) as session:
                session.ingest(events[0])
            different, _, _ = make_demo(steps=1, seed=8)
            # Seeds vary generated events, not the reviewed policy. Exercise
            # an actual risk-policy change rather than expecting a false mismatch.
            different.max_pending = policy.max_pending + 1
            self.assertNotEqual(different.to_dict(), policy.to_dict())
            with self.assertRaisesRegex(ValueError, "policy fingerprint mismatch"):
                Session(path, different)


if __name__ == "__main__":
    unittest.main()

