import copy
import unittest

from causal_router.engine import Engine, Policy


SHA = "a" * 64
SRC = "b" * 64


def policy(**kw):
    return Policy(feature_names=("trend",), reviewed={
        "alpha@v1": {"evidence_digest": SHA, "source_digest": SRC,
                     "asset": "NQ", "timeframe": "20m"},
        "beta@v1": {"evidence_digest": SHA, "source_digest": SRC,
                    "asset": "NQ", "timeframe": "20m"},
    }, sources=(SRC,), **kw)


def candidate(name="alpha", ts=0):
    return dict(id="register-" + name, type="candidate", observed_at=ts,
                candidate_id=name, version="v1", evidence_digest=SHA,
                source_digest=SRC, asset="NQ", timeframe="20m",
                feature_names=["trend"], expires_at=100000, split="development")


def decision(n=1, ts=100):
    return dict(id=f"d{n}", type="decision", observed_at=ts, decided_at=ts,
                asset="NQ", timeframe="20m", source_digest=SRC,
                feature_names=["trend"], features=[1.0], event_time=ts-1,
                available_at=ts-1, split="shadow")


def outcome(n=1, ts=150, rewards=None):
    return dict(id=f"o{n}", type="outcome", observed_at=ts, event_time=ts,
                decision_id=f"d{n}", rewards=rewards or {"alpha@v1": .8},
                split="shadow", reward_basis="net_risk_units")


class EngineTests(unittest.TestCase):
    def make(self, **kw):
        e = Engine(policy(**kw))
        e.apply(candidate())
        return e

    def test_registration_is_not_self_approval(self):
        e = Engine(policy())
        c = candidate()
        c["evidence_digest"] = "c" * 64
        with self.assertRaises(ValueError): e.apply(c)
        self.assertEqual(e.summary()["candidates"], 0)

    def test_future_features_rejected_without_mutation(self):
        e = self.make()
        d = decision()
        d["available_at"] = 101
        before = e.summary()
        with self.assertRaises(ValueError): e.apply(d)
        self.assertEqual(before, e.summary())

    def test_cold_start_abstains_then_learns(self):
        e = self.make(min_feedback=2, uncertainty_penalty=0, min_score=.1)
        self.assertIsNone(e.apply(decision())["selected"])
        e.apply(outcome())
        e.apply(decision(2, 200))
        e.apply(outcome(2, 250))
        d = e.apply(decision(3, 300))
        self.assertEqual(d["selected"], "alpha@v1")
        self.assertEqual(d["authority"], "research_only")
        self.assertFalse(d["execution_allowed"])

    def test_partial_feedback_is_not_zero_imputed(self):
        e = self.make()
        e.apply(candidate("beta"))
        e.apply(decision())
        before = e.summary()
        with self.assertRaises(ValueError): e.apply(outcome())
        self.assertEqual(e.summary(), before)
        r = e.apply(outcome(rewards={"alpha@v1": .8, "beta@v1": -.2}))
        self.assertEqual(r["learned"], 2)

    def test_duplicate_outcome_cannot_train_twice(self):
        e = self.make()
        e.apply(decision())
        e.apply(outcome())
        o = outcome()
        o["id"] = "another-id"
        with self.assertRaises(ValueError): e.apply(o)

    def test_protected_holdout_never_enters_learning(self):
        e = self.make()
        for maker in (decision, outcome):
            event = maker()
            event["split"] = "final_holdout"
            with self.assertRaises(ValueError): e.apply(event)

    def test_feedback_cannot_predate_decision(self):
        e = self.make()
        e.apply(decision())
        o = outcome()
        o["event_time"] = 99
        with self.assertRaises(ValueError): e.apply(o)

    def test_frozen_decisions_unchanged_by_future_learning(self):
        e = self.make()
        recorded = e.apply(decision())
        before = copy.deepcopy(recorded)
        e.apply(outcome())
        e.apply(decision(2, 200))
        self.assertEqual(recorded, before)

    def test_expired_candidate_does_not_participate(self):
        e = self.make()
        self.assertEqual(e.apply(decision(ts=100001))["reason"], "no_eligible_candidates")

    def test_revocation_stops_new_decisions_not_historical_feedback(self):
        e = self.make()
        e.apply(decision())
        e.apply(dict(id="r1", type="revoke", observed_at=110, candidate="alpha@v1"))
        self.assertIsNone(e.apply(decision(2, 120))["selected"])
        self.assertEqual(e.apply(outcome())["learned"], 1)

    def test_unknown_fields_nan_and_stale_features_rejected(self):
        e = self.make(max_feature_age=10)
        for patch in ({"oops": 1}, {"features": [float("nan")]},
                      {"event_time": 20}, {"source_digest": "c" * 64},
                      {"feature_names": ["new_schema"]}, {"features": [2.]},
                      {"observed_at": True}, {"decided_at": 99}):
            d = decision()
            d.update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError): e.apply(d)

    def test_bad_reward_batch_is_atomic(self):
        e = self.make()
        e.apply(candidate("beta"))
        e.apply(decision())
        before = e.summary()
        with self.assertRaises(ValueError):
            e.apply(outcome(rewards={"alpha@v1": .8, "beta@v1": float("nan")}))
        self.assertEqual(e.summary(), before)

    def test_failure_dossier_contains_frozen_forecast_not_postfit(self):
        e = self.make()
        d = e.apply(decision())
        e.apply(outcome(rewards={"alpha@v1": -.7}))
        f = e.summary()["failure_reviews"][0]
        self.assertEqual(f["prediction"], d["predictions"]["alpha@v1"]["mean"])
        self.assertEqual(f["net_reward"], -.7)
        self.assertFalse(f["automatic_strategy_modification"])

    def test_future_injection_preserves_entire_decision_prefix(self):
        from causal_router.demo import make_demo
        p, events, _ = make_demo(50, 17)
        altered = copy.deepcopy(events)
        for event in altered:
            if event["type"] == "outcome" and event["observed_at"] >= 35000:
                event["rewards"] = {k: -v for k, v in event["rewards"].items()}
        outputs = []
        for stream in (events, altered):
            e = Engine(p)
            prefix = []
            for event in stream:
                result = e.apply(event)
                if event["observed_at"] < 35000: prefix.append(result)
            outputs.append(prefix)
        self.assertEqual(outputs[0], outputs[1])

    def test_unrelated_slot_learning_cannot_change_nq_prediction(self):
        p = policy(min_feedback=1)
        p.reviewed["gamma@v1"] = {**p.reviewed["alpha@v1"], "asset": "GC"}
        e = Engine(p)
        e.apply(candidate())
        c = candidate("gamma")
        c["asset"] = "GC"
        e.apply(c)
        e.apply(decision())
        d = decision(2, 110)
        d["asset"] = "GC"
        e.apply(d)
        e.apply(outcome(2, 150, {"gamma@v1": .9}))
        r = e.apply(decision(3, 200))
        self.assertEqual(r["predictions"]["alpha@v1"]["mean"], 0.)
        self.assertEqual(r["predictions"]["alpha@v1"]["count"], 0)

    def test_extra_switch_friction_is_reported_and_charged(self):
        e = self.make(min_feedback=1, uncertainty_penalty=0, switch_penalty=.1)
        e.apply(candidate("beta"))
        e.apply(decision())
        e.apply(outcome(rewards={"alpha@v1": .8, "beta@v1": .9}))
        e.apply(decision(2, 200))
        e.apply(outcome(2, 250, {"alpha@v1": .8, "beta@v1": -.9}))
        switched = e.apply(decision(3, 300))
        self.assertEqual(switched["selected"], "alpha@v1")
        self.assertEqual(switched.get("switch_cost"), .1)
        before = e.summary()["metrics"]["selected_net_reward"]
        e.apply(outcome(3, 350, {"alpha@v1": .5, "beta@v1": 0.}))
        self.assertAlmostEqual(e.summary()["metrics"]["selected_net_reward"] - before, .4)

    def test_research_drawdown_cap_abstains_but_keeps_shadow_learning(self):
        e = self.make(min_feedback=1, uncertainty_penalty=0, max_drawdown=.5)
        e.apply(decision())
        e.apply(outcome())
        e.apply(decision(2, 200))
        e.apply(outcome(2, 250, {"alpha@v1": -.8}))
        r = e.apply(decision(3, 300))
        self.assertEqual(r["reason"], "research_drawdown_limit")
        self.assertIsNone(r["selected"])
        self.assertEqual(e.apply(outcome(3, 350))["learned"], 1)


if __name__ == "__main__": unittest.main()
