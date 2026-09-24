"""Seeded synthetic scenarios, explicitly unsuitable as market evidence."""

import hashlib
import math
import random

from .engine import Policy, integer


def make_demo(steps=500, seed=7):
    integer(steps, "steps", 1, 20000)
    integer(seed, "seed", 0, 2**32-1)
    rng = random.Random(seed)
    source = hashlib.sha256(b"synthetic-demo-not-market-data").hexdigest()
    slots = [("NQ", "20m"), ("GC", "20m"), ("BTCUSD", "60m")]
    families = ("trend", "reverse", "defensive")
    reviews, events, fixed = {}, [], {}
    for asset, tf in slots:
        fixed[asset + "/" + tf] = dict.fromkeys(families, 0.)
        for family in families:
            name = asset + "-" + family
            evidence = hashlib.sha256(("synthetic-" + name).encode()).hexdigest()
            reviews[name + "@v1"] = {"asset": asset, "timeframe": tf,
                                    "evidence_digest": evidence, "source_digest": source}
            events.append({"type": "candidate", "id": "register-" + name,
                           "observed_at": 0, "candidate_id": name, "version": "v1",
                           "asset": asset, "timeframe": tf, "evidence_digest": evidence,
                           "source_digest": source, "feature_names": ["trend", "volatility"],
                           "expires_at": (steps+100)*1000, "split": "development"})
    for i in range(steps):
        asset, tf = slots[i % len(slots)]
        local_step = i // len(slots)
        ts = (i+1)*1000
        x = [round(math.sin(local_step / 7.), 8), round(rng.uniform(-1, 1), 8)]
        events.append({"type": "decision", "id": f"d{i}", "observed_at": ts,
                       "decided_at": ts, "asset": asset, "timeframe": tf,
                       "source_digest": source, "feature_names": ["trend", "volatility"],
                       "features": x, "event_time": ts-5, "available_at": ts-2,
                       "split": "shadow"})
        # The hidden regime is NOT exposed as a feature. Every outcome includes
        # explicit utility costs and matures after the prediction was frozen.
        regime = 1 if (local_step // max(10, steps//9)) % 2 == 0 else -1
        noise = rng.uniform(-.07, .07)
        values = {"trend": regime*.62*x[0] + noise - .025,
                  "reverse": -regime*.62*x[0] - noise - .025,
                  "defensive": .04 - .02*abs(x[1]) + noise/2}
        for family, reward in values.items(): fixed[asset + "/" + tf][family] += reward
        events.append({"type": "outcome", "id": f"o{i}", "decision_id": f"d{i}",
                       "observed_at": ts + rng.randint(1, 8)*1000 + 250,
                       "event_time": ts + 500,
                       "rewards": {asset+"-"+k+"@v1": round(v, 10) for k, v in values.items()},
                       "reward_basis": "net_risk_units", "split": "shadow"})
    events.sort(key=lambda e: (e["observed_at"], e["id"]))
    policy = Policy(feature_names=("trend", "volatility"), reviewed=reviews,
                    sources=(source,), min_feedback=8, uncertainty_penalty=.15,
                    switch_penalty=.02, min_score=0., max_drawdown=5.)
    baselines = {"abstain": 0., "fixed_family_by_slot": fixed,
                 "best_fixed_per_slot_hindsight": sum(max(v.values()) for v in fixed.values()),
                 "hindsight_is_not_deployable": True}
    return policy, events, baselines
