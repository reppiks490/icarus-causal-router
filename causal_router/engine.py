"""Causal, atomic event transitions. All outputs remain research-only."""

import copy
import math
import re
from dataclasses import asdict, dataclass

from .learning import AdaptiveRegressor


def number(value, name, low, high):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{name}: finite number in [{low}, {high}] required")
    return float(value)


def integer(value, name, low=0, high=2**53-1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}: integer in [{low}, {high}] required")
    return value


def identity(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
        raise ValueError("invalid identity")
    return value


def digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("SHA-256 hex digest required")
    return value


@dataclass
class Policy:
    feature_names: tuple
    reviewed: dict
    sources: tuple
    min_feedback: int = 8
    max_feature_age: int = 60000
    uncertainty_penalty: float = .25
    switch_penalty: float = .02
    min_score: float = 0.
    max_drawdown: float = 5.
    max_pending: int = 10000

    def __post_init__(self):
        self.feature_names = tuple(self.feature_names)
        self.sources = tuple(self.sources)
        if not 1 <= len(self.feature_names) <= 64:
            raise ValueError("1..64 feature names required")
        for name in self.feature_names: identity(name)
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("duplicate feature names")
        integer(self.min_feedback, "min_feedback", 1, 1000000)
        integer(self.max_feature_age, "max_feature_age", 1)
        integer(self.max_pending, "max_pending", 1, 100000)
        number(self.uncertainty_penalty, "uncertainty_penalty", 0, 100)
        number(self.switch_penalty, "switch_penalty", 0, 2)
        number(self.min_score, "min_score", -1, 1)
        number(self.max_drawdown, "max_drawdown", .01, 1000000)
        if not isinstance(self.reviewed, dict) or len(self.reviewed) > 1000:
            raise ValueError("reviewed mapping required, at most 1000 candidates")
        if not self.sources or len(self.sources) != len(set(self.sources)):
            raise ValueError("distinct approved source digests required")
        for source in self.sources: digest(source)
        for key, record in self.reviewed.items():
            if not isinstance(key, str) or len(key.split("@")) != 2:
                raise ValueError("review key must be candidate@version")
            for part in key.split("@"): identity(part)
            if not isinstance(record, dict) or set(record) != {
                "evidence_digest", "source_digest", "asset", "timeframe"
            }:
                raise ValueError("review record schema mismatch")
            digest(record["evidence_digest"])
            digest(record["source_digest"])
            if record["source_digest"] not in self.sources:
                raise ValueError("reviewed candidate source not approved")
            identity(record["asset"])
            identity(record["timeframe"])
        self.reviewed = copy.deepcopy(self.reviewed)

    def to_dict(self):
        result = asdict(self)
        result["feature_names"] = list(self.feature_names)
        result["sources"] = list(self.sources)
        return result


FIELDS = {
    "candidate": {"candidate_id", "version", "evidence_digest", "source_digest",
                  "asset", "timeframe", "feature_names", "expires_at", "split"},
    "decision": {"decided_at", "asset", "timeframe", "source_digest", "feature_names",
                 "features", "event_time", "available_at", "split"},
    "outcome": {"event_time", "decision_id", "rewards", "split", "reward_basis"},
    "revoke": {"candidate"},
}


class Engine:
    def __init__(self, policy):
        self.policy = copy.deepcopy(policy)
        self.candidates = {}
        self.models = {}
        self.pending = {}
        self.completed = set()
        self.seen = set()
        self.last_observed = -1
        self.slot_clock = {}
        self.slot_steps = {}
        self.incumbents = {}
        self.risk = {}
        self.failures = []
        self.metrics = {"decisions": 0, "abstentions": 0, "settled": 0,
                        "learned_rewards": 0, "selected_net_reward": 0.,
                        "selected_settled": 0, "prediction_squared_error": 0.}

    def apply(self, event):
        # Isolate an entire transition: validation or numerical failure cannot
        # leave one candidate trained while another candidate's label fails.
        trial = copy.deepcopy(self)
        result = trial._apply(copy.deepcopy(event))
        self.__dict__ = trial.__dict__
        return copy.deepcopy(result)

    def _apply(self, event):
        if not isinstance(event, dict): raise ValueError("event object required")
        kind = event.get("type")
        if not isinstance(kind, str) or kind not in FIELDS:
            raise ValueError("unsupported event type")
        if set(event) != FIELDS[kind] | {"id", "type", "observed_at"}:
            raise ValueError("missing or unknown event fields")
        identity(event["id"])
        if event["id"] in self.seen: raise ValueError("duplicate event id")
        ts = integer(event["observed_at"], "observed_at")
        if ts < self.last_observed: raise ValueError("observation clock moved backwards")
        if "split" in event and event["split"] not in ("development", "shadow"):
            raise ValueError("protected/unknown split cannot enter adaptive learning")
        result = getattr(self, "_" + kind)(event)
        self.seen.add(event["id"])
        self.last_observed = ts
        return {"authority": "research_only", "execution_allowed": False, **result}

    def _schema(self, event):
        if not isinstance(event["feature_names"], list) or tuple(event["feature_names"]) != self.policy.feature_names:
            raise ValueError("feature schema mismatch")

    def _candidate(self, event):
        key = identity(event["candidate_id"]) + "@" + identity(event["version"])
        if key in self.candidates: raise ValueError("candidate version is immutable")
        approved = self.policy.reviewed.get(key)
        if approved is None or any(event[k] != v for k, v in approved.items()):
            raise ValueError("candidate does not match operator review manifest")
        self._schema(event)
        if integer(event["expires_at"], "expires_at") <= event["observed_at"]:
            raise ValueError("candidate already expired")
        self.candidates[key] = {**event, "revoked": False}
        self.models[key] = AdaptiveRegressor(len(self.policy.feature_names))
        return {"registered": key}

    def _decision(self, event):
        ts = integer(event["decided_at"], "decided_at")
        if ts != event["observed_at"]:
            raise ValueError("decision must occur at observation time, not be backdated")
        asset, tf = identity(event["asset"]), identity(event["timeframe"])
        slot = asset + "/" + tf
        if ts <= self.slot_clock.get(slot, -1):
            raise ValueError("slot decision clock must strictly increase")
        self._schema(event)
        event_time = integer(event["event_time"], "event_time")
        available = integer(event["available_at"], "available_at")
        if not event_time <= available <= ts:
            raise ValueError("feature event/availability exceeds decision time")
        if ts - event_time > self.policy.max_feature_age:
            raise ValueError("stale features")
        if digest(event["source_digest"]) not in self.policy.sources:
            raise ValueError("unapproved feature source")
        x = event["features"]
        if not isinstance(x, list) or len(x) != len(self.policy.feature_names):
            raise ValueError("feature dimension mismatch")
        for value in x: number(value, "feature", -1, 1)
        if len(self.pending) >= self.policy.max_pending:
            raise ValueError("pending feedback budget exhausted")
        predictions, ranking = {}, []
        incumbent = self.incumbents.get(slot)
        step = self.slot_steps.get(slot, 0) + 1
        for key, c in sorted(self.candidates.items()):
            if (c["asset"] != asset or c["timeframe"] != tf or c["revoked"]
                    or c["expires_at"] <= ts or c["source_digest"] != event["source_digest"]):
                continue
            prediction = self.models[key].predict(x, step=step)
            score = (prediction["mean"] - self.policy.uncertainty_penalty * prediction["uncertainty"]
                     - (self.policy.switch_penalty if incumbent and key != incumbent else 0.))
            predictions[key] = {**prediction, "score": score}
            if prediction["count"] >= self.policy.min_feedback:
                ranking.append((score, key))
        risk = self.risk.setdefault(slot, {"net": 0., "peak": 0., "drawdown": 0.})
        ranking.sort(key=lambda pair: (-pair[0], pair[1]))
        selected = None
        reason = "no_eligible_candidates"
        if predictions:
            reason = "insufficient_feedback"
            if ranking:
                reason = "nonpositive_risk_adjusted_edge"
                if ranking[0][0] > self.policy.min_score:
                    selected, reason = ranking[0][1], "research_candidate_selected"
            if risk["drawdown"] >= self.policy.max_drawdown:
                selected, reason = None, "research_drawdown_limit"
        if selected: self.incumbents[slot] = selected
        self.slot_clock[slot] = ts
        self.slot_steps[slot] = step
        self.metrics["decisions"] += 1
        self.metrics["abstentions"] += selected is None
        result = {"decision_id": event["id"], "slot": slot, "decided_at": ts,
                  "selected": selected, "reason": reason, "predictions": predictions,
                  "switch_cost": (self.policy.switch_penalty if selected and incumbent and selected != incumbent else 0.),
                  "uncertainty_is_calibrated": False}
        # Even abstained decisions collect full shadow labels for warm-up.
        if predictions:
            self.pending[event["id"]] = {"result": copy.deepcopy(result), "features": list(x)}
        return result

    def _outcome(self, event):
        identity(event["decision_id"])
        pending = self.pending.get(event["decision_id"])
        if pending is None: raise ValueError("unknown or already settled decision")
        d = pending["result"]
        end = integer(event["event_time"], "event_time")
        if not d["decided_at"] < end <= event["observed_at"]:
            raise ValueError("outcome must mature after decision and before receipt")
        if event["reward_basis"] != "net_risk_units":
            raise ValueError("net, common risk-unit reward basis required")
        rewards = event["rewards"]
        if not isinstance(rewards, dict) or set(rewards) != set(d["predictions"]):
            raise ValueError("complete frozen shadow candidate rewards required")
        for reward in rewards.values(): number(reward, "net reward", -1, 1)
        for key, prediction in sorted(d["predictions"].items()):
            reward = float(rewards[key])
            self.models[key].update(pending["features"], reward, prediction)
            self.metrics["prediction_squared_error"] += (prediction["mean"] - reward)**2
            if reward < 0:
                self.failures.append({"decision_id": event["decision_id"], "candidate": key,
                                      "slot": d["slot"], "prediction": prediction["mean"],
                                      "net_reward": reward, "available_at": event["observed_at"],
                                      "features": pending["features"],
                                      "hypothesis": "forecast_miss" if prediction["mean"] > 0 else "anticipated_downside",
                                      "automatic_strategy_modification": False})
        self.failures = self.failures[-200:]
        selected = d["selected"]
        if selected:
            reward = float(rewards[selected]) - d["switch_cost"]
            self.metrics["selected_net_reward"] += reward
            self.metrics["selected_settled"] += 1
            r = self.risk[d["slot"]]
            r["net"] += reward
            r["peak"] = max(r["peak"], r["net"])
            r["drawdown"] = max(r["drawdown"], r["peak"] - r["net"])
        self.metrics["settled"] += 1
        self.metrics["learned_rewards"] += len(rewards)
        del self.pending[event["decision_id"]]
        self.completed.add(event["decision_id"])
        return {"learned": len(rewards), "decision_id": event["decision_id"]}

    def _revoke(self, event):
        key = event["candidate"]
        if not isinstance(key, str) or key not in self.candidates:
            raise ValueError("unknown candidate")
        self.candidates[key]["revoked"] = True
        return {"revoked": key}

    def summary(self):
        return copy.deepcopy({"authority": "research_only", "execution_allowed": False,
                              "candidates": len(self.candidates), "events": len(self.seen),
                              "pending": len(self.pending), "last_observed": self.last_observed,
                              "metrics": self.metrics, "risk_by_slot": self.risk,
                              "failure_reviews": self.failures})
