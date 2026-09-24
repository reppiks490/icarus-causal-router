"""Deterministic, synthetic checks for the research-only numerical core."""

import math
import unittest

from causal_router.learning import AdaptiveRegressor


def expected_weights(weights, members, reward, rate, share):
    raw = [
        weight * math.exp(-rate * ((reward - member) ** 2) / 4.0)
        for weight, member in zip(weights, members)
    ]
    total = sum(raw)
    return [(1.0 - share) * value / total + share / len(raw) for value in raw]


class AdaptiveRegressorTests(unittest.TestCase):
    def test_cold_start_and_defensive_prediction_snapshot(self):
        model = AdaptiveRegressor(1, half_lives=(1.0, 4.0))
        before = model.predict([0.5])
        self.assertEqual(before["mean"], 0.0)
        self.assertEqual(before["members"], [0.0, 0.0])
        self.assertEqual(before["weights"], [0.5, 0.5])
        self.assertEqual(before["count"], 0)
        self.assertEqual(before["uncertainty"], 1.0)
        before["weights"][0] = 0.0
        before["members"][0] = 1.0
        self.assertEqual(model.predict([0.5])["weights"], [0.5, 0.5])
        self.assertEqual(model.predict([0.5])["members"], [0.0, 0.0])

    def test_hand_calculated_ridge_and_intercept(self):
        model = AdaptiveRegressor(1, half_lives=(1.0,), ridge=1.0)
        model.update([1.0], 1.0, model.predict([1.0]))
        # G = [[1, 1], [1, 1]], b = [1, 1].
        # (G + I) beta = b gives beta = [1/3, 1/3].
        self.assertAlmostEqual(model.predict([1.0])["mean"], 2.0 / 3.0)
        self.assertAlmostEqual(model.predict([0.0])["mean"], 1.0 / 3.0)
        self.assertEqual(model.predict([1.0])["count"], 1)

        model.update([1.0], -1.0, model.predict([1.0]))
        # lambda = 1/2: G = 1.5 * xx', b = [-0.5, -0.5].
        # Ridge stays I (it is not discounted), so beta = [-1/8, -1/8].
        self.assertAlmostEqual(model.predict([1.0])["mean"], -0.25)
        self.assertAlmostEqual(model.predict([0.0])["mean"], -0.125)

    def test_fixed_share_normalizes_bounded_frozen_losses(self):
        rate, share = 2.0, 0.1
        model = AdaptiveRegressor(
            1, half_lives=(1.0, 8.0), learning_rate=rate, share=share
        )
        for reward in (1.0, -1.0, 1.0):
            model.update([1.0], reward, model.predict([1.0]))
        snapshot = model.predict([1.0])
        target = expected_weights(
            snapshot["weights"], snapshot["members"], -1.0, rate, share
        )
        model.update([1.0], -1.0, snapshot)
        observed = model.predict([1.0])["weights"]
        for actual, expected in zip(observed, target):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(observed), 1.0)
        self.assertTrue(all(share / 2.0 <= value <= 1.0 for value in observed))

    def test_delayed_update_uses_supplied_pre_outcome_members(self):
        model = AdaptiveRegressor(
            1, half_lives=(1.0, 16.0), learning_rate=3.0, share=0.0
        )
        for reward in (1.0, 1.0, -1.0):
            model.update([1.0], reward, model.predict([1.0]))
        old = model.predict([1.0])
        model.update([1.0], -1.0, old)
        current = model.predict([1.0])
        target = expected_weights(current["weights"], old["members"], 1.0, 3.0, 0.0)
        recomputed = expected_weights(
            current["weights"], current["members"], 1.0, 3.0, 0.0
        )
        self.assertGreater(abs(target[0] - recomputed[0]), 1e-5)
        model.update([1.0], 1.0, old)
        self.assertAlmostEqual(model.predict([1.0])["weights"][0], target[0])

    def test_short_horizon_adapts_faster_after_regime_change(self):
        model = AdaptiveRegressor(1, half_lives=(2.0, 64.0), ridge=1.0)
        for _ in range(24):
            model.update([1.0], 1.0, model.predict([1.0]))
        for _ in range(12):
            model.update([1.0], -1.0, model.predict([1.0]))
        prediction = model.predict([1.0])
        self.assertLess(prediction["members"][0], prediction["members"][1])
        self.assertLess(prediction["members"][0], 0.0)
        self.assertEqual(prediction["count"], 36)
        self.assertTrue(math.isfinite(prediction["uncertainty"]))
        self.assertGreaterEqual(prediction["uncertainty"], 0.0)
        self.assertLessEqual(prediction["uncertainty"], 1.0)

    def test_invalid_constructor_arguments(self):
        invalid = (
            {"dimension": 0}, {"dimension": 65}, {"dimension": True},
            {"half_lives": ()}, {"half_lives": (0.0,)},
            {"half_lives": (float("inf"),)}, {"half_lives": (float("nan"),)},
            {"ridge": 0.0}, {"ridge": float("inf")},
            {"learning_rate": -1.0}, {"learning_rate": float("nan")},
            {"share": -0.1}, {"share": 1.1}, {"share": float("nan")},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                AdaptiveRegressor(**({"dimension": 1} | kwargs))

    def test_invalid_features_and_reward_are_atomic(self):
        model = AdaptiveRegressor(2)
        for bad in ([0.0], [0.0, 0.0, 0.0], [1.01, 0.0],
                    [float("nan"), 0.0], [float("inf"), 0.0],
                    [True, 0.0], "0,0"):
            with self.subTest(features=bad), self.assertRaises((TypeError, ValueError)):
                model.predict(bad)
        snapshot = model.predict([0.0, 0.0])
        for bad in (-1.01, 1.01, float("nan"), float("inf"), True):
            with self.subTest(reward=bad), self.assertRaises((TypeError, ValueError)):
                model.update([0.0, 0.0], bad, snapshot)
        with self.assertRaises((TypeError, ValueError)):
            model.update([float("nan"), 0.0], 1.0, snapshot)
        self.assertEqual(model.predict([0.0, 0.0])["count"], 0)

    def test_invalid_frozen_prediction_does_not_mutate(self):
        model = AdaptiveRegressor(1, half_lives=(2.0, 4.0))
        valid = model.predict([1.0])
        for bad in (
            {}, {**valid, "members": [0.0]},
            {**valid, "members": [float("nan"), 0.0]},
            {**valid, "members": [1.1, 0.0]},
        ):
            with self.subTest(prediction=bad), self.assertRaises((TypeError, ValueError)):
                model.update([1.0], 1.0, bad)
        self.assertEqual(model.predict([1.0])["count"], 0)
        self.assertEqual(model.predict([1.0])["weights"], [0.5, 0.5])

    def test_long_sequence_remains_finite_and_normalized(self):
        model = AdaptiveRegressor(2)
        for index in range(300):
            x = [(-1.0 if index % 2 else 1.0), index % 7 / 7.0]
            reward = -1.0 if index % 3 else 1.0
            model.update(x, reward, model.predict(x))
        prediction = model.predict([1.0, -1.0])
        self.assertEqual(prediction["count"], 300)
        self.assertAlmostEqual(sum(prediction["weights"]), 1.0)
        self.assertTrue(all(math.isfinite(v) for v in prediction["members"]))
        self.assertTrue(math.isfinite(prediction["mean"]))

    def test_maximum_dimension_and_repeatability(self):
        models = [
            AdaptiveRegressor(64, half_lives=(1e6,), ridge=1e-3)
            for _ in range(2)
        ]
        features = [1.0] * 64
        for model in models:
            for _ in range(40):
                model.update(features, 1.0, model.predict(features))
        self.assertEqual(models[0].predict(features), models[1].predict(features))
        self.assertTrue(math.isfinite(models[0].predict(features)["mean"]))

    def test_unobserved_feature_direction_retains_uncertainty(self):
        model = AdaptiveRegressor(1, half_lives=(64.0,), ridge=1.0)
        for _ in range(40):
            model.update([0.0], 0.0, model.predict([0.0]))
        known = model.predict([0.0])["uncertainty"]
        unseen = model.predict([1.0])["uncertainty"]
        self.assertGreater(unseen, known + 0.15)
        self.assertGreater(unseen, 0.5)

    def test_residual_uncertainty_uses_frozen_prequential_error(self):
        ordinary = AdaptiveRegressor(1, half_lives=(2.0,), ridge=1.0)
        surprising = AdaptiveRegressor(1, half_lives=(2.0,), ridge=1.0)
        for model in (ordinary, surprising):
            for _ in range(20):
                model.update([0.0], 0.0, model.predict([0.0]))
        frozen = ordinary.predict([0.0])
        ordinary.update([0.0], 1.0, frozen)
        surprising.update([0.0], 1.0, {**frozen, "members": [-1.0]})
        self.assertEqual(ordinary.predict([0.0])["mean"], surprising.predict([0.0])["mean"])
        self.assertGreater(
            surprising.predict([0.0])["uncertainty"],
            ordinary.predict([0.0])["uncertainty"] + 0.05,
        )

    def test_explicit_decision_clock_decays_without_feedback(self):
        model = AdaptiveRegressor(1, half_lives=(1.0,), ridge=1.0)
        frozen = model.predict([0.0], step=1)
        self.assertEqual(frozen["step"], 1)
        model.update([0.0], 1.0, frozen)
        self.assertAlmostEqual(model.predict([0.0], step=1)["mean"], 0.5)
        self.assertAlmostEqual(model.predict([0.0], step=2)["mean"], 1.0 / 3.0)
        self.assertLess(model.predict([0.0], step=10)["mean"], 0.01)

    def test_late_old_label_has_less_effect_than_recent_label(self):
        old = AdaptiveRegressor(1, half_lives=(1.0,), ridge=1.0)
        recent = AdaptiveRegressor(1, half_lives=(1.0,), ridge=1.0)
        old_snapshot = old.predict([0.0], step=1)
        old.predict([0.0], step=10)
        recent_snapshot = recent.predict([0.0], step=10)
        old.update([0.0], 1.0, old_snapshot)
        recent.update([0.0], 1.0, recent_snapshot)
        self.assertLess(old.predict([0.0], step=10)["mean"], 0.01)
        self.assertAlmostEqual(recent.predict([0.0], step=10)["mean"], 0.5)

    def test_late_outcome_order_does_not_change_ridge_members(self):
        models = [AdaptiveRegressor(1, half_lives=(2.0, 8.0)) for _ in range(2)]
        snapshots = []
        for model in models:
            a = model.predict([1.0], step=2)
            b = model.predict([-1.0], step=2)
            model.predict([0.0], step=12)
            snapshots.append((a, b))
        a, b = snapshots[0]
        models[0].update([1.0], 1.0, a)
        models[0].update([-1.0], -1.0, b)
        a, b = snapshots[1]
        models[1].update([-1.0], -1.0, b)
        models[1].update([1.0], 1.0, a)
        for x in ([0.0], [1.0], [-1.0]):
            for left, right in zip(
                models[0].predict(x, step=12)["members"],
                models[1].predict(x, step=12)["members"],
            ):
                self.assertAlmostEqual(left, right)

    def test_receipt_does_not_decay_existing_stats_again(self):
        model = AdaptiveRegressor(1, half_lives=(1.0,), ridge=1.0)
        frozen = model.predict([0.0], step=1)
        model.predict([0.0], step=10)
        model.update([0.0], 1.0, frozen)
        model.update([0.0], 1.0, frozen)
        contribution = 2.0 * 2.0 ** -9
        self.assertAlmostEqual(
            model.predict([0.0], step=10)["mean"], contribution / (1.0 + contribution)
        )

    def test_decision_clock_rejects_invalid_or_backward_step(self):
        model = AdaptiveRegressor(1)
        for bad in (-1, True, 1.5, float("nan")):
            with self.subTest(step=bad), self.assertRaises((TypeError, ValueError)):
                model.predict([0.0], step=bad)
        model.predict([0.0], step=5)
        with self.assertRaises(ValueError):
            model.predict([0.0], step=4)


if __name__ == "__main__":
    unittest.main()
