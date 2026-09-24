"""Bounded, deterministic delayed-feedback regression for research routing.

Predictions are clipped to [-1, 1]; out-of-range rewards are rejected.
``uncertainty`` combines ridge leverage, frozen prequential residual error,
horizon disagreement, and a cold-start term. It has no statistical coverage,
trading, or profitability guarantee.
"""

from __future__ import annotations

import math


def _bounded_number(value: object, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return number


def _solve_ridge(
    gram: list[list[float]], target: list[float], ridge: float, features: list[float]
) -> tuple[list[float], float]:
    """Solve ridge coefficients and z'(G + ridge I)^-1 z by Cholesky."""
    size = len(target)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for col in range(row + 1):
            residual = gram[row][col] + (ridge if row == col else 0.0)
            residual -= math.fsum(lower[row][k] * lower[col][k] for k in range(col))
            if row == col:
                if not math.isfinite(residual) or residual <= 0.0:
                    raise ArithmeticError("ridge system is not numerically positive definite")
                lower[row][col] = math.sqrt(residual)
            else:
                lower[row][col] = residual / lower[col][col]

    forward = [0.0] * size
    for row in range(size):
        prior = math.fsum(lower[row][col] * forward[col] for col in range(row))
        forward[row] = (target[row] - prior) / lower[row][row]

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        later = math.fsum(lower[col][row] * solution[col] for col in range(row + 1, size))
        solution[row] = (forward[row] - later) / lower[row][row]
    whitened = [0.0] * size
    for row in range(size):
        prior = math.fsum(lower[row][col] * whitened[col] for col in range(row))
        whitened[row] = (features[row] - prior) / lower[row][row]
    leverage = math.fsum(value * value for value in whitened)
    if not math.isfinite(leverage) or leverage < 0.0:
        raise ArithmeticError("non-finite ridge leverage")
    return solution, leverage


class AdaptiveRegressor:
    """Multi-timescale discounted ridge with fixed-share exponential weights.

    Each horizon ages G and b by decision-clock distance, then adds
    ``decay**label_age * z z'`` and ``decay**label_age * z * reward``.
    Here ``z = [1, *x]``. A fixed ridge
    penalty is added only when solving, so old data decay but the penalty does
    not. Predictions are clipped to [-1, 1] before scoring. Weight losses are
    squared errors divided by four, hence bounded in [0, 1].

    Dimension is 1..64; 1..16 half-lives are allowed in [1, 1e6]. Ridge is
    in [1e-3, 1e6], learning_rate in (0, 10], and share in [0, 1].
    All observations and rewards must be finite and in [-1, 1].
    """

    def __init__(
        self,
        dimension: int,
        half_lives: tuple[float, ...] = (16.0, 64.0, 256.0),
        ridge: float = 1.0,
        learning_rate: float = 0.5,
        share: float = 0.02,
    ) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("dimension must be an integer")
        if not 1 <= dimension <= 64:
            raise ValueError("dimension must be in [1, 64]")
        if not isinstance(half_lives, (tuple, list)) or not 1 <= len(half_lives) <= 16:
            raise ValueError("half_lives must contain 1..16 values")

        self.dimension = dimension
        self.half_lives = tuple(
            _bounded_number(value, "half_life", 1.0, 1e6) for value in half_lives
        )
        self.ridge = _bounded_number(ridge, "ridge", 1e-3, 1e6)
        self.learning_rate = _bounded_number(learning_rate, "learning_rate", 0.0, 10.0)
        if self.learning_rate == 0.0:
            raise ValueError("learning_rate must be positive")
        self.share = _bounded_number(share, "share", 0.0, 1.0)

        self._decays = tuple(2.0 ** (-1.0 / value) for value in self.half_lives)
        size = dimension + 1
        self._grams = [
            [[0.0] * size for _ in range(size)] for _ in self.half_lives
        ]
        self._targets = [[0.0] * size for _ in self.half_lives]
        self._residual_ewma = [0.0] * len(self.half_lives)
        self._weights = [1.0 / len(self.half_lives)] * len(self.half_lives)
        self._count = 0
        self._clock = 0
        self._explicit_clock = False

    def _advance(self, step: int) -> None:
        if type(step) is not int or not self._clock <= step <= 2**53-1:
            raise ValueError("step must be a nonnegative monotone integer")
        gap = step - self._clock
        if gap:
            for h, (decay, gram, target) in enumerate(zip(self._decays, self._grams, self._targets)):
                factor = decay ** gap
                for row in range(len(target)):
                    target[row] *= factor
                    for col in range(len(target)):
                        gram[row][col] *= factor
                self._residual_ewma[h] *= factor
        self._clock = step

    def _features(self, x: list[float]) -> list[float]:
        if not isinstance(x, (list, tuple)) or len(x) != self.dimension:
            raise ValueError(f"x must contain exactly {self.dimension} features")
        return [1.0] + [
            _bounded_number(value, f"x[{index}]", -1.0, 1.0)
            for index, value in enumerate(x)
        ]

    def predict(self, x: list[float], step: int | None = None) -> dict:
        """Explicit steps age data by decisions; omitted steps only inspect.

        If explicit steps are never supplied, every update advances one implicit
        step and treats the label as current. Delayed callers must use explicit
        steps (as Engine does). Scores remain heuristics, not confidence bounds.
        """
        features = self._features(x)
        if step is not None:
            self._advance(step)
            self._explicit_clock = True
        members: list[float] = []
        leverage_scores: list[float] = []
        for gram, target in zip(self._grams, self._targets):
            beta, leverage = _solve_ridge(gram, target, self.ridge, features)
            raw = math.fsum(coefficient * value for coefficient, value in zip(beta, features))
            if not math.isfinite(raw):
                raise ArithmeticError("non-finite ridge prediction")
            members.append(max(-1.0, min(1.0, raw)))
            leverage_scores.append(leverage / (1.0 + leverage))

        mean = math.fsum(weight * value for weight, value in zip(self._weights, members))
        mean = max(-1.0, min(1.0, mean))
        disagreement = math.fsum(
            weight * (value - mean) ** 2
            for weight, value in zip(self._weights, members)
        )
        leverage_term = math.fsum(
            weight * value for weight, value in zip(self._weights, leverage_scores)
        )
        residual_term = math.fsum(
            weight * value for weight, value in zip(self._weights, self._residual_ewma)
        )
        uncertainty = math.sqrt(min(
            1.0, disagreement + leverage_term + residual_term + 1.0 / (self._count + 1)
        ))
        return {
            "mean": mean,
            "uncertainty": uncertainty,
            "members": members,
            "weights": self._weights.copy(),
            "count": self._count,
            "step": self._clock,
        }

    def update(self, x: list[float], reward: float, prediction: dict) -> None:
        features = self._features(x)
        observed = _bounded_number(reward, "reward", -1.0, 1.0)
        if not isinstance(prediction, dict):
            raise TypeError("prediction must be a pre-outcome prediction dictionary")
        members = prediction.get("members")
        if not isinstance(members, (tuple, list)) or len(members) != len(self._weights):
            raise ValueError("prediction members do not match the ensemble")
        frozen = [
            _bounded_number(value, f"prediction.members[{index}]", -1.0, 1.0)
            for index, value in enumerate(members)
        ]
        recorded_count = prediction.get("count")
        if (isinstance(recorded_count, bool) or not isinstance(recorded_count, int)
                or not 0 <= recorded_count <= self._count):
            raise ValueError("prediction count is not a valid past snapshot")
        recorded_step = prediction.get("step")
        if type(recorded_step) is not int or not 0 <= recorded_step <= self._clock:
            raise ValueError("prediction step is not a valid past snapshot")

        losses = [(observed - value) ** 2 / 4.0 for value in frozen]
        scores = [
            weight * math.exp(-self.learning_rate * loss)
            for weight, loss in zip(self._weights, losses)
        ]
        total = math.fsum(scores)
        if not math.isfinite(total) or total <= 0.0:
            raise ArithmeticError("ensemble weight normalization failed")
        count = len(scores)
        new_weights = [
            (1.0 - self.share) * (score / total) + self.share / count
            for score in scores
        ]
        norm = math.fsum(new_weights)
        self._weights = [weight / norm for weight in new_weights]

        if not self._explicit_clock:
            self._advance(self._clock + 1)
            recorded_step = self._clock

        size = len(features)
        for h, (decay, gram, target) in enumerate(zip(self._decays, self._grams, self._targets)):
            age_weight = decay ** (self._clock - recorded_step)
            self._residual_ewma[h] += age_weight * (1.-decay) * losses[h]
            for row in range(size):
                target[row] += age_weight * features[row] * observed
                for col in range(row + 1):
                    value = gram[row][col] + age_weight * features[row] * features[col]
                    gram[row][col] = value
                    gram[col][row] = value
        self._count += 1
