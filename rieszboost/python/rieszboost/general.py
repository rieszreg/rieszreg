"""Slow general path: Friedman gradient boosting on the augmented dataset with
arbitrary sklearn-compatible base learners.

Each round:
  1. residual r_j = -(2 a_j F_j + b_j)
  2. fit base_learner.fit(X_aug, r)
  3. line search: γ* = Σ r_j h_j / (2 Σ a_j h_j²)
  4. F += learning_rate · γ* · h(X_aug)

Supports the same finite-point linear-functional class the fast engine does,
but with any base learner exposing .fit(X, y) / .predict(X). Slower than the
xgboost path but lets you swap in kernel ridge, random forests, MLPs, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .engine import build_augmented
from .tracer import trace


@dataclass
class GeneralRieszBooster:
    learners: list
    steps: list[float]            # learning_rate · γ_round, one per round
    base_score: float
    feature_keys: tuple[str, ...]
    best_iteration: int | None = None
    best_score: float | None = None
    history: list[float] = field(default_factory=list)

    def _end_index(self) -> int:
        if self.best_iteration is not None:
            return self.best_iteration + 1
        return len(self.learners)

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.base_score)
        for h, step in zip(self.learners[: self._end_index()], self.steps[: self._end_index()]):
            F = F + step * np.asarray(h.predict(X))
        return F

    def predict(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        X = np.asarray(
            [[row[k] for k in self.feature_keys] for row in rows], dtype=float
        )
        return self.predict_array(X)

    def riesz_loss(self, rows: Sequence[dict[str, Any]], m: Callable) -> float:
        aug = build_augmented(rows, m, self.feature_keys)
        F = self.predict_array(aug.features)
        return float(np.sum(aug.a * F**2 + aug.b * F) / len(rows))


def _line_search_step(a: np.ndarray, h: np.ndarray, r: np.ndarray) -> float:
    """Closed-form γ* minimizing L(F + γ h) given residual r = -(2aF + b)."""
    denom = float(2.0 * np.sum(a * h * h))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(r * h) / denom)


def general_fit(
    rows: Sequence[dict[str, Any]],
    m: Callable,
    feature_keys: Sequence[str],
    *,
    base_learner: Callable[[], Any],
    valid_rows: Sequence[dict[str, Any]] | None = None,
    num_boost_round: int = 100,
    early_stopping_rounds: int | None = None,
    learning_rate: float = 0.1,
    init: str | float = 0.0,
) -> GeneralRieszBooster:
    """Friedman MART on the augmented dataset with `base_learner()` as the
    weak-learner factory (returns a fresh sklearn-compatible estimator)."""
    aug = build_augmented(rows, m, feature_keys)

    if init == "m1":
        per_row = [sum(c for c, _ in trace(m, z)) for z in rows]
        base_score = float(np.mean(per_row))
    elif isinstance(init, (int, float)):
        base_score = float(init)
    else:
        raise ValueError(f"init must be 0, a float, or 'm1'; got {init!r}")

    F = np.full(aug.features.shape[0], base_score)

    have_valid = valid_rows is not None
    if early_stopping_rounds is not None and not have_valid:
        raise ValueError("early_stopping_rounds requires valid_rows to be provided")

    if have_valid:
        aug_val = build_augmented(valid_rows, m, feature_keys)
        F_val = np.full(aug_val.features.shape[0], base_score)
        n_val = len(valid_rows)
    else:
        aug_val = None
        F_val = None
        n_val = 0

    learners: list = []
    steps: list[float] = []
    history: list[float] = []
    best_score = float("inf")
    best_iter: int | None = None
    no_improve = 0

    for it in range(num_boost_round):
        residual = -(2.0 * aug.a * F + aug.b)
        learner = base_learner()
        learner.fit(aug.features, residual)
        h_train = np.asarray(learner.predict(aug.features))

        gamma = _line_search_step(aug.a, h_train, residual)
        step = learning_rate * gamma
        F = F + step * h_train

        learners.append(learner)
        steps.append(step)

        if have_valid:
            h_val = np.asarray(learner.predict(aug_val.features))
            F_val = F_val + step * h_val
            val_loss = float(np.sum(aug_val.a * F_val**2 + aug_val.b * F_val) / n_val)
            history.append(val_loss)
            if val_loss < best_score - 1e-12:
                best_score = val_loss
                best_iter = it
                no_improve = 0
            else:
                no_improve += 1
            if early_stopping_rounds is not None and no_improve >= early_stopping_rounds:
                break

    return GeneralRieszBooster(
        learners=learners,
        steps=steps,
        base_score=base_score,
        feature_keys=tuple(feature_keys),
        best_iteration=best_iter,
        best_score=best_score if best_iter is not None else None,
        history=history,
    )
