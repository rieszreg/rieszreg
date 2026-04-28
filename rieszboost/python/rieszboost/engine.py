"""Fast engine: data augmentation + xgboost custom objective.

For each training row i, the user's m extracts a finite list of (coefficient,
point) pairs. We assemble an augmented dataset where every row j contributes a
loss term

    a_j * alpha(z_j)^2 + b_j * alpha(z_j)

with gradient 2*a_j*F_j + b_j and Hessian 2*a_j. The original row i contributes
(a=1, b=0) at point z_i (the alpha^2 term in the Riesz loss); each pair (c, p)
from m(z_i) contributes (a=0, b=-2c) at point p. Duplicate points within a row
are merged by summing (a, b).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import xgboost as xgb

from .tracer import trace


@dataclass
class AugmentedDataset:
    features: np.ndarray  # shape (n_aug, n_features)
    a: np.ndarray         # shape (n_aug,) — quadratic coefficient
    b: np.ndarray         # shape (n_aug,) — linear coefficient
    origin_index: np.ndarray  # shape (n_aug,) — index into original rows


def _row_to_features(point: dict[str, Any], feature_keys: Sequence[str]) -> np.ndarray:
    return np.asarray([point[k] for k in feature_keys], dtype=float)


def build_augmented(
    rows: Sequence[dict[str, Any]],
    m: Callable,
    feature_keys: Sequence[str],
) -> AugmentedDataset:
    """Trace m on each row and assemble the augmented (features, a, b) arrays."""
    feats: list[np.ndarray] = []
    a_list: list[float] = []
    b_list: list[float] = []
    origin: list[int] = []

    for i, z in enumerate(rows):
        # Per-row accumulator: point_key -> (a, b)
        acc: dict[tuple, tuple[float, float]] = {}
        # Original row contributes the alpha^2 term at z itself.
        z_pt = {k: z[k] for k in feature_keys}
        z_key = tuple(z_pt[k] for k in feature_keys)
        acc[z_key] = (1.0, 0.0)

        # Linear functional contributes -2c * alpha(p) for each (c, p).
        for coef, point in trace(m, z):
            missing = [k for k in feature_keys if k not in point]
            if missing:
                raise ValueError(
                    f"m evaluated alpha at a point missing keys {missing}; "
                    f"all feature_keys {list(feature_keys)} must be specified."
                )
            key = tuple(point[k] for k in feature_keys)
            cur_a, cur_b = acc.get(key, (0.0, 0.0))
            acc[key] = (cur_a, cur_b - 2.0 * coef)

        for key, (a, b) in acc.items():
            feats.append(np.asarray(key, dtype=float))
            a_list.append(a)
            b_list.append(b)
            origin.append(i)

    return AugmentedDataset(
        features=np.vstack(feats) if feats else np.zeros((0, len(feature_keys))),
        a=np.asarray(a_list, dtype=float),
        b=np.asarray(b_list, dtype=float),
        origin_index=np.asarray(origin, dtype=np.int64),
    )


def _make_objective(a: np.ndarray, b: np.ndarray, eps: float = 1e-6):
    """xgboost custom objective. Gradient = 2a*F + b; Hessian = 2a (with eps
    floor so leaves with only b-rows get a finite second-order step)."""
    hess = 2.0 * a + eps

    def obj(preds: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        del dtrain
        grad = 2.0 * a * preds + b
        return grad, hess

    return obj


def fit(
    rows: Sequence[dict[str, Any]],
    m: Callable,
    feature_keys: Sequence[str],
    *,
    num_boost_round: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    reg_lambda: float = 1.0,
    subsample: float = 1.0,
    base_score: float = 0.0,
    seed: int = 0,
    init: str | float = 0.0,
) -> "RieszBooster":
    """Fit a Riesz representer to the user's m via the fast augmented-data path."""
    aug = build_augmented(rows, m, feature_keys)

    # Initialization: 0 (default) or 'm1' (alpha = m(z, 1)).
    if init == "m1":
        # Trace m on the first row with alpha returning 1, then... actually
        # init='m1' for Riesz representer means setting alpha_0 = sum of
        # coefficients from m. Computed per-row at predict time would be
        # ideal; for boosting we set base_score to mean over rows of sum(c).
        per_row = []
        for z in rows:
            per_row.append(sum(c for c, _ in trace(m, z)))
        base_score = float(np.mean(per_row))
    elif isinstance(init, (int, float)):
        base_score = float(init)
    else:
        raise ValueError(f"init must be 0, a float, or 'm1'; got {init!r}")

    dtrain = xgb.DMatrix(aug.features)
    params = {
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "reg_lambda": reg_lambda,
        "subsample": subsample,
        "base_score": base_score,
        "seed": seed,
        "disable_default_eval_metric": 1,
    }
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        obj=_make_objective(aug.a, aug.b),
    )
    return RieszBooster(
        booster=booster,
        feature_keys=tuple(feature_keys),
        base_score=base_score,
    )


@dataclass
class RieszBooster:
    booster: xgb.Booster
    feature_keys: tuple[str, ...]
    base_score: float

    def predict(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        X = np.asarray(
            [[row[k] for k in self.feature_keys] for row in rows], dtype=float
        )
        return self.booster.predict(xgb.DMatrix(X))

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(xgb.DMatrix(np.asarray(X, dtype=float)))

    def riesz_loss(
        self,
        rows: Sequence[dict[str, Any]],
        m: Callable,
    ) -> float:
        """Empirical Riesz loss alpha(z)^2 - 2*m(z, alpha) on rows."""
        preds = self.predict(rows)
        m_vals = np.zeros(len(rows))
        for i, z in enumerate(rows):
            for coef, point in trace(m, z):
                m_vals[i] += coef * float(
                    self.predict_array(
                        np.asarray([[point[k] for k in self.feature_keys]])
                    )[0]
                )
        return float(np.mean(preds**2 - 2.0 * m_vals))
