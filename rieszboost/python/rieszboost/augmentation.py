"""Trace `m` on each row and build the augmented dataset for the squared
Riesz loss. Each augmented row j contributes a per-row loss term

    a_j · ψ(α(z̃_j)) + (b_j / 2) · φ'(α(z̃_j))

(the squared-loss case ψ = t², φ' = 2t simplifies to `a_j F² + b_j F`).
The original row contributes (a=1, b=0) at z_i; each (coef, point) pair
from m(z_i) contributes (a=0, b=-2·coef) at the point. Duplicate points
within a row are merged by summing (a, b).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .estimand import Estimand
from .tracer import trace


@dataclass
class AugmentedDataset:
    features: np.ndarray   # (n_aug, n_features)
    a: np.ndarray          # (n_aug,) — quadratic coefficient
    b: np.ndarray          # (n_aug,) — linear coefficient
    origin_index: np.ndarray  # (n_aug,) — index into original rows
    n_rows: int            # number of original rows


def build_augmented(
    rows: Sequence[dict[str, Any]],
    estimand: Estimand,
) -> AugmentedDataset:
    feature_keys = estimand.feature_keys

    feats: list[np.ndarray] = []
    a_list: list[float] = []
    b_list: list[float] = []
    origin: list[int] = []

    for i, z in enumerate(rows):
        acc: dict[tuple, tuple[float, float]] = {}
        z_key = tuple(z[k] for k in feature_keys)
        acc[z_key] = (1.0, 0.0)

        for coef, point in trace(estimand, z):
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
        n_rows=len(rows),
    )
