"""Dispatch between augmentation-style and moment-style backends.

`RieszEstimator.fit` calls `fit_rows` for moment-only backends; otherwise builds an
`AugmentedDataset` and calls `fit_augmented`. Backends implementing both
methods default to `fit_augmented` for back-compat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from rieszreg import (
    ATE,
    AugmentedDataset,
    FitResult,
    RieszEstimator,
    SquaredLoss,
)


class _ConstPredictor:
    kind = "dispatch-test"

    def __init__(self, value: float = 0.0):
        self.value = float(value)

    def predict_eta(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.value, dtype=float)

    def predict_alpha(self, features: np.ndarray) -> np.ndarray:
        return self.predict_eta(features)

    def save(self, dir_path) -> None:  # pragma: no cover - not used
        pass


@dataclass
class _AugOnlyBackend:
    calls: list[str] = field(default_factory=list)
    last_aug: AugmentedDataset | None = None

    def fit_augmented(self, aug_train, aug_valid, loss, **kwargs) -> FitResult:
        self.calls.append("fit_augmented")
        self.last_aug = aug_train
        return FitResult(predictor=_ConstPredictor(0.0))


@dataclass
class _MomentOnlyBackend:
    calls: list[str] = field(default_factory=list)
    last_X: np.ndarray | None = None
    last_aug: AugmentedDataset | None = None
    last_estimand_name: str | None = None

    def fit_rows(self, X_train, X_valid, estimand, loss, *, aug_train, aug_valid, **kwargs) -> FitResult:
        self.calls.append("fit_rows")
        self.last_X = X_train
        self.last_aug = aug_train
        self.last_estimand_name = estimand.name
        return FitResult(predictor=_ConstPredictor(0.0))


@dataclass
class _BothBackend:
    calls: list[str] = field(default_factory=list)

    def fit_augmented(self, aug_train, aug_valid, loss, **kwargs) -> FitResult:
        self.calls.append("fit_augmented")
        return FitResult(predictor=_ConstPredictor(0.0))

    def fit_rows(self, X_train, X_valid, estimand, loss, **kwargs) -> FitResult:
        self.calls.append("fit_rows")
        return FitResult(predictor=_ConstPredictor(0.0))


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 50
    return pd.DataFrame({"a": (rng.uniform(size=n) > 0.5).astype(float),
                         "x": rng.normal(size=n)})


def test_aug_only_backend_routes_to_fit_augmented(df):
    backend = _AugOnlyBackend()
    est = RieszEstimator(estimand=ATE(), backend=backend, loss=SquaredLoss())
    est.fit(df)
    assert backend.calls == ["fit_augmented"]
    assert backend.last_aug is not None
    # ATE on n=50 produces 3 augmented rows per original (orig + 2 counterfactuals).
    assert backend.last_aug.n_rows == 50


def test_moment_only_backend_routes_to_fit_rows(df):
    backend = _MomentOnlyBackend()
    est = RieszEstimator(estimand=ATE(), backend=backend, loss=SquaredLoss())
    est.fit(df)
    assert backend.calls == ["fit_rows"]
    assert backend.last_estimand_name == "ATE"
    # Columns arrive in feature_keys order (treatment first).
    np.testing.assert_array_equal(backend.last_X, df[["a", "x"]].to_numpy())


def test_backend_implementing_both_defaults_to_fit_augmented(df):
    backend = _BothBackend()
    est = RieszEstimator(estimand=ATE(), backend=backend, loss=SquaredLoss())
    est.fit(df)
    assert backend.calls == ["fit_augmented"]


def test_moment_path_passes_validation_rows(df):
    backend = _MomentOnlyBackend()
    backend.validation_fraction = 0.2
    est = RieszEstimator(
        estimand=ATE(),
        backend=backend,
        loss=SquaredLoss(),
    )
    est.fit(df)
    assert backend.calls == ["fit_rows"]
    # With backend.validation_fraction=0.2, train has 40 rows, valid has 10.
    assert backend.last_X.shape == (40, 2)
    assert backend.last_aug.n_rows == 40
