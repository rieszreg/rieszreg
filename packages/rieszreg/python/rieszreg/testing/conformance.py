"""sklearn-conformance helpers for Riesz estimators.

These check the load-bearing properties of the public API: clone, get_params,
GridSearchCV, cross_val_predict, Pipeline composition. Implementation packages
import + run them against their concrete estimator.
"""

from __future__ import annotations

from typing import Callable

from sklearn.base import clone


def assert_clone_roundtrip(make_estimator: Callable):
    """`clone` produces an unfit copy with identical params."""
    est = make_estimator()
    twin = clone(est)
    assert type(twin) is type(est)
    assert twin.get_params(deep=False) == est.get_params(deep=False)
    # Cloned estimator must not carry fit-time attributes.
    for attr in ("predictor_", "best_iteration_", "loss_"):
        assert not hasattr(twin, attr), f"clone leaked fit attribute {attr!r}"


def assert_get_params_round_trip(make_estimator: Callable):
    """Rebuilding from `get_params()` gives the same params."""
    est = make_estimator()
    params = est.get_params(deep=False)
    twin = type(est)(**params)
    assert twin.get_params(deep=False) == params
