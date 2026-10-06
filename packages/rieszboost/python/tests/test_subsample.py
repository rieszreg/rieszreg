"""``XGBoostBackend(subsample=...)`` samples individuals, not augmented rows:
each round keeps or drops all of an individual's augmented rows together."""

from __future__ import annotations

import numpy as np
import pytest

from rieszboost import RieszBooster
from rieszboost.backends.xgboost import _make_objective
from rieszreg import ATE, SquaredLoss
from rieszreg.testing import dgps


def test_subsample_keeps_or_drops_each_individuals_rows_together():
    """ATE puts individual i at augmented rows i and n + i. Over many rounds
    both rows share one draw, and about `subsample` of individuals survive."""
    n, frac = 2000, 0.3
    rng = np.random.default_rng(0)
    feats = np.column_stack([rng.integers(0, 2, n).astype(float), rng.normal(size=n)])
    aug = ATE().augment(feats)
    obj = _make_objective(aug, SquaredLoss(), hessian_floor="auto",
                          gradient_only=False, subsample=frac, seed=0)
    preds = rng.normal(size=len(aug.features))

    kept_frac = []
    for _ in range(20):
        grad, hess = obj(preds, None)
        kept = hess > 0  # "auto" floors every row's Hessian above 0
        per_individual = np.zeros(n)
        np.add.at(per_individual, aug.origin_index, kept)
        assert set(np.unique(per_individual)) <= {0.0, 2.0}
        assert np.all(grad[~kept] == 0)
        kept_frac.append(np.mean(per_individual == 2.0))
    assert abs(np.mean(kept_frac) - frac) < 0.01


def test_subsample_recovers_ate_representer():
    dgp = dgps.linear_gaussian_ate()
    train = dgp.sample(2000, np.random.default_rng(0))
    test = dgp.sample(5000, np.random.default_rng(10))
    alpha0 = dgp.true_alpha(test)

    def rel_rmse(subsample):
        alpha = RieszBooster(ATE(), n_estimators=400, subsample=subsample).fit(train).predict(test)
        return np.sqrt(np.mean((alpha - alpha0) ** 2)) / np.sqrt(np.mean(alpha0**2))

    full, sub = rel_rmse(1.0), rel_rmse(0.5)
    assert sub < 0.3
    assert sub < 1.25 * full


def test_subsample_outside_unit_interval_raises():
    df = dgps.linear_gaussian_ate().sample(200, np.random.default_rng(0))
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="subsample must be in"):
            RieszBooster(ATE(), subsample=bad, n_estimators=5).fit(df)
