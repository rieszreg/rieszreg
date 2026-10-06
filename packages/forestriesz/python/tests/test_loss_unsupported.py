"""Non-quadratic Bregman losses raise NotImplementedError with a clear message."""

from __future__ import annotations

import pytest

from rieszreg import BoundedSquaredLoss, KLLoss

from forestriesz import ForestRieszRegressor, TSM


@pytest.mark.parametrize(
    "loss",
    [KLLoss(), BoundedSquaredLoss(lo=-1.0, hi=10.0)],
)
def test_non_squared_loss_raises(loss, logistic_tsm_df):
    est = ForestRieszRegressor(
        estimand=TSM(level=1),
        loss=loss,
        n_estimators=10,
        random_state=0,
    )
    with pytest.raises(NotImplementedError, match="SquaredLoss only"):
        est.fit(logistic_tsm_df)


def test_constant_basis_error_mentions_sieve(small_df):
    """Forcing the constant basis on ATE triggers the row-constant check."""
    from forestriesz import ATE

    est = ForestRieszRegressor(
        estimand=ATE(),
        riesz_feature_fns=None,    # force constant; default would have used "auto"
        n_estimators=10,
        random_state=0,
    )
    with pytest.raises(ValueError, match="AugForestRieszRegressor"):
        est.fit(small_df)
