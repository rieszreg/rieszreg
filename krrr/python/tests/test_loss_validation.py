"""KernelRidgeBackend supports SquaredLoss only; non-quadratic losses raise."""

from __future__ import annotations

import numpy as np
import pytest

from rieszreg import (
    ATE,
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    SquaredLoss,
    TSM,
)

from krrr import KernelRieszRegressor


@pytest.mark.parametrize(
    "unsupported_loss",
    [KLLoss(), BernoulliLoss(), BoundedSquaredLoss(lo=0.0, hi=1.0)],
)
def test_unsupported_losses_raise_at_fit(binary_ate_data, unsupported_loss):
    """Fitting with KL / Bernoulli / BoundedSquared raises NotImplementedError."""
    df, _, _ = binary_ate_data
    krr = KernelRieszRegressor(
        estimand=TSM(level=1) if unsupported_loss.name in ("kl", "bernoulli") else ATE(),
        loss=unsupported_loss,
        lambda_grid=np.logspace(-3, 0, 4),
        validation_fraction=0.25,
    )
    with pytest.raises(NotImplementedError, match="SquaredLoss"):
        krr.fit(df)


def test_squared_loss_default(binary_ate_data):
    """No `loss=` defaults to SquaredLoss and fits cleanly."""
    df, _, _ = binary_ate_data
    krr = KernelRieszRegressor(
        estimand=ATE(),
        lambda_grid=np.logspace(-3, 0, 4),
        validation_fraction=0.25,
    ).fit(df)
    assert isinstance(krr.loss_, SquaredLoss)
