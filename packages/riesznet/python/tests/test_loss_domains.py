"""Each non-squared built-in loss trains and keeps α̂ inside its link's range."""

from __future__ import annotations

import numpy as np

from rieszreg import (
    ATE,
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    TSM,
)

from riesznet import RieszNet


def test_kl_loss_runs_on_tsm(logistic_tsm_df):
    est = RieszNet(
        estimand=TSM(level=1),
        hidden_sizes=(8,),
        epochs=10,
        loss=KLLoss(max_eta=10.0),
        random_state=0,
    )
    est.fit(logistic_tsm_df)
    pred = est.predict(logistic_tsm_df)
    assert pred.shape == (len(logistic_tsm_df),)
    # KL link is exp → α > 0.
    assert np.all(pred > 0)


def test_bernoulli_loss_runs_on_tsm(logistic_tsm_df):
    est = RieszNet(
        estimand=TSM(level=1),
        hidden_sizes=(8,),
        epochs=10,
        loss=BernoulliLoss(max_abs_eta=10.0),
        random_state=0,
    )
    est.fit(logistic_tsm_df)
    pred = est.predict(logistic_tsm_df)
    # Sigmoid link → α ∈ (0, 1).
    assert np.all((pred > 0.0) & (pred < 1.0))


def test_bounded_squared_runs_on_ate(linear_gaussian_ate_df):
    est = RieszNet(
        estimand=ATE(),
        hidden_sizes=(8,),
        epochs=10,
        loss=BoundedSquaredLoss(lo=-10.0, hi=10.0, max_abs_eta=10.0),
        random_state=0,
    )
    est.fit(linear_gaussian_ate_df)
    pred = est.predict(linear_gaussian_ate_df)
    assert np.all((pred > -10.0) & (pred < 10.0))
