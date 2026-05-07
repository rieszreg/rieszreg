"""All four built-in losses are supported. A bogus loss spec raises NotImplementedError."""

from __future__ import annotations

import functools

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

from riesznet import RieszNet
from riesznet.losses_torch import is_supported, validate_supported


def test_squared_supported():
    assert is_supported(SquaredLoss())


def test_kl_supported():
    assert is_supported(KLLoss(max_eta=10.0))


def test_bernoulli_supported():
    assert is_supported(BernoulliLoss(max_abs_eta=10.0))


def test_bounded_squared_supported():
    assert is_supported(BoundedSquaredLoss(lo=0.1, hi=5.0, max_abs_eta=10.0))


class _DummyLoss:
    name = "dummy"

    def to_spec(self):
        return {"type": "DummyLoss", "args": {}}

    def link_to_alpha(self, eta):
        return eta

    def alpha_to_eta(self, alpha):
        return alpha

    def loss_row(self, a, b, alpha):
        return a * alpha**2 + b * alpha

    def gradient(self, a, b, eta):
        return 2.0 * a * eta + b

    def hessian(self, a, b, eta, hessian_floor):
        return np.maximum(2.0 * a, hessian_floor)

    def best_constant_init(self, m_bar):
        return float(m_bar)

    def validate_coefficients(self, b):
        return

    def link(self, eta):
        return eta


def test_unsupported_loss_raises(linear_gaussian_ate_df):
    est = RieszNet(
        estimand=ATE(),
        hidden_sizes=(4,),
        epochs=2,
        loss=_DummyLoss(),
        random_state=0,
    )
    with pytest.raises(NotImplementedError, match="DummyLoss"):
        est.fit(linear_gaussian_ate_df)


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
