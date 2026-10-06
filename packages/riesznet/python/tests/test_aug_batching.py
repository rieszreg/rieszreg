"""The neural backend's per-row loss, built from ``estimand.augment``, is the
trace-based Riesz loss; CSR minibatching partitions it exactly; and the
reported validation score is the numpy Riesz loss."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from rieszreg import (
    ATE,
    FiniteEvalEstimand,
    KLLoss,
    SquaredLoss,
    TSM,
    trace,
)
from riesznet import RieszNet
from riesznet.backend import _AugTensors, _batch_loss, _row_batches
from riesznet.losses_torch import TorchRieszLoss
from riesznet.modules import build_mlp


def _custom_shift():
    # m(α)(z) = α(a + 1, x) − 2 α(a, x): the original point also carries a
    # moment coefficient, exercising merged duplicate points.
    return FiniteEvalEstimand(
        feature_keys=("a", "x"),
        m=lambda alpha: lambda z, y=None: alpha(a=z["a"] + 1.0, x=z["x"]) - 2.0 * alpha(**z),
    )


def _model(seed=0):
    torch.manual_seed(seed)
    return build_mlp(2, hidden_sizes=(8,)).double()


def _eta(model, X):
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float64)).squeeze(-1).numpy()


@pytest.mark.parametrize("estimand", [ATE(), _custom_shift()], ids=["ate", "custom"])
@pytest.mark.parametrize("loss", [SquaredLoss(), KLLoss(max_eta=10.0)], ids=["squared", "kl"])
def test_augmented_per_row_loss_equals_trace_form(estimand, loss):
    """L_i = h̃(α(x_i)) + Σ_j (−coef_j) h'(α(p_j)) over trace(estimand, z_i)."""
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.choice([0.0, 1.0], 30), rng.normal(size=30)])
    model = _model()

    data = _AugTensors.build(estimand.augment(X), "cpu", torch.float64)
    base = torch.tensor(0.0, dtype=torch.float64)
    with torch.no_grad():
        got = _batch_loss(model, base, TorchRieszLoss(loss), data).item()

    per_row = []
    for x in X:
        z = {"a": x[0], "x": x[1]}
        L = loss.tilde_potential(loss.link_to_alpha(_eta(model, x[None])))[0]
        for coef, point in trace(estimand, z):
            eta_p = _eta(model, np.array([[point["a"], point["x"]]]))
            L -= coef * loss.potential_deriv(loss.link_to_alpha(eta_p))[0]
        per_row.append(L)
    np.testing.assert_allclose(got, np.mean(per_row), rtol=1e-10)


def test_minibatches_partition_the_full_batch_loss():
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.choice([0.0, 1.0], 101), rng.normal(size=101)])
    data = _AugTensors.build(TSM(level=1.0).augment(X), "cpu", torch.float64)
    model, base, tl = _model(), torch.tensor(0.3, dtype=torch.float64), TorchRieszLoss(SquaredLoss())
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        full = _batch_loss(model, base, tl, data).item() * data.n_rows
        parts = sum(
            _batch_loss(model, base, tl, data, rows).item() * rows.shape[0]
            for rows in _row_batches(data.n_rows, 16, gen)
        )
    np.testing.assert_allclose(parts, full, rtol=1e-12)


def test_kl_best_score_is_the_validation_riesz_loss(logistic_tsm_df):
    df = logistic_tsm_df
    train, valid = df.iloc[:300], df.iloc[300:]
    est = RieszNet(
        estimand=TSM(level=1.0), loss=KLLoss(), hidden_sizes=(8,), epochs=15,
        early_stopping_rounds=100, dtype="float64",
    ).fit(train, eval_set=valid)
    assert est.best_score_ == pytest.approx(est.riesz_loss(valid), rel=1e-8)


def test_no_early_stopping_keeps_final_weights(small_df):
    """Without early stopping, eval_set must not roll the model back."""
    kw = dict(estimand=ATE(), hidden_sizes=(4,), epochs=8, batch_size=None)
    with_eval = RieszNet(**kw).fit(small_df, eval_set=small_df)
    without = RieszNet(**kw).fit(small_df)
    np.testing.assert_allclose(with_eval.predict(small_df), without.predict(small_df))
    assert with_eval.best_iteration_ is None
