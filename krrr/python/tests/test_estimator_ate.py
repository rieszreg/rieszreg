"""End-to-end ATE: KernelRieszRegressor recovers the inverse-propensity
representer up to regularization bias.
"""

from __future__ import annotations

import numpy as np

from krrr import ATE, Gaussian, KernelRieszRegressor


def test_ate_correlates_with_truth(binary_ate_data):
    df, truth, _ = binary_ate_data
    krr = KernelRieszRegressor(
        estimand=ATE("a", ("x",)),
        kernel=Gaussian(length_scale="median"),
        lambda_grid=np.logspace(-4, 0, 12),
        solver="direct",
        validation_fraction=0.25,
        random_state=0,
    )
    krr.fit(df)
    alpha_hat = krr.predict(df)
    assert alpha_hat.shape == (len(df),)
    assert np.all(np.isfinite(alpha_hat))
    # Predictions are in the right ballpark — large positive for treated,
    # negative for control.
    assert np.corrcoef(alpha_hat, truth)[0, 1] > 0.85
    # selected_lambda surfaced
    assert krr.lambda_ in list(np.logspace(-4, 0, 12))


def test_ate_score_higher_on_train_than_random(binary_ate_data):
    df, _, _ = binary_ate_data
    krr = KernelRieszRegressor(
        estimand=ATE("a", ("x",)),
        lambda_grid=np.logspace(-3, 0, 6),
        validation_fraction=0.2,
    ).fit(df)
    s_train = krr.score(df)
    # Random α: scrambled predictions
    rng = np.random.default_rng(0)
    krr2 = KernelRieszRegressor(
        estimand=ATE("a", ("x",)),
        lambda_grid=[10.0],  # heavily over-regularized -> ~0
        validation_fraction=0.2,
    ).fit(df)
    s_random = krr2.score(df)
    # higher score = lower loss; trained should beat over-regularized
    assert s_train > s_random
