"""Comparison of moment-style vs augmentation-style backends.

These tests are not strict pass/fail — they assert the two backends produce
*comparable* fits on the estimands where both work, and they print a small
benchmark table so you can eyeball the trade-off.

Run with ``pytest -s`` to see the printed timing / RMSE table.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from rieszreg.testing import dgps

from forestriesz import ATE, AugForestRieszRegressor, ForestRieszRegressor, TSM


def _benchmark_one(make_moment, make_aug, dgp, n, seed=0):
    rng = np.random.default_rng(seed)
    train = dgp.sample(n, rng)
    test = dgp.sample(2 * n, rng)
    truth = dgp.true_alpha(test)

    moment = make_moment()
    t0 = time.perf_counter()
    moment.fit(train)
    moment_fit_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    moment_pred = moment.predict(test)
    moment_pred_t = time.perf_counter() - t0
    moment_rmse = float(np.sqrt(np.mean((moment_pred - truth) ** 2)))

    aug = make_aug()
    t0 = time.perf_counter()
    aug.fit(train)
    aug_fit_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    aug_pred = aug.predict(test)
    aug_pred_t = time.perf_counter() - t0
    aug_rmse = float(np.sqrt(np.mean((aug_pred - truth) ** 2)))

    return {
        "moment_fit_s": moment_fit_t,
        "moment_pred_s": moment_pred_t,
        "moment_rmse": moment_rmse,
        "aug_fit_s": aug_fit_t,
        "aug_pred_s": aug_pred_t,
        "aug_rmse": aug_rmse,
        "fit_slowdown": aug_fit_t / max(moment_fit_t, 1e-9),
        "rmse_ratio": aug_rmse / max(moment_rmse, 1e-9),
    }


@pytest.fixture
def kw():
    return dict(n_estimators=200, min_samples_leaf=10, random_state=0)


def test_ate_aug_matches_moment_quality(kw):
    """At reasonable n the two backends should hit similar RMSE on ATE."""
    dgp = dgps.linear_gaussian_ate()
    rng = np.random.default_rng(0)
    train = dgp.sample(2000, rng)
    test = dgp.sample(4000, rng)
    truth = dgp.true_alpha(test)

    moment = ForestRieszRegressor(estimand=ATE(), **kw).fit(train)
    aug = AugForestRieszRegressor(estimand=ATE(), **kw).fit(train)
    m_rmse = float(np.sqrt(np.mean((moment.predict(test) - truth) ** 2)))
    a_rmse = float(np.sqrt(np.mean((aug.predict(test) - truth) ** 2)))

    # Aug should be within 2× of moment RMSE; both should be < 1.0 absolute.
    assert m_rmse < 1.0
    assert a_rmse < 1.0
    assert a_rmse < 2.0 * m_rmse, f"aug rmse {a_rmse:.3f} far worse than moment {m_rmse:.3f}"


def test_tsm_aug_matches_moment_quality(kw):
    dgp = dgps.logistic_tsm(level=1.0)
    rng = np.random.default_rng(0)
    train = dgp.sample(2000, rng)
    test = dgp.sample(4000, rng)
    truth = dgp.true_alpha(test)

    moment = ForestRieszRegressor(estimand=TSM(level=1), **kw).fit(train)
    aug = AugForestRieszRegressor(estimand=TSM(level=1), **kw).fit(train)
    m_rmse = float(np.sqrt(np.mean((moment.predict(test) - truth) ** 2)))
    a_rmse = float(np.sqrt(np.mean((aug.predict(test) - truth) ** 2)))

    assert m_rmse < 1.0
    assert a_rmse < 1.0
    assert a_rmse < 2.0 * m_rmse, f"aug rmse {a_rmse:.3f} far worse than moment {m_rmse:.3f}"


def test_print_benchmark_table(capsys, kw):
    """Side-effect test: print a table comparing the two backends.
    Always passes; run with ``pytest -s`` to see output."""

    rows = []
    for name, dgp, est_factory in [
        ("ATE n=500", dgps.linear_gaussian_ate(), ATE),
        ("ATE n=2000", dgps.linear_gaussian_ate(), ATE),
        ("ATE n=5000", dgps.linear_gaussian_ate(), ATE),
        ("TSM n=500", dgps.logistic_tsm(1.0), lambda: TSM(level=1)),
        ("TSM n=2000", dgps.logistic_tsm(1.0), lambda: TSM(level=1)),
        ("TSM n=5000", dgps.logistic_tsm(1.0), lambda: TSM(level=1)),
    ]:
        n = int(name.split("n=")[1])
        result = _benchmark_one(
            make_moment=lambda: ForestRieszRegressor(estimand=est_factory(), **kw),
            make_aug=lambda: AugForestRieszRegressor(estimand=est_factory(), **kw),
            dgp=dgp,
            n=n,
        )
        rows.append({"setting": name, "n": n, **result})

    df = pd.DataFrame(rows)
    cols = [
        "setting",
        "moment_fit_s",
        "aug_fit_s",
        "fit_slowdown",
        "moment_rmse",
        "aug_rmse",
        "rmse_ratio",
    ]
    pd.set_option("display.float_format", lambda v: f"{v:8.3f}")
    print("\n--- aug-style vs moment-style backend benchmark ---")
    print(df[cols].to_string(index=False))
