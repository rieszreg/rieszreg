"""Fitting, diagnosing and reloading don't leak state between models, and the
reported validation loss is the Riesz loss of the fitted α̂."""

from __future__ import annotations

import numpy as np

from krrr import ATE, TSM, Gaussian, KernelRieszRegressor, Sum, Tensor


def _krr(**kwargs):
    kwargs.setdefault("estimand", ATE(treatment="a", covariates=("x",)))
    kwargs.setdefault("lambda_grid", [1e-3, 1e-2, 1e-1])
    return KernelRieszRegressor(**kwargs)


def test_shared_kernel_object_does_not_couple_models(binary_ate_data):
    df, _, _ = binary_ate_data
    df1, df2 = df.iloc[:150], df.iloc[150:].assign(x=lambda d: 10.0 * d["x"])
    k = Gaussian()  # "median" bandwidth, resolved per fit
    m1 = _krr(kernel=k).fit(df1)
    _krr(kernel=k).fit(df2)
    assert np.isnan(k._resolved)  # the user's kernel is never resolved in place
    np.testing.assert_array_equal(m1.predict(df1), _krr(kernel=Gaussian()).fit(df1).predict(df1))


def test_tensor_with_one_kernel_object_resolves_each_factor(binary_ate_data):
    df, _, _ = binary_ate_data
    X = np.column_stack([df["x"], 100.0 * df["x"]])
    k = Gaussian()
    t = Tensor(k, [0], k, [1]).fit_data(X)
    assert t.b._ls() > 50.0 * t.a._ls()
    # The same object nested inside a factor must not couple the factors either.
    t = Tensor(Sum(k, Gaussian(length_scale=1.0)), [0], k, [1]).fit_data(X)
    assert t.b._ls() > 50.0 * t.a.a._ls()


def test_diagnose_leaves_predictions_unchanged(binary_ate_data):
    df, _, _ = binary_ate_data
    krr = _krr(solver="direct").fit(df.iloc[:200])
    before = krr.predict(df)
    diag = krr.diagnose(df.iloc[200:].assign(x=lambda d: 5.0 * d["x"]))
    np.testing.assert_array_equal(krr.predict(df), before)
    assert diag.effective_dof is not None and diag.effective_dof > 0


def test_best_score_is_riesz_loss_on_eval_set(binary_ate_data):
    """TSM starts from α = m̄ = 1, a nonzero base_score folded into the targets;
    the reported validation loss must still be the Riesz loss of α̂."""
    df, _, _ = binary_ate_data
    krr = _krr(estimand=TSM(level=1, treatment="a", covariates=("x",)), validation_fraction=0.0)
    krr.fit(df.iloc[:200], eval_set=df.iloc[200:])
    assert krr.base_score_ == 1.0
    np.testing.assert_allclose(krr.best_score_, krr.riesz_loss(df.iloc[200:]), rtol=1e-10)


def test_no_validation_picks_largest_lambda(binary_ate_data):
    df, _, _ = binary_ate_data
    krr = _krr(lambda_grid=[1.0, 1e-3, 1e-2], validation_fraction=0.0).fit(df)
    assert krr.lambda_ == 1.0


def test_save_load_round_trips_kernel_and_lambda_grid(binary_ate_data, tmp_path):
    df, _, _ = binary_ate_data
    krr = _krr(kernel=Gaussian(length_scale=0.3), lambda_grid=[1e-3, 1e-2], solver="direct").fit(df)
    krr.save(tmp_path / "m")
    loaded = KernelRieszRegressor.load(tmp_path / "m")
    np.testing.assert_array_equal(loaded.predict(df), krr.predict(df))
    np.testing.assert_array_equal(loaded.predict_path(df), krr.predict_path(df))
    assert loaded.kernel == krr.kernel
    assert list(loaded.lambda_grid) == list(krr.lambda_grid)
    assert loaded.solver == "direct"
    assert loaded.lambda_ == krr.lambda_


def test_diagnose_reports_unconverged_conjugate_gradient():
    """A λ whose CG solve stopped at cg_max_iter is flagged by diagnose()."""
    import pandas as pd

    from krrr import KernelRieszRegressor
    from rieszreg import ATE

    rng = np.random.default_rng(0)
    x = rng.normal(size=(1500, 2))
    a = (rng.uniform(size=1500) < 1 / (1 + np.exp(-x[:, 0]))).astype(float)
    df = pd.DataFrame({"a": a, "x0": x[:, 0], "x1": x[:, 1]})
    fit = lambda **kw: KernelRieszRegressor(ATE(), solver="nystrom_cg", **kw).fit(df)
    stopped = fit(cg_max_iter=2, lambda_grid=[1e-5]).diagnose(df)
    assert any("did not converge" in w for w in stopped.extra_warnings)
    assert not fit(lambda_grid=[1e-2]).diagnose(df).extra_warnings
