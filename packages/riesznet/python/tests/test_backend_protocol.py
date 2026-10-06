"""TorchBackend satisfies Backend; TorchPredictor satisfies Predictor."""

from __future__ import annotations

import functools

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

import riesznet
from riesznet import RieszNet, TorchBackend, TorchPredictor
from riesznet.modules import build_adam, build_mlp


def test_backend_exposes_fit_augmented():
    backend = TorchBackend(
        module_factory=functools.partial(build_mlp),
        optimizer_factory=functools.partial(build_adam),
    )
    assert callable(getattr(backend, "fit_augmented", None))
    assert not hasattr(backend, "fit_rows")


def test_predictor_protocol_surface():
    assert TorchPredictor.kind == "riesznet"
    for attr in ("predict_eta", "predict_alpha", "save"):
        assert callable(getattr(TorchPredictor, attr)), attr


def test_predictor_loader_registered():
    from rieszreg.backends.base import _PREDICTOR_LOADERS

    assert "riesznet" in _PREDICTOR_LOADERS, (
        "Importing riesznet must register the predictor loader. "
        "Check that backend.py runs `register_predictor_loader('riesznet', ...)`."
    )


def test_namespace_reexports_shared_user_api():
    from rieszreg import user_api

    assert set(user_api.__all__) <= set(riesznet.__all__)
    assert {"RieszNet", "TorchBackend"} <= set(riesznet.__all__)


def test_load_works_after_plain_import(tmp_path, linear_gaussian_ate_df):
    """A saved RieszNet reloads via RieszEstimator.load in a fresh process
    that has only run `import riesznet` (torch not yet imported)."""
    import subprocess
    import sys

    est = riesznet.RieszNet(estimand=riesznet.ATE(), epochs=2, hidden_sizes=(4,))
    est.fit(linear_gaussian_ate_df)
    est.save(tmp_path / "m")
    code = (
        "import riesznet, sys; from rieszreg import RieszEstimator; "
        "assert 'torch' not in sys.modules; "
        f"RieszEstimator.load({str(tmp_path / 'm')!r})"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_rieszreg_orchestrator_composes_with_torch_backend(linear_gaussian_ate_df):
    """End-to-end: composing TorchBackend with rieszreg.RieszEstimator works."""
    from rieszreg import ATE, RieszEstimator

    backend = TorchBackend(
        module_factory=functools.partial(build_mlp, hidden_sizes=(16,)),
        optimizer_factory=functools.partial(build_adam, lr=1e-2),
        epochs=20,
    )
    est = RieszEstimator(estimand=ATE(), backend=backend, random_state=0)
    est.fit(linear_gaussian_ate_df)
    pred = est.predict(linear_gaussian_ate_df)
    assert pred.shape == (len(linear_gaussian_ate_df),)
    assert np.all(np.isfinite(pred))


_threads_seen: list[int] = []


def _thread_recording_factory(input_dim):
    _threads_seen.append(torch.get_num_threads())
    return build_mlp(input_dim, hidden_sizes=(4,))


def test_n_jobs_sets_threads_for_fit_and_restores_them(linear_gaussian_ate_df):
    from rieszreg import ATE, RieszEstimator

    before = torch.get_num_threads()
    n_jobs = 1 if before > 1 else 2
    backend = TorchBackend(
        module_factory=_thread_recording_factory,
        optimizer_factory=functools.partial(build_adam),
        epochs=2, n_jobs=n_jobs,
    )
    est = RieszEstimator(estimand=ATE(), backend=backend).fit(linear_gaussian_ate_df)
    assert _threads_seen[-1] == n_jobs
    assert torch.get_num_threads() == before
    est.predict(linear_gaussian_ate_df)
    assert torch.get_num_threads() == before


def test_n_jobs_minus_one_uses_all_cores_and_bad_values_raise(linear_gaussian_ate_df):
    import os

    from rieszreg import ATE, RieszEstimator

    def backend(n_jobs):
        return TorchBackend(
            module_factory=_thread_recording_factory,
            optimizer_factory=functools.partial(build_adam),
            epochs=1, n_jobs=n_jobs,
        )

    RieszEstimator(estimand=ATE(), backend=backend(-1)).fit(linear_gaussian_ate_df)
    assert _threads_seen[-1] == os.cpu_count()
    with pytest.raises(ValueError, match="n_jobs must be"):
        RieszEstimator(estimand=ATE(), backend=backend(0)).fit(linear_gaussian_ate_df)


def test_fit_leaves_global_torch_rng_alone(linear_gaussian_ate_df):
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    RieszNet(estimand=riesznet.ATE(), hidden_sizes=(4,), epochs=2).fit(linear_gaussian_ate_df)
    torch.testing.assert_close(torch.rand(3), expected)


def test_rows_are_held_out_only_under_early_stopping(linear_gaussian_ate_df):
    """Without early stopping every row trains (the standardization mean is
    the full-data mean); with it, validation_fraction of the rows is held out."""
    from rieszreg import ATE, RieszEstimator

    df = linear_gaussian_ate_df

    def fitted_x_mean(**kw):
        backend = TorchBackend(
            module_factory=functools.partial(build_mlp, hidden_sizes=(4,)),
            optimizer_factory=functools.partial(build_adam),
            epochs=2, validation_fraction=0.2, **kw,
        )
        assert backend.holdout_fraction() == (0.2 if kw else 0.0)
        return RieszEstimator(estimand=ATE(), backend=backend).fit(df).predictor_.feature_loc[1]

    assert fitted_x_mean() == pytest.approx(df["x"].mean())
    assert fitted_x_mean(early_stopping_rounds=5) != pytest.approx(df["x"].mean())


def test_fits_in_parallel_threads_match_sequential_fits(small_df):
    """torch's random state is process-wide; fits in threads take turns, so
    each still depends only on its data and random_state."""
    def fit(seed):
        net = RieszNet(estimand=riesznet.ATE(), hidden_sizes=(8,), dropout=0.2, epochs=5, random_state=seed)
        return net.fit(small_df).predict(small_df)

    sequential = [fit(s) for s in range(4)]
    with ThreadPoolExecutor(4) as ex:
        threaded = list(ex.map(fit, range(4)))
    for a, b in zip(sequential, threaded):
        np.testing.assert_array_equal(a, b)
