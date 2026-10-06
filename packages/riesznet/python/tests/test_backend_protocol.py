"""TorchBackend satisfies Backend; TorchPredictor satisfies Predictor."""

from __future__ import annotations

import functools

import numpy as np

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
