"""Phase 10: ``splitter='python'`` is deprecated.

The Cython paths (exact / hist / random) are byte-equivalent or
numerically very close to the legacy Python splitter on the four
built-in losses, and substantially faster. Phase 10 marks the
``splitter='python'`` path as deprecated. v0.0.3 will remove it.

For custom losses outside the four built-ins, the recommended path
is :func:`riesztree.fast.register_fast_leaf_solver` instead of
falling back to the Python splitter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riesztree import ATE, RieszTreeRegressor


def _make_df(n=300, p=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, p))
    pi = 1.0 / (1.0 + np.exp(-0.5 * X[:, 0]))
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    cols = {f"x{j}": X[:, j] for j in range(p)}
    cols["a"] = a
    return pd.DataFrame(cols)


def _ate(p):
    return ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))


def test_splitter_python_emits_deprecation_warning():
    """Setting splitter='python' explicitly fires a DeprecationWarning
    pointing at the v0.0.3 removal."""
    # Reset the once-only flag so this test is order-independent.
    from riesztree import grow as grow_mod
    grow_mod._PYTHON_SPLITTER_WARNED = False

    df = _make_df(n=200, p=3)
    with pytest.warns(DeprecationWarning, match="splitter='python' is deprecated"):
        RieszTreeRegressor(
            estimand=_ate(3), max_depth=4, splitter="python"
        ).fit(df)


def test_splitter_python_warning_fires_only_once():
    """The DeprecationWarning is one-shot per process to avoid noise on
    repeated cross-validation fits."""
    from riesztree import grow as grow_mod
    grow_mod._PYTHON_SPLITTER_WARNED = False

    df = _make_df(n=200, p=3)
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        for _ in range(3):
            RieszTreeRegressor(
                estimand=_ate(3), max_depth=4, splitter="python"
            ).fit(df)
    matching = [w for w in caught if "splitter='python' is deprecated" in str(w.message)]
    assert len(matching) == 1, f"expected 1 warning, got {len(matching)}"


def test_splitter_exact_does_not_warn():
    """The default path emits no DeprecationWarning."""
    from riesztree import grow as grow_mod
    grow_mod._PYTHON_SPLITTER_WARNED = False

    df = _make_df(n=200, p=3)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        RieszTreeRegressor(estimand=_ate(3), max_depth=4).fit(df)


def test_splitter_python_still_produces_correct_fit():
    """The deprecated path still works correctly. Only the warning is new."""
    from riesztree import grow as grow_mod
    grow_mod._PYTHON_SPLITTER_WARNED = False

    df = _make_df(n=200, p=3)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        est = RieszTreeRegressor(
            estimand=_ate(3), max_depth=4, splitter="python"
        ).fit(df)
    a = est.predict(df)
    assert np.isfinite(a).all()
