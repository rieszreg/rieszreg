"""Phase 5: ``register_fast_leaf_solver`` plugs a Numba ``@cfunc`` (or
any C-callable address) into the Cython splitter for a user Loss.

The registered cfunc is called from the Cython splitter's tight loop
at C speed — no Python dispatch per evaluation. The smoke tests below
register a SquaredLoss-equivalent kernel for a fresh Loss subclass
and verify that the resulting fitted tree matches a vanilla SquaredLoss
fit on the same DGP.
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from rieszreg import ATE, Loss, SquaredLoss
from riesztree import RieszTreeRegressor
from riesztree.fast import register_fast_leaf_solver
from riesztree.fast._splitter import (
    LOSS_USER_CFUNC,
    _USER_LOSS_REGISTRY,
    loss_kind_for,
)

numba = pytest.importorskip("numba")


# ---------------------------------------------------------------------------
# A user Loss subclass that does the same math as SquaredLoss. Keeping
# it isinstance-incompatible with SquaredLoss forces the splitter to fall
# back to the user registry rather than the built-in dispatch.

class _MySquaredLoss(Loss):
    """Identical math to SquaredLoss, but a fresh Loss subclass —
    the built-in `loss_kind_for` returns ``None`` for it."""

    name = "_MySquaredLoss"

    # The augmented Bregman API the orchestrator consumes — match
    # SquaredLoss point-for-point.
    def potential(self, alpha):
        return SquaredLoss().potential(alpha)

    def potential_deriv(self, alpha):
        return SquaredLoss().potential_deriv(alpha)

    def potential_deriv_inv(self, eta):
        return SquaredLoss().potential_deriv_inv(eta)

    def tilde_potential(self, alpha):
        return SquaredLoss().tilde_potential(alpha)

    def alpha_to_eta(self, alpha):
        return SquaredLoss().alpha_to_eta(alpha)

    def eta_to_alpha(self, eta):
        return SquaredLoss().eta_to_alpha(eta)

    def link_to_alpha(self, eta):
        return SquaredLoss().link_to_alpha(eta)

    def to_spec(self):
        return {"name": "_MySquaredLoss"}


# A Numba @cfunc with the SquaredLoss leaf-loss math.

@numba.cfunc("float64(float64, float64)", cache=True, nopython=True)
def _mysquared_leaf_loss(D, C):
    if D <= 0.0:
        return 0.0
    return -C * C / D


def _mysquared_alpha_at_opt(D, C):
    return 0.0 if D <= 0.0 else -C / D


@pytest.fixture
def _registered_mysquared():
    """Register the cfunc + alpha for ``_MySquaredLoss`` for the test,
    then clear the registry afterwards so tests don't bleed state."""
    register_fast_leaf_solver(
        _MySquaredLoss, _mysquared_leaf_loss, _mysquared_alpha_at_opt
    )
    yield
    _USER_LOSS_REGISTRY.pop(_MySquaredLoss, None)


def _make_df(n=600, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, p))
    pi = 1.0 / (1.0 + np.exp(-0.5 * X[:, 0]))
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    cols = {f"x{j}": X[:, j] for j in range(p)}
    cols["a"] = a
    return pd.DataFrame(cols)


def _ate(p):
    return ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))


# ---------------------------------------------------------------------------

def test_register_uses_user_cfunc_kind():
    """After registration, ``loss_kind_for`` returns ``LOSS_USER_CFUNC``
    for the user's loss subclass."""
    register_fast_leaf_solver(
        _MySquaredLoss, _mysquared_leaf_loss, _mysquared_alpha_at_opt
    )
    try:
        kind, _lo, _hi, addr = loss_kind_for(_MySquaredLoss())
        assert kind == LOSS_USER_CFUNC
        assert addr == _mysquared_leaf_loss.address
        # Built-in still resolves to the built-in path:
        kind2, _, _, addr2 = loss_kind_for(SquaredLoss())
        assert kind2 != LOSS_USER_CFUNC
        assert addr2 == 0
    finally:
        _USER_LOSS_REGISTRY.pop(_MySquaredLoss, None)


def test_register_rejects_invalid_address():
    with pytest.raises(ValueError, match="invalid C-callable address"):
        register_fast_leaf_solver(_MySquaredLoss, 0, _mysquared_alpha_at_opt)


def test_register_rejects_non_callable_alpha():
    with pytest.raises(TypeError, match="alpha_at_opt"):
        register_fast_leaf_solver(_MySquaredLoss, _mysquared_leaf_loss, 42)


def test_user_cfunc_fit_matches_builtin_squared_loss(_registered_mysquared):
    """Fit with the user-registered loss vs vanilla SquaredLoss on the
    same seed; predictions must agree exactly."""
    df = _make_df(n=600, p=5)
    estimand = _ate(5)
    builtin = RieszTreeRegressor(
        estimand=estimand, loss=SquaredLoss(), max_depth=4, random_state=0
    ).fit(df)
    user = RieszTreeRegressor(
        estimand=estimand, loss=_MySquaredLoss(), max_depth=4, random_state=0
    ).fit(df)
    np.testing.assert_array_equal(builtin.predict(df), user.predict(df))


def test_unregistered_loss_raises_helpful_error():
    """Without registration, fit raises NotImplementedError pointing the
    user at the registration hook."""
    df = _make_df(n=200, p=3)
    with pytest.raises(NotImplementedError, match="register_fast_leaf_solver"):
        RieszTreeRegressor(
            estimand=_ate(3), loss=_MySquaredLoss(), max_depth=4
        ).fit(df)
