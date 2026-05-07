"""Per-leaf Bregman optimization.

For SquaredLoss the per-leaf optimum has a closed form `θ = (Σ J)⁻¹ (Σ A)`
that EconML's `LinearMomentGRFCriterion` solves directly. For non-quadratic
Bregman losses the leaf-level loss is

    L_leaf(θ) = Σ_{r ∈ leaf} loss.aug_loss_alpha(D_r, C_r, link(θ · φ(z_r)))

which is convex in θ (the loss's gradient is monotone) but non-linear,
so we Newton-iterate. The per-row gradient and Hessian come from
``loss.aug_grad_eta(D, C, η)`` and ``loss.aug_hess_eta(D, C, η, floor)``,
which work in η-space.

Used by `AugForestRieszBackend` to post-hoc replace each leaf's stored value
when the user asks for `KLLoss`, `BernoulliLoss`, or `BoundedSquaredLoss`.
The tree structure is still chosen by the squared-loss MSE criterion — splits
that maximize variance reduction in α* = −Σ C / Σ D also separate the
monotonically-related Bregman optima well.
"""

from __future__ import annotations

import numpy as np

from rieszreg import Loss


def solve_leaf_bregman(
    loss: Loss,
    is_original_leaf: np.ndarray,
    potential_deriv_coef_leaf: np.ndarray,
    phi_leaf: np.ndarray,
    *,
    base_score: float = 0.0,
    init_theta: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    hessian_floor: float = 1e-6,
) -> np.ndarray:
    """Find the per-leaf θ minimizing the augmented loss over rows in this leaf,
    where η_r = θ · φ(z_r) + base_score.

    The base_score offset matches the predictor's `predict_eta` formula —
    when a non-zero base_score is in play, the leaf θ represents the
    deviation from base_score and Newton evaluates gradients at
    η_r = θ · φ_r + base_score, not θ · φ_r.

    Parameters
    ----------
    loss
        The Bregman loss. Provides ``aug_grad_eta(D, C, η)`` and
        ``aug_hess_eta(D, C, η, floor)`` per row in η-space.
    is_original_leaf, potential_deriv_coef_leaf
        Augmented coefficients (D, C) for the rows in this leaf, shape ``(m,)``.
    phi_leaf
        Basis evaluations at those rows, shape ``(m, p)``. For locally
        constant fits ``p = 1`` and ``phi_leaf[:, 0] = 1``.
    base_score
        Predictor offset: η at predict time is ``θ · φ(z) + base_score``.
    init_theta
        Starting θ for the Newton iteration. Defaults to all-zeros (so the
        first η = base_score, which is loss.default_init_alpha()'s η).
    max_iter, tol, hessian_floor
        Newton stopping rule and numerical safeguards.

    Returns
    -------
    np.ndarray, shape ``(p,)``: the leaf-level θ.
    """
    if phi_leaf.ndim == 1:
        phi_leaf = phi_leaf.reshape(-1, 1)
    p = phi_leaf.shape[1]

    if init_theta is None:
        init_theta = np.zeros(p, dtype=float)

    if is_original_leaf.size == 0:
        return init_theta.copy()

    if p == 1:
        return _newton_scalar(
            loss, is_original_leaf, potential_deriv_coef_leaf, phi_leaf[:, 0],
            base_score=base_score, init_theta=float(init_theta[0]),
            max_iter=max_iter, tol=tol, hessian_floor=hessian_floor,
        )
    return _newton_multivariate(
        loss, is_original_leaf, potential_deriv_coef_leaf, phi_leaf,
        base_score=base_score, init_theta=init_theta,
        max_iter=max_iter, tol=tol, hessian_floor=hessian_floor,
    )


def _newton_scalar(loss, is_original, potential_deriv_coef, phi, *, base_score, init_theta, max_iter, tol, hessian_floor):
    """Scalar Newton iteration for p = 1.

    Per-row η = θ · φ_r + base_score. Chain rule: ∂η/∂θ = φ_r.
    """
    theta = float(init_theta)
    phi2 = phi * phi
    for _ in range(max_iter):
        eta = theta * phi + base_score
        g_eta = loss.aug_grad_eta(is_original, potential_deriv_coef, eta)
        h_eta = loss.aug_hess_eta(is_original, potential_deriv_coef, eta, hessian_floor)
        G = float(np.sum(g_eta * phi))
        H = float(np.sum(h_eta * phi2))
        if H <= 0.0:
            break
        step = -G / H
        theta_new = theta + step
        if abs(step) < tol:
            return np.array([theta_new], dtype=float)
        theta = theta_new
    return np.array([theta], dtype=float)


def _newton_multivariate(loss, is_original, potential_deriv_coef, phi, *, base_score, init_theta, max_iter, tol, hessian_floor):
    """Multivariate Newton for p > 1 (sieve case)."""
    p = phi.shape[1]
    theta = init_theta.astype(float).copy()
    for _ in range(max_iter):
        eta = phi @ theta + base_score
        g_eta = loss.aug_grad_eta(is_original, potential_deriv_coef, eta)
        h_eta = loss.aug_hess_eta(is_original, potential_deriv_coef, eta, hessian_floor)
        G = phi.T @ g_eta
        H = (phi * h_eta[:, None]).T @ phi
        try:
            step = -np.linalg.solve(H, G)
        except np.linalg.LinAlgError:
            H_ridged = H + 1e-8 * np.eye(p)
            try:
                step = -np.linalg.solve(H_ridged, G)
            except np.linalg.LinAlgError:
                break
        theta = theta + step
        if float(np.max(np.abs(step))) < tol:
            break
    return theta


def compute_leaf_eta_table(
    forest,
    X_aug: np.ndarray,
    is_original_aug: np.ndarray,
    potential_deriv_coef_aug: np.ndarray,
    phi_aug: np.ndarray,
    loss: Loss,
    *,
    base_score: float = 0.0,
) -> dict[tuple[int, int], np.ndarray]:
    """Walk every (tree, leaf) pair and store the Bregman per-leaf optimum.

    Returns a dict mapping ``(tree_idx, leaf_node_id)`` to the θ vector
    of shape ``(p,)``.
    """
    leaves_per_tree = forest.apply(X_aug)        # (n_aug, n_trees)
    n_aug, n_trees = leaves_per_tree.shape

    table: dict[tuple[int, int], np.ndarray] = {}
    for t in range(n_trees):
        leaf_ids_t = leaves_per_tree[:, t]
        for leaf_id in np.unique(leaf_ids_t):
            mask = leaf_ids_t == leaf_id
            theta = solve_leaf_bregman(
                loss=loss,
                is_original_leaf=is_original_aug[mask],
                potential_deriv_coef_leaf=potential_deriv_coef_aug[mask],
                phi_leaf=phi_aug[mask],
                base_score=base_score,
            )
            table[(int(t), int(leaf_id))] = theta
    return table
