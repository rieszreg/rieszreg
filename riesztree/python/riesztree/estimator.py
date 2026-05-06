"""RieszTreeRegressor — sklearn-compatible single-tree Riesz regressor.

Subclass of ``rieszreg.RieszEstimator`` defaulting to ``RieszTreeBackend``
and surfacing tree-specific hyperparameters on the constructor. Composes
with ``GridSearchCV``, ``cross_val_predict``, ``Pipeline``.

Hyperparameter names mirror :class:`sklearn.tree.DecisionTreeRegressor`
where the augmented Bregman-Riesz setting allows. The v0.0.1 names
``pruning_alpha`` and ``max_leaves`` are accepted as deprecated aliases
for ``ccp_alpha`` and ``max_leaf_nodes`` and emit ``FutureWarning``.
"""

from __future__ import annotations

import warnings
from typing import Sequence

from rieszreg import Estimand, LossSpec, RieszEstimator, SquaredLoss

from .backend import RieszTreeBackend


class RieszTreeRegressor(RieszEstimator):
    """Single-tree Riesz regression.

    Fits the Riesz representer ``α₀`` of a linear functional by greedy splits
    on the augmented Bregman-Riesz loss. Each leaf stores the closed-form
    per-leaf optimum ``α_ℓ* = -C_ℓ / D_ℓ`` (loss-aware: projected to the
    loss's α-domain).

    Parameters
    ----------
    estimand : rieszreg.Estimand
        Carries ``feature_keys`` and the ``m(alpha)(z, y)`` operator.
    loss : rieszreg.LossSpec, default=None
        Bregman-Riesz loss. ``None`` resolves to ``SquaredLoss()``. Built-in
        support: ``SquaredLoss``, ``KLLoss``, ``BernoulliLoss``,
        ``BoundedSquaredLoss``.
    max_depth : int, default=8
        Maximum tree depth.
    min_samples_split : int, default=20
        Minimum count of original (D > 0) augmented rows in a node before
        considering a split.
    min_samples_leaf : int, default=10
        Minimum count of original rows in each child of a candidate split.
    min_weight_fraction_leaf : float, default=0.0
        sklearn parity: leaves must contain at least
        ``ceil(min_weight_fraction_leaf * n_original_total)`` original rows
        (combined with ``min_samples_leaf`` via ``max(...)``). Default 0.0.
    max_leaf_nodes : int, default=31
        Cap for leafwise growth. Ignored when ``growth_policy="depthwise"``.
    max_features : int, float, {"sqrt", "log2"}, or None, default=None
        Per-split feature-subsampling rule (sklearn convention).
    growth_policy : {"depthwise", "leafwise"}, default="depthwise"
        Tree-growth strategy.
    min_impurity_decrease : float, default=0.0
        Reject splits with gain ≤ this threshold.
    ccp_alpha : float, default=0.0
        Cost-complexity pruning penalty. ``0`` disables pruning.
    early_stopping_rounds : int or None, default=None
        Stop growing when held-out augmented loss has not improved for that
        many consecutive accepted splits. ``None`` disables.
    validation_fraction : float, default=0.1
        Held-out fraction the orchestrator splits off before augmentation
        when early stopping is enabled. Ignored when no holdout is needed.
    categorical_features : sequence of int, optional
        Column indices (into ``estimand.feature_keys``) treated as integer
        category labels; the splitter sorts levels by within-level α* and
        sweeps contiguous splits, per the standard CART convention.
    init : float or None
        α-space initialization, threaded through to ``RieszEstimator``.
    random_state : int, default=0
        Seeds the per-split feature subsample under ``max_features``.
    pruning_alpha : float or None, default=None
        Deprecated. Alias for ``ccp_alpha``; emits ``FutureWarning``.
    max_leaves : int or None, default=None
        Deprecated. Alias for ``max_leaf_nodes``; emits ``FutureWarning``.
    """

    def __init__(
        self,
        estimand: Estimand,
        loss: LossSpec | None = None,
        max_depth: int = 8,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        min_weight_fraction_leaf: float = 0.0,
        max_leaf_nodes: int = 31,
        max_features: object = None,
        growth_policy: str = "depthwise",
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        early_stopping_rounds: int | None = None,
        validation_fraction: float = 0.1,
        categorical_features: Sequence[int] | None = None,
        init: float | None = None,
        random_state: int = 0,
        # Deprecated aliases — keep at the end for backwards-compatible
        # positional behaviour. ``None`` sentinel → not user-supplied.
        pruning_alpha: float | None = None,
        max_leaves: int | None = None,
    ):
        super().__init__(
            estimand=estimand,
            backend=None,
            loss=loss,
            init=init,
            random_state=random_state,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.growth_policy = growth_policy
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.categorical_features = categorical_features
        # Deprecated aliases stored as-is so sklearn's clone() round-trips
        # through get_params / __init__. Resolution happens in
        # _resolved_backend at fit-time so set_params can still flip them.
        self.pruning_alpha = pruning_alpha
        self.max_leaves = max_leaves

    # ---- alias resolution ----

    def _resolved_ccp_alpha(self) -> float:
        if self.pruning_alpha is not None:
            warnings.warn(
                "`pruning_alpha` is deprecated; use `ccp_alpha` instead "
                "(matches sklearn.tree.DecisionTreeRegressor).",
                FutureWarning,
                stacklevel=3,
            )
            return float(self.pruning_alpha)
        return float(self.ccp_alpha)

    def _resolved_max_leaf_nodes(self) -> int:
        if self.max_leaves is not None:
            warnings.warn(
                "`max_leaves` is deprecated; use `max_leaf_nodes` instead "
                "(matches sklearn.tree.DecisionTreeRegressor).",
                FutureWarning,
                stacklevel=3,
            )
            return int(self.max_leaves)
        return int(self.max_leaf_nodes)

    # ---- backend construction ----

    def _resolved_loss(self) -> LossSpec:
        return self.loss if self.loss is not None else SquaredLoss()

    def _resolved_backend(self) -> RieszTreeBackend:
        cat = (
            tuple(int(i) for i in self.categorical_features)
            if self.categorical_features is not None
            else ()
        )
        # Validation fraction is only consumed when a holdout is needed.
        val_frac = (
            float(self.validation_fraction)
            if self.early_stopping_rounds is not None
            else 0.0
        )
        return RieszTreeBackend(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_leaf_nodes=self._resolved_max_leaf_nodes(),
            max_features=self.max_features,
            growth_policy=self.growth_policy,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self._resolved_ccp_alpha(),
            early_stopping_rounds=self.early_stopping_rounds,
            validation_fraction=val_frac,
            categorical_features=cat,
            random_state=self.random_state,
        )

    def fit(self, Z, y=None, eval_set=None, eval_y=None) -> "RieszTreeRegressor":
        """Fit the tree. After dispatch, we patch the predictor's
        ``feature_keys`` from the resolved estimand so save/load round-trips
        carry the column ordering."""
        super().fit(Z, y=y, eval_set=eval_set, eval_y=eval_y)
        self.predictor_.feature_keys = tuple(self.feature_keys_)
        return self

    # ---- save/load ----

    def _save_hyperparameters(self) -> dict:
        base = super()._save_hyperparameters()
        base.update(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            max_features=self.max_features,
            growth_policy=self.growth_policy,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            early_stopping_rounds=self.early_stopping_rounds,
            validation_fraction=self.validation_fraction,
            categorical_features=(
                list(int(i) for i in self.categorical_features)
                if self.categorical_features is not None
                else None
            ),
        )
        return base

    @classmethod
    def _construct_for_load(
        cls, *, estimand, loss, hyperparameters: dict
    ) -> "RieszTreeRegressor":
        cat = hyperparameters.get("categorical_features")
        return cls(
            estimand=estimand,
            loss=loss,
            max_depth=hyperparameters.get("max_depth", 8),
            min_samples_split=hyperparameters.get("min_samples_split", 20),
            min_samples_leaf=hyperparameters.get("min_samples_leaf", 10),
            min_weight_fraction_leaf=hyperparameters.get(
                "min_weight_fraction_leaf", 0.0
            ),
            max_leaf_nodes=hyperparameters.get(
                "max_leaf_nodes", hyperparameters.get("max_leaves", 31)
            ),
            max_features=hyperparameters.get("max_features"),
            growth_policy=hyperparameters.get("growth_policy", "depthwise"),
            min_impurity_decrease=hyperparameters.get("min_impurity_decrease", 0.0),
            ccp_alpha=hyperparameters.get(
                "ccp_alpha", hyperparameters.get("pruning_alpha", 0.0)
            ),
            early_stopping_rounds=hyperparameters.get("early_stopping_rounds"),
            validation_fraction=hyperparameters.get("validation_fraction", 0.1),
            categorical_features=tuple(int(i) for i in cat) if cat else None,
            init=hyperparameters.get("init"),
            random_state=hyperparameters.get("random_state", 0),
        )
