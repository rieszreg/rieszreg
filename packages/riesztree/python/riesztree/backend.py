"""RieszTreeBackend — augmentation-style single-tree backend.

Implements ``rieszreg.Backend.fit_augmented``. Consumes the precomputed
``AugmentedDataset``, picks the loss-aware splitter, and grows / prunes
the tree according to the constructor's hyperparameters.

Universal across the four built-in losses (SquaredLoss / KLLoss /
BernoulliLoss / BoundedSquaredLoss). Custom Loss subclasses need
:func:`riesztree.fast.register_fast_leaf_solver`; unregistered ones raise
NotImplementedError from the leaf-solver dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from rieszreg import AugmentedDataset, FitResult, Loss

from .grow import _Grower
from .predictor import RieszTreePredictor
from .pruning import cost_complexity_prune
from .tree import check_categorical


@dataclass
class RieszTreeBackend:
    """Single-tree backend for the augmentation-style entry point.

    Hyperparameters mirror :class:`sklearn.tree.DecisionTreeRegressor`
    where the augmented Bregman-Riesz setting allows.

    Parameters
    ----------
    max_depth
        Maximum tree depth. ``None`` means unlimited. Default 8.
    min_samples_split
        Minimum count of original (D > 0) augmented rows in a node before
        considering a split. Default 20.
    min_samples_leaf
        Minimum count of original rows in each child of a candidate split.
        Default 10.
    min_weight_fraction_leaf
        sklearn-parity: leaves must contain at least
        ``ceil(min_weight_fraction_leaf * n_original_total)`` original rows
        (combined with ``min_samples_leaf`` via ``max(...)``). Default 0.0.
    max_leaf_nodes
        Cap for leafwise growth; ``None`` means unlimited. Ignored when
        ``growth_policy="depthwise"``. Default 31.
    max_features
        Per-split feature-subsampling rule, as in
        :class:`sklearn.tree.DecisionTreeRegressor`. ``None``, ``"sqrt"``,
        ``"log2"``, an int, or a float in ``(0, 1]``. Default ``None``.
    growth_policy
        ``"depthwise"`` (recursive depth-first) or ``"leafwise"`` (best-first).
        Default ``"depthwise"``.
    min_impurity_decrease
        Reject splits with gain ≤ this threshold. Default 0.0 (sklearn).
    ccp_alpha
        Cost-complexity penalty. ``0`` (default) disables pruning.
    early_stopping_rounds
        Stop growing when held-out augmented loss has not improved for that
        many consecutive accepted splits. ``None`` (default) disables.
    validation_fraction
        Held-out fraction the orchestrator splits off before augmentation
        when early stopping or pruning needs a holdout. Default 0.0.
    categorical_features
        Tuple of column indices (into the estimand's ``feature_keys``)
        whose values should be treated as integer category labels rather
        than ordered numerics. Default ``()``.
    splitter
        ``"exact"`` (default) routes continuous-feature splits through
        the Cython sweep in :mod:`riesztree.fast._splitter_c`;
        ``"hist"`` uses the histogram-based Cython splitter
        (:mod:`riesztree.fast._splitter_hist`) with quantile pre-binning;
        ``"random"`` (sklearn ExtraTrees-style) draws a single uniform
        threshold per feature per leaf and evaluates the gain there.
        Custom losses registered via
        :func:`riesztree.fast.register_fast_leaf_solver` need ``"exact"``.
    """

    max_depth: int | None = 8
    min_samples_split: int = 20
    min_samples_leaf: int = 10
    min_weight_fraction_leaf: float = 0.0
    max_leaf_nodes: int | None = 31
    max_features: object = None     # int | float | str | None
    growth_policy: str = "depthwise"
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0
    early_stopping_rounds: int | None = None
    validation_fraction: float = 0.0
    categorical_features: tuple[int, ...] = field(default_factory=tuple)
    splitter: str = "exact"
    max_bins: int = 255      # used when splitter == "hist"

    def fit_augmented(
        self,
        aug_train: AugmentedDataset,
        aug_valid: AugmentedDataset | None,
        loss: Loss,
        *,
        base_score: float,
        random_state: int | None,
    ) -> FitResult:
        # Leaves store the loss-optimal α directly, so there is no boosting
        # offset: base_score (and hence `init`) has no effect on a tree.
        del base_score
        check_categorical(aug_train.features, self.categorical_features)
        X_binned, mapper = self._bin(aug_train.features, random_state)
        return self._fit_binned(
            aug_train, aug_valid, loss,
            random_state=random_state, X_binned=X_binned, mapper=mapper,
        )

    def _bin(self, features: np.ndarray, random_state: int | None):
        """``(X_binned, mapper)`` for ``splitter="hist"``, else ``(None, None)``."""
        if self.splitter != "hist":
            return None, None
        from .fast._binner import fit_bin_mapper, transform
        mapper = fit_bin_mapper(features, max_bins=self.max_bins, random_state=random_state)
        return transform(features, mapper), mapper

    def _fit_binned(
        self,
        aug_train: AugmentedDataset,
        aug_valid: AugmentedDataset | None,
        loss: Loss,
        *,
        random_state: int | None,
        X_binned: np.ndarray | None = None,
        mapper=None,
    ) -> FitResult:
        """Grow (and prune) one tree. With ``splitter="hist"``, ``X_binned``
        is ``aug_train.features`` already binned by ``mapper``, so a forest
        can bin once and share the mapper across trees."""
        if self.growth_policy not in ("depthwise", "leafwise"):
            raise ValueError(
                f"growth_policy must be 'depthwise' or 'leafwise'; got "
                f"{self.growth_policy!r}."
            )
        unbounded = 2**31 - 1
        has_valid = aug_valid is not None and aug_valid.n_rows > 0
        cat_feats = tuple(int(i) for i in self.categorical_features)
        g = _Grower(
            aug_train.features, aug_train.is_original, aug_train.potential_deriv_coef, loss,
            max_depth=unbounded if self.max_depth is None else int(self.max_depth),
            min_samples_split=self.min_samples_split,
            min_orig_leaf=self.min_samples_leaf,
            categorical_features=cat_feats,
            max_features=self.max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            aug_valid=aug_valid if has_valid else None,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=random_state,
            splitter=self.splitter,
            X_binned=X_binned,
            mapper=mapper,
        )
        if self.growth_policy == "depthwise":
            tree = g.grow_depthwise()
        else:
            tree = g.grow_leafwise(
                unbounded if self.max_leaf_nodes is None else int(self.max_leaf_nodes)
            )

        if self.ccp_alpha > 0:
            tree = cost_complexity_prune(tree, loss, ccp_alpha=self.ccp_alpha)

        predictor = RieszTreePredictor(tree=tree, loss=loss, categorical_features=cat_feats)
        return FitResult(
            predictor=predictor,
            best_score=(
                aug_valid.mean_loss(loss, predictor.predict_alpha(aug_valid.features))
                if has_valid else None
            ),
        )
