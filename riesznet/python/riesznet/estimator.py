"""RieszNet — sklearn-compatible neural-network Riesz regressor.

Subclass of ``rieszreg.RieszEstimator`` that defaults the backend to a simple
MLP trained with Adam, surfacing the common knobs (``hidden_sizes``,
``activation``, ``dropout``, ``learning_rate``, ``weight_decay``, ``epochs``,
``device``, ``dtype``) on the constructor.

Power users who want a custom architecture instantiate ``TorchBackend``
directly and pass it to ``RieszEstimator(estimand, backend=...)``.
"""

from __future__ import annotations

import functools
from typing import Sequence

import numpy as np

from rieszreg import Estimand, LossSpec, RieszEstimator, SquaredLoss
from rieszreg.estimator import _features_from_rows, _rows_from_Z

from .backend import TorchBackend, auto_snapshot_epochs
from .modules import build_adam, build_mlp


class RieszNet(RieszEstimator):
    """Neural-network Riesz regressor with a default-MLP convenience surface.

    Trains the Riesz representer α₀ of a linear functional θ(P) = E[m(Z, g₀)]
    by minimizing the per-row Bregman-Riesz loss with PyTorch.

    Parameters
    ----------
    estimand : rieszreg.Estimand
        Carries ``feature_keys`` and the ``m(alpha)(z, y)`` operator.
    hidden_sizes : tuple of int, default (64, 64)
        MLP hidden-layer widths. Empty tuple gives a linear model.
    activation : {"relu", "tanh", "gelu", "elu", "silu", "leaky_relu"}, default "relu"
    dropout : float, default 0.0
        Dropout probability applied after each activation.
    learning_rate : float, default 1e-3
        Adam learning rate.
    weight_decay : float, default 0.0
    epochs : int, default 200
    batch_size : int or None, default 64
        Number of original rows per minibatch. ``None`` is full-batch GD.
        64 is a common default for tabular MLPs; reduce if validation loss is
        unstable, increase to speed up training when n is large.
    device : {"cpu", "cuda", "mps", "auto"}, default "cpu"
    dtype : {"float32", "float64"}, default "float32"
    grad_clip_norm : float or None, default None
    loss : rieszreg.LossSpec, default SquaredLoss()
        Any of ``SquaredLoss``, ``KLLoss``, ``BernoulliLoss``,
        ``BoundedSquaredLoss``.
    init : float or None
        α-space initialization. ``None`` (default) sets α to the empirical
        loss-minimizing constant ``m̄ = E[m(Z, 1)]`` projected into the
        loss's α-domain. Pass a float to override.
    validation_fraction : float, default 0.0
        Fraction of training data held out for early stopping. Required when
        ``early_stopping_rounds`` is set.
    early_stopping_rounds : int or None
        Stop fitting after this many epochs without validation-loss
        improvement; restore best-validation weights at end of fit.
    snapshot_epochs : sequence of int or None, default None
        Epoch ticks at which to snapshot ``state_dict`` during training so
        ``predict_path(Z, epochs=...)`` can return α̂ at each tick. ``None``
        builds an auto-grid of ~20 log-spaced ticks across ``[1, epochs]``
        (see :func:`riesznet.backend.auto_snapshot_epochs`). Pass an empty
        sequence to disable snapshotting entirely.
    random_state : int, default 0
    """

    def __init__(
        self,
        estimand: Estimand,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 200,
        batch_size: int | None = 64,
        device: str = "cpu",
        dtype: str = "float32",
        grad_clip_norm: float | None = None,
        loss: LossSpec | None = None,
        init: float | str | None = None,
        validation_fraction: float = 0.0,
        early_stopping_rounds: int | None = None,
        snapshot_epochs: Sequence[int] | None = None,
        random_state: int = 0,
    ):
        super().__init__(
            estimand=estimand,
            backend=None,                # built lazily in _resolved_backend
            loss=loss,
            init=init,
            random_state=random_state,
        )
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.grad_clip_norm = grad_clip_norm
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.snapshot_epochs = snapshot_epochs

    # ---- defaults / backend construction ----

    def _resolved_loss(self) -> LossSpec:
        return self.loss if self.loss is not None else SquaredLoss()

    def _resolved_snapshot_epochs(self) -> tuple[int, ...]:
        if self.snapshot_epochs is None:
            return auto_snapshot_epochs(int(self.epochs))
        ticks = sorted({int(e) for e in self.snapshot_epochs})
        for e in ticks:
            if not (1 <= e <= int(self.epochs)):
                raise ValueError(
                    f"snapshot_epochs entry {e} must satisfy 1 ≤ e ≤ epochs"
                    f"={int(self.epochs)}."
                )
        return tuple(ticks)

    def _resolved_backend(self) -> TorchBackend:
        module_factory = functools.partial(
            build_mlp,
            hidden_sizes=tuple(int(h) for h in self.hidden_sizes),
            activation=self.activation,
            dropout=float(self.dropout),
        )
        optimizer_factory = functools.partial(
            build_adam,
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        return TorchBackend(
            module_factory=module_factory,
            optimizer_factory=optimizer_factory,
            scheduler_factory=None,
            epochs=int(self.epochs),
            batch_size=self.batch_size,
            device=self.device,
            dtype=self.dtype,
            grad_clip_norm=self.grad_clip_norm,
            early_stopping_rounds=self.early_stopping_rounds,
            validation_fraction=self.validation_fraction,
            snapshot_epochs=self._resolved_snapshot_epochs(),
        )

    def predict_path(
        self, Z, epochs: Sequence[int] | None = None
    ) -> np.ndarray:
        """Predict α̂ at every snapshot epoch from one training run.

        Returns an array of shape ``(n_rows, n_epochs)`` whose column ``j``
        is the prediction obtained from the network's ``state_dict`` at
        snapshot epoch ``epochs[j]`` (or every retained snapshot when
        ``epochs`` is ``None``). Each column equals a fresh fit at
        ``epochs=epochs[j]`` to within Adam's deterministic-trajectory
        tolerance (bit-equal under fixed seed and identical data ordering).
        """
        if not hasattr(self, "predictor_"):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet. Call .fit() first."
            )
        rows = _rows_from_Z(Z, self.estimand)
        feats = _features_from_rows(rows, self.estimand)
        return self.predictor_.predict_alpha_path(feats, epochs)

    # ---- save/load ----

    def _save_hyperparameters(self) -> dict:
        base = super()._save_hyperparameters()
        base.update(
            hidden_sizes=list(self.hidden_sizes),
            activation=self.activation,
            dropout=float(self.dropout),
            learning_rate=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
            epochs=int(self.epochs),
            batch_size=self.batch_size,
            device=self.device,
            dtype=self.dtype,
            grad_clip_norm=self.grad_clip_norm,
            early_stopping_rounds=self.early_stopping_rounds,
            validation_fraction=float(self.validation_fraction),
            snapshot_epochs=(
                list(int(e) for e in self.snapshot_epochs)
                if self.snapshot_epochs is not None
                else None
            ),
        )
        return base

    @classmethod
    def _construct_for_load(
        cls, *, estimand, loss, hyperparameters: dict
    ) -> "RieszNet":
        hs = hyperparameters.get("hidden_sizes", [64, 64])
        return cls(
            estimand=estimand,
            hidden_sizes=tuple(int(h) for h in hs),
            activation=hyperparameters.get("activation", "relu"),
            dropout=float(hyperparameters.get("dropout", 0.0)),
            learning_rate=float(hyperparameters.get("learning_rate", 1e-3)),
            weight_decay=float(hyperparameters.get("weight_decay", 0.0)),
            epochs=int(hyperparameters.get("epochs", 200)),
            batch_size=hyperparameters.get("batch_size", 64),
            device=hyperparameters.get("device", "cpu"),
            dtype=hyperparameters.get("dtype", "float32"),
            grad_clip_norm=hyperparameters.get("grad_clip_norm"),
            loss=loss,
            init=hyperparameters.get("init"),
            validation_fraction=hyperparameters.get("validation_fraction", 0.0),
            early_stopping_rounds=hyperparameters.get("early_stopping_rounds"),
            snapshot_epochs=hyperparameters.get("snapshot_epochs"),
            random_state=hyperparameters.get("random_state", 0),
        )
