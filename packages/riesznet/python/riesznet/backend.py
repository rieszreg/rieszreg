"""TorchBackend — implements ``rieszreg.Backend``.

Consumes the ``AugmentedDataset``, groups the augmented rows by origin row,
and minimizes the per-row Bregman-Riesz loss with a PyTorch training loop
over minibatches of original rows. Returns a ``FitResult`` whose predictor
is a ``TorchPredictor``.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import importlib
import json
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable

import numpy as np
import torch

from rieszreg import AugmentedDataset, FitResult, Loss, register_predictor_loader
from rieszreg.losses import loss_from_spec

from .losses_torch import TorchRieszLoss


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_device(spec: str, dtype: str = "float32") -> torch.device:
    """``"auto"`` picks CUDA, then MPS (float32 only; MPS has no float64),
    then CPU. An explicit device is used as given."""
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if dtype != "float64" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(spec: str) -> torch.dtype:
    if spec == "float32":
        return torch.float32
    if spec == "float64":
        return torch.float64
    raise ValueError(f"dtype must be 'float32' or 'float64'; got {spec!r}")


def _factory_metadata(factory: Callable) -> dict:
    """Snapshot a callable factory as JSON-friendly metadata.

    Supports top-level functions (``func.__module__`` + ``func.__qualname__``)
    and ``functools.partial`` over them. Closures, lambdas, and locally-defined
    classes raise — the user must define the factory at module top level.
    """
    inner, partial_kwargs = factory, None
    if isinstance(factory, functools.partial):
        if factory.args:
            raise ValueError(
                "TorchBackend save/load requires `module_factory` partials to "
                "use keyword args only (no positional args), so the factory "
                "round-trips faithfully through JSON metadata."
            )
        inner, partial_kwargs = factory.func, dict(factory.keywords or {})
        try:
            json.dumps(partial_kwargs)
        except TypeError as e:
            raise ValueError(
                "TorchBackend save/load requires `module_factory` partial "
                "kwargs to be JSON-serializable (numbers, strings, bools, "
                "lists/tuples, dicts, None). Got non-serializable kwarg in "
                f"{partial_kwargs!r}: {e}"
            ) from e
    qualname = getattr(inner, "__qualname__", None)
    module = getattr(inner, "__module__", None)
    if qualname is None or module is None or "." in qualname or qualname.startswith("<"):
        raise ValueError(
            "TorchBackend save/load requires `module_factory` to be a "
            "top-level callable (e.g. a module-level `def`). Closures, "
            "lambdas, and class methods cannot be reconstructed by qualname."
        )
    return {"qualname": qualname, "module": module, "partial_kwargs": partial_kwargs}


def _factory_from_metadata(meta: dict) -> Callable:
    mod = importlib.import_module(meta["module"])
    inner = getattr(mod, meta["qualname"])
    pk = meta.get("partial_kwargs")
    if pk:
        return functools.partial(inner, **pk)
    return inner


# torch's random state is process-wide, and whether its thread count is
# depends on the build (per-thread under OpenMP). A fit (and a predict that
# sets n_jobs) holds this lock while it changes them, so fits in parallel
# threads run one at a time instead of mixing each other's seeds, and thread
# counts can't leak between them on any build.
_TORCH_STATE_LOCK = threading.RLock()


@contextlib.contextmanager
def _num_threads(n_jobs: int | None):
    """Run the block with torch's intra-op thread count set to ``n_jobs``
    (``None`` leaves it alone, -1 means all cores), then restore the
    previous count."""
    if n_jobs is None:
        yield
        return
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if int(n_jobs) < 1:
        raise ValueError(f"n_jobs must be None, -1 or a positive integer; got {n_jobs!r}.")
    with _TORCH_STATE_LOCK:
        prev = torch.get_num_threads()
        torch.set_num_threads(int(n_jobs))
        try:
            yield
        finally:
            torch.set_num_threads(prev)


def _standardizer(aug: AugmentedDataset) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and standard deviation of the observed rows (D ≠ 0).
    Constant columns get scale 1. The test is relative, as in sklearn: a
    constant 0.1 has a float std of about 1e-16, not 0."""
    X = aug.features[aug.is_original != 0]
    loc, scale = X.mean(axis=0), X.std(axis=0)
    constant = scale <= 10 * np.finfo(float).eps * np.maximum(1.0, np.abs(loc))
    return loc, np.where(constant, 1.0, scale)


def _standardize(X: np.ndarray, loc: np.ndarray | None, scale: np.ndarray | None) -> np.ndarray:
    return X if loc is None else (X - loc) / scale


@dataclass
class _AugTensors:
    """An ``AugmentedDataset`` on the training device, sorted by origin row
    with CSR offsets so a minibatch of original rows gathers only its own
    augmented rows."""

    features: torch.Tensor
    is_original: torch.Tensor
    potential_deriv_coef: torch.Tensor
    origin: torch.Tensor
    offsets: torch.Tensor  # (n_rows + 1,) start of each row's block
    n_rows: int

    @classmethod
    def build(cls, aug: AugmentedDataset, device, dtype, loc=None, scale=None) -> "_AugTensors":
        # Rows with D = C = 0 contribute nothing; drop them.
        keep = (aug.is_original != 0) | (aug.potential_deriv_coef != 0)
        order = np.flatnonzero(keep)[np.argsort(aug.origin_index[keep], kind="stable")]
        origin = aug.origin_index[order]
        counts = np.bincount(origin, minlength=aug.n_rows)

        def t(a, dt=dtype):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)

        return cls(
            features=t(_standardize(aug.features[order], loc, scale)),
            is_original=t(aug.is_original[order]),
            potential_deriv_coef=t(aug.potential_deriv_coef[order]),
            origin=t(origin, torch.long),
            offsets=t(np.concatenate([[0], np.cumsum(counts)]), torch.long),
            n_rows=int(aug.n_rows),
        )

    def batch(self, rows: torch.Tensor | None):
        """``(features, D, C, local_origin, n_batch)`` for original rows
        ``rows`` (``None`` means every row)."""
        if rows is None:
            return self.features, self.is_original, self.potential_deriv_coef, self.origin, self.n_rows
        starts = self.offsets[rows]
        counts = self.offsets[rows + 1] - starts
        local = torch.repeat_interleave(torch.arange(rows.shape[0], device=rows.device), counts)
        block_start = torch.cumsum(counts, 0) - counts
        idx = starts[local] + torch.arange(local.shape[0], device=rows.device) - block_start[local]
        return (
            self.features[idx], self.is_original[idx],
            self.potential_deriv_coef[idx], local, int(rows.shape[0]),
        )


def _batch_loss(model, base, torch_loss: TorchRieszLoss, data: _AugTensors, rows=None):
    """Mean per-row Riesz loss over original rows ``rows`` (all when None)."""
    feats, D, C, origin, n = data.batch(rows)
    eta = model(feats).squeeze(-1) + base
    return torch_loss.per_row(eta, D, C, origin, n).mean()


def _row_batches(
    n_rows: int, batch_size: int | None, generator: torch.Generator
) -> list[torch.Tensor | None]:
    """Row-index batches for one epoch; ``[None]`` means one full batch."""
    if batch_size is None or batch_size >= n_rows:
        return [None]
    perm = torch.randperm(n_rows, generator=generator)
    return [perm[i : i + batch_size] for i in range(0, n_rows, batch_size)]


# ----------------------------------------------------------------------
# Predictor
# ----------------------------------------------------------------------


def auto_snapshot_epochs(max_epochs: int) -> tuple[int, ...]:
    """Default tick grid for ``RieszNet.snapshot_epochs``.

    Returns roughly 20 epoch ticks spanning ``[1, max_epochs]``: dense at
    the start (1, 2, 5, 10) and evenly spaced thereafter at stride
    ``max(1, max_epochs // 20)``. Always includes 1 and ``max_epochs``.
    """
    if max_epochs < 1:
        return ()
    rec = max(1, int(max_epochs) // 20)
    seeded = {1, int(max_epochs)}
    seeded.update({2, 5, 10})
    seeded.update(range(rec, int(max_epochs) + 1, rec))
    return tuple(sorted(e for e in seeded if 1 <= e <= int(max_epochs)))


@dataclass
class TorchPredictor:
    """Wraps the trained ``nn.Module`` for prediction.

    Implements the ``rieszreg.Predictor`` protocol (``predict_eta``,
    ``predict_alpha``, ``save`` + classmethod ``load``).

    When fit with ``snapshot_epochs`` set, also stores a per-epoch
    ``state_dict`` snapshot dictionary so ``predict_eta_path`` /
    ``predict_alpha_path`` can return α̂ at every snapshot epoch in one call.
    ``feature_loc`` / ``feature_scale`` standardize inputs before the model
    sees them (``None`` when the backend was fit with ``standardize=False``).

    The predictor puts the model in eval mode on its device once, at
    construction, and never changes it afterward. Each snapshot is evaluated
    on its own copy of the model, built on first use, so concurrent
    predictions are safe.
    """

    model: torch.nn.Module
    loss: Loss
    base_score: float
    input_dim: int
    dtype: str
    device: str
    module_factory: Callable[[int], torch.nn.Module]
    snapshot_state_dicts: dict[int, dict[str, torch.Tensor]] | None = None
    snapshot_epochs: tuple[int, ...] | None = None
    feature_loc: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    n_jobs: int | None = None

    kind: ClassVar[str] = "riesznet"

    def __post_init__(self):
        device = _resolve_device(self.device, self.dtype)
        # A model saved on a GPU still predicts on a machine without one.
        if (device.type == "cuda" and not torch.cuda.is_available()) or (
            device.type == "mps" and not torch.backends.mps.is_available()
        ):
            device = torch.device("cpu")
        self._device = device
        self.model.to(device=device, dtype=_resolve_dtype(self.dtype)).eval()
        self._snapshot_models: dict[int, torch.nn.Module] = {}

    def _snapshot_model(self, epoch: int) -> torch.nn.Module:
        model = self._snapshot_models.get(epoch)
        if model is None:
            model = copy.deepcopy(self.model)
            model.load_state_dict(self.snapshot_state_dicts[epoch])
            self._snapshot_models[epoch] = model
        return model

    # ---- prediction ----

    def _input_tensor(self, features: np.ndarray) -> torch.Tensor:
        """Check the feature width and return the standardized ``features``
        as a tensor on the model's device."""
        X = np.atleast_2d(np.array(features, dtype=float))
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"TorchPredictor expects {self.input_dim} input features, "
                f"got {X.shape[1]}."
            )
        X = _standardize(X, self.feature_loc, self.feature_scale)
        return torch.as_tensor(X, dtype=_resolve_dtype(self.dtype), device=self._device)

    def _eta(self, X_t: torch.Tensor, model: torch.nn.Module | None = None) -> np.ndarray:
        """η at ``X_t`` from the fitted model, or from ``model`` (a snapshot)."""
        with torch.no_grad(), _num_threads(self.n_jobs):
            eta_t = (self.model if model is None else model)(X_t).squeeze(-1) + self.base_score
        return eta_t.cpu().numpy().astype(float)

    def predict_eta(self, features: np.ndarray) -> np.ndarray:
        return self._eta(self._input_tensor(features))

    def predict_alpha(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.loss.link_to_alpha(self.predict_eta(features)))

    # ---- path predict ----

    def _resolve_snapshot_epochs(
        self, epochs: Iterable[int] | None
    ) -> list[int]:
        if self.snapshot_state_dicts is None or self.snapshot_epochs is None:
            raise RuntimeError(
                "predict_path requires snapshot_epochs (auto or explicit) "
                "to have been set at fit time."
            )
        if epochs is None:
            return list(self.snapshot_epochs)
        chosen: list[int] = []
        stored = set(self.snapshot_epochs)
        for e in epochs:
            ek = int(e)
            if ek not in stored:
                raise ValueError(
                    f"epoch={ek!r} not in stored snapshot_epochs "
                    f"{tuple(self.snapshot_epochs)}."
                )
            chosen.append(ek)
        return chosen

    def predict_eta_path(
        self, features: np.ndarray, epochs: Iterable[int] | None = None
    ) -> np.ndarray:
        chosen = self._resolve_snapshot_epochs(epochs)
        X_t = self._input_tensor(features)
        return np.column_stack([self._eta(X_t, self._snapshot_model(ep)) for ep in chosen])

    def predict_alpha_path(
        self, features: np.ndarray, epochs: Iterable[int] | None = None
    ) -> np.ndarray:
        eta = self.predict_eta_path(features, epochs)
        return np.asarray(self.loss.link_to_alpha(eta))

    # ---- serialization ----

    def save(self, dir_path) -> None:
        # Raises for a factory that can't be re-imported by qualname, before
        # anything is written.
        factory_meta = _factory_metadata(self.module_factory)
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        # Always save weights on CPU so load works on machines without the
        # original device.
        cpu_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        torch.save(cpu_state, path / "state_dict.pt")
        meta = {
            "kind": self.kind,
            "loss": self.loss.to_spec(),
            "base_score": float(self.base_score),
            "input_dim": int(self.input_dim),
            "dtype": self.dtype,
            "device": self.device,
            "factory": factory_meta,
            "snapshot_epochs": (
                list(self.snapshot_epochs)
                if self.snapshot_epochs is not None
                else None
            ),
            "feature_loc": None if self.feature_loc is None else self.feature_loc.tolist(),
            "feature_scale": None if self.feature_scale is None else self.feature_scale.tolist(),
            "n_jobs": self.n_jobs,
        }
        with open(path / "predictor.json", "w") as f:
            json.dump(meta, f, indent=2)

        if self.snapshot_state_dicts is not None and self.snapshot_epochs:
            snap_dir = path / "snapshots"
            snap_dir.mkdir(exist_ok=True)
            for ep in self.snapshot_epochs:
                cpu_sd = {
                    k: v.detach().cpu()
                    for k, v in self.snapshot_state_dicts[ep].items()
                }
                torch.save(cpu_sd, snap_dir / f"epoch_{ep}.pt")

    @classmethod
    def load(cls, dir_path, *, base_score=None, loss=None, best_iteration=None):
        path = Path(dir_path)
        with open(path / "predictor.json") as f:
            meta = json.load(f)
        loss_obj = loss if loss is not None else loss_from_spec(meta["loss"])
        bs = float(meta["base_score"]) if base_score is None else float(base_score)
        dtype = meta.get("dtype", "float32")
        device = meta.get("device", "cpu")

        factory = _factory_from_metadata(meta["factory"])
        loc, scale = meta.get("feature_loc"), meta.get("feature_scale")
        model = factory(int(meta["input_dim"]))
        torch_dtype = _resolve_dtype(dtype)
        model.to(dtype=torch_dtype)
        state = torch.load(path / "state_dict.pt", map_location="cpu")
        model.load_state_dict(state)

        snap_epochs = meta.get("snapshot_epochs")
        snap_dir = path / "snapshots"
        snap_state_dicts: dict[int, dict[str, torch.Tensor]] | None = None
        if snap_epochs and snap_dir.is_dir():
            snap_state_dicts = {}
            for ep in snap_epochs:
                snap_state_dicts[int(ep)] = torch.load(
                    snap_dir / f"epoch_{int(ep)}.pt", map_location="cpu"
                )

        return cls(
            model=model,
            loss=loss_obj,
            base_score=bs,
            input_dim=int(meta["input_dim"]),
            dtype=dtype,
            device=device,
            module_factory=factory,
            snapshot_state_dicts=snap_state_dicts,
            snapshot_epochs=(
                tuple(int(e) for e in snap_epochs) if snap_epochs else None
            ),
            feature_loc=None if loc is None else np.asarray(loc, dtype=float),
            feature_scale=None if scale is None else np.asarray(scale, dtype=float),
            n_jobs=meta.get("n_jobs"),
        )


register_predictor_loader("riesznet", TorchPredictor.load)


# ----------------------------------------------------------------------
# Backend
# ----------------------------------------------------------------------


def _default_module_factory(input_dim: int):  # pragma: no cover - placeholder
    raise RuntimeError(
        "TorchBackend.module_factory must be supplied. "
        "Use the convenience class `riesznet.RieszNet` for a default MLP."
    )


def _default_optimizer_factory(params):  # pragma: no cover - placeholder
    raise RuntimeError(
        "TorchBackend.optimizer_factory must be supplied. "
        "Use the convenience class `riesznet.RieszNet` for a default Adam."
    )


@dataclass
class TorchBackend:
    """Neural-network Riesz regression backend (PyTorch).

    Implements ``rieszreg.Backend.fit_augmented``: groups the augmented rows
    by origin row and minimizes the per-row Bregman-Riesz loss with a
    PyTorch training loop over minibatches of original rows.

    Parameters
    ----------
    module_factory : Callable[[int], nn.Module]
        ``input_dim -> nn.Module`` returning a module that maps
        ``(batch, input_dim) -> (batch, 1)`` (or ``(batch,)``) producing the
        η-space score. Must be importable by qualname for save/load — pass a
        top-level function or a ``functools.partial`` over one. Closures and
        lambdas raise on save.
    optimizer_factory : Callable[[Iterable[Parameter]], Optimizer]
        ``params -> torch.optim.Optimizer``. Same importability constraint.
    scheduler_factory : Callable[[Optimizer], Any] or None
        Optional ``optimizer -> LRScheduler`` factory. Stepped once per epoch.
    epochs : int, default 200
    batch_size : int or None, default None
        Number of original rows per minibatch. ``None`` means full-batch.
    device : {"cpu", "cuda", "mps", "auto"}, default "cpu"
    dtype : {"float32", "float64"}, default "float32"
    grad_clip_norm : float or None, default None
        Global L2 gradient-norm clip applied before each optimizer step.
    early_stopping_rounds : int or None, default None
        Stop after this many epochs without validation-loss improvement and
        restore the best-validation weights.
    validation_fraction : float, default 0.1
        Fraction of training rows held out for early stopping when no
        ``eval_set`` is passed. Rows are held out only when
        ``early_stopping_rounds`` is set.
    snapshot_epochs : tuple of int, default ()
        Epochs at which to store the weights for ``predict_eta_path``.
    standardize : bool, default True
        Center and scale each input column by its mean and standard deviation
        over the observed training rows. The counterfactual rows get the
        same transformation, so it is valid for every estimand.
    n_jobs : int or None, default None
        torch intra-op threads during fit and predict. ``None`` keeps torch's
        current setting (all cores unless changed); -1 uses all cores.

    torch's random state is process-wide (and, on some builds, so is its
    thread count), so fits running in parallel threads of one process take
    turns. Run parallel fits in separate processes (joblib's default) to fit
    them at the same time.
    """

    module_factory: Callable[[int], torch.nn.Module] = field(
        default=_default_module_factory
    )
    optimizer_factory: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer] = field(
        default=_default_optimizer_factory
    )
    scheduler_factory: Callable[[torch.optim.Optimizer], Any] | None = None
    epochs: int = 200
    batch_size: int | None = None
    device: str = "cpu"
    dtype: str = "float32"
    grad_clip_norm: float | None = None
    early_stopping_rounds: int | None = None
    validation_fraction: float = 0.1
    snapshot_epochs: tuple[int, ...] = ()
    standardize: bool = True
    n_jobs: int | None = None

    def holdout_fraction(self) -> float:
        """Rows to hold out: ``validation_fraction`` under early stopping, else 0."""
        return self.validation_fraction if self.early_stopping_rounds is not None else 0.0

    def fit_augmented(
        self,
        aug_train: AugmentedDataset,
        aug_valid: AugmentedDataset | None,
        loss: Loss,
        *,
        base_score: float,
        random_state: int,
    ) -> FitResult:
        if aug_valid is None and self.early_stopping_rounds is not None:
            raise ValueError(
                "early_stopping_rounds requires a validation set. Set "
                "validation_fraction>0 (or pass eval_set=) when fitting."
            )
        device = _resolve_device(self.device, self.dtype)
        seed = int(random_state)
        # fork_rng restores the caller's torch random state, on the CPU and
        # on the training device, after the fit.
        cpu = device.type == "cpu"
        with _TORCH_STATE_LOCK, _num_threads(self.n_jobs), torch.random.fork_rng(
            devices=[] if cpu else None, device_type="cuda" if cpu else device.type,
        ):
            if cpu:
                torch.default_generator.manual_seed(seed)
            else:
                torch.manual_seed(seed)
            return self._fit(aug_train, aug_valid, loss, base_score, seed, device)

    def _fit(self, aug_train, aug_valid, loss, base_score, seed, device) -> FitResult:
        torch_loss = TorchRieszLoss(loss)
        gen = torch.Generator().manual_seed(seed)

        dtype = _resolve_dtype(self.dtype)
        input_dim = int(aug_train.features.shape[1])

        # ---- data: each row's evaluation points, grouped by row ----
        loc, scale = _standardizer(aug_train) if self.standardize else (None, None)
        train = _AugTensors.build(aug_train, device, dtype, loc, scale)
        valid = (
            _AugTensors.build(aug_valid, device, dtype, loc, scale)
            if aug_valid is not None else None
        )

        # ---- build model + optimizer ----
        model = self.module_factory(input_dim).to(device=device, dtype=dtype)
        optimizer = self.optimizer_factory(model.parameters())
        scheduler = (
            self.scheduler_factory(optimizer)
            if self.scheduler_factory is not None
            else None
        )
        base = torch.tensor(float(base_score), device=device, dtype=dtype)

        best_score = math.inf
        best_iter = None
        early_stopping = self.early_stopping_rounds is not None
        best_state = None
        no_improve = 0
        history: list[float] = []
        snap_set = {int(e) for e in self.snapshot_epochs}
        snapshots: dict[int, dict[str, torch.Tensor]] = {}

        for epoch in range(int(self.epochs)):
            model.train()
            for rows in _row_batches(train.n_rows, self.batch_size, gen):
                optimizer.zero_grad()
                batch = None if rows is None else rows.to(device=device)
                loss_val = _batch_loss(model, base, torch_loss, train, batch)
                loss_val.backward()
                if self.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(self.grad_clip_norm)
                    )
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            done = epoch + 1
            if done in snap_set:
                snapshots[done] = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

            if valid is not None:
                model.eval()
                with torch.no_grad():
                    val = float(_batch_loss(model, base, torch_loss, valid).item())
                history.append(val)
                if val < best_score - 1e-12:
                    best_score = val
                    best_iter = epoch
                    if early_stopping:
                        best_state = {
                            k: v.detach().clone() for k, v in model.state_dict().items()
                        }
                    no_improve = 0
                else:
                    no_improve += 1
                if early_stopping and no_improve >= int(self.early_stopping_rounds):
                    break

        # Early stopping restores the best-validation weights; without it the
        # final weights are kept (as in the boosting backends).
        if best_state is not None:
            model.load_state_dict(best_state)

        # Only retain snapshot epochs that were actually reached during
        # training (early stopping may end the loop before later ticks).
        retained_epochs = tuple(sorted(snapshots.keys()))
        predictor = TorchPredictor(
            model=model,
            loss=loss,
            base_score=float(base_score),
            input_dim=input_dim,
            dtype=self.dtype,
            device=str(device),
            module_factory=self.module_factory,
            snapshot_state_dicts=snapshots if retained_epochs else None,
            snapshot_epochs=retained_epochs if retained_epochs else None,
            feature_loc=loc,
            feature_scale=scale,
            n_jobs=self.n_jobs,
        )

        return FitResult(
            predictor=predictor,
            best_iteration=best_iter if early_stopping else None,
            best_score=best_score if early_stopping and best_iter is not None else None,
            history=history if history else None,
        )


__all__ = ["TorchBackend", "TorchPredictor"]
