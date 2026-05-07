"""TorchBackend — implements ``rieszreg.MomentBackend``.

Consumes raw rows + the estimand directly (the moment-style entry point),
evaluates per-row moments via ``rieszreg.trace``, and minimizes the per-row
Bregman-Riesz loss with a PyTorch training loop. Returns a ``FitResult`` whose
predictor is a ``TorchPredictor``.
"""

from __future__ import annotations

import functools
import importlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable

import numpy as np
import torch

from rieszreg import (
    Estimand,
    FitResult,
    Loss,
    register_predictor_loader,
    trace,
)
from rieszreg.losses import loss_from_spec

from . import losses_torch
from .losses_torch import per_row_riesz_loss, validate_supported


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


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
    if isinstance(factory, functools.partial):
        inner = factory.func
        partial_kwargs = dict(factory.keywords or {})
        if factory.args:
            raise ValueError(
                "TorchBackend save/load requires `module_factory` partials to "
                "use keyword args only (no positional args), so the factory "
                "round-trips faithfully through JSON metadata."
            )
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
        if qualname is None or module is None or "." in (qualname or ""):
            raise ValueError(
                "TorchBackend save/load requires `module_factory` to be a "
                "top-level callable (e.g. a module-level `def`). Closures, "
                "lambdas, and class methods cannot be reconstructed by qualname."
            )
        return {"qualname": qualname, "module": module, "partial_kwargs": partial_kwargs}
    qualname = getattr(factory, "__qualname__", None)
    module = getattr(factory, "__module__", None)
    if qualname is None or module is None or "." in (qualname or ""):
        raise ValueError(
            "TorchBackend save/load requires `module_factory` to be a "
            "top-level callable (e.g. a module-level `def`). Closures, "
            "lambdas, and class methods cannot be reconstructed by qualname."
        )
    return {"qualname": qualname, "module": module, "partial_kwargs": None}


def _factory_from_metadata(meta: dict) -> Callable:
    mod = importlib.import_module(meta["module"])
    inner = getattr(mod, meta["qualname"])
    pk = meta.get("partial_kwargs")
    if pk:
        return functools.partial(inner, **pk)
    return inner


def _prepare_rows(
    rows: list[dict[str, Any]],
    estimand: Estimand,
    ys: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trace each row once; return packed arrays.

    ``ys`` is the sklearn-style per-row outcome; pass ``None`` when the
    estimand's m doesn't depend on Y.

    Returns
    -------
    x : (n, d) original-row feature matrix.
    pts : (N, d) stacked feature matrix for all trace points across all rows.
    coefs : (N,) trace coefficients matching ``pts``.
    pt_to_row : (N,) int index — which original row each trace point belongs to.
    """
    feature_keys = estimand.feature_keys
    d = len(feature_keys)
    if not rows:
        return (
            np.zeros((0, d), dtype=float),
            np.zeros((0, d), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=np.int64),
        )

    x = np.asarray(
        [[row[k] for k in feature_keys] for row in rows], dtype=float
    )

    pt_list: list[list[float]] = []
    coef_list: list[float] = []
    p2r_list: list[int] = []
    for i, row in enumerate(rows):
        y_i = ys[i] if ys is not None else None
        for coef, point in trace(estimand, row, y_i):
            missing = [k for k in feature_keys if k not in point]
            if missing:
                raise ValueError(
                    f"m evaluated alpha at a point missing keys {missing}; "
                    f"all feature_keys {list(feature_keys)} must be specified."
                )
            pt_list.append([point[k] for k in feature_keys])
            coef_list.append(float(coef))
            p2r_list.append(i)

    if not pt_list:
        pts = np.zeros((0, d), dtype=float)
    else:
        pts = np.asarray(pt_list, dtype=float)
    coefs = np.asarray(coef_list, dtype=float)
    pt_to_row = np.asarray(p2r_list, dtype=np.int64)
    return x, pts, coefs, pt_to_row


def _row_batches(
    n_rows: int, batch_size: int | None, generator: torch.Generator
) -> list[torch.Tensor]:
    """Return a list of LongTensor row-index batches for one epoch."""
    if batch_size is None or batch_size >= n_rows:
        return [torch.arange(n_rows, dtype=torch.long)]
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
    """

    model: torch.nn.Module
    loss: Loss
    base_score: float
    input_dim: int
    dtype: str
    device: str
    factory_metadata: dict
    feature_keys: tuple[str, ...] = field(default_factory=tuple)
    snapshot_state_dicts: dict[int, dict[str, torch.Tensor]] | None = None
    snapshot_epochs: tuple[int, ...] | None = None

    kind: ClassVar[str] = "riesznet"

    # ---- prediction ----

    def _torch_dtype(self) -> torch.dtype:
        return _resolve_dtype(self.dtype)

    def _torch_device(self) -> torch.device:
        try:
            return torch.device(self.device)
        except (RuntimeError, ValueError):
            return torch.device("cpu")

    def predict_eta(self, features: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(features, dtype=float))
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"TorchPredictor expects {self.input_dim} input features, "
                f"got {X.shape[1]}."
            )
        device = self._torch_device()
        dtype = self._torch_dtype()
        X_t = torch.as_tensor(X, dtype=dtype, device=device)
        self.model.to(device=device, dtype=dtype)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                eta_t = self.model(X_t).squeeze(-1) + self.base_score
        finally:
            if was_training:
                self.model.train()
        return eta_t.detach().cpu().numpy().astype(float)

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
        X = np.atleast_2d(np.asarray(features, dtype=float))
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"TorchPredictor expects {self.input_dim} input features, "
                f"got {X.shape[1]}."
            )
        device = self._torch_device()
        dtype = self._torch_dtype()
        X_t = torch.as_tensor(X, dtype=dtype, device=device)
        self.model.to(device=device, dtype=dtype)

        original_state = {
            k: v.detach().clone() for k, v in self.model.state_dict().items()
        }
        was_training = self.model.training
        self.model.eval()
        out = np.empty((X.shape[0], len(chosen)), dtype=float)
        try:
            for j, ep in enumerate(chosen):
                self.model.load_state_dict(self.snapshot_state_dicts[ep])
                with torch.no_grad():
                    eta_t = self.model(X_t).squeeze(-1) + self.base_score
                out[:, j] = eta_t.detach().cpu().numpy().astype(float)
        finally:
            self.model.load_state_dict(original_state)
            if was_training:
                self.model.train()
        return out

    def predict_alpha_path(
        self, features: np.ndarray, epochs: Iterable[int] | None = None
    ) -> np.ndarray:
        eta = self.predict_eta_path(features, epochs)
        return np.asarray(self.loss.link_to_alpha(eta))

    # ---- serialization ----

    def save(self, dir_path) -> None:
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
            "factory": self.factory_metadata,
            "feature_keys": list(self.feature_keys),
            "snapshot_epochs": (
                list(self.snapshot_epochs)
                if self.snapshot_epochs is not None
                else None
            ),
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
            factory_metadata=meta["factory"],
            feature_keys=tuple(meta.get("feature_keys", ())),
            snapshot_state_dicts=snap_state_dicts,
            snapshot_epochs=(
                tuple(int(e) for e in snap_epochs) if snap_epochs else None
            ),
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

    Implements ``rieszreg.MomentBackend.fit_rows``: consumes raw rows and the
    estimand, evaluates per-row moments via ``rieszreg.trace``, and minimizes
    the per-row Bregman-Riesz loss with a PyTorch training loop.

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
    validation_fraction: float = 0.0
    snapshot_epochs: tuple[int, ...] = ()

    def fit_rows(
        self,
        rows_train: list[dict[str, Any]],
        rows_valid: list[dict[str, Any]] | None,
        estimand: Estimand,
        loss: Loss,
        *,
        base_score: float,
        random_state: int,
        hyperparams: dict[str, Any],
        ys_train: list | None = None,
        ys_valid: list | None = None,
    ) -> FitResult:
        del hyperparams  # torch backend has no string-keyed passthrough

        validate_supported(loss)

        # ---- precompute traces ----
        train_x, train_pts, train_coefs, train_p2r = _prepare_rows(
            rows_train, estimand, ys_train
        )
        if rows_valid:
            valid_x, valid_pts, valid_coefs, valid_p2r = _prepare_rows(
                rows_valid, estimand, ys_valid
            )
        else:
            valid_x = valid_pts = valid_coefs = valid_p2r = None

        # ---- seeding ----
        seed = int(random_state)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        gen = torch.Generator().manual_seed(seed)

        device = _resolve_device(self.device)
        dtype = _resolve_dtype(self.dtype)
        input_dim = int(train_x.shape[1])

        # ---- build model + optimizer ----
        model = self.module_factory(input_dim).to(device=device, dtype=dtype)
        optimizer = self.optimizer_factory(model.parameters())
        scheduler = (
            self.scheduler_factory(optimizer)
            if self.scheduler_factory is not None
            else None
        )

        train_x_t = torch.as_tensor(train_x, dtype=dtype, device=device)
        train_pts_t = torch.as_tensor(train_pts, dtype=dtype, device=device)
        train_coefs_t = torch.as_tensor(train_coefs, dtype=dtype, device=device)
        train_p2r_t = torch.as_tensor(train_p2r, dtype=torch.long, device=device)
        if valid_x is not None:
            valid_x_t = torch.as_tensor(valid_x, dtype=dtype, device=device)
            valid_pts_t = torch.as_tensor(valid_pts, dtype=dtype, device=device)
            valid_coefs_t = torch.as_tensor(valid_coefs, dtype=dtype, device=device)
            valid_p2r_t = torch.as_tensor(valid_p2r, dtype=torch.long, device=device)
        base = torch.tensor(float(base_score), device=device, dtype=dtype)
        n_train = train_x_t.shape[0]

        best_score = math.inf
        best_iter = None
        best_state = None
        no_improve = 0
        history: list[float] = []

        snap_set = {int(e) for e in self.snapshot_epochs}
        snapshots: dict[int, dict[str, torch.Tensor]] = {}

        if rows_valid is None and self.early_stopping_rounds:
            raise ValueError(
                "early_stopping_rounds requires a validation set. Set "
                "validation_fraction>0 (or pass eval_set=) when fitting."
            )

        for epoch in range(int(self.epochs)):
            model.train()
            for batch_row_idx in _row_batches(n_train, self.batch_size, gen):
                batch_row_idx = batch_row_idx.to(device=device)
                optimizer.zero_grad()

                # Pick out trace rows whose origin is in this batch and remap
                # global row indices to local-batch indices.
                B = batch_row_idx.shape[0]
                row_to_local = torch.full(
                    (n_train,), -1, dtype=torch.long, device=device
                )
                row_to_local[batch_row_idx] = torch.arange(B, device=device)
                local_p2r_full = row_to_local[train_p2r_t]
                mask = local_p2r_full >= 0
                pts_b = train_pts_t[mask]
                coefs_b = train_coefs_t[mask]
                local_p2r = local_p2r_full[mask]
                x_b = train_x_t[batch_row_idx]

                # One forward pass on (orig + trace points) for efficiency.
                if pts_b.shape[0] > 0:
                    all_feat = torch.cat([x_b, pts_b], dim=0)
                else:
                    all_feat = x_b
                eta_all = model(all_feat).squeeze(-1) + base
                eta_orig = eta_all[:B]
                eta_pts = eta_all[B:]

                per_row = per_row_riesz_loss(
                    loss, eta_orig, eta_pts, coefs_b, local_p2r, B
                )
                loss_val = per_row.mean()
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

            if valid_x is not None:
                val = self._validation_loss(
                    model, base, loss,
                    valid_x_t, valid_pts_t, valid_coefs_t, valid_p2r_t,
                )
                history.append(val)
                if val < best_score - 1e-12:
                    best_score = val
                    best_iter = epoch
                    best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                if self.early_stopping_rounds and no_improve >= int(self.early_stopping_rounds):
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Build factory metadata once at end-of-fit so any importability error
        # surfaces early, before the user tries to save.
        factory_meta = _factory_metadata(self.module_factory)

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
            factory_metadata=factory_meta,
            feature_keys=tuple(estimand.feature_keys),
            snapshot_state_dicts=snapshots if retained_epochs else None,
            snapshot_epochs=retained_epochs if retained_epochs else None,
        )

        return FitResult(
            predictor=predictor,
            best_iteration=best_iter,
            best_score=best_score if best_iter is not None else None,
            history=history if history else None,
        )

    @staticmethod
    def _validation_loss(
        model: torch.nn.Module,
        base: torch.Tensor,
        loss: Loss,
        x_t: torch.Tensor,
        pts_t: torch.Tensor,
        coefs_t: torch.Tensor,
        p2r_t: torch.Tensor,
    ) -> float:
        n_valid = x_t.shape[0]
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                if pts_t.shape[0] > 0:
                    all_feat = torch.cat([x_t, pts_t], dim=0)
                else:
                    all_feat = x_t
                eta_all = model(all_feat).squeeze(-1) + base
                eta_orig = eta_all[:n_valid]
                eta_pts = eta_all[n_valid:]
                per_row = per_row_riesz_loss(
                    loss, eta_orig, eta_pts, coefs_t, p2r_t, n_valid
                )
                return float(per_row.mean().item())
        finally:
            if was_training:
                model.train()


__all__ = ["TorchBackend", "TorchPredictor"]
