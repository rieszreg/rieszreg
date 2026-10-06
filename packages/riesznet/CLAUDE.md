# riesznet

> **Read the family design doc first.** It lives at the monorepo root:
> [`../../DESIGN.md`](../../DESIGN.md). Part B is the contract this package implements —
> anything in this CLAUDE.md is riesznet-specific notes layered on top.

Neural-network backend for the [RieszReg meta-package](../README.md), in the spirit of [Chernozhukov et al. (2021)](https://arxiv.org/abs/2110.03031). Trains the Riesz representer α(x) only — outcome regression is the user's responsibility.

This package depends on `rieszreg` for the shared abstractions (`Estimand`, `Loss`, `Backend` Protocol, `Diagnostics`, `RieszEstimator` orchestrator, `AugmentedDataset`). `riesznet` contributes:

- `TorchBackend` — `Backend.fit_augmented` Protocol implementation. Receives the augmented dataset, groups each original row's evaluation points by `origin_index`, and minimizes the per-row Bregman-Riesz loss with a PyTorch training loop.
- `TorchPredictor` — wraps the trained `nn.Module` and registers itself with the rieszreg loader registry.
- `RieszNet` — convenience subclass of `rieszreg.RieszEstimator` exposing simple-MLP defaults (`hidden_sizes`, `activation`, `dropout`, `learning_rate`, `epochs`, ...). Power users instantiate `TorchBackend(module_factory=..., optimizer_factory=...)` directly for full architecture control.
- R6 wrapper subclassing `rieszreg::RieszEstimatorR6`.

## Living-doc rule (README + meta-project docs)

`README.md` is a living document — update it in the same edit whenever a change touches the public API surface. If a change makes any line in the README false or outdated, the change is not done until the README is fixed.

The user guide is the unified Quarto site at [`../docs/`](../docs/). The neural-specific page is [`../docs/backends/neural.qmd`](../docs/backends/neural.qmd). Any change to the neural backend that affects user-facing behavior must update that page in the same edit. On bilingual pages, update BOTH the `{python}` and `{r}` tabs.

## R wrapper scope

The R6 wrapper exposes the simple-MLP knobs (`hidden_sizes`, `activation`, `dropout`, `learning_rate`, `weight_decay`, `epochs`, `device`). Custom torch architectures (custom `nn.Module`, custom optimizer, custom scheduler) are Python-only — the factory callables don't survive the reticulate boundary cleanly. R users who need a custom architecture write the factory in Python and call into Python via reticulate.

## API design rule

Object-oriented factory `RieszNet(estimand=, hidden_sizes=, ...)`, `BaseEstimator`-compatible `fit / predict / score / diagnose`, composes with `sklearn.model_selection` (`GridSearchCV`, `cross_val_predict`, `Pipeline`). No `feature_keys` on `fit()` / `predict()`. Cross-fitting is `cross_val_predict`; tuning is `GridSearchCV`.

## Layout

- `python/riesznet/`
  - `backend.py` — `TorchBackend` (`Backend.fit_augmented`), `TorchPredictor`, predictor-loader registration.
  - `losses_torch.py` — `TorchRieszLoss`: torch-autograd `h̃(α(η))` and `h'(α(η))` for each Bregman loss (resolved once per loss) and the per-row Riesz loss.
  - `modules.py` — top-level default factories (`build_mlp`, `build_adam`) for the convenience class.
  - `estimator.py` — `RieszNet` convenience subclass of `RieszEstimator`.
- `r/riesznet/` — R6 wrapper via reticulate, ~120 lines.
- `examples/` — runnable demonstrations (ATE, TSM).

## Run tests

```sh
.venv/bin/python -m pytest python/tests -v
```

R parity:

```sh
Rscript -e '
  Sys.setenv(RETICULATE_PYTHON = file.path(getwd(), ".venv/bin/python"))
  pkgload::load_all("../rieszreg/r/rieszreg")
  pkgload::load_all("r/riesznet")
  testthat::test_dir("r/riesznet/tests/testthat")
'
```

## Architecture notes

### Dependency on rieszreg

`riesznet` depends on `rieszreg` and reuses, without modification:

- `Estimand`, `Estimand.augment`, `AugmentedDataset` — the moment-functional abstraction. The neural backend groups the augmented rows by `origin_index` to get each original row's loss terms.
- `Loss`, all four built-in losses — the Bregman-Riesz loss framework.
- `Diagnostics`, `diagnose` — base diagnostics.
- `RieszEstimator` — orchestration; `RieszNet` is a thin subclass.

The integration point is `rieszreg`'s `Backend` Protocol (`rieszreg/backends/base.py`). `TorchBackend.fit_augmented(...)` consumes the augmented datasets the orchestrator builds and returns a `FitResult`. `TorchPredictor` registers itself for the registry-based save/load path on import via `register_predictor_loader("riesznet", ...)`.

### Per-row Bregman-Riesz loss

For each original row `z_i`, the per-row Riesz loss sums the augmented rows `r` with `origin_index == i`:

```
L_i = Σ_r [ D_r · h̃(α(z_r)) + C_r · h'(α(z_r)) ]
```

This is `Loss.aug_loss_alpha` summed per original row, so values (not just gradients) match the numpy losses and `riesz_loss`. `losses_torch.py` writes `h̃(α(η))` and `h'(α(η))` directly per loss:

| Loss | `h̃(α(η))` | `h'(α(η))` |
|---|---|---|
| `SquaredLoss` | `η²` | `2η` |
| `KLLoss` | `exp(η)` | `η + 1` |
| `BernoulliLoss` | `softplus(η)` | `η` |
| `BoundedSquaredLoss` | `(lo + R·σ(η))²` | `2(lo + R·σ(η))` |

η is clamped per the loss spec's `max_eta` / `max_abs_eta` for numerical stability, matching the existing backends.

### Training loop

`_AugTensors` holds the augmented rows sorted by origin with CSR offsets. A minibatch of original rows gathers only its own augmented rows (O(batch) per step), runs one forward pass, and `scatter_add`s the per-row terms. Validation loss is the same formula on the held-out rows. The convenience class defaults to `batch_size=64`; set `batch_size=None` for full-batch GD on small problems. With `early_stopping_rounds=None` all epochs run and the final weights are kept, even when an `eval_set` is passed.

### Save / load

Saves the model's `state_dict` plus a JSON metadata blob carrying the `module_factory` qualname (and `functools.partial` kwargs if applicable). Load re-imports the factory by qualname, rebuilds the module, and calls `load_state_dict`. The default `RieszNet` MLP path uses `functools.partial(riesznet.modules.build_mlp, ...)` so it round-trips cleanly. Closures, lambdas, and locally-defined classes raise on save with a clear error.

### Hyperparameter forwarding

`epochs` and `batch_size` live on `TorchBackend` as dataclass fields. The optimizer's LR is the source of truth (set by `optimizer_factory`). `RieszNet` exposes `learning_rate` and `weight_decay` as ctor args and folds them into the default `Adam` factory via `functools.partial`. `RieszEstimator` no longer carries any iterative-method knobs — those are owned by the backend.

### Device / dtype

Default `cpu` and `float32`. `device="cuda"` and `device="mps"` work if the corresponding torch backend is available; fitting on an unavailable device raises. A saved model whose device isn't available predicts on CPU, so GPU-saved models load anywhere. `device="auto"` skips MPS for `float64`. `dtype="float64"` works at a small speed penalty. Bitwise reproducibility on CUDA is not promised; the loop seeds `torch`, `torch.cuda`, and a `torch.Generator` for the DataLoader, but does not enable `torch.use_deterministic_algorithms`.

### What's lazy-imported

`torch` is a hard dependency, not lazy; `riesznet`'s import surface always includes the model classes. `pandas` is needed only for tests.

## What works today (v0.0.1)

- **`RieszNet(BaseEstimator)`** — sklearn-compatible. Composes with `GridSearchCV`, `cross_val_predict`, `clone`, `Pipeline`. Same `fit / predict / score / diagnose` surface as `RieszBooster`, `KernelRieszRegressor`, `ForestRieszRegressor`.
- **All six built-in estimands** via the rieszreg re-exports. Custom `Estimand`s also work through the tracer-based default `augment`.
- **All four built-in losses**: `SquaredLoss`, `KLLoss`, `BernoulliLoss`, `BoundedSquaredLoss` (autograd-friendly torch implementations matching the analytic gradients).
- **Architecture flexibility**: pass any `nn.Module` factory via `TorchBackend(module_factory=..., optimizer_factory=...)`. The convenience class `RieszNet` exposes a simple MLP path with `hidden_sizes`, `activation`, `dropout`.
- **Save / load**: `state_dict` + JSON metadata. Built-in estimands round-trip automatically; the default MLP factory round-trips cleanly via qualname; user-defined factories must be importable by qualname.
- **Early stopping**: `early_stopping_rounds` measures epochs without validation-loss improvement; restores best-validation weights at end of fit.
- **Diagnostics**: inherits `rieszreg.Diagnostics`.
- **R wrapper**: simple-MLP knobs only.

## Known sharp edges

- **`torch.save(model, path)` is not used** — only `state_dict`. Users with one-off `nn.Module` subclasses defined in notebook cells will get a clear `RuntimeError` on `save()` pointing at the workaround (move the class to a top-level module).
- **No multi-GPU / distributed training** — single-device only in v1.
- **No mixed precision** — `dtype` is one of `float32` / `float64` end-to-end.
- **No deterministic CUDA flag** — bitwise reproducibility on GPU is not promised; the seeding is best-effort.
- **Default init is the empirical loss-minimizing constant** (`m̄ = E[m(Z, 1)]` projected to the loss's α-domain). Override with `init=<float>` when you have a better prior. The `"m1"` string mode from earlier versions is gone — the default now does what `"m1"` did but generalized correctly to non-squared Bregman losses.

## What's next

- Mixed-precision training for large architectures.
- Distributed / multi-GPU.
- Optional deterministic-algorithms flag for fully-reproducible CUDA runs.
