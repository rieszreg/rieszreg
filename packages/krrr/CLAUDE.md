# krrr

> **Read the family design doc first.** It lives in the rieszreg meta-package
> at `rieszreg/DESIGN.md` (clone [rieszreg/rieszreg](https://github.com/rieszreg/rieszreg) as a sibling, then it's at
> [`../rieszreg/DESIGN.md`](../rieszreg/DESIGN.md)). Part B is the contract this package implements —
> anything in this CLAUDE.md is krrr-specific notes layered on top.

Kernel-ridge backend for the [RieszReg meta-package](../README.md), implementing Singh ([arXiv:2102.11076](https://arxiv.org/abs/2102.11076)) for the full set of estimands the rieszreg framework supports.

This package depends on `rieszreg` for the shared abstractions (`Estimand`, `Loss`, `AugmentedDataset`, `Diagnostics`, `Backend` Protocol, `RieszEstimator` orchestrator). `krrr` contributes:

- `KernelRidgeBackend` — `Backend` Protocol implementation; closed-form solve over the augmented dataset.
- `KernelRieszRegressor` — convenience subclass of `rieszreg.RieszEstimator` with kernel-specific hyperparameters (`kernel`, `lambda_grid`, `solver`, ...) on the constructor.
- Kernel algebra (`Gaussian`, `Matern`, `Linear`, `Polynomial`, `Tensor`, `Sum`, `Product`, `Scaled`) and four solvers (`direct`, `nystrom_cg`, `rff`, optional `falkon`).
- R6 wrapper subclassing `rieszreg::RieszEstimatorR6`.

This package depends only on `rieszreg`.

## Living-doc rule (README + meta-project docs)

`README.md` is a living document — update it in the same edit whenever a change touches the public API surface (new kernel, new solver, new option on `KernelRieszRegressor`, change to defaults). If a change makes any line in the README false or outdated, the change is not done until the README is fixed.

The user guide is the unified Quarto site at [`../docs/`](../docs/). The kernel-specific page is [`../docs/backends/kernel.qmd`](../docs/backends/kernel.qmd). Any change to the kernel backend that affects user-facing behavior must update that page in the same edit. On bilingual pages, update BOTH the `{python}` and `{r}` tabs.

The pre-commit hook at the monorepo root (`../../.githooks/pre-commit`) enforces this; activate it once per clone with `bash ../../scripts/setup-hooks.sh`. The same lint runs in CI via the `lint-docs` job.

## Per-estimand example rule

Estimand factories live in `rieszreg`. When a *kernel-side* feature is added (new kernel, new solver), add a corresponding example in `examples/` that exercises it on a realistic DGP.

## R wrapper scope

The R6 wrapper exposes built-in estimands only. Custom `m()` is Python-only — the `LinearForm` tracer lives in `rieszreg` and is not ported. R users who need a brand-new functional write the `m()` in Python and call into it from R via reticulate.

## API design rule

The public API should feel like **ngboost / sklearn**:

- Object-oriented factory `KernelRieszRegressor(estimand=, kernel=, lambda_grid=, solver=, ...)`. `BaseEstimator`-compatible `fit / predict / score / diagnose`. Anything that can't compose with `sklearn.model_selection` (`GridSearchCV`, `cross_val_predict`, `Pipeline`) is a regression and should be fixed.
- **No `feature_keys` (or other input-schema args) on `fit()` / `predict()`.** The estimand owns its input schema.
- Cross-fitting is `sklearn.model_selection.cross_val_predict`. No bespoke `crossfit()`.
- Hyperparameter tuning is `sklearn.model_selection.GridSearchCV`. No `tune_riesz()`.

R-side mirrors this: R6 classes (`KernelRieszRegressor$new(estimand=, kernel=, ...)$fit(df)$predict(df)`).

## Layout

- `python/krrr/` — `KernelRidgeBackend`, kernels, solvers, and the `KernelRieszRegressor` convenience class. `pyproject.toml` declares `rieszreg>=0.0.1` as the dependency.
- `r/krrr/` — R6 wrapper via reticulate. `KernelRieszRegressor` subclasses `rieszreg::RieszEstimatorR6` (~50 lines locally). Estimand and loss factories are re-exported from `rieszreg` via NAMESPACE.
- `examples/` — runnable demonstrations of each feature.
- `.venv/` — local Python venv (gitignored).

## Run tests

```sh
.venv/bin/python -m pytest python/tests -v
```

## Architecture notes

### Dependency on rieszreg

`krrr` depends on `rieszreg` and reuses, without modification:

- `Estimand`, `Tracer`/`LinearForm`, `Estimand.augment`, `AugmentedDataset` — the moment-functional abstraction and its data-augmentation engine.
- `Loss`, `SquaredLoss` — the Bregman-Riesz loss framework. (KLLoss / Bernoulli / BoundedSquared are NOT yet supported by the kernel backend; v0.2.)
- `Diagnostics`, `diagnose` — base diagnostics (`KernelDiagnostics` extends with kernel-specific extras).
- `RieszEstimator` — orchestration; `KernelRieszRegressor` is a thin subclass with the kernel backend defaulted.

The integration point is `rieszreg`'s `Backend` Protocol (`rieszreg/backends/base.py`). `KernelRidgeBackend.fit_augmented(...)` consumes an `AugmentedDataset` + `Loss` and returns a `FitResult` whose predictor exposes `predict_eta` / `predict_alpha`. `KernelPredictor` registers itself for the registry-based save/load path on import via `register_predictor_loader("krrr", ...)`.

### Augmentation → kernel solve

Given the augmented dataset (per-point quadratic coefficient `a_k` and linear coefficient `b_k`), the squared Riesz loss is

    L_n(α) = (1/n) Σ_k [a_k α(p_k)² + b_k α(p_k)] + λ ‖α‖²_H

The representer theorem gives `α̂ = Σ_k γ_k k(·, p_k)` and the first-order condition is

    (diag(a) K + n λ I) γ = − b / 2

`Estimand.augment` produces `a_k ∈ {0, 1}` (1 for the original observation row, 0 for counterfactual evaluation points introduced by `m`). Partition `o = {a_k > 0}` and `c = {a_k = 0}`:

- Row k ∈ c reduces to `n λ γ_k = − b_k / 2`, so `γ_c = − b_c / (2 n λ)` is closed-form.
- Row k ∈ o solves a symmetric PSD system via the substitution γ̃ = D^{1/2} γ:

      (K̃_oo + n λ I) γ̃ = D^{−1/2} (−b_o/2 + K_oc b_c / (2 n λ))

  with K̃_oo = D^{1/2} K_oo D^{1/2}, D = diag(a_o).

A single eigendecomposition of K̃_oo solves the entire λ path in O(n_o²) per λ.

For TSM1 with a Gaussian kernel and `a_k ≡ 1`, this recovers the closed form in [Singh (2021, arXiv:2102.11076)](https://arxiv.org/abs/2102.11076) and the R reference at `~/Desktop/dml-tmle/code/R/learners/krrr.R`. See `python/tests/test_reference_parity.py` for a 1e-8 round-trip check.

### Solver tier

| Solver | Best for | Cost | Notes |
|---|---|---|---|
| `direct` | n_aug ≤ 3000 | O(n³) eigendecomposition once + O(n²) per λ | Exact. Default for small n. |
| `nystrom_cg` | n_aug ≤ 50k | O(m² n) per λ + a few CG iterations | Nyström-preconditioned CG on the symmetric o-block. m landmarks. |
| `rff` | very large n, shift-invariant kernel | O(n D + D³) per λ | Random Fourier features (Rahimi-Recht); primal solve. |
| `falkon` | very large n, GPU available | depends | Optional dependency `pip install krrr[falkon]`. Wraps the `falkon` package. |

`solver="auto"` dispatches by `n_aug`.

### What's lazy-imported

`falkon`, `pykeops`, and `pandas` are optional. The core path uses only numpy, scipy, scikit-learn, and rieszreg.

## What works today (v0.0.1)

- **`KernelRieszRegressor(BaseEstimator)`** — sklearn-compatible. Composes with `GridSearchCV`, `cross_val_predict`, `clone`, `Pipeline`. Same `fit / predict / score / diagnose` surface as `RieszBooster`.
- **All five built-in estimands** via the rieszreg re-exports: `ATE`, `ATT`, `TSM`, `AdditiveShift`, `LocalShift`. Custom `FiniteEvalEstimand`s also work (the augmentation engine is identical). `StochasticIntervention` is currently stubbed in rieszreg and will be reintroduced.
- **Kernels**: `Gaussian`, `Matern(nu={0.5, 1.5, 2.5})`, `Linear`, `Polynomial`, `Tensor` (tensor product over disjoint feature subsets), `Sum`, `Product`, `Scaled`. Spec-round-trippable.
- **Bandwidth selection**: `length_scale={float, "median", "scott", "silverman"}`. Median heuristic (default) resolves on `fit_data`.
- **Solvers**: `direct`, `nystrom_cg`, `rff`, optional `falkon`. `solver="auto"` dispatches by n_aug.
- **λ selection** via validation Riesz loss (default: 20% holdout). `lambda_grid` is a sequence; the chosen value surfaces as `regressor.lambda_`.
- **Loss**: `SquaredLoss` only (closed-form linear-system solve). KLLoss / BernoulliLoss / BoundedSquaredLoss require Newton iteration on the kernel system; planned for v0.2.
- **Save / load**: directory format with JSON metadata + npz tensors. Round-trip works for built-in estimands automatically; custom estimands require `estimand=` on load.
- **Diagnostics**: `KernelDiagnostics` extends `rieszreg.Diagnostics` with chosen λ, support size, effective d.o.f., condition number, and ill-conditioning warnings.
- **R wrapper**: R6 mirror via reticulate. Built-in estimands only.
- **31 Python tests** covering: backend protocol satisfaction, end-to-end ATE recovery, all six built-in estimands + custom, save/load round-trip, solver equivalence (direct vs nystrom_cg vs rff), kernel correctness (PSD, algebra, spec round-trip), sklearn integration (clone, GridSearchCV, cross_val_predict), and TSM1 numerical parity with the dml-tmle krrr.R reference.

## Known sharp edges

- **λ scaling.** For consistency theory the regularizer should scale O(1/n). The default grid `np.logspace(-4, 0, 21)` covers a wide range; cross-fitting users should re-tune per fold (sklearn's `cross_val_predict` does this if KRRR is wrapped in `GridSearchCV`).
- **Median-heuristic bandwidth on the augmented dataset.** The median is computed on the *augmented* points (originals + counterfactuals from `m`). This adapts to the joint scale of `(treatment, covariates)` automatically; for shift-style estimands it includes the shifted treatment values, which is usually what you want.
- **`solver="falkon"` drops the K_oc b_c coupling on the o-block.** The standalone Falkon API only solves vanilla KRR, not the modified-RHS system the augmentation produces. For estimands where n_c is small or λ is moderate, the bias is small; for tight overlap or extreme λ it is not. Use `solver="nystrom_cg"` if exactness matters more than scale.
- **`KLLoss`, `BernoulliLoss`, `BoundedSquaredLoss` raise.** These need an iterative kernel Newton scheme that v0.1 doesn't implement. Planned for v0.2.

## What's next

See README's `## On the roadmap` section. Headlines: KLLoss / Newton-iterated KRR; KeOps lazy kernel ops; learned bandwidth via marginal-likelihood; benchmarks at n = 10⁵–10⁶.
