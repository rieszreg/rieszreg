# RieszReg

A family of packages for **Riesz regression** — direct estimation of the Riesz representer α of a linear estimand ψ = E[m(μ)(Z)], the building block of one-step, TMLE, and DML estimators in semiparametric inference.

## Layout

```
RieszReg/
├── rieszreg/         # meta-package: shared abstractions
│   ├── python/       # Estimand, LossSpec, augmentation, RieszEstimator, testing utilities
│   └── r/rieszreg/   # shared R6 base class, estimand + loss factories
├── rieszboost/       # gradient-boosting learner (Lee & Schuler 2025)
│   ├── python/       # XGBoostBackend, SklearnBackend, RieszBooster
│   └── r/rieszboost/ # R6 wrapper subclassing rieszreg::RieszEstimatorR6
├── krrr/             # kernel ridge learner (Singh 2021)
│   ├── python/       # KernelRidgeBackend, kernels, solvers, KernelRieszRegressor
│   └── r/krrr/       # R6 wrapper subclassing rieszreg::RieszEstimatorR6
├── forestriesz/      # random-forest learner (Chernozhukov et al. ICML 2022)
│   ├── python/       # ForestRieszBackend, ForestRieszRegressor, default_riesz_features
│   └── r/forestriesz/ # R6 wrapper subclassing rieszreg::RieszEstimatorR6
├── riesznet/         # neural-network learner (Chernozhukov et al. 2021, Riesz-rep only)
│   ├── python/       # TorchBackend, TorchPredictor, RieszNet
│   └── r/riesznet/   # R6 wrapper subclassing rieszreg::RieszEstimatorR6
├── docs/             # unified Quarto user guide (sklearn-style sectioning)
├── reference/        # arXiv paper index, shared across packages
├── .githooks/        # pre-commit hook + shared lint-docs.sh (living-doc rule + tone lint)
├── .github/workflows # CI (pytest + R parity, docs deploy, doc-tone lint)
├── scripts/          # setup-hooks.sh (one-time per-clone hook activation)
└── rieszreg/DESIGN.md  # meta-package design + learner-package contract
```

The user guide is a single Quarto site at [`docs/`](docs/) — sklearn-style, not per-package — covering Concepts, Get started, Usage, Backends (one sub-page per backend package), R interface, Developing, and References.

## Install

The five packages live in sibling GitHub repos:
[rieszreg](https://github.com/rieszreg/rieszreg) (this repo, the meta-package + unified docs),
[rieszboost](https://github.com/rieszreg/rieszboost),
[krrr](https://github.com/rieszreg/krrr),
[forestriesz](https://github.com/rieszreg/forestriesz),
[riesznet](https://github.com/rieszreg/riesznet).
Clone them as siblings into a parent directory; the docs builds and CI assume
that layout.

```sh
mkdir RieszReg && cd RieszReg
git clone https://github.com/rieszreg/rieszreg.git
git clone https://github.com/rieszreg/rieszboost.git
git clone https://github.com/rieszreg/krrr.git
git clone https://github.com/rieszreg/forestriesz.git
git clone https://github.com/rieszreg/riesznet.git
python3 -m venv .venv
.venv/bin/pip install -e rieszreg/python
.venv/bin/pip install -e rieszboost/python      # gradient-boosting backend
.venv/bin/pip install -e krrr/python            # kernel-ridge backend
.venv/bin/pip install -e forestriesz/python     # random-forest backend
.venv/bin/pip install -e riesznet/python        # neural-network backend
```

`rieszboost`'s `XGBoostBackend` requires OpenMP; on macOS, `brew install libomp` once.

## Quickstart

Pick any learner package and compose it with `RieszEstimator`:

```python
from rieszreg import RieszEstimator, ATE

# Gradient boosting
from rieszboost.backends import XGBoostBackend
est = RieszEstimator(estimand=ATE(), backend=XGBoostBackend())

# Kernel ridge
from krrr import KernelRidgeBackend, Gaussian
est = RieszEstimator(estimand=ATE(), backend=KernelRidgeBackend(kernel=Gaussian()))

# Random forest
from forestriesz import ForestRieszBackend
est = RieszEstimator(estimand=ATE(), backend=ForestRieszBackend(n_estimators=500))

# Neural network
from riesznet import TorchBackend
est = RieszEstimator(estimand=ATE(), backend=TorchBackend(epochs=200))

est.fit(df)
alpha_hat = est.predict(df)
```

Each learner package also ships a convenience subclass (`RieszBooster`, `KernelRieszRegressor`, `ForestRieszRegressor`, `RieszNet`) with backend-specific hyperparameters on the constructor. See the [backends comparison](https://rieszreg.github.io/rieszreg/backends/) to choose.

## Status

- **rieszreg** v0.0.1 — feature-complete: estimand machinery, four Bregman losses, augmentation engine, both `Backend` (augmentation-style) and `MomentBackend` (moment-style) Protocols with orchestrator dispatch, RieszEstimator, base R6 class, testing utilities. 71 Python tests passing.
- **rieszboost** v0.0.1 — sklearn-compatible `RieszBooster` with `XGBoostBackend` (default) and `SklearnBackend`; 112 Python tests + 11 R parity tests passing.
- **krrr** v0.0.1 — sklearn-compatible `KernelRieszRegressor`; four solvers (direct, Nyström-CG, RFF, optional Falkon); 36 Python tests + 1 R parity test passing.
- **forestriesz** v0.0.1 — sklearn-compatible `ForestRieszRegressor` on EconML's `BaseGRF`; locally constant + locally linear sieve fits; honest-split `predict_interval`; 55 Python tests + 1 R parity test passing.
- **riesznet** v0.0.1 — sklearn-compatible `RieszNet` (default-MLP) and `TorchBackend` (arbitrary `nn.Module` factories) trained with PyTorch autograd against any of the four built-in Bregman losses; 41 Python tests + 1 R parity test passing.

## Related work

A few existing tools cover overlapping ground.

[**genriesz**](https://github.com/MasaKat0/genriesz) (Kato, 2026) is a single Python package implementing the Bregman-unified Riesz regression framework from [arXiv:2601.07752](https://arxiv.org/abs/2601.07752). It exposes `LinearFunctional` and `BregmanGenerator` abstractions, analogous to this project's `Estimand` and `Loss`. It ships several basis-function classes (polynomial, random Fourier features, Nyström, KNN catchments, random-forest leaves, PyTorch embeddings) inside the package itself. Third parties cannot publish their own learners against a stable protocol. It is Python only.

[**EconML**](https://github.com/py-why/EconML) (Microsoft) provides `RieszNet`, `ForestRiesz`, and an `automatic_debiased_ml` module. The `forestriesz` package in this repo wraps EconML's `BaseGRF`. EconML is monolithic, with no third-party backend protocol, and Python only.

[**DoubleML**](https://docs.doubleml.org/) (Bach, Chernozhukov, Kurz, Spindler) is a mature DML library with parallel Python and R implementations. It expects the user to supply outcome and propensity nuisances using sklearn-compatible learners. Riesz regression is not the focal abstraction.

[**tlverse**](https://tlverse.org/) (van der Laan group) is an R-only family of packages (`sl3`, `tmle3`, `lmtp`, `hal9001`, …) organized around TMLE and SuperLearner. The meta-package + sibling-backends shape is the closest organizational match to this project.

What's distinctive here:

- The `Backend` / `MomentBackend` split, exposed as a stable Protocol, lets a third-party learner package depend on `rieszreg` and ship as its own PyPI/CRAN release. New learners do not require a PR upstream.
- The split itself reflects two structurally different fitting strategies: augmentation-style (kernel ridge, gradient boosting via `fit_augmented`) vs. moment-style (forests, neural nets via `fit_rows`).
- Cross-language Python + R coverage at the family level via R6 wrappers per package, not just bindings to a Python core.

## Tests

```sh
.venv/bin/python -m pytest rieszreg/python/tests -q
.venv/bin/python -m pytest rieszboost/python/tests -q
.venv/bin/python -m pytest krrr/python/tests -q
.venv/bin/python -m pytest forestriesz/python/tests -q
.venv/bin/python -m pytest riesznet/python/tests -q

Rscript -e '
  RETICULATE_PYTHON <- file.path(getwd(), ".venv/bin/python")
  Sys.setenv(RETICULATE_PYTHON = RETICULATE_PYTHON)
  pkgload::load_all("rieszreg/r/rieszreg")
  pkgload::load_all("rieszboost/r/rieszboost")
  testthat::test_dir("rieszboost/r/rieszboost/tests/testthat")
  pkgload::load_all("krrr/r/krrr")
  testthat::test_dir("krrr/r/krrr/tests/testthat")
  pkgload::load_all("forestriesz/r/forestriesz")
  testthat::test_dir("forestriesz/r/forestriesz/tests/testthat")
  pkgload::load_all("riesznet/r/riesznet")
  testthat::test_dir("riesznet/r/riesznet/tests/testthat")
'
```

## Contributing a new learner package

`RIESZREG_DESIGN.md` (Part B) is the contract: depend on `rieszreg`, implement either the `Backend` Protocol (augmentation-style — for kernel ridge, gradient boosting) or the `MomentBackend` Protocol (moment-style — for random forests, neural nets), satisfy the sklearn-conformance subset, contribute docs pages to `docs/`, follow the doc-tone and living-doc rules. The pre-commit hook at `.githooks/pre-commit` enforces the doc-tone and API-changes-update-docs rules; activate it once per clone with `bash scripts/setup-hooks.sh`. The `lint-docs` job in `.github/workflows/test.yml` mirrors the doc-tone check in CI.

## References

The meta-project's [`reference/`](reference/) directory indexes the foundational papers (Lee & Schuler 2025, Chernozhukov et al., Singh, Hines & Miles, Kato, van der Laan et al.) with arXiv IDs and a refetch script.
