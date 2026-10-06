# RieszReg

A family of packages for **Riesz regression** — direct estimation of the Riesz representer α of a linear estimand ψ = E[m(μ)(Z)], the building block of one-step, TMLE, and DML estimators in semiparametric inference.

New here? Start with the [`rieszreg` README](packages/rieszreg/README.md): key terms, a quickstart, cross-fitting with sklearn, and turning $\hat\alpha$ into an ATE estimate.

## Packages

All six packages live in this uv workspace under [`packages/`](packages/):

| Package | Learner class | Method |
|---|---|---|
| [`rieszreg`](packages/rieszreg/) | `RieszEstimator` | Shared estimands, losses, and sklearn glue |
| [`rieszboost`](packages/rieszboost/) | `RieszBooster` | Gradient boosting (Lee & Schuler 2025) |
| [`krrr`](packages/krrr/) | `KernelRieszRegressor` | Kernel ridge regression (Singh 2021) |
| [`forestriesz`](packages/forestriesz/) | `AugForestRieszRegressor`, `ForestRieszRegressor` | Random forest (Chernozhukov et al. 2022) |
| [`riesztree`](packages/riesztree/) | `RieszTreeRegressor` | A single decision tree |
| [`riesznet`](packages/riesznet/) | `RieszNet` | Neural network (Chernozhukov et al. 2021) |

Each package has a Python module under `python/` and an R6 wrapper under `r/`. The user guide is a single Quarto site at [`docs/`](docs/).

## Install

```sh
git clone https://github.com/rieszreg/rieszreg.git
cd rieszreg
uv sync --all-packages --all-extras
```

`rieszboost` needs OpenMP; on macOS, run `brew install libomp` once.

## Quickstart

Every learner is an sklearn estimator. Pick one, tell it the estimand and the treatment column, then `fit` and `predict`:

```python
from rieszboost import RieszBooster, ATE

est = RieszBooster(estimand=ATE(treatment="treated"))
est.fit(Z)                 # Z: DataFrame with the treatment column + covariates
alpha_hat = est.predict(Z)
```

Swap `RieszBooster` for `KernelRieszRegressor` (krrr), `AugForestRieszRegressor` (forestriesz), `RieszTreeRegressor` (riesztree), or `RieszNet` (riesznet). Each package re-exports the estimands and losses, so one import line is enough. See the [backends comparison](https://rieszreg.github.io/rieszreg/backends/) to choose.

## Related work

A few existing tools cover overlapping ground.

[**genriesz**](https://github.com/MasaKat0/genriesz) (Kato, 2026) is a single Python package implementing the Bregman-unified Riesz regression framework from [arXiv:2601.07752](https://arxiv.org/abs/2601.07752). It exposes `LinearFunctional` and `BregmanGenerator` abstractions, analogous to this project's `Estimand` and `Loss`. It ships several basis-function classes (polynomial, random Fourier features, Nyström, KNN catchments, random-forest leaves, PyTorch embeddings) inside the package itself. Third parties cannot publish their own learners against a stable protocol. It is Python only.

[**EconML**](https://github.com/py-why/EconML) (Microsoft) provides `RieszNet`, `ForestRiesz`, and an `automatic_debiased_ml` module. The `forestriesz` package in this repo wraps EconML's `BaseGRF`. EconML is monolithic, with no third-party backend protocol, and Python only.

[**DoubleML**](https://docs.doubleml.org/) (Bach, Chernozhukov, Kurz, Spindler) is a mature DML library with parallel Python and R implementations. It expects the user to supply outcome and propensity nuisances using sklearn-compatible learners. Riesz regression is not the focal abstraction.

[**tlverse**](https://tlverse.org/) (van der Laan group) is an R-only family of packages (`sl3`, `tmle3`, `lmtp`, `hal9001`, …) organized around TMLE and SuperLearner. The meta-package + sibling-backends shape is the closest organizational match to this project.

What's distinctive here:

- The `Backend` / `MomentBackend` split, exposed as a stable Protocol, lets a third-party learner package depend on `rieszreg` and ship as its own PyPI/CRAN release. New learners do not require a PR upstream.
- The split itself reflects two structurally different fitting strategies: augmentation-style (kernel ridge, gradient boosting, trees, neural nets via `fit_augmented`) vs. moment-style (the EconML-based forest via `fit_rows`).
- Cross-language Python + R coverage at the family level via R6 wrappers per package, not just bindings to a Python core.

## Tests

Python tests use the uv workspace:

```sh
uv sync --all-packages --all-extras
for pkg in rieszreg rieszboost krrr forestriesz riesznet riesztree; do
  uv run pytest "packages/$pkg/python/tests" -q
done
```

R parity tests use a one-time pak install of the workspace, then `library()` + `testthat::test_dir`:

```sh
Rscript tools/r/install.R   # installs all 6 R packages from packages/*/r/*/

RETICULATE_PYTHON=$(uv run python -c 'import sys; print(sys.executable)') \
  Rscript -e '
    library(rieszreg)
    for (pkg in c("rieszboost", "krrr", "forestriesz", "riesznet", "riesztree")) {
      library(pkg, character.only = TRUE)
      testthat::test_dir(file.path("packages", pkg, "r", pkg, "tests", "testthat"))
    }
  '
```

## Contributing a new learner package

[`DESIGN.md`](DESIGN.md) (Part B) is the contract: depend on `rieszreg`, implement either the `Backend` Protocol (augmentation-style — for kernel ridge, gradient boosting, trees, neural nets) or the `MomentBackend` Protocol (moment-style — for learners that fit on the original rows, such as the EconML-based forest), satisfy the sklearn-conformance subset, contribute docs pages to `docs/`, follow the doc-tone and living-doc rules. The pre-commit hook at `.githooks/pre-commit` enforces the doc-tone and API-changes-update-docs rules; activate it once per clone with `bash scripts/setup-hooks.sh`. The `lint-docs` job in `.github/workflows/test.yml` mirrors the doc-tone check in CI.

## References

The meta-project's [`reference/`](reference/) directory indexes the foundational papers (Lee & Schuler 2025, Chernozhukov et al., Singh, Hines & Miles, Kato, van der Laan et al.) with arXiv IDs and a refetch script.
