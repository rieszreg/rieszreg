# RieszReg monorepo

**This is research code, not production.** Breaking changes are fine. Don't write deprecation shims, don't preserve old API names "for compatibility", don't paranoidly validate user inputs at every entry point. Tests should verify the libraries do what they're meant to do statistically and conceptually — not that warnings fire on bad input.

> **Read [`DESIGN.md`](DESIGN.md) first.** Authoritative architecture for the family — Part A is the meta-package, Part B is the contract every learner package must satisfy. Anything in this file is operational notes; anything in DESIGN.md is contract.

> **Skills load on the relevant edits.** Do not duplicate their content here:
> - [`rieszreg-notation`](.claude/skills/rieszreg-notation/SKILL.md) — math/prose/code naming conventions and doc-tone rules. Triggers on user-guide prose, READMEs, docstrings, comments.
> - [`rieszreg-coding`](.claude/skills/rieszreg-coding/SKILL.md) — Tier-1/2/3 placement, OOP-first, fewer-lines preference, sklearn-first, research-code mindset, test philosophy. Triggers on any code edit in `packages/`.
> - [`rieszreg-dev-environment`](.claude/skills/rieszreg-dev-environment/SKILL.md) — `uv sync` workflow, worktree usage, Quarto rendering ops.
> - [`rieszreg-backend-contract`](.claude/skills/rieszreg-backend-contract/SKILL.md) — checklist for adding a new learner. Triggers on new backend implementations.
> - [`rieszreg-cross-package-change`](.claude/skills/rieszreg-cross-package-change/SKILL.md) — prose-and-examples sweep after a public-API rename.

A uv-workspace monorepo for the Riesz-regression package family. Six members under `packages/`:

| Member | Role |
|---|---|
| [`packages/rieszreg`](packages/rieszreg) | meta-package: shared `Estimand`, `Loss`, `RieszEstimator`, augmentation, diagnostics, `Backend` / `MomentBackend` Protocols, R6 base class |
| [`packages/rieszboost`](packages/rieszboost) | gradient-boosting learner (Lee & Schuler 2025; `fit_augmented`) |
| [`packages/krrr`](packages/krrr) | kernel-ridge learner (Singh 2021; `fit_augmented`) |
| [`packages/forestriesz`](packages/forestriesz) | random-forest learner (Chernozhukov et al. ICML 2022; `fit_rows`) |
| [`packages/riesznet`](packages/riesznet) | neural-network learner (Chernozhukov et al. 2021; `fit_rows`) |
| [`packages/riesztree`](packages/riesztree) | single-tree learner (Schuler 2026; `fit_augmented`) |

Each member has the structure `packages/<pkg>/python/<pkg>/` (Python source + tests) and `packages/<pkg>/r/<pkg>/` (R wrapper). The five non-meta members declare `dependencies = ["rieszreg"]` in their `pyproject.toml`; uv resolves that to the local workspace member, not PyPI.

## Layout

```
RieszReg/
├── pyproject.toml            workspace root
├── uv.lock
├── DESIGN.md                 architectural contract
├── docs/                     unified Quarto site (covers all six)
├── reference/                arXiv PDFs
├── .github/workflows/        test.yml (matrix all six × 2 OS × 2 Py), docs.yml, release.yml
└── packages/
    └── <pkg>/
        ├── pyproject.toml
        ├── python/<pkg>/      Python source
        ├── python/tests/
        └── r/<pkg>/           R wrapper (DESCRIPTION + R/ + tests/testthat/)
```

## Run tests

```sh
uv sync --all-packages --all-extras
uv run pytest packages/<pkg>/python/tests -q          # one package
for p in rieszreg rieszboost krrr forestriesz riesznet riesztree; do
  uv run pytest packages/$p/python/tests -q
done                                                   # all packages
```

R parity tests:

```sh
uv run Rscript -e '
  pkgload::load_all("packages/rieszreg/r/rieszreg")
  for (p in c("rieszboost","krrr","forestriesz","riesznet","riesztree")) {
    pkgload::load_all(file.path("packages", p, "r", p))
    testthat::test_dir(file.path("packages", p, "r", p, "tests", "testthat"))
  }
'
```

See [`rieszreg-dev-environment`](.claude/skills/rieszreg-dev-environment/SKILL.md) for worktree usage and Quarto rendering.

## Adding a new estimand, loss, or learner

- **Estimand or loss** → `packages/rieszreg/python/rieszreg/{estimands,losses}/`. Steps in [`DESIGN.md`](DESIGN.md) §A.3.
- **Learner (new backend)** → new `packages/<pkg>/`, follow the contract in [`DESIGN.md`](DESIGN.md) Part B. The [`rieszreg-backend-contract`](.claude/skills/rieszreg-backend-contract/SKILL.md) skill is the triggered short form of that contract.

## Releases

Per-package PyPI releases via tag prefix: `<pkg>-vX.Y.Z` (e.g. `rieszboost-v0.1.0`). The `.github/workflows/release.yml` workflow uses Trusted Publisher OIDC — set up the trusted-publisher relationship on PyPI per package before the first release.
