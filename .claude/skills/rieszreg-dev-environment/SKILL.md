---
name: rieszreg-dev-environment
description: Operational rules for the rieszreg uv-workspace monorepo. Use when running `uv sync` or `uv run pytest`, creating or working inside a `git worktree`, rendering the Quarto docs site (`docs/`), or doing anything that touches the workspace `.venv`. Triggers any time you cd into a fresh checkout of `rieszreg/rieszreg` or one of its worktrees.
---

# rieszreg dev environment

The repo is a single `uv` workspace at `github.com/rieszreg/rieszreg`. Six packages live under `packages/<pkg>/`. One `.venv/` at the workspace root, with all six packages editable-installed via uv. Worktrees get their own `.venv/` (one `uv sync` per worktree).

## First-time setup

```sh
git clone https://github.com/rieszreg/rieszreg.git
cd rieszreg
uv sync --all-packages --all-extras    # creates .venv with all six pkgs editable
```

`uv` (Astral's package manager) must be installed: `brew install uv` on macOS, or `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## Running things

Two equivalent forms:

```sh
uv run pytest packages/rieszreg/python/tests -q     # uv handles activation
.venv/bin/python -m pytest packages/rieszreg/python/tests -q     # direct
```

Use `uv run <cmd>` when in doubt — it ensures the workspace `.venv` is current before running.

For an interactive REPL on the workspace:

```sh
uv run python
```

## Worktrees

`git worktree add ../wt-feature feature-branch` creates a worktree. Each worktree has its own `.venv/` once you run `uv sync` inside it. From the worktree:

```sh
cd ../wt-feature
uv sync --all-packages --all-extras
.venv/bin/python -c "import rieszreg; print(rieszreg.__file__)"
# expected: <wt-feature>/packages/rieszreg/python/rieszreg/__init__.py
```

The path printed must be **inside the worktree**. If it's not — if it points to the main checkout or any other location — there is a leftover `rieszreg/` directory at the workspace root namespace-shadowing the editable install. Remove it: `rm -rf rieszreg` (the directory at the root, *not* `packages/rieszreg/`).

## Adding extras you need ad-hoc

The workspace pyproject has a `[dependency-groups] docs = [...]` group with matplotlib, pandas, causaldata, xgboost, jax, torch. To get those in your venv:

```sh
uv sync --all-packages --all-extras --group docs
```

Other groups can be added to the root `pyproject.toml` — keep them at the workspace root, not in member pyprojects, when they're cross-cutting.

## Quarto docs

The unified docs site lives at `docs/` and Quarto-renders to `docs/_site/` (gitignored). The `_freeze/` cache **is** committed — that's what lets the gh-pages publish workflow skip re-executing chunks.

```sh
quarto render docs/                       # full site (10+ minutes)
quarto render docs/<page>.qmd             # single page
```

If a render seems stuck, check whether chunks actually executed:

```sh
ls -lt docs/_freeze/<page>/execute-results/   # most-recently-modified files
```

If the timestamps are old, the chunks haven't run and Quarto is using the cache. To force fresh execution: delete the relevant `_freeze/<page>/` and re-render.

To kill a stuck render and any orphan chunk processes:

```sh
pkill -f "quarto render"
ps -ef | grep -E '(quarto|knitr|R --slave|python.*chunk)' | grep -v grep
```

## R parity tests

R uses the workspace's Python venv via reticulate. Run from the workspace root:

```sh
uv run Rscript -e '
  pkgload::load_all("packages/rieszreg/r/rieszreg")
  for (p in c("rieszboost","krrr","forestriesz","riesznet","riesztree")) {
    pkgload::load_all(file.path("packages", p, "r", p))
    testthat::test_dir(file.path("packages", p, "r", p, "tests", "testthat"))
  }
'
```

R packages are *not* uv-managed — uv only handles Python. R deps come from CRAN at test time. The `rieszreg` R package isn't on CRAN; sibling R packages declare it in their `DESCRIPTION` Imports, which means `pak::pkg_deps` (used by `r-lib/actions/setup-r-dependencies`) can't resolve a sibling's deps directly. CI works around this by setting up R deps from rieszreg's DESCRIPTION (which only has CRAN deps) and relying on `pkgload::load_all` for the local rieszreg + sibling at test time.

## Sanity-check before claiming a local verification worked

- `uv run pytest <path> -q` collected and ran the expected test count (compare with CI numbers in the workflow logs).
- `quarto render`: did the chunks actually execute? `ls -lt docs/_freeze/<page>/` timestamps should be recent.
- R parity test: did `pkgload::load_all` succeed and was `testthat::test_dir` actually invoked? Look for `[ FAIL 0 | … ]` or `OK` in the output.

If the worktree environment can't run a verification, say so explicitly — don't pretend the fix is verified. Push to a feature branch and let CI verify.
