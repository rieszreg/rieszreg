# This docs directory is deprecated

The rieszboost user guide moved to the unified RieszReg meta-project docs at
[`/docs/`](../../docs/) (top of the repo).

The boosting-specific page lives at [`/docs/backends/boosting.qmd`](../../docs/backends/boosting.qmd).
Concept pages (estimands, losses, sklearn integration, save/load, diagnostics, R interface) and the
quickstart are shared across all backends; per-backend tuning advice lives under `docs/backends/`.

The .qmd files in this directory are kept for git-history continuity. They are no longer rendered;
their content has been migrated. The pre-commit hook's "must update docs" rule now points at
`/docs/` instead of `rieszboost/docs/`.
