#!/usr/bin/env bash
# One-time activation of the repo's pre-commit hook for the current clone.
# Run from the repo root:  bash scripts/setup-hooks.sh

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

git config core.hooksPath .githooks

echo "Activated .githooks/ for this clone."
echo "Pre-commit checks now run on every git commit."
echo "Bypass for a single commit with: git commit --no-verify"
