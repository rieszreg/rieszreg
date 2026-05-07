#!/usr/bin/env bash
# Doc-tone linter for the rieszreg monorepo.
#
# Greps a unified diff (staged in pre-commit, or against a base ref in CI) for
# two anti-patterns in user-facing prose (qmd / README.md):
#   1. Design-decision metacommentary ("intentionally no", "by design", ...).
#   2. AI-flavored hedge phrases ("the workhorse", "almost never", ...).
#
# Usage:
#   .githooks/lint-docs.sh --cached            # for pre-commit hook (staged diff)
#   .githooks/lint-docs.sh --base <ref>        # for CI (diff <ref>...HEAD)
#
# Exits 0 if no violations, 1 if violations found, 2 on usage error.

set -euo pipefail

mode=""
base=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cached)
      mode="cached"
      shift
      ;;
    --base)
      mode="base"
      base="${2:-}"
      shift 2
      ;;
    -h|--help)
      sed -n '2,15p' "$0" | sed 's/^# //; s/^#//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$mode" ]]; then
  echo "Usage: $0 --cached | --base <ref>" >&2
  exit 2
fi
if [[ "$mode" == "base" && -z "$base" ]]; then
  echo "--base requires a ref argument" >&2
  exit 2
fi

# Files we lint: any qmd page (root or per-package docs/) and any README.md.
doc_path_re='^(docs/.*\.qmd|packages/[^/]+/docs/.*\.qmd|README\.md|packages/[^/]+/README\.md)$'

if [[ "$mode" == "cached" ]]; then
  changed=$(git diff --cached --name-only)
else
  changed=$(git diff --name-only "$base"...HEAD)
fi

doc_files=$(echo "$changed" | grep -E "$doc_path_re" || true)

if [[ -z "$doc_files" ]]; then
  exit 0
fi

forbidden_regex='(intentionally (no|absent|out of scope|not)|by design|the design rule|design rule (is|says)|no bespoke|no separate [a-z_]+= ?argument|almost (every|never)|right (choice|for almost)|the workhorse|the natural (api|way|approach|choice)|don'\''t reinvent|rather than reinvent|reinvent it|we chose [a-z]+ over)'

if [[ "$mode" == "cached" ]]; then
  diff_cmd=(git diff --cached --unified=0 -- $doc_files)
else
  diff_cmd=(git diff --unified=0 "$base"...HEAD -- $doc_files)
fi

hits=$("${diff_cmd[@]}" | grep -E '^\+[^+]' | grep -iE "$forbidden_regex" || true)

if [[ -z "$hits" ]]; then
  exit 0
fi

echo "ERROR: doc tone violations in changed prose."
echo
echo "Lines being added contain phrases that read as design-decision"
echo "metacommentary or AI-flavored hedges. Rewrite to describe what"
echo "the API IS, not what it isn't or why we chose this design."
echo
echo "Offending lines:"
echo "$hits" | sed 's/^/  /'
echo
echo "Examples to avoid:"
echo "  BAD : 'There is intentionally no bespoke crossfit() function'"
echo "  GOOD: 'Cross-fit with sklearn.model_selection.cross_val_predict'"
echo
echo "  BAD : 'XGBoostBackend is right for almost every problem'"
echo "  GOOD: 'XGBoostBackend is the default. Use SklearnBackend for non-tree base learners.'"
echo
echo "Tone target: the ngboost user guide. Direct, instructive, short."
echo
if [[ "$mode" == "cached" ]]; then
  echo "Bypass for false positives with: git commit --no-verify"
fi
exit 1
