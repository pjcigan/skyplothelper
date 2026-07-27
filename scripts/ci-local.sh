#!/usr/bin/env bash
# Run the CI checks locally, the way GitHub Actions runs them, so you can
# verify a commit will pass before pushing.
#
# Two modes:
#   ./scripts/ci-local.sh          # fast: lint + type-check + tests in the
#                                  #   CURRENT environment (use your dev env)
#   ./scripts/ci-local.sh --clean  # faithful: build a throwaway conda env with
#                                  #   the pinned baseline stack and run there
#                                  #   (catches dependency-version drift CI sees
#                                  #   but your dev env doesn't)
#
# The fast mode mirrors CI's logic; the clean mode mirrors CI's environment.
# Run fast per-commit; run --clean before a push or release.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTEST_FLAGS=""
# The dev conda env carries two broken pytest plugins that abort collection;
# a clean CI env does not. Disable them only when they are present.
if python -c "import pytest_doctestplus" 2>/dev/null; then
    PYTEST_FLAGS="$PYTEST_FLAGS -p no:doctestplus"
fi
if python -c "import pytest_filter_subpackage" 2>/dev/null; then
    PYTEST_FLAGS="$PYTEST_FLAGS -p no:filter_subpackage"
fi
# Parallelize when pytest-xdist is available (CI does; matches `-n auto` there).
if python -c "import xdist" 2>/dev/null; then
    PYTEST_FLAGS="$PYTEST_FLAGS -n auto"
fi

run_checks() {
    echo "== ruff =="
    ruff check skyplothelper tests
    echo "== mypy =="
    mypy
    echo "== pytest =="
    # shellcheck disable=SC2086
    pytest tests/ -q $PYTEST_FLAGS
}

if [[ "${1:-}" == "--clean" ]]; then
    ENV=sph-cilocal
    echo ">> Building clean baseline env '$ENV' (conda)…"
    conda create -y -n "$ENV" python=3.12 pip >/dev/null
    conda run -n "$ENV" python -m pip install -e ".[dev,test]" \
        -c ci/constraints-baseline.txt -q
    echo ">> Running CI checks in '$ENV'…"
    conda run -n "$ENV" bash -c "cd '$PWD' && \
        ruff check skyplothelper tests && mypy && pytest tests/ -q -n auto"
    echo ">> Done. Remove the env with:  conda env remove -n $ENV"
else
    run_checks
fi
