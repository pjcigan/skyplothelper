"""pytest-mpl configuration for the local visual regression suite.

These tests are excluded from the default ``pytest`` run (see the
``addopts = "--ignore=tests/visual_baselines"`` line in
``pyproject.toml``) because:

* Visual baselines are environment-sensitive (matplotlib version,
  available fonts, DPI, backend) and don't survive cross-platform CI
  cleanly.
* The 779 assertion-based tests in ``tests/`` provide the
  cross-platform regression coverage; this suite is a *local* dev
  tool for catching unintended visual changes during region-renderer
  refactors.

Baselines live in ``tests/visual_baselines/baseline/`` (gitignored).
See ``tests/visual_baselines/README.md`` for run / regen instructions.
"""

import os
import sys

import matplotlib

# Force the Agg backend so the suite is usable in headless environments.
matplotlib.use("Agg")

# Make the merge-verification render scripts importable. They live
# under ``tests/integration/visual/`` and use a
# sys.path-relative ``from _common import ...`` pattern; the suite
# imports the figure-builders from those modules.
_RENDER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "integration",
                 "visual"))
if _RENDER_DIR not in sys.path:
    sys.path.insert(0, _RENDER_DIR)

# Also expose this directory itself so test files can import the
# shared ``_helpers`` module.
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
