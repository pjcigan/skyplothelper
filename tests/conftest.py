"""Shared fixtures for the assertion test suite."""

import matplotlib as mpl
import pytest


@pytest.fixture(autouse=True)
def _restore_rcparams():
    """Snapshot rcParams before each test and restore them after.

    Several tests mutate global matplotlib state (``set_base_style`` /
    ``set_theme`` / ``set_palette`` / ``style_wcs_axes`` and friends). Without
    isolation those mutations leak into later tests — e.g. a base preset that
    sets ``xtick.labelsize`` changing what a downstream auto-fontsize test sees
    as the default ceiling. Restoring per test keeps every test self-contained.
    """
    saved = mpl.rcParams.copy()
    try:
        yield
    finally:
        mpl.rcParams.update(saved)
