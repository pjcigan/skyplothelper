"""Shared fixtures for the assertion test suite."""

import matplotlib as mpl
import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Skip (not fail) any test that needs the generate-on-demand Earth data.

    The Earth-boundary datasets (coastlines / tectonic plates / time zones /
    land / lakes / rivers) are produced locally by
    ``skyplothelper.prepare_earth_data()`` and are NOT shipped in the repo or
    wheel (see the project ``.gitignore``). On a fresh checkout / CI without the
    data, a test that draws a geographic overlay raises a ``FileNotFoundError``
    pointing at ``prepare_earth_data``; this turns that specific failure into a
    skip so the geo tests run fully when the data is present and are skipped —
    not failed — when it isn't. This lives at the top level so it covers every
    test directory, not just ``tests/integration/``. Any other error propagates.
    """
    try:
        return (yield)
    except FileNotFoundError as exc:
        if "prepare_earth_data" in str(exc):
            pytest.skip(
                "Earth-boundary data not present — run "
                "skyplothelper.prepare_earth_data() to generate it "
                f"({str(exc).splitlines()[0][:80]})")
        raise


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
