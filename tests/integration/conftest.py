"""Integration-test conftest.

The Earth-boundary datasets (coastlines / tectonic plates / time zones / land /
lakes / rivers) are **generate-on-demand** — produced locally by
``skyplothelper.prepare_earth_data()`` and NOT shipped in the repo or wheel (see
the project ``.gitignore``). So on a fresh checkout / CI without the data, any
test that draws a geographic overlay would fail with a ``FileNotFoundError``.

This hook turns that specific failure into a **skip** (with a clear reason), so
the geo-overlay tests run fully when the data is present locally and are skipped
— not failed — when it isn't. Only ``FileNotFoundError``s that point at
``prepare_earth_data`` are caught; any other error propagates normally.
"""

import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except FileNotFoundError as exc:
        if "prepare_earth_data" in str(exc):
            pytest.skip(
                "Earth-boundary data not present — run "
                "skyplothelper.prepare_earth_data() to generate it "
                f"({str(exc).splitlines()[0][:80]})")
        raise
