"""Import sanity check.

Confirms the package imports cleanly with no optional dependencies and
that the version string is wired up through `_version.py`.
"""


def test_package_imports():
    import skyplothelper  # noqa: F401


def test_version_set():
    import skyplothelper

    assert skyplothelper.__version__ == "1.0.0"
