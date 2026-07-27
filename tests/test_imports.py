"""Import sanity check.

Confirms the package imports cleanly with no optional dependencies and
that the version string is wired up through `_version.py`.
"""


def test_package_imports():
    import skyplothelper  # noqa: F401


def test_version_set():
    import re

    import skyplothelper
    from skyplothelper._version import __version__ as source_version

    # The exported version matches the single source of truth (skyplothelper/
    # _version.py) and looks like a release version — checked without hard-
    # coding the number, so a version bump doesn't break this test.
    assert skyplothelper.__version__ == source_version
    assert re.match(r"^\d+\.\d+\.\d+", skyplothelper.__version__)
