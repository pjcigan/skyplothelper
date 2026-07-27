"""Shared helpers for the merge-verification visual gallery scripts."""

import os
import sys


# ----------------------------------------------------------------------
# sys.path shielding: drop any directory that has a top-level
# ``skyplothelper.py`` single-file module but no companion
# ``skyplothelper/`` package directory.
#
# Some setups (e.g. a legacy ``PATHmodules`` dir on ``PYTHONPATH``)
# ship a single-file ``skyplothelper.py`` that shadows the installed
# package. When that shadow wins, ``import skyplothelper.overlays``
# fails with the very confusing
#     ModuleNotFoundError: No module named 'skyplothelper.overlays';
#     'skyplothelper' is not a package
# This shim runs before any skyplothelper import (every render script
# imports ``_common`` first) and removes the offending entry.
# ----------------------------------------------------------------------
def _strip_shadowing_skyplothelper_paths() -> list[str]:
    """Return list of dropped path entries (for diagnostics)."""
    dropped = []
    keep = []
    for entry in sys.path:
        if not entry:
            keep.append(entry)
            continue
        try:
            shadow = os.path.join(entry, "skyplothelper.py")
            real_pkg = os.path.join(entry, "skyplothelper", "__init__.py")
            if os.path.isfile(shadow) and not os.path.isfile(real_pkg):
                dropped.append(entry)
                continue
        except OSError:
            pass
        keep.append(entry)
    sys.path[:] = keep
    return dropped


_dropped_paths = _strip_shadowing_skyplothelper_paths()
# Also evict any partial skyplothelper module that may already have
# been loaded from a shadow path before this shim ran.
if _dropped_paths:
    for _name in [k for k in list(sys.modules)
                  if k == "skyplothelper" or k.startswith("skyplothelper.")]:
        del sys.modules[_name]


import matplotlib  # noqa: E402  (must follow the path-shim above)

# In --save mode default to Agg; --show needs an interactive backend.
if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_mode():
    """Return 'show' if --show on argv, else 'save'."""
    return "show" if "--show" in sys.argv else "save"


def save_or_show(fig, name, mode=None, dpi=120):
    """Save the figure as ``<OUTPUT_DIR>/<name>.png`` or show it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        File stem (no extension). Use a sortable prefix like
        ``"globe_01_scatter"``.
    mode : {'save', 'show'} or None
        If None, falls back to ``get_mode()`` (i.e., respects --show).
    dpi : int
        Output resolution for save mode.
    """
    if mode is None:
        mode = get_mode()
    if mode == "show":
        plt.show()
    else:
        path = os.path.join(OUTPUT_DIR, f"{name}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  saved: {path}")
        plt.close(fig)


def banner(title):
    """Print a section banner so terminal output is scannable."""
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")
