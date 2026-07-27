"""Helper for declaring pytest-mpl tests from a PANELS registry.

Keeps each ``tests/visual_baselines/test_*.py`` file to a couple of
lines: import the render module's PANELS dict, hand it to
``make_panel_tests(globals(), PANELS)``, done.

Style note
----------
pytest-mpl defaults to matplotlib's ``classic`` style when running
tests, which remaps the modern color cycle so e.g. ``"C2"`` resolves
to ``"r"`` (red) instead of ``#2ca02c`` (green). Panels written
against the modern cycle would baseline with the wrong colors —
indistinguishable from a real regression on visual review. Pinning
``style="default"`` on every generated test keeps baselines visually
matched to the gallery script output (which itself uses the default
matplotlib style).
"""

import pytest


def make_panel_tests(target_globals, panels, tolerance=10,
                     style="default"):
    """Inject a ``test_<panel>`` function into *target_globals* for each
    entry in *panels*. Each generated function is decorated with
    ``@pytest.mark.mpl_image_compare(filename="<panel>.png",
    tolerance=tolerance, style=style)`` and returns the figure produced
    by the panel's builder.

    Parameters
    ----------
    target_globals : dict
        The caller's ``globals()``. The generated test functions are
        inserted here so pytest discovers them as if they were defined
        at module top level.
    panels : dict[str, callable]
        Panel name → no-arg builder returning a Figure. Typically
        imported from a ``render_*.py`` script's ``PANELS`` registry.
    tolerance : float, optional
        pytest-mpl RMS tolerance (default 10).
    style : str, optional
        Matplotlib style applied during baseline rendering and
        comparison. Default ``"default"`` matches the gallery
        script's appearance (modern color cycle).
    """
    for panel_name, builder in panels.items():
        # Bind `builder` via a default arg so each generated function
        # captures the correct callable (otherwise all functions would
        # share the last loop iteration's binding).
        @pytest.mark.mpl_image_compare(filename=f"{panel_name}.png",
                                       tolerance=tolerance,
                                       style=style)
        def test_fn(_b=builder):
            return _b()
        test_fn.__name__ = f"test_{panel_name}"
        test_fn.__qualname__ = test_fn.__name__
        target_globals[test_fn.__name__] = test_fn
