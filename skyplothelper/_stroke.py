"""Shared helper for the ``stroke_color`` / ``stroke_lw`` kwarg pattern.

Many decorators and overlays in skyplothelper expose a uniform pair of
``stroke_color`` / ``stroke_lw`` kwargs to draw a stroke outline behind
text / lines / patches for legibility on textured backgrounds (the
classical cartographic "black text on a thin white stroke" look, or
the inverse for light cores on noisy bodies).

The stroke is rendered via :class:`matplotlib.patheffects.withStroke`:
the core artist is drawn on top of a wider stroke in ``stroke_color``,
so the visible stroke on each side is ``(stroke_lw - core_lw) / 2``.

Importing the helper from one place keeps the kwarg semantics and
disable-on-``None`` behavior consistent across decorators.
"""

from __future__ import annotations

from typing import Any

import matplotlib.patheffects as PathEffects


def _stroke_path_effects(stroke_color: Any,
                         stroke_lw: float | None) -> list[Any] | None:
    """Build a ``[withStroke]`` ``path_effects`` list, or ``None``.

    Returns ``None`` (signal: no stroke) when ``stroke_color`` is
    ``None`` or ``stroke_lw <= 0``. The returned list is suitable to
    pass as the ``path_effects=`` kwarg to ``ax.plot`` / ``ax.annotate``
    or to ``artist.set_path_effects(...)`` on existing artists.

    ``withStroke`` already renders the core artist on top of the stroke in
    one effect, so this is a single-element list — bit-identical to the
    hand-rolled ``[pe.withStroke(...)]`` idiom. (An appended
    ``PathEffects.Normal()`` would redraw the core a second time, darkening
    glyph edges at the antialiasing level; that idiom is only needed with the
    *plain* ``Stroke`` effect, which draws the outline alone.)
    """
    if stroke_color is None or stroke_lw is None or float(stroke_lw) <= 0:
        return None
    return [
        PathEffects.withStroke(linewidth=float(stroke_lw),
                                foreground=stroke_color),
    ]
