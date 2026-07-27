"""Shared text-layout helpers for rotated, perpendicularly-offset labels.

These utilities resolve rotation and alignment for matplotlib ``Text``
artists that sit at a perpendicular offset from a line or curve and
rotate to follow a tangent direction. They were factored out of
:mod:`skyplothelper.overlays.ruler` (the canonical caller) so the
two-dimensional tick / label conventions can be applied uniformly by
:func:`~skyplothelper.ticks.add_curved_lon_ticks` and the
:class:`~skyplothelper.coord_overlay.CoordinateOverlay` axis-overlay machinery.

Functions
---------
:func:`_normalize_readable_angle`
    Fold an angle into ``(-90, 90]`` so the text never reads upside-down
    — the same convention matplotlib's ``AxisLabel`` uses.
:func:`_resolve_rotation_deg`
    Convert a rotation *mode* (``'auto'`` / ``'horizontal'`` /
    ``'perpendicular'`` / numeric) plus a local tangent angle to a
    concrete rotation in degrees.
:func:`_resolve_text_anchor`
    Pick ``(ha, va)`` so that, after rotation, the text's near edge
    sits at the anchor point — making the caller-supplied offset the
    actual visible padding, regardless of rotation or font size.

All three are private (single-leading-underscore) — they're stable
within the package but not part of the public API.
"""

from __future__ import annotations

import re
from typing import Any, cast

import numpy as np

from ._compat import coord_ticklabels

_VALID_ROTATIONS = ('auto', 'horizontal', 'perpendicular')


def _normalize_readable_angle(angle_deg: float) -> float:
    """Fold *angle_deg* into ``(-90, 90]`` so text reads right-side-up.

    matplotlib text rotates CCW from horizontal in display space.
    Angles in ``(90, 270]`` produce upside-down text (read right-to-left
    from below the line); folding them by 180° lands them in a
    right-side-up reading orientation. The remaining ``(270, 360)``
    range folds by 360° into ``(-90, 0)``.

    The convention matches matplotlib's ``AxisLabel`` behavior for
    tilted axis labels.
    """
    a = angle_deg % 360.0
    if 90.0 < a <= 270.0:
        a -= 180.0
    if a > 180.0:
        a -= 360.0
    return a


def _resolve_rotation_deg(mode: str | float,
                          tangent_angle_deg: float) -> float:
    """Resolve a rotation mode to a numeric angle in degrees.

    Modes:

    * ``'auto'`` — parallel to the local tangent of the line / curve.
    * ``'horizontal'`` — always 0° (matplotlib's default).
    * ``'perpendicular'`` — 90° rotated from the tangent.
    * a numeric value — literal rotation in degrees CCW from horizontal.

    The result is normalized for readability via
    :func:`_normalize_readable_angle`.
    """
    if isinstance(mode, (int, float)) and not isinstance(mode, bool):
        return _normalize_readable_angle(float(mode))
    if mode == 'horizontal':
        return 0.0
    if mode == 'auto':
        return _normalize_readable_angle(tangent_angle_deg)
    if mode == 'perpendicular':
        return _normalize_readable_angle(tangent_angle_deg + 90.0)
    raise ValueError(
        f"rotation mode must be one of {_VALID_ROTATIONS!r} or a "
        f"numeric angle (degrees), got {mode!r}")


def _resolve_text_anchor(rotation_deg: float, side_sign: int,
                         perp_x: float, perp_y: float) -> tuple[str, str]:
    """Pick ``(ha, va)`` so a text's near edge lands at the anchor.

    Without this, ``ha='center', va='center'`` parks the bbox center
    on the anchor — so the visible gap between an offset anchor and
    the nearest text edge depends on font size *and* rotation. We
    instead anchor the *near edge* (the bbox edge that, after rotation,
    faces the line), so the caller-supplied offset is the actual
    visible padding regardless of how the text is rotated.

    Implementation: express the outward perpendicular direction
    (``side_sign × perp``) in the text's local (unrotated) frame by
    applying the inverse rotation. Whichever local axis dominates
    picks ``ha`` or ``va``; the sign picks left/right or top/bottom.

    Parameters
    ----------
    rotation_deg : float
        The resolved rotation that will be applied to the text, in
        degrees CCW from horizontal.
    side_sign : int
        ``+1`` if the label sits on the +perpendicular side of the
        line (the side ``(perp_x, perp_y)`` points to), ``-1`` for the
        opposite side. Callers without a flip-side concept can just
        pass ``+1`` and supply the actual outward perpendicular.
    perp_x, perp_y : float
        Perpendicular direction in *display* coords (typically the
        tangent rotated by 90° CCW). The function uses the signed
        outward = ``side_sign * (perp_x, perp_y)`` to choose the
        anchor.

    Returns
    -------
    ha, va : str
        ``ha`` is one of ``'left'``, ``'center'``, ``'right'``;
        ``va`` is one of ``'top'``, ``'center'``, ``'bottom'``.

    Examples
    --------
    * Horizontal label above a horizontal line (rotation=0,
      perp=+y) → ``('center', 'bottom')``: text grows upward.
    * Perpendicular label above a horizontal line (rotation=90,
      perp=+y) → ``('left', 'center')``: text rotates CCW, then
      grows upward from the anchored left edge.
    * Auto-rotation on any line: outward maps to local +up
      regardless of tangent angle, so always
      ``('center', 'bottom')`` (+1 side) / ``('center', 'top')``
      (−1 side).
    """
    out_x = side_sign * perp_x
    out_y = side_sign * perp_y
    rad = np.radians(rotation_deg)
    c = np.cos(rad)
    s = np.sin(rad)
    # Outward direction expressed in the text's local (unrotated)
    # frame: local = R(−θ) · display, where R(θ) is CCW rotation.
    u = c * out_x + s * out_y
    v = -s * out_x + c * out_y
    if abs(v) >= abs(u):
        return ('center', 'bottom' if v > 0 else 'top')
    return ('left' if u > 0 else 'right', 'center')


# ---------------------------------------------------------------------------
# Auto-scale tick-label fontsize
# ---------------------------------------------------------------------------
#
# Heuristic helpers used by make_wcs_frame / make_globe_frame to size tick
# labels for the available axes width. Goal: at typical figure sizes the
# rcParams default is preserved; at small / multi-panel axes the fontsize
# shrinks just enough to avoid crowding. The two helpers below split
# responsibility: ``_auto_label_fontsize`` does the geometric math (given
# the label width hints), and ``_n_chars_for_wcs_coord`` introspects an
# astropy WCSAxes coord to feed a sensible width hint into the math.

# Average proportional-font character width as a fraction of the fontsize
# in points. Roughly the median for ``DejaVu Sans`` / ``Arial`` digits
# and the common WCS-label glyphs (h, m, s, °, ', ").
_CHAR_WIDTH_FACTOR = 0.55

# Multiplicative gap between adjacent labels — used to ensure the
# chosen fontsize leaves visible whitespace, not just touching glyphs.
_LABEL_SAFETY_FACTOR = 1.3

# Fallback character-count when an astropy coord can't be introspected
# (or for non-WCS axes). Sized to the widest common label class —
# sexagesimal HMS / DMS — so the heuristic stays conservative when
# uncertain (slightly too small beats overlapping).
_DEFAULT_N_CHARS = 10


def _auto_label_fontsize(ax: Any, n_chars_hint: int = _DEFAULT_N_CHARS,
                         n_ticks_hint: int = 6, floor: float = 6.0,
                         ceiling: float | None = None,
                         axis: str = 'x') -> float:
    """Pick a tick-label fontsize sized to fit the available axes width.

    Heuristic::

        fontsize_pt = axes_width_pt
                    / (n_ticks * n_chars * _CHAR_WIDTH_FACTOR
                       * _LABEL_SAFETY_FACTOR)

    clipped to ``[floor, ceiling]``. The intent is to *shrink* labels
    on tight panels (small figsize, dense subplot grids) without ever
    growing past the user's rcParams default — so plots at typical
    sizes are visually unchanged.

    Parameters
    ----------
    ax : matplotlib Axes
        Source of the axes geometry. The axes' rendered window extent
        (``ax.get_window_extent()``) is the binding measurement.
    n_chars_hint : int
        Expected average characters per tick label. Default
        :data:`_DEFAULT_N_CHARS` (10 — HMS / DMS sexagesimal sized).
        Use :func:`_n_chars_for_wcs_coord` to pick a sharper value
        from an astropy WCSAxes coord.
    n_ticks_hint : int
        Expected number of ticks across the axis. Default 6 — matches
        the ``make_wcs_frame`` ``lon_spacing='auto'`` default of ~8
        ticks across the visible range, allowing for some edge
        clipping on non-rectangular projections.
    floor : float
        Minimum fontsize in points. Default 6.0 (legible at typical
        screen / print resolution; smaller and the glyph detail
        breaks down).
    ceiling : float or None
        Maximum fontsize in points. ``None`` (default) → the
        matplotlib rcParams default for the matching axis
        (``xtick.labelsize`` or ``ytick.labelsize``). The auto-scale
        is purely a *shrink* — it never grows past this value, so
        plots at typical figure sizes are visually unchanged.
    axis : {'x', 'y'}
        Which axis dimension to size from. Default ``'x'`` since
        longitude labels (HMS / DMS) are typically the binding
        constraint.

    Returns
    -------
    float
        The chosen fontsize in points.
    """
    from matplotlib import rcParams

    if ceiling is None:
        key = 'xtick.labelsize' if axis == 'x' else 'ytick.labelsize'
        # mpl 3.11 types rcParams.get with Literal keys; cast to Any so a plain
        # str key resolves on both the baseline and latest stubs.
        raw = cast("Any", rcParams).get(key, 10.0)
        # rcParams may contain string sizes ('medium', 'small'); the
        # auto-scale needs a number, so fall back to 10pt when we
        # can't coerce.
        try:
            ceiling = float(raw)
        except (TypeError, ValueError):
            ceiling = 10.0

    try:
        bbox = ax.get_window_extent()
        ax_extent_px = bbox.width if axis == 'x' else bbox.height
        ax_extent_pt = ax_extent_px * 72.0 / ax.figure.dpi
    except (AttributeError, ValueError):
        return float(ceiling)

    if ax_extent_pt <= 0:
        return float(ceiling)

    denom = max(
        n_ticks_hint * n_chars_hint
        * _CHAR_WIDTH_FACTOR * _LABEL_SAFETY_FACTOR,
        1.0,
    )
    max_fit = ax_extent_pt / denom
    return float(max(floor, min(ceiling, max_fit)))


# Matches a mathtext block — ``$...$`` — used by astropy to render
# unit superscripts (^h, ^m, ^s, ^circ, etc.). When estimating the
# rendered width of a label string we collapse each block to a single
# character (the typical block holds one suffix glyph).
_MATHTEXT_BLOCK_RE = re.compile(r'\$[^$]*\$')


def _approx_rendered_chars(s: str) -> int:
    """Estimate the rendered character count of a tick label string.

    Strips astropy's mathtext markup (``$\\mathregular{^h}$`` and
    friends) by collapsing each ``$...$`` block to one character — the
    typical block holds a single unit-superscript glyph (h / m / s /
    °), so 1 is a fair proxy for its rendered width. Bias is slightly
    toward overcounting (superscripts render narrower than full-height
    glyphs), which biases the auto-fontsize toward the conservative
    side.
    """
    return len(_MATHTEXT_BLOCK_RE.sub('°', s))


def _n_chars_for_wcs_coord(coord: Any,
                           fallback: int = _DEFAULT_N_CHARS) -> int:
    """Best-effort character-count estimate for an astropy WCSAxes coord.

    Two-stage lookup:

    1. **Post-draw introspection** — when the figure has been drawn
       and ``coord.ticklabels.text`` is populated, read the actual
       rendered label strings (stripping LaTeX / mathtext markup via
       :func:`_approx_rendered_chars`) and return the maximum length.
       This is the accurate path and is what
       :func:`_apply_auto_label_fontsize_to_wcs` uses since it triggers
       after the make_wcs_frame / make_globe_frame ``canvas.draw()``.

       In astropy ≥ 5, ``coord.ticklabels.text`` is a
       ``defaultdict[str, list[str]]`` keyed by axis location
       (``'c'`` / ``'h'`` / ``'b'`` / ``'l'`` / ``'r'``). We take the
       union across all locations, filter out 1-character degenerate
       entries (sometimes appear as ``'$'`` placeholders), and return
       the widest rendered label.

    2. **Pre-draw fallback** — when no rendered labels exist yet,
       bucket by ``coord.get_format_unit()`` + the decimal-vs-sexagesimal
       formatter setting:

       * sexagesimal (hourangle / deg, non-decimal) → 10 chars
       * decimal                                    → 7 chars
       * unknown                                    → *fallback*

    The decimal-vs-sexagesimal lookup reads
    ``coord._formatter_locator.decimal`` — a private attribute, but
    its name and semantics have been stable across recent astropy
    releases; the helper degrades gracefully to the conservative
    sexagesimal bucket if astropy moves the API.
    """
    # Stage 1 — post-draw introspection of actual rendered labels.
    try:
        tick_obj = coord_ticklabels(coord)
        text_attr = getattr(tick_obj, 'text', None)
        if isinstance(text_attr, dict):
            texts = [s for v in text_attr.values() for s in v if s]
        elif isinstance(text_attr, (list, tuple)):
            texts = [s for s in text_attr if s]
        else:
            texts = []
        widths = [_approx_rendered_chars(s) for s in texts]
        # Filter out single-char artifacts (astropy sometimes emits
        # ``'$'`` placeholders in the per-location list).
        widths = [w for w in widths if w >= 2]
        if widths:
            return max(widths)
    except (AttributeError, IndexError, TypeError):
        pass

    # Stage 2 — pre-draw fallback via format-unit detection.
    try:
        unit = coord.get_format_unit()
        unit_str = (unit.to_string()
                    if hasattr(unit, 'to_string') else str(unit)).lower()
    except (AttributeError, ValueError):
        return fallback

    is_decimal = False
    try:
        is_decimal = bool(coord._formatter_locator.decimal)
    except AttributeError:
        pass

    if is_decimal:
        return 7
    if 'hour' in unit_str or 'deg' in unit_str:
        return 10
    return fallback


def _apply_auto_label_fontsize_to_wcs(ax: Any, *, n_ticks_hint: int = 6,
                                       floor: float = 6.0,
                                       ceiling: float | None = None
                                       ) -> float | None:
    """Compute + apply auto-fontsize to both coords of a WCSAxes.

    Wraps :func:`_auto_label_fontsize` with per-coord character-count
    introspection from :func:`_n_chars_for_wcs_coord`, applies the
    chosen fontsize via :meth:`coord.set_ticklabel`, caches the value
    on the axes as ``ax._sph_auto_label_fontsize`` (so downstream
    helpers like :func:`~skyplothelper.coord_overlay.add_overlay_ticks`
    can pick it up without an explicit kwarg), and returns it so
    callers (``make_wcs_frame`` / ``make_globe_frame``) can also
    forward it into overlay-mode ``label_kwargs`` directly.

    Sizing uses the *wider* of the two coords' label widths (typically
    longitude, since HMS / DMS is wider than decimal-ish latitude
    labels). Calling this on a non-WCSAxes is a no-op that returns
    ``None``.
    """
    if not hasattr(ax, 'coords'):
        return None
    try:
        n_chars_lon = _n_chars_for_wcs_coord(ax.coords[0])
        n_chars_lat = _n_chars_for_wcs_coord(ax.coords[1])
    except (AttributeError, IndexError):
        return None
    n_chars = max(n_chars_lon, n_chars_lat)
    fontsize = _auto_label_fontsize(
        ax, n_chars_hint=n_chars, n_ticks_hint=n_ticks_hint,
        floor=floor, ceiling=ceiling, axis='x')
    try:
        ax.coords[0].set_ticklabel(fontsize=fontsize)
        ax.coords[1].set_ticklabel(fontsize=fontsize)
    except (AttributeError, IndexError):
        pass
    # Cache for downstream pickup (e.g. by add_overlay_ticks when the
    # gallery calls it directly, bypassing _apply_tick_style's
    # label_kwargs forwarding).
    ax._sph_auto_label_fontsize = fontsize
    return fontsize


def _n_chars_for_mpl_axis(ax: Any, axis: str = 'x',
                          fallback: int = _DEFAULT_N_CHARS) -> int:
    """Character-count estimate from a plain matplotlib axis's
    rendered tick labels.

    Reads the visible (non-empty) tick label texts via
    :meth:`Axes.get_xticklabels` / :meth:`Axes.get_yticklabels`, strips
    mathtext markup via :func:`_approx_rendered_chars`, and returns the
    maximum rendered width. Falls back to *fallback* when no labels
    exist (e.g. pre-draw axes or all-empty labels — typical when the
    user hasn't set ticks yet).
    """
    try:
        labels = (ax.get_xticklabels() if axis == 'x'
                  else ax.get_yticklabels())
    except (AttributeError, ValueError):
        return fallback
    widths = []
    for lbl in labels:
        try:
            text = lbl.get_text()
        except AttributeError:
            continue
        if text and text.strip():
            widths.append(_approx_rendered_chars(text))
    return max(widths) if widths else fallback


def _apply_auto_label_fontsize_to_mpl(ax: Any, *, axis: str = 'both',
                                       n_ticks_hint: int = 6,
                                       floor: float = 6.0,
                                       ceiling: float | None = None
                                       ) -> float | None:
    """Compute + apply auto-fontsize to a plain matplotlib axes.

    Same heuristic as the WCS path, but introspects via
    :func:`_n_chars_for_mpl_axis` and applies through
    :meth:`Axes.tick_params` (the matplotlib-canonical way to set tick
    label size). When ``axis='both'`` the wider of the two axes' label
    widths drives a single fontsize applied to both — matches the WCS
    behavior so the two paths feel symmetric.

    Returns the chosen fontsize (or ``None`` if the call can't
    proceed, e.g. *ax* isn't a matplotlib Axes).
    """
    if not hasattr(ax, 'tick_params'):
        return None
    if axis == 'both':
        n_chars = max(
            _n_chars_for_mpl_axis(ax, 'x'),
            _n_chars_for_mpl_axis(ax, 'y'),
        )
        size_axis = 'x'   # use the wider dimension as the constraint
    else:
        n_chars = _n_chars_for_mpl_axis(ax, axis)
        size_axis = axis
    fontsize = _auto_label_fontsize(
        ax, n_chars_hint=n_chars, n_ticks_hint=n_ticks_hint,
        floor=floor, ceiling=ceiling, axis=size_axis)
    try:
        ax.tick_params(axis=axis, labelsize=fontsize)
    except (AttributeError, ValueError):
        pass
    ax._sph_auto_label_fontsize = fontsize
    return fontsize


def _is_cartopy_axes(ax: Any) -> bool:
    """True if *ax* is a cartopy GeoAxes (or subclass).

    Cartopy is an optional dependency; this check uses string-based
    class-tree introspection so we don't import cartopy just to test
    membership. Returns ``False`` on anything that isn't a recognizable
    cartopy axes.
    """
    try:
        for cls in type(ax).__mro__:
            mod = getattr(cls, '__module__', '') or ''
            if mod.startswith('cartopy.'):
                return True
    except (AttributeError, TypeError):
        pass
    return False


def _gridliners_on_cartopy_axes(ax: Any) -> list[Any]:
    """List of cartopy Gridliner artists on *ax*.

    Iterates the axes' children and filters by class (rather than
    importing cartopy at module load) — keeps the optional-dependency
    contract intact. Returns an empty list when none exist.
    """
    out: list[Any] = []
    try:
        children = ax.get_children()
    except AttributeError:
        return out
    for child in children:
        cls = type(child)
        if (cls.__name__ == 'Gridliner'
                and getattr(cls, '__module__', '').startswith('cartopy.')):
            out.append(child)
    return out


def _n_chars_for_cartopy_gridliner(gridliner: Any,
                                   fallback: int = _DEFAULT_N_CHARS) -> int:
    """Character-count estimate from a cartopy Gridliner's rendered labels.

    Reads the union of ``xlabel_artists`` and ``ylabel_artists`` (Text
    artists populated post-draw), strips mathtext via
    :func:`_approx_rendered_chars`, and returns the widest non-trivial
    label. Falls back to *fallback* when no labels have rendered yet.
    """
    texts = []
    for attr in ('xlabel_artists', 'ylabel_artists'):
        try:
            artists = getattr(gridliner, attr, None) or []
        except AttributeError:
            artists = []
        for art in artists:
            try:
                t = art.get_text()
            except AttributeError:
                continue
            if t and t.strip():
                texts.append(t)
    widths = [_approx_rendered_chars(s) for s in texts if s]
    widths = [w for w in widths if w >= 2]
    return max(widths) if widths else fallback


def _apply_auto_label_fontsize_to_cartopy(ax: Any, *, axis: str = 'both',
                                           n_ticks_hint: int = 6,
                                           floor: float = 6.0,
                                           ceiling: float | None = None
                                           ) -> float | None:
    """Compute + apply auto-fontsize to all Gridliner artists on a
    cartopy GeoAxes.

    Cartopy gridliner labels live in a separate machinery from the
    standard tick labels: they're managed by
    :class:`~cartopy.mpl.gridliner.Gridliner` objects that re-create
    their Text artists on each draw based on ``xlabel_style`` /
    ``ylabel_style`` dicts. This helper:

    1. Finds every Gridliner attached to *ax*.
    2. Picks one rendered label-width hint via
       :func:`_n_chars_for_cartopy_gridliner`.
    3. Computes the fontsize via :func:`_auto_label_fontsize`.
    4. Updates each gridliner's label-style dicts so subsequent draws
       pick up the new size *and* directly resets each currently-rendered
       label artist so the change is visible without requiring a redraw.

    Returns the chosen fontsize, or ``None`` when no gridliners are
    present (e.g. ``make_cartopy_frame(grid=False)``) — nothing to
    size, no warning needed.
    """
    gridliners = _gridliners_on_cartopy_axes(ax)
    if not gridliners:
        return None

    # Width hint: use the widest rendered label across all gridliners
    # so we never size for the narrower one and leave the wider crowded.
    n_chars = max(
        (_n_chars_for_cartopy_gridliner(gl) for gl in gridliners),
        default=_DEFAULT_N_CHARS,
    )
    fontsize = _auto_label_fontsize(
        ax, n_chars_hint=n_chars, n_ticks_hint=n_ticks_hint,
        floor=floor, ceiling=ceiling, axis='x')

    for gl in gridliners:
        # Update the style dicts so the *next* draw picks up the size.
        if axis in ('x', 'both'):
            x_style = dict(getattr(gl, 'xlabel_style', None) or {})
            x_style['size'] = fontsize
            try:
                gl.xlabel_style = x_style
            except AttributeError:
                pass
        if axis in ('y', 'both'):
            y_style = dict(getattr(gl, 'ylabel_style', None) or {})
            y_style['size'] = fontsize
            try:
                gl.ylabel_style = y_style
            except AttributeError:
                pass
        # Directly update the currently-rendered label artists so the
        # change shows up without waiting for a redraw.
        for attr in ('xlabel_artists', 'ylabel_artists'):
            target_axis = 'x' if attr.startswith('x') else 'y'
            if axis not in ('both', target_axis):
                continue
            for art in getattr(gl, attr, None) or []:
                try:
                    art.set_fontsize(fontsize)
                except AttributeError:
                    pass

    ax._sph_auto_label_fontsize = fontsize
    return fontsize
