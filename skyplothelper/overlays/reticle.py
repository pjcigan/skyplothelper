"""Reticle / target-marker decorator (the :class:`Reticle` class).

A :class:`Reticle` draws a small, fixed-pixel-size mark anchored at a
sky or pixel position — the conventional crosshair used to highlight a
compact source on a finding chart, point out the location of a transient,
or indicate the pointing center of an observation. Four styles cover the
classical astronomy use cases:

* ``'plus'`` — vertical + horizontal arms with a central gap.
* ``'x'``    — same geometry rotated 45° (useful when a plus would
  visually collide with RA/Dec gridlines).
* ``'L'``    — two arms meeting at a right angle; only one quadrant of
  the plus, so the user can point the L's open side toward whatever
  quadrant of the source is least busy.
* ``'circle'`` — open circle of radius ``size`` (the finding-chart
  convention; ``circle_gap_deg=`` sculpts a broken-circle variant).

The mark sits at a specified ``(lon, lat)`` (or pixel ``(x, y)``) with
``transform=ax.get_transform('world')`` plumbing when the input is
world coords. Geometry is computed in **display points** so
``size`` / ``gap`` stay pixel-stable regardless of axes zoom — matching
the :class:`Ruler` / :func:`~skyplothelper.globe.add_compass_rose`
conventions.

Defaults are tuned for the canonical use case of highlighting a target
on a dark sky background: a ``color='white'`` body with a thin
``stroke_color='black'`` stroke via :class:`matplotlib.patheffects.withStroke`,
so the mark reads cleanly over both bright and dark patches. Every
appearance kwarg is overridable for light-background plots.

Coordinate input divergence from :class:`Ruler`
-----------------------------------------------
The canonical use case for a reticle is "I have a target at this RA/Dec,
highlight it"; the canonical use case for a Ruler is "measure between
these two pixels on the image I'm looking at". The two decorators therefore
default numeric tuples to *different* coordinate systems:

* :class:`Reticle` (and :func:`add_reticle`) — a numeric ``(lon, lat)``
  tuple is treated as **world** coordinates in degrees. Pass
  ``coord_type='pixel'`` (or use :meth:`Reticle.from_pixel`) for
  pixel-space input.
* :class:`Ruler` — numeric tuples are pixel coords; world input goes
  through :meth:`Ruler.from_world` or a :class:`SkyCoord`.

Use the explicit :meth:`Reticle.from_world` / :meth:`Reticle.from_pixel`
factories whenever the call site benefits from unambiguous coordinate
intent.

Label placement
---------------
When ``label=`` is set, the text is drawn at a compass offset from the
reticle center via :func:`matplotlib.axes.Axes.annotate` with
``textcoords='offset points'`` — so the gap between the reticle and the
label stays pixel-stable. ``label_side='auto'`` (the default) picks the
corner direction (NE/NW/SE/SW) pointing into the largest free region of
the axes at the time of attach, so the label naturally lands in the
emptiest quadrant. Pin via ``label_side='NE'`` (any of the eight compass
points are valid) if you need a stable placement across zooms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.lines import Line2D

from .._stroke import _stroke_path_effects

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

__all__ = ['Reticle', 'add_reticle']


_VALID_STYLES = ('plus', 'x', 'L', 'circle')
_VALID_COORD_TYPES = ('world', 'pixel')

# Symbolic aliases for the canonical style names — the glyph-shaped
# forms users naturally reach for ('+' for a plus, 'o' for a circle).
# Normalized once at construction time so internal dispatch only ever
# sees canonical strings.
_STYLE_ALIASES = {
    '+': 'plus',
    'o': 'circle',
}

# Compass-direction → (dx_sign, dy_sign, ha, va) for label offset placement.
# Signs are in display-axes orientation: +x is right, +y is up.
_LABEL_DIRECTIONS = {
    'N':  (0,  +1, 'center', 'bottom'),
    'NE': (+1, +1, 'left',   'bottom'),
    'E':  (+1, 0, 'left',   'center'),
    'SE': (+1, -1, 'left',   'top'),
    'S':  (0,  -1, 'center', 'top'),
    'SW': (-1, -1, 'right',  'top'),
    'W':  (-1, 0, 'right',  'center'),
    'NW': (-1, +1, 'right',  'bottom'),
}

_VALID_LABEL_SIDES = ('auto',) + tuple(_LABEL_DIRECTIONS.keys())


# ---------------------------------------------------------------------------
# Coordinate resolution
# ---------------------------------------------------------------------------

def _resolve_anchor(coord: SkyCoord | tuple[float, float], ax: Any, coord_type: str,
                    frame: Any) -> tuple[float, float, Any]:
    """Resolve *coord* to ``(anchor_x, anchor_y, anchor_transform)``.

    The returned ``(x, y)`` pair plus ``anchor_transform`` is what
    matplotlib needs to position the reticle in the axes:

    * ``coord_type='pixel'`` — numeric ``(x, y)`` tuple in pixel/data
      coords; anchor_transform = :attr:`ax.transData`.
    * ``coord_type='world'`` with :class:`SkyCoord` — projected to
      pixel coords via :meth:`ax.wcs.world_to_pixel` (frame-aware);
      anchor_transform = :attr:`ax.transData`.
    * ``coord_type='world'`` with numeric ``(lon, lat)`` tuple + ``frame=``
      — wrapped as a :class:`SkyCoord` and projected via
      :meth:`ax.wcs.world_to_pixel`; anchor_transform = :attr:`ax.transData`.
    * ``coord_type='world'`` with numeric ``(lon, lat)`` tuple, no frame
      — anchor_transform = :meth:`ax.get_transform('world') <astropy.visualization.wcsaxes.WCSAxes.get_transform>`,
      so astropy handles the projection natively (works for any WCS, not
      just celestial).
    """
    from astropy.coordinates import SkyCoord

    if coord_type not in _VALID_COORD_TYPES:
        raise ValueError(
            f"coord_type must be one of {_VALID_COORD_TYPES!r}, "
            f"got {coord_type!r}")

    if coord_type == 'pixel':
        try:
            x, y = coord
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"coord_type='pixel' expects an (x, y) numeric tuple, "
                f"got {coord!r}") from exc
        return float(x), float(y), ax.transData

    # coord_type == 'world'
    if isinstance(coord, SkyCoord):
        if not coord.isscalar:
            raise ValueError(
                f"SkyCoord must be a scalar position, got vector of "
                f"size {coord.size}")
        if not hasattr(ax, 'wcs'):
            raise ValueError(
                "SkyCoord input requires a WCSAxes (ax with .wcs) to "
                "project to pixel coords. For non-WCS axes, pass an "
                "(x, y) tuple with coord_type='pixel' instead.")
        x, y = ax.wcs.world_to_pixel(coord)
        return float(x), float(y), ax.transData

    # Numeric (lon, lat) tuple
    try:
        lon, lat = coord
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"coord_type='world' expects a SkyCoord or (lon, lat) "
            f"numeric tuple in degrees, got {coord!r}") from exc

    if frame is not None:
        if not hasattr(ax, 'wcs'):
            raise ValueError(
                "frame= conversion requires a WCSAxes (ax with .wcs).")
        sc = SkyCoord(float(lon), float(lat), unit='deg', frame=frame)
        x, y = ax.wcs.world_to_pixel(sc)
        return float(x), float(y), ax.transData

    # frame=None — use astropy's native get_transform('world') so the
    # input lon/lat is interpreted in the axes' WCS frame directly.
    if not hasattr(ax, 'wcs'):
        raise ValueError(
            "coord_type='world' requires a WCSAxes (ax with .wcs). "
            "For non-WCS axes, pass an (x, y) tuple with coord_type='pixel'.")
    return float(lon), float(lat), ax.get_transform('world')


# ---------------------------------------------------------------------------
# Style geometry — segments in display-point coords, relative to (0, 0) center
# ---------------------------------------------------------------------------

def _rotate(points: Sequence[tuple[float, float]],
            rotation_deg: float) -> list[tuple[float, float]]:
    """Rotate a list of ``(x, y)`` tuples by *rotation_deg* CCW.

    Returns a new list. Identity (rotation_deg ≈ 0) returns the input
    unchanged to keep the geometry tests literal at the canonical
    rotation.
    """
    if rotation_deg == 0:
        return list(points)
    theta = np.deg2rad(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    return [(c * x - s * y, s * x + c * y) for (x, y) in points]


def _reticle_segments(style: str, size: float, gap: float, rotation: float,
                      circle_npts: int, circle_gap_deg: float
                      ) -> list[list[tuple[float, float]]]:
    """Build the line segments for *style* in display-point coords.

    Returns a list of ``[(x0, y0), (x1, y1), ...]`` polylines centered on
    ``(0, 0)``. Each polyline becomes one :class:`Line2D` artist.

    Style decomposition (before rotation):

    * ``'plus'`` — four 1-segment arms along ±x / ±y axes, inner endpoint
      at ``gap``, outer endpoint at ``size``.
    * ``'x'`` — equivalent to ``'plus'`` rotated 45°.
    * ``'L'`` — two arms pointing into the lower-left (-x and -y) with
      the same gap/size; ``rotation=0`` leaves the upper-right quadrant
      open, ``rotation=90`` rotates the L counter-clockwise so the
      open quadrant is upper-left, and so on.
    * ``'circle'`` — closed circle of radius ``size`` sampled at
      ``circle_npts`` points. ``circle_gap_deg > 0`` removes an angular
      wedge centered on the +x axis (rotation=0 → gap on the right);
      ``gap`` is ignored (the canonical finding-chart marker does not
      have a central gap).
    """
    arms: list[list[tuple[float, float]]]
    if style == 'plus':
        arms = [
            [(0,    gap),   (0,    size)],   # N
            [(0,   -gap),   (0,   -size)],   # S
            [(gap,  0),     (size,  0)],     # E
            [(-gap, 0),     (-size, 0)],     # W
        ]
        return [_rotate(arm, rotation) for arm in arms]

    if style == 'x':
        # Same as plus rotated 45° (then apply caller's rotation on top).
        return _reticle_segments(
            'plus', size, gap, rotation + 45.0,
            circle_npts, circle_gap_deg)

    if style == 'L':
        arms = [
            [(-gap, 0),     (-size, 0)],     # W
            [(0,   -gap),   (0,    -size)],  # S
        ]
        return [_rotate(arm, rotation) for arm in arms]

    if style == 'circle':
        if circle_gap_deg > 0:
            # Broken circle: skip the angular wedge centered on +x.
            half = circle_gap_deg / 2.0
            start = np.deg2rad(half)
            end = np.deg2rad(360.0 - half)
            theta = np.linspace(start, end, max(circle_npts, 8))
        else:
            theta = np.linspace(0.0, 2 * np.pi, max(circle_npts, 8))
        ring = [(size * np.cos(t), size * np.sin(t)) for t in theta]
        return [_rotate(ring, rotation)]

    raise ValueError(
        f"style must be one of {_VALID_STYLES!r}, got {style!r}")


def _outer_extent(segments: Sequence[Sequence[tuple[float, float]]]) -> float:
    """Maximum distance from origin across all segment vertices.

    Used to size the label offset so it sits just past the reticle bbox
    regardless of style.
    """
    flat = [pt for seg in segments for pt in seg]
    if not flat:
        return 0.0
    return max(np.hypot(x, y) for (x, y) in flat)


# ---------------------------------------------------------------------------
# Label-side resolution
# ---------------------------------------------------------------------------

def _resolve_auto_label_side(ax: Any, anchor_x: float, anchor_y: float,
                             anchor_transform: Any) -> str:
    """Pick the corner direction pointing into the largest free quadrant.

    Computes the anchor's display-coord position via *anchor_transform*,
    then scores each of the four diagonal directions (NE/NW/SE/SW) by
    the minimum of the two perpendicular room dimensions toward that
    corner — i.e. the tightest constraint along the route the label
    would walk. The corner with the largest minimum wins.

    Falls back to ``'NE'`` if the anchor position can't be evaluated
    (degenerate axes bbox or similar).
    """
    try:
        cx, cy = anchor_transform.transform((anchor_x, anchor_y))
        bbox = ax.get_window_extent()
        room_E = bbox.x1 - cx
        room_W = cx - bbox.x0
        room_N = bbox.y1 - cy
        room_S = cy - bbox.y0
    except (AttributeError, ValueError):
        return 'NE'

    scores = {
        'NE': min(room_N, room_E),
        'NW': min(room_N, room_W),
        'SE': min(room_S, room_E),
        'SW': min(room_S, room_W),
    }
    return max(scores, key=lambda k: scores[k])


def _place_offset_label(ax: Any, anchor_x: float, anchor_y: float,
                        anchor_transform: Any, *, label: Any, side: str,
                        outer_extent: float, label_offset: float,
                        color: Any, fontsize: Any = None,
                        zorder: float | None = None,
                        path_effects: Any = None,
                        label_kwargs: dict[str, Any] | None = None) -> Any:
    """Annotate *label* at a compass *side* offset from an anchor point.

    Shared by :class:`Reticle` and the procedural instrument markers in
    :mod:`skyplothelper.overlays.instruments`: both anchor a pixel-stable
    sprite at ``(anchor_x, anchor_y)`` and want the text to sit just past
    the sprite's bounding box in a compass ``side`` direction, with the
    gap held constant in display points across zoom / pan.

    *outer_extent* is the sprite's radius from its center in display
    points (the reticle's outermost vertex, or a marker's half-size); the
    label is offset a further *label_offset* points beyond it. Returns the
    :class:`~matplotlib.text.Annotation` artist.
    """
    dx_sign, dy_sign, ha, va = _LABEL_DIRECTIONS[side]
    # Offset = sprite extent + extra gap; for diagonal corners we don't
    # divide by sqrt(2) since the label hugs the bounding box, not the
    # circumscribed circle. For pure-axis directions (N/S/E/W) one
    # component is zero; for corners both carry the full radial.
    radial = outer_extent + label_offset
    kw: dict[str, Any] = dict(
        xy=(anchor_x, anchor_y),
        xycoords=anchor_transform,
        xytext=(dx_sign * radial, dy_sign * radial),
        textcoords='offset points',
        ha=ha, va=va,
        color=color,
    )
    if zorder is not None:
        kw['zorder'] = zorder
    if fontsize is not None:
        kw['fontsize'] = fontsize
    # The legibility stroke reaches the label too (not just the sprite/arms),
    # so light text stays readable over busy sky; user label_kwargs still wins.
    if path_effects is not None:
        kw['path_effects'] = path_effects
    if label_kwargs:
        kw.update(label_kwargs)
    return ax.annotate(label, **kw)


# ---------------------------------------------------------------------------
# Reticle
# ---------------------------------------------------------------------------

class Reticle:
    """Target-highlight reticle with pixel-stable geometry.

    Parameters
    ----------
    coord : :class:`~astropy.coordinates.SkyCoord` or tuple
        Anchor position. Default interpretation depends on ``coord_type``:

        * ``coord_type='world'`` (default) — a :class:`SkyCoord` or
          ``(lon, lat)``-degree tuple. SkyCoords are projected via
          ``ax.wcs.world_to_pixel`` so cross-frame input (e.g. galactic
          coords on an ICRS WCS) is handled automatically.
        * ``coord_type='pixel'`` — ``(x, y)`` tuple in axes data
          (pixel) coordinates. The right choice for non-WCS axes or
          when the user already knows the pixel position.

        See :meth:`Reticle.from_world` / :meth:`Reticle.from_pixel` for
        explicit factory methods that document the coord type at the
        call site.

    Shape
    -----
    style : {'plus', 'x', 'L', 'circle'}
        Reticle shape. Symbolic aliases ``'+'`` (→ ``'plus'``) and
        ``'o'`` (→ ``'circle'``) are accepted as a convenience.

        * ``'plus'`` (default) — vertical + horizontal arms with a
          central gap. The classical target-acquisition reticle.
        * ``'x'`` — same geometry rotated 45°; useful when a plus
          would collide with RA/Dec gridlines.
        * ``'L'`` — two arms meeting at a right angle. ``rotation=0``
          opens the upper-right quadrant (arms point into the
          lower-left); positive ``rotation`` walks the L around
          counter-clockwise.
        * ``'circle'`` — open circle of radius ``size`` — the
          finding-chart convention. ``gap`` is ignored;
          ``circle_gap_deg`` cuts an angular wedge for a broken-circle
          variant.
    size : float
        Outer half-extent of the reticle in display points. For
        ``'plus'`` / ``'x'`` / ``'L'`` this is the arm length; for
        ``'circle'`` it is the circle radius. Default ``12``.
    gap : float
        Half-extent of the empty zone at the reticle center in display
        points. Inner endpoints of each arm sit at ``gap`` so the source
        itself isn't obscured. Default ``4``. Ignored for ``'circle'``.
    rotation : float
        Whole-reticle rotation in degrees counter-clockwise from the
        canonical orientation. Default ``0``. For ``'L'``, the four
        quadrant orientations are ``rotation=0`` (open upper-right),
        ``90`` (open upper-left), ``180`` (open lower-left), ``270``
        (open lower-right). For ``'circle'`` with
        ``circle_gap_deg > 0``, rotation moves the gap around the ring.

    Coordinate input
    ----------------
    coord_type : {'world', 'pixel'}
        How to interpret *coord*. Default ``'world'``. Note this defaults
        differently from :class:`Ruler` (which treats numeric tuples as
        pixel coords) — see the module docstring.
    frame : str or :class:`~astropy.coordinates.BaseCoordinateFrame`, optional
        Astropy frame name (e.g. ``'galactic'``, ``'fk5'``) for a numeric
        ``(lon, lat)`` tuple, when the input frame differs from the
        axes' WCS frame. When given, the tuple is wrapped as a
        :class:`SkyCoord` and projected via ``ax.wcs.world_to_pixel``.
        Ignored for :class:`SkyCoord` input (which carries its own
        frame) and for ``coord_type='pixel'``. Default ``None``
        (interpret the tuple in the axes-native frame via
        ``ax.get_transform('world')``).

    Appearance
    ----------
    color : matplotlib color
        Body color of the reticle arms. Default ``'white'`` — the
        dark-sky-readable default. Pass a dark color for light
        backgrounds.
    lw : float
        Line width of the reticle arms in display points. Default
        ``1.2``.
    stroke_color : matplotlib color or None
        Color of the optional stroke drawn under each arm (and under the
        label) via :class:`matplotlib.patheffects.withStroke`, for legibility
        across mixed bright/dark backgrounds. Default ``'black'``;
        pass ``None`` to disable the stroke entirely.
    stroke_lw : float
        Total stroke width in display points (the body draws on top of
        this stroke, so the visible stroke on each side is
        ``(stroke_lw - lw) / 2``). Default ``2.4`` — gives a 0.6 pt
        stroke on each side of the default ``lw=1.2`` body.
    circle_npts : int
        Sample count for the ``'circle'`` style polyline. Default ``64``
        — visually smooth across typical sizes.
    circle_gap_deg : float
        Angular wedge cut from the ``'circle'`` style, in degrees
        centered on the ``+x`` axis (before ``rotation`` is applied).
        Default ``0`` (full closed circle). Set e.g. ``30`` for the
        broken-circle "open ring" finding-chart variant.

    Label
    -----
    label : str, optional
        Text drawn next to the reticle. Default ``None`` (no label).
    label_side : str
        Side of the reticle to place the label on. ``'auto'`` (default)
        picks the corner direction (NE / NW / SE / SW) pointing into
        the largest free quadrant of the axes at the time of attach.
        Compass-point shorthands ``'N'``, ``'NE'``, ``'E'``, ``'SE'``,
        ``'S'``, ``'SW'``, ``'W'``, ``'NW'`` pin the side explicitly.
    label_offset : float
        Extra gap in display points between the reticle's outer extent
        and the label anchor. Default ``2``. The total label offset
        from the reticle center is ``size + label_offset`` (so labels
        always clear the reticle bbox regardless of the chosen side).
    label_color : matplotlib color, optional
        Label color. Defaults to the body ``color``.
    label_fontsize : float or str, optional
        Forwarded to :class:`matplotlib.text.Text`. Default inherits
        from rcParams.
    label_kwargs : dict, optional
        Extra kwargs forwarded to :meth:`matplotlib.axes.Axes.annotate`.
        Useful for setting weight, style, path-effects, etc.

    Other
    -----
    zorder : int
        Z-order applied to the arm artists and the label. Default ``5``.
    **line_kwargs
        Extra kwargs forwarded to each arm's :class:`~matplotlib.lines.Line2D`
        (e.g. ``alpha=``, ``ls=``).

    Returns
    -------
    Reticle
        The container — call :meth:`add_to` to attach it to an axes, or
        use :func:`add_reticle` to construct + attach in one call. The
        container also exposes :meth:`remove`, :meth:`set_color`,
        :meth:`set_label`, :meth:`set_size`, and an :attr:`arm_artists`
        / :attr:`label_artist` accessor pair for fine-grained edits.

    Notes
    -----
    Geometry is computed in display points and rebuilt on
    :meth:`add_to`. The reticle does **not** auto-reflow on pan/zoom by
    default — the arms stay pixel-stable through the
    ``get_transform('world')`` / ``transData`` anchor + display-points
    offsets, with no further callback needed for the shape itself. The
    one zoom-sensitive part is the ``label_side='auto'`` heuristic,
    which freezes at attach time; pass an explicit ``label_side=`` for
    stability under heavy zoom.

    Examples
    --------
    Highlight an ICRS target on a sky map (the default coord_type)::

        sph.add_reticle(ax, (266.4, -29.0), label='Sgr A*')

    Same target via :class:`SkyCoord`, frame-converted automatically::

        from astropy.coordinates import SkyCoord
        sgrA = SkyCoord(266.4, -29.0, unit='deg', frame='icrs')
        sph.add_reticle(ax, sgrA, label='Sgr A*', label_side='NE')

    L-shape pointing into the upper-left, with a custom stroke::

        sph.add_reticle(ax, target_coord, style='L', rotation=90,
                        color='yellow', stroke_color='black', stroke_lw=3.0)

    Broken-circle finding-chart marker::

        sph.add_reticle(ax, target_coord, style='circle', size=20,
                        circle_gap_deg=40, label='Candidate A')
    """

    def __init__(self, coord: SkyCoord | tuple[float, float], *,
                 style: str = 'plus', size: float = 12.0, gap: float = 4.0,
                 rotation: float = 0.0,
                 coord_type: str = 'world', frame: Any = None,
                 color: Any = 'white', lw: float = 1.2,
                 stroke_color: Any = 'black', stroke_lw: float = 2.4,
                 circle_npts: int = 64, circle_gap_deg: float = 0.0,
                 label: str | None = None, label_side: str = 'auto',
                 label_offset: float = 2.0,
                 label_color: Any = None,
                 label_fontsize: float | str | None = None,
                 label_kwargs: dict[str, Any] | None = None,
                 zorder: int = 5, **line_kwargs: Any) -> None:

        style = _STYLE_ALIASES.get(style, style)
        if style not in _VALID_STYLES:
            raise ValueError(
                f"style must be one of {_VALID_STYLES!r} (or aliases "
                f"{list(_STYLE_ALIASES)!r}), got {style!r}")
        if label_side not in _VALID_LABEL_SIDES:
            raise ValueError(
                f"label_side must be one of {_VALID_LABEL_SIDES!r}, "
                f"got {label_side!r}")

        self._coord = coord
        self._style = style
        self._size = float(size)
        self._gap = float(gap)
        self._rotation = float(rotation)
        self._coord_type = coord_type
        self._frame = frame

        self._color = color
        self._lw = float(lw)
        self._stroke_color = stroke_color
        self._stroke_lw = float(stroke_lw)
        self._circle_npts = int(circle_npts)
        self._circle_gap_deg = float(circle_gap_deg)

        self._label = label
        self._label_side = label_side
        self._label_offset = float(label_offset)
        self._label_color = label_color
        self._label_fontsize = label_fontsize
        self._label_kwargs = dict(label_kwargs) if label_kwargs else {}

        self._zorder = float(zorder)
        self._line_kwargs = dict(line_kwargs)

        # Attached state
        self._host_axes: Any = None
        self._anchor_box: Any = None
        self._arm_artists: list[Any] = []
        self._label_artist: Any = None
        self._resolved_label_side: str | None = None  # cached after 'auto'

    # ----- factories ---------------------------------------------------

    @classmethod
    def from_world(cls, coord: SkyCoord | tuple[float, float], *, frame: Any = None,
                   **kwargs: Any) -> Reticle:
        """Construct a Reticle from a world coordinate (SkyCoord or
        ``(lon, lat)``-degree tuple). Explicit form of
        ``Reticle(coord, coord_type='world', frame=frame, ...)``.
        """
        kwargs.pop('coord_type', None)
        return cls(coord, coord_type='world', frame=frame, **kwargs)

    @classmethod
    def from_pixel(cls, xy: Any, **kwargs: Any) -> Reticle:
        """Construct a Reticle from an ``(x, y)`` pixel/data tuple.
        Explicit form of ``Reticle(xy, coord_type='pixel', ...)``.
        """
        kwargs.pop('coord_type', None)
        kwargs.pop('frame', None)
        return cls(xy, coord_type='pixel', **kwargs)

    # ----- attach / detach ---------------------------------------------

    def add_to(self, ax: Any) -> Reticle:
        """Attach the reticle to *ax*. Returns ``self`` for chaining."""
        self._host_axes = ax
        self._build_artists(ax)
        return self

    def remove(self) -> None:
        """Remove all reticle artists from the host axes."""
        anchor_box = getattr(self, '_anchor_box', None)
        if (anchor_box is not None
                and getattr(anchor_box, 'axes', None) is not None):
            anchor_box.remove()
        self._anchor_box = None
        self._arm_artists = []
        if (self._label_artist is not None
                and getattr(self._label_artist, 'axes', None) is not None):
            self._label_artist.remove()
        self._label_artist = None
        self._host_axes = None
        self._resolved_label_side = None

    # ----- core build --------------------------------------------------

    def _build_artists(self, ax: Any) -> None:
        """Construct + attach the reticle arms and the optional label.

        Geometry follows the :func:`~skyplothelper.globe.add_compass_rose`
        pattern: a single :class:`~matplotlib.offsetbox.AnchoredOffsetbox`
        positions a :class:`~matplotlib.offsetbox.DrawingArea` (whose
        contents are sized in display points) at the anchor point.
        This gives free pixel-stable rendering across figure resize,
        pan, and zoom without any explicit draw-event callbacks.
        """
        from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea

        anchor_x, anchor_y, anchor_transform = _resolve_anchor(
            self._coord, ax, self._coord_type, self._frame)

        segments = _reticle_segments(
            self._style, self._size, self._gap, self._rotation,
            self._circle_npts, self._circle_gap_deg)

        if self._stroke_color is not None:
            stroke_effect = _stroke_path_effects(self._stroke_color,
                                                 self._stroke_lw)
        else:
            stroke_effect = None

        # Size the DrawingArea generously so all geometry fits with
        # room for the stroke outline. extent = outermost vertex + stroke
        # half-width + 1 pt of slack.
        outer = _outer_extent(segments)
        stroke_pad = max(0.0, self._stroke_lw - self._lw) / 2.0
        pad = outer + stroke_pad + 1.0
        total = 2.0 * pad
        cx = cy = pad

        da = DrawingArea(total, total, 0, 0)
        self._arm_artists = []
        for segment in segments:
            xs = [cx + pt[0] for pt in segment]
            ys = [cy + pt[1] for pt in segment]
            arm = Line2D(
                xs, ys,
                color=self._color, lw=self._lw,
                solid_capstyle='round',
                **self._line_kwargs)
            if stroke_effect is not None:
                arm.set_path_effects(stroke_effect)
            da.add_artist(arm)
            self._arm_artists.append(arm)

        anchor = AnchoredOffsetbox(
            loc='center', child=da, pad=0.0,
            frameon=False,
            bbox_to_anchor=(anchor_x, anchor_y),
            bbox_transform=anchor_transform,
            borderpad=0,
        )
        anchor.set_zorder(self._zorder)
        ax.add_artist(anchor)
        self._anchor_box = anchor

        # Label uses ax.annotate with offset-points text — independently
        # pixel-stable, anchored to the same world/pixel point.
        if self._label is not None:
            side = self._label_side
            if side == 'auto':
                side = _resolve_auto_label_side(
                    ax, anchor_x, anchor_y, anchor_transform)
            self._resolved_label_side = side
            self._label_artist = self._build_label(
                ax, anchor_x, anchor_y, anchor_transform, side)
        else:
            self._resolved_label_side = None

    # ----- label -------------------------------------------------------

    def _build_label(self, ax: Any, anchor_x: float, anchor_y: float,
                     anchor_transform: Any, side: str) -> Any:
        """Draw the label at *side* from the reticle center."""
        outer = _outer_extent(_reticle_segments(
            self._style, self._size, self._gap, self._rotation,
            self._circle_npts, self._circle_gap_deg))
        color = (self._label_color
                 if self._label_color is not None else self._color)
        # The arm stroke also backs the label, so a light label stays legible
        # over busy sky (the stroke's whole purpose), matching the arms.
        stroke_effect = (_stroke_path_effects(self._stroke_color,
                                              self._stroke_lw)
                         if self._stroke_color is not None else None)
        # Delegate to the shared compass-offset placer; user overrides in
        # ``label_kwargs`` take precedence (applied last inside the helper).
        return _place_offset_label(
            ax, anchor_x, anchor_y, anchor_transform,
            label=self._label, side=side, outer_extent=outer,
            label_offset=self._label_offset, color=color,
            fontsize=self._label_fontsize, zorder=self._zorder,
            path_effects=stroke_effect,
            label_kwargs=self._label_kwargs)

    # ----- accessors ---------------------------------------------------

    @property
    def arm_artists(self) -> list[Any]:
        """List of arm :class:`Line2D` artists (read-only view)."""
        return list(self._arm_artists)

    @property
    def label_artist(self) -> Any:
        """The label :class:`~matplotlib.text.Annotation` artist, or ``None``."""
        return self._label_artist

    @property
    def resolved_label_side(self) -> str | None:
        """The compass direction the label was placed at (after
        ``'auto'`` resolution). ``None`` if no label is set."""
        return self._resolved_label_side

    # ----- setters -----------------------------------------------------

    def set_color(self, color: Any, *,
                  stroke_color: Any = '__unset__') -> Reticle:
        """Update the body color (and optionally the stroke color).
        Returns ``self``."""
        self._color = color
        for arm in self._arm_artists:
            arm.set_color(color)
        if stroke_color != '__unset__':
            self._stroke_color = stroke_color
            if stroke_color is not None:
                stroke_effect = _stroke_path_effects(stroke_color,
                                                     self._stroke_lw)
                for arm in self._arm_artists:
                    arm.set_path_effects(stroke_effect)
            else:
                for arm in self._arm_artists:
                    arm.set_path_effects([])
        if (self._label_artist is not None
                and self._label_color is None):
            self._label_artist.set_color(color)
        return self

    def set_label(self, text: str | None, *,
                  side: str | None = None) -> Reticle:
        """Update the label text (and optionally its side). Rebuilds
        the label artist in place. Returns ``self``."""
        self._label = text
        if side is not None:
            if side not in _VALID_LABEL_SIDES:
                raise ValueError(
                    f"side must be one of {_VALID_LABEL_SIDES!r}, "
                    f"got {side!r}")
            self._label_side = side
        if self._host_axes is not None:
            if (self._label_artist is not None
                    and getattr(self._label_artist, 'axes', None) is not None):
                self._label_artist.remove()
            self._label_artist = None
            if text is not None:
                ax = self._host_axes
                anchor_x, anchor_y, anchor_transform = _resolve_anchor(
                    self._coord, ax, self._coord_type, self._frame)
                resolved = self._label_side
                if resolved == 'auto':
                    resolved = _resolve_auto_label_side(
                        ax, anchor_x, anchor_y, anchor_transform)
                self._resolved_label_side = resolved
                self._label_artist = self._build_label(
                    ax, anchor_x, anchor_y, anchor_transform, resolved)
            else:
                self._resolved_label_side = None
        return self

    def set_size(self, size: float, *, gap: float | None = None) -> Reticle:
        """Resize the reticle. Requires a rebuild — strips existing
        artists and re-attaches if the reticle is attached. Returns
        ``self``."""
        self._size = float(size)
        if gap is not None:
            self._gap = float(gap)
        if self._host_axes is not None:
            ax = self._host_axes
            self.remove()
            self.add_to(ax)
        return self

    def __repr__(self) -> str:
        if self._label:
            tag = f' label={self._label!r}'
        else:
            tag = ''
        return (f"<Reticle style={self._style!r} coord={self._coord!r} "
                f"size={self._size}{tag}>")


# ---------------------------------------------------------------------------
# Top-level convenience helper
# ---------------------------------------------------------------------------

def add_reticle(ax: Any, coord: SkyCoord | tuple[float, float], **kwargs: Any) -> Reticle:
    """Construct a :class:`Reticle` and attach it to *ax* in one call.

    Equivalent to ``Reticle(coord, **kwargs).add_to(ax)``. See
    :class:`Reticle` for the full parameter documentation.

    Returns
    -------
    Reticle
        The attached reticle, suitable for later restyling via
        :meth:`~Reticle.set_color` / :meth:`~Reticle.set_label` /
        :meth:`~Reticle.set_size` or removal via
        :meth:`~Reticle.remove`.

    Examples
    --------
    >>> import skyplothelper as sph
    >>> ax = sph.make_wcs_frame(111, 'TAN', frame='ICRS', center=180)
    >>> sph.add_reticle(ax, (180.0, 0.0))            # (ra, dec) in deg
    """
    return Reticle(coord, **kwargs).add_to(ax)
