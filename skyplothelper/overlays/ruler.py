"""Two-point distance annotation with tick marks (the :class:`Ruler`).

A :class:`Ruler` connects two points on an axes with a straight line or
great-circle arc and annotates the connecting line with regularly-spaced
tick marks and labels. The publication use case is a "distance between
source A and source B" annotation on a science image, complementing the
corner-anchored scale bars in :mod:`skyplothelper.overlays.annotations`
(:func:`add_sizebar` / :func:`add_sizebar_asec`).

How Ruler differs from related annotations
------------------------------------------
* :func:`~skyplothelper.overlays.annotations.add_sizebar` /
  :func:`~skyplothelper.overlays.annotations.add_sizebar_asec` are
  *corner-anchored*, single-label scale bars that stay pinned to an
  axes corner during pan / zoom. They render a fixed length with one
  caption — "5 arcsec" — and are the right pick for the canonical
  publication scale bar.
* Coordinate overlays (e.g. :func:`~skyplothelper.add_coord_overlay`)
  produce multi-spine *axes* — a full secondary grid of longitude /
  latitude lines drawn across the frame, with tick labels on each
  spine.
* A :class:`Ruler` is *free-floating between any two points* in the
  data, with ticks and labels along that single spine. It can
  reproduce a corner scale bar (corner anchored data coords) or a
  twinx-style spine label (long horizontal ruler) when desired,
  but the canonical use is "annotate the angular / physical
  distance between source A and source B."

Two construction paths mirror :class:`~skyplothelper.overlays.beam.Beam`:

* :class:`Ruler(xy1, xy2, ax=..., ...) <Ruler>` — endpoints in axes data
  (pixel) coordinates. ``ax=`` is optional but enables arcsec-aware
  labels by reading the pixel scale from the axes' WCS.
* :meth:`Ruler.from_world` — endpoints as :class:`~astropy.coordinates.SkyCoord`
  instances (or ``(lon, lat)`` degree tuples). Defaults to
  ``geodesic=True`` since world-coordinate inputs naturally imply an
  on-sky measurement.

Set ``geodesic=True`` for the great-circle path (essential for separations
of more than a few degrees, where the straight-line projection diverges
from the true on-sky distance); leave it ``False`` for a straight line
in data coordinates, which is faster and adequate for sub-degree fields.

Tick marks are perpendicular to the local tangent of the line in
**display** coordinates, so they look balanced regardless of the
axes' data aspect ratio. The default tick spacing is auto-picked as
``1 / 2 / 5 × 10^n`` to give roughly four major ticks across the line;
override with ``tick_interval=`` (in the active unit) or ``n_ticks=``.

Labels follow an angular-unit auto-selection that spans the full range:
``≥1°`` → deg, ``≥1′`` → arcmin, ``≥1″`` → arcsec, ``≥1 mas`` → mas,
``≥1 μas`` → μas, else nas — so a sub-arcsec VLBI-scale ruler labels in
mas / μas / nas rather than a tiny fraction of an arcsec. Caller can pin via
``label_unit='arcsec'|'arcmin'|'deg'|'mas'|'uas'|'nas'`` or supply a custom
``label_fmt=callable(value_arcsec, unit) -> str``.

For physical-unit labels — most commonly *projected physical distance at a
given redshift* — pass ``convert=`` as a dict driving an astropy
cosmology, an arbitrary distance for nearby resolved sources, or a
custom callable::

    Ruler(xy1, xy2, ax=ax,
          convert=dict(redshift=0.5, cosmo='Planck18', unit='kpc'))
    Ruler(xy1, xy2, ax=ax,
          convert=dict(distance=10, distance_unit='pc', unit='au'))
    Ruler(xy1, xy2, ax=ax,
          convert=lambda asec: asec * 2.5, convert_unit='kpc')

When set, ``convert=`` takes precedence over ``label_unit=`` (tick
labels render in the converted unit); ``label_fmt=`` still wins over
both.

Live reflow
-----------
When attached via :meth:`Ruler.add_to`, the ruler connects to its
axes' ``resize_event`` / ``xlim_changed`` / ``ylim_changed``
callbacks and rebuilds its tick orientations + label positions in
place whenever the layout changes. The result: tick lengths stay
visually consistent during pan / zoom / window-resize, and labels
remain anchored a fixed display-points distance from the line.
Call :meth:`Ruler.remove` to detach and disconnect the callbacks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from matplotlib.lines import Line2D

from .._stroke import _stroke_path_effects
from .._text_layout import (
    _normalize_readable_angle,
    _resolve_rotation_deg,
    _resolve_text_anchor,
)

__all__ = ['Ruler']


_VALID_TICK_SIDES = ('both', 'left', 'right', 'none')
# How a bare numeric (x, y) pair is interpreted. 'auto' = world on a sky
# frame, pixel on a plain axes. A SkyCoord always means world and ignores
# this, since it carries its own frame.
_VALID_RULER_COORD_TYPES = ('auto', 'world', 'pixel')
# Minors additionally accept 'auto' — follow whatever tick_side resolves to.
_VALID_MINOR_SIDES = ('auto', *_VALID_TICK_SIDES)
_VALID_UNITS = ('auto', 'arcsec', 'arcmin', 'deg', 'mas', 'uas', 'nas', 'pix')


def _auto_angle_unit(mag_asec: float) -> str:
    """Pick an angular unit for a magnitude in arcsec (the ``'auto'`` rule).

    Promotes across the full angular range so a sub-arcsec (VLBI-scale) ruler
    labels in mas / μas / nas instead of a tiny fraction of an arcsec: ``≥1°``
    → deg, ``≥1′`` → arcmin, ``≥1″`` → arcsec, ``≥1 mas`` → mas, ``≥1 μas`` →
    μas, else nas. The Ruler resolves this ONCE per ruler from the largest
    tick magnitude, so every tick shares one unit (no mixing).
    """
    mag = abs(mag_asec)
    if mag >= 3600.0:
        return 'deg'
    if mag >= 60.0:
        return 'arcmin'
    if mag >= 1.0:
        return 'arcsec'
    if mag >= 1e-3:        # ≥ 1 mas
        return 'mas'
    if mag >= 1e-6:        # ≥ 1 μas
        return 'uas'
    return 'nas'           # sub-μas → nanoarcsec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nice_interval(span: float, target_n: int = 4) -> float:
    """Pick a 1/2/5×10^n interval that gives ~*target_n* steps across *span*.

    Used when neither ``tick_interval`` nor an explicit ``n_ticks`` is
    supplied. The largest of ``(1, 2, 5) × 10^n`` that is ``<= span /
    target_n`` is selected, which lands the tick count between roughly
    ``target_n`` and ``2 × target_n`` across the line — close enough
    for an unobtrusive publication ruler without manual tuning.
    """
    if not np.isfinite(span) or span <= 0:
        return 1.0
    raw = span / max(target_n, 1)
    exp = np.floor(np.log10(raw))
    base = raw / 10.0 ** exp
    if base >= 5.0:
        step = 5.0
    elif base >= 2.0:
        step = 2.0
    else:
        step = 1.0
    return float(step * 10.0 ** exp)


def _auto_minor_subdivisions(interval: float) -> int:
    """Subdivisions for ``minor_ticks='auto'``. Mirrors matplotlib's
    AutoMinorLocator: a major step leading with 2 splits into 4, everything
    else into 5 — so minors land on readable fractions of the interval.

    Module-level (rather than a Ruler method) because it is backend-agnostic
    arithmetic: the plotly ``add_ruler`` imports it too, so both backends
    resolve ``minor_ticks='auto'`` identically.
    """
    if not np.isfinite(interval) or interval <= 0:
        return 5
    mantissa = interval / 10.0 ** np.floor(np.log10(abs(interval)))
    return 4 if np.isclose(mantissa, 2.0, rtol=1e-3) else 5


def _format_numeric(value: float, fmt: str | None = None) -> str:
    """Format the numeric portion of a tick label.

    Honors a printf-style format string when given (e.g. ``'%.2f'``),
    else falls back to 4-sig-fig with trailing-zero trim (``'5.0'`` →
    ``'5'``, ``'1.33333'`` → ``'1.333'``).
    """
    if fmt is not None:
        return fmt % value
    return f"{value:.4g}"


def _format_angle_label(value_asec: float, unit: str = 'auto',
                        fmt: str | None = None) -> str:
    """Format an angular value with auto-selected unit.

    ``'auto'`` promotes across the full angular range via
    :func:`_auto_angle_unit` (deg → arcmin → arcsec → mas → μas), so a
    sub-arcsec value labels in mas / μas rather than a fraction of an arcsec.
    ``'μas'`` is accepted as an alias for ``'uas'``.
    """
    if unit == 'μas':
        unit = 'uas'
    if unit not in _VALID_UNITS:
        raise ValueError(
            f"label_unit must be one of {_VALID_UNITS!r}, got {unit!r}")
    if unit == 'auto':
        unit = _auto_angle_unit(value_asec)
    if unit == 'arcsec':
        return f"{_format_numeric(value_asec, fmt)}″"     # ″
    if unit == 'arcmin':
        return f"{_format_numeric(value_asec / 60.0, fmt)}′"   # ′
    if unit == 'deg':
        return f"{_format_numeric(value_asec / 3600.0, fmt)}°"  # °
    if unit == 'mas':
        return f"{_format_numeric(value_asec * 1e3, fmt)} mas"
    if unit == 'uas':
        return f"{_format_numeric(value_asec * 1e6, fmt)} μas"
    if unit == 'nas':
        return f"{_format_numeric(value_asec * 1e9, fmt)} nas"
    # unit == 'pix' (only reached if caller pinned explicitly)
    return f"{_format_numeric(value_asec, fmt)} px"


def _pixscale_asec_from_ax(ax: Any) -> float | None:
    """Best-effort pixel scale in arcsec/pix from a WCSAxes' WCS.

    Returns ``None`` for non-WCS axes or when the pixel scale can't be
    determined.
    """
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        # proj_plane_pixel_scales folds in the CD/PC matrix, so this is
        # correct for HST-style WCS where wcs.cdelt is [1, 1] and the real
        # scale lives in the CD matrix (a raw cdelt read gives 3600"/px).
        scales = proj_plane_pixel_scales(ax.wcs)   # world units/px (deg)
        return float(scales[1]) * 3600.0
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _to_skycoord(value: Any) -> Any:
    """Coerce *value* (SkyCoord, or 2-tuple of degrees) to a SkyCoord."""
    from astropy.coordinates import SkyCoord
    if isinstance(value, SkyCoord):
        return value
    try:
        lon, lat = value
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"expected SkyCoord or (lon, lat) tuple, got {value!r}"
        ) from exc
    return SkyCoord(float(lon), float(lat), unit='deg')


def _resolve_xy_to_pixel(value: Any, ax: Any = None,
                         coord_type: str = 'auto',
                         ) -> tuple[float, float]:
    """Coerce *value* to a ``(x_pix, y_pix)`` tuple in data coords.

    * :class:`~astropy.coordinates.SkyCoord` (scalar) → projected to
      pixel coords via ``ax.wcs.world_to_pixel``. Requires *ax* to
      have a WCS.
    * Numeric ``(x, y)`` tuple / list / ``ndarray`` of length 2 →
      used directly as pixel (data) coordinates.

    Note that *numeric tuples are always treated as pixel coords*,
    even when *ax* has a WCS — this matches matplotlib's data-coord
    convention and keeps the canonical ``Ruler(xy1, xy2)`` usable on
    plain non-WCS axes. For literal world (lon, lat) tuples, use
    :meth:`Ruler.from_world` or wrap them in a :class:`SkyCoord`.
    """
    from astropy.coordinates import SkyCoord
    if isinstance(value, SkyCoord):
        if value.isscalar is False:
            raise ValueError(
                f"SkyCoord must be a scalar (single position), "
                f"got vector of size {value.size}")
        if ax is None or not hasattr(ax, 'wcs'):
            raise ValueError(
                "SkyCoord input requires ax= with a WCS to project "
                "to pixel coords — pass ax= at construction, or "
                "convert manually via ax.wcs.world_to_pixel(coord) "
                "and supply the resulting (x, y) tuple instead.")
        x, y = ax.wcs.world_to_pixel(value)
        return (float(x), float(y))
    # Numeric pair — what it MEANS depends on coord_type.
    try:
        x, y = value
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"expected SkyCoord or an (x, y) numeric pair, "
            f"got {value!r}") from exc

    if coord_type not in _VALID_RULER_COORD_TYPES:
        raise ValueError(
            f"coord_type must be one of {_VALID_RULER_COORD_TYPES!r}, "
            f"got {coord_type!r}")

    has_wcs = ax is not None and getattr(ax, 'wcs', None) is not None
    mode = coord_type
    if mode == 'auto':
        # On a sky frame a bare pair means sky degrees — matching Reticle and
        # the add_* family. On a plain axes it means data coordinates, which is
        # what matplotlib means there anyway. So the same call reads naturally
        # in both contexts; pin coord_type= when you want to be explicit.
        mode = 'world' if has_wcs else 'pixel'

    if mode == 'pixel':
        return (float(x), float(y))

    if not has_wcs:
        raise ValueError(
            "coord_type='world' needs an axes with a WCS to project sky "
            "coordinates to pixels. Pass ax= with a sky frame, or use "
            "coord_type='pixel' for data coordinates.")
    # A bare pair is interpreted in the axes' NATIVE frame, matching the
    # convention used throughout the geometry helpers; a SkyCoord (handled
    # above) is the way to supply coordinates from a different frame.
    px, py = ax.wcs.world_to_pixel_values(float(x), float(y))
    return (float(px), float(py))


# ---------------------------------------------------------------------------
# convert= helpers — angular → physical unit conversion for tick labels
# ---------------------------------------------------------------------------

# Arcseconds per radian — small-angle conversion factor used in
# ``_make_distance_converter``.
_ASEC_PER_RAD = 206264.80624709636


def _resolve_cosmology(cosmo: Any) -> Any:
    """Coerce ``cosmo`` to an :class:`astropy.cosmology.Cosmology` instance.

    Accepts a Cosmology instance directly, or a name string looked up in
    :mod:`astropy.cosmology` (e.g. ``'Planck18'``, ``'WMAP9'``).
    """
    from astropy import cosmology

    if isinstance(cosmo, cosmology.Cosmology):
        return cosmo
    if isinstance(cosmo, str):
        try:
            return getattr(cosmology, cosmo)
        except AttributeError:
            raise ValueError(
                f"unknown cosmology name {cosmo!r}; pick from "
                f"astropy.cosmology (e.g. 'Planck18', 'WMAP9', "
                f"'Planck15')") from None
    raise TypeError(
        f"cosmo must be a Cosmology instance or name string; "
        f"got {type(cosmo).__name__}")


def _make_redshift_converter(
        redshift: float, cosmo: Any = 'Planck18', unit: Any = 'kpc',
) -> tuple[Callable[[float], float], Any]:
    """Build an ``(arcsec → value_in_unit)`` callable for a redshift.

    Uses :meth:`astropy.cosmology.Cosmology.kpc_proper_per_arcmin`
    (proper transverse separation per unit angle on the sky), converted
    from kpc to *unit* via :mod:`astropy.units`.
    """
    from astropy import units as u

    cosmo_obj = _resolve_cosmology(cosmo)
    # kpc proper per arcmin → per arcsec
    kpc_per_asec = float(
        cosmo_obj.kpc_proper_per_arcmin(redshift).value) / 60.0
    try:
        target_unit = getattr(u, unit) if isinstance(unit, str) else unit
        factor = (1.0 * u.kpc).to(target_unit).value
    except (AttributeError, u.UnitConversionError) as exc:
        raise ValueError(
            f"unit must be a length-like astropy unit (got {unit!r})"
        ) from exc
    per_asec = kpc_per_asec * factor

    def fn(value_asec: float) -> float:
        return value_asec * per_asec

    return fn, unit


def _make_distance_converter(
        distance: float, distance_unit: Any = 'pc', unit: Any = 'au',
) -> tuple[Callable[[float], float], Any]:
    """Build an ``(arcsec → value_in_unit)`` small-angle callable.

    Useful for nearby resolved objects (Solar System bodies, stellar
    streams, MW open clusters). Computes
    ``size = distance × tan(θ) ≈ distance × θ_rad`` and converts to
    *unit*. The small-angle approximation is accurate to <1 ppm for
    separations below 1°.
    """
    from astropy import units as u

    try:
        dist_quantity = distance * (
            getattr(u, distance_unit) if isinstance(distance_unit, str)
            else distance_unit)
        target_unit = getattr(u, unit) if isinstance(unit, str) else unit
        dist_in_unit = dist_quantity.to(target_unit).value
    except (AttributeError, u.UnitConversionError) as exc:
        raise ValueError(
            f"distance_unit / unit must be length-like astropy units "
            f"(got distance_unit={distance_unit!r}, unit={unit!r})"
        ) from exc
    per_asec = dist_in_unit / _ASEC_PER_RAD

    def fn(value_asec: float) -> float:
        return value_asec * per_asec

    return fn, unit


def _normalize_convert(
        convert: Any, convert_unit: Any = None,
) -> tuple[Callable[[float], float] | None, Any]:
    """Resolve ``convert=`` into ``(fn, unit_str)`` or ``(None, None)``.

    * ``None``                     → ``(None, None)`` (no conversion).
    * callable                     → ``(callable, convert_unit)``.
    * ``{'redshift': z, ...}``      → cosmology-based converter.
    * ``{'distance': d, ...}``     → small-angle distance converter.
    """
    if convert is None:
        return None, None
    if callable(convert):
        return convert, convert_unit
    if isinstance(convert, dict):
        d = dict(convert)  # shallow copy so we can pop
        if 'redshift' in d:
            return _make_redshift_converter(
                redshift=d.pop('redshift'),
                cosmo=d.pop('cosmo', 'Planck18'),
                unit=d.pop('unit', 'kpc'))
        if 'distance' in d:
            return _make_distance_converter(
                distance=d.pop('distance'),
                distance_unit=d.pop('distance_unit', 'pc'),
                unit=d.pop('unit', 'au'))
        raise ValueError(
            f"convert dict must contain 'redshift' or 'distance'; "
            f"got keys {sorted(convert.keys())}")
    raise TypeError(
        f"convert must be None, callable, or dict; "
        f"got {type(convert).__name__}")


def _format_converted_label(value: float, unit: Any,
                            fmt: str | None = None) -> str:
    """Format a converted value + unit string (``'12.3 kpc'``)."""
    num = _format_numeric(value, fmt)
    if unit:
        return f"{num} {unit}"
    return num


# ---------------------------------------------------------------------------
# Ruler-specific side helpers
# ---------------------------------------------------------------------------
#
# Rotation / anchor helpers live in :mod:`skyplothelper._text_layout` so
# they can be shared with :func:`~skyplothelper.ticks.add_curved_lon_ticks`
# and :class:`~skyplothelper.coord_overlay.TickOverlay`. The side
# helpers below are ruler-specific (they encode the
# ``label_side='auto' == tick_side`` and ``title_side='auto' == opposite
# of labels`` conventions, which other overlays don't share).


def _resolve_label_side_sign(label_side: str, tick_side: str) -> int:
    """Pick the perpendicular sign (+1 / -1) for label placement.

    * ``'auto'``  — same side as the ticks ('+1' for both/none/left,
                    '-1' for right).
    * ``'left'``  — +1 (always, regardless of tick_side).
    * ``'right'`` — -1.
    """
    if label_side == 'auto':
        if tick_side == 'right':
            return -1
        return +1
    if label_side == 'left':
        return +1
    if label_side == 'right':
        return -1
    raise ValueError(
        f"label_side must be 'auto', 'left', or 'right'; got {label_side!r}")


def _resolve_title_side_sign(title_side: str, label_sign: int) -> int:
    """Pick the perpendicular sign for the title text.

    ``'auto'`` defaults to *opposite* the label side, so labels and
    title flank the line cleanly. Explicit ``'left'`` / ``'right'``
    map to +1 / -1.
    """
    if title_side == 'auto':
        return -label_sign
    if title_side == 'left':
        return +1
    if title_side == 'right':
        return -1
    raise ValueError(
        f"title_side must be 'auto', 'left', or 'right'; got {title_side!r}")


# ---------------------------------------------------------------------------
# Ruler
# ---------------------------------------------------------------------------

class Ruler:
    """Two-point distance annotation with tick marks and labels.

    Parameters
    ----------
    xy1, xy2 : (float, float) or :class:`~astropy.coordinates.SkyCoord`
        Endpoints, as scalar :class:`SkyCoord` objects or numeric pairs.
        A SkyCoord carries its own frame, is projected via
        ``ax.wcs.world_to_pixel``, and ignores ``coord_type`` entirely
        (it requires *ax* with a WCS at construction time).

        What a **numeric pair** means is set by ``coord_type``.
    coord_type : {'auto', 'world', 'pixel'}, optional
        How to read a numeric ``(x, y)`` pair. Default ``'auto'``:

        * on an axes **with a WCS**, the pair is **sky degrees** in the axes'
          native frame — matching :class:`~skyplothelper.Reticle` and the
          ``add_*`` shape helpers;
        * on a **plain** axes it is data coordinates, which is what
          matplotlib means there anyway.

        Pin ``'pixel'`` to place a ruler at a screen position on a sky frame
        (the scale-bar-like use), or ``'world'`` to be explicit. ``'world'``
        without a WCS raises rather than guessing.

        See also :meth:`Ruler.from_axes_fraction` for axes-fraction
        ``(0, 1)`` endpoint coordinates that stay dynamically
        pinned during pan / zoom (the twin-axis use case).
    ax : matplotlib Axes, optional
        Host axes. Optional at construction — if supplied, the pixel
        scale is read from ``ax.wcs`` (when present) so tick labels
        come out in arcsec / arcmin / degrees instead of pixels.
        Required at construction when *xy1* or *xy2* is a SkyCoord.
        If omitted at construction, the axes seen by :meth:`add_to`
        supplies the same information.

        **Passing** ``ax=`` **does not draw the ruler** — it only enables
        coordinate/pixel-scale resolution. You must call ``.add_to(ax)`` to
        add the artists (constructing a ``Ruler`` and never calling
        :meth:`add_to` is a silent no-op). The one-line idiom is
        ``sph.Ruler(xy1, xy2, ax=ax, ...).add_to(ax)``. Works on any
        WCSAxes, including all-sky projections.
    pixscale_asec : float, optional
        Pixel scale in arcsec/pix. Overrides any value derived from
        ``ax``. Useful for non-WCS axes or when the WCS is anisotropic
        and the relevant scale isn't ``|CDELT2|``.
    geodesic : bool, optional
        If ``True``, draw a great-circle arc between the endpoints
        (sampled in world coords via ``ax.wcs`` and reprojected back to
        pixels). Required for separations more than a few degrees, where
        the straight-line pixel projection diverges from the on-sky
        path. Default ``False``.
    n_geodesic_pts : int, optional
        Number of samples along the geodesic arc. Default ``64``.

    Tick configuration
    ------------------
    n_ticks : 'auto' or int, optional
        ``'auto'`` (default) picks a 1/2/5×10^n interval that gives
        roughly four major ticks across the line. An integer forces
        exactly that many evenly-spaced ticks (including both endpoints).
        Ignored when ``tick_interval`` or ``tick_positions`` is given.
    tick_interval : float, optional
        Explicit tick spacing in the active unit (arcsec if a pixel
        scale is known, otherwise pixels). Overrides ``n_ticks``.
    tick_positions : sequence of float, optional
        Explicit list of tick positions in the active unit. Overrides
        both ``n_ticks`` and ``tick_interval``. Out-of-range positions
        (outside ``[-lambda0*total, (1-lambda0)*total]``) are silently
        skipped.
    lambda0 : float, optional
        Fractional position along the ruler in ``[0, 1]`` where the
        value-0 tick lands. Default ``0`` — value-0 at xy1, all tick
        values positive (matches the canonical scale-bar semantic).
        ``0.5`` puts the zero at the midpoint and produces a
        symmetric ±-valued ruler; intermediate values are also
        valid. See the :meth:`Ruler.from_zero` factory for the
        coordinate-based version of this same idea.
    tick_length : float, optional
        Half-length of each major tick in display points. Default
        ``4`` — a compact publication-style tick that doesn't crowd
        the labels.
    tick_side : {'both', 'left', 'right', 'none'}, optional
        Which side of the line to draw ticks on. ``'both'`` (default)
        gives a centered-tick "ruler" look; ``'left'`` / ``'right'``
        give a one-sided look (relative to the xy1→xy2 direction);
        ``'none'`` suppresses tick marks entirely (line + labels only).
        Pair a one-sided ``tick_side`` with ``label_side`` when using a
        ruler as a stand-in for a twin axis, so ticks and labels both
        face away from the plot.

    Minor tick configuration
    ------------------------
    minor_ticks : int, 'auto', bool or None, optional
        Unlabeled minor ticks between the major ticks — small gradations
        that also make the ruler's orientation easier to read. The value is
        a **subdivision count**, matching
        :class:`matplotlib.ticker.AutoMinorLocator`: ``n`` splits each major
        interval into ``n`` sub-intervals, i.e. ``n-1`` minors between
        adjacent majors (``minor_ticks=5`` → 4 minors). ``'auto'`` (or
        ``True``) picks the subdivision from the major interval — 4 for a
        step leading with 2, otherwise 5. ``None`` (default) / ``False``
        turns minors off. Minors coinciding with a major tick are dropped,
        and an endpoint carrying an endcap is skipped.
    minor_tick_interval : float, optional
        Explicit minor spacing in the active unit (arcsec if a pixel scale
        is known, otherwise pixels). Overrides ``minor_ticks``.
    minor_tick_length : float, optional
        Half-length of each minor tick in display points. Default ``None``
        → half of ``tick_length``.
    minor_tick_side : {'auto', 'both', 'left', 'right', 'none'}, optional
        Side for the minor ticks. ``'auto'`` (default) follows ``tick_side``.
    minor_tick_color : color, optional
        Minor tick color. Defaults to the major tick color.
    minor_tick_lw : float, optional
        Minor tick linewidth. Defaults to the major tick linewidth.

    Label configuration
    -------------------
    labels : bool, optional
        Show per-tick labels. Default ``True``.
    label_unit : {'auto','arcsec','arcmin','deg','mas','uas','nas','pix'}, \
optional
        Unit for tick labels. ``'auto'`` (default) promotes across the full
        angular range (deg → arcmin → arcsec → mas → μas → nas), resolved once
        per ruler from its largest tick so every tick shares one unit — a
        sub-arcsec ruler reads in mas / μas / nas. Pin explicitly to force a
        unit (``'μas'`` accepted for ``'uas'``).
    label_fmt : callable, optional
        Custom label formatter — signature
        ``fmt(value_arcsec, unit) -> str``. When supplied, overrides
        all built-in formatting (including ``fmt=`` and ``convert=``).
    fmt : str, optional
        printf-style format string applied to the numeric portion of
        each label (e.g. ``'%.2f'``, ``'%3d'``). Works alongside the
        auto-unit selection and ``convert=``; the unit suffix /
        converted-unit string is appended after the formatted number.
        Ignored when ``label_fmt`` is supplied.
    label_fontsize : float or str, optional
        Forwarded to :func:`matplotlib.text.Text`. Default inherits from
        rcParams.
    label_color : color, optional
        Label color. Defaults to the main line color.
    label_offset : float, optional
        Extra spacing between the tick tip and the label, in display
        points. Default ``2``.
    label_side : {'auto', 'left', 'right'}, optional
        Side of the ruler line for label placement (and for the
        opposite-side default of ``title_side=``). ``'auto'`` matches
        the tick side when one-sided (``tick_side='left'`` →
        ``label_side='left'``); ``'left'`` is the +perpendicular side
        relative to the xy1→xy2 direction. Default ``'auto'``.
    label_rotation : {'auto', 'horizontal', 'perpendicular'} or float, optional
        Rotation of each tick label:

        * ``'auto'`` (default) — parallel to the local tangent of
          the ruler. Reads "along the bar" — the publication standard
          for distance annotations. For geodesic rulers the angle is
          computed per-tick from the local tangent.
        * ``'horizontal'`` — always 0° (every label upright).
        * ``'perpendicular'`` — 90° from the tangent (labels stand
          off perpendicular to the line).
        * a numeric value — literal rotation in degrees CCW from
          horizontal.

        All rotations are normalized to keep text right-side-up
        (matplotlib's usual axis-label convention).
    label_rotation_add : float, optional
        Extra rotation added on top of ``label_rotation``, in degrees.
        Useful for flipping ``'auto'`` to its perpendicular by passing
        ``90``, or tilting labels off-axis without giving up the
        auto-tangent behavior. Default ``0``.
    convert : None, callable, or dict, optional
        Convert tick values from arcsec to a physical / custom unit
        for the labels. Forms:

        * ``None`` (default) — no conversion; labels render in
          arcsec / arcmin / deg per ``label_unit``.
        * **Callable** ``f(value_arcsec) -> float`` — pair with
          ``convert_unit='kpc'`` (or similar) for the label suffix.
        * **Dict** ``{'redshift': z, 'cosmo': 'Planck18', 'unit': 'kpc'}``
          — uses :meth:`astropy.cosmology.Cosmology.kpc_proper_per_arcmin`
          for redshift-based projected physical distance. ``cosmo``
          defaults to ``'Planck18'`` and accepts any name from
          :mod:`astropy.cosmology` or a Cosmology instance; ``unit``
          defaults to ``'kpc'`` and accepts any length-like astropy
          unit string.
        * **Dict** ``{'distance': d, 'distance_unit': 'pc', 'unit': 'au'}``
          — small-angle ``size ≈ distance × θ`` for nearby resolved
          sources. ``distance_unit`` defaults to ``'pc'``, ``unit``
          to ``'au'``.

        When ``convert=`` is set, it takes precedence over
        ``label_unit=`` for the tick labels (``label_fmt=`` still wins
        over both). Requires a known pixel scale — either from the
        axes' WCS or from ``pixscale_asec=`` — since the conversion
        operates on arcsec values.
    convert_unit : str, optional
        Label suffix paired with a callable ``convert=``. Ignored for
        the dict forms (which carry their own ``unit`` entry).

    Title
    -----
    title : str, optional
        Caption rendered at the midpoint of the ruler on the side
        opposite the tick labels — useful for compactifying labels
        (set ``title='Size in kpc'`` to drop the per-tick unit
        suffix). Default ``None`` (no title).
    title_fontsize : float or str, optional
        Title font size. Defaults to rcParams.
    title_color : color, optional
        Title color. Defaults to the main line color.
    title_offset : float, optional
        Title gap beyond the tick tip on the title side, in display
        points — same semantic as ``label_offset``. Default ``3``, a
        touch larger than ``label_offset``'s default of ``2`` so the
        title sits one notch farther out than the tick labels.
    title_rotation : {'auto', 'horizontal', 'perpendicular'} or float, optional
        Rotation mode for the title (same vocabulary as
        ``label_rotation``). Default ``'auto'`` (parallel to the
        line at its midpoint).
    title_side : {'auto', 'left', 'right'}, optional
        Side of the ruler the title sits on. ``'auto'`` (default)
        puts it opposite the labels.

    Endcaps
    -------
    endcap_style : {'none', 'tick', 'arrow'}, optional
        Visually distinguish the ruler's endpoints from regular ticks:

        * ``'none'`` (default) — endpoints render as regular ticks,
          subject to the collision rule (the auto-tick logic drops
          an endpoint label when it would fall too close to the last
          regular tick).
        * ``'tick'`` — endpoints render as a longer tick (scaled by
          ``endcap_length_scale``), perpendicular to the line. Same
          orientation as regular ticks.
        * ``'arrow'`` — endpoints render as an outward-pointing
          arrowhead along the tangent direction (away from the line
          midpoint). The publication "interval covers from here to
          there" look. The arrowhead replaces the regular endpoint
          tick.
    endcaps : {'both', 'start', 'end', 'none'}, optional
        Which endpoint(s) get an endcap. ``'both'`` (default) draws
        a cap at xy1 *and* xy2; ``'start'`` only at xy1; ``'end'``
        only at xy2; ``'none'`` suppresses both even when
        ``endcap_style`` is set. Ignored when
        ``endcap_style='none'``.
    endcap_length_scale : float, optional
        For ``endcap_style='tick'``, the endpoint tick length is
        ``tick_length × endcap_length_scale``. Default ``1.5``
        (50 % longer than regular ticks).
    endcap_size : float, optional
        For ``endcap_style='arrow'``, the arrowhead size in
        matplotlib's ``mutation_scale`` units. Default ``8``.
    endcap_color : color, optional
        Endcap color. Defaults to ``tick_color`` (which in turn
        defaults to the main line color).
    endcap_lw : float, optional
        Endcap line width in points. Defaults to ``tick_lw`` (which
        defaults to the main line lw).
    endcap_label : {'auto', True, False}, optional
        Controls whether the endpoint receives a tick label:

        * ``'auto'`` (default) — labeled when ``endcap_style !=
          'none'`` (the endcap visually disambiguates from the
          previous tick) *or* when the regular collision rule
          allows it.
        * ``True`` — always label the endpoint, overriding the
          collision rule.
        * ``False`` — never label the endpoint, even when an
          endcap is drawn.

        Note: when ``endcap_label`` re-introduces a label that the
        regular collision rule would have dropped (the canonical
        "11.3″ ruler with 2″ ticks" case), the endpoint label may
        sit close to the last regular tick's label — that's
        intrinsic to the case (the cap is needed *because* the
        endpoint falls between regular ticks). Three ways to
        avoid the crowding:

        * ``endcap_label=False`` — keep the cap as a pure
          extent / direction marker, no distance label at the
          endpoint;
        * coarser ``tick_interval=`` — the regular ticks spread
          out, the endpoint label sits in clear space;
        * explicit ``tick_positions=[...]`` — list the exact
          positions you want, omitting the regular tick nearest
          the endpoint.

    Main line styling
    -----------------
    color : color, optional
        Main line color. Default ``'k'``.
    lw : float, optional
        Main line width in points. Default ``1.0``.
    ls : str, optional
        Main line linestyle. Default ``'-'``.
    alpha : float, optional
        Alpha for line, ticks and labels. Default ``1.0``.
    zorder : float, optional
        Draw order. Default ``5``.
    path_effects : list, optional
        Forwarded to the main line and tick artists (typically a
        :class:`~matplotlib.patheffects.withStroke` for legibility on
        busy backgrounds).

    Tick styling
    ------------
    tick_color, tick_lw, tick_ls : optional
        Independent tick styling. Each inherits from the corresponding
        main-line attribute when ``None`` (default).

    Examples
    --------
    >>> # Two points 50 px apart on a plain axes
    >>> ruler = Ruler((20, 50), (70, 50))
    >>> ruler.add_to(ax)

    >>> # WCS-aware: arcsec tick labels read from ax.wcs
    >>> ruler = Ruler((20, 50), (70, 50), ax=ax)
    >>> ruler.add_to(ax)

    >>> # SkyCoord endpoints with geodesic on-sky distance
    >>> from astropy.coordinates import SkyCoord
    >>> a = SkyCoord(180.0, +5.0, unit='deg')
    >>> b = SkyCoord(184.5, -3.0, unit='deg')
    >>> Ruler.from_world(a, b, ax=ax, color='C0', lw=1.2).add_to(ax)

    >>> # Pin to arcmin labels with one-sided ticks
    >>> Ruler((20, 50), (90, 50), ax=ax,
    ...       label_unit='arcmin', tick_side='left',
    ...       tick_interval=30.0).add_to(ax)

    >>> # Minor ticks for fine gradations (4 minors per major interval)
    >>> Ruler((20, 50), (90, 50), ax=ax, tick_interval=30.0,
    ...       minor_ticks=5).add_to(ax)

    >>> # Twin-axis stand-in: ticks + labels on one side, minors between
    >>> Ruler.from_axes_fraction((1.0, 0.0), (1.0, 1.0), ax=ax,
    ...       tick_side='right', label_side='right',
    ...       minor_ticks='auto').add_to(ax)

    >>> # Projected physical distance at z=0.5 (Planck18 cosmology)
    >>> Ruler((20, 50), (80, 50), ax=ax,
    ...       convert=dict(redshift=0.5, unit='kpc')).add_to(ax)
    """

    def __init__(self, xy1: Any, xy2: Any, *,
                 ax: Any = None,
                 pixscale_asec: float | None = None,
                 coord_type: str = 'auto',
                 geodesic: bool = False,
                 n_geodesic_pts: int = 64,
                 # Tick config
                 n_ticks: int | str | None = 'auto',
                 tick_interval: float | None = None,
                 tick_positions: Sequence[float] | None = None,
                 lambda0: float = 0.0,
                 tick_length: float = 4.0,
                 tick_side: str = 'both',
                 # Minor tick config
                 minor_ticks: int | str | bool | None = None,
                 minor_tick_interval: float | None = None,
                 minor_tick_length: float | None = None,
                 minor_tick_side: str = 'auto',
                 minor_tick_color: Any = None,
                 minor_tick_lw: float | None = None,
                 # Label config
                 labels: bool = True,
                 label_unit: str = 'auto',
                 label_fmt: Callable[..., str] | None = None,
                 fmt: str | None = None,
                 label_fontsize: float | str | None = None,
                 label_color: Any = None,
                 label_offset: float = 2.0,
                 label_side: str = 'auto',
                 label_rotation: str | float = 'auto',
                 label_rotation_add: float = 0.0,
                 # Unit conversion (overrides label_unit when set)
                 convert: Any = None,
                 convert_unit: str | None = None,
                 # Title
                 title: str | None = None,
                 title_fontsize: float | str | None = None,
                 title_color: Any = None,
                 title_offset: float = 3.0,
                 title_rotation: str | float = 'auto',
                 title_side: str = 'auto',
                 title_beyond_labels: bool = False,
                 # Endcaps
                 endcap_style: str = 'none',
                 endcaps: str = 'both',
                 endcap_length_scale: float = 1.5,
                 endcap_size: float = 8.0,
                 endcap_color: Any = None,
                 endcap_lw: float | None = None,
                 endcap_label: str | bool = 'auto',
                 # Main line styling
                 color: Any = 'k', lw: float = 1.0, ls: str = '-',
                 alpha: float = 1.0, zorder: float = 5,
                 path_effects: Any = None,
                 stroke_color: Any = None, stroke_lw: float = 2.5,
                 clip_on: bool = True,
                 # Tick styling
                 tick_color: Any = None, tick_lw: float | None = None,
                 tick_ls: str | None = None) -> None:
        # Endpoint resolution: SkyCoord → pixel via ax.wcs;
        # numeric tuple → pixel as-is (matches matplotlib's data-coord
        # convention).
        self._coord_type = coord_type
        self._xy1 = _resolve_xy_to_pixel(xy1, ax, coord_type)
        self._xy2 = _resolve_xy_to_pixel(xy2, ax, coord_type)
        self._ax_ref = ax
        if pixscale_asec is None and ax is not None:
            pixscale_asec = _pixscale_asec_from_ax(ax)
        self._pixscale_asec = (None if pixscale_asec is None
                                else float(pixscale_asec))
        self._geodesic = bool(geodesic)
        self._n_geodesic_pts = int(n_geodesic_pts)

        # Tick config
        if n_ticks != 'auto' and n_ticks is not None:
            n_ticks = int(n_ticks)
            if n_ticks < 2:
                raise ValueError(
                    f"n_ticks must be 'auto' or an integer >= 2, "
                    f"got {n_ticks}")
        self._n_ticks = n_ticks
        self._tick_interval = (None if tick_interval is None
                                else float(tick_interval))
        self._tick_positions = (None if tick_positions is None
                                 else [float(t) for t in tick_positions])
        lambda0 = float(lambda0)
        if not 0.0 <= lambda0 <= 1.0:
            raise ValueError(
                f"lambda0 must be in [0, 1] (fractional position along "
                f"the ruler where the value-0 tick lands), got {lambda0}")
        self._lambda0 = lambda0
        self._tick_length = float(tick_length)
        if tick_side not in _VALID_TICK_SIDES:
            raise ValueError(
                f"tick_side must be one of {_VALID_TICK_SIDES!r}, "
                f"got {tick_side!r}")
        self._tick_side = tick_side

        # Minor tick config. ``minor_ticks`` is a SUBDIVISION count (matching
        # matplotlib's AutoMinorLocator): n splits each major interval into n
        # sub-intervals, i.e. n-1 minors between adjacent majors.
        if minor_ticks is False:
            minor_ticks = None
        if minor_ticks is True:
            minor_ticks = 'auto'
        if (minor_ticks is not None and minor_ticks != 'auto'
                and not isinstance(minor_ticks, int)):
            raise ValueError(
                "minor_ticks must be None/False (off), True/'auto', or an "
                f"int subdivision count, got {minor_ticks!r}")
        if isinstance(minor_ticks, int) and minor_ticks < 2:
            raise ValueError(
                "minor_ticks is a subdivision count and must be >= 2 "
                f"(n splits each major interval into n), got {minor_ticks!r}")
        self._minor_ticks = minor_ticks
        self._minor_tick_interval = (None if minor_tick_interval is None
                                     else float(minor_tick_interval))
        self._minor_tick_length = (None if minor_tick_length is None
                                   else float(minor_tick_length))
        if minor_tick_side not in _VALID_MINOR_SIDES:
            raise ValueError(
                f"minor_tick_side must be one of {_VALID_MINOR_SIDES!r}, "
                f"got {minor_tick_side!r}")
        self._minor_tick_side = minor_tick_side
        self._minor_tick_color = minor_tick_color
        self._minor_tick_lw = (None if minor_tick_lw is None
                               else float(minor_tick_lw))

        # Label config
        self._labels = bool(labels)
        if label_unit == 'μas':
            label_unit = 'uas'
        if label_unit not in _VALID_UNITS:
            raise ValueError(
                f"label_unit must be one of {_VALID_UNITS!r}, "
                f"got {label_unit!r}")
        self._label_unit = label_unit
        self._label_fmt = label_fmt
        self._fmt = fmt
        self._label_fontsize = label_fontsize
        self._label_color = label_color
        self._label_offset = float(label_offset)
        if label_side not in ('auto', 'left', 'right'):
            raise ValueError(
                f"label_side must be 'auto', 'left', or 'right'; "
                f"got {label_side!r}")
        self._label_side = label_side
        self._label_rotation = label_rotation
        self._label_rotation_add = float(label_rotation_add)

        # Unit conversion. Stored unresolved (caller-supplied) for
        # transparency; ``_resolve_convert()`` returns the
        # ``(fn, unit_str)`` tuple at label-render time.
        self._convert = convert
        self._convert_unit = convert_unit

        # Title config
        self._title = title
        self._title_fontsize = title_fontsize
        self._title_color = title_color
        self._title_offset = float(title_offset)
        self._title_rotation = title_rotation
        if title_side not in ('auto', 'left', 'right'):
            raise ValueError(
                f"title_side must be 'auto', 'left', or 'right'; "
                f"got {title_side!r}")
        self._title_side = title_side
        self._title_beyond_labels = bool(title_beyond_labels)

        # Endcap config
        if endcap_style not in ('none', 'tick', 'arrow'):
            raise ValueError(
                f"endcap_style must be 'none', 'tick', or 'arrow'; "
                f"got {endcap_style!r}")
        self._endcap_style = endcap_style
        if endcaps not in ('both', 'start', 'end', 'none'):
            raise ValueError(
                f"endcaps must be 'both', 'start', 'end', or 'none'; "
                f"got {endcaps!r}")
        self._endcaps = endcaps
        self._endcap_length_scale = float(endcap_length_scale)
        self._endcap_size = float(endcap_size)
        self._endcap_color = endcap_color
        self._endcap_lw = (None if endcap_lw is None else float(endcap_lw))
        if endcap_label not in ('auto', True, False):
            raise ValueError(
                f"endcap_label must be 'auto', True, or False; "
                f"got {endcap_label!r}")
        self._endcap_label = endcap_label

        # Main line styling
        self._color = color
        self._lw = float(lw)
        self._ls = ls
        self._alpha = float(alpha)
        self._zorder = float(zorder)
        # ``stroke_color`` / ``stroke_lw`` are a convenience shortcut for
        # the cartographic-style outline stroke; ``path_effects=`` is the
        # explicit escape hatch and wins when both are given.
        if path_effects is None:
            path_effects = _stroke_path_effects(stroke_color, stroke_lw)
        self._path_effects = path_effects
        # Remembered so a later ``set_line(stroke_color=...)`` can keep the
        # width the ruler was built with instead of snapping back to the
        # default.
        self._stroke_lw = float(stroke_lw)
        self._clip_on = bool(clip_on)

        # Tick styling
        self._tick_color = tick_color
        self._tick_lw = (None if tick_lw is None else float(tick_lw))
        self._tick_ls = tick_ls

        # Built artists (filled in by add_to)
        self._line_artist: Any = None
        self._tick_artists: list[Any] = []
        self._minor_tick_artists: list[Any] = []
        self._label_artists: list[Any] = []
        self._title_artist: Any = None
        self._endcap_artists: list[Any] = []
        self._host_axes: Any = None

        # Live-reflow callback bookkeeping (filled in by add_to,
        # disconnected by remove). The ``_relayout_in_progress``
        # reentrancy guard prevents the redraw triggered by rebuild
        # from re-entering the layout-change callback.
        self._reflow_cid_resize: Any = None
        self._reflow_cid_xlim: Any = None
        self._reflow_cid_ylim: Any = None
        self._reflow_fig: Any = None
        self._relayout_in_progress = False

        # Axes-fraction pinning (set by :meth:`from_axes_fraction`).
        # When True, endpoints are re-resolved from the stored
        # fractional positions on every ``_build_artists`` call —
        # keeps the ruler glued to the same visual axes-frame
        # position during pan / zoom / resize. False for all other
        # construction paths (pixel / SkyCoord / polar / zero / world).
        self._uses_axes_frac = False
        self._xy1_axfrac: tuple[float, float] | None = None
        self._xy2_axfrac: tuple[float, float] | None = None

    # ----- factories ----------------------------------------------------

    @classmethod
    def from_world(cls, coord1: Any, coord2: Any, ax: Any, *,
                   geodesic: bool = True, **kwargs: Any) -> Ruler:
        """Build a :class:`Ruler` from two world-coordinate endpoints.

        The endpoints are projected to data (pixel) coords via
        ``ax.wcs``. ``geodesic`` defaults to ``True`` here since
        world-coord inputs naturally imply an on-sky measurement.

        Parameters
        ----------
        coord1, coord2 : SkyCoord or (lon, lat) tuple
            Endpoints in world coordinates. A 2-tuple is interpreted as
            degrees in the same frame as the axes' WCS.
        ax : WCSAxes
            Required — supplies the WCS for the world→pixel projection
            and (when constructed) is also the default host axes used by
            :meth:`add_to`.
        geodesic : bool, optional
            Whether to draw the line as a great-circle arc. Default
            ``True``.
        **kwargs
            Forwarded to :class:`Ruler`.

        Returns
        -------
        ruler : Ruler
        """
        if not hasattr(ax, 'wcs'):
            raise ValueError(
                "Ruler.from_world requires an axes with a WCS "
                "(WCSAxes); pass the data-coord endpoints to the "
                "canonical Ruler(xy1, xy2, ...) constructor instead.")
        c1 = _to_skycoord(coord1)
        c2 = _to_skycoord(coord2)
        # Use wcs.world_to_pixel — accepts SkyCoord directly.
        wcs = ax.wcs
        x1, y1 = wcs.world_to_pixel(c1)
        x2, y2 = wcs.world_to_pixel(c2)
        # These are ALREADY projected to pixels by this factory, so pin the
        # interpretation — otherwise coord_type='auto' would re-read them as
        # world degrees on a WCS axes and project them a second time.
        return cls((float(x1), float(y1)), (float(x2), float(y2)),
                   coord_type='pixel',
                    ax=ax, geodesic=geodesic, **kwargs)

    @classmethod
    def from_polar(cls, xy: Any, length: float, angle: float, *,
                   ax: Any = None, pixscale_asec: float | None = None,
                   length_unit: str = 'arcsec',
                   angle_convention: str = 'fits',
                   **kwargs: Any) -> Ruler:
        """Build a :class:`Ruler` from one endpoint, a length, and an
        angle (the Kapteyn ``rulersize`` / ``rulerangle`` style).

        Useful when the second endpoint is best expressed as "5 arcmin
        in direction PA=30°" — e.g. a literature value for a jet
        extent at a published position angle.

        Parameters
        ----------
        xy : (float, float) or :class:`~astropy.coordinates.SkyCoord`
            First endpoint, as a pixel ``(x, y)`` tuple or a scalar
            SkyCoord (auto-projected via ``ax.wcs``).
        length : float
            Ruler length, in units set by ``length_unit``.
        angle : float
            Direction from xy1 toward xy2, in degrees, interpreted per
            ``angle_convention``.
        ax : matplotlib Axes, optional
            Host axes. Used to derive ``pixscale_asec`` when not given
            explicitly. Required for ``length_unit`` other than
            ``'pix'``.
        pixscale_asec : float, optional
            Override / fallback pixel scale (arcsec/pix).
        length_unit : {'arcsec', 'arcmin', 'deg', 'pix'}, optional
            Units of *length*. Default ``'arcsec'``.
        angle_convention : {'fits', 'plot'}, optional
            ``'fits'`` (default) interprets *angle* as a position angle
            in degrees east of north on an N-up, E-left image
            (CDELT1 < 0 — the standard astronomical convention). When
            the host axes has CDELT1 > 0 (E to the right), the
            direction is flipped automatically. ``'plot'`` interprets
            *angle* as matplotlib's CCW-from-+x convention.
        **kwargs
            Forwarded to :class:`Ruler` (``color``, ``lw``,
            ``label_unit``, ``title``, …).

        Returns
        -------
        ruler : Ruler

        Notes
        -----
        Defaults differ from the canonical constructor:

        * ``endcap_style='arrow'`` — an outward arrowhead at the end,
          matching the "this much in that direction" use case
          ("60″ at PA=45°").
        * ``endcaps='end'`` — arrow only at the *end* endpoint
          (the anchor at ``xy`` is just the start point).

        Override either via the corresponding kwarg if you want a
        plain ruler or an arrow on both ends.

        Examples
        --------
        >>> # 5 arcmin bar at PA = 30° (E of N), anchored at (50, 50):
        >>> Ruler.from_polar((50, 50), length=5.0, angle=30.0,
        ...                   ax=ax, length_unit='arcmin',
        ...                   title='5\\' bar').add_to(ax)
        """
        if pixscale_asec is None and ax is not None:
            pixscale_asec = _pixscale_asec_from_ax(ax)

        # Resolve length to pixels.
        if length_unit == 'pix':
            length_pix = float(length)
        else:
            if pixscale_asec is None:
                raise ValueError(
                    f"Ruler.from_polar with length_unit={length_unit!r} "
                    f"requires a known pixel scale — pass an axes with "
                    f"a WCS or set pixscale_asec= explicitly.")
            if length_unit == 'arcsec':
                length_asec = float(length)
            elif length_unit == 'arcmin':
                length_asec = float(length) * 60.0
            elif length_unit == 'deg':
                length_asec = float(length) * 3600.0
            else:
                raise ValueError(
                    f"length_unit must be 'arcsec', 'arcmin', 'deg', "
                    f"or 'pix'; got {length_unit!r}")
            length_pix = length_asec / pixscale_asec

        # Resolve angle to matplotlib CCW-from-+x in pixel space.
        if angle_convention == 'plot':
            angle_plot_deg = float(angle)
        elif angle_convention == 'fits':
            # FITS PA = degrees east of north. On an N-up E-left image
            # (CDELT1 < 0), PA=0 → +y, PA=90 → -x → 90 + angle CCW
            # from +x. For E-right images, the east direction is +x,
            # so we negate.
            e_to_left = True
            if ax is not None and hasattr(ax, 'wcs'):
                try:
                    # CD/PC-aware handedness probe (a raw CDELT1 sign is
                    # wrong for CD-matrix WCS where CDELT1 is +1).
                    from ..wcs_frame import _east_increases_right
                    e_to_left = not _east_increases_right(ax.wcs)
                except Exception:
                    pass
            if e_to_left:
                angle_plot_deg = 90.0 + float(angle)
            else:
                angle_plot_deg = 90.0 - float(angle)
        else:
            raise ValueError(
                f"angle_convention must be 'fits' or 'plot', "
                f"got {angle_convention!r}")

        angle_rad = np.radians(angle_plot_deg)
        # Resolve the anchor coordinate (accepts SkyCoord or pixel tuple).
        x1, y1 = _resolve_xy_to_pixel(xy, ax)
        x2 = x1 + length_pix * np.cos(angle_rad)
        y2 = y1 + length_pix * np.sin(angle_rad)
        # from_polar defaults to an outward arrowhead at the end — the
        # "this much in that direction" use case reads naturally with
        # an arrow ("60'' at PA=45°" → arrow points 45° E of N).
        # Caller can override with endcap_style='none' / 'tick'.
        kwargs.setdefault('endcap_style', 'arrow')
        kwargs.setdefault('endcaps', 'end')
        # Already pixel coords — see from_world.
        return cls((x1, y1), (x2, y2), coord_type='pixel',
                    ax=ax, pixscale_asec=pixscale_asec, **kwargs)

    @classmethod
    def from_zero(cls, xy: Any, extent: float, angle: float, *,
                  extent_back: float | None = None,
                  ax: Any = None, pixscale_asec: float | None = None,
                  length_unit: str = 'arcsec',
                  angle_convention: str = 'fits',
                  **kwargs: Any) -> Ruler:
        """Build a Ruler with the value-0 tick at *xy*, extending
        *extent* in the +angle direction and *extent_back* in the
        opposite direction.

        The user-friendly companion to :meth:`from_polar`: instead
        of specifying a *start* coordinate and a length, specify
        the *zero* coordinate and how far the ruler extends on
        each side. Internally this picks an ``xy1``/``xy2`` and a
        ``lambda0`` such that the value-0 tick lands at *xy*.

        Parameters
        ----------
        xy : (float, float) or :class:`~astropy.coordinates.SkyCoord`
            Anchor point — pixel ``(x, y)`` tuple, or a scalar
            SkyCoord auto-projected via ``ax.wcs``. The value-0 tick
            will be drawn at this point.
        extent : float
            Length of the ruler in the +angle direction, in units
            set by *length_unit*.
        angle : float
            Direction in degrees from *xy* toward the +extent end,
            interpreted per *angle_convention*.
        extent_back : float, optional
            Length on the −angle side. Default ``None`` is treated
            as ``extent`` (symmetric ruler centered on *xy*). Pass
            a different value for an asymmetric ruler (e.g.
            ``extent=10, extent_back=5`` for 10″ forward + 5″ back).
        ax, pixscale_asec, length_unit, angle_convention :
            Same semantics as :meth:`from_polar`.
        **kwargs :
            Forwarded to :class:`Ruler`. Common pairings:
            ``endcap_style='arrow', endcaps='both'`` for a
            bidirectional arrow look (the natural visual for a
            symmetric ruler around a coordinate); ``label_unit=``
            to pin units; etc.

        Returns
        -------
        ruler : Ruler

        Examples
        --------
        >>> # Symmetric ±15″ ruler at the field center, with
        >>> # arrows on both ends:
        >>> Ruler.from_zero((50, 50), extent=15.0, angle=0.0,
        ...                   ax=ax, length_unit='arcsec',
        ...                   angle_convention='plot',
        ...                   endcap_style='arrow').add_to(ax)

        >>> # Asymmetric: 30 kpc back, 60 kpc forward (PA=45°):
        >>> Ruler.from_zero(xy_source, extent=60.0,
        ...                   extent_back=30.0, angle=45.0,
        ...                   ax=ax, length_unit='arcsec').add_to(ax)
        """
        if extent_back is None:
            extent_back = extent
        extent = float(extent)
        extent_back = float(extent_back)
        if extent + extent_back <= 0:
            raise ValueError(
                "extent + extent_back must be positive; got "
                f"extent={extent}, extent_back={extent_back}")

        if pixscale_asec is None and ax is not None:
            pixscale_asec = _pixscale_asec_from_ax(ax)

        def _to_pixels(length_value: float) -> float:
            if length_unit == 'pix':
                return length_value
            if pixscale_asec is None:
                raise ValueError(
                    f"Ruler.from_zero with length_unit="
                    f"{length_unit!r} requires a known pixel scale "
                    f"— pass an axes with a WCS or set "
                    f"pixscale_asec= explicitly.")
            if length_unit == 'arcsec':
                length_asec = length_value
            elif length_unit == 'arcmin':
                length_asec = length_value * 60.0
            elif length_unit == 'deg':
                length_asec = length_value * 3600.0
            else:
                raise ValueError(
                    f"length_unit must be 'arcsec', 'arcmin', 'deg', "
                    f"or 'pix'; got {length_unit!r}")
            return length_asec / pixscale_asec

        ext_fwd_pix = _to_pixels(extent)
        ext_back_pix = _to_pixels(extent_back)

        # Resolve angle to matplotlib CCW-from-+x in pixel space
        # (shared with from_polar).
        if angle_convention == 'plot':
            angle_plot_deg = float(angle)
        elif angle_convention == 'fits':
            e_to_left = True
            if ax is not None and hasattr(ax, 'wcs'):
                try:
                    # CD/PC-aware handedness probe (a raw CDELT1 sign is
                    # wrong for CD-matrix WCS where CDELT1 is +1).
                    from ..wcs_frame import _east_increases_right
                    e_to_left = not _east_increases_right(ax.wcs)
                except Exception:
                    pass
            if e_to_left:
                angle_plot_deg = 90.0 + float(angle)
            else:
                angle_plot_deg = 90.0 - float(angle)
        else:
            raise ValueError(
                f"angle_convention must be 'fits' or 'plot', "
                f"got {angle_convention!r}")

        angle_rad = np.radians(angle_plot_deg)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        # Resolve the zero-point coordinate (accepts SkyCoord or
        # pixel tuple).
        x0, y0 = _resolve_xy_to_pixel(xy, ax)
        x1 = x0 - ext_back_pix * dx
        y1 = y0 - ext_back_pix * dy
        x2 = x0 + ext_fwd_pix * dx
        y2 = y0 + ext_fwd_pix * dy

        # lambda0 = back_fraction along xy1 → xy2.
        total_pix = ext_back_pix + ext_fwd_pix
        lambda0_resolved = ext_back_pix / total_pix

        # Default endcap style for from_zero: arrows on BOTH ends —
        # the natural visual for a "ruler extending from a zero
        # coordinate in both directions". Caller can override.
        kwargs.setdefault('endcap_style', 'arrow')
        kwargs.setdefault('endcaps', 'both')

        # Already pixel coords — see from_world.
        return cls((x1, y1), (x2, y2), coord_type='pixel',
                    ax=ax, pixscale_asec=pixscale_asec,
                    lambda0=lambda0_resolved, **kwargs)

    @classmethod
    def from_axes_fraction(cls, xy1: Sequence[float], xy2: Sequence[float],
                           *, ax: Any, **kwargs: Any) -> Ruler:
        """Build a Ruler from axes-fraction (0–1) endpoint coordinates.

        Useful for placing a Ruler at a specific *visual* position on
        the axes — e.g. as a pseudo twin-axis spine just outside the
        plot frame (``y_frac = -0.05`` below the bottom edge), or
        spanning the bottom edge from corner to corner (``(0, 0)``
        and ``(1, 0)``).

        Unlike the other factories, the ruler stays *dynamically
        pinned* to the axes-fraction position: panning, zooming, or
        resizing the figure re-projects the endpoints to pixel coords
        each time, so the ruler remains glued to the same visual
        position on the axes (matplotlib twin-axis behavior). The
        live-reflow callbacks set up by :meth:`add_to` drive this
        re-projection.

        Default ``clip_on=False`` so the ruler renders cleanly when
        placed *outside* the axes bounding box (the typical
        twin-axis layout). Override via ``clip_on=True`` to clip
        to the axes box like a normal artist.

        Parameters
        ----------
        xy1, xy2 : (float, float)
            Endpoints in axes-fraction coordinates (each in
            ``[0, 1]`` for inside-the-axes; negative or > 1 places
            the ruler outside the frame, which is the typical
            stacked-twin layout).
        ax : matplotlib Axes
            Required — supplies ``ax.transAxes`` for the
            fraction → display projection.
        **kwargs :
            Forwarded to :class:`Ruler`. Common pairings:
            ``tick_interval=`` and ``label_unit=`` for the spine
            tick layout, ``convert=`` for a separate unit on a
            stacked twin (e.g. ``convert=lambda asec: ...``),
            ``title=`` for a spine label.

        Returns
        -------
        ruler : Ruler

        Examples
        --------
        >>> # Twin spine 5% below the bottom edge, ticks in arcsec:
        >>> Ruler.from_axes_fraction((0, -0.05), (1, -0.05), ax=ax,
        ...                           tick_interval=30.0,
        ...                           title='Offset (arcsec)').add_to(ax)

        >>> # Second stacked twin 12% below, ticks converted to kpc:
        >>> Ruler.from_axes_fraction((0, -0.12), (1, -0.12), ax=ax,
        ...                           tick_interval=5.0,
        ...                           convert=lambda asec: asec * 0.5,
        ...                           convert_unit='kpc',
        ...                           title='Distance (kpc)').add_to(ax)
        """
        # Default clip_on=False so rulers outside the axes box still
        # render — the typical twin-axis layout.
        kwargs.setdefault('clip_on', False)

        # Capture axes-fraction values for dynamic re-projection.
        f1 = (float(xy1[0]), float(xy1[1]))
        f2 = (float(xy2[0]), float(xy2[1]))

        # Compute initial pixel coords for the constructor (will be
        # refreshed on every layout change via the
        # ``_uses_axes_frac`` flag set below).
        pixel_xy1, pixel_xy2 = cls._axfrac_to_pixel(ax, f1, f2)

        # Already pixel coords — see from_world.
        ruler = cls(pixel_xy1, pixel_xy2, ax=ax, coord_type='pixel',
                    **kwargs)
        ruler._uses_axes_frac = True
        ruler._xy1_axfrac = f1
        ruler._xy2_axfrac = f2
        return ruler

    @staticmethod
    def _axfrac_to_pixel(
            ax: Any, f1: tuple[float, float], f2: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Project axes-fraction endpoints to data (pixel) coords
        using the axes' current ``transAxes`` and inverse
        ``transData``."""
        trans_ax = ax.transAxes
        inv = ax.transData.inverted()
        p1_disp = trans_ax.transform(f1)
        p2_disp = trans_ax.transform(f2)
        p1_data = inv.transform(p1_disp)
        p2_data = inv.transform(p2_disp)
        return ((float(p1_data[0]), float(p1_data[1])),
                (float(p2_data[0]), float(p2_data[1])))

    def _refresh_endpoints_from_axes_frac(self, ax: Any) -> None:
        """Recompute ``self._xy1`` / ``self._xy2`` from the cached
        axes-fraction positions, using *ax*'s current transform.
        Called by :meth:`_build_artists` whenever
        ``self._uses_axes_frac`` is True (i.e. the ruler was built
        via :meth:`from_axes_fraction`)."""
        # Invariant: only reached when built via from_axes_fraction,
        # which always populates both cached axes-fraction positions.
        assert self._xy1_axfrac is not None and self._xy2_axfrac is not None
        self._xy1, self._xy2 = self._axfrac_to_pixel(
            ax, self._xy1_axfrac, self._xy2_axfrac)

    def _compute_title_offset_beyond_labels(
            self, ax: Any, mx_d: float, my_d: float, m_px: float,
            m_py: float, title_sign: float, dpi: float) -> float:
        """Auto-compute the display-pixel title offset so the title
        sits past the labels' rendered bounding boxes.

        Called from :meth:`_build_artists` when
        ``self._title_beyond_labels`` is True *and* the title is on
        the same side as the labels. Measures each label's window
        extent, projects each corner onto the title's outward
        perpendicular direction, takes the max, and adds
        ``title_offset`` as an additional gap. Force-draws the
        figure first so the label bboxes are populated.
        """
        fig = ax.figure
        fig.canvas.draw()
        try:
            renderer = fig.canvas.get_renderer()
        except AttributeError:
            # Non-Agg backends without get_renderer — fall back to
            # the default gap-beyond-tick-tip semantic.
            tick_len_disp = self._tick_length * dpi / 72.0
            return tick_len_disp + self._title_offset * dpi / 72.0
        outward_x = title_sign * m_px
        outward_y = title_sign * m_py
        max_perp = 0.0
        for lab in self._label_artists:
            bbox = lab.get_window_extent(renderer=renderer)
            for cx, cy in ((bbox.x0, bbox.y0), (bbox.x1, bbox.y0),
                           (bbox.x0, bbox.y1), (bbox.x1, bbox.y1)):
                proj = (cx - mx_d) * outward_x + (cy - my_d) * outward_y
                if proj > max_perp:
                    max_perp = proj
        return max_perp + self._title_offset * dpi / 72.0

    # ----- properties ---------------------------------------------------

    @property
    def xy1(self) -> tuple[float, float]:
        return self._xy1

    @property
    def xy2(self) -> tuple[float, float]:
        return self._xy2

    @property
    def geodesic(self) -> bool:
        return self._geodesic

    @property
    def pixscale_asec(self) -> float | None:
        """Pixel scale in arcsec/pix, or ``None`` if unknown."""
        return self._pixscale_asec

    @property
    def line_artist(self) -> Any:
        """The main :class:`~matplotlib.lines.Line2D`, or ``None`` if
        the ruler hasn't been added to an axes yet."""
        return self._line_artist

    @property
    def tick_artists(self) -> list[Any]:
        """The major-tick :class:`Line2D` artists, in tick order."""
        return list(self._tick_artists)

    @property
    def minor_tick_artists(self) -> list[Any]:
        """The minor-tick :class:`Line2D` artists, in tick order (empty
        unless ``minor_ticks`` / ``minor_tick_interval`` is set)."""
        return list(self._minor_tick_artists)

    @property
    def label_artists(self) -> list[Any]:
        """The per-tick label :class:`~matplotlib.text.Text` artists."""
        return list(self._label_artists)

    @property
    def endcap_artists(self) -> list[Any]:
        """The endcap artists (one or two :class:`Line2D` for
        ``endcap_style='tick'``, one or two
        :class:`~matplotlib.patches.FancyArrowPatch` for
        ``endcap_style='arrow'``)."""
        return list(self._endcap_artists)

    @property
    def title_artist(self) -> Any:
        """The title :class:`~matplotlib.text.Text` artist, or ``None``
        if the ruler has no title or has not been attached yet."""
        return self._title_artist

    @property
    def title(self) -> str | None:
        """Current title string (``None`` if unset)."""
        return self._title

    # ----- distance / sampling ------------------------------------------

    def angular_distance_asec(self) -> float | None:
        """Return the line length in arcsec, or ``None`` if unknown.

        Uses the great-circle separation between the endpoints (via the
        axes' WCS) when ``geodesic=True``; otherwise falls back to
        ``pixel_distance × pixscale_asec``. Returns ``None`` when no
        pixel scale or WCS is available — labels then fall back to
        pixel units.
        """
        if (self._geodesic and self._ax_ref is not None
                and hasattr(self._ax_ref, 'wcs')):
            from ..globe.spherical import great_circle_distance
            wcs = self._ax_ref.wcs
            lon1, lat1 = wcs.pixel_to_world_values(
                self._xy1[0], self._xy1[1])
            lon2, lat2 = wcs.pixel_to_world_values(
                self._xy2[0], self._xy2[1])
            radians = great_circle_distance(
                float(lon1), float(lat1), float(lon2), float(lat2))
            return float(np.degrees(radians)) * 3600.0
        # Straight-line pixel distance × pixscale
        dx = self._xy2[0] - self._xy1[0]
        dy = self._xy2[1] - self._xy1[1]
        d_pix = float(np.hypot(dx, dy))
        if self._pixscale_asec is not None:
            return d_pix * self._pixscale_asec
        return None

    def _line_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(xs, ys)`` arrays of points along the line in data
        coords. Two points for a straight line, *n_geodesic_pts* for
        the great-circle arc."""
        if (self._geodesic and self._ax_ref is not None
                and hasattr(self._ax_ref, 'wcs')):
            from ..globe.spherical import great_circle_arc
            wcs = self._ax_ref.wcs
            lon1, lat1 = wcs.pixel_to_world_values(
                self._xy1[0], self._xy1[1])
            lon2, lat2 = wcs.pixel_to_world_values(
                self._xy2[0], self._xy2[1])
            lons, lats = great_circle_arc(
                float(lon1), float(lat1),
                float(lon2), float(lat2),
                n_pts=self._n_geodesic_pts)
            xs, ys = wcs.world_to_pixel_values(lons, lats)
            return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        return (np.array([self._xy1[0], self._xy2[0]], dtype=float),
                np.array([self._xy1[1], self._xy2[1]], dtype=float))

    # ----- tick generation ---------------------------------------------

    def _ruler_total(self) -> float:
        """Ruler length in the active unit (arcsec when a pixel scale is
        known, otherwise pixels). ``0`` for a degenerate ruler."""
        total = self.angular_distance_asec()
        if total is None:
            dx = self._xy2[0] - self._xy1[0]
            dy = self._xy2[1] - self._xy1[1]
            total = float(np.hypot(dx, dy))
        return float(total)

    def _resolved_major_interval(self, total: float) -> float | None:
        """The major-tick spacing actually in use, in the active unit.

        ``None`` when the majors come from explicit ``tick_positions`` (there
        is no single spacing to subdivide) — callers that need a spacing then
        fall back to the observed tick gaps.
        """
        if self._tick_positions is not None:
            return None
        if self._tick_interval is not None:
            return float(self._tick_interval)
        if isinstance(self._n_ticks, int):
            return total / (self._n_ticks - 1)
        return _nice_interval(total, target_n=4)

    def _minor_tick_positions(self) -> list[float]:
        """Return ``t_frac`` positions for the minor ticks (never labeled).

        Minors subdivide the major interval; any position coinciding with a
        major tick is dropped so the two never overprint.
        """
        if self._minor_ticks is None and self._minor_tick_interval is None:
            return []
        total = self._ruler_total()
        if total <= 0:
            return []
        zero_d = self._lambda0 * total
        d_min, d_max = -zero_d, total - zero_d

        majors = [d for d, _ in self._major_tick_positions()]
        if self._minor_tick_interval is not None:
            step = float(self._minor_tick_interval)
        else:
            subdiv = self._minor_ticks
            if subdiv is None:
                return []
            major = self._resolved_major_interval(total)
            if major is None:
                # Majors came from explicit tick_positions: subdivide their
                # observed spacing so 'subdivision' still has a meaning.
                if len(majors) < 2:
                    return []
                major = float(np.median(np.diff(sorted(majors))))
            n = (_auto_minor_subdivisions(major)
                 if subdiv == 'auto' else int(subdiv))
            if n < 2 or major <= 0:
                return []
            step = major / n
        if step <= 0:
            return []

        tol = 1e-6 * step
        out = []
        k_min = int(np.ceil((d_min - 1e-9) / step))
        k_max = int(np.floor((d_max + 1e-9) / step))
        for k in range(k_min, k_max + 1):
            d = k * step
            if any(abs(d - dm) < tol for dm in majors):
                continue
            out.append((d + zero_d) / total)
        return out

    def _major_tick_positions(self) -> list[tuple[float, float]]:
        """Return a list of ``(d_label, t_frac)`` for each major tick.

        ``d_label`` is the *signed* distance from the value-0 tick
        (positive toward xy2, negative toward xy1 when
        ``lambda0 > 0``). The label value displayed to the user is
        ``d_label`` directly. ``t_frac`` is the fractional position
        along the line in ``[0, 1]``.

        When ``lambda0 == 0`` (default), ``d_label`` is identical to
        the cumulative distance from xy1.
        """
        total = self._ruler_total()
        if total <= 0:
            return []

        zero_d = self._lambda0 * total      # abs distance from xy1 to zero tick
        d_min = -zero_d                      # d_label at xy1
        d_max = total - zero_d               # d_label at xy2

        # Explicit tick positions win over any auto / interval logic.
        # User-supplied positions are in d_label units (signed when
        # lambda0 > 0).
        if self._tick_positions is not None:
            positions = []
            for d_label in self._tick_positions:
                d_label = float(d_label)
                if d_label < d_min - 1e-6 or d_label > d_max + 1e-6:
                    continue   # out-of-range — silently skip
                t = (d_label + zero_d) / total
                positions.append((d_label, t))
            return positions

        interval = self._resolved_major_interval(total)
        if interval is None or interval <= 0:
            return []

        # Regular ticks at k * interval (signed) inside [d_min, d_max].
        k_min = int(np.ceil((d_min - 1e-9) / interval))
        k_max = int(np.floor((d_max + 1e-9) / interval))
        positions = [
            (k * interval, (k * interval + zero_d) / total)
            for k in range(k_min, k_max + 1)
        ]

        # Endpoint inclusion: apply collision rule + endcap_label
        # logic at BOTH endpoints (start at d_min, end at d_max).
        # When lambda0 == 0 the start is always at d=0 which is on the
        # regular grid, so only the end may need separate handling.
        # For lambda0 != 0 both endpoints
        # may not coincide with a regular tick.
        def _should_include(d_endpoint: float, is_end: bool) -> bool:
            # Already in positions?
            for d_existing, _ in positions:
                if abs(d_existing - d_endpoint) < 1e-6:
                    return False
            # endcap_label overrides
            if self._endcap_label is True:
                return True
            if self._endcap_label is False:
                return False
            # 'auto': include if an endcap is drawn on this side, OR if
            # gap to the nearest regular tick exceeds half the interval.
            side = 'end' if is_end else 'start'
            cap_here = (self._endcap_style != 'none'
                        and self._endcaps in ('both', side))
            if cap_here:
                return True
            if positions:
                nearest = positions[-1][0] if is_end else positions[0][0]
                gap = abs(d_endpoint - nearest)
                return gap > 0.5 * interval
            return True

        if _should_include(d_min, is_end=False):
            positions.insert(0, (d_min, 0.0))
        if _should_include(d_max, is_end=True):
            positions.append((d_max, 1.0))

        return positions

    def _interpolate_along_line(self, xs: np.ndarray, ys: np.ndarray,
                                t_frac: float) -> tuple[float, float]:
        """Interpolate to ``(x, y)`` at fractional position *t_frac*
        in ``[0, 1]`` along the line, by cumulative pixel arc length."""
        if len(xs) == 2:
            return (xs[0] + t_frac * (xs[1] - xs[0]),
                    ys[0] + t_frac * (ys[1] - ys[0]))
        seg = np.hypot(np.diff(xs), np.diff(ys))
        total = float(seg.sum())
        if total == 0:
            return float(xs[0]), float(ys[0])
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        target = t_frac * total
        idx = int(np.searchsorted(cum, target))
        if idx <= 0:
            return float(xs[0]), float(ys[0])
        if idx >= len(cum):
            return float(xs[-1]), float(ys[-1])
        seg_t = (target - cum[idx - 1]) / max(seg[idx - 1], 1e-12)
        return (float(xs[idx - 1] + seg_t * (xs[idx] - xs[idx - 1])),
                float(ys[idx - 1] + seg_t * (ys[idx] - ys[idx - 1])))

    def _local_tangent(self, xs: np.ndarray, ys: np.ndarray, t_frac: float,
                       eps: float = 0.01) -> tuple[float, float]:
        """Unit tangent vector (in data coords) at fractional position
        *t_frac* along the line. Forward-difference for smooth geodesic
        arcs; constant for straight lines."""
        if len(xs) == 2:
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
            n = np.hypot(dx, dy)
            if n == 0:
                return (1.0, 0.0)
            return (dx / n, dy / n)
        t_a = max(0.0, t_frac - eps)
        t_b = min(1.0, t_frac + eps)
        if t_a == t_b:
            t_a, t_b = max(0.0, t_frac - eps), min(1.0, t_frac + 2 * eps)
        x_a, y_a = self._interpolate_along_line(xs, ys, t_a)
        x_b, y_b = self._interpolate_along_line(xs, ys, t_b)
        dx, dy = x_b - x_a, y_b - y_a
        n = np.hypot(dx, dy)
        if n == 0:
            return (1.0, 0.0)
        return (dx / n, dy / n)

    # ----- style resolution --------------------------------------------

    def _resolved_tick_color(self) -> Any:
        return self._tick_color if self._tick_color is not None else self._color

    def _resolved_tick_lw(self) -> float:
        return self._tick_lw if self._tick_lw is not None else self._lw

    def _resolved_tick_ls(self) -> str:
        return self._tick_ls if self._tick_ls is not None else '-'

    # Minors default to inheriting the major ticks' look and side, so turning
    # them on is a one-knob change; each aspect can still be pinned separately.
    def _resolved_minor_side(self) -> str:
        return (self._tick_side if self._minor_tick_side == 'auto'
                else self._minor_tick_side)

    def _resolved_minor_length(self) -> float:
        return (self._minor_tick_length if self._minor_tick_length is not None
                else 0.5 * self._tick_length)

    def _resolved_minor_color(self) -> Any:
        return (self._minor_tick_color if self._minor_tick_color is not None
                else self._resolved_tick_color())

    def _resolved_minor_lw(self) -> float:
        return (self._minor_tick_lw if self._minor_tick_lw is not None
                else self._resolved_tick_lw())

    def _resolved_label_color(self) -> Any:
        return (self._label_color if self._label_color is not None
                else self._color)

    def _resolve_convert(self) -> tuple[Callable[[float], float] | None, Any]:
        """Return ``(fn, unit_str)`` for the configured ``convert=``,
        or ``(None, None)`` when no conversion is active. Resolution
        is deferred until label-render time so the user can swap
        ``convert=`` via :meth:`set_labels` without rebuilding the
        ruler from scratch."""
        return _normalize_convert(self._convert, self._convert_unit)

    def _resolved_endcap_color(self) -> Any:
        if self._endcap_color is not None:
            return self._endcap_color
        return self._resolved_tick_color()

    def _resolved_endcap_lw(self) -> float:
        if self._endcap_lw is not None:
            return self._endcap_lw
        return self._resolved_tick_lw()

    def _draw_endcap(self, ax: Any, xs: np.ndarray, ys: np.ndarray,
                     t_end: float, sides: Sequence[int],
                     tick_len_disp: float, trans: Any, inv: Any) -> None:
        """Render one endcap at fractional position ``t_end`` (0.0 or
        1.0). Style is governed by ``self._endcap_style``."""
        # Anchor + tangent at the endpoint (in display coords).
        cx, cy = self._interpolate_along_line(xs, ys, t_end)
        tx, ty = self._local_tangent(xs, ys, t_end)
        cx_d, cy_d = trans.transform((cx, cy))
        tx_d_pt, ty_d_pt = trans.transform((cx + tx, cy + ty))
        d_tx, d_ty = tx_d_pt - cx_d, ty_d_pt - cy_d
        n = np.hypot(d_tx, d_ty)
        if n == 0:
            return
        d_tx, d_ty = d_tx / n, d_ty / n
        # Outward along the tangent: +tangent at the end (t=1), or
        # −tangent at the start (t=0) — points AWAY from the midpoint.
        out_sign = +1.0 if t_end >= 0.5 else -1.0
        out_tx = out_sign * d_tx
        out_ty = out_sign * d_ty
        # Perpendicular in display coords (for the 'tick' endcap shape).
        p_x, p_y = -d_ty, d_tx

        color = self._resolved_endcap_color()
        lw = self._resolved_endcap_lw()

        if self._endcap_style == 'tick':
            # Longer tick: tick_length × endcap_length_scale, in both
            # perpendicular directions selected by ``sides``.
            cap_len_disp = tick_len_disp * self._endcap_length_scale
            for sign in sides:
                end_disp = (cx_d + sign * p_x * cap_len_disp,
                            cy_d + sign * p_y * cap_len_disp)
                end_data = inv.transform(end_disp)
                cap = Line2D(
                    [cx, end_data[0]], [cy, end_data[1]],
                    color=color, lw=lw,
                    ls=self._resolved_tick_ls(),
                    alpha=self._alpha,
                    zorder=self._zorder,
                    path_effects=self._path_effects,
                    clip_on=self._clip_on)
                ax.add_line(cap)
                self._endcap_artists.append(cap)
        elif self._endcap_style == 'arrow':
            # Arrowhead at the endpoint, pointing outward along the
            # tangent. Use a short FancyArrowPatch from a fiducial
            # point JUST INSIDE the line (a few display pixels) to
            # the endpoint, with the arrowhead at the endpoint.
            from matplotlib.patches import FancyArrowPatch
            # Base offset: a small step along −outward (i.e. inward).
            base_step_disp = 1.0       # 1 pixel — minimal stem so the
                                       # head reads as the cap, not as
                                       # a duplicate line segment.
            base_disp = (cx_d - out_tx * base_step_disp,
                         cy_d - out_ty * base_step_disp)
            tip_disp = (cx_d, cy_d)
            base_data = inv.transform(base_disp)
            tip_data = inv.transform(tip_disp)
            arrow = FancyArrowPatch(
                posA=(base_data[0], base_data[1]),
                posB=(tip_data[0], tip_data[1]),
                arrowstyle='-|>',
                mutation_scale=self._endcap_size,
                color=color, lw=lw,
                shrinkA=0, shrinkB=0,
                zorder=self._zorder,
                path_effects=self._path_effects,
                clip_on=self._clip_on,
            )
            ax.add_patch(arrow)
            self._endcap_artists.append(arrow)

    # ----- axes wiring -------------------------------------------------

    def add_to(self, ax: Any) -> Ruler:
        """Add the ruler (line + ticks + labels) to *ax* in data
        coordinates. Returns ``self`` for chaining.

        Once attached, the ruler connects to the axes'
        ``resize_event`` / ``xlim_changed`` / ``ylim_changed``
        callbacks and automatically rebuilds its tick orientations
        and label positions when the figure resizes or the data
        limits change — so tick lengths stay visually consistent
        during pan / zoom / window-resize. Call :meth:`remove`
        to detach (and disconnect the callbacks).
        """
        # Re-adding: cleanly detach the previous artists + reflow callbacks
        # first, so a second add_to doesn't duplicate artists or leak
        # callbacks (the stored cids would otherwise be overwritten).
        if self._host_axes is not None:
            self._disconnect_reflow()
            self._strip_artists()
        if self._ax_ref is None:
            self._ax_ref = ax
            if self._pixscale_asec is None:
                self._pixscale_asec = _pixscale_asec_from_ax(ax)
        self._host_axes = ax
        self._build_artists(ax)
        self._connect_reflow(ax)
        return self

    def _build_artists(self, ax: Any) -> None:
        """Construct the line + ticks + labels + endcaps + title
        artists at the current axes layout. Public entry is
        :meth:`add_to`; :meth:`_on_layout_change` calls back into
        this helper after stripping the previous artists during
        live reflow."""
        # When the ruler was built via :meth:`from_axes_fraction`,
        # refresh ``self._xy1`` / ``self._xy2`` from the cached
        # axes-fraction positions using the axes' CURRENT transform —
        # so the ruler stays glued to the same visual position
        # regardless of pan / zoom / resize.
        if self._uses_axes_frac:
            self._refresh_endpoints_from_axes_frac(ax)

        # 1) Main line
        xs, ys = self._line_samples()
        self._line_artist = Line2D(
            xs, ys,
            color=self._color, lw=self._lw, ls=self._ls,
            alpha=self._alpha, zorder=self._zorder,
            path_effects=self._path_effects,
            clip_on=self._clip_on)
        ax.add_line(self._line_artist)

        # 2) Ticks + labels
        self._tick_artists = []
        self._minor_tick_artists = []
        self._label_artists = []
        if (self._tick_side == 'none' and not self._labels
                and self._resolved_minor_side() == 'none'):
            return

        positions = self._major_tick_positions()
        if not positions:
            return

        # Resolve the convert= kwarg (None unless caller asked for
        # physical-unit labels). Requires a known pixel scale because
        # the conversion operates on arcsec values; raise early so the
        # caller gets a clear message instead of nonsense pixel-as-arcsec
        # labels.
        convert_fn, convert_unit_str = self._resolve_convert()
        if convert_fn is not None and self._pixscale_asec is None:
            raise ValueError(
                "convert= requires a known pixel scale: pass an "
                "axes with a WCS, or set pixscale_asec= explicitly.")

        # Determine which sides get tick marks
        sides: list[int] = []
        if self._tick_side in ('both', 'left'):
            sides.append(+1)
        if self._tick_side in ('both', 'right'):
            sides.append(-1)

        # Label side resolves via explicit kwarg (auto = match tick
        # side: left/right/both → +1, right → -1).
        label_sign = _resolve_label_side_sign(
            self._label_side, self._tick_side)

        # Convert tick / label spacing from points to display pixels
        # (transData operates in display = pixel coords).
        dpi = ax.figure.dpi
        tick_len_disp = self._tick_length * dpi / 72.0
        label_off_disp = (tick_len_disp +
                          self._label_offset * dpi / 72.0)

        trans = ax.transData
        inv = trans.inverted()

        # Resolve a single ``'auto'`` unit for the WHOLE ruler from its largest
        # tick magnitude, so every tick shares ONE unit. Per-tick auto can't:
        # the value-0 tick has no scale (→ would fall to the smallest unit),
        # and a ruler straddling a boundary would mix units (e.g. "40″, 1′" or
        # "500 mas, 1″"). A measurement scale wants uniform units. Only for
        # angular labels (known pixel scale, no convert / label_fmt).
        label_unit = self._label_unit
        if (label_unit == 'auto' and self._pixscale_asec is not None
                and self._label_fmt is None):
            max_mag = max((abs(d) for d, _ in positions), default=0.0)
            label_unit = _auto_angle_unit(max_mag)

        # Minor ticks (never labeled). Drawn before the majors so that where
        # the two are close the major tick and its label sit on top.
        minor_sides: list[int] = []
        minor_side = self._resolved_minor_side()
        if minor_side in ('both', 'left'):
            minor_sides.append(+1)
        if minor_side in ('both', 'right'):
            minor_sides.append(-1)
        if minor_sides:
            minor_len_disp = self._resolved_minor_length() * dpi / 72.0
            for t in self._minor_tick_positions():
                # An endcap visually replaces any tick at that endpoint.
                if self._endcap_style != 'none' and (
                        (t <= 1e-9 and self._endcaps in ('both', 'start'))
                        or (t >= 1.0 - 1e-9
                            and self._endcaps in ('both', 'end'))):
                    continue
                cx, cy = self._interpolate_along_line(xs, ys, t)
                tx, ty = self._local_tangent(xs, ys, t)
                cx_d, cy_d = trans.transform((cx, cy))
                tx_d, ty_d = trans.transform((cx + tx, cy + ty))
                d_tx, d_ty = tx_d - cx_d, ty_d - cy_d
                n_t = np.hypot(d_tx, d_ty)
                if n_t == 0:
                    continue
                d_tx, d_ty = d_tx / n_t, d_ty / n_t
                p_x, p_y = -d_ty, d_tx
                for sign in minor_sides:
                    end_disp = (cx_d + sign * p_x * minor_len_disp,
                                cy_d + sign * p_y * minor_len_disp)
                    end_data = inv.transform(end_disp)
                    mtick = Line2D(
                        [cx, end_data[0]], [cy, end_data[1]],
                        color=self._resolved_minor_color(),
                        lw=self._resolved_minor_lw(),
                        ls=self._resolved_tick_ls(),
                        alpha=self._alpha,
                        zorder=self._zorder,
                        path_effects=self._path_effects,
                        clip_on=self._clip_on)
                    ax.add_line(mtick)
                    self._minor_tick_artists.append(mtick)

        for d_user, t in positions:
            cx, cy = self._interpolate_along_line(xs, ys, t)
            tx, ty = self._local_tangent(xs, ys, t)

            # Display-coord tangent (so the perpendicular looks balanced
            # on both sides regardless of data aspect).
            cx_d, cy_d = trans.transform((cx, cy))
            tx_d, ty_d = trans.transform((cx + tx, cy + ty))
            d_tx, d_ty = tx_d - cx_d, ty_d - cy_d
            n = np.hypot(d_tx, d_ty)
            if n == 0:
                continue
            d_tx, d_ty = d_tx / n, d_ty / n
            # Perpendicular in display coords (90° CCW of tangent).
            p_x, p_y = -d_ty, d_tx

            # Skip the regular tick at an endpoint where an endcap
            # will be drawn (the endcap visually replaces the tick).
            is_start = t <= 1e-9
            is_end = t >= 1.0 - 1e-9
            cap_here = (self._endcap_style != 'none'
                        and ((is_start
                              and self._endcaps in ('both', 'start'))
                             or (is_end
                                 and self._endcaps in ('both', 'end'))))

            # Tick marks (suppressed at endpoints with an endcap)
            if not cap_here:
                for sign in sides:
                    end_disp = (cx_d + sign * p_x * tick_len_disp,
                                cy_d + sign * p_y * tick_len_disp)
                    end_data = inv.transform(end_disp)
                    tick = Line2D(
                        [cx, end_data[0]], [cy, end_data[1]],
                        color=self._resolved_tick_color(),
                        lw=self._resolved_tick_lw(),
                        ls=self._resolved_tick_ls(),
                        alpha=self._alpha,
                        zorder=self._zorder,
                        path_effects=self._path_effects,
                        clip_on=self._clip_on)
                    ax.add_line(tick)
                    self._tick_artists.append(tick)

            # Label
            if self._labels:
                lab_disp = (cx_d + label_sign * p_x * label_off_disp,
                            cy_d + label_sign * p_y * label_off_disp)
                lab_x, lab_y = inv.transform(lab_disp)

                if self._label_fmt is not None:
                    text = self._label_fmt(d_user, self._label_unit)
                elif convert_fn is not None:
                    text = _format_converted_label(
                        convert_fn(d_user), convert_unit_str,
                        fmt=self._fmt)
                elif self._pixscale_asec is None:
                    # Non-WCS axes: d_user is already in pixels.
                    text = _format_angle_label(
                        d_user, unit='pix', fmt=self._fmt)
                elif label_unit == 'pix':
                    # WCS axes, pinned 'pix': d_user is in arcsec, so
                    # convert to pixels for the 'px' label (the formatter
                    # doesn't know the pixel scale).
                    text = _format_angle_label(
                        d_user / self._pixscale_asec, unit='pix',
                        fmt=self._fmt)
                else:
                    text = _format_angle_label(
                        d_user, unit=label_unit, fmt=self._fmt)

                tangent_angle_deg = float(
                    np.degrees(np.arctan2(d_ty, d_tx)))
                rotation = (
                    _resolve_rotation_deg(
                        self._label_rotation, tangent_angle_deg)
                    + self._label_rotation_add)
                rotation = _normalize_readable_angle(rotation)
                ha, va = _resolve_text_anchor(
                    rotation, label_sign, p_x, p_y)

                kwargs: dict[str, Any] = dict(
                    ha=ha, va=va,
                    color=self._resolved_label_color(),
                    alpha=self._alpha,
                    zorder=self._zorder,
                    rotation=rotation,
                    rotation_mode='anchor',
                    clip_on=self._clip_on)
                if self._label_fontsize is not None:
                    kwargs['fontsize'] = self._label_fontsize
                # The same legibility stroke that backs the line/ticks also
                # backs the tick labels — text is what most needs it on a busy
                # image (the whole point of the cartographic outline stroke).
                if self._path_effects is not None:
                    kwargs['path_effects'] = self._path_effects
                label = ax.text(lab_x, lab_y, text, **kwargs)
                self._label_artists.append(label)

        # 3) Endcaps (drawn AT the endpoints in place of the regular
        # tick when ``endcap_style != 'none'``).
        self._endcap_artists = []
        if self._endcap_style != 'none' and self._endcaps != 'none':
            ends_to_cap: list[float] = []
            if self._endcaps in ('both', 'start'):
                ends_to_cap.append(0.0)
            if self._endcaps in ('both', 'end'):
                ends_to_cap.append(1.0)
            for t_end in ends_to_cap:
                self._draw_endcap(ax, xs, ys, t_end, sides,
                                   tick_len_disp, trans, inv)

        # 4) Title (rendered at the midpoint, on the opposite side
        # from labels by default).
        self._title_artist = None
        if self._title:
            mid_x, mid_y = self._interpolate_along_line(xs, ys, 0.5)
            mtx, mty = self._local_tangent(xs, ys, 0.5)
            mx_d, my_d = trans.transform((mid_x, mid_y))
            mtx_d, mty_d = trans.transform((mid_x + mtx, mid_y + mty))
            m_dx, m_dy = mtx_d - mx_d, mty_d - my_d
            n = np.hypot(m_dx, m_dy)
            if n > 0:
                m_dx, m_dy = m_dx / n, m_dy / n
                # Perpendicular in display coords.
                m_px, m_py = -m_dy, m_dx
                title_sign = _resolve_title_side_sign(
                    self._title_side, label_sign)
                # Title offset semantic depends on the title_beyond_labels
                # flag:
                # * False (default): "gap beyond tick tip" — same as
                #   label_offset's semantic, suitable for the typical
                #   opposite-side title placement.
                # * True + title on the same side as labels: auto-
                #   compute clearance so the title sits PAST the
                #   labels' bounding boxes (the matplotlib twin-axis
                #   look). ``title_offset`` is reinterpreted as the
                #   *additional* gap beyond the labels.
                if (self._title_beyond_labels
                        and self._label_artists
                        and label_sign == title_sign):
                    title_off_disp = self._compute_title_offset_beyond_labels(
                        ax, mx_d, my_d, m_px, m_py, title_sign, dpi)
                else:
                    title_off_disp = (tick_len_disp
                                       + self._title_offset * dpi / 72.0)
                t_disp = (mx_d + title_sign * m_px * title_off_disp,
                          my_d + title_sign * m_py * title_off_disp)
                t_x, t_y = inv.transform(t_disp)
                tangent_angle_deg = float(
                    np.degrees(np.arctan2(m_dy, m_dx)))
                rotation = _resolve_rotation_deg(
                    self._title_rotation, tangent_angle_deg)
                rotation = _normalize_readable_angle(rotation)
                ha, va = _resolve_text_anchor(
                    rotation, title_sign, m_px, m_py)
                tkw: dict[str, Any] = dict(
                    ha=ha, va=va,
                    color=(self._title_color
                            if self._title_color is not None
                            else self._color),
                    alpha=self._alpha,
                    zorder=self._zorder,
                    rotation=rotation,
                    rotation_mode='anchor',
                    clip_on=self._clip_on)
                if self._title_fontsize is not None:
                    tkw['fontsize'] = self._title_fontsize
                if self._path_effects is not None:
                    tkw['path_effects'] = self._path_effects
                self._title_artist = ax.text(t_x, t_y, self._title, **tkw)

    # ----- live reflow on resize / pan / zoom --------------------------

    def _connect_reflow(self, ax: Any) -> None:
        """Connect the matplotlib lifecycle callbacks that re-layout
        the ruler when the figure resizes or the axes' data limits
        change. Called from :meth:`add_to`; the connection ids are
        stored so :meth:`remove` can disconnect cleanly."""
        fig = ax.figure
        # Use lambdas to swallow the event payload — the callback
        # logic doesn't need it (it just re-runs the layout pass).
        cid_resize = fig.canvas.mpl_connect(
            'resize_event', lambda _e: self._on_layout_change())
        cid_xlim = ax.callbacks.connect(
            'xlim_changed', lambda _a: self._on_layout_change())
        cid_ylim = ax.callbacks.connect(
            'ylim_changed', lambda _a: self._on_layout_change())
        self._reflow_cid_resize = cid_resize
        self._reflow_cid_xlim = cid_xlim
        self._reflow_cid_ylim = cid_ylim
        self._reflow_fig = fig

    def _disconnect_reflow(self) -> None:
        """Tear down callbacks set up by :meth:`_connect_reflow`."""
        if self._reflow_fig is not None:
            try:
                self._reflow_fig.canvas.mpl_disconnect(
                    self._reflow_cid_resize)
            except Exception:
                pass
        if self._host_axes is not None:
            try:
                self._host_axes.callbacks.disconnect(self._reflow_cid_xlim)
            except Exception:
                pass
            try:
                self._host_axes.callbacks.disconnect(self._reflow_cid_ylim)
            except Exception:
                pass
        self._reflow_cid_resize = None
        self._reflow_cid_xlim = None
        self._reflow_cid_ylim = None
        self._reflow_fig = None

    def _on_layout_change(self) -> None:
        """Rebuild artists in place when the layout has changed.

        Strips the current artists and re-runs :meth:`_build_artists`
        with the host axes' current transform. A reentrancy guard
        prevents the redraw triggered by the rebuild from re-entering
        this callback in a loop."""
        if self._host_axes is None:
            return
        if self._relayout_in_progress:
            return
        self._relayout_in_progress = True
        try:
            self._strip_artists()
            self._build_artists(self._host_axes)
        finally:
            self._relayout_in_progress = False

    def _strip_artists(self) -> None:
        """Detach all per-layout artists from the axes (line, ticks,
        labels, endcaps, title) and reset the corresponding state.
        Does NOT touch the reflow callbacks or ``_host_axes``."""
        for artist_list in (self._tick_artists, self._minor_tick_artists,
                             self._label_artists, self._endcap_artists):
            for a in artist_list:
                if getattr(a, 'axes', None) is not None:
                    a.remove()
        if (self._title_artist is not None
                and getattr(self._title_artist, 'axes', None) is not None):
            self._title_artist.remove()
        if (self._line_artist is not None
                and getattr(self._line_artist, 'axes', None) is not None):
            self._line_artist.remove()
        self._line_artist = None
        self._tick_artists = []
        self._minor_tick_artists = []
        self._label_artists = []
        self._endcap_artists = []
        self._title_artist = None

    def remove(self) -> None:
        """Remove the ruler from its axes and disconnect the
        live-reflow callbacks."""
        self._disconnect_reflow()
        self._strip_artists()
        self._host_axes = None

    # ----- component setters -------------------------------------------

    def set_line(self, *, color: Any = None, lw: float | None = None,
                 ls: str | None = None, alpha: float | None = None,
                 zorder: float | None = None, path_effects: Any = None,
                 stroke_color: Any = '__unset__',
                 stroke_lw: float | None = None) -> Ruler:
        """Update main-line styling. Applies in place to the attached
        artist if already added; otherwise stores for the next
        :meth:`add_to`. Pass ``stroke_color=None`` to disable the
        stroke explicitly. Returns ``self``."""
        if color is not None:
            self._color = color
        if lw is not None:
            self._lw = float(lw)
        if ls is not None:
            self._ls = ls
        if alpha is not None:
            self._alpha = float(alpha)
        if zorder is not None:
            self._zorder = float(zorder)
        if path_effects is not None:
            self._path_effects = path_effects
        elif stroke_color != '__unset__':
            # Convenience shortcut: rebuild path_effects from stroke
            # kwargs. Use the current stroke width if the user didn't
            # supply a new one along with the color change.
            sw = stroke_lw if stroke_lw is not None else self._stroke_lw
            if stroke_lw is not None:
                self._stroke_lw = float(stroke_lw)
            self._path_effects = _stroke_path_effects(stroke_color, sw)
        if self._line_artist is not None:
            self._line_artist.set_color(self._color)
            self._line_artist.set_linewidth(self._lw)
            self._line_artist.set_linestyle(self._ls)
            self._line_artist.set_alpha(self._alpha)
            self._line_artist.set_zorder(self._zorder)
            # Re-apply path_effects whenever the user set them via either
            # ``path_effects=`` or the ``stroke_color`` / ``stroke_lw``
            # shortcut. matplotlib's ``set_path_effects(None)`` clears
            # effects, so passing ``self._path_effects`` (which may be
            # ``None``) is the right "disable" behavior too.
            if path_effects is not None or stroke_color != '__unset__':
                self._line_artist.set_path_effects(self._path_effects or [])
        return self

    def set_ticks(self, *, color: Any = None, lw: float | None = None,
                  ls: str | None = None, length: float | None = None,
                  side: str | None = None) -> Ruler:
        """Update tick styling. ``color`` / ``lw`` / ``ls`` apply in
        place; ``length`` / ``side`` only take effect on the next
        :meth:`add_to` (orientation is computed at attach time).
        Returns ``self``."""
        if color is not None:
            self._tick_color = color
        if lw is not None:
            self._tick_lw = float(lw)
        if ls is not None:
            self._tick_ls = ls
        if length is not None:
            self._tick_length = float(length)
        if side is not None:
            if side not in _VALID_TICK_SIDES:
                raise ValueError(
                    f"tick_side must be one of {_VALID_TICK_SIDES!r}, "
                    f"got {side!r}")
            self._tick_side = side
        for tick in self._tick_artists:
            tick.set_color(self._resolved_tick_color())
            tick.set_linewidth(self._resolved_tick_lw())
            tick.set_linestyle(self._resolved_tick_ls())
        # Minors inherit the majors' color/lw unless pinned via minor_tick_*,
        # so a restyle has to reach them too.
        for mtick in self._minor_tick_artists:
            mtick.set_color(self._resolved_minor_color())
            mtick.set_linewidth(self._resolved_minor_lw())
            mtick.set_linestyle(self._resolved_tick_ls())
        return self

    def set_labels(self, *, color: Any = None,
                   fontsize: float | str | None = None,
                   unit: str | None = None,
                   label_fmt: Callable[..., str] | None = None,
                   fmt: str | None = None,
                   offset: float | None = None, show: bool | None = None,
                   side: str | None = None,
                   rotation: str | float | None = None,
                   rotation_add: float | None = None,
                   convert: Any = None,
                   convert_unit: str | None = None) -> Ruler:
        """Update label styling. ``color`` / ``fontsize`` apply in
        place to attached artists; everything else only takes effect
        on the next :meth:`add_to`. Returns ``self``.

        Parameter names mirror the constructor (``label_fmt`` for a
        custom callable formatter, ``fmt`` for a printf string,
        ``side`` for ``label_side=``, ``rotation`` for
        ``label_rotation=``, ``rotation_add`` for
        ``label_rotation_add=``).
        """
        if color is not None:
            self._label_color = color
        if fontsize is not None:
            self._label_fontsize = fontsize
        if unit is not None:
            if unit == 'μas':
                unit = 'uas'
            if unit not in _VALID_UNITS:
                raise ValueError(
                    f"label_unit must be one of {_VALID_UNITS!r}, "
                    f"got {unit!r}")
            self._label_unit = unit
        if label_fmt is not None:
            self._label_fmt = label_fmt
        if fmt is not None:
            self._fmt = fmt
        if offset is not None:
            self._label_offset = float(offset)
        if show is not None:
            self._labels = bool(show)
        if side is not None:
            if side not in ('auto', 'left', 'right'):
                raise ValueError(
                    f"label_side must be 'auto', 'left', or "
                    f"'right'; got {side!r}")
            self._label_side = side
        if rotation is not None:
            self._label_rotation = rotation
        if rotation_add is not None:
            self._label_rotation_add = float(rotation_add)
        if convert is not None:
            self._convert = convert
        if convert_unit is not None:
            self._convert_unit = convert_unit
        for lab in self._label_artists:
            lab.set_color(self._resolved_label_color())
            if fontsize is not None:
                lab.set_fontsize(fontsize)
        return self

    def set_title(self, title: str | None = None, *,
                  fontsize: float | str | None = None, color: Any = None,
                  offset: float | None = None,
                  rotation: str | float | None = None,
                  side: str | None = None) -> Ruler:
        """Update the ruler title. Text + color + fontsize apply in
        place when a title is already rendered; layout-affecting
        kwargs (``offset`` / ``rotation`` / ``side``) take effect
        on the next :meth:`add_to`. Returns ``self``."""
        if title is not None:
            self._title = title
        if fontsize is not None:
            self._title_fontsize = fontsize
        if color is not None:
            self._title_color = color
        if offset is not None:
            self._title_offset = float(offset)
        if rotation is not None:
            self._title_rotation = rotation
        if side is not None:
            if side not in ('auto', 'left', 'right'):
                raise ValueError(
                    f"title_side must be 'auto', 'left', or "
                    f"'right'; got {side!r}")
            self._title_side = side
        # Apply in-place updates to the existing title artist (if any).
        if self._title_artist is not None:
            if title is not None:
                self._title_artist.set_text(self._title)
            if color is not None:
                self._title_artist.set_color(color)
            if fontsize is not None:
                self._title_artist.set_fontsize(fontsize)
        return self

    # ----- repr --------------------------------------------------------

    def __repr__(self) -> str:
        mode = 'geodesic' if self._geodesic else 'straight'
        d = self.angular_distance_asec()
        if d is None:
            dlabel = 'unknown distance'
        else:
            dlabel = _format_angle_label(d, unit=self._label_unit)
        return (f"<Ruler xy1={self._xy1} xy2={self._xy2} "
                f"{mode}, {dlabel}>")
