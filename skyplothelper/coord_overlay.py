"""Coordinate-system overlays with full tick + label support.

Architecture inspired by the Kapteyn Astronomical Institute's
``kapteyn.wcsgrat`` module (3-clause BSD); skyplothelper's
implementation is independent and adapted to our use cases.

The overlay is built in stages, referenced by number throughout this
module: (1) gridlines, (2) tick discovery on bbox edges and arbitrary
frame curves, (3) tick-mark rendering, (4) tick-label rendering, and
(5) the same machinery applied to PCO / conic / AIT / MOL projection
boundary ticks.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator, Sequence
from typing import Any, Callable

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord
from matplotlib import rcParams

from ._stroke import _stroke_path_effects
from ._text_layout import _resolve_text_anchor
from .overlays.planes import _find_world_polyline_splits
from .wcs_frame import (
    _get_wcs_center_lat,
    _get_wcs_center_lon,
    _get_wcs_frame_name,
)

# Map the short names used by ``_get_wcs_frame_name`` and the
# user-facing API to the astropy frame names ``SkyCoord.transform_to``
# understands.
_FRAME_ALIASES = {
    'icrs': 'icrs',
    'fk5': 'fk5',
    'fk4': 'fk4',
    'galactic': 'galactic',
    'supergalactic': 'supergalactic',
    'ecliptic': 'geocentrictrueecliptic',
    'geocentrictrueecliptic': 'geocentrictrueecliptic',
    'barycentrictrueecliptic': 'barycentrictrueecliptic',
}

# Astropy component attribute names per frame (lon, lat).
_FRAME_COMPONENTS = {
    'icrs': ('ra', 'dec'),
    'fk5': ('ra', 'dec'),
    'fk4': ('ra', 'dec'),
    'galactic': ('l', 'b'),
    'supergalactic': ('sgl', 'sgb'),
    'geocentrictrueecliptic': ('lon', 'lat'),
    'barycentrictrueecliptic': ('lon', 'lat'),
    'heliocentrictrueecliptic': ('lon', 'lat'),
}


def _resolve_frame(name: str) -> str:
    """Resolve a frame name/alias to its astropy frame name.

    Delegates to the single canonical table in :mod:`skyplothelper.core.coords`
    rather than keeping a second one. The local copy had drifted: it knew only
    the long spellings, so ``convert_frame('gal')`` worked while
    ``CoordinateOverlay(frame='gal')`` silently passed ``'gal'`` straight to
    astropy — and ``image_to_healpix`` inherited the same gap through this
    function.
    """
    from .core.coords import _resolve_frame as _canonical
    return _canonical(name)


def _frame_components(name: str) -> tuple[str, str]:
    return _FRAME_COMPONENTS.get(name.lower(), ('ra', 'dec'))


def _field_world_extent(
    ax: Any, overlay_frame: str | None = None,
) -> tuple[float, float, float, float] | None:
    """Visible field extent ``(lon_lo, lon_hi, lat_lo, lat_hi)`` in host-frame
    degrees, with longitude unwrapped about the field center (so a field
    straddling lon=0, e.g. 359..2, gives a contiguous range). Returns ``None``
    if the view projects to no finite world coords. Assumes a small,
    pole-free field — true for the zoomed frames this serves.

    When ``overlay_frame`` is given and differs from the host axes frame, the
    sampled corners are transformed into that frame first, so the returned
    extent is in the overlay frame's coords (e.g. a galactic extent over an
    equatorial field). The longitude unwrap then references a sampled point
    rather than the host's center (there is no host center in the overlay
    frame).

    Call after ``ax.figure.canvas.draw()`` so the axes limits are valid.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    gx, gy = np.meshgrid(np.linspace(x0, x1, 5), np.linspace(y0, y1, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lon, lat = ax.wcs.pixel_to_world_values(gx.ravel(), gy.ravel())
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    good = np.isfinite(lon) & np.isfinite(lat)
    if not good.any():
        return None
    lon, lat = lon[good], lat[good]
    if overlay_frame is not None:
        host = _resolve_frame(_get_wcs_frame_name(ax))
        dst = _resolve_frame(overlay_frame)
        if host != dst:
            sc = SkyCoord(lon * u.deg, lat * u.deg, frame=host)
            tr = sc.transform_to(dst)
            a, b = _frame_components(dst)
            lon = np.asarray(getattr(tr, a).deg, dtype=float)
            lat = np.asarray(getattr(tr, b).deg, dtype=float)
            # No host center exists in the overlay frame — unwrap about a
            # sampled point so a field straddling lon=0 stays contiguous.
            ref = float(lon[0])
            lon_rel = ((lon - ref + 180.0) % 360.0) - 180.0
            return (ref + float(lon_rel.min()), ref + float(lon_rel.max()),
                    float(lat.min()), float(lat.max()))
    center_lon = float(_get_wcs_center_lon(ax))
    lon_rel = ((lon - center_lon + 180.0) % 360.0) - 180.0
    return (center_lon + float(lon_rel.min()),
            center_lon + float(lon_rel.max()),
            float(lat.min()), float(lat.max()))


def _field_graticule_vals(
    ax: Any, nbins: int = 5, overlay_frame: str | None = None,
) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None]:
    """Nice ``(lon_vals, lat_vals)`` spanning the axes' visible field extent.

    The all-sky default graticule (``arange(0, 360, 30)`` /
    ``arange(-75, 76, 15)``) contains no values inside a small zoomed field
    (TAN / SIN / ZEA with a few-degree ``fov_deg``), so in-frame ticks find
    no gridline intersections. This runs a
    :class:`~matplotlib.ticker.MaxNLocator` nice-number step over the field's
    world extent. Returns ``(None, None)`` if the field projects to no finite
    world coords.

    ``overlay_frame`` is forwarded to :func:`_field_world_extent` so the nice
    values land in the overlay frame's coords for a cross-frame overlay (e.g.
    galactic graticule over an equatorial field).

    Call after ``ax.figure.canvas.draw()`` so the axes limits are valid.
    """
    from matplotlib.ticker import MaxNLocator

    ext = _field_world_extent(ax, overlay_frame=overlay_frame)
    if ext is None:
        return None, None
    lon_lo, lon_hi, lat_lo, lat_hi = ext

    def _nice(lo: float, hi: float) -> npt.NDArray[np.float64]:
        # Steps restricted to {1, 2, 5}×10ᵏ (no 2.5) so the values never carry
        # a half-integer that the separation-based label precision would round
        # away — see _label_decimals.
        vals = np.asarray(
            MaxNLocator(nbins=nbins, steps=[1, 2, 5, 10]).tick_values(lo, hi),
            dtype=np.float64)
        # MaxNLocator pads to round numbers outside [lo, hi]; keep only the
        # values that actually fall within the visible field.
        return vals[(vals >= lo) & (vals <= hi)]

    lon_vals = _nice(lon_lo, lon_hi) % 360.0
    lat_vals = _nice(lat_lo, lat_hi)
    return (lon_vals if lon_vals.size else None,
            lat_vals if lat_vals.size else None)


def _insert_pixel_jump_breaks(xy_pix: npt.NDArray[np.float64], ax: Any,
                              jump_fraction: float = 0.35,
                              ) -> npt.NDArray[np.float64]:
    """Insert NaN rows in ``xy_pix`` wherever consecutive samples
    jump by more than ``jump_fraction`` × max(bbox width, height).

    Cross-frame ``from_const_lat`` / ``from_const_lon`` curves trace
    closed loops on the sphere; projected into pixel space those
    loops typically wrap once across the host's antimeridian,
    producing a long straight segment that connects opposite edges
    of the projection. That wrap segment is not a real arc — meridian
    intersections against it are spurious. Breaking the polyline
    with NaN rows lets the downstream intersection step (which
    splits on NaN) avoid the wrap line entirely.

    Returns ``xy_pix`` unchanged when no jumps are detected.
    """
    if len(xy_pix) < 2:
        return xy_pix
    d = np.linalg.norm(np.diff(xy_pix, axis=0), axis=1)
    bb = ax.bbox
    threshold = jump_fraction * max(bb.width, bb.height)
    jump_after = np.where(d > threshold)[0]
    if len(jump_after) == 0:
        return xy_pix
    # Insert a NaN row AFTER each jump-from index. Walk back-to-front
    # so indices stay valid.
    out = xy_pix.copy()
    nan_row = np.array([[np.nan, np.nan]])
    for idx in jump_after[::-1]:
        out = np.vstack([out[:idx + 1], nan_row, out[idx + 1:]])
    return out


def _lonlat_to_axes_frame(ax: Any, lons: npt.ArrayLike, lats: npt.ArrayLike,
                          frame: str | None,
                          ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Transform (lons, lats) from ``frame`` to the host axes' frame.

    No-op when ``frame`` is ``None`` or resolves to the host frame.
    Mirrors :meth:`CoordinateOverlay._to_axes_frame` but works at
    module level for the :class:`_FrameCurve` factories, which don't
    have a bound overlay instance.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if frame is None:
        return lons, lats
    src = _resolve_frame(frame)
    dst = _resolve_frame(_get_wcs_frame_name(ax))
    if src == dst:
        return lons, lats
    sc = SkyCoord(lons * u.deg, lats * u.deg, frame=src)
    transformed = sc.transform_to(dst)
    a, b = _frame_components(dst)
    return getattr(transformed, a).deg, getattr(transformed, b).deg


_HOUR_SEP = {'super': r'$^\mathregular{h}$', 'unicode': '\u02b0',
             'plain': 'h', 'letter': 'h', 'none': ''}


def _resolve_hour_sep(sep: str) -> str:
    """Resolve a longitude-hour separator spec to its suffix string.

    Accepts a preset name ('super' = mathtext superscript, the
    default; 'unicode'; 'plain'/'letter'; 'none') or any literal
    suffix string.
    """
    return _HOUR_SEP.get(sep, sep)


def _min_spacing(vals: npt.ArrayLike) -> float | None:
    """Smallest positive gap between adjacent (sorted, unique) values, used as
    the label-precision hint. Robust to a longitude wrap (the one large gap a
    field straddling lon=0 introduces is ignored since we take the minimum)."""
    v = np.unique(np.asarray(vals, dtype=float))
    if v.size < 2:
        return None
    diffs = np.diff(v)
    diffs = diffs[diffs > 1e-9]
    return float(diffs.min()) if diffs.size else None


def _label_decimals(step: float | None) -> int:
    """Number of decimal places needed to render labels spaced ``step`` apart
    distinctly. ``None`` (no spacing hint) → 1, preserving the historical
    ``.1f`` default. A ~degree-or-coarser step → integer labels; a fine field
    step (e.g. 0.5°/0.033ʰ) → enough decimals so adjacent labels differ."""
    if not step or step <= 0 or not np.isfinite(step):
        return 1
    return int(min(4, max(0, np.ceil(-np.log10(step)))))


def _format_tick_label(value: float, kind: str, frame: str,
                       fmt: str | Callable[[float], str] = 'auto',
                       sep: str = 'super', step: float | None = None) -> str:
    """Format a single coordinate value into a tick-label string.

    Latitudes always render in degrees. Longitudes render in hours
    for ICRS / FK5 / FK4 (RA is conventionally in hours) and in
    degrees for galactic / ecliptic / supergalactic.

    Parameters
    ----------
    value : float
        Coordinate value in degrees.
    kind : {'lon', 'lat'}
    frame : str
        Overlay frame name (used to choose hour vs degree for lon
        when ``fmt='auto'``).
    fmt : 'auto' | 'hour' | 'deg' | callable
        Format. A callable receives ``value`` and returns a string.
    step : float, optional
        Spacing (in degrees) between adjacent tick values. When given, the
        label precision adapts so closely-spaced field-scale ticks stay
        distinct (a 30° all-sky graticule still renders as integers; a 0.5°
        field renders one decimal, a 0.033ʰ RA field renders two). ``None``
        keeps the historical single-decimal default.

    Returns
    -------
    label : str
    """
    if callable(fmt):
        return fmt(value)
    # Integer-shortcut ("30°" not "30.0°") only on the legacy step-less path
    # or a truly coarse (nd==0) graticule. On the field path (explicit step,
    # nd>0) labels render at uniform decimals so neighbours stay consistent
    # ("9.97ʰ", "10.00ʰ", "10.03ʰ" rather than "9.97ʰ", "10ʰ", "10.03ʰ").
    if kind == 'lat':
        nd = _label_decimals(step)
        d = float(value)
        if nd == 0 or (step is None and abs(d - round(d)) < 1e-3):
            i = int(round(d))
            return "0°" if i == 0 else f"{i:+d}°"
        return f"{d:+.{nd}f}°"
    # kind == 'lon'
    if fmt == 'auto':
        fmt = 'hour' if frame.lower() in ('icrs', 'fk5', 'fk4') else 'deg'
    if fmt == 'hour':
        suffix = _resolve_hour_sep(sep)
        nd = _label_decimals(None if step is None else step / 15.0)
        h = (float(value) / 15.0) % 24
        if nd == 0 or (step is None and abs(h - round(h)) < 1e-3):
            return f"{int(round(h)) % 24:d}{suffix}"
        return f"{h:.{nd}f}{suffix}"
    if fmt == 'deg':
        # Absolute longitude convention is [0, 360) (galactic, ecliptic,
        # supergalactic, ...) — do NOT fold to a signed [-180, 180) range,
        # which renders l=182° as -178°. A center-relative signed display,
        # if ever wanted, is the caller's job via _unwrap_lon_about_center.
        nd = _label_decimals(step)
        d = float(value) % 360.0
        if nd == 0 or (step is None and abs(d - round(d)) < 1e-3):
            return f"{int(round(d)) % 360:d}°"
        return f"{d:.{nd}f}°"
    if fmt == 'west':
        # West-longitude labeling (route b): ``value`` is east-longitude; show
        # its west-longitude equivalent with a W/E hemisphere suffix (east −71°
        # → "71°W"; 0° / 180° get no letter). Mirrors the native-tick path
        # (``_install_west_longitude_labels``) so overlay (in-frame) labels on
        # the non-FITS all-sky planet frames match the FITS ones.
        nd = _label_decimals(step)
        w = -(((float(value) + 180.0) % 360.0) - 180.0)  # west-positive (-180,180]
        if w <= -180.0 + 1e-9:
            w = 180.0
        mag = abs(w)
        if nd == 0 or (step is None and abs(mag - round(mag)) < 1e-3):
            body = f"{int(round(mag)):d}°"
        else:
            body = f"{mag:.{nd}f}°"
        if mag < 1e-6 or abs(mag - 180.0) < 1e-6:
            return body
        return body + ('W' if w > 0 else 'E')
    raise ValueError(f"Unknown lon-tick format {fmt!r}")


class _GridLine:
    """Single constant-coord curve (one meridian or one parallel).

    Stored in the overlay frame's native lon/lat so the curve is
    independent of the parent axes; transformation to axes pixels
    happens at plot time.
    """

    __slots__ = ('kind', 'value', 'lons', 'lats', '_axes_xy')

    def __init__(self, kind: str, value: float, lons: npt.ArrayLike,
                 lats: npt.ArrayLike) -> None:
        self.kind = kind        # 'lon' (meridian) or 'lat' (parallel)
        self.value = float(value)
        self.lons = np.asarray(lons, dtype=float)
        self.lats = np.asarray(lats, dtype=float)
        # Cached (axes_lon, axes_lat) from the overlay→axes frame transform —
        # filled once and shared by plotting and tick discovery.
        self._axes_xy: tuple[np.ndarray, np.ndarray] | None = None


class _FrameCurve:
    """A curve along which gridline ticks are anchored.

    Polyline in axes-display (pixel) coordinates. The default frame
    curves of a :class:`CoordinateOverlay` are the four edges of the
    parent axes bounding box. Callers can replace them with arbitrary
    curves (e.g. the elliptical boundary of an AIT projection, the
    boundary outline of a PCO/conic projection, or a constant-lat
    parallel for the axis-curve tick mode) via
    :meth:`CoordinateOverlay.set_frame_curves`.

    Parameters
    ----------
    xy_pix : (N, 2) array_like
        Polyline vertices in display (pixel) coordinates.
    name : str, optional
        Human-readable identifier (e.g. ``'left'``, ``'boundary'``,
        ``'lat=0'``).
    closed : bool
        If True, the polyline is treated as a closed loop; an extra
        closing point is appended if the polyline is not already
        closed.
    kind : {None, 'lon', 'lat'}
        Gridline-kind filter for tick discovery. ``None`` (default)
        means *both* meridians (lon gridlines) and parallels (lat
        gridlines) intersect this curve. ``'lon'`` restricts to
        meridians (typical for a constant-lat parallel that hosts
        lon labels — the axis-curve mode).
        ``'lat'`` restricts to parallels (typical for a constant-lon
        meridian hosting lat labels).
    axis_curve : bool
        ``True`` marks this curve as an interior reference curve
        (constant-lat parallel or constant-lon meridian — the
        ``'axis'`` / ``'lat=N'`` / ``'lon=N'`` family from
        :func:`add_overlay_ticks`). ``False`` (default) is for
        boundary / spine curves. The flag enables a one-tick-per-
        gridline filter in :meth:`CoordinateOverlay.discover_ticks`:
        on axis curves a single overlay gridline can produce
        multiple intersections (sphere-loop wraps, dense-sample
        endpoint duplicates) but the user typically wants ONE label
        per gridline; boundary curves keep their multiple crossings
        (each spine intersection is a legitimate edge tick).
    """

    __slots__ = ('xy_pix', 'name', 'closed', 'kind', 'axis_curve')

    def __init__(self, xy_pix: npt.ArrayLike, name: str | None = None,
                 closed: bool = False, kind: str | None = None,
                 axis_curve: bool = False) -> None:
        if kind not in (None, 'lon', 'lat'):
            raise ValueError(
                f"kind must be None, 'lon', or 'lat', got {kind!r}")
        xy = np.asarray(xy_pix, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy_pix must have shape (N, 2)")
        if closed and len(xy) >= 1 and not np.allclose(xy[0], xy[-1]):
            xy = np.vstack([xy, xy[:1]])
        self.xy_pix = xy
        self.name = name
        self.closed = bool(closed)
        self.kind = kind
        self.axis_curve = bool(axis_curve)

    @classmethod
    def from_bbox_edge(cls, ax: Any, edge: str) -> _FrameCurve:
        """Build the bbox edge of *ax* as a 2-point pixel polyline.

        Parameters
        ----------
        ax : WCSAxes
        edge : {'left', 'right', 'bottom', 'top'}
        """
        x0, y0, x1, y1 = ax.bbox.extents
        edges = {
            'left':   np.array([[x0, y0], [x0, y1]]),
            'right':  np.array([[x1, y0], [x1, y1]]),
            'bottom': np.array([[x0, y0], [x1, y0]]),
            'top':    np.array([[x0, y1], [x1, y1]]),
        }
        if edge not in edges:
            raise ValueError(
                f"edge must be one of {sorted(edges)}, got {edge!r}")
        return cls(edges[edge], name=edge)

    # Axis-aligned outward direction for bbox edges; everything else
    # falls back to the centroid heuristic in ``outward_at``.
    _BBOX_OUTWARD = {
        'left':   (-1., 0.),
        'right':  (1., 0.),
        'bottom': (0., -1.),
        'top':    (0., 1.),
    }

    def outward_at(self, xy_pix: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Unit vector pointing 'outward' at *xy_pix*.

        For bbox edges this is the axis-aligned direction away from
        the parent axes interior. For closed frame curves (e.g. a
        projection boundary) it is the unit vector from the curve
        centroid through ``xy_pix``. Tick-mark rendering uses this to
        decide which of the two perpendicular tick orientations points
        away from the frame.

        Parameters
        ----------
        xy_pix : (2,) array_like
            Position in display coordinates.
        """
        if self.name in self._BBOX_OUTWARD:
            return np.array(self._BBOX_OUTWARD[self.name])
        # Drop the duplicated closing vertex for closed curves so it
        # doesn't bias the centroid toward the first/last point.
        verts = self.xy_pix[:-1] if self.closed else self.xy_pix
        centroid = verts.mean(axis=0)
        v = np.asarray(xy_pix, dtype=float) - centroid
        n = np.linalg.norm(v)
        if n == 0:
            return np.array([1., 0.])
        return v / n

    @classmethod
    def from_world_polyline(cls, ax: Any, lonlat: npt.ArrayLike,
                            name: str | None = None, closed: bool = False,
                            kind: str | None = None) -> _FrameCurve:
        """Build a frame curve from a world-coordinate polyline.

        The (lon, lat) vertices are transformed to display pixels via
        ``ax.get_transform('world')``. :func:`add_overlay_ticks` uses
        this to register each non-rectangular projection's geometric
        boundary as the frame edge.

        Parameters
        ----------
        ax : WCSAxes
        lonlat : (N, 2) array_like
            Vertices in world coords (degrees).
        name : str, optional
        closed : bool
        kind : {None, 'lon', 'lat'}
            Gridline-kind filter. See :class:`_FrameCurve`.
        """
        lonlat = np.asarray(lonlat, dtype=float)
        xy_pix = ax.get_transform('world').transform(lonlat)
        return cls(xy_pix, name=name, closed=closed, kind=kind)

    @classmethod
    def from_const_lat(cls, ax: Any, lat: float,
                       lon_range: tuple[float, float] | None = None,
                       n: int = 200, name: str | None = None,
                       kind: str | None = 'lon',
                       frame: str | None = None) -> _FrameCurve:
        """Build a constant-lat parallel as a frame curve in pixel space.

        Axis-curve mode: meridians intersect this curve to
        produce lon ticks positioned along the chosen lat parallel
        (e.g. the equator on a Robinson, bowing with the projection).

        Parameters
        ----------
        ax : WCSAxes
        lat : float
            Latitude in degrees.
        lon_range : (lo, hi), optional
            Lon range to sample. If ``None``, defaults to the visible
            lon range of *ax* trimmed by ``±0.01°`` so the sampled
            polyline doesn't span the wrap point (which would cause a
            jump-split — the same wrap-edge issue the boundary-tick
            polyline extension already bridges, so trimming inside the
            wrap is the cleaner choice here). For cross-frame curves
            (``frame`` differs from the host), defaults to the full
            ``(0.01, 359.99)`` so the overlay-frame parallel is
            sampled in its entirety before transformation.
        n : int
            Samples along the parallel.
        name : str, optional
            Defaults to ``f'lat={lat}'``.
        kind : {None, 'lon', 'lat'}
            Default ``'lon'`` (meridians intersect this curve to
            place lon ticks).
        frame : str, optional
            Coordinate frame the ``lat`` value lives in. ``None``
            (default) means the host axes' frame — current behavior.
            A different frame causes the sampled (lon, lat) polyline
            to be transformed via :class:`~astropy.coordinates.SkyCoord`
            to the host frame before pixel-projection, so the
            resulting pixel-space curve traces the *overlay*'s
            constant-lat parallel through the host projection.
        """
        if lon_range is None:
            if (frame is not None
                    and _resolve_frame(frame)
                    != _resolve_frame(_get_wcs_frame_name(ax))):
                lon_range = (0.01, 359.99)
            else:
                from .wcs_frame import _get_wcs_center_lon
                c = _get_wcs_center_lon(ax)
                lon_range = (c - 180. + 0.01, c + 180. - 0.01)
        lons = np.linspace(float(lon_range[0]), float(lon_range[1]), int(n))
        lats = np.full_like(lons, float(lat))
        lons, lats = _lonlat_to_axes_frame(ax, lons, lats, frame)
        xy_pix = ax.get_transform('world').transform(
            np.column_stack([lons, lats]))
        if frame is not None and _resolve_frame(frame) != _resolve_frame(
                _get_wcs_frame_name(ax)):
            xy_pix = _insert_pixel_jump_breaks(xy_pix, ax)
        return cls(xy_pix, name=name or f'lat={lat}', kind=kind,
                   axis_curve=True)

    @classmethod
    def from_const_lon(cls, ax: Any, lon: float,
                       lat_range: tuple[float, float] = (-89.9999, 89.9999),
                       n: int = 200, name: str | None = None,
                       kind: str | None = 'lat',
                       frame: str | None = None) -> _FrameCurve:
        """Build a constant-lon meridian as a frame curve in pixel space.

        Axis-curve mode: parallels intersect this curve to
        produce lat ticks positioned along the chosen lon meridian
        (e.g. the central meridian on a Robinson).

        Parameters
        ----------
        ax : WCSAxes
        lon : float
            Longitude in degrees.
        lat_range : (lo, hi), optional
            Latitude range to sample. Default just shy of the poles
            to keep SkyCoord transformations well-defined for
            cross-frame overlays.
        n : int
            Samples along the meridian.
        name : str, optional
            Defaults to ``f'lon={lon}'``.
        kind : {None, 'lon', 'lat'}
            Default ``'lat'`` (parallels intersect this curve to
            place lat ticks).
        frame : str, optional
            Coordinate frame the ``lon`` value lives in. ``None``
            (default) means the host axes' frame — current behavior.
            A different frame causes the sampled (lon, lat) polyline
            to be transformed via :class:`~astropy.coordinates.SkyCoord`
            to the host frame before pixel-projection, so the
            resulting pixel-space curve traces the *overlay*'s
            constant-lon meridian through the host projection.
        """
        lats = np.linspace(float(lat_range[0]), float(lat_range[1]), int(n))
        lons = np.full_like(lats, float(lon))
        lons, lats = _lonlat_to_axes_frame(ax, lons, lats, frame)
        xy_pix = ax.get_transform('world').transform(
            np.column_stack([lons, lats]))
        if frame is not None and _resolve_frame(frame) != _resolve_frame(
                _get_wcs_frame_name(ax)):
            xy_pix = _insert_pixel_jump_breaks(xy_pix, ax)
        return cls(xy_pix, name=name or f'lon={lon}', kind=kind,
                   axis_curve=True)


class _GridTick:
    """A single tick: where a gridline crosses a frame curve.

    Holds the display-pixel position, the gridline tangent angle at
    that point (degrees, measured in display space), and references
    back to the parent gridline and frame curve. Both tick-mark
    rendering and tick-label placement consume these.

    Attributes
    ----------
    xy_pix : (2,) ndarray
        Position in axes-display coordinates.
    tangent_deg : float
        Direction of the gridline at the crossing, in degrees, in
        display space (atan2 of dy/dx along the gridline segment
        producing the intersection).
    gridline : _GridLine
        The originating gridline.
    frame_curve : _FrameCurve
        The frame curve this tick sits on.
    value : float
        Convenience copy of ``gridline.value`` (the gridline's
        constant lon or lat).
    kind : {'lon', 'lat'}
        Convenience copy of ``gridline.kind``.
    """

    __slots__ = ('xy_pix', 'tangent_deg', 'gridline', 'frame_curve',
                 'value', 'kind')

    def __init__(self, xy_pix: npt.ArrayLike, tangent_deg: float,
                 gridline: _GridLine, frame_curve: _FrameCurve) -> None:
        self.xy_pix = np.asarray(xy_pix, dtype=float)
        self.tangent_deg = float(tangent_deg)
        self.gridline = gridline
        self.frame_curve = frame_curve
        self.value = gridline.value
        self.kind = gridline.kind


# Floor extension (display pixels) added at each end of every
# gridline polyline before boundary intersection. The actual
# extension auto-scales upward to ``1.5 × median segment length``
# when that exceeds the floor, so a sparsely-sampled gridline still
# bridges its own one-step wrap-side gap (e.g. a Robinson parallel
# stops one ~3 px sample step short of the right edge). Small enough
# that gridlines legitimately ending inside the frame don't sprout
# spurious extra intersections.
_ENDPOINT_PAD_PIX = 2.0


def _extrapolate_polyline_endpoints(xy: npt.ArrayLike, min_extension: float,
                                    ) -> npt.NDArray[np.float64]:
    """Extend a 2D polyline at both ends along its endpoint tangents.

    The extension magnitude is ``max(min_extension, 1.5 × median
    segment length)`` so the added margin scales with the polyline's
    own sampling density. Used by
    :meth:`CoordinateOverlay._get_pixel_segments` so the intersection
    algorithm catches gridlines whose true mathematical endpoint lies
    ON the boundary curve but whose densified sampling lands a
    fraction of a pixel — or one sample step — short.

    Parameters
    ----------
    xy : (M, 2) array_like
    min_extension : float
        Floor on the per-end extension in display pixels.

    Returns
    -------
    out : (M', 2) ndarray
        The polyline with one extra vertex prepended (along the
        first segment's reverse tangent) and one appended (along the
        last segment's forward tangent). ``M' = M + 2`` unless an
        endpoint segment is degenerate, in which case that end is
        left unchanged.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.shape[0] < 2:
        return xy
    seg_lens = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    seg_lens = seg_lens[seg_lens > 0]
    if len(seg_lens) == 0:
        return xy
    extension = max(float(min_extension), 1.5 * float(np.median(seg_lens)))
    d0 = xy[1] - xy[0]
    n0 = np.linalg.norm(d0)
    dn = xy[-1] - xy[-2]
    nn = np.linalg.norm(dn)
    prefix = [xy[0] - (d0 / n0) * extension] if n0 > 0 else []
    suffix = [xy[-1] + (dn / nn) * extension] if nn > 0 else []
    if not prefix and not suffix:
        return xy
    return np.vstack(prefix + [xy] + suffix)


def _intersect_polylines(poly_a: npt.ArrayLike, poly_b: npt.ArrayLike,
                         dedup_tol: float = 0.5,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find every intersection of two 2D polylines.

    Both inputs must live in the same coordinate system and contain
    no NaN values (split polylines into clean sub-arrays first).

    Parameters
    ----------
    poly_a, poly_b : (N, 2) array_like
        Polylines with at least two points each.
    dedup_tol : float, optional
        Merge tolerance in poly_a's coordinate units. Intersections
        within this distance of an already-kept intersection are
        dropped. Default 0.5 (half a display pixel). Set to 0 to
        disable deduplication.

        Necessary because :func:`_extrapolate_polyline_endpoints`
        intentionally pads each polyline by ~one sample step at
        each end to bridge sub-pixel sampling gaps; when a polyline
        endpoint *already* sits on a frame curve (e.g. a CAR
        parallel starting at lon=0 on the left bbox spine), the
        prepended extension and the original first segment both
        cross at the same boundary point.

    Returns
    -------
    points : (K, 2) ndarray
        Intersection points.
    a_seg : (K,) ndarray of int
        Index of the originating segment in poly_a (between
        ``poly_a[i]`` and ``poly_a[i+1]``). Lets callers compute a
        local tangent at the intersection from the segment direction.
    a_t : (K,) ndarray of float in [0, 1]
        Parametric position along the originating segment in poly_a.
    """
    a = np.asarray(poly_a, dtype=float)
    b = np.asarray(poly_b, dtype=float)
    empty = (np.empty((0, 2)), np.empty(0, dtype=int), np.empty(0))
    if a.shape[0] < 2 or b.shape[0] < 2:
        return empty

    a1 = a[:-1]
    a2 = a[1:]
    b1 = b[:-1]
    b2 = b[1:]

    r = a2 - a1                               # (Na-1, 2)
    s = b2 - b1                               # (Nb-1, 2)

    rxs = (r[:, None, 0] * s[None, :, 1]
           - r[:, None, 1] * s[None, :, 0])    # (Na-1, Nb-1)
    diff = b1[None, :, :] - a1[:, None, :]     # (Na-1, Nb-1, 2)
    t_num = diff[..., 0] * s[None, :, 1] - diff[..., 1] * s[None, :, 0]
    u_num = diff[..., 0] * r[:, None, 1] - diff[..., 1] * r[:, None, 0]

    with np.errstate(divide='ignore', invalid='ignore'):
        t = t_num / rxs
        u = u_num / rxs

    valid = ((rxs != 0)
             & np.isfinite(t) & np.isfinite(u)
             & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1))
    if not valid.any():
        return empty

    flat = np.argwhere(valid)
    a_idx = flat[:, 0]
    b_idx = flat[:, 1]
    t_hit = t[a_idx, b_idx]
    points = a1[a_idx] + t_hit[:, None] * r[a_idx]

    if dedup_tol > 0 and len(points) > 1:
        # Walk in poly_a-traversal order (a_idx, then a_t) and drop
        # any intersection within dedup_tol of an already-kept one.
        order = np.lexsort((t_hit, a_idx))
        points = points[order]
        a_idx = a_idx[order]
        t_hit = t_hit[order]
        keep = [0]
        for i in range(1, len(points)):
            if np.linalg.norm(points[i] - points[keep[-1]]) > dedup_tol:
                keep.append(i)
        keep_arr = np.array(keep)
        points = points[keep_arr]
        a_idx = a_idx[keep_arr]
        t_hit = t_hit[keep_arr]

    return points, a_idx, t_hit


class CoordinateOverlay:
    """A coordinate-system grid overlaid on a WCSAxes.

    Draws meridians and parallels from an *overlay frame* (e.g.
    ``'galactic'``) on top of an axes whose own WCS may use a
    different frame (e.g. ICRS), with optional tick marks and tick
    labels along the gridlines.

    Parameters
    ----------
    ax : WCSAxes
        The parent axes. Must have a ``.wcs`` attribute or the
        ``_sph_frame`` / ``_sph_center_lon`` non-FITS hints used
        elsewhere in skyplothelper.
    frame : str
        Astropy frame name for the overlay grid: ``'galactic'``,
        ``'icrs'``, ``'fk5'``, ``'fk4'``, ``'ecliptic'`` (alias for
        ``'geocentrictrueecliptic'``), ``'supergalactic'``, ...
    lon_vals : array_like, optional
        Longitudes of meridians to draw, in degrees. Default
        ``np.arange(0, 360, 30)``.
    lat_vals : array_like, optional
        Latitudes of parallels to draw, in degrees. Default
        ``np.arange(-75, 76, 15)``.
    n_samples : int, optional
        Number of points along each gridline (default 200). Higher
        values smooth out projection-induced curvature.

    Attributes
    ----------
    lon_gridlines, lat_gridlines : list of _GridLine
        The constructed meridians / parallels (overlay-frame samples).
    lon_artists, lat_artists : list of list of Line2D
        Plotted segments per gridline, populated by :meth:`plot`.
        Each gridline can split into multiple segments when its
        path crosses a projection discontinuity (antimeridian wrap
        or a multi-face seam).

    Examples
    --------
    >>> import skyplothelper as sph
    >>> ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)
    >>> sph.CoordinateOverlay(ax, frame='galactic').plot(color='orange')
    """

    def __init__(self, ax: Any, frame: str = 'galactic',
                 lon_vals: npt.ArrayLike | None = None,
                 lat_vals: npt.ArrayLike | None = None,
                 n_samples: int = 200) -> None:
        self.ax = ax
        self.frame = _resolve_frame(frame)
        self._target_frame = _resolve_frame(_get_wcs_frame_name(ax))
        self._center_lon = _get_wcs_center_lon(ax)

        # On a zoomed (non-all-sky) field, the all-sky default graticule
        # (30°/15°) lands no meridian/parallel inside a few-degree view, so
        # the overlay would come back with zero ticks / a near-empty grid.
        # Derive field-scale nice values from the visible extent expressed in
        # THIS overlay's frame (so a galactic overlay over an equatorial
        # field gets galactic-spaced values, not the host frame's). All-sky /
        # globe frames keep the 30°/15° look; explicit lon_vals / lat_vals
        # always win. Centralized here so add_overlay_ticks AND
        # add_coord_overlay (which builds the overlay directly) both adapt.
        auto_lon = auto_lat = None
        if ((lon_vals is None or lat_vals is None)
                and not getattr(ax, '_sph_is_allsky', True)
                and not getattr(ax, '_sph_is_globe', False)):
            auto_lon, auto_lat = _field_graticule_vals(
                ax, overlay_frame=self.frame)
        self.lon_vals = (np.asarray(lon_vals, dtype=float)
                         if lon_vals is not None
                         else auto_lon if auto_lon is not None
                         else np.arange(0., 360., 30.))
        self.lat_vals = (np.asarray(lat_vals, dtype=float)
                         if lat_vals is not None
                         else auto_lat if auto_lat is not None
                         else np.arange(-75., 76., 15.))
        self.n_samples = int(n_samples)

        self.lon_gridlines: list[_GridLine] = []
        self.lat_gridlines: list[_GridLine] = []
        self.lon_artists: list[list[Any]] = []
        self.lat_artists: list[list[Any]] = []
        # built lazily by discover_ticks
        self.frame_curves: list[_FrameCurve] | None = None
        self.gridticks: list[_GridTick] = []  # populated by discover_ticks
        # Tracks whether discover_ticks has run at least once. Lets
        # ``render_ticks`` / ``render_labels`` auto-discover only when
        # discovery genuinely hasn't happened, instead of treating an
        # empty gridticks list (e.g. after a clip filter removed every
        # candidate) as the trigger — which would re-discover and
        # silently undo the filter.
        self._ticks_discovered = False
        self.tick_artists: list[Any] = []      # populated by render_ticks
        self.frame_curve_artists: list[Any] = []  # populated by draw_frame_curves
        self.label_artists: list[Any] = []     # populated by render_labels
        # How far past xy_pix the last-rendered tick extends *outward*.
        # Used as the default outward offset for label placement so
        # labels clear the tick endpoint (and only count outward extent
        # — inward-pointing ticks need no extra pad).
        self._tick_outward_extent: float | None = None

        self._build_gridlines()

    def _build_gridlines(self) -> None:
        # Stop just shy of the poles (sub-arcsecond) so SkyCoord
        # transformations stay well-defined for every meridian's first
        # and last sample. Tight enough that the endpoint gap is
        # sub-pixel even on a zenithal projection zoomed near a pole.
        lat_lo, lat_hi = -89.9999, 89.9999
        lon_lo, lon_hi = 0., 360.
        # On a zoomed field, sample meridians/parallels over the field extent
        # (padded) rather than the whole sky: the full-sky default at
        # n_samples puts only ~1-2 samples inside a few-degree field, too
        # sparse for reliable gridline×axis-curve crossings. For a CROSS-frame
        # overlay this matters most for the parallels — sampled over the full
        # 0–360° overlay-frame lon, only a handful of points land in the
        # window and the wrap/clip then drops the curve entirely (lat ticks
        # vanish). Compute the extent in the OVERLAY frame (via the
        # overlay_frame= helper) so both meridians and parallels densify over
        # the visible arc. All-sky / globe frames keep full-sky.
        if (not getattr(self.ax, '_sph_is_allsky', True)
                and not getattr(self.ax, '_sph_is_globe', False)):
            ext = _field_world_extent(self.ax, overlay_frame=self.frame)
            if ext is not None:
                e_lon_lo, e_lon_hi, e_lat_lo, e_lat_hi = ext
                lat_pad = 0.25 * (e_lat_hi - e_lat_lo)
                lon_pad = 0.25 * (e_lon_hi - e_lon_lo)
                lat_lo = max(-89.9999, e_lat_lo - lat_pad)
                lat_hi = min(89.9999, e_lat_hi + lat_pad)
                lon_lo, lon_hi = e_lon_lo - lon_pad, e_lon_hi + lon_pad
        meridian_lats = np.linspace(lat_lo, lat_hi, self.n_samples)
        for lon in self.lon_vals:
            lons = np.full_like(meridian_lats, float(lon))
            self.lon_gridlines.append(
                _GridLine('lon', lon, lons, meridian_lats))

        parallel_lons = np.linspace(lon_lo, lon_hi, self.n_samples)
        for lat in self.lat_vals:
            lats = np.full_like(parallel_lons, float(lat))
            self.lat_gridlines.append(
                _GridLine('lat', lat, parallel_lons, lats))

    def _to_axes_frame(self, lons: npt.ArrayLike, lats: npt.ArrayLike,
                       ) -> tuple[Any, Any]:
        """Transform (lons, lats) from overlay frame to axes frame."""
        if self.frame == self._target_frame:
            return np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)
        sc = SkyCoord(np.asarray(lons) * u.deg, np.asarray(lats) * u.deg,
                      frame=self.frame)
        transformed = sc.transform_to(self._target_frame)
        a, b = _frame_components(self._target_frame)
        return getattr(transformed, a).deg, getattr(transformed, b).deg

    def _wrap_axes_lon(self, lon_deg: npt.ArrayLike) -> npt.NDArray[np.float64]:
        c = self._center_lon
        return ((np.asarray(lon_deg, dtype=float) - c + 180.) % 360.) + c - 180.

    def plot(self, lon_style: dict[str, Any] | None = None,
             lat_style: dict[str, Any] | None = None,
             **kwargs: Any) -> CoordinateOverlay:
        """Render the overlay gridlines on the parent axes.

        Parameters
        ----------
        lon_style, lat_style : dict, optional
            Per-axis style overrides (color, lw, ls, alpha, zorder,
            anything ``ax.plot`` accepts).
        **kwargs : dict
            Default style applied to both meridians and parallels.
            Per-axis ``lon_style`` / ``lat_style`` take precedence
            where the keys overlap.

        Returns
        -------
        self : CoordinateOverlay
            Returned for chaining.
        """
        base = dict(color='gray', lw=0.5, alpha=0.3, ls=':', zorder=2)
        base.update(kwargs)
        lon_kw = dict(base)
        if lon_style:
            lon_kw.update(lon_style)
        lat_kw = dict(base)
        if lat_style:
            lat_kw.update(lat_style)

        transform = self.ax.get_transform('world')

        self.lon_artists = [self._plot_one(gl, transform, lon_kw)
                            for gl in self.lon_gridlines]
        self.lat_artists = [self._plot_one(gl, transform, lat_kw)
                            for gl in self.lat_gridlines]
        return self

    def _fill_axes_frame_cache(self) -> None:
        """Transform every gridline to the axes frame in ONE SkyCoord call.

        The cross-frame ``transform_to`` has high fixed overhead, and its result
        is consumed twice (plotting + tick discovery), so transform all
        meridians+parallels together once and cache the split per gridline.
        Same-frame overlays skip the transform entirely.
        """
        pending = [gl for gl in (self.lon_gridlines + self.lat_gridlines)
                   if gl._axes_xy is None]
        if not pending:
            return
        if self.frame == self._target_frame:
            for gl in pending:
                gl._axes_xy = (gl.lons, gl.lats)
            return
        lengths = [len(gl.lons) for gl in pending]
        all_lon = np.concatenate([gl.lons for gl in pending])
        all_lat = np.concatenate([gl.lats for gl in pending])
        axes_lon, axes_lat = self._to_axes_frame(all_lon, all_lat)   # 1 transform
        idx = np.cumsum(lengths)[:-1]
        for gl, lo, la in zip(pending, np.split(axes_lon, idx),
                              np.split(axes_lat, idx)):
            gl._axes_xy = (lo, la)

    def _axes_frame_segments(self, gl: _GridLine,
                             ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield post-split (axes_lon, axes_lat) segments for a gridline.

        The gridline is transformed to the axes frame, wrapped around
        the axes center longitude, and broken wherever a projection
        discontinuity (antimeridian or multi-face seam) is detected.
        Each yielded segment has at least two points.
        """
        if gl._axes_xy is None:
            self._fill_axes_frame_cache()
        axes_lon, axes_lat = gl._axes_xy               # type: ignore[misc]
        axes_lon = self._wrap_axes_lon(axes_lon)
        splits = _find_world_polyline_splits(self.ax, axes_lon, axes_lat)
        for seg_lon, seg_lat in zip(np.split(axes_lon, splits),
                                    np.split(axes_lat, splits)):
            if len(seg_lon) >= 2:
                yield seg_lon, seg_lat

    def _plot_one(self, gl: _GridLine, transform: Any,
                  style: dict[str, Any]) -> list[Any]:
        artists = []
        for seg_lon, seg_lat in self._axes_frame_segments(gl):
            ln, = self.ax.plot(seg_lon, seg_lat, transform=transform, **style)
            artists.append(ln)
        return artists

    # ===== Tick discovery =====

    def _get_pixel_segments(self, gl: _GridLine) -> list[npt.NDArray[np.float64]]:
        """Return display-space polyline segments for a gridline.

        Each segment is an (M, 2) array of (x_pix, y_pix). Multiple
        segments are returned whenever the gridline crosses a
        projection discontinuity. Each segment is extrapolated by a
        few display pixels at both ends so endpoints that fall a
        fraction of a pixel short of an enclosing boundary curve
        (e.g. meridians clamped at lat=±89.9999 on a Robinson, or
        parallels stopping just shy of an antimeridian wrap) cross
        the boundary in the segment-intersection step rather than
        being missed.
        """
        transform = self.ax.get_transform('world')
        out = []
        for seg_lon, seg_lat in self._axes_frame_segments(gl):
            xy = transform.transform(np.column_stack([seg_lon, seg_lat]))
            out.append(_extrapolate_polyline_endpoints(xy, _ENDPOINT_PAD_PIX))
        return out

    def _default_frame_curves(self) -> list[_FrameCurve]:
        return [_FrameCurve.from_bbox_edge(self.ax, e)
                for e in ('left', 'right', 'bottom', 'top')]

    def set_frame_curves(self, curves: Sequence[_FrameCurve]) -> CoordinateOverlay:
        """Replace the default bbox-edge frame curves with custom ones.

        The default frame is the four axes bbox edges. For projections
        whose natural boundary is a curve (e.g. AIT/MOL ellipses,
        PCO/conic outlines), pass one or more :class:`_FrameCurve`
        instances built via :meth:`_FrameCurve.from_world_polyline`.

        Parameters
        ----------
        curves : sequence of _FrameCurve

        Returns
        -------
        self : CoordinateOverlay
        """
        self.frame_curves = list(curves)
        return self

    def draw_frame_curves(self, color: str = 'k', lw: float = 1.0,
                          ls: str = '-', zorder: int = 2,
                          clip_on: bool = False,
                          **kwargs: Any) -> CoordinateOverlay:
        """Draw the registered frame curves as visible polylines.

        Mostly a convenience for visualizing custom frame curves
        (e.g. an inner-box reference or a projection boundary). Bbox-edge frame curves are usually already drawn
        by matplotlib's axes spines, so this rarely adds anything
        useful for the default case.

        Parameters
        ----------
        color, lw, ls, zorder, clip_on : matplotlib line styling.
        **kwargs : additional ``Line2D`` kwargs.

        Returns
        -------
        self : CoordinateOverlay
        """
        from matplotlib.lines import Line2D

        if self.frame_curves is None:
            self.frame_curves = self._default_frame_curves()

        # Convert frozen display pixels to axes data coords once, then
        # plot with ax.transData so rendering survives a savefig at a
        # different dpi than the construction dpi.
        inv = self.ax.transData.inverted()
        artists = []
        for fc in self.frame_curves:
            xy_data = inv.transform(fc.xy_pix)
            line = Line2D(xy_data[:, 0], xy_data[:, 1],
                          color=color, lw=lw, ls=ls, zorder=zorder,
                          clip_on=clip_on,
                          transform=self.ax.transData, **kwargs)
            self.ax.add_line(line)
            artists.append(line)
        self.frame_curve_artists = artists
        return self

    def _uniform_outward_for_axis_curves(self) -> dict[int, npt.NDArray[np.float64]]:
        """Per-axis-curve outward direction shared across all of its ticks.

        For axis curves (constant-lat parallels / constant-lon
        meridians used as interior reference curves), the centroid-
        based ``_FrameCurve.outward_at`` flips sign across the curve
        midpoint when the curve is roughly horizontal or vertical,
        producing the "labels above on one side, below on the other"
        artifact (e.g. lon labels on a Robinson equator). Resolving
        this at draw time by rotating the curve's average tangent
        90° clockwise gives a single curve-wide outward — lon labels
        sit uniformly *below* a roughly-horizontal parallel, lat
        labels sit uniformly *right of* a roughly-vertical meridian,
        matching standard astronomy-axis convention.

        Boundary curves keep their per-point centroid-based outward
        (each spine-edge intersection genuinely faces a different
        direction).

        Returns
        -------
        dict
            ``{id(frame_curve): outward_unit_vector}`` for every
            ``axis_curve=True`` frame curve currently in
            ``self.frame_curves``. Lookup fails for boundary curves —
            callers should fall back to ``fc.outward_at(xy_pix)``.
        """
        out = {}
        for fc in {tick.frame_curve for tick in self.gridticks}:
            if not getattr(fc, 'axis_curve', False):
                continue
            d = np.diff(fc.xy_pix, axis=0)
            d = d[np.isfinite(d).all(axis=1)]
            if len(d) == 0:
                continue
            avg_t = d.mean(axis=0)
            n = np.linalg.norm(avg_t)
            if n == 0:
                continue
            avg_t = avg_t / n
            out[id(fc)] = np.array([avg_t[1], -avg_t[0]])
        return out

    def render_ticks(self, length: float = 6., lw: float = 1.0,
                     color: str | None = None, direction: str = 'out',
                     zorder: int = 3, clip_on: bool = False,
                     **kwargs: Any) -> CoordinateOverlay:
        """Render tick marks at every discovered gridline×frame crossing.

        Each tick is a short line in the gridline's tangent direction
        at the crossing point, oriented so it extends past the frame
        curve. Length is in display pixels so the visual size stays
        constant regardless of axes aspect or zoom.

        :meth:`discover_ticks` is invoked automatically if it hasn't
        been already.

        Parameters
        ----------
        length : float
            Tick length in display pixels.
        lw : float
            Tick linewidth.
        color : str
            Tick color.
        direction : {'out', 'in', 'both'}
            ``'out'`` (default) extends the tick outward past the
            frame; ``'in'`` extends inward; ``'both'`` straddles the
            frame with total length ``2 * length``.
        zorder : int
        clip_on : bool
            Default ``False`` so ticks can extend past the axes bbox
            without being clipped (essential for the natural 'out'
            direction).
        **kwargs : additional ``Line2D`` kwargs.

        Returns
        -------
        self : CoordinateOverlay
        """
        from matplotlib.lines import Line2D

        if direction not in ('out', 'in', 'both'):
            raise ValueError(
                f"direction must be 'out', 'in', or 'both', "
                f"got {direction!r}")

        if color is None:
            color = rcParams['xtick.color']

        if not self._ticks_discovered:
            self.discover_ticks()

        # Tick endpoints are computed in display pixels at construction
        # dpi (gridline tangent direction × pixel length), then snapped
        # to axes data coords via transData.inverted() so the rendered
        # line survives a savefig at a different dpi.
        uniform_outward = self._uniform_outward_for_axis_curves()
        inv = self.ax.transData.inverted()
        artists = []
        for tick in self.gridticks:
            tan = np.array([np.cos(np.radians(tick.tangent_deg)),
                            np.sin(np.radians(tick.tangent_deg))])
            outward = uniform_outward.get(
                id(tick.frame_curve),
                tick.frame_curve.outward_at(tick.xy_pix))
            if np.dot(tan, outward) < 0:
                tan = -tan

            if direction == 'both':
                start_pix = tick.xy_pix - length * tan
                end_pix = tick.xy_pix + length * tan
            else:
                d = tan if direction == 'out' else -tan
                start_pix = tick.xy_pix
                end_pix = tick.xy_pix + length * d

            start_data = inv.transform(start_pix)
            end_data = inv.transform(end_pix)
            line = Line2D([start_data[0], end_data[0]],
                          [start_data[1], end_data[1]],
                          color=color, lw=lw, zorder=zorder,
                          clip_on=clip_on,
                          transform=self.ax.transData, **kwargs)
            # dynamic marker attrs so a post-build restyle (style_wcs_axes)
            # can recolor these overlay tick marks AND flip their direction:
            # the anchor (boundary intersection) and the OUTWARD endpoint
            # (always xy_pix + length·outward-tangent, independent of the
            # direction rendered here) are stored in dpi-stable data coords so
            # style_wcs_axes can re-derive an 'in' / 'out' / 'both' segment
            # without the overlay object. astropy can't draw inward ticks on a
            # curved spine, so these sph-drawn ticks are the only direction-
            # controllable ones on an elliptical all-sky frame.
            anchor_data = inv.transform(tick.xy_pix)
            out_data = inv.transform(tick.xy_pix + length * tan)
            line._sph_overlay_tick = True  # type: ignore[attr-defined]
            line._sph_overlay_kind = tick.kind  # type: ignore[attr-defined]
            line._sph_tick_anchor = (  # type: ignore[attr-defined]
                float(anchor_data[0]), float(anchor_data[1]))
            line._sph_tick_out_end = (  # type: ignore[attr-defined]
                float(out_data[0]), float(out_data[1]))
            self.ax.add_line(line)
            artists.append(line)

        self.tick_artists = artists
        # 'in' ticks contribute 0 outward extent; 'out' and 'both' both
        # extend `length` pixels past xy_pix in the outward direction.
        self._tick_outward_extent = 0.0 if direction == 'in' else float(length)
        return self

    def render_labels(self, fontsize: float = 10, color: str | None = None,
                      pad: float | None = None,
                      rotate: str | float | Callable[[_GridTick], float] = 'tangent',
                      fmt: str | Callable[[float], str] = 'auto',
                      sep: str = 'super',
                      mode: str = 'auto', zorder: int = 4,
                      clip_on: bool = False, ha: str = 'auto',
                      va: str = 'auto', **kwargs: Any) -> CoordinateOverlay:
        """Render formatted text labels at every discovered tick.

        Each label is placed past the tick endpoint along the
        gridline tangent (so the visual flow is gridline → tick →
        label), rotated to the tangent direction by default. Position
        is snapshotted in axes data coords so labels survive a
        savefig at a different dpi than the canvas dpi.

        :meth:`discover_ticks` is invoked automatically if it hasn't
        been already.

        Parameters
        ----------
        fontsize, color : matplotlib text styling.
        pad : float, optional
            Display-pixel distance from each tick's intersection
            point to its label position, measured along the outward
            gridline tangent. If ``None`` (default), the pad is
            ``tick_outward_extent + 5`` — enough buffer for the
            label's own bbox half-width at typical font sizes so the
            label edge clears both the tick endpoint and the frame
            curve. ``tick_outward_extent`` is the outward-pointing
            length from :meth:`render_ticks` — ``length`` for
            ``direction='out'`` or ``'both'``, ``0`` for
            ``direction='in'``, and treated as ``0`` if
            ``render_ticks`` has not been called.
        rotate : {'tangent', 'tangent_upright', 'horizontal', float, callable}
            ``'tangent'`` (default, aliased as ``'tangent_noflip'``) aligns
            labels with the gridline tangent *continuously* so
            spatially-adjacent labels never snap 180°, choosing a single
            branch per placement group (the labels of one kind sharing a
            frame edge) so the group reads upright for the current view —
            e.g. the parallels on a tilted globe, whose raw tangents all
            point "into" the page, are flipped upright as a set. Where a
            group genuinely sweeps through vertical (a globe's converging
            meridians, an all-sky limb) labels lean past ±90° rather than
            snapping ("follow the tangent", "stay upright", and "never
            flip" cannot all hold there — the default keeps continuity).
            ``'tangent_upright'`` instead clamps every label to
            ``(-90, 90]`` so each is guaranteed upright, at the cost of a
            180° snap where the tangent crosses ±90° partway along a curve.
            ``'tangent_perp'`` (alias ``'tangent_upright_perp'``) rotates
            *perpendicular* to the tangent (tangent + 90°), clamped upright —
            the gridline-relative spelling of :func:`apply_boundary_labels`'
            ``orient='parallel'``. Labels read across the gridline they sit on
            (upright on a steep parallel where plain ``'tangent'`` would lay
            them on their sides). The general form ``'tangent+N'`` /
            ``'tangent-N'`` (e.g. ``'tangent+90'``, ``'tangent-30'``) rotates
            by any offset off the local tangent, likewise clamped readable.
            ``'horizontal'`` forces 0°. A float sets a fixed rotation
            (upright and unflipped, but ignores the curve). A callable
            receives the :class:`_GridTick` and returns a float in
            degrees — e.g. ``lambda t: t.tangent_deg`` is the raw,
            uncorrected tangent (continuous but may render upside-down).
        fmt : 'auto' | 'hour' | 'deg' | callable
            Formatter for longitude labels. ``'auto'`` picks hours
            for equatorial frames, degrees otherwise. Latitudes
            always render as degrees. A callable receives the value
            in degrees and returns the label string.
        sep : str, optional
            Hour-suffix style for hour labels: ``'super'`` (mathtext
            superscript, the default — renders in any font),
            ``'unicode'`` (``ʰ``), ``'plain'`` / ``'letter'`` (``h``),
            ``'none'``, or any literal suffix. Ignored for degrees.
        mode : {'auto', 'complete'}
            ``'auto'`` (default) hides labels whose bbox overlaps an
            earlier label's bbox — first-wins in tick-discovery
            order. ``'complete'`` shows every candidate label
            regardless of overlap; useful as a debug / inspect view
            or when the caller plans to shrink ``fontsize`` after
            creation so overlaps resolve themselves.
        zorder : int
        clip_on : bool
            Default False so labels just past the frame edge stay
            visible.
        ha, va : str
            Horizontal / vertical text alignment. Default ``'auto'``
            both — picks the alignment so the bbox's near edge sits
            at the offset position (via
            :func:`~skyplothelper._text_layout._resolve_text_anchor`).
            Means the caller's ``pad`` is the actual visible gap
            between tick and label, regardless of rotation or font
            size. Pass ``'center'`` / ``'left'`` / etc. to override
            with matplotlib's standard anchoring instead.
        **kwargs : additional ``ax.text`` kwargs.

        Returns
        -------
        self : CoordinateOverlay
        """
        if color is None:
            color = rcParams['xtick.color']
        if mode not in ('auto', 'complete'):
            raise ValueError(
                f"mode must be 'auto' (filter overlapping labels) or "
                f"'complete' (show every candidate label), got {mode!r}")

        if not self._ticks_discovered:
            self.discover_ticks()

        if pad is None:
            outward_extent = (self._tick_outward_extent
                              if self._tick_outward_extent is not None else 0.0)
            # 5 px past the tick endpoint when ``ha=va='auto'`` —
            # near-edge anchoring means this is the actual visible
            # gap (vs. the legacy ``14 px`` which targeted bbox-
            # center anchoring and bled ~7-10 px into the label's
            # own half-bbox depending on rotation).
            pad = outward_extent + 5.0

        uniform_outward = self._uniform_outward_for_axis_curves()

        # Per-axis tick spacing → adaptive label precision (so a few-degree
        # field's closely-spaced labels stay distinct; the 30°/15° all-sky
        # graticule still resolves to integer labels).
        lon_step = _min_spacing(self.lon_vals)
        lat_step = _min_spacing(self.lat_vals)

        # For 'tangent_noflip', pre-compute the rotation per placement group.
        # The raw tangent already varies continuously along a placement curve
        # (spatially-adjacent labels have near-equal tangents), so the only
        # free choice is a single 0°/180° branch per group — applied uniformly
        # so no NEW flip is introduced between neighbors. Grouping is by
        # (placement curve, lon/lat): all the lon (or lat) labels sharing a
        # curve get ONE branch, chosen to keep that set upright on balance.
        # This flips a whole upside-down set upright (e.g. a tilted globe's
        # parallels, whose tangents all point "into" the page) while leaving a
        # set that genuinely sweeps through vertical (a globe's converging
        # meridians, an all-sky limb) to lean — never snapping mid-curve.
        # (Grouping by gridline instead would put each meridian in its own
        # group and clamp each independently — reintroducing the very flip the
        # preset exists to avoid.)
        noflip_rot: dict[int, float] = {}
        if rotate in ('tangent', 'tangent_noflip'):
            groups: dict[tuple[int, str], list[Any]] = {}
            for tick in self.gridticks:
                groups.setdefault((id(tick.frame_curve), tick.kind), []) \
                    .append(tick)
            for group in groups.values():
                raws = [((t.tangent_deg + 180.0) % 360.0) - 180.0
                        for t in group]
                # Flip the whole group 180° iff that leaves more labels
                # upright (|r| <= 90); ties keep the raw branch.
                upside = sum(abs(r) > 90.0 for r in raws)
                offset = 180.0 if upside > len(raws) - upside else 0.0
                for t, r in zip(group, raws):
                    noflip_rot[id(t)] = ((r + offset + 180.0) % 360.0) - 180.0

        # Interpretive tangent-relative rotation: 'tangent+90', 'tangent-30',
        # etc. rotate each label by that many degrees off the local gridline
        # tangent (then clamped readable-upright). The named 'tangent_perp' /
        # 'tangent_upright_perp' are the +90 special case. ``None`` unless a
        # tangent-offset spelling matches, in which case the per-tick branch
        # below uses it.
        tangent_offset: float | None = None
        if isinstance(rotate, str):
            if rotate in ('tangent_perp', 'tangent_upright_perp'):
                tangent_offset = 90.0
            else:
                _m = re.fullmatch(r'tangent\s*([+-]\s*\d+(?:\.\d+)?)',
                                  rotate.strip())
                if _m:
                    tangent_offset = float(_m.group(1).replace(' ', ''))

        inv = self.ax.transData.inverted()
        artists = []
        for tick in self.gridticks:
            tan = np.array([np.cos(np.radians(tick.tangent_deg)),
                            np.sin(np.radians(tick.tangent_deg))])
            outward = uniform_outward.get(
                id(tick.frame_curve),
                tick.frame_curve.outward_at(tick.xy_pix))
            if np.dot(tan, outward) < 0:
                tan = -tan

            pos_pix = tick.xy_pix + pad * tan
            pos_data = inv.transform(pos_pix)

            if rotate in ('tangent', 'tangent_noflip'):
                # Default. Continuous (no 180° snap between neighbors), with a
                # per-placement-group branch chosen to keep labels upright for
                # the current view (see noflip_rot above). Where a group
                # genuinely sweeps through vertical, labels still lean past ±90°
                # rather than flipping. 'tangent_noflip' is an explicit alias.
                r = noflip_rot[id(tick)]
            elif rotate == 'tangent_upright':
                # Clamp each label to (-90, 90] so every one is upright — at the
                # cost of a 180° snap where the tangent crosses ±90° mid-curve.
                r = ((tick.tangent_deg + 90.0) % 180.0) - 90.0
            elif rotate == 'horizontal':
                r = 0.0
            elif tangent_offset is not None:
                # Tangent + offset ('tangent+90'/'tangent-30'/'tangent_perp'),
                # clamped readable-upright. 'tangent_perp' (offset 90) is the
                # gridline-relative spelling of apply_boundary_labels'
                # orient='parallel' — the label reads ACROSS the gridline it
                # sits on: a ~90° (near-vertical) local tangent maps to ~0°
                # (horizontal) text, a flat parallel to vertical text.
                r = ((tick.tangent_deg + tangent_offset + 90.0) % 180.0) - 90.0
            elif callable(rotate):
                r = float(rotate(tick))
            else:
                r = float(rotate)

            text = _format_tick_label(
                tick.value, tick.kind, self.frame, fmt, sep=sep,
                step=lon_step if tick.kind == 'lon' else lat_step)

            # ``ha`` / ``va`` ``'auto'`` → pick alignment so the
            # bbox's near edge (relative to outward) sits at
            # ``pos_pix``. Caller-supplied values pass through
            # unchanged.
            if ha == 'auto' or va == 'auto':
                ha_auto, va_auto = _resolve_text_anchor(
                    r, 1, tan[0], tan[1])
                ha_final = ha_auto if ha == 'auto' else ha
                va_final = va_auto if va == 'auto' else va
            else:
                ha_final = ha
                va_final = va

            # ``rotation_mode='anchor'`` pivots the rotation around
            # the resolved (ha, va) point — required for the near-edge
            # anchoring to hold AFTER rotation. Without it matplotlib
            # rotates around the bbox center and then re-applies
            # ha/va, which makes the visible near-edge drift away
            # from the anchor on rotated labels.
            artist = self.ax.text(
                pos_data[0], pos_data[1], text,
                rotation=r, ha=ha_final, va=va_final,
                rotation_mode='anchor',
                fontsize=fontsize, color=color,
                zorder=zorder, clip_on=clip_on,
                transform=self.ax.transData, **kwargs)
            artist._sph_overlay_ticklabel = True
            artist._sph_overlay_kind = tick.kind
            artists.append(artist)

        self.label_artists = artists
        if mode == 'auto':
            self._hide_overlapping_labels()
        return self

    def _hide_overlapping_labels(self) -> None:
        """First-wins overlap suppression on label bboxes."""
        if not self.label_artists:
            return
        # Force a draw so each Text artist's bbox is populated.
        self.ax.figure.canvas.draw()
        renderer = self.ax.figure.canvas.get_renderer()
        visible_bboxes: list[Any] = []
        for artist in self.label_artists:
            bbox = artist.get_window_extent(renderer=renderer)
            if any(bbox.overlaps(prev) for prev in visible_bboxes):
                artist.set_visible(False)
            else:
                visible_bboxes.append(bbox)

    def discover_ticks(self, dedup_axis_curve_ticks: bool = True,
                       ) -> CoordinateOverlay:
        """Find every gridline × frame-curve crossing.

        Populates :attr:`gridticks` with one :class:`_GridTick` per
        intersection, each holding the display-pixel position, the
        local gridline tangent angle in degrees, and references to
        the originating gridline and frame curve.

        The parent axes must have been drawn at least once (e.g.
        ``fig.canvas.draw()``) so that ``ax.get_transform('world')``
        produces valid display coordinates.

        Parameters
        ----------
        dedup_axis_curve_ticks : bool
            When ``True`` (default), at most one tick is kept per
            (gridline, frame_curve) pair when the frame curve is an
            interior reference curve (``axis_curve=True``). The tick
            nearest the axes bbox center is kept. Suppresses wrap-
            induced duplicates and aesthetic clutter from the
            "in-frame label" mode where each overlay gridline should
            be labeled once at a clean interior position. Boundary
            curves (``axis_curve=False``) are exempt — their multiple
            crossings (one per spine intersection) are legitimate.

        Returns
        -------
        self : CoordinateOverlay
        """
        if self.frame_curves is None:
            self.frame_curves = self._default_frame_curves()

        ticks: list[_GridTick] = []
        bb = self.ax.bbox
        bbox_center = np.array([bb.x0 + 0.5 * bb.width,
                                bb.y0 + 0.5 * bb.height])
        # Pre-split each frame curve at NaN rows (cross-frame curves
        # may have wrap-jump breaks inserted by ``_insert_pixel_jump_breaks``
        # to avoid spurious crossings on the antimeridian wrap line).
        # Cache so we only split once per discover call.
        fc_sub_polylines: dict[int, list[np.ndarray]] = {}
        for fc in self.frame_curves:
            xy = fc.xy_pix
            if np.isnan(xy).any():
                nan_mask = np.isnan(xy).any(axis=1)
                break_idx = np.where(nan_mask)[0]
                # np.split on the NaN rows then drop them by slicing.
                pieces = np.split(xy, break_idx)
                cleaned = [p[~np.isnan(p).any(axis=1)] for p in pieces]
                fc_sub_polylines[id(fc)] = [p for p in cleaned if len(p) >= 2]
            else:
                fc_sub_polylines[id(fc)] = [xy]
        for gl in self.lon_gridlines + self.lat_gridlines:
            for seg in self._get_pixel_segments(gl):
                for fc in self.frame_curves:
                    # kind filter: a curve tagged with kind='lon' only
                    # accepts meridian intersections; 'lat' only
                    # parallels; None (default) accepts both.
                    if fc.kind is not None and fc.kind != gl.kind:
                        continue
                    for sub in fc_sub_polylines[id(fc)]:
                        points, a_idx, _ = _intersect_polylines(seg, sub)
                        for i in range(len(points)):
                            seg_dir = seg[a_idx[i] + 1] - seg[a_idx[i]]
                            angle = np.degrees(np.arctan2(seg_dir[1],
                                                          seg_dir[0]))
                            ticks.append(_GridTick(points[i], angle,
                                                    gl, fc))

        if dedup_axis_curve_ticks:
            # For each (gridline, axis_curve frame_curve) pair, keep
            # the single tick nearest the bbox center. Boundary curves
            # are exempt (axis_curve=False).
            groups: dict[tuple[int, Any], list[_GridTick]] = {}
            order: list[tuple[int, Any]] = []
            for t in ticks:
                key: tuple[int, Any]
                if not getattr(t.frame_curve, 'axis_curve', False):
                    key = (id(t), 'keep-all')  # unique → never dedupes
                else:
                    key = (id(t.gridline), id(t.frame_curve))
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append(t)
            kept: list[_GridTick] = []
            for key in order:
                bucket = groups[key]
                if len(bucket) == 1:
                    kept.append(bucket[0])
                else:
                    dists = [np.linalg.norm(t.xy_pix - bbox_center)
                             for t in bucket]
                    kept.append(bucket[int(np.argmin(dists))])
            ticks = kept

        self.gridticks = ticks
        self._ticks_discovered = True
        return self


def add_coord_overlay(ax: Any, frame: str = 'galactic',
                      lon_vals: npt.ArrayLike | None = None,
                      lat_vals: npt.ArrayLike | None = None,
                      n_samples: int = 200,
                      lon_style: dict[str, Any] | None = None,
                      lat_style: dict[str, Any] | None = None,
                      **kwargs: Any) -> CoordinateOverlay:
    """Draw a coordinate-system grid overlay on a WCSAxes.

    Convenience wrapper around :class:`CoordinateOverlay` that builds
    the overlay and renders it in one call. Returns the overlay so
    the caller can do further customization.

    Parameters
    ----------
    ax : WCSAxes
    frame : str
        Astropy frame for the overlay (``'galactic'``, ``'icrs'``,
        ``'fk5'``, ``'ecliptic'``, ``'supergalactic'``, ...).
    lon_vals, lat_vals : array_like, optional
        Meridian longitudes / parallel latitudes in degrees.
    n_samples : int
        Sample points per gridline.
    lon_style, lat_style : dict, optional
        Per-axis style overrides.
    **kwargs
        Common style for all gridlines (e.g. ``color='blue'``,
        ``alpha=0.5``). Defaults: gray, lw=0.5, alpha=0.3, dotted.

    Returns
    -------
    overlay : CoordinateOverlay

    Examples
    --------
    >>> fig, ax = sph.allsky_figure('AIT', frame='ICRS')
    >>> sph.add_coord_overlay(ax, frame='galactic', color='C0',
    ...                       alpha=0.4)

    >>> # Custom meridian / parallel sets, asymmetric styling
    >>> sph.add_coord_overlay(ax, frame='galactic',
    ...                       lon_vals=[0, 90, 180, 270],
    ...                       lat_vals=[-30, 0, 30],
    ...                       lon_style={'color': 'red'},
    ...                       lat_style={'color': 'blue'})
    """
    overlay = CoordinateOverlay(ax, frame=frame, lon_vals=lon_vals,
                                lat_vals=lat_vals, n_samples=n_samples)
    overlay.plot(lon_style=lon_style, lat_style=lat_style, **kwargs)
    return overlay


# Alias for users familiar with kapteyn's "graticule" terminology.
add_graticule_overlay = add_coord_overlay


# ===== Overlay ticks/labels on projection boundaries =====

# Map astropy's short spine keys to the _FrameCurve.outward_at
# axis-aligned names for rectangular frames.
_SPINE_NAME_MAP = {
    'b': 'bottom', 't': 'top', 'l': 'left', 'r': 'right',
}


def _frame_to_curves(ax: Any) -> list[_FrameCurve]:
    """Extract the visible boundary of *ax* as a list of _FrameCurves.

    astropy WCSAxes exposes the frame boundary via ``ax.coords.frame``.
    Custom frames (CircularFrame, EllipticalFrame, SinusoidalFrame,
    ParabolicFrame, RobinsonFrame, KavrayskiyFrame, Eckert4Frame,
    WinkelTripelFrame, McBrydeFrame, ...) provide a single spine
    keyed ``'c'`` that traces the projection's outer envelope as a
    closed polyline in display pixels. Default rectangular frames
    expose four spines keyed ``'b'`` / ``'t'`` / ``'l'`` / ``'r'``.

    Returns
    -------
    curves : list of _FrameCurve
    """
    frame = ax.coords.frame
    keys = list(frame.keys())
    if 'c' in keys:
        pix = np.asarray(frame['c']._get_pixel(), dtype=float)
        return [_FrameCurve(pix, name='boundary', closed=True)]
    return [_FrameCurve(np.asarray(frame[k]._get_pixel(), dtype=float),
                        name=_SPINE_NAME_MAP.get(k, k))
            for k in keys]


def _suppress_default_ticks(ax: Any, which: str) -> None:
    """Hide astropy's default WCSAxes tick marks and labels.

    Parameters
    ----------
    ax : WCSAxes
    which : {'both', 'lon', 'lat', 'none'}
    """
    if which not in ('both', 'lon', 'lat', 'none'):
        raise ValueError(
            f"suppress_default must be 'both', 'lon', 'lat', or 'none', "
            f"got {which!r}")
    if which in ('both', 'lon'):
        ax.coords[0].set_ticks_visible(False)
        ax.coords[0].set_ticklabel_visible(False)
    if which in ('both', 'lat'):
        ax.coords[1].set_ticks_visible(False)
        ax.coords[1].set_ticklabel_visible(False)


def _remove_auto_overlay_ticks(ax: Any, kinds: set[str]) -> None:
    """Remove the *auto*-drawn overlay tick marks + labels of the given kinds.

    The auto in-frame / boundary tick overlay that ``make_wcs_frame`` /
    ``make_globe_frame`` draws (tagged ``_sph_auto_overlay``) is a distinct
    artist system from astropy's native ticks — ``suppress_default`` /
    ``set_ticklabel_visible`` can't reach it. So when the user calls
    :func:`add_overlay_ticks` to place their own primary ticks/labels, drop
    the auto default for *only the kinds the new call is drawing* (so e.g.
    re-placing the Dec labels keeps the auto RA-on-gridlines labels), and
    leave the user's own previously-added overlays (not tagged auto) so
    layering still works.
    """
    for artist in list(ax.lines) + list(ax.texts):
        if (getattr(artist, '_sph_auto_overlay', False)
                and getattr(artist, '_sph_overlay_kind', None) in kinds):
            artist.remove()


def _resolve_curve_spec(ax: Any, spec: str | _FrameCurve, kind: str,
                        frame: str | None = None) -> _FrameCurve | None:
    """Resolve a ``lon_at`` / ``lat_at`` spec to a :class:`_FrameCurve`.

    Accepts:
    - ``'boundary'`` → sentinel (caller substitutes the projection
      boundary curves; returned as ``None`` to indicate this)
    - ``'axis'`` → constant-coord curve at the axes' center
      (``lat=center_lat`` for ``kind='lon'``, ``lon=center_lon`` for
      ``kind='lat'``). When ``frame`` resolves to a non-host frame,
      the curve is built at ``(0, 0)`` in the *overlay* frame
      instead — i.e. the overlay's equator (``kind='lon'``) or
      prime meridian (``kind='lat'``).
    - ``'lat=N'`` / ``'lon=N'`` → ``from_const_lat`` /
      ``from_const_lon`` at the parsed value, interpreted in
      ``frame`` (default: host axes' frame).
    - a :class:`_FrameCurve` instance → used directly (kind back-
      filled if absent; ``frame`` ignored — caller built the curve
      themselves and is assumed to have put it in the right frame).

    Parameters
    ----------
    ax : WCSAxes
    spec : str | _FrameCurve
    kind : {'lon', 'lat'}
    frame : str, optional
        Coordinate frame for ``'axis'`` / ``'lat=N'`` / ``'lon=N'``
        specs. ``None`` (default) interprets the spec in the host
        axes' frame. A different frame causes the constructed curve
        to live in the overlay frame's coords (transformed via
        SkyCoord to the host frame before pixel-projection).
    """
    if isinstance(spec, _FrameCurve):
        if spec.kind is None:
            spec.kind = kind
        return spec
    if not isinstance(spec, str):
        raise ValueError(
            f"{kind}_at must be 'boundary', 'axis', 'lat=N', 'lon=N', "
            f"or a _FrameCurve, got {spec!r}")
    if spec == 'boundary':
        return None
    is_cross_frame = (frame is not None
                      and _resolve_frame(frame)
                      != _resolve_frame(_get_wcs_frame_name(ax)))
    if spec == 'axis':
        if is_cross_frame:
            # Center the axis curve on the FIELD, in overlay-frame coords.
            # On a zoomed field the overlay frame's origin (0, 0) is generally
            # nowhere in view, so a lon=0 / lat=0 axis curve would miss the
            # field entirely (the lat-tick / lon-tick axis curve then finds no
            # crossings). All-sky cross-frame keeps the overlay-frame origin
            # (the natural center of a whole-sky graticule).
            c_lon = c_lat = 0.0
            if (not getattr(ax, '_sph_is_allsky', True)
                    and not getattr(ax, '_sph_is_globe', False)):
                ext = _field_world_extent(ax, overlay_frame=frame)
                if ext is not None:
                    e_lon_lo, e_lon_hi, e_lat_lo, e_lat_hi = ext
                    c_lon = 0.5 * (e_lon_lo + e_lon_hi)
                    c_lat = 0.5 * (e_lat_lo + e_lat_hi)
            if kind == 'lon':
                return _FrameCurve.from_const_lat(
                    ax, c_lat, kind='lon', frame=frame)
            return _FrameCurve.from_const_lon(
                ax, c_lon, kind='lat', frame=frame)
        # On a zoomed field frame the all-sky default sampling range (full
        # lon / lat span at fixed n) puts only ~1 sample inside a few-degree
        # field — too sparse for the densified segment intersection to find
        # any ticks. Sample the axis curve across the FIELD extent instead
        # (with a small pad so the outermost gridline still crosses it).
        # All-sky frames keep the full-sky sampling (unchanged behavior).
        field_ext = (None if (getattr(ax, '_sph_is_allsky', True)
                              or getattr(ax, '_sph_is_globe', False))
                     else _field_world_extent(ax))
        if kind == 'lon':
            lon_range = None
            if field_ext is not None:
                lo, hi, _, _ = field_ext
                pad = 0.05 * (hi - lo)
                lon_range = (lo - pad, hi + pad)
            return _FrameCurve.from_const_lat(
                ax, _get_wcs_center_lat(ax), kind='lon', lon_range=lon_range)
        lat_range = (-89.9999, 89.9999)
        if field_ext is not None:
            _, _, lo, hi = field_ext
            pad = 0.05 * (hi - lo)
            lat_range = (max(-89.9999, lo - pad), min(89.9999, hi + pad))
        return _FrameCurve.from_const_lon(
            ax, _get_wcs_center_lon(ax), kind='lat', lat_range=lat_range)
    if spec.startswith('lat='):
        return _FrameCurve.from_const_lat(
            ax, float(spec[4:]), kind='lon', frame=frame)
    if spec.startswith('lon='):
        return _FrameCurve.from_const_lon(
            ax, float(spec[4:]), kind='lat', frame=frame)
    raise ValueError(
        f"{kind}_at spec {spec!r} not recognized — use 'boundary', "
        f"'axis', 'lat=N', 'lon=N', or a _FrameCurve")


def add_overlay_ticks(ax: Any, lon_vals: npt.ArrayLike | None = None,
                      lat_vals: npt.ArrayLike | None = None,
                      lon_at: str | _FrameCurve | None = 'boundary',
                      lat_at: str | _FrameCurve | None = 'boundary',
                      boundary: _FrameCurve | npt.ArrayLike | None = None,
                      frame: str | None = None,
                      suppress_default: str = 'both', n_samples: int = 200,
                      tick_kwargs: dict[str, Any] | None = None,
                      label_kwargs: dict[str, Any] | None = None,
                      stroke_lw: float | None = None,
                      stroke_color: str | None = None,
                      lon_tick_kwargs: dict[str, Any] | None = None,
                      lat_tick_kwargs: dict[str, Any] | None = None,
                      lon_label_kwargs: dict[str, Any] | None = None,
                      lat_label_kwargs: dict[str, Any] | None = None,
                      show_ticks: bool = True,
                      show_labels: bool = True,
                      _auto: bool = False) -> CoordinateOverlay:
    """Place overlay-style ticks + labels on a projection curve.

    Two placement modes, switched per-axis via ``lon_at`` / ``lat_at``:

    **Boundary mode** (``'boundary'``, default): ticks discovered
    against the projection's natural visible boundary curve —
    circular for SIN / ARC / ZEA / STG / AZP / SZP / AIR, elliptical
    for AIT / MOL, sinusoidal for SFL, parabolic for PAR, and the
    various custom curves from the non-FITS pseudocylindricals
    (Robinson / Kavrayskiy / Eckert IV / Winkel Tripel / McBryde).
    Rectangular frames fall back to their four bbox spines. Cleanest
    for circular zenithal sub-frame views and for annotating an
    inset frame edge.

    **Axis-curve mode** (``'axis'``, ``'lat=N'``, ``'lon=N'``): ticks
    discovered against an internal reference curve — typically a
    parallel for lon labels and a meridian for lat labels. Labels
    follow the projection's curvature (bowing with a Robinson
    equator, fanning out around a SIN polar meridian) rather than
    stacking on the bbox-rectangle. Closest to traditional
    astronomy plot conventions.

    The two axes are independent: you can use boundary placement
    for lat labels and axis placement for lon labels, or mix in any
    combination.

    Parameters
    ----------
    ax : WCSAxes
    lon_vals, lat_vals : array_like, optional
        Meridian / parallel values to label, in degrees. Defaults
        match :class:`CoordinateOverlay` (``arange(0, 360, 30)`` /
        ``arange(-75, 76, 15)``). When both ``lon_at`` and ``lat_at``
        resolve to the projection boundary on an envelope frame,
        the lon-default is filtered to exclude the antimeridian
        (which lies along the boundary and would produce spurious
        near-collinear crossings).
    lon_at, lat_at : str or _FrameCurve
        Where to place lon / lat ticks. Accepted forms:

        - ``'boundary'`` (default) — projection boundary
        - ``'axis'`` — central parallel for lon, central meridian
          for lat
        - ``'lat=N'`` / ``'lon=N'`` — constant-coord curve at the
          parsed value in degrees
        - a :class:`_FrameCurve` instance — used directly

        Pass ``None`` to skip that kind of tick entirely. ``'lat=N'``
        only makes sense for ``lon_at``, ``'lon=N'`` only for
        ``lat_at``; the wrong combination raises.
    boundary : _FrameCurve or (N, 2) array_like, optional
        Explicit closed boundary curve. When provided, replaces the
        projection's astropy frame spine wherever a boundary is
        needed — for clipping out-of-frame axis-curve ticks, and
        wherever ``lon_at`` / ``lat_at`` is ``'boundary'``. Accepts
        either a :class:`_FrameCurve` (used as-is) or a raw
        ``(N, 2)`` world-coord polyline (auto-wrapped via
        :meth:`_FrameCurve.from_world_polyline` with ``closed=True``).

        Needed for projections like BON (Bonne pseudoconic), PCO
        (polyconic), HPX (HEALPix all-sky), and the conic family
        (COD / COE / COO / COP) whose astropy frame spine doesn't
        trace the actual visible region. See
        :mod:`skyplothelper.projections._boundaries` for the
        per-projection boundary helpers.
    frame : str, optional
        Overlay frame name. ``None`` (default) uses the axes' own
        frame.
    suppress_default : {'both', 'lon', 'lat', 'none'}
        Which astropy default tick marks / labels to hide before
        rendering the overlay ones. Default ``'both'``.
    n_samples : int
        Samples per gridline for the intersection algorithm.
    tick_kwargs : dict, optional
        Forwarded to :meth:`CoordinateOverlay.render_ticks` (length,
        lw, color, direction, ...). Shared by both axes; override per
        axis with ``lon_tick_kwargs`` / ``lat_tick_kwargs``.
    label_kwargs : dict, optional
        Forwarded to :meth:`CoordinateOverlay.render_labels`
        (fontsize, color, pad, rotate, fmt, sep, mode, ...). Shared by
        both axes; override per axis with ``lon_label_kwargs`` /
        ``lat_label_kwargs``. The ``rotate`` key takes the same values
        as :meth:`~CoordinateOverlay.render_labels` — including
        ``'tangent_perp'`` / ``'tangent+90'`` / ``'tangent-30'`` for a
        tangent-relative offset (the overlay spelling of
        :func:`apply_boundary_labels`' ``orient='parallel'``).
    stroke_lw : float, optional
        Stroke (outline) linewidth for the labels — readability on busy
        backgrounds, parity with :func:`format_ticklabels`. Applied only
        when both ``stroke_lw`` and ``stroke_color`` are given, and only
        if ``label_kwargs`` doesn't already set ``path_effects``.
    stroke_color : str, optional
        Stroke color (e.g. ``'w'`` for a white outline on a dark map).
    lon_tick_kwargs, lat_tick_kwargs : dict, optional
        Per-axis ``render_ticks`` overrides, merged on top of
        ``tick_kwargs`` for that axis only. Giving any per-axis dict
        renders the two coordinates in separate passes so lon and lat
        can differ (e.g. inward lon ticks, outward lat ticks).
    lon_label_kwargs, lat_label_kwargs : dict, optional
        Per-axis ``render_labels`` overrides, merged on top of
        ``label_kwargs`` for that axis only — different ``rotate`` /
        ``sep`` / ``color`` / ``fmt`` for lon vs lat in one call.
    show_ticks, show_labels : bool, optional
        Skip tick-mark rendering / label rendering when ``False``.
        Defaults to ``True`` for both — preserves the historical
        "render both" behavior. The :meth:`discover_ticks` step still
        runs in either case, so the returned overlay's
        ``gridticks`` attribute is populated for downstream use.

    Returns
    -------
    overlay : CoordinateOverlay
        Returned so callers can re-style or rediscover ticks later.

    Examples
    --------
    >>> # Boundary mode (the default)
    >>> sph.add_overlay_ticks(ax)

    >>> # Axis-curve mode: lon labels along the equator,
    >>> # lat labels along the central meridian
    >>> sph.add_overlay_ticks(ax, lon_at='axis', lat_at='axis')

    >>> # Mixed: lon labels along lat=20° parallel, lat labels on
    >>> # the projection boundary
    >>> sph.add_overlay_ticks(ax, lon_at='lat=20')

    >>> # Lon-only axis-curve labels, leave default lat labels alone
    >>> sph.add_overlay_ticks(ax, lon_at='axis', lat_at=None,
    ...                       suppress_default='lon')
    """
    from matplotlib.path import Path

    _suppress_default_ticks(ax, suppress_default)

    # Kind-aware replace of the auto in-frame/boundary overlay (option 3): a
    # USER call (``_auto=False``) drops the auto-drawn overlay for only the
    # kinds it is about to draw — so re-placing Dec labels keeps the auto RA
    # labels, and nothing doubles. The internal auto call (``_auto=True``)
    # skips this (it IS the auto set) and tags its artists below.
    if not _auto:
        _kinds = set()
        if lon_at is not None:
            _kinds.add('lon')
        if lat_at is not None:
            _kinds.add('lat')
        if _kinds:
            _remove_auto_overlay_ticks(ax, _kinds)

    # Frame curves must be sampled *after* a draw so display coords
    # are valid (both boundary and constant-coord curve construction
    # use the live transform).
    ax.figure.canvas.draw()

    # Resolve an explicit ``boundary=`` argument up-front. Accepts a
    # raw world-coord polyline OR a pre-built _FrameCurve; the rest
    # of the function then treats it as the projection boundary
    # (substitutes for 'boundary' specs and serves as the clip
    # polygon for axis-curve mode).
    if boundary is not None and not isinstance(boundary, _FrameCurve):
        boundary = _FrameCurve.from_world_polyline(
            ax, np.asarray(boundary, dtype=float), closed=True,
            name='boundary')

    # Resolve curve specs. 'boundary' for either axis falls back to
    # the projection boundary curves; if both are 'boundary' the
    # shared-boundary codepath kicks in (single shared curve set).
    # The overlay's frame is plumbed through so that 'axis' / 'lat=N' /
    # 'lon=N' curves live in the overlay frame's coords when the
    # caller specifies a non-default ``frame=`` (the "in-frame label"
    # mode used for cross-frame compound plots).
    overlay_frame = frame if frame is not None else _get_wcs_frame_name(ax)
    lon_curve = (_resolve_curve_spec(ax, lon_at, 'lon', frame=overlay_frame)
                 if lon_at is not None else None)
    lat_curve = (_resolve_curve_spec(ax, lat_at, 'lat', frame=overlay_frame)
                 if lat_at is not None else None)
    boundary_for_lon = lon_at == 'boundary' and lon_at is not None
    boundary_for_lat = lat_at == 'boundary' and lat_at is not None

    if boundary is not None:
        boundary_curves = [boundary] if (boundary_for_lon
                                          or boundary_for_lat) else []
    else:
        boundary_curves = _frame_to_curves(ax) if (
            boundary_for_lon or boundary_for_lat) else []
    # Tag boundary curves with the appropriate kind. If both axes
    # use the boundary, leave kind=None so the same curves serve
    # both (the shared-boundary behavior). Otherwise split: copy the
    # boundary curves and tag the copies.
    curves = []
    if boundary_for_lon and boundary_for_lat:
        curves.extend(boundary_curves)
    else:
        if boundary_for_lon:
            for bc in boundary_curves:
                curves.append(_FrameCurve(bc.xy_pix, name=bc.name,
                                          closed=bc.closed, kind='lon'))
        if boundary_for_lat:
            for bc in boundary_curves:
                curves.append(_FrameCurve(bc.xy_pix, name=bc.name,
                                          closed=bc.closed, kind='lat'))
    if lon_curve is not None and not boundary_for_lon:
        curves.append(lon_curve)
    if lat_curve is not None and not boundary_for_lat:
        curves.append(lat_curve)

    # On envelope frames (single closed boundary curve), the lon=
    # antimeridian meridian is parameterized to lie *along* the
    # boundary in pixel space; densified segment intersection
    # produces a cascade of near-collinear false crossings. Filter
    # the antimeridian out of the default lon_vals when lon ticks
    # are bound to the boundary. Axis-curve mode doesn't have this
    # degeneracy.
    is_envelope_boundary = (
        boundary_for_lon
        and len(boundary_curves) == 1
        and boundary_curves[0].closed)
    if is_envelope_boundary and lon_vals is None:
        center_lon = _get_wcs_center_lon(ax)
        anti = (center_lon + 180.0) % 360.0
        default = np.arange(0., 360., 30.)
        lon_vals = default[
            np.abs(((default - anti + 180.) % 360.) - 180.) > 0.5]

    # FOV-adaptive default graticule values (so a zoomed field's overlay
    # isn't empty) are resolved inside CoordinateOverlay.__init__ now — for
    # both same-frame and cross-frame overlays, in the overlay's own frame.
    # Only the envelope-boundary antimeridian filter above needs to touch
    # lon_vals here.
    overlay = CoordinateOverlay(ax, frame=overlay_frame,
                                lon_vals=lon_vals, lat_vals=lat_vals,
                                n_samples=n_samples)
    overlay.set_frame_curves(curves)
    overlay.discover_ticks()

    # Axis-curve curves (constant-lat / constant-lon polylines) can place
    # ticks outside the visible frame. Two cases:
    #  - SIN globe (limited fov): lon=90/270 along the lat=center parallel
    #    map to valid pixels OUTSIDE the circular spine.
    #  - Rectilinear field frames (TAN/SIN with fov_deg): a constant-coord
    #    curve sampled across the sky runs toward infinity under a gnomonic
    #    projection, so a crossing can land thousands of pixels off-canvas —
    #    which then explodes ``savefig(bbox_inches='tight')``.
    # Always clip discovered ticks to the axes bbox (catches both, on ANY
    # frame shape, incl. non-finite tick positions). When a single closed
    # boundary polygon is available, additionally require ticks inside it
    # (the SIN-spine phantom-tick filter).
    if not (boundary_for_lon and boundary_for_lat) and overlay.gridticks:
        bb = ax.bbox
        pts = np.array([t.xy_pix for t in overlay.gridticks], dtype=float)
        margin = 5.0
        with np.errstate(invalid='ignore'):
            inside = (
                (pts[:, 0] >= bb.x0 - margin)
                & (pts[:, 0] <= bb.x1 + margin)
                & (pts[:, 1] >= bb.y0 - margin)
                & (pts[:, 1] <= bb.y1 + margin))  # NaN/inf → False (dropped)
        if boundary is not None:
            clip_curves = [boundary]
        elif boundary_curves:
            clip_curves = boundary_curves
        else:
            clip_curves = _frame_to_curves(ax)
        if len(clip_curves) == 1 and clip_curves[0].closed:
            # Sanitize the polygon: NaN samples (projection singularities,
            # e.g. COP's cone apex) and wildly-extrapolated samples break
            # ``Path.contains_points``. Drop both before the polygon test.
            poly = clip_curves[0].xy_pix
            poly = poly[np.isfinite(poly).all(axis=1)]
            bw = max(bb.width, bb.height)
            sane = ((np.abs(poly[:, 0] - bb.x0) < 4 * bw)
                    & (np.abs(poly[:, 1] - bb.y0) < 4 * bw))
            poly = poly[sane]
            if len(poly) >= 3:
                inside = inside & Path(poly).contains_points(pts, radius=1.0)
        overlay.gridticks = [t for t, ok in zip(overlay.gridticks, inside)
                             if ok]

    # Base label kwargs: honor an auto-fontsize value cached on the axes by
    # make_wcs_frame / make_globe_frame's auto_fontsize hook, so callers
    # (galleries, downstream helpers) that invoke add_overlay_ticks directly
    # pick up the same sizing the frame builder chose — unless they pass their
    # own ``label_kwargs={'fontsize': ...}``. ``stroke_lw`` / ``stroke_color``
    # add a readability stroke to the labels (parity with format_ticklabels),
    # unless the caller already supplied their own ``path_effects``.
    base_label_kwargs = dict(label_kwargs or {})
    cached_fs = getattr(ax, '_sph_auto_label_fontsize', None)
    if cached_fs is not None and 'fontsize' not in base_label_kwargs:
        base_label_kwargs['fontsize'] = cached_fs
    stroke_pe = _stroke_path_effects(stroke_color, stroke_lw)
    if stroke_pe is not None and 'path_effects' not in base_label_kwargs:
        base_label_kwargs['path_effects'] = stroke_pe
    base_tick_kwargs = dict(tick_kwargs or {})
    # The stroke backs the tick MARKS too, not only the labels — otherwise a
    # stroked overlay reads inconsistently (outlined labels, bare marks).
    if stroke_pe is not None and 'path_effects' not in base_tick_kwargs:
        base_tick_kwargs['path_effects'] = stroke_pe

    # Per-axis styling: when any lon_*/lat_* override is given, render the two
    # kinds separately so each axis can carry its own tick / label kwargs
    # (color, rotate, sep, ...). Partition the discovered ticks by kind, render
    # each subset with base ∪ per-axis kwargs, then restore the full set and
    # combine the artist lists. (Overlap hiding runs within each axis's pass;
    # cross-axis label overlaps — rare — are not cross-checked.)
    per_axis = any(k is not None for k in (lon_tick_kwargs, lat_tick_kwargs,
                                           lon_label_kwargs, lat_label_kwargs))
    if per_axis:
        all_ticks = list(overlay.gridticks)
        combined_ticks: list[Any] = []
        combined_labels: list[Any] = []
        for kind, extra_tk, extra_lk in (
                ('lon', lon_tick_kwargs, lon_label_kwargs),
                ('lat', lat_tick_kwargs, lat_label_kwargs)):
            overlay.gridticks = [t for t in all_ticks if t.kind == kind]
            if not overlay.gridticks:
                continue
            if show_ticks:
                overlay.render_ticks(**{**base_tick_kwargs, **(extra_tk or {})})
                combined_ticks.extend(overlay.tick_artists)
            if show_labels:
                overlay.render_labels(
                    **{**base_label_kwargs, **(extra_lk or {})})
                combined_labels.extend(overlay.label_artists)
        overlay.gridticks = all_ticks
        overlay.tick_artists = combined_ticks
        overlay.label_artists = combined_labels
    else:
        if show_ticks:
            overlay.render_ticks(**base_tick_kwargs)
        if show_labels:
            overlay.render_labels(**base_label_kwargs)
    # Tag the auto-drawn overlay (the make_wcs_frame / make_globe_frame default)
    # so a later USER add_overlay_ticks call can replace it kind-aware above.
    if _auto:
        for artist in overlay.tick_artists + overlay.label_artists:
            artist._sph_auto_overlay = True
    return overlay
