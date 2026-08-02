"""Plotly export core — make_figure / project / add_scatter / add_healpix /
constellation overlays.

All functions in this module use lazy ``plotly`` imports so that the
plotly dependency is genuinely optional. ``import skyplothelper`` works
without plotly installed; the import only triggers when a user calls
one of these functions, at which point a clean ``ImportError`` points
at the install command.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord


from ..projections.project import project as _project

__all__ = [
    'make_figure',
    'project',
    'add_scatter',
    'add_healpix',
    'add_healpix_sparse',
    'add_constellation_boundaries',
    'add_constellation_lines',
    'add_constellation_labels',
    'add_constellation_polygon',
    'add_great_circle',
    'add_plane_overlay',
    'add_geodesic_circle',
    'add_spherical_polygon',
    'add_lonlat_box',
    'add_frame_band',
    'add_great_circle_band',
    'add_sky_vectors',
    'add_coord_labels',
    'add_frame_edge',
    'add_reticle',
    'add_ruler',
    'add_compound_region',
    'make_compound_region',
]


# -- Wrap-edge polygon split ------------------------------------------------

def _split_polygon_at_wrap(
    lons: npt.ArrayLike, lats: npt.ArrayLike, center: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a closed polygon at the wrap edge ``lon = center ± 180``.

    Plotly-side adapter over the shared D3-style antimeridian clipper
    :func:`skyplothelper.geometry._antimeridian._antimeridian_clip`
    (also used by the mpl side and the future CompoundRegion plotly
    port). The clipper does Sutherland-Hodgman polygon clipping
    against the wrap meridian, returning open segments each
    annotated with ``entry_lat`` / ``exit_lat`` boundary markers.

    This function closes each segment into a renderable fill polygon
    via :func:`_close_clipped_segment` — along the wrap meridian for
    same-side crossings (the common band / region case) or through
    the pole for opposite-side crossings (the polar-polygon case,
    e.g. constellation outlines that wrap around the celestial pole).

    Parameters
    ----------
    lons, lats : array-like
        Vertex coordinates of a closed polygon (``lons[0] == lons[-1]``,
        same for ``lats``). Lons in any range — the function shifts
        internally.
    center : float
        Longitude (degrees) of the projection center. The wrap edge is
        at ``center ± 180°``.

    Returns
    -------
    pieces : list of (ndarray, ndarray)
        ``(lons, lats)`` arrays — each a closed sub-polygon. Caller
        projects and emits traces.
    """
    from ..geometry._antimeridian import (
        _antimeridian_clip,
        _close_clipped_segment,
    )

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    segments = _antimeridian_clip(lons, lats, center)
    if not segments:
        return []

    # ``_antimeridian_clip`` returns a single segment with
    # ``entry_lat=None`` / ``exit_lat=None`` when no edge crosses the wrap
    # meridian. Shift into the projection window and pass it through — but a
    # pole-touching tile spans a wide longitude range (its corners reach
    # ``lon = center ± 180`` without an edge crossing it), so the modulo shift
    # maps that boundary vertex to the opposite canvas edge and the fill would
    # streak across the figure. When the shifted polygon still has a
    # ``|d_lon| > 180`` jump, split it at the jump (the pre-clip behavior)
    # instead of returning it whole.
    if (len(segments) == 1
            and segments[0]['entry_lat'] is None
            and segments[0]['exit_lat'] is None):
        shifted = ((lons - center + 180.0) % 360.0) - 180.0 + center
        if np.any(np.abs(np.diff(shifted)) > 180.0):
            return _split_polygon_by_jump(shifted, lats, center)
        return [(shifted, lats)]

    pieces = []
    for seg in segments:
        closed_lons, closed_lats = _close_clipped_segment(seg, center)
        if len(closed_lons) >= 3:
            pieces.append((closed_lons, closed_lats))
    return pieces


def _split_polygon_by_jump(
    shifted: npt.ArrayLike, lats: npt.ArrayLike, center: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a wrap-shifted closed polygon at ``|d_lon| > 180`` jumps into
    one or two edge-hugging sub-polygons.

    Fallback for the degenerate pole-touching tile case the antimeridian
    clipper reports as non-crossing: the polygon's vertices straddle the wrap
    meridian (one sits at ``center ± 180``) without an edge crossing it, so it
    must be cut at the jump and capped at each wrap edge rather than drawn
    whole. Caps are densified along the wrap edge so the closing edge follows
    the curved AIT / MOL frame silhouette.
    """
    from ..geometry._antimeridian import _densify_along_wrap_edge

    shifted = np.asarray(shifted, dtype=float)
    lats = np.asarray(lats, dtype=float)
    wrap_hi = center + 180.0
    wrap_lo = center - 180.0

    high_lons, high_lats = [], []   # vertices on the lon >= center side
    low_lons, low_lats = [], []     # vertices on the lon <= center side

    n_edges = len(shifted) - 1   # closed polygon → last == first
    for i in range(n_edges):
        l1, l2 = float(shifted[i]), float(shifted[i + 1])
        a1, a2 = float(lats[i]), float(lats[i + 1])
        if l1 >= center:
            high_lons.append(l1)
            high_lats.append(a1)
        if l1 <= center:
            low_lons.append(l1)
            low_lats.append(a1)
        if abs(l2 - l1) > 180.0:
            # Edge straddles the wrap edge — cap both sides at the latitude
            # of the crossing (same sphere point, opposite canvas edges).
            if l1 > center:
                d_to_edge = wrap_hi - l1
                d_from_edge = l2 - wrap_lo
            else:
                d_to_edge = l1 - wrap_lo
                d_from_edge = wrap_hi - l2
            total = d_to_edge + d_from_edge
            if total <= 0:
                continue
            cap_lat = a1 + (d_to_edge / total) * (a2 - a1)
            high_lons.append(wrap_hi)
            high_lats.append(cap_lat)
            low_lons.append(wrap_lo)
            low_lats.append(cap_lat)

    pieces = []
    for plons, plats, wrap_lon in ((high_lons, high_lats, wrap_hi),
                                   (low_lons, low_lats, wrap_lo)):
        if len(plons) >= 3:
            plons.append(plons[0])
            plats.append(plats[0])
            dlons, dlats = _densify_along_wrap_edge(plons, plats, wrap_lon)
            pieces.append((dlons, dlats))
    return pieces



def _split_polyline_at_wrap(
    lons: npt.ArrayLike, lats: npt.ArrayLike, center: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN at wrap-edge crossings in an open polyline.

    Polyline counterpart to :func:`_split_polygon_at_wrap`. Detects
    consecutive vertex pairs whose ``d_lon > 180`` after shifting into
    ``[center - 180, center + 180]``, and inserts a ``NaN`` between
    them so plotly's ``go.Scatter(mode='lines')`` breaks the polyline
    cleanly at the wrap edge rather than drawing a long sweep across
    the canvas. The lon arrays returned are also already shifted into
    the canonical projection window.

    Parameters
    ----------
    lons, lats : array-like
        Open-polyline coordinates in degrees.
    center : float
        Projection center longitude (degrees).

    Returns
    -------
    out_lons, out_lats : ndarray
        Coordinates with ``np.nan`` inserted at wrap-edge crossings.
        Ready to project (NaN propagates cleanly through the WCS
        path) and then hand to ``go.Scatter(x=..., y=...)``.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    shifted = ((lons - center + 180.0) % 360.0) - 180.0 + center
    d = np.abs(np.diff(shifted))
    jumps = np.where(d > 180.0)[0]
    if len(jumps) == 0:
        return shifted, lats
    out_lons: list[float] = []
    out_lats: list[float] = []
    prev = 0
    for j in jumps:
        out_lons.extend(shifted[prev:j + 1])
        out_lats.extend(lats[prev:j + 1])
        out_lons.append(np.nan)
        out_lats.append(np.nan)
        prev = int(j) + 1
    out_lons.extend(shifted[prev:])
    out_lats.extend(lats[prev:])
    return np.asarray(out_lons), np.asarray(out_lats)


# -- Hover-template helpers --------------------------------------------------

def _resolve_hover_line(
    hover: bool | str | None, name: str | None, fig: Any = None,
) -> tuple[str | None, str | None]:
    """Resolve the ``hover`` parameter for line / polyline traces.

    Parameters
    ----------
    hover : False / None / True / str
        ``False`` or ``None`` keeps the current ``hoverinfo='skip'``
        behavior. ``True`` requests the auto template — name (if set)
        plus per-vertex RA/Dec from ``customdata[0]/[1]``. A string is
        used directly as the hovertemplate (``<extra></extra>`` is
        appended if not present, to suppress the trace-name box).
    name : str or None
        Trace name. Included as a bold header line when ``hover=True``.

    Returns
    -------
    hovertemplate : str or None
        ``None`` when hover is disabled; otherwise the resolved template.
    hoverinfo : str or None
        ``'skip'`` when hover is disabled; ``None`` (i.e. let plotly use
        the template) otherwise.
    """
    if hover is False or hover is None:
        return None, 'skip'
    if hover is True:
        prefix = f"<b>{name}</b><br>" if name else ""
        return _default_hover(fig, prefix=prefix), None
    tpl = str(hover)
    if '<extra>' not in tpl:
        tpl += '<extra></extra>'
    return tpl, None


def _resolve_hover_fill(
    hover: bool | str | None, name: str | None,
    auto_detail: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Resolve the ``hover`` parameter for filled (``hoveron='fills'``) traces.

    Parameters
    ----------
    hover : False / None / True / str
    name : str or None
        Bold-header line when ``hover=True``.
    auto_detail : str or None
        Extra lines (already HTML-formatted) appended after the name
        when ``hover=True``. E.g. ``"center: (...)<br>radius: ...°"``
        for a geodesic disc.

    Returns
    -------
    text : str or None
        The single-string ``text`` to assign to the trace.
    hovertemplate : str or None
        ``'%{text}<extra></extra>'`` when hover is enabled, else None.
    hoverinfo : str or None
        ``'skip'`` when disabled, else None.
    """
    if hover is False or hover is None:
        return None, None, 'skip'
    if hover is True:
        parts = []
        if name:
            parts.append(f"<b>{name}</b>")
        if auto_detail:
            parts.append(auto_detail)
        text = "<br>".join(parts)
        return text, '%{text}<extra></extra>', None
    return str(hover), '%{text}<extra></extra>', None


# -- HEALPix tile-resolution heuristic ---------------------------------------

def _resolve_tile_resolution(tile_resolution: int | str, nside: int) -> int:
    """Resolve ``tile_resolution`` against ``nside`` for HEALPix tile sampling.

    Returns the ``step`` argument to pass to ``healpy.boundaries`` —
    the number of sample points per tile edge.

    ``'auto'`` (default) targets roughly 1° spacing along each tile
    edge: ``max(2, ceil(58 / nside))``. The 58 is a rough handle on
    the angular edge length of a HEALPix tile at ``nside`` (which
    scales as ``~58.6°/nside``). This keeps low-nside polar-cap tiles
    well-densified (so their projected chord-approximated boundaries
    don't expose neighbor-tile color through sliver gaps near the
    poles in the AIT silhouette) while staying light at high nside.

    Explicit integers pass through ``max(1, int(...))`` and bypass the
    heuristic — users who want manual control still get it.
    """
    if isinstance(tile_resolution, str):
        if tile_resolution.lower() != 'auto':
            raise ValueError(
                f"tile_resolution must be 'auto' or an integer, "
                f"got {tile_resolution!r}")
        return max(2, int(np.ceil(58.0 / max(1, int(nside)))))
    return max(1, int(tile_resolution))


# -- Colorbar for fill-colored traces ----------------------------------------

def _add_scale_colorbar(
    fig: Any, go: Any, *, colorscale: str | list[Any],
    cmin: float, cmax: float, title: str | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Attach a standalone colorbar to ``fig`` for a set of fill-colored traces.

    A trace's ``fillcolor`` is a flat color — it carries no colorscale, so
    plotly has nothing to build a colorbar from. Tile maps (HEALPix) sample
    the colorscale themselves and paint each tile a solid color, which means
    the key has to come from a companion trace: an invisible marker whose
    ``marker.colorscale`` / ``cmin`` / ``cmax`` reproduce the mapping the
    tiles were painted with. The point itself is placed at ``(None, None)``
    so it contributes no data and cannot disturb the axis ranges.

    ``cmin`` / ``cmax`` are taken from the same ``vmin`` / ``vmax`` used to
    color the tiles, so the bar can never disagree with what it labels.
    """
    cbar: dict[str, Any] = {}
    if title is not None:
        cbar['title'] = title
    if colorbar_kwargs:
        cbar.update(colorbar_kwargs)
    trace = go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(
            colorscale=colorscale, cmin=float(cmin), cmax=float(cmax),
            color=[float(cmin)], size=0.1, opacity=0,
            showscale=True, colorbar=cbar,
        ),
        hoverinfo='skip', showlegend=False, name='',
    )
    fig.add_trace(trace)
    return trace


# -- Lazy plotly import ------------------------------------------------------

def _import_plotly() -> Any:
    """Import ``plotly.graph_objects`` lazily with a friendly error."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "skyplothelper.plotly requires the optional `plotly` package. "
            "Install with `pip install plotly` or "
            "`pip install skyplothelper[plotly]`."
        ) from exc
    return go


# -- Theme presets -----------------------------------------------------------

_THEMES = {
    'dark': {
        'bg':       '#0a0a14',
        'fg':       '#dcdcdc',
        'grid':     '#3a3a4a',
        'limb':     '#dcdcdc',
    },
    'light': {
        'bg':       '#ffffff',
        'fg':       '#1a1a1a',
        'grid':     '#cccccc',
        'limb':     '#1a1a1a',
    },
}


# -- project() re-export -----------------------------------------------------

def project(lons: SkyCoord | npt.ArrayLike, lats: npt.ArrayLike | None = None,
            projection: str = 'AIT', center: float = 0.0,
            lat_center: float = 0.0, frame: str | None = None,
            direction: str = 'sky') -> tuple[np.ndarray, np.ndarray]:
    """Re-export of :func:`skyplothelper.project`.

    Identical signature and behavior — provided here so users working
    primarily in ``skyplothelper.plotly`` can find the primitive
    alongside the other plotly helpers without a second import.

    See :func:`skyplothelper.projections.project` for the full
    documentation.
    """
    return _project(lons, lats, projection=projection, center=center,
                    lat_center=lat_center, frame=frame, direction=direction)


# -- make_figure -------------------------------------------------------------

def _grid_line_positions(lo: float, hi: float, spacing: float,
                         hard_lo: float | None = None,
                         hard_hi: float | None = None) -> list[float]:
    """Graticule line positions: multiples of ``spacing`` within ``[lo, hi]``.

    Anchored at round degrees (0, ±spacing, ±2·spacing, …) so lines land on
    tidy values whether the figure is all-sky or a zoomed field. ``hard_lo`` /
    ``hard_hi`` drop lines that sit *on* an excluded bound — the poles for
    latitude, and the wrap meridian for all-sky longitude (which the caller
    re-adds once as an explicit mirror pair).
    """
    if spacing <= 0:
        return []
    eps = spacing * 1e-9
    k0 = int(np.ceil(lo / spacing - 1e-9))
    k1 = int(np.floor(hi / spacing + 1e-9))
    vals = [k * spacing for k in range(k0, k1 + 1)]
    if hard_lo is not None:
        vals = [v for v in vals if v > hard_lo + eps]
    if hard_hi is not None:
        vals = [v for v in vals if v < hard_hi - eps]
    return vals


def _fov_extent(projection: str, center: float, lat_center: float,
                direction: str, fov_deg: float) -> tuple[float, float,
                                                          float, float]:
    """Projection-plane bounding box ``(x0, x1, y0, y1)`` of an ``fov_deg``
    field centered on ``(center, lat_center)``.

    Densely samples a sky window that spans ``fov_deg`` vertically and
    ``fov_deg / cos(lat_center)`` horizontally — so the *visible* field is
    ``fov_deg`` wide regardless of the meridian convergence — projects it,
    and returns the finite bounding box. Sampling rather than solving in
    closed form keeps this projection-agnostic: it works for every code the
    shared :func:`project` primitive handles.
    """
    half = float(fov_deg) / 2.0
    lat_lo = max(-89.999, lat_center - half)
    lat_hi = min(89.999, lat_center + half)
    coslat = max(0.05, float(np.cos(np.radians(lat_center))))
    lon_half = half / coslat
    lons = np.linspace(center - lon_half, center + lon_half, 121)
    lats = np.linspace(lat_lo, lat_hi, 121)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    x, y = _project(grid_lon, grid_lat, projection=projection, center=center,
                    lat_center=lat_center, direction=direction)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fx, fy = x[np.isfinite(x)], y[np.isfinite(y)]
    if not fx.size or not fy.size:
        raise ValueError(
            f"fov_deg={fov_deg} around center=({center}, {lat_center}) "
            f"projects to nothing on {projection!r} — the whole field is "
            f"off-sky. Check the center, or use a smaller fov_deg.")
    return float(fx.min()), float(fx.max()), float(fy.min()), float(fy.max())


def make_figure(projection: str = 'AIT', center: float = 0.0,
                lat_center: float = 0.0, direction: str = 'sky',
                frame: str | None = None, lon_units: str = 'auto',
                theme: str = 'light', width: int = 900, height: int = 500,
                show_grid: bool = True,
                grid_lon_spacing: float | None = None,
                grid_lat_spacing: float | None = None,
                fov_deg: float | None = None,
                extent: tuple[float, float, float, float] | None = None,
                title: str | None = None) -> Any:
    """Build a :class:`plotly.graph_objects.Figure` scaffold for a sky map.

    The returned figure has no axis chrome (no ticks / numbers /
    axis lines), equal aspect ratio, and a background appropriate for
    the chosen theme. Users layer their own traces on top via
    :func:`add_scatter`, :func:`add_healpix`, the constellation
    overlays, or any custom ``go.*`` trace.

    Parameters
    ----------
    projection : str
        Projection code (any name accepted by :func:`project` — FITS
        codes ``'AIT'``, ``'MOL'``, ``'SIN'``, etc., or
        skyplothelper-extended names like ``'robinson'``,
        ``'kavrayskiy'``). Default ``'AIT'``.
    center, lat_center : float
        Projection center longitude / latitude (degrees). Default
        ``0, 0``.
    direction : str
        x-axis orientation. Default ``'sky'`` — RA increasing leftward.
        ``'geographic'`` increases rightward. Accepts aliases (``'astro'``,
        ``'geo'``, ``'earth'``, …).
    frame : str, optional
        The figure's **display frame** (``'icrs'`` / ``'galactic'`` /
        ``'ecliptic'`` / …), recorded on the figure and honored by the data
        and overlay helpers: :func:`add_scatter`, :func:`add_spherical_polygon`
        and the rest transform a :class:`~astropy.coordinates.SkyCoord` *into*
        this frame before projecting, so a galactic catalog lands correctly on
        a galactic figure rather than being read as ICRS. It also lets
        :func:`add_coord_labels` default its longitude tick units sensibly
        (hours for equatorial frames, degrees otherwise), mirroring the
        matplotlib ``make_wcs_frame`` behavior. ``None`` (default) is ICRS.
    lon_units : {'auto', 'hours', 'degrees'}
        Longitude tick units for :func:`add_coord_labels` (which reads this
        from the figure). ``'auto'`` (default) uses the ``frame`` hint (hours
        for equatorial, degrees otherwise); ``'hours'`` / ``'degrees'`` force
        it. Matches the matplotlib ``make_wcs_frame(lon_units=...)`` argument.
    theme : {'light', 'dark'}
        Color theme. Default ``'light'``. Use ``'dark'`` for plots
        meant to be viewed on screen / over dark backgrounds.
    width, height : int
        Figure dimensions in pixels. Default ``900 x 500`` (matches
        the natural ~2:1 aspect of all-sky projections).
    show_grid : bool
        Draw a lon/lat graticule on the figure. Default ``True``.
    grid_lon_spacing, grid_lat_spacing : float, optional
        Graticule spacing in degrees. ``None`` (default) means ``30`` for an
        all-sky figure, or an auto spacing (~6 lines across the field) when
        ``fov_deg`` is given — a 2° field with 30° spacing would draw no
        lines. Pass an explicit value to override either.
    fov_deg : float, optional
        Field of view in degrees for a zoomed / field-style figure, centered
        on ``(center, lat_center)``. Mirrors ``make_wcs_frame(fov_deg=...)``.
        The graticule is windowed to the field and the axes are ranged to its
        projected extent — without it, a non-all-sky projection (TAN, SIN,
        AZP, …) draws a full-sky graticule and autoscales to a useless or
        divergent extent. ``None`` (default) keeps the all-sky behavior, which
        is correct for the bounded projections (AIT, MOL, CAR, …). Mutually
        exclusive with ``extent``.
    extent : tuple of float, optional
        Explicit axis range ``(x0, x1, y0, y1)`` in projection-plane degrees,
        for when the ``fov_deg`` auto-window isn't the framing you want. Sets
        the range directly and does not window the graticule. Mutually
        exclusive with ``fov_deg``.
    title : str or None
        Optional figure title.

    Returns
    -------
    fig : plotly.graph_objects.Figure

    Examples
    --------
    >>> fig = make_figure(projection='AIT', center=180, theme='dark')
    >>> fig.show()
    >>> # An 8° gnomonic field around the Orion Nebula:
    >>> fig = make_figure(projection='TAN', center=83.8, lat_center=-5.4,
    ...                    fov_deg=8)
    """
    go = _import_plotly()
    if theme not in _THEMES:
        raise ValueError(
            f"theme must be 'light' or 'dark', got {theme!r}")
    if fov_deg is not None and extent is not None:
        raise TypeError(
            "make_figure() got both ``fov_deg`` and ``extent`` — pass only "
            "one (``fov_deg`` auto-windows a field; ``extent`` sets the "
            "projection-plane range explicitly).")
    if fov_deg is not None and fov_deg <= 0:
        raise ValueError(f"fov_deg must be positive, got {fov_deg}")
    th = _THEMES[theme]

    from ..projections.project import resolve_direction
    from ..wcs_frame import _auto_tick_spacing
    direction = resolve_direction(direction)

    # Resolve the axis range and the graticule window. ``fov_deg`` windows
    # the graticule to the field and ranges the axes to its projected extent;
    # ``extent`` ranges the axes but leaves the graticule full-sky.
    axis_range: tuple[float, float, float, float] | None = None
    if fov_deg is not None:
        axis_range = _fov_extent(projection, center, lat_center, direction,
                                 fov_deg)
        if grid_lon_spacing is None:
            grid_lon_spacing = _auto_tick_spacing(fov_deg)
        if grid_lat_spacing is None:
            grid_lat_spacing = _auto_tick_spacing(fov_deg)
    elif extent is not None:
        axis_range = (float(extent[0]), float(extent[1]),
                      float(extent[2]), float(extent[3]))
    if grid_lon_spacing is None:
        grid_lon_spacing = 30.0
    if grid_lat_spacing is None:
        grid_lat_spacing = 30.0

    fig = go.Figure()

    if show_grid:
        # Under fov_deg the graticule is windowed to the field; otherwise it
        # spans the whole sphere. A windowed lat/lon line is sampled only over
        # the field's own lon/lat span, so it doesn't streak off to the
        # all-sky extent (which for a zoomed TAN is enormous).
        if fov_deg is not None:
            half = float(fov_deg) / 2.0
            coslat = max(0.05, float(np.cos(np.radians(lat_center))))
            lon_lo, lon_hi = center - half / coslat, center + half / coslat
            lat_lo = max(-89.999, lat_center - half)
            lat_hi = min(89.999, lat_center + half)
        else:
            lon_lo, lon_hi = center - 180.0, center + 180.0
            lat_lo, lat_hi = -89.9, 89.9

        # Latitude lines (constant lat, varying lon).
        lat_lines = _grid_line_positions(lat_lo, lat_hi, grid_lat_spacing,
                                         hard_lo=-90.0, hard_hi=90.0)
        for lat in lat_lines:
            lon_grid = np.linspace(lon_lo, lon_hi, 181)
            lat_grid = np.full_like(lon_grid, lat)
            xg, yg = _project(lon_grid, lat_grid, projection=projection,
                              center=center, lat_center=lat_center,
                              direction=direction)
            if not np.isfinite(np.asarray(xg, dtype=float)).any():
                continue
            fig.add_trace(go.Scatter(
                x=xg, y=yg, mode='lines',
                line=dict(color=th['grid'], width=0.6),
                showlegend=False, hoverinfo='skip',
            ))

        # Longitude lines (constant lon, varying lat).
        if fov_deg is None:
            # Exclude the wrap meridian from the base list (hard_hi), then add
            # it back once as an explicit mirror. ``center - 180`` and
            # ``center + 180`` are the same great circle but project to
            # opposite frame edges of an all-sky projection, so it bounds the
            # frame on both sides; drawing it once would leave the far edge
            # open.
            lon_lines = _grid_line_positions(lon_lo, lon_hi, grid_lon_spacing,
                                             hard_hi=lon_hi)
            lon_lines.append(center + 180.0)
        else:
            lon_lines = _grid_line_positions(lon_lo, lon_hi, grid_lon_spacing)
        for lon in lon_lines:
            lat_grid = np.linspace(lat_lo, lat_hi, 91)
            lon_grid = np.full_like(lat_grid, lon)
            xg, yg = _project(lon_grid, lat_grid, projection=projection,
                              center=center, lat_center=lat_center,
                              direction=direction)
            # On a zoomed or hemispheric projection (TAN, SIN) the wrap
            # meridian lies outside the visible hemisphere and projects to
            # all-NaN. Skip it rather than add an empty trace.
            if not np.isfinite(np.asarray(xg, dtype=float)).any():
                continue
            fig.add_trace(go.Scatter(
                x=xg, y=yg, mode='lines',
                line=dict(color=th['grid'], width=0.6),
                showlegend=False, hoverinfo='skip',
            ))

    xaxis = dict(visible=False, showgrid=False, zeroline=False,
                 scaleanchor='y', scaleratio=1, constrain='domain')
    yaxis: dict[str, Any] = dict(visible=False, showgrid=False,
                                 zeroline=False)
    if axis_range is not None:
        # Ascending range for both directions. The east-left ('sky') vs
        # east-right ('geographic') flip is already baked into the projected
        # data by ``project()`` (its x_sign), exactly as for the all-sky
        # figures, which set no range and autoscale ascending. Reversing the
        # axis here would undo that flip and mirror a sky field east-to-west.
        x0, x1, y0, y1 = axis_range
        xaxis['range'] = [x0, x1]
        yaxis['range'] = [y0, y1]
    fig.update_layout(
        width=width, height=height,
        paper_bgcolor=th['bg'], plot_bgcolor=th['bg'],
        font=dict(color=th['fg']),
        title=dict(text=title) if title else None,
        margin=dict(l=20, r=20, t=40 if title else 20, b=20),
        showlegend=False, xaxis=xaxis, yaxis=yaxis,
    )

    # Tag the figure with its projection setup so subsequent
    # ``add_*`` calls can default to matching values without the user
    # having to repeat them. Stored on ``fig.layout.meta`` (free-form
    # dict slot plotly preserves through serialization). ``direction`` is
    # already resolved to its canonical form above.
    from ..projections.project import resolve_lon_units
    fig.update_layout(meta=dict(
        sph_projection=projection, sph_center=center,
        sph_lat_center=lat_center, sph_direction=direction,
        sph_frame=(None if frame is None else str(frame).lower()),
        sph_lon_units=resolve_lon_units(lon_units),
        # Resolved theme colors, so overlays can pick a legible foreground
        # without re-deriving it. Without these the readers downstream fell
        # back to sniffing paper_bgcolor for a leading '#0', which happened to
        # work for the built-in dark theme and failed for any other dark
        # background the user set themselves.
        sph_fg=th['fg'], sph_bg=th['bg'],
    ))
    return fig


def _meta_defaults(
    fig: Any, projection: str | None, center: float | None,
    lat_center: float | None, direction: str | None,
) -> tuple[str, float, float, str]:
    """Resolve projection kwargs against the figure's ``meta`` tag set by
    :func:`make_figure`. User-supplied values always win; otherwise
    fall back to the figure's stored config."""
    meta = (getattr(fig, 'layout', None) and getattr(fig.layout, 'meta', None)) or {}
    return (
        projection if projection is not None else meta.get('sph_projection', 'AIT'),
        center if center is not None else meta.get('sph_center', 0.0),
        lat_center if lat_center is not None else meta.get('sph_lat_center', 0.0),
        direction if direction is not None else meta.get('sph_direction', 'sky'),
    )


def _display_frame(fig: Any) -> str:
    """The coordinate frame the figure's x/y actually represent.

    ``make_figure(frame=...)`` records the frame the plotted longitudes and
    latitudes are in; overlays defined in some *other* frame (the galactic
    plane on an equatorial map, say) must be converted into it before they
    can be projected.

    Undeclared figures resolve to ICRS. That is the historical assumption
    baked into this module, and it keeps every default figure rendering
    identically — only a figure that explicitly declares a non-equatorial
    frame changes, which is precisely the case that used to be wrong.
    """
    from ..geometry._parsing import _resolve_sky_frame
    meta = (getattr(fig, 'layout', None)
            and getattr(fig.layout, 'meta', None)) or {}
    meta = meta if isinstance(meta, dict) else {}
    return _resolve_sky_frame(meta.get('sph_frame') or 'icrs')


def _to_display_deg(coords: SkyCoord, fig: Any) -> tuple[Any, Any]:
    """A SkyCoord → ``(lon, lat)`` degrees in the figure's display frame."""
    from ..geometry._parsing import _coords_to_frame_deg
    return _coords_to_frame_deg(coords, _display_frame(fig))


def _theme_fg(fig: Any, default: str = '#1a1a1a') -> str:
    """A foreground color legible against *fig*'s background.

    Prefers the theme color `make_figure` stamped on the figure. The
    background sniff is only a fallback for figures built by hand or restyled
    afterwards — it recognizes plotly's own color spellings rather than just a
    leading ``'#0'``, which misread any dark background outside the built-in
    theme as light.
    """
    meta = (getattr(fig, 'layout', None)
            and getattr(fig.layout, 'meta', None)) or {}
    if isinstance(meta, dict) and meta.get('sph_fg'):
        return str(meta['sph_fg'])
    bg = ''
    if getattr(fig, 'layout', None) is not None:
        bg = (fig.layout.paper_bgcolor or fig.layout.plot_bgcolor or '')
    return '#dcdcdc' if _is_dark_color(bg) else default


def _is_dark_color(color: Any) -> bool:
    """Is *color* dark enough to need light ink on top?

    Luminance rather than a string prefix, so ``'#1D1C1A'`` and
    ``'rgb(20,20,20)'`` are recognized as dark like any other dark color.
    """
    if not color:
        return False
    try:
        from matplotlib.colors import to_rgb
        text = str(color).strip().lower()
        if text.startswith('rgb'):
            nums = [float(v) for v in
                    text[text.index('(') + 1:text.index(')')].split(',')[:3]]
            r, g, b = (v / 255.0 for v in nums)
        else:
            r, g, b = to_rgb(text)
    except Exception:
        return False
    return (0.299 * r + 0.587 * g + 0.114 * b) < 0.4


def _hover_labels(fig: Any) -> tuple[str, str]:
    """Short ``(lon, lat)`` names for this figure's frame, e.g. ``('l', 'b')``.

    Hover text used to say ``RA``/``Dec`` unconditionally. That was harmless
    while the display frame was always equatorial, but the figure's ``frame=``
    is now honored, so a galactic figure really does plot galactic longitude —
    and calling it RA mislabels it.
    """
    from ..constants import frame_short_labels
    meta = (getattr(fig, 'layout', None)
            and getattr(fig.layout, 'meta', None)) or {}
    meta = meta if isinstance(meta, dict) else {}
    return frame_short_labels(meta.get('sph_frame') or 'icrs')


def _default_hover(fig: Any, prefix: str = '', extra: str = '') -> str:
    """The standard two-line coordinate hover, labeled for *fig*'s frame."""
    lon_name, lat_name = _hover_labels(fig)
    return (f"{prefix}"
            f"{lon_name}: %{{customdata[0]:.3f}}°<br>"
            f"{lat_name}: %{{customdata[1]:.3f}}°"
            f"{extra}<extra></extra>")


def _resolve_lonlat(fig: Any, lons: SkyCoord | npt.ArrayLike, lats: npt.ArrayLike | None,
                    caller: str) -> tuple[Any, Any]:
    """``(SkyCoord | lons, lats)`` → degrees in the figure's display frame.

    The plotly twin of the matplotlib wrappers' ``_resolve``: a SkyCoord is
    converted into the figure's frame, while bare numbers are taken at face
    value (they are *already* display coordinates). Shared parsing lives in
    ``geometry._parsing`` so both backends agree on what a coordinate
    argument may be, and on the error text when it is malformed.
    """
    from ..geometry._parsing import _coords_or_arrays_deg
    return _coords_or_arrays_deg(lons, lats, _display_frame(fig), caller)


def _overlay_projector(
    fig: Any, projection: str, center: float, lat_center: float,
    direction: str,
) -> Any:
    """The :class:`Projector` a standalone overlay helper projects through.

    A FITS figure (carrying ``sph_wcs_header`` in meta) yields its WCS
    pixel / offset projector so the overlay lands on the displayed image;
    any other figure yields the all-sky :class:`SkyplothelperProjector`
    built from the already-resolved projection kwargs (so an explicit
    ``projection=`` / ``center=`` override is honored). Seam handling is the
    projector's own concern — the all-sky one splits at the wrap meridian,
    the seamless FITS frames don't — so the helpers project through
    ``project_points`` / ``project_polyline`` / ``project_polygon_pieces``
    without threading projection kwargs through each call."""
    meta = (getattr(fig, 'layout', None) and getattr(fig.layout, 'meta', None)) or {}
    if meta.get('sph_wcs_header'):
        from .fits import _fits_projector_from_figure
        return _fits_projector_from_figure(fig)
    from .projector import SkyplothelperProjector
    return SkyplothelperProjector(projection=projection, center=center,
                                  lat_center=lat_center, direction=direction)


# -- add_scatter -------------------------------------------------------------

# ``'auto'`` rather than a literal template string: the labels depend on the
# figure's frame, which isn't known until call time.
_DEFAULT_HOVER = 'auto'


def add_scatter(fig: Any, lons: SkyCoord | npt.ArrayLike, lats: npt.ArrayLike | None = None, *,
                projection: str | None = None, center: float | None = None,
                lat_center: float | None = None, direction: str | None = None,
                hovertemplate: str | None = 'auto',
                customdata: npt.ArrayLike | None = None,
                mode: str = 'markers', name: str | None = None,
                **trace_kwargs: Any) -> Any:
    """Project ``(lons, lats)`` and add a ``go.Scatter`` trace to ``fig``.

    By default the hover shows projected (lon, lat) — formatted as
    RA / Dec — via plotly's ``customdata`` channel. Callers can pass
    a different ``hovertemplate`` and/or ``customdata`` array to show
    other per-point info (e.g. source name, flux, redshift). When
    ``customdata`` is supplied it must be a 2D array — column 0 is
    interpreted as longitude, column 1 as latitude, and any further
    columns are appended for use in the hover template.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Target figure (typically built by :func:`make_figure`).
    lons, lats : array-like
        Sky coordinates in degrees.
    projection, center, lat_center, direction : optional
        Projection kwargs. Default to the values recorded on the
        figure by :func:`make_figure`; pass to override.
    hovertemplate : str
        Plotly hover template. ``'auto'`` (default) labels the
        coordinates for the figure's frame — ``RA``/``Dec`` on an
        equatorial figure, ``l``/``b`` on a galactic one.
    customdata : array-like, optional
        Per-point auxiliary data for the hover template. If ``None``,
        ``customdata = np.column_stack([lons, lats])`` so the
        default hover template Just Works.
    mode : str
        ``go.Scatter`` mode. Default ``'markers'``.
    name : str, optional
        Trace name (for the legend, if enabled).
    **trace_kwargs
        Forwarded to ``go.Scatter`` — useful: ``marker={'size': 4,
        'color': 'red'}``, ``line={'width': 1, 'color': 'gold'}``, etc.

    Returns
    -------
    trace : plotly.graph_objects.Scatter
        The added trace (also accessible via ``fig.data[-1]``).
    """
    go = _import_plotly()
    projection, center, lat_center, direction = _meta_defaults(
        fig, projection, center, lat_center, direction)
    lons, lats = _resolve_lonlat(fig, lons, lats, 'add_scatter')
    if hovertemplate == 'auto':
        hovertemplate = _default_hover(fig)
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    x, y = _project(lons, lats, projection=projection, center=center,
                    lat_center=lat_center, direction=direction)
    if customdata is None:
        customdata = np.column_stack([lons.ravel(), lats.ravel()])
    scatter_kw = dict(
        x=x.ravel(), y=y.ravel(), mode=mode, name=name,
        customdata=customdata, hovertemplate=hovertemplate,
        showlegend=name is not None,
    )
    scatter_kw.update(trace_kwargs)   # let the caller override any default
    trace = go.Scatter(**scatter_kw)
    fig.add_trace(trace)
    return trace


# -- add_healpix -------------------------------------------------------------

def add_healpix(fig: Any, hp_values: npt.ArrayLike, nside: int, *,
                nest: bool = False,
                projection: str | None = None, center: float | None = None,
                lat_center: float | None = None, direction: str | None = None,
                colorscale: str | list[Any] = 'Viridis',
                vmin: float | None = None, vmax: float | None = None,
                hover_format: str | Callable[..., str] | None = None,
                tile_resolution: int | str = 'auto', line_width: float = 0,
                line_color: str | None = None,
                opacity: float = 1.0,
                add_colorbar: bool = False, cbar_title: str | None = None,
                colorbar_kwargs: dict[str, Any] | None = None) -> list[Any]:
    """Add a HEALPix map as polygon tiles with per-tile hover.

    Each non-NaN tile becomes a separate ``go.Scatter(fill='toself')``
    trace whose vertices are the projected tile corners. Hovering
    anywhere over a tile reveals its center coordinates and value via
    the ``customdata`` channel. This is the **novel feature** of the
    skyplothelper plotly module — no existing astropy-adjacent
    package provides HEALPix + plotly hover-over cleanly.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    hp_values : array-like
        HEALPix map values, shape ``(12 * nside**2,)``. NaN tiles are
        skipped.
    nside : int
        HEALPix resolution parameter (a power of two is required for
        NESTED indexing; RING accepts any valid nside).
    nest : bool
        ``True`` for NESTED indexing, ``False`` (default) for RING.
    projection, center, lat_center, direction : optional
        Projection kwargs (default to figure's metadata).
    colorscale : str or list
        Plotly colorscale name (``'Viridis'``, ``'Plasma'``, ``'Cividis'``,
        ...). Default ``'Viridis'``.
    vmin, vmax : float, optional
        Color normalization range. Default: data min / max ignoring NaN.
    hover_format : str or callable
        Per-tile hover content. Either a callable returning the text
        to show — accepting either ``(lon, lat, value)`` or
        ``(lon, lat, value, ipix)`` (arity auto-detected) — or a
        Python format string with any of the placeholders ``{lon}``,
        ``{lat}``, ``{value}``, ``{ipix}``. Default ``None`` →
        ``"<lon>: {lon:.3f}°<br><lat>: {lat:.3f}°<br>value: {value:.6g}<br>ipix: {ipix}"``,
        where the coordinate names follow the figure's frame.
        Use ``<br>`` for line breaks (plotly renders the result as HTML).
    tile_resolution : int or ``'auto'``
        Sample points per tile edge for the polygon outline. ``'auto'``
        (default) targets ~1° spacing per edge via
        ``max(2, ceil(58 / nside))`` — gives nside=4 → 15 samples/edge
        (enough to keep polar-cap tile chords flush with the AIT
        silhouette), nside=16 → 4, nside=64+ → 2. Pass an explicit
        integer to override.
    line_width : float
        Edge line width for tile polygons. Default ``0`` (no edges).
    line_color : str, optional
        Edge color for tile polygons. Default ``None`` uses each tile's
        own fill color (edges blend in); pass a color for contrasting
        tile boundaries.
    opacity : float
        Per-tile fill opacity. Default ``1.0``.
    add_colorbar : bool
        Attach a colorbar keyed to ``colorscale`` / ``vmin`` / ``vmax``.
        Default ``False``. Each tile is painted a flat ``fillcolor``,
        which carries no colorscale, so the bar rides on an extra
        invisible companion trace (appended last to the return value).
    cbar_title : str, optional
        Colorbar title. Ignored unless ``add_colorbar=True``.
    colorbar_kwargs : dict, optional
        Passed through to ``marker.colorbar``, for placement and
        styling. A horizontal bar beneath the map, for instance::

            colorbar_kwargs=dict(orientation='h', x=0.5, y=-0.05,
                                 len=0.6, thickness=14)

    Returns
    -------
    traces : list of plotly.graph_objects.Scatter
        The added traces (one per non-NaN tile), followed by the
        invisible colorbar trace when ``add_colorbar=True``.

    Notes
    -----
    Requires ``healpy`` to be importable.

    For very large maps (``nside >= 128``, i.e. ~200k tiles) plotly's
    per-trace overhead makes the figure heavy and slow to render in
    the browser. Consider downgrading to ``nside ≤ 64`` (49k tiles)
    or pre-aggregating the map before passing here.
    """
    go = _import_plotly()
    try:
        import healpy as hp
    except ImportError as exc:
        raise ImportError(
            "add_healpix requires the optional `healpy` package. "
            "Install with `pip install healpy` or "
            "`pip install skyplothelper[healpix]`."
        ) from exc

    projection, center, lat_center, direction = _meta_defaults(
        fig, projection, center, lat_center, direction)
    hp_values = np.asarray(hp_values, dtype=float)
    npix = hp_values.size
    if npix != 12 * nside ** 2:
        raise ValueError(
            f"hp_values size {npix} does not match 12 * nside**2 = "
            f"{12 * nside ** 2} for nside={nside}")

    finite = np.isfinite(hp_values)
    if vmin is None:
        vmin = float(np.min(hp_values[finite])) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.max(hp_values[finite])) if finite.any() else 1.0

    # Per-tile corner coords. ``hp.boundaries`` gives 3D unit vectors;
    # convert to lon/lat. tile_resolution → 4*step boundary points.
    step = _resolve_tile_resolution(tile_resolution, nside)
    # Tile center coords for hover display.
    cen_theta, cen_phi = hp.pix2ang(nside, np.arange(npix), nest=nest)
    cen_lon = np.degrees(cen_phi)
    cen_lat = 90.0 - np.degrees(cen_theta)

    # Plotly colorscale → callable for value → RGB.
    from plotly.colors import sample_colorscale

    # Resolve hover_format → callable or default template.
    _hover_text: Callable[[float, float, float, int], str]
    if hover_format is None:
        _lon_name, _lat_name = _hover_labels(fig)

        def _hover_text(lon: float, lat: float, val: float, ipix: int) -> str:
            return (f"{_lon_name}: {lon:.3f}°<br>{_lat_name}: {lat:.3f}°<br>"
                    f"value: {val:.6g}<br>ipix: {ipix}")
    elif callable(hover_format):
        # Accept either 3-arg ``(lon, lat, value)`` or 4-arg
        # ``(lon, lat, value, ipix)`` callbacks. Inspect once; fall back
        # to the 3-arg call shape if the signature can't be introspected
        # (e.g. a C-implemented callable).
        import inspect
        try:
            _params = inspect.signature(hover_format).parameters
            _accepts_ipix = (
                len(_params) >= 4
                or any(p.kind == inspect.Parameter.VAR_POSITIONAL
                       or p.kind == inspect.Parameter.VAR_KEYWORD
                       for p in _params.values())
            )
        except (TypeError, ValueError):
            _accepts_ipix = False
        if _accepts_ipix:
            _hover_text = hover_format
        else:
            def _hover_text(lon: float, lat: float, val: float,
                            ipix: int) -> str:
                return hover_format(lon, lat, val)
    else:
        # Treat as a Python format string with {lon}, {lat}, {value},
        # {ipix}. Unused placeholders are silently ignored by str.format.
        _fmt = str(hover_format)
        def _hover_text(lon: float, lat: float, val: float, ipix: int) -> str:
            return _fmt.format(lon=lon, lat=lat, value=val, ipix=ipix)

    traces = []
    for ipix in np.where(finite)[0]:
        value = hp_values[ipix]
        # Tile boundary as a 3xN array of unit vectors.
        verts = hp.boundaries(nside, ipix, step=step, nest=nest)
        # Convert to lon, lat.
        x_v, y_v, z_v = verts[0], verts[1], verts[2]
        r = np.sqrt(x_v ** 2 + y_v ** 2 + z_v ** 2)
        b_lat = np.degrees(np.arcsin(z_v / r))
        b_lon = np.degrees(np.arctan2(y_v, x_v)) % 360.0
        # Close the polygon.
        b_lon = np.append(b_lon, b_lon[0])
        b_lat = np.append(b_lat, b_lat[0])
        # Tiles that straddle the projection's wrap edge (longitude =
        # center ± 180°) need to be split into two sub-polygons — one
        # on each side of the wrap — otherwise their projected corners
        # land on opposite sides of the figure and ``fill='toself'``
        # paints a streak across the canvas. ``_split_polygon_at_wrap``
        # returns one or two ``(lons, lats)`` polygons; we render each.
        pieces = _split_polygon_at_wrap(b_lon, b_lat, center)
        # Map value → RGB once per tile (shared across both wrap-split
        # sub-polygons).
        norm_val = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        norm_val = float(np.clip(norm_val, 0.0, 1.0))
        rgb = sample_colorscale(colorscale, [norm_val])[0]
        # Pre-format the hover text once per tile — same string for
        # every vertex, so the hover template can use ``%{text}``
        # without per-point customdata-indexing (which doesn't work
        # under ``hoveron='fills'``).
        hover_text = _hover_text(float(cen_lon[ipix]),
                                  float(cen_lat[ipix]), float(value),
                                  int(ipix))
        for piece_lons, piece_lats in pieces:
            x, y = _project(piece_lons, piece_lats, projection=projection,
                            center=center, lat_center=lat_center,
                            direction=direction)
            trace = go.Scatter(
                x=x, y=y, mode='lines', fill='toself',
                fillcolor=rgb, opacity=opacity,
                line=dict(width=line_width,
                          color=line_color if line_color is not None else rgb),
                # ``text`` as a single string applies uniformly to
                # every vertex AND to the fill-hover. ``%{text}``
                # substitution works in both modes.
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                # Empty ``name`` prevents plotly from generating
                # ``"trace N"`` auto-labels that survive
                # ``<extra></extra>`` suppression as a small black
                # hoverlabel on ``hoveron='fills'`` traces.
                name='',
                showlegend=False,
                # ``hoveron='fills'`` (not 'fills+points') restricts
                # hover to the colored fill area. Without it, zooming
                # in close to polygon vertices triggers a per-vertex
                # hover that overlays the canvas with trace-name
                # tooltips for each nearby tile.
                hoveron='fills',
            )
            fig.add_trace(trace)
            traces.append(trace)
    if add_colorbar:
        traces.append(_add_scale_colorbar(
            fig, go, colorscale=colorscale, cmin=vmin, cmax=vmax,
            title=cbar_title, colorbar_kwargs=colorbar_kwargs))
    return traces


# -- Constellation overlays --------------------------------------------------

def _world_transform_kwargs(
    fig: Any, projection: str | None, center: float | None,
    lat_center: float | None, direction: str | None,
) -> tuple[str, float, float, str]:
    """Common boilerplate for sphering kwarg defaults from the figure."""
    return _meta_defaults(fig, projection, center, lat_center, direction)


def add_constellation_boundaries(fig: Any, *, projection: str | None = None,
                                  center: float | None = None,
                                  lat_center: float | None = None,
                                  direction: str | None = None,
                                  data_file: str | None = None,
                                  color: str = '#888888', width: float = 0.5,
                                  opacity: float = 0.5) -> Any:
    """Add IAU constellation boundary line segments as a plotly trace.

    Reuses the same data loader as
    :func:`skyplothelper.add_constellation_boundaries` on the
    matplotlib side, so the underlying segment set is identical.
    Antimeridian-crossing segments are split into two short legs
    (same logic as the mpl version) to keep the projected polyline
    on one side of the figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    projection, center, lat_center, direction : optional
        Projection kwargs (default to figure's metadata).
    data_file : str, optional
        Override the default boundary data source. See
        :func:`skyplothelper.add_constellation_boundaries`.
    color : str
        Line color. Default subtle gray ``'#888888'``.
    width : float
        Line width. Default ``0.5``.
    opacity : float
        Line opacity. Default ``0.5``.

    Returns
    -------
    trace : plotly.graph_objects.Scatter
        Single concatenated-line trace with NaN-separated segments.
    """
    go = _import_plotly()
    from ..overlays.constellations import _load_constellation_boundaries
    projection, center, lat_center, direction = _world_transform_kwargs(
        fig, projection, center, lat_center, direction)
    data = _load_constellation_boundaries(data_file)
    segments = data.get('segments', [])

    # Build a single Scatter trace with NaN-separated polylines —
    # this is much faster than one trace per segment.
    xs: list[float] = []
    ys: list[float] = []
    for seg in segments:
        ra1, dec1, ra2, dec2 = seg[0], seg[1], seg[2], seg[3]
        # Detect a crossing of the PROJECTION seam (center ± 180) — not
        # the fixed ICRS antimeridian. A segment straddling the seam on a
        # non-180 center projects to opposite frame edges and would
        # streak across the canvas; densify + wrap-split breaks it into
        # two legs at the seam edge instead.
        n1 = ((ra1 - center + 180.0) % 360.0) - 180.0
        n2 = ((ra2 - center + 180.0) % 360.0) - 180.0
        if abs(n2 - n1) > 180.0:
            # The endpoints straddle the seam — densify the SHORT way across
            # it (unwrap ra2 relative to ra1) so the points cross the seam
            # edge exactly once, where _split_polyline_at_wrap breaks them.
            # A raw linspace(ra1, ra2) would walk the long way around the
            # sphere whenever the two RAs are numerically far apart (e.g.
            # 359°->1° at center=180 → 359,335,...,1), giving small per-step
            # diffs that the wrap-splitter can't see, so the leg streaks
            # straight across the canvas.
            ra2_unwrapped = ra2
            if ra2 - ra1 > 180.0:
                ra2_unwrapped -= 360.0
            elif ra2 - ra1 < -180.0:
                ra2_unwrapped += 360.0
            dl = np.linspace(ra1, ra2_unwrapped, 16)
            db = np.linspace(dec1, dec2, 16)
            sl, sb = _split_polyline_at_wrap(dl, db, center)
            xseg, yseg = _project(sl, sb, projection=projection,
                                   center=center, lat_center=lat_center,
                                   direction=direction)
            xs.extend(list(xseg))
            xs.append(np.nan)
            ys.extend(list(yseg))
            ys.append(np.nan)
            continue
        if abs(ra2 - ra1) > 90:
            continue
        xseg, yseg = _project([ra1, ra2], [dec1, dec2],
                               projection=projection, center=center,
                               lat_center=lat_center, direction=direction)
        xs.extend([xseg[0], xseg[1], np.nan])
        ys.extend([yseg[0], yseg[1], np.nan])

    trace = go.Scatter(
        x=xs, y=ys, mode='lines',
        line=dict(color=color, width=width),
        opacity=opacity, hoverinfo='skip', showlegend=False,
    )
    fig.add_trace(trace)
    return trace


def add_constellation_lines(fig: Any, *, projection: str | None = None,
                             center: float | None = None,
                             lat_center: float | None = None,
                             direction: str | None = None,
                             constellations: Sequence[str] | None = None,
                             rank_max: int | None = None,
                             data_file: str | None = None,
                             color: str = '#C7A86A', width: float = 0.7,
                             opacity: float = 0.8) -> Any:
    """Add IAU constellation asterism (connect-the-dots) lines.

    Reuses the data loader from
    :func:`skyplothelper.add_constellation_lines`.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    projection, center, lat_center, direction : optional
    constellations : iterable of str, optional
        Restrict to these 3-letter IAU codes. Case-insensitive.
    rank_max : int, optional
        Restrict to lines with rank ≤ this value (1=brightest only).
    data_file : str, optional
    color : str
        Line color. Default warm gold ``'#C7A86A'``.
    width : float
        Line width. Default ``0.7``.
    opacity : float
        Default ``0.8``.
    """
    go = _import_plotly()
    from ..overlays.constellations import _load_constellation_lines
    projection, center, lat_center, direction = _world_transform_kwargs(
        fig, projection, center, lat_center, direction)
    data = _load_constellation_lines(data_file)
    cst = data['cst']
    rank = data['rank']
    ra = data['ra']
    dec = data['dec']
    seg_offsets = data['seg_offsets']
    cst_seg_ids = data['cst_seg_ids']

    if constellations is None:
        cst_mask = np.ones(len(cst), dtype=bool)
    else:
        wanted = {c.upper() for c in constellations}
        cst_mask = np.array([c in wanted for c in cst], dtype=bool)

    xs: list[float] = []
    ys: list[float] = []
    n_segments = len(seg_offsets) - 1
    for s in range(n_segments):
        cst_idx = int(cst_seg_ids[s])
        if not cst_mask[cst_idx]:
            continue
        if rank_max is not None and int(rank[cst_idx]) > int(rank_max):
            continue
        a = int(seg_offsets[s])
        b = int(seg_offsets[s + 1])
        if b - a < 2:
            continue
        seg_ra = np.asarray(ra[a:b], dtype=float)
        seg_dec = np.asarray(dec[a:b], dtype=float)
        # Break the polyline at PROJECTION-seam (center ± 180) crossings,
        # not the fixed ICRS antimeridian — otherwise asterism legs that
        # straddle the seam on a non-180 center streak across the canvas.
        seg_ra, seg_dec = _split_polyline_at_wrap(seg_ra, seg_dec, center)
        xseg, yseg = _project(seg_ra, seg_dec, projection=projection,
                               center=center, lat_center=lat_center,
                               direction=direction)
        xs.extend(list(xseg) + [np.nan])
        ys.extend(list(yseg) + [np.nan])

    trace = go.Scatter(
        x=xs, y=ys, mode='lines',
        line=dict(color=color, width=width),
        opacity=opacity, hoverinfo='skip', showlegend=False,
    )
    fig.add_trace(trace)
    return trace


def add_constellation_labels(fig: Any, *, projection: str | None = None,
                              center: float | None = None,
                              lat_center: float | None = None,
                              direction: str | None = None,
                              labels: str = 'abbr',
                              constellations: Sequence[str] | None = None,
                              apply_default_offsets: bool = True,
                              color: str = '#888888', fontsize: int = 11,
                              opacity: float = 0.7) -> Any:
    """Add constellation name labels as text annotations.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    projection, center, lat_center, direction : optional
    labels : {'abbr', 'name', 'both'}
        Label content. Default ``'abbr'`` (3-letter IAU code).
    constellations : iterable of str, optional
    apply_default_offsets : bool
        Apply hand-tuned positioning offsets for crowded constellations
        on the default AIT(center=180) all-sky view. Default ``True``.
    color : str
        Text color. Default subtle gray ``'#888888'``.
    fontsize : int
        Default ``11``.
    opacity : float
        Default ``0.7``.
    """
    go = _import_plotly()
    from ..overlays.constellations import (
        _CONSTELLATION_CENTERS,
        _CONSTELLATION_LABEL_OFFSETS_DEG,
        _CONSTELLATION_NAMES,
    )
    projection, center, lat_center, direction = _world_transform_kwargs(
        fig, projection, center, lat_center, direction)

    keys = ([c.upper() for c in constellations] if constellations
            else sorted(_CONSTELLATION_CENTERS.keys()))

    label_lons: list[float] = []
    label_lats: list[float] = []
    label_texts: list[str] = []
    for abbr in keys:
        if abbr not in _CONSTELLATION_CENTERS:
            continue
        ra, dec = _CONSTELLATION_CENTERS[abbr]
        if apply_default_offsets:
            dra, ddec = _CONSTELLATION_LABEL_OFFSETS_DEG.get(
                abbr, (0.0, 0.0))
            ra = (ra + dra) % 360.0
            dec = dec + ddec
        if labels == 'name':
            text = _CONSTELLATION_NAMES.get(abbr, abbr)
        elif labels == 'both':
            text = f"{abbr} {_CONSTELLATION_NAMES.get(abbr, '')}"
        else:
            text = abbr
        label_lons.append(ra)
        label_lats.append(dec)
        label_texts.append(text)

    xs, ys = _project(label_lons, label_lats, projection=projection,
                       center=center, lat_center=lat_center,
                       direction=direction)
    trace = go.Scatter(
        x=xs, y=ys, mode='text', text=label_texts,
        textfont=dict(color=color, size=fontsize),
        opacity=opacity, hoverinfo='skip', showlegend=False,
    )
    fig.add_trace(trace)
    return trace


# -- Great-circle / plane overlays -----------------------------------------

# Frame names accepted by add_great_circle, matching the matplotlib side.
_GREAT_CIRCLE_FRAMES = {'galactic', 'ecliptic', 'geocentrictrueecliptic',
                        'supergalactic', 'pole', 'icrs', 'fk5', 'fk4'}


def add_great_circle(fig: Any, *, frame: str = 'galactic',
                     lat_offset: float = 0.0,
                     pole_lon: float = 0.0, pole_lat: float = 90.0,
                     n_points: int = 500,
                     projection: str | None = None, center: float | None = None,
                     lat_center: float | None = None,
                     direction: str | None = None,
                     color: str = 'dimgray', width: float = 1.0,
                     opacity: float = 1.0,
                     name: str | None = None, hover: bool | str = False,
                     **trace_kwargs: Any) -> Any:
    """Draw a great circle (or small circle at a latitude offset).

    Plotly counterpart to :func:`skyplothelper.add_great_circle`. The
    circle is sampled in the source ``frame`` at ``lat = lat_offset``
    (so ``lat_offset = 0`` is the great circle itself; nonzero values
    give parallels of that frame). The samples are then projected via
    :func:`sph.project` into the figure's projection and emitted as a
    single ``go.Scatter(mode='lines')`` trace with antimeridian /
    wrap-edge crossings split via NaN-separator polyline pieces.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    frame : str
        Source frame the circle is defined in. Accepts ``'galactic'``,
        ``'ecliptic'`` (alias ``'geocentrictrueecliptic'``),
        ``'supergalactic'``, ``'icrs'`` / ``'fk5'`` / ``'fk4'``, or
        ``'pole'`` (custom — define via ``pole_lon`` / ``pole_lat``).
        Default ``'galactic'``.
    lat_offset : float
        Latitude offset (degrees) within the source frame.
        ``0`` (default) traces the great circle; nonzero gives a
        small circle (parallel).
    pole_lon, pole_lat : float
        Pole of the custom great circle in ICRS degrees. Only used
        when ``frame='pole'``.
    n_points : int
        Number of samples along the circle. Default ``500``.
    projection, center, lat_center, direction : optional
        Projection kwargs (default to the figure's metadata).
    color : str
    width : float
    opacity : float
    name : str, optional
        Trace name (shows in legend if not ``None``).
    hover : False / True / str
        Hover behavior. ``False`` (default) emits ``hoverinfo='skip'``
        — no hover info, matching the original silent behavior.
        ``True`` enables an auto template showing the trace name (when
        set) and per-vertex RA/Dec. Pass a custom hovertemplate string
        for arbitrary content (``%{customdata[0]}`` / ``%{customdata[1]}``
        index lon and lat; ``<extra></extra>`` is auto-appended if not
        present).
    **trace_kwargs
        Forwarded to ``go.Scatter``.

    Returns
    -------
    trace : plotly.graph_objects.Scatter
    """
    go = _import_plotly()
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    projection, center, lat_center, direction = _meta_defaults(
        fig, projection, center, lat_center, direction)

    frame_key = str(frame).lower()
    if frame_key not in _GREAT_CIRCLE_FRAMES:
        raise ValueError(
            f"Unknown frame {frame!r}. Use one of "
            f"{sorted(_GREAT_CIRCLE_FRAMES)}.")

    lons_sample = np.linspace(0, 360, n_points, endpoint=False)
    lats_sample = np.full(n_points, float(lat_offset))

    if frame_key == 'galactic':
        coords = SkyCoord(l=lons_sample, b=lats_sample,
                          frame='galactic', unit='deg')
    elif frame_key in ('ecliptic', 'geocentrictrueecliptic'):
        coords = SkyCoord(lon=lons_sample, lat=lats_sample,
                          frame='geocentrictrueecliptic', unit='deg')
    elif frame_key == 'supergalactic':
        coords = SkyCoord(sgl=lons_sample, sgb=lats_sample,
                          frame='supergalactic', unit='deg')
    elif frame_key == 'pole':
        pole = SkyCoord(pole_lon, pole_lat, unit='deg',
                        frame=_display_frame(fig))
        pa = np.linspace(0, 360, n_points, endpoint=False)
        sep = 90. - lat_offset
        coords = pole.directional_offset_by(pa * u.deg, sep * u.deg)
    else:
        # icrs / fk5 / fk4 — input lons/lats already in destination frame
        coords = SkyCoord(lons_sample, lats_sample, frame=frame_key,
                          unit='deg')

    # Into the figure's display frame — NOT unconditionally ICRS, which used
    # to draw the galactic plane as a sinusoid on a galactic map.
    plot_lon, plot_lat = _to_display_deg(coords, fig)

    # Only a GREAT circle (lat_offset == 0) is sorted by longitude: it's
    # monotonic in RA, so the sort just rotates the wrap seam to the array ends.
    # A SMALL circle (lat_offset != 0) that doesn't enclose the pole is
    # double-valued in RA, so sorting would interleave its upper/lower branches
    # into a zig-zag fan — it must stay in the linspace (or directional_offset_by
    # position-angle) path order, where consecutive samples are spatially
    # adjacent. split_polyline handles the wrap crossings from either order.
    if lat_offset == 0:
        wrapped = ((plot_lon - center + 180.0) % 360.0) - 180.0 + center
        order = np.argsort(wrapped)
        plot_lon, plot_lat = wrapped[order], plot_lat[order]

    # Insert NaN at wrap-edge crossings → polyline breaks cleanly (a no-op
    # on a seamless FITS frame). The split coords also feed the per-vertex
    # hover customdata below, so they must align with the projected (x, y).
    proj = _overlay_projector(fig, projection, center, lat_center, direction)
    seg_lons, seg_lats = proj.split_polyline(plot_lon, plot_lat)
    x, y = proj.project_points(seg_lons, seg_lats)
    hovertemplate, hoverinfo = _resolve_hover_line(hover, name, fig)
    customdata = (np.column_stack([seg_lons, seg_lats])
                  if hovertemplate is not None else None)
    trace_kw: dict[str, Any] = dict(
        x=x, y=y, mode='lines',
        line=dict(color=color, width=width),
        opacity=opacity, name=name,
        showlegend=name is not None,
    )
    trace_kw.update(trace_kwargs)
    if hovertemplate is not None:
        trace_kw['hovertemplate'] = hovertemplate
        trace_kw['customdata'] = customdata
    else:
        trace_kw['hoverinfo'] = hoverinfo
    trace = go.Scatter(**trace_kw)
    fig.add_trace(trace)
    return trace


_PLANE_DEFAULTS = {
    'galactic':      ('dimgray',    'Galactic plane'),
    'ecliptic':      ('goldenrod',  'Ecliptic plane'),
    'supergalactic': ('steelblue',  'Supergalactic plane'),
}


def add_plane_overlay(fig: Any, *, plane: str = 'galactic',
                      color: str | None = None, width: float = 1.0,
                      opacity: float = 1.0,
                      name: str | None = None,
                      parallels: Sequence[float] | None = None,
                      parallel_opacity: float = 0.4,
                      parallel_width: float | None = None,
                      projection: str | None = None, center: float | None = None,
                      lat_center: float | None = None,
                      direction: str | None = None,
                      hover: bool | str = False,
                      **trace_kwargs: Any) -> list[Any]:
    """Add a coordinate-plane overlay (galactic / ecliptic / supergalactic)
    with optional parallels.

    Plotly counterpart to :func:`skyplothelper.add_plane_overlay`. A
    thin wrapper around :func:`add_great_circle` that supplies sensible
    color and label defaults for the named planes, and optionally
    draws additional parallels at the requested latitude offsets.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    plane : {'galactic', 'ecliptic', 'supergalactic'}
        Which plane to draw.
    color : str, optional
        Line color. ``None`` (default) uses the per-plane default
        (dimgray / goldenrod / steelblue).
    width, opacity : float
    name : str, optional
        Trace name (default: ``"Galactic plane"`` / etc.).
    parallels : iterable of float, optional
        Latitude offsets for additional parallel circles
        (e.g. ``[-10, 10]`` for ±10° galactic latitudes).
    parallel_opacity : float
        Opacity for the parallels. Default ``0.4``.
    parallel_width : float, optional
        Line width for parallels. ``None`` (default) uses ``width``.
    projection, center, lat_center, direction : optional
        Projection kwargs (default to figure metadata).
    hover : False / True / str
        Forwarded to each underlying :func:`add_great_circle` call.
        When ``hover=True``, parallels receive a name that includes
        their latitude offset (e.g. ``"Galactic plane (b=+10.0°)"``) so
        the hoverbox identifies which parallel is under the cursor.
    **trace_kwargs
        Forwarded to ``go.Scatter``.

    Returns
    -------
    traces : list of go.Scatter
        Single trace if no parallels; otherwise main + parallels.
    """
    plane_key = str(plane).lower()
    default_color, default_name = _PLANE_DEFAULTS.get(
        plane_key, ('dimgray', f'{plane} plane'))
    use_color = color if color is not None else default_color
    use_name = name if name is not None else default_name

    traces = []
    traces.append(add_great_circle(
        fig, frame=plane_key, lat_offset=0.0,
        color=use_color, width=width, opacity=opacity,
        name=use_name, hover=hover,
        projection=projection, center=center,
        lat_center=lat_center, direction=direction,
        **trace_kwargs,
    ))
    if parallels:
        pw = parallel_width if parallel_width is not None else width
        for off in parallels:
            # When auto-hover is on, give each parallel a distinguishing
            # name so the hoverbox identifies its latitude offset.
            parallel_name = (f"{use_name} (b={float(off):+.1f}°)"
                              if hover is True else None)
            traces.append(add_great_circle(
                fig, frame=plane_key, lat_offset=float(off),
                color=use_color, width=pw, opacity=parallel_opacity,
                name=parallel_name, hover=hover,
                projection=projection, center=center,
                lat_center=lat_center, direction=direction,
                **trace_kwargs,
            ))
    return traces


# -- Geodesic / small circles ----------------------------------------------

def add_geodesic_circle(fig: Any, lon: SkyCoord | float,
                        lat: float | None = None,
                         radius_deg: float | None = None, *,
                         resolution: int = 200,
                         projection: str | None = None,
                         center: float | None = None,
                         lat_center: float | None = None,
                         direction: str | None = None,
                         color: str = 'dimgray', width: float = 1.0,
                         opacity: float = 1.0,
                         fill: bool = False, fillcolor: str | None = None,
                         name: str | None = None, hover: bool | str = False,
                         **trace_kwargs: Any) -> list[Any]:
    """Draw a geodesic (small) circle of given angular radius.

    Plotly counterpart to :func:`skyplothelper.add_geodesic_circle`.
    Samples the circle via :func:`skyplothelper.geometry.shapes.geodesic_circle`
    (the same primitive the matplotlib side uses), projects each
    boundary sample, and emits either an open polyline or a filled
    polygon trace with wrap-edge splitting.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    lon, lat : float
        Center coordinates in degrees.
    radius_deg : float
        Angular radius in degrees.
    resolution : int
        Number of boundary samples. Default ``200``.
    projection, center, lat_center, direction : optional
        Projection kwargs (default to figure metadata).
    color : str
        Edge / line color.
    width : float
    opacity : float
    fill : bool
        ``True`` to emit a filled polygon (``go.Scatter(fill='toself')``);
        ``False`` (default) emits an open line outline.
    fillcolor : str, optional
        Fill color when ``fill=True``. Default ``None`` (plotly auto).
    name : str, optional
    hover : False / True / str
        Hover behavior. ``False`` (default) emits ``hoverinfo='skip'``.
        ``True`` enables an auto template — for ``fill=False`` lines,
        name + per-vertex RA/Dec; for ``fill=True`` discs, name +
        center coords + angular radius (single label, shown anywhere
        over the fill via ``hoveron='fills'``). A string is used
        directly as the hovertemplate (or as fill ``text``).
    **trace_kwargs
        Forwarded to ``go.Scatter``.

    Returns
    -------
    traces : list of go.Scatter
        Single trace if the circle doesn't straddle the wrap edge;
        two traces if it does (wrap-split sub-polygons / sub-polylines).
    """
    go = _import_plotly()
    from ..geometry.shapes import geodesic_circle
    projection, center, lat_center, direction = _meta_defaults(
        fig, projection, center, lat_center, direction)

    # A SkyCoord occupies the single ``lon`` slot, so the radius — the one
    # argument that follows — shifts up into ``lat``.
    if hasattr(lon, 'transform_to'):
        if radius_deg is None:
            lon, lat, radius_deg = lon, None, lat
        lon, lat = _to_display_deg(lon, fig)
    if lat is None or radius_deg is None:
        raise TypeError(
            "add_geodesic_circle: needs a center and a radius — either "
            "(lon, lat, radius_deg) in degrees, or (SkyCoord, radius_deg).")

    proj = _overlay_projector(fig, projection, center, lat_center, direction)
    lons, lats = geodesic_circle(float(lon), float(lat),
                                  float(radius_deg), resolution)
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    # Close the boundary for the polygon path; close-or-not for the
    # polyline path doesn't matter visually since the last sample is
    # adjacent to the first.
    lons_closed = np.append(lons, lons[0])
    lats_closed = np.append(lats, lats[0])

    traces = []
    if fill:
        auto_detail = (f"center: ({float(lon):.3f}°, {float(lat):.3f}°)<br>"
                       f"radius: {float(radius_deg):.3f}°")
        fill_text, fill_template, fill_hoverinfo = _resolve_hover_fill(
            hover, name, auto_detail=auto_detail)
        for x, y in proj.project_polygon_pieces(lons_closed, lats_closed):
            trace_kw: dict[str, Any] = dict(
                x=x, y=y, mode='lines', fill='toself',
                fillcolor=fillcolor,
                line=dict(color=color, width=width),
                opacity=opacity, name=name,
                showlegend=name is not None,
                hoveron='fills',
            )
            trace_kw.update(trace_kwargs)
            if fill_template is not None:
                trace_kw['text'] = fill_text
                trace_kw['hovertemplate'] = fill_template
                # Empty trace name suppresses ``"trace N"`` auto-labels
                # that survive ``<extra></extra>`` under ``hoveron='fills'``.
                trace_kw['name'] = ''
                trace_kw['showlegend'] = False
            else:
                trace_kw['hoverinfo'] = fill_hoverinfo
            trace = go.Scatter(**trace_kw)
            fig.add_trace(trace)
            traces.append(trace)
    else:
        seg_lons, seg_lats = proj.split_polyline(lons_closed, lats_closed)
        x, y = proj.project_points(seg_lons, seg_lats)
        hovertemplate, hoverinfo = _resolve_hover_line(hover, name, fig)
        trace_kw = dict(
            x=x, y=y, mode='lines',
            line=dict(color=color, width=width),
            opacity=opacity, name=name,
            showlegend=name is not None,
        )
        trace_kw.update(trace_kwargs)
        if hovertemplate is not None:
            trace_kw['hovertemplate'] = hovertemplate
            trace_kw['customdata'] = np.column_stack([seg_lons, seg_lats])
        else:
            trace_kw['hoverinfo'] = hoverinfo
        trace = go.Scatter(**trace_kw)
        fig.add_trace(trace)
        traces.append(trace)
    return traces


# -- Spherical polygon -----------------------------------------------------

def add_spherical_polygon(fig: Any, lons: SkyCoord | npt.ArrayLike,
                           lats: npt.ArrayLike | None = None, *,
                           resolution: int = 100,
                           geodesic: bool | str = 'auto',
                           geodesic_threshold: float = 10.0,
                           projection: str | None = None,
                           center: float | None = None,
                           lat_center: float | None = None,
                           direction: str | None = None,
                           color: str = 'dimgray', width: float = 1.0,
                           opacity: float = 1.0,
                           fill: bool = True, fillcolor: str | None = None,
                           name: str | None = None, hover: bool | str = False,
                           **trace_kwargs: Any) -> list[Any]:
    """Render an arbitrary spherical polygon, with wrap-edge splitting.

    Plotly counterpart to :func:`skyplothelper.add_spherical_polygon`.
    Densifies each polygon edge via
    :func:`skyplothelper.geometry._densify._densify_polygon_edges` so
    that edges follow projection curvature correctly, then splits at
    the projection's wrap edge into one or two sub-polygons and emits
    a ``go.Scatter(fill='toself')`` trace per piece.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    lons, lats : array-like
        Polygon vertices in degrees. Order matters (CCW is the
        conventional positive orientation on a map but it's not
        enforced — the fill will be ambiguous for self-intersecting
        polygons).
    resolution : int
        Edge densification target — the higher, the more curved-edge
        accuracy. Default ``100``. Same semantics as the matplotlib
        helper.
    geodesic : {'auto', True, False}
        Whether to interpret edges as geodesics (great-circle arcs)
        rather than rhumb-line / linear interpolation. ``'auto'``
        (default) picks geodesic for polygons whose largest edge
        exceeds ``geodesic_threshold`` degrees.
    geodesic_threshold : float
        Edge-length (degrees) threshold for ``geodesic='auto'``.
        Default ``10.0``.
    projection, center, lat_center, direction : optional
        Projection kwargs (default to figure metadata).
    color : str
        Edge color.
    width : float
        Edge line width. Use ``width=0`` to suppress the edge entirely.
    opacity : float
        Per-piece alpha.
    fill : bool
        ``True`` (default) emits a filled polygon; ``False`` emits an
        open polyline (boundary only).
    fillcolor : str, optional
        Polygon fill color. Default ``None`` (plotly auto, uses
        ``color`` faded).
    name : str, optional
    hover : False / True / str
        Hover behavior. ``False`` (default) emits ``hoverinfo='skip'``.
        ``True`` enables an auto template — name only (no derived
        per-vertex info beyond RA/Dec on outline mode). A string is
        used directly as the hovertemplate (or as fill ``text``).
    **trace_kwargs
        Forwarded to ``go.Scatter``.

    Returns
    -------
    traces : list of go.Scatter
        One trace if the polygon doesn't straddle the wrap edge;
        two traces if it does.

    Examples
    --------
    A spherical triangle from the celestial north pole down to two
    points on the equator::

        sph.plotly.add_spherical_polygon(
            fig,
            lons=[0, 60, 30, 0],
            lats=[89, 0, 0, 89],
            fillcolor='rgba(255,200,100,0.4)',
            color='orange', width=1.5)
    """
    go = _import_plotly()
    from ..geometry._densify import _densify_polygon_edges
    projection, center, lat_center, direction = _meta_defaults(
        fig, projection, center, lat_center, direction)

    lons, lats = _resolve_lonlat(fig, lons, lats, 'add_spherical_polygon')
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if lons.shape != lats.shape:
        raise ValueError(
            f"lons and lats must have the same shape, got "
            f"{lons.shape} vs {lats.shape}")
    # Ensure the polygon is closed (last vertex == first).
    if not (lons[0] == lons[-1] and lats[0] == lats[-1]):
        lons = np.append(lons, lons[0])
        lats = np.append(lats, lats[0])

    # Edge densification.
    if geodesic == 'auto':
        d_lon = np.diff(lons)
        d_lat = np.diff(lats)
        max_edge = float(np.max(np.hypot(d_lon, d_lat))) if len(d_lon) else 0
        use_geodesic = max_edge >= float(geodesic_threshold)
    else:
        use_geodesic = bool(geodesic)
    dense_lons, dense_lats = _densify_polygon_edges(
        lons, lats, resolution=int(resolution), geodesic=use_geodesic)

    if fill:
        pieces = _split_polygon_at_wrap(dense_lons, dense_lats, center)
        mode = 'lines'
        fill_text, fill_template, fill_hoverinfo = _resolve_hover_fill(
            hover, name)
    else:
        # Open polyline: NaN-separate at wrap crossings, single piece.
        pl_lons, pl_lats = _split_polyline_at_wrap(dense_lons, dense_lats,
                                                    center)
        pieces = [(pl_lons, pl_lats)]
        mode = 'lines'
        line_template, line_hoverinfo = _resolve_hover_line(hover, name, fig)

    traces = []
    for piece_lons, piece_lats in pieces:
        x, y = _project(piece_lons, piece_lats,
                        projection=projection, center=center,
                        lat_center=lat_center, direction=direction)
        if fill:
            trace_kw: dict[str, Any] = dict(
                x=x, y=y, mode=mode, fill='toself',
                fillcolor=fillcolor,
                line=dict(color=color, width=width),
                opacity=opacity, name=name,
                showlegend=name is not None,
                hoveron='fills',
            )
            trace_kw.update(trace_kwargs)
            if fill_template is not None:
                trace_kw['text'] = fill_text
                trace_kw['hovertemplate'] = fill_template
                # Suppress ``"trace N"`` label that survives ``<extra></extra>``
                # under ``hoveron='fills'``.
                trace_kw['name'] = ''
                trace_kw['showlegend'] = False
            else:
                trace_kw['hoverinfo'] = fill_hoverinfo
            trace = go.Scatter(**trace_kw)
        else:
            trace_kw = dict(
                x=x, y=y, mode=mode,
                line=dict(color=color, width=width),
                opacity=opacity, name=name,
                showlegend=name is not None,
            )
            trace_kw.update(trace_kwargs)
            if line_template is not None:
                trace_kw['hovertemplate'] = line_template
                trace_kw['customdata'] = np.column_stack(
                    [piece_lons, piece_lats])
            else:
                trace_kw['hoverinfo'] = line_hoverinfo
            trace = go.Scatter(**trace_kw)
        fig.add_trace(trace)
        traces.append(trace)
    return traces


# -- Constellation / box / band wrappers ------------------------------------

def add_constellation_polygon(fig: Any, constellation: str, *,
                                step_deg: float = 0.5,
                                projection: str | None = None,
                                center: float | None = None,
                                lat_center: float | None = None,
                                direction: str | None = None,
                                color: str = 'steelblue', width: float = 1.0,
                                opacity: float = 0.25,
                                fill: bool = True, fillcolor: str | None = None,
                                name: str | None = None,
                                hover: bool | str = False,
                                **trace_kwargs: Any) -> list[Any]:
    """Fill a single IAU constellation as a closed polygon.

    Plotly counterpart to
    :func:`skyplothelper.add_constellation_polygon`. Reuses the same
    bundled corner-list data and densification (parallel/meridian
    walker at ``step_deg``) as the matplotlib side, and delegates each
    constituent polygon to :func:`add_spherical_polygon` for rendering
    — so wrap-edge splitting and frame-curve following are inherited
    automatically. Serpens has two polygons (Caput + Cauda) and both
    are drawn under a single call.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    constellation : str
        IAU 3-letter abbreviation (case-insensitive). Use
        :func:`skyplothelper.list_constellations` to enumerate.
    step_deg : float
        Edge densification step in degrees. Default ``0.5°`` (matches
        the mpl-side constellation boundary helper).
    projection, center, lat_center, direction : optional
        Projection kwargs (default to figure metadata).
    color : str
        Outline color.
    width : float
        Outline line width. Use ``width=0`` to suppress the outline.
    opacity : float
        Per-piece alpha. Default ``0.25`` — translucent highlight.
    fill : bool
        ``True`` (default) emits filled polygons; ``False`` emits
        outlines only.
    fillcolor : str, optional
        Plotly fill color (e.g. ``'rgba(70,130,180,0.25)'``). Default
        ``None`` (plotly auto, derived from ``color``).
    name : str, optional
        Trace name (used by hover/legend).
    hover : False / True / str
        Hover behavior — see :func:`add_spherical_polygon`.
    **trace_kwargs
        Forwarded to ``go.Scatter``.

    Returns
    -------
    traces : list of plotly.graph_objects.Scatter
        Concatenated traces across all polygons (usually 1; 2 for
        Serpens; potentially more once wrap-edge splitting kicks in).

    Raises
    ------
    KeyError
        If ``constellation`` doesn't match any IAU code.
    """
    from ..overlays.constellations import _load_constellation_polygons

    polygons = _load_constellation_polygons(step_deg=step_deg)
    key = str(constellation).upper()
    if key not in polygons:
        raise KeyError(
            f"add_constellation_polygon: unknown IAU code "
            f"{constellation!r}.")

    traces = []
    for lons, lats in polygons[key]:
        traces.extend(add_spherical_polygon(
            fig, lons, lats,
            resolution=1, geodesic=False,
            projection=projection, center=center,
            lat_center=lat_center, direction=direction,
            color=color, width=width, opacity=opacity,
            fill=fill, fillcolor=fillcolor,
            name=name, hover=hover,
            **trace_kwargs,
        ))
    return traces


def add_lonlat_box(fig: Any, lat_min: float, lat_max: float,
                    lon_min: float, lon_max: float, *,
                    frame: str = 'galactic', resolution: int = 100,
                    projection: str | None = None, center: float | None = None,
                    lat_center: float | None = None,
                    direction: str | None = None,
                    color: str = 'steelblue', width: float = 1.0,
                    opacity: float = 0.25,
                    fill: bool = True, fillcolor: str | None = None,
                    name: str | None = None, hover: bool | str = False,
                    **trace_kwargs: Any) -> list[Any]:
    """Add a closed lon/lat-aligned box defined in another coordinate frame.

    Plotly counterpart to :func:`skyplothelper.add_lonlat_box`. Builds
    the four-edge box outline in ``frame``, densifying each
    lon-constant edge at ``resolution`` points so antimeridian / pole /
    wrap handling stays well-behaved after the ICRS conversion, then
    delegates to :func:`add_spherical_polygon`.

    Useful for surveys whose footprint is most naturally described as
    a lon/lat rectangle in a non-axes frame — e.g. eROSITA's western
    galactic hemisphere (``l=180..360, b=-90..+90`` in galactic).

    Polar-touching edges (``lat_max >= 89.9`` / ``lat_min <= -89.9``)
    collapse to a point and are dropped from the outline walk.
    Longitude wrap (``lon_max < lon_min``) is normalized so the box is
    the slice the user intended, not its complement.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    lat_min, lat_max : float
        Latitude limits in degrees, in ``frame``.
    lon_min, lon_max : float
        Longitude limits in degrees, in ``frame``. ``lon_max < lon_min``
        is interpreted as a wraparound box (``lon_max`` bumped by 360°).
    frame : str
        Source frame for the box: ``'galactic'``, ``'ecliptic'`` (alias
        for ``'geocentrictrueecliptic'``), ``'icrs'``, ``'fk5'``, etc.
    resolution : int
        Samples along each lon-constant edge. Default ``100``.
    projection, center, lat_center, direction : optional
        Projection kwargs (default to figure metadata).
    color : str
        Outline color.
    width : float
        Outline line width.
    opacity : float
        Per-piece alpha. Default ``0.25``.
    fill : bool
        ``True`` (default) emits a filled polygon; ``False`` emits
        outline only.
    fillcolor : str, optional
        Plotly fill color.
    name : str, optional
    hover : False / True / str
    **trace_kwargs
        Forwarded to ``go.Scatter``.

    Returns
    -------
    traces : list of plotly.graph_objects.Scatter
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    frame_key = 'geocentrictrueecliptic' if str(frame).lower() == 'ecliptic' \
        else str(frame).lower()

    lat_lo = float(lat_min)
    lat_hi = float(lat_max)
    if lat_lo >= lat_hi:
        raise ValueError("lat_min must be less than lat_max")
    lon_lo = float(lon_min)
    lon_hi = float(lon_max)
    if lon_hi < lon_lo:
        lon_hi += 360.0  # wraparound box

    # Walk the box: bottom edge L→R, right edge B→T, top edge R→L,
    # left edge T→B. Collapse polar edges to a single point so the
    # outline doesn't trace around the pole.
    n = max(2, int(resolution))
    bot_lons = np.linspace(lon_lo, lon_hi, n)
    top_lons = np.linspace(lon_hi, lon_lo, n)
    bot_lats = np.full(n, lat_lo)
    top_lats = np.full(n, lat_hi)

    pieces: list[tuple[np.ndarray, np.ndarray]] = []
    if lat_lo > -89.9:
        pieces.append((bot_lons, bot_lats))
    else:
        pieces.append((np.array([lon_lo]), np.array([lat_lo])))
    # right edge — straight vertical at lon_hi, two endpoints suffice
    pieces.append((np.array([lon_hi, lon_hi]), np.array([lat_lo, lat_hi])))
    if lat_hi < 89.9:
        pieces.append((top_lons, top_lats))
    else:
        pieces.append((np.array([lon_hi]), np.array([lat_hi])))
    # left edge — straight vertical at lon_lo, close back to start
    pieces.append((np.array([lon_lo, lon_lo]), np.array([lat_hi, lat_lo])))

    # Concatenate edges into a single polygon vertex list. The
    # vertical edges' start/end overlap with the adjacent horizontal
    # edges' endpoints — strip the duplicates so consecutive edges
    # don't introduce zero-length segments.
    src_lons: list[float] = []
    src_lats: list[float] = []
    for i, (pl, pt) in enumerate(pieces):
        if i == 0:
            src_lons.extend(pl.tolist())
            src_lats.extend(pt.tolist())
        else:
            src_lons.extend(pl[1:].tolist())
            src_lats.extend(pt[1:].tolist())

    # Transform source-frame box vertices into the figure's display frame.
    src_coords = SkyCoord(np.asarray(src_lons) * u.deg,
                           np.asarray(src_lats) * u.deg, frame=frame_key)
    polygon_lons, polygon_lats = _to_display_deg(src_coords, fig)

    return add_spherical_polygon(
        fig, polygon_lons, polygon_lats,
        resolution=1, geodesic=False,
        projection=projection, center=center,
        lat_center=lat_center, direction=direction,
        color=color, width=width, opacity=opacity,
        fill=fill, fillcolor=fillcolor,
        name=name, hover=hover,
        **trace_kwargs,
    )


def _frame_band_polygon(lat_min: float, lat_max: float, frame_key: str,
                        resolution: int,
                        connector_step_deg: float = 2.0,
                        display_frame: str = 'icrs',
                        ) -> tuple[Any, Any]:
    """Build a closed polygon outline of a frame latitude band as
    ``(polygon_lons, polygon_lats)`` arrays in ICRS degrees.

    Samples the top edge L→R at ``resolution`` points, then a densified
    "connector" edge across the band width at the right wrap-edge
    seam, then the bottom edge R→L, then the densified left-seam
    connector. The connectors are sampled at a target step of
    ``connector_step_deg`` so the polygon's seam doesn't project as a
    single band-width chord (which would otherwise show up as a
    visible "snap" at whichever ICRS lon the seam lands — e.g. near
    the galactic anti-center for galactic bands). Used by both
    :func:`add_frame_band` and :func:`add_great_circle_band` (which
    builds a custom rotated frame via
    :class:`astropy.coordinates.SkyOffsetFrame`).
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    n = max(8, int(resolution))
    span = max(1e-3, float(lat_max) - float(lat_min))
    m = max(20, int(np.ceil(span / float(connector_step_deg))) + 2)
    eps = 0.05
    lons_fwd = np.linspace(-180.0 + eps, 180.0 - eps, n)
    lons_rev = lons_fwd[::-1]
    top_lats = np.full(n, float(lat_max))
    bot_lats = np.full(n, float(lat_min))

    # Densified connector at the right seam (src_lon = 180 - eps),
    # spanning the full band width from lat_max down to lat_min.
    conn_right_lats = np.linspace(float(lat_max), float(lat_min), m)[1:-1]
    conn_right_lons = np.full(len(conn_right_lats), lons_fwd[-1])

    # Densified connector at the left seam (src_lon = -180 + eps),
    # spanning lat_min up to lat_max.
    conn_left_lats = np.linspace(float(lat_min), float(lat_max), m)[1:-1]
    conn_left_lons = np.full(len(conn_left_lats), lons_fwd[0])

    src_lons = np.concatenate([
        lons_fwd, conn_right_lons, lons_rev, conn_left_lons,
    ])
    src_lats = np.concatenate([
        top_lats, conn_right_lats, bot_lats, conn_left_lats,
    ])

    src_coords = SkyCoord(src_lons * u.deg, src_lats * u.deg,
                           frame=frame_key)
    from ..geometry._parsing import _coords_to_frame_deg
    return _coords_to_frame_deg(src_coords, display_frame)


def add_frame_band(fig: Any, lat_min: float, lat_max: float, *,
                    frame: str = 'galactic', resolution: int = 360,
                    projection: str | None = None, center: float | None = None,
                    lat_center: float | None = None,
                    direction: str | None = None,
                    color: str = 'steelblue', width: float = 1.0,
                    opacity: float = 0.25,
                    fill: bool = True, fillcolor: str | None = None,
                    name: str | None = None, hover: bool | str = False,
                    **trace_kwargs: Any) -> list[Any]:
    """Add a latitude band defined in another coordinate frame.

    Plotly counterpart to :func:`skyplothelper.add_frame_band` (patch
    backend). Samples the band's two latitude boundary parallels in
    ``frame``, closes the polygon, transforms to ICRS, and delegates
    to :func:`add_spherical_polygon` — which handles wrap-edge
    splitting and frame-curve densification.

    Note: the plotly path is the sphere-side polygon approach (no
    shapely / CompoundRegion). For visually thin bands like ±10°
    galactic or ±5° ecliptic the projected fill is clean across the
    wrap edge. For wider bands or unusual tilts the wrap-split may
    introduce visible cap chords at the projection boundary; switch
    to outline-only (``fill=False``) in that case.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    lat_min, lat_max : float
        Band latitude limits in degrees in ``frame``.
    frame : str
        Source frame: ``'galactic'``, ``'ecliptic'``,
        ``'supergalactic'``, ``'icrs'``, ``'fk5'``, ``'fk4'``.
    resolution : int
        Samples along each boundary parallel. Default ``360``.
    projection, center, lat_center, direction : optional
        Projection kwargs (default to figure metadata).
    color : str
    width : float
    opacity : float
        Default ``0.25``.
    fill : bool
        ``True`` (default) emits a filled band; ``False`` emits
        outline only.
    fillcolor : str, optional
    name : str, optional
    hover : False / True / str
    **trace_kwargs

    Returns
    -------
    traces : list of plotly.graph_objects.Scatter
    """
    if lat_min >= lat_max:
        raise ValueError("lat_min must be less than lat_max")
    frame_key = 'geocentrictrueecliptic' if str(frame).lower() == 'ecliptic' \
        else str(frame).lower()
    polygon_lons, polygon_lats = _frame_band_polygon(
        float(lat_min), float(lat_max), frame_key, resolution,
        display_frame=_display_frame(fig))
    # When fill is on, render fill via the polygon with NO visible
    # edge, then overlay the band's two real parallel edges as
    # independent ``add_great_circle`` polylines. The polygon's
    # internal seam connectors are still part of the fill outline
    # but never get drawn — only the parallels carry the visible
    # boundary lines, so users don't see a seam cutting across the
    # band.
    if fill:
        traces = list(add_spherical_polygon(
            fig, polygon_lons, polygon_lats,
            resolution=1, geodesic=False,
            projection=projection, center=center,
            lat_center=lat_center, direction=direction,
            color=color, width=0, opacity=opacity,
            fill=True, fillcolor=fillcolor,
            name=name, hover=hover,
            **trace_kwargs,
        ))
        if width > 0:
            for lat in (float(lat_max), float(lat_min)):
                traces.append(add_great_circle(
                    fig, frame=frame, lat_offset=lat,
                    projection=projection, center=center,
                    lat_center=lat_center, direction=direction,
                    color=color, width=width, opacity=opacity,
                ))
        return traces
    return add_spherical_polygon(
        fig, polygon_lons, polygon_lats,
        resolution=1, geodesic=False,
        projection=projection, center=center,
        lat_center=lat_center, direction=direction,
        color=color, width=width, opacity=opacity,
        fill=False, fillcolor=fillcolor,
        name=name, hover=hover,
        **trace_kwargs,
    )


def add_great_circle_band(fig: Any, ra_pole: float, dec_pole: float,
                           half_width: float, *,
                           resolution: int = 360,
                           projection: str | None = None,
                           center: float | None = None,
                           lat_center: float | None = None,
                           direction: str | None = None,
                           color: str = 'steelblue', width: float = 1.0,
                           opacity: float = 0.25,
                           fill: bool = True, fillcolor: str | None = None,
                           name: str | None = None, hover: bool | str = False,
                           **trace_kwargs: Any) -> list[Any]:
    """Add a band along an arbitrary great circle.

    The great circle is defined by its pole — every point on the great
    circle is exactly 90° from the pole. The band extends
    ``half_width`` degrees on either side. Plotly counterpart to
    :func:`skyplothelper.add_great_circle_band`. Generalizes latitude
    bands (pole at celestial pole), galactic bands (pole at galactic
    pole), and ecliptic bands (pole at ecliptic pole) to arbitrary
    orientations — e.g. satellite orbital planes, scanning-law strips,
    or custom avoidance zones.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    ra_pole, dec_pole : float
        RA / Dec of the great-circle pole in degrees (ICRS).
    half_width : float
        Half-width of the band in degrees.
    resolution : int
        Samples along each boundary edge. Default ``360``.
    projection, center, lat_center, direction : optional
    color : str
    width : float
    opacity : float
    fill : bool
    fillcolor : str, optional
    name : str, optional
    hover : False / True / str
    **trace_kwargs

    Returns
    -------
    traces : list of plotly.graph_objects.Scatter
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    if not (0 < float(half_width) < 90):
        raise ValueError("half_width must be in (0, 90) degrees")

    # The band boundaries are the two small circles at CONSTANT angular
    # distance (90 - half_width) inside and (90 + half_width) outside the pole
    # — i.e. exactly half_width on either side of the great circle (which is
    # 90° from the pole). Build them with directional_offset_by. (A
    # SkyOffsetFrame latitude parallel is NOT such a curve: its
    # distance-from-the-great-circle varies with longitude, so the old
    # top/bottom edges swept across the whole band width and crossed each
    # other, collapsing the projected fill to two self-intersection lenses.)
    # The pole is given in the figure's own display frame, like every other
    # bare lon/lat this module takes.
    pole = SkyCoord(float(ra_pole) * u.deg, float(dec_pole) * u.deg,
                     frame=_display_frame(fig))
    n = max(8, int(resolution))
    hw = float(half_width)
    r_inner = 90.0 - hw          # small circle inside the great circle
    r_outer = 90.0 + hw          # small circle outside it
    eps = 0.05
    pa_fwd = np.linspace(eps, 360.0 - eps, n)
    pa_rev = pa_fwd[::-1]
    # Densified radial connectors across the band width (target ~2°/step) so
    # the keyhole seam doesn't project as a single chord snap.
    m = max(20, int(np.ceil(2.0 * hw / 2.0)) + 2)
    conn_out = np.linspace(r_inner, r_outer, m)[1:-1]   # inner -> outer edge
    conn_in = np.linspace(r_outer, r_inner, m)[1:-1]    # outer -> inner edge

    def _circle(pa_arr: Any, sep_arr: Any) -> Any:
        return pole.directional_offset_by(
            np.atleast_1d(pa_arr) * u.deg, np.atleast_1d(sep_arr) * u.deg)

    # Keyhole annulus: inner edge forward, radial connector out, outer edge
    # reversed, radial connector back — a single closed ring, no hole needed.
    parts = [
        _circle(pa_fwd, np.full(n, r_inner)),
        _circle(np.full(len(conn_out), pa_fwd[-1]), conn_out),
        _circle(pa_rev, np.full(n, r_outer)),
        _circle(np.full(len(conn_in), pa_fwd[0]), conn_in),
    ]
    # ``directional_offset_by`` keeps the pole's frame, which is already the
    # display frame — read the components frame-agnostically (a galactic
    # figure has no ``.ra``).
    from ..geometry._parsing import _spherical_deg
    _parts_deg = [_spherical_deg(p) for p in parts]
    band_ra = np.concatenate([d[0] for d in _parts_deg])
    band_dec = np.concatenate([d[1] for d in _parts_deg])
    band_ra = np.append(band_ra, band_ra[0])
    band_dec = np.append(band_dec, band_dec[0])
    # Same fill + separate-parallel-boundary pattern as add_frame_band:
    # the polygon's internal seams stay invisible, only the two real
    # boundary parallels are drawn as lines via add_great_circle's
    # ``frame='pole'`` path.
    if fill:
        traces = list(add_spherical_polygon(
            fig, band_ra, band_dec,
            resolution=1, geodesic=False,
            projection=projection, center=center,
            lat_center=lat_center, direction=direction,
            color=color, width=0, opacity=opacity,
            fill=True, fillcolor=fillcolor,
            name=name, hover=hover,
            **trace_kwargs,
        ))
        if width > 0:
            for off in (float(half_width), -float(half_width)):
                traces.append(add_great_circle(
                    fig, frame='pole',
                    pole_lon=float(ra_pole), pole_lat=float(dec_pole),
                    lat_offset=off,
                    projection=projection, center=center,
                    lat_center=lat_center, direction=direction,
                    color=color, width=width, opacity=opacity,
                ))
        return traces
    return add_spherical_polygon(
        fig, band_ra, band_dec,
        resolution=1, geodesic=False,
        projection=projection, center=center,
        lat_center=lat_center, direction=direction,
        color=color, width=width, opacity=opacity,
        fill=False, fillcolor=fillcolor,
        name=name, hover=hover,
        **trace_kwargs,
    )


# -- HEALPix sparse ---------------------------------------------------------

def add_healpix_sparse(fig: Any, pixel_indices: npt.ArrayLike,
                        values: npt.ArrayLike, nside: int, *,
                        nest: bool = False,
                        projection: str | None = None,
                        center: float | None = None,
                        lat_center: float | None = None,
                        direction: str | None = None,
                        colorscale: str | list[Any] = 'Viridis',
                        vmin: float | None = None, vmax: float | None = None,
                        hover_format: str | Callable[..., str] | None = None,
                        tile_resolution: int | str = 'auto',
                        line_width: float = 0,
                        line_color: str | None = None,
                        opacity: float = 1.0,
                        add_colorbar: bool = False,
                        cbar_title: str | None = None,
                        colorbar_kwargs: dict[str, Any] | None = None,
                        ) -> list[Any]:
    """Render a sparse subset of HEALPix tiles with per-tile hover.

    Companion to :func:`add_healpix` for cases where only a small
    fraction of tiles carry data — zoom-ins, query results, masked
    surveys. Avoids materializing a full ``12 * nside**2`` array.
    Plotly counterpart to the patch backend of
    :func:`skyplothelper.plot_healpix_sparse`.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    pixel_indices : array-like of int
        HEALPix pixel indices to render.
    values : array-like of float
        Per-pixel values (same length as ``pixel_indices``). NaN tiles
        are skipped.
    nside : int
        HEALPix resolution parameter.
    nest : bool
        ``True`` for NESTED indexing, ``False`` (default) for RING.
    projection, center, lat_center, direction : optional
    colorscale : str or list
        Plotly colorscale. Default ``'Viridis'``.
    vmin, vmax : float, optional
        Color normalization. Default: finite-values min / max.
    hover_format : str or callable
        Per-tile hover content — see :func:`add_healpix`.
    tile_resolution : int or ``'auto'``
        Edge sample density per tile — see :func:`add_healpix` for the
        ``'auto'`` heuristic. Default ``'auto'``.
    line_width : float
        Tile edge line width. Default ``0``.
    line_color : str, optional
        Tile edge color. Default ``None`` uses each tile's fill color (edges
        invisible against the fill); set a contrasting color to make the tile
        boundaries stand out.
    opacity : float
        Default ``1.0``.
    add_colorbar : bool
        Attach a colorbar keyed to ``colorscale`` / ``vmin`` / ``vmax``
        via an invisible companion trace — see :func:`add_healpix`.
        Default ``False``.
    cbar_title : str, optional
        Colorbar title. Ignored unless ``add_colorbar=True``.
    colorbar_kwargs : dict, optional
        Passed through to ``marker.colorbar`` for placement / styling.

    Returns
    -------
    traces : list of plotly.graph_objects.Scatter
        One or two traces per non-NaN tile (two when the tile straddles
        the projection wrap edge), followed by the invisible colorbar
        trace when ``add_colorbar=True``.
    """
    go = _import_plotly()
    try:
        import healpy as hp
    except ImportError as exc:
        raise ImportError(
            "add_healpix_sparse requires the optional `healpy` package. "
            "Install with `pip install healpy` or "
            "`pip install skyplothelper[healpix]`."
        ) from exc

    projection, center, lat_center, direction = _meta_defaults(
        fig, projection, center, lat_center, direction)
    proj = _overlay_projector(fig, projection, center, lat_center, direction)
    pixel_indices = np.asarray(pixel_indices, dtype=int)
    values = np.asarray(values, dtype=float)
    if pixel_indices.shape != values.shape:
        raise ValueError(
            f"pixel_indices and values must have the same shape, got "
            f"{pixel_indices.shape} vs {values.shape}")

    finite = np.isfinite(values)
    if vmin is None:
        vmin = float(np.min(values[finite])) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.max(values[finite])) if finite.any() else 1.0

    step = _resolve_tile_resolution(tile_resolution, nside)
    # Tile center coords for the hover content (lookup only on the
    # rendered subset, not the full sky).
    sel_idx = pixel_indices[finite]
    sel_vals = values[finite]
    cen_theta, cen_phi = hp.pix2ang(nside, sel_idx, nest=nest)
    cen_lon = np.degrees(cen_phi)
    cen_lat = 90.0 - np.degrees(cen_theta)

    from plotly.colors import sample_colorscale

    _hover_text: Callable[[float, float, float, int], str]
    if hover_format is None:
        _lon_name, _lat_name = _hover_labels(fig)

        def _hover_text(lon: float, lat: float, val: float, ipix: int) -> str:
            return (f"{_lon_name}: {lon:.3f}°<br>{_lat_name}: {lat:.3f}°<br>"
                    f"value: {val:.6g}<br>ipix: {ipix}")
    elif callable(hover_format):
        import inspect
        try:
            _params = inspect.signature(hover_format).parameters
            _accepts_ipix = (
                len(_params) >= 4
                or any(p.kind == inspect.Parameter.VAR_POSITIONAL
                       or p.kind == inspect.Parameter.VAR_KEYWORD
                       for p in _params.values())
            )
        except (TypeError, ValueError):
            _accepts_ipix = False
        if _accepts_ipix:
            _hover_text = hover_format
        else:
            def _hover_text(lon: float, lat: float, val: float,
                            ipix: int) -> str:
                return hover_format(lon, lat, val)
    else:
        _fmt = str(hover_format)
        def _hover_text(lon: float, lat: float, val: float, ipix: int) -> str:
            return _fmt.format(lon=lon, lat=lat, value=val, ipix=ipix)

    traces = []
    for k in range(len(sel_idx)):
        ipix = int(sel_idx[k])
        value = float(sel_vals[k])
        verts = hp.boundaries(nside, ipix, step=step, nest=nest)
        x_v, y_v, z_v = verts[0], verts[1], verts[2]
        r = np.sqrt(x_v ** 2 + y_v ** 2 + z_v ** 2)
        b_lat = np.degrees(np.arcsin(z_v / r))
        b_lon = np.degrees(np.arctan2(y_v, x_v)) % 360.0
        b_lon = np.append(b_lon, b_lon[0])
        b_lat = np.append(b_lat, b_lat[0])
        norm_val = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        norm_val = float(np.clip(norm_val, 0.0, 1.0))
        rgb = sample_colorscale(colorscale, [norm_val])[0]
        hover_text = _hover_text(float(cen_lon[k]), float(cen_lat[k]),
                                  value, ipix)
        edge = rgb if line_color is None else line_color
        for x, y in proj.project_polygon_pieces(b_lon, b_lat):
            trace = go.Scatter(
                x=x, y=y, mode='lines', fill='toself',
                fillcolor=rgb, opacity=opacity,
                line=dict(width=line_width, color=edge),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='',
                showlegend=False,
                hoveron='fills',
            )
            fig.add_trace(trace)
            traces.append(trace)
    if add_colorbar:
        traces.append(_add_scale_colorbar(
            fig, go, colorscale=colorscale, cmin=vmin, cmax=vmax,
            title=cbar_title, colorbar_kwargs=colorbar_kwargs))
    return traces


# -- Sky-vector arrows ------------------------------------------------------

_VECTOR_UNIT_FACTORS = {
    'deg': 1.0, 'arcmin': 1 / 60., 'arcsec': 1 / 3600.,
    'mas': 1 / 3.6e6, 'uas': 1 / 3.6e9,
}


def add_sky_vectors(fig: Any, lon: SkyCoord | npt.ArrayLike, lat: npt.ArrayLike | None = None,
                    dlon: npt.ArrayLike | None = None,
                    dlat: npt.ArrayLike | None = None, *,
                    scale: float | str = 1.0, units: str = 'arcsec',
                    cos_dec: bool = True,
                    auto_target_deg: float = 2.0, pivot: str = 'middle',
                    projection: str | None = None, center: float | None = None,
                    lat_center: float | None = None,
                    direction: str | None = None,
                    color: Any = 'steelblue', opacity: float = 0.85,
                    width: float = 1.5, arrow_size: float = 10,
                    shaft_color: str = 'auto',
                    color_by_magnitude: bool = False, cmap: str = 'Viridis',
                    cmin: float | None = None, cmax: float | None = None,
                    add_colorbar: bool = False, cbar_title: str | None = None,
                    name: str | None = None, hover: bool | str = False,
                    **trace_kwargs: Any) -> tuple[Any, Any]:
    """Add 2D sky-vector arrows (quiver) on the figure.

    Plotly counterpart to :func:`skyplothelper.plot_sky_vectors`.
    Renders ``(dlon, dlat)`` vectors at each ``(lon, lat)`` anchor as
    a shaft polyline plus a marker arrowhead — emitted as two
    ``go.Scatter`` traces (so per-arrow color and per-arrow rotation
    work via the marker channel). Works for proper-motion arrows,
    catalog position differences, VSH-fit residuals, simulated
    velocity fields, or any other "this point + this vector" data.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    lon, lat : array-like
        Anchor positions in degrees (ICRS).
    dlon, dlat : array-like
        Vector components — longitudinal (RA-direction) and
        latitudinal (Dec-direction) parts. Units set by ``units``.
        Typical conventions:

        * Proper motion: ``dlon = μ_α cos δ``, ``dlat = μ_δ``
          (both in mas/yr); ``cos_dec=True`` (default).
        * Catalog position differences ``ra2 - ra1`` /
          ``dec2 - dec1`` that are *not* cosδ-scaled —
          ``cos_dec=False``.
        * VSH residuals: predicted minus observed PM components
          in μas / mas.
    scale : float or ``'auto'``
        Arrow scale factor. Multiplies each vector's magnitude (in
        ``units``); the product is the arrow's on-sky length,
        converted to degrees for plotting — so it is
        degrees-per-unit-magnitude only when ``units='deg'``.
        ``'auto'`` picks a scale so the median arrow spans
        ``auto_target_deg`` degrees.
    units : str
        ``'deg'`` / ``'arcmin'`` / ``'arcsec'`` / ``'mas'`` / ``'uas'``.
    cos_dec : bool
        If ``True`` (default), ``dlon`` already includes the
        cos δ factor. If ``False``, the correction is applied
        internally.
    auto_target_deg : float
        Median-arrow target length (degrees) under ``scale='auto'``.
    pivot : {'middle', 'tail', 'tip'}
        Anchor position on the arrow shaft. Default ``'middle'``.
    projection, center, lat_center, direction : optional
        Projection kwargs (default to figure metadata).
    color : str
        Arrow color. Ignored when ``color_by_magnitude=True``.
    opacity : float
    width : float
        Shaft line width.
    arrow_size : float
        Arrowhead marker size in pixels. Default ``10``.
    shaft_color : ``'auto'`` / ``'match'`` / str
        How the arrow shaft is colored. ``'auto'`` (default) emits a
        single shaft trace using ``color`` if it's a string, else a
        neutral gray — fast, one trace for all shafts regardless of
        N. ``'match'`` emits one shaft trace per arrow, each sampled
        from the same colorscale as the heads (only meaningful when a
        per-arrow color array is in use via ``color_by_magnitude=True``
        or a numerical ``color=`` sequence; falls back to ``'auto'``
        otherwise). Any other string overrides ``color`` for the shaft
        only.

        Trade-off: ``'match'`` produces N traces instead of 1, which
        is fine up to a few hundred arrows but gets sluggish for very
        large fields (thousands).
    color_by_magnitude : bool
        Color arrows by ``hypot(dlon, dlat)``. Default ``False``.
    cmap : str
        Plotly colorscale (used when ``color_by_magnitude=True``).
    cmin, cmax : float, optional
        Color normalization range. Default: data min / max.
    add_colorbar : bool
        Attach a plotly colorbar to the arrowhead trace when
        ``color_by_magnitude=True``. Default ``False``.
    cbar_title : str, optional
        Colorbar title.
    name : str, optional
        Trace name (legend / hover header). Applied to the
        arrowhead trace.
    hover : False / True / str
        Hover on the arrowhead trace. ``True`` shows name +
        anchor RA/Dec + vector magnitude in ``units``. A custom
        string is used verbatim as the hovertemplate (customdata
        columns: ``[0]=anchor lon, [1]=anchor lat, [2]=magnitude
        in input units, [3]=position angle (deg)``).
    **trace_kwargs
        Forwarded to the arrowhead ``go.Scatter`` trace.

    Returns
    -------
    shaft : go.Scatter or list of go.Scatter
        The shaft trace(s). A single NaN-separated polyline in
        ``shaft_color='auto'`` or explicit-string modes; a list of
        one trace per arrow when ``shaft_color='match'`` and a
        per-arrow color array is in use.
    head : go.Scatter
        The per-arrow marker arrowhead trace.

    Notes
    -----
    Arrows whose shaft crosses the projection wrap edge (rare unless
    the input magnitude × scale is very large) render as straight
    lines across the canvas; the wrap-split helpers used elsewhere
    don't apply per-arrow. Lower ``scale`` or pre-clip such inputs.
    """
    go = _import_plotly()
    projection, center, lat_center, direction = _meta_defaults(
        fig, projection, center, lat_center, direction)
    proj = _overlay_projector(fig, projection, center, lat_center, direction)

    # A SkyCoord fills the single ``lon`` slot; the two magnitude arrays that
    # follow can't shift unambiguously, so they must be named. Same rule (and
    # same message) as the matplotlib ``plot_sky_vectors``.
    if hasattr(lon, 'transform_to') and lat is not None:
        raise TypeError(
            "add_sky_vectors: pass either a SkyCoord (with dlon=/dlat= as "
            "keywords) or separate lon/lat arrays — not a SkyCoord followed "
            "by positional magnitudes.")
    lon, lat = _resolve_lonlat(fig, lon, lat, 'add_sky_vectors')
    if dlon is None or dlat is None:
        raise TypeError("add_sky_vectors: dlon and dlat are required.")

    lon_a: np.ndarray = np.atleast_1d(np.asarray(lon, dtype=float))
    lat_a: np.ndarray = np.atleast_1d(np.asarray(lat, dtype=float))
    dlon_a: np.ndarray = np.atleast_1d(np.asarray(dlon, dtype=float))
    dlat_a: np.ndarray = np.atleast_1d(np.asarray(dlat, dtype=float))

    if units.lower() not in _VECTOR_UNIT_FACTORS:
        raise ValueError(
            f"units must be one of {sorted(_VECTOR_UNIT_FACTORS)}, "
            f"got {units!r}")
    ufac = _VECTOR_UNIT_FACTORS[units.lower()]

    magnitudes = np.hypot(dlon_a, dlat_a)
    if isinstance(scale, str):
        if scale.lower() != 'auto':
            raise ValueError(f"scale must be 'auto' or a float, got {scale!r}")
        median_mag = (np.median(magnitudes[magnitudes > 0])
                       if np.any(magnitudes > 0) else 1.0)
        scale_val = (auto_target_deg / (ufac * median_mag)
                      if median_mag * ufac > 0 else 1.0)
    else:
        scale_val = float(scale)

    dlon_deg = dlon_a * ufac * scale_val
    dlat_deg = dlat_a * ufac * scale_val
    if not cos_dec:
        cos_lat = np.cos(np.radians(lat_a))
        cos_lat = np.where(cos_lat > 1e-6, cos_lat, 1e-6)
        dlon_deg = dlon_deg / cos_lat

    # Resolve pivot — shifts the shaft so the data point sits at the
    # requested fraction along the arrow. ``cos δ`` correction in
    # sphere space: a step of ``dlon_deg`` in (RA-cos δ-scaled)
    # screen-equivalent corresponds to ``dlon_deg / cos δ`` in raw
    # RA degrees. We accept the small-angle simplification that the
    # cos δ at the anchor is representative across the short arrow
    # span.
    cos_anchor = np.cos(np.radians(lat_a))
    cos_anchor = np.where(cos_anchor > 1e-6, cos_anchor, 1e-6)
    pivot_frac = {'tail': 0.0, 'middle': 0.5, 'tip': 1.0}.get(pivot)
    if pivot_frac is None:
        raise ValueError("pivot must be 'tail', 'middle', or 'tip', "
                         f"got {pivot!r}")
    start_lon = lon_a - pivot_frac * (dlon_deg / cos_anchor)
    start_lat = lat_a - pivot_frac * dlat_deg
    end_lon = start_lon + (dlon_deg / cos_anchor)
    end_lat = start_lat + dlat_deg

    x0, y0 = proj.project_points(start_lon, start_lat)
    x1, y1 = proj.project_points(end_lon, end_lat)

    # Build the shaft polyline (NaN-separated so a single go.Scatter
    # renders all arrows). An arrow whose base→head segment crosses the
    # projection wrap seam (center ± 180) projects to opposite frame
    # edges, so a raw base→head canvas line would streak across the whole
    # frame. Those few arrows are densified and wrap-split so the shaft
    # breaks cleanly at the seam edge; the rest keep the simple two-point
    # segment.
    n = len(lon_a)
    # Seam-straddle detection is relative to the projector's own wrap
    # meridian; on a seamless FITS frame a small displacement never
    # straddles, so those arrows take the plain two-point shaft below.
    norm0 = ((start_lon - proj.center + 180.0) % 360.0) - 180.0
    norm1 = ((end_lon - proj.center + 180.0) % 360.0) - 180.0
    straddle = np.abs(norm1 - norm0) > 180.0

    shaft_pieces: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        if straddle[i]:
            dl = np.linspace(start_lon[i], end_lon[i], 16)
            db = np.linspace(start_lat[i], end_lat[i], 16)
            sx, sy = proj.project_polyline(dl, db)
        else:
            sx = np.array([x0[i], x1[i]])
            sy = np.array([y0[i], y1[i]])
        shaft_pieces.append((sx, sy))

    nan1 = np.array([np.nan])
    if n:
        shaft_x = np.concatenate(
            [a for sx, sy in shaft_pieces for a in (sx, nan1)])
        shaft_y = np.concatenate(
            [a for sx, sy in shaft_pieces for a in (sy, nan1)])
    else:
        shaft_x = np.array([])
        shaft_y = np.array([])

    # Arrowhead orientation in the projected plane. Plotly's
    # ``marker.angle`` is measured clockwise from north (i.e. from the
    # +y axis), so we convert atan2(dx, dy) — note the swapped argument
    # order — to degrees. ``(x1 - x0)`` is the head direction for ordinary
    # arrows; for a seam-straddling arrow that is the across-frame streak
    # vector, so use the local projected tangent at the head instead.
    head_dx = x1 - x0
    head_dy = y1 - y0
    if np.any(straddle):
        pre_lon = end_lon - 1e-3 * (end_lon - start_lon)
        pre_lat = end_lat - 1e-3 * (end_lat - start_lat)
        xp, yp = proj.project_points(pre_lon, pre_lat)
        head_dx = np.where(straddle, x1 - xp, head_dx)
        head_dy = np.where(straddle, y1 - yp, head_dy)
    head_angle = np.degrees(np.arctan2(head_dx, head_dy))

    # Per-arrow color array. ``color_by_magnitude`` overrides the
    # uniform ``color`` parameter; a numerical sequence in ``color``
    # is also forwarded as a colorscale-mapped array.
    head_marker_color: Any = color
    head_marker_extras: dict[str, Any] = {}
    using_array_color = False
    if color_by_magnitude:
        head_marker_color = magnitudes
        head_marker_extras = dict(
            colorscale=cmap,
            cmin=float(cmin) if cmin is not None else float(np.min(magnitudes)),
            cmax=float(cmax) if cmax is not None else float(np.max(magnitudes)),
        )
        if add_colorbar:
            head_marker_extras['showscale'] = True
            head_marker_extras['colorbar'] = dict(
                title=cbar_title or f"magnitude ({units})")
        using_array_color = True
    elif (hasattr(color, '__len__') and not isinstance(color, str)
          and not isinstance(color, dict)):
        head_marker_color = np.asarray(color)
        head_marker_extras = dict(colorscale=cmap)
        if cmin is not None:
            head_marker_extras['cmin'] = float(cmin)
        if cmax is not None:
            head_marker_extras['cmax'] = float(cmax)
        if add_colorbar:
            head_marker_extras['showscale'] = True
            head_marker_extras['colorbar'] = dict(title=cbar_title or '')
        using_array_color = True

    # Resolve ``shaft_color``. ``'match'`` requires a per-arrow color
    # array (color_by_magnitude or array-color); without one it falls
    # back to ``'auto'`` behavior so the contract stays predictable.
    match_active = (shaft_color == 'match' and using_array_color)
    if shaft_color == 'auto' or (shaft_color == 'match' and not using_array_color):
        shaft_color_resolved = color if isinstance(color, str) else 'gray'
    elif match_active:
        shaft_color_resolved = None  # set per-arrow below
    elif isinstance(shaft_color, str):
        shaft_color_resolved = shaft_color
    else:
        raise ValueError(
            f"shaft_color must be 'auto', 'match', or a color string, "
            f"got {shaft_color!r}")

    if match_active:
        # Plotly lines don't accept a per-segment colorscale on a
        # single trace; emit one short shaft trace per arrow,
        # sampling the colorscale at this arrow's normalized value.
        from plotly.colors import sample_colorscale
        cmin_val = float(head_marker_extras.get('cmin',
                                                 np.min(head_marker_color)))
        cmax_val = float(head_marker_extras.get('cmax',
                                                 np.max(head_marker_color)))
        span = cmax_val - cmin_val if cmax_val > cmin_val else 1.0
        color_arr = np.asarray(head_marker_color, dtype=float)
        norm = np.clip((color_arr - cmin_val) / span, 0.0, 1.0)
        shaft_trace: Any = []
        for i in range(n):
            rgb = sample_colorscale(cmap, [float(norm[i])])[0]
            t = go.Scatter(
                x=shaft_pieces[i][0], y=shaft_pieces[i][1],
                mode='lines',
                line=dict(color=rgb, width=width),
                opacity=opacity, hoverinfo='skip', showlegend=False,
            )
            fig.add_trace(t)
            shaft_trace.append(t)
    else:
        shaft_trace = go.Scatter(
            x=shaft_x, y=shaft_y, mode='lines',
            line=dict(color=shaft_color_resolved, width=width),
            opacity=opacity, hoverinfo='skip', showlegend=False,
        )
        fig.add_trace(shaft_trace)

    # Position angle in degrees (E of N convention, from anchor) —
    # for the hover customdata. Note: ``head_angle`` above is
    # measured in the projected plane (clockwise from +y for plotly
    # rotation); for hover we report the on-sky PA derived from
    # ``(dlon, dlat)`` so the value is interpretable independent of
    # projection.
    sky_pa = np.degrees(np.arctan2(dlon_a, dlat_a)) % 360.0
    customdata = np.column_stack([lon_a.ravel(), lat_a.ravel(),
                                    magnitudes.ravel(), sky_pa.ravel()])

    head_kw: dict[str, Any] = dict(
        x=x1, y=y1, mode='markers', name=name,
        showlegend=name is not None and not using_array_color,
        marker=dict(
            symbol='arrow-up', size=arrow_size,
            angle=head_angle,
            color=head_marker_color, **head_marker_extras,
        ),
        opacity=opacity, customdata=customdata,
    )
    if hover is False or hover is None:
        head_kw['hoverinfo'] = 'skip'
    elif hover is True:
        prefix = f"<b>{name}</b><br>" if name else ""
        head_kw['hovertemplate'] = _default_hover(
            fig, prefix=prefix,
            extra=(f"<br>|v|: %{{customdata[2]:.3g}} {units}<br>"
                   "PA: %{customdata[3]:.1f}°"))
    else:
        tpl = str(hover)
        if '<extra>' not in tpl:
            tpl += '<extra></extra>'
        head_kw['hovertemplate'] = tpl
    head_kw.update(trace_kwargs)
    head_trace = go.Scatter(**head_kw)
    fig.add_trace(head_trace)

    return shaft_trace, head_trace


# -- Coordinate tick labels --------------------------------------------------

def _format_lon(lon: float, fmt: str = 'deg') -> str:
    """Format a longitude value for tick-label display."""
    lon_wrapped = lon % 360.0
    if fmt == 'hours':
        # Hour angle: 0–24h, one decimal if needed.
        h = lon_wrapped / 15.0
        return f"{h:g}h"
    # Default: degrees.
    return f"{int(round(lon_wrapped))}°"


def _format_lat(lat: float, fmt: str = 'deg') -> str:
    """Format a latitude value for tick-label display."""
    return f"{int(round(lat)):+d}°"


def add_coord_labels(fig: Any, *, projection: str | None = None,
                     center: float | None = None,
                     lat_center: float | None = None,
                     direction: str | None = None,
                     lon_spacing: float = 30, lat_spacing: float = 15,
                     lon_format: str | Callable[[float], str] = 'auto',
                     lat_format: str | Callable[[float], str] = 'deg',
                     color: str | None = None, fontsize: int = 11,
                     show_lon: bool = True, show_lat: bool = True,
                     placement: str = 'frame',
                     lon_offset_px: int = 6, lat_offset_px: int = 6,
                     lat_exterior: bool = False) -> list[Any]:
    """Add lon/lat coordinate tick labels to a plotly sky figure.

    Plotly counterpart to WCSAxes tick labels. Projects evenly spaced
    longitudes (at the equator) and latitudes (at the wrap-edge
    meridian) through the figure's projection primitive, then places
    text annotations at the projected positions.

    Two placement modes:

    * ``'frame'`` (default) — labels follow the projection silhouette.
      Lon labels sit just below the equator at each labeled longitude
      (where the meridian crosses ``lat=0``). Lat labels sit at the
      frame edge where each parallel meets the wrap meridian — just
      inside by default, or outside with ``lat_exterior=True``. Natural
      fit for AIT / MOL / SIN / other curved pseudocylindricals.

    * ``'canvas'`` — labels sit along the figure's canvas edges (lon
      along the bottom in paper-y, lat along the left in paper-x).
      Better fit for cylindrical / rectangular frames or TAN zooms
      where the equator may not be visible.

    Uses :func:`plotly.graph_objects.Figure.add_annotation` so the
    labels live in ``fig.layout.annotations`` (not as data traces) —
    no impact on legend, hover, or zoom behavior. Re-run to refresh
    after the figure is regenerated; there's no "clear" pass.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Target figure (typically built by :func:`make_figure`).
    projection, center, lat_center, direction : optional
        Projection kwargs (default to figure metadata stamped by
        :func:`make_figure`).
    lon_spacing, lat_spacing : float
        Tick spacing in degrees. Defaults ``30°`` for lon and ``15°``
        for lat — the conventional all-sky cadence.
    lon_format : ``'auto'`` / ``'deg'`` / ``'hours'`` / callable
        Longitude label format. ``'auto'`` (default) picks ``'hours'`` for an
        equatorial figure ``frame`` (set on :func:`make_figure`) and ``'deg'``
        otherwise. ``'deg'`` emits e.g. ``"60°"``; ``'hours'`` emits ``"4h"``.
        A callable receives the raw lon value (degrees) and returns the string.
    lat_format : ``'deg'`` / callable
        Latitude label format. ``'deg'`` (default) emits e.g. ``"+30°"``.
        A callable receives the raw lat value (degrees) and returns the string.
    color : str, optional
        Label color. Defaults to the figure's foreground color
        (recorded by :func:`make_figure` for dark/light themes); use
        a string to override.
    fontsize : int
        Label font size. Default ``11``.
    show_lon, show_lat : bool
        Independent toggles for the two axes. Default both ``True``.
    placement : ``'frame'`` / ``'canvas'``
        Where to anchor the labels — see top of docstring. Default
        ``'frame'``.
    lon_offset_px, lat_offset_px : int
        Pixel offset that pushes the labels just past the equator (lon)
        or frame edge (lat) so they don't overlap the gridlines.
        Defaults ``6``.
    lat_exterior : bool
        ``placement='frame'`` only. When ``False`` (default), latitude
        labels sit just *inside* the frame edge (the historical
        behavior). When ``True``, they sit *outside* the frame edge —
        the conventional axis-tick look, useful when interior labels
        would overlap dense sky content. No effect on ``'canvas'``
        placement (already exterior to the data area).

    Returns
    -------
    annotations : list of plotly Annotation
        The annotation objects appended to ``fig.layout.annotations``
        (matches the order they were added).
    """
    if placement not in ('frame', 'canvas'):
        raise ValueError(
            f"placement must be 'frame' or 'canvas', got {placement!r}")
    projection, center, lat_center, direction = _meta_defaults(
        fig, projection, center, lat_center, direction)

    # Resolve 'auto' longitude units. An explicit make_figure(lon_units=...)
    # wins; otherwise fall back to the frame hint (equatorial → hours, like
    # the matplotlib make_wcs_frame default; every other frame, or none, →
    # degrees). An explicit lon_format='deg'/'hours'/callable always wins.
    if lon_format == 'auto':
        _fmeta = (getattr(fig, 'layout', None)
                  and getattr(fig.layout, 'meta', None)) or {}
        _fmeta = _fmeta if isinstance(_fmeta, dict) else {}
        _units = _fmeta.get('sph_lon_units', 'auto')
        if _units == 'hours':
            lon_format = 'hours'
        elif _units == 'degrees':
            lon_format = 'deg'
        else:  # 'auto' → defer to the frame hint
            _fr = _fmeta.get('sph_frame')
            lon_format = 'hours' if _fr in (
                'icrs', 'fk5', 'fk4', 'fk4-no-e', 'equatorial', 'j2000',
                'b1950') else 'deg'

    # Pull a sane default label color from the make_figure theme tag
    # so dark themes get bright labels without forcing the user to
    # pass ``color=`` every time.
    if color is None:
        # Default to a light fg if the figure was built dark, else dark.
        color = _theme_fg(fig)

    def _resolve_lon_fmt(val: float) -> str:
        if callable(lon_format):
            return lon_format(val)
        return _format_lon(val, fmt=lon_format)

    def _resolve_lat_fmt(val: float) -> str:
        if callable(lat_format):
            return lat_format(val)
        return _format_lat(val, fmt=lat_format)

    added: list[Any] = []

    # ``Figure.add_annotation`` returns the figure itself, not the
    # annotation. Snapshot the count beforehand and slice
    # ``fig.layout.annotations`` after to recover the freshly-added
    # annotation objects for the return value.
    n_before = len(fig.layout.annotations) if fig.layout.annotations else 0

    if show_lon:
        # Longitude ticks at evenly-spaced lons across the full 360°
        # window centered on ``center``. Drop labels too close to the
        # wrap edges (they collide with the frame silhouette).
        lons = np.arange(-180.0 + center, 180.0 + center, float(lon_spacing))
        wrap_pad = float(lon_spacing) * 0.4
        keep = ((lons > center - 180.0 + wrap_pad)
                & (lons < center + 180.0 - wrap_pad))
        lons = lons[keep]
        if len(lons) > 0:
            if placement == 'frame':
                # Place each label at the projected (lon, 0) point —
                # i.e. on the equator gridline — and shift it down by
                # ``lon_offset_px`` so it sits just below the line.
                xs, ys = _project(lons, np.zeros_like(lons),
                                   projection=projection, center=center,
                                   lat_center=lat_center,
                                   direction=direction)
                for lon_val, x_val, y_val in zip(lons, xs, ys):
                    if not (np.isfinite(x_val) and np.isfinite(y_val)):
                        continue
                    fig.add_annotation(
                        x=float(x_val), y=float(y_val),
                        xref='x', yref='y',
                        text=_resolve_lon_fmt(float(lon_val)),
                        showarrow=False,
                        yshift=-int(lon_offset_px),
                        font=dict(color=color, size=int(fontsize)),
                        xanchor='center', yanchor='top',
                    )
            else:  # 'canvas'
                xs, _ = _project(lons, np.zeros_like(lons),
                                  projection=projection, center=center,
                                  lat_center=lat_center,
                                  direction=direction)
                for lon_val, x_val in zip(lons, xs):
                    if not np.isfinite(x_val):
                        continue
                    fig.add_annotation(
                        x=float(x_val), y=0.02,
                        xref='x', yref='paper',
                        text=_resolve_lon_fmt(float(lon_val)),
                        showarrow=False,
                        font=dict(color=color, size=int(fontsize)),
                        xanchor='center', yanchor='top',
                    )

    if show_lat:
        # Latitude ticks at evenly-spaced lats avoiding the poles
        # (which collapse to a point on most projections).
        lats = np.arange(-90.0 + float(lat_spacing), 90.0,
                          float(lat_spacing))
        if len(lats) > 0:
            if placement == 'frame':
                # Pick the projection wrap-edge meridian that maps furthest
                # toward the canvas LEFT, then place all lat labels on it.
                # Works for both ``direction='sky'`` (RA-leftward) and
                # 'geographic' since the test is empirical.
                eps = 0.05
                x_left_test, _ = _project(
                    np.array([center - 180.0 + eps]), np.array([0.0]),
                    projection=projection, center=center,
                    lat_center=lat_center, direction=direction)
                x_right_test, _ = _project(
                    np.array([center + 180.0 - eps]), np.array([0.0]),
                    projection=projection, center=center,
                    lat_center=lat_center, direction=direction)
                if float(x_left_test[0]) <= float(x_right_test[0]):
                    edge_lon = center - 180.0 + eps
                    edge_x = float(x_left_test[0])
                else:
                    edge_lon = center + 180.0 - eps
                    edge_x = float(x_right_test[0])
                # Offset the labels away from (exterior) or toward
                # (interior) the frame interior. The frame is symmetric
                # about x=0 for centered projections, so the sign of the
                # chosen edge's x is the outward (away-from-frame)
                # direction; interior is the opposite.
                outward = -1 if edge_x < 0 else 1
                sign = outward if lat_exterior else -outward
                xshift = sign * int(lat_offset_px)
                xanchor = 'right' if xshift < 0 else 'left'
                xs_lat, ys_lat = _project(
                    np.full_like(lats, edge_lon), lats,
                    projection=projection, center=center,
                    lat_center=lat_center, direction=direction)
                for lat_val, x_val, y_val in zip(lats, xs_lat, ys_lat):
                    if not (np.isfinite(x_val) and np.isfinite(y_val)):
                        continue
                    fig.add_annotation(
                        x=float(x_val), y=float(y_val),
                        xref='x', yref='y',
                        text=_resolve_lat_fmt(float(lat_val)),
                        showarrow=False,
                        xshift=xshift,
                        font=dict(color=color, size=int(fontsize)),
                        xanchor=xanchor, yanchor='middle',
                    )
            else:  # 'canvas'
                _, ys = _project(np.full_like(lats, float(center)), lats,
                                  projection=projection, center=center,
                                  lat_center=lat_center,
                                  direction=direction)
                for lat_val, y_val in zip(lats, ys):
                    if not np.isfinite(y_val):
                        continue
                    fig.add_annotation(
                        x=0.02, y=float(y_val),
                        xref='paper', yref='y',
                        text=_resolve_lat_fmt(float(lat_val)),
                        showarrow=False,
                        font=dict(color=color, size=int(fontsize)),
                        xanchor='right', yanchor='middle',
                    )

    added = list(fig.layout.annotations[n_before:])
    return added


# -- Projection frame edge --------------------------------------------------

def _limb_lonlat(center: float, lat_center: float, radius_deg: float,
                 n: int) -> tuple[npt.NDArray[np.float64],
                                  npt.NDArray[np.float64]]:
    """(lon, lat) of the small circle at angular radius ``radius_deg``
    about ``(center, lat_center)`` — the great-circle locus that is the
    limb (visible-hemisphere boundary) of a zenithal globe when
    ``radius_deg`` ≈ 90.

    Standard spherical-triangle sweep: bearing ``th`` runs 0→2π around
    the center, giving the destination lat/lon at fixed angular distance.
    """
    th = np.linspace(0.0, 2.0 * np.pi, int(n))
    clr, clt, rr = np.deg2rad([center, lat_center, radius_deg])
    lat = np.arcsin(
        np.sin(clt) * np.cos(rr) + np.cos(clt) * np.sin(rr) * np.cos(th))
    lon = clr + np.arctan2(
        np.sin(th) * np.sin(rr) * np.cos(clt),
        np.cos(rr) - np.sin(clt) * np.sin(lat))
    return np.rad2deg(lon), np.rad2deg(lat)


def add_frame_edge(fig: Any, *, projection: str | None = None,
                   center: float | None = None,
                   lat_center: float | None = None,
                   direction: str | None = None,
                   resolution: int = 361, color: str | None = None,
                   width: float = 1.0, opacity: float = 1.0) -> Any:
    """Draw the projection's silhouette as a closed curve.

    The silhouette is traced one of two ways depending on the
    projection's frame shape (looked up in the projection registry):

    * **Circular globes** (the zenithal projections SIN, ARC, ZEA,
      STG, AZP, SZP, AIR) — the boundary is the *limb*: the great
      circle 90° from the projection center. The wrap meridian would
      run down the far, invisible hemisphere here (every sample
      un-projectable), so the limb small-circle is traced instead.
    * **Pseudo-cylindrical / elliptical all-sky** (AIT, MOL, PAR,
      SFL) — the wrap meridian (``lon = center ± 180``, pole to pole
      and back) IS the canvas boundary, so it is sampled and
      projected directly.

    For zoomed or rectangular projections (TAN, CYL, PCC) the canvas
    itself is the frame; the wrap-meridian curve is still returned but
    an explicit edge is usually unnecessary.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Target figure (typically built by :func:`make_figure`).
    projection, center, lat_center, direction : optional
        Projection kwargs (default to figure metadata stamped by
        :func:`make_figure`).
    resolution : int
        Number of vertices along each side of the wrap meridian.
        Default ``361`` (giving ``2*n - 1`` vertices in the full
        silhouette, dense enough for a smooth curve).
    color : str, optional
        Line color. Defaults to a theme-aware foreground (bright on
        dark themes, dark on light) so the edge stays legible
        without manual override.
    width : float
        Line width in pixels. Default ``1.0``.
    opacity : float
        Default ``1.0``.

    Returns
    -------
    trace : plotly.graph_objects.Scatter
        The frame-edge polyline trace.
    """
    go = _import_plotly()
    projection, center, lat_center, direction = _meta_defaults(
        fig, projection, center, lat_center, direction)

    if color is None:
        color = _theme_fg(fig)

    # Circular (zenithal globe) frames need the limb, not the wrap
    # meridian — the anti-meridian lies on the far hemisphere there, so
    # every wrap-meridian sample is un-projectable (NaN). Dispatch on the
    # registry's frame_shape; fall back to the wrap meridian if the
    # projection can't be resolved.
    is_circular = False
    try:
        from ..projections.registry import _resolve_projection
        _, _info = _resolve_projection(projection)
        is_circular = _info.frame_shape == 'circular'
    except (ValueError, AttributeError):
        pass

    if is_circular:
        # Trace the limb (great circle 90° from center), a hair inside
        # 90° so the exact-edge pinch doesn't produce a degenerate point.
        lon, lat = _limb_lonlat(center, lat_center, 89.95,
                                2 * int(resolution) - 1)
        limb_x, limb_y = _project(
            lon, lat, projection=projection, center=center,
            lat_center=lat_center, direction=direction)
        xs = np.concatenate([limb_x, [limb_x[0]]])
        ys = np.concatenate([limb_y, [limb_y[0]]])
    else:
        # Sample both sides of the wrap meridian at evenly spaced lats.
        # Keep just inside ±90 so the projection's pole-pinch doesn't
        # produce a degenerate ``project`` result.
        eps = 0.05
        lats = np.linspace(-89.95, 89.95, int(resolution))
        right_lons = np.full_like(lats, center + 180.0 - eps)
        left_lons = np.full_like(lats, center - 180.0 + eps)

        right_x, right_y = _project(
            right_lons, lats,
            projection=projection, center=center,
            lat_center=lat_center, direction=direction)
        left_x, left_y = _project(
            left_lons, lats[::-1],
            projection=projection, center=center,
            lat_center=lat_center, direction=direction)

        xs = np.concatenate([right_x, left_x, [right_x[0]]])
        ys = np.concatenate([right_y, left_y, [right_y[0]]])

    trace = go.Scatter(
        x=xs, y=ys, mode='lines',
        line=dict(color=color, width=float(width)),
        opacity=opacity, hoverinfo='skip', showlegend=False,
    )
    fig.add_trace(trace)
    return trace


# -- Reticle ----------------------------------------------------------------

_RETICLE_VALID_STYLES = ('plus', 'x', 'L', 'circle')
_RETICLE_STYLE_ALIASES = {'+': 'plus', 'o': 'circle'}

# Compass-direction → (dx_sign, dy_sign, xanchor, yanchor) for label
# offset placement. Mirrors the matplotlib ``_LABEL_DIRECTIONS`` table
# in skyplothelper/overlays/reticle.py, with mpl ha/va mapped to
# plotly annotation xanchor/yanchor (note plotly uses 'middle' where
# mpl uses 'center' for vertical centering).
_RETICLE_LABEL_DIRECTIONS = {
    'N':  (0,  +1, 'center', 'bottom'),
    'NE': (+1, +1, 'left',   'bottom'),
    'E':  (+1,  0, 'left',   'middle'),
    'SE': (+1, -1, 'left',   'top'),
    'S':  (0,  -1, 'center', 'top'),
    'SW': (-1, -1, 'right',  'top'),
    'W':  (-1,  0, 'right',  'middle'),
    'NW': (-1, +1, 'right',  'bottom'),
}
_RETICLE_VALID_LABEL_SIDES = ('auto',) + tuple(_RETICLE_LABEL_DIRECTIONS)


def _rotate_xy(
    pts: Sequence[tuple[float, float]], rotation_deg: float,
) -> list[tuple[float, float]]:
    """Rotate a sequence of (x, y) tuples by ``rotation_deg`` CCW."""
    if rotation_deg == 0:
        return list(pts)
    th = np.deg2rad(rotation_deg)
    c, s = np.cos(th), np.sin(th)
    return [(c * x - s * y, s * x + c * y) for x, y in pts]


def _reticle_segments_px(
    style: str, size: float, gap: float, rotation: float, circle_npts: int,
    circle_gap_deg: float,
) -> list[list[tuple[float, float]]]:
    """Build reticle arm segments in pixel offsets from the anchor.

    Mirrors the matplotlib ``_reticle_segments`` decomposition in
    :mod:`skyplothelper.overlays.reticle`. Returns a list of polylines
    (each a sequence of ``(dx, dy)`` pixel offsets) ready for plotly
    ``add_shape(type='path', xsizemode='pixel', ysizemode='pixel')``.
    """
    if style == 'plus':
        arms: list[list[tuple[float, float]]] = [
            [(0.0,  gap),   (0.0,  size)],   # N
            [(0.0, -gap),   (0.0, -size)],   # S
            [(gap,  0.0),   (size,  0.0)],   # E
            [(-gap, 0.0),   (-size, 0.0)],   # W
        ]
        return [_rotate_xy(arm, rotation) for arm in arms]
    if style == 'x':
        # Same as plus rotated 45°, then apply caller's rotation on top.
        return _reticle_segments_px(
            'plus', size, gap, rotation + 45.0,
            circle_npts, circle_gap_deg)
    if style == 'L':
        arms = [
            [(-gap, 0.0),   (-size, 0.0)],   # W
            [(0.0, -gap),   (0.0,  -size)],  # S
        ]
        return [_rotate_xy(arm, rotation) for arm in arms]
    if style == 'circle':
        if circle_gap_deg > 0:
            half = circle_gap_deg / 2.0
            start = np.deg2rad(half)
            end = np.deg2rad(360.0 - half)
            theta = np.linspace(start, end, max(circle_npts, 8))
        else:
            theta = np.linspace(0.0, 2 * np.pi, max(circle_npts, 8))
        ring = [(size * np.cos(t), size * np.sin(t)) for t in theta]
        return [_rotate_xy(ring, rotation)]
    raise ValueError(
        f"style must be one of {_RETICLE_VALID_STYLES!r} (or aliases "
        f"{list(_RETICLE_STYLE_ALIASES)!r}), got {style!r}")


def _segments_to_path(segment: Sequence[tuple[float, float]]) -> str:
    """Convert a pixel-offset polyline to an SVG path string for
    ``plotly.add_shape(type='path')``."""
    head = segment[0]
    out = [f"M{head[0]},{head[1]}"]
    for pt in segment[1:]:
        out.append(f"L{pt[0]},{pt[1]}")
    return ' '.join(out)


def _resolve_auto_label_side_plotly(
    anchor_x: float, anchor_y: float, projection: str, center: float,
    lat_center: float, direction: str,
) -> str:
    """Pick the corner direction pointing into the largest free quadrant
    of the projected data extent.

    Plotly counterpart to
    :func:`skyplothelper.overlays.reticle._resolve_auto_label_side`.
    Samples a coarse lon/lat grid through the projection primitive to
    get the data x/y range, then scores NE/NW/SE/SW by the tightest
    perpendicular room toward each corner.
    """
    lons = np.linspace(-180.0 + center, 180.0 + center, 37)
    lats = np.linspace(-85.0, 85.0, 19)
    LL, BB = np.meshgrid(lons, lats)
    xs_grid, ys_grid = _project(LL.ravel(), BB.ravel(),
                                  projection=projection, center=center,
                                  lat_center=lat_center,
                                  direction=direction)
    if not np.any(np.isfinite(xs_grid)):
        return 'NE'
    x_min = float(np.nanmin(xs_grid))
    x_max = float(np.nanmax(xs_grid))
    y_min = float(np.nanmin(ys_grid))
    y_max = float(np.nanmax(ys_grid))
    room_E = x_max - anchor_x
    room_W = anchor_x - x_min
    room_N = y_max - anchor_y
    room_S = anchor_y - y_min
    scores = {
        'NE': min(room_N, room_E),
        'NW': min(room_N, room_W),
        'SE': min(room_S, room_E),
        'SW': min(room_S, room_W),
    }
    return max(scores, key=lambda k: scores[k])


def add_reticle(fig: Any, lon: SkyCoord | float,
                lat: float | None = None, *,
                style: str = 'plus', size: float = 12.0, gap: float = 4.0,
                rotation: float = 0.0,
                projection: str | None = None, center: float | None = None,
                lat_center: float | None = None, direction: str | None = None,
                color: str = 'white', lw: float = 1.2,
                stroke_color: str | None = 'black', stroke_lw: float = 2.4,
                circle_npts: int = 64, circle_gap_deg: float = 0.0,
                label: str | None = None, label_side: str = 'auto',
                label_offset: float = 2.0,
                label_color: str | None = None,
                label_fontsize: int = 11) -> tuple[list[Any], Any]:
    """Add a target-highlight reticle anchored at a sky position.

    Plotly counterpart to :class:`skyplothelper.Reticle`. The reticle
    sits at ``(lon, lat)`` and is drawn at a fixed pixel size — its
    arms stay the same size on screen regardless of zoom — via plotly's
    ``xsizemode='pixel'`` / ``ysizemode='pixel'`` shape mode anchored
    to the projected data position. The classical finding-chart
    crosshair + open-circle markers, with optional contrast stroke
    and label.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    lon, lat : float
        Anchor position in degrees (ICRS).
    style : {'plus', 'x', 'L', 'circle'} or {'+', 'o'}
        Reticle shape. ``'plus'`` (default) — vertical + horizontal
        arms with a central gap (the classical target-acquisition
        reticle). ``'x'`` — same geometry rotated 45°. ``'L'`` — two
        arms in one quadrant (``rotation=0`` opens the upper-right;
        positive ``rotation`` walks the L CCW). ``'circle'`` —
        open ring; pair with ``circle_gap_deg`` for a broken-circle
        variant.
    size : float
        Outer half-extent (arm length / circle radius) in pixels.
        Default ``12``.
    gap : float
        Central empty zone half-extent in pixels. Inner endpoints of
        each arm sit at ``gap`` so the target itself isn't obscured.
        Default ``4``. Ignored for ``'circle'``.
    rotation : float
        Whole-reticle rotation in degrees CCW. Default ``0``.
    projection, center, lat_center, direction : optional
        Projection kwargs (default to figure metadata).
    color : str
        Body color of the arms. Default ``'white'`` — the
        dark-sky-readable default.
    lw : float
        Arm line width in pixels. Default ``1.2``.
    stroke_color : str or None
        Color of the optional stroke drawn under each arm — the plotly
        equivalent of the matplotlib ``patheffects.withStroke``
        outline. Default ``'black'``; pass ``None`` to disable.
    stroke_lw : float
        Total stroke line width in pixels. The body draws on top of
        the stroke, so the visible stroke on each side is
        ``(stroke_lw - lw) / 2``. Default ``2.4``.
    circle_npts : int
        Sample count for the ``'circle'`` style polyline. Default
        ``64`` — visually smooth across typical sizes.
    circle_gap_deg : float
        Angular wedge cut from the ``'circle'`` style (degrees,
        centered on +x before rotation). Default ``0`` (closed ring).
    label : str, optional
        Text drawn next to the reticle.
    label_side : {'auto', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'}
        Side to place the label on. ``'auto'`` (default) picks the
        corner pointing into the largest free quadrant of the
        projected data extent.
    label_offset : float
        Extra gap in pixels between the reticle outer extent and the
        label. Default ``2``.
    label_color : str, optional
        Label color. Defaults to the body ``color``.
    label_fontsize : int
        Default ``11``.

    Returns
    -------
    shapes : list of plotly Shape
        The reticle arm shapes appended to ``fig.layout.shapes``
        (each style includes both stroke and body layers when
        ``stroke_color`` is enabled, so the list is at least
        ``len(segments) * 2``).
    label_ann : plotly Annotation or None
        The label annotation if ``label`` was supplied, else ``None``.
    """
    style_resolved = _RETICLE_STYLE_ALIASES.get(style, style)
    if style_resolved not in _RETICLE_VALID_STYLES:
        raise ValueError(
            f"style must be one of {_RETICLE_VALID_STYLES!r} "
            f"(or aliases {list(_RETICLE_STYLE_ALIASES)!r}), "
            f"got {style!r}")
    if label_side not in _RETICLE_VALID_LABEL_SIDES:
        raise ValueError(
            f"label_side must be one of {_RETICLE_VALID_LABEL_SIDES!r}, "
            f"got {label_side!r}")

    projection, center, lat_center, direction = _meta_defaults(
        fig, projection, center, lat_center, direction)

    lon, lat = _resolve_lonlat(fig, lon, lat, 'add_reticle')

    xs_a, ys_a = _project(np.array([float(lon)]), np.array([float(lat)]),
                            projection=projection, center=center,
                            lat_center=lat_center, direction=direction)
    anchor_x = float(xs_a[0])
    anchor_y = float(ys_a[0])

    segments = _reticle_segments_px(
        style_resolved, float(size), float(gap), float(rotation),
        int(circle_npts), float(circle_gap_deg))

    n_shapes_before = (len(fig.layout.shapes)
                       if fig.layout.shapes else 0)

    use_stroke = stroke_color is not None and stroke_lw > lw

    for segment in segments:
        if len(segment) < 2:
            continue
        path = _segments_to_path(segment)
        # Stroke layer drawn first so the body lands on top.
        if use_stroke:
            fig.add_shape(
                type='path', path=path,
                xref='x', yref='y',
                xanchor=anchor_x, yanchor=anchor_y,
                xsizemode='pixel', ysizemode='pixel',
                line=dict(color=stroke_color, width=float(stroke_lw)),
            )
        fig.add_shape(
            type='path', path=path,
            xref='x', yref='y',
            xanchor=anchor_x, yanchor=anchor_y,
            xsizemode='pixel', ysizemode='pixel',
            line=dict(color=color, width=float(lw)),
        )

    shapes_added = list(fig.layout.shapes[n_shapes_before:])

    label_ann = None
    if label is not None:
        side = label_side
        if side == 'auto':
            side = _resolve_auto_label_side_plotly(
                anchor_x, anchor_y, projection, center,
                lat_center, direction)
        dx_sign, dy_sign, xanchor, yanchor = (
            _RETICLE_LABEL_DIRECTIONS[side])
        outer = max(
            (np.hypot(x, y) for seg in segments for (x, y) in seg),
            default=0.0)
        radial = float(outer) + float(label_offset)
        offset_x = dx_sign * radial
        offset_y = dy_sign * radial
        text_color = (label_color if label_color is not None else color)

        n_anns_before = (len(fig.layout.annotations)
                         if fig.layout.annotations else 0)
        fig.add_annotation(
            x=anchor_x, y=anchor_y,
            xref='x', yref='y',
            text=str(label),
            showarrow=False,
            xshift=offset_x, yshift=offset_y,
            font=dict(color=text_color, size=int(label_fontsize)),
            xanchor=xanchor, yanchor=yanchor,
        )
        label_ann = fig.layout.annotations[n_anns_before]

    return shapes_added, label_ann


# -- Ruler ------------------------------------------------------------------

_RULER_TICK_SIDES = ('both', 'left', 'right', 'none')
_RULER_SIDES = ('auto', 'left', 'right')
# Minors additionally accept 'auto' — follow whatever tick_side resolves to.
_RULER_MINOR_SIDES = ('auto', *_RULER_TICK_SIDES)
_RULER_UNITS = {'uas': 3.6e9, 'mas': 3.6e6, 'arcsec': 3600.0,
                'arcmin': 60.0, 'deg': 1.0}
_RULER_ENDCAP_STYLES = ('none', 'tick', 'arrow')
_RULER_ENDCAPS = ('both', 'start', 'end', 'none')


def _haversine_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in degrees between two sphere points."""
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return float(np.degrees(2 * np.arcsin(np.sqrt(a))))


def _ruler_resolve_unit(
    label_unit: str, total_deg: float,
) -> tuple[str, float]:
    """Pick a display unit + scaling factor (units per degree)."""
    if label_unit == 'auto':
        if total_deg < 1.0 / 3.6e6:        # < 1 mas → microarcsec
            return 'uas', 3.6e9
        if total_deg < 1.0 / 3600.0:       # < 1 arcsec → milliarcsec (VLBI)
            return 'mas', 3.6e6
        if total_deg < 1.0:
            return 'arcsec', 3600.0
        if total_deg < 60.0:
            return 'arcmin', 60.0
        return 'deg', 1.0
    if label_unit not in _RULER_UNITS:
        raise ValueError(
            f"label_unit must be 'auto' or one of {sorted(_RULER_UNITS)}, "
            f"got {label_unit!r}")
    return label_unit, _RULER_UNITS[label_unit]


def _ruler_tick_positions(
    total_in_unit: float, n_ticks: int | str,
    tick_interval: float | None, tick_positions: npt.ArrayLike | None,
    lambda0: float = 0.0,
) -> np.ndarray:
    """Resolve tick positions in display units along the ruler.

    Returns signed values measured from the value-0 tick, which sits at
    fractional position ``lambda0`` along the ruler (matching the mpl
    :class:`~skyplothelper.Ruler` convention). With the default
    ``lambda0=0`` the zero tick is at xy1 and all values are ``>= 0``.
    """
    from ..overlays.ruler import _nice_interval

    if tick_positions is not None:
        return np.asarray(tick_positions, dtype=float)

    # Value range spanned by the ruler, signed about the zero tick.
    zero = lambda0 * total_in_unit
    d_min = -zero
    d_max = total_in_unit - zero

    if isinstance(n_ticks, int):
        return np.linspace(d_min, d_max, max(2, n_ticks))
    if tick_interval is not None:
        step = float(tick_interval)
    elif n_ticks == 'auto':
        step = _nice_interval(total_in_unit, target_n=4)
    else:
        raise ValueError(
            f"n_ticks must be 'auto' or an int, got {n_ticks!r}")
    if step <= 0:
        return np.array([d_min, d_max])
    # Regular ticks at k * step (signed) inside [d_min, d_max].
    k_min = int(np.ceil((d_min - 1e-9) / step))
    k_max = int(np.floor((d_max + 1e-9) / step))
    return np.array([k * step for k in range(k_min, k_max + 1)], dtype=float)


def _ruler_major_interval(
    total_in_unit: float, n_ticks: int | str,
    tick_interval: float | None, tick_positions: npt.ArrayLike | None,
    majors: np.ndarray,
) -> float | None:
    """The major-tick spacing actually in use, in display units.

    Mirrors the resolution order of :func:`_ruler_tick_positions`. With
    explicit ``tick_positions`` there is no single spacing, so fall back to
    the median gap of the resolved majors. ``None`` when undeterminable.
    """
    if tick_positions is not None:
        if len(majors) < 2:
            return None
        return float(np.median(np.diff(np.sort(majors))))
    if isinstance(n_ticks, int):
        return total_in_unit / max(1, max(2, n_ticks) - 1)
    if tick_interval is not None:
        return float(tick_interval)
    from ..overlays.ruler import _nice_interval
    return _nice_interval(total_in_unit, target_n=4)


def _ruler_minor_positions(
    total_in_unit: float, majors: np.ndarray, n_ticks: int | str,
    tick_interval: float | None, tick_positions: npt.ArrayLike | None,
    minor_ticks: int | str | None, minor_tick_interval: float | None,
    lambda0: float = 0.0,
) -> np.ndarray:
    """Signed minor-tick values (display units) about the zero tick.

    Same semantics as the mpl :class:`~skyplothelper.Ruler`: ``minor_ticks``
    is a subdivision count of the major interval, and any minor coinciding
    with a major is dropped so the two never overprint.
    """
    from ..overlays.ruler import _auto_minor_subdivisions

    if minor_ticks is None and minor_tick_interval is None:
        return np.array([], dtype=float)
    if total_in_unit <= 0:
        return np.array([], dtype=float)
    zero = lambda0 * total_in_unit
    d_min, d_max = -zero, total_in_unit - zero

    if minor_tick_interval is not None:
        step = float(minor_tick_interval)
    else:
        subdiv = minor_ticks
        if subdiv is None:
            return np.array([], dtype=float)
        major = _ruler_major_interval(total_in_unit, n_ticks, tick_interval,
                                      tick_positions, majors)
        if major is None or major <= 0:
            return np.array([], dtype=float)
        n = (_auto_minor_subdivisions(major)
             if subdiv == 'auto' else int(subdiv))
        if n < 2:
            return np.array([], dtype=float)
        step = major / n
    if step <= 0:
        return np.array([], dtype=float)

    tol = 1e-6 * step
    k_min = int(np.ceil((d_min - 1e-9) / step))
    k_max = int(np.floor((d_max + 1e-9) / step))
    out = [k * step for k in range(k_min, k_max + 1)]
    if len(majors):
        out = [d for d in out
               if not np.any(np.abs(majors - d) < tol)]
    return np.array(out, dtype=float)


# mpl linestyle spellings → plotly ``line.dash`` names. Plotly's own dash
# names (and ``None`` for solid) pass through unchanged.
_RULER_DASH_MAP = {
    '-': 'solid', 'solid': 'solid',
    '--': 'dash', 'dashed': 'dash',
    ':': 'dot', 'dotted': 'dot',
    '-.': 'dashdot', 'dashdot': 'dashdot',
}


def _ruler_dash(ls: str | None) -> str | None:
    """Map an mpl-style linestyle to a plotly ``line.dash`` value."""
    if ls is None:
        return None
    return _RULER_DASH_MAP.get(ls, ls)


def _slerp_one_point(
    lon1: float, lat1: float, lon2: float, lat2: float, frac: float,
) -> tuple[float, float]:
    """Single-point version of geometry._slerp."""
    from ..geometry._densify import _slerp
    if frac <= 0.0:
        return float(lon1), float(lat1)
    if frac >= 1.0:
        return float(lon2), float(lat2)
    lons, lats = _slerp(lon1, lat1, lon2, lat2, 3)
    # _slerp returns 3 evenly-spaced points incl. endpoints; the
    # caller-supplied frac is rarely 0.5, so directly interpolate
    # along the chord in (lon, lat) and re-slerp at finer step.
    # Simpler: sample at higher density and linear-interp.
    n = 64
    lons, lats = _slerp(lon1, lat1, lon2, lat2, n)
    t = frac * (n - 1)
    i = int(np.clip(np.floor(t), 0, n - 2))
    u = t - i
    return (float(lons[i] + u * (lons[i + 1] - lons[i])),
            float(lats[i] + u * (lats[i + 1] - lats[i])))


def _ruler_textangle(
    math_angle_deg: float, rotation_spec: str | float,
) -> float:
    """Resolve text rotation (degrees, plotly textangle convention).

    Plotly ``textangle`` is measured clockwise from horizontal, while
    the local tangent angle is measured counter-clockwise. The
    function returns the value to pass as ``textangle`` so the text
    sits flat along the local tangent (``'auto'``), horizontal, or
    perpendicular, plus a numeric override.

    Auto-rotation also flips the text 180° when it would otherwise
    render upside down (math angle outside ``[-90, 90]``) — keeps
    every label right-side-up.
    """
    if rotation_spec == 'horizontal':
        return 0.0
    if rotation_spec == 'perpendicular':
        a = math_angle_deg + 90.0
    elif rotation_spec == 'auto':
        a = math_angle_deg
    else:
        return float(rotation_spec)
    # Keep right-side-up.
    while a > 90.0:
        a -= 180.0
    while a < -90.0:
        a += 180.0
    return -a  # CW for plotly


def add_ruler(fig: Any, lon1: float, lat1: float, lon2: float, lat2: float, *,
              projection: str | None = None, center: float | None = None,
              lat_center: float | None = None, direction: str | None = None,
              geodesic: bool = False, n_geodesic_pts: int = 64,
              n_ticks: int | str = 'auto', tick_interval: float | None = None,
              tick_positions: npt.ArrayLike | None = None,
              tick_length: float = 4.0, tick_side: str = 'both',
              minor_ticks: int | str | bool | None = None,
              minor_tick_interval: float | None = None,
              minor_tick_length: float | None = None,
              minor_tick_side: str = 'auto',
              minor_tick_color: str | None = None,
              minor_tick_lw: float | None = None,
              lambda0: float = 0.0,
              labels: bool = True, label_unit: str = 'auto',
              fmt: str | None = None,
              label_fmt: Callable[[float, str], str] | None = None,
              label_offset: float = 2.0, label_side: str = 'auto',
              label_rotation: str | float = 'auto', label_fontsize: int = 10,
              title: str | None = None, title_offset: float = 10.0,
              title_side: str = 'auto',
              title_rotation: str | float = 'auto', title_fontsize: int = 11,
              endcap_style: str = 'none', endcaps: str = 'both',
              endcap_length_scale: float = 1.5,
              color: str = 'white', lw: float = 1.0, opacity: float = 1.0,
              tick_color: str | None = None, tick_lw: float | None = None,
              tick_ls: str | None = None,
              endcap_color: str | None = None, endcap_lw: float | None = None,
              stroke_color: str | None = 'black', stroke_lw: float = 2.4,
              name: str | None = None,
              hover: bool | str = False,
              ) -> tuple[Any, list[Any], list[Any]]:
    """Add a two-point distance ruler with ticks + labels.

    Plotly counterpart to :class:`skyplothelper.Ruler` (focused
    subset). Draws a chord (or great-circle arc, ``geodesic=True``)
    between two sphere positions, places tick marks at evenly-spaced
    intervals along the arc, and labels each tick with the angular
    distance from the start. Tick marks and label offsets are sized
    in pixels so they stay stable under zoom — same pixel-stable
    pattern as :func:`add_reticle`.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    lon1, lat1 : float
        Starting endpoint in degrees (ICRS).
    lon2, lat2 : float
        Ending endpoint in degrees (ICRS).
    projection, center, lat_center, direction : optional
        Projection kwargs (default to figure metadata).
    geodesic : bool
        If ``True``, sample the great-circle arc between the
        endpoints; otherwise draw a straight chord in projection
        coords. Default ``False``. Use ``True`` for separations more
        than a few degrees, where the projected chord visibly
        diverges from the on-sky geodesic.
    n_geodesic_pts : int
        Samples along the geodesic arc. Default ``64``.
    n_ticks : ``'auto'`` or int
        ``'auto'`` (default) picks a 1/2/5×10^n interval giving ~4
        ticks across the ruler. An integer forces that many evenly
        spaced ticks.
    tick_interval : float, optional
        Explicit tick spacing in the active ``label_unit`` (overrides
        ``n_ticks``).
    tick_positions : sequence of float, optional
        Explicit tick positions in the active ``label_unit``
        (overrides ``n_ticks`` and ``tick_interval``).
    tick_length : float
        Half-length of each tick in pixels. Default ``4``.
    tick_side : {'both', 'left', 'right', 'none'}
        Which side of the line ticks extend on (relative to the
        ``xy1 → xy2`` direction in projection). Default ``'both'``.
    minor_ticks : int, 'auto', bool or None
        Unlabeled minor ticks between the majors — fine gradations that also
        make the ruler's orientation easier to read. A **subdivision count**
        (matching matplotlib's AutoMinorLocator and the mpl
        :class:`~skyplothelper.Ruler`): ``n`` splits each major interval into
        ``n``, i.e. ``n-1`` minors between adjacent majors. ``'auto'`` /
        ``True`` picks 4 or 5 from the major step; ``None`` (default) /
        ``False`` is off. Minors coinciding with a major are dropped, and
        endpoints carrying an endcap are skipped.
    minor_tick_interval : float, optional
        Explicit minor spacing in the active ``label_unit``. Overrides
        ``minor_ticks``.
    minor_tick_length : float, optional
        Half-length of each minor tick in pixels. Default ``None`` → half of
        ``tick_length``.
    minor_tick_side : {'auto', 'both', 'left', 'right', 'none'}
        Side for the minor ticks. ``'auto'`` (default) follows ``tick_side``.
    minor_tick_color : str, optional
        Minor tick color. Defaults to the major tick color.
    minor_tick_lw : float, optional
        Minor tick width. Defaults to the major tick width.
    lambda0 : float
        Fractional position along the ruler in ``[0, 1]`` where the
        value-0 tick lands. Default ``0`` (zero at the start, all tick
        values ``>= 0``). ``0.5`` centers the zero and produces a
        symmetric ±-valued ruler.
    labels : bool
        Show per-tick labels. Default ``True``.
    label_unit : {'auto', 'uas', 'mas', 'arcsec', 'arcmin', 'deg'}
        Unit for tick values. ``'auto'`` (default) picks based on the
        total span (uas under 1 mas, mas under 1 arcsec, arcsec under 1°,
        arcmin under 60°, else deg) — ``'mas'`` / ``'uas'`` suit compact /
        VLBI fields shown in the offset-coords FITS viewer.
    fmt : str, optional
        printf-style format for the numeric portion of each label
        (e.g. ``'%.1f'``). Default 4-sig-fig with trim. Ignored when
        ``label_fmt`` is supplied.
    label_fmt : callable, optional
        Custom label formatter with signature
        ``label_fmt(value_arcsec, unit) -> str`` (``value_arcsec`` is the
        signed tick value in arcseconds, ``unit`` the resolved unit name).
        Overrides ``fmt`` and the built-in unit suffix when given.
    label_offset : float
        Pixel gap between the tick tip and the label. Default ``2``.
    label_side : {'auto', 'left', 'right'}
        Side of the line for label placement. ``'auto'`` (default)
        matches the tick side when one-sided; otherwise defaults to
        ``'right'``.
    label_rotation : {'auto', 'horizontal', 'perpendicular'} or float
        Label rotation. ``'auto'`` (default) — parallel to the line
        at each tick. Numeric values are in degrees CCW.
    label_fontsize : int
        Default ``10``.
    title : str, optional
        Caption at the line's midpoint. Sits on the opposite side
        from the labels by default.
    title_offset : float
        Pixel gap beyond the tick tip on the title side. Default
        ``10``.
    title_side : {'auto', 'left', 'right'}
        Title side. ``'auto'`` (default) is opposite the labels.
    title_rotation : {'auto', 'horizontal', 'perpendicular'} or float
        Default ``'auto'``.
    title_fontsize : int
        Default ``11``.
    endcap_style : {'none', 'tick', 'arrow'}
        How endpoint marks render. Default ``'none'``.
    endcaps : {'both', 'start', 'end', 'none'}
        Which endpoints get an endcap. Default ``'both'``.
    endcap_length_scale : float
        For ``endcap_style='tick'``, the endpoint tick length is
        ``tick_length × endcap_length_scale``. Default ``1.5``.
    color : str
        Main line / tick / label color. Default ``'white'``.
    lw : float
        Main line width in pixels. Default ``1.0``.
    opacity : float
        Default ``1.0``.
    tick_color : str, optional
        Tick-mark color. Defaults to ``color``.
    tick_lw : float, optional
        Tick-mark line width in pixels. Defaults to ``lw``.
    tick_ls : str, optional
        Tick-mark line style — an mpl spelling (``'-'``, ``'--'``,
        ``':'``, ``'-.'``) or a plotly ``line.dash`` name. Default solid.
    endcap_color : str, optional
        Endcap color. Defaults to ``tick_color``.
    endcap_lw : float, optional
        Endcap line width in pixels. Defaults to ``tick_lw``.
    stroke_color : str or None
        Optional stroke under the main line for legibility. Default
        ``'black'``; pass ``None`` to disable.
    stroke_lw : float
        Total stroke width in pixels. Default ``2.4``.
    name : str, optional
        Trace name for the main line.
    hover : False / True / str
        Hover behavior — see :func:`add_great_circle`.

    Returns
    -------
    main_trace : plotly.graph_objects.Scatter
        The main line trace.
    tick_shapes : list of plotly Shape
        The tick / endcap shapes (each tick contributes 1 or 2
        depending on ``tick_side``; stroke adds another layer).
    label_anns : list of plotly Annotation
        Tick label annotations plus the title annotation if set.
    """
    go = _import_plotly()
    if tick_side not in _RULER_TICK_SIDES:
        raise ValueError(
            f"tick_side must be one of {_RULER_TICK_SIDES}, got {tick_side!r}")
    if minor_tick_side not in _RULER_MINOR_SIDES:
        raise ValueError(
            f"minor_tick_side must be one of {_RULER_MINOR_SIDES}, "
            f"got {minor_tick_side!r}")
    if minor_ticks is False:
        minor_ticks = None
    if minor_ticks is True:
        minor_ticks = 'auto'
    if (minor_ticks is not None and minor_ticks != 'auto'
            and not isinstance(minor_ticks, int)):
        raise ValueError(
            "minor_ticks must be None/False (off), True/'auto', or an int "
            f"subdivision count, got {minor_ticks!r}")
    if isinstance(minor_ticks, int) and minor_ticks < 2:
        raise ValueError(
            "minor_ticks is a subdivision count and must be >= 2 "
            f"(n splits each major interval into n), got {minor_ticks!r}")
    if label_side not in _RULER_SIDES:
        raise ValueError(
            f"label_side must be one of {_RULER_SIDES}, got {label_side!r}")
    if title_side not in _RULER_SIDES:
        raise ValueError(
            f"title_side must be one of {_RULER_SIDES}, got {title_side!r}")
    if endcap_style not in _RULER_ENDCAP_STYLES:
        raise ValueError(
            f"endcap_style must be one of {_RULER_ENDCAP_STYLES}, "
            f"got {endcap_style!r}")
    if endcaps not in _RULER_ENDCAPS:
        raise ValueError(
            f"endcaps must be one of {_RULER_ENDCAPS}, got {endcaps!r}")
    lambda0 = float(lambda0)
    if not 0.0 <= lambda0 <= 1.0:
        raise ValueError(
            f"lambda0 must be in [0, 1] (fractional position of the "
            f"value-0 tick), got {lambda0}")

    # Resolve tick / endcap styling: ticks default to the main line's
    # color / width; endcaps in turn default to the tick styling.
    tick_color = color if tick_color is None else tick_color
    tick_lw = lw if tick_lw is None else float(tick_lw)
    tick_dash = _ruler_dash(tick_ls)
    endcap_color = tick_color if endcap_color is None else endcap_color
    endcap_lw = tick_lw if endcap_lw is None else float(endcap_lw)
    # Minors in turn default to the major ticks' look / side / half-length,
    # so turning them on is a one-knob change.
    minor_tick_color = (tick_color if minor_tick_color is None
                        else minor_tick_color)
    minor_tick_lw = (tick_lw if minor_tick_lw is None
                     else float(minor_tick_lw))
    minor_tick_length = (0.5 * float(tick_length) if minor_tick_length is None
                         else float(minor_tick_length))
    resolved_minor_side = (tick_side if minor_tick_side == 'auto'
                           else minor_tick_side)

    projection, center, lat_center, direction = _meta_defaults(
        fig, projection, center, lat_center, direction)
    proj = _overlay_projector(fig, projection, center, lat_center, direction)

    # Total angular distance + display unit.
    total_deg = _haversine_deg(lon1, lat1, lon2, lat2)
    unit_name, unit_scale = _ruler_resolve_unit(label_unit, total_deg)
    total_in_unit = total_deg * unit_scale

    # Main-line samples.
    if geodesic:
        from ..geometry._densify import _slerp
        # _slerp excludes its endpoint (it's built for chaining densified
        # segments). The ruler line is a single span, so append the true
        # endpoint — otherwise the drawn line stops one step short while
        # the endpoint tick / arrow cap sit at the exact endpoint, leaving
        # a visible gap.
        line_lons, line_lats = _slerp(lon1, lat1, lon2, lat2,
                                        int(n_geodesic_pts))
        line_lons = np.append(line_lons, float(lon2))
        line_lats = np.append(line_lats, float(lat2))
    else:
        line_lons = np.array([float(lon1), float(lon2)])
        line_lats = np.array([float(lat1), float(lat2)])
    line_x, line_y = proj.project_points(line_lons, line_lats)
    # Main-line trace coords, wrap-split at the projection seam so a
    # geodesic spanning the seam (or a seam-crossing chord) breaks
    # cleanly at the wrap edge instead of streaking across the canvas.
    # ``line_x`` / ``line_y`` stay un-split for the tangent computation
    # below; for a line that doesn't cross the seam (or a seamless FITS
    # frame) the split is a no-op so ``main_x`` / ``main_y`` equal
    # ``line_x`` / ``line_y``.
    main_x, main_y = proj.project_polyline(line_lons, line_lats)

    # Tick positions (signed display-unit values about the zero tick at
    # fraction lambda0) and the corresponding (lon, lat).
    positions = _ruler_tick_positions(total_in_unit, n_ticks,
                                        tick_interval, tick_positions, lambda0)
    zero = lambda0 * total_in_unit
    d_min, d_max = -zero, total_in_unit - zero
    positions = positions[(positions >= d_min - 1e-9)
                           & (positions <= d_max + 1e-9)]
    if total_in_unit > 0:
        fracs = (positions + zero) / total_in_unit
    else:
        fracs = np.zeros_like(positions)

    def _frac_geometry(
        fs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(x, y, tangents, perp_left)`` for fractional positions *fs*.

        Shared by the major and minor tick passes — both need the same
        position + orientation math, only the drawn length differs.
        """
        lons = np.empty_like(fs)
        lats = np.empty_like(fs)
        for k, f in enumerate(fs):
            if geodesic:
                tl, tb = _slerp_one_point(lon1, lat1, lon2, lat2, float(f))
            else:
                tl = float(lon1 + f * (lon2 - lon1))
                tb = float(lat1 + f * (lat2 - lat1))
            lons[k] = tl
            lats[k] = tb
        px, py = proj.project_points(lons, lats)

        # Per-tick tangent direction in projection (unit vector). For a
        # chord this is constant; for a geodesic it varies with position.
        tg = np.zeros((len(fs), 2))
        for k in range(len(fs)):
            if geodesic and len(fs) > 0:
                df = 1.0 / max(2, int(n_geodesic_pts))
                f1 = max(0.0, float(fs[k]) - df / 2)
                f2 = min(1.0, float(fs[k]) + df / 2)
                l1, b1 = _slerp_one_point(lon1, lat1, lon2, lat2, f1)
                l2, b2 = _slerp_one_point(lon1, lat1, lon2, lat2, f2)
                x1, y1 = proj.project_points(np.array([l1]), np.array([b1]))
                x2, y2 = proj.project_points(np.array([l2]), np.array([b2]))
                dx = float(x2[0] - x1[0])
                dy = float(y2[0] - y1[0])
            else:
                dx = float(line_x[-1] - line_x[0])
                dy = float(line_y[-1] - line_y[0])
            norm = np.hypot(dx, dy)
            if norm > 0:
                tg[k] = (dx / norm, dy / norm)
            else:
                tg[k] = (1.0, 0.0)

        # Perpendicular unit vector per tick (left-of-line direction). The
        # figure's xaxis/yaxis are locked at scaleratio=1 by make_figure so
        # a unit vector in projection coords equals one in pixel coords.
        return px, py, tg, np.column_stack([-tg[:, 1], tg[:, 0]])

    tick_x, tick_y, tangents, perp_left = _frac_geometry(fracs)

    # Resolve label / title sides. The "left" side is +perp_left.
    if label_side == 'auto':
        resolved_label_side = ('left' if tick_side == 'left'
                                else 'right' if tick_side == 'right'
                                else 'right')
    else:
        resolved_label_side = label_side
    if title_side == 'auto':
        resolved_title_side = ('right' if resolved_label_side == 'left'
                                else 'left')
    else:
        resolved_title_side = title_side

    label_sign = +1.0 if resolved_label_side == 'left' else -1.0
    title_sign = +1.0 if resolved_title_side == 'left' else -1.0

    n_shapes_before = (len(fig.layout.shapes)
                        if fig.layout.shapes else 0)
    n_anns_before = (len(fig.layout.annotations)
                      if fig.layout.annotations else 0)

    # Main line trace, with optional stroke under the body.
    use_stroke = stroke_color is not None and stroke_lw > lw
    main_traces: list[Any] = []
    if use_stroke:
        main_traces.append(go.Scatter(
            x=main_x, y=main_y, mode='lines',
            line=dict(color=stroke_color, width=float(stroke_lw)),
            opacity=opacity, hoverinfo='skip', showlegend=False,
        ))
    main_kw: dict[str, Any] = dict(
        x=main_x, y=main_y, mode='lines',
        line=dict(color=color, width=float(lw)),
        opacity=opacity, name=name,
        showlegend=name is not None,
    )
    if hover is False or hover is None:
        main_kw['hoverinfo'] = 'skip'
    elif hover is True:
        main_kw['hovertemplate'] = (
            (f"<b>{name}</b><br>" if name else "")
            + f"length: {total_in_unit:.4g} {unit_name}<extra></extra>"
        )
    else:
        tpl = str(hover)
        if '<extra>' not in tpl:
            tpl += '<extra></extra>'
        main_kw['hovertemplate'] = tpl
    main_traces.append(go.Scatter(**main_kw))
    for t in main_traces:
        fig.add_trace(t)
    main_trace = main_traces[-1]

    # Tick mark sides (relative to the line direction).
    tick_side_signs: list[float]
    if tick_side == 'both':
        tick_side_signs = [+1.0, -1.0]
    elif tick_side == 'left':
        tick_side_signs = [+1.0]
    elif tick_side == 'right':
        tick_side_signs = [-1.0]
    else:   # 'none'
        tick_side_signs = []

    # Minor ticks (never labeled), added before the majors so a major tick
    # sits on top wherever the two nearly touch.
    minor_side_signs: list[float] = []
    if resolved_minor_side in ('both', 'left'):
        minor_side_signs.append(+1.0)
    if resolved_minor_side in ('both', 'right'):
        minor_side_signs.append(-1.0)
    if minor_side_signs:
        minor_vals = _ruler_minor_positions(
            total_in_unit, positions, n_ticks, tick_interval, tick_positions,
            minor_ticks, minor_tick_interval, lambda0)
        if len(minor_vals):
            minor_fracs = (minor_vals + zero) / total_in_unit
            m_x, m_y, _m_tg, m_perp = _frac_geometry(minor_fracs)
            for k in range(len(minor_fracs)):
                # An endcap visually replaces any tick at that endpoint.
                if endcap_style != 'none' and (
                        (minor_fracs[k] <= 1e-9
                         and endcaps in ('both', 'start'))
                        or (minor_fracs[k] >= 1.0 - 1e-9
                            and endcaps in ('both', 'end'))):
                    continue
                for sign in minor_side_signs:
                    dx_px = sign * m_perp[k, 0] * minor_tick_length
                    dy_px = sign * m_perp[k, 1] * minor_tick_length
                    if use_stroke:
                        fig.add_shape(
                            type='line', xref='x', yref='y',
                            xanchor=float(m_x[k]), yanchor=float(m_y[k]),
                            x0=0, y0=0, x1=dx_px, y1=dy_px,
                            xsizemode='pixel', ysizemode='pixel',
                            line=dict(color=stroke_color,
                                      width=float(stroke_lw)),
                        )
                    fig.add_shape(
                        type='line', xref='x', yref='y',
                        xanchor=float(m_x[k]), yanchor=float(m_y[k]),
                        x0=0, y0=0, x1=dx_px, y1=dy_px,
                        xsizemode='pixel', ysizemode='pixel',
                        line=dict(color=minor_tick_color,
                                  width=float(minor_tick_lw),
                                  dash=tick_dash),
                    )

    # Tick marks (pixel-stable shapes). Skip endpoint ticks when an
    # endcap style is requested at that endpoint.
    for k in range(len(positions)):
        is_start = (k == 0)
        is_end = (k == len(positions) - 1)
        if endcap_style != 'none':
            if (is_start and endcaps in ('both', 'start')
                    or is_end and endcaps in ('both', 'end')):
                continue
        for sign in tick_side_signs:
            dx_px = sign * perp_left[k, 0] * float(tick_length)
            dy_px = sign * perp_left[k, 1] * float(tick_length)
            if use_stroke:
                fig.add_shape(
                    type='line', xref='x', yref='y',
                    xanchor=float(tick_x[k]), yanchor=float(tick_y[k]),
                    x0=0, y0=0, x1=dx_px, y1=dy_px,
                    xsizemode='pixel', ysizemode='pixel',
                    line=dict(color=stroke_color, width=float(stroke_lw)),
                )
            fig.add_shape(
                type='line', xref='x', yref='y',
                xanchor=float(tick_x[k]), yanchor=float(tick_y[k]),
                x0=0, y0=0, x1=dx_px, y1=dy_px,
                xsizemode='pixel', ysizemode='pixel',
                line=dict(color=tick_color, width=float(tick_lw),
                          dash=tick_dash),
            )

    # Endcaps (tick / arrow). Substitute for the regular endpoint tick.
    if endcap_style == 'tick' and len(positions) > 0:
        tick_endcap_indices: list[int] = []
        if endcaps in ('both', 'start'):
            tick_endcap_indices.append(0)
        if endcaps in ('both', 'end'):
            tick_endcap_indices.append(len(positions) - 1)
        cap_len = float(tick_length) * float(endcap_length_scale)
        for k in tick_endcap_indices:
            for sign in (+1.0, -1.0):
                dx_px = sign * perp_left[k, 0] * cap_len
                dy_px = sign * perp_left[k, 1] * cap_len
                if use_stroke:
                    fig.add_shape(
                        type='line', xref='x', yref='y',
                        xanchor=float(tick_x[k]), yanchor=float(tick_y[k]),
                        x0=0, y0=0, x1=dx_px, y1=dy_px,
                        xsizemode='pixel', ysizemode='pixel',
                        line=dict(color=stroke_color, width=float(stroke_lw)),
                    )
                fig.add_shape(
                    type='line', xref='x', yref='y',
                    xanchor=float(tick_x[k]), yanchor=float(tick_y[k]),
                    x0=0, y0=0, x1=dx_px, y1=dy_px,
                    xsizemode='pixel', ysizemode='pixel',
                    line=dict(color=endcap_color, width=float(endcap_lw)),
                )
    elif endcap_style == 'arrow' and len(positions) > 0:
        # Arrowhead pointing outward along the tangent. Sized in
        # pixels via a small triangle shape per endpoint.
        arrow_endcap_indices: list[tuple[int, float]] = []
        if endcaps in ('both', 'start'):
            arrow_endcap_indices.append((0, -1.0))    # outward = -tangent
        if endcaps in ('both', 'end'):
            arrow_endcap_indices.append((len(positions) - 1, +1.0))
        head_size = float(tick_length) * float(endcap_length_scale)
        for k, t_sign in arrow_endcap_indices:
            tx_px = t_sign * tangents[k, 0] * head_size
            ty_px = t_sign * tangents[k, 1] * head_size
            wing_x = perp_left[k, 0] * head_size * 0.6
            wing_y = perp_left[k, 1] * head_size * 0.6
            # Triangle: tip at (tx_px, ty_px), wings at ±perp * 0.6
            path = (f"M {tx_px},{ty_px} L {-wing_x},{-wing_y} "
                    f"L {wing_x},{wing_y} Z")
            if use_stroke:
                fig.add_shape(
                    type='path', path=path,
                    xref='x', yref='y',
                    xanchor=float(tick_x[k]), yanchor=float(tick_y[k]),
                    xsizemode='pixel', ysizemode='pixel',
                    fillcolor=stroke_color,
                    line=dict(color=stroke_color, width=float(stroke_lw)),
                )
            fig.add_shape(
                type='path', path=path,
                xref='x', yref='y',
                xanchor=float(tick_x[k]), yanchor=float(tick_y[k]),
                xsizemode='pixel', ysizemode='pixel',
                fillcolor=endcap_color,
                line=dict(color=endcap_color, width=float(endcap_lw)),
            )

    # Tick labels.
    from ..overlays.ruler import _format_numeric
    if labels and len(positions) > 0:
        for k, pos in enumerate(positions):
            # Where the label anchors: tick tip on the label side
            # plus label_offset pixels.
            radial = float(tick_length) + float(label_offset)
            dx_px = label_sign * perp_left[k, 0] * radial
            dy_px = label_sign * perp_left[k, 1] * radial
            tangent_angle = float(
                np.degrees(np.arctan2(tangents[k, 1], tangents[k, 0])))
            textangle = _ruler_textangle(tangent_angle, label_rotation)
            if label_fmt is not None:
                # mpl-compatible signature: (signed value in arcsec, unit).
                pos_asec = float(pos) / unit_scale * 3600.0
                text_val = label_fmt(pos_asec, unit_name)
            else:
                text_val = (f"{_format_numeric(float(pos), fmt=fmt)} "
                            f"{unit_name}")
            fig.add_annotation(
                x=float(tick_x[k]), y=float(tick_y[k]),
                xref='x', yref='y',
                text=text_val, showarrow=False,
                xshift=dx_px, yshift=dy_px,
                textangle=textangle,
                font=dict(color=color, size=int(label_fontsize)),
                xanchor='center', yanchor='middle',
            )

    # Title at the midpoint, on the opposite side from labels.
    if title is not None and len(fracs) > 0:
        mid_idx = len(fracs) // 2
        # Use the midpoint tangent for rotation, midpoint position
        # for anchoring.
        mid_x = float(tick_x[mid_idx])
        mid_y = float(tick_y[mid_idx])
        # When ticks are present, push title past tick tip + offset;
        # when no ticks, push past just label_offset.
        title_radial = (float(tick_length) + float(title_offset)
                        if tick_side != 'none' else float(title_offset))
        dx_px = title_sign * perp_left[mid_idx, 0] * title_radial
        dy_px = title_sign * perp_left[mid_idx, 1] * title_radial
        tangent_angle = float(np.degrees(
            np.arctan2(tangents[mid_idx, 1], tangents[mid_idx, 0])))
        textangle = _ruler_textangle(tangent_angle, title_rotation)
        fig.add_annotation(
            x=mid_x, y=mid_y, xref='x', yref='y',
            text=str(title), showarrow=False,
            xshift=dx_px, yshift=dy_px,
            textangle=textangle,
            font=dict(color=color, size=int(title_fontsize)),
            xanchor='center', yanchor='middle',
        )

    tick_shapes = list(fig.layout.shapes[n_shapes_before:])
    label_anns = list(fig.layout.annotations[n_anns_before:])
    return main_trace, tick_shapes, label_anns


# -- CompoundRegion (plotly bridge) -----------------------------------------

def make_compound_region(fig: Any) -> Any:
    """Construct a :class:`skyplothelper.CompoundRegion` driven by a
    plotly-figure projector.

    Convenience wrapper for ``CompoundRegion(SkyplothelperProjector.from_figure(fig))``
    — reads the projection / center metadata stamped on ``fig`` by
    :func:`make_figure` so the resulting region accumulates geometry
    in the figure's own canvas coords and renders directly through
    :func:`add_compound_region`.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Target figure (typically built by :func:`make_figure`).

    Returns
    -------
    region : skyplothelper.CompoundRegion
        An empty region ready for ``.add_circle(...).subtract_circle(...)``
        chaining. Render with :func:`add_compound_region`.
    """
    from .. import CompoundRegion
    from .projector import SkyplothelperProjector
    return CompoundRegion(SkyplothelperProjector.from_figure(fig))


def _shapely_to_svg_path(geom: Any) -> str:
    """Build an SVG path string from a shapely Polygon / MultiPolygon.

    Each polygon contributes an exterior subpath (M / L / ... / Z)
    followed by any interior (hole) subpaths. With plotly's default
    ``'evenodd'`` fill rule this renders holes correctly.
    """
    polys = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
    parts: list[str] = []
    for poly in polys:
        if not hasattr(poly, 'exterior'):
            continue
        ext_x, ext_y = poly.exterior.xy
        if len(ext_x) < 3:
            continue
        seg = [f"M {float(ext_x[0])},{float(ext_y[0])}"]
        for x, y in zip(ext_x[1:], ext_y[1:]):
            seg.append(f"L {float(x)},{float(y)}")
        seg.append("Z")
        parts.append(" ".join(seg))
        for interior in poly.interiors:
            int_x, int_y = interior.xy
            if len(int_x) < 3:
                continue
            seg = [f"M {float(int_x[0])},{float(int_y[0])}"]
            for x, y in zip(int_x[1:], int_y[1:]):
                seg.append(f"L {float(x)},{float(y)}")
            seg.append("Z")
            parts.append(" ".join(seg))
    return " ".join(parts) if parts else ""


def add_compound_region(fig: Any, region: Any, *, color: str = 'steelblue',
                        fillcolor: str | None = None, width: float = 1.0,
                        opacity: float = 0.4,
                        name: str | None = None, hover: bool | str = False,
                        hover_anchor: str = 'area',
                        legend_per_polygon: bool = False,
                        edge_buffer: float = 0.5) -> tuple[Any, Any]:
    """Render a :class:`skyplothelper.CompoundRegion` on a plotly figure.

    The region must have been built with a plotly-side projector (see
    :func:`make_compound_region`) so its accumulated geometry lives in
    the figure's canvas coordinates. The fill is emitted as a single
    ``fig.add_shape(type='path', ...)`` SVG path with interior rings
    included as additional subpaths under plotly's ``'evenodd'`` fill
    rule — so set-algebra holes (e.g. a band-minus-circle annulus)
    render correctly.

    The polygon's outline is then drawn as a separate set of
    ``go.Scatter`` polylines, with segments that lie on the
    projection silhouette **suppressed** — those are
    antimeridian-stitching artifacts added when projecting a
    sphere-wrapping shape into 2-D canvas coords, and would otherwise
    show up as visible "perpendicular arcs" along the wrap edge or
    pole-region stitches. Only the polygon's real boundary (band
    parallels, circle / polygon edges, interior holes) is rendered.

    When ``name`` is supplied, an invisible ``go.Scatter`` overlay
    carries the legend entry and the optional hover template — plotly
    shapes themselves don't support hover or legend natively. By
    default the overlay traces the region's outline so the tooltip
    fires anywhere over it (see ``hover_anchor``).

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    region : :class:`skyplothelper.CompoundRegion`
        Region accumulated against a plotly projector.
    color : str
        Outline color. Default ``'steelblue'``.
    fillcolor : str, optional
        Fill color (RGBA recommended for opacity). Default ``None``
        (no fill, outline only).
    width : float
        Outline line width. Default ``1.0``. Pass ``0`` to suppress
        the outline entirely (fill only).
    opacity : float
        Default ``0.4``.
    name : str, optional
        Legend entry / hover header. When set, an invisible Scatter
        overlay carries the hover; otherwise the shape is silent.
    hover : False / True / str
        Hover behavior on the legend overlay (only meaningful with
        ``name``). ``True`` shows the name; a string is used as the
        hovertemplate verbatim.
    hover_anchor : {'area', 'point'}
        Where the hover tooltip fires. ``'area'`` (default) makes the
        whole region hoverable, like every other filled helper —
        subtracted holes correctly stay silent. ``'point'`` restricts
        hover to a single invisible marker at the region's
        representative point, which suits slivers and very small
        regions where an area target sits under the cursor itself.

        One asymmetry, inherited from plotly: hover over a *fill* is
        rendered from the trace's plain text, not its hovertemplate.
        So under ``'area'`` a custom ``hover`` string is displayed
        verbatim and ``%{...}`` placeholders do not substitute; use
        ``'point'`` if you need them.
    legend_per_polygon : bool
        When ``True`` and the region resolves to multiple disconnected
        pieces, emit one legend overlay per piece (named ``f"{name} 1"``,
        ``f"{name} 2"``, ...) instead of a single overlay for the whole
        region. Requires ``name``; ignored for single-piece regions.
        Default ``False``.
    edge_buffer : float
        Pixel tolerance for the silhouette-edge suppression. Boundary
        segments within this distance of the projection silhouette
        are treated as stitching artifacts and omitted from the
        outline. Default ``0.5`` — large enough to absorb sub-pixel
        offsets from the silhouette sampling without eating into
        real boundaries.

    Returns
    -------
    shape : plotly Shape or None
        The fill path shape appended to ``fig.layout.shapes``, or
        ``None`` if the region was empty.
    overlay : plotly.graph_objects.Scatter, list of Scatter, or None
        The invisible legend overlay, or ``None`` when ``name`` is not
        supplied. A ``list`` of overlays when ``legend_per_polygon=True``
        splits a multi-piece region. Under ``hover_anchor='area'`` a
        further invisible trace per overlay carries the hover itself.
    """
    go = _import_plotly()
    if hover_anchor not in ('area', 'point'):
        raise ValueError(
            f"hover_anchor must be 'area' or 'point', got {hover_anchor!r}")
    geom = getattr(region, '_geom', None)
    if geom is None or geom.is_empty:
        return None, None

    path = _shapely_to_svg_path(geom)
    if not path:
        return None, None

    n_shapes_before = (len(fig.layout.shapes)
                        if fig.layout.shapes else 0)
    # Fill only (no line) on the shape itself — the outline goes
    # through a separate filtering pass below so silhouette stitches
    # don't appear as visible "perpendicular arc" artifacts.
    fig.add_shape(
        type='path', path=path,
        xref='x', yref='y',
        line=dict(width=0),
        fillcolor=fillcolor,
        opacity=float(opacity),
        fillrule='evenodd',
    )
    shape = fig.layout.shapes[n_shapes_before]

    # Boundary rendering: walk the polygon's boundary, drop the parts
    # lying on the frame silhouette (stitching artifacts), emit each
    # remaining piece as a Scatter polyline.
    if float(width) > 0:
        frame_poly = getattr(getattr(region, 'projector', None),
                              'frame_polygon', None)
        for poly in (geom.geoms if hasattr(geom, 'geoms') else [geom]):
            if not hasattr(poly, 'exterior'):
                continue
            rings = [poly.exterior] + list(poly.interiors)
            for ring in rings:
                _emit_filtered_boundary(
                    fig, go, ring, frame_poly,
                    color=color, width=float(width),
                    opacity=opacity, edge_buffer=float(edge_buffer))

    overlay: Any = None
    if name is not None:
        # Plotly shapes don't show up in the legend or carry hover, so
        # an invisible Scatter supplies the legend entry + optional
        # hover (area-shaped by default, see _add_region_legend_overlay).
        pieces = [p for p in (geom.geoms if hasattr(geom, 'geoms')
                              else [geom]) if hasattr(p, 'exterior')]
        if legend_per_polygon and len(pieces) > 1:
            overlay = [
                _add_region_legend_overlay(
                    fig, go, piece, f"{name} {i + 1}", color, hover,
                    hover_anchor)
                for i, piece in enumerate(pieces)]
        else:
            overlay = _add_region_legend_overlay(
                fig, go, geom, name, color, hover, hover_anchor)

    return shape, overlay


def _catalog_skycoord(fig: Any, catalog: Any) -> Any:
    """Normalize a slider catalog to a SkyCoord in the figure's display frame.

    Accepts a SkyCoord (used verbatim — its own frame is authoritative) or an
    ``(lon, lat)`` degree pair, which is read in the figure's display frame
    (the same convention as :func:`add_scatter`). Returning a SkyCoord lets
    both the projection (canvas markers) and the containment test agree on the
    frame even when the figure is not ICRS.
    """
    from astropy.coordinates import SkyCoord
    if isinstance(catalog, SkyCoord):
        return catalog
    import astropy.units as u
    lon, lat = catalog
    return SkyCoord(np.asarray(lon, dtype=float) * u.deg,
                    np.asarray(lat, dtype=float) * u.deg,
                    frame=_display_frame(fig))


def compound_region_states(
    fig: Any, region_factory: Callable[..., Any],
    param_values: Sequence[dict[str, Any]], catalog: Any = None, *,
    edge_buffer: float = 0.5,
) -> list[dict[str, Any]]:
    """Precompute per-parameter render data for a region that grows under a
    slider — the static, kernel-free half of the region-explorer pattern.

    For each parameter dict in *param_values*, build
    ``region_factory(**params)`` (a :class:`~skyplothelper.CompoundRegion`
    accumulated against *fig*'s projector, e.g. via
    :func:`make_compound_region`) and return everything a slider step needs:
    the set-algebra fill path, the silhouette-filtered outline (both as an SVG
    path and as ``None``-separated x/y), and — if a *catalog* is supplied —
    the per-source containment plus inside / outside counts. Pure compute: no
    traces or shapes are added, so the same states drive either a static
    slider (:func:`add_region_slider`) or a live Dash app
    (:func:`skyplothelper.plotly.dash_region.region_explorer_app`).

    A union that splits into disjoint lobes (or carries set-algebra holes)
    is already a *single* fill path and a single outline path here — the
    ``(Multi)Polygon`` is flattened into one SVG string with the pieces as
    subpaths — so a slider step swaps one ``shapes[i].path``, never a varying
    number of shapes.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure the region is projected against (built by
        :func:`make_figure`).
    region_factory : callable
        ``region_factory(**params) -> CompoundRegion``. Written once by the
        caller; consumed identically here and by the Dash app.
    param_values : sequence of dict
        One parameter dict per slider step, each spread into *region_factory*.
    catalog : SkyCoord or (lon, lat), optional
        Sources to reclassify inside / outside per step. A SkyCoord is used in
        its own frame; a bare ``(lon, lat)`` pair is read in the figure's
        display frame. When ``None``, containment fields are ``None`` / 0.
    edge_buffer : float
        Silhouette-suppression tolerance for the outline, in canvas pixels
        (see :func:`add_compound_region`). Default ``0.5``.

    Returns
    -------
    list of dict
        One dict per step with keys ``params``, ``fill_path``,
        ``outline_path``, ``outline_x``, ``outline_y``, ``contains`` (bool
        ndarray or ``None``), ``contains_int`` (0/1 list or ``None``),
        ``n_inside``, ``n_outside``.
    """
    if not callable(region_factory):
        raise TypeError(
            "region_factory must be callable: "
            "region_factory(**params) -> CompoundRegion")
    skycoord = None if catalog is None else _catalog_skycoord(fig, catalog)

    states: list[dict[str, Any]] = []
    for params in param_values:
        region = region_factory(**params)
        geom = getattr(region, '_geom', None)
        frame_poly = getattr(getattr(region, 'projector', None),
                             'frame_polygon', None)
        if geom is None or geom.is_empty:
            fill_path, outline_path = '', ''
            ox: list[Any] = []
            oy: list[Any] = []
        else:
            fill_path = _shapely_to_svg_path(geom)
            outline_path = _region_outline_path(geom, frame_poly, edge_buffer)
            ox, oy = _region_outline_xy(geom, frame_poly, edge_buffer)

        if skycoord is not None:
            contains = np.asarray(region.contains_points(skycoord), dtype=bool)
            contains_int: list[int] | None = contains.astype(int).tolist()
            n_inside = int(contains.sum())
            n_outside = int(contains.size) - n_inside
        else:
            contains = None
            contains_int = None
            n_inside = n_outside = 0

        states.append(dict(
            params=dict(params), fill_path=fill_path,
            outline_path=outline_path, outline_x=ox, outline_y=oy,
            contains=contains, contains_int=contains_int,
            n_inside=n_inside, n_outside=n_outside))
    return states


def _region_step_label(state: dict[str, Any], index: int,
                       label_format: Any) -> str:
    """Slider-step label from a state dict. ``None`` -> ``k=v`` of the params;
    a callable is passed the state; a format string sees the params plus
    ``i`` / ``n_inside`` / ``n_outside``."""
    if label_format is None:
        parts = []
        for k, v in state['params'].items():
            parts.append(f"{k}={v:g}" if isinstance(v, (int, float))
                         and not isinstance(v, bool) else f"{k}={v}")
        return ", ".join(parts)
    if callable(label_format):
        return str(label_format(state))
    return label_format.format(i=index, n_inside=state['n_inside'],
                               n_outside=state['n_outside'], **state['params'])


def _project_catalog(fig: Any, catalog: Any) -> tuple[Any, Any]:
    """Project a slider catalog to this figure's canvas coords, the same way
    :func:`add_scatter` does (frame-resolved via the figure's display frame)."""
    skycoord = _catalog_skycoord(fig, catalog)
    lon_fig, lat_fig = _resolve_lonlat(fig, skycoord, None, 'region catalog')
    projection, center, lat_center, direction = _meta_defaults(
        fig, None, None, None, None)
    return _project(np.asarray(lon_fig, dtype=float),
                    np.asarray(lat_fig, dtype=float),
                    projection=projection, center=center,
                    lat_center=lat_center, direction=direction)


def _draw_region_and_catalog(
    fig: Any, state: dict[str, Any], cx: Any = None, cy: Any = None, *,
    color: str, fillcolor: str | None, width: float, opacity: float,
    inside_color: str, outside_color: str, marker_size: float,
    marker_symbol: Any, name: str | None,
) -> tuple[int, int, int | None]:
    """Draw a region's fill + outline shapes and (optionally) the catalog
    markers for one state, returning ``(fill_idx, outline_idx, cat_idx)``.

    Shared by the static :func:`add_region_slider` and the live
    :mod:`~skyplothelper.plotly.dash_region` app so both render a step
    identically — the fill and outline are two ``layout.shape`` paths (a
    fixed two-shape footprint even when the region splits into disjoint
    lobes) and the catalog is one marker trace colored 0/1 through a two-stop
    colorscale.
    """
    go = _import_plotly()
    n_shapes = len(fig.layout.shapes) if fig.layout.shapes else 0
    fill_idx, outline_idx = n_shapes, n_shapes + 1
    fig.add_shape(type='path', path=state['fill_path'], xref='x', yref='y',
                  line=dict(width=0), fillcolor=fillcolor,
                  opacity=float(opacity), fillrule='evenodd')
    fig.add_shape(type='path', path=state['outline_path'], xref='x', yref='y',
                  line=dict(color=color, width=float(width)),
                  fillcolor='rgba(0,0,0,0)', opacity=float(opacity))
    cat_idx: int | None = None
    if cx is not None:
        marker: dict[str, Any] = dict(
            color=state['contains_int'], cmin=0, cmax=1,
            size=float(marker_size),
            colorscale=[[0.0, _to_plotly_color(outside_color)],
                        [1.0, _to_plotly_color(inside_color)]])
        if marker_symbol is not None:
            marker['symbol'] = marker_symbol
        cat_idx = len(fig.data)
        fig.add_trace(go.Scatter(
            x=cx.ravel(), y=cy.ravel(), mode='markers', marker=marker,
            name=name, showlegend=name is not None, hoverinfo='skip'))
    return fill_idx, outline_idx, cat_idx


def add_region_slider(
    fig: Any, region_factory: Callable[..., Any],
    param_values: Sequence[dict[str, Any]], catalog: Any = None, *,
    color: str = 'steelblue', fillcolor: str | None = None,
    width: float = 1.0, opacity: float = 0.4, edge_buffer: float = 0.5,
    inside_color: str = 'crimson', outside_color: str = '0.5',
    marker_size: float = 4.0, marker_symbol: Any = None,
    name: str | None = None, active: int = 0,
    slider_label: str | None = None, label_format: Any = None,
) -> list[dict[str, Any]]:
    """Attach a parameter slider that grows a :class:`~skyplothelper.CompoundRegion`
    and reclassifies a source catalog inside / outside it — all precomputed, so
    the figure stays fully static (works in nbviewer / the Sphinx docs, no live
    kernel).

    One-call wrapper over :func:`compound_region_states`: it computes the
    states, draws the region (a fill ``shape`` + a stroked outline ``shape``)
    and the catalog markers, then wires a plotly slider whose every step swaps
    the two shape paths and restyles the markers. The catalog's coordinates are
    sent once; each step ships only the two paths and a compact 0/1 membership
    array (rendered through a two-stop colorscale), not a fresh hex color per
    source — roughly half the payload on a several-thousand-source catalog.

    The genuinely-live counterpart (a continuous slider that re-runs the set
    algebra in Python on every drag) is
    :func:`skyplothelper.plotly.dash_region.region_explorer_app`, which
    consumes the same *region_factory*; use it when you have a running kernel
    (local / Binder / Colab).

    Parameters
    ----------
    fig, region_factory, param_values, catalog, edge_buffer :
        As in :func:`compound_region_states`.
    color, fillcolor, width, opacity :
        Region outline color / fill color / outline width / fill opacity, as
        in :func:`add_compound_region`.
    inside_color, outside_color : str
        Marker colors for sources inside / outside the region. Default
        crimson / mid-grey.
    marker_size : float
        Source marker size. Default ``4``.
    marker_symbol : array-like or str, optional
        Per-source (or single) plotly marker symbol — e.g. one symbol for
        *defining* and another for *standard* sources. Static across steps.
    name : str, optional
        Legend name for the catalog trace.
    active : int
        Initially-selected step. Default ``0``.
    slider_label : str, optional
        Prefix shown before the active step's label on the slider.
    label_format : str or callable, optional
        Step-label control (see :func:`_region_step_label`). Default labels
        each step by its ``k=v`` parameters.

    Returns
    -------
    list of dict
        The computed states (as :func:`compound_region_states` returns), for
        inspection; the slider, shapes and markers are added to *fig* in place.
    """
    if not param_values:
        raise ValueError("param_values must be a non-empty sequence")
    if not 0 <= active < len(param_values):
        raise ValueError(
            f"active={active} out of range for {len(param_values)} steps")

    states = compound_region_states(fig, region_factory, param_values,
                                    catalog, edge_buffer=edge_buffer)
    init = states[active]

    cx = cy = None
    if catalog is not None:
        cx, cy = _project_catalog(fig, catalog)
    # Fill + outline as two swappable layout-shape paths, catalog as one
    # 0/1-colored marker trace (see _draw_region_and_catalog). Sharing this
    # with the Dash app keeps a static step and a live drag pixel-identical.
    fill_idx, outline_idx, cat_idx = _draw_region_and_catalog(
        fig, init, cx, cy, color=color, fillcolor=fillcolor, width=width,
        opacity=opacity, inside_color=inside_color, outside_color=outside_color,
        marker_size=marker_size, marker_symbol=marker_symbol, name=name)

    steps = []
    for i, st in enumerate(states):
        relayout = {f'shapes[{fill_idx}].path': st['fill_path'],
                    f'shapes[{outline_idx}].path': st['outline_path']}
        if cat_idx is not None:
            args: list[Any] = [{'marker.color': [st['contains_int']]},
                               relayout, [cat_idx]]
        else:
            args = [{}, relayout]
        steps.append(dict(method='update', args=args,
                          label=_region_step_label(st, i, label_format)))

    slider: dict[str, Any] = dict(active=active, steps=steps)
    if slider_label is not None:
        slider['currentvalue'] = dict(prefix=f"{slider_label}: ")
    fig.update_layout(sliders=[slider])
    return states


def _region_rings_xy(geom: Any) -> tuple[list[Any], list[Any]]:
    """Flatten every ring of a (Multi)Polygon into ``None``-separated
    ``(xs, ys)`` for a ``fill='toself'`` Scatter.

    Interior rings are included on purpose. Plotly hover-tests a filled
    trace by counting how many of its subpath polygons contain the
    cursor and toggling on each hit — an even-odd rule. A point inside a
    subtracted hole is contained by both the exterior ring and the hole
    ring, toggles twice, and so correctly reports *no* hover. (The rings
    are invisible here — transparent fill, zero-width line — so the fact
    that plotly would *paint* a hole ring as a positive area never
    shows.)
    """
    xs: list[Any] = []
    ys: list[Any] = []
    for piece in (geom.geoms if hasattr(geom, 'geoms') else [geom]):
        if not hasattr(piece, 'exterior'):
            continue
        for ring in [piece.exterior, *piece.interiors]:
            rx, ry = ring.xy
            if not len(rx):
                continue
            if xs:
                xs.append(None)
                ys.append(None)
            xs.extend(float(v) for v in rx)
            ys.extend(float(v) for v in ry)
    return xs, ys


def _add_region_legend_overlay(
    fig: Any, go: Any, geom: Any, name: str, color: str,
    hover: bool | str, hover_anchor: str = 'area',
) -> Any:
    """Add the invisible ``go.Scatter`` legend / hover overlay for a
    region (or a single disconnected piece of it).

    The region's own fill is a ``layout.shape`` SVG path, and plotly
    shapes carry neither a legend entry nor hover — hence this overlay.
    It is anchored at the geometry's representative point (a point
    guaranteed to lie inside it, unlike the centroid / bounds midpoint,
    which for wavy bands can land far outside the polygon) and is what
    this function returns.

    Under ``hover_anchor='area'`` a *second*, equally invisible trace is
    appended: the region's rings under ``fill='toself'`` +
    ``hoveron='fills'``, so the tooltip fires anywhere over the region
    the way it does for every other filled helper. It has to be its own
    trace because plotly renders a fill-hover label from ``trace.text``
    and falls back to ``trace.name`` — a named trace would print its
    name a second time in the tooltip's side box. So the marker keeps
    the name (and the legend entry) while the fill trace, named ``''``,
    keeps the hover.

    With hover disabled there is nothing to hover, so only the marker is
    emitted; it exists purely to carry the legend entry.
    """
    hover_on = hover is not False and hover is not None
    area = hover_on and hover_anchor == 'area'

    rep_point = geom.representative_point()
    overlay_kw: dict[str, Any] = dict(
        x=[float(rep_point.x)],
        y=[float(rep_point.y)],
        mode='markers',
        marker=dict(size=1, color=color, opacity=0),
        name=name, showlegend=True,
    )
    if not hover_on or area:
        # Either nothing to hover, or the hover lives on the fill trace.
        overlay_kw['hoverinfo'] = 'skip'
    elif hover is True:
        overlay_kw['hovertemplate'] = f"<b>{name}</b><extra></extra>"
    else:
        tpl = str(hover)
        if '<extra>' not in tpl:
            tpl += '<extra></extra>'
        overlay_kw['hovertemplate'] = tpl
    overlay = go.Scatter(**overlay_kw)
    fig.add_trace(overlay)

    if area:
        xs, ys = _region_rings_xy(geom)
        fill_text, fill_template, _ = _resolve_hover_fill(hover, name)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines', fill='toself',
            # Transparent: the visible fill is the layout shape. Plotly
            # hover-tests fills against the trace's polygon coords, not
            # the rendered pixels, so an invisible fill still hovers.
            # The line color is never drawn (width 0) but plotly does
            # read it for the hover label's background, since the
            # transparent fillcolor gives it nothing to tint with.
            fillcolor='rgba(0,0,0,0)',
            line=dict(color=color, width=0),
            hoveron='fills',
            text=fill_text, hovertemplate=fill_template,
            # Empty trace name suppresses the ``"trace N"`` side box
            # plotly would otherwise render next to the fill tooltip.
            name='', showlegend=False,
        ))
    return overlay


def _filtered_ring_pieces(
    ring: Any, frame_poly: Any, edge_buffer: float,
) -> list[tuple[list[float], list[float]]]:
    """Split a shapely LinearRing into visible polyline pieces.

    Drops the segments that lie within ``edge_buffer`` of ``frame_poly``'s
    exterior — those are antimeridian stitching artifacts that would otherwise
    show as visible "arcs" along the canvas wrap edge. Returns a list of
    ``(xs, ys)`` pieces. Pure (no plotly), so it is the single home shared by
    the static outline (:func:`_emit_filtered_boundary`) and the
    slider-friendly outline (:func:`_region_outline_xy` /
    :func:`_region_outline_path`) — the three cannot drift.
    """
    coords = list(ring.coords)
    if len(coords) < 2:
        return []
    if frame_poly is None:
        # No silhouette to filter against — the whole ring is one piece.
        return [([pt[0] for pt in coords], [pt[1] for pt in coords])]

    from shapely.geometry import Point
    frame_ext = frame_poly.exterior
    # Per-vertex flag: is this vertex within ``edge_buffer`` of the
    # silhouette? Edges between two near-silhouette vertices are
    # dropped; edges with at least one interior vertex are kept.
    near_frame = [frame_ext.distance(Point(*pt)) < edge_buffer
                   for pt in coords]
    pieces: list[tuple[list[float], list[float]]] = []
    cur_xs: list[float] = []
    cur_ys: list[float] = []
    for i in range(len(coords) - 1):
        if near_frame[i] and near_frame[i + 1]:
            # Stitching edge — flush any active piece, then skip.
            if len(cur_xs) >= 2:
                pieces.append((cur_xs, cur_ys))
            cur_xs, cur_ys = [], []
            continue
        if not cur_xs:
            cur_xs.append(coords[i][0])
            cur_ys.append(coords[i][1])
        cur_xs.append(coords[i + 1][0])
        cur_ys.append(coords[i + 1][1])
    if len(cur_xs) >= 2:
        pieces.append((cur_xs, cur_ys))
    return pieces


def _emit_filtered_boundary(
    fig: Any, go: Any, ring: Any, frame_poly: Any, *,
    color: str, width: float, opacity: float, edge_buffer: float,
) -> None:
    """Render ``ring`` (a shapely LinearRing) as ``go.Scatter`` polylines,
    with the projection-silhouette stitching segments suppressed (see
    :func:`_filtered_ring_pieces`)."""
    for xs, ys in _filtered_ring_pieces(ring, frame_poly, edge_buffer):
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color=color, width=width),
            opacity=opacity, hoverinfo='skip', showlegend=False,
        ))


def _region_outline_xy(
    geom: Any, frame_poly: Any, edge_buffer: float = 0.5,
) -> tuple[list[Any], list[Any]]:
    """Flatten a region's silhouette-filtered boundary into one
    ``None``-separated ``(xs, ys)`` pair — the form a single ``go.Scatter``
    line trace (or a live Dash repaint) restyles per slider step. Same filter
    as :func:`add_compound_region`'s multi-trace outline, so the static and
    slider outlines match."""
    xs: list[Any] = []
    ys: list[Any] = []
    for poly in (geom.geoms if hasattr(geom, 'geoms') else [geom]):
        if not hasattr(poly, 'exterior'):
            continue
        for ring in [poly.exterior, *poly.interiors]:
            for pxs, pys in _filtered_ring_pieces(ring, frame_poly, edge_buffer):
                if xs:
                    xs.append(None)
                    ys.append(None)
                xs.extend(pxs)
                ys.extend(pys)
    return xs, ys


def _region_outline_path(
    geom: Any, frame_poly: Any, edge_buffer: float = 0.5,
) -> str:
    """Silhouette-filtered outline as an SVG path of *open* polylines
    (``M ... L ...`` per piece, no ``Z``). Each disjoint boundary piece is
    a separate subpath, so a single ``layout.shape`` of type ``'path'`` with
    ``fillcolor='rgba(0,0,0,0)'`` strokes the whole outline — updated per
    slider step by swapping one ``shapes[i].path`` (no per-step trace churn,
    even when the union splits into disjoint lobes)."""
    parts: list[str] = []
    for poly in (geom.geoms if hasattr(geom, 'geoms') else [geom]):
        if not hasattr(poly, 'exterior'):
            continue
        for ring in [poly.exterior, *poly.interiors]:
            for pxs, pys in _filtered_ring_pieces(ring, frame_poly, edge_buffer):
                if len(pxs) < 2:
                    continue
                seg = [f"M {float(pxs[0])},{float(pys[0])}"]
                seg.extend(f"L {float(x)},{float(y)}"
                           for x, y in zip(pxs[1:], pys[1:]))
                parts.append(" ".join(seg))
    return " ".join(parts)


# ===========================================================================
# Multi-dimensional legend (plotly) — render skyplothelper legend blocks as
# native plotly legend entries via invisible named traces.
# ===========================================================================

# matplotlib marker -> plotly symbol. Unlisted markers fall back to 'circle'.
_MPL_TO_PLOTLY_SYMBOL: dict[str, str] = {
    'o': 'circle', '.': 'circle', 's': 'square', 'D': 'diamond',
    'd': 'diamond', '^': 'triangle-up', 'v': 'triangle-down',
    '<': 'triangle-left', '>': 'triangle-right', 'p': 'pentagon',
    'h': 'hexagon', 'H': 'hexagon2', '*': 'star', '+': 'cross-thin',
    'x': 'x-thin', 'P': 'cross', 'X': 'x',
}

# matplotlib linestyle -> plotly dash.
_MPL_TO_PLOTLY_DASH: dict[str, str] = {
    '-': 'solid', '--': 'dash', '-.': 'dashdot', ':': 'dot',
    'solid': 'solid', 'dashed': 'dash', 'dashdot': 'dashdot', 'dotted': 'dot',
}


def _to_plotly_color(c: Any) -> str:
    """Convert a matplotlib color spec ('C0', '0.4', named, hex, rgba) to a
    plotly-usable string. ``None`` / ``'none'`` become fully transparent."""
    if c is None or c == 'none':
        return 'rgba(0,0,0,0)'
    from matplotlib.colors import to_hex
    try:
        # 6-digit hex only — plotly rejects 8-digit; any alpha rides on
        # marker.opacity instead.
        return to_hex(c, keep_alpha=False)
    except (ValueError, TypeError):
        return str(c)


def _entry_marker(style: dict[str, Any], default_size: float) -> dict[str, Any]:
    """Translate a legend entry's style dict into a plotly ``marker`` dict."""
    mk = style.get('marker', 'o')
    symbol = _MPL_TO_PLOTLY_SYMBOL.get(mk, 'circle')
    face = style.get('facecolor', style.get('color'))
    edge = style.get('edgecolor', 'none')
    open_marker = face in (None, 'none')
    if open_marker and not symbol.endswith('-open'):
        # A plotly '-open' symbol is stroked in marker.color, so an open swatch
        # needs a visible ink there: prefer the edge color, then the entry's
        # own color, then the (transparent) face — never leave it transparent
        # when a color was actually given.
        symbol = symbol + '-open'
        ink = (edge if edge not in ('none', None)
               else style.get('color', face))
        color = _to_plotly_color(ink)
    else:
        color = _to_plotly_color(face)
    marker: dict[str, Any] = dict(
        symbol=symbol, color=color,
        size=float(style.get('markersize', default_size)))
    if style.get('alpha') is not None:
        marker['opacity'] = float(style['alpha'])
    if style.get('angle') is not None:
        marker['angle'] = float(style['angle'])
    if edge not in ('none', None) and not open_marker:
        marker['line'] = dict(color=_to_plotly_color(edge),
                              width=float(style.get('linewidth', 1.0)))
    return marker


def _entry_line(style: dict[str, Any]) -> dict[str, Any]:
    """Translate a line entry's style dict into a plotly ``line`` dict."""
    return dict(
        color=_to_plotly_color(style.get('color', style.get('facecolor'))),
        dash=_MPL_TO_PLOTLY_DASH.get(style.get('linestyle', '-'), 'solid'),
        width=float(style.get('linewidth', 2.0)))


def add_legend(fig: Any, blocks: Any, *, default_size: float = 10.0,
               warn_unsupported: bool = True) -> list[Any]:
    """Render skyplothelper legend blocks as native plotly legend entries.

    Each block (``ColorBlock``, ``ShapeBlock``, ``SizeBlock``, …, built the
    same way as for the matplotlib :class:`~skyplothelper.MultiLegend`) becomes
    a legend group — an invisible named ``go.Scatter`` trace per entry, grouped
    with ``legendgroup`` + ``legendgrouptitle``. This gives plotly a
    multi-channel key, including **graduated size / alpha** legends that plotly
    has no native equivalent for.

    Channel coverage mirrors what plotly markers can express: color, shape,
    size, edge, fill (open/solid), alpha, orientation, and line (dash/width);
    a ``ColorbarBlock`` emits a real plotly colorbar. Hatch, translucent
    region patches, free text, and custom-artist blocks are matplotlib-only
    refinements — skipped (with a warning unless ``warn_unsupported=False``).

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    blocks : LegendBlock or sequence of LegendBlock
        The channel blocks to render (e.g. ``[sph.ColorBlock(...),
        sph.SizeBlock(...)]``).
    default_size : float
        Marker size (px) for entries whose block doesn't set one.
    warn_unsupported : bool
        Warn when a block has no plotly equivalent (default True).

    Returns
    -------
    list of plotly traces
        The invisible legend traces added (also in ``fig.data``).
    """
    go = _import_plotly()
    from ..legend import ColorbarBlock, LegendBlock

    if isinstance(blocks, LegendBlock):
        blocks = [blocks]

    traces: list[Any] = []
    for bi, block in enumerate(blocks):
        title = block.title

        if isinstance(block, ColorbarBlock):
            import matplotlib.pyplot as plt
            from matplotlib.colors import to_hex
            cmap = (plt.get_cmap(block.cmap) if isinstance(block.cmap, str)
                    else block.cmap)
            colorscale = [[i / 9.0, to_hex(cmap(i / 9.0))] for i in range(10)]
            traces.append(_add_scale_colorbar(
                fig, go, colorscale=colorscale, cmin=block.vmin,
                cmax=block.vmax, title=title))
            continue

        kind = block.swatch_kind
        has_hatch = any('hatch' in s for _lbl, s in block.entries)
        if kind in ('text', 'custom', 'glyph') or (kind == 'region' and has_hatch):
            if warn_unsupported:
                warnings.warn(
                    f"add_legend: block {title!r} ({kind}) has no plotly "
                    "equivalent and was skipped", stacklevel=2)
            continue

        grp = f"sphleg{bi}"
        first = True
        for label, entry_style in block.entries:
            # Merge the block's shared base_style under the per-entry style,
            # the same way the matplotlib side does (block._resolved_style) —
            # otherwise a block that carries its swatch color / facecolor in
            # base_style (e.g. ShapeBlock's shared color) loses it here and
            # renders an invisible swatch.
            style = block._resolved_style(entry_style)
            trace_kw: dict[str, Any] = dict(
                x=[None], y=[None], name=str(label), legendgroup=grp,
                showlegend=True, hoverinfo='skip')
            if first and title is not None:
                trace_kw['legendgrouptitle'] = dict(text=title)
            first = False
            if kind == 'line':
                trace_kw['mode'] = 'lines'
                trace_kw['line'] = _entry_line(style)
                if style.get('marker'):
                    trace_kw['mode'] = 'lines+markers'
                    trace_kw['marker'] = _entry_marker(style, default_size)
            else:
                # marker / patch / region -> a marker swatch (chips are squares).
                trace_kw['mode'] = 'markers'
                mstyle = ({**style, 'marker': 's'}
                          if kind in ('patch', 'region') else style)
                trace_kw['marker'] = _entry_marker(mstyle, default_size)
            trace = go.Scatter(**trace_kw)
            fig.add_trace(trace)
            traces.append(trace)

    fig.update_layout(showlegend=True)
    return traces
