"""Latitude / longitude / great-circle / cross-frame band overlays.

``add_latitude_band``, ``add_longitude_band``, ``add_great_circle_band``,
and ``add_frame_band`` are thin wrappers around the
:class:`~skyplothelper.geometry.compound.CompoundRegion` machinery
— a single source of truth for spherical-region projection,
antimeridian clipping, frame intersection, and edge handling.

The one exception is ``add_frame_band(backend='contour')``, which
uses a fundamentally different rasterization-then-contour
algorithm (membership-test per pixel) that doesn't fit the
shapely-region model. It stays as its own implementation here.
"""

from __future__ import annotations

import warnings  # noqa: F401
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
import numpy.typing as npt

try:
    from shapely.geometry import LineString, Polygon  # noqa: F401
    from shapely.ops import unary_union  # noqa: F401
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

from .._stroke import _stroke_path_effects
from ._antimeridian import (  # noqa: F401  (used by contour fallback)
    _antimeridian_clip,
    _stitch_and_project,
)
from ._api import _resolve_backend
from ._frame_geom import (
    _fix_hairline_kwargs,  # noqa: F401  (kept for completeness; renderer uses it)
    _get_projection_center,  # noqa: F401  (used by contour fallback)
    _paths_to_geom,  # noqa: F401  (used by contour fallback)
    _shapely_to_paths,  # noqa: F401  (used by contour fallback)
)
from ._parsing import (  # noqa: F401  (_parse_coord kept for consumers)
    _coords_to_frame_deg,
    _parse_angle,
    _parse_coord,
)
from ._projection import (  # noqa: F401  (kept for backcompat consumers)
    _project_shape,
    _render_complement,
    _simple_project,
)
from .compound import CompoundRegion
from .shapes import geodesic_circle  # noqa: F401  (used by contour fallback if relevant)

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord


def _apply_stroke_kwargs(kwargs: dict[str, Any], stroke_color: Any,
                         stroke_lw: float) -> None:
    """Inject a legibility stroke (``path_effects``) into a band helper's
    render kwargs, in place.

    No-op (so the rendered output is unchanged) when ``stroke_color`` is
    ``None`` or the caller already supplied an explicit ``path_effects=``.
    """
    if stroke_color is not None and 'path_effects' not in kwargs:
        pe = _stroke_path_effects(stroke_color, stroke_lw)
        if pe is not None:
            kwargs['path_effects'] = pe


# ===== Latitude band =====

def add_latitude_band(ax: Any, lat_min: Any, lat_max: Any, lon_min: Any = None,
                      lon_max: Any = None, resolution: int = 360,
                      complement: bool = False, clip: str = 'auto',
                      stroke_color: Any = None, stroke_lw: float = 3.0,
                      **kwargs: Any) -> Any:
    """
    Add a latitude band.

    Parameters
    ----------
    ax : WCSAxes
    lat_min, lat_max : float or Quantity
        Latitude limits in degrees (ICRS).
    lon_min, lon_max : float, Quantity, or None
        Longitude limits (default: full sky).
    resolution : int
        Longitude sampling density.
    complement : bool
        If True, fill everything EXCEPT the band.
    clip : str
        Projection-seam handling pipeline. ``'auto'`` (default)
        resolves to ``'d3'`` for closed-region patches. See
        :func:`~skyplothelper.add_spherical_polygon` for the full
        description.
    stroke_color : color, optional
        Legibility stroke around the band outline (via ``path_effects``).
        Default ``None`` — no stroke.
    stroke_lw : float, optional
        Total stroke width in points. Default ``3.0``; applies only when
        ``stroke_color`` is set.
    **kwargs
        Passed to PathPatch.
    """
    lat_lo = _parse_angle(lat_min)
    lat_hi = _parse_angle(lat_max)
    if lat_lo is None or lat_hi is None or lat_lo >= lat_hi:
        raise ValueError("lat_min must be less than lat_max")
    region = CompoundRegion(ax).add_latitude_band(
        lat_lo, lat_hi, resolution=resolution, clip=clip)
    if lon_min is not None or lon_max is not None:
        _EPS = 1e-4
        lo = _parse_angle(lon_min) if lon_min is not None else (-180 + _EPS)
        hi = _parse_angle(lon_max) if lon_max is not None else ( 180 - _EPS)
        region.intersect_longitude_band(lo, hi, resolution=resolution,
                                          clip=clip)
    if complement:
        region.complement()
    _apply_stroke_kwargs(kwargs, stroke_color, stroke_lw)
    return region.render(**kwargs)


# ============================================================
# Vertex generators for standard shapes
# ============================================================


# ===== Longitude band =====

def add_longitude_band(ax: Any, lon_min: Any, lon_max: Any, lat_min: Any = None,
                       lat_max: Any = None, resolution: int = 360,
                       complement: bool = False, clip: str = 'auto',
                       stroke_color: Any = None, stroke_lw: float = 3.0,
                       **kwargs: Any) -> Any:
    """
    Add a longitude (RA) band to WCSAxes.

    Parameters
    ----------
    ax : WCSAxes
    lon_min, lon_max : float or Quantity
        Longitude limits in degrees (ICRS).  The band covers the
        shorter arc from lon_min to lon_max going eastward.
    lat_min, lat_max : float, Quantity, or None
        Latitude limits (default: full range -90 to +90).
    resolution : int
        Sampling density along latitude edges.
    complement : bool
        If True, fill everything except the band.
    clip : str
        Projection-seam handling pipeline. ``'auto'`` (default)
        resolves to ``'d3'`` for closed-region patches.
    stroke_color : color, optional
        Legibility stroke around the band outline (via ``path_effects``).
        Default ``None`` — no stroke.
    stroke_lw : float, optional
        Total stroke width in points. Default ``3.0``; applies only when
        ``stroke_color`` is set.
    **kwargs
        Passed to PathPatch.
    """
    region = CompoundRegion(ax).add_longitude_band(
        lon_min, lon_max, lat_min=lat_min, lat_max=lat_max,
        resolution=resolution, clip=clip)
    if complement:
        region.complement()
    _apply_stroke_kwargs(kwargs, stroke_color, stroke_lw)
    return region.render(**kwargs)



# ===== Great-circle band =====

def add_great_circle_band(ax: Any, ra_pole: SkyCoord | float, dec_pole: Any = None,
                          half_width: Any = None, resolution: int = 500,
                          complement: bool = False, clip: str = 'auto',
                          stroke_color: Any = None, stroke_lw: float = 3.0,
                          **kwargs: Any) -> Any:
    """
    Add a band along an arbitrary great circle to WCSAxes.

    The great circle is defined by its pole — every point on the great
    circle is exactly 90° from the pole.  The band extends ``half_width``
    degrees on either side.

    This generalises latitude bands (pole at celestial pole), galactic
    bands (pole at galactic pole), and ecliptic bands (pole at ecliptic
    pole) to arbitrary orientations — e.g., satellite orbital planes,
    scanning-law strips, or custom avoidance zones.

    Parameters
    ----------
    ax : WCSAxes
    ra_pole : float or SkyCoord
        RA of the great-circle pole in degrees (ICRS), or a SkyCoord.
    dec_pole : float or None
        Dec of the pole in degrees.  When *ra_pole* is a SkyCoord this
        positional slot holds *half_width* instead.
    half_width : float, Quantity, or None
        Half-width of the band in degrees.
    resolution : int
        Boundary point count per edge.
    complement : bool
        If True, fill everything except the band.
    clip : str
        Projection-seam handling pipeline. ``'auto'`` (default)
        resolves to ``'d3'`` for closed-region patches.
    stroke_color : color, optional
        Legibility stroke around the band outline (via ``path_effects``).
        Default ``None`` — no stroke.
    stroke_lw : float, optional
        Total stroke width in points. Default ``3.0``; applies only when
        ``stroke_color`` is set.
    **kwargs
        Passed to PathPatch.
    """
    region = CompoundRegion(ax).add_great_circle_band(
        ra_pole, dec_pole, half_width=half_width, resolution=resolution,
        clip=clip)
    if complement:
        region.complement()
    _apply_stroke_kwargs(kwargs, stroke_color, stroke_lw)
    return region.render(**kwargs)



# ===== Cross-frame lon/lat box =====

def add_lonlat_box(ax: Any, lat_min: Any, lat_max: Any, lon_min: Any,
                   lon_max: Any, frame: str = 'galactic', resolution: int = 100,
                   complement: bool = False, clip: str = 'auto',
                   stroke_color: Any = None, stroke_lw: float = 3.0,
                   **kwargs: Any) -> Any:
    """
    Add a closed lon/lat-aligned box defined in another coordinate frame.

    Cross-frame analogue of :func:`add_latitude_band` /
    :func:`add_longitude_band`: the box ``(lat_min, lat_max, lon_min,
    lon_max)`` is defined in *frame* (e.g. galactic), converted to the
    axes' frame via :class:`~astropy.coordinates.SkyCoord`, and
    projected through the same antimeridian-clip + stitch machinery
    used by :func:`add_frame_band`.

    Useful for surveys whose footprint is most naturally described as
    a lon/lat rectangle in a non-axes frame — e.g. eROSITA's western
    galactic hemisphere (``l=180..360, b=-90..+90`` in galactic).

    Polar-touching edges (``lat_max >= 89.9`` / ``lat_min <= -89.9``)
    are handled: the corresponding edge collapses to a point and is
    omitted from the outline walk. Longitude wrap
    (``lon_max < lon_min``) is normalized so the box is the slice the
    user intended, not its complement.

    Parameters
    ----------
    ax : WCSAxes
    lat_min, lat_max : float or Quantity
        Latitude limits in degrees, in *frame*.
    lon_min, lon_max : float or Quantity
        Longitude limits in degrees, in *frame*. ``lon_max < lon_min``
        is interpreted as a wraparound box (``lon_max`` is bumped by
        360°).
    frame : str
        Source frame for the box: ``'galactic'``, ``'ecliptic'``
        (alias for ``'geocentrictrueecliptic'``), ``'icrs'``,
        ``'fk5'``, etc.
    resolution : int
        Samples along each lon-constant edge (the densification that
        matters for antimeridian-wrap handling). Default 100.
    complement : bool
        If True, fill everything EXCEPT the box.
    stroke_color : color, optional
        Legibility stroke around the box outline (via ``path_effects``).
        Default ``None`` — no stroke.
    stroke_lw : float, optional
        Total stroke width in points. Default ``3.0``; applies only when
        ``stroke_color`` is set.
    **kwargs
        Passed to the resulting :class:`~matplotlib.patches.PathPatch`.
    """
    region = CompoundRegion(ax).add_lonlat_box(
        lat_min, lat_max, lon_min, lon_max,
        frame=frame, resolution=resolution, clip=clip)
    if complement:
        region.complement()
    _apply_stroke_kwargs(kwargs, stroke_color, stroke_lw)
    return region.render(**kwargs)


# ===== Cross-frame band (add_frame_band + helpers) =====

def add_frame_band(ax: Any, lat_min: Any, lat_max: Any, frame: str = 'galactic',
                   backend: str = 'patch', resolution: int | str = 'auto',
                   clip: str = 'auto', stroke_color: Any = None,
                   stroke_lw: float = 3.0, **kwargs: Any) -> Any:
    """
    Add a latitude band defined in another coordinate frame to WCSAxes.

    Two rendering backends are available:

    ``'patch'`` (default)
        D3-style antimeridian pre-clipping: clips exclusion caps against
        the antimeridian before projecting, then stitches segments via
        boundary walks.  Produces ``PathPatch`` objects with smooth vector
        edges — no rasterization grid, no staircase artifacts.  Suitable
        for layered compositing and vector export (PDF/SVG).  ~2x faster
        than contour.

    ``'contour'``
        Rasterizes the membership function at each pixel and renders via
        ``contourf``.  Completely robust fallback — evaluates membership
        independently at each pixel, so topology is always correct.
        Edges are smoothed via a sub-pixel box blur.

    Parameters
    ----------
    ax : WCSAxes
        The axes to render on.
    lat_min, lat_max : float or Quantity
        Band latitude limits in degrees, in the source frame.
    frame : str
        Astropy coordinate frame name: ``'galactic'``, ``'ecliptic'``,
        ``'geocentrictrueecliptic'``, etc.
    backend : str, optional
        ``'patch'`` (default) or ``'contour'``. Plural / singular
        forms are accepted interchangeably.
    resolution : int or ``'auto'``, optional
        Rendering resolution; its meaning depends on *backend*. Default
        ``'auto'``:

        * ``backend='patch'`` — number of longitude samples per cap
          boundary edge. ``'auto'`` resolves to 500.
        * ``backend='contour'`` — pixel grid resolution. ``'auto'`` sets
          the grid to 2x the axes pixel extent, giving smooth edges that
          match the plot scale (~720 grid points for a standard 360x180
          all-sky plot, ~0.14 s). Pass an explicit integer for manual
          control (lower = faster, higher = smoother).
    stroke_color : color, optional
        Legibility stroke around the band outline (via ``path_effects``).
        Default ``None`` — no stroke.
    stroke_lw : float, optional
        Total stroke width in points. Default ``3.0``; applies only when
        ``stroke_color`` is set.
    **kwargs
        Styling keyword arguments.  Both backends accept ``facecolor``,
        ``edgecolor``, ``alpha``.  The patch backend additionally accepts
        any ``PathPatch`` keyword.  The contour backend accepts
        ``linewidth`` / ``lw`` for the edge contour, and ``smooth``
        (bool, default False) to apply a sub-pixel box blur that
        eliminates staircase contour edges.

    Returns
    -------
    artists
        For ``backend='contour'``: tuple of ``(contourf_set, contour_set)``
        or just ``contourf_set`` if no edge is drawn.
        For ``backend='patch'``: list of ``PathPatch`` objects.

    Notes
    -----
    Requires ``astropy.coordinates.SkyCoord`` for frame transformations.
    The axes WCS should be ICRS (equatorial) for the transformation to
    work correctly.
    """
    backend = _resolve_backend(backend, helper_name='add_frame_band',
                                valid=('patch', 'contour'))

    lat_min = _parse_angle(lat_min)
    lat_max = _parse_angle(lat_max)
    if lat_min >= lat_max:
        raise ValueError("lat_min must be less than lat_max")

    _apply_stroke_kwargs(kwargs, stroke_color, stroke_lw)
    if backend == 'contour':
        # Contour pipeline is rasterise-based; clip= has no effect here.
        return _add_frame_band_contour(ax, lat_min, lat_max, frame,
                                        resolution=resolution, **kwargs)
    # Patch backend: 'auto' resolution means 500 boundary samples per edge.
    res = 500 if resolution == 'auto' else int(resolution)
    return _add_frame_band_patch(ax, lat_min, lat_max, frame,
                                  resolution=res, clip=clip, **kwargs)


def _smooth_mask(mask: npt.ArrayLike,
                 valid: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Apply a small box blur to a binary membership mask to smooth contour edges.

    The binary 0/1 mask produces staircase artifacts when contoured because
    the 0.5 level snaps to grid cell boundaries.  A small uniform blur
    creates a smooth gradient at the boundary, so the 0.5 contour traces a
    smooth curve instead.

    The blur is confined to valid (non-NaN) pixels.  Outside-projection
    pixels remain NaN so contourf ignores them.

    Parameters
    ----------
    mask : ndarray
        2D array with 0.0 (outside band), 1.0 (inside band), NaN (outside
        projection).
    valid : ndarray of bool
        Which pixels are inside the projection.

    Returns
    -------
    ndarray
        Smoothed mask with the same NaN pattern.
    """
    mask = np.asarray(mask, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    # Replace NaN with 0 for filtering, track valid pixels
    m = np.where(valid, mask, 0.0)

    # Uniform (box) filter — pure numpy, no scipy dependency.
    # Kernel size 3×3 is just enough to smooth the staircase
    # without visibly shifting boundaries.
    kernel_size = 3
    from numpy.lib.stride_tricks import sliding_window_view
    pad = kernel_size // 2
    # Pad with edge values to avoid shrinking
    m_pad = np.pad(m, pad, mode='edge')
    v_pad = np.pad(valid.astype(float), pad, mode='constant',
                   constant_values=0)
    # Windowed sum / windowed count (only average over valid pixels)
    windows_m = sliding_window_view(m_pad, (kernel_size, kernel_size))
    windows_v = sliding_window_view(v_pad, (kernel_size, kernel_size))
    smoothed = np.sum(windows_m, axis=(-2, -1)) / np.maximum(
        np.sum(windows_v, axis=(-2, -1)), 1)

    # Restore NaN for outside-projection pixels
    smoothed[~valid] = np.nan
    return smoothed


def _add_frame_band_contour(ax: Any, lat_min: float, lat_max: float,
                            frame: str, resolution: int | str = 'auto',
                            **kwargs: Any) -> Any:
    """
    Contour-based cross-frame band rendering (rasterize-then-render).

    Evaluates the latitude membership function at every pixel in the
    axes, then renders the result via ``contourf``.  This is completely
    robust because the membership test is evaluated independently at each
    pixel -- no boundary polygon projection, splitting, or reconstruction
    is needed.

    The rendering is projection-agnostic: it works identically for
    Aitoff, Mollweide, Sanson-Flamsteed, Parabolic, PlateCarree, or any
    other WCS projection supported by astropy.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from matplotlib.colors import to_rgba

    if frame == 'ecliptic':
        frame = 'geocentrictrueecliptic'

    wcs = ax.wcs

    # Build pixel grid covering the axes
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Resolve resolution: 'auto' = 2x the axes pixel extent
    if resolution == 'auto':
        nx = max(200, int(2 * (xlim[1] - xlim[0])))
    else:
        nx = int(resolution)
    ny = max(1, int(nx * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])))

    px = np.linspace(xlim[0], xlim[1], nx)
    py = np.linspace(ylim[0], ylim[1], ny)
    pxx, pyy = np.meshgrid(px, py)

    # Pixel -> world (ICRS)
    ra, dec = wcs.pixel_to_world_values(pxx, pyy)

    # Mask invalid pixels (outside projection)
    valid = np.isfinite(ra) & np.isfinite(dec)

    # Transform to source frame and build a signed-distance-in-latitude
    # field instead of a binary mask. For each pixel:
    #   inside the band:   +min(lat - lat_min, lat_max - lat)   (degrees inward)
    #   outside the band:  -min(|lat - lat_min|, |lat - lat_max|) (degrees outward)
    # The level-0 contour then traces the lat_min / lat_max curves through
    # the pixel grid via matplotlib's sub-pixel linear interpolation, so the
    # rendered edges are smooth — no staircase, no box-blur kludge needed.
    field = np.full_like(ra, np.nan)
    if np.any(valid):
        coords = SkyCoord(ra[valid] * u.deg, dec[valid] * u.deg, frame='icrs')
        src = coords.transform_to(frame)
        # Handle both lat-lon and l-b frame conventions
        src_lat = src.lat.deg if hasattr(src, 'lat') else src.b.deg
        d_lo = src_lat - lat_min
        d_hi = lat_max - src_lat
        inside = (d_lo >= 0) & (d_hi >= 0)
        # Signed distance: positive inside, negative outside.
        signed = np.where(
            inside,
            np.minimum(d_lo, d_hi),
            -np.minimum(np.abs(d_lo), np.abs(d_hi)),
        )
        field[valid] = signed

    # The signed-distance field above is already smooth to sub-pixel
    # accuracy, so the optional ``smooth=`` box blur defaults to off;
    # it remains available for explicit user control.
    smooth = kwargs.pop('smooth', False)
    if smooth:
        # Box-blur the field; keeps the level-0 crossing in place to
        # first order but rounds off corners at sharp orientation
        # changes.
        field = _smooth_mask(field, valid)

    # Extract styling kwargs
    facecolor = kwargs.pop('facecolor', kwargs.pop('fc', 'lightblue'))
    alpha = kwargs.pop('alpha', 0.5)
    edgecolor = kwargs.pop('edgecolor', kwargs.pop('ec', 'none'))
    linewidth = kwargs.pop('linewidth', kwargs.pop('lw', 1.0))

    # Render fill via contourf (level 0 separates inside/outside).
    rgba = to_rgba(facecolor, alpha=alpha)
    fill_set = ax.contourf(pxx, pyy, field,
                           levels=[0.0, np.nanmax(field) + 1.0],
                           colors=[rgba],
                           transform=ax.get_transform('pixel'),
                           antialiased=True)

    # Render edge via contour (if requested) — also at level 0.
    edge_set = None
    if edgecolor not in ('none', None):
        edge_set = ax.contour(pxx, pyy, field, levels=[0.0],
                              colors=[edgecolor], linewidths=linewidth,
                              transform=ax.get_transform('pixel'))
        # Route the legibility stroke onto the drawn edge contour (ax.contour
        # takes no path_effects arg, so set it on the resulting artist).
        pe = kwargs.get('path_effects')
        if pe is not None:
            try:
                edge_set.set_path_effects(pe)      # mpl >= 3.8: a Collection
            except AttributeError:                 # older: per-collection
                for _coll in edge_set.collections:
                    _coll.set_path_effects(pe)

    return (fill_set, edge_set) if edge_set is not None else fill_set


def _add_frame_band_patch(ax: Any, lat_min: float, lat_max: float,
                          frame: str, resolution: int = 500,
                          clip: str = 'auto', **kwargs: Any) -> Any:
    """
    Patch-based cross-frame band rendering — thin wrapper around
    :class:`~skyplothelper.geometry.compound.CompoundRegion`'s
    ``add_frame_band``.

    For the band's latitude *boundary* lines (the ``edgecolor=`` /
    ``linewidth=`` / ``linestyle=`` styling), we still draw separate
    `_project_frame_boundary_line` polylines so the edge follows
    the actual cross-frame latitude curves rather than the shapely
    polygon outline (which can dip down along the projection-frame
    edge). The fill itself comes from CompoundRegion.
    """
    edgecolor = kwargs.pop('edgecolor', kwargs.pop('ec', 'none'))
    linewidth = kwargs.pop('linewidth', kwargs.pop('lw', 1.0))
    linestyle = kwargs.pop('linestyle', kwargs.pop('ls', '-'))

    region = CompoundRegion(ax).add_frame_band(
        lat_min, lat_max, frame=frame, resolution=resolution, clip=clip)
    # Render fill only (edges drawn separately below as latitude curves).
    fill_kwargs = dict(kwargs)
    fill_kwargs['edgecolor'] = 'none'
    fill_kwargs['linewidth'] = 0
    patches = region.render(**fill_kwargs)

    # Resolve the frame name for the edge-line tracer
    fr = 'geocentrictrueecliptic' if frame == 'ecliptic' else frame

    # Edge lines — project boundary latitudes as separate line plots.
    if edgecolor not in ('none', None):
        edge_lats = []
        if lat_min > -89.9:
            edge_lats.append(lat_min)
        if lat_max < 89.9:
            edge_lats.append(lat_max)
        for elat in edge_lats:
            segments = _project_frame_boundary_line(
                ax, elat, fr, n_pts=1000)
            for sx, sy in segments:
                # The legibility stroke has to land on THESE drawn boundary
                # lines — the visible band outline — not on the fill patch,
                # whose edge is transparent + lw=0 (so a path_effects there is
                # invisible). Route it here from kwargs.
                ax.plot(sx, sy, color=edgecolor, linewidth=linewidth,
                        linestyle=linestyle,
                        transform=ax.get_transform('pixel'),
                        zorder=kwargs.get('zorder', 5),
                        path_effects=kwargs.get('path_effects'))

    return patches


def _project_frame_boundary_line(
    ax: Any, lat_val: float, frame: str, n_pts: int = 1000,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Project a latitude line from a source frame to pixel-space segments.

    Returns a list of ``(x_array, y_array)`` segments, split at projection
    boundary gaps.  Used to draw clean band boundary lines that follow
    the latitude curves without tracing the frame boundary.

    Parameters
    ----------
    ax : WCSAxes
    lat_val : float
        Latitude in degrees in the source frame.
    frame : str
        Astropy frame name (already resolved, e.g. 'geocentrictrueecliptic').
    n_pts : int
        Number of longitude samples.

    Returns
    -------
    list of (ndarray, ndarray)
        Pixel-space segments.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    lons = np.linspace(0, 360, n_pts, endpoint=False)
    coords = SkyCoord(lons * u.deg, np.full(n_pts, lat_val) * u.deg,
                       frame=frame)
    # world_to_pixel_values expects the WCS's NATIVE frame, so convert into
    # that — not unconditionally to ICRS, which mis-projects the line on a
    # galactic / ecliptic axes (a flat b=0 line came out as a wide arc).
    from ..wcs_frame import _get_wcs_frame_name
    ax_lon, ax_lat = _coords_to_frame_deg(coords, _get_wcs_frame_name(ax))
    x, y = ax.wcs.world_to_pixel_values(ax_lon, ax_lat)

    valid = np.isfinite(x) & np.isfinite(y)
    dx = np.diff(x)
    dy = np.diff(y)
    dists = np.sqrt(dx**2 + dy**2)

    # Gap threshold: 10% of frame width
    xlim = ax.get_xlim()
    gap_thresh = 0.1 * (xlim[1] - xlim[0])

    # An edge (i, i+1) breaks the polyline when either endpoint is invalid or
    # the pixel gap exceeds the threshold. Find all breaks at once (vectorized)
    # and split into contiguous runs, instead of a Python loop over every point.
    edge_bad = (~valid[:-1]) | (~valid[1:]) | (dists > gap_thresh)
    breaks = np.nonzero(edge_bad)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(x)]))       # exclusive

    segments = []
    for s, e in zip(starts, ends):
        mask = valid[s:e]
        if np.count_nonzero(mask) > 1:
            segments.append((x[s:e][mask], y[s:e][mask]))

    return segments




# ============================================================
# CompoundRegion — boolean shape combinations
# ============================================================
