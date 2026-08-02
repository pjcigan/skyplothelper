"""Frame polygon detection and geometry conversion helpers.

Helpers for extracting the visible frame boundary from a WCSAxes,
detecting self-intersections, splitting rings, and converting between
matplotlib Paths and shapely geometries.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt  # noqa: F401  (used by some helpers via plt.gca)
import numpy as np
import numpy.typing as npt
from astropy.wcs import WCS
from matplotlib.patches import PathPatch  # noqa: F401
from matplotlib.path import Path

try:
    from shapely.geometry import LineString, Point, Polygon  # noqa: F401
    from shapely.ops import polygonize, unary_union  # noqa: F401
    from shapely.validation import make_valid  # noqa: F401
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    from cartopy import geodesic  # noqa: F401
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False


# Offset from pole for containment checks
POLE_OFFSET_DEG = 0.1


# ===== Frame extraction helpers =====

def _get_projection_center(ax: Any) -> float:
    """Get the central longitude of the WCS projection."""
    try:
        return float(ax.wcs.wcs.crval[0])
    except Exception:
        return 0.0


# ============================================================
# Frame boundary
# ============================================================

def _get_frame_polygon(ax: Any, n_pts: int = 500) -> Any:
    """
    Get the frame boundary as a shapely Polygon.

    Combines two sources for maximum correctness:
    1. The visual frame boundary (from the WCSAxes frame/patch) —
       what the user actually sees drawn on screen.
    2. The WCS projection boundary (empirically traced) —
       the valid region of the projection.

    Returns their intersection so shapes are clipped to both the drawn
    frame AND the valid projection region.  This handles standard frames
    (elliptical, rectangular), custom frames (hexagonal, etc.), and any
    single-boundary all-sky projection (Aitoff, Mollweide, Sanson-Flamsteed,
    Parabolic, PlateCarree, etc.) without hardcoding.
    """
    # --- Interrupted / oddball frames: use the analytic visible boundary ---
    # Some projections have a true visible-region silhouette that the package
    # already computes (and trusts for the image data clip — _DATA_CLIP_CODES
    # in wcs_frame) but that the generic _get_wcs_boundary antimeridian trace
    # below gets WRONG:
    #   * Conics (COD/COE/COO/COP): the lon_center±180 meridian is the two
    #     straight wedge edges, so the trace closes into a thin bogus sliver
    #     (NOT the visible fan). With area > 1 the empty-result guard never
    #     trips, so every shape clips away — tissot/circles silently vanish.
    #   * Interrupted projections (HPX diamonds, XPH butterfly): the visible
    #     region is the zigzag/butterfly outline, but the generic trace yields
    #     the bounding rectangle, so a polar shape STREAKS across the V-notches
    #     between facets instead of being clipped to them.
    # In both cases the authoritative boundary (∩ the drawn axes) is correct.
    # _projection_boundary returns None for a field-view conic and for standard
    # projections, so they fall through to the generic path unchanged. The quad
    # cubes (TSC/CSC/QSC) are deliberately NOT here: their cross outline can't
    # separate the interior face-to-face seams, so the boundary clip only
    # partially helps — left on the generic path pending interior-seam handling.
    # Lazy import: wcs_frame imports this module.
    _BOUNDARY_CLIP_CODES = ('COD', 'COE', 'COO', 'COP', 'HPX', 'XPH')
    try:
        from ..wcs_frame import _axes_fits_code, _projection_boundary
        if _axes_fits_code(ax) in _BOUNDARY_CLIP_CODES:
            pb = _projection_boundary(ax)
            if pb is not None:
                verts = pb.vertices[np.isfinite(pb.vertices).all(axis=1)]
                bound = Polygon(verts)
                if not bound.is_valid:
                    bound = make_valid(bound)
                visual = _get_visual_frame(ax)
                if visual is not None:
                    bound = _safe_intersection(bound, visual)
                if bound.geom_type == 'MultiPolygon':
                    bound = max(bound.geoms, key=lambda p: p.area)
                if not bound.is_empty and bound.area > 1:
                    return bound
    except Exception:
        pass

    # --- Source 1: Visual frame boundary ---
    visual_poly = _get_visual_frame(ax)

    # --- Source 2: WCS projection boundary ---
    wcs_poly = _get_wcs_boundary(ax.wcs, n_pts)

    # Intersect both for most conservative clip
    if visual_poly is not None and wcs_poly is not None:
        try:
            result = _safe_intersection(visual_poly, wcs_poly)
            if not result.is_empty and result.area > 1:
                if result.geom_type == 'MultiPolygon':
                    result = max(result.geoms, key=lambda p: p.area)
                return result
        except Exception:
            pass
        # The intersection was empty or invalid. ``_get_wcs_boundary``
        # traces the lon_center±180 antimeridian, which is meaningful
        # for all-sky projections (Aitoff, Mollweide, ...) but
        # extrapolates wildly for zenithal/conic field projections
        # (TAN, SIN, ARC, ...) where the antimeridian is far outside
        # the projection's valid domain. In that case the user's
        # actual axes (``visual_poly``) is correct; ignore the bogus
        # ``wcs_poly``.
        return visual_poly

    # Fall back to whichever single source is available
    if visual_poly is not None:
        return visual_poly
    if wcs_poly is not None:
        return wcs_poly

    # Last resort: bounding box
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    from shapely.geometry import box
    return box(xlim[0], ylim[0], xlim[1], ylim[1])


def _get_visual_frame(ax: Any) -> Any:
    """
    Extract the visual frame boundary from the WCSAxes frame spines.

    Tries the frame's contour spine first (works for EllipticalFrame
    and custom frames), then falls back to the axes patch path.
    """
    try:
        # Try frame contour spine (e.g. 'c' for EllipticalFrame)
        if hasattr(ax, 'coords') and hasattr(ax.coords, 'frame'):
            frame = ax.coords.frame
            # Look for a contour spine — the first one with >10 vertices
            for key in frame:
                spine = frame[key]
                if (hasattr(spine, 'data') and spine.data is not None
                        and len(spine.data) > 10):
                    poly = Polygon(spine.data)
                    if poly.is_valid and poly.area > 1:
                        return poly

        # Fallback: axes patch path
        path = ax.patch.get_path()
        if path is not None and len(path.vertices) > 4:
            poly = Polygon(path.vertices)
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda p: p.area)
            if not poly.is_empty and poly.area > 1:
                return poly
    except Exception:
        pass
    return None


# Minimum separation between the two antimeridian traces, as a fraction of the
# traced extent, for the ring they close to be a real map boundary rather than a
# degenerate sliver. Measured across every supported projection: genuine all-sky
# boundaries score >= 0.09 (PCO is the tightest; most are 0.3-0.9), while the
# degenerate zenithal / quadcube traces score <= 3e-4. 0.01 sits between them
# with ~10x margin on one side and ~40x on the other.
_ANTIMERIDIAN_SEPARATION_MIN = 0.01


def _antimeridian_traces_separate(xa: Any, ya: Any, xb: Any, yb: Any) -> bool:
    """Do the two antimeridian traces land on *opposite* map edges?

    The traces sample the same meridian a hair either side of it, so they
    coincide unless that meridian is genuinely the map's cut edge. Compares the
    typical point-to-point separation against the overall traced extent; the
    inputs must already be paired by latitude.
    """
    ok = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(xb) & np.isfinite(yb)
    if int(np.sum(ok)) < 6:
        return False
    sep = np.hypot(xa[ok] - xb[ok], ya[ok] - yb[ok])
    xs = np.concatenate([xa[ok], xb[ok]])
    ys = np.concatenate([ya[ok], yb[ok]])
    diag = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
    if not np.isfinite(diag) or diag <= 0:
        return False
    return bool(float(np.median(sep)) / diag >= _ANTIMERIDIAN_SEPARATION_MIN)


def _get_wcs_boundary(wcs: WCS, n_pts: int = 500) -> Any:
    """
    Derive the projection boundary by tracing the world-coordinate
    domain boundary through the WCS.

    For all-sky projections the boundary is the anti-center meridian
    (lon_center ± 180°) traversed south-to-north on one side, then
    north-to-south on the other. Returns ``None`` when that meridian is not
    actually the map's cut edge (see below), leaving the caller on the drawn
    frame alone.
    """
    try:
        lon_center = float(wcs.wcs.crval[0])
    except Exception:
        lon_center = 0.0

    eps = 1e-3
    half = n_pts // 2
    lats = np.linspace(-90 + eps, 90 - eps, half)

    lon_a = lon_center - 180 + eps
    xa, ya = wcs.world_to_pixel_values(np.full(half, lon_a), lats)

    lon_b = lon_center + 180 - eps
    xb, yb = wcs.world_to_pixel_values(np.full(half, lon_b), lats[::-1])

    # ``lon_a`` and ``lon_b`` are the SAME meridian, offset by ±eps. On a map
    # whose edge is that meridian (cylindrical / pseudocylindrical / HPX ...)
    # the two traces land on opposite edges and the ring between them is the map
    # boundary. On a zenithal projection the antimeridian is an interior curve,
    # so both traces fall on top of each other and the "ring" degenerates into a
    # sliver hugging that curve. The sliver's area can still clear the ``> 1``
    # guard below, and intersecting it with the drawn frame then clips away every
    # shape drawn on the axes -- which is what collapsed ZEA and ARC to a thin
    # meridian. Pair the traces by latitude (``xb`` was traced north-to-south)
    # and reject the degenerate case outright.
    if not _antimeridian_traces_separate(xa, ya, xb[::-1], yb[::-1]):
        return None

    all_x = np.concatenate([xa, xb])
    all_y = np.concatenate([ya, yb])

    valid = np.isfinite(all_x) & np.isfinite(all_y)
    if np.sum(valid) >= 10:
        poly = Polygon(zip(all_x[valid], all_y[valid]))
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.geom_type == 'MultiPolygon':
            poly = max(poly.geoms, key=lambda p: p.area)
        if not poly.is_empty and poly.area > 1:
            return poly
    return None


# ============================================================
# World-coordinate boundary detection
# ============================================================


# ===== Self-intersection / ring-split helpers =====

def _find_self_intersections(x: npt.ArrayLike,
                             y: npt.ArrayLike) -> list[tuple[Any, Any, int]]:
    """Find self-intersection points of a closed polygon path in pixel space."""
    x = np.asarray(x)
    y = np.asarray(y)
    xc = np.append(x, x[0])
    yc = np.append(y, y[0])
    ls = LineString(zip(xc, yc))

    if ls.is_simple:
        return []

    noded = unary_union(ls)
    all_coords = []
    if noded.geom_type == 'MultiLineString':
        for seg in noded.geoms:
            all_coords.extend(list(seg.coords))
    elif noded.geom_type == 'LineString':
        all_coords = list(noded.coords)
    else:
        return []

    coord_counts = Counter(tuple(c) for c in all_coords)
    intersection_pts = [c for c, count in coord_counts.items() if count >= 3]

    result = []
    for pt in intersection_pts:
        dists = (x - pt[0])**2 + (y - pt[1])**2
        idx = int(np.argmin(dists))
        result.append((pt[0], pt[1], idx))

    result.sort(key=lambda r: r[2])
    return result


def _split_ring_at_indices(
        x: np.ndarray, y: np.ndarray, split_indices: Sequence[int]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a coordinate ring at the given indices."""
    if len(split_indices) < 2:
        return [(x, y)]

    idxs = sorted(split_indices)
    segments = []
    for i in range(len(idxs)):
        start = idxs[i]
        end = idxs[(i + 1) % len(idxs)]
        if end > start:
            seg_x, seg_y = x[start:end + 1], y[start:end + 1]
        else:
            seg_x = np.concatenate([x[start:], x[:end + 1]])
            seg_y = np.concatenate([y[start:], y[:end + 1]])
        if len(seg_x) >= 3:
            segments.append((seg_x, seg_y))
    return segments



# ===== Path / shapely conversion =====

def _shapely_to_paths(geom: Any, min_area: float = 1.0) -> list[Path]:
    """Convert a shapely geometry to a list of matplotlib Paths.

    Filters out zero-area sliver artifacts from shapely polygon
    operations. The default threshold of 1 pixel² is appropriate for
    rendering full-sized regions like survey footprints, where any
    sub-pixel piece is almost certainly a numerical artifact rather
    than a real shape. Pass a smaller value (e.g. 0.0) for callers
    that render small primitives such as individual HEALPix tile
    halves at the antimeridian, where each piece can legitimately
    be a fraction of a pixel² yet must still render.
    """
    paths: list[Path] = []
    if geom.is_empty:
        return paths
    if geom.geom_type == 'Polygon':
        if geom.area > min_area:
            paths.append(_poly_to_path(geom))
    elif geom.geom_type == 'MultiPolygon':
        for p in geom.geoms:
            if not p.is_empty and p.area > min_area:
                paths.append(_poly_to_path(p))
    elif geom.geom_type == 'GeometryCollection':
        for g in geom.geoms:
            paths.extend(_shapely_to_paths(g, min_area=min_area))
    return paths


def _geom_to_clip_path(geom: Any, frame_poly: Any = None,
                       complement: bool = False) -> Path:
    """Build a single matplotlib clip ``Path`` from a shapely geometry (already
    in the axes' pixel / data coordinates), for use with ``set_clip_path``.

    With ``complement=True`` the path covers ``frame_poly`` minus ``geom``
    (e.g. ocean = frame minus land). Small pieces are kept (``min_area=0``) so
    nothing silently drops out of a mask.
    """
    if complement and frame_poly is not None:
        geom = (frame_poly if (geom is None or geom.is_empty)
                else frame_poly.difference(geom))
    if geom is None or getattr(geom, 'is_empty', True):
        return Path(np.empty((0, 2)))
    paths = _shapely_to_paths(geom, min_area=0.0)
    return Path.make_compound_path(*paths) if paths else Path(np.empty((0, 2)))


def _poly_to_path(poly: Any) -> Path:
    verts = list(poly.exterior.coords)
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    for interior in poly.interiors:
        hole = list(interior.coords)
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(hole) - 2) + [Path.CLOSEPOLY])
        verts.extend(hole)
    return Path(np.array(verts), codes)


def _safe_intersection(geom: Any, clip_geom: Any) -> Any:
    """``geom.intersection(clip_geom)`` with the off-limb noise silenced.

    Clipping a projected shape against the frame silhouette is the core step
    of the region pipeline. On extreme projections (COE / HPX / PCO / BON, any
    SIN-limb straddle) or for the degenerate sub-pixel slivers a high-nside
    HEALPix tiling feeds in, the projected geometry legitimately carries NaN /
    degenerate vertices, and shapely 2.x flags the GEOS topology-invalid as a
    ``RuntimeWarning: invalid value encountered in intersection``. The empty /
    partial result is exactly what callers want (the visible sliver is kept),
    so the warning is pure noise.

    shapely 2.x surfaces this through TWO channels depending on the failure:
    a NaN coordinate trips numpy's floating-point ``invalid`` flag (caught by
    ``np.errstate``), while a degenerate-topology result is emitted via
    ``warnings.warn`` (NOT caught by ``np.errstate`` — the high-nside case).
    Guard both, scoped to this one op and to the specific message so genuine
    RuntimeWarnings elsewhere still surface.
    """
    with warnings.catch_warnings(), np.errstate(invalid='ignore'):
        warnings.filterwarnings(
            'ignore',
            message='.*invalid value encountered in intersection.*',
            category=RuntimeWarning)
        return geom.intersection(clip_geom)


def _union_and_clip(geoms: list[Any], frame_poly: Any) -> list[Path]:
    """
    Union a list of shapely geometries, clip to frame, and convert to paths.

    Using shapely's ``unary_union`` at output collection points:
    - Eliminates tiny overlaps between segment halves (e.g. from
      ``_compact_pixel_split``)
    - Produces clean MultiPolygon output without duplicate coverage
    - Simplifies downstream rendering (single patch per connected region)

    Parameters
    ----------
    geoms : list of shapely geometry
        Individual segment geometries (may overlap slightly).
    frame_poly : shapely Polygon
        Frame boundary for final clipping.

    Returns
    -------
    list of matplotlib Path objects.
    """
    if not geoms:
        return []
    combined = unary_union([g for g in geoms if g is not None and not g.is_empty])
    if combined.is_empty:
        return []
    try:
        clipped = _safe_intersection(combined, frame_poly)
    except Exception:
        clipped = combined
    return _shapely_to_paths(clipped) if not clipped.is_empty else []


# ============================================================
# Public API
# ============================================================

def _fix_hairline_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Prevent matplotlib hairline artifacts on PathPatch edges.

    When ``edgecolor='none'``, matplotlib still renders a visible
    antialiased edge at the default ``linewidth=1.0``.  This helper
    sets ``linewidth=0`` when edgecolor is 'none' and the user hasn't
    explicitly specified a linewidth.
    """
    ec = kwargs.get('edgecolor', kwargs.get('ec', None))
    if ec in ('none', None) and 'linewidth' not in kwargs and 'lw' not in kwargs:
        kwargs['linewidth'] = 0
    return kwargs



# ===== Paths-to-geometry =====

def _paths_to_geom(paths: list[Path], min_area: float = 1.0) -> Any:
    """Convert list of matplotlib Paths to a shapely geometry (union).

    ``min_area`` (px²) filters sub-pixel sliver artifacts from the
    Path→Polygon conversion. The default of 1.0 suits full-sized
    regions; pass a smaller value (e.g. 0.0) when the caller
    legitimately renders sub-pixel pieces — deep-field survey
    footprints driving ``add_spherical_polygon`` with
    ``min_piece_area=0`` route through here, so the filter must match
    the upstream ``_stitch_and_project`` threshold.
    """
    polys: list[Any] = []
    for path in paths:
        try:
            verts = np.asarray(path.vertices)
            if len(verts) >= 4:  # need at least 3 + closing
                p = Polygon(verts)
                if not p.is_valid:
                    p = make_valid(p)
                if not p.is_empty and p.area > min_area:
                    if p.geom_type == 'MultiPolygon':
                        polys.extend(g for g in p.geoms if g.area > min_area)
                    elif p.geom_type == 'Polygon':
                        polys.append(p)
                    elif p.geom_type == 'GeometryCollection':
                        polys.extend(g for g in p.geoms
                                     if g.geom_type == 'Polygon'
                                     and g.area > min_area)
        except Exception:
            continue
    if not polys:
        return None
    return unary_union(polys) if len(polys) > 1 else polys[0]


def _complement_detect(clipped: Any, frame_poly: Any,
                       expected_frac: float | None) -> Any:
    """Flip a frame-clipped region to its complement when that is the
    more plausible reading of an ambiguous stitch result.

    A polygon that straddles the projection wrap edge (or encloses a
    pole) can emerge from the projection/stitch step as the *outside* of
    the intended region. ``expected_frac`` (the region's expected
    solid-angle fraction) disambiguates: if the clipped area is closer to
    ``(1 - expected_frac)`` of the frame than to ``expected_frac``, the
    stitch produced the complement, so flip it back.

    This is the shared core of the heuristic — both the matplotlib
    pipeline (``_stitch_and_project``) and the plotly projector route
    through here so the two backends can never drift on it.
    """
    if expected_frac is None or clipped is None or clipped.is_empty:
        return clipped
    frame_area = frame_poly.area
    expected_area = float(expected_frac) * frame_area
    complement_area = (1.0 - float(expected_frac)) * frame_area
    if abs(clipped.area - complement_area) < abs(clipped.area - expected_area):
        try:
            return frame_poly.difference(clipped)
        except Exception:
            return clipped
    return clipped


def _expected_frac_from_vertices(lons: npt.ArrayLike, lats: npt.ArrayLike,
                                 lon_center: float) -> float:
    """Estimate a polygon's solid-angle fraction of the sphere from its
    (lon, lat) vertices — the ``expected_frac`` hint :func:`_complement_detect`
    uses to decide whether a stitched region came out as its complement.

    The estimate is the area of the polygon's lon×sin(lat) bounding cell:
    ``(lon_span / 360) * |sin(lat_max) - sin(lat_min)| / 2``. It is only an
    upper-bound proxy (a convex cap fills roughly half its cell), but that is
    all the heuristic needs — it only has to sit far closer to the true small
    fraction than to ``1 - fraction`` to pick the right side. ``lon_span`` is
    measured center-relative and folded past 180° so a wrap-straddling shape
    reports its true (small) span, not ~360°.

    Factored out of :func:`_prepare_region_vertices` so the matplotlib FITS
    dispatch (``clip='d3'``) and the non-FITS ``WCSNonFitsProjector`` share one
    formula instead of two that can drift.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    sin_range = abs(np.sin(np.radians(np.max(lats)))
                    - np.sin(np.radians(np.min(lats))))
    lon_norm = ((lons - lon_center + 180) % 360) - 180
    lon_span = float(np.ptp(lon_norm))
    if lon_span > 180:
        lon_span = 360 - lon_span
    return (lon_span / 360.0) * float(sin_range) / 2.0
