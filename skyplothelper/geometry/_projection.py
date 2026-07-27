"""Projection pipeline for spherical regions on WCSAxes.

The core pipeline that takes spherical (lon, lat) coordinates and
projects them through a WCSAxes' transform to produce matplotlib paths
suitable for PathPatch rendering. Handles antimeridian crossings,
pole containment, and frame-boundary closure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.patches import PathPatch  # noqa: F401
from matplotlib.path import Path

try:
    from shapely.geometry import LineString, Point, Polygon  # noqa: F401
    from shapely.ops import polygonize, unary_union  # noqa: F401
    from shapely.validation import make_valid  # noqa: F401
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

from ._densify import _angular_separation, _densify_polygon_edges, _slerp, _unwrap_lon  # noqa: F401
from ._frame_geom import (
    POLE_OFFSET_DEG,
    _find_self_intersections,
    _get_frame_polygon,
    _get_projection_center,
    _paths_to_geom,
    _safe_intersection,
    _shapely_to_paths,
    _split_ring_at_indices,
    _union_and_clip,
)

# ===== World-space crossings + closure =====

def _detect_world_crossings(
    lons: npt.ArrayLike, lats: npt.ArrayLike, lon_center: float,
) -> tuple[list[int], list[float]]:
    """
    Detect where the shape crosses the projection boundary in world coords.

    The projection boundary is at lon_center ± 180°. We normalize longitudes
    relative to center and look for jumps > 180° between consecutive vertices.

    Returns:
        crossing_indices: list of indices where crossings occur
        crossing_lats: list of interpolated latitudes at each crossing
    """
    # Normalize to [-180, 180) relative to projection center
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    lon_norm = ((lons - lon_center + 180) % 360) - 180

    dlon = np.diff(lon_norm)
    crossing_mask = np.abs(dlon) > 180
    crossing_indices = np.where(crossing_mask)[0]

    if len(crossing_indices) == 0:
        return [], []

    # Interpolate the latitude at each crossing
    crossing_lats = []
    for idx in crossing_indices:
        lon_a, lat_a = lon_norm[idx], lats[idx]
        lon_b, lat_b = lon_norm[idx + 1], lats[idx + 1]

        # Determine which boundary (±180)
        boundary = 180.0 if lon_a > 0 else -180.0

        # Unwrap lon_b for interpolation
        if lon_a > 0 and lon_b < 0:
            lon_b_unwrap = lon_b + 360
        else:
            lon_b_unwrap = lon_b - 360

        denom = lon_b_unwrap - lon_a
        if abs(denom) > 1e-10:
            t = (boundary - lon_a) / denom
        else:
            t = 0.5
        t = np.clip(t, 0, 1)

        lat_crossing = lat_a + t * (lat_b - lat_a)
        crossing_lats.append(lat_crossing)

    return crossing_indices.tolist(), crossing_lats


def _split_world_coords(
    lons: npt.ArrayLike, lats: npt.ArrayLike,
    crossing_indices: list[int], crossing_lats: list[float],
    lon_center: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Split world coordinates at boundary crossings, inserting intersection points.

    Each segment gets boundary points (at ±180° relative to center) prepended/appended.
    Returns list of (segment_lons, segment_lats) tuples.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    lon_norm = ((lons - lon_center + 180) % 360) - 180

    n_crossings = len(crossing_indices)
    if n_crossings == 0:
        return [(lon_norm + lon_center, lats)]

    # Build segments between crossings
    boundaries = [0] + [ci + 1 for ci in crossing_indices] + [len(lons)]
    segments = []

    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        seg_lon = lon_norm[s:e].copy()
        seg_lat = lats[s:e].copy()

        if len(seg_lon) < 1:
            continue

        # Prepend boundary point from previous crossing
        if i > 0:
            ci = i - 1  # which crossing
            lat_b = crossing_lats[ci]
            # This segment's lons: determine which side of boundary
            if len(seg_lon) > 0:
                if seg_lon[0] < 0:
                    seg_lon = np.concatenate([[-180.0], seg_lon])
                else:
                    seg_lon = np.concatenate([[180.0], seg_lon])
                seg_lat = np.concatenate([[lat_b], seg_lat])

        # Append boundary point from next crossing
        if i < n_crossings:
            ci = i
            lat_b = crossing_lats[ci]
            if len(seg_lon) > 0:
                if seg_lon[-1] > 0:
                    seg_lon = np.append(seg_lon, 180.0)
                else:
                    seg_lon = np.append(seg_lon, -180.0)
                seg_lat = np.append(seg_lat, lat_b)

        if len(seg_lon) >= 3:
            segments.append((seg_lon + lon_center, seg_lat))

    return segments


def _close_world_segment(
    seg_lons: np.ndarray, seg_lats: np.ndarray, lon_center: float,
    pole_side: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Close an open world-coordinate segment by tracing along the boundary.

    For segments that start and end at the projection boundary (±180° from center),
    we need to connect them along the boundary meridian.

    For pole-covering shapes: trace through the (offset) pole.
    For non-pole shapes: trace along the boundary meridian between the two latitudes.
    """
    if len(seg_lons) < 3:
        return seg_lons, seg_lats

    # Check if endpoints are at the boundary
    lon_norm_start = ((seg_lons[0] - lon_center + 180) % 360) - 180
    lon_norm_end = ((seg_lons[-1] - lon_center + 180) % 360) - 180

    at_boundary = (abs(abs(lon_norm_start) - 180) < 1.0 and
                   abs(abs(lon_norm_end) - 180) < 1.0)

    if not at_boundary:
        # Already closed or doesn't need boundary closure
        return (np.append(seg_lons, seg_lons[0]),
                np.append(seg_lats, seg_lats[0]))

    lat_start = seg_lats[0]
    lat_end = seg_lats[-1]
    n_close = 30

    if pole_side is not None:
        # Close through the pole (with offset to avoid singularity)
        pole_lat = (90.0 - POLE_OFFSET_DEG) if pole_side == 'north' else (-90.0 + POLE_OFFSET_DEG)

        # Trace: end → pole along end boundary → pole along start boundary → start
        close_lats = np.concatenate([
            np.linspace(lat_end, pole_lat, n_close),
            np.linspace(pole_lat, lat_start, n_close)
        ])
        close_lons = np.concatenate([
            np.full(n_close, seg_lons[-1]),   # same boundary lon as end
            np.full(n_close, seg_lons[0])     # same boundary lon as start
        ])
    else:
        # Close along the boundary between the two latitudes
        close_lats = np.linspace(lat_end, lat_start, n_close)
        close_lons = np.full(n_close, seg_lons[-1])

    all_lons = np.concatenate([seg_lons, close_lons, [seg_lons[0]]])
    all_lats = np.concatenate([seg_lats, close_lats, [seg_lats[0]]])

    return all_lons, all_lats


# ============================================================
# Pole-offset containment check
# ============================================================

def _check_pole_containment(poly: Any, ax: Any, pole_side: str) -> bool:
    """
    Check if a projected polygon contains the pole using the offset trick.

    Projects a point at lat = ±(90 - POLE_OFFSET_DEG) and checks containment.
    This avoids the singularity at exactly ±90° where longitude is undefined.
    """
    wcs = ax.wcs
    lon_center = _get_projection_center(ax)

    pole_lat = (90.0 - POLE_OFFSET_DEG) if pole_side == 'north' else (-90.0 + POLE_OFFSET_DEG)
    pole_px, pole_py = wcs.world_to_pixel_values([lon_center], [pole_lat])

    if not (np.isfinite(pole_px[0]) and np.isfinite(pole_py[0])):
        return False

    return poly.contains(Point(pole_px[0], pole_py[0]))


# ============================================================
# Pixel-space self-intersection fallback
# ============================================================


# ===== Pixel-space closure helpers =====

def _close_pixel_segment(
    seg_x: np.ndarray, seg_y: np.ndarray, frame_poly: Any,
    pole_side: str | None = None,
) -> Any | None:
    """
    Close an open pixel-space segment along the frame boundary.

    Uses the segment's position relative to frame center to determine
    which direction to trace along the boundary.
    """
    if len(seg_x) < 3:
        return None

    # Already closed?
    d = np.sqrt((seg_x[-1] - seg_x[0])**2 + (seg_y[-1] - seg_y[0])**2)
    if d < 2.0:
        poly = Polygon(zip(seg_x, seg_y))
        if not poly.is_valid:
            poly = make_valid(poly)
        return _safe_intersection(poly, frame_poly) if not poly.is_empty else None

    # Get frame boundary ring
    frame_ring = np.array(frame_poly.exterior.coords)
    n_ring = len(frame_ring) - 1  # last == first
    frame_cx = np.mean(frame_ring[:n_ring, 0])
    frame_cy = np.mean(frame_ring[:n_ring, 1])

    def nearest_frame_idx(px: float, py: float) -> int:
        return int(np.argmin((frame_ring[:n_ring, 0] - px)**2 +
                              (frame_ring[:n_ring, 1] - py)**2))

    idx_start = nearest_frame_idx(seg_x[0], seg_y[0])
    idx_end = nearest_frame_idx(seg_x[-1], seg_y[-1])

    # Generate two candidate paths
    def trace(i_from: int, i_to: int, step: int) -> list[int]:
        indices = []
        i = i_from % n_ring
        target = i_to % n_ring
        for _ in range(n_ring + 1):
            indices.append(i)
            if i == target:
                break
            i = (i + step) % n_ring
        return indices

    path_cw = trace(idx_end, idx_start, +1)
    path_ccw = trace(idx_end, idx_start, -1)

    seg_pts = np.column_stack([seg_x, seg_y])
    seg_mean_x = np.mean(seg_x)

    best = None
    for bpath in [path_cw, path_ccw]:
        bpts = frame_ring[bpath]
        all_pts = np.vstack([seg_pts, bpts])

        try:
            poly = Polygon(all_pts)
            if not poly.is_valid:
                poly = make_valid(poly)
            clipped = _safe_intersection(poly, frame_poly)
            if clipped.is_empty:
                continue

            # Determine if this closure direction is correct:
            # Boundary should be on SAME side of frame center as the segment
            b_mean_x = np.mean(frame_ring[bpath, 0])
            b_mean_y = np.mean(frame_ring[bpath, 1])

            if pole_side == 'north':
                correct = b_mean_y > frame_cy
            elif pole_side == 'south':
                correct = b_mean_y < frame_cy
            else:
                correct = (seg_mean_x - frame_cx) * (b_mean_x - frame_cx) >= 0

            if correct:
                return clipped
            elif best is None:
                best = clipped
        except Exception:
            continue

    return best


def _close_segment_centroid(
    seg_x: np.ndarray, seg_y: np.ndarray, frame_poly: Any,
) -> Any | None:
    """
    Close an open pixel-space segment by walking the frame boundary,
    choosing the correct direction via segment centroid containment.

    For polygons crossing the projection boundary (1 or more pixel jumps),
    rolling at a jump produces an open segment whose endpoints are near the
    frame boundary.  This function closes the segment by walking the frame
    boundary from the last point back to the first, producing two candidate
    polygons (CW and CCW walks).  The correct candidate is the one that
    contains the mean position (centroid) of the segment's pixel coordinates.

    This replaces the previous ``_close_segment_short_walk`` which used an
    area-minimization heuristic.  Area minimization fails for shapes spanning
    >50% of the frame (e.g. wide cross-frame latitude bands), where the
    correct closure is the *larger* piece.

    Parameters
    ----------
    seg_x, seg_y : ndarray
        Open segment pixel coordinates.
    frame_poly : shapely Polygon
        Frame boundary for clipping.

    Returns
    -------
    shapely geometry or None
    """
    if len(seg_x) < 3:
        return None

    # Already closed?
    d = np.sqrt((seg_x[-1] - seg_x[0])**2 + (seg_y[-1] - seg_y[0])**2)
    if d < 2.0:
        poly = Polygon(zip(seg_x, seg_y))
        if not poly.is_valid:
            poly = make_valid(poly)
        return _safe_intersection(poly, frame_poly) if not poly.is_empty else None

    # Segment centroid — the "known interior" test point
    centroid_x = np.mean(seg_x)
    centroid_y = np.mean(seg_y)
    test_pt = Point(centroid_x, centroid_y)

    # Frame boundary ring
    fr = np.array(frame_poly.exterior.coords)
    n_ring = len(fr) - 1  # last == first

    def nearest_frame_idx(px: float, py: float) -> int:
        return int(np.argmin((fr[:n_ring, 0] - px)**2 +
                              (fr[:n_ring, 1] - py)**2))

    idx_start = nearest_frame_idx(seg_x[0], seg_y[0])
    idx_end = nearest_frame_idx(seg_x[-1], seg_y[-1])

    def walk_boundary(i_from: int, i_to: int, step: int) -> list[int]:
        pts = []
        i = i_from % n_ring
        target = i_to % n_ring
        for _ in range(n_ring + 1):
            pts.append(i)
            if i == target:
                break
            i = (i + step) % n_ring
        return pts

    walk_pos = walk_boundary(idx_end, idx_start, +1)
    walk_neg = walk_boundary(idx_end, idx_start, -1)

    seg_pts = np.column_stack([seg_x, seg_y])
    candidates = []

    for bwalk in [walk_pos, walk_neg]:
        try:
            all_pts = np.vstack([seg_pts, fr[bwalk]])
            poly = Polygon(all_pts)
            if not poly.is_valid:
                poly = make_valid(poly)
            clipped = _safe_intersection(poly, frame_poly)
            if clipped.is_empty:
                continue
            candidates.append(clipped)
        except Exception:
            continue

    if not candidates:
        return None

    # Pick the candidate that contains the segment centroid
    for c in candidates:
        if c.contains(test_pt):
            return c

    # Fallback: if neither contains the centroid (edge case — centroid
    # might land on the boundary or in a tiny gap), pick the larger one.
    # For well-formed polygons this rarely triggers.
    return max(candidates, key=lambda c: c.area)


# ============================================================
# Main projection function
# ============================================================

def _extract_pole_cap_smooth(
    xv: np.ndarray, yv: np.ndarray, pole_side: str, frame_poly: Any,
) -> Any | None:
    """
    Extract a smooth pole cap by splitting the projected ring and closing
    along the frame boundary.

    Algorithm (inspired by cartopy's _attach_lines_to_boundary):
    1. Split the projected ring at its extreme-x vertices into two halves
    2. Take the EQUATOR-side half (the circle boundary closest to equator)
    3. Walk the frame boundary from the right endpoint to the left endpoint
       going the LONG way around (through the pole)
    4. Build a closed polygon from equator arc + boundary walk, clip to frame

    This produces perfectly smooth edges using actual projected coordinates
    instead of the binned envelope approach.

    Parameters
    ----------
    xv, yv : ndarray
        Projected pixel coordinates (valid points only).
    pole_side : str
        'north' or 'south'.
    frame_poly : shapely Polygon
        Frame boundary.

    Returns
    -------
    shapely geometry or None
    """
    if len(xv) < 3:
        return None

    # Split ring at extreme-x vertices into two halves
    idx_left = int(np.argmin(xv))
    idx_right = int(np.argmax(xv))

    if idx_left < idx_right:
        ha_x, ha_y = xv[idx_left:idx_right + 1], yv[idx_left:idx_right + 1]
        hb_x = np.concatenate([xv[idx_right:], xv[:idx_left + 1]])
        hb_y = np.concatenate([yv[idx_right:], yv[:idx_left + 1]])
    else:
        ha_x = np.concatenate([xv[idx_left:], xv[:idx_right + 1]])
        ha_y = np.concatenate([yv[idx_left:], yv[:idx_right + 1]])
        hb_x, hb_y = xv[idx_right:idx_left + 1], yv[idx_right:idx_left + 1]

    # Equator-side = lower mean-y for north pole, higher for south
    if pole_side == 'north':
        eq_x, eq_y = (ha_x, ha_y) if np.mean(ha_y) < np.mean(hb_y) else (hb_x, hb_y)
    else:
        eq_x, eq_y = (ha_x, ha_y) if np.mean(ha_y) > np.mean(hb_y) else (hb_x, hb_y)

    if len(eq_x) < 2:
        return None

    # Ensure left-to-right ordering
    if eq_x[0] > eq_x[-1]:
        eq_x, eq_y = eq_x[::-1], eq_y[::-1]

    # Close via frame boundary walk through the pole
    return _close_arc_via_boundary(eq_x, eq_y, pole_side, frame_poly)


def _close_open_segment_via_boundary(
    seg_x: np.ndarray, seg_y: np.ndarray, pole_side: str, frame_poly: Any,
) -> Any | None:
    """
    Close an open pixel-space segment (from a boundary-crossing pole shape)
    along the frame boundary, walking through the pole.

    For pole-covering circles that also cross the projection boundary,
    rolling at the pixel jump produces a single open segment. This function
    closes it by walking the frame boundary from one endpoint to the other,
    going through the pole.

    Parameters
    ----------
    seg_x, seg_y : ndarray
        Open segment pixel coordinates.
    pole_side : str
        'north' or 'south'.
    frame_poly : shapely Polygon
        Frame boundary.

    Returns
    -------
    shapely geometry or None
    """
    if len(seg_x) < 3:
        return None

    return _close_arc_via_boundary(seg_x, seg_y, pole_side, frame_poly)


def _close_arc_via_boundary(
    arc_x: np.ndarray, arc_y: np.ndarray, pole_side: str, frame_poly: Any,
) -> Any | None:
    """
    Close an arc by walking the frame boundary from the arc's last point
    back to its first point, going through the pole side.

    This is the core boundary-walk algorithm shared by both the smooth
    pole cap extraction and the open-segment closure.

    Parameters
    ----------
    arc_x, arc_y : ndarray
        Arc pixel coordinates (ordered, open — first != last).
    pole_side : str
        'north' or 'south' — determines walk direction.
    frame_poly : shapely Polygon
        Frame boundary for walking and clipping.

    Returns
    -------
    shapely geometry or None
    """
    fr = np.array(frame_poly.exterior.coords)
    n_fr = len(fr) - 1  # last == first in ring

    # Find nearest frame boundary vertices to arc endpoints
    def nearest_frame_idx(px: float, py: float) -> int:
        return int(np.argmin((fr[:n_fr, 0] - px)**2 + (fr[:n_fr, 1] - py)**2))

    idx_start = nearest_frame_idx(arc_x[0], arc_y[0])
    idx_end = nearest_frame_idx(arc_x[-1], arc_y[-1])

    # Walk frame from arc end back to arc start, going through pole
    def walk_boundary(i_from: int, i_to: int, step: int) -> list[int]:
        pts = []
        i = i_from % n_fr
        target = i_to % n_fr
        for _ in range(n_fr + 1):
            pts.append(i)
            if i == target:
                break
            i = (i + step) % n_fr
        return pts

    walk_pos = walk_boundary(idx_end, idx_start, +1)
    walk_neg = walk_boundary(idx_end, idx_start, -1)

    # Correct walk goes through the pole side
    mean_y_pos = np.mean(fr[walk_pos, 1])
    mean_y_neg = np.mean(fr[walk_neg, 1])

    if pole_side == 'north':
        bwalk = walk_pos if mean_y_pos > mean_y_neg else walk_neg
    else:
        bwalk = walk_pos if mean_y_pos < mean_y_neg else walk_neg

    # Build closed polygon: arc + boundary walk
    final_x = np.concatenate([arc_x, fr[bwalk, 0]])
    final_y = np.concatenate([arc_y, fr[bwalk, 1]])

    try:
        poly = Polygon(zip(final_x, final_y))
        if not poly.is_valid:
            poly = make_valid(poly)
        clipped = _safe_intersection(poly, frame_poly)
        return clipped if not clipped.is_empty else None
    except Exception:
        return None


# ============================================================
# CASE 0: Pixel-space compact fast-path
# ============================================================


# ===== Pixel-jump detection + splitting =====

def _detect_pixel_jumps_circular(
    xv: np.ndarray, yv: np.ndarray, frame_poly: Any,
) -> tuple[np.ndarray, float]:
    """
    Detect large discontinuities in pixel-space coordinates,
    treating the coordinate arrays as circular (last→first is also checked).

    Uses an adaptive threshold: 10× the median normal spacing between
    consecutive points. This correctly detects boundary jumps even at
    high latitudes where projection compression reduces the jump size.

    Returns
    -------
    jump_indices : ndarray
        Indices where jumps occur (between i and i+1).
        Index len(xv)-1 means the jump is between last and first element.
    threshold : float
        The threshold used (for diagnostics).
    """
    # Circular diff: append first point so last→first is included
    dx = np.diff(np.append(xv, xv[0]))
    dy = np.diff(np.append(yv, yv[0]))
    dists = np.sqrt(dx**2 + dy**2)

    # Adaptive threshold: use the 90th percentile of distances as
    # "normal" reference (robust against the jumps inflating the median)
    normal_dists = dists[dists <= np.percentile(dists, 90)]
    if len(normal_dists) == 0:
        normal_dists = dists

    median_normal = np.median(normal_dists)

    # Jump = 10× normal spacing, with floor (20 px) and ceiling (50% frame)
    frame_ring = np.array(frame_poly.exterior.coords)
    frame_extent = max(
        frame_ring[:, 0].max() - frame_ring[:, 0].min(),
        frame_ring[:, 1].max() - frame_ring[:, 1].min(),
    )
    threshold = np.clip(10 * median_normal, 20, 0.5 * frame_extent)

    jump_indices = np.where(dists > threshold)[0]

    # --- Gap filter ---
    # Densely-sampled polygons (1000+ points) can have "distortion
    # stretching" near the frame boundary that produces distances barely
    # above the threshold floor.  Real boundary crossings produce much
    # larger jumps.  If there's a clear gap (factor of 5×) between the
    # largest and smallest candidate jump distances, keep only the group
    # near the largest values — those are the real boundary crossings.
    if len(jump_indices) > 2:
        cand_dists = dists[jump_indices]
        d_max, d_min = cand_dists.max(), cand_dists.min()
        if d_max > 5 * d_min:
            # Clear gap: keep only candidates > max/3
            keep_mask = cand_dists > d_max / 3
            jump_indices = jump_indices[keep_mask]

    return jump_indices, threshold


def _roll_to_expose_jumps(
    xv: np.ndarray, yv: np.ndarray, jump_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Roll pixel coordinate arrays so both jump points fall in the
    interior of the array (not at the start/end boundary).

    Finds the midpoint of the longest gap between jumps and rolls
    the array there, then recomputes jump indices with a linear diff.

    Parameters
    ----------
    xv, yv : ndarray
        Pixel coordinates.
    jump_indices : ndarray
        Circular jump indices (from _detect_pixel_jumps_circular).

    Returns
    -------
    xv_rolled, yv_rolled : ndarray
        Rolled coordinate arrays.
    new_jump_indices : ndarray
        Jump indices in the rolled (linear, non-circular) array.
    """
    n = len(xv)

    if len(jump_indices) < 2:
        return xv, yv, jump_indices

    # Find the longest gap between consecutive jump indices (circular)
    sorted_jumps = np.sort(jump_indices)
    gaps = np.diff(sorted_jumps)
    wrap_gap = n - sorted_jumps[-1] + sorted_jumps[0]
    all_gaps = np.append(gaps, wrap_gap)

    longest_gap_idx = np.argmax(all_gaps)

    if longest_gap_idx < len(gaps):
        start_of_gap = sorted_jumps[longest_gap_idx] + 1
        gap_len = int(all_gaps[longest_gap_idx])
    else:
        start_of_gap = (sorted_jumps[-1] + 1) % n
        gap_len = int(wrap_gap)

    # Roll to the midpoint of the longest gap
    roll_to = (start_of_gap + gap_len // 2) % n

    xv_rolled = np.roll(xv, -roll_to)
    yv_rolled = np.roll(yv, -roll_to)

    # Recompute jump indices in the rolled array (linear diff only)
    dx = np.diff(xv_rolled)
    dy = np.diff(yv_rolled)
    dists = np.sqrt(dx**2 + dy**2)

    # Find the top-N jumps (same count as input) by magnitude
    n_jumps = len(jump_indices)
    if n_jumps <= len(dists):
        top_n = np.argsort(dists)[-n_jumps:]
        new_jumps = np.sort(top_n)
    else:
        new_jumps = np.array([], dtype=int)

    return xv_rolled, yv_rolled, new_jumps


def _compact_pixel_split(
    xv: np.ndarray, yv: np.ndarray, jump_indices: np.ndarray, frame_poly: Any,
) -> list[Any]:
    """
    Handle a compact boundary-crossing shape (2 pixel jumps) by stitching
    the two segments together via short frame boundary walks.

    For a shape crossing the projection boundary once (2 jumps = enter + exit),
    this reconnects the two halves:

        segA → short boundary walk → segB → short boundary walk → segA

    The short walks at each jump bridge the projection-boundary gap via
    the shorter path along the frame boundary.  The resulting single closed
    polygon is validated and clipped to frame.

    This stitch approach avoids the closure-direction ambiguity that plagued
    the previous independent-segment closure strategy, which used a same-side
    heuristic that failed at certain projection centers (e.g. center=180°).

    Parameters
    ----------
    xv, yv : ndarray
        Pixel coordinates (already rolled so jumps are in interior).
    jump_indices : ndarray of length 2
        Where the discontinuities fall in the rolled arrays.
    frame_poly : shapely Polygon
        Frame boundary for clipping and closure.

    Returns
    -------
    list of matplotlib Path objects, or empty list on failure.
    """
    if len(jump_indices) != 2:
        return []

    j1, j2 = int(jump_indices[0]), int(jump_indices[1])

    # Segment A: indices [j1+1 ... j2] (between the two jumps)
    seg_a_x = xv[j1 + 1 : j2 + 1]
    seg_a_y = yv[j1 + 1 : j2 + 1]

    # Segment B: indices [j2+1 ... end, 0 ... j1] (wraps around)
    seg_b_x = np.concatenate([xv[j2 + 1 :], xv[: j1 + 1]])
    seg_b_y = np.concatenate([yv[j2 + 1 :], yv[: j1 + 1]])

    if len(seg_a_x) < 2 or len(seg_b_x) < 2:
        return []

    # Frame boundary ring
    fr = np.array(frame_poly.exterior.coords)
    n_fr = len(fr) - 1  # last == first

    def nearest_frame_idx(px: float, py: float) -> int:
        return int(np.argmin((fr[:n_fr, 0] - px)**2 +
                              (fr[:n_fr, 1] - py)**2))

    def boundary_walk_both(i_from: int, i_to: int) -> list[list[int]]:
        """Return both CW and CCW boundary walks."""
        results = []
        for step in [+1, -1]:
            pts = []
            i = i_from % n_fr
            target = i_to % n_fr
            for _ in range(n_fr + 1):
                pts.append(i)
                if i == target:
                    break
                i = (i + step) % n_fr
            results.append(pts)
        return results  # [cw_walk, ccw_walk]

    # Stitch: segA → walk1 → segB → walk2 → segA
    fi_a_end = nearest_frame_idx(seg_a_x[-1], seg_a_y[-1])
    fi_b_start = nearest_frame_idx(seg_b_x[0], seg_b_y[0])
    walks1 = boundary_walk_both(fi_a_end, fi_b_start)

    fi_b_end = nearest_frame_idx(seg_b_x[-1], seg_b_y[-1])
    fi_a_start = nearest_frame_idx(seg_a_x[0], seg_a_y[0])
    walks2 = boundary_walk_both(fi_b_end, fi_a_start)

    # Combined segment centroid for validation
    all_seg_x = np.concatenate([seg_a_x, seg_b_x])
    all_seg_y = np.concatenate([seg_a_y, seg_b_y])
    centroid = Point(np.mean(all_seg_x), np.mean(all_seg_y))

    # Try all 4 walk combinations, pick the one containing the centroid
    best_result = None
    for w1 in walks1:
        for w2 in walks2:
            parts_x = [seg_a_x]
            parts_y = [seg_a_y]
            if len(w1) > 1:
                parts_x.append(fr[w1[1:], 0])
                parts_y.append(fr[w1[1:], 1])
            parts_x.append(seg_b_x)
            parts_y.append(seg_b_y)
            if len(w2) > 1:
                parts_x.append(fr[w2[1:], 0])
                parts_y.append(fr[w2[1:], 1])

            sx = np.concatenate(parts_x)
            sy = np.concatenate(parts_y)
            if len(sx) < 3:
                continue

            try:
                poly = Polygon(zip(sx, sy))
                if not poly.is_valid:
                    poly = make_valid(poly)
                clipped = _safe_intersection(poly, frame_poly)
                if clipped.is_empty:
                    continue

                if clipped.contains(centroid):
                    # Plausibility: if the centroid-containing result covers
                    # >50% of the frame, it's likely the complement (the
                    # combined centroid of a narrow wrapping strip falls at
                    # the frame center, inside the complement).  Track it
                    # but keep looking for a smaller candidate.
                    if clipped.area < 0.5 * frame_poly.area:
                        return _shapely_to_paths(clipped)

                # Track smallest candidate as fallback
                if best_result is None or clipped.area < best_result.area:
                    best_result = clipped
            except Exception:
                continue

    # Fallback: return smallest candidate (avoids complement)
    if best_result is not None and not best_result.is_empty:
        # If the smallest candidate is still >50% of frame, the stitch
        # approach produced the complement for all walk combinations.
        # Fall back to closing each segment independently using the
        # same-side heuristic, which works correctly for narrow wrapping
        # strips where the centroid-based approaches fail.
        if best_result.area > 0.5 * frame_poly.area:
            all_geoms = []
            for seg_x, seg_y in [(seg_a_x, seg_a_y), (seg_b_x, seg_b_y)]:
                if len(seg_x) < 3:
                    continue
                result = _close_pixel_segment(
                    seg_x, seg_y, frame_poly, pole_side=None)
                if result is not None and not result.is_empty:
                    # Also apply area plausibility per-segment
                    if result.area < 0.5 * frame_poly.area:
                        all_geoms.append(result)
            if all_geoms:
                return _union_and_clip(all_geoms, frame_poly)
        return _shapely_to_paths(best_result)

    return []


def _multi_jump_pixel_split(
    xv: np.ndarray, yv: np.ndarray, jump_indices: np.ndarray, frame_poly: Any,
) -> list[Any]:
    """
    Handle N≥2 pixel-space discontinuities by detecting reciprocal jump
    pairs and splitting the polygon into a main body and loop(s).

    Cross-frame latitude bands often produce "reciprocal" jump pairs where
    the polygon visits a distant part of the frame and returns to the same
    pixel location.  These aren't real boundary crossings — the polygon is
    continuous through them.

    .. note::
       The primary use case for >2-jump polygons (wide cross-frame bands)
       is now handled at a higher level by ``add_frame_band``'s latitude
       chunking strategy, which avoids producing >2-jump polygons in the
       first place.  This function remains as a fallback for cases where
       ``_project_shape`` is called directly with pre-transformed
       coordinates.  It may be safe to remove in a future version if all
       cross-frame rendering goes through ``add_frame_band``.

    Algorithm:
    1. Detect reciprocal pairs (jumps whose from/to endpoints match in
       reverse within tolerance).
    2. For each reciprocal pair (A, B where A < B):
       - **Main body** (B+1 → end → 0 → A): continuous, no real jumps
         between them → handle as 0-jump polygon (direct clip).
       - **Loop** (A+1 → B): may contain real jumps → split at those
         real jumps and close each piece via centroid closure.
    3. If no reciprocal pairs found, fall back to splitting at all jumps
       and closing each segment independently.

    Parameters
    ----------
    xv, yv : ndarray
        Pixel coordinates (already rolled so all jumps are in interior).
    jump_indices : ndarray
        Sorted jump indices in the rolled arrays (length ≥ 2).
    frame_poly : shapely Polygon
        Frame boundary for clipping and closure.

    Returns
    -------
    list of matplotlib Path objects, or empty list on failure.
    """
    n_jumps = len(jump_indices)
    if n_jumps < 2:
        return []

    jumps = sorted(int(j) for j in jump_indices)
    n = len(xv)

    # ---- Step 1: Detect reciprocal jump pairs ----
    tol = 25.0
    reciprocal_pairs = []  # list of (idx_in_jumps_A, idx_in_jumps_B)
    used = set()

    for i in range(n_jumps):
        if i in used:
            continue
        ji = jumps[i]
        ni = (ji + 1) % n
        for j in range(i + 1, n_jumps):
            if j in used:
                continue
            jj = jumps[j]
            nj = (jj + 1) % n
            d1 = np.sqrt((xv[ji] - xv[nj])**2 + (yv[ji] - yv[nj])**2)
            d2 = np.sqrt((xv[ni] - xv[jj])**2 + (yv[ni] - yv[jj])**2)
            if d1 < tol and d2 < tol:
                reciprocal_pairs.append((i, j))
                used.add(i)
                used.add(j)
                break

    # ---- Step 2: Handle reciprocal pairs ----
    if reciprocal_pairs:
        all_paths = []

        for pair_i, pair_j in reciprocal_pairs:
            jA = jumps[pair_i]   # earlier reciprocal jump index
            jB = jumps[pair_j]   # later reciprocal jump index

            # Main body: (jB+1) → end → 0 → jA  (continuous, no real jumps)
            if jB + 1 < n:
                body_x = np.concatenate([xv[jB + 1:], xv[:jA + 1]])
                body_y = np.concatenate([yv[jB + 1:], yv[:jA + 1]])
            else:
                body_x = xv[:jA + 1]
                body_y = yv[:jA + 1]

            if len(body_x) >= 3:
                poly = Polygon(zip(body_x, body_y))
                if not poly.is_valid:
                    poly = make_valid(poly)
                clipped = _safe_intersection(poly, frame_poly)
                if not clipped.is_empty:
                    all_paths.extend(_shapely_to_paths(clipped))

            # Loop: (jA+1) → jB  (may contain real jumps)
            loop_x = xv[jA + 1: jB + 1]
            loop_y = yv[jA + 1: jB + 1]

            # Find real jumps inside the loop (indices relative to loop)
            real_in_loop = []
            for k in range(n_jumps):
                if k in used:
                    continue  # skip reciprocal jumps
                jk = jumps[k]
                if jA < jk < jB:
                    real_in_loop.append(jk - (jA + 1))  # relative to loop start

            if len(loop_x) >= 3:
                if len(real_in_loop) == 0:
                    # No real jumps in loop → direct clip
                    poly = Polygon(zip(loop_x, loop_y))
                    if not poly.is_valid:
                        poly = make_valid(poly)
                    clipped = _safe_intersection(poly, frame_poly)
                    if not clipped.is_empty:
                        all_paths.extend(_shapely_to_paths(clipped))
                elif len(real_in_loop) == 1:
                    # 1 real jump → roll at it and centroid closure
                    rj = real_in_loop[0]
                    seg_x = np.roll(loop_x, -(rj + 1))
                    seg_y = np.roll(loop_y, -(rj + 1))
                    result = _close_segment_centroid(seg_x, seg_y, frame_poly)
                    if result is not None and not result.is_empty:
                        all_paths.extend(_shapely_to_paths(result))
                else:
                    # Multiple real jumps in loop → split and close each
                    rjs = sorted(real_in_loop)
                    for si in range(len(rjs)):
                        start = rjs[si] + 1
                        end = rjs[(si + 1) % len(rjs)]
                        if si < len(rjs) - 1:
                            seg_x = loop_x[start:end + 1]
                            seg_y = loop_y[start:end + 1]
                        else:
                            seg_x = np.concatenate([loop_x[start:],
                                                     loop_x[:end + 1]])
                            seg_y = np.concatenate([loop_y[start:],
                                                     loop_y[:end + 1]])
                        if len(seg_x) >= 3:
                            result = _close_segment_centroid(
                                seg_x, seg_y, frame_poly)
                            if result is not None and not result.is_empty:
                                all_paths.extend(_shapely_to_paths(result))

        if all_paths:
            return all_paths

    # ---- Step 3: No reciprocal pairs — general fallback ----
    # Split at all jumps, close each segment via centroid closure.
    segments = []
    for i in range(n_jumps):
        start = jumps[i] + 1
        end = jumps[(i + 1) % n_jumps]
        if i < n_jumps - 1:
            seg_x = xv[start:end + 1]
            seg_y = yv[start:end + 1]
        else:
            if start < n:
                seg_x = np.concatenate([xv[start:], xv[:end + 1]])
                seg_y = np.concatenate([yv[start:], yv[:end + 1]])
            else:
                seg_x = xv[:end + 1]
                seg_y = yv[:end + 1]
        if len(seg_x) >= 3:
            segments.append((seg_x, seg_y))

    all_paths = []
    for seg_x, seg_y in segments:
        result = _close_segment_centroid(seg_x, seg_y, frame_poly)
        if result is not None and not result.is_empty:
            all_paths.extend(_shapely_to_paths(result))
    return all_paths


# ============================================================
# Main projection function
# ============================================================


# ===== Main project pipeline =====

def _project_shape(
    ax: Any, lons: npt.ArrayLike, lats: npt.ArrayLike,
    lat_center: float | None = None, radius_deg: float | None = None,
) -> list[Any]:
    """
    Project spherical shape to matplotlib paths.

    Decision tree:
    1. Pole-covering (boundary walk approach):
       a. 0 jumps → arc-split + boundary walk (smooth pole cap)
       b. 1 jump  → roll at jump, close open segment via boundary walk
       c. 2 jumps → split, close pole-side segments via boundary walk
    0. Compact boundary crossing (2 pixel jumps, non-pole) → pixel split + roll
    0m. Multi-jump boundary crossing (>2 jumps, non-pole) → multi pixel split
    0b. Single boundary crossing (1 jump, non-pole) → centroid-aware closure
    3. Non-pole + world-coord crossing → world split (fallback)
    4. Non-pole + self-intersecting → pixel split at intersections
    5. Simple valid polygon → direct clip
    """
    if not SHAPELY_AVAILABLE:
        return _simple_project(ax, lons, lats)

    wcs = ax.wcs
    lon_center = _get_projection_center(ax)
    frame_poly = _get_frame_polygon(ax)

    # Determine pole coverage
    pole_side = None
    if lat_center is not None and radius_deg is not None:
        if lat_center + radius_deg >= 90:
            pole_side = 'north'
        elif lat_center - radius_deg <= -90:
            pole_side = 'south'

    # Project to pixel coordinates
    x, y = wcs.world_to_pixel_values(lons, lats)
    finite = np.isfinite(x) & np.isfinite(y)
    # Tight-FOV projections (TAN/SIN/etc. on a small cdelt) can map
    # off-FOV input samples to *finite-but-absurd* pixel coords
    # (millions of pixels from the frame). Including such vertices
    # produces polygons whose edges sweep across the entire visible
    # region. Filter to a generous bbox around the frame so true
    # off-FOV samples are
    # dropped without affecting AIT/MOL/SIN allsky cases (where
    # finite projections always land within frame bounds).
    fxmin, fymin, fxmax, fymax = frame_poly.bounds
    fw = fxmax - fxmin
    fh = fymax - fymin
    margin_w = 10.0 * fw
    margin_h = 10.0 * fh
    in_bbox = (
        (x >= fxmin - margin_w) & (x <= fxmax + margin_w) &
        (y >= fymin - margin_h) & (y <= fymax + margin_h)
    )
    valid = finite & in_bbox
    if np.sum(valid) < 3:
        return []
    xv, yv = x[valid], y[valid]

    # Detect pixel-space discontinuities (needed for multiple cases)
    jump_indices, _threshold = _detect_pixel_jumps_circular(xv, yv, frame_poly)
    n_jumps = len(jump_indices)

    # ---- CASE 1: Pole-covering → boundary walk approach ----
    if pole_side is not None:
        if n_jumps == 0:
            # Case 1a: smooth ring, no boundary crossing
            result = _extract_pole_cap_smooth(xv, yv, pole_side, frame_poly)
            if result is not None:
                return _shapely_to_paths(result)

        elif n_jumps == 1:
            # Case 1b: one boundary crossing (e.g., lon=180, center=0)
            # Roll at the jump → single open segment → close through pole
            j = int(jump_indices[0])
            seg_x = np.roll(xv, -(j + 1))
            seg_y = np.roll(yv, -(j + 1))
            result = _close_open_segment_via_boundary(
                seg_x, seg_y, pole_side, frame_poly)
            if result is not None:
                return _shapely_to_paths(result)

        elif n_jumps == 2:
            # Case 1c: two boundary crossings + pole
            # Split at jumps; close pole-side via boundary walk, other via frame
            xv_r, yv_r, new_jumps = _roll_to_expose_jumps(xv, yv, jump_indices)
            if len(new_jumps) == 2:
                j1, j2 = int(new_jumps[0]), int(new_jumps[1])
                seg_a_x = xv_r[j1 + 1 : j2 + 1]
                seg_a_y = yv_r[j1 + 1 : j2 + 1]
                seg_b_x = np.concatenate([xv_r[j2 + 1 :], xv_r[: j1 + 1]])
                seg_b_y = np.concatenate([yv_r[j2 + 1 :], yv_r[: j1 + 1]])

                fr = np.array(frame_poly.exterior.coords)
                fc_y = np.mean(fr[:, 1])
                all_geoms = []
                for seg_x, seg_y in [(seg_a_x, seg_a_y), (seg_b_x, seg_b_y)]:
                    if len(seg_x) < 3:
                        continue
                    is_pole_seg = (
                        (pole_side == 'north' and np.mean(seg_y) > fc_y) or
                        (pole_side == 'south' and np.mean(seg_y) < fc_y))
                    if is_pole_seg:
                        result = _close_open_segment_via_boundary(
                            seg_x, seg_y, pole_side, frame_poly)
                    else:
                        result = _close_pixel_segment(
                            seg_x, seg_y, frame_poly, pole_side=None)
                    if result is not None and not result.is_empty:
                        all_geoms.append(result)
                paths = _union_and_clip(all_geoms, frame_poly)
                if paths:
                    return paths

        # Fallback for pole cases that didn't produce results above
        result = _extract_pole_cap_smooth(xv, yv, pole_side, frame_poly)
        if result is not None:
            return _shapely_to_paths(result)

    # ---- CASE 0: Compact boundary crossing → pixel split + roll ----
    if n_jumps == 2 and pole_side is None:
        xv_r, yv_r, new_jumps = _roll_to_expose_jumps(xv, yv, jump_indices)
        if len(new_jumps) == 2:
            result = _compact_pixel_split(xv_r, yv_r, new_jumps, frame_poly)
            if result:
                return result

    # ---- CASE 0m: Multi-jump boundary crossing (>2 jumps, non-pole) ----
    # Cross-frame latitude bands (e.g. wide ecliptic bands in equatorial
    # frame) can produce many pixel jumps when the transformed polygon
    # boundary crosses the projection boundary repeatedly.  Roll to expose
    # all jumps, then split at every jump and close each piece via
    # centroid-based boundary walk.
    if n_jumps > 2 and pole_side is None:
        xv_r, yv_r, new_jumps = _roll_to_expose_jumps(xv, yv, jump_indices)
        if len(new_jumps) >= 2:
            result = _multi_jump_pixel_split(xv_r, yv_r, new_jumps, frame_poly)
            if result:
                return result

    # ---- Fast-path: 0 pixel jumps → polygon is continuous, skip to clip ----
    # Pixel continuity is the ground truth.  If the projected coordinates
    # have no large gaps, the polygon doesn't need splitting regardless of
    # what the world-coordinate crossing detector thinks.  This handles
    # full-sky latitude bands whose edges touch (but don't cross) the
    # projection boundary.
    if n_jumps == 0 and pole_side is None:
        poly = Polygon(zip(xv, yv))
        if not poly.is_valid:
            poly = make_valid(poly)
        clipped = _safe_intersection(poly, frame_poly)
        return _shapely_to_paths(clipped) if not clipped.is_empty else []

    # ---- CASE 0b: Single boundary crossing (1 jump, non-pole) ----
    # Polygons spanning >180° longitude cross the boundary once, producing
    # a single pixel jump.  Roll at the jump to create an open segment,
    # then close via centroid-aware boundary walk.
    if n_jumps == 1 and pole_side is None:
        # ── False-positive jump filter ──
        # The boundary-walk closure is correct ONLY when the polygon
        # actually spans > 180° in longitude — i.e. its area covers
        # more than half the globe in lon. At high latitude the
        # AIT/MOL/etc. projections have a WCS singularity at center±180°
        # — same physical point, two different pixel positions — that
        # ``_detect_pixel_jumps_circular`` flags as a "jump." For such
        # false positives the segment endpoints sit deep inside the
        # frame, and walking the limb between them gobbles up a wedge
        # of the projection.
        #
        # Discriminator: circular lon coverage (360° minus the largest
        # gap between consecutive sorted vertex longitudes). A small
        # high-lat tile straddling the singularity covers ≪ 180° in
        # lon; a true >180° wrap polygon covers > 180°.
        rel_lons = ((np.asarray(lons, float) - lon_center + 180) % 360) - 180
        finite_rel = rel_lons[np.isfinite(rel_lons)]
        if len(finite_rel) >= 2:
            sorted_rel = np.sort(finite_rel)
            gaps = np.diff(sorted_rel)
            wrap_gap = (sorted_rel[0] + 360) - sorted_rel[-1]
            largest_gap = max(gaps.max(), wrap_gap)
            lon_coverage = 360.0 - largest_gap
        else:
            lon_coverage = 0.0
        if lon_coverage > 180.0:
            j = int(jump_indices[0])
            seg_x = np.roll(xv, -(j + 1))
            seg_y = np.roll(yv, -(j + 1))
            result = _close_segment_centroid(seg_x, seg_y, frame_poly)
            if result is not None and not result.is_empty:
                return _shapely_to_paths(result)
        # Otherwise, fall through to CASE 5 (simple clip) — the jump
        # is a high-lat WCS singularity, not a real wrap.

    # ---- CASE 3: Non-pole + world-coord crossing → world split (fallback) ----
    crossing_indices, crossing_lats = _detect_world_crossings(lons, lats, lon_center)

    if len(crossing_indices) > 0:
        segments = _split_world_coords(lons, lats, crossing_indices,
                                        crossing_lats, lon_center)
        all_geoms = []
        for seg_lons, seg_lats in segments:
            c_lons, c_lats = _close_world_segment(
                seg_lons, seg_lats, lon_center, pole_side=None)
            sx, sy = wcs.world_to_pixel_values(c_lons, c_lats)
            sv = np.isfinite(sx) & np.isfinite(sy)
            if np.sum(sv) < 3:
                continue
            try:
                poly = Polygon(zip(sx[sv], sy[sv]))
                if not poly.is_valid:
                    poly = make_valid(poly)
                clipped = _safe_intersection(poly, frame_poly)
                if not clipped.is_empty:
                    all_geoms.append(clipped)
            except Exception:
                continue
        return _union_and_clip(all_geoms, frame_poly) if all_geoms else []

    # ---- CASE 4: Non-pole + self-intersecting → pixel split ----
    intersections = _find_self_intersections(xv, yv)
    if len(intersections) > 0:
        split_idxs = [ipt[2] for ipt in intersections]
        px_segments = _split_ring_at_indices(xv, yv, split_idxs)
        all_geoms = []
        for seg_x, seg_y in px_segments:
            result = _close_pixel_segment(seg_x, seg_y, frame_poly)
            if result is not None and not result.is_empty:
                all_geoms.append(result)
        return _union_and_clip(all_geoms, frame_poly) if all_geoms else []

    # ---- CASE 5: Simple valid polygon → direct clip ----
    poly = Polygon(zip(xv, yv))
    if not poly.is_valid:
        poly = make_valid(poly)
    clipped = _safe_intersection(poly, frame_poly)
    return _shapely_to_paths(clipped) if not clipped.is_empty else []


def _simple_project(ax: Any, lons: npt.ArrayLike, lats: npt.ArrayLike) -> list[Path]:
    """Fallback without shapely."""
    wcs = ax.wcs
    x, y = wcs.world_to_pixel_values(lons, lats)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return []
    xv, yv = x[valid], y[valid]
    verts = np.column_stack([xv, yv])
    verts = np.vstack([verts, verts[0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return [Path(verts, codes)]


# ============================================================
# Shapely → matplotlib conversion
# ============================================================


# ===== Complement renderer =====

def _render_complement(
    ax: Any, paths: list[Any], frame_poly: Any, **kwargs: Any,
) -> list[Any]:
    """
    Render the complement of a projected shape (everything except the shape).

    Takes the already-projected paths from ``_project_shape``, converts
    to a shapely geometry, computes ``frame_poly - shape``, and renders
    the difference.

    Edge rendering is separated: fill is drawn without edges, then the
    shape boundary (not the frame boundary) is overlaid as line plots.

    Parameters
    ----------
    ax : WCSAxes
    paths : list of matplotlib Path
        Projected shape paths from ``_project_shape``.
    frame_poly : shapely Polygon
        Frame boundary.
    **kwargs
        Styling kwargs (facecolor, edgecolor, alpha, linewidth, etc.).

    Returns
    -------
    list of PathPatch
        The complement fill patches added to the axes.
    """
    # Compute complement geometry
    if paths:
        shape_geom = _paths_to_geom(paths)
        if shape_geom is not None and not shape_geom.is_empty:
            try:
                comp_geom = frame_poly.difference(shape_geom)
            except Exception:
                comp_geom = frame_poly
        else:
            comp_geom = frame_poly
    else:
        comp_geom = frame_poly

    if comp_geom.is_empty:
        return []

    # Separate fill from edges
    edgecolor = kwargs.pop('edgecolor', kwargs.pop('ec', 'none'))
    linewidth = kwargs.pop('linewidth', kwargs.pop('lw', 1.0))
    linestyle = kwargs.pop('linestyle', kwargs.pop('ls', '-'))

    # Fill complement without edges
    fill_kwargs = dict(kwargs)
    fill_kwargs['edgecolor'] = 'none'
    fill_kwargs['linewidth'] = 0

    comp_paths = _shapely_to_paths(comp_geom)
    patches = []
    for path in comp_paths:
        patch = PathPatch(path, **fill_kwargs)
        ax.add_patch(patch)
        patches.append(patch)

    # Edge lines on the SHAPE boundary (not the frame boundary)
    if edgecolor not in ('none', None) and paths:
        xlim = ax.get_xlim()
        gap_thresh = 0.1 * (xlim[1] - xlim[0])

        for sp in paths:
            verts = sp.vertices
            dx = np.abs(np.diff(verts[:, 0]))
            dy = np.abs(np.diff(verts[:, 1]))
            dists = np.sqrt(dx**2 + dy**2)

            gaps = np.where(dists > gap_thresh)[0]
            boundaries = [0] + (gaps + 1).tolist() + [len(verts)]
            for i in range(len(boundaries) - 1):
                s, e = boundaries[i], boundaries[i + 1]
                seg = verts[s:e]
                if len(seg) >= 2:
                    ax.plot(seg[:, 0], seg[:, 1], color=edgecolor,
                            linewidth=linewidth, linestyle=linestyle,
                            transform=ax.get_transform('pixel'),
                            zorder=kwargs.get('zorder', 5))

    return patches

