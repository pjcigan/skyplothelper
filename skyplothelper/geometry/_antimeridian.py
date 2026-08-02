"""D3-style antimeridian clipping and segment-stitching.

``_antimeridian_clip`` does Sutherland-Hodgman-style spherical polygon
clipping; ``_stitch_and_project`` reassembles split segments after
projection through the WCS.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.path import Path

try:
    from shapely.geometry import LineString, Point, Polygon  # noqa: F401
    from shapely.ops import polygonize, unary_union  # noqa: F401
    from shapely.validation import make_valid  # noqa: F401
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

from ._frame_geom import (  # noqa: F401
    _complement_detect,
    _get_projection_center,
    _paths_to_geom,
    _safe_intersection,
    _shapely_to_paths,
)
from ._projection import _project_shape  # noqa: F401


def _antimeridian_clip(lons: npt.ArrayLike, lats: npt.ArrayLike,
                       lon_center: float) -> list[dict[str, Any]]:
    """
    Clip a closed spherical polygon against the antimeridian.

    Inspired by D3's ``clipAntimeridian``: walks the polygon edge by edge,
    detects antimeridian crossings (``|delta_lon| > 180``), and splits into
    segments that each lie entirely within ``[-180, 180]`` of the projection
    center.  Boundary intersection points (at ``lon_center +/- 180``) are
    inserted at each crossing.

    Parameters
    ----------
    lons, lats : ndarray
        Polygon vertices in degrees (closed: first == last).
    lon_center : float
        Projection center longitude.

    Returns
    -------
    list of dict
        Each segment has ``lons``, ``lats`` (ndarray), ``entry_lat``,
        ``exit_lat`` (float or None).
    """
    lon_norm = ((np.asarray(lons, float) - lon_center + 180) % 360) - 180
    lats = np.asarray(lats, float)
    n = len(lon_norm)
    if n < 3:
        return []

    if not (np.isclose(lon_norm[0], lon_norm[-1]) and np.isclose(lats[0], lats[-1])):
        lon_norm = np.append(lon_norm, lon_norm[0])
        lats = np.append(lats, lats[0])
        n = len(lon_norm)

    # Disambiguate vertices exactly at the antimeridian (|lon_norm| == 180).
    # Python's `%` always returns the negative side, so a vertex at
    # `lon = lon_center + 180` becomes `lon_norm = -180` even when the
    # polygon's other vertices sit firmly in the positive-lon_norm half.
    # Without disambiguation, a polygon edge from lon_norm=170 to a vertex
    # AT lon=180 reads as dlon=-350 — flagged as a false antimeridian
    # crossing that splits the polygon into spurious halves. Choose the
    # sign that matches the (non-antimeridian) neighbour so the dlon to
    # adjacent vertices stays small.
    eps = 1e-6
    at_anti = np.abs(np.abs(lon_norm) - 180.0) < eps
    if np.any(at_anti):
        non_anti_signs = np.where(at_anti, 0.0, np.sign(lon_norm))
        # Loop a few times so chains of antimeridian vertices propagate
        # the sign from the nearest non-antimeridian neighbour.
        for _ in range(n):
            for i in range(n):
                if at_anti[i]:
                    # Look at neighbours (cyclic) for a nonzero sign
                    prev_i = (i - 1) % (n - 1)
                    next_i = (i + 1) % (n - 1)
                    prev_sign = non_anti_signs[prev_i]
                    next_sign = non_anti_signs[next_i]
                    if prev_sign != 0:
                        lon_norm[i] = prev_sign * 180.0
                        non_anti_signs[i] = prev_sign
                        at_anti[i] = False
                    elif next_sign != 0:
                        lon_norm[i] = next_sign * 180.0
                        non_anti_signs[i] = next_sign
                        at_anti[i] = False

    dlon = np.diff(lon_norm)
    crossing_mask = np.abs(dlon) > 180
    crossing_indices = np.where(crossing_mask)[0]

    if len(crossing_indices) == 0:
        return [{'lons': lon_norm + lon_center, 'lats': lats.copy(),
                 'entry_lat': None, 'exit_lat': None}]

    def _crossing_info(idx: int) -> tuple[Any, int]:
        la, lb = lon_norm[idx], lon_norm[idx + 1]
        lat_a, lat_b = lats[idx], lats[idx + 1]
        if la > 0:
            boundary = 180.0
            lb_unwrap = lb + 360
        else:
            boundary = -180.0
            lb_unwrap = lb - 360
        denom = lb_unwrap - la
        t = (boundary - la) / denom if abs(denom) > 1e-10 else 0.5
        t = np.clip(t, 0, 1)
        lat_cross = lat_a + t * (lat_b - lat_a)
        side = +1 if boundary > 0 else -1
        return lat_cross, side

    n_cross = len(crossing_indices)
    segments = []

    for ci in range(n_cross):
        cross_idx = crossing_indices[ci]
        next_cross_idx = crossing_indices[(ci + 1) % n_cross]

        lat_entry, side_exit_prev = _crossing_info(cross_idx)
        lat_exit, side_exit_this = _crossing_info(next_cross_idx)

        seg_lons = [-side_exit_prev * 180.0]
        seg_lats = [lat_entry]

        i = (cross_idx + 1) % (n - 1)
        target = next_cross_idx % (n - 1)
        count = 0
        while count < n:
            seg_lons.append(lon_norm[i])
            seg_lats.append(lats[i])
            if i == target:
                break
            i = (i + 1) % (n - 1)
            count += 1

        seg_lons.append(side_exit_this * 180.0)
        seg_lats.append(lat_exit)

        segments.append({
            'lons': np.array(seg_lons) + lon_center,
            'lats': np.array(seg_lats),
            'entry_lat': lat_entry,
            'exit_lat': lat_exit,
        })

    return segments


def _stitch_and_project(segments: list[dict[str, Any]], ax: Any,
                        frame_poly: Any, expected_frac: float | None = None,
                        min_piece_area: float = 5.0) -> list[Path]:
    """
    Project clipped segments and stitch them into one polygon.

    D3-style global rejoin: segments from ``_antimeridian_clip`` are in
    polygon order, so they are stitched together with short boundary walks
    at each junction.  This produces ONE closed polygon with no per-segment
    closure ambiguity.

    For the no-crossing case (single segment, no boundary), projects
    directly and applies area plausibility to detect/fix complements.

    Parameters
    ----------
    segments : list of dict
        From ``_antimeridian_clip``.
    ax : WCSAxes
    frame_poly : shapely Polygon
    expected_frac : float or None
        Expected solid-angle fraction for area plausibility.
    min_piece_area : float
        Minimum pixel area a stitched polygon piece must have to be
        kept after the per-side stitch step. The default of 5 pixels²
        filters out zero-area slivers from self-intersecting stitches
        for full-sized regions like survey footprints. Pass a smaller
        value (e.g. 0.1) for callers that render small primitives such
        as individual HEALPix tiles, where each antimeridian-split half
        can be fractions of a pixel² yet must still render.

    Returns
    -------
    list of matplotlib Path objects
    """
    wcs = ax.wcs

    # No-crossing case: project directly
    if len(segments) == 1 and segments[0]['entry_lat'] is None:
        seg = segments[0]
        x, y = wcs.world_to_pixel_values(seg['lons'], seg['lats'])
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 3:
            return []
        poly = Polygon(zip(x[valid], y[valid]))
        if not poly.is_valid:
            poly = make_valid(poly)
        clipped = _safe_intersection(poly, frame_poly)
        if clipped.is_empty:
            return []
        clipped = _complement_detect(clipped, frame_poly, expected_frac)
        return _shapely_to_paths(clipped, min_area=min(1.0, min_piece_area)) if not clipped.is_empty else []

    # Single segment WITH boundary crossing — a near-360° wrap: a pole-enclosing
    # cap, or a large region crossing the antimeridian once. CLOSE it in lon/lat
    # first (walk the wrap meridian for a same-side crossing, or up-and-over the
    # pole for an opposite-side one) and only THEN project. Closing in lon/lat
    # yields a well-formed fill polygon on curved frames (AIT / MOL / globe),
    # where the old project-then-guess path (``_project_shape`` + the complement
    # heuristic) picked the WRONG side for pole caps (empty / complement on MOL).
    # ``_close_clipped_segment`` is shared with the plotly backend, so the two
    # close pole-enclosing polygons identically.
    if len(segments) == 1 and segments[0]['entry_lat'] is not None:
        center = _get_projection_center(ax)
        clons, clats = _close_clipped_segment(segments[0], center)
        # Densify along each wrap meridian so the closing edge traces the curved
        # frame silhouette instead of chording across it.
        for _wl in (center + 180.0, center - 180.0):
            clons, clats = _densify_along_wrap_edge(clons, clats, _wl)
        x, y = wcs.world_to_pixel_values(clons, clats)
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 3:
            return []
        poly = Polygon(zip(x[valid], y[valid]))
        if not poly.is_valid:
            poly = make_valid(poly)
        clipped = _safe_intersection(poly, frame_poly)
        if clipped.is_empty:
            return []
        clipped = _complement_detect(clipped, frame_poly, expected_frac)
        return _shapely_to_paths(clipped, min_area=min(1.0, min_piece_area)) if not clipped.is_empty else []

    # Multi-segment: project each, stitch via boundary walks
    fr = np.array(frame_poly.exterior.coords)
    n_fr = len(fr) - 1

    def nearest_frame_idx(px: float, py: float) -> int:
        return int(np.argmin((fr[:n_fr, 0] - px)**2 + (fr[:n_fr, 1] - py)**2))

    def short_boundary_walk(i_from: int, i_to: int) -> list[int]:
        walk_pos, walk_neg = [], []
        i = i_from % n_fr
        target = i_to % n_fr
        for _ in range(n_fr + 1):
            walk_pos.append(i)
            if i == target:
                break
            i = (i + 1) % n_fr
        i = i_from % n_fr
        for _ in range(n_fr + 1):
            walk_neg.append(i)
            if i == target:
                break
            i = (i - 1) % n_fr
        return walk_pos if len(walk_pos) <= len(walk_neg) else walk_neg

    px_segs: list[dict[str, Any] | None] = []
    for seg in segments:
        x, y = wcs.world_to_pixel_values(seg['lons'], seg['lats'])
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 2:
            px_segs.append(None)
            continue
        px_segs.append({'x': x[valid], 'y': y[valid]})

    n_segs = len(px_segs)

    # Stitch: seg0 -> walk -> seg1 -> walk -> ... -> seg0
    # But first check if walks are "short" (segments are adjacent on the
    # boundary) or "long" (segments are on opposite sides of the frame).
    # Long walks indicate a small polygon crossing the antimeridian —
    # in that case, close each segment independently instead of stitching.

    # Check walk lengths
    max_walk_len = 0
    for si in range(n_segs):
        pseg = px_segs[si]
        pnext = px_segs[(si + 1) % n_segs]
        if pseg is None or pnext is None:
            continue
        fi_end = nearest_frame_idx(pseg['x'][-1], pseg['y'][-1])
        fi_start = nearest_frame_idx(pnext['x'][0], pnext['y'][0])
        if fi_end != fi_start:
            walk = short_boundary_walk(fi_end, fi_start)
            max_walk_len = max(max_walk_len, len(walk))

    use_independent = max_walk_len > n_fr // 3

    # Projection center longitude — needed for projected boundary walks
    lon_center = _get_projection_center(ax)

    if use_independent:
        # Stitch-by-side with projected boundary walks.
        #
        # Groups segments by which frame edge they lie on, then stitches
        # within each group.  Instead of walking discrete frame polygon
        # vertices (which causes chord artifacts), traces the frame edge
        # by projecting points along the antimeridian at known crossing
        # latitudes.
        #
        # For complex shapes with connector folds (zigzag raster scans),
        # make_valid correctly decomposes the self-intersecting stitched
        # polygon into individual strip pieces.

        frame_cx = np.mean(fr[:n_fr, 0])

        # Determine which boundary lon maps to which pixel edge
        left_blon: float | None = lon_center + 180 - 0.001  # → left pixel edge
        right_blon: float | None = lon_center - 180 + 0.001  # → right pixel edge
        test_left = wcs.world_to_pixel_values(left_blon, 0.0)
        test_right = wcs.world_to_pixel_values(right_blon, 0.0)
        if not (np.isfinite(test_left[0]) and np.isfinite(test_right[0])):
            # Fallback: use frame vertex walking (shouldn't happen)
            left_blon, right_blon = None, None
        elif test_left[0] > test_right[0]:
            # Swap if needed so left_blon → lower x
            left_blon, right_blon = right_blon, left_blon

        left_group = []
        right_group = []
        for si in range(n_segs):
            gseg = px_segs[si]
            if gseg is None:
                continue
            if np.mean(gseg['x']) < frame_cx:
                left_group.append(si)
            else:
                right_group.append(si)

        def _projected_boundary_walk(
                lat_from: float, lat_to: float, boundary_lon: float
        ) -> tuple[np.ndarray, np.ndarray]:
            """Project points along the antimeridian to trace the frame edge."""
            n_pts = max(5, int(abs(lat_from - lat_to) * 2))
            walk_lats = np.linspace(lat_from, lat_to, n_pts)
            walk_x, walk_y = wcs.world_to_pixel_values(
                np.full(n_pts, boundary_lon), walk_lats)
            valid = np.isfinite(walk_x) & np.isfinite(walk_y)
            if np.sum(valid) > 0:
                return walk_x[valid], walk_y[valid]
            return np.array([]), np.array([])

        def _stitch_edge_group(group: list[int],
                               boundary_lon: float | None) -> list[Any]:
            """Stitch segments on one frame edge with projected boundary walks."""
            if not group or boundary_lon is None:
                return []

            parts_x, parts_y = [], []
            for gi in range(len(group)):
                si = group[gi]
                seg = px_segs[si]
                if seg is None:
                    continue

                # Anchor: exact frame edge position at entry latitude
                entry_lat = segments[si]['entry_lat']
                ax_pt, ay_pt = wcs.world_to_pixel_values(boundary_lon, entry_lat)
                if np.isfinite(ax_pt) and np.isfinite(ay_pt):
                    parts_x.append(np.array([float(ax_pt)]))
                    parts_y.append(np.array([float(ay_pt)]))

                # Segment pixel path
                parts_x.append(seg['x'])
                parts_y.append(seg['y'])

                # Anchor: exact frame edge position at exit latitude
                exit_lat = segments[si]['exit_lat']
                ex_pt, ey_pt = wcs.world_to_pixel_values(boundary_lon, exit_lat)
                if np.isfinite(ex_pt) and np.isfinite(ey_pt):
                    parts_x.append(np.array([float(ex_pt)]))
                    parts_y.append(np.array([float(ey_pt)]))

                # Projected boundary walk to next segment
                next_si = group[(gi + 1) % len(group)]
                next_entry_lat = segments[next_si]['entry_lat']
                wx, wy = _projected_boundary_walk(exit_lat, next_entry_lat,
                                                   boundary_lon)
                if len(wx) > 0:
                    parts_x.append(wx)
                    parts_y.append(wy)

            all_x = np.concatenate(parts_x)
            all_y = np.concatenate(parts_y)
            # Deduplicate consecutive vertices that coincide (within sub-pixel
            # tolerance). For polar HEALPix tiles where a segment's entry_lat
            # equals its exit_lat, the entry-anchor + exit-anchor + projected
            # boundary walk all collapse to the same point, producing a
            # degenerate self-intersecting polygon that shapely.make_valid
            # turns into a sub-pixel multipolygon. The downstream
            # frame_poly.intersection then clips to empty. Deduping makes the
            # polygon well-formed so it survives the intersection.
            if len(all_x) > 1:
                dx = np.diff(all_x)
                dy = np.diff(all_y)
                keep = np.concatenate(([True], (dx**2 + dy**2) > 1e-10))
                all_x = all_x[keep]
                all_y = all_y[keep]
            if len(all_x) < 3:
                return []
            poly = Polygon(zip(all_x, all_y))
            if not poly.is_valid:
                poly = make_valid(poly)
            clipped = _safe_intersection(poly, frame_poly)
            pieces = _extract_substantive_pieces(clipped, min_area=min_piece_area)
            # Fallback for thin slivers: when a tile is mostly on one side
            # of the antimeridian and the other half is just a narrow
            # strip along the seam, the per-side stitch above produces
            # a degenerate polygon whose segment and boundary-walk both
            # run along the seam in opposite directions — collapsing
            # to ~zero area and getting clipped out. In that case, fall
            # back to projecting each segment's vertices into a simple
            # closed polygon (no per-side anchors / walk) so the thin
            # sliver still renders. The per-side stitching has a
            # directional asymmetry that works on the bulk side but
            # fails on the sliver side.
            if not pieces:
                fallback = []
                for si in group:
                    seg = px_segs[si]
                    if seg is None or len(seg['x']) < 3:
                        continue
                    p = Polygon(zip(seg['x'], seg['y']))
                    if not p.is_valid:
                        p = make_valid(p)
                    p_clip = _safe_intersection(p, frame_poly)
                    if not p_clip.is_empty and p_clip.area > 0:
                        fallback.extend(
                            _extract_substantive_pieces(p_clip, min_area=0.0))
                if fallback:
                    return fallback
            return pieces

        def _extract_substantive_pieces(geom: Any,
                                        min_area: float = 5.0) -> list[Any]:
            """Filter non-trivial polygon pieces (remove zero-area slivers)."""
            pieces = []
            if geom.geom_type == 'Polygon':
                if geom.area > min_area:
                    pieces.append(geom)
            elif geom.geom_type in ('MultiPolygon', 'GeometryCollection'):
                for g in geom.geoms:
                    if hasattr(g, 'area') and g.area > min_area:
                        if g.geom_type == 'Polygon':
                            pieces.append(g)
                        elif g.geom_type == 'MultiPolygon':
                            pieces.extend(p for p in g.geoms if p.area > min_area)
            return pieces

        all_pieces = []
        for group, blon in [(left_group, left_blon), (right_group, right_blon)]:
            for p in _stitch_edge_group(group, blon):
                if p.area < 0.4 * frame_poly.area:
                    all_pieces.append(p)

        if not all_pieces:
            return []
        combined = unary_union(all_pieces)
        clipped = _safe_intersection(combined, frame_poly)
        if clipped.is_empty:
            return []
        clipped = _complement_detect(clipped, frame_poly, expected_frac)
        return _shapely_to_paths(clipped, min_area=min(1.0, min_piece_area)) if not clipped.is_empty else []

    # Short walks → stitch segments together
    parts_x, parts_y = [], []

    for si in range(n_segs):
        sseg = px_segs[si]
        if sseg is None:
            continue
        parts_x.append(sseg['x'])
        parts_y.append(sseg['y'])

        snext = px_segs[(si + 1) % n_segs]
        if snext is None:
            continue

        fi_end = nearest_frame_idx(sseg['x'][-1], sseg['y'][-1])
        fi_start = nearest_frame_idx(snext['x'][0], snext['y'][0])

        if fi_end != fi_start:
            walk = short_boundary_walk(fi_end, fi_start)
            if len(walk) > 1:
                parts_x.append(fr[walk[1:], 0])
                parts_y.append(fr[walk[1:], 1])

    if not parts_x:
        return []

    all_x = np.concatenate(parts_x)
    all_y = np.concatenate(parts_y)
    if len(all_x) < 3:
        return []

    try:
        poly = Polygon(zip(all_x, all_y))
        if not poly.is_valid:
            poly = make_valid(poly)
        clipped = _safe_intersection(poly, frame_poly)
    except Exception:
        return []

    if clipped.is_empty:
        return []

    clipped = _complement_detect(clipped, frame_poly, expected_frac)

    return _shapely_to_paths(clipped, min_area=min(1.0, min_piece_area)) if not clipped.is_empty else []



# ---------------------------------------------------------------------------
# Segment closure in lon/lat (shared with the plotly backend)
#
# ``_antimeridian_clip`` returns open segments (the polygon boundary cut at the
# wrap edge). To fill a segment we must close it back into a polygon. Closing
# in *lon/lat* — walking the wrap meridian, or up-and-over the pole — and only
# THEN projecting gives a well-formed fill polygon on curved frames (AIT / MOL /
# globe), where projecting first and guessing the side (the old ``_project_shape``
# + complement heuristic) breaks for pole-enclosing caps. Both the matplotlib
# fill (``_stitch_and_project``) and the plotly projector import these so the
# two backends close segments identically.
# ---------------------------------------------------------------------------

def _close_clipped_segment(
    segment: dict[str, Any], center: float, n_per_deg: float = 2,
    min_intermediates: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Close a single :func:`_antimeridian_clip` segment into a fill polygon
    (in absolute lon coords).

    Two closure modes, distinguished by whether the segment's entry and exit
    boundary points sit on the same wrap meridian side:

    * **Same-side closure** — the segment enters and exits at the same wrap
      meridian (e.g. both at ``lon = center + 180``). The closure walks back
      along that meridian from ``exit_lat`` to ``entry_lat``, densified at
      ``n_per_deg`` vertices per degree so the projected closing edge follows
      the curved frame silhouette on AIT / MOL.
    * **Opposite-side closure** (polar case) — the segment enters on one wrap
      meridian and exits on the other. Closure walks from the exit boundary
      point UP (or down) to the nearest pole along its wrap meridian, crosses
      to the other wrap meridian at the pole, and walks back DOWN to the entry
      boundary point. This is the clean polar-cap closure for pole-enclosing
      polygons that the projected-then-guess path can't represent.

    Returns the closed sub-polygon as ``(lons, lats)`` arrays with last vertex
    == first vertex.
    """
    seg_lons = list(segment['lons'])
    seg_lats = list(segment['lats'])
    entry_lat = segment['entry_lat']
    exit_lat = segment['exit_lat']
    entry_lon = float(seg_lons[0])
    exit_lon = float(seg_lons[-1])

    out_lons = list(seg_lons)
    out_lats = list(seg_lats)

    if abs(entry_lon - exit_lon) < 1e-6:
        # Same-side closure along the wrap meridian from
        # (exit_lon, exit_lat) back to (entry_lon, entry_lat).
        wrap_lon = entry_lon
        d_lat = float(entry_lat) - float(exit_lat)
        n = max(int(min_intermediates),
                int(round(n_per_deg * abs(d_lat))))
        for k in range(1, n + 1):
            t = k / (n + 1)
            out_lons.append(wrap_lon)
            out_lats.append(float(exit_lat) + t * d_lat)
        out_lons.append(entry_lon)
        out_lats.append(float(entry_lat))
    else:
        # Opposite-side (polar) closure: go from exit boundary up to the
        # nearest pole, cross to the other wrap meridian at the pole, walk
        # back down to the entry boundary.
        avg_lat = (float(entry_lat) + float(exit_lat)) / 2.0
        pole_lat = 90.0 if avg_lat >= 0 else -90.0
        d_to_pole = pole_lat - float(exit_lat)
        n_to_pole = max(int(min_intermediates),
                        int(round(n_per_deg * abs(d_to_pole))))
        for k in range(1, n_to_pole + 1):
            t = k / (n_to_pole + 1)
            out_lons.append(exit_lon)
            out_lats.append(float(exit_lat) + t * d_to_pole)
        out_lons.append(exit_lon)
        out_lats.append(pole_lat)
        # Cross at the pole (same sphere point, different wrap-meridian edge).
        out_lons.append(entry_lon)
        out_lats.append(pole_lat)
        d_from_pole = float(entry_lat) - pole_lat
        n_from_pole = max(int(min_intermediates),
                          int(round(n_per_deg * abs(d_from_pole))))
        for k in range(1, n_from_pole + 1):
            t = k / (n_from_pole + 1)
            out_lons.append(entry_lon)
            out_lats.append(pole_lat + t * d_from_pole)
        out_lons.append(entry_lon)
        out_lats.append(float(entry_lat))

    return np.asarray(out_lons), np.asarray(out_lats)


def _densify_along_wrap_edge(
    plons: npt.ArrayLike, plats: npt.ArrayLike, wrap_lon: float,
    n_per_deg: float = 2, min_intermediates: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert intermediate vertices along edges that lie on a wrap edge.

    For projections with curved frame silhouettes (AIT, MOL, pseudocylindrical,
    …), a polygon edge from ``(wrap_lon, lat_a)`` to ``(wrap_lon, lat_b)`` — two
    consecutive vertices at the same wrap meridian — projects to a chord across
    the curved frame silhouette rather than following it. This post-processes a
    wrap-split sub-polygon by inserting N intermediate vertices at the same
    ``wrap_lon`` with linearly interpolated latitudes on each such edge, so the
    projected polygon traces the frame curve.

    Parameters
    ----------
    plons, plats : sequence of float
        Closed sub-polygon vertices (``plons[0] == plons[-1]``, same for lats).
    wrap_lon : float
        The wrap-edge longitude this sub-polygon is on (``center + 180`` or
        ``center - 180``).
    n_per_deg : float
        Intermediate vertices per degree of latitude span. Default ``2``.
    min_intermediates : int
        Minimum intermediate vertices per along-wrap edge. Default ``5``.
    """
    plons = np.asarray(plons, dtype=float)
    plats = np.asarray(plats, dtype=float)
    out_lons = [float(plons[0])]
    out_lats = [float(plats[0])]
    for i in range(1, len(plons)):
        l0, l1 = float(plons[i - 1]), float(plons[i])
        a0, a1 = float(plats[i - 1]), float(plats[i])
        on_wrap_both = (abs(l0 - wrap_lon) < 1e-6
                        and abs(l1 - wrap_lon) < 1e-6)
        if on_wrap_both and abs(a1 - a0) > 1e-3:
            n = max(int(min_intermediates),
                    int(round(n_per_deg * abs(a1 - a0))))
            for k in range(1, n + 1):
                t = k / (n + 1)
                out_lons.append(wrap_lon)
                out_lats.append(a0 + t * (a1 - a0))
        out_lons.append(l1)
        out_lats.append(a1)
    return np.asarray(out_lons), np.asarray(out_lats)
