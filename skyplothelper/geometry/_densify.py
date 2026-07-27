"""Edge densification helpers for spherical polygon edges.

Optionally uses ``cartopy.geodesic`` for high-accuracy great-circle
densification; falls back to a vectorised slerp otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

try:
    from cartopy import geodesic as _geodesic  # noqa: F401
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False


def _slerp(lon_a: float, lat_a: float, lon_b: float, lat_b: float,
           n_pts: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Spherical linear interpolation (great circle path) between two points.

    Returns n_pts points along the great circle from (lon_a, lat_a) to
    (lon_b, lat_b), excluding the endpoint to avoid duplication.

    Parameters
    ----------
    lon_a, lat_a, lon_b, lat_b : float
        Endpoints in degrees.
    n_pts : int
        Number of interpolated points (excluding endpoint).

    Returns
    -------
    lons, lats : ndarray
        Interpolated coordinates in degrees.
    """
    # Convert to unit vectors
    lon1, lat1 = np.radians(lon_a), np.radians(lat_a)
    lon2, lat2 = np.radians(lon_b), np.radians(lat_b)

    p1 = np.array([np.cos(lat1)*np.cos(lon1),
                    np.cos(lat1)*np.sin(lon1),
                    np.sin(lat1)])
    p2 = np.array([np.cos(lat2)*np.cos(lon2),
                    np.cos(lat2)*np.sin(lon2),
                    np.sin(lat2)])

    # Angle between the two points
    dot = np.clip(np.dot(p1, p2), -1.0, 1.0)
    omega = np.arccos(dot)

    t = np.linspace(0, 1, n_pts, endpoint=False)

    if omega < 1e-10:
        # Points are (nearly) identical — just replicate
        return np.full(n_pts, lon_a), np.full(n_pts, lat_a)

    # Slerp
    sin_omega = np.sin(omega)
    coeff1 = np.sin((1 - t) * omega) / sin_omega
    coeff2 = np.sin(t * omega) / sin_omega
    pts = coeff1[:, np.newaxis] * p1 + coeff2[:, np.newaxis] * p2

    # Convert back to lon/lat
    lats_out = np.degrees(np.arcsin(np.clip(pts[:, 2], -1, 1)))
    lons_out = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))

    # Pole-lon disambiguation: when an endpoint is the geographic pole
    # (lat=±90), its lon is mathematically undefined and `vec2ang`-style
    # back-projection returns lon=0 by convention. Inside a slerp path
    # from/to the pole, every interior point lies along the meridian of
    # the OTHER endpoint, so the pole's natural meridian lon is `lon_b`
    # (or `lon_a` when slerping into the pole). Without this fix, the
    # default lon=0 at the pole creates a spurious lon jump (often >180°
    # from the next interior point's lon), which `_antimeridian_clip`
    # interprets as an antimeridian crossing — triggering polygon splits
    # that shouldn't exist for tiles whose boundary just touches the pole.
    if abs(lat_a) > 89.999 and len(lons_out) > 0:
        lons_out[0] = lon_b
    return lons_out, lats_out


def _angular_separation(lon_a: float, lat_a: float, lon_b: float,
                        lat_b: float) -> Any:
    """Angular separation between two points in degrees (Vincenty formula)."""
    lon1, lat1 = np.radians(lon_a), np.radians(lat_a)
    lon2, lat2 = np.radians(lon_b), np.radians(lat_b)
    dlon = lon2 - lon1
    num = np.sqrt((np.cos(lat2)*np.sin(dlon))**2 +
                  (np.cos(lat1)*np.sin(lat2) -
                   np.sin(lat1)*np.cos(lat2)*np.cos(dlon))**2)
    den = (np.sin(lat1)*np.sin(lat2) +
           np.cos(lat1)*np.cos(lat2)*np.cos(dlon))
    return np.degrees(np.arctan2(num, den))


def _densify_polygon_edges(lons: npt.ArrayLike, lats: npt.ArrayLike,
                           resolution: int = 100,
                           geodesic: bool | str = 'auto',
                           geodesic_threshold: float = 10.0
                           ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Densify polygon edges by interpolating intermediate points.

    Sparse polygons (triangles, pentagons, etc.) need densification so that:
    1. Edges follow projection curvature instead of straight pixel-space lines
    2. Enough vertices exist for boundary-crossing detection and splitting

    Every edge is interpolated the "short way" around the sphere (the
    direction with the smaller absolute lon delta after wrap). This is
    the universal convention for user-authored polygon vertex sequences
    — each edge is intended to connect its two endpoints by the shorter
    of the two possible arcs. The historical "centroid distance"
    heuristic that picked between short- and long-way picked
    incorrectly for the closure edge of polygons whose vertex sequences
    walk ~360° in lon (e.g. circumpolar surveys like Euclid Northern
    mainland with vertices at lon=0, 25, …, 335, 355), producing a
    horizontal "snap-across" at high latitude on AIT/MOL renders.

    Parameters
    ----------
    lons, lats : ndarray
        Polygon vertices (closed — first == last).
    resolution : int
        Number of points per edge.
    geodesic : bool or 'auto'
        If True, always use great-circle interpolation (accurate for large
        survey footprints).  If False, use linear interpolation in lon/lat.
        If 'auto' (default), use geodesic for edges longer than
        ``geodesic_threshold`` degrees and linear for shorter edges.
    geodesic_threshold : float
        Edge length in degrees above which geodesic interpolation is used
        when ``geodesic='auto'``.  Default 10°.

    Returns
    -------
    dense_lons, dense_lats : ndarray
        Densified polygon vertices (closed).
    """
    lons = np.asarray(lons, float)
    lats = np.asarray(lats, float)
    n_verts = len(lons) - 1  # exclude closing vertex

    dense_lons_parts = []
    dense_lats_parts = []

    for i in range(n_verts):
        lon_a, lat_a = lons[i], lats[i]
        lon_b, lat_b = lons[i + 1], lats[i + 1]

        # Determine whether to use geodesic for this edge
        if geodesic is True:
            use_geodesic = True
        elif geodesic is False:
            use_geodesic = False
        else:  # 'auto'
            edge_len = _angular_separation(lon_a, lat_a, lon_b, lat_b)
            use_geodesic = edge_len > geodesic_threshold
            # Linear lon/lat interpolation breaks down for any edge that
            # touches the geographic pole: the pole is at (any_lon, ±90),
            # so sweeping lon from `lon_a` to `lon_b` traces a path that
            # bows AROUND the pole at intermediate latitudes instead of
            # going up a single meridian to it. Force slerp (via xyz) for
            # pole-touching edges regardless of edge length — slerp handles
            # the pole correctly because xyz vectors are well-defined there.
            if abs(lat_a) > 89.999 or abs(lat_b) > 89.999:
                use_geodesic = True

        if use_geodesic:
            # Great-circle interpolation — inherently goes the short way.
            edge_lons, edge_lats = _slerp(lon_a, lat_a, lon_b, lat_b,
                                          resolution)
        else:
            # Linear lon/lat interpolation, short-way around the sphere.
            # Unwrap ``lon_b`` so the signed delta from ``lon_a`` has
            # magnitude ≤ 180° — that's the short-way arc.
            dlon = lon_b - lon_a
            if dlon > 180:
                lon_b_unwrap = lon_b - 360
            elif dlon < -180:
                lon_b_unwrap = lon_b + 360
            else:
                lon_b_unwrap = lon_b

            t = np.linspace(0, 1, resolution, endpoint=False)
            edge_lons = lon_a + t * (lon_b_unwrap - lon_a)
            edge_lats = lat_a + t * (lat_b - lat_a)

            # Re-wrap longitudes to [-180, 180]
            edge_lons = ((edge_lons + 180) % 360) - 180

        dense_lons_parts.append(edge_lons)
        dense_lats_parts.append(edge_lats)

    dense_lons = np.concatenate(dense_lons_parts)
    dense_lats = np.concatenate(dense_lats_parts)

    # Close the polygon
    dense_lons = np.append(dense_lons, dense_lons[0])
    dense_lats = np.append(dense_lats, dense_lats[0])

    return dense_lons, dense_lats


def _unwrap_lon(lon: float, ref: float) -> float:
    """Unwrap a longitude relative to a reference, keeping within ±180° of it."""
    d = lon - ref
    if d > 180:
        return lon - 360
    elif d < -180:
        return lon + 360
    return lon
