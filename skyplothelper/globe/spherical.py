"""Spherical geometry helpers.

Vectorised lon/lat <-> xyz, great-circle distance/arc/midpoint/bearing,
small-circle generation, and orthographic projection forward/inverse.

Intentionally independent of :mod:`skyplothelper.geometry`. The two
modules sit at different layers:

* This module is pure spherical math — lon/lat arrays in, lon/lat (or
  xyz, or projected x/y) arrays out — with no WCSAxes dependency.
  Globe code, plain-matplotlib orthographic scatter, and a handful of
  overlay helpers (:mod:`overlays.ruler`) consume it directly.
* :mod:`skyplothelper.geometry` is the WCS-renderable region pipeline:
  every public entry point lands in pixel space and intersects the
  axes frame polygon. It carries antimeridian-clip and pole-handling
  special cases that only make sense once you're heading for a
  rendered patch.

The Vincenty distance and slerp math is restated in
``geometry/_densify`` rather than imported from here, because that
copy carries pipeline-specific adjustments (e.g. pole-longitude
disambiguation for the antimeridian splitter) that the pure-math
forms here intentionally don't have. Keeping the two implementations
side-by-side preserves the package layering (geometry → globe would
invert it) and lets each module evolve to its own callers' needs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ..constants import planet_radii  # noqa: F401  (great_circle_distance default)

# ===== Orthographic projection =====

def orthographic_forward(
    longitude: npt.ArrayLike, latitude: npt.ArrayLike,
    lon_0: float = 180., lat_0: float = 0., R: float = 1.,
) -> npt.NDArray[np.float64]:
    """
    Forward orthographic map projection: (lon, lat) -> (x, y).

    Parameters
    ----------
    longitude, latitude : float or ndarray
        Input coordinates in degrees.
    lon_0, lat_0 : float
        Projection center in degrees.
    R : float
        Sphere radius. Default 1.

    Returns
    -------
    xy : ndarray, shape (..., 2)
        Projected x, y coordinates.
    """
    lat_r = np.radians(np.asarray(latitude, dtype=float))
    lon_r = np.radians(np.asarray(longitude, dtype=float) - lon_0)
    lat0_r = np.radians(lat_0)

    x = R * np.cos(lat_r) * np.sin(lon_r)
    y = R * (np.cos(lat0_r) * np.sin(lat_r) -
             np.sin(lat0_r) * np.cos(lat_r) * np.cos(lon_r))
    return np.column_stack([x, y])


def orthographic_inverse(
    x: npt.ArrayLike, y: npt.ArrayLike,
    lon_0: float = 180., lat_0: float = 0., R: float = 1.,
) -> npt.NDArray[np.float64]:
    """
    Inverse orthographic map projection: (x, y) -> (lon, lat).

    Parameters
    ----------
    x, y : float or ndarray
        Projected coordinates.
    lon_0, lat_0 : float
        Projection center in degrees.
    R : float
        Sphere radius. Default 1.

    Returns
    -------
    lonlat : ndarray, shape (..., 2)
        Recovered longitude, latitude in degrees.
    """
    lat0_r = np.radians(lat_0)
    rho = np.hypot(x, y)
    c = np.arcsin(rho / R)

    longitude = lon_0 + np.degrees(np.arctan2(
        x * np.sin(c),
        rho * np.cos(c) * np.cos(lat0_r) - y * np.sin(c) * np.sin(lat0_r)))
    latitude = np.degrees(np.arcsin(
        np.cos(c) * np.sin(lat0_r) +
        y * np.sin(c) * np.cos(lat0_r) / rho))
    return np.column_stack([longitude, latitude])


# ===== Orthographic visibility =====

def orthographic_visibility(
    longitude: npt.ArrayLike, latitude: npt.ArrayLike,
    lon_0: float = 0., lat_0: float = 0.,
) -> Any:
    """
    Determine which points are on the front (visible) hemisphere of an
    orthographic projection.

    Parameters
    ----------
    longitude, latitude : array_like
        Coordinates in degrees.
    lon_0, lat_0 : float
        Projection center in degrees.

    Returns
    -------
    visible : ndarray of bool
        True for points with angular distance from center < 90°.
    """
    lat_r = np.radians(latitude)
    lon_r = np.radians(np.asarray(longitude, dtype=float) - lon_0)
    lat0_r = np.radians(lat_0)
    cos_c = (np.sin(lat0_r) * np.sin(lat_r) +
             np.cos(lat0_r) * np.cos(lat_r) * np.cos(lon_r))
    return cos_c > 0



# ===== Internal ortho-project helper =====

def _ortho_project(
    lons: npt.ArrayLike, lats: npt.ArrayLike, lon_0: float, lat_0: float,
    R_sphere: float = 1.,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Project arrays of lon/lat (deg) to orthographic x/y (in R units)."""
    lat_r = np.radians(lats)
    lon_r = np.radians(np.asarray(lons) - lon_0)
    lat0_r = np.radians(lat_0)
    x = R_sphere * np.cos(lat_r) * np.sin(lon_r)
    y = R_sphere * (np.cos(lat0_r) * np.sin(lat_r) -
                    np.sin(lat0_r) * np.cos(lat_r) * np.cos(lon_r))
    return np.asarray(x), np.asarray(y)



# ===== Cartesian / great-circle / small-circle =====

def lonlat_to_xyz(
    lon: npt.ArrayLike, lat: npt.ArrayLike, R: float = 1.,
) -> npt.NDArray[np.float64]:
    """
    Convert longitude/latitude (degrees) to Cartesian [x, y, z].

    Parameters
    ----------
    lon, lat : array_like
        Longitude and latitude in degrees.
    R : float
        Radius. Default 1 (unit sphere).

    Returns
    -------
    xyz : ndarray, shape (..., 3)
        Cartesian coordinates.
    """
    lon_r = np.radians(np.asarray(lon, dtype=float))
    lat_r = np.radians(np.asarray(lat, dtype=float))
    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return np.stack([x, y, z], axis=-1)


def xyz_to_lonlat(
    xyz: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Convert Cartesian [x, y, z] to longitude/latitude (degrees).

    Parameters
    ----------
    xyz : array_like, shape (..., 3)
        Cartesian coordinates.

    Returns
    -------
    lon, lat : ndarray
        Longitude (-180, 180] and latitude [-90, 90] in degrees.
    """
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.degrees(np.arcsin(np.clip(z / np.where(r > 0, r, 1.), -1, 1)))
    lon = np.degrees(np.arctan2(y, x))
    return lon, lat


def great_circle_distance(
    lon1: npt.ArrayLike, lat1: npt.ArrayLike,
    lon2: npt.ArrayLike, lat2: npt.ArrayLike,
    R: float | None = None, body: str | None = None,
) -> Any:
    """
    Great-circle distance between two points using the Vincenty formula.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : float or array_like
        Coordinates in degrees.
    R : float or None
        Sphere radius in km. If None, uses body radius or returns radians.
    body : str or None
        Body name for automatic radius lookup (e.g. 'earth').

    Returns
    -------
    distance : float or ndarray
        Distance in km if R or body is given, otherwise in radians.
    """
    if R is None and body is not None:
        R = planet_radii.get(body.lower(), 6371.0)

    lat1_r = np.radians(np.asarray(lat1, dtype=float))
    lat2_r = np.radians(np.asarray(lat2, dtype=float))
    dlon_r = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))

    # Vincenty formula (robust near antipodes)
    a = np.cos(lat2_r) * np.sin(dlon_r)
    b = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon_r)
    c = np.sin(lat1_r) * np.sin(lat2_r) + np.cos(lat1_r) * np.cos(lat2_r) * np.cos(dlon_r)
    sigma = np.arctan2(np.sqrt(a**2 + b**2), c)

    if R is not None:
        return R * sigma
    return sigma


def great_circle_arc(
    lon1: float, lat1: float, lon2: float, lat2: float, n_pts: int = 100,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Generate points along the great-circle arc between two points.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : float
        Endpoint coordinates in degrees.
    n_pts : int
        Number of points along the arc.

    Returns
    -------
    lons, lats : ndarray
        Coordinates along the arc in degrees.
    """
    p1 = lonlat_to_xyz(lon1, lat1)
    p2 = lonlat_to_xyz(lon2, lat2)
    # Slerp
    dot = np.clip(np.dot(p1, p2), -1., 1.)
    omega = np.arccos(dot)
    if abs(omega) < 1e-12:
        return np.full(n_pts, lon1), np.full(n_pts, lat1)
    t = np.linspace(0, 1, n_pts)
    sin_omega = np.sin(omega)
    pts = (np.sin((1 - t[:, None]) * omega) * p1 +
           np.sin(t[:, None] * omega) * p2) / sin_omega
    return xyz_to_lonlat(pts)


def midpoint(
    lon1: float, lat1: float, lon2: float, lat2: float,
) -> tuple[float, float]:
    """Great-circle midpoint between two points (degrees)."""
    lons, lats = great_circle_arc(lon1, lat1, lon2, lat2, n_pts=3)
    return float(lons[1]), float(lats[1])


def initial_bearing(
    lon1: float, lat1: float, lon2: float, lat2: float,
) -> Any:
    """
    Initial bearing (forward azimuth) from point 1 to point 2.

    Returns
    -------
    bearing : float
        Bearing in degrees [0, 360).
    """
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlon_r = np.radians(lon2 - lon1)
    x = np.sin(dlon_r) * np.cos(lat2_r)
    y = (np.cos(lat1_r) * np.sin(lat2_r) -
         np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon_r))
    return np.degrees(np.arctan2(x, y)) % 360


def destination_point(
    lon: float, lat: float, bearing_deg: float, distance_rad: float,
) -> tuple[Any, Any]:
    """
    Compute destination point given start, bearing, and angular distance.

    Parameters
    ----------
    lon, lat : float
        Start point in degrees.
    bearing_deg : float
        Initial bearing in degrees.
    distance_rad : float
        Angular distance in radians.

    Returns
    -------
    lon2, lat2 : float
        Destination in degrees.
    """
    lat_r = np.radians(lat)
    brg_r = np.radians(bearing_deg)
    d = distance_rad
    lat2 = np.arcsin(np.sin(lat_r) * np.cos(d) +
                      np.cos(lat_r) * np.sin(d) * np.cos(brg_r))
    lon2 = np.radians(lon) + np.arctan2(
        np.sin(brg_r) * np.sin(d) * np.cos(lat_r),
        np.cos(d) - np.sin(lat_r) * np.sin(lat2))
    return np.degrees(lon2), np.degrees(lat2)


def small_circle(
    center_lon: float, center_lat: float, radius_deg: float, n_pts: int = 180,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Generate points along a small circle (constant angular radius).

    Parameters
    ----------
    center_lon, center_lat : float
        Center in degrees.
    radius_deg : float
        Angular radius in degrees.
    n_pts : int
        Number of points.

    Returns
    -------
    lons, lats : ndarray
        Coordinates in degrees.
    """
    bearings = np.linspace(0, 360, n_pts, endpoint=False)
    radius_rad = np.radians(radius_deg)
    lons, lats = [], []
    for b in bearings:
        lo, la = destination_point(center_lon, center_lat, b, radius_rad)
        lons.append(lo)
        lats.append(la)
    return np.array(lons), np.array(lats)
