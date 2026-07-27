"""Projection-specific visible-region boundary curves.

Each helper here returns a closed ``(N, 2)`` array of ``(lon, lat)``
world coordinates tracing the natural visible boundary of one
projection family. Callers convert these to pixel-space frame
curves via :meth:`skyplothelper.coord_overlay._FrameCurve.from_world_polyline`
(with ``closed=True``), then pass them to
:func:`skyplothelper.coord_overlay.add_overlay_ticks` as the
``boundary=`` argument — used both for clipping out-of-frame
axis-curve ticks and (optionally) as the tick-discovery curve.

This module exists because several projections (BON cardioid, PCO
egg envelope, HPX HEALPix stepped diamond, the conic wedges) have
visible regions that don't match their astropy frame's spine
('rectangular' for BON / PCO / conics, 'elliptical' for HPX). The
boundary-curve mode of ``add_overlay_ticks`` needs a true boundary
to function; this is where each projection family's true boundary
lives.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def _crval(ax: Any) -> tuple[float, float]:
    """Return ``(crval_lon, crval_lat)`` from the axes' WCS, or
    sensible defaults if WCS isn't available."""
    if hasattr(ax, 'wcs') and ax.wcs is not None:
        return float(ax.wcs.wcs.crval[0]), float(ax.wcs.wcs.crval[1])
    lon = getattr(ax, '_sph_center_lon', 180.0)
    lat = getattr(ax, '_sph_center_lat', 0.0)
    return float(lon), float(lat)


def world_rect_boundary(
        ax: Any, lat_max: float = 89.99,
        n: int = 400, lat_lo: float | None = None,
        lat_hi: float | None = None) -> npt.NDArray[np.float64]:
    """Closed boundary tracing the four sides of the world coordinate
    rectangle: ``lat`` arcs + ``lon=CRVAL±180`` meridians.

    For projections whose full-sky visible region is the image of
    this world rectangle (conic wedges, polyconic egg, ...), this
    generic helper produces a closed (lon, lat) polyline that maps
    to the visible boundary in pixel space. For projections where
    one or more sides collapse to a point (Bonne south pole, conic
    apex), the redundant near-coincident pixels along the collapsed
    side are tolerated by downstream clipping (matplotlib's Path
    point-in-polygon handles them correctly).

    Parameters
    ----------
    ax : WCSAxes
    lat_max : float
        Symmetric top / bottom arc latitude in degrees, used when
        ``lat_lo`` / ``lat_hi`` are not given. Stops just shy of ±90
        so SkyCoord-style transforms stay well-defined.
    n : int
        Sample points per side (so total polyline length is ~4n).
    lat_lo, lat_hi : float, optional
        Explicit (asymmetric) bottom / top arc latitudes. Override
        the symmetric ``±lat_max`` when given — used for conic wedges
        whose visible latitude range is clipped on the divergent far
        side (see :func:`conic_visible_lat_range`).

    Returns
    -------
    lonlat : (~4n, 2) ndarray
        Closed boundary as a ``(lon, lat)`` polyline in degrees.
    """
    if lat_lo is None:
        lat_lo = -lat_max
    if lat_hi is None:
        lat_hi = lat_max
    crval_lon, _ = _crval(ax)
    lon_w = crval_lon - 180.0
    lon_e = crval_lon + 180.0
    lons = np.linspace(lon_w, lon_e, n)
    lats = np.linspace(lat_lo, lat_hi, n)
    top = np.column_stack([lons, np.full(n, lat_hi)])
    right = np.column_stack([np.full(n, lon_e), lats[::-1]])
    bottom = np.column_stack([lons[::-1], np.full(n, lat_lo)])
    left = np.column_stack([np.full(n, lon_w), lats])
    return np.vstack([top, right, bottom, left])


def conic_visible_lat_range(
        fits_code: str, pv2_1: float) -> tuple[float, float]:
    """Visible (frame / clip) latitude range for an all-sky conic.

    With the reference point on the standard parallel (CRVAL2 = PV2_1,
    the kapteyn all-sky recipe), the cone's apex sits at the pole on
    the same side as ``pv2_1`` and the wedge opens toward the far pole.
    COD (equidistant) and COE (equal-area) stay finite across the whole
    sphere, so they show the full ``±89.99°`` range. COO (orthomorphic)
    and COP (perspective) diverge toward the far pole, so kapteyn clips
    them 30° past the equator on the far side (``wylim=(-30, 90)`` for a
    northern standard parallel); we mirror that bound for a southern one.

    Returns ``(lat_lo, lat_hi)`` in degrees.
    """
    if fits_code in ('COD', 'COE'):
        return -89.99, 89.99
    # COO / COP: drop the divergent far-pole cap (60° around the far pole).
    if pv2_1 >= 0.0:
        return -30.0, 89.99
    return -89.99, 30.0


def _conic_pv2_1(ax: Any, default: float = 45.0) -> float:
    """Read ``PV2_1`` (the conic standard parallel) from the axes' WCS."""
    try:
        for axis, param, val in ax.wcs.wcs.get_pv():
            if axis == 2 and param == 1:
                return float(val)
    except Exception:
        pass
    return default


def conic_boundary(ax: Any, n: int = 400) -> npt.NDArray[np.float64]:
    """Closed boundary polygon for a conic projection
    (COD / COE / COO / COP), used to clip the data, overlay
    gridlines, and ticks to the projection's natural visible wedge.

    With the reference point centered on the standard parallel
    (CRVAL2 = PV2_1), the visible region is a clean wedge: the apex
    pole (a near-point at ``lat = ±89.99``), the two ``lon = CRVAL±180``
    seam meridians forming the wedge sides, and a bottom parallel arc at
    the far edge of the visible latitude range. COO / COP diverge toward
    the far pole, so their wedge is clipped to
    :func:`conic_visible_lat_range`; COD / COE show the full sphere.

    Parameters
    ----------
    ax : WCSAxes
    n : int
        Sample points per side.
    """
    code = ''
    try:
        code = str(ax.wcs.wcs.ctype[0]).split('-')[-1].strip().upper()
    except Exception:
        pass
    pv2_1 = _conic_pv2_1(ax)
    lat_lo, lat_hi = conic_visible_lat_range(code, pv2_1)
    return world_rect_boundary(ax, n=n, lat_lo=lat_lo, lat_hi=lat_hi)


def polyconic_boundary(ax: Any, n: int = 400) -> npt.NDArray[np.float64]:
    """Closed boundary of a polyconic projection (PCO).

    The visible "egg" envelope is the image of the world rectangle —
    the lon=CRVAL±180 seams form the curved sides of the egg, and
    the lat=±90 arcs collapse to the polar tips. Delegates to
    :func:`world_rect_boundary`. (The egg IS the visible region, even
    though the lon=±180 trace alone is non-monotonic.)

    Parameters
    ----------
    ax : WCSAxes
    n : int
        Sample points per side.
    """
    return world_rect_boundary(ax, n=n)


def healpix_boundary(
        ax: Any, eq_lat: float | None = None) -> npt.NDArray[np.float64]:
    """Closed boundary of the HEALPix all-sky projection (HPX).

    Traces the *actual* visible region — the stepped diamond formed
    by the union of HEALPix's 12 base pixels:

    - 4 north polar pixels (rhombii with one vertex at the north
      pole and bases at ``lat = arcsin(2/3) ≈ 41.81°``)
    - The equatorial band
    - 4 south polar pixels (mirror image)

    The outline returns just the *vertices* of the stepped-diamond
    perimeter — 16 corners total (4 north peaks, 4 north-side
    boundary corners, 4 south peaks, 4 south-side boundary
    corners). When plotted with ``transform=ax.get_transform('world')``,
    matplotlib renders straight pixel-space lines between
    consecutive vertices — and HEALPix base-pixel edges are
    *exactly* straight lines in HPX projection space (that's the
    whole point of the projection). Providing intermediate
    interpolated (lon, lat) samples would project to curves,
    incorrectly making the polar tile sides look concave.

    Parameters
    ----------
    ax : WCSAxes
    eq_lat : float, optional
        Equator-to-pole-cap latitude boundary in degrees. Default
        ``arcsin(2/3) ≈ 41.81°`` — the standard HEALPix value.

    Returns
    -------
    lonlat : (N, 2) ndarray
        Closed boundary in degrees. ``N = 17`` (16 distinct
        corners plus the closing vertex).
    """
    crval_lon, _ = _crval(ax)
    if eq_lat is None:
        eq_lat = float(np.degrees(np.arcsin(2.0 / 3.0)))

    peak_offsets = (-135.0, -45.0, 45.0, 135.0)
    base_offsets = (-180.0, -90.0, 0.0, 90.0, 180.0)
    peak_lons = [crval_lon + off for off in peak_offsets]
    base_lons = [crval_lon + off for off in base_offsets]

    verts = []
    # North polar zigzag (west → east): base, peak, base, peak, ...
    verts.append((base_lons[0], eq_lat))
    for k in range(4):
        verts.append((peak_lons[k], 90.0))
        verts.append((base_lons[k + 1], eq_lat))
    # Right edge of equatorial band: drop straight to -eq_lat
    verts.append((base_lons[-1], -eq_lat))
    # South polar zigzag (east → west)
    for k in range(3, -1, -1):
        verts.append((peak_lons[k], -90.0))
        verts.append((base_lons[k], -eq_lat))
    # Close along left edge of equatorial band
    verts.append((base_lons[0], eq_lat))

    return np.asarray(verts, dtype=float)


def bonne_boundary(
        ax: Any, lat_max: float = 89.99,
        n: int = 400) -> npt.NDArray[np.float64]:
    """Closed boundary of a BON (Bonne pseudoconic) projection.

    The natural visible region is the cardioid traced by the
    ``lon = CRVAL ± 180°`` meridians on each side. The boundary
    polyline runs from south pole up the west side (``lon=CRVAL-180``)
    to the north pole, then back down the east side (``lon=CRVAL+180``)
    to the south pole, forming a single closed curve.

    Parameters
    ----------
    ax : WCSAxes
        Reads ``ax.wcs.wcs.crval[0]`` for the projection center.
    lat_max : float
        Latitude (degrees) at which to clamp the meridian sampling.
        Stops just shy of ±90 so the wrap-around at the poles is
        well-defined.
    n : int
        Sample points per side.

    Returns
    -------
    lonlat : (2*n, 2) ndarray
        Closed boundary as a ``(lon, lat)`` polyline in degrees.
    """
    crval_lon, _ = _crval(ax)
    lats = np.linspace(-lat_max, lat_max, n)
    west = np.column_stack([np.full(n, crval_lon - 180.0), lats])
    east = np.column_stack([np.full(n, crval_lon + 180.0), lats[::-1]])
    return np.vstack([west, east])
