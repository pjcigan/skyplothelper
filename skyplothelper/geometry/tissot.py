"""Tissot indicatrix grid.

A grid of small geodesic circles spaced regularly across the sphere —
classic visualization of a projection's distortion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from .._stroke import _stroke_path_effects
from ._parsing import _parse_angle
from .shapes import add_geodesic_circle


def tissot(ax: Axes, rad_deg: Any = 5,
           lons: npt.ArrayLike | None = None,
           lats: npt.ArrayLike | None = None, resolution: int = 100,
           clip: str = 'auto', stroke_color: Any = None,
           stroke_lw: float | None = None, **kwargs: Any) -> list[Any]:
    """Add Tissot indicatrices — a grid of equal-radius geodesic circles that
    visualize a projection's local distortion.

    Each indicatrix is a true geodesic (small) circle of the *same* angular
    radius; how it renders — stretched, sheared, or resized — at each grid
    point reveals the projection's local scale and shape distortion. Handy on
    the flat planet frames from :func:`~skyplothelper.make_planet_frame` and on
    all-sky projections.

    Parameters
    ----------
    ax : WCSAxes
        A FITS-projection frame (all-sky, globe, or flat planet map). Non-FITS
        projections (Robinson, Eckert, ...) are not yet supported.
    rad_deg : float or Quantity
        Indicatrix radius in degrees (default 5).
    lons, lats : array-like or None
        Grid positions. Defaults to a 6x6 grid (longitudes every 60 deg,
        latitudes from -80 to +80).
    resolution : int
        Boundary points per circle.
    clip : str
        Projection-seam handling pipeline forwarded to each
        :func:`~skyplothelper.add_geodesic_circle`. ``'auto'``
        (default) resolves to ``'d3'``.
    stroke_color, stroke_lw :
        Optional legibility stroke around each indicatrix (the shared sph
        stroke, applied via path effects).
    **kwargs
        Passed to PathPatch (``facecolor``, ``edgecolor``, ``alpha``, ...).
    """
    if getattr(ax, 'wcs', None) is None:
        raise NotImplementedError(
            "tissot needs a FITS-projection frame (ax.wcs is None on non-FITS "
            "projections like Robinson). Use a FITS projection (CAR/MOL/AIT/"
            "...) or a globe.")

    _pe = _stroke_path_effects(stroke_color, stroke_lw)
    if _pe is not None:
        kwargs.setdefault('path_effects', _pe)

    rad_deg = _parse_angle(rad_deg)
    # Fresh ndarray locals (not the ArrayLike params) so the shape-typed numpy
    # stubs keep a concrete ndarray type through meshgrid/flatten.
    lon_arr = (np.linspace(-180, 180, 6, endpoint=False)
               if lons is None else np.asarray(lons))
    lat_arr = np.linspace(-80, 80, 6) if lats is None else np.asarray(lats)
    if lon_arr.ndim == 1 or lat_arr.ndim == 1:
        lon_arr, lat_arr = np.meshgrid(lon_arr, lat_arr)
    flat_lon, flat_lat = lon_arr.flatten(), lat_arr.flatten()
    all_patches = []
    for lon, lat in zip(flat_lon, flat_lat):
        all_patches.extend(add_geodesic_circle(
            ax, lon, lat, rad_deg, resolution, clip=clip, **kwargs))
    return all_patches

