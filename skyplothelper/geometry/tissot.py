"""Tissot indicatrix grid.

A grid of small geodesic circles spaced regularly across the sphere —
classic visualization of a projection's distortion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from ._parsing import _parse_angle
from .shapes import add_geodesic_circle


def tissot(ax: Axes, rad_deg: Any = 5,
           lons: npt.ArrayLike | None = None,
           lats: npt.ArrayLike | None = None, resolution: int = 100,
           clip: str = 'auto', **kwargs: Any) -> list[Any]:
    """Add Tissot indicatrices.

    Parameters
    ----------
    ax : WCSAxes
    rad_deg : float or Quantity
        Indicatrix radius in degrees (default 5).
    lons, lats : array-like or None
        Grid positions.
    resolution : int
        Boundary points per circle.
    clip : str
        Projection-seam handling pipeline forwarded to each
        :func:`~skyplothelper.add_geodesic_circle`. ``'auto'``
        (default) resolves to ``'d3'``.
    **kwargs
        Passed to PathPatch.
    """
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

