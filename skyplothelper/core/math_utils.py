"""Angle wrapping and value-range remapping helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def map_to_newrange(
    oldval: npt.ArrayLike,
    oldminmax: Sequence[float],
    newminmax: Sequence[float],
) -> npt.NDArray[np.float64]:
    """
    Map value(s) from one numerical range to another via linear interpolation.

    Parameters
    ----------
    oldval : float or array-like
        Input value(s) to remap.
    oldminmax : list
        [min_old, max_old]
    newminmax : list
        [min_new, max_new]

    Returns
    -------
    newval : float or ndarray

    Example
    -------
    map_to_newrange(98.6, [32, 212], [0, 100])  # 37.0
    """
    oldrange = oldminmax[1] - oldminmax[0]
    newrange = newminmax[1] - newminmax[0]
    return (newrange / oldrange) * (np.asarray(oldval) - oldminmax[0]) + newminmax[0]


def rescale_data_range(
    array_in: npt.ArrayLike,
    newmin: float = 0,
    newmax: float = 1,
    axis: int | None = None,
) -> npt.NDArray[np.float64]:
    """
    Rescale data to a new range via min-max normalization.

    Parameters
    ----------
    array_in : array-like
    newmin, newmax : float
    axis : None or int

    Returns
    -------
    arr_rescaled : ndarray
    """
    arr = np.asarray(array_in, dtype=float)
    # keepdims so a per-axis min/max broadcasts back against ``arr`` for
    # any ``axis`` (a bare reduction would fail to broadcast on axis != 0).
    amin = np.nanmin(arr, axis=axis, keepdims=True)
    amax = np.nanmax(arr, axis=axis, keepdims=True)
    return newmin + (arr - amin) * (newmax - newmin) / (amax - amin)


# ---- Angle wrapping ----

def wrap_range(
    valin: npt.ArrayLike, range_lo: float, range_hi: float
) -> npt.NDArray[np.float64]:
    """Wrap values to the range [range_lo, range_hi]."""
    valin = np.asarray(valin)
    return np.mod(valin - range_lo, range_hi - range_lo) + range_lo

def wrap_center_pmrange(
    valin: npt.ArrayLike, center_val: float, pm_range: float
) -> npt.NDArray[np.float64]:
    """Wrap values to center_val +/- pm_range."""
    valin = np.asarray(valin)
    return np.mod(valin - (center_val - pm_range), 2 * pm_range) + (center_val - pm_range)

def wrap_360(valin: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Wrap to [0, 360]."""
    return wrap_range(valin, 0., 360.)

def wrap_pm180(valin: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Wrap to [-180, +180]."""
    return wrap_center_pmrange(valin, 0., 180.)

def wrap_pm90(valin: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Wrap to [-90, +90]."""
    return wrap_center_pmrange(valin, 0., 90.)

def wrap_pmPI(valin: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Wrap to [-pi, +pi]."""
    return wrap_center_pmrange(valin, 0., np.pi)


def wrap_24hr(timein: npt.ArrayLike, component: bool = False) -> npt.NDArray[np.float64]:
    """Wrap a time value to the [0, 24] hour range."""
    if component:
        from .coords import dms2deg  # lazy import to avoid circularity risk
        # component=True is documented to take DMS/HMS string or [h, m, s]
        # components, which dms2deg parses; the broader ArrayLike param
        # type covers both this and the numeric pass-through path below.
        timein = dms2deg(timein)  # type: ignore[arg-type]
    return wrap_range(timein, 0., 24.)
