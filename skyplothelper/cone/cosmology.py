"""Redshift -> radial-coordinate conversions for cone wedge plots.

Used by ``cone_scatter_z`` and ``make_cone_frame`` when the radial
variable is comoving distance or lookback time. astropy is an optional
soft dependency; if not installed only ``r_variable="redshift"`` works.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import astropy.units as u
    _HAVE_ASTROPY = True
except ImportError:
    u = None
    _HAVE_ASTROPY = False


_R_VARIABLES = ('redshift', 'comoving_distance', 'lookback_time')


def redshift_to_r(z: npt.ArrayLike, r_variable: str = 'redshift',
                  cosmology: Any = None,
                  r_unit: str = 'Mpc') -> npt.NDArray[np.float64]:
    """
    Convert a redshift array to the requested radial variable.

    Parameters
    ----------
    z : array_like
        Redshift values.
    r_variable : {'redshift', 'comoving_distance', 'lookback_time'}
        Which radial quantity to return. ``'redshift'`` is the identity.
    cosmology : astropy.cosmology instance or None
        Required for any conversion other than ``'redshift'``. E.g.
        ``astropy.cosmology.Planck18``.
    r_unit : str or astropy.units.Unit
        Output unit for distance (``'Mpc'``, ``'Gpc'``, ...) or time
        (``'Gyr'``, ``'Myr'``, ...). Ignored when ``r_variable='redshift'``.

    Returns
    -------
    r : ndarray
        Converted values as a plain numpy array (unitless floats).
    """
    z = np.asarray(z, dtype=float)
    if r_variable == 'redshift':
        return z
    if cosmology is None:
        raise ValueError(
            f"redshift_to_r: r_variable='{r_variable}' requires a cosmology "
            "(e.g. `from astropy.cosmology import Planck18; "
            "cosmology=Planck18`).")
    if not _HAVE_ASTROPY:
        raise ImportError(
            "redshift_to_r: astropy is required for cosmology conversions.")
    if r_variable == 'comoving_distance':
        q = cosmology.comoving_distance(z).to(r_unit)
    elif r_variable == 'lookback_time':
        q = cosmology.lookback_time(z).to(r_unit)
    else:
        raise ValueError(
            f"Unknown r_variable={r_variable!r}; use one of {_R_VARIABLES}.")
    return q.value


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------

