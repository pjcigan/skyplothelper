"""Vector Spherical Harmonics (VSH) forward model.

Evaluate the VSH vector field at sky positions given the VSH
coefficients — the *forward* direction (coefficients → field), the
companion to the sky-vector renderers :func:`skyplothelper.plot_sky_vectors`
and :func:`skyplothelper.plotly.add_sky_vectors`, which draw a
pre-computed ``(dlon, dlat)`` field but don't generate one. (VSH
*fitting* — field → coefficients — is out of scope here.)

The real-valued formulation follows Mignard & Klioner 2012,
Titov+ 2011/2013, and Charlot+ 2020 (ICRF3). Two degrees are
implemented — the canonical 16-parameter set:

* **ℓ = 1** — a rigid rotation ``(R_1, R_2, R_3)`` plus a glide
  ``(D_1, D_2, D_3)`` (the dipole / spin+glide that dominate frame
  comparisons).
* **ℓ = 2** — the quadrupole, as poloidal ("electric") ``E_2m`` and
  toroidal ("magnetic") ``M_2m`` terms with real / imaginary parts.

The field components, with ``α`` = longitude (RA) and ``δ`` =
latitude (Dec)::

    Δα·cosδ =
        R_1·cosα·sinδ + R_2·sinα·sinδ − R_3·cosδ
      − D_1·sinα + D_2·cosα
      + M_20·sin(2δ)
      + (E_21_Re·sinα + E_21_Im·cosα)·sinδ
      − (M_21_Re·cosα − M_21_Im·sinα)·cos(2δ)
      − 2·(E_22_Re·sin(2α) + E_22_Im·cos(2α))·cosδ
      − (M_22_Re·cos(2α) − M_22_Im·sin(2α))·sin(2δ)

    Δδ =
      − R_1·sinα + R_2·cosα
      − D_1·cosα·sinδ − D_2·sinα·sinδ + D_3·cosδ
      + E_20·sin(2δ)
      − (E_21_Re·cosα − E_21_Im·sinα)·cos(2δ)
      − (M_21_Re·sinα + M_21_Im·cosα)·sinδ
      − (E_22_Re·cos(2α) − E_22_Im·sin(2α))·sin(2δ)
      + 2·(M_22_Re·sin(2α) + M_22_Im·cos(2α))·cosδ

The amplitudes carry whatever angular unit the caller uses (µas in
ICRF3 work); the trigonometric factors are dimensionless, so the
returned field is in the *same* unit as the input coefficients. The
module is pure-numpy by design — no astropy / lmfit / pandas — so an
interactive demo can recompute the field cheaply on every slider move.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

# Canonical 16-parameter order (Mignard / Charlot ICRF3). The first six
# are the ℓ=1 rotation + glide; the remaining ten are the ℓ=2 quadrupole.
VSH_PARAM_NAMES = (
    'R_1', 'R_2', 'R_3',
    'D_1', 'D_2', 'D_3',
    'E_20', 'M_20',
    'E_21_Re', 'E_21_Im', 'M_21_Re', 'M_21_Im',
    'E_22_Re', 'E_22_Im', 'M_22_Re', 'M_22_Im',
)


def _resolve_params(params: Sequence[float] | dict[str, float] | npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce ``params`` to a length-16 float array in canonical order.

    Accepts a length-6 sequence (ℓ=1 rotation + glide; ℓ=2 zero-filled),
    a length-16 sequence (full ℓ=1 + ℓ=2), or a dict keyed by
    :data:`VSH_PARAM_NAMES` (missing keys default to 0).
    """
    if isinstance(params, dict):
        return np.array([float(params.get(n, 0.0)) for n in VSH_PARAM_NAMES])
    arr = np.asarray(params, dtype=float).ravel()
    if arr.size == 6:
        full = np.zeros(16)
        full[:6] = arr
        return full
    if arr.size == 16:
        return arr
    raise ValueError(
        "params must be length 6 (R, D), length 16 (R, D + ℓ=2 "
        "quadrupole), or a dict keyed by VSH_PARAM_NAMES; got length "
        f"{arr.size}")


def vsh_field(
    lon: npt.ArrayLike, lat: npt.ArrayLike,
    params: Sequence[float] | dict[str, float] | npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluate the VSH vector field at sky positions.

    Parameters
    ----------
    lon, lat : float or array-like
        Sky positions (RA / Dec, or any lon / lat) in **degrees**.
    params : sequence or dict
        VSH coefficients. A length-6 sequence
        ``(R_1, R_2, R_3, D_1, D_2, D_3)`` for rotation + glide only, a
        length-16 sequence adding the ℓ=2 quadrupole (order =
        :data:`VSH_PARAM_NAMES`), or a dict keyed by those names.

    Returns
    -------
    dlon_coslat, dlat : ndarray
        The longitudinal and latitudinal field components, ``Δα·cosδ``
        and ``Δδ``, in the **same angular units as** ``params``. The
        ``Δα·cosδ`` convention means the pair feeds the sky-vector
        renderers directly as ``dlon`` / ``dlat`` with ``cos_dec=True``
        (the default).

    Examples
    --------
    Zero-length arrows that grow as a rotation ``R3`` is dialed in::

        import numpy as np
        import skyplothelper as sph
        lon = np.repeat(np.arange(0, 360, 30), 5)
        lat = np.tile(np.arange(-60, 61, 30), 12)
        dlon, dlat = sph.vsh_field(lon, lat, [0, 0, R3, 0, 0, 0])
        sph.plotly.add_sky_vectors(fig, lon, lat, dlon, dlat)
    """
    (R1, R2, R3, D1, D2, D3, E20, M20,
     E21Re, E21Im, M21Re, M21Im,
     E22Re, E22Im, M22Re, M22Im) = _resolve_params(params)

    a = np.radians(np.asarray(lon, dtype=float))
    d = np.radians(np.asarray(lat, dtype=float))
    sa, ca = np.sin(a), np.cos(a)
    sd, cd = np.sin(d), np.cos(d)
    s2a, c2a = np.sin(2 * a), np.cos(2 * a)
    s2d, c2d = np.sin(2 * d), np.cos(2 * d)

    dlon_coslat = (
        R1 * ca * sd + R2 * sa * sd - R3 * cd
        - D1 * sa + D2 * ca
        + M20 * s2d
        + (E21Re * sa + E21Im * ca) * sd
        - (M21Re * ca - M21Im * sa) * c2d
        - 2 * (E22Re * s2a + E22Im * c2a) * cd
        - (M22Re * c2a - M22Im * s2a) * s2d
    )
    dlat = (
        - R1 * sa + R2 * ca
        - D1 * ca * sd - D2 * sa * sd + D3 * cd
        + E20 * s2d
        - (E21Re * ca - E21Im * sa) * c2d
        - (M21Re * sa + M21Im * ca) * sd
        - (E22Re * c2a - E22Im * s2a) * s2d
        + 2 * (M22Re * s2a + M22Im * c2a) * cd
    )
    return dlon_coslat, dlat


def vsh_shift_sources(
    lon: npt.ArrayLike, lat: npt.ArrayLike,
    params: Sequence[float] | dict[str, float] | npt.ArrayLike,
    scale: float = 1.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Displace sky source positions by the VSH field.

    The primary "apply the VSH field as a coordinate offset" routine —
    use it to rotate a whole catalog/frame (pass the catalog positions)
    or any specified set of sources. Pass small ``params`` with a large
    ``scale`` to exaggerate an otherwise-invisible (e.g. µas-level)
    deformation. See :func:`vsh_shift_frame` for the convenience that
    generates a uniform grid and shifts it in one call.

    The ``Δα·cosδ`` component is added straight to the longitude (the
    convention of the reference ``VSH_shift``), so the displacement is the
    on-sky angular offset rather than a raw Δlon — exact at the equator and
    a good visual approximation elsewhere for the small shifts these demos
    use.

    Parameters
    ----------
    lon, lat : float or array-like
        Sky positions in degrees.
    params : sequence or dict
        VSH coefficients (see :func:`vsh_field`).
    scale : float
        Multiplier applied to the field before it is added to the
        (degree-valued) coordinates. The field comes out in the **same
        units as** ``params`` (the harmonics are dimensionless), so
        ``scale`` is what carries it into degrees — set it to "how many
        degrees is one unit of your parameters":

        * params already in **degrees** → ``scale=1`` (the default).
        * params in **arcseconds** → ``scale=1/3600`` (``= 1/(60*60)``).
        * params in **µas** → ``scale=1/(3600e6)``.

        For *visualization* you will usually pass a deliberately larger
        ``scale`` than the faithful conversion, to exaggerate an
        otherwise-imperceptible (e.g. µas-level) deformation into something
        visible on the plot. Default ``1.0``.

    Returns
    -------
    lon_shifted, lat_shifted : ndarray
        Displaced positions in degrees — longitude wrapped to
        ``[0, 360)`` and latitude clipped to ``[-90, 90]``.

    Examples
    --------
    Apply a *true* 1-arcsecond polar spin (R₃) to a small catalog whose VSH
    parameters are given in arcseconds — ``scale=1/3600`` converts the
    arcsecond-valued field into the degree coordinates::

        import numpy as np
        import skyplothelper as sph
        ra = np.array([10.0, 80.0, 200.0])
        dec = np.array([-30.0, 0.0, 45.0])
        ra2, dec2 = sph.vsh_shift_sources(
            ra, dec, [0, 0, 1.0, 0, 0, 0], scale=1 / 3600)

    Exaggerate a small glide so the shift is visible on an all-sky plot
    (parameters treated as degrees and amplified 5×)::

        ra2, dec2 = sph.vsh_shift_sources(
            ra, dec, [0, 0, 0, 0, 2.0, 1.0], scale=5.0)
    """
    dlon_coslat, dlat = vsh_field(lon, lat, params)
    lon_shifted = (np.asarray(lon, dtype=float) + scale * dlon_coslat) % 360.0
    lat_shifted = np.clip(np.asarray(lat, dtype=float) + scale * dlat,
                          -90.0, 90.0)
    return lon_shifted, lat_shifted


def vsh_shift_frame(
    params: Sequence[float] | dict[str, float] | npt.ArrayLike,
    n_lon: int = 24, n_lat: int = 11, scale: float = 1.0, lat_max: float = 85.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
           npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate a uniform grid of sky sources and shift it by the VSH field.

    Convenience wrapper over :func:`vsh_shift_sources` for visualizing how
    a VSH parameter set deforms a *whole frame*: it lays down a regular
    lon/lat grid of "sources" and returns both the original and shifted
    positions, ready to draw as displacement arrows
    (:func:`skyplothelper.plot_displacement`) or — together with
    :func:`vsh_field` on the returned grid — as a vector field
    (:func:`skyplothelper.plot_sky_vectors`).

    Parameters
    ----------
    params : sequence or dict
        VSH coefficients (see :func:`vsh_field`).
    n_lon, n_lat : int
        Grid resolution: ``n_lon`` longitudes evenly spaced over
        ``[0, 360)`` and ``n_lat`` latitudes over ``[-lat_max, lat_max]``.
        Defaults ``24`` × ``11``.
    scale : float
        Multiplier carrying the field into degrees before displacing — the
        unit-conversion / exaggeration knob documented under
        :func:`vsh_shift_sources` (e.g. ``scale=1/3600`` for parameters in
        arcseconds, or a large value to exaggerate a tiny deformation).
        Default ``1.0``.
    lat_max : float
        Latitude cap for the grid, in degrees — kept below 90° so the
        grid doesn't pile up at the poles. Default ``85``.

    Returns
    -------
    lon, lat, lon_shifted, lat_shifted : ndarray
        Flattened original grid positions and their shifted counterparts,
        all in degrees.

    Examples
    --------
    Visualize how a rotation + glide deforms a whole frame, as before→after
    displacement arrows::

        import matplotlib.pyplot as plt
        import skyplothelper as sph
        ax = sph.make_wcs_frame(111, projection="AIT", center=180)
        lon, lat, lon2, lat2 = sph.vsh_shift_frame(
            [3, -2, 6, 0, 5, 2], n_lon=18, n_lat=9, scale=1.0)
        sph.plot_displacement(ax, lon, lat, lon2, lat2, color="C2")
        plt.show()

    Or show the same deformation as a vector field by evaluating
    :func:`vsh_field` on the returned grid::

        params = [3, -2, 6, 0, 5, 2]
        lon, lat, _, _ = sph.vsh_shift_frame(params)
        dlon, dlat = sph.vsh_field(lon, lat, params)
        sph.plot_sky_vectors(ax, lon, lat, dlon, dlat,
                             units="deg", scale="auto")
    """
    lons = np.linspace(0.0, 360.0, int(n_lon), endpoint=False)
    lats = np.linspace(-float(lat_max), float(lat_max), int(n_lat))
    mesh_lon, mesh_lat = np.meshgrid(lons, lats)
    flat_lon = mesh_lon.ravel()
    flat_lat = mesh_lat.ravel()
    lon_shifted, lat_shifted = vsh_shift_sources(
        flat_lon, flat_lat, params, scale=scale)
    return flat_lon, flat_lat, lon_shifted, lat_shifted
