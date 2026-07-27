"""Pure-numpy ``(lon, lat) → (x, y)`` projection helper.

Single public function :func:`project` that turns a pair of sky-
coordinate arrays into 2D projected coordinates in degrees of the
projection plane, suitable for direct use in any plot library that
doesn't already have skyplothelper's WCSAxes machinery hooked up —
notably the :mod:`skyplothelper.plotly` web-export submodule, but also
plain matplotlib axes, bokeh, or custom-rendered output (SVG, etc.).

The function is the single source of truth for ``(lon, lat) → (x, y)``
projection in skyplothelper. Both the FITS-WCS path (AIT, MOL, SIN,
TAN, CAR, MER, SFL, PAR, BON, CEA, CYP, STG, ARC, ZEA, …) and the
custom-mpl-transform projections (robinson, kavrayskiy, mcbryde,
winkel_tripel, eckert4) route through here, and any future export
backend reuses the same primitive rather than re-implementing
projection math.

Output convention: ``(x, y)`` in degrees of intermediate-world
projection-plane. Each projection has its own natural extent
(``CAR``: ``x ∈ [-180, +180]``; ``AIT``: ``x ∈ [-114.6, +114.6]``
at the equator; Robinson: ``x ∈ [-152.85, +152.85]``; etc.) so
downstream code is expected to set axis limits per-projection.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord


from ._math import (
    _eckert4_forward,
    _kavrayskiy_forward,
    _mcbryde_forward,
    _robinson_forward,
    _winkel_forward,
)

__all__ = ['project']


# Largest projection-plane magnitude (degrees) treated as meaningful.
# The widest *bounded* projection here spans ~404° (MER sampled to lat 89.9),
# so nothing a plot could use comes close; a coordinate past this lies within
# ~0.002° of a projection's singularity, where the value is floating-point
# noise rather than geometry. See ``_mask_singularities``.
_MAX_PLANE_COORD_DEG = 1e6


def _mask_singularities(x: np.ndarray,
                        y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN out points that land on a projection's singularity.

    wcslib does not fail there. ``cos(90°)`` evaluates to ``6.1e-17`` rather
    than ``0``, so a point sitting exactly on the gnomonic horizon — or on a
    perspective conic's divergence — comes back as a finite magnitude around
    ``1e17``–``1e19`` instead of the ``NaN`` returned just beyond it. On TAN
    the entire meridian at ``center ± 90`` lies on that horizon, and wcslib
    returns a mixture of ``~1e19`` and ``NaN`` along it.

    Such values are noise, and they poison whatever consumes them: an
    autoscaled axis collapses to a point, and a projected polygon fed to
    shapely acquires vertices ``1e19`` away.

    The mask is applied per *point*, not per coordinate. An overflowed ``x``
    leaves its partner ``y`` finite and small (the horizon meridian yields
    pairs like ``(-9.4e17, 57.3)``), and that ``57.3`` is every bit as
    meaningless as its partner — so a pair is dropped whole.
    """
    bad = (~np.isfinite(x) | ~np.isfinite(y)
           | (np.abs(x) > _MAX_PLANE_COORD_DEG)
           | (np.abs(y) > _MAX_PLANE_COORD_DEG))
    if not bad.any():
        return x, y
    return np.where(bad, np.nan, x), np.where(bad, np.nan, y)


# Projection codes that need a zenithal (non-zero ``lat_center``)
# dummy header rather than the cylindrical / pseudocylindrical
# ``dummy_allsky_hdr`` default. Same set the dispatcher uses.
_ZENITHAL_FITS_CODES = frozenset({'SIN', 'TAN', 'ARC', 'ZEA', 'STG',
                                   'AZP', 'SZP', 'NCP', 'AIR'})

# skyplothelper-extended custom-transform projections. Names are
# normalized to lowercase here.
_CUSTOM_FORWARDS = {
    'robinson':      _robinson_forward,
    'kavrayskiy':    _kavrayskiy_forward,
    'mcbryde':       _mcbryde_forward,
    'winkel_tripel': _winkel_forward,
    'eckert4':       _eckert4_forward,
}


# Longitude-orientation aliases, resolved to the two canonical values
# ('sky' = lon/RA increasing leftward, the astronomical "looking out"
# convention; 'geographic' = lon increasing rightward, the cartographic /
# terrestrial-globe convention). Case-insensitive; shared by every entry
# point that takes a ``direction`` (project, make_figure, make_wcs_frame,
# make_globe_frame) so the vocabulary is uniform across backends.
_DIRECTION_ALIASES = {
    'sky': 'sky', 'astro': 'sky', 'astronomical': 'sky',
    'geographic': 'geographic', 'geo': 'geographic',
    'earth': 'geographic', 'cartographic': 'geographic',
}


def resolve_direction(direction: str) -> str:
    """Normalize a longitude-orientation name to ``'sky'`` or
    ``'geographic'``, accepting intuitive aliases (``'astro'``, ``'geo'``,
    ``'earth'``, …). Case-insensitive. Raises ``ValueError`` on an
    unknown value."""
    key = str(direction).strip().lower()
    try:
        return _DIRECTION_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"direction must be one of {sorted(_DIRECTION_ALIASES)} "
            f"(all aliases for 'sky' / 'geographic'), got {direction!r}")


# Longitude tick-unit aliases, resolved to 'auto' / 'hours' / 'degrees'.
# 'auto' defers to the frame + direction policy of whatever consumes it
# (equatorial sky → hours; geographic / galactic / ecliptic → degrees);
# 'hours' / 'degrees' force the unit. Shared by the mpl frame builders and
# the plotly figure so the vocabulary is uniform across backends. Kept here
# (numpy-only module) so the plotly side need not import matplotlib.
_LON_UNITS_ALIASES = {
    'auto': 'auto',
    'hours': 'hours', 'hour': 'hours', 'hms': 'hours', 'h': 'hours',
    'degrees': 'degrees', 'degree': 'degrees', 'deg': 'degrees', 'd': 'degrees',
}


def resolve_lon_units(lon_units: str) -> str:
    """Normalize a longitude-units choice to ``'auto'`` / ``'hours'`` /
    ``'degrees'`` (case-insensitive; accepts ``hms`` / ``h`` and ``deg`` /
    ``d``). Raises ``ValueError`` on an unknown value."""
    key = str(lon_units).strip().lower()
    try:
        return _LON_UNITS_ALIASES[key]
    except KeyError:
        raise ValueError(
            "lon_units must be 'auto', 'hours', or 'degrees' (or aliases "
            f"hms/h, deg/d), got {lon_units!r}")


def _wrap_lon(lons: npt.ArrayLike, center: float) -> npt.NDArray[np.float64]:
    """Shift longitudes into ``[center - 180, center + 180]`` so that
    the projection center sits at ``0`` in the natural frame the custom
    forward functions expect (which assume the meridian-of-zero is the
    projection center)."""
    lons = np.asarray(lons, dtype=float)
    return ((lons - center + 180.0) % 360.0) - 180.0


@functools.lru_cache(maxsize=128)
def _cached_projection_wcs(proj_upper: str, center: float, lat_center: float,
                           pv2_1: float | None, pv2_2: float | None) -> Any:
    """Build (and memoize) the astropy WCS for a projection's parameters.

    The WCS + its FITS header depend only on the projection and center, NOT on
    the coordinates being projected, so callers that ``project()`` many small
    batches against the same frame (e.g. per-segment constellation boundaries,
    wrap-splitting) reuse one WCS instead of rebuilding a header + WCS tens of
    thousands of times. ``wcs_world2pix`` is read-only, so sharing the cached
    object across calls is safe.

    Lazy imports dodge the circular dependency through ``wcs_frame`` (which
    pulls ticks → projections.frames) and keep the entry point cheap to import.
    """
    from astropy.wcs import WCS

    from ..wcs_frame import dummy_allsky_hdr, dummy_ortho_hdr

    # For zenithal projections we force ``LONPOLE=180`` (the FITS standard for
    # non-pole CRVAL2 with ``delta_0 < theta_0``), matching ``make_wcs_frame``'s
    # SIN orientation. (This differs from ``make_globe_frame``'s ``lonpole=0``,
    # which rotates the orthographic disk — a tilted-earth view, not the default
    # ``project()`` behavior.) Conic / Bonne codes route through the else branch
    # and need a standard parallel, which ``dummy_allsky_hdr`` supplies (45°).
    if proj_upper in _ZENITHAL_FITS_CODES:
        hdr = dummy_ortho_hdr(center_LONdeg=center, center_LATdeg=lat_center,
                              projection=proj_upper, lonpole=180.0)
    else:
        hdr = dummy_allsky_hdr(center_LONdeg=center, projection=proj_upper,
                               pv2_1=pv2_1, pv2_2=pv2_2)
    return WCS(hdr)


def project(lons: SkyCoord | npt.ArrayLike, lats: npt.ArrayLike | None = None, projection: str = 'AIT',
            center: float = 0.0, lat_center: float = 0.0,
            frame: str | None = None,
            direction: str = 'sky',
            pv2_1: float | None = None,
            pv2_2: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Project ``(lon, lat)`` arrays to 2D ``(x, y)`` coords.

    Pure-numpy entry point — no figure, axes, or matplotlib state is
    touched. Returns plain ``ndarray`` outputs that the caller can
    drop into any plot library.

    Parameters
    ----------
    lons, lats : array-like, or SkyCoord in ``lons``
        A ``SkyCoord`` may be passed as ``lons``, replacing both AND supplying
        the source frame — ``frame=`` then becomes unnecessary.
        Longitude and latitude arrays in degrees. Any shape; outputs
        match the input shape.
    projection : str, optional
        Projection code. Either a FITS WCS code understood by astropy
        (``'AIT'`` / ``'MOL'`` / ``'SIN'`` / ``'TAN'`` / ``'CAR'`` /
        ``'MER'`` / ``'SFL'`` / ``'PAR'`` / ``'BON'`` / ``'CEA'`` /
        ``'CYP'`` / ``'STG'`` / ``'ARC'`` / ``'ZEA'`` / ...) or a
        skyplothelper-extended custom-transform name (``'robinson'``,
        ``'kavrayskiy'``, ``'mcbryde'``, ``'winkel_tripel'``,
        ``'eckert4'``). Case-insensitive. Default ``'AIT'``.
    center : float, optional
        Longitude (degrees) of the projection center. Default 0.
    lat_center : float, optional
        Latitude (degrees) of the projection center. Only meaningful
        for zenithal projections (SIN / TAN / ARC / ZEA / STG / ...)
        — cylindrical and pseudocylindrical projections always center
        at the equator and ignore this. Default 0.
    frame : str or None, optional
        Source coordinate frame. If given (e.g. ``'icrs'``,
        ``'galactic'``, ``'fk5'``), the input ``(lons, lats)`` are
        transformed from this frame into the projection's native frame
        before projecting. ``None`` (default) means input coordinates
        are already in the projection's native frame.
    direction : str, optional
        Orientation convention for the x-axis. ``'sky'`` (default):
        longitude / RA increases to the *left* on the plot (the
        astronomical "looking outward" convention — RA=+90 lands on the
        left side of an all-sky plot). ``'geographic'``: longitude
        increases to the *right* (the standard cartographic convention
        for terrestrial maps). Intuitive aliases are accepted and
        resolved case-insensitively: ``'astro'`` / ``'astronomical'`` →
        ``'sky'``; ``'geo'`` / ``'earth'`` / ``'cartographic'`` →
        ``'geographic'``.
    pv2_1, pv2_2 : float, optional
        FITS PV2_1 / PV2_2 parameters for the conic (COD / COE / COO /
        COP) and Bonne (BON) projections, which are undefined without a
        standard parallel. ``pv2_1`` sets that parallel in degrees;
        ``pv2_2`` sets the spread between the two standard parallels for
        conics (BON ignores it). When unset, those five default to
        ``pv2_1=45``, matching :func:`skyplothelper.make_wcs_frame` so a
        conic renders identically through either backend. Silently
        ignored for every other projection.

    Returns
    -------
    x, y : ndarray
        Projected coordinates in degrees of the projection plane. Each
        projection has its own natural extent — set plot axis limits
        accordingly. Shape matches the input.

    Notes
    -----
    Output scale convention: degrees of intermediate-world projection-
    plane. For example:

    * ``CAR`` (Plate Carrée): ``x ∈ [-180, +180]``, ``y ∈ [-90, +90]``.
    * ``AIT`` (Aitoff): ``x ∈ [-114.6, +114.6]`` at the equator,
      curving inward to the poles.
    * ``MOL`` (Mollweide): ``x ∈ [-180, +180]`` at equator (Mollweide
      is "honest" along the equator), curving to the poles.
    * ``SIN`` (orthographic): disk of radius ``180/π ≈ 57.296°``
      centered at ``(0, 0)`` in projection space.
    * ``robinson``: ``x ∈ [-152.85, +152.85]`` at the equator.

    Points "behind" the projection (e.g. the back hemisphere of a SIN
    orthographic view) return ``NaN`` from the FITS WCS path. So do
    points sitting *on* a projection's singularity — the gnomonic (TAN)
    horizon, a perspective conic's divergence — where wcslib itself
    returns a finite but meaningless magnitude of order ``1e17``–``1e19``
    rather than failing. Both coordinates of such a point are ``NaN``,
    never one of the pair.

    The custom-transform forwards return finite (but visually wrong)
    values for out-of-range inputs; callers needing visibility
    filtering should consult :func:`orthographic_visibility` for
    zenithal cases.

    Examples
    --------
    >>> import numpy as np
    >>> from skyplothelper import project
    >>> lon = np.linspace(0, 360, 10)
    >>> lat = np.zeros_like(lon)
    >>> x, y = project(lon, lat, projection='AIT', center=180)
    >>> x.shape == lat.shape
    True

    Project galactic-frame coords onto an ICRS-frame map::

        x, y = project(gal_l, gal_b, projection='MOL', frame='galactic')
    """
    # A SkyCoord replaces both arrays AND supplies the source frame, so an
    # explicit frame= is unnecessary (and is honored if given).
    if hasattr(lons, 'transform_to'):
        from ..geometry._parsing import _spherical_deg
        # Bound to a local because `hasattr` does not narrow the parameter's
        # union for a type checker, and the duck-type check is deliberate --
        # the package detects SkyCoord without importing astropy here.
        coord: Any = lons
        if lats is not None:
            raise TypeError(
                "project: a SkyCoord replaces both coordinate arguments; pass "
                "later arguments by keyword (e.g. projection='MOL').")
        if frame is None:
            frame = coord.frame.name
        lons, lats = _spherical_deg(coord)
    elif lats is None:
        raise TypeError("project: needs a SkyCoord or both lons and lats.")
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if lons.shape != lats.shape:
        raise ValueError(
            f"lons and lats must have the same shape, got "
            f"{lons.shape} vs {lats.shape}")

    direction = resolve_direction(direction)
    # Sign factor applied to the x output. 'sky' convention has RA /
    # longitude increasing leftward (negative x for positive lon);
    # 'geographic' has longitude increasing rightward (positive x for
    # positive lon). The FITS WCS path natively produces 'geographic'
    # output before this flip; the custom-transform forwards also
    # produce 'geographic'.
    x_sign = -1.0 if direction == 'sky' else 1.0

    proj_key = str(projection).strip()
    proj_upper = proj_key.upper()
    proj_lower = proj_key.lower()

    # ----- Optional frame transformation -----
    if frame is not None:
        # Lazy import — frame conversion is only needed if the user
        # explicitly asked for it, and we'd rather not pay the astropy
        # SkyCoord import cost on every call.
        from astropy.coordinates import SkyCoord

        src = SkyCoord(lons, lats, unit='deg', frame=frame)
        # The destination frame is implied by the projection's typical
        # use — for now assume ICRS as the projection's native frame.
        # Callers that need a different destination frame can do the
        # SkyCoord transform themselves and pass ``frame=None``.
        dst = src.icrs
        lons = np.asarray(dst.ra.deg, dtype=float)
        lats = np.asarray(dst.dec.deg, dtype=float)

    # ----- Custom-transform projections (skyplothelper-extended) -----
    if proj_lower in _CUSTOM_FORWARDS:
        fwd = _CUSTOM_FORWARDS[proj_lower]
        # The custom forwards assume center=0 internally; pre-shift.
        lon_shifted = _wrap_lon(lons, center)
        x_nat, y_nat = fwd(lon_shifted, lats)
        # Custom forwards return projected coords in "radians of
        # natural scale" — multiply by 180/π to bring into degrees of
        # projection plane (consistent with the FITS path's output).
        scale = 180.0 / np.pi
        return (x_sign * np.asarray(x_nat) * scale,
                np.asarray(y_nat) * scale)

    # ----- FITS WCS path -----
    # The FITS header + WCS depend only on (projection, center, lat_center,
    # pv2_*), NOT on the data being projected — so they're memoized. Callers
    # that project() many small batches against the same frame (per-segment
    # constellation boundaries, wrap-splitting) then reuse one WCS instead of
    # rebuilding a header + WCS on every call. The dummy header's CDELT sets the
    # pixel scale; we convert pixel → intermediate-world (deg) below, so output
    # is independent of NAXIS / CDELT.
    wcs = _cached_projection_wcs(proj_upper, float(center), float(lat_center),
                                 pv2_1, pv2_2)
    coords = np.column_stack([lons.ravel(), lats.ravel()])
    pix = wcs.wcs_world2pix(coords, 0)
    # Convert pixel coords to intermediate-world (projection-plane
    # degrees) by subtracting the reference pixel and multiplying by
    # CDELT. The product (pix - crpix) * cdelt always yields the
    # 'geographic'-convention intermediate world (positive longitude
    # → positive x) regardless of the FITS-standard CDELT1 < 0 sign,
    # because the two factors carry compensating signs. Then apply
    # x_sign for the user's chosen direction.
    cdelt = wcs.wcs.cdelt
    crpix = wcs.wcs.crpix - 1.0   # FITS → 0-indexed
    x = (pix[:, 0] - crpix[0]) * cdelt[0]
    y = (pix[:, 1] - crpix[1]) * cdelt[1]
    x = x_sign * x
    # Only the FITS-WCS path needs this: the custom-transform forwards are
    # closed-form and bounded, and they deliberately return finite (if
    # visually wrong) values out of range — see the Notes above.
    x, y = _mask_singularities(x, y)
    return x.reshape(lons.shape), y.reshape(lons.shape)
