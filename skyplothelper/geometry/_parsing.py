"""SkyCoord / Quantity input helpers (internal).

Duck-typed helpers that detect SkyCoord and astropy Quantity values without
requiring astropy at import time. Used across the geometry module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from astropy.wcs import WCS

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time and stays out of the import graph.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord


def _wcs_frame(wcs: WCS | None) -> str:
    """Detect the native coordinate frame of a WCS from its CTYPE.

    Returns one of ``'icrs'``, ``'galactic'``, or
    ``'geocentrictrueecliptic'``.
    """
    if wcs is None:
        return 'icrs'
    try:
        ctype1 = wcs.wcs.ctype[0][:4].upper()
    except Exception:
        return 'icrs'
    if ctype1 == 'GLON':
        return 'galactic'
    elif ctype1 == 'ELON':
        return 'geocentrictrueecliptic'
    return 'icrs'  # 'RA--' or any equatorial system


def _spherical_deg(coord: SkyCoord) -> tuple[Any, Any]:
    """Frame-agnostic ``(lon, lat)`` degrees from a SkyCoord.

    Uses the spherical representation rather than frame-specific attribute
    names (``ra``/``dec`` vs ``l``/``b`` vs ``lon``/``lat``), so it works for
    any frame — including ones with no hard-coded branch here.
    """
    s = coord.spherical
    return s.lon.deg, s.lat.deg


def _resolve_sky_frame(frame_name: str | None) -> str:
    """sph frame name/alias → the astropy frame to transform into.

    Accepts both the canonical names reported by ``_get_wcs_frame_name(ax)`` /
    ``Projector.wcs_frame`` (``'galactic'``, ``'ecliptic'``, …) and the looser
    user-facing strings the frame builders take (``'gal'``, ``'Galactic'``,
    ``'ecl'``, …).

    Substring matching deliberately mirrors ``wcs_frame._resolve_ctype`` — the
    two must agree, or a frame built from one vocabulary gets its coordinates
    converted with the other. **Order matters:** ``'supergalactic'`` contains
    ``'gal'``, so ``'super'`` is tested first, exactly as ``_resolve_ctype``
    does. (Resolved locally rather than importing ``_resolve_ctype``: this
    module is imported by ``ticks``, which ``wcs_frame`` imports in turn, so
    the import would close a cycle.)

    Body-fixed / non-sky frames (``'itrs'``, ``'heliographic'``) and anything
    unrecognized fall back to ICRS, preserving long-standing globe behavior —
    a celestial SkyCoord has no meaningful transform into them without an
    obstime.
    """
    fr = (frame_name or 'icrs').lower()
    if 'super' in fr:
        return 'supergalactic'
    if 'gal' in fr:
        return 'galactic'
    if 'ecl' in fr:
        return 'geocentrictrueecliptic'
    if fr in ('fk5', 'fk4'):
        return fr
    return 'icrs'


def _coords_to_frame_deg(coords: SkyCoord, frame_name: str | None,
                         ) -> tuple[Any, Any]:
    """SkyCoord → ``(lon_deg, lat_deg)`` degrees in the named sph frame.

    The single place user coordinates get put into an axes' native frame.
    Centralized because every module that needs this was hand-rolling it, and
    the copies drifted: modules that dispatched on the frame were correct,
    modules that reached for ``.icrs`` instead silently mis-projected on
    galactic / ecliptic axes. Backends stay responsible only for reporting
    their frame *name*; the conversion itself lives here.
    """
    return _spherical_deg(coords.transform_to(_resolve_sky_frame(frame_name)))


def _coords_or_arrays_deg(lons_or_coord: SkyCoord | npt.ArrayLike,
                          lats: npt.ArrayLike | None = None,
                          frame_name: str | None = None,
                          caller: str = 'this function',
                          ) -> tuple[Any, Any]:
    """Accept a SkyCoord array (with *lats* omitted) OR two array-likes.

    The uniform "SkyCoord in the first coordinate slot" form for the array
    plotting/binning helpers. Deliberately not the one-argument positional
    shift used by the scalar shape builders: these callers have trailing
    positional parameters of their own, so a shift would misbind them.

    ``frame_name=None`` means *preserve the SkyCoord's own frame* — the right
    contract for helpers with no axes/frame context (the HEALPix binning
    family works in whatever frame the caller's data is in). Pass a frame name
    when there is one to convert into.
    """
    if hasattr(lons_or_coord, 'transform_to'):  # SkyCoord duck-type
        if lats is not None:
            raise TypeError(
                f"{caller}: a SkyCoord replaces BOTH coordinate arguments, so "
                "the latitude slot must be left empty. If you meant to pass a "
                "following argument positionally, give it by keyword instead "
                "(e.g. f(coords, nside=64) rather than f(coords, 64)).")
        if frame_name is None:
            return _spherical_deg(lons_or_coord)
        return _coords_to_frame_deg(lons_or_coord, frame_name)
    if lats is None:
        raise TypeError(
            f"{caller}: needs either a SkyCoord or both lon and lat arrays.")
    return (np.asarray(lons_or_coord, dtype=float),
            np.asarray(lats, dtype=float))


def _parse_coord(lon_or_coord: SkyCoord | float, lat: float | None = None,
                 wcs: WCS | None = None,
                 preserve_frame: bool = False,
                 frame_name: str | None = None,
                 ) -> tuple[float, float, bool]:
    """Accept ``(lon, lat)`` floats or a SkyCoord → ``(lon_deg, lat_deg, shifted)``.

    When a SkyCoord is passed as the first argument, the value in *lat*
    is actually the next positional parameter (e.g. radius, width).
    The *shifted* flag tells the caller to re-interpret *lat*'s
    original value as that next parameter.

    SkyCoord inputs are converted to the WCS native frame.  Raw float
    inputs are always assumed to be in the WCS native frame (no
    conversion applied), matching the existing convention where floats go
    directly into ``wcs.world_to_pixel_values()``.

    ``preserve_frame`` is for callers with **no WCS context** — the bare
    vertex builders (``geodesic_circle``/``rectangle``/``ellipse``), which just
    return coordinate arrays. Without it a non-ICRS SkyCoord would be silently
    converted to ICRS (``_wcs_frame(None)`` is ``'icrs'``), so asking for a
    circle around a galactic position returned one centered somewhere else
    entirely. With it, output frame == input frame. It is deliberately opt-in:
    callers that legitimately work in ICRS (``catalog.py``) rely on the
    coercion, so the default must not change.
    """
    if hasattr(lon_or_coord, 'transform_to'):  # SkyCoord duck-type
        # frame_name wins when given: a backend may know its frame without
        # having a WCS object at all (the plotly projector carries a frame
        # STRING), in which case falling back on the WCS would silently mean
        # ICRS. Otherwise a WCS wins, and preserve_frame decides what "no WCS"
        # means.
        if frame_name is not None:
            c = lon_or_coord.transform_to(_resolve_sky_frame(frame_name))
        elif wcs is None and preserve_frame:
            c = lon_or_coord
        else:
            c = lon_or_coord.transform_to(_wcs_frame(wcs))
        if not c.isscalar:
            raise ValueError(
                "Scalar SkyCoord expected for center position; "
                "use _parse_coords() for arrays")
        lon_deg, lat_deg = _spherical_deg(c)
        return float(lon_deg), float(lat_deg), True
    if lat is None:
        raise ValueError("lat is required when lon is not a SkyCoord")
    return float(lon_or_coord), float(lat), False


def _parse_coords(lons_or_coord: SkyCoord | npt.ArrayLike,
                  lats: npt.ArrayLike | None = None,
                  wcs: WCS | None = None, preserve_frame: bool = False,
                  frame_name: str | None = None,
                  ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Accept ``(lons, lats)`` arrays or a SkyCoord array → ``(lons_deg, lats_deg)``.

    Converts to the WCS native frame, same as :func:`_parse_coord`.
    No shift flag needed — array SkyCoord replaces both positional args.
    ``preserve_frame`` has the same meaning as in :func:`_parse_coord`.
    """
    if hasattr(lons_or_coord, 'transform_to'):  # SkyCoord duck-type
        if frame_name is not None:
            c = lons_or_coord.transform_to(_resolve_sky_frame(frame_name))
        elif wcs is None and preserve_frame:
            c = lons_or_coord
        else:
            c = lons_or_coord.transform_to(_wcs_frame(wcs))
        lon_deg, lat_deg = _spherical_deg(c)
        return np.asarray(lon_deg, float), np.asarray(lat_deg, float)
    if lats is None:
        raise ValueError("lats required when lons is not a SkyCoord")
    return np.asarray(lons_or_coord, float), np.asarray(lats, float)


def _parse_angle(value: Any) -> float | None:
    """Accept a float (degrees) or astropy Quantity → float degrees."""
    if value is None:
        return None
    if hasattr(value, 'to'):  # Quantity duck-type
        return float(value.to('deg').value)
    return float(value)
