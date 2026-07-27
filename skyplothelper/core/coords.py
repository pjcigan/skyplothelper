"""Sexagesimal/decimal conversions, angular separation, and frame
conversion helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

# ---- Angle conversions ----

def _cascade_sexagesimal(deg: int, mins: int, sec: float,
                         decimal_places: int | None) -> tuple[int, int, float]:
    """Round *sec* and carry any 60 overflow up through minutes and degrees.

    Operates on non-negative magnitude components — the caller owns the
    sign. When *decimal_places* is an int the seconds are rounded to that
    many decimals first (the display path, so e.g. ``59.998`` at 2 dp
    becomes ``00`` with a minute carried, never ``60.00``); when ``None`` a
    float tolerance is used instead (the numeric-list path, so a float
    artifact like ``59.9999999999`` from ``-12.7 deg`` collapses to an
    exact ``0``).

    Shared by :func:`deg2dms` and :func:`dec2sex` so a sexagesimal value
    never displays ``60`` in any field, in either output mode.
    """
    if decimal_places is None:
        if sec >= 60 - 1e-9:
            sec = 0.0
            mins += 1
    else:
        sec = round(sec, decimal_places)
        if sec >= 60:
            sec -= 60
            mins += 1
    if mins >= 60:
        mins -= 60
        deg += 1
    return deg, mins, sec


def deg2dms(valin: float, return_type: Any = list, str_delimiter: str = ':',
            str_decimal_places: int = 2,
            str_zero_pad: bool = True) -> Any:
    """
    Convert decimal degrees to DMS components.

    Parameters
    ----------
    valin : float
    return_type : type
        list, tuple, np.array, or str
    str_delimiter : str
    str_decimal_places : int
    str_zero_pad : bool

    Returns
    -------
    value_DMS : list, tuple, ndarray, or str

    Examples
    --------
    deg2dms(1.123456789)             # [1, 7, 24.44...]
    deg2dms(1.123456789, str)        # '01:07:24.44'
    deg2dms(-0.001)                  # [0, 0, -3.6]   (sign on first
    deg2dms(-0.001, str)             # '-00:00:03.60'  nonzero field)
    deg2dms(-12.7, str)              # '-12:42:00.00'  (rollover)

    Notes
    -----
    For values where the degree component truncates to zero (``abs(valin)`` < 1 deg),
    the sign is propagated to the first non-zero component in list/tuple/
    ndarray output. For string output, the sign is always the leading '-'.
    ``dms2deg`` reads back both conventions correctly.

    When the computed or rounded seconds value lands at 60 (e.g. from
    floating-point accumulation in `-12.7 deg -> 41m 59.9999999s`, or from
    rounding `59.998s` to 2 decimal places), the carry is propagated up
    through minutes and degrees so the final output never displays 60.
    """
    sign = -1 if valin < 0 else 1
    absval = abs(valin)
    ddeg = int(absval)
    dmins_f = (absval - ddeg) * 60
    dmins = int(dmins_f)
    dsec = (dmins_f - dmins) * 60

    if return_type is str or return_type == 'str':
        # Round to display precision, then cascade any 60-overflow up.
        ddeg, dmins, dsec_r = _cascade_sexagesimal(
            ddeg, dmins, dsec, str_decimal_places)
        # Seconds column width: "DD.DDD..." = 2 integer digits + '.' + N decimals.
        # With no decimals, just the 2 integer digits.
        if str_zero_pad:
            deg_pad = '0>2'
            sec_pad = f'0>{3 + str_decimal_places}' if str_decimal_places > 0 else '0>2'
        else:
            deg_pad = ''
            sec_pad = ''
        prefix = '-' if sign < 0 else ''
        return '{6}{0:{7}d}{3}{1:{7}d}{3}{2:{5}.{4}f}'.format(
            ddeg, dmins, dsec_r, str_delimiter,
            str_decimal_places, sec_pad, prefix, deg_pad)
    else:
        # For list output, cascade when dsec is within float tolerance of 60
        # so the user never sees [d, m, 59.9999999] for a value that's
        # mathematically exactly [d, m+1, 0].
        ddeg, dmins, dsec = _cascade_sexagesimal(ddeg, dmins, dsec, None)
        # Apply sign to the first non-zero component so list round-trips
        # through dms2deg. Order of preference: degree, minute, second.
        if sign < 0:
            if ddeg != 0:
                ddeg = -ddeg
            elif dmins != 0:
                dmins = -dmins
            else:
                dsec = -dsec
        return return_type([int(ddeg), int(dmins), dsec])


def dms2deg(valin: str | Sequence[float]) -> float:
    """
    Convert DMS string ('dd:mm:ss.s') or [d, m, s] list to decimal degrees.

    Accepts sign on any component of a list (e.g. ``[0, 0, -3.6]`` for
    -0.001 deg) and leading '-' on strings. See ``deg2dms``.
    """
    if isinstance(valin, str):
        stripped = valin.strip()
        sign = -1 if stripped.startswith('-') else 1
        body = stripped.lstrip('+-')
        comps = [float(v) for v in body.replace('d', ':')
                 .replace('m', ':').replace('s', '').split(':')]
        return sign * (abs(comps[0]) + comps[1] / 60. + comps[2] / 3600.)
    else:
        comps = list(valin)
        # Pick sign from the first non-zero component - matches the
        # convention used by deg2dms for list output.
        sign = 1
        for c in comps:
            if c < 0:
                sign = -1
                break
            elif c > 0:
                break
        return sign * (abs(comps[0]) + abs(comps[1]) / 60.
                       + abs(comps[2]) / 3600.)


def deg2hour(valin: float) -> list[float]:
    """Convert decimal degrees to [hours, minutes, seconds] list."""
    rmins, rsec = divmod(24. / 360 * valin * 3600, 60)
    rh, rmins = divmod(rmins, 60)
    return [int(rh), int(rmins), rsec]


def hour2deg(valin: str | Sequence[float]) -> float:
    """Convert HMS string ('hh:mm:ss.s') or [h, m, s] list to decimal degrees."""
    if isinstance(valin, str):
        cleaned = valin.lower().replace('h', ':').replace('m', ':').replace('s', '')
        comps = [float(v) * 360. / 24 for v in cleaned.split(':')]
    else:
        comps = [v * 360. / 24 for v in valin]
    return comps[0] + comps[1] / 60. + comps[2] / 3600.


def dec2sex(longin: float, latin: float, as_string: bool = False,
            decimal_places: int = 2, str_format: str | Sequence[str] = ':',
            RAhours: bool = True, order: str = 'radec') -> Any:
    """
    Convert decimal degree coordinates to sexagesimal format.

    Parameters
    ----------
    longin : float
        Longitude (e.g., RA) in decimal degrees
    latin : float
        Latitude (e.g., DEC) in decimal degrees
    as_string : bool
        Return as formatted strings
    decimal_places : int
        Decimal places for seconds
    str_format : str or list
        ':', ' ', 'HMS', 'hms', 'DMS', 'dms', or list of 6 delimiters
    RAhours : bool
        If True, divide longitude by 15 to convert to hours
    order : str
        'radec'/'lonlat' or 'decra'/'latlon'

    Returns
    -------
    [lon_components, lat_components] as lists or strings
    """
    RAfactor = 24. / 360. if RAhours else 1.
    if order[:3].lower() in ('lat', 'dec'):
        longin, latin = float(latin), float(longin)
    else:
        longin, latin = float(longin), float(latin)

    # Decompose into non-negative magnitude components; the sign is carried
    # separately so the 60-overflow cascade (below) works correctly and a
    # negative value with a zero degree field still renders '-00'.
    long_sign = -1 if longin < 0 else 1
    longmins, longsecs = divmod(RAfactor * abs(longin) * 3600, 60)
    longdegs, longmins = divmod(longmins, 60)
    longdegs, longmins = int(longdegs), int(longmins)

    lat_sign = -1 if latin < 0 else 1
    lat_abs = abs(latin)
    latdegs = int(lat_abs)
    latmins_f = (lat_abs - latdegs) * 60
    latmins = int(latmins_f)
    latsecs = (latmins_f - latmins) * 60

    if as_string:
        # Round to display precision and carry any 60 up through the fields
        # (fixes float artifacts, e.g. -12.7 deg -> 41m 59.9999s -> 42m 00s).
        longdegs, longmins, longsecs = _cascade_sexagesimal(
            longdegs, longmins, longsecs, decimal_places)
        latdegs, latmins, latsecs = _cascade_sexagesimal(
            latdegs, latmins, latsecs, decimal_places)

        delimiter_map = {
            ':': [':', ':', '', ':', ':', ''],
            ' ': [' ', ' ', '', ' ', ' ', ''],
            'DMS': ['D', 'M', 'S', 'D', 'M', 'S'],
            'dms': ['d', 'm', 's', 'd', 'm', 's'],
            'HMS': ['H', 'M', 'S', 'D', 'M', 'S'],
            'hms': ['h', 'm', 's', 'd', 'm', 's'],
        }
        # ``str_format`` is either a preset key (string) or a custom list of
        # six delimiters. Only string keys can index ``delimiter_map`` — a
        # list is unhashable, so guard the lookup and pass a custom list
        # through directly.
        delimiters: Sequence[str]
        if isinstance(str_format, str):
            delimiters = delimiter_map.get(str_format, str_format)
        else:
            delimiters = str_format
        if len(delimiters) < 6:
            raise ValueError('str_format must have six delimiters if custom list')

        lon_prefix = '-' if long_sign < 0 else ''
        lat_prefix = '-' if lat_sign < 0 else ''
        lonstring = '{8}{0:0>2d}{5}{1:0>2d}{6}{2:0>{4}.{3}f}{7}'.format(
            longdegs, longmins, longsecs,
            decimal_places, decimal_places + 3,
            delimiters[0], delimiters[1], delimiters[2], lon_prefix)
        latstring = '{8}{0:0>2d}{5}{1:0>2d}{6}{2:0>{4}.{3}f}{7}'.format(
            latdegs, latmins, latsecs,
            decimal_places, decimal_places + 3,
            delimiters[3], delimiters[4], delimiters[5], lat_prefix)

        if order[:3].lower() in ('lat', 'dec'):
            return [latstring, lonstring]
        return [lonstring, latstring]
    else:
        # Numeric list: cascade any near-60 seconds, then re-attach the sign
        # on the degree field (a negative-zero degree keeps the sign of a
        # small negative value so it round-trips through sex2dec / dms2deg).
        longdegs, longmins, longsecs = _cascade_sexagesimal(
            longdegs, longmins, longsecs, None)
        latdegs, latmins, latsecs = _cascade_sexagesimal(
            latdegs, latmins, latsecs, None)
        lon_deg_out = -0.0 if (longdegs == 0 and long_sign < 0) else long_sign * longdegs
        lat_deg_out = -0.0 if (latdegs == 0 and lat_sign < 0) else lat_sign * latdegs
        lon_comp = [lon_deg_out, longmins, longsecs]
        lat_comp = [lat_deg_out, latmins, latsecs]
        if order[:3].lower() in ('lat', 'dec'):
            return lat_comp, lon_comp
        return lon_comp, lat_comp


def sex2dec(longin: str, latin: str, RAhours: bool = True,
            order: str = 'radec') -> list[float]:
    """
    Convert sexagesimal coordinate strings to decimal degrees.

    Parameters
    ----------
    longin, latin : str
        Sexagesimal strings (e.g., '05:34:31.9', '05h34m31.9s', '-22d00m52.2s')
    RAhours : bool
        If True, multiply longitude result by 15 (hours -> degrees)
    order : str
        'radec'/'lonlat' or 'decra'/'latlon'

    Returns
    -------
    [lon_decimal, lat_decimal] in degrees
    """
    if order[:3].lower() in ('lat', 'dec'):
        longin, latin = str(latin), str(longin)

    RAfactor = 360. / 24. if (RAhours or 'h' in longin.lower()) else 1.

    # Parse longitude
    lon_clean = longin.lower().replace('h', ':').replace('d', ':').replace('m', ':').replace('s', '')
    if ':' in lon_clean:
        lo = [float(v) * RAfactor for v in lon_clean.split(':')[:3]]
    else:
        lo = [float(v) * RAfactor for v in longin.split()[:3]]
    longout = lo[0] + lo[1] / 60. + lo[2] / 3600.

    # Parse latitude
    lat_clean = latin.lower().replace('d', ':').replace('m', ':').replace('s', '')
    if ':' in lat_clean:
        la = [float(v) for v in lat_clean.split(':')[:3]]
    else:
        la = [float(v) for v in latin.split()[:3]]
    if la[0] < 0 or latin.strip().startswith('-'):
        latout = la[0] - la[1] / 60. - la[2] / 3600.
    else:
        latout = la[0] + la[1] / 60. + la[2] / 3600.

    if order[:3].lower() in ('lat', 'dec'):
        return [latout, longout]
    return [longout, latout]


# ---- Angular distance (Vincenty) ----

def angulardistance(coords1_deg: npt.ArrayLike, coords2_deg: npt.ArrayLike,
                    pythag_approx: bool = False, returncomponents: bool = False,
                    input_precision: npt.DTypeLike = np.float64) -> Any:
    """
    Angular distance between two sky positions using the Vincenty formula.

    Parameters
    ----------
    coords1_deg, coords2_deg : array-like, shape (2,) or (N, 2)
        [lon, lat] in decimal degrees (same frame)
    pythag_approx : bool
        Use small-angle approximation (valid for sep < ~1 deg)
    returncomponents : bool
        Also return (dRA*cos(DEC), dDEC) components
    input_precision : numpy dtype
        Use np.float128 for sub-mas precision (where available)

    Returns
    -------
    separation_deg : float or ndarray
    dRA_deg, dDEC_deg : (if returncomponents=True)
    """
    coords1_rad = np.array(coords1_deg, dtype=input_precision) * (np.pi / 180)
    coords2_rad = np.array(coords2_deg, dtype=input_precision) * (np.pi / 180)

    if coords1_rad.ndim == 1:
        lon1, lat1 = coords1_rad[0], coords1_rad[1]
        lon2, lat2 = coords2_rad[0], coords2_rad[1]
    else:
        lon1, lat1 = coords1_rad[:, 0], coords1_rad[:, 1]
        lon2, lat2 = coords2_rad[:, 0], coords2_rad[:, 1]

    lat_mean = 0.5 * (lat1 + lat2)

    if pythag_approx:
        dRA_rad = (lon2 - lon1) * np.cos(lat_mean)
        dDEC_rad = lat2 - lat1
        sep_rad = np.sqrt(dRA_rad**2 + dDEC_rad**2)
        if returncomponents:
            return np.degrees(sep_rad), np.degrees(dRA_rad), np.degrees(dDEC_rad)
        return np.degrees(sep_rad)

    # Full Vincenty
    sin_dlon = np.sin(lon1 - lon2)
    cos_dlon = np.cos(lon1 - lon2)
    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)

    A = cos_lat2 * sin_dlon
    B = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_dlon
    numerator = np.array(np.hypot(A, B), dtype=input_precision)
    denominator = np.array(sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_dlon,
                           dtype=input_precision)
    sep_rad = np.arctan2(numerator, denominator)

    if returncomponents:
        dRA_rad = (lon2 - lon1) * np.cos(lat_mean)
        dDEC_rad = lat2 - lat1
        return np.degrees(sep_rad), np.degrees(dRA_rad), np.degrees(dDEC_rad)
    return np.degrees(sep_rad)


def RAcosDEC_err(RA_deg: float, DEC_deg: float, uncRA_asec: float,
                 uncDEC_asec: float, cov_RADEC: float = 0.,
                 return_units: str = 'arcsec') -> Any:
    """
    Total formal uncertainty on RA*cos(DEC).

    Parameters
    ----------
    RA_deg, DEC_deg : float
        Position in degrees
    uncRA_asec, uncDEC_asec : float
        Uncertainties in arcseconds
    cov_RADEC : float
        Covariance between RA and DEC (in arcsec^2)
    return_units : str
        'arcsec', 'mas', 'uas'

    Returns
    -------
    sigma_RAcosDEC : float
    """
    DEC_rad = np.radians(DEC_deg)
    RA_rad = np.radians(RA_deg)

    sigma_RAcosD = np.sqrt(
        (np.cos(DEC_rad) * uncRA_asec) ** 2
        + (RA_rad * np.sin(DEC_rad) * uncDEC_asec) ** 2
        - 2 * cov_RADEC * RA_rad * np.cos(DEC_rad) * np.sin(DEC_rad)
    )

    unit_factors = {'arcsec': 1., 'mas': 1e3, 'uas': 1e6, 'microarcsec': 1e6}
    return sigma_RAcosD * unit_factors.get(return_units.lower(), 1.)


# ---- Frame conversion helpers ----
#
# Thin wrappers around astropy.SkyCoord so users making sky plots can
# do quick lon/lat ↔ lon/lat conversions without instantiating a
# SkyCoord by hand. Inputs are decimal degrees (scalar or ndarray);
# outputs are a tuple of (lon, lat) in decimal degrees in the target
# frame.

# Frame aliases — accept common short names.
_FRAME_ALIASES = {
    'icrs': 'icrs',
    'fk5': 'fk5',
    'fk4': 'fk4',
    'equatorial': 'icrs',
    'eq': 'icrs',
    'gal': 'galactic',
    'galactic': 'galactic',
    'super': 'supergalactic',
    'supergalactic': 'supergalactic',
    'sgal': 'supergalactic',
    'ecl': 'geocentrictrueecliptic',
    'ecliptic': 'geocentrictrueecliptic',
    'geocentrictrueecliptic': 'geocentrictrueecliptic',
    'barycentric': 'barycentrictrueecliptic',
    'barycentrictrueecliptic': 'barycentrictrueecliptic',
    'helio': 'heliocentrictrueecliptic',
    'heliocentric': 'heliocentrictrueecliptic',
    'heliocentrictrueecliptic': 'heliocentrictrueecliptic',
    'cirs': 'cirs',
    'altaz': 'altaz',
}


def _resolve_frame(name: str | None) -> str:
    if name is None:
        return 'icrs'
    key = name.lower().replace(' ', '_').replace('-', '_')
    if key in _FRAME_ALIASES:
        return _FRAME_ALIASES[key]
    return key  # pass through unknown names — astropy may still recognize


def convert_frame(lon_deg: npt.ArrayLike, lat_deg: npt.ArrayLike,
                  from_frame: str = 'icrs',
                  to_frame: str = 'galactic') -> tuple[Any, Any]:
    """
    Convert (lon, lat) coordinates from one celestial frame to another.

    Parameters
    ----------
    lon_deg, lat_deg : float or array-like
        Longitude and latitude in the source frame, in decimal degrees.
        Scalars or arrays of matching shape.
    from_frame, to_frame : str
        Source / target frame. Common aliases accepted: ``'icrs'``,
        ``'fk5'``, ``'fk4'``, ``'galactic'`` (alias ``'gal'``),
        ``'supergalactic'`` (``'sgal'``), ``'ecliptic'``
        (``'ecl'`` — geocentric true ecliptic), ``'heliocentric'``
        (``'helio'`` — heliocentric true ecliptic). Any other string
        is passed through to astropy's frame resolver.

    Returns
    -------
    lon_out_deg, lat_out_deg : float or ndarray
        Coordinates in the target frame.

    Examples
    --------
    >>> from skyplothelper.core.coords import convert_frame
    >>> # Galactic center coordinates in ICRS
    >>> convert_frame(0.0, 0.0, 'galactic', 'icrs')
    (266.4..., -28.9...)
    >>> # Many positions at once
    >>> import numpy as np
    >>> ras = np.array([180.0, 270.0])
    >>> decs = np.array([0.0, -45.0])
    >>> l, b = convert_frame(ras, decs, 'icrs', 'galactic')

    Notes
    -----
    For more control (custom equinox, motion correction, observation
    epoch, etc.) use ``astropy.coordinates.SkyCoord`` directly.
    """
    # Local import — astropy is a hard dependency of the package, but
    # keeping it lazy keeps `import skyplothelper` startup cheap on
    # systems where the user only wants the format helpers.
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    src = _resolve_frame(from_frame)
    dst = _resolve_frame(to_frame)
    coord = SkyCoord(np.asarray(lon_deg) * u.deg,
                     np.asarray(lat_deg) * u.deg,
                     frame=src)
    out = coord.transform_to(dst)
    # Get spherical lon/lat regardless of the target frame's attribute
    # names (icrs has ra/dec; galactic has l/b; ecliptic has lon/lat).
    rep = out.represent_as('unitspherical')
    lon_out = rep.lon.to(u.deg).value
    lat_out = rep.lat.to(u.deg).value
    return lon_out, lat_out


def icrs_to_galactic(ra_deg: npt.ArrayLike, dec_deg: npt.ArrayLike) -> tuple[Any, Any]:
    """ICRS (RA, Dec) → Galactic (l, b). Decimal degrees in/out."""
    return convert_frame(ra_deg, dec_deg, 'icrs', 'galactic')


def galactic_to_icrs(l_deg: npt.ArrayLike, b_deg: npt.ArrayLike) -> tuple[Any, Any]:
    """Galactic (l, b) → ICRS (RA, Dec). Decimal degrees in/out."""
    return convert_frame(l_deg, b_deg, 'galactic', 'icrs')


def icrs_to_ecliptic(ra_deg: npt.ArrayLike, dec_deg: npt.ArrayLike) -> tuple[Any, Any]:
    """ICRS (RA, Dec) → Ecliptic (lon, lat). Decimal degrees in/out."""
    return convert_frame(ra_deg, dec_deg, 'icrs', 'ecliptic')


def ecliptic_to_icrs(lon_deg: npt.ArrayLike, lat_deg: npt.ArrayLike) -> tuple[Any, Any]:
    """Ecliptic (lon, lat) → ICRS (RA, Dec). Decimal degrees in/out."""
    return convert_frame(lon_deg, lat_deg, 'ecliptic', 'icrs')


def galactic_to_ecliptic(l_deg: npt.ArrayLike, b_deg: npt.ArrayLike) -> tuple[Any, Any]:
    """Galactic (l, b) → Ecliptic (lon, lat). Decimal degrees in/out."""
    return convert_frame(l_deg, b_deg, 'galactic', 'ecliptic')


def ecliptic_to_galactic(lon_deg: npt.ArrayLike, lat_deg: npt.ArrayLike) -> tuple[Any, Any]:
    """Ecliptic (lon, lat) → Galactic (l, b). Decimal degrees in/out."""
    return convert_frame(lon_deg, lat_deg, 'ecliptic', 'galactic')


def icrs_to_supergalactic(ra_deg: npt.ArrayLike, dec_deg: npt.ArrayLike) -> tuple[Any, Any]:
    """ICRS (RA, Dec) → Supergalactic (SGL, SGB). Decimal degrees."""
    return convert_frame(ra_deg, dec_deg, 'icrs', 'supergalactic')


def supergalactic_to_icrs(sgl_deg: npt.ArrayLike, sgb_deg: npt.ArrayLike) -> tuple[Any, Any]:
    """Supergalactic (SGL, SGB) → ICRS (RA, Dec). Decimal degrees."""
    return convert_frame(sgl_deg, sgb_deg, 'supergalactic', 'icrs')
