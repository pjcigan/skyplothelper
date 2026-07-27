"""Nightshade blend for daylight / terminator visualization.

``pseudofits_from_image`` wraps a 2-D RGB image in a fake Plate Carrée
WCS so it can be reprojected. ``make_nightshade_blend`` overlays a
day/night terminator on an Earth (or planet) map by building a per-pixel
alpha channel.

Two blend back-ends are available (``blend=`` argument):

* ``'elevation'`` (default) — a physically-grounded blend driven by the
  actual **solar elevation angle** at every pixel. The day→night transition
  spans the *twilight band* (configurable via ``h_day`` / ``h_night``) and
  the shape of the fade is chosen from a small set of transfer curves
  (``curve='linear' | 'smoothstep' | 'twilight'``, default ``'smoothstep'``).
  This mode is resolution-independent, has the correct latitude/orientation
  dependence built in, and needs neither cartopy nor scipy.

* ``'gaussian'`` — cartopy's :class:`~cartopy.feature.nightshade.Nightshade`
  is used to get the binary day/night region, which is then smoothed with
  a fixed-width Gaussian. The transition width is set in *pixels*
  (``blend_sigma``); it is fast and looks good, but it is a purely
  **cosmetic** image-space smoothing with no physical scale — a tunable
  soft edge for aesthetic effect, not a model of real twilight.

The relevant physics, for reference:

The only quantity that matters is the solar elevation angle ``h`` (the
Sun's altitude above the local horizon), a smooth scalar field over the
globe::

    h(lon, lat) = arcsin( sin(lat)·sin(δ) + cos(lat)·cos(δ)·cos(lon − lon₀) )

where ``δ`` is the solar declination and ``lon₀`` the sub-solar longitude
(both from the date; see :func:`_subsolar_lonlat`). The terminator is the
``h = 0`` contour, and the day→night blend is exactly the twilight band:

    apparent sunset   h ≈ −0.83°   (Sun's upper limb at the horizon)
    civil twilight    h = −6°
    nautical twilight h = −12°
    astronomical      h = −18°     (sky fully dark)
"""

from __future__ import annotations

import os
import warnings  # noqa: F401
from typing import Any

import astropy.io.fits as pyfits
import matplotlib.colors as mcolors  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy.wcs import WCS  # noqa: F401

try:
    import cartopy.crs as ccrs
    from cartopy.feature.nightshade import Nightshade
    from matplotlib.path import Path as MplPath
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False

try:
    import scipy.interpolate as interp
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from ..images.reprojection import reproject_rgb_map  # noqa: F401


def _shapely_geoms_to_path(geoms: Any) -> Any:
    """Build a matplotlib Path from shapely (Multi)Polygon geometries.

    Replaces the deprecated ``cartopy.mpl.patch.geos_to_path`` (which cartopy
    will remove) — the Nightshade night region is a shapely polygon and all we
    need it for is a ``contains_points`` membership test, so trace each ring's
    coords directly. Verified to give a byte-identical night mask.
    """
    verts: list[Any] = []
    codes: list[Any] = []
    for g in geoms:
        polys = list(g.geoms) if g.geom_type == 'MultiPolygon' else [g]
        for p in polys:
            for ring in (p.exterior, *p.interiors):
                pts = np.asarray(ring.coords, dtype=float)
                if len(pts) < 3:
                    continue
                verts.extend(pts.tolist())
                codes.append(MplPath.MOVETO)
                codes.extend([MplPath.LINETO] * (len(pts) - 1))
    return MplPath(verts, codes)


def pseudofits_from_image(input_path: str | os.PathLike[str] | npt.NDArray[Any],
                          fitsproj: str = 'CAR',
                          gmst_deg: float = 0., geo: bool = False) -> Any:
    """
    Wrap a flat-map image (JPG, PNG, etc.) in a pseudo-FITS ImageHDU with
    appropriate WCS header for reprojection.

    Parameters
    ----------
    input_path : str, path-like, or ndarray
        Path to an image file, **or** an in-memory ``(ny, nx, nchannels)``
        RGB(A) array. The array form is what lets a computed raster be draped
        directly — e.g. the RGBA output of :func:`make_nightshade_blend` —
        without writing it to disk first.
    fitsproj : str, optional
        Fits projection code for the image. Default 'CAR' (Plate Carrée).
    gmst_deg : float, optional
        Greenwich Mean Sidereal Time offset in degrees. Default 0.
    geo : bool, optional
        If True, use geographic coordinate types (TLON/TLAT with ITRF).
        If False (default), use celestial (RA/DEC with ICRS).

    Returns
    -------
    hdu : ImageHDU
        FITS ImageHDU containing the image data and WCS header.

    Notes
    -----
    The ``geo`` flag sets the coordinate *system* of the WCS: ``geo=True``
    → geographic ``TLON``/``TLAT`` (ITRF); ``geo=False`` → celestial
    ``RA``/``DEC`` (ICRS). When draping the result onto a frame with
    :func:`reproject_rgb_map`, the target frame must use the **same**
    system, or reproject applies a real coordinate transform between them.
    In particular a ``geo=True`` texture belongs on a geographic frame — a
    globe from :func:`make_planet_frame` (already ITRS) or a flat frame
    built with ``make_wcs_frame(..., frame='ITRS')``. Draping it onto a
    default (celestial) frame instead triggers an ITRS↔ICRS rotation that
    slides the map in longitude by tens of degrees.

    Examples
    --------
    Load a NASA Blue Marble image::

        earth_hdu = pseudofits_from_image('blue_marble.jpg', geo=True)

    Drape a computed nightshade blend onto a globe (no temp file)::

        rgba = make_nightshade_blend(day_rgb, date)
        hdu = pseudofits_from_image(rgba, geo=True)
        draped = reproject_rgb_map(hdu, globe_header, shape_out=shape)
    """
    # Accept an in-memory raster as well as a path. make_nightshade_blend()
    # returns an RGBA array, so requiring a file meant the one thing you'd
    # want to drape was the one thing you couldn't feed in -- the caller had
    # to round-trip through a temp file. Everything else in the header is
    # synthesized from the array shape, so nothing else is needed.
    if isinstance(input_path, (str, os.PathLike)):
        # fspath() so a pathlib.Path works too — matplotlib's imread signature
        # is narrower than os.PathLike.
        img = plt.imread(os.fspath(input_path))
    else:
        img = np.asarray(input_path)
    if img.ndim != 3:
        raise ValueError(
            "pseudofits_from_image expects an (ny, nx, nchannels) RGB(A) "
            f"raster; got an array with shape {img.shape}. For a 2-D map, "
            "stack it to 3 channels first (e.g. np.dstack([m, m, m])).")
    imheight, imwidth, n_channels = img.shape

    lon_type = 'TLON' if geo else 'RA--'
    lat_type = 'TLAT' if geo else 'DEC-'
    rdsys = 'ITRF' if geo else 'ICRS'

    header = pyfits.Header(dict(
        NAXIS=3,
        NAXIS1=n_channels, NAXIS2=imwidth, NAXIS3=imheight,
        CRPIX2=imwidth / 2, CRPIX3=imheight / 2,
        CRVAL2=gmst_deg % 360., CRVAL3=0,
        CDELT2=360 / imwidth,
        CDELT3=-180 / imheight,
        CTYPE2=f'{lon_type}-{fitsproj}',
        CTYPE3=f'{lat_type}-{fitsproj}',
        CUNIT2='deg', CUNIT3='deg',
        RADESYSa=rdsys,
    ).items())
    return pyfits.ImageHDU(img[:, :, :], header)


def make_nightshade_blend(
    rgb_image: np.ndarray, date: Any, blend: str = 'elevation',
    blend_sigma: float = 20,
    lonlat_extent: tuple[float, float, float, float] = (-180, 180, -90, 90),
    day_blend: bool = False, hard_terminator_edge: bool = False,
    mode: str = 'fast', fastmode_xy: tuple[int, int] = (300, 150),
    min_blend_pct: float = 1.,
    h_day: float = 0.0, h_night: float = -18.0,
    curve: str = 'smoothstep', twilight_decay: float = 6.0,
    inspect_results: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Build a day/night terminator alpha channel for an RGB Earth/planet map.

    In the default (nighttime) orientation, the returned RGBA image has
    daytime pixels transparent and nighttime pixels opaque — designed so
    you can lay this image (e.g. a city-lights / Black Marble map) on top
    of a daytime base map and have the two cross-fade through the
    terminator. Set ``day_blend=True`` to invert the alpha (opaque day,
    transparent night) for the reverse layering.

    Parameters
    ----------
    rgb_image : ndarray, shape (H, W, 3)
        Input RGB image with values in [0, 1] (not [0, 255]). Assumed to
        be a Plate Carrée map covering ``lonlat_extent``, north-up (row 0
        is the northern edge).
    date : astropy.time.Time, datetime, or str
        Observation time (UTC). An :class:`astropy.time.Time` is preferred —
        it is what the rest of the astropy ecosystem passes around, and a FITS
        ``DATE-OBS`` parses straight into one. A ``datetime`` or an ISO string
        work equally well.
        UTC date/time for which to compute the terminator.
    blend : {'gaussian', 'elevation'}, optional
        Which terminator model to use. Default ``'elevation'`` (physical;
        also the lightest on dependencies).

        * ``'gaussian'`` — cartopy Nightshade binary mask, Gaussian-smoothed
          by ``blend_sigma`` pixels (uses ``hard_terminator_edge``, ``mode``,
          ``fastmode_xy``, ``min_blend_pct``). Requires cartopy and scipy.
          This is a **cosmetic** smoothing knob, *not* a physical model: the
          transition width is an arbitrary number of pixels with no
          connection to the real twilight band, so use it for aesthetic
          exaggeration / a controllable soft edge rather than physical
          accuracy. Prefer ``'elevation'`` when the width should mean
          something.
        * ``'elevation'`` — physical solar-elevation blend over the twilight
          band (uses ``h_day``, ``h_night``, ``curve``, ``twilight_decay``).
          The transition spans the real twilight angles and is
          resolution-independent. Requires neither cartopy nor scipy.

    blend_sigma : float, optional
        *(gaussian)* Gaussian sigma for the terminator transition, in
        pixels. Default 20.
    lonlat_extent : tuple, optional
        (lon_min, lon_max, lat_min, lat_max) extent of the input image.
    day_blend : bool, optional
        If True, invert the alpha so the *daytime* side is opaque (for
        overlaying a daytime image on a nighttime base).
    hard_terminator_edge : bool, optional
        *(gaussian)* If False (default), the Gaussian smooths the binary
        day/night mask symmetrically, giving a soft transition centered on
        the terminator with no discontinuity. If True, the night interior
        is forced fully opaque and only the daytime side is blended — which
        marks the exact terminator but leaves a visible step (the night
        side jumps to full opacity right at the line). Default False for the
        smoother, more natural look.
    mode : str, optional
        *(gaussian)* 'fast' (default) or 'exact'. Fast mode convolves a
        small dummy array then upscales (~550 ms vs >60 s for large images).
    fastmode_xy : tuple of int, optional
        *(gaussian)* (x_pixels, y_pixels) for the fast-mode dummy array.
    min_blend_pct : float, optional
        *(gaussian)* Minimum blend sigma as a percentage of the image
        x-dimension. Default 1.
    h_day : float, optional
        *(elevation)* Solar elevation (degrees) at/above which a pixel is
        full daytime (night alpha = 0). Default 0.0 (the geometric
        terminator). Use −0.833 for apparent sunset (Sun's upper limb at
        the horizon), or a small positive value to start the fade slightly
        before sunset.
    h_night : float, optional
        *(elevation)* Solar elevation (degrees) at/below which a pixel is
        full night (night alpha = 1). Default −18.0 (end of astronomical
        twilight, sky fully dark). Set −6 for civil or −12 for nautical
        twilight to get a tighter terminator.
    curve : {'linear', 'smoothstep', 'twilight'}, optional
        *(elevation)* Transfer function mapping solar elevation across the
        twilight band to night-layer opacity. Default ``'smoothstep'``.

        * ``'linear'`` — opacity ramps linearly with elevation across the
          band. Because perceived sky brightness is roughly the *logarithm*
          of illuminance and twilight illuminance falls ~log-linearly with
          the Sun's depression angle, a linear ramp in elevation is close
          to the *perceptually* uniform fade.
        * ``'smoothstep'`` — a raised-cosine S-curve, ``0.5·(1 − cos(π·t))``,
          which is flat-tangent (C¹) at both the day and night ends so the
          transition eases in and out. A good general-purpose default.
        * ``'twilight'`` — a *radiometric* model: the twilight sky's
          illuminance falls roughly exponentially with the Sun's depression
          angle, so this curve darkens the night side quickly just past the
          terminator and then levels off, producing a thin bright twilight
          arc hugging the terminator. ``twilight_decay`` sets the falloff
          rate.
    twilight_decay : float, optional
        *(elevation, curve='twilight')* Number of e-foldings of illuminance
        loss across the twilight band — larger ⇒ a sharper, more
        terminator-hugging bright band; smaller ⇒ a softer fade. Default 6.0.
    inspect_results : bool, optional
        If True, show diagnostic plots of the terminator and blended image.

    Returns
    -------
    rgba_image : ndarray, shape (H, W, 4)
        RGBA image with the blended terminator alpha channel, values in
        [0, 1].

    Raises
    ------
    ImportError
        If ``blend='gaussian'`` and cartopy or scipy is not installed.
    ValueError
        If ``blend`` (or ``curve``) is not a recognized option.

    Notes
    -----
    The terminator overlay is a *terrestrial* feature, so display it on an
    Earth globe built with the geographic (east-right) orientation. Use
    :func:`~skyplothelper.globe.make_planet_frame` (geographic + ITRS by
    default); a plain :func:`~skyplothelper.globe.make_globe_frame` defaults
    to the astronomical east-left view, which would mirror the Earth.

    Examples
    --------
    Overlay a physically-blended nightshade on an Earth globe::

        import datetime
        from skyplothelper.globe import make_planet_frame, make_nightshade_blend

        rgba = make_nightshade_blend(city_lights_rgb, datetime.datetime.utcnow(),
                                     blend='elevation', curve='smoothstep')
        ax = make_planet_frame(111, center_LONdeg=0, center_LATdeg=23.44)
        ax.imshow(daytime_rgb, ...)   # base map
        ax.imshow(rgba, ...)          # night map, cross-faded at the terminator
    """
    if blend == 'gaussian':
        return _gaussian_blend(
            rgb_image, date, blend_sigma, lonlat_extent, day_blend,
            hard_terminator_edge, mode, fastmode_xy, min_blend_pct,
            inspect_results)
    elif blend == 'elevation':
        return _elevation_blend(
            rgb_image, date, lonlat_extent, day_blend,
            h_day, h_night, curve, twilight_decay, inspect_results)
    raise ValueError(
        f"blend must be 'gaussian' or 'elevation', got {blend!r}")


# =============================================================================
# Physical (solar-elevation) blend
# =============================================================================

def _julian_day(date: Any) -> float:
    """Julian Date (days) for a UTC time.

    Accepts an :class:`astropy.time.Time`, a ``datetime``, or anything
    ``Time`` can parse (e.g. an ISO string). A ``Time`` already carries the
    JD, so it is read directly.

    The ``datetime`` path uses Python's proleptic-Gregorian ordinal:
    ``toordinal()`` counts days since 0001-01-01, whose midnight is
    JD 1721424.5; add the fractional day. (E.g. 2000-01-01 12:00 UTC →
    exactly JD 2451545.0, the J2000 epoch.) That path stays self-contained —
    no IERS tables or astropy time machinery needed — and is plenty accurate
    for placing a terminator.
    """
    if hasattr(date, 'jd'):                      # astropy Time duck-type
        return float(np.atleast_1d(date.utc.jd)[0])
    if hasattr(date, 'toordinal'):               # datetime / date
        frac = (getattr(date, 'hour', 0)
                + getattr(date, 'minute', 0) / 60.0
                + getattr(date, 'second', 0) / 3600.0
                + getattr(date, 'microsecond', 0) / 3.6e9) / 24.0
        return date.toordinal() + 1721424.5 + frac
    # Anything else (ISO string, MJD-carrying object, ...) — let astropy try.
    from astropy.time import Time
    return float(np.atleast_1d(Time(date).utc.jd)[0])


def _subsolar_lonlat(date: Any) -> tuple[float, float]:
    """Sub-solar point ``(lon, lat)`` in degrees for a UTC ``datetime``.

    The sub-solar point is where the Sun is directly overhead: its latitude
    is the solar declination ``δ`` and its longitude is the meridian facing
    the Sun. Implements the low-precision solar-position series from Vallado,
    *Fundamentals of Astrodynamics and Applications* (Algorithm 29) — good
    to ~0.01°, far finer than a map pixel. Pure-numpy and self-contained, so
    the elevation blend depends only on numpy.
    """
    # Julian centuries since the J2000.0 epoch.
    t_ut1 = (_julian_day(date) - 2451545.0) / 36525.0

    # Mean longitude and mean anomaly of the Sun (degrees).
    mean_lon = (280.460 + 36000.771 * t_ut1) % 360.0
    mean_anom = np.radians((357.5277233 + 35999.05034 * t_ut1) % 360.0)

    # Ecliptic longitude (equation of center applied to the mean longitude).
    ecl_lon = np.radians(mean_lon
                         + 1.914666471 * np.sin(mean_anom)
                         + 0.019994643 * np.sin(2.0 * mean_anom))

    # Obliquity of the ecliptic, and the resulting solar declination.
    eps = np.radians(23.439291 - 0.0130042 * t_ut1)
    dec = np.degrees(np.arcsin(np.sin(eps) * np.sin(ecl_lon)))

    # Solar right ascension (atan2 keeps it in the correct quadrant).
    ra = np.degrees(np.arctan2(np.cos(eps) * np.sin(ecl_lon), np.cos(ecl_lon)))

    # Greenwich mean sidereal time (seconds → degrees). The sub-solar
    # longitude is the Sun's RA minus the Greenwich hour angle, i.e.
    # lon = RA − GMST, wrapped to [-180, 180].
    gmst_s = (67310.54841
              + (876600.0 * 3600.0 + 8640184.812866) * t_ut1
              + 0.093104 * t_ut1**2
              - 6.2e-6 * t_ut1**3)
    gmst_deg = (gmst_s % 86400.0) / 240.0
    lon = (ra - gmst_deg + 180.0) % 360.0 - 180.0
    return float(lon), float(dec)


def _solar_elevation_field(
    lon_deg: npt.NDArray[np.float64], lat_deg: npt.NDArray[np.float64],
    subsolar_lon: float, subsolar_dec: float,
) -> npt.NDArray[np.float64]:
    """Solar elevation angle (degrees) on a lon/lat grid.

    Vectorized form of the standard relation

        sin(h) = sin(lat)·sin(δ) + cos(lat)·cos(δ)·cos(lon − lon₀)

    where ``h`` is the Sun's altitude above the horizon, ``δ`` the solar
    declination, and ``lon₀`` the sub-solar longitude. ``h = +90°`` at the
    sub-solar point, ``0°`` on the terminator, and ``−90°`` at the
    antisolar point. ``lon_deg`` and ``lat_deg`` broadcast against each
    other (pass shape ``(1, W)`` and ``(H, 1)`` to get an ``(H, W)`` field).
    """
    lat_r = np.radians(lat_deg)
    dec_r = np.radians(subsolar_dec)
    hour_angle_r = np.radians(lon_deg - subsolar_lon)
    sin_h = (np.sin(lat_r) * np.sin(dec_r)
             + np.cos(lat_r) * np.cos(dec_r) * np.cos(hour_angle_r))
    return np.degrees(np.arcsin(np.clip(sin_h, -1.0, 1.0)))


def _terminator_alpha(
    elev_deg: npt.NDArray[np.float64], h_day: float, h_night: float,
    curve: str, twilight_decay: float,
) -> npt.NDArray[np.float64]:
    """Map a solar-elevation field to night-layer opacity in ``[0, 1]``.

    ``t`` is the normalized depth into the twilight band: ``0`` where the
    Sun is at/above ``h_day`` (full day) and ``1`` where it is at/below
    ``h_night`` (full night). The named ``curve`` then shapes how opacity
    rises across that band — see :func:`make_nightshade_blend` for the
    physical motivation of each.
    """
    span = float(h_day - h_night)
    if span <= 0:
        raise ValueError(
            f"h_day ({h_day}) must be greater than h_night ({h_night})")
    t = np.clip((h_day - elev_deg) / span, 0.0, 1.0)

    if curve == 'linear':
        # Even ramp in elevation. Since perceived brightness ≈ log(illuminance)
        # and twilight illuminance is ~log-linear in depression angle, this is
        # close to a perceptually-uniform day→night fade.
        return t
    if curve == 'smoothstep':
        # Raised-cosine S-curve: zero slope at both ends (C¹-continuous), so
        # the fade eases in at sunset and eases out into full night.
        return 0.5 - 0.5 * np.cos(np.pi * t)
    if curve == 'twilight':
        # Radiometric model. The twilight sky's illuminance falls roughly
        # exponentially with the Sun's depression angle: I(t) = exp(-k·t).
        # Re-scale so the day fraction is exactly 1 at the day edge and 0 at
        # the night edge, then night opacity = 1 − day_fraction. The result
        # darkens fast just past the terminator (a thin bright twilight arc)
        # and then flattens.
        k = float(twilight_decay)
        if k <= 0:
            return t
        day_fraction = (np.exp(-k * t) - np.exp(-k)) / (1.0 - np.exp(-k))
        return 1.0 - day_fraction
    raise ValueError(
        f"curve must be 'linear', 'smoothstep', or 'twilight', got {curve!r}")


def _elevation_blend(
    rgb_image: np.ndarray, date: Any,
    lonlat_extent: tuple[float, float, float, float], day_blend: bool,
    h_day: float, h_night: float, curve: str, twilight_decay: float,
    inspect_results: bool,
) -> npt.NDArray[np.float64]:
    """Physical solar-elevation terminator blend (see make_nightshade_blend)."""
    height, width = rgb_image.shape[:2]
    lon_min, lon_max, lat_min, lat_max = lonlat_extent

    # Pixel-center lon/lat axes. Image row 0 is the *northern* edge, so
    # latitude runs from lat_max (top) down to lat_min (bottom); building it
    # this way means the alpha rows line up with the image rows with no flip.
    lon = np.linspace(lon_min, lon_max, width)[None, :]        # (1, W)
    lat = np.linspace(lat_max, lat_min, height)[:, None]       # (H, 1)

    subsolar_lon, subsolar_dec = _subsolar_lonlat(date)
    elev = _solar_elevation_field(lon, lat, subsolar_lon, subsolar_dec)

    # Night-layer opacity, then optionally invert for day-on-night layering.
    night_alpha = _terminator_alpha(elev, h_day, h_night, curve, twilight_decay)
    alpha = (1.0 - night_alpha) if day_blend else night_alpha

    rgba = np.zeros((height, width, 4), dtype=float)
    rgba[:, :, :3] = rgb_image[:, :, :3]
    rgba[:, :, 3] = alpha
    rgba = rgba.clip(0.0, 1.0)

    if inspect_results:
        _inspect_elevation(rgb_image, rgba, elev, lonlat_extent)
    return rgba


def _inspect_elevation(
    rgb: npt.ArrayLike, rgba: npt.ArrayLike,
    elev: npt.NDArray[np.float64], extent: Any,
) -> None:
    """Diagnostic plot for the elevation blend: elevation field + result."""
    fig, axes_raw = plt.subplots(2, 1, figsize=(8, 8))
    axes: Any = axes_raw  # matplotlib stubs type this Axes|ndarray; we index it
    im = axes[0].imshow(elev, extent=extent, origin='upper', cmap='RdBu_r',
                        vmin=-90, vmax=90, aspect='auto')
    # Mark the terminator (h = 0) and the twilight band edges.
    axes[0].contour(elev, levels=[-18, -6, 0], extent=extent, origin='upper',
                    colors='k', linewidths=[0.5, 0.5, 1.0])
    axes[0].set_title('Solar elevation (°) — terminator at h=0')
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    axes[1].imshow(np.asarray(rgba), extent=extent, origin='upper',
                   aspect='auto')
    axes[1].set_title('Blended result')
    plt.show()
    plt.close(fig)


# =============================================================================
# Gaussian (image-space) blend — original implementation
# =============================================================================

def _gaussian_blend(
    rgb_image: np.ndarray, date: Any, blend_sigma: float,
    lonlat_extent: tuple[float, float, float, float], day_blend: bool,
    hard_terminator_edge: bool, mode: str, fastmode_xy: tuple[int, int],
    min_blend_pct: float, inspect_results: bool,
) -> npt.NDArray[np.float64]:
    """Cartopy Nightshade mask smoothed by a fixed-width Gaussian.

    The terminator comes from cartopy's binary day/night polygon; the soft
    edge is a Gaussian blur of that mask whose width is fixed in *pixels*
    (``blend_sigma``). See :func:`make_nightshade_blend` for how this
    compares to the physical ``'elevation'`` blend.
    """
    if not _HAS_CARTOPY:
        raise ImportError(
            "cartopy is required for make_nightshade_blend(blend='gaussian'). "
            'Install with: conda install -c conda-forge cartopy '
            "(or use blend='elevation', which needs neither cartopy nor scipy).")
    if not _HAS_SCIPY:
        raise ImportError(
            "scipy is required for make_nightshade_blend(blend='gaussian'). "
            "Install with: pip install scipy "
            "(or use blend='elevation', which needs neither cartopy nor scipy).")

    # cartopy duck-types the date (it calls .utcoffset()), so a Time or an ISO
    # string reaches it as an AttributeError. The 'elevation' branch parses
    # every accepted type itself; normalize here so both branches honor the
    # same contract.
    from .._timeinput import _to_datetime
    nshade = Nightshade(_to_datetime(date, caller='make_nightshade_blend'),
                        alpha=0.2)
    geoms = list(nshade.geometries())
    path = _shapely_geoms_to_path(geoms)

    # Create pixel coordinate arrays
    if mode == 'fast':
        fx, fy = fastmode_xy
        x = np.linspace(lonlat_extent[0], lonlat_extent[1], fx)
        y = np.linspace(lonlat_extent[2], lonlat_extent[3], fy)
        if min_blend_pct != 0:
            blend_sigma = max(
                blend_sigma / rgb_image.shape[1] * fx,
                (min_blend_pct / 100) * fx)
    else:
        x = np.linspace(lonlat_extent[0], lonlat_extent[1], rgb_image.shape[1])
        y = np.linspace(lonlat_extent[2], lonlat_extent[3], rgb_image.shape[0])

    xx, yy = np.meshgrid(x, y)

    # Determine which pixels are in the nightshade polygon
    pts_proj = nshade.crs.transform_points(
        ccrs.PlateCarree(), *np.vstack([xx.ravel(), yy.ravel()]))
    msk_night = path.contains_points(pts_proj[:, :2]).reshape(xx.shape)

    # Apply gaussian blur for smooth transition
    msk_fuzzy = gaussian_filter(msk_night.astype(float), blend_sigma)

    if mode == 'fast':
        # Upscale masks to full image resolution
        ratio = rgb_image.shape[1] / fx
        f_fuzzy = interp.RectBivariateSpline(
            np.arange(fy), np.arange(fx), msk_fuzzy)
        msk_fuzzy = f_fuzzy(np.arange(rgb_image.shape[0]) / ratio,
                            np.arange(rgb_image.shape[1]) / ratio)
        f_hard = interp.RectBivariateSpline(
            np.arange(fy), np.arange(fx), msk_night)
        msk_night = np.around(
            f_hard(np.arange(rgb_image.shape[0]) / ratio,
                   np.arange(rgb_image.shape[1]) / ratio), decimals=0)

    if hard_terminator_edge:
        # Night interior stays fully opaque; blend extends into daytime only
        msk_fuzzy *= (msk_night - 1) * -1  # zero out interior night pixels
        from astropy.visualization import LogStretch
        msk_fuzzy = LogStretch()(
            _rescale_range(msk_fuzzy, 0, 1))
        msk_fuzzy += msk_night

    # Flip vertically for matplotlib's origin='lower' convention
    msk_fuzzy = np.flip(msk_fuzzy, axis=0)

    img_mask = (1 - msk_fuzzy) if day_blend else msk_fuzzy

    # Assemble RGBA
    rgba = np.zeros(rgb_image.shape[:2] + (4,))
    rgba[:, :, :3] = rgb_image.copy()
    rgba[:, :, 3] = img_mask
    rgba = rgba.clip(0., 1.)

    if inspect_results:
        _inspect_nightshade(rgb_image, rgba, nshade, img_mask, lonlat_extent)

    return rgba


def _rescale_range(arr: npt.ArrayLike, new_min: float = 0,
                   new_max: float = 1) -> npt.NDArray[np.float64]:
    """Rescale array to [new_min, new_max] range."""
    arr = np.asarray(arr, dtype=float)
    old_min = np.nanmin(arr)
    old_max = np.nanmax(arr)
    if old_max == old_min:
        return np.full_like(arr, (new_min + new_max) / 2)
    return new_min + (arr - old_min) * (new_max - new_min) / (old_max - old_min)


def _inspect_nightshade(rgb: npt.ArrayLike, rgba: npt.ArrayLike, nshade: Any,
                        mask: npt.ArrayLike, extent: Any) -> None:
    """Diagnostic plot for make_nightshade_blend()."""
    if not _HAS_CARTOPY:
        return
    plt.clf()
    plt.close('all')
    ax1: Any = plt.subplot(211, projection=ccrs.PlateCarree(),
                           transform=nshade.crs)
    ax1.imshow(rgb, extent=extent, transform=ccrs.PlateCarree(),
               origin='upper', zorder=1)
    ax1.add_feature(nshade, facecolor='r')
    ax1.set_title('Original + Nightshade')
    ax2 = plt.subplot(212, projection=ccrs.PlateCarree(),
                      transform=nshade.crs, sharex=ax1, sharey=ax1)
    ax2.imshow(rgba, extent=extent, transform=ccrs.PlateCarree(),
               origin='upper', zorder=2)
    ax2.set_title('Blended result')
    plt.show()
    plt.clf()
    plt.close('all')


# =============================================================================
# Formatting Helpers
# =============================================================================
