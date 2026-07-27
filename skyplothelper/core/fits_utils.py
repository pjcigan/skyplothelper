"""FITS header pixel-scale, beam, and image utilities."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import astropy.io.fits as pyfits
import numpy as np
import numpy.typing as npt
from astropy.wcs import WCS


def getcdelts(hdrin: pyfits.Header, getrot: bool = False) -> tuple[Any, ...]:
    """
    Calculate CDELT1 and CDELT2 from header CD or PC matrix cards.

    Parameters
    ----------
    hdrin : astropy.io.fits.Header
    getrot : bool
        If True, also return the rotation as a standard FITS ``CROTA2``
        in degrees (``arctan2(-CD1_2, CD2_2)``), which round-trips through
        :func:`getcdmatrix` and a standard WCS reader such as astropy.

    Returns
    -------
    cdelt1, cdelt2 : float
        Pixel scale in degrees (cdelt1 typically negative for RA)
    crota : float (if getrot=True)
    """
    if 'CDELT1' in hdrin and 'CDELT2' in hdrin:
        cdelt1 = hdrin['CDELT1']
        cdelt2 = hdrin['CDELT2']
        # CDELT gives the scale directly. The rotation can live in EITHER a
        # PC matrix (modern convention) or CROTA2/CROTA1 (older AIPS style),
        # and a header may carry CDELT alongside either — so consult the PC
        # matrix first and only fall back to CROTA. Form CD = PC*CDELT and use
        # the standard CROTA2 = arctan2(-CD1_2, CD2_2) so the returned angle
        # matches a standard WCS reader (and the CD-matrix branch below).
        if 'PC1_1' in hdrin or 'PC1_2' in hdrin:
            pc1_2 = hdrin.get('PC1_2', 0.0)
            pc2_2 = hdrin.get('PC2_2', 1.0)
            crota = np.degrees(np.arctan2(-pc1_2 * cdelt1, pc2_2 * cdelt2))
        else:
            crota = hdrin.get('CROTA2', hdrin.get('CROTA1', 0.))
    else:
        # No CDELT pair: the transform must come from a full CD matrix.
        # (A PC matrix without CDELT is under-specified — it carries no
        # scale — so it isn't a valid standalone source here.)
        try:
            cd1_1 = hdrin['CD1_1']
            cd1_2 = hdrin['CD1_2']
            cd2_1 = hdrin['CD2_1']
            cd2_2 = hdrin['CD2_2']
        except KeyError:
            raise ValueError(
                'Header has no valid CDELT, CD, or PC matrix cards')
        cdelt1 = np.sqrt(cd1_1**2 + cd1_2**2) * np.sign(cd1_1)
        cdelt2 = np.sqrt(cd2_1**2 + cd2_2**2) * np.sign(cd2_2)
        # Standard FITS CROTA2 (arctan2(-CD1_2, CD2_2)); full-quadrant and
        # round-trips through getcdmatrix / a standard WCS reader.
        crota = np.degrees(np.arctan2(-cd1_2, cd2_2))

    if getrot:
        return cdelt1, cdelt2, crota
    return cdelt1, cdelt2


def getdegperpix(hdrin: pyfits.Header) -> Any:
    """Degrees per pixel side from header."""
    cdelt1, cdelt2 = getcdelts(hdrin)
    if abs(abs(cdelt1) - abs(cdelt2)) < 1e-6:
        return abs(cdelt2)
    else:
        warnings.warn('CDELT1 and CDELT2 differ significantly, returning mean')
        return np.mean([abs(cdelt1), abs(cdelt2)])


def getasecperpix(hdrin: pyfits.Header) -> Any:
    """Arcseconds per pixel side from header (assumes CDELTs in degrees)."""
    return getdegperpix(hdrin) * 3600.


def getsteradperpix(hdrin: pyfits.Header) -> Any:
    """Steradians per pixel from header (assumes CDELTs in degrees)."""
    return np.radians(getdegperpix(hdrin)) ** 2


def getcdmatrix(hdrin: pyfits.Header, crot: float | None = None) -> tuple[Any, Any, Any, Any]:
    """Compute the CD matrix ``(CD1_1, CD1_2, CD2_1, CD2_2)`` from a header.

    Uses the CD cards directly if present, else builds them from a PC matrix
    times CDELT, else from CDELT + a rotation (CROTA2 / CROTA / ``crot``). The
    inverse of :func:`getcdelts`; useful for header surgery / rotation.

    Parameters
    ----------
    hdrin : astropy.io.fits.Header
    crot : float, optional
        Rotation in degrees, used when the header has no CD / PC / CROTA cards.

    Returns
    -------
    cd1_1, cd1_2, cd2_1, cd2_2 : float
    """
    try:
        return (hdrin['CD1_1'], hdrin['CD1_2'],
                hdrin['CD2_1'], hdrin['CD2_2'])
    except KeyError:
        pass
    if crot is None:
        crot = hdrin.get('CROTA2', hdrin.get('CROTA', 0.))
    try:
        cdelt1 = float(hdrin['CDELT1'])
        cdelt2 = float(hdrin['CDELT2'])
    except KeyError:
        raise ValueError('Header has no CD, PC, or CDELT cards to build a '
                         'CD matrix from')
    try:
        cd1_1 = hdrin['PC1_1'] * cdelt1
        cd1_2 = hdrin['PC1_2'] * cdelt1
        cd2_1 = hdrin['PC2_1'] * cdelt2
        cd2_2 = hdrin['PC2_2'] * cdelt2
    except KeyError:
        c, s = np.cos(np.radians(crot)), np.sin(np.radians(crot))
        cd1_1, cd1_2 = cdelt1 * c, cdelt1 * s
        cd2_1, cd2_2 = -cdelt2 * s, cdelt2 * c
    return cd1_1, cd1_2, cd2_1, cd2_2


def beampars_asec_fromhdr(hdrin: pyfits.Header, PAdef: str = 'plot') -> list[Any]:
    """
    Return beam parameters [BMAJ_asec, BMIN_asec, BPA_deg] from FITS header.

    Parameters
    ----------
    hdrin : astropy.io.fits.Header
    PAdef : str
        'plot' (CCW from x-axis) or 'astro' (E from N, add 90 deg)
    """
    try:
        bmaj_asec = hdrin['BMAJ'] * 3600.
        bmin_asec = hdrin['BMIN'] * 3600.
        bpa = hdrin['BPA']
    except KeyError:
        raise KeyError('Beam parameters (BMAJ, BMIN, BPA) not found in header')
    if 'ast' in PAdef.lower():
        bpa += 90.
    return [bmaj_asec, bmin_asec, bpa]


def pixperbeam_from_hdr(hdrin: pyfits.Header) -> Any:
    """Pixels per beam area from header BMAJ/BMIN and pixel scale."""
    FWHMmaj = hdrin['BMAJ']
    FWHMmin = hdrin['BMIN']
    return np.pi / (4. * np.log(2)) * FWHMmaj * FWHMmin / getdegperpix(hdrin)**2


def pixperbeam_from_pars(BMAJ: float, BMIN: float, pixarea: float,
                         bpars: str = 'FWHM') -> Any:
    """Pixels per beam area from beam width parameters and pixel area."""
    if bpars.upper() == 'FWHM':
        scale = np.pi / (4. * np.log(2))
    elif 'SIG' in bpars.upper():
        scale = 2. * np.pi
    else:
        raise ValueError('bpars must be FWHM or sigma')
    return scale * BMAJ * BMIN / pixarea


def makesimpleheader(headerin: pyfits.Header | str, naxis: int = 2,
                     radesys: str | None = None,
                     equinox: float | None = None) -> pyfits.Header:
    """
    Create a simplified 2D FITS header from a complex one (resolving a PC/CD
    matrix into simple CDELTs plus a CROTA2 rotation).

    Parameters
    ----------
    headerin : astropy.io.fits.Header or str (filepath)
    naxis : int
    radesys, equinox : optional overrides

    Returns
    -------
    simpleheader : astropy.io.fits.Header
    """
    if isinstance(headerin, str):
        headerin = pyfits.getheader(headerin)

    wcs_temp = WCS(naxis=naxis)

    # Scale + rotation resolved from the source in one shot: getcdelts reduces
    # a CD or PC matrix to CDELTs plus a standard FITS CROTA2.
    cd1, cd2, crota2 = getcdelts(headerin, getrot=True)

    if naxis > 2:
        wcs_temp.wcs.crpix = [float(headerin['CRPIX1']), float(headerin['CRPIX2']),
                              float(headerin['CRPIX3'])]
        wcs_temp.wcs.crval = [float(headerin['CRVAL1']), float(headerin['CRVAL2']),
                              float(headerin['CRVAL3'])]
        wcs_temp.wcs.ctype = [headerin['CTYPE1'], headerin['CTYPE2'], headerin['CTYPE3']]
        try:
            wcs_temp.wcs.cunit = [headerin['CUNIT1'], headerin['CUNIT2'], headerin['CUNIT3']]
        except KeyError:
            pass
        wcs_temp.wcs.cdelt = [cd1, cd2, headerin['CDELT3']]
    else:
        wcs_temp.wcs.crpix = [float(headerin['CRPIX1']), float(headerin['CRPIX2'])]
        wcs_temp.wcs.crval = [float(headerin['CRVAL1']), float(headerin['CRVAL2'])]
        wcs_temp.wcs.ctype = [headerin['CTYPE1'], headerin['CTYPE2']]
        try:
            wcs_temp.wcs.cunit = [headerin['CUNIT1'], headerin['CUNIT2']]
        except KeyError:
            pass
        wcs_temp.wcs.cdelt = [cd1, cd2]

    # Copy frame info
    for key in ['RADESYS', 'RADECSYS']:
        try:
            wcs_temp.wcs.radesys = headerin[key]
            break
        except KeyError:
            pass
    try:
        wcs_temp.wcs.equinox = headerin['EQUINOX']
    except KeyError:
        pass

    if radesys is not None:
        wcs_temp.wcs.radesys = radesys
    if equinox is not None:
        wcs_temp.wcs.equinox = equinox

    simpleheader = wcs_temp.to_header()
    simpleheader['NAXIS'] = naxis
    for key in ['NAXIS1', 'NAXIS2']:
        try:
            simpleheader[key] = int(headerin[key])
        except KeyError:
            pass

    # Carry over useful cards
    for card in ['BMAJ', 'BMIN', 'BPA', 'BUNIT', 'OBJECT',
                 'TELESCOP', 'RESTFRQ', 'LONPOLE', 'LATPOLE']:
        try:
            simpleheader[card] = headerin[card]
        except KeyError:
            pass

    # Preserve rotation. getcdelts collapsed the source's PC/CD matrix to
    # scale-only CDELTs, so re-inject its (standard FITS) CROTA2 — otherwise a
    # CD/PC-rotated image would silently come out axis-aligned. A CROTA2 can
    # only represent a rotation when the pixels are square, so warn (but still
    # inject the best-fit angle) for a rotated image with non-square pixels,
    # since the CDELT+CROTA2 form cannot capture the resulting skew.
    if abs(crota2) > 1e-9:
        if abs(abs(cd1) - abs(cd2)) > 1e-6 * abs(cd2):
            warnings.warn(
                'makesimpleheader: the source has a rotation with non-square '
                'pixels; a CDELT+CROTA2 header cannot represent the induced '
                'skew, so the injected rotation is approximate. Keep the '
                'original header if exact astrometry is required.')
        simpleheader['CROTA2'] = float(crota2)

    return simpleheader


def convsky2pix(hdrin: pyfits.Header, ra_deg: float, dec_deg: float,
                precise: bool = True, origin: int = 0) -> list[Any]:
    """
    Convert sky coordinates (degrees) to pixel coordinates using header WCS.

    Parameters
    ----------
    hdrin : astropy.io.fits.Header
    ra_deg, dec_deg : float
    precise : bool
        If False, round to nearest integer pixel
    origin : int
        0-based (numpy) or 1-based (FITS) pixel indexing

    Returns
    -------
    [x_pix, y_pix] : list
    """
    wcs = WCS(hdrin, naxis=2)
    pixarr = wcs.wcs_world2pix([[ra_deg, dec_deg]], origin)
    if precise:
        return [pixarr[0][0], pixarr[0][1]]
    return [int(np.round(pixarr[0][0])), int(np.round(pixarr[0][1]))]


def convpix2sky(hdrin: pyfits.Header, x_pix: float, y_pix: float,
                origin: int = 0) -> list[Any]:
    """
    Convert pixel coordinates to sky coordinates (degrees) using header WCS.

    Parameters
    ----------
    hdrin : astropy.io.fits.Header
    x_pix, y_pix : float
    origin : int

    Returns
    -------
    [ra_deg, dec_deg] : list
    """
    wcs = WCS(hdrin, naxis=2)
    skyarr = wcs.wcs_pix2world([[x_pix, y_pix]], origin)
    return [skyarr[0][0], skyarr[0][1]]


def squeeze_image(data: npt.ArrayLike, header: pyfits.Header | None = None,
                  verbose: bool = True) -> tuple[np.ndarray, pyfits.Header | None]:
    """
    Squeeze a FITS image to 2D, cleaning up the header to match.

    Many FITS files have degenerate (length-1) frequency and/or Stokes
    axes even when the data is effectively 2D. This function removes
    those dimensions from the data array and strips the corresponding
    header cards (NAXIS3/4, CTYPE3/4, CRVAL3/4, etc.), preserving
    the metadata values for extraction before removal.

    Truly multi-channel data (where an axis has length > 1) will raise
    a ValueError rather than silently dropping data.

    Parameters
    ----------
    data : ndarray
        FITS image data (2D, 3D, or 4D).
    header : astropy Header, optional
        FITS header. If provided, a cleaned copy is returned with
        extra-axis cards removed.
    verbose : bool
        If True, print info about squeezed dimensions.

    Returns
    -------
    data_2d : ndarray
        2D image array.
    header_2d : Header or None
        Cleaned header (if input header was provided), or None.

    Raises
    ------
    ValueError
        If data has a non-degenerate axis beyond 2D that cannot be
        squeezed (e.g., a true spectral cube with NAXIS3 > 1).

    Examples
    --------
    >>> # Squeeze a 4D FITS array [1, 1, 256, 256] -> [256, 256]
    >>> data_2d, hdr_2d = sph.squeeze_image(hdu.data, hdu.header)

    >>> # Use with quicklook_plot (called automatically internally)
    >>> result = sph.quicklook_plot(data_4d, header=hdr_4d)

    Notes
    -----
    Preserved before removal: CRVAL3 (frequency), CTYPE3 (axis type),
    and CRVAL4 (Stokes) values are stored in the cleaned header as
    comment cards so downstream functions can still extract them for
    metadata display (e.g., frequency in ``quicklook_plot``).
    """
    data = np.asarray(data)

    if data.ndim == 2:
        return data, header.copy() if header is not None else None

    if data.ndim < 2:
        raise ValueError(f"Data must be at least 2D, got {data.ndim}D")

    # Identify degenerate (length-1) axes
    shape = data.shape
    # FITS convention: axes are reversed relative to numpy
    # For a 4D FITS [NAXIS1, NAXIS2, NAXIS3, NAXIS4], numpy shape is
    # (NAXIS4, NAXIS3, NAXIS2, NAXIS1)
    non_spatial = shape[:-2]  # leading axes (Stokes, freq, etc.)

    non_degenerate = [i for i, s in enumerate(non_spatial) if s > 1]
    if non_degenerate:
        # There's a real multi-valued axis - can't squeeze to 2D
        axis_names = []
        for i in non_degenerate:
            fits_axis = data.ndim - i  # convert numpy->FITS axis number
            if header is not None:
                ctype = header.get(f'CTYPE{fits_axis}', f'axis {fits_axis}')
                axis_names.append(f'{ctype} (NAXIS{fits_axis}={shape[i]})')
            else:
                axis_names.append(f'axis {i} (size={shape[i]})')
        raise ValueError(
            f"Cannot squeeze to 2D: non-degenerate axes present: "
            f"{', '.join(axis_names)}. "
            f"Input shape: {shape}. "
            f"Select a slice manually (e.g., data[0, 0, :, :])."
        )

    # All leading axes are length-1 - safe to squeeze
    data_2d = data.squeeze()

    if data_2d.ndim != 2:
        # Pathological case (shouldn't happen with the check above)
        while data_2d.ndim > 2:
            data_2d = data_2d[0]

    if verbose and data.ndim > 2:
        squeezed_axes = data.ndim - 2
        print(f"squeeze_image: {data.shape} -> {data_2d.shape} "
              f"({squeezed_axes} degenerate axis{'es' if squeezed_axes > 1 else ''} removed)")

    if header is None:
        return data_2d, None

    # Clean header: remove extra-axis cards but preserve metadata
    hdr_2d = header.copy()

    # Cards associated with axes 3 and 4
    _AXIS_CARDS = ['NAXIS', 'CTYPE', 'CRVAL', 'CRPIX', 'CDELT',
                   'CUNIT', 'CROTA']

    # Preserve frequency/Stokes info as comments before removing
    for ax_num in [3, 4]:
        ctype = header.get(f'CTYPE{ax_num}', '')
        crval = header.get(f'CRVAL{ax_num}', '')
        if ctype or crval:
            hdr_2d.add_comment(
                f'Original axis {ax_num}: {ctype} = {crval}')

    # Remove the cards
    for ax_num in [3, 4]:
        for card in _AXIS_CARDS:
            key = f'{card}{ax_num}'
            if key in hdr_2d:
                del hdr_2d[key]
        # Also remove PC/CD matrix entries for these axes
        for i in [1, 2, 3, 4]:
            for key_pattern in [f'PC{ax_num}_{i}', f'PC{i}_{ax_num}',
                                f'CD{ax_num}_{i}', f'CD{i}_{ax_num}']:
                if key_pattern in hdr_2d:
                    del hdr_2d[key_pattern]

    # Update NAXIS
    hdr_2d['NAXIS'] = 2
    if 'WCSAXES' in hdr_2d:
        hdr_2d['WCSAXES'] = 2

    return data_2d, hdr_2d


def _axis_cards(axes: Iterable[int]) -> list[str]:
    """WCS card names belonging to the given FITS axis numbers (for the
    header-only dimension strippers). Both PC spellings (``PCn_m`` and the
    zero-padded ``PC0n_0m``) plus CD entries are covered."""
    cards = []
    for ax in axes:
        for base in ('NAXIS', 'CTYPE', 'CRVAL', 'CRPIX', 'CDELT', 'CUNIT',
                     'CROTA'):
            cards.append(f'{base}{ax}')
        for i in (1, 2, 3, 4):
            cards += [f'PC{ax}_{i}', f'PC{i}_{ax}',
                      f'CD{ax}_{i}', f'CD{i}_{ax}',
                      f'PC{ax:02d}_{i:02d}', f'PC{i:02d}_{ax:02d}']
    return cards


def force_hdr_to_2D(hdrin: pyfits.Header) -> pyfits.Header:
    """Return a copy of a header with all 3rd/4th-axis cards removed.

    Strips ``NAXIS3/4``, the ``CTYPE/CRVAL/CRPIX/CDELT/CUNIT/CROTA`` cards for
    axes 3-4, and the associated PC / CD matrix entries, then sets ``NAXIS=2``
    (and ``WCSAXES=2`` if present). Unlike :func:`squeeze_image`, this operates
    on a *bare header* (no data array) — handy for making a cube / Stokes
    header usable by 2-D plotters (WCSAxes, the plotly FITS viewer).
    """
    hdr = hdrin.copy()
    for key in _axis_cards((3, 4)):
        if key in hdr:
            del hdr[key]
    hdr['NAXIS'] = 2
    if 'WCSAXES' in hdr:
        hdr['WCSAXES'] = 2
    return hdr


def force_hdr_to_3D(hdrin: pyfits.Header) -> pyfits.Header:
    """Return a copy of a header with all 4th-axis cards removed (``NAXIS=3``).

    The cube-preserving counterpart to :func:`force_hdr_to_2D` — strips only
    the 4th-axis (e.g. Stokes) cards and sets ``NAXIS=3`` (and ``WCSAXES=3``
    if present).
    """
    hdr = hdrin.copy()
    for key in _axis_cards((4,)):
        if key in hdr:
            del hdr[key]
    hdr['NAXIS'] = 3
    if 'WCSAXES' in hdr:
        hdr['WCSAXES'] = 3
    return hdr


_FLOAT_WCS_CARDS = (
    'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2',
    'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
    'CROTA', 'CROTA2', 'EQUINOX', 'LONPOLE', 'LATPOLE',
)


def force_hdr_floats(hdrin: pyfits.Header) -> pyfits.Header:
    """Return a copy of a header with WCS-critical cards coerced to ``float``.

    Some pipelines write numeric WCS keywords (``CRPIX`` / ``CRVAL`` /
    ``CDELT`` / ``CD`` / ``PC`` / ``CROTA`` / ``EQUINOX`` / ...) as *strings*,
    which breaks astropy WCS parsing. This makes such a header plottable again.
    Cards that are absent or not convertible are left untouched.
    """
    hdr = hdrin.copy()
    for key in _FLOAT_WCS_CARDS:
        if key in hdr:
            try:
                hdr[key] = float(hdr[key])
            except (TypeError, ValueError):
                pass
    return hdr


def header_coord_grids(hdr_or_wcs: pyfits.Header | WCS,
                       shape: tuple[int, int] | None = None,
                       x: npt.ArrayLike | None = None,
                       y: npt.ArrayLike | None = None,
                       return_1d: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel world (lon/lat, e.g. RA/Dec) coordinates for an image grid.

    Builds arrays of the sky coordinates at every pixel of a 2-D image from a
    FITS header or WCS — for hover read-outs, masking by sky position, or any
    coordinate-aware analysis. A proper function replacing the ad-hoc lambdas
    that compute pixel-grid coordinate arrays from CDELT-style header values.

    Parameters
    ----------
    hdr_or_wcs : astropy.io.fits.Header or astropy.wcs.WCS
        Source of the WCS (reduced to its 2-D celestial part if higher-D).
    shape : (ny, nx), optional
        Image shape, needed only for a WCS input that carries no
        ``pixel_shape`` when ``x`` / ``y`` are not given.
    x, y : array-like, optional
        Pixel coordinate vectors to evaluate at (0-based). Default is the full
        ``0 .. nx-1`` / ``0 .. ny-1`` grid; pass a strided subset to match a
        downsampled display.
    return_1d : bool
        If True, return the 1-D coordinate values along each axis (lon across
        the center row, lat down the center column) instead of full 2-D grids
        — cheaper, and exact for a non-rotated WCS.

    Returns
    -------
    lon, lat : ndarray
        ``(ny, nx)`` world-coordinate grids in degrees (or 1-D arrays when
        ``return_1d=True``).
    """
    if isinstance(hdr_or_wcs, WCS):
        wcs = hdr_or_wcs
        if shape is not None:
            ny, nx = int(shape[0]), int(shape[1])
        elif wcs.pixel_shape is not None:
            nx, ny = int(wcs.pixel_shape[0]), int(wcs.pixel_shape[1])
        elif x is not None and y is not None:
            nx, ny = len(np.atleast_1d(x)), len(np.atleast_1d(y))
        else:
            raise ValueError("WCS input needs shape=(ny, nx) or x/y vectors")
    else:
        wcs = WCS(hdr_or_wcs)
        nx, ny = int(hdr_or_wcs['NAXIS1']), int(hdr_or_wcs['NAXIS2'])
    wcs2d = wcs.celestial if getattr(wcs, 'naxis', 2) > 2 else wcs

    if x is None:
        x = np.arange(nx)
    if y is None:
        y = np.arange(ny)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if return_1d:
        cy = y[len(y) // 2] if len(y) else 0.0
        cx = x[len(x) // 2] if len(x) else 0.0
        lon1d, _ = wcs2d.pixel_to_world_values(x, np.full_like(x, cy))
        _, lat1d = wcs2d.pixel_to_world_values(np.full_like(y, cx), y)
        return np.asarray(lon1d), np.asarray(lat1d)

    xx, yy = np.meshgrid(x, y)
    lon, lat = wcs2d.pixel_to_world_values(xx, yy)
    return np.asarray(lon), np.asarray(lat)
