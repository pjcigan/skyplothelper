"""Diagnostics and module-wide registry aggregators.

``describe_wcs`` prints a human-readable WCS summary (center coords, pixel
scale, projection type, FOV, beam parameters). ``saved_plot_size_reducer``
shrinks saved PNGs via PIL palette mode. The ``list_*`` family is
re-exported from each module's home for one-stop discoverability.
"""

from __future__ import annotations

import os
from typing import Any

import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord  # noqa: F401
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# Re-exported list_* aggregators (one-stop discovery from one namespace).
from .cartopy_backend import list_cartopy_projections  # noqa: F401
from .images.levels import list_stretches  # noqa: F401
from .overlays.constellations import list_constellations  # noqa: F401
from .overlays.surveys import list_surveys  # noqa: F401
from .projections.registry import _resolve_projection
from .queries import list_skyview_surveys  # noqa: F401


def describe_wcs(wcs_or_hdr: Any, name: str = '') -> dict[str, Any]:
    """
    Print a human-readable summary of a WCS or FITS header.

    Displays center coordinates (sexagesimal for equatorial), pixel
    scale, projection type, coordinate frame, image dimensions, FOV,
    beam parameters, and telescope/object metadata.  Auto-selects
    display units (deg/arcmin/arcsec/mas/μas) based on scale.

    Parameters
    ----------
    wcs_or_hdr : WCS, Header, or str
        An astropy WCS object, FITS header, or path to a FITS file.
    name : str, optional
        Label for the printout header (e.g. filename or object name).

    Returns
    -------
    info : dict
        Dictionary of extracted WCS properties for programmatic use.
        Keys include: projection, frame, crval1, crval2, center_str,
        pixscale_arcsec, naxis1, naxis2, fov_arcsec, fov_arcmin,
        fov_deg, beam_maj_arcsec, beam_min_arcsec, beam_pa,
        equinox, object, telescope.

    Examples
    --------
    >>> describe_wcs('observation.fits')
    WCS Summary: observation.fits
      Projection:  TAN (Gnomonic)
      Frame:       ICRS (J2000.0)
      Center:      12h30m49.42s +12d23m28.0s (187.706°, 12.391°)
      Pixel scale: 1.000" × 1.000" /pix
      Image size:  512 × 512 px
      FOV:         8.53' × 8.53'
      Beam:        5.000" × 3.000", PA 30.0°
      Object:      M87
      Telescope:   VLBA

    >>> info = describe_wcs(header)
    >>> print(info['fov_arcmin'])
    """
    # Parse input
    if isinstance(wcs_or_hdr, str):
        hdr = pyfits.getheader(wcs_or_hdr)
        wcs = WCS(hdr, naxis=2)
        if not name:
            name = wcs_or_hdr
    elif isinstance(wcs_or_hdr, WCS):
        wcs = wcs_or_hdr
        hdr = wcs.to_header()
    else:
        hdr = wcs_or_hdr
        wcs = WCS(hdr, naxis=2)

    info: dict[str, Any] = {}

    # Projection type
    ctype1 = str(hdr.get('CTYPE1', wcs.wcs.ctype[0] if hasattr(wcs.wcs, 'ctype') else ''))
    proj_code = ctype1.split('-')[-1].strip() if '-' in ctype1 else '???'
    info['projection'] = proj_code

    proj_name = proj_code
    try:
        _, proj_info = _resolve_projection(proj_code)
        proj_name = f"{proj_code} ({proj_info.description})"
    except (ValueError, KeyError, TypeError):
        pass
    info['projection_name'] = proj_name

    # Frame
    ctype_prefix = ctype1[:4].upper()
    frame_map = {'RA--': 'ICRS', 'GLON': 'Galactic', 'GLAT': 'Galactic',
                 'ELON': 'Ecliptic', 'SLON': 'Supergalactic'}
    frame_name = frame_map.get(ctype_prefix, ctype_prefix)
    radesys = hdr.get('RADESYS', hdr.get('RADECSYS', ''))
    if radesys and frame_name == 'ICRS':
        frame_name = radesys
    info['frame'] = frame_name

    # Center coordinates
    crval1 = float(wcs.wcs.crval[0])
    crval2 = float(wcs.wcs.crval[1])
    info['crval1'] = crval1
    info['crval2'] = crval2

    if 'RA' in ctype1.upper():
        sc = SkyCoord(crval1, crval2, unit='deg', frame='icrs')
        center_str = (f"{sc.ra.to_string(unit=u.hourangle, sep='hms', precision=2)} "
                      f"{sc.dec.to_string(unit=u.degree, sep='dms', precision=1, alwayssign=True)}"
                      f" ({crval1:.3f}°, {crval2:.3f}°)")
    else:
        center_str = f"{crval1:.3f}°, {crval2:.3f}°"
    info['center_str'] = center_str

    # Pixel scale
    try:
        pix_scales = proj_plane_pixel_scales(wcs) * 3600  # arcsec
    except Exception:
        try:
            cdelt = wcs.wcs.get_cdelt()
            pix_scales = np.abs(cdelt[:2]) * 3600
        except Exception:
            pix_scales = None
    info['pixscale_arcsec'] = tuple(pix_scales) if pix_scales is not None else None

    def _fmt_scale(asec: float) -> str:
        """Auto-unit format for angular scale in arcsec."""
        if asec >= 60:
            return f"{asec/60:.3f}'"
        elif asec >= 0.1:
            return f'{asec:.3f}"'
        elif asec >= 0.0001:
            return f"{asec*1000:.2f} mas"
        else:
            return f"{asec*1e6:.1f} μas"

    # Image size
    if wcs.pixel_shape:
        nx, ny = int(wcs.pixel_shape[0]), int(wcs.pixel_shape[1])
    else:
        nx = int(hdr.get('NAXIS1', wcs.wcs.crpix[0] * 2))
        ny = int(hdr.get('NAXIS2', wcs.wcs.crpix[1] * 2))
    info['naxis1'] = nx
    info['naxis2'] = ny

    # FOV
    if pix_scales is not None and nx > 0 and ny > 0:
        fov_x = nx * pix_scales[0]
        fov_y = ny * pix_scales[1]
        info['fov_arcsec'] = (fov_x, fov_y)
        info['fov_arcmin'] = (fov_x / 60, fov_y / 60)
        info['fov_deg'] = (fov_x / 3600, fov_y / 3600)

    # Beam
    bmaj = hdr.get('BMAJ', None)
    bmin = hdr.get('BMIN', None)
    bpa = hdr.get('BPA', 0)
    if bmaj is not None:
        info['beam_maj_arcsec'] = float(bmaj) * 3600
        info['beam_min_arcsec'] = float(bmin) * 3600 if bmin else info['beam_maj_arcsec']
        info['beam_pa'] = float(bpa)

    # Equinox
    equinox = hdr.get('EQUINOX', hdr.get('EPOCH', None))
    if equinox is not None:
        info['equinox'] = float(equinox)

    # --- Print ---
    label = f"WCS Summary: {name}" if name else "WCS Summary"
    print(label)
    print(f"  Projection:  {proj_name}")
    eq_str = f" (J{info['equinox']:.1f})" if 'equinox' in info else ''
    print(f"  Frame:       {frame_name}{eq_str}")
    print(f"  Center:      {center_str}")

    if pix_scales is not None:
        print(f"  Pixel scale: {_fmt_scale(pix_scales[0])} × "
              f"{_fmt_scale(pix_scales[1])} /pix")

    if nx > 0:
        print(f"  Image size:  {nx} × {ny} px")

    if 'fov_arcsec' in info:
        fx, fy = info['fov_arcsec']
        if max(fx, fy) >= 3600:
            print(f"  FOV:         {fx/3600:.2f}° × {fy/3600:.2f}°")
        elif max(fx, fy) >= 60:
            print(f"  FOV:         {fx/60:.2f}' × {fy/60:.2f}'")
        elif max(fx, fy) >= 0.1:
            print(f'  FOV:         {fx:.2f}" × {fy:.2f}"')
        else:
            print(f"  FOV:         {fx*1000:.1f} mas × {fy*1000:.1f} mas")

    if bmaj is not None:
        print(f'  Beam:        {_fmt_scale(info["beam_maj_arcsec"])} × '
              f'{_fmt_scale(info["beam_min_arcsec"])}, PA {info["beam_pa"]:.1f}°')

    obj = hdr.get('OBJECT', None)
    if obj:
        info['object'] = obj
        print(f"  Object:      {obj}")

    tel = hdr.get('TELESCOP', None)
    instr = hdr.get('INSTRUME', None)
    if tel or instr:
        parts = [p for p in [tel, instr] if p]
        info['telescope'] = ' / '.join(parts)
        print(f"  Telescope:   {info['telescope']}")

    return info


# ===== saved_plot_size_reducer =====

def saved_plot_size_reducer(savepath: str, suffix: str = '',
                            fileformat: str = 'PNG',
                            colordepth: int = 256) -> None:
    """
    Reduce saved image file size by converting to palette mode.

    Requires PIL/Pillow.
    """
    if not _HAS_PIL:
        raise ImportError('PIL/Pillow required for saved_plot_size_reducer')
    im = Image.open(savepath)
    im2 = im.convert('P', palette=Image.Palette.ADAPTIVE, colors=colordepth)
    base = os.path.splitext(savepath)[0]
    outsavepath = base + suffix + '.' + fileformat.lower()
    im2.save(outsavepath, optimize=True, format=fileformat)
