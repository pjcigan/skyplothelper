"""Astroquery-backed name resolution and image fetching.

Optional dependency: ``astroquery``. If not installed, calling these will
raise an informative ImportError. ``overlay_cutout`` additionally needs
``reproject``.
"""

from __future__ import annotations

import socket
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

# Default client timeout (seconds) for the remote query wrappers. A live
# service with no timeout can hang a caller in an uninterruptible socket read;
# this bounds the wait and lets a slow/hung service fail fast instead.
_DEFAULT_QUERY_TIMEOUT = 60.0


@contextmanager
def _socket_timeout(seconds: float | None) -> Iterator[None]:
    """Temporarily set the default socket timeout for the enclosed query.

    Forces sockets opened during the query to time out (raising instead of
    blocking forever on a stalled read), then restores the previous default.
    ``None`` leaves the global default untouched.
    """
    old = socket.getdefaulttimeout()
    if seconds is not None:
        socket.setdefaulttimeout(float(seconds))
    try:
        yield
    finally:
        socket.setdefaulttimeout(old)

try:
    import astroquery  # noqa: F401  (probe import for the optional-dep flag)
    _HAS_ASTROQUERY = True
except ImportError:
    _HAS_ASTROQUERY = False


def _require_astroquery(submodule: str) -> Any:
    """Import check for astroquery submodules.

    Also patches known issues in older astroquery versions:
    - SkyView.URL: older versions use http:// which NASA now silently
      drops; patched to https://.
    - SkyView.TIMEOUT: older versions have no default timeout, causing
      requests to hang indefinitely; set to 60s.
    """
    try:
        import importlib
        mod = importlib.import_module(f'astroquery.{submodule}')
    except ImportError:
        raise ImportError(
            f"astroquery.{submodule} is required for this function. "
            "Install with: pip install astroquery")

    if submodule == 'skyview' and hasattr(mod, 'SkyView'):
        sv = mod.SkyView
        if hasattr(sv, 'URL') and sv.URL.startswith('http://'):
            sv.URL = sv.URL.replace('http://', 'https://', 1)
        if not getattr(sv, 'TIMEOUT', None):
            sv.TIMEOUT = 60

    return mod


def resolve_name(name: str, service: str = 'simbad') -> Any:
    """
    Resolve an astronomical object name to sky coordinates.

    Parameters
    ----------
    name : str
        Object name (e.g. 'M31', 'Crab Nebula', '3C273', 'Sgr A*').
    service : str
        Name resolution service: 'simbad' (default), 'ned', or 'all'
        (tries SIMBAD first, then NED on failure).

    Returns
    -------
    coord : SkyCoord
        ICRS coordinates of the resolved object.

    Examples
    --------
    >>> coord = resolve_name('M31')
    >>> print(coord.to_string('hmsdms'))

    >>> coord = resolve_name('NGC 1275', service='ned')
    """
    if service.lower() in ('simbad', 'all'):
        try:
            return SkyCoord.from_name(name, frame='icrs')
        except Exception as e:
            if service.lower() == 'simbad':
                raise
            # Fall through to NED
            simbad_err = e

    if service.lower() in ('ned', 'all'):
        try:
            mod = _require_astroquery('ipac.ned')
            result = mod.Ned.query_object(name)
            ra = float(result['RA'][0])
            dec = float(result['DEC'][0])
            return SkyCoord(ra, dec, unit='deg', frame='icrs')
        except Exception as e:
            if service.lower() == 'all':
                raise ValueError(
                    f"Could not resolve '{name}' via SIMBAD ({simbad_err}) "
                    f"or NED ({e})")
            raise

    raise ValueError(f"Unknown service '{service}'. Use 'simbad', 'ned', or 'all'.")


def _normalize_simbad_table(table: Any) -> Any:
    """Normalize a SIMBAD result to a version-stable schema.

    astroquery's SIMBAD output changed with its TAP migration: newer versions
    return lowercase ``main_id`` / ``ra`` / ``dec`` (RA/Dec as float **degrees**),
    where older versions gave uppercase ``MAIN_ID`` / ``RA`` / ``DEC`` (RA/Dec as
    sexagesimal **strings**). This guarantees ``main_id`` (str) plus ``ra`` /
    ``dec`` as float degrees regardless of the installed astroquery, so callers
    get a stable contract. Other columns pass through unchanged; an unrecognized
    schema (no id/ra/dec found) is returned as-is.
    """
    if table is None or len(table) == 0:
        return table
    lower = {c.lower(): c for c in table.colnames}

    # Identifier column -> 'main_id'.
    id_src = lower.get('main_id')
    if id_src is None:
        for cand in ('id', 'typed_id', 'user_specified_id'):
            if cand in lower:
                id_src = lower[cand]
                break
    if id_src is not None and id_src != 'main_id':
        if 'main_id' in table.colnames:
            table.remove_column('main_id')
        table.rename_column(id_src, 'main_id')

    # RA/Dec -> float degrees in lowercase 'ra'/'dec' (in place, keeping order).
    ra_src, dec_src = lower.get('ra'), lower.get('dec')
    if ra_src is None or dec_src is None:
        return table                       # unknown schema — pass through
    ra_col, dec_col = table[ra_src], table[dec_src]
    if ra_col.dtype.kind in ('U', 'S', 'O'):
        # Old schema: sexagesimal strings — RA in hours, Dec in degrees.
        coords = SkyCoord([str(x) for x in ra_col], [str(x) for x in dec_col],
                          unit=(u.hourangle, u.deg))
        ra_deg, dec_deg = coords.ra.deg, coords.dec.deg
    else:
        ra_deg, dec_deg = np.asarray(ra_col, dtype=float), np.asarray(dec_col, dtype=float)
    table[ra_src] = ra_deg
    table[dec_src] = dec_deg
    for src, canon in ((ra_src, 'ra'), (dec_src, 'dec')):
        if src != canon:
            if canon in table.colnames:
                table.remove_column(canon)
            table.rename_column(src, canon)
    return table


def query_simbad(name_or_coord: Any, radius: Any = None,
                 timeout: float | None = _DEFAULT_QUERY_TIMEOUT) -> Any:
    """
    Query SIMBAD for basic source information.

    Parameters
    ----------
    name_or_coord : str or SkyCoord
        Object name or sky coordinates.
    radius : Quantity or float, optional
        Search radius for cone search (when using coordinates).
        If float, interpreted as arcseconds. Default 10 arcsec.
    timeout : float or None, optional
        Client timeout in seconds (default 60). Bounds the wait so a
        slow/hung service fails fast instead of blocking; ``None`` disables it.

    Returns
    -------
    result : astropy.table.Table
        SIMBAD query result, normalized to a **version-stable schema**:
        ``main_id`` (str) and ``ra`` / ``dec`` as float **degrees**, regardless
        of the installed astroquery (which otherwise varies the column names and
        RA/Dec format across versions — see :func:`_normalize_simbad_table`).
        The service's other columns pass through unchanged.

    Examples
    --------
    >>> tbl = query_simbad('3C273')
    >>> ra_deg, dec_deg = tbl['ra'][0], tbl['dec'][0]

    >>> tbl = query_simbad(SkyCoord(180, 45, unit='deg'), radius=60)
    """
    mod = _require_astroquery('simbad')
    simbad = mod.Simbad()
    if timeout is not None:
        simbad.TIMEOUT = timeout            # astroquery's own request timeout

    with _socket_timeout(timeout):
        if isinstance(name_or_coord, str):
            result = simbad.query_object(name_or_coord)
        elif isinstance(name_or_coord, SkyCoord):
            if radius is None:
                radius = 10 * u.arcsec
            elif not hasattr(radius, 'unit'):
                radius = float(radius) * u.arcsec
            result = simbad.query_region(name_or_coord, radius=radius)
        else:
            raise TypeError(
                "name_or_coord must be a string or SkyCoord, "
                f"got {type(name_or_coord)}")

    return _normalize_simbad_table(result)


def _find_deg_col(colnames: Sequence[str], axis: str) -> str | None:
    """Find a degree-valued RA/Dec column by fuzzy name (ra, ra_deg, RA(deg),
    RAJ2000, ...) — a defensive net for schema variants."""
    for c in colnames:
        k = c.lower().replace(' ', '').replace('_', '')
        if k == axis or k == axis + 'deg' or k.startswith(axis + '(') or k == axis + 'j2000':
            return c
    return None


def _normalize_ned_table(table: Any) -> Any:
    """Augment a NED result with stable lowercase ``main_id`` / ``ra`` / ``dec``
    (float degrees), *preserving* NED's native columns ('Object Name', 'RA',
    'DEC', 'Type', ...). NED's schema is stable and callers/tutorials use the
    native names, so this adds the canonical columns rather than renaming.
    """
    if table is None or len(table) == 0:
        return table
    lower = {c.lower(): c for c in table.colnames}
    if 'main_id' not in table.colnames:
        for cand in ('object name', 'object_name', 'objname', 'main_id'):
            if cand in lower:
                table['main_id'] = table[lower[cand]]
                break
    ra_src = lower.get('ra') or _find_deg_col(table.colnames, 'ra')
    dec_src = lower.get('dec') or _find_deg_col(table.colnames, 'dec')
    if ra_src is not None and dec_src is not None:
        if 'ra' not in table.colnames:
            table['ra'] = np.asarray(table[ra_src], dtype=float)
        if 'dec' not in table.colnames:
            table['dec'] = np.asarray(table[dec_src], dtype=float)
    return table


def query_ned(name_or_coord: Any, radius: Any = None,
              timeout: float | None = _DEFAULT_QUERY_TIMEOUT) -> Any:
    """
    Query NED (NASA/IPAC Extragalactic Database) for source information.

    Parameters
    ----------
    name_or_coord : str or SkyCoord
        Object name or sky coordinates.
    radius : Quantity or float, optional
        Search radius for cone search. If float, arcseconds.
    timeout : float or None, optional
        Client timeout in seconds (default 60); ``None`` disables it.

    Returns
    -------
    result : astropy.table.Table
        NED's native table ('Object Name', 'RA', 'DEC', 'Type', ...) plus
        stable lowercase ``main_id`` / ``ra`` / ``dec`` (float degrees) added
        for a version-consistent contract shared with :func:`query_simbad`.

    Examples
    --------
    >>> tbl = query_ned('NGC 1275')
    >>> ra_deg, dec_deg = tbl['ra'][0], tbl['dec'][0]

    >>> tbl = query_ned(SkyCoord(49.95, 41.51, unit='deg'), radius=120)
    """
    mod = _require_astroquery('ipac.ned')
    if timeout is not None:
        mod.Ned.TIMEOUT = timeout

    with _socket_timeout(timeout):
        if isinstance(name_or_coord, str):
            result = mod.Ned.query_object(name_or_coord)
        elif isinstance(name_or_coord, SkyCoord):
            if radius is None:
                radius = 1 * u.arcmin
            elif not hasattr(radius, 'unit'):
                radius = float(radius) * u.arcsec
            result = mod.Ned.query_region(name_or_coord, radius=radius)
        else:
            raise TypeError(
                "name_or_coord must be a string or SkyCoord, "
                f"got {type(name_or_coord)}")

    return _normalize_ned_table(result)


def download_skyview(coord: SkyCoord | str, survey: str = 'DSS2 Red', size: float = 0.25,
                     pixels: int = 500,
                     cache: bool = True) -> tuple[np.ndarray, Any]:
    """
    Download an image cutout from SkyView.

    Parameters
    ----------
    coord : str or SkyCoord
        Center position (object name or SkyCoord).
    survey : str
        SkyView survey name. Common options:

        - Optical: 'DSS2 Red', 'DSS2 Blue', 'DSS2 IR', 'SDSSg', 'SDSSr'
        - IR: '2MASS-J', '2MASS-H', '2MASS-K', 'WISE 3.4', 'WISE 12'
        - Radio: 'NVSS', 'FIRST', 'VLSS'
        - UV: 'GALEX Near UV', 'GALEX Far UV'
        - X-ray: 'RASS', 'SwiftXRT'

        Use ``list_skyview_surveys()`` for the full list.
    size : float
        Field of view in degrees.
    pixels : int
        Image size in pixels (square).
    cache : bool
        Use astroquery cache. Default True.

    Returns
    -------
    data : ndarray
        2D image array.
    header : astropy.io.fits.Header
        FITS header with WCS information.

    Examples
    --------
    >>> data, hdr = download_skyview('M51', survey='DSS2 Red', size=0.3)
    >>> data, hdr = download_skyview(
    ...     SkyCoord(10.684, 41.269, unit='deg'),
    ...     survey='2MASS-J', size=0.5)
    """
    mod = _require_astroquery('skyview')

    if isinstance(coord, str):
        coord = resolve_name(coord)

    hdu_list = mod.SkyView.get_images(
        position=coord, survey=[survey],
        radius=size * u.deg,
        pixels=pixels, cache=cache
    )

    if not hdu_list or len(hdu_list) == 0:
        raise RuntimeError(
            f"SkyView returned no images for survey='{survey}'")

    hdu = hdu_list[0][0]
    return hdu.data.astype(float), hdu.header


def download_hips(coord: SkyCoord | str, hips_id: str = 'CDS/P/DSS2/red',
                  size: float = 0.25,
                  pixels: int = 500,
                  fmt: str = 'fits') -> tuple[np.ndarray, Any]:
    """
    Download an image cutout from the CDS HiPS2FITS service.

    HiPS (Hierarchical Progressive Surveys) provides all-sky image
    data at multiple resolutions. This function fetches a cutout
    projected as a tangent-plane FITS image.

    Parameters
    ----------
    coord : str or SkyCoord
        Center position.
    hips_id : str
        HiPS survey identifier. Common options:

        - 'CDS/P/DSS2/red' — DSS2 Red (default)
        - 'CDS/P/DSS2/blue' — DSS2 Blue
        - 'CDS/P/2MASS/H' — 2MASS H-band
        - 'CDS/P/GALEXGR6/AIS/NUV' — GALEX NUV
        - 'CDS/P/Fermi/color' — Fermi all-sky
        - 'CDS/P/HLA/B' — Hubble Legacy Archive blue

        Browse all at: https://aladin.cds.unistra.fr/hips/list
    size : float
        Field of view in degrees.
    pixels : int
        Image size in pixels.
    fmt : str
        Output format: 'fits' or 'png'.

    Returns
    -------
    data : ndarray
        2D image array (for FITS) or RGB array (for PNG).
    header : astropy.io.fits.Header or None
        FITS header (None for PNG format).

    Examples
    --------
    >>> data, hdr = download_hips('Crab Nebula', size=0.15)
    >>> data, hdr = download_hips(
    ...     SkyCoord(83.63, 22.01, unit='deg'),
    ...     hips_id='CDS/P/2MASS/K', size=0.3)
    """
    if isinstance(coord, str):
        coord = resolve_name(coord)

    # hips2fits via astroquery.hips2fits (if available)
    try:
        from astroquery.hips2fits import hips2fits

        wcs_hips = WCS(naxis=2)
        wcs_hips.wcs.crpix = [pixels / 2 + 0.5, pixels / 2 + 0.5]
        wcs_hips.wcs.cdelt = [-size / pixels, size / pixels]
        wcs_hips.wcs.crval = [coord.ra.deg, coord.dec.deg]
        wcs_hips.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        # hips2fits reads the output image size off the WCS itself; without
        # pixel_shape it raises "The WCS passed does not contain the size of
        # the pixel image."
        wcs_hips.pixel_shape = (pixels, pixels)

        result = hips2fits.query_with_wcs(
            hips=hips_id,
            wcs=wcs_hips,
            get_query_payload=False,
            format=fmt,
        )

        if fmt == 'fits':
            return result[0].data.astype(float), result[0].header
        else:
            # PNG: result is a PIL Image or bytes
            return np.array(result), None

    except ImportError:
        # Fallback: direct URL fetch
        import io
        import urllib.request

        url = (
            f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
            f"?hips={hips_id}"
            f"&ra={coord.ra.deg}&dec={coord.dec.deg}"
            f"&fov={size}&width={pixels}&height={pixels}"
            f"&projection=TAN&format={fmt}"
        )

        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()

        if fmt == 'fits':
            from astropy.io import fits
            hdul = fits.open(io.BytesIO(content))
            return hdul[0].data.astype(float), hdul[0].header
        else:
            from matplotlib.image import imread as mpl_imread
            img = mpl_imread(io.BytesIO(content))
            return img, None


def list_skyview_surveys(pattern: str | None = None) -> list[str] | None:
    """
    List available SkyView survey names.

    Parameters
    ----------
    pattern : str, optional
        Filter surveys by substring match (case-insensitive).
        E.g. ``list_skyview_surveys('2MASS')``

    Returns
    -------
    surveys : list of str
        Matching survey names.
    """
    mod = _require_astroquery('skyview')

    all_surveys = mod.SkyView.survey_dict

    if pattern is None:
        for category, names in sorted(all_surveys.items()):
            print(f"\n{category}:")
            for name in names:
                print(f"  {name}")
        return None

    matches = []
    pat = pattern.lower()
    for category, names in all_surveys.items():
        for name in names:
            if pat in name.lower():
                matches.append(name)

    if matches:
        for m in sorted(matches):
            print(f"  {m}")
    else:
        print(f"  No surveys matching '{pattern}'")
    return matches




def resolve_names(names: Sequence[str], service: str = 'simbad',
                  on_error: str = 'warn') -> tuple[Any, list[str]]:
    """
    Batch-resolve a list of object names to sky coordinates.

    Parameters
    ----------
    names : list of str
        Object names to resolve.
    service : str
        'simbad' (default), 'ned', or 'all'.
    on_error : str
        'warn' (default) — skip failures with a warning.
        'raise' — raise on first failure.
        'skip' — silently skip failures.

    Returns
    -------
    coords : SkyCoord
        Array of resolved coordinates. Failed lookups are set to
        NaN coordinates and flagged in ``failed``.
    failed : list of str
        Names that could not be resolved.

    Examples
    --------
    >>> coords, failed = resolve_names(['M31', 'M51', 'NGC1275'])
    >>> print(coords.to_string('hmsdms'))
    """
    ra_list, dec_list = [], []
    failed = []

    for name in names:
        try:
            c = resolve_name(name, service=service)
            ra_list.append(c.ra.deg)
            dec_list.append(c.dec.deg)
        except Exception as e:
            ra_list.append(np.nan)
            dec_list.append(np.nan)
            failed.append(name)
            if on_error == 'raise':
                raise
            elif on_error == 'warn':
                warnings.warn(f"Could not resolve '{name}': {e}")

    coords = SkyCoord(ra=ra_list, dec=dec_list, unit='deg', frame='icrs')
    return coords, failed


def search_vizier(catalog_id: str, coord: SkyCoord | tuple[float, float] | str, radius: Any = 5,
                  columns: Sequence[str] | None = None,
                  row_limit: int = 5000) -> Any:
    """
    Query a Vizier catalog by cone search.

    Thin wrapper around ``astroquery.vizier.Vizier.query_region``
    for the common use case of fetching sources around a position.

    Parameters
    ----------
    catalog_id : str
        Vizier catalog identifier (e.g. 'II/246' for 2MASS PSC,
        'I/355' for Gaia DR3, 'VIII/65' for NVSS).
    coord : str, SkyCoord, or (ra, dec)
        Center position: an object name, a SkyCoord, or a bare
        ``(ra, dec)`` pair in degrees (like the ``catalog`` helpers).
    radius : float or Quantity
        Search radius. If float, interpreted as arcminutes.
    columns : list of str, optional
        Column names to retrieve. If None, returns all columns.
    row_limit : int
        Maximum number of rows. Default 5000. Use -1 for unlimited.
        Vizier truncates server-side, so a result that comes back at
        exactly this length is probably incomplete; a warning is issued
        when that happens, since a truncated table otherwise looks
        exactly like a complete one.

    Returns
    -------
    result : astropy.table.Table or None
        Query result, or None if no sources found.

    Examples
    --------
    >>> # 2MASS sources within 5' of M31
    >>> tbl = search_vizier('II/246', 'M31', radius=5)

    >>> # Gaia DR3 with specific columns
    >>> tbl = search_vizier('I/355/gaiadr3', SkyCoord(180, 45, unit='deg'),
    ...                     radius=10, columns=['RA_ICRS', 'DE_ICRS', 'Gmag'])
    """
    mod = _require_astroquery('vizier')

    if isinstance(coord, str):
        coord = resolve_name(coord)
    elif not hasattr(coord, 'frame') and not hasattr(coord, 'ra'):
        # Accept a bare (ra, dec) pair in degrees, like the catalog helpers
        # (astroquery's query_region rejects tuples on its own).
        ra, dec = coord
        from astropy.coordinates import SkyCoord
        coord = SkyCoord(float(ra), float(dec), unit='deg')

    if not hasattr(radius, 'unit'):
        radius = float(radius) * u.arcmin

    viz = mod.Vizier(columns=columns or ['**'], row_limit=row_limit)
    results = viz.query_region(coord, radius=radius, catalog=catalog_id)

    if results is None or len(results) == 0:
        return None

    tbl = results[0]
    # Vizier truncates server-side at row_limit and says nothing. A truncated
    # table is indistinguishable from a complete one -- it plots perfectly
    # plausibly -- so a crowded field silently becomes a partial catalog.
    # Documenting the limit cannot tell a user that THEIR query hit it.
    if row_limit > 0 and len(tbl) >= row_limit:
        warnings.warn(
            f"search_vizier: got {len(tbl)} rows, which is the row_limit "
            f"({row_limit}) -- the catalog result is probably truncated and "
            f"this field may hold more sources. Raise row_limit, or pass "
            f"row_limit=-1 for no limit.", stacklevel=2)
    return tbl


def overlay_cutout(ax: Any, coord: SkyCoord | str, survey: str = 'DSS2 Red',
                   size: float | None = None,
                   cmap: str = 'gray_r', alpha: float = 0.8, zorder: int = 0,
                   **kwargs: Any) -> Any:
    """
    Download and overlay a sky image cutout on an existing WCSAxes.

    One-liner for adding background images to plots: downloads a
    cutout from SkyView, reprojects it to match the axes' WCS, and
    displays it via ``imshow``.

    Parameters
    ----------
    ax : WCSAxes
        Target axes (must have a WCS projection).
    coord : str or SkyCoord
        Center position for the cutout.
    survey : str
        SkyView survey name (default 'DSS2 Red').
    size : float, optional
        Cutout FOV in degrees. If None, auto-computed from the
        axes' WCS field of view (with 20% padding).
    cmap : str
        Colormap for the cutout display.
    alpha : float
        Transparency of the overlay.
    zorder : int
        Drawing order (default 0 = below other artists).
    **kwargs
        Additional kwargs passed to ``ax.imshow()``.

    Returns
    -------
    im : AxesImage
        The imshow artist, or None if download/reproject failed.

    Examples
    --------
    >>> overlay_cutout(ax, 'M51', survey='DSS2 Red')
    >>> overlay_cutout(ax, SkyCoord(83.63, 22.01, unit='deg'),
    ...               survey='2MASS-J', alpha=0.5)
    """
    try:
        from reproject import reproject_interp
    except ImportError:
        warnings.warn("reproject is required for overlay_cutout. "
                      "Install with: pip install reproject")
        return None

    # Get the axes' WCS and determine FOV
    ax_wcs = ax.wcs if hasattr(ax, 'wcs') else WCS(ax.header)

    if size is None:
        # Estimate FOV from pixel scale and axes size
        try:
            pix_scales = proj_plane_pixel_scales(ax_wcs) * 3600  # arcsec
            nx = ax_wcs.pixel_shape[0] if ax_wcs.pixel_shape else 500
            ny = ax_wcs.pixel_shape[1] if ax_wcs.pixel_shape else 500
            size = max(nx * pix_scales[0], ny * pix_scales[1]) / 3600 * 1.2
        except Exception:
            size = 0.25  # fallback

    # Download
    try:
        data, hdr = download_skyview(coord, survey=survey, size=size,
                                     pixels=max(500, int(size * 2000)))
    except Exception as e:
        warnings.warn(f"overlay_cutout download failed: {e}")
        return None

    # Reproject to match axes WCS
    try:
        target_hdr = ax_wcs.to_header()
        # Need NAXIS for reproject
        if 'NAXIS1' not in target_hdr:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            target_hdr['NAXIS1'] = int(xlim[1] - xlim[0])
            target_hdr['NAXIS2'] = int(ylim[1] - ylim[0])
        target_hdr['NAXIS'] = 2

        reprojected, _ = reproject_interp(
            (data, WCS(hdr)), target_hdr,
            shape_out=(int(target_hdr['NAXIS2']), int(target_hdr['NAXIS1'])))
        # NaN is left in place deliberately. matplotlib renders it fully
        # transparent for every colormap, so where the cutout underfills the
        # axes nothing is painted. Filling with 0 instead painted those
        # regions an OPAQUE colormap value -- invisible only in the specific
        # case of gray_r on a white page, and an 80%-opaque white rectangle
        # on a dark theme. It also coupled this line to the cmap= default two
        # functions away: changing one without the other produced solid
        # corners.
    except Exception as e:
        warnings.warn(f"overlay_cutout reproject failed: {e}")
        return None

    # Display
    im = ax.imshow(reprojected, origin='lower', cmap=cmap,
                   alpha=alpha, zorder=zorder, **kwargs)
    return im
