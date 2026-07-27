"""FITS image loading and reprojection helpers.

``load_sky_image`` reads a 2-D RGB(A) raster + builds a matching
equirectangular WCS; ``reproject_background`` and ``reproject_rgb_map``
project arbitrary FITS images onto a target WCS using the optional
``reproject`` package.
"""

from __future__ import annotations

import warnings
from typing import Any

import astropy.io.fits as pyfits
import astropy.units as u  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord  # noqa: F401
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel, proj_plane_pixel_scales

from ..wcs_frame import _resolve_ctype

try:
    from reproject import reproject_interp
    _HAS_REPROJECT = True
except ImportError:
    _HAS_REPROJECT = False


def _require_reproject() -> None:
    if not _HAS_REPROJECT:
        raise ImportError(
            "This functionality requires the reproject package. "
            "Install with: pip install reproject"
        )


# reproject_interp order-string -> scipy.ndimage.map_coordinates spline order.
_ORDER_CODES = {"nearest": 0, "nearest-neighbor": 0, "nearest_neighbor": 0,
                "bilinear": 1, "biquadratic": 2, "bicubic": 3}


def _order_code(order: Any) -> int:
    if isinstance(order, str):
        return _ORDER_CODES.get(order.lower(), 1)
    return int(order)


# Cylindrical projections whose pixel x-axis is a linear function of longitude,
# so a full-sky image in one of them is genuinely periodic along x and its first
# and last columns are neighbors. Deliberately NOT every full-sky projection:
# in AIT/MOL the left and right edges are the projection boundary rather than a
# seam, and wrapping there would fold unrelated sky together.
_LON_PERIODIC_CODES = frozenset({'CAR', 'CEA', 'MER', 'CYP'})


def _lon_is_periodic(wcs: WCS, nx: int) -> bool:
    """Is this source periodic along its pixel x-axis (full 360° of longitude)?"""
    try:
        ctype = str(wcs.wcs.ctype[0]).upper()
        scale = float(abs(proj_plane_pixel_scales(wcs)[0]))
    except Exception:
        return False
    if ctype[-3:] not in _LON_PERIODIC_CODES:
        return False
    if ctype[:4] not in ('RA--', 'GLON', 'ELON', 'TLON', 'SLON', 'HLON'):
        return False
    return bool(scale > 0 and abs(nx * scale - 360.0) < 1.0)


# Body-fixed longitude types, as written by ``pseudofits_from_image(geo=True)``
# and ``make_planet_frame``. A drape between one of these and a celestial frame
# is the mismatch the module docstring warns about.
_BODY_FIXED_LON = ('TLON', 'PLON', 'HLON')


def _lon_type(wcs: WCS) -> str:
    try:
        return str(wcs.wcs.ctype[0]).upper()[:4]
    except Exception:
        return ''


def _warn_if_geo_celestial_mismatch(source_wcs: WCS, target_wcs: WCS) -> None:
    """Warn when a drape crosses the geographic / celestial boundary.

    Such a pair has no epoch-free transform: ITRS↔ICRS is an Earth-rotation,
    ``obstime``-dependent conversion, so a texture drape (which is not an
    observation at an instant) comes out rotated by an essentially arbitrary
    amount and is not reproducible if the assumed epoch changes.

    The conversion itself is correct and long-documented -- matching the frames
    is the caller's job. What was missing is any runtime signal that they have
    not been matched, so the rotation looked like a rendering bug rather than
    the documented consequence of a mismatch.
    """
    s, t = _lon_type(source_wcs), _lon_type(target_wcs)
    if not s or not t:
        return
    s_geo, t_geo = s in _BODY_FIXED_LON, t in _BODY_FIXED_LON
    if s_geo != t_geo:
        warnings.warn(
            f"reprojecting between a body-fixed frame ({s}) and a celestial "
            f"one ({t}): this applies a real, epoch-dependent Earth-rotation "
            "transform, so the texture will be rotated in longitude and the "
            "result depends on the assumed obstime. Match the frames instead "
            "-- drape a geo=True texture onto make_planet_frame() or "
            "make_wcs_frame(frame='ITRS'), and a geo=False texture onto a "
            "celestial frame.", stacklevel=3)


def _reproject_shared(image: np.ndarray, source_wcs: WCS, target_wcs: WCS,
                      shape_out: Any, order: Any = "bilinear",
                      downscale: float = 1.0) -> np.ndarray:
    """Reproject every channel of *image* through ONE shared coordinate map.

    ``reproject_interp`` rebuilds the output→input coordinate transform on each
    call; for an RGB(A) raster that transform is identical across the color
    channels, so computing it once here and interpolating each channel with
    ``scipy.ndimage.map_coordinates`` is ~3-5x faster than looping
    ``reproject_interp`` per channel (scipy ships with reproject, so this adds
    no dependency). The output matches ``reproject_interp`` to a ~1e-4 mean
    difference; NaN marks pixels outside the source footprint / projection,
    matching reproject.

    ``downscale > 1`` samples the (full-frame) target grid every ``downscale``
    pixels — reproject cost scales with the output pixel count — then upsamples
    the result back to the full frame, a cheap draft-resolution lever.
    """
    from scipy.ndimage import map_coordinates, zoom

    _warn_if_geo_celestial_mismatch(source_wcs, target_wcs)

    ny, nx = int(shape_out[0]), int(shape_out[1])
    icode = _order_code(order)
    d = float(downscale)
    # A grid that always spans the FULL frame (0..ny, 0..nx); coarser when d>1.
    ys = np.arange(0, ny, d) if d > 1.0 else np.arange(ny)
    xs = np.arange(0, nx, d) if d > 1.0 else np.arange(nx)
    xx, yy = np.meshgrid(xs, ys)
    with np.errstate(invalid="ignore", divide="ignore"):
        # pixel_to_pixel, NOT the *_values pair. The *_values APIs are
        # deliberately frame-agnostic: they exchange bare numbers in each
        # WCS's own frame, so feeding target RA/Dec into a GLON/GLAT source
        # WCS is accepted silently and reads them as galactic. That made
        # cross-frame drapes a no-op -- the source layout was resampled onto
        # the target grid untransformed -- while same-frame drapes, where the
        # two frames coincide, stayed correct and hid it.
        src_x, src_y = pixel_to_pixel(target_wcs, source_wcs, xx, yy)
    invalid = ~(np.isfinite(src_x) & np.isfinite(src_y))   # outside projection

    nchan = image.shape[2] if image.ndim == 3 else 0
    planes = image.transpose(2, 0, 1) if nchan else image[None]

    # A full-sky cylindrical source is periodic along x: its first and last
    # columns are neighbors on the sky. map_coordinates does not know that, so
    # a target pixel landing between them interpolated off the array edge and
    # took the fill value instead -- a one-pixel dark chord along the source's
    # seam. Wrap the sampled column, then append a copy of column 0 so the
    # interval [nx-1, nx] interpolates between the true neighbors.
    #
    # Same-frame drapes hid this: there the seam coincides with the map edge.
    # Cross-frame drapes put it mid-map, so fixing the frame transform is what
    # made it visible.
    if _lon_is_periodic(source_wcs, planes.shape[2]):
        src_x = np.mod(src_x, planes.shape[2])
        planes = np.concatenate([planes, planes[:, :, :1]], axis=2)

    coords = np.asarray([src_y, src_x])           # map_coordinates: (row, col)
    gh, gw = xx.shape
    small = np.empty((planes.shape[0], gh, gw), dtype=float)
    with np.errstate(invalid="ignore"):
        for i, plane in enumerate(planes):
            # map_coordinates preserves the input dtype, so an INTEGER texture
            # (uint8 JPG) would come back integer and the off-footprint
            # ``r[invalid] = np.nan`` below could not be assigned. Promote only
            # those — an already-float plane is passed through untouched, since
            # forcing float32 up to float64 perturbs the interpolation (~3e-8
            # on every pixel: harmless numerically, but enough to churn every
            # rendered figure that drapes a float32 texture).
            p = plane if np.issubdtype(plane.dtype, np.floating) else \
                np.asarray(plane, dtype=float)
            r = map_coordinates(p, coords, order=icode, mode="constant",
                                cval=np.nan).reshape(gh, gw)
            r[invalid] = np.nan
            small[i] = r

    if d > 1.0 and (gh, gw) != (ny, nx):          # draft: upsample to full frame
        small = np.stack([
            zoom(np.nan_to_num(small[i], nan=0.0), (ny / gh, nx / gw), order=1)
            for i in range(small.shape[0])])

    return np.moveaxis(small, 0, -1) if nchan else small[0]


def reproject_rgb_map(input_hdu: Any, *args: Any,
                      **kwargs: Any) -> np.ndarray:
    """
    Reproject an RGB(A) FITS HDU onto a target WCS for all-sky or other
    projection plotting. Handles RGBA as well as RGB.

    Extra positional / keyword args are forwarded to
    ``reproject.reproject_interp`` — typically the output header (or WCS) and
    ``shape_out=(ny, nx)``.

    Notes
    -----
    The input HDU's WCS and the target must share a coordinate **system**.
    Reprojecting a geographic texture (``TLON``/``TLAT`` from
    ``pseudofits_from_image(..., geo=True)``) onto a celestial target
    (``RA``/``DEC``), or vice versa, makes reproject apply the real transform
    between the two frames — e.g. an ITRS↔ICRS rotation that slides an Earth
    map by tens of degrees in longitude. Match them: drape a ``geo=True``
    texture onto a globe (:func:`make_planet_frame`) or a flat
    ``make_wcs_frame(..., frame='ITRS')``; use a ``geo=False`` texture with a
    celestial frame. (A legitimate cross-system drape, e.g. Galactic → ICRS,
    *should* transform — this note is only about unintended mismatches.)
    """
    if not _HAS_REPROJECT:
        raise ImportError('reproject package required')
    data = input_hdu.data
    wcs = WCS(input_hdu).celestial

    # Fast path: the target (output header/WCS) + shape are resolvable, so
    # reproject all channels through one shared coordinate map instead of one
    # reproject_interp per channel (identical transform each time).
    target = args[0] if args else kwargs.get('output_projection')
    twcs, shape_out = _resolve_target_wcs_shape(target, kwargs.get('shape_out'))
    if twcs is not None and shape_out is not None:
        order = kwargs.get('order', 'bilinear')
        with np.errstate(invalid='ignore'):
            return _reproject_shared(data, wcs, twcs, shape_out,
                                     order).astype(data.dtype)

    # Fallback (exotic forwarding): the original per-channel loop.
    Nchannels = data.shape[-1]
    with np.errstate(invalid='ignore'):
        reprojected = np.moveaxis(np.stack([
            reproject_interp((data[:, :, i], wcs), *args, **kwargs)[0].astype(data.dtype)
            for i in range(Nchannels)
        ]), 0, -1)
    return reprojected


def _resolve_target_wcs_shape(target: Any, shape_out: Any,
                              ) -> tuple[Any, Any]:
    """Resolve a reproject target (header / WCS / ``(array, wcs)``) to
    ``(target_wcs, (ny, nx))``, or ``(None, None)`` if it can't be determined
    (caller then falls back to ``reproject_interp``)."""
    if target is None:
        return None, None
    twcs: Any = None
    if isinstance(target, WCS):
        twcs = target
    elif isinstance(target, (tuple, list)) and len(target) == 2:
        arr, w = target
        twcs = w if isinstance(w, WCS) else WCS(w, naxis=2)
        if shape_out is None:
            shape_out = np.asarray(arr).shape[:2]
    elif isinstance(target, pyfits.Header):
        twcs = WCS(target, naxis=2)
        if shape_out is None and 'NAXIS1' in target and 'NAXIS2' in target:
            shape_out = (int(target['NAXIS2']), int(target['NAXIS1']))
    else:
        return None, None
    if shape_out is None:
        pshape = getattr(twcs, 'pixel_shape', None)
        if pshape is None:
            return None, None
        shape_out = (int(pshape[1]), int(pshape[0]))
    return twcs, (int(shape_out[0]), int(shape_out[1]))


def load_sky_image(filepath: str, frame: str = 'ICRS', center: float = 180.,
                   flip_y: bool = True) -> tuple[np.ndarray, pyfits.Header]:
    """
    Load an RGB(A) raster image and create a matching equirectangular
    WCS header, ready for reprojection onto all-sky axes.

    Handles the common preprocessing pitfalls: uint8 → float
    normalization, axis orientation, RGBA vs RGB, and correct WCS
    header construction matching the image dimensions.

    Parameters
    ----------
    filepath : str
        Path to image file (.jpg, .png, .tif, etc. — anything
        ``matplotlib.pyplot.imread()`` can load).
    frame : str
        Coordinate frame for the WCS header: 'ICRS', 'Galactic', etc.
    center : float
        Center longitude of the equirectangular projection in degrees.
        Default ``180``. **Match this to how your panorama is centered:** a
        Milky-Way panorama centered on the galactic center needs
        ``center=0``, while the ``180`` default suits an RA-centered
        equatorial panorama (RA = 0 at the left/right image edge). Getting
        this wrong shifts the whole background by ``center`` degrees.
    flip_y : bool
        If True (default), flip the image vertically. Most raster
        images have y=0 at the top (latitude +90°), but FITS and
        imshow expect y=0 at the bottom (latitude -90°).

    Returns
    -------
    image : ndarray, shape (ny, nx, 3) or (ny, nx, 4)
        RGB(A) image array, float in [0, 1].
    header : astropy Header
        Equirectangular (CAR) FITS header matching the image dimensions.

    Notes
    -----
    The returned header uses Plate Carrée (CAR) projection with CDELT
    calibrated so the image spans exactly 360° × 180°. This is the
    standard assumption for panoramic all-sky background images.

    **JPEG vs PNG:** JPEG files are loaded as uint8 [0–255] and
    normalized to float [0, 1]. PNG files may be uint8 or already
    float [0, 1] depending on the file — both are handled correctly.
    RGBA PNG files are preserved as 4-channel.

    Examples
    --------
    >>> # Milky Way panorama, centered on the galactic center → center=0
    >>> img, hdr = sph.load_sky_image('milkyway_panorama.jpg',
    ...                               frame='Galactic', center=0)
    >>> ax = sph.make_wcs_frame(111, 'AIT', center=0, frame='Galactic')
    >>> reprojected = sph.reproject_background(img, hdr, ax)
    >>> ax.imshow(reprojected)

    >>> # RA-centered equatorial panorama uses the center=180 default
    >>> img, hdr = sph.load_sky_image('allsky_equatorial.png')
    """
    # Load image
    image = plt.imread(filepath)

    # Normalize to float [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.
    elif image.dtype in (np.float32, np.float64):
        # Already float — clip to [0, 1] just in case
        image = np.clip(image, 0., 1.)
    else:
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= image.max()

    # Flip y-axis (raster images have y=0 at top, FITS has y=0 at bottom)
    if flip_y:
        image = np.flip(image, axis=0)

    # Handle grayscale → RGB
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    ny, nx = image.shape[:2]

    # Build equirectangular (Plate Carrée) header
    ctt1, ctt2, rs = _resolve_ctype(frame)
    hdr = pyfits.Header({
        'NAXIS': 2, 'NAXIS1': nx, 'NAXIS2': ny,
        'CRPIX1': nx / 2. + 0.5, 'CRPIX2': ny / 2. + 0.5,
        'CRVAL1': center, 'CRVAL2': 0.,
        'CDELT1': -360. / nx, 'CDELT2': 180. / ny,
        'CUNIT1': 'deg', 'CUNIT2': 'deg',
        'CTYPE1': f'{ctt1}-CAR', 'CTYPE2': f'{ctt2}-CAR',
        'RADESYS': rs,
    })

    return image, hdr


def reproject_background(image: np.ndarray, source_header: pyfits.Header,
                         ax_or_header: Any,
                         order: str = 'bilinear',
                         downscale: float = 1.0) -> np.ndarray:
    """
    Reproject an RGB(A) background image to match a WCSAxes or header.

    Handles multi-channel images (RGB/RGBA), NaN masking, and dtype
    preservation. All channels are reprojected through a single shared
    coordinate map (much faster than one reproject per channel).

    Parameters
    ----------
    image : ndarray, shape (ny, nx, 3) or (ny, nx, 4)
        RGB(A) image array (float [0, 1]).
    source_header : astropy Header
        WCS header for the input image (e.g., from ``load_sky_image()``).
    ax_or_header : WCSAxes or Header
        Target: either a WCSAxes (uses its ``.wcs.to_header()`` and
        pixel shape) or a FITS header defining the output grid.
    order : str
        Interpolation order: 'nearest', 'bilinear' (default).
    downscale : float
        Draft-resolution lever (default ``1.0`` = full frame). ``>1`` reprojects
        onto a grid coarsened by this factor (reproject cost scales with output
        pixel count), then upsamples back to the full frame so
        ``ax.imshow(result)`` still lands correctly — e.g. ``downscale=2`` is
        ~4x fewer output pixels for quick draft renders.

    Returns
    -------
    reprojected : ndarray, shape (ny_out, nx_out, 3 or 4)
        Reprojected image, float [0, 1], NaN-free (NaN → 0).

    Notes
    -----
    Requires the ``reproject`` package. All channels are reprojected through a
    single shared coordinate map (via ``scipy.ndimage.map_coordinates``, which
    ships with reproject) rather than one reproject per channel.

    Examples
    --------
    >>> # A galactic-center Milky Way panorama needs center=0 (not the
    >>> # center=180 default, which suits RA-centered equatorial panoramas).
    >>> img, hdr = sph.load_sky_image('milkyway.jpg', frame='Galactic',
    ...                               center=0)
    >>> ax = sph.make_wcs_frame(111, 'AIT', center=0, frame='Galactic')
    >>> bg = sph.reproject_background(img, hdr, ax)
    >>> ax.imshow(bg)
    """
    if not _HAS_REPROJECT:
        raise ImportError(
            "reproject package required for reproject_background(). "
            "Install with: pip install reproject"
        )

    # Determine target header and shape
    if hasattr(ax_or_header, 'wcs'):
        # WCSAxes — extract header and pixel shape
        target_wcs = ax_or_header.wcs
        target_hdr = target_wcs.to_header()
        # Build the output for the FULL frame, not the current xlim/ylim span.
        # The header's CRPIX is full-frame; sizing to the span while the view
        # origin isn't 0 (e.g. a limb-framed SIN inset whose xlim/ylim don't
        # start near 0) mis-places / clips the raster under imshow. A full-frame
        # output aligns pixel 0 with data 0, so ``ax.imshow(result)`` (default
        # extent) lands correctly regardless of the view.
        pshape = getattr(target_wcs, 'pixel_shape', None)
        if pshape is not None:
            nx_out, ny_out = int(pshape[0]), int(pshape[1])
        else:
            # No declared frame size — cover pixel 0 to the view's far edge so
            # the output still aligns pixel 0 with data 0 under imshow.
            xlim = ax_or_header.get_xlim()
            ylim = ax_or_header.get_ylim()
            nx_out = max(1, int(round(max(xlim) + 0.5)))
            ny_out = max(1, int(round(max(ylim) + 0.5)))
        target_hdr['NAXIS1'] = nx_out
        target_hdr['NAXIS2'] = ny_out
        shape_out = (ny_out, nx_out)
    else:
        target_hdr = ax_or_header
        shape_out = (int(target_hdr['NAXIS2']), int(target_hdr['NAXIS1']))

    source_wcs = WCS(source_header, naxis=2)
    target_wcs = WCS(target_hdr, naxis=2)
    with np.errstate(invalid='ignore'):
        result = _reproject_shared(image, source_wcs, target_wcs, shape_out,
                                   order, downscale=downscale)
    # Replace NaN with 0 (outside projection boundary), match old dtype/range.
    result = np.nan_to_num(result, nan=0.).astype(np.float32)
    return np.clip(result, 0., 1.)


# Convenience functions for stretching, scaling, and clipping single-
# channel image arrays.  Wraps astropy.visualization stretches and
# matplotlib normalizations behind a simple string-based API.
# For multi-color RGB compositing (greyRGBize, colorize, combine),
# see the multicolorfits package.
#
# Core functions:
#   clip_percentile / clip_sigma / clip_zscale — interval detection
#   auto_interval — unified interval dispatcher
#   rescale_image — main workhorse: array → [0, 1] with stretch + clip
#   make_norm — matplotlib Normalize factory
#   adjust_gamma — NaN-safe gamma correction
#   list_stretches — print available stretch names
#
# Stretch names (case-insensitive):
#   linear, sqrt, squared, log, asinh, sinh, power, histeq
#
# Clip/interval methods:
#   percentile, sigma, zscale, minmax, manual


