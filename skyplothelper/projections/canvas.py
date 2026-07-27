"""Canvas-pixel sampling for crisp image rendering on WCSAxes.

The standard "build a regular lon/lat grid → ``pcolormesh`` with
``transform='world'``" pipeline mathematically reprojects each
lon/lat cell into the canvas. At high latitudes lon-cells become
thin slivers in projection space, producing visible banding /
fuzziness — even at very dense sampling.

The fix is the approach taken by ``healpy.mollview``: build the
sampling grid in **canvas-pixel space** (the axes' WCS pixel
grid), inverse-project each output pixel to ``(lon, lat)`` on the
sphere, look up the data value at that sphere position, and
display the resulting 2D array via ``ax.imshow`` (or
``ax.pcolormesh``) without any further projection. Every output
pixel maps to exactly one canvas pixel — no aliasing.

``project_to_canvas`` is the general primitive: pass any
``lookup_fn(lons, lats) → values`` and get back a 2D array
aligned to the axes. ``healpix_to_canvas`` is the HEALPix
specialization (handles frame conversion + nearest / bilinear
lookup). The same machinery works for any sky-data source —
gridded arrays (via interpolation), simulation outputs, analytic
functions — with a custom ``lookup_fn``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt


def project_to_canvas(
        ax: Any,
        lookup_fn: Callable[..., Any],
        *,
        output_shape: tuple[int, int] | None = None,
        extent: tuple[float, float, float, float] | None = None,
        blank_value: float = np.nan,
) -> tuple[npt.NDArray[np.float64], tuple[float, float, float, float]]:
    """Sample arbitrary sky data at every projection-pixel of a WCSAxes.

    Builds a regular grid in canvas (projection-pixel) space,
    inverse-projects each pixel center to ``(lon, lat)`` on the
    sphere, calls ``lookup_fn(lons, lats)`` for the data value,
    and returns a 2D array aligned to the canvas — ready for
    ``ax.imshow(arr, extent=extent, origin='lower')`` with no
    further ``transform=`` needed.

    Parameters
    ----------
    ax : WCSAxes
        Target axes. Must have a ``wcs`` attribute with
        ``pixel_to_world_values`` (any standard astropy WCSAxes).
    lookup_fn : callable
        ``lookup_fn(lons, lats) -> values``. Both inputs are 1D
        ndarrays of degree values in the **axes' own coordinate
        frame** (whatever ``ax.wcs.wcs.ctype`` declares — ICRS,
        galactic, ecliptic, supergalactic). Helpers like
        ``healpix_to_canvas`` handle frame conversion internally.
        Called once with every valid output-pixel coordinate.
    output_shape : (ny, nx) or None
        Output array shape. ``None`` (default) matches the visible
        canvas extent in WCS pixels (≈1 output pixel per WCS
        pixel). Pass a larger shape for dense saves; smaller for
        speed. Output pixels are square in **canvas** space, not
        in lon/lat — that is the entire point of this function.
    extent : (xmin, xmax, ymin, ymax) or None
        Canvas-pixel extent to sample. Defaults to
        ``ax.get_xlim() + ax.get_ylim()`` so the array covers
        exactly the currently-visible axes area. The same tuple
        should be passed to ``ax.imshow(..., extent=extent)``.
    blank_value : scalar
        Value for output pixels where the inverse projection is
        invalid (off-projection / NaN). Default ``np.nan`` — pair
        with ``cmap.set_bad('white')`` (or your preferred
        background) for clean compositing.

    Returns
    -------
    arr : ndarray, shape ``output_shape``
    extent : tuple
        ``(xmin, xmax, ymin, ymax)`` — pass to ``ax.imshow``.

    Notes
    -----
    The output is sampled once at the moment of the call. If the
    figure is later resized dramatically (much larger canvas) or
    the axes' xlim/ylim are zoomed in deeply, the array may
    appear pixelated because matplotlib upsamples a coarse array
    to fill the screen. Mitigations: pass a denser
    ``output_shape``, or re-call after the resize. Typical
    ``tight_layout`` / aspect-preserving resizes leave the result
    visually unchanged because the data-coord system is
    untouched and matplotlib's interpolation handles minor
    rescaling fine.

    Examples
    --------
    >>> # Custom analytic field on the sphere
    >>> def field(lons, lats):
    ...     return np.cos(np.radians(lons)) * np.sin(np.radians(lats))
    >>> arr, extent = sph.project_to_canvas(ax, field)
    >>> ax.imshow(arr, extent=extent, origin='lower', cmap='RdBu_r')

    >>> # HEALPix-specific (use the ``healpix_to_canvas`` shortcut)
    >>> arr, extent = sph.healpix_to_canvas(hpxmap, ax)
    >>> ax.imshow(arr, extent=extent, origin='lower', cmap='viridis')
    """
    if not hasattr(ax, 'wcs') or ax.wcs is None:
        raise ValueError(
            "project_to_canvas requires a WCSAxes with a .wcs "
            "attribute (got a plain matplotlib axes).")

    if extent is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        extent = (float(xlim[0]), float(xlim[1]),
                  float(ylim[0]), float(ylim[1]))
    xmin, xmax, ymin, ymax = extent

    if output_shape is None:
        # Default: ~2 output pixels per visible WCS pixel, with a
        # floor of 2000 in the wider dim so even very-coarse WCS
        # grids (e.g. the default ``make_wcs_frame`` AIT grid of
        # 360 px wide) produce sub-pixel-density output at typical
        # figsize/dpi combinations. The 2000 floor is chosen so the
        # polar regions of an all-sky view look smooth (where the
        # 4 polar HEALPix tiles' wedge boundaries are the most
        # demanding feature to resolve cleanly).
        # Pass an explicit ``output_shape`` for tighter density
        # control (denser for high-DPI saves, sparser for speed).
        nx_natural = abs(xmax - xmin)
        ny_natural = abs(ymax - ymin)
        target_wide = max(2000, int(round(2 * max(nx_natural, ny_natural))))
        if nx_natural >= ny_natural:
            nx = target_wide
            ny = max(int(round(target_wide * ny_natural / max(nx_natural, 1))), 100)
        else:
            ny = target_wide
            nx = max(int(round(target_wide * nx_natural / max(ny_natural, 1))), 100)
        output_shape = (ny, nx)
    ny, nx = output_shape

    # Sample at output-pixel **centers**, not edges — matches imshow's
    # extent convention (the array's [0, 0] cell occupies pixel-coord
    # rectangle [xmin, xmin+dx] × [ymin, ymin+dy], and its center is
    # at (xmin + dx/2, ymin + dy/2)).
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    x = xmin + (np.arange(nx) + 0.5) * dx
    y = ymin + (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y)

    # Inverse-project: canvas pixels → world (lon, lat).
    lons, lats = ax.wcs.pixel_to_world_values(X, Y)
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    # Off-projection pixels (e.g. outside the AIT ellipse, or
    # behind the SIN globe limb) come back as NaN.
    valid = np.isfinite(lons) & np.isfinite(lats)

    arr = np.full(output_shape, blank_value, dtype=float)
    if valid.any():
        # Normalize lons to [0, 360) so lookup_fn sees a consistent
        # range — astropy WCS can return negative lons depending on
        # the projection / center convention.
        valid_lons = lons[valid] % 360.0
        valid_lats = lats[valid]
        arr[valid] = lookup_fn(valid_lons, valid_lats)

    return arr, extent


def healpix_to_canvas(
        healpix_map: np.ndarray,
        ax: Any,
        *,
        frame: str = 'icrs',
        nest: bool = False,
        interp: bool = False,
        output_shape: tuple[int, int] | None = None,
        extent: tuple[float, float, float, float] | None = None,
        blank_value: float = np.nan,
) -> tuple[npt.NDArray[np.float64], tuple[float, float, float, float]]:
    """Sample a HEALPix map at every canvas-pixel of a WCSAxes.

    Convenience wrapper around ``project_to_canvas`` for HEALPix
    inputs. Handles frame conversion automatically when the map
    and the axes use different sky frames (e.g. galactic-frame
    HEALPix on an ICRS AIT axes).

    Parameters
    ----------
    healpix_map : ndarray, shape ``(npix,)``
        Full HEALPix array.
    ax : WCSAxes
    frame : {'icrs', 'galactic', 'ecliptic', 'supergalactic'}
        Coordinate frame the HEALPix array is indexed in. Default
        ``'icrs'``. If different from the axes' own frame, every
        canvas pixel is transformed before the HEALPix lookup.
    nest : bool
        If True, the HEALPix array uses NESTED ordering; otherwise
        RING (the HEALPix default).
    interp : bool
        If False (default), use nearest-pixel lookup — exact tile
        boundaries, fastest. If True, use bilinear interpolation
        via ``hp.get_interp_val`` — smoother edges, ~4× slower.
    output_shape, extent, blank_value
        See ``project_to_canvas``.

    Returns
    -------
    arr, extent : as ``project_to_canvas``

    Examples
    --------
    >>> arr, ext = sph.healpix_to_canvas(hpxmap, ax)
    >>> ax.imshow(arr, extent=ext, origin='lower',
    ...           cmap='viridis', vmin=-1, vmax=1)
    """
    try:
        import healpy as hp
    except ImportError:
        raise ImportError(
            "healpix_to_canvas requires the 'healpy' package "
            "(pip install healpy).")

    nside = hp.npix2nside(len(healpix_map))

    # Resolve frame conversion, if any.
    from ..wcs_frame import _get_wcs_frame_name
    axes_frame = _get_wcs_frame_name(ax).lower()
    target_frame = frame.lower()

    if axes_frame == target_frame:
        if interp:
            def lookup(lons: npt.ArrayLike, lats: npt.ArrayLike) -> Any:
                return hp.get_interp_val(
                    healpix_map, lons, lats, lonlat=True, nest=nest)
        else:
            def lookup(lons: npt.ArrayLike, lats: npt.ArrayLike) -> Any:
                return healpix_map[hp.ang2pix(
                    nside, lons, lats, lonlat=True, nest=nest)]
    else:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        # astropy frame names differ slightly from ours.
        astropy_name = {
            'icrs': 'icrs',
            'galactic': 'galactic',
            'ecliptic': 'geocentrictrueecliptic',
            'supergalactic': 'supergalactic',
        }
        src = astropy_name.get(axes_frame, 'icrs')
        tgt = astropy_name.get(target_frame, 'icrs')

        def _to_target(
                lons: npt.ArrayLike,
                lats: npt.ArrayLike) -> tuple[Any, Any]:
            sc = SkyCoord(lons * u.deg, lats * u.deg, frame=src)
            tr = sc.transform_to(tgt)
            if target_frame == 'galactic':
                return tr.l.deg, tr.b.deg
            if target_frame == 'ecliptic':
                return tr.lon.deg, tr.lat.deg
            if target_frame == 'supergalactic':
                return tr.sgl.deg, tr.sgb.deg
            return tr.ra.deg, tr.dec.deg

        if interp:
            def lookup(lons: npt.ArrayLike, lats: npt.ArrayLike) -> Any:
                tlon, tlat = _to_target(lons, lats)
                return hp.get_interp_val(
                    healpix_map, tlon, tlat, lonlat=True, nest=nest)
        else:
            def lookup(lons: npt.ArrayLike, lats: npt.ArrayLike) -> Any:
                tlon, tlat = _to_target(lons, lats)
                return healpix_map[hp.ang2pix(
                    nside, tlon, tlat, lonlat=True, nest=nest)]

    return project_to_canvas(ax, lookup, output_shape=output_shape,
                              extent=extent, blank_value=blank_value)
