"""WCS frame construction.

``make_wcs_frame`` is the main entry point: it builds an astropy WCSAxes
for any of the registry's projections (FITS or non-FITS), wires up the
correct frame class and projection center, and returns the axes ready for
plotting. Also includes header generators for the four common dummy-WCS
shapes, the ``apply_boundary_labels`` boundary-label drawer, the
``clip_to_frame`` artist clipper, offset-WCS helpers, and the legacy
``make_WCS_*_subplot_frame`` aliases.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import astropy.io.fits as pyfits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales, skycoord_to_pixel
from matplotlib import rcParams

try:
    from astropy.visualization.wcsaxes.frame import RectangularFrame
except ImportError:
    RectangularFrame = None  # fallback handled in make_wcs_frame

# --- skyplothelper internals ---
from ._stroke import _stroke_path_effects
from ._text_layout import _resolve_text_anchor
from .plotting import _attach_sky_methods
from .projections._math import (
    _eckert4_forward,
    _kavrayskiy_forward,
    _mcbryde_forward,
    _robinson_forward,
    _winkel_forward,
)
from .projections.frames import (
    _HAS_CURVEDTRANSFORM,
    Eckert4Transform,
    KavrayskiyTransform,
    McBrydeTransform,
    ObliqueAspectTransform,
    RobinsonTransform,
    WinkelTripelTransform,
)
from .projections.project import resolve_lon_units
from .projections.registry import _FRAME_CLASSES, _resolve_projection
from .ticks import format_ticklabels

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

# ===== clip_to_frame, apply_boundary_labels =====

def clip_to_frame(ax: Any) -> None:
    """
    Clip all data artists on a WCSAxes to the frame boundary.

    For non-FITS projections (Robinson, Kavrayskiy, Eckert IV, Winkel Tripel,
    McBryde-Thomas), scatter points and other data artists can project to
    valid positions outside the frame boundary. Call this after plotting to
    clip them to the frame shape.

    For FITS-based projections with custom frames (SFL, PAR), this also
    works and can catch edge cases. For EllipticalFrame projections
    (AIT, MOL), clipping is usually handled automatically by astropy.

    Parameters
    ----------
    ax : WCSAxes

    Notes
    -----
    Call ``fig.canvas.draw()`` before ``clip_to_frame()`` to ensure the
    frame boundary patch is initialized. Avoid calling ``tight_layout()``
    after clipping (it's generally incompatible with WCSAxes).

    **Key insight:** On WCSAxes, ``ax.patch`` IS the frame boundary
    PathPatch, and its transform uses a simple BboxTransform that maps
    data coords → display coords WITHOUT going through the
    CurvedTransform. This is why ``ax.patch`` works as a clip path
    while ``ax.transData`` (which includes the projection transform)
    does not.

    Examples
    --------
    >>> ax = sph.make_wcs_frame(111, 'robinson', center=180)
    >>> ax.scatter(lon, lat, transform=ax.get_transform('world'))
    >>> fig.canvas.draw()
    >>> sph.clip_to_frame(ax)
    """
    # ax.patch on WCSAxes IS the frame boundary PathPatch, with a
    # transform that maps data coords → display coords via a simple
    # BboxTransform (not through the CurvedTransform). This makes it
    # the correct clip source for all artists.
    patch = ax.patch

    for artist in ax.get_children():
        if artist is patch:
            continue
        if hasattr(artist, 'set_clip_path') and not isinstance(artist, plt.Text):
            try:
                artist.set_clip_path(patch)
            except Exception:
                pass


def apply_boundary_labels(ax: Any, coord_index: int = 1,
                         lat_values: Sequence[float] | None = None,
                         side: str = 'both', orient: str = 'perpendicular',
                         fontsize: float | None = None,
                         color: str | None = None, pad: float = 4,
                         stroke_lw: float | None = None,
                         stroke_color: str | None = None,
                         fmt_func: Callable[[float], str] | None = None,
                         **text_kwargs: Any) -> list[Any]:
    """
    Draw tick labels at the frame boundary with configurable orientation.

    For latitude labels on curved all-sky frames (sinusoidal, parabolic,
    elliptical, Robinson, etc.), this places labels at the intersection of
    each latitude grid line with the frame boundary, with three orientation
    options.

    Call this AFTER ``format_ticklabels()`` — it suppresses astropy's
    auto tick labels for the specified coordinate and draws its own.

    Parameters
    ----------
    ax : WCSAxes
        Must have a WCS-based or CurvedTransform-based projection.
    coord_index : int
        Which coordinate's labels to make tangent-aligned.
        1 = latitude/Dec (default), 0 = longitude/RA.
    lat_values : list of float, optional
        Tick values to label (degrees). If None, uses [-60, -30, 0, 30, 60].
    side : str
        'left', 'right', or 'both' — which boundary edge(s) to label.
    orient : str
        Label orientation style, named relative to the frame boundary:

        - ``'perpendicular'`` (default) : Label text is rotated to follow
          the direction the grid line would extend outward *across* the frame
          boundary.  Good for visually connecting each label to its grid line.
        - ``'parallel'`` : Label text runs *along* the frame boundary curve
          (rotated to the local boundary tangent).  Gives a compact look
          that hugs the frame edge.
        - ``'horizontal'`` : No rotation — all labels are horizontal.
          Clean and simple, good for publication figures where rotation
          might look cluttered.

        (These boundary-relative names replace the older ``'radial'`` /
        ``'extension'`` and ``'tangent'`` — the latter clashed with
        ``make_wcs_frame``'s ``tick_rotation='tangent'``, which means
        something different: aligning to a *grid line's* tangent.)
    fontsize : float, optional
        Defaults to current ticklabel size.
    color : str
    pad : float
        Padding from the frame boundary in points.
    stroke_lw : float, optional
    stroke_color : str, optional
    fmt_func : callable, optional
        Custom label formatter. Takes a float (degrees) and returns a string.
        Default: ``'{:+.0f}°'.format(val)``
    **text_kwargs
        Additional kwargs passed to ax.text().

    Returns
    -------
    labels : list
        The text artists created.

    Examples
    --------
    >>> format_ticklabels(ax, style='publication')
    >>> apply_boundary_labels(ax)  # perpendicular (grid-line extension) labels

    >>> apply_boundary_labels(ax, orient='parallel')  # hug the frame curve
    >>> apply_boundary_labels(ax, orient='horizontal')  # flat labels

    >>> # Custom values and formatting
    >>> apply_boundary_labels(ax, lat_values=[-45, -15, 15, 45],
    ...     fmt_func=lambda v: f'{v:+.0f}°', fontsize=8)

    Notes
    -----
    **Root cause of ax.annotate() not working:** ``ax.annotate()`` gets
    clipped by custom frame boundary paths even with ``clip_on=False``.
    This function uses ``ax.text()`` with display-space offset conversion
    instead.

    **Frame boundary awareness:** Includes ``_inside_frame()`` filtering
    using the analytical boundary function — critical for SFL/PAR
    projections where WCS pixel coords extend beyond the custom frame
    boundary (CDELT calibrated for Aitoff scale).

    **Alternative to astropy's tick labels:** This function sidesteps
    astropy's ``exclude_overlapping`` behavior entirely, making it a
    good workaround when astropy drops labels at crowded positions
    (e.g., the 300° Galactic longitude on Aitoff at small panel sizes).
    """
    if lat_values is None:
        lat_values = [-60, -30, 0, 30, 60]

    if fontsize is None:
        fontsize = rcParams.get('xtick.labelsize', 10)
    if color is None:
        color = rcParams['xtick.color']

    if fmt_func is None:
        def fmt_func(v: float) -> str:
            return f'{v:+.0f}\u00B0' if v != 0 else '0\u00B0'

    orient = orient.lower()
    if orient not in ('perpendicular', 'parallel', 'horizontal'):
        raise ValueError(
            f"orient must be 'perpendicular', 'parallel', or 'horizontal', "
            f"got '{orient}'")

    # Suppress astropy's auto tick labels for this coord entirely —
    # our manually placed tangent labels replace them
    ax.coords[coord_index].set_ticklabel_visible(False)

    # Get pixel extent center (used for outward normal direction)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xcen = 0.5 * (xmin + xmax)
    ycen = 0.5 * (ymin + ymax)
    half_w = 0.5 * (xmax - xmin)
    half_h = 0.5 * (ymax - ymin)

    # Determine frame boundary function for inside/outside testing.
    # Custom frames have _boundary_x(t) giving normalized half-width at
    # normalized height t ∈ [-1, 1].  EllipticalFrame uses sqrt(1-t²).
    frame_obj = ax.coords.frame
    if hasattr(frame_obj, '_boundary_x'):
        _bnd_func = frame_obj._boundary_x
    else:
        def _bnd_func(t: Any) -> Any:
            return np.sqrt(np.maximum(1 - t ** 2, 0))

    def _inside_frame(x_arr: Any, y_arr: Any) -> Any:
        """Boolean mask: True where (x, y) pixel coords are inside frame."""
        t = (y_arr - ycen) / half_h if half_h > 0 else np.zeros_like(y_arr)
        t = np.clip(t, -1, 1)
        bx = _bnd_func(t)  # normalized half-width at each y
        x_bnd = half_w * bx  # pixel half-width at each y
        return np.abs(x_arr - xcen) <= x_bnd + 0.5  # small tolerance

    # Map lat values to normalized t (pixel y position)
    # For WCS-based axes, convert lat → pixel y
    wcs = ax.wcs if hasattr(ax, 'wcs') and ax.wcs is not None else None

    labels = []
    pe = _stroke_path_effects(stroke_color, stroke_lw)

    # Get center longitude for the projection
    if wcs is not None:
        center_lon = wcs.wcs.crval[0]
    else:
        center_lon = getattr(ax, '_sph_center_lon', 0)

    # --- Find actual grid-boundary intersections ---
    # Sweep longitudes at each latitude and project to pixel coords to find
    # where the constant-Dec line reaches the frame edges. The longitude sweep
    # is identical for every latitude, so project ALL latitudes in one
    # world→pixel transform (instead of one per latitude) — same result.
    n_sweep = 500
    lon_sweep = np.linspace(center_lon - 179.9, center_lon + 179.9, n_sweep)
    lats_arr = np.asarray(list(lat_values), dtype=float)
    world_all = np.column_stack([np.tile(lon_sweep, len(lats_arr)),
                                 np.repeat(lats_arr, n_sweep)])
    if wcs is not None:
        pix_all = wcs.wcs_world2pix(world_all, 0)
    else:
        fwd = ax.coords._transform.inverted()  # world→projected
        pix_all = fwd.transform(world_all)
    pix_all = pix_all.reshape(len(lats_arr), n_sweep, 2)

    for li, lat_val in enumerate(lat_values):
        x_pix_arr = pix_all[li, :, 0]
        y_pix_arr = pix_all[li, :, 1]

        # Filter out NaN/inf from projection
        valid = np.isfinite(x_pix_arr) & np.isfinite(y_pix_arr)
        if valid.sum() < 2:
            continue
        x_fin = x_pix_arr[valid]
        y_fin = y_pix_arr[valid]
        lon_fin = lon_sweep[valid]  # keep matching longitudes

        # Further filter to points inside the frame boundary.
        # This is critical for projections (SFL, PAR) where WCS pixel
        # coords extend beyond the custom frame boundary.
        inside = _inside_frame(x_fin, y_fin)
        if inside.sum() < 2:
            continue
        x_valid = x_fin[inside]
        y_valid = y_fin[inside]
        lon_valid = lon_fin[inside]

        # Right boundary: maximum x position (among inside-frame points)
        idx_right = np.argmax(x_valid)
        # Left boundary: minimum x position
        idx_left = np.argmin(x_valid)

        def _refine_boundary_y(idx: Any, x_arr: Any, y_arr: Any,
                               lon_arr: Any, x_sign: float) -> Any:
            """Refine the grid-boundary intersection with a fine re-sweep.

            Does a tight re-sweep around the boundary longitude to get a
            more accurate y-position at the frame edge, important when
            the grid line approaches the boundary tangentially.
            """
            lon_bnd = lon_arr[idx]
            # Fine sweep: ±2° around the boundary longitude
            dlon = 2.0
            lon_fine = np.linspace(lon_bnd - dlon, lon_bnd + dlon, 200)
            lat_fine = np.full(200, lat_val)
            if wcs is not None:
                pix_fine = wcs.wcs_world2pix(
                    np.column_stack([lon_fine, lat_fine]), 0)
            else:
                fwd = ax.coords._transform.inverted()
                pix_fine = fwd.transform(
                    np.column_stack([lon_fine, lat_fine]))
            xf, yf = pix_fine[:, 0], pix_fine[:, 1]
            ok = np.isfinite(xf) & np.isfinite(yf) & _inside_frame(xf, yf)
            if ok.sum() < 2:
                return y_arr[idx]
            if x_sign > 0:
                return yf[ok][np.argmax(xf[ok])]
            else:
                return yf[ok][np.argmin(xf[ok])]

        def _tangent_from_sweep(idx: Any, x_arr: Any, y_arr: Any) -> float:
            """Grid-line tangent angle in display coords near boundary."""
            n = len(x_arr)
            hw = max(2, n // 50)
            i_lo = max(0, idx - hw)
            i_hi = min(n - 1, idx + hw)
            d1 = ax.transData.transform((x_arr[i_lo], y_arr[i_lo]))
            d2 = ax.transData.transform((x_arr[i_hi], y_arr[i_hi]))
            return np.degrees(np.arctan2(d2[1] - d1[1], d2[0] - d1[0]))

        def _boundary_tangent_angle(y_pix: float, x_sign: float) -> float:
            """Frame boundary tangent angle in display coords at pixel y."""
            t_loc = np.clip((y_pix - ycen) / half_h, -0.999, 0.999) if half_h > 0 else 0
            dt = 0.005
            t_p = np.clip(t_loc + dt, -1, 1)
            t_m = np.clip(t_loc - dt, -1, 1)
            bx_p = _bnd_func(np.array([t_p]))[0]
            bx_m = _bnd_func(np.array([t_m]))[0]
            x1 = xcen + x_sign * half_w * bx_m
            y1 = ycen + half_h * t_m
            x2 = xcen + x_sign * half_w * bx_p
            y2 = ycen + half_h * t_p
            d1 = ax.transData.transform((x1, y1))
            d2 = ax.transData.transform((x2, y2))
            return np.degrees(np.arctan2(d2[1] - d1[1], d2[0] - d1[0]))

        def _readable_angle(a: float) -> float:
            a = a % 360
            if a > 90 and a <= 270:
                a -= 180
            elif a > 270:
                a -= 360
            return a

        label_str = fmt_func(lat_val)

        sides_to_do = []
        if side in ('right', 'both'):
            grid_ang = _tangent_from_sweep(idx_right, x_valid, y_valid)
            y_r = _refine_boundary_y(idx_right, x_valid, y_valid,
                                      lon_valid, +1)
            t_r = np.clip((y_r - ycen) / half_h, -1, 1) if half_h > 0 else 0
            x_r = xcen + half_w * _bnd_func(np.array([t_r]))[0]
            bnd_ang = _boundary_tangent_angle(y_r, +1)
            sides_to_do.append(('right', x_r, y_r, grid_ang, bnd_ang))
        if side in ('left', 'both'):
            grid_ang = _tangent_from_sweep(idx_left, x_valid, y_valid)
            y_l = _refine_boundary_y(idx_left, x_valid, y_valid,
                                      lon_valid, -1)
            t_l = np.clip((y_l - ycen) / half_h, -1, 1) if half_h > 0 else 0
            x_l = xcen - half_w * _bnd_func(np.array([t_l]))[0]
            bnd_ang = _boundary_tangent_angle(y_l, -1)
            sides_to_do.append(('left', x_l, y_l, grid_ang, bnd_ang))

        for side_name, x_bnd, y_bnd, grid_angle, bnd_angle in sides_to_do:
            # Choose label rotation based on orient mode
            if orient == 'perpendicular':
                rot = _readable_angle(grid_angle)
            elif orient == 'parallel':
                rot = _readable_angle(bnd_angle)
            else:  # horizontal
                rot = 0

            # Offset perpendicular to the frame boundary, pointing outward.
            # Always use the boundary tangent for offset direction so padding
            # is uniform regardless of label orient mode.
            bnd_rad = np.radians(bnd_angle)
            disp_bnd = ax.transData.transform((x_bnd, y_bnd))
            disp_cen = ax.transData.transform((xcen, ycen))
            # Two candidate normals: ±90° from boundary tangent; keep the one
            # pointing AWAY from the frame center — the outward normal.
            for sign_n in (+1, -1):
                nx = -sign_n * np.sin(bnd_rad)
                ny = sign_n * np.cos(bnd_rad)
                dx_to_cen = disp_cen[0] - disp_bnd[0]
                dy_to_cen = disp_cen[1] - disp_bnd[1]
                if nx * dx_to_cen + ny * dy_to_cen < 0:
                    break

            pad_disp = pad * ax.figure.dpi / 72.
            disp_off = (disp_bnd[0] + nx * pad_disp,
                        disp_bnd[1] + ny * pad_disp)
            data_off = ax.transData.inverted().transform(disp_off)

            # Derive (ha, va) from the outward normal so the label's NEAR
            # EDGE — not its bbox center — lands at the offset point. This
            # centers each label on its gridline and justifies it away from
            # the frame uniformly across orient modes, fixing the old fixed
            # ha + va='center' which shifted rotated labels tangentially off
            # their tick (most visible for orient='parallel', near-vertical
            # text on the L/R edges). Shared with the overlay-tick label path
            # (coord_overlay.render_labels) via _resolve_text_anchor.
            ha, va = _resolve_text_anchor(rot, +1, nx, ny)

            txt = ax.text(
                data_off[0], data_off[1], label_str,
                fontsize=fontsize, color=color,
                ha=ha, va=va,
                rotation=rot,
                rotation_mode='anchor',
                clip_on=False,
                **text_kwargs
            )
            if pe is not None:
                txt.set_path_effects(pe)
            labels.append(txt)

    return labels




# ===== Frame-resolution helper =====

def _resolve_ctype(radesys: str) -> tuple[str, str, str]:
    """Determine CTYPE prefix and RADESYS from input frame string."""
    rs = radesys.lower()
    if 'super' in rs:
        return 'SLON', 'SLAT', 'ICRS'
    elif 'gal' in rs:
        return 'GLON', 'GLAT', 'ICRS'
    elif 'hel' in rs:
        return 'HLON', 'HLAT', 'ICRS'
    elif 'ecl' in rs:
        return 'ELON', 'ELAT', 'ICRS'
    elif rs == 'itrs':
        return 'TLON', 'TLAT', 'ICRS'
    else:
        return 'RA--', 'DEC-', radesys.upper()


# ===== Header generators + make_wcs_frame + offset helpers =====

# Conic (COD/COE/COO/COP) and pseudoconic (BON) projections are undefined
# without a standard parallel: wcslib rejects the header at ``wcsset`` time
# with "ERROR 5 (invalid parameter)" if PV2_1 is absent. 45° is a sensible
# mid-latitude default that gives a well-behaved frame for all five.
_PV_REQUIRED_FITS_CODES = frozenset({'COD', 'COE', 'COO', 'COP', 'BON'})
_DEFAULT_PV2_1 = 45.0

# The four true conics additionally put the reference point *on* the standard
# parallel (CRVAL2 = PV2_1), which places the cone apex at the pole so the
# wedge frames cleanly. Bonne is excluded: it keeps its own reference latitude.
_CONIC_FITS_CODES = frozenset({'COD', 'COE', 'COO', 'COP'})


def _apply_pv_cards(hdr: pyfits.Header, fits_code: str,
                    pv2_1: float | None, pv2_2: float | None,
                    set_conic_crval2: bool = False) -> None:
    """Inject the PV cards a conic / Bonne header needs, in place.

    Shared by :func:`dummy_allsky_hdr` and :func:`make_wcs_frame` so the
    matplotlib frames and the backend-agnostic :func:`skyplothelper.project`
    primitive agree on the standard parallel. Cards are NOT injected for any
    other projection — ``pv2_1`` / ``pv2_2`` passed to AIT/MOL/SIN/... are
    silently ignored, matching the documented contract.
    """
    if fits_code.upper() not in _PV_REQUIRED_FITS_CODES:
        return
    hdr['PV2_1'] = _DEFAULT_PV2_1 if pv2_1 is None else float(pv2_1)
    if pv2_2 is not None:
        hdr['PV2_2'] = float(pv2_2)
    if set_conic_crval2 and fits_code.upper() in _CONIC_FITS_CODES:
        hdr['CRVAL2'] = hdr['PV2_1']


def dummy_allsky_hdr(center_LONdeg: float = 180, radesys: str = 'ICRS',
                     projection: str = 'AIT',
                     npix: tuple[int, int] = (360, 180),
                     pv2_1: float | None = None,
                     pv2_2: float | None = None) -> pyfits.Header:
    """
    Create a dummy all-sky FITS header for the specified projection.

    Parameters
    ----------
    center_LONdeg : float
    radesys : str
    projection : str
        FITS projection code: 'AIT', 'MOL', 'HPX', 'SFL', 'PAR', 'CAR', etc.
    npix : tuple
        (NAXIS1, NAXIS2)
    pv2_1, pv2_2 : float, optional
        FITS PV2_1 / PV2_2 parameters for conic (COD/COE/COO/COP) and Bonne
        (BON) projections — see :func:`make_wcs_frame`. Those five require
        PV2_1 and default to 45°; every other projection ignores both.

    Returns
    -------
    hdr : astropy.io.fits.Header
    """
    ctt1, ctt2, rs = _resolve_ctype(radesys)
    hdr = pyfits.Header({
        'NAXIS': 2, 'NAXIS1': npix[0], 'NAXIS2': npix[1],
        'CRPIX1': npix[0] / 2 + 0.5, 'CRPIX2': npix[1] / 2 + 0.5,
        'CRVAL1': center_LONdeg, 'CRVAL2': 0.,
        'CDELT1': -2 * np.sqrt(2) / np.pi * 360 / npix[0],
        'CDELT2':  2 * np.sqrt(2) / np.pi * 180 / npix[1],
        'CUNIT1': 'deg', 'CUNIT2': 'deg',
        'CTYPE1': f'{ctt1}-{projection}', 'CTYPE2': f'{ctt2}-{projection}',
        'RADESYS': rs,
    })
    _apply_pv_cards(hdr, projection, pv2_1, pv2_2, set_conic_crval2=True)
    return hdr


def dummy_ortho_hdr(center_LONdeg: float = 0, center_LATdeg: float = 0.,
                    radesys: str = 'ICRS', projection: str = 'SIN',
                    lonpole: float = 0., latpole: float = 0.,
                    Naxispix: int = 180) -> pyfits.Header:
    """Create a dummy orthographic (globe) FITS header."""
    ctt1, ctt2, rs = _resolve_ctype(radesys)
    lonsign = 1 if ctt1 != 'RA--' else -1
    hdr = pyfits.Header({
        'NAXIS': 2, 'NAXIS1': Naxispix, 'NAXIS2': Naxispix,
        'CRPIX1': Naxispix / 2 + 0.5, 'CRPIX2': Naxispix / 2 + 0.5,
        'CRVAL1': center_LONdeg, 'CRVAL2': center_LATdeg,
        'CDELT1': 2 / np.pi * lonsign, 'CDELT2': 2 / np.pi,
        'CUNIT1': 'deg', 'CUNIT2': 'deg',
        'CTYPE1': f'{ctt1}-{projection}', 'CTYPE2': f'{ctt2}-{projection}',
        'RADESYS': rs,
        'LONPOLE': lonpole, 'LATPOLE': latpole,
    })
    return hdr


def dummy_offset_hdr(centercoords_deg: Any = (0., 0.),
                     offset_units: str = 'deg',
                     naxis_xy: tuple[int, int] = (100, 100),
                     cdelts: tuple[float, float] = (1., 1.),
                     refpixel_xy: Any = 'center', radesys: str = 'ICRS',
                     lonpole: float = 0.,
                     latpole: float = 0.) -> pyfits.Header:
    """Create a dummy FITS header for offset/relative coordinates."""
    if isinstance(centercoords_deg, SkyCoord):
        ctt1, ctt2, rs = _resolve_ctype(radesys)
        if 'gal' in radesys.lower():
            centercoords_deg = [centercoords_deg.galactic.l.deg,
                                centercoords_deg.galactic.b.deg]
        else:
            centercoords_deg = [centercoords_deg.icrs.ra.deg,
                                centercoords_deg.icrs.dec.deg]

    if refpixel_xy == 'origin':
        refpixel_xy = [0., 0.]
    elif refpixel_xy == 'center':
        refpixel_xy = [naxis_xy[0] * 0.5, naxis_xy[1] * 0.5]

    hdr = pyfits.Header({
        'NAXIS': 2, 'WCSAXES': 2,
        'NAXIS1': naxis_xy[0], 'NAXIS2': naxis_xy[1],
        'CRPIX1': refpixel_xy[0], 'CRPIX2': refpixel_xy[1],
        'CRVAL1': centercoords_deg[0], 'CRVAL2': centercoords_deg[1],
        'CDELT1': cdelts[0], 'CDELT2': cdelts[1],
        'CUNIT1': offset_units, 'CUNIT2': offset_units,
        'CTYPE1': 'XOFFSET', 'CTYPE2': 'YOFFSET',
        'RADESYS': radesys,
        'LONPOLE': lonpole, 'LATPOLE': latpole,
    })
    return hdr


def dummy_standard_hdr(centercoords_deg: Any = (0., 0.), cunit: str = 'deg',
                       naxis_xy: tuple[int, int] = (100, 100),
                       cdelts: tuple[float, float] = (1. / 3600, 1. / 3600),
                       refpixel_xy: Any = 'center', radesys: str = 'ICRS',
                       equinox: float = 2000.0, projection: str = 'TAN',
                       lonpole: float = 0.,
                       latpole: float = 0.) -> pyfits.Header:
    """Create a dummy standard celestial FITS header (e.g., TAN projection)."""
    ctt1, ctt2, rs = _resolve_ctype(radesys)
    lonsign = -1 if ctt1 == 'RA--' else 1

    if isinstance(centercoords_deg, SkyCoord):
        if 'gal' in radesys.lower():
            centercoords_deg = [centercoords_deg.galactic.l.deg,
                                centercoords_deg.galactic.b.deg]
        else:
            centercoords_deg = [centercoords_deg.icrs.ra.deg,
                                centercoords_deg.icrs.dec.deg]

    if refpixel_xy == 'origin':
        refpixel_xy = [0., 0.]
    elif refpixel_xy == 'center':
        refpixel_xy = [naxis_xy[0] * 0.5, naxis_xy[1] * 0.5]

    hdr = pyfits.Header({
        'NAXIS': 2, 'WCSAXES': 2,
        'NAXIS1': naxis_xy[0], 'NAXIS2': naxis_xy[1],
        'CRPIX1': refpixel_xy[0] + 0.5, 'CRPIX2': refpixel_xy[1] + 0.5,
        'CRVAL1': centercoords_deg[0], 'CRVAL2': centercoords_deg[1],
        'CDELT1': cdelts[0] * lonsign, 'CDELT2': cdelts[1],
        'CUNIT1': cunit, 'CUNIT2': cunit,
        'CTYPE1': f'{ctt1}-{projection}', 'CTYPE2': f'{ctt2}-{projection}',
        'RADESYS': rs, 'EQUINOX': equinox,
        'LONPOLE': lonpole, 'LATPOLE': latpole,
    })
    return hdr


# Projections whose top edge sits right at axes-fraction y≈1.0, causing
# the topmost tick labels (at the pole) to collide with the default
# title position. Used by `_pad_title_for_pole_top_projection()`.
_POLE_TOP_FITS_PROJECTIONS = frozenset({'SFL', 'PAR', 'CAR', 'SIN'})
_POLE_TOP_NON_FITS_FRAME_SHAPES = frozenset({
    'sinusoidal', 'parabolic', 'rectangular',
    'eckert4', 'kavrayskiy', 'mcbryde', 'winkel_tripel',
})

# Frame shapes that get in-frame tick labels by default (``tick_style='auto'``
# on :func:`make_wcs_frame`). Selected because their astropy-default
# boundary labels are either visibly buggy (doubled boundary +
# central-curve fallback, spurious-tick walks on closed spines) or
# horizontal-and-disconnected from the curves they annotate.
# ``elliptical`` (AIT/MOL) and ``rectangular`` are intentionally omitted
# — astropy's defaults render cleanly there. Users can opt in for those
# (or out for anything in this set) via the ``tick_style`` kwarg.
_IN_FRAME_TICK_AUTO_FRAME_SHAPES = frozenset({
    'circular',
    'sinusoidal',
    'parabolic',
    'robinson',
    'kavrayskiy',
    'eckert4',  # NB: the eckert_iv projection's frame_shape key is 'eckert4'
    'winkel_tripel',
    'mcbryde',
})

# Projections whose astropy native rendering produces no visible tick
# labels — independent of frame_shape. These get auto-routed to
# 'in_frame' as well so the labels actually show up. Their natural
# visible region (PCO egg, HPX stepped diamond, BON cardioid, the
# quadcube cross) sits inside the rectangular frame, so astropy's default
# boundary ticks land in the invalid region / on the canvas rectangle and
# read poorly; in-frame central-crosshair labels are the clean default.
# The cube cross is also poorly served by boundary ticks (its edges run
# parallel to the gridlines, so few clean intersections), so it gets
# in-frame too rather than 'boundary'.
_IN_FRAME_TICK_AUTO_FITS_CODES = frozenset({
    'PCO', 'HPX', 'XPH', 'BON', 'TSC', 'CSC', 'QSC',
    # All four conics take in-frame too, for consistency with the other
    # oddballs. (Boundary-edge ticks read nicely on the full-sphere COD/COE
    # wedges, but the clipped COO/COP wedges are too cramped; central-
    # crosshair labels are the uniform, predictable default. Boundary ticks
    # remain available for any of them via tick_style='boundary'.)
    'COD', 'COE', 'COO', 'COP',
})

# Projections that auto-route to boundary-edge ticks. Currently none — every
# interrupted projection reads better with in-frame central-crosshair labels
# (above), and boundary mode is left as an explicit opt-in (tick_style=
# 'boundary', wired to the true edge via _boundary_tick_curve). Kept as the
# hook if a future projection's edge is the natural place for its labels.
_BOUNDARY_TICK_AUTO_FITS_CODES: frozenset[str] = frozenset()


# Frame shapes that need a "hybrid" tick path: keep native astropy
# for the longitude coord (which renders fine) but route the latitude
# coord through skyplothelper's overlay machinery to work around
# upstream tick-rendering bugs. EllipticalFrame (AIT / MOL) is here
# because astropy's tangent-angle calculation for lat tick marks on
# the ellipse-curve spine returns NaN at most positions — only 2 of
# the ~10 lat tick marks render, leaving the panel visually broken.
# The hybrid lat overlay places labels horizontally at the standard
# spacing on the ellipse boundary, matching the publication look the
# native rendering was trying to produce.
_HYBRID_LAT_OVERLAY_FRAME_SHAPES = frozenset({
    'elliptical',
})


def _boundary_tick_curve(ax: Any, fits_code: str | None) -> Any:
    """Return a ``boundary=`` argument for :func:`add_overlay_ticks` that
    traces the projection's TRUE visible edge, for the interrupted /
    non-rectangular all-sky projections whose astropy frame spine is the
    enclosing canvas rectangle (so the default 'boundary' ticks would land
    on the rectangle, not the diamond / wedge / egg / cross).

    Analytic-boundary families (HPX/BON/PCO/conics) return their ``(lon,lat)``
    world polyline (``add_overlay_ticks`` wraps it via
    ``_FrameCurve.from_world_polyline``); the pixel-space families
    (cubes/XPH) return a pre-built ``_FrameCurve`` from
    :func:`_projection_boundary`. Returns ``None`` for standard projections
    (MOL ellipse, Robinson edge, ...) whose astropy spine already traces the
    real edge — there the caller keeps the existing spine-based behavior.
    """
    code = (fits_code or '').upper()
    from .projections import _boundaries
    world_helpers: dict[str, Any] = {
        'HPX': _boundaries.healpix_boundary,
        'BON': _boundaries.bonne_boundary,
        'PCO': _boundaries.polyconic_boundary,
        'COD': _boundaries.conic_boundary,
        'COE': _boundaries.conic_boundary,
        'COO': _boundaries.conic_boundary,
        'COP': _boundaries.conic_boundary,
    }
    helper = world_helpers.get(code)
    if helper is not None:
        return helper(ax)
    if code in ('TSC', 'CSC', 'QSC', 'XPH'):
        from .coord_overlay import _FrameCurve
        path = _projection_boundary(ax)
        if path is not None:
            return _FrameCurve(np.asarray(path.vertices, dtype=float),
                               name='boundary', closed=True)
    return None


def _overlay_tick_direction_from_rc() -> str:
    """Map ``rcParams['xtick.direction']`` to a :meth:`render_ticks` direction.

    Lets the hybrid / boundary / in-frame overlay tick marks honor the active
    base style's tick direction at build time — e.g. a ``'structural'`` base
    sets ``xtick.direction='in'`` (which WCSAxes ignore for their own ticks,
    and which astropy can't draw on a curved spine anyway), so without this
    the sph-drawn elliptical boundary ticks would always point outward. The
    default ``'out'`` leaves rendering unchanged.
    """
    return {'in': 'in', 'out': 'out', 'inout': 'both'}.get(
        rcParams.get('xtick.direction', 'out'), 'out')


def _apply_tick_style(ax: Any, frame_shape: str, tick_style: str,
                      tick_rotation: Any, label_fontsize: float | None = None,
                      fits_code: str | None = None,
                      lon_label_fmt: str | None = None,
                      lon_spacing: float | None = None,
                      lat_spacing: float | None = None) -> None:
    """Wire :func:`add_overlay_ticks` into a freshly-built WCSAxes per
    the requested tick style.

    Parameters
    ----------
    ax : WCSAxes
    frame_shape : str
        Frame-shape key (``'circular'``, ``'sinusoidal'``, ...). Used
        to resolve ``tick_style='auto'``.
    tick_style : {'auto', 'in_frame', 'boundary', 'native'}
        Where to draw tick labels.

        - ``'auto'`` — default. ``'in_frame'`` when ``frame_shape`` is
          in :data:`_IN_FRAME_TICK_AUTO_FRAME_SHAPES`; ``'native'``
          otherwise.
        - ``'in_frame'`` — labels along the central parallel + central
          meridian inside the visible region. Calls
          ``add_overlay_ticks(lon_at='axis', lat_at='axis',
          suppress_default='both')``.
        - ``'boundary'`` — labels on the projection's natural
          boundary curve (the MOL ellipse, the Robinson edge, etc.),
          one tick per gridline×spine intersection, tangent-rotated.
          Calls ``add_overlay_ticks(lon_at='boundary',
          lat_at='boundary', suppress_default='both')``. Both
          ``'in_frame'`` and ``'boundary'`` bypass astropy's default
          tick-discovery — useful for avoiding upstream bugs with
          spurious / one-sided ticks on certain frame classes.
        - ``'native'`` — no overlay; whatever astropy's WCSAxes
          renders by default.
    tick_rotation : {'tangent', 'tangent_upright', 'horizontal'} or float or callable
        Forwarded to
        :meth:`CoordinateOverlay.render_labels` as ``rotate=``. Ignored
        when ``tick_style='native'``. ``'tangent'`` (default, aliased
        ``'tangent_noflip'``) follows the gridline tangent continuously —
        no flip between adjacent labels, with a per-placement-group branch
        keeping labels upright for the current view (leaning past vertical
        only where a group genuinely sweeps through vertical).
        ``'tangent_upright'`` instead clamps each label upright, flipping
        180° where the tangent crosses ±90°.

    Notes
    -----
    Lazy-imports :func:`add_overlay_ticks` to avoid a circular import.
    Calls ``ax.figure.canvas.draw()`` first because the underlying
    intersection algorithm needs valid display coords.
    """
    if tick_style not in ('auto', 'in_frame', 'boundary', 'native'):
        raise ValueError(
            f"tick_style must be 'auto', 'in_frame', 'boundary', or "
            f"'native', got {tick_style!r}")
    # Track whether the auto resolver routed us to native specifically
    # for a frame shape that needs the hybrid lat-overlay workaround.
    # Only triggered from the auto path; explicit ``tick_style='native'``
    # preserves bare astropy behavior as an escape hatch.
    apply_hybrid_lat_overlay = False
    # Auto-routing is a convenience: if the overlay attach fails on some axes
    # (e.g. a draw of a strongly divergent projection like the COP perspective
    # conic), the frame must still build — so the auto path swallows overlay
    # errors. An EXPLICIT tick_style raises, so the user sees the problem.
    auto_routed = tick_style == 'auto'
    if tick_style == 'auto':
        fits_code_upper = (fits_code or '').upper()
        # The per-FITS-code in-frame/boundary defaults assume the WHOLE-SKY net
        # is in view (their label positions span the full -90..+90 / all-lon
        # graticule). On a ZOOMED field frame (fov_deg/cdelt) most of that net
        # is off-panel, so the labels scatter across the figure — fall back to
        # native (spine) ticks there. Frame-shape routing (globe 'circular',
        # the custom pseudocylindricals) is NOT gated: globes show a full
        # hemisphere and the customs have no zoomed field mode.
        allsky_frame = bool(getattr(ax, '_sph_is_allsky', True))
        if allsky_frame and fits_code_upper in _BOUNDARY_TICK_AUTO_FITS_CODES:
            tick_style = 'boundary'
        elif (frame_shape in _IN_FRAME_TICK_AUTO_FRAME_SHAPES
                or (allsky_frame
                    and fits_code_upper in _IN_FRAME_TICK_AUTO_FITS_CODES)):
            tick_style = 'in_frame'
        else:
            tick_style = 'native'
            if frame_shape in _HYBRID_LAT_OVERLAY_FRAME_SHAPES:
                apply_hybrid_lat_overlay = True
    if tick_style == 'native':
        if apply_hybrid_lat_overlay:
            # Lon stays native; lat goes through skyplothelper's
            # overlay machinery so its tangent angles are computed
            # correctly. Horizontal labels at standard 30° spacing
            # on the ellipse boundary keeps the publication aesthetic
            # the native path was trying to achieve.
            from .coord_overlay import add_overlay_ticks
            overlay_label_kwargs: dict[str, Any] = {'rotate': 'horizontal'}
            if label_fontsize is not None:
                overlay_label_kwargs['fontsize'] = label_fontsize
            ax.figure.canvas.draw()
            try:
                add_overlay_ticks(
                    ax,
                    lat_at='boundary', lon_at=None,
                    lat_vals=np.arange(-60.0, 61.0, 30.0),
                    suppress_default='lat',
                    label_kwargs=overlay_label_kwargs,
                    tick_kwargs={'direction': _overlay_tick_direction_from_rc()},
                    _auto=True,
                )
            except Exception:
                # Auto-routing is a convenience; never break the
                # caller if the overlay attach fails on some axes.
                pass
        return
    # The projection's TRUE visible edge (None for standard projections,
    # whose astropy frame spine already traces it).
    boundary_curve = _boundary_tick_curve(ax, fits_code)
    in_frame_view_clip = False
    if tick_style == 'in_frame':
        lon_at = lat_at = 'axis'
        # In-frame labels span the full projection net (every gridline value
        # along the central crosshair). On a ZOOMED field frame most of that
        # net is off-panel, so without a clip the labels scatter across the
        # figure. Clip them to the axes VIEW rectangle (built after the draw,
        # below, once the bbox is valid) — permissive enough never to over-clip
        # the all-sky central labels, but it drops the off-view ones. EXCEPTION:
        # the lat-clipped conics (COO/COP) keep their WEDGE boundary, which also
        # drops the far-pole latitude labels that would otherwise float in the
        # empty bbox below the wedge.
        if (fits_code or '').upper() not in ('COO', 'COP'):
            in_frame_view_clip = True
            boundary_curve = None
    else:  # 'boundary'
        lon_at = lat_at = 'boundary'
        # Labels placed ON the true edge instead of the canvas rectangle.
    label_kwargs: dict[str, Any] = ({} if tick_rotation == 'tangent'
                                     else {'rotate': tick_rotation})
    if label_fontsize is not None:
        label_kwargs['fontsize'] = label_fontsize
    # Overlay labels format longitude themselves (the native ax.coords
    # format_unit doesn't reach them), so pass the desired lon unit through
    # to render_labels' ``fmt`` ('hour' / 'deg'; lat is always degrees).
    if lon_label_fmt is not None:
        label_kwargs['fmt'] = lon_label_fmt
    from .coord_overlay import add_overlay_ticks
    # On a zoomed field frame the all-sky default graticule (30°/15°) contains
    # no values inside the field, so in-frame ticks find no gridline crossings.
    # Derive field-scale nice values from the view extent instead. All-sky
    # frames keep the default graticule (unchanged behavior); circular-limb
    # globes also keep it (their graticule spans the full hemisphere — a field
    # restriction would stop it short of the limb).
    allsky_frame = bool(getattr(ax, '_sph_is_allsky', True))
    globe_frame = bool(getattr(ax, '_sph_is_globe', False))
    tick_vals_kwargs: dict[str, Any] = {}
    try:
        ax.figure.canvas.draw()
        if tick_style == 'in_frame' and not allsky_frame and not globe_frame:
            from .coord_overlay import _field_graticule_vals
            lon_v, lat_v = _field_graticule_vals(ax)
            if lon_v is not None:
                tick_vals_kwargs['lon_vals'] = lon_v
            if lat_v is not None:
                tick_vals_kwargs['lat_vals'] = lat_v
        elif (tick_style == 'in_frame' and frame_shape == 'circular'
              and lon_spacing is not None and lat_spacing is not None):
            # Globe (make_globe_frame passes its lon/lat spacing): place the
            # in-frame labels on the same graticule as the grid lines instead
            # of the all-sky overlay default (30°/15°) — so a coarser spacing
            # actually coarsens the labels and each one sits on its grid line.
            tick_vals_kwargs['lon_vals'] = np.arange(0.0, 360.0, lon_spacing)
            tick_vals_kwargs['lat_vals'] = np.arange(
                -90.0 + lat_spacing, 90.0, lat_spacing)
        if in_frame_view_clip:
            # View rectangle in display pixels (bbox now valid post-draw).
            from .coord_overlay import _FrameCurve
            bb = ax.bbox
            rect = np.array([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1],
                             [bb.x0, bb.y1], [bb.x0, bb.y0]])
            boundary_curve = _FrameCurve(rect, name='view', closed=True)
        add_overlay_ticks(ax, lon_at=lon_at, lat_at=lat_at,
                          boundary=boundary_curve,
                          suppress_default='both',
                          label_kwargs=label_kwargs,
                          tick_kwargs={
                              'direction': _overlay_tick_direction_from_rc()},
                          _auto=True,
                          **tick_vals_kwargs)
    except Exception:
        if not auto_routed:
            raise  # explicit request — surface the failure
        # Auto-routed: tick styling is cosmetic, never fail the frame build.


def _restrict_field_edge_ticks(ax: Any) -> None:
    """Pin single-field tick marks to their natural spines.

    On a flat single-field frame (TAN/SIN/ZEA/cube-face zoom) the meridians
    converge toward the pole, so they fan across the panel and cross the
    LEFT/RIGHT spines near the top corners; the parallels likewise dip across
    the BOTTOM. astropy's default per-spine tick heuristic (raw visible-axes
    ``'bltr'``) then scatters a few lon ticks onto the side spines and a lat
    tick onto the bottom — which naively reads as an error. Pinning lon to
    bottom/top and lat to left/right removes those strays.

    Safe for the north-up, axis-aligned fields ``make_wcs_frame`` builds:
    meridians always enter/exit through the top & bottom edges and parallels
    through the sides, so every *real* tick is kept — only the converging-corner
    strays drop. (A manually rolled WCS, where a meridian legitimately exits a
    side spine, is the case ``edge_ticks='all'`` is for.) Cosmetic — never fails
    the frame build.
    """
    try:
        ax.coords[0].set_ticks_position('bt')   # lon: bottom/top
        ax.coords[1].set_ticks_position('lr')   # lat: left/right
    except Exception:
        pass


def _suppress_curved_minor_ticks(ax: Any) -> None:
    """Turn off astropy's native minor ticks on a curved all-sky / globe frame.

    WCSAxes honor ``rcParams['xtick.minor.visible']`` (True under the
    structural / journal / press / ... base presets). On a curved all-sky or
    circular-globe spine astropy scatters those minor ticks into a dense row
    along the central parallel and all the way around the limb — clutter, not
    useful subdivisions — whereas on a flat field they ARE useful, so only the
    curved frames are suppressed here. No-op under the mpl default (minor ticks
    already off), so the default-style look and the visual baselines are
    unchanged; it only removes the clutter that a minor-tick base style would
    otherwise add. A caller who genuinely wants globe minor ticks can re-enable
    via ``ax.coords[i].display_minor_ticks(True)``.
    """
    try:
        ax.coords[0].display_minor_ticks(False)
        ax.coords[1].display_minor_ticks(False)
    except Exception:
        pass


def _pad_title_for_pole_top_projection(ax: Any, extra_pt: float = 18) -> None:
    """
    Inject extra default ``pad`` into ``ax.set_title`` for projections
    whose topmost tick labels crowd the default title position.

    Monkey-patches the bound ``set_title`` method so that subsequent
    user calls like ``ax.set_title('My Title')`` automatically render
    the title with clearance above the topmost tick label. The user
    can still pass an explicit ``pad=`` kwarg to override.
    """
    _orig_set_title = ax.set_title
    _default_extra = extra_pt

    def _set_title_with_extra_pad(label: str = '', *args: Any,
                                  pad: float | None = None,
                                  **kwargs: Any) -> Any:
        if pad is None:
            pad = rcParams.get('axes.titlepad', 6.0) + _default_extra
        return _orig_set_title(label, *args, pad=pad, **kwargs)

    ax.set_title = _set_title_with_extra_pad


def _draw_allsky_lon_boundary(ax: Any, hdr: Any, lat_max: float = 89.99,
                              n: int = 400, color: str | None = None,
                              lw: float = 0.7, zorder: float = 1.5) -> None:
    """Draw the lon = CRVAL ± 180° meridian as an explicit boundary
    overlay.

    For projections whose natural valid region sits inside a larger
    rectangular frame (BON and PCO), the projection's iconic outline IS
    the lon=CRVAL±180° meridian — but astropy's gridline densifier
    doesn't draw this wraparound seam cleanly. This helper samples the
    two seam meridians via
    :func:`skyplothelper.projections._boundaries.bonne_boundary`
    (just the west + east lon=CRVAL±180 meridians, no top/bottom arcs)
    and plots them as a line overlay — restoring the BON cardioid and
    the PCO double-lobe egg. Sampling the meridians directly (rather
    than relying on a closed polygon) handles PCO's non-monotonic seam,
    which loops past both poles.
    """
    from .projections._boundaries import bonne_boundary

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(hdr)

    # Shim object with the .wcs attribute bonne_boundary expects.
    # Using the just-built ``wcs`` rather than ``ax.wcs`` keeps the
    # CRVAL semantics identical to the original implementation.
    class _AxShim:
        wcs: Any
    shim = _AxShim()
    shim.wcs = wcs

    lonlat = bonne_boundary(shim, lat_max=lat_max, n=n)
    transform = ax.get_transform('pixel')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        px, py = wcs.world_to_pixel_values(lonlat[:, 0], lonlat[:, 1])
    # The projection silhouette IS the frame edge on an all-sky plot (there is
    # no rectangular spine), so it follows the frame color rather than a
    # literal black that disappears on a dark theme.
    if color is None:
        color = rcParams['axes.edgecolor']
    ax.plot(px, py, transform=transform,
            color=color, lw=lw, zorder=zorder)


# Projections whose default astropy graticule drops wrap-side / multi-face
# segments (HEALPix diamonds, the polyconic egg, the Bonne cardioid, the
# quadcube faces) or spills gridlines outside the visible region (conic
# wedges). For these, make_wcs_frame replaces the default grid with a
# same-frame coordinate overlay (which samples gridlines densely and handles
# wrap-side segments), clipped to the projection's true visible boundary via
# _projection_boundary. For the quadcubes (TSC/CSC/QSC) the cross-perimeter
# clip trims the off-center meridian segments that otherwise run out into the
# empty bbox corners, so the graticule stays inside the cross.
_BACKFILL_GRID_CODES = frozenset(
    {'HPX', 'XPH', 'BON', 'PCO', 'COD', 'COE', 'COO', 'COP',
     'TSC', 'CSC', 'QSC'})

# Of the backfilled projections, those whose visible-region boundary is
# reliable enough to confine the overlay gridlines to (by NaN-masking samples
# outside the pixel-space boundary polygon). COD/COE earn this: centered on the
# standard parallel (CRVAL2 = PV2_1) their full-sphere wedge boundary is a clean
# apex-to-far-parallel curve. COO/COP are NOT here — they diverge toward the far
# pole, so the polygon mis-clips; they use the dedicated world-latitude-band
# clip in _backfill_overlay_grid instead. The polyconic egg stays excluded
# (double-lobed, per-flavor unreliable).
_GRIDLINE_MASK_CODES = frozenset(
    {'HPX', 'BON', 'XPH', 'TSC', 'CSC', 'QSC', 'COD', 'COE'})


def _measure_allsky_proj_extent(
        ctype1: str, ctype2: str, center_lon: float, center_lat: float,
        pv2_1: float | None = None, pv2_2: float | None = None,
        lat_range: tuple[float, float] = (-89.99, 89.99),
        ) -> tuple[float, float, float, float]:
    """Measure a projection's all-sky envelope in the projection plane.

    Projects a dense ``(lon, lat)`` grid through a unit-scale WCS (CDELT=±1°,
    reference at the projection origin) and returns
    ``(x_min, x_max, y_min, y_max)`` in degrees: the (generally asymmetric)
    left/right and bottom/top of the visible region. Used to fit a frame
    snugly to projections whose envelope is lopsided — the Bonne cardioid
    (whose south cusp sits well above the symmetric ``-y_max`` bound), the
    conic wedges (centered on the standard parallel, far pole clipped via
    ``lat_range``), and the quadcube cross (whose net is offset in x — the
    pole column sits over a face that is right-of-center, leaving an empty
    arm if framed symmetrically).
    """
    hdr = pyfits.Header({
        'NAXIS': 2, 'NAXIS1': 3, 'NAXIS2': 3, 'CRPIX1': 1.0, 'CRPIX2': 1.0,
        'CRVAL1': center_lon, 'CRVAL2': center_lat,
        'CDELT1': -1.0, 'CDELT2': 1.0, 'CUNIT1': 'deg', 'CUNIT2': 'deg',
        'CTYPE1': ctype1, 'CTYPE2': ctype2,
    })
    if pv2_1 is not None:
        hdr['PV2_1'] = float(pv2_1)
    if pv2_2 is not None:
        hdr['PV2_2'] = float(pv2_2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(hdr)
        lon = np.linspace(center_lon - 180.0, center_lon + 180.0, 361)
        lat = np.linspace(lat_range[0], lat_range[1], 361)
        glon, glat = np.meshgrid(lon, lat)
        px, py = wcs.world_to_pixel_values(glon.ravel(), glat.ravel())
    # intermediate (projection-plane) coords in degrees: (pix - CRPIX) * CDELT
    xproj = -(np.asarray(px) - 1.0)
    yproj = np.asarray(py) - 1.0
    return (float(np.nanmin(xproj)), float(np.nanmax(xproj)),
            float(np.nanmin(yproj)), float(np.nanmax(yproj)))


def _backfill_overlay_grid(
        ax: Any, lon_spacing: float, lat_spacing: float,
        color: str, alpha: float, fits_code: str,
        lw: float | None = None, ls: str | None = None) -> None:
    """Replace astropy's default grid with a dense same-frame overlay.

    astropy's gridline densifier truncates or skips wrap-side segments on
    full-sky non-rectangular envelopes (HPX diamonds, the PCO egg, the BON
    cardioid) and lets conic-wedge gridlines spill across the rectangular
    bbox. The ``CoordinateOverlay`` machinery samples each meridian/parallel
    densely and handles the wrap-side / multi-face segments cleanly. For
    projections whose visible region is bounded by a custom curve
    (BON/PCO/conics) the overlay gridlines are clipped to that boundary so
    they don't wander into the empty bbox corners; HEALPix (HPX/XPH) needs no
    clip (the stepped diamond fills its frame).
    """
    # Lazy import: coord_overlay imports wcs_frame, so a module-level import
    # would be circular.
    from .coord_overlay import add_coord_overlay
    from .projections._boundaries import _conic_pv2_1, conic_visible_lat_range

    # NB: do NOT call ax.coords[i].grid(draw_grid=False) here — counter to its
    # name it *enables* astropy's default grid (drawing the very incomplete
    # lines we're replacing). The backfill path never enables the default grid,
    # so leaving it untouched keeps it off; the dense overlay below is the grid.

    # Conics only render a visible-latitude wedge: COO/COP drop the divergent
    # far-pole cap (clipped to conic_visible_lat_range), COD/COE span the full
    # sphere. Cap the parallels we draw to that band so no sub-wedge parallel
    # is generated, and remember the band to clip the meridian samples below.
    lat_lo, lat_hi = -90.0, 90.0
    is_conic = fits_code in ('COD', 'COE', 'COO', 'COP')
    if is_conic:
        lat_lo, lat_hi = conic_visible_lat_range(fits_code, _conic_pv2_1(ax))

    lon_vals = np.arange(0.0, 360.0, lon_spacing)
    # Parallels excluding the poles (a single point, not a line), capped to the
    # visible latitude band (a no-op for the full-sphere ±90 default).
    lat_vals = np.arange(-90.0 + lat_spacing, 90.0 - lat_spacing / 2.0,
                         lat_spacing)
    lat_vals = lat_vals[(lat_vals >= lat_lo) & (lat_vals <= lat_hi)]
    overlay = add_coord_overlay(
        ax, frame=_get_wcs_frame_name(ax), lon_vals=lon_vals,
        lat_vals=lat_vals, color=color, alpha=alpha,
        lw=(0.5 if lw is None else lw), ls=(':' if ls is None else ls))

    # COO/COP diverge toward the far pole, so the pixel-space boundary polygon
    # is unreliable for them (it let the graticule flare past the wedge) —
    # clip the samples in WORLD space instead, dropping any whose latitude
    # falls outside the visible band. matplotlib breaks each polyline at the
    # NaN, so the seam meridians end cleanly at the bottom parallel and nothing
    # spills past the wedge. COD/COE span the full sphere and clip cleanly with
    # the polygon mask below, so they keep that (unchanged) path.
    if fits_code in ('COO', 'COP'):
        for artists in overlay.lon_artists + overlay.lat_artists:
            for ln in artists:
                xd, yd = ln.get_data()
                xd = np.asarray(xd, dtype=float)
                yd = np.asarray(yd, dtype=float)
                bad = (yd < lat_lo) | (yd > lat_hi)
                xd[bad] = np.nan
                yd[bad] = np.nan
                ln.set_data(xd, yd)
        return

    # Confine the overlay gridlines to the projection's visible region by
    # NaN-masking the samples that fall outside its boundary (HPX diamond /
    # BON cardioid / XPH butterfly / quadcube cross). matplotlib's
    # ``set_clip_path`` does NOT clip these world-transform Line2D artists, so
    # masking the line data (matplotlib breaks the line at NaN) is the
    # reliable mechanism. Only applied where the boundary is reliable.
    if fits_code in _GRIDLINE_MASK_CODES:
        path = _projection_boundary(ax)
        if path is not None:
            for artists in overlay.lon_artists + overlay.lat_artists:
                for ln in artists:
                    xd, yd = ln.get_data()
                    xd = np.asarray(xd, dtype=float)
                    yd = np.asarray(yd, dtype=float)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        px, py = ax.wcs.world_to_pixel_values(xd, yd)
                    pts = np.column_stack([px, py])
                    bad = ~path.contains_points(pts)
                    # Also break any segment whose MIDPOINT falls outside the
                    # boundary. The polar-cap interruptions (HPX diamond / XPH
                    # butterfly / cube faces) split a parallel across a notch:
                    # both sample endpoints sit on facet edges (so the per-
                    # point test reads them as inside), but the straight
                    # segment connecting them crosses the empty notch. Testing
                    # the midpoint catches that bridge; NaN the segment's end
                    # sample to split the polyline there. Threshold-free, and a
                    # no-op on the smooth continuous gridlines.
                    if len(pts) > 1:
                        mids = 0.5 * (pts[:-1] + pts[1:])
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            mid_out = ~path.contains_points(mids)
                        bad[1:] |= mid_out
                    xd[bad] = np.nan
                    yd[bad] = np.nan
                    ln.set_data(xd, yd)


def _boundary_clip_path(
        ax: Any, lonlat: np.ndarray, margin: float = 20.0) -> Any:
    """Build a matplotlib ``Path`` (in data coords) from a ``(lon, lat)``
    boundary polyline, for use as a clip path.

    The polyline is sanitized in pixel space first — NaN samples (projection
    singularities, e.g. the COP cone apex) and samples far outside the axes
    bbox get dropped so the clip polygon can't collapse or invert when a
    boundary side extrapolates toward infinity. Returns ``None`` if too few
    samples survive.
    """
    from matplotlib.path import Path

    pix = ax.get_transform('world').transform(lonlat)
    bb = ax.bbox
    finite = np.isfinite(pix).all(axis=1)
    inside = ((pix[:, 0] >= bb.x0 - margin) & (pix[:, 0] <= bb.x1 + margin)
              & (pix[:, 1] >= bb.y0 - margin) & (pix[:, 1] <= bb.y1 + margin))
    keep = finite & inside
    if keep.sum() < 3:
        return None
    return Path(ax.transData.inverted().transform(pix[keep]))


def _axes_fits_code(ax: Any) -> str | None:
    """Return the FITS projection code of a WCSAxes (e.g. 'HPX'), or None."""
    try:
        return str(ax.wcs.wcs.ctype[0]).split('-')[-1].strip().upper()
    except Exception:
        return None


def _cube_plus_path(ax: Any) -> Any:
    """Visible-region outline (a plus/cross) of a quadcube projection.

    The all-sky quadcube (TSC/CSC/QSC) unfolds into a cross: an equatorial
    band of four faces (lat ∈ ±45°, full longitude) plus the two polar faces
    stacked above/below in a central column (lon ∈ CRVAL ± 45°). This is the
    world-edge perimeter kapteyn's ``getperimeter`` draws; here the band and
    column pixel extents come from those face-edge world points (lat = ±45° on
    the central meridian, lon = CRVAL ± 45° on the equator), and the cross is
    built as a 12-vertex plus that fills the frame in each arm.

    Returns a closed ``matplotlib.path.Path`` in data coordinates, or ``None``.
    """
    from matplotlib.path import Path
    try:
        crval_lon = float(ax.wcs.wcs.crval[0])
    except Exception:
        return None
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # band bottom/top (lat=±45 on the central meridian); column left/right
        # edges (lon=CRVAL±45 on the equator). Pixel == data coords on WCSAxes.
        _, yb_a = ax.wcs.world_to_pixel_values(crval_lon, -45.0)
        _, yb_b = ax.wcs.world_to_pixel_values(crval_lon, 45.0)
        xc_a, _ = ax.wcs.world_to_pixel_values(crval_lon - 45.0, 0.0)
        xc_b, _ = ax.wcs.world_to_pixel_values(crval_lon + 45.0, 0.0)
    vals = [float(v) for v in (yb_a, yb_b, xc_a, xc_b)]
    if not all(np.isfinite(vals)):
        return None
    yb_lo, yb_hi = min(yb_a, yb_b), max(yb_a, yb_b)
    xc_lo, xc_hi = min(xc_a, xc_b), max(xc_a, xc_b)
    xl_lo, xl_hi = min(x0, x1), max(x0, x1)
    yl_lo, yl_hi = min(y0, y1), max(y0, y1)
    verts = [(xc_lo, yl_hi), (xc_hi, yl_hi), (xc_hi, yb_hi), (xl_hi, yb_hi),
             (xl_hi, yb_lo), (xc_hi, yb_lo), (xc_hi, yl_lo), (xc_lo, yl_lo),
             (xc_lo, yb_lo), (xl_lo, yb_lo), (xl_lo, yb_hi), (xc_lo, yb_hi),
             (xc_lo, yl_hi)]
    return Path(np.array(verts, dtype=float))


# Unified visible-region boundary provider for the interrupted / non-
# rectangular projections, returning a data-coord clip Path (or None). One
# dispatcher backs the data clip, the gridline clip, and the outline draw:
#   * HPX/BON/PCO/conics — closed analytic (lon,lat) curve from `_boundaries`
#   * XPH                — NaN-edge limb (no closed-form boundary)
#   * TSC/CSC/QSC        — the quadcube cross perimeter
def _projection_boundary(ax: Any) -> Any:
    code = _axes_fits_code(ax)
    if code is None:
        return None
    if code in ('TSC', 'CSC', 'QSC'):
        return _cube_plus_path(ax)
    # XPH butterfly, BON cardioid, PCO egg: trace the TRUE visible region via
    # the NaN-edge detector. It works in ANY aspect (it finds wherever the
    # projection goes NaN), so it captures the cardioid/egg concave top+bottom
    # notches and — unlike the analytic bonne_boundary/polyconic_boundary (the
    # lon=CRVAL±180 meridians, which trace the edge only in the equatorial
    # aspect) — stays correct under an oblique center_lat. Used for the data
    # clip so pcolormesh cells can't bridge those notches. (For PCO this is the
    # NOTCH bridging only; its overlapping-lobe interior is a separate
    # won't-fix — see _DATA_CLIP_CODES / the seam-mask notes.)
    if code in ('XPH', 'BON', 'PCO'):
        return _visible_region_clip_path(ax)
    # The conic wedge boundary is only valid for the all-sky frame (reference
    # on the standard parallel, CRVAL2 = PV2_1). A field-view conic (fov_deg /
    # cdelt zoom) centers elsewhere, so the wedge would mis-clip / blank it —
    # skip the boundary entirely there.
    if code in ('COD', 'COE', 'COO', 'COP') and not getattr(
            ax, '_sph_is_allsky', True):
        return None
    from .projections import _boundaries
    analytic: dict[str, Any] = {
        'HPX': _boundaries.healpix_boundary,
        'COD': _boundaries.conic_boundary,
        'COE': _boundaries.conic_boundary,
        'COO': _boundaries.conic_boundary,
        'COP': _boundaries.conic_boundary,
    }
    helper = analytic.get(code)
    if helper is not None:
        return _boundary_clip_path(ax, helper(ax))
    return None


# Codes whose DATA we clip to the visible region: those that both genuinely
# bleed (cells project outside the visible region) and have a reliable
# boundary. BON/PCO need it because pcolormesh cells bridge the cardioid/egg
# concave top+bottom notches (most visible under an oblique center_lat) — the
# NaN-edge boundary clips them. (PCO's overlapping-lobe interior is a separate
# won't-fix: the lobes double-value the data INSIDE the envelope, which a
# boundary clip can't separate; the clip still fixes the notch bridging.)
# Conics bleed badly (data fills the bbox past the wedge) and have a reliable
# wedge boundary once centered on the standard parallel.
_DATA_CLIP_CODES = frozenset(
    {'HPX', 'XPH', 'BON', 'PCO', 'TSC', 'CSC', 'QSC',
     'COD', 'COE', 'COO', 'COP'})


def _visible_region_clip_path(ax: Any, n: int = 600) -> Any:
    """Trace a projection's visible-region outline by NaN-edge detection.

    For projections whose visible region has no clean closed-form boundary
    (the HEALPix butterfly XPH), find it numerically the way kapteyn's
    ``scanborder`` does: inverse-project a dense pixel grid over the axes and
    locate the valid→invalid (finite→NaN) transition. The boundary contour of
    the finite-coordinate mask is the visible-region outline. Cached on the
    axes since the same path is reused for the data clip, the outline, and the
    gridline clip.

    Returns a closed ``matplotlib.path.Path`` in data coordinates, or ``None``
    if the region fills the frame (no NaN edge) or the axes has no WCS.
    """
    cached = getattr(ax, '_sph_visible_clip_path', None)
    if cached is not None:
        return cached
    if not hasattr(ax, 'wcs') or ax.wcs is None:
        return None
    import contourpy
    from matplotlib.path import Path

    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    xs = np.linspace(min(x0, x1), max(x0, x1), n)
    ys = np.linspace(min(y0, y1), max(y0, y1), n)
    grid_x, grid_y = np.meshgrid(xs, ys)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lon, lat = ax.wcs.pixel_to_world_values(grid_x, grid_y)
    mask = (np.isfinite(lon) & np.isfinite(lat)).astype(float)
    if mask.min() == mask.max():
        return None  # all-valid (no NaN edge) or all-invalid
    segs = [np.asarray(s) for s in contourpy.contour_generator(xs, ys, mask).lines(0.5)]
    if not segs:
        return None
    seg = max(segs, key=lambda s: s.shape[0])
    if not np.allclose(seg[0], seg[-1]):
        seg = np.vstack([seg, seg[0]])  # close the loop for clipping
    path = Path(seg)
    ax._sph_visible_clip_path = path
    return path


def clip_to_projection_boundary(ax: Any, artist: Any) -> Any:
    """Clip a data artist to the projection's visible-region boundary.

    Interrupted / non-rectangular projections (HPX, BON, PCO, the conics)
    leave empty regions inside their rectangular axes frame. A ``pcolormesh``
    (or ``imshow``) drawn there bleeds color into those empty regions — most
    visibly HEALPix data spilling above the stepped-diamond peaks. This clips
    *artist* to the projection's true outline so the data stays inside the
    visible region. It is a no-op for projections without a special boundary
    (AIT, MOL, the cylindricals, …) and for non-WCS axes.

    Parameters
    ----------
    ax : WCSAxes
    artist : matplotlib artist
        e.g. the ``QuadMesh`` returned by ``ax.pcolormesh``.

    Returns
    -------
    artist : the same artist (clipped in place), for chaining.
    """
    code = _axes_fits_code(ax)
    if code not in _DATA_CLIP_CODES:
        return artist
    try:
        path = _projection_boundary(ax)
        if path is not None:
            artist.set_clip_path(path, transform=ax.transData)
    except Exception:
        pass  # clipping is a refinement — never fail the plot
    return artist


_NICE_TICK_STEPS_DEG = np.array([
    # Sub-mas regime (μas) — relevant for space VLBI / EHT-scale offset frames.
    # 1e-10 deg = 0.36 μas; 5e-8 deg = 180 μas.
    1e-10, 2e-10, 5e-10,
    1e-9, 2e-9, 5e-9,
    1e-8, 2e-8, 5e-8,
    # Sub-arcsec regime (mas) — standard VLBI offset frames
    1e-7, 2e-7, 5e-7,
    1e-6, 2e-6, 5e-6,
    1e-5, 2e-5, 5e-5,
    1e-4, 2e-4, 5e-4,
    # Arcsec
    1.0 / 3600, 2.0 / 3600, 5.0 / 3600, 10.0 / 3600, 30.0 / 3600,
    # Arcmin
    1.0 / 60, 2.0 / 60, 5.0 / 60, 10.0 / 60, 30.0 / 60,
    # Degrees
    1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 45.0, 60.0, 90.0,
])


def _auto_tick_spacing(fov_deg: float, n_target: int = 6) -> float:
    """Pick a sensible tick spacing in degrees for a given FOV.

    Targets ``~n_target`` ticks across the FOV by selecting the
    ``_NICE_TICK_STEPS_DEG`` value closest to ``fov_deg / n_target``
    on a log scale. Used by ``make_wcs_frame`` when
    ``lon_spacing='auto'`` or ``lat_spacing='auto'`` (the default
    since 0.6.x, replacing the hard-coded 30°). Without this,
    sub-degree TAN fields end up with zero ticks landing within the
    visible range and astropy WCSAxes hides the axis entirely.

    ``n_target=6`` is a deliberate choice: it puts ~30° ticks on a
    full-sky AIT panel (matching pre-auto behavior), ~10' on a 1°
    TAN field, and ~30" on a 200" annotation panel — generally five
    or six ticks across the visible range, which reads as "labeled
    every ~15-20% of the axis" without crowding.
    """
    if fov_deg <= 0:
        return 30.0
    target = fov_deg / max(n_target, 1)
    idx = int(np.argmin(np.abs(np.log10(_NICE_TICK_STEPS_DEG)
                               - np.log10(target))))
    return float(_NICE_TICK_STEPS_DEG[idx])


def _clamp_spacing(spacing_deg: float, domain_deg: float, axis_label: str,
                   max_ticks: int = 1000) -> float:
    """Failsafe for ``set_ticks(spacing=)`` / ``arange`` over a full coordinate
    domain: if the spacing would enumerate more than ``max_ticks`` ticks across
    ``domain_deg``, fall back to the sensible auto spacing and warn.

    astropy's spacing locator (and an explicit ``arange``) walk the whole
    coordinate domain, so a degenerately small spacing — from a tiny
    ``fov_deg`` or an explicit ``lon_spacing`` / ``lat_spacing`` — would
    otherwise generate millions of ticks and freeze the draw. A friendly
    warning + fallback beats a frozen machine.
    """
    if not np.isfinite(spacing_deg) or spacing_deg <= 0:
        return spacing_deg
    n_est = domain_deg / spacing_deg
    if n_est > max_ticks:
        fallback = _auto_tick_spacing(domain_deg)
        warnings.warn(
            f"make_wcs_frame: {axis_label} tick spacing {spacing_deg:g}° would "
            f"place ~{n_est:.0f} ticks across the {domain_deg:g}° domain "
            f"(>{max_ticks}); falling back to {fallback:g}°. For finer ticks "
            f"use a zoomed field (smaller fov_deg) so ticks span only the "
            f"visible region.",
            UserWarning, stacklevel=3)
        return fallback
    return spacing_deg


def _field_tick_values(center_deg: float, spacing_deg: float, span_deg: float,
                       pad: float = 0.6,
                       max_ticks: int = 200) -> npt.NDArray[np.float64]:
    """Explicit tick values on the ``spacing_deg`` grid spanning a field of
    angular width ``span_deg`` about ``center_deg``.

    Used for zoomed field frames instead of astropy's ``set_ticks(spacing=)``,
    which enumerates ticks across the *whole* coordinate domain (lon 0–360°,
    lat ±90°): a degenerately small ``fov_deg`` (e.g. 50 mas → ~2e-6° spacing)
    would otherwise generate ~1e8 candidate ticks and freeze the draw. Placing
    explicit values across only the visible field bounds the count to
    ``~span/spacing`` regardless of how small the field is. ``max_ticks`` is a
    final failsafe clamp.
    """
    if not np.isfinite(spacing_deg) or spacing_deg <= 0:
        spacing_deg = max(span_deg, 1e-12)
    half = max(pad * span_deg, spacing_deg)
    lo = np.floor((center_deg - half) / spacing_deg) * spacing_deg
    n = int((2.0 * half) / spacing_deg) + 2
    n = min(max(n, 1), max_ticks)
    return lo + spacing_deg * np.arange(n)


def _apply_lon_units(ax: Any, lon_units: str, auto_to_degrees: bool) -> None:
    """Set the longitude axis format unit from a resolved ``lon_units``.

    ``auto_to_degrees`` is the frame/direction policy outcome for the
    ``'auto'`` case (True → degrees, False → leave the existing default,
    which is hour-angle for equatorial frames). An explicit
    ``'hours'`` / ``'degrees'`` always wins over the policy."""
    if lon_units == 'hours':
        ax.coords[0].set_format_unit(u.hourangle)
    elif lon_units == 'degrees' or (lon_units == 'auto' and auto_to_degrees):
        ax.coords[0].set_format_unit(u.deg, decimal=True)


def _overlay_lon_fmt(lon_units: str, geographic: bool) -> str | None:
    """Longitude ``fmt`` for the coord-overlay tick labels (globes / in-frame
    styles): ``'hour'`` / ``'deg'`` to force, or ``None`` to defer to the
    overlay's frame-based auto (equatorial → hours, else degrees). For
    ``'auto'`` a geographic frame forces degrees (overriding the RA-hours
    default of an east-right celestial frame)."""
    if lon_units == 'hours':
        return 'hour'
    if lon_units == 'degrees':
        return 'deg'
    return 'deg' if geographic else None


def make_wcs_frame(subplotnumber: Any = 111, projection: str = 'AIT',
                   center: SkyCoord | float | tuple[float, float] | None = None,
                   center_lon: float | None = None,
                   center_lat: float | None = None,
                   frame: str = 'ICRS', direction: str = 'sky',
                   lon_units: str = 'auto',
                   lon_spacing: float | str = 'auto',
                   lat_spacing: float | str = 'auto',
                   grid: bool = True, gridcolor: str = '0.5',
                   gridalpha: float = 0.5,
                   gridlw: float | None = None,
                   gridls: str | None = None,
                   aspect: Any = 'auto', npix: Any = None,
                   shape: str | None = None, cdelt: float | None = None,
                   fov_deg: float | None = None,
                   lonpole: float = 0., latpole: float = 0.,
                   equinox: float | None = 2000.0,
                   obstime: Any = None,
                   apply_format_defaults: bool = True,
                   pv2_1: float | None = None, pv2_2: float | None = None,
                   return_hdr: bool = False, fig: Any = None,
                   subplot_kw: dict[str, Any] | None = None,
                   tick_style: str = 'auto', tick_rotation: Any = 'tangent',
                   edge_ticks: str = 'auto',
                   auto_fontsize: bool = True,
                   outline_color: Any = None, outline_lw: float | None = None,
                   **kwargs: Any) -> Any:
    """
    Create a WCSAxes plot frame for any supported projection.

    Unified entry point for all-sky, globe, and field frames. Accepts
    both FITS 3-letter codes and human-readable projection names
    (case-insensitive); the frame ``shape`` ('elliptical', 'circular',
    'rectangular') selects the all-sky / globe / field flavor.

    Parameters
    ----------
    subplotnumber : int, tuple, SubplotSpec, or Axes
        Subplot specification. Accepts the standard matplotlib
        forms — int (e.g. ``111``), tuple (e.g. ``(2, 3, 1)``),
        or a ``SubplotSpec``. Also accepts a pre-existing
        matplotlib ``Axes`` object: the axes is removed and
        replaced by a WCSAxes at the same ``SubplotSpec``
        position. Useful for ``plt.subplots`` / ``GridSpec``
        workflows where the user pre-builds the panel layout
        and wants to convert one panel into a WCSAxes:

        >>> fig, axes = plt.subplots(2, 3)
        >>> sky_ax = make_wcs_frame(axes[0, 1], 'AIT', center=180)

        The original axes is ``.remove()``-d, so anything already
        drawn on it is lost. ``Axes`` objects without a
        ``SubplotSpec`` (e.g. from ``fig.add_axes(rect)``) are
        rejected — pass a SubplotSpec or position-style int
        instead.
    projection : str
        Projection name. FITS codes ('AIT', 'MOL', 'SIN', 'TAN', 'CAR', etc.)
        or human-readable names ('aitoff', 'mollweide', 'orthographic',
        'gnomonic', 'plate_carree', etc.) — case-insensitive.
        Use list_projections() to see all available options.
    center : float, tuple, or SkyCoord, optional
        Center of the projection. A single float gives the center longitude;
        a ``(lon, lat)`` tuple also sets the center latitude. A scalar
        :class:`~astropy.coordinates.SkyCoord` is also accepted and is
        converted into the frame **being built** (not blindly to ICRS), so a
        galactic SkyCoord centers where you asked on ``frame='galactic'`` and
        converts correctly on an equatorial frame. Defaults to 180
        (lon only). Overridden by ``center_lon``/``center_lat`` if provided.
        All-sky projections honor ``center_lat`` for an oblique aspect (the
        custom non-FITS projections via a spherical rotation), EXCEPT the
        HEALPix / quadcube projections (HPX, XPH, TSC, CSC, QSC), whose
        pole-relative tiling stays locked to the equatorial aspect
        (``center_lat`` is ignored for those). The quadcubes also read
        cleanest at a face-aligned ``center_lon`` of 0/90/180/270. The conics
        (COD/COE/COO/COP), the polyconic (PCO), and the perspective zenithals
        (TAN, AIR, AZP, SZP) are field/regional projections — best shown as a
        zoomed field (``fov_deg``) or graticule-only rather than draped with
        whole-sky data.
    center_lon : float, optional
        Center longitude in degrees. If provided (with optional ``center_lat``),
        takes precedence over ``center``.
    center_lat : float, optional
        Center latitude in degrees. Defaults to 0 if only ``center_lon`` is given.
    frame : str
        Coordinate frame: 'ICRS', 'Galactic', 'Supergalactic', 'Ecliptic',
        'ITRS' (geographic TLON/TLAT), etc. Use ``frame='ITRS'`` for an
        Earth/planet map so a geographic raster texture (from
        ``pseudofits_from_image(..., geo=True)``) drapes onto it without a
        cross-system transform — see :func:`reproject_rgb_map`.
    direction : str, optional
        Longitude orientation. ``'sky'`` (default) puts longitude / RA
        increasing to the *left* (astronomical convention); ``'geographic'``
        puts it increasing to the *right* (cartographic convention). Applies
        uniformly to all-sky, field, and globe views regardless of frame.
        Accepts aliases (``'astro'``, ``'geo'``, ``'earth'``, …).
    lon_units : {'auto', 'hours', 'degrees'}
        Longitude tick units. ``'auto'`` (default) picks by frame +
        orientation: hour-angle for a sky-oriented equatorial frame, degrees
        for geographic / galactic / ecliptic. ``'hours'`` / ``'degrees'``
        force the unit (e.g. an ICRS all-sky map labeled in degrees). Latitude
        is always in degrees. Aliases ``hms`` / ``h`` and ``deg`` / ``d``.
    lon_spacing, lat_spacing : float or 'auto'
        Tick/grid spacing in degrees. ``'auto'`` (default) picks a
        sensible spacing based on the field of view (~8 ticks across
        the visible range, snapped to nice 1/2/5×10ⁿ values from
        sub-arcsec to 90°). Pass an explicit value to override.
    grid : bool
        Draw coordinate grid
    gridcolor : str
        Grid line color (default ``'0.5'`` — medium gray, visible
        on both white and dark backgrounds)
    gridalpha : float
        Grid line transparency (default 0.5)
    gridlw : float, optional
        Grid line width. ``None`` (default) inherits
        ``rcParams['grid.linewidth']``.
    gridls : str, optional
        Grid line style. ``None`` (default) keeps each branch's historical
        look: dotted for the non-FITS all-sky projections and for the
        densified backfill grid, and ``rcParams['grid.linestyle']`` for the
        plain WCSAxes path. Those defaults deliberately still differ — see
        the note at the grid call — so this only makes them reachable.
    outline_color : color, optional
        Color of the projection silhouette drawn for projections whose
        valid region is not the full rectangle (BON, PCO, HPX, quadcube,
        the conics). ``None`` (default) follows
        ``rcParams['axes.edgecolor']``, so it matches the frame and stays
        visible on a dark theme.
    outline_lw : float, optional
        Line width of that silhouette. ``None`` (default) uses 0.7.
    aspect : matplotlib aspect kwarg
        'auto', 'equal', 1, etc. Defaults to 'auto' for all-sky, 1 for globe.
    npix : tuple or int, optional
        Number of pixels (NAXIS1, NAXIS2). If int, used for both axes.
        Defaults chosen per projection type.
    shape : str, optional
        Override the default frame shape for this projection.
        'elliptical', 'rectangular', 'circular', 'sinusoidal'
    lonpole, latpole : float
        FITS LONPOLE/LATPOLE parameters (degrees)
    equinox : float
    obstime : str, astropy Time, datetime, or None
        Observation time for the ``DATE-OBS`` header card. Anything
        :class:`astropy.time.Time` accepts works, including a ``Time`` itself
        — which is what a FITS ``DATE-OBS`` parses into.
    apply_format_defaults : bool
        Auto-apply frame-appropriate tick formatting (decimal degrees
        for Galactic/Supergalactic/Ecliptic, HMS/DMS for equatorial)
    pv2_1, pv2_2 : float, optional
        FITS PV2_1 / PV2_2 parameters for conic (COD/COE/COO/COP) and
        Bonne (BON) projections. ``pv2_1`` sets the standard parallel
        in degrees; ``pv2_2`` sets the spread (half-distance) between
        the two standard parallels for conics (BON ignores it). When
        unset, conic / Bonne projections use ``pv2_1=45``, a sensible
        mid-latitude default. The values are silently ignored for
        every other projection (AIT/MOL/SIN/...).
    return_hdr : bool
        If True, also return the dummy FITS header
    fig : matplotlib Figure, optional
        Figure to add axes to. If None, uses the current figure
        (``plt.gcf()``). This is an axis-builder, not a figure-builder;
        size the figure yourself via ``plt.figure(figsize=...)`` and
        pass it here.
    subplot_kw : dict, optional
        Additional kwargs passed to plt.subplot()
    tick_style : {'auto', 'in_frame', 'boundary', 'native'}, optional
        Where tick labels are drawn.

        - ``'auto'`` (default) — picks ``'in_frame'`` for projections
          whose astropy defaults are buggy or visually disconnected
          (``frame_shape`` in ``{circular, sinusoidal, parabolic,
          robinson, kavrayskiy, eckert_iv, winkel_tripel, mcbryde}``);
          ``'native'`` for everything else.
        - ``'in_frame'`` — labels along the central parallel + central
          meridian, inside the visible region. Works on both all-sky
          frames and zoomed single-field frames (TAN/SIN/ZEA with
          ``fov_deg``/``cdelt``): on a field frame the tick values and
          label precision adapt to the field's extent. (``'auto'`` still
          uses native ticks for zoomed fields; pass ``'in_frame'``
          explicitly to opt in.)
        - ``'boundary'`` — labels on the projection's natural boundary
          curve (one per gridline×spine crossing). Routes through
          :func:`~skyplothelper.coord_overlay.add_overlay_ticks` —
          tangent-rotated and free of the astropy spurious-tick
          / one-sided-tick bugs that affect some frame classes.

          Note: for elliptical projections whose meridians converge at
          the poles (AIT, MOL), explicit ``'boundary'`` will pile lon
          labels at the apex top and bottom. The default ``'auto'``
          routes lon to native horizontal placement at the axes bottom
          for these frames, which is usually what you want. If you
          want lat-boundary + lon-along-a-southern-parallel instead,
          compose manually::

              ax = sph.make_wcs_frame(111, 'AIT', center=0,
                                      tick_style='native')
              sph.add_overlay_ticks(ax, lat_at='lat=-60',
                                    lon_at=None,
                                    suppress_default='lat')

        - ``'native'`` — no overlay; whatever astropy renders by
          default. Useful as an escape hatch or when chaining the
          ``add_overlay_ticks`` / ``add_curved_lon_ticks`` helpers
          yourself.

          On a **curved / circular-globe** frame (SIN/ZEA, the
          pseudocylindricals) ``'native'`` inherits astropy's
          ``CircularFrame`` limitations: tick *marks* sit on the straight
          central-cross spines (a flat row across the globe center) or
          render in only one quadrant of the limb (astropy returns a NaN
          tangent at most curved-spine positions), and the spines are named
          ``'c'`` / ``'h'`` / ``'v'`` — so ``set_ticklabel_position('b')``
          (a rectangular-frame name) hides everything. The default
          ``'auto'`` deliberately routes these frames to the in-frame
          overlay, which computes its own tangents and draws clean
          curved marks + labels — prefer it (or ``add_overlay_ticks``)
          over ``'native'`` on globes.
    tick_rotation : {'tangent', 'tangent_upright', 'horizontal'}, float, or callable, optional
        Rotation of each tick label, forwarded to
        :meth:`~skyplothelper.coord_overlay.CoordinateOverlay.render_labels`
        as ``rotate=``. Default ``'tangent'`` (aliased ``'tangent_noflip'``)
        follows the gridline tangent continuously — no flip between adjacent
        labels, with a per-placement-group branch keeping labels upright for
        the current view, leaning past vertical only where a group genuinely
        sweeps through vertical. ``'tangent_upright'`` instead clamps each
        label upright, flipping 180° where the tangent crosses ±90°. Ignored
        when ``tick_style='native'``.
    edge_ticks : {'auto', 'all'}, optional
        Which frame spines carry tick marks on flat single-field frames
        (TAN/SIN/ZEA and cube-face zooms — *not* all-sky or globe frames).
        Default ``'auto'``.

        - ``'auto'`` — pin longitude ticks to the bottom/top spines and
          latitude ticks to the left/right spines. On a single field the
          meridians converge toward the pole, so they fan across the panel
          and clip the side spines near the top corners (and the parallels
          dip across the bottom); astropy's default heuristic then scatters
          a few stray RA ticks onto the left/right edges and a Dec tick onto
          the bottom, which can look like an error. Pinning removes those
          strays while keeping every real tick, because the fields built here
          are north-up and axis-aligned.
        - ``'all'`` — leave astropy's automatic per-spine assignment (ticks
          wherever a gridline meets any spine). Use this for a manually
          rolled / rotated WCS, where a meridian may legitimately exit a
          side spine and ``'auto'`` would suppress a real tick.

        All-sky, globe, and pseudocylindrical frames ignore this setting
        (they have their own boundary/in-frame tick handling). For full
        manual control, call ``ax.coords[i].set_ticks_position(...)`` after
        construction.
    auto_fontsize : bool, optional
        Auto-shrink tick label fontsize to fit the available axes
        width. Default ``True``. The chosen size is clipped to
        ``[6pt, rcParams['xtick.labelsize']]`` so plots at typical
        figure sizes are visually unchanged — the auto-scale only
        kicks in to *shrink* labels on small / multi-panel layouts.
        Pass ``auto_fontsize=False`` to opt out and use astropy's
        defaults. Explicit overrides via
        ``ax.coords[i].set_ticklabel(fontsize=...)`` after frame
        construction always win.
    **kwargs
        Additional keyword arguments passed to plt.subplot()

    Returns
    -------
    ax : WCSAxes
        The constructed axes. If ``return_hdr=True``, returns
        ``(ax, hdr)`` instead.

    Examples
    --------
    >>> ax = sph.make_wcs_frame(111, 'aitoff', center=180, frame='ICRS')
    >>> ax = sph.make_wcs_frame(111, 'mollweide', center=0, frame='Galactic')
    >>> ax = sph.make_wcs_frame(111, 'SIN', center=(180, 45))  # globe
    >>> fig = plt.figure(figsize=(6, 6))
    >>> ax = sph.make_wcs_frame(projection='SIN', center=(180, 30), fig=fig)
    >>> ax = sph.make_wcs_frame(111, 'TAN', center=(83.6, 22.0))  # field view
    >>> ax = sph.make_wcs_frame(111, 'sinusoidal', center=0, frame='Galactic')
    >>> sph.list_projections()  # see all available projections

    Notes
    -----
    **LONPOLE handling:** For zenithal projections (TAN, SIN, ARC, etc.),
    the FITS standard default LONPOLE is 180° (not 0°). The ``lonpole``
    parameter default of 0° is correct for all-sky pseudocylindrical
    projections but would flip both RA and Dec directions for field views.
    This function automatically omits LONPOLE/LATPOLE from the header
    for zenithal projections when left at defaults, letting wcslib use
    the correct FITS standard values.

    **Non-FITS projections** (Robinson, Kavrayskiy, Eckert IV, Winkel
    Tripel, McBryde) use ``WCSAxes`` with a custom ``CurvedTransform``
    and are positioned via matplotlib's ``SubplotSpec`` system for
    proper layout integration. They support ``fig.subplots_adjust()``
    and ``bbox_inches='tight'``.

    **tight_layout:** Generally incompatible with WCSAxes (both FITS
    and non-FITS). Use ``fig.subplots_adjust()`` or manual positioning.
    """
    # A SkyCoord ``center`` is normalized to a (lon, lat) degree pair up front,
    # converted into the frame being built rather than blindly to ICRS, so the
    # numeric paths further down need no SkyCoord branches of their own.
    if hasattr(center, 'transform_to'):
        from .geometry._parsing import _coords_to_frame_deg
        center = _coords_to_frame_deg(center, frame)

    # ``figsize=`` is a figure-level setting — reject it with a
    # pointer to the right place. This is an axis-builder; the figure
    # comes from ``plt.figure(figsize=...)`` (or ``plt.gcf()``) and is
    # passed via ``fig=``.
    if 'figsize' in kwargs:
        raise TypeError(
            "make_wcs_frame() does not accept ``figsize=`` — it is "
            "an axis-builder, not a figure-builder. Create the figure "
            "first via ``plt.figure(figsize=...)`` and pass it as "
            "``fig=fig``."
        )

    if edge_ticks not in ('auto', 'all'):
        raise ValueError(
            f"make_wcs_frame(): edge_ticks must be 'auto' or 'all', "
            f"got {edge_ticks!r}."
        )

    # ``subplotnumber`` accepts:
    #   * int (e.g. 111) or tuple (e.g. (2, 3, 1))
    #   * a ``SubplotSpec`` (gridspec output)
    #   * a pre-existing matplotlib ``Axes`` object — used so the
    #     user can swap an axes from their own subplot/gridspec
    #     workflow for a WCSAxes at the same grid position. The
    #     existing axes is ``.remove()``-d and we extract its
    #     ``SubplotSpec`` to position the new WCSAxes identically.
    #     Loses anything already drawn on the original axes
    #     (warn-worthy but not failure-worthy).
    from matplotlib.axes import Axes
    if isinstance(subplotnumber, Axes):
        existing_ax = subplotnumber
        existing_fig = existing_ax.figure
        if fig is None:
            fig = existing_fig
        elif fig is not existing_fig:
            raise ValueError(
                "make_wcs_frame() received both an explicit ``fig=`` "
                "and a ``subplotnumber=`` Axes from a different "
                "figure — pass at most one figure context."
            )
        spec = existing_ax.get_subplotspec()
        if spec is None:
            raise ValueError(
                "Pre-existing Axes passed via ``subplotnumber=`` has "
                "no SubplotSpec (probably created via "
                "``fig.add_axes(rect)``). Pass a SubplotSpec, an int "
                "subplot number, or an Axes that lives in a subplot "
                "grid (``plt.subplots`` / ``GridSpec``)."
            )
        existing_ax.remove()
        subplotnumber = spec

    # ``fov_deg`` is a convenience alternative to ``cdelt`` for
    # zoomed / field-style frames: the user specifies the desired
    # field-of-view in degrees and ``cdelt`` is computed from
    # ``fov_deg / max(npix)``. Mutually exclusive with explicit
    # ``cdelt`` — raise rather than silently choose one.
    if fov_deg is not None:
        if cdelt is not None:
            raise TypeError(
                "make_wcs_frame() got both ``cdelt`` and ``fov_deg`` "
                "— pass only one (``fov_deg`` is the convenience "
                "alternative; pass ``cdelt`` for explicit control)."
            )
        # Use the wider axis to size the FOV so the requested FOV
        # is the visible extent in that dimension. ``npix`` may
        # still be ``None`` here — fall back to a sensible default
        # for offset-style frames so the math is well-defined.
        if npix is None:
            npix = (500, 500)
        npix_tuple = (int(npix), int(npix)) if isinstance(
            npix, (int, float)) else tuple(int(x) for x in npix)
        cdelt = float(fov_deg) / max(npix_tuple)

    # Resolve center from center_lon/center_lat or center parameter
    if center_lon is not None:
        if center_lat is not None:
            center = (center_lon, center_lat)
        else:
            center = center_lon
    elif center is None:
        center = 180

    # Pure axis-builder: the user is expected to create the figure
    # via ``plt.figure(...)`` (or rely on ``plt.gcf()``); this
    # function only adds the WCSAxes inside that figure. No
    # ``figsize=`` shortcut.

    # Resolve projection name
    proj_key, proj_info = _resolve_projection(projection)
    fits_code = proj_info.fits_code

    if fits_code is None:
        # Non-FITS projection — check if we have a CurvedTransform implementation
        _NON_FITS_TRANSFORMS = {}
        if _HAS_CURVEDTRANSFORM:
            _NON_FITS_TRANSFORMS['robinson'] = RobinsonTransform
            _NON_FITS_TRANSFORMS['kavrayskiy'] = KavrayskiyTransform
            _NON_FITS_TRANSFORMS['eckert_iv'] = Eckert4Transform
            _NON_FITS_TRANSFORMS['winkel_tripel'] = WinkelTripelTransform
            _NON_FITS_TRANSFORMS['mcbryde'] = McBrydeTransform

        if proj_key not in _NON_FITS_TRANSFORMS:
            raise NotImplementedError(
                f"Projection '{projection}' ({proj_info.description}) is not yet "
                f"implemented. Non-FITS projections require custom transforms — "
                f"planned for a future version."
            )

        # Build non-FITS projection using CurvedTransform + coord_meta
        if isinstance(center, (list, tuple)):
            center_lon = float(center[0])
            center_lat = float(center[1]) if len(center) > 1 else 0.0
        else:
            center_lon = float(center)
            center_lat = 0.0

        transform_cls = _NON_FITS_TRANSFORMS[proj_key]
        # WCSAxes transform parameter expects pixel→world direction.
        # Our "pixel" space IS the projected coordinate space, so we need
        # the inverse transform: projected→world.
        if center_lat != 0.0:
            # Oblique aspect: a spherical rotation centers BOTH lon and lat,
            # then the equatorial projection runs on the rotated graticule (so
            # the projection itself is built with center_lon=0). The custom
            # CurvedTransforms have no oblique math of their own; this is how
            # they match the FITS path's CRVAL2 latitude shift.
            world_to_proj: Any = (
                ObliqueAspectTransform(center_lon=center_lon, center_lat=center_lat)
                + transform_cls(center_lon=0.0))
        else:
            world_to_proj = transform_cls(center_lon=center_lon)
        # Longitude orientation: the custom CurvedTransforms produce east-RIGHT
        # projected x (geographic). For the package default 'sky'/astro
        # (east-left) we flip the projected x with an affine scale — the same
        # net effect the FITS path gets from a negative CDELT1, but applied in
        # projected space so the symmetric xlim and the frame-boundary code stay
        # unchanged. ('geographic' leaves it east-right.)
        from .projections.project import resolve_direction
        if resolve_direction(direction) == 'sky':
            from matplotlib.transforms import Affine2D
            world_to_proj_oriented: Any = world_to_proj + Affine2D().scale(-1.0, 1.0)
        else:
            world_to_proj_oriented = world_to_proj
        proj_to_world = world_to_proj_oriented.inverted()

        # Frame-dependent coordinate metadata
        ctt1, ctt2, rs = _resolve_ctype(frame)
        is_equatorial = ctt1 == 'RA--'

        coord_meta = {
            'type': ('longitude', 'latitude'),
            'wrap': (360 * u.deg, None),
            'unit': (u.deg, u.deg),
        }
        if is_equatorial:
            coord_meta['name'] = ('Right Ascension', 'Declination')
        elif 'GLON' in ctt1:
            coord_meta['name'] = ('Galactic Longitude', 'Galactic Latitude')
        else:
            coord_meta['name'] = ('Longitude', 'Latitude')

        # Determine frame shape
        frame_shape = shape or proj_info.frame_shape
        frame_class = _FRAME_CLASSES.get(frame_shape)

        if fig is None:
            fig = plt.gcf()

        # Parse subplot number into SubplotSpec for proper layout integration.
        # This ensures non-FITS axes participate in tight_layout and are
        # centered within their subplot cell, matching plt.subplot() behavior.
        from matplotlib.gridspec import GridSpec, SubplotSpec

        if isinstance(subplotnumber, SubplotSpec):
            ss = subplotnumber
        elif isinstance(subplotnumber, int) and subplotnumber >= 100:
            nrows = subplotnumber // 100
            ncols = (subplotnumber % 100) // 10
            idx = (subplotnumber % 10) - 1  # 0-based
            gs = GridSpec(nrows, ncols, figure=fig)
            ss = gs[idx // ncols, idx % ncols]
        elif isinstance(subplotnumber, tuple) and len(subplotnumber) == 3:
            nrows, ncols, idx = subplotnumber
            gs = GridSpec(nrows, ncols, figure=fig)
            ss = gs[(idx - 1) // ncols, (idx - 1) % ncols]
        else:
            gs = GridSpec(1, 1, figure=fig)
            ss = gs[0, 0]

        rect = ss.get_position(fig).bounds

        ax_kwargs = {}
        if frame_class is not None:
            ax_kwargs['frame_class'] = frame_class

        ax = WCSAxes(fig, rect, transform=proj_to_world,
                     coord_meta=coord_meta, **ax_kwargs)
        ax.set_subplotspec(ss)  # integrate with matplotlib layout engine
        fig.add_axes(ax)

        # Suppress astropy's auto axis labels — format_ticklabels will set
        # them as needed. Astropy regenerates
        # the default `pos.eq.ra` / `pos.eq.dec` UCD labels from coord_meta
        # at draw time when the explicit label is the empty string, so
        # use a single space to lock the slot empty without retriggering
        # the default.
        ax.coords[0].set_axislabel(' ')
        ax.coords[1].set_axislabel(' ')

        # Store metadata for helper functions that normally read from WCS
        ax._sph_frame = frame.lower() if frame.lower() in (
            'galactic', 'supergalactic', 'ecliptic') else 'icrs'
        ax._sph_center_lon = center_lon
        ax._sph_center_lat = center_lat
        ax._sph_proj_key = proj_key

        # Set limits based on projection extents — compute from forward transform
        # x extent from equator at ±180°, y extent from center meridian at ±90°
        _fwd_funcs = {
            'robinson': _robinson_forward,
            'kavrayskiy': _kavrayskiy_forward,
            'eckert_iv': _eckert4_forward,
            'winkel_tripel': _winkel_forward,
            'mcbryde': _mcbryde_forward,
        }
        if proj_key in _fwd_funcs:
            fwd = _fwd_funcs[proj_key]
            x_raw, _ = fwd(180., 0.)
            _, y_raw = fwd(0., 90.)
            x_ext = abs(float(x_raw))
            y_ext = abs(float(y_raw))
            ax.set_xlim(-x_ext, x_ext)
            ax.set_ylim(-y_ext, y_ext)

        if aspect != 'auto':
            ax.set_aspect(aspect)

        # Grid. This branch's historical look is dotted at the rcParams
        # width; the WCSAxes branch below forces neither and inherits both.
        # gridlw / gridls only make those reachable -- deliberately NOT
        # unified, since agreeing on one look would move existing renders on
        # whichever branch lost. That is a styling decision, not a cleanup.
        if grid:
            ax.grid(color=gridcolor, alpha=gridalpha,
                    ls=(':' if gridls is None else gridls),
                    **({} if gridlw is None else {'lw': gridlw}))

        # Resolve 'auto' spacings — non-FITS projections are all-sky.
        lon_spacing_deg: float = (_auto_tick_spacing(360.0)
                                  if lon_spacing == 'auto'
                                  else float(lon_spacing))
        lat_spacing_deg: float = (_auto_tick_spacing(180.0)
                                  if lat_spacing == 'auto'
                                  else float(lat_spacing))
        # Guard against a degenerate explicit spacing freezing the draw (these
        # span the full 360°/180° domain — lon via spacing=, lat via arange).
        lon_spacing_deg = _clamp_spacing(lon_spacing_deg, 360.0, 'longitude')
        lat_spacing_deg = _clamp_spacing(lat_spacing_deg, 180.0, 'latitude')
        ax.coords[0].set_ticks(spacing=lon_spacing_deg * u.deg)
        # For non-FITS pseudo-cylindricals, the polar parallel (lat=±90)
        # is a degenerate flat line in projection space, not a point — a
        # single tick at the boundary is geometrically meaningless and
        # astropy ends up placing extras (one per spine intersection at
        # lat=±90, plus one from the central-meridian 'v' spine). Use
        # explicit lat tick values that exclude ±90.
        lat_ticks_deg = np.arange(
            -90 + lat_spacing_deg, 90 - lat_spacing_deg / 2, lat_spacing_deg)
        ax.coords[1].set_ticks(values=lat_ticks_deg * u.deg)
        # Curved all-sky spine: drop the base-style minor ticks (they clutter).
        _suppress_curved_minor_ticks(ax)

        # Auto-apply formatting
        lon_label_fmt = None
        if apply_format_defaults:
            format_ticklabels(ax, style='publication')
            # Longitude units: explicit lon_units wins; 'auto' defers to the
            # orientation (geographic → degrees, since hours are an RA / sky
            # convention and an east-right map reads in degrees). Set the
            # native unit and the overlay label format (in-frame styles draw
            # their own labels, which don't read the native format unit).
            from .projections.project import resolve_direction
            _geo = resolve_direction(direction) == 'geographic'
            _lon_u = resolve_lon_units(lon_units)
            _apply_lon_units(ax, _lon_u, _geo)
            lon_label_fmt = _overlay_lon_fmt(_lon_u, _geo)

        # Pad title position for projections whose topmost tick labels
        # would otherwise collide with the default title location.
        if proj_key in _POLE_TOP_NON_FITS_FRAME_SHAPES or \
           frame_shape in _POLE_TOP_NON_FITS_FRAME_SHAPES:
            _pad_title_for_pole_top_projection(ax)

        _apply_tick_style(ax, frame_shape, tick_style, tick_rotation,
                          lon_label_fmt=lon_label_fmt)

        _attach_sky_methods(ax)
        if return_hdr:
            return ax, None
        return ax

    # Determine frame shape (user override or from registry)
    frame_shape = shape or proj_info.frame_shape
    frame_class = _FRAME_CLASSES.get(frame_shape)

    # Parse center coordinates
    if isinstance(center, (list, tuple)):
        center_lon, center_lat = center
    else:
        center_lon = float(center)
        center_lat = 0.

    # Determine if this is a globe/field view vs all-sky. An
    # ``allsky=True`` projection (e.g. CSC / TSC / QSC / XPH) can
    # also be rendered as a zoomed field view by passing an
    # explicit ``cdelt`` or ``fov_deg`` — useful for projections
    # whose all-sky presentation is awkward (cube unfolds, polar
    # butterfly) and which are more naturally shown as a
    # field-view zoom onto one face / one pole. The explicit
    # parameter forces the field-view CDELT branch.
    is_globe = (proj_info.proj_class == 'zenithal' and
                frame_shape in ('circular', 'rectangular'))
    is_field_override = (cdelt is not None and proj_info.allsky
                         and not is_globe)
    is_allsky = proj_info.allsky and not is_globe and not is_field_override

    # Set default npix
    if npix is None:
        if is_globe:
            npix = (360, 360)
        elif is_allsky:
            npix = (360, 180)
        else:
            npix = (360, 360)
    elif isinstance(npix, (int, float)):
        npix = (int(npix), int(npix))

    # Set default aspect
    if aspect == 'auto' and is_globe:
        aspect = 1

    # Build the appropriate dummy header
    ctt1, ctt2, rs = _resolve_ctype(frame)
    # Longitude orientation is set by ``direction``, not the frame: 'sky'
    # (default) puts lon/RA increasing leftward (negative CDELT1) for every
    # frame; 'geographic' puts it increasing rightward. This is uniform
    # across all-sky / field / globe views.
    from .projections.project import resolve_direction
    lonsign = -1.0 if resolve_direction(direction) == 'sky' else 1.0

    if is_allsky:
        # CDELT sizing: AIT/MOL use elliptical 2sqrt(2)/sqrt(2) constants
        # (these constants give x_max=2sqrt(2)·180/π ≈ 162° and
        # y_max=sqrt(2)·180/π ≈ 81° in the projection plane, matching the
        # actual extents of those projections). Other projection classes
        # have different boundary extents — most notably CEA, whose
        # y at lat=90 is only 1 rad ≈ 57.3°, so the AIT-sized frame
        # leaves blank top/bottom bands. Dispatch on fits_code so each
        # projection's CDELT scales its own boundary into ``npix``.
        # Per-projection projection-plane extent at the all-sky
        # boundary. For each FITS projection class, ``x_max_deg`` is
        # the projection x at (lon=±180°, lat=0°) and ``y_max_deg``
        # is the projection y at (lon=0°, lat=±90°), in degrees.
        # CDELT then scales these into ``npix`` so the visible frame
        # exactly contains the projection's natural bounding box.
        # y_min_deg stays None for the usual vertically-symmetric envelopes
        # (the header then uses -y_max_deg); projections whose envelope is
        # lopsided in y (BON) set it explicitly to fit the frame snugly.
        # x_min_deg is the analog for horizontally-offset envelopes (the
        # quadcube cross, whose net sits left-of-center); None → -x_max_deg.
        y_min_deg: float | None = None
        x_min_deg: float | None = None
        if fits_code in ('AIT', 'MOL'):
            # Elliptical pseudocylindricals (Hammer-Aitoff,
            # Mollweide): natural ellipse with x_max = 2sqrt(2)·180/π,
            # y_max = sqrt(2)·180/π.
            x_max_deg = 2 * np.sqrt(2) * 180. / np.pi
            y_max_deg = np.sqrt(2) * 180. / np.pi
        elif proj_info.proj_class == 'cylindrical':
            # CAR/CEA/MER/CYP: x_max = 180° always; y_max varies:
            #  * CAR (Plate Carrée): y = lat → y_max = 90°.
            #  * CEA (cyl. equal-area): y = sin(lat)·180/π →
            #    y_max = 180/π ≈ 57.3°.
            #  * MER (Mercator): clamp at lat=±85° (Web Mercator
            #    convention; Google Maps / OSM standard).
            #  * CYP (cyl. perspective, default WCS PV1=0): diverges
            #    as tan(lat); ±85° here would compress the equator
            #    to a tiny strip, so match CAR's y_max=90° (visible
            #    lat clamps to ~±58°).
            x_max_deg = 180.0
            if fits_code == 'CAR':
                y_max_deg = 90.0
            elif fits_code == 'CEA':
                y_max_deg = 180. / np.pi
            elif fits_code == 'MER':
                lat_clamp = np.radians(85.)
                y_max_deg = 180. / np.pi * np.log(
                    np.tan(np.pi / 4. + lat_clamp / 2.))
            else:  # CYP and any future cylindrical
                y_max_deg = 90.0
        elif fits_code == 'XPH':
            # HEALPix polar (butterfly) — actual bounding box from
            # dense (lon, lat) sampling: x ∈ ±127°, y ∈ ±127°.
            x_max_deg = 128.0
            y_max_deg = 128.0
        elif fits_code == 'BON':
            # Bonne pseudoconic: heart-cardioid shape inside a
            # rectangular frame (kapteyn-style). The cardioid is
            # vertically lopsided — the south cusp sits at y_proj ≈ -91°
            # while the "ear" tops reach ≈ +141° (for the default
            # standard parallel PV2_1=45). A symmetric ±y_max range would
            # leave a large dead band below the cusp, so measure the true
            # envelope and fit y asymmetrically (cusp → bottom of frame).
            # The extent is independent of center_lat (oblique just rotates
            # the graticule inside the same envelope) but depends on PV2_1.
            _bon_pv = 45.0 if pv2_1 is None else float(pv2_1)
            x_lo, x_hi, y_lo, y_hi = _measure_allsky_proj_extent(
                f'{ctt1}-BON', f'{ctt2}-BON', center_lon, center_lat, _bon_pv)
            x_max_deg = max(abs(x_lo), abs(x_hi)) + 3.0  # symmetric in x
            y_max_deg = y_hi + 3.0
            y_min_deg = y_lo - 3.0
        elif fits_code == 'PCO':
            # Polyconic — the egg shape extends past ±90° in y at
            # intermediate latitudes (parallels are circle-arcs whose
            # centers sit at y_proj = lat + cot(lat) above the
            # equator, so high-lat arcs reach far above y=lat).
            # Probe gives y_extent ±138°; round to ±140°. x_extent
            # is ±180° (the egg's widest point at the equator).
            x_max_deg = 180.0
            y_max_deg = 140.0
        elif fits_code in ('CSC', 'TSC', 'QSC'):
            # Quadcube unfolding into a cross. The net is OFFSET in x: the
            # polar-face column sits over a face that is right-of-center of
            # the equatorial belt, so the net spans (in projection-plane deg)
            # roughly x ∈ [-44, +315], NOT a symmetric ±180. A symmetric
            # frame (x centered on the reference) therefore leaves a large
            # empty arm on one side (the kapteyn figures avoid this by
            # framing the cross tightly). Measure the true net bbox and frame
            # it snugly so the cross fills the panel with no empty arm. The
            # net is center_lon-independent in the projection plane, so the
            # measurement is taken once at the actual center.
            x_lo, x_hi, y_lo, y_hi = _measure_allsky_proj_extent(
                f'{ctt1}-{fits_code}', f'{ctt2}-{fits_code}',
                center_lon, 0.0)
            # Small margin only: the cube net edges ARE the visible face edges
            # (unlike the cardioid/wedge, which sit inside their frame), so the
            # data reaches the net bbox exactly. A large margin would leave a
            # visible empty border (a ~3 px gap around the cross); ~0.5° keeps
            # the boundary outline off the axes spine without a visible gap.
            x_min_deg = x_lo - 0.5
            x_max_deg = x_hi + 0.5
            y_max_deg = y_hi + 0.5
            y_min_deg = y_lo - 0.5
        elif fits_code in ('COD', 'COE', 'COO', 'COP'):
            # Conic (kapteyn all-sky recipe). The reference point sits ON the
            # standard parallel (CRVAL2 = PV2_1, set below), so the cone apex
            # is centered at the near pole and the wedge opens toward the far
            # pole — a clean wedge instead of the lopsided bbox-filling smear
            # that an equatorial reference produces. COD/COE stay finite over
            # the whole sphere; COO/COP diverge toward the far pole, so their
            # visible latitude range is clipped (conic_visible_lat_range). The
            # envelope is measured at CRVAL2 = PV2_1 to match the frame.
            from .projections import _boundaries
            _conic_pv = 45.0 if pv2_1 is None else float(pv2_1)
            _lat_lo, _lat_hi = _boundaries.conic_visible_lat_range(
                fits_code, _conic_pv)
            x_lo, x_hi, y_lo, y_hi = _measure_allsky_proj_extent(
                f'{ctt1}-{fits_code}', f'{ctt2}-{fits_code}', center_lon,
                _conic_pv, _conic_pv, pv2_2, lat_range=(_lat_lo, _lat_hi))
            x_max_deg = max(abs(x_lo), abs(x_hi)) + 3.0  # symmetric in x
            y_max_deg = y_hi + 3.0
            y_min_deg = y_lo - 3.0
        else:
            # Remaining all-sky FITS projections — sinusoidal,
            # parabolic, HEALPix-rect — have a natural rectangular
            # bounding box at lon=±180°, lat=±90°.
            #   PAR (parabolic), SFL (sinusoidal), HPX (HEALPix).
            x_max_deg = 180.0
            y_max_deg = 90.0
        # All-sky CDELT1 sign follows ``direction`` (via ``lonsign``):
        # negative = lon increases left ('sky', default), positive = right
        # ('geographic'). Same orientation rule as the field / globe path.
        # x range is [x_min_deg, x_max_deg]; symmetric (±x_max) unless a
        # horizontally-offset envelope (the quadcube cross) set x_min_deg
        # explicitly. CRPIX1 is shifted so the projection origin (CRVAL1)
        # sits at intermediate x=0 while the pixel grid spans the requested
        # x range — reduces to the centered ``npix/2 + 0.5`` when symmetric.
        if x_min_deg is None:
            x_min_deg = -x_max_deg
        x_center_deg = (x_min_deg + x_max_deg) / 2.0
        cd1_allsky = lonsign * (x_max_deg - x_min_deg) / npix[0]
        crpix1_allsky = npix[0] / 2 + 0.5 - x_center_deg / cd1_allsky
        # y range is [y_min_deg, y_max_deg]; symmetric (±y_max) unless a
        # lopsided-envelope projection (BON) set y_min_deg explicitly. CRPIX2
        # is placed so the projection origin (CRVAL2) sits at intermediate
        # y=0 while the pixel grid spans the requested y range.
        if y_min_deg is None:
            y_min_deg = -y_max_deg
        cd2_allsky = (y_max_deg - y_min_deg) / npix[1]
        crpix2_allsky = 0.5 - y_min_deg / cd2_allsky
        # HEALPix / quadcube tilings are pole-relative; an oblique aspect
        # would misalign the tiles with the data, so they stay equatorial.
        # Conics center the reference point on the standard parallel
        # (CRVAL2 = PV2_1) so the apex sits at the pole and the wedge frames
        # cleanly; their aspect is fixed by PV2_1, so center_lat is ignored
        # too (like the pole-locked tilings above).
        if fits_code in ('HPX', 'XPH', 'TSC', 'CSC', 'QSC'):
            allsky_crval2 = 0.0
        elif fits_code in _CONIC_FITS_CODES:
            allsky_crval2 = _DEFAULT_PV2_1 if pv2_1 is None else float(pv2_1)
        else:
            allsky_crval2 = center_lat
        hdr = pyfits.Header({
            'NAXIS': 2, 'NAXIS1': npix[0], 'NAXIS2': npix[1],
            'CRPIX1': crpix1_allsky, 'CRPIX2': crpix2_allsky,
            # All-sky frames honor center_lat for an oblique aspect (the data
            # + graticule rotate; the projection's bounding box is unchanged
            # because the whole sphere always maps into the same envelope).
            # center_lat defaults to 0 (the equatorial aspect), so this is
            # additive — passing only a center longitude is unchanged.
            # Exception: the HEALPix / quadcube projections are constructed
            # relative to the pole (their tiling only aligns with the data at
            # the equatorial aspect), so they stay locked to CRVAL2=0.
            'CRVAL1': center_lon, 'CRVAL2': allsky_crval2,
            'CDELT1': cd1_allsky,
            'CDELT2': cd2_allsky,
            'CUNIT1': 'deg', 'CUNIT2': 'deg',
            'CTYPE1': f'{ctt1}-{fits_code}',
            'CTYPE2': f'{ctt2}-{fits_code}',
            'RADESYS': rs,
        })
    else:
        # Globe / field view header.
        # On a circular-limb globe (SIN/orthographic), ``fov_deg`` is the
        # angular DIAMETER of the visible cap, mapped through the orthographic
        # relation r = sin θ so the cap exactly fills the circular frame.
        # Clamp to the physical hemisphere (180° diameter = the limb); beyond
        # that sin θ turns over and the globe can't show more. ``None`` →
        # full hemisphere. An explicit ``cdelt`` (the low-level escape hatch)
        # is honored literally. Flat field projections (TAN, ...) are not
        # circular globes and keep the plain fov_deg → cdelt path below.
        is_circular_globe = is_globe and frame_shape == 'circular'
        if is_circular_globe and fov_deg is not None:
            globe_fov = float(fov_deg)
            if globe_fov > 180.0:
                warnings.warn(
                    f"make_wcs_frame: fov_deg={globe_fov:g}° exceeds the "
                    f"{fits_code} globe's physical hemisphere; clamping to "
                    "180° (the full visible hemisphere).",
                    UserWarning, stacklevel=2)
                globe_fov = 180.0
            rho = np.radians(max(globe_fov, 1e-6) / 2.0)  # angular cap radius
            half_deg = np.sin(rho) * (180. / np.pi)       # SIN plane radius
            cd1 = 2 * half_deg / npix[0] * lonsign
            cd2 = 2 * half_deg / npix[1]
        elif cdelt is not None:
            cd1 = -abs(cdelt) if lonsign < 0 else abs(cdelt)
            cd2 = abs(cdelt)
        elif is_globe:
            # No fov_deg/cdelt → full hemisphere. For SIN the hemisphere
            # radius in the projection plane is exactly 1 radian = 180/pi
            # degrees; scale so it fills the pixel grid (limb = frame circle).
            hemisphere_radius = 180. / np.pi  # ~57.296 degrees
            cd1 = 2 * hemisphere_radius / npix[0] * lonsign
            cd2 = 2 * hemisphere_radius / npix[1]
        else:
            cd1 = 2 / np.pi * lonsign
            cd2 = 2 / np.pi
        hdr = pyfits.Header({
            'NAXIS': 2, 'NAXIS1': npix[0], 'NAXIS2': npix[1],
            'CRPIX1': npix[0] / 2 + 0.5, 'CRPIX2': npix[1] / 2 + 0.5,
            'CRVAL1': center_lon, 'CRVAL2': center_lat,
            'CDELT1': cd1, 'CDELT2': cd2,
            'CUNIT1': 'deg', 'CUNIT2': 'deg',
            'CTYPE1': f'{ctt1}-{fits_code}',
            'CTYPE2': f'{ctt2}-{fits_code}',
            'RADESYS': rs,
        })
        # LONPOLE/LATPOLE: for zenithal projections (TAN, SIN, ARC, etc.),
        # the FITS standard default is LONPOLE=180, LATPOLE=90. Only include
        # these cards when the user has explicitly set non-default values.
        # For non-zenithal field views, include them as given.
        is_zenithal = proj_info.proj_class == 'zenithal'
        if is_zenithal:
            # Omit if user left at function defaults (0, 0) — let wcslib
            # use the correct FITS standard defaults (180°, 90°)
            if lonpole != 0.:
                hdr['LONPOLE'] = lonpole
            if latpole != 0.:
                hdr['LATPOLE'] = latpole
        else:
            hdr['LONPOLE'] = lonpole
            hdr['LATPOLE'] = latpole
        if equinox is not None:
            hdr['EQUINOX'] = equinox

    if obstime is not None:
        # .utc is load-bearing: FITS defines DATE-OBS as UTC, so a TT/TDB Time
        # has to be converted rather than formatted as-is (~69 s for TT).
        from ._timeinput import to_time
        hdr['DATE-OBS'] = to_time(obstime, _caller='make_wcs_frame').utc.isot

    # PV parameters for conic / pseudoconic projections. CRVAL2 is left alone
    # here: the all-sky branch above already set it to the standard parallel,
    # and a field-view conic keeps the user's requested center_lat.
    _apply_pv_cards(hdr, fits_code, pv2_1, pv2_2)

    # Create WCS and axis
    wcs = WCS(hdr)

    subplot_kwargs = {}
    if subplot_kw is not None:
        subplot_kwargs.update(subplot_kw)
    subplot_kwargs.update(kwargs)

    if frame_class is not None:
        subplot_kwargs['frame_class'] = frame_class

    if fig is not None:
        if isinstance(subplotnumber, tuple):
            ax = fig.add_subplot(*subplotnumber, projection=wcs, **subplot_kwargs)
        else:
            ax = fig.add_subplot(subplotnumber, projection=wcs, **subplot_kwargs)
    else:
        if isinstance(subplotnumber, tuple):
            ax = plt.subplot(*subplotnumber, projection=wcs, **subplot_kwargs)
        else:
            ax = plt.subplot(subplotnumber, projection=wcs, **subplot_kwargs)

    ax.set_xlim(-0.5, npix[0] - 0.5)
    ax.set_ylim(-0.5, npix[1] - 0.5)

    # Record whether this is the all-sky frame (vs a zoomed field/globe view).
    # The conic wedge boundary assumes the all-sky standard-parallel setup
    # (CRVAL2 = PV2_1); it must NOT be applied to a field-view conic (a
    # fov_deg / cdelt zoom centered elsewhere), or the clip would blank the
    # data. _projection_boundary reads this to gate the conic boundary.
    ax._sph_is_allsky = bool(is_allsky)
    # A circular-limb globe (off-pole zenithal hemisphere). Its graticule spans
    # the full hemisphere up to the limb, so the field-extent restriction used
    # for flat rectangular field frames (which shrinks gridlines/tick values to
    # the visible patch) must NOT apply here — it would stop the gridlines short
    # of the limb. The overlay machinery reads this flag.
    ax._sph_is_globe = bool(is_globe and frame_shape == 'circular')

    if aspect != 'auto':
        ax.set_aspect(aspect)

    # Suppress astropy's auto axis labels. Without this, rectangular-frame
    # projections (SFL, PAR, CAR, SIN globe, ...) pick up the WCS CTYPE UCD
    # codes like 'pos.eq.ra'/'pos.eq.dec' as default labels. Elliptical
    # frames (AIT, MOL) don't suffer this because EllipticalFrame suppresses
    # them internally, but we clear both for consistency. format_ticklabels()
    # will set appropriate labels later when apply_format_defaults is True.
    # Use a single space rather than the empty
    # string — astropy regenerates the UCD-derived default at draw time
    # when the explicit label is `''`, so the explicit-but-blank space
    # locks the slot empty across redraws.
    ax.coords[0].set_axislabel(' ')
    ax.coords[1].set_axislabel(' ')

    # Resolve 'auto' tick spacings from the actual FOV (CDELT × NPIX).
    # Without this, sub-degree TAN fields end up with zero ticks
    # within the visible range and astropy WCSAxes hides the axis
    # entirely (visible_axes=[]).
    lon_fov_deg = abs(hdr.get('CDELT1', 1.0)) * npix[0]
    lat_fov_deg = abs(hdr.get('CDELT2', 1.0)) * npix[1]
    if lon_spacing == 'auto':
        lon_spacing = _auto_tick_spacing(lon_fov_deg)
    if lat_spacing == 'auto':
        lat_spacing = _auto_tick_spacing(lat_fov_deg)

    # Configure ticks
    for a_i in [0, 1]:
        ax.coords[a_i].set_ticklabel(exclude_overlapping=True)
    # A circular-limb globe shows the full hemisphere up to the limb (and over
    # the pole), so its graticule must span the whole coordinate domain like an
    # all-sky frame — NOT the visible-field-extent values used for flat field
    # frames (which would shrink the grid inward and leave an empty over-pole
    # cap). Rectangular TAN/SIN field views are not circular globes, so they
    # keep the field-extent path (and its degenerate-fov freeze guard).
    is_circular_globe = is_globe and frame_shape == 'circular'
    if is_allsky or is_circular_globe:
        # Whole-sky / globe frame: spacing= over the full coord domain is safe
        # for the auto-resolved spacing (fov ≈ 360°/180° → a dozen ticks), but
        # an explicit degenerate spacing would freeze the draw — clamp it.
        lon_spacing = _clamp_spacing(float(lon_spacing), 360.0, 'longitude')
        lat_spacing = _clamp_spacing(float(lat_spacing), 180.0, 'latitude')
        ax.coords[0].set_ticks(spacing=lon_spacing * u.deg)
        ax.coords[1].set_ticks(spacing=lat_spacing * u.deg)
        # Curved all-sky / circular-globe spine: drop the base-style minor
        # ticks (astropy scatters them into a dense central row + all-around
        # limb here); the flat field branch below keeps its useful minors.
        _suppress_curved_minor_ticks(ax)
    else:
        # Zoomed field frame: never hand astropy spacing= over the full domain
        # — it enumerates ticks across all of lon 0–360° / lat ±90°, so a tiny
        # fov_deg (e.g. 50 mas → ~2e-6° spacing) generates ~1e8 candidate ticks
        # and freezes the draw. Place explicit tick values across only the
        # visible field extent (mirrors the all-sky lat handling above), which
        # bounds the count regardless of how small the field is.
        crval1 = float(hdr.get('CRVAL1', 0.0))
        crval2 = float(hdr.get('CRVAL2', 0.0))
        # The lon-coordinate span widens by 1/cos(dec) vs the on-sky width.
        cosd = max(float(np.cos(np.radians(crval2))), 1e-6)
        lon_vals = _field_tick_values(crval1, float(lon_spacing),
                                      lon_fov_deg / cosd)
        lat_vals = _field_tick_values(crval2, float(lat_spacing), lat_fov_deg)
        ax.coords[0].set_ticks(values=lon_vals * u.deg)
        ax.coords[1].set_ticks(values=lat_vals * u.deg)

    if grid:
        # Projections with interrupted / non-rectangular visible regions need
        # a dense same-frame overlay instead of astropy's default grid, which
        # drops wrap-side segments (HPX/XPH/BON/PCO) or spills past the
        # visible wedge (conics). Everything else uses the default grid.
        # ONLY for the all-sky frame: the backfill + its boundary mask are
        # built for the whole-sky envelope. On a ZOOMED field frame the visible
        # region is a smooth single face, where astropy's default grid is clean
        # and the all-sky machinery instead bleeds whole-net segments through
        # the (over-approximating) boundary — worst on QSC's curved faces.
        if is_allsky and fits_code.upper() in _BACKFILL_GRID_CODES:
            try:
                _backfill_overlay_grid(ax, float(lon_spacing),
                                       float(lat_spacing), gridcolor,
                                       gridalpha, fits_code.upper(),
                                       lw=gridlw, ls=gridls)
            except Exception as exc:
                warnings.warn(
                    f"make_wcs_frame: gridline backfill failed for "
                    f"{fits_code} ({type(exc).__name__}: {exc}); falling back "
                    f"to the default grid.", UserWarning, stacklevel=2)
                ax.coords.grid(color=gridcolor, alpha=gridalpha,
                           **({} if gridlw is None else {'linewidth': gridlw}),
                           **({} if gridls is None else {'linestyle': gridls}))
        else:
            ax.coords.grid(color=gridcolor, alpha=gridalpha,
                           **({} if gridlw is None else {'linewidth': gridlw}),
                           **({} if gridls is None else {'linestyle': gridls}))

    # Apply frame-appropriate formatting defaults
    pb_lon_label_fmt = None
    if apply_format_defaults:
        frame_lower = frame.lower()
        if any(f in frame_lower for f in ('gal', 'super', 'ecl', 'hel')):
            ax.coords[0].set_format_unit(u.deg, decimal=True)
            ax.coords[1].set_format_unit(u.deg, decimal=True)
        # Longitude units: explicit lon_units wins; 'auto' → degrees for the
        # geographic orientation (lonsign > 0), else the equatorial default
        # (RA hours). Runs after the galactic block so an explicit override
        # still applies there. The overlay label format (for SIN-globe /
        # in-frame styles, which draw their own labels) is threaded separately.
        _lon_u = resolve_lon_units(lon_units)
        _apply_lon_units(ax, _lon_u, lonsign > 0)
        pb_lon_label_fmt = _overlay_lon_fmt(_lon_u, lonsign > 0)

    # Pad title position for FITS projections whose topmost tick labels
    # (pole meridians) would otherwise collide with the default title.
    # Elliptical frames (AIT, MOL) taper away from y=1.0 at top and
    # don't need padding; rectangular-ish ones (SFL, PAR, CAR) do, as
    # does the SIN globe where the circle reaches y=1.0 at the top.
    if fits_code.upper() in _POLE_TOP_FITS_PROJECTIONS:
        _pad_title_for_pole_top_projection(ax)

    # Draw the projection's natural visible-region outline as a solid line for
    # projections whose region sits inside a larger rectangular frame and whose
    # edges astropy's gridline densifier doesn't trace cleanly: the HPX stepped
    # diamond, the XPH butterfly, the quadcube (TSC/CSC/QSC) cross, and the
    # conic (COD/COE/COO/COP) wedge — all via the unified `_projection_boundary`
    # (data-coord Path). The conic path traces the apex, the two seam meridians,
    # and the bottom parallel (the kapteyn "boundary parallel"). BON keeps its
    # own cardioid drawer. PCO draws its egg envelope (the lon=CRVAL±180
    # meridians) via _draw_allsky_lon_boundary — the same lon-meridian drawer
    # BON uses — which handles the non-monotonic double-lobe by sampling the
    # two seam meridians directly rather than relying on a closed polygon.
    if is_allsky:
        code = fits_code.upper()
        if code in ('BON', 'PCO'):
            _draw_allsky_lon_boundary(ax, hdr, color=outline_color,
                                      lw=(0.7 if outline_lw is None
                                          else float(outline_lw)))
        elif code in ('HPX', 'XPH', 'TSC', 'CSC', 'QSC',
                      'COD', 'COE', 'COO', 'COP'):
            try:
                path = _projection_boundary(ax)
                if path is not None:
                    ax.plot(path.vertices[:, 0], path.vertices[:, 1],
                            transform=ax.transData,
                            color=(outline_color if outline_color is not None
                                   else rcParams['axes.edgecolor']),
                            lw=(0.7 if outline_lw is None
                                else float(outline_lw)),
                            zorder=1.5)
            except Exception:
                pass  # outline is cosmetic — never fail the frame build

    # Auto-shrink the tick-label fontsize to the available axes width
    # before _apply_tick_style runs. Done first so the chosen size is
    # picked up by both the native astropy labels (via set_ticklabel)
    # and any overlay-mode labels (forwarded through label_fontsize).
    # A canvas.draw() is needed so label texts exist for character-
    # count introspection — _apply_tick_style also draws, but draws
    # are idempotent within a single frame build. The whole block is
    # try/excepted: auto-fontsize is a convenience, never a reason for
    # make_wcs_frame to fail.
    auto_fs = None
    if auto_fontsize:
        from .autosize import auto_size_ticklabels
        try:
            ax.figure.canvas.draw()
        except Exception:
            pass
        try:
            auto_fs = auto_size_ticklabels(ax)
        except Exception as exc:
            warnings.warn(
                f"make_wcs_frame: auto_fontsize failed "
                f"({type(exc).__name__}: {exc}); falling back to "
                f"rcParams default. Pass auto_fontsize=False to suppress.",
                UserWarning, stacklevel=2)

    _apply_tick_style(ax, frame_shape, tick_style, tick_rotation,
                      label_fontsize=auto_fs, fits_code=fits_code,
                      lon_label_fmt=pb_lon_label_fmt)

    # Pin tick marks to their natural spines on flat single-field frames so
    # converging meridians don't scatter stray ticks onto the side/bottom
    # edges (see _restrict_field_edge_ticks). Gated to the field path — the
    # same condition that routes to field-extent tick values above — so
    # all-sky and circular-limb globe frames keep astropy's full assignment.
    # Also restricted to the default rectangular frame (``frame_class is
    # None``): the 'bt'/'lr' spine codes only name spines on RectangularFrame,
    # and a non-rectangular field frame (e.g. AIT/MOL at a small FOV) both
    # ignores them and — on astropy >= 8 — warns that they are unrecognized.
    is_circular_globe = is_globe and frame_shape == 'circular'
    if (edge_ticks == 'auto' and frame_class is None
            and not (is_allsky or is_circular_globe)):
        _restrict_field_edge_ticks(ax)

    _attach_sky_methods(ax)
    if return_hdr:
        return ax, hdr
    return ax


###############################################################################
#                                                                             #
#            OFFSET COORDINATE WCS                                 #
#                                                                             #
###############################################################################


_OFFSET_UNIT_FACTORS = {
    'degree': 1., 'degrees': 1., 'deg': 1,
    'arcmin': 60, 'amin': 60, 'arcminute': 60,
    'arcsec': 3600, 'asec': 3600, 'arcsecond': 3600,
    'mas': 3600e3, 'marcsec': 3600e3, 'milliarcsec': 3600e3,
    'uas': 3600e6, 'microarcsec': 3600e6, 'microarcsecond': 3600e6,
}


def _parse_centercoords(centercoords: Any,
                        coord_units: tuple[str, str] = ('deg', 'deg')) -> Any:
    """Parse center coordinates into a SkyCoord."""
    if isinstance(centercoords, SkyCoord):
        return SkyCoord(centercoords)
    elif isinstance(centercoords, (list, tuple)):
        if isinstance(centercoords[0], (int, float)):
            return SkyCoord(*centercoords, unit=coord_units)
        else:
            return SkyCoord(*centercoords)
    raise ValueError('centercoords must be SkyCoord or [lon, lat] list')


def WCS_to_offsetWCS(wcs: Any, centercoords: Any, offset_units: str = 'arcsec',
                     coord_units: tuple[str, str] = ('deg', 'deg')) -> Any:
    """
    Convert a celestial WCS to a locally linear offset WCS.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
    centercoords : SkyCoord or list
    offset_units : str

    Returns
    -------
    new_wcs : astropy.wcs.WCS
    """
    center = _parse_centercoords(centercoords, coord_units)
    xp, yp = skycoord_to_pixel(center, wcs)

    new_wcs = WCS(naxis=2)
    new_wcs.wcs.crpix = xp + 1, yp + 1
    new_wcs.wcs.crval = 0., 0.
    new_wcs.wcs.cdelt = proj_plane_pixel_scales(wcs) * _OFFSET_UNIT_FACTORS[offset_units]
    new_wcs.wcs.ctype = 'XOFFSET', 'YOFFSET'
    new_wcs.wcs.cunit = offset_units, offset_units

    return new_wcs

def offset_coord_WCS(hdrin: Any, centercoords: Any, offset_units: str = 'arcsec',
                     coord_units: tuple[str, str] = ('deg', 'deg')) -> Any:
    """
    Create an offset WCS from a FITS header and center coordinates.

    Parameters
    ----------
    hdrin : astropy.io.fits.Header
    centercoords : SkyCoord or list
    offset_units : str

    Returns
    -------
    wcs : astropy.wcs.WCS
    """
    center = _parse_centercoords(centercoords, coord_units)
    wcs = WCS(hdrin)

    cpix_x, cpix_y = skycoord_to_pixel(center, wcs)
    cdelts_deg = proj_plane_pixel_scales(wcs)

    wcs.naxis = 2
    wcs.wcs.crpix = cpix_x + 1, cpix_y + 1
    wcs.wcs.crval = 0., 0.
    wcs.wcs.cdelt = cdelts_deg * _OFFSET_UNIT_FACTORS[offset_units]
    wcs.wcs.ctype = 'XOFFSET', 'YOFFSET'
    wcs.wcs.cunit = offset_units, offset_units

    return wcs


# ===== WCSAxes introspection helpers =====

def _is_wcsaxes(ax: Any) -> bool:
    """Check if an axes object is a WCSAxes instance."""
    return hasattr(ax, 'get_transform') and hasattr(ax, 'wcs')


def _get_wcs_frame_name(ax: Any) -> str:
    """
    Get the coordinate frame name from a WCSAxes object.
    Returns 'icrs', 'galactic', 'ecliptic', or 'supergalactic'.
    """
    # Check for non-FITS projection metadata first
    if hasattr(ax, '_sph_frame'):
        return ax._sph_frame
    if not hasattr(ax, 'wcs') or ax.wcs is None:
        return 'icrs'
    ctype = ax.wcs.wcs.ctype[0][:4].upper()
    frame_map = {
        'RA--': 'icrs', 'GLON': 'galactic', 'ELON': 'ecliptic',
        'SLON': 'supergalactic', 'TLON': 'itrs', 'HLON': 'heliographic',
    }
    return frame_map.get(ctype, 'icrs')


def _get_wcs_center_lon(ax: Any) -> float:
    """Get the center longitude of a WCSAxes from its CRVAL."""
    if hasattr(ax, '_sph_center_lon'):
        return ax._sph_center_lon
    if hasattr(ax, 'wcs') and ax.wcs is not None:
        return ax.wcs.wcs.crval[0]
    return 180.


def _get_wcs_center_lat(ax: Any) -> float:
    """Get the center latitude of a WCSAxes from its CRVAL."""
    if hasattr(ax, '_sph_center_lat'):
        return ax._sph_center_lat
    if hasattr(ax, 'wcs') and ax.wcs is not None:
        return ax.wcs.wcs.crval[1]
    return 0.


def _east_increases_right(wcs: Any) -> bool:
    """Whether +longitude maps to +x (east to the *right*) on screen for this
    WCS — the projection-agnostic test of display orientation. The CDELT1 sign
    alone doesn't tell you this (a CD/PC matrix carries the scale, so CDELT1
    is often +1; and orthographic SIN inverts the handedness relative to
    cylindrical/TAN via CDELT2), so probe the actual world→pixel mapping."""
    lon0, lat0 = float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1])
    x0, _ = wcs.world_to_pixel_values(lon0, lat0)
    x1, _ = wcs.world_to_pixel_values(lon0 + 1e-3, lat0)
    return float(np.ravel(x1)[0]) > float(np.ravel(x0)[0])
