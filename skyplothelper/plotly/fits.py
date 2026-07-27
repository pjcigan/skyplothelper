"""FITS-image viewer for plotly figures, driven by an astropy WCS.

Displays a FITS image in **pixel coordinates** as a ``go.Heatmap`` (a clean
regular grid, like a WCSAxes), with WCS-aware axis ticks and hover. sph sky
overlays project into the same pixel space via :class:`WCSPixelProjector`
(``_project_xy = wcs.world_to_pixel``), so catalog markers
(:func:`add_fits_scatter`) and :class:`~skyplothelper.CompoundRegion` regions
(:func:`make_fits_compound_region` → :func:`~skyplothelper.plotly.add_compound_region`)
land on the image for free.

Two coordinate label modes share one projector and differ only in the tick /
hover *formatting*:

* ``coords='absolute'`` — world RA / Dec.
* ``coords='offset'`` — angular offset (arcsec / mas / …) from a reference
  point, the natural mode for compact / VLBI fields.

The "dynamic" layer (recompute ticks + beam on zoom) is kept framework-agnostic:
:func:`fits_ticks_for_range` and :func:`beam_shape_for_range` are stateless
per-view computers an app's relayout handler calls. (A Dash convenience layer
that wires them is a separate, optional follow-on.)
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord


from .core import _THEMES, _import_plotly

# --- small internal helpers ------------------------------------------------

def _squeeze_2d(data: npt.ArrayLike) -> np.ndarray:
    """Drop degenerate (length-1) leading axes so a radio cube's
    ``(1, 1, ny, nx)`` data becomes ``(ny, nx)``. Errors on a genuinely
    >2-D image (an axis with length > 1)."""
    arr = np.asarray(data)
    sq = np.squeeze(arr)
    if sq.ndim != 2:
        raise ValueError(
            f"add_fits_image needs 2-D image data (after squeezing degenerate "
            f"axes); got shape {arr.shape} -> {sq.shape}. Select a single "
            f"plane first.")
    return sq


def _celestial_wcs(wcs: Any) -> Any:
    """The 2-D celestial sub-WCS (handles 4-D radio-cube WCS)."""
    return wcs.celestial if getattr(wcs, 'naxis', 2) > 2 else wcs


def _pixel_scale_deg(wcs: Any) -> float:
    """Mean pixel scale in degrees/pixel from a 2-D celestial WCS."""
    from astropy.wcs.utils import proj_plane_pixel_scales
    return float(np.mean(np.abs(proj_plane_pixel_scales(_celestial_wcs(wcs)))))


def _mpl_colorscale(cmap: Any, n: int = 64) -> Any:
    """Sample a matplotlib colormap name into a plotly colorscale list.

    Robust across plotly versions / colormap-name casing (matplotlib is a
    hard dependency). A list / pre-built colorscale passes through."""
    if isinstance(cmap, (list, tuple)):
        return cmap
    try:
        from matplotlib import colormaps
        cmap_obj = colormaps[cmap]
    except Exception:
        # matplotlib.cm.get_cmap was removed in mpl 3.11; plt.get_cmap is the
        # stable name-lookup fallback across the supported versions.
        import matplotlib.pyplot as plt
        cmap_obj = plt.get_cmap(cmap)
    out = []
    for i in range(n):
        f = i / (n - 1)
        r, g, b, _ = cmap_obj(f)
        out.append([f, f'rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})'])
    return out


# _auto_offset_unit names the µas unit 'μas'; OffsetFormatter keys it 'uas'.
_OFFSET_UNIT_ALIASES = {'μas': 'uas', 'uas': 'uas', 'mas': 'mas',
                        'arcsec': 'arcsec', 'arcmin': 'arcmin'}


def _resolve_offset_unit(offset_units: str | None, fov_deg: float) -> str:
    """Resolve an offset unit name to an OffsetFormatter key, auto-picking
    from the field of view when ``'auto'``."""
    from ..ticks import _auto_offset_unit
    if offset_units is None or offset_units == 'auto':
        _, name = _auto_offset_unit(fov_deg)
    else:
        name = offset_units
    key = _OFFSET_UNIT_ALIASES.get(name)
    if key is None:
        raise ValueError(
            f"offset_units must be one of {sorted(set(_OFFSET_UNIT_ALIASES))} "
            f"or 'auto', got {offset_units!r}")
    return key


def _nice_ticks(lo: float, hi: float,
                max_ticks: int = 6) -> npt.NDArray[np.float64]:
    """A few nice-number tick positions within ``[lo, hi]`` (pixel coords)."""
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    span = hi - lo
    if span <= 0:
        return np.array([lo])
    raw = span / max(2, max_ticks)
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    else:
        step = 10 * mag
    start = np.ceil(lo / step) * step
    ticks = np.arange(start, hi + 0.5 * step, step)
    return ticks[(ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)]


def _minor_ticks(major: npt.ArrayLike, lo: float, hi: float,
                 n: int = 5) -> npt.NDArray[np.float64]:
    """Minor-tick pixel positions subdividing the (uniformly spaced) ``major``
    ticks into ``n`` intervals, extended to cover the full ``[lo, hi]`` range.

    Needed only for the absolute-coord axes, which set explicit ``tickvals``
    (``tickmode='array'``) — plotly then suppresses its automatic minor ticks,
    so they must be supplied as an array too. Offset axes use native ticks and
    get minor ticks for free from the layout.
    """
    major = np.asarray(major, dtype=float)
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    if major.size < 2:
        return np.array([])
    step = float(np.median(np.diff(major))) / n
    if step <= 0:
        return np.array([])
    start = np.ceil((lo - major[0]) / step) * step + major[0]
    minor = np.arange(start, hi + 0.5 * step, step)
    # Drop positions coincident with a major tick.
    keep = np.min(np.abs(minor[:, None] - major[None, :]), axis=1) > 0.5 * step
    minor = minor[keep]
    return minor[(minor >= lo - 1e-9) & (minor <= hi + 1e-9)]


def _radec_str(value_deg: float, spacing_deg: float, is_ra: bool) -> str:
    """Format an absolute world value (deg) with precision adapted to the
    tick spacing, in decimal degrees."""
    prec = int(np.clip(np.ceil(-np.log10(max(spacing_deg, 1e-12))) + 1, 1, 8))
    return f"{value_deg:.{prec}f}°"


def _stamp_fits_meta(fig: Any, wcs2d: Any, *,
                     image_shape: tuple[int, int] | None = None,
                     coords: str | None = None,
                     ref_coord: Sequence[float] | None = None,
                     offset_units: str | None = None) -> None:
    """Merge FITS-viewer keys into ``fig.layout.meta`` (kept JSON-serializable;
    the WCS is stored as a header string)."""
    meta = dict((getattr(fig, 'layout', None)
                 and getattr(fig.layout, 'meta', None)) or {})
    meta['sph_fits'] = True
    if wcs2d is not None:
        meta['sph_wcs_header'] = wcs2d.to_header_string()
    if image_shape is not None:
        meta['sph_image_shape'] = [int(image_shape[0]), int(image_shape[1])]
    if coords is not None:
        meta['sph_coords'] = coords
    if ref_coord is not None:
        meta['sph_ref_coord'] = [float(ref_coord[0]), float(ref_coord[1])]
    if offset_units is not None:
        meta['sph_offset_units'] = offset_units
    fig.update_layout(meta=meta)


# --- figure scaffold -------------------------------------------------------

def make_fits_figure(wcs: Any, *, theme: str = 'light', width: int = 700,
                     height: int = 700, title: str | None = None) -> Any:
    """Build an empty plotly figure scaffold for a FITS image.

    Square-aspect, pixel-coordinate axes (the WCS already encodes any RA
    flip, so — unlike :func:`~skyplothelper.plotly.make_figure` — there is no
    sky-direction reversal and no all-sky graticule). Pass the result to
    :func:`add_fits_image`.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        Image WCS (degenerate axes are reduced to the celestial 2-D WCS).
    theme : {'light', 'dark'}
        Color theme. Default ``'light'``.
    width, height : int
        Figure size in pixels. Default ``700 x 700`` (square, matching the
        1:1 pixel aspect).
    title : str, optional

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    go = _import_plotly()
    if theme not in _THEMES:
        raise ValueError(f"theme must be 'light' or 'dark', got {theme!r}")
    th = _THEMES[theme]
    fig = go.Figure()
    fig.update_layout(
        width=width, height=height,
        paper_bgcolor=th['bg'], plot_bgcolor=th['bg'],
        font=dict(color=th['fg']),
        title=dict(text=title) if title else None,
        margin=dict(l=60, r=20, t=40 if title else 20, b=50),
        showlegend=False,
        xaxis=dict(visible=True, showgrid=False, zeroline=False,
                   scaleanchor='y', scaleratio=1, constrain='domain',
                   color=th['fg'], showline=True, linecolor=th['fg'],
                   mirror='ticks', ticks='inside', tickcolor=th['fg'],
                   ticklen=6, minor=dict(ticks='inside', ticklen=3,
                                         tickcolor=th['fg']),
                   automargin=False),
        yaxis=dict(visible=True, showgrid=False, zeroline=False,
                   color=th['fg'], showline=True, linecolor=th['fg'],
                   mirror='ticks', ticks='inside', tickcolor=th['fg'],
                   ticklen=6, minor=dict(ticks='inside', ticklen=3,
                                         tickcolor=th['fg']),
                   automargin=False),
    )
    _stamp_fits_meta(fig, _celestial_wcs(wcs))
    return fig


# --- the image -------------------------------------------------------------

def add_fits_image(fig: Any, data: npt.ArrayLike, wcs: Any, *,
                   coords: str = 'absolute', stretch: str = 'linear',
                   clip: str = 'percentile', plo: float = 0.5,
                   phi: float = 99.5, vmin: float | None = None,
                   vmax: float | None = None, colormap: Any = 'inferno',
                   colorbar: bool = False,
                   ref_coord: Sequence[float] | None = None,
                   offset_units: str = 'auto', header: Any = None,
                   beam_maj: float | None = None, beam_min: float | None = None,
                   beam_pa: float = 0.0, beam_corner: str = 'lower left',
                   beam_color: str | None = None,
                   beam_fillcolor: str | None = None,
                   field_size: float | None = None, hover: str | bool = 'full',
                   bunit: str | None = None, display_factor: float = 1.0,
                   max_pixels: int = 2_000_000) -> Any:
    """Add a FITS image to a plotly figure, driven by an astropy WCS.

    The image is drawn in **pixel coordinates** as a ``go.Heatmap``; sph sky
    overlays project onto it via :class:`WCSPixelProjector`. Kwarg names mirror
    the matplotlib :func:`skyplothelper.quicklook_plot` where sensible.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Target figure from :func:`make_fits_figure`.
    data : ndarray
        Image data (degenerate Stokes / frequency axes are squeezed away).
    wcs : astropy.wcs.WCS
        Image WCS (reduced to its celestial 2-D sub-WCS).
    coords : {'absolute', 'offset'}
        Tick / hover labeling: absolute world RA/Dec, or angular offset from
        ``ref_coord``. Default ``'absolute'``.
    stretch, clip, plo, phi, vmin, vmax : passed to
        :func:`skyplothelper.images.levels.rescale_image` to map data to a
        ``[0, 1]`` display array. ``stretch`` accepts ``'linear'``, ``'sqrt'``,
        ``'log'``, ``'asinh'``, ``'symmetric_log'``, … .
    colormap : str or list
        Matplotlib colormap name (or a plotly colorscale list). Default
        ``'inferno'``.
    colorbar : bool
        Show a colorbar labeled with raw data levels. Default ``False``.
    ref_coord : (ra, dec) or None
        Reference for ``coords='offset'`` (degrees). Defaults to the WCS
        reference point (``CRVAL``).
    offset_units : {'auto', 'arcsec', 'arcmin', 'mas', 'uas'}
        Offset unit. ``'auto'`` picks from the field of view. Default ``'auto'``.
    header : astropy Header, optional
        If given and ``beam_*`` are unset, the beam is read from the header
        (``BMAJ`` / ``BMIN`` / ``BPA``).
    beam_maj, beam_min : float, optional
        Beam FWHM axes in **arcseconds**. If given, a corner beam is drawn.
    beam_pa : float
        Beam position angle (deg, N through E). Default ``0``.
    beam_color, beam_fillcolor : str, optional
        Outline / fill of the beam ellipse. ``None`` (default) keeps the
        white-on-translucent-white pair, which reads over most astronomical
        colormaps; set them for a light background or a reversed colormap
        such as ``gray_r``, where white is invisible.
    beam_corner : {'lower left', 'lower right', 'upper left', 'upper right'}
        View corner the beam is nestled into. Default ``'lower left'``.
    field_size : float, optional
        Initial field width in the offset unit (arcsec by default), centered
        on ``ref_coord``. ``None`` shows the full image.
    hover : {'full', 'value', False}
        Hover content. ``'full'`` (default) shows world RA/Dec, the brightness
        value, and the pixel index. ``'value'`` shows just value + pixel (skips
        the per-pixel RA/Dec grid — cheaper for very large images). ``False``
        disables hover.
    bunit : str, optional
        Brightness unit label for hover / colorbar. Defaults to the header's
        ``BUNIT`` when ``header`` is given.
    display_factor : float
        Multiplier applied to the displayed brightness in the hover readout and
        colorbar labels (the stretch / image are unchanged) — e.g. ``1e3`` with
        ``bunit='mJy/beam'`` to show a Jy/beam image in mJy/beam. Default ``1``.
    max_pixels : int
        Above this pixel count the display grid is block-mean downsampled.
        Default ``2e6`` (so a 1k² image shows full-res).

    Returns
    -------
    trace : plotly.graph_objects.Heatmap
        The image trace appended to ``fig.data``.
    """
    go = _import_plotly()
    if coords not in ('absolute', 'offset'):
        raise ValueError(f"coords must be 'absolute' or 'offset', got {coords!r}")
    if hover not in ('full', 'value', False, None):
        raise ValueError(f"hover must be 'full', 'value', or False, got {hover!r}")

    wcs2d = _celestial_wcs(wcs)
    img = _squeeze_2d(data).astype(float)
    ny, nx = img.shape

    # Brightness unit: explicit, else from the header (BUNIT).
    if bunit is None and header is not None:
        bunit = (header.get('BUNIT') or '').strip() or None
    # Coordinate label precision for hover, adapted to the pixel scale.
    dpp = _pixel_scale_deg(wcs2d)
    coord_prec = int(np.clip(np.ceil(-np.log10(max(dpp, 1e-12))) + 1, 3, 9))

    # Optional block-mean downsample of the *display* grid (the WCS/overlays
    # keep the original resolution; only the heatmap is coarsened).
    step = 1
    if nx * ny > max_pixels:
        step = int(np.ceil(np.sqrt(nx * ny / max_pixels)))
        warnings.warn(
            f"image is {nx}x{ny} (> max_pixels={max_pixels}); displaying a "
            f"{step}x block-mean downsample. Overlays/ticks stay full-res.")

    disp = img[::step, ::step] if step > 1 else img
    dny, dnx = disp.shape

    # Normalize to [0,1], then re-insert NaN at blanks so they render
    # transparent (rescale_image fills NaN with a fixed color otherwise).
    from ..images.levels import rescale_image
    finite = np.isfinite(disp)
    z01 = rescale_image(disp, stretch=stretch, clip=clip, plo=plo, phi=phi,
                        vmin=vmin, vmax=vmax, fill_nan=0.0)
    z = np.where(finite, z01, np.nan)

    ref = (tuple(ref_coord) if ref_coord is not None
           else (float(wcs2d.wcs.crval[0]), float(wcs2d.wcs.crval[1])))

    # Pixel vectors of the displayed cells (used for the world-coord hover
    # grid regardless of which units the axes show).
    xpix = np.arange(0, nx, step, dtype=float)[:dnx]
    ypix = np.arange(0, ny, step, dtype=float)[:dny]

    # Display-coordinate vectors. ``absolute`` shows pixel coords; ``offset``
    # shows angular offset from the reference (linear in pixel — the standard
    # constant-mas/pixel "relative coordinate" representation), so plotly's
    # native numeric ticking gives round, zoom-adaptive labels with no server.
    if coords == 'offset':
        from ..ticks import OffsetFormatter
        from .projector import _offset_linear_map
        unit_key = _resolve_offset_unit(offset_units, _img_fov_deg(wcs2d, nx, ny))
        unit_lbl, factor = OffsetFormatter._UNIT_LABELS[unit_key]
        sx, sy, cx, cy = _offset_linear_map(wcs2d, ref, factor)
        xc = sx * (xpix - cx)            # east-offset per cell (signed)
        yc = sy * (ypix - cy)            # north-offset per cell
        x_title, y_title = (f"Relative RA ({unit_lbl})",
                            f"Relative Dec ({unit_lbl})")
        stored_units = unit_key
    else:
        unit_key = None
        xc, yc = xpix, ypix
        # Axes are pixel-valued, but the ticks are labeled with world RA/Dec
        # (decimal degrees), so the titles name the world coordinate.
        x_title, y_title = 'RA', 'Dec'
        stored_units = offset_units

    heat_kw = dict(
        z=z, x=xc, y=yc,
        colorscale=_mpl_colorscale(colormap),
        showscale=bool(colorbar), zsmooth=False, connectgaps=False,
        name='', hoverongaps=False,
    )
    if colorbar:
        heat_kw['colorbar'] = _raw_colorbar(disp, stretch, clip, plo, phi,
                                            vmin, vmax, display_factor, bunit)
    _apply_hover(heat_kw, hover, disp, xpix, ypix, xc, yc, wcs2d, coords,
                 unit_key, display_factor, bunit, coord_prec)

    trace = go.Heatmap(**heat_kw)
    fig.add_trace(trace)

    _stamp_fits_meta(fig, wcs2d, image_shape=(ny, nx), coords=coords,
                     ref_coord=ref, offset_units=stored_units)

    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)

    # Initial axis ranges (full image or a field_size box about the reference)
    # in the active display units.
    _set_fits_ranges(fig, coords, xc, yc, ref, wcs2d, unit_key, field_size)

    # Absolute mode shows custom RA/Dec tick labels (static at this view; full
    # zoom-adaptive labels need the dynamic layer). Offset mode uses plotly's
    # native numeric ticks (round + zoom-adaptive) — no ticktext needed.
    if coords == 'absolute':
        xr = [float(fig.layout.xaxis.range[0]), float(fig.layout.xaxis.range[1])]
        yr = [float(fig.layout.yaxis.range[0]), float(fig.layout.yaxis.range[1])]
        # Square pixels (scaleanchor) + a colorbar expand the *visible* extent
        # past the set range on one axis, so ticks computed from the set range
        # stop short of that axis' edge. Tick over the displayed extent (same
        # estimate the beam uses) so ticks fill the whole visible field.
        xr, yr = _displayed_ranges(fig, xr, yr, colorbar)
        ticks = fits_ticks_for_range(wcs2d, xr, yr, coords='absolute',
                                     ref_coord=ref)
        fig.update_xaxes(**ticks['xaxis'])
        fig.update_yaxes(**ticks['yaxis'])

    # Optional beam, read from the header if not given explicitly.
    if beam_maj is None and header is not None:
        from ..core.fits_utils import beampars_asec_fromhdr
        try:
            bmaj_a, bmin_a, bpa = beampars_asec_fromhdr(header)
            beam_maj, beam_min, beam_pa = bmaj_a, bmin_a, bpa
        except Exception:
            pass
    if beam_maj is not None and beam_min is not None:
        xr = [float(fig.layout.xaxis.range[0]), float(fig.layout.xaxis.range[1])]
        yr = [float(fig.layout.yaxis.range[0]), float(fig.layout.yaxis.range[1])]
        # Square pixels (scaleanchor) + a colorbar make plotly expand one axis
        # to fill the plot area, so the *visible* extent is larger than the set
        # range. Pin the beam to that estimated displayed extent, else it sits
        # off the true (expanded) corner.
        xr, yr = _displayed_ranges(fig, xr, yr, colorbar)
        # Beam styling was unreachable from here: both builders expose
        # ``line_color`` / ``fillcolor`` but neither was forwarded, so the
        # ellipse stayed white on a light theme or a ``gray_r`` colormap.
        _beam_kw: dict[str, Any] = {}
        if beam_color is not None:
            _beam_kw['line_color'] = beam_color
        if beam_fillcolor is not None:
            _beam_kw['fillcolor'] = beam_fillcolor
        if coords == 'offset':
            shape = _beam_ellipse_offset(beam_maj, beam_min, beam_pa, factor,
                                         xr, yr, corner=beam_corner,
                                         **_beam_kw)
        else:
            shape = beam_shape_for_range(
                wcs2d, xr, yr, bmaj_arcsec=beam_maj, bmin_arcsec=beam_min,
                bpa_deg=beam_pa, corner=beam_corner, pad_frac=0.012,
                **_beam_kw)
        fig.add_shape(**shape)

    return trace


def _img_fov_deg(wcs2d: Any, nx: int, ny: int) -> float:
    """Approximate field-of-view diagonal-ish width in degrees."""
    return _pixel_scale_deg(wcs2d) * max(nx, ny)


def _displayed_ranges(
    fig: Any, xr: list[float], yr: list[float], has_colorbar: bool,
) -> tuple[list[float], list[float]]:
    """Estimate the axis ranges actually *displayed* under square-pixel
    (``scaleanchor``) constraints.

    With equal data aspect, plotly fits the requested range into the plot area
    and expands the slacker axis to fill the rest at the same data-per-pixel
    scale. The visible extent therefore exceeds the set range on one axis —
    which matters for corner-pinned decorations like the beam. Returns the
    aspect-expanded ``(xr, yr)`` (centers and reversal preserved). A rough
    colorbar allowance is subtracted from the plot width when present.
    """
    w = float(fig.layout.width or 700)
    h = float(fig.layout.height or 700)
    m = fig.layout.margin
    dom_w = w - float(m.l or 0) - float(m.r or 0)
    dom_h = h - float(m.t or 0) - float(m.b or 0)
    if has_colorbar:
        dom_w -= 0.13 * w          # colorbar bar + tick labels, approximate
    if dom_w <= 0 or dom_h <= 0:
        return xr, yr
    xspan, yspan = abs(xr[1] - xr[0]), abs(yr[1] - yr[0])
    dpp = max(xspan / dom_w, yspan / dom_h)   # data units per pixel
    xc, yc = 0.5 * (xr[0] + xr[1]), 0.5 * (yr[0] + yr[1])
    hx, hy = 0.5 * dpp * dom_w, 0.5 * dpp * dom_h
    sx = 1.0 if xr[1] >= xr[0] else -1.0
    sy = 1.0 if yr[1] >= yr[0] else -1.0
    return ([xc - sx * hx, xc + sx * hx], [yc - sy * hy, yc + sy * hy])


def _set_fits_ranges(fig: Any, coords: str, xc: np.ndarray,
                     yc: np.ndarray, ref: Sequence[float], wcs2d: Any,
                     unit_key: str | None, field_size: float | None) -> None:
    """Set the initial axis ranges (full image, or a ``field_size`` arcsec box
    about the reference) in the active display units. The offset x-axis is
    reversed so east (positive relative RA) is on the left, per convention."""
    if coords == 'offset':
        from ..ticks import OffsetFormatter
        assert unit_key is not None  # set whenever coords == 'offset'
        factor = OffsetFormatter._UNIT_LABELS[unit_key][1]
        if field_size is not None:
            half = 0.5 * float(field_size) * factor / 3600.0   # arcsec → unit
            xr, yr = [half, -half], [-half, half]
        else:
            xr = [float(np.max(xc)), float(np.min(xc))]        # reversed (E left)
            yr = [float(np.min(yc)), float(np.max(yc))]
        fig.update_xaxes(range=xr, autorange=False)
        fig.update_yaxes(range=yr, autorange=False)
    else:
        if field_size is not None:
            half_pix = 0.5 * (float(field_size) / 3600.0) / _pixel_scale_deg(wcs2d)
            cx, cy = wcs2d.world_to_pixel_values(ref[0], ref[1])
            cx, cy = float(np.ravel(cx)[0]), float(np.ravel(cy)[0])
            xr, yr = [cx - half_pix, cx + half_pix], [cy - half_pix, cy + half_pix]
        else:
            xr = [float(xc[0]) - 0.5, float(xc[-1]) + 0.5]
            yr = [float(yc[0]) - 0.5, float(yc[-1]) + 0.5]
        fig.update_xaxes(range=xr)
        fig.update_yaxes(range=yr)


def _beam_ellipse_offset(bmaj_arcsec: float, bmin_arcsec: float,
                         bpa_deg: float, factor: float, xr: Sequence[float],
                         yr: Sequence[float], corner: str = 'lower left',
                         pad_frac: float = 0.025, n: int = 72,
                         line_color: str = 'white', line_width: float = 1.2,
                         fillcolor: str = 'rgba(255,255,255,0.25)',
                         ) -> dict[str, Any]:
    """Beam ellipse as a plotly shape in **offset** display coords.

    The offset frame is east=+x, north=+y by construction, so the beam PA
    (N through E) maps directly to ``(sin PA, cos PA)`` — no WCS Jacobian
    needed. Nestled into one of the four view corners (``'lower left'`` by
    default) with a small gap, sized to the beam rather than a fixed fraction
    of the view.
    """
    if corner not in _BEAM_CORNERS:
        raise ValueError(f"corner must be one of {list(_BEAM_CORNERS)}")
    a = 0.5 * bmaj_arcsec * factor / 3600.0   # semi-major, offset units
    b = 0.5 * bmin_arcsec * factor / 3600.0
    x0, x1 = sorted([float(xr[0]), float(xr[1])])
    y0, y1 = sorted([float(yr[0]), float(yr[1])])
    dx, dy = x1 - x0, y1 - y0
    pa = np.radians(bpa_deg)
    maj = (np.sin(pa), np.cos(pa))
    mino = (np.sin(pa + np.pi / 2), np.cos(pa + np.pi / 2))
    bx, by = _beam_corner_center(a, b, maj, mino, x0, x1, y0, y1, dx, dy,
                                 corner, pad_frac, e_left=True)
    t = np.linspace(0.0, 2 * np.pi, n)
    px = bx + a * np.cos(t) * maj[0] + b * np.sin(t) * mino[0]
    py = by + a * np.cos(t) * maj[1] + b * np.sin(t) * mino[1]
    return dict(type='path', path=_ellipse_path(px, py), xref='x', yref='y',
                layer='above', line=dict(color=line_color, width=line_width),
                fillcolor=fillcolor, name='sph_fits_beam')


def _beam_corner_center(a: float, b: float, maj: Sequence[float],
                        mino: Sequence[float], x0: float, x1: float,
                        y0: float, y1: float, dx: float, dy: float,
                        corner: str, pad_frac: float,
                        e_left: bool = False) -> tuple[float, float]:
    """Center for a beam ellipse nestled into a view corner.

    Insets by the ellipse's bounding-box half-extent plus a small gap (sized
    to the beam, not the whole view), so a small beam sits right in the corner
    rather than floating far inside it. ``e_left`` flips the x sense for the
    east-left (reversed-x) offset axes.
    """
    hx = np.hypot(a * maj[0], b * mino[0])    # bbox half-width
    hy = np.hypot(a * maj[1], b * mino[1])    # bbox half-height
    # Gap is mostly beam-sized (so a small beam hugs the corner) with a small
    # fraction-of-view floor.
    gx = hx + max(0.7 * hx, pad_frac * dx)
    gy = hy + max(0.7 * hy, pad_frac * dy)
    want_left = corner.endswith('left')
    if e_left:
        want_left = not want_left          # east on the left → swap x side
    bx = (x0 + gx) if want_left else (x1 - gx)
    by = (y0 + gy) if corner.startswith('lower') else (y1 - gy)
    return bx, by


def _raw_colorbar(disp: np.ndarray, stretch: str, clip: str, plo: float,
                  phi: float, vmin: float | None, vmax: float | None,
                  display_factor: float, bunit: str | None) -> dict[str, Any]:
    """Colorbar whose ticks sit at raw data levels mapped through the forward
    stretch (z is the stretched [0,1] array, so a raw-unit bar needs this).
    Tick labels are scaled by ``display_factor`` and titled with ``bunit``."""
    from ..images.levels import rescale_image
    finite = disp[np.isfinite(disp)]
    if finite.size == 0:
        return dict()
    lo = vmin if vmin is not None else np.percentile(finite, plo)
    hi = vmax if vmax is not None else np.percentile(finite, phi)
    levels = np.linspace(lo, hi, 5)
    # Forward-stretch each level to its [0,1] colorbar position.
    pos = rescale_image(np.array([levels]), stretch=stretch, clip='manual',
                        vmin=lo, vmax=hi, fill_nan=0.0).ravel()
    cb: dict[str, Any] = dict(
        tickvals=list(pos),
        ticktext=[f"{v * display_factor:.3g}" for v in levels])
    if bunit:
        cb['title'] = dict(text=bunit, side='right')
    return cb


def _apply_hover(heat_kw: dict[str, Any], hover: str | bool, disp: np.ndarray,
                 xpix: np.ndarray, ypix: np.ndarray, xc: np.ndarray,
                 yc: np.ndarray, wcs2d: Any, coords: str, unit_key: str | None,
                 display_factor: float, bunit: str | None,
                 coord_prec: int) -> None:
    """Configure hover on the heatmap trace dict in-place.

    ``'full'`` (default) shows world RA/Dec, the brightness value (scaled by
    ``display_factor``, labeled with ``bunit``), and the display-axis position
    (offset units or pixel). ``'value'`` skips the per-pixel RA/Dec grid
    (cheaper for huge images). The RA/Dec grid is always computed from the
    pixel coords, independent of the display units on the axes.
    """
    if hover is False or hover is None:
        heat_kw['hoverinfo'] = 'skip'
        return
    vlbl = f" {bunit}" if bunit else ""
    val = disp * display_factor
    if coords == 'offset':
        from ..ticks import OffsetFormatter
        assert unit_key is not None  # set whenever coords == 'offset'
        u = OffsetFormatter._UNIT_LABELS[unit_key][0]
        pos_line = f"offset: %{{x:.3g}}, %{{y:.3g}} {u}"
    else:
        pos_line = "pixel: %{x:.0f}, %{y:.0f}"
    if hover == 'value':
        heat_kw['customdata'] = val
        heat_kw['hovertemplate'] = (
            f"value: %{{customdata:.4g}}{vlbl}<br>{pos_line}<extra></extra>")
        return
    # 'full': RA/Dec at the top, then value, then the display position.
    from ..core.fits_utils import header_coord_grids
    ra, dec = header_coord_grids(wcs2d, x=xpix, y=ypix)
    heat_kw['customdata'] = np.dstack([ra, dec, val])
    heat_kw['hovertemplate'] = (
        f"RA: %{{customdata[0]:.{coord_prec}f}}°  "
        f"Dec: %{{customdata[1]:.{coord_prec}f}}°"
        f"<br>value: %{{customdata[2]:.4g}}{vlbl}<br>{pos_line}<extra></extra>")


# --- overlays --------------------------------------------------------------

def make_fits_compound_region(fig: Any) -> Any:
    """Construct a :class:`~skyplothelper.CompoundRegion` driven by a FITS
    figure's projector (read from ``fig`` metadata): a pixel projector for
    ``coords='absolute'`` figures, an offset projector for ``'offset'`` ones,
    so regions land in whatever units the image is displayed in.

    Render the result with :func:`~skyplothelper.plotly.add_compound_region`."""
    from .. import CompoundRegion
    return CompoundRegion(_fits_projector_from_figure(fig))


def _fits_projector_from_figure(fig: Any) -> Any:
    from .projector import WCSOffsetProjector, WCSPixelProjector
    meta = (getattr(fig, 'layout', None)
            and getattr(fig.layout, 'meta', None)) or {}
    if meta.get('sph_coords') == 'offset':
        return WCSOffsetProjector.from_figure(fig)
    return WCSPixelProjector.from_figure(fig)


def _project_world_to_fig(
    fig: Any, lons: SkyCoord | npt.ArrayLike, lats: npt.ArrayLike | None = None, wcs: Any = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Project sky ``(lon, lat)`` into a figure's coords.

    On a FITS figure (``sph_wcs_header`` in meta, or an explicit ``wcs=``)
    project via ``wcs.world_to_pixel_values``; otherwise fall back to the
    all-sky projection-by-name (``_project`` against ``make_figure`` meta).
    The single seam a future refactor can reuse to make every standalone
    overlay helper FITS-aware."""
    meta = (getattr(fig, 'layout', None)
            and getattr(fig.layout, 'meta', None)) or {}
    if hasattr(lons, 'transform_to'):
        # A SkyCoord must land in whatever frame this figure's coordinates
        # are in. On a FITS figure that is the image WCS's own frame (read
        # from the header, NOT the make_figure hint, which a FITS figure
        # doesn't set), so a galactic map takes galactic numbers.
        from ..geometry._parsing import _spherical_deg
        target = None
        _w = wcs if wcs is not None else (
            getattr(_fits_projector_from_figure(fig), 'wcs', None)
            if meta.get('sph_wcs_header') else None)
        if _w is not None:
            from astropy.wcs.utils import wcs_to_celestial_frame
            try:
                target = wcs_to_celestial_frame(_celestial_wcs(_w))
            except Exception:
                target = None
        if target is None:
            from .core import _display_frame
            target = _display_frame(fig)
        lons, lats = _spherical_deg(lons.transform_to(target))
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if wcs is None and meta.get('sph_wcs_header'):
        # FITS figure — use its projector (pixel or offset) so markers land in
        # the same coords as the displayed image.
        proj = _fits_projector_from_figure(fig)
        x, y = proj._project_xy(lons, lats)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if wcs is not None:
        x, y = _celestial_wcs(wcs).world_to_pixel_values(lons, lats)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    # all-sky fallback
    from ..projections.project import project as _project
    from .core import _meta_defaults
    projection, center, lat_center, direction = _meta_defaults(
        fig, None, None, None, None)
    return _project(lons, lats, projection=projection, center=center,
                    lat_center=lat_center, direction=direction)


def add_fits_scatter(fig: Any, lons: SkyCoord | npt.ArrayLike, lats: npt.ArrayLike | None = None, *,
                     wcs: Any = None, mode: str = 'markers', marker: Any = None,
                     line: Any = None, name: str | None = None,
                     hovertext: Any = None, hoverinfo: str | None = None,
                     **kwargs: Any) -> Any:
    """Overlay catalog markers (or polylines) on a FITS image.

    Projects ``(lons, lats)`` (ICRS degrees) to image pixels via the figure's
    WCS (or an explicit ``wcs=``) and adds a ``go.Scatter`` trace.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        A FITS figure (from :func:`add_fits_image` / :func:`make_fits_figure`).
    lons, lats : array-like
        Sky positions in degrees.
    wcs : astropy.wcs.WCS, optional
        Override / supply the WCS if the figure carries none.
    mode : str
        Plotly scatter mode (``'markers'`` default; ``'lines'`` for a polyline).
    marker, line, name, hovertext, hoverinfo, **kwargs :
        Forwarded to ``go.Scatter``.

    Returns
    -------
    trace : plotly.graph_objects.Scatter
    """
    go = _import_plotly()
    x, y = _project_world_to_fig(fig, lons, lats, wcs=wcs)
    kw = dict(x=x, y=y, mode=mode, name=name, showlegend=name is not None)
    if marker is not None:
        kw['marker'] = marker
    if line is not None:
        kw['line'] = line
    if hovertext is not None:
        kw['hovertext'] = hovertext
        kw.setdefault('hoverinfo', 'text')
    if hoverinfo is not None:
        kw['hoverinfo'] = hoverinfo
    kw.update(kwargs)
    trace = go.Scatter(**kw)
    fig.add_trace(trace)
    return trace


# --- stateless per-view computers -----------------------------------------

def fits_ticks_for_range(wcs: Any, xrange: Sequence[float],
                         yrange: Sequence[float], *, coords: str = 'absolute',
                         ref_coord: Sequence[float] | None = None,
                         offset_units: str = 'auto', precision: int = 1,
                         max_ticks: int = 6) -> dict[str, Any]:
    """Compute WCS axis ticks for the current pixel view (stateless).

    Maps nice-number pixel tick positions in the given ranges to world coords
    and formats them as absolute RA/Dec or offset-from-reference. An app's
    relayout handler feeds the current ranges and patches the figure's axes
    with the returned ``tickvals`` / ``ticktext``.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        Image WCS (reduced to 2-D celestial).
    xrange, yrange : (lo, hi)
        Current pixel ranges (e.g. straight from a plotly relayout event).
    coords : {'absolute', 'offset'}
    ref_coord : (ra, dec) or None
        Offset reference (deg). Defaults to ``CRVAL``.
    offset_units : {'auto', 'arcsec', 'arcmin', 'mas', 'uas'}
    precision : int
        Decimal places for offset labels.
    max_ticks : int

    Returns
    -------
    dict
        Per-axis tick spec, ready for ``fig.update_xaxes(**out['xaxis'])`` etc.::

            {'xaxis': {'tickvals': [...], 'ticktext': [...]},
             'yaxis': {'tickvals': [...], 'ticktext': [...]}}
    """
    wcs2d = _celestial_wcs(wcs)
    ref = (tuple(ref_coord) if ref_coord is not None
           else (float(wcs2d.wcs.crval[0]), float(wcs2d.wcs.crval[1])))
    x0, x1 = float(xrange[0]), float(xrange[1])
    y0, y1 = float(yrange[0]), float(yrange[1])
    xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    xt = _nice_ticks(x0, x1, max_ticks)
    yt = _nice_ticks(y0, y1, max_ticks)
    # World values along each axis (other coord held at the view center).
    ra_x, _ = wcs2d.pixel_to_world_values(xt, np.full_like(xt, yc))
    _, dec_y = wcs2d.pixel_to_world_values(np.full_like(yt, xc), yt)
    ra_x = np.atleast_1d(ra_x)
    dec_y = np.atleast_1d(dec_y)

    if coords == 'absolute':
        sp_x = abs(np.mean(np.diff(ra_x))) if len(ra_x) > 1 else 1e-4
        sp_y = abs(np.mean(np.diff(dec_y))) if len(dec_y) > 1 else 1e-4
        xtext = [_radec_str(v, sp_x, True) for v in ra_x]
        ytext = [_radec_str(v, sp_y, False) for v in dec_y]
    else:
        from ..ticks import OffsetFormatter
        fov_deg = abs(x1 - x0) * _pixel_scale_deg(wcs2d)
        unit_key = _resolve_offset_unit(offset_units, fov_deg)
        fmt_x = OffsetFormatter(ref[0], unit=unit_key, precision=precision,
                                cos_factor=np.cos(np.radians(ref[1])))
        fmt_y = OffsetFormatter(ref[1], unit=unit_key, precision=precision,
                                cos_factor=1.0)
        # Unwrap RA about the reference so a 0/360 straddle stays small.
        ra_adj = ref[0] + (((ra_x - ref[0] + 180.0) % 360.0) - 180.0)
        xtext = [fmt_x(v) for v in ra_adj]
        ytext = [fmt_y(v) for v in dec_y]

    x_minor = _minor_ticks(xt, x0, x1)
    y_minor = _minor_ticks(yt, y0, y1)
    return {
        'xaxis': {'tickvals': list(map(float, xt)), 'ticktext': xtext,
                  'minor': {'tickvals': list(map(float, x_minor))}},
        'yaxis': {'tickvals': list(map(float, yt)), 'ticktext': ytext,
                  'minor': {'tickvals': list(map(float, y_minor))}},
    }


_BEAM_CORNERS = {
    'lower left': (0.0, 0.0), 'lower right': (1.0, 0.0),
    'upper left': (0.0, 1.0), 'upper right': (1.0, 1.0),
}


def beam_shape_for_range(wcs: Any, xrange: Sequence[float],
                         yrange: Sequence[float], *, bmaj_arcsec: float,
                         bmin_arcsec: float, bpa_deg: float = 0.0,
                         corner: str = 'lower left', pad_frac: float = 0.008,
                         n: int = 72, line_color: str = 'white',
                         line_width: float = 1.2,
                         fillcolor: str = 'rgba(255,255,255,0.25)',
                         ) -> dict[str, Any]:
    """Compute a beam-ellipse plotly shape pinned to a view corner (stateless).

    Sized in pixels (so it is angularly correct under the square-pixel axes)
    and re-pinned to a corner of the *current* view, so an app's relayout
    handler can replace it each zoom. Orientation is derived from the WCS local
    North/East directions (handles CD rotation / RA flip) rather than treating
    ``bpa_deg`` as a screen angle.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
    xrange, yrange : (lo, hi)
        Current pixel ranges.
    bmaj_arcsec, bmin_arcsec : float
        Beam FWHM axes in arcseconds.
    bpa_deg : float
        Position angle (deg, N through E).
    corner : {'lower left', 'lower right', 'upper left', 'upper right'}
    pad_frac : float
        Inset from the corner, as a fraction of the current view.

    Returns
    -------
    dict
        A plotly shape dict (``type='path'``) for ``fig.add_shape(**shape)`` or
        an app's ``layout.shapes`` patch.
    """
    wcs2d = _celestial_wcs(wcs)
    if corner not in _BEAM_CORNERS:
        raise ValueError(f"corner must be one of {list(_BEAM_CORNERS)}")
    x0, x1 = sorted([float(xrange[0]), float(xrange[1])])
    y0, y1 = sorted([float(yrange[0]), float(yrange[1])])
    dx, dy = (x1 - x0), (y1 - y0)

    dpp = _pixel_scale_deg(wcs2d)
    a = 0.5 * (bmaj_arcsec / 3600.0) / dpp   # semi-major, pixels
    b = 0.5 * (bmin_arcsec / 3600.0) / dpp

    # Sample the local North / East directions near the target corner.
    fx, fy = _BEAM_CORNERS[corner]
    rcx = x0 + (0.1 if fx == 0.0 else 0.9) * dx
    rcy = y0 + (0.1 if fy == 0.0 else 0.9) * dy
    ra0, dec0 = wcs2d.pixel_to_world_values(rcx, rcy)
    ra0 = float(np.ravel(ra0)[0])
    dec0 = float(np.ravel(dec0)[0])
    eps = dpp  # ~1 pixel in degrees
    nx_, ny_ = _pix_delta(wcs2d, ra0, dec0, 0.0, eps)            # North (+dec)
    ex_, ey_ = _pix_delta(wcs2d, ra0, dec0,
                          eps / max(np.cos(np.radians(dec0)), 1e-6), 0.0)  # East
    n_hat = _unit(nx_, ny_)
    e_hat = _unit(ex_, ey_)

    pa = np.radians(bpa_deg)
    # Major axis = PA from North toward East; minor axis perpendicular.
    maj = (np.cos(pa) * n_hat[0] + np.sin(pa) * e_hat[0],
           np.cos(pa) * n_hat[1] + np.sin(pa) * e_hat[1])
    mino = (np.cos(pa + np.pi / 2) * n_hat[0] + np.sin(pa + np.pi / 2) * e_hat[0],
            np.cos(pa + np.pi / 2) * n_hat[1] + np.sin(pa + np.pi / 2) * e_hat[1])

    # Nestle into the corner by the ellipse bounding box + a small gap.
    cx, cy = _beam_corner_center(a, b, maj, mino, x0, x1, y0, y1, dx, dy,
                                 corner, pad_frac)

    t = np.linspace(0.0, 2 * np.pi, n)
    px = cx + a * np.cos(t) * maj[0] + b * np.sin(t) * mino[0]
    py = cy + a * np.cos(t) * maj[1] + b * np.sin(t) * mino[1]
    path = _ellipse_path(px, py)
    return dict(type='path', path=path, xref='x', yref='y', layer='above',
                line=dict(color=line_color, width=line_width),
                fillcolor=fillcolor, name='sph_fits_beam')


def _pix_delta(wcs2d: Any, ra0: float, dec0: float, dra: float,
               ddec: float) -> tuple[float, float]:
    """Pixel-space displacement from projecting a small (dra, ddec) world step."""
    x0, y0 = wcs2d.world_to_pixel_values(ra0, dec0)
    x1, y1 = wcs2d.world_to_pixel_values(ra0 + dra, dec0 + ddec)
    return (float(np.ravel(x1)[0]) - float(np.ravel(x0)[0]),
            float(np.ravel(y1)[0]) - float(np.ravel(y0)[0]))


def _unit(vx: float, vy: float) -> tuple[float, float]:
    norm = float(np.hypot(vx, vy)) or 1.0
    return (vx / norm, vy / norm)


def _ellipse_path(px: np.ndarray, py: np.ndarray) -> str:
    """SVG path string from sampled ellipse points."""
    parts = [f"M {float(px[0])},{float(py[0])}"]
    for x, y in zip(px[1:], py[1:]):
        parts.append(f"L {float(x)},{float(y)}")
    parts.append("Z")
    return " ".join(parts)
