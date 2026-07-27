"""Quicklook image plotters.

``quicklook_plot`` is the flagship — auto-stretches and renders any FITS
image with sensible defaults; ``quicklook_fits`` is a thin path-input
wrapper; ``simpleimageplot`` is a minimal plain-mpl variant.

Each axis plotter has a ``*_figure`` sibling that owns figure creation
(``simpleimage_figure``, ``quicklook_figure``); the axis plotter takes
``ax=None`` and is the building block users compose into multi-panel
publication figures. Both return a NamedTuple result with every
artist created internally (image, colorbar, scalebar, north arrow,
info text, contours …) so they can be adjusted after the fact.
"""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import astropy.io.fits as pyfits
import astropy.units as u  # noqa: F401
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord  # noqa: F401
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from .._colorbar import apply_minor_ticks, str_formatter, tick_text
from .._compat import _safe_ticklabel_kwargs, coord_ticks
from .._stroke import _stroke_path_effects
from ..core.fits_utils import (
    squeeze_image,
)
from ..overlays.annotations import (
    add_colorbar,
    style_ax_colors,
)
from ..ticks import (
    _auto_offset_unit,
    apply_offset_ticks,
    format_ticklabels,
)
from .levels import (
    make_norm,
)


def _check_image_kwargs_no_overlap(image_kwargs: dict[str, Any] | None,
                                   **direct: Any) -> None:
    """Raise ``TypeError`` if a key is supplied both as a direct
    named kwarg and inside ``image_kwargs``."""
    if not image_kwargs:
        return
    overlap = [k for k in direct if k in image_kwargs]
    if overlap:
        raise TypeError(
            f"keyword(s) {overlap!r} supplied both as a named "
            "argument and inside image_kwargs. Pass each option "
            "in exactly one place."
        )


# ===== simpleimageplot =====

class SimpleImageResult(NamedTuple):
    """Return type for :func:`simpleimageplot` and
    :func:`simpleimage_figure`. Tuple-unpackable
    (``fig, ax, im, cbar = simpleimageplot(...)``) and
    attribute-accessible (``result.colorbar.set_label(...)``)."""
    fig:      object  # matplotlib.figure.Figure
    ax:       object  # WCSAxes
    image:    object  # AxesImage
    colorbar: object  # matplotlib.colorbar.Colorbar | None


def simpleimageplot(image_arr: npt.ArrayLike, ax: Any = None, *,
                    axtitle: str = '', cmap: Any = 'gist_yarg',
                    tickcolor: str = 'w', labelcolor: str = 'k',
                    minorticks: bool = True, colorbar: bool = False,
                    cbar_label: str | None = None,
                    image_kwargs: dict[str, Any] | None = None
                    ) -> SimpleImageResult:
    """Plot a single 2D image (or RGB array) onto an existing WCSAxes.

    Pure axis-plotter — does not create a figure. Use
    :func:`simpleimage_figure` for a one-line figure-builder
    convenience that owns ``hdrin`` (FITS header → WCS), ``figsize``,
    ``dpi``, ``facecolor``, ``savepath`` and delegates here.

    Parameters
    ----------
    image_arr : ndarray
        2D image or RGB array.
    ax : WCSAxes, optional
        Target axes. If ``None``, creates a 5×5 figure with a plain
        ``Axes`` (no WCS). For WCS-aware tick formatters, pre-create
        a WCSAxes via ``make_wcs_frame`` (or
        ``simpleimage_figure``) and pass it here.
    axtitle : str
        Axes title.
    cmap : str or Colormap
    tickcolor, labelcolor : str
        Tick line and label colors (typical: white ticks on dark
        images, black labels on white background).
    minorticks : bool
        Show minor tick marks.
    colorbar : bool
        Add a colorbar (default ``False`` — simple plots typically
        omit it).
    cbar_label : str, optional
    image_kwargs : dict, optional
        Extra kwargs for ``ax.imshow``. Collision with direct kwargs
        (``cmap``) raises ``TypeError``.

    Returns
    -------
    SimpleImageResult
        NamedTuple ``(fig, ax, image, colorbar)``.
    """
    image_kwargs = dict(image_kwargs) if image_kwargs else {}
    _check_image_kwargs_no_overlap(image_kwargs, cmap=cmap)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    im = ax.imshow(image_arr, origin='lower', interpolation='nearest',
                   cmap=cmap, **image_kwargs)
    ax.set_title(axtitle, color=labelcolor, size=12)

    # WCSAxes-specific tick formatting (skip if plain Axes).
    if hasattr(ax, 'coords'):
        ax.coords.frame.set_color(tickcolor)
        for i in (0, 1):
            ax.coords[i].display_minor_ticks(minorticks)
            ax.coords[i].set_ticks(number=6, size=8, color=tickcolor)
            ax.coords[i].set_ticklabel(size=10, color=labelcolor)
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.coords[1].set_major_formatter('dd:mm:ss')
        ax.coords[0].set_separator(
            (r'$^\mathregular{H}$', "'", '"'))

    cbar = None
    if colorbar:
        cbar = add_colorbar(im, ax=ax, label=cbar_label)
        # Match the colorbar's tick labels + axis label to the figure's text
        # color, so they don't keep add_colorbar's black default and vanish on
        # a dark facecolor. Use labelcolor (the margin-text color), NOT
        # tickcolor — that one is white for ticks drawn OVER the dark image.
        cbar.ax.tick_params(labelcolor=labelcolor)
        cbar.ax.yaxis.label.set_color(labelcolor)

    return SimpleImageResult(fig=fig, ax=ax, image=im, colorbar=cbar)


def simpleimage_figure(image_arr: npt.ArrayLike, hdrin: pyfits.Header, *,
                       figsize: tuple[float, float] | None = None,
                       dpi: int = 150, facecolor: str = 'w',
                       savepath: str | None = None,
                       **plot_kwargs: Any) -> SimpleImageResult:
    """Quick single-image figure builder — uses a FITS header to
    set up a WCSAxes, plots the image via :func:`simpleimageplot`,
    and optionally saves the figure.

    Parameters
    ----------
    image_arr : ndarray
    hdrin : astropy.io.fits.Header
        FITS header (used to build the WCS for the axes).
    figsize : tuple, optional
    dpi : int
    facecolor : str
        Figure background color.
    savepath : str, optional
        If given, save the figure to this path.
    **plot_kwargs
        Forwarded to :func:`simpleimageplot` (``axtitle``, ``cmap``,
        ``tickcolor``, ``labelcolor``, ``minorticks``, ``colorbar``,
        ``image_kwargs`` …).

    Returns
    -------
    SimpleImageResult
        ``(fig, ax, image, colorbar)``.

    Examples
    --------
    >>> import skyplothelper as sph
    >>> res = sph.simpleimage_figure(image, header, cmap='sph.deepsky',
    ...                              colorbar=True)
    >>> res.ax, res.image                     # the WCSAxes + AxesImage
    """
    fig = plt.figure(figsize=figsize)
    fig.set_facecolor(facecolor)
    wcs = WCS(hdrin)
    ax = fig.add_subplot(111, projection=wcs)
    result = simpleimageplot(image_arr, ax=ax, **plot_kwargs)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight', dpi=dpi,
                    facecolor=fig.get_facecolor())

    return result



# ===== Quicklook helpers =====

# The light/dark ink pair for marks drawn over an image, used as a PAIR --
# whichever is the ink, the other is the stroke -- so the stroke is always the
# ink's opposite. A fixed stroke color cannot work here, since the ink itself
# flips with the colormap and a dark stroke behind dark ink does nothing.
#
# The light half stays pure white deliberately. A warm off-white (#D9D5C5 was
# tried) looks better on the cool maps but blends on the warm ones: nebula and
# deepsky both reach (0.925, 0.900, 0.742) at their bright end, so cream
# contours crossing a bright core are cream on cream -- contrast 1.06, dE 13.5.
# A neutral '0.9' is worse still on those maps, not better: their bright end
# sits at luminance ~0.897, so it is nearly luminance-matched (contrast 1.01).
# White is the only light ink that clears every bundled colormap's bright end.
INK_LIGHT = 'w'
# The dark half can afford the softening the light half cannot: it is selected
# for reversed maps, whose relevant regions are light, so there is no
# equivalent collision to design around.
INK_DARK = '#1A1A1A'

# The beam is drawn in a fixed corner, so unlike the contours it is not
# reliably over the image at all -- on a masked map that corner is the page.
# A neutral mid-gray reads on either, and distinguishes the beam (an
# instrument property) from the contours (the data).
BEAM_INK = '0.65'


def _over_image_ink(colormap: Any, drawing_image: bool,
                    fallback: Any) -> Any:
    """Ink for marks drawn ON the image (contours, beam).

    Sampled from the colormap's lower quartile rather than hard-coded white:
    that is where most contour length sits, and it flips correctly for the
    reversed maps common in optical work (``gray_r``, ``Blues``), where a
    light ink would be invisible. With no image beneath, the caller's own
    color is right and is returned unchanged.
    """
    if not drawing_image:
        return fallback
    try:
        cmap = (plt.get_cmap(colormap) if isinstance(colormap, str)
                else colormap)
        r, g, b = cmap(0.25)[:3]
    except Exception:
        return fallback
    return (INK_LIGHT if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5
            else INK_DARK)


def _contrast_ink(color: Any, fallback: Any = INK_DARK) -> Any:
    """The other half of the ink pair — whichever of :data:`INK_LIGHT` /
    :data:`INK_DARK` contrasts with *color*.

    Used to resolve a stroke against its own ink, for both the contours and
    the beam. Without this the two could coincide: the ink flips with the
    colormap, so a fixed dark stroke sat behind dark ink on every reversed
    map and did nothing.
    """
    try:
        r, g, b = mcolors.to_rgb(color)
    except Exception:
        return fallback
    return (INK_DARK if (0.299 * r + 0.587 * g + 0.114 * b) > 0.5
            else INK_LIGHT)


def _estimate_rms(image_data: np.ndarray,
                  corner_fraction: float = 0.1) -> float:
    """Estimate RMS noise from image corners (source-free regions)."""
    h, w = image_data.shape
    cs = max(min(int(h * corner_fraction), int(w * corner_fraction)), 2)
    corners = np.concatenate([
        image_data[:cs, :cs].ravel(),
        image_data[:cs, -cs:].ravel(),
        image_data[-cs:, :cs].ravel(),
        image_data[-cs:, -cs:].ravel(),
    ])
    return float(np.nanstd(corners[np.isfinite(corners)])) if len(corners) > 0 else 0.


def _estimate_background(image_data: np.ndarray,
                        corner_fraction: float = 0.1) -> float:
    """Estimate the background (sky) *level* from the image corners.

    ``_estimate_rms`` measures the corner *noise* (std); this measures the
    corner *median*, i.e. where the empty sky sits. Radio/interferometric maps
    are zero-mean so this is ~0, but an optical/IR frame on a bright sky
    pedestal (a DSS or survey cutout) has a large positive level here. Anchoring
    the display ``vmin`` at this level rather than at zero is what keeps the
    empty sky black instead of landing mid-colormap and washing the frame out.
    Uses the median (not mean) so a few bright pixels in a corner don't bias it.
    """
    h, w = image_data.shape
    cs = max(min(int(h * corner_fraction), int(w * corner_fraction)), 2)
    corners = np.concatenate([
        image_data[:cs, :cs].ravel(),
        image_data[:cs, -cs:].ravel(),
        image_data[-cs:, :cs].ravel(),
        image_data[-cs:, -cs:].ravel(),
    ])
    corners = corners[np.isfinite(corners)]
    return float(np.median(corners)) if len(corners) > 0 else 0.


def _stat_fmt(value: float) -> str:
    """Pick a format spec for an info-block statistic from its own magnitude.

    Peak and RMS are formatted independently: a bright peak (e.g. 3 Jy/beam)
    and a sub-mJy RMS live many orders of magnitude apart, so a single
    peak-derived fixed-decimal format truncates the RMS to ``0.000``. Values
    at or above 0.01 keep the familiar fixed-decimal ladder; below that we
    switch to 3 significant figures so faint RMS/levels stay readable.
    """
    a = abs(value)
    if a >= 100:
        return '.1f'
    if a >= 1:
        return '.3f'
    if a >= 0.01:
        return '.4f'
    return '.3g'


def _auto_band_identifier(header: pyfits.Header | None) -> str:
    """
    Auto-detect observation band/frequency/wavelength from FITS header.

    Checks CRVAL3+CTYPE3 (frequency/wavelength axis), WAVELEN, FILTER,
    and BAND header cards. Returns a human-readable string or ''.
    """
    if header is None:
        return ''

    # Try frequency axis (radio/mm/sub-mm)
    ctype3 = header.get('CTYPE3', '').strip().upper()
    crval3 = header.get('CRVAL3', None)

    if crval3 is not None and ('FREQ' in ctype3 or crval3 > 1e6):
        freq_hz = float(crval3)
        if freq_hz > 1e12:
            return f'{freq_hz/1e12:.2f} THz'
        elif freq_hz > 1e9:
            return f'{freq_hz/1e9:.2f} GHz'
        elif freq_hz > 1e6:
            return f'{freq_hz/1e6:.1f} MHz'
        elif freq_hz > 1e3:
            return f'{freq_hz/1e3:.1f} kHz'
        return f'{freq_hz:.0f} Hz'

    # Try wavelength axis
    if crval3 is not None and 'WAVE' in ctype3:
        wave = float(crval3)
        cunit3 = header.get('CUNIT3', 'm').strip().lower()
        # Convert to meters
        if cunit3 == 'angstrom' or cunit3 == 'a':
            wave_m = wave * 1e-10
        elif cunit3 == 'nm':
            wave_m = wave * 1e-9
        elif cunit3 == 'um' or cunit3 == 'micron':
            wave_m = wave * 1e-6
        elif cunit3 == 'mm':
            wave_m = wave * 1e-3
        elif cunit3 == 'cm':
            wave_m = wave * 1e-2
        else:
            wave_m = wave  # assume meters

        if wave_m < 1e-9:
            return f'{wave_m*1e10:.1f} Å'
        elif wave_m < 1e-6:
            return f'{wave_m*1e9:.1f} nm'
        elif wave_m < 1e-3:
            return f'{wave_m*1e6:.1f} μm'
        elif wave_m < 1:
            return f'{wave_m*1e3:.1f} mm'
        return f'{wave_m*1e2:.1f} cm'

    # Try common optical/IR header cards
    filt = header.get('FILTER', header.get('FILTER1', ''))
    if filt:
        return str(filt).strip()

    wavelen = header.get('WAVELEN', header.get('WAVELENG', None))
    if wavelen is not None:
        wave_m = float(wavelen)
        # Guess units from magnitude
        if wave_m > 100:  # likely Angstroms
            return f'{wave_m:.0f} Å'
        elif wave_m > 0.1:  # likely microns
            return f'{wave_m:.2f} μm'
        else:  # meters
            return f'{wave_m*1e9:.1f} nm'

    band = header.get('BAND', '')
    if band:
        return str(band).strip()

    return ''




# ===== Flagship plotter =====

class QuicklookResult(NamedTuple):
    """Return type for :func:`quicklook_plot` and
    :func:`quicklook_figure`.

    Tuple-unpackable (``fig, ax, image, cbar, contour_set,
    neg_contour_set, info_text = quicklook_plot(...)``) and
    attribute-accessible. Every internally-created artist is
    reachable so the user can adjust it after the fact.
    """
    fig:               object  # matplotlib.figure.Figure
    ax:                object  # WCSAxes
    image:             object  # AxesImage (None if image=False)
    colorbar:          object  # Colorbar (None if colorbar=False)
    contour_set:       object  # ContourSet (None if contours=False)
    neg_contour_set:   object  # ContourSet (None if no negative contours)
    info_text:         object  # Text (None if show_info=False)


def quicklook_plot(image_data: Any, ax: Any = None, *,
                   header: pyfits.Header | None = None, wcs: Any = None,
                   # -- Annotation ---------------------------------------------
                   source_name: str | None = None, obs_date: str | None = None,
                   label: str | None = None, show_info: bool = True,
                   info_color: str | None = None,
                   # -- Statistics (None = auto-compute from data) -------------
                   peak: float | None = None, rms: float | None = None,
                   unit: str | None = None,
                   # -- Beam ---------------------------------------------------
                   beam_maj: float | None = None, beam_min: float | None = None,
                   beam_pa: float | None = None,
                   beam_style: str = 'crosshair',
                   # -- Contours -----------------------------------------------
                   contours: bool = True, levels: npt.ArrayLike | None = None,
                   contour_start: float = 3, contour_factor: float = 2,
                   negative_contours: bool = True, n_negative: int | None = 1,
                   cbar_format: Any = None, cbar_minor_ticks: Any = None,
                   color: str = 'k', contour_cmap: Any = None,
                   contour_color: Any = None, contour_alpha: float = 0.9,
                   contour_stroke_color: Any = 'auto',
                   contour_stroke_lw: float = 0.7,
                   beam_color: Any = None,
                   beam_stroke_color: Any = 'auto',
                   beam_stroke_lw: float = 1.4,
                   contour_lw: float | str = 0.5,
                   contour_labelstyle: str = 'RMS',
                   # -- Image / colorbar ---------------------------------------
                   image: bool = True, colorbar: bool = True,
                   colormap: Any = 'sph.deepsky',
                   norm: Any = None, stretch: str | None = None,
                   vmin: float | None = None, vmax: float | None = None,
                   display_factor: float = 1.,
                   # -- Axes / coordinates -------------------------------------
                   offset_coords: bool = False, ref_coord: Any = None,
                   offset_units: str = 'mas',
                   field_size: float | None = None,
                   # -- Grid ---------------------------------------------------
                   grid: bool = False, gridcolor: str = '0.3',
                   gridalpha: float = 0.5,
                   # -- Appearance ---------------------------------------------
                   tick_style: str = 'publication',
                   mpl_style: str | None = 'professional',
                   frame_color: str | None = None,
                   frame_stroke: Any = None,
                   figure_font: str = 'DejaVu Sans',
                   facecolor: str = 'w', axcolor: str = 'k',
                   **kwargs: Any) -> QuicklookResult:
    """
    Create an informative quick-look plot of a FITS image.

    Produces a self-contained figure with the image (contours and/or
    pixel map), offset coordinate axes, and metadata annotations:
    source name, date, label, peak/RMS statistics, contour
    levels, beam size, and reference position.

    Parameters
    ----------
    image_data : str or 2D ndarray
        FITS filename (str) or 2D image array. Multi-dimensional
        FITS data cubes are squeezed to 2D.
    header : astropy Header, optional
        FITS header for metadata extraction.
    wcs : astropy WCS, optional
        WCS object. Created from header if not provided.
    source_name : str, optional
        Source name (overrides OBJECT header card).
    obs_date : str, optional
        Observation date (overrides DATE-OBS).
    label : str, optional
        Label to print above top left frame corner (e.g. '8.4 GHz',
        'F814W', '2.2 μm'). If 'auto', attempts to auto-detect
        from CRVAL3/CTYPE3, FILTER, WAVELEN, or BAND header cards.
        Set to '' to suppress.
    show_info : bool
        Show metadata text below the plot.
    info_color : str, optional
        Color for all text annotations (label, and info text below
        the plot). Defaults to ``color``. Set explicitly when using
        light contour colors (e.g., ``color='w', info_color='k'``).
    peak, rms : float, optional
        Peak brightness and RMS noise. Auto-computed if None.
    unit : str, optional
        Brightness unit label. When omitted, ``BUNIT`` from *header* is
        used if present (with the display factor folded into an SI prefix
        for Jy-family units), otherwise the radio ``Jy/beam`` family is
        assumed.
    beam_maj, beam_min : float, optional
        Beam axes in degrees. Read from BMAJ/BMIN if not provided.
        Fits standard for BMAJ/BMIN is FWHM in degrees.
        Set to 0 to suppress beam display.
    beam_pa : float, optional
        Beam PA in degrees (overrides BPA header card).
        Fits standard for BPA is degrees E from N (CCW from up).
    beam_style : str
        Beam display style: 'filled' (solid fill), 'crosshair'
        (outline with major/minor axis lines, publication standard),
        'hatch' (hatched fill). Default 'crosshair'. Add the term
        'grid' to the style ('filledgrid', 'crosshairgrid') to additionally
        add a fine grid-style hatch, which can be useful for further
        differentiating the beam ellipse from background contours.
    contours : bool
        Draw contours. Default True.
    levels : array-like, optional
        Explicit contour levels (overrides contour_start/contour_factor).
    contour_start : float
        First contour as multiple of RMS. Default 3.
    contour_factor : float
        Geometric contour spacing factor. Default 2.
    negative_contours : bool
        Show dashed negative contours. Default True.
    cbar_format : str or Formatter, optional
        Colorbar tick label format. ``None`` (default) picks the decimal
        precision from the displayed range, so a 0-3 Jy bar no longer
        collapses to "0 1 2 3". A ``'%.2f'``-style string or a matplotlib
        Formatter both work.
    cbar_minor_ticks : sequence, Locator, False, optional
        Colorbar minor ticks. ``None`` (default) subdivides evenly on a
        linear bar and uses 1/2/3/5 x 10^k multiples on a compressed
        (log / sqrt / asinh) one, spanning whatever decades the data
        occupies. Pass positions or a Locator to set them, or ``False`` to
        remove them.
    n_negative : int or None
        Maximum number of negative contour levels. Default 1; ``None``
        draws as many as the data's minimum admits. Only applies to the
        automatic geometric sequence, not to explicit ``levels``.
    color : str
        Fallback color for contours, beam, and annotation text. The
        annotation text is drawn *outside* the axes, so this stays dark by
        default; marks drawn *on* the image pick their own ink (see
        ``contour_color`` / ``beam_color``). Ignored for contours if
        ``contour_cmap`` is set.
    contour_color : color, optional
        Contour color. ``None`` (default) samples the image colormap's lower
        quartile — where most contour length sits — and picks white over a
        dark map (``sph.deepsky``, ``viridis``) or black over a reversed one
        (``gray_r``, ``Blues``), so contours stay visible either way. With
        ``image=False`` there is nothing to read against and ``color`` is
        used unchanged.
    contour_alpha : float
        Contour opacity. Default 0.9, so the image beneath is not drowned
        out. Pass 1.0 for solid lines.
    contour_stroke_color, contour_stroke_lw : optional
        A very fine stroke behind the contours, keeping them readable where
        they cross the bright end of a through-black colormap. Defaults
        ``'auto'`` and 0.7. ``'auto'`` resolves to whichever of
        :data:`INK_LIGHT` / :data:`INK_DARK` contrasts with the contour ink
        actually in use — necessary because that ink itself flips with the
        colormap, so a fixed stroke color would coincide with it on reversed
        maps. Note ``stroke_lw`` is the TOTAL width, so the visible edge per
        side is ``(stroke_lw - contour_lw) / 2`` — 0.1 at the defaults. Pass
        ``contour_stroke_color=None`` to disable.
    beam_color : color, optional
        Beam ellipse color. ``None`` (default) uses :data:`BEAM_INK`, a
        neutral mid-gray, whenever an image is drawn — distinct from the
        contour ink on purpose, since the beam is an instrument property
        rather than data, and since the beam's corner is not reliably over
        the image at all. With ``image=False`` it follows the contour ink.
    beam_stroke_color, beam_stroke_lw : optional
        Legibility stroke behind the beam, defaulting to ``'auto'`` and 1.4.
        The beam is drawn in a fixed corner rather than on the data, so on a
        masked map — a moment map, a cutout with blank sky,
        ``facecolor='none'`` — that corner is transparent and ink chosen for
        the image can match the page exactly, making the beam disappear.
        ``'auto'`` contrasts against the beam's own ink; pass a color to
        override, or ``None`` to disable the stroke.
    contour_cmap : str or Colormap, optional
        Colormap for contours — levels are colored by value. Overrides
        ``color``. Any valid matplotlib colormap name or object.
    contour_lw : float or str
        Contour linewidth. A float applies uniformly. The string
        ``'scaled'`` makes linewidth increase with contour level
        (0.4 at lowest → 1.6 at highest), a common radio convention.
    contour_labelstyle : str (``'actual'`` or ``'RMS'``)
        Style for printing contour levels in lower info block.  'actual'
        will print the actual values (e.g., "5.64, 11.3, 22.5,...") while
        'RMS' will print multiples of RMS (e.g., "RMS (1.88 mJy/bm) x 1, 2, 4, ...")
    image : bool
        Show the underlying pixel image. Default True. Pass ``False`` for
        the classic contour-only (difmap-style) rendering; the colorbar
        then drops out too, since there is nothing to scale.
    colormap : str or Colormap
        Colormap for the pixel image. Default ``'sph.deepsky'`` — a bundled
        through-black skyplothelper map (see :mod:`skyplothelper.colormaps`)
        for a unified look; any matplotlib colormap name or object works too.
    colorbar : bool
        Show a colorbar. Default True — a colormapped image without a scale
        is only half a plot. Automatically skipped when ``image=False``,
        since there is then no mappable to describe.
    norm : matplotlib Normalize, optional
        Image normalization (e.g. ``SymLogNorm``, ``LogNorm``,
        ``PowerNorm``). Applied to imshow when ``image=True``.
    stretch : str, optional
        Shorthand for common normalizations (avoids importing
        ``matplotlib.colors``): 'log', 'sqrt', 'asinh', 'symlog',
        'power2'. Ignored if ``norm`` is explicitly set.
    vmin, vmax : float, optional
        Explicit color limits for imshow **in display units** (i.e. after
        *display_factor* scaling).  Bypass auto-stretch when set.
        If None, auto-scaled.
    display_factor : float
        Multiplicative factor from native to display units.  Common choices:
        ``1e3`` (Jy/beam → mJy/beam), ``1e6`` (Jy/beam → µJy/beam).
    offset_coords : bool
        Replace absolute RA/Dec ticks with an offset axis centered on
        *ref_coord* (or the image center).  Requires *header* or *wcs*.
    ref_coord : `~astropy.coordinates.SkyCoord`, optional
        Reference position for offset axes.  Defaults to image center.
    offset_units : str
        Angular unit for offset ticks ('mas', 'arcsec', …).
    field_size : float or None
        Crop the axes to this width (in offset_units when offset, else mas).
    grid : bool
        Show coordinate grid.
    gridcolor : str
        Color for overlay grid
    gridalpha : float
        Alpha transparency value for overlay grid, in range [0,1]
    tick_style : str
        Tick label formatting style passed to ``format_ticklabels()``.
    mpl_style : str or None
        Matplotlib RC style to apply locally. 'professional' applies
        publication-quality settings (inward ticks, minor ticks, clean
        fonts). 'default' or None uses current rcParams. Any valid
        matplotlib style name also works (e.g. 'seaborn-v0_8').
    frame_color : color, optional
        Color for the spines and tick *marks* (tick labels / title are
        unaffected). If None (default), the frame is black in contour mode
        and **medium gray ('0.5')** in image mode — so it stays visible over
        a filled colormap that runs through black (the ``image=True``
        ``sph.deepsky`` default) without a hardcoded light color that would
        fail in dark mode.
    frame_stroke : dict or color or None, optional
        Opt-in stroke around the frame + tick marks (off by default; the gray
        ``frame_color`` above is the safe default). Pass:

        * a **dict** of stroke params, e.g. ``{'color': 'white', 'lw': 1.6}``
          — keys ``'color'`` (default white) and ``'lw'`` (default 1.6, a thin
          ~0.4-pt-per-side stroke). Use a dark stroke for dark-mode figures.
        * a bare color (e.g. ``'white'``) — shorthand for ``{'color': ...}``.
        * ``None`` (default) — no stroke.

        The stroke is one continuous rectangle for the frame (clean corners)
        plus an end-wrapping two-pass on the native tick marks; tick labels
        are left unstroked. Applied via the public
        :func:`~skyplothelper.apply_frame_stroke`.
    figure_font : str
        Font family for all text.
    facecolor : str
        Any valid matplotlib color string.  If set to 'none', make background
        transparent (useful for presentations).
    axcolor : str
        Any valid matplotlib color string.  The color to make the axis elements
        (spine, text, etc).  Useful to set it to white for black facecolor.
    **kwargs
        Additional kwargs passed to ``ax.contour()``.

        For figure-level control — ``figsize``, ``output_file``, ``dpi`` —
        use :func:`quicklook_figure`, which owns figure creation and wraps
        this function.

    Returns
    -------
    result : QuicklookResult
        NamedTuple ``(fig, ax, image, colorbar, contour_set,
        neg_contour_set, info_text)`` — tuple-unpackable and
        attribute-accessible. Each artist field is ``None`` when its
        feature is disabled.

    Notes
    -----
    **Offset coordinates:** When WCS is available, the axes show offsets
    from the image center in auto-detected units (arcmin, arcsec, mas,
    or μas based on field of view). The center coordinates are shown in
    the info text below the plot.

    **Band auto-detection:** Checks header cards in order: CRVAL3+CTYPE3
    (frequency → GHz/MHz; wavelength → nm/μm), FILTER/FILTER1 (optical
    filter names), WAVELEN/WAVELENG (wavelength), BAND. Covers radio
    through X-ray FITS conventions.

    **Beam styles:** 'crosshair' (default) draws the beam outline with
    lines along the major and minor axes — the standard for published
    VLBI/radio maps. 'filled' uses a solid fill. 'hatch' uses diagonal
    line hatching.

    Examples
    --------
    >>> # Classic radio contour plot
    >>> result = sph.quicklook_plot('source.fits', color='#2ca02c')

    >>> # Optical image with colorbar
    >>> result = sph.quicklook_plot('hst_image.fits', contours=False,
    ...     image=True, colorbar=True, colormap='gray_r')

    >>> # Colormap-colored contours on colormap background
    >>> result = sph.quicklook_plot(data, header=hdr,
    ...     image=True, colormap='magma', contour_cmap='cool',
    ...     stretch='symlog')

    >>> # Scaled linewidths (thicker at higher contours)
    >>> result = sph.quicklook_plot('source.fits', contour_lw='scaled')

    >>> # Explicit contour levels
    >>> result = sph.quicklook_plot(data, header=hdr,
    ...     levels=[0.001, 0.002, 0.005, 0.01, 0.02])

    >>> # Log-stretch background with white contours
    >>> result = sph.quicklook_plot(data, header=hdr,
    ...     image=True, stretch='log', colormap='inferno',
    ...     color='w', info_color='0.3')
    """
    # A MomentMap carries its own data + WCS + header + order; recognize it and
    # apply moment-appropriate defaults (order-based colormap, a "Moment N"
    # corner label, a centered diverging range for the velocity field, and no
    # misleading peak/rms). Runs before the colormap resolution below so the
    # order default feeds it. Lazy import keeps quicklook <-> cube decoupled.
    from .cube import _MOMENT_CMAP, MomentMap
    if isinstance(image_data, MomentMap):
        mm = image_data
        if header is None:
            header = mm.header
        if wcs is None and mm.wcs is not None:
            wcs = mm.wcs
        if unit is None:
            unit = mm.units
        image = True
        contours = False
        if colormap == 'sph.deepsky':
            colormap = _MOMENT_CMAP.get(mm.order, 'sph.deepsky')
        if label is None:
            label = f"Moment {mm.order}"                   # compact corner tag
        finite = mm.data[np.isfinite(mm.data)]
        if mm.order == 1 and vmin is None and vmax is None and norm is None:
            if finite.size:
                mid = float(np.median(finite))
                half = float(np.percentile(np.abs(finite - mid), 99)) or 1.0
                vmin, vmax = mid - half, mid + half        # white at systemic
        if mm.order in (1, 2):
            show_info = False        # peak/rms are meaningless for v-field/σ
        # Supply stats from the finite pixels so quicklook skips its sigma-clip
        # RMS estimate — moment maps are NaN-heavy (masked), which trips a
        # spurious "ddof <= 0" warning in that estimator.
        if finite.size > 1:
            if peak is None:
                peak = float(np.nanmax(finite))
            if rms is None:
                rms = float(np.nanstd(finite))
        image_data = mm.data

    # Resolve a bundled ``'sph.*'`` colormap name to its Colormap object via
    # the registration-independent path (colormaps.get_colormap falls back to
    # the LUT-built table), so the ``'sph.deepsky'`` default — and any sph.*
    # name a caller passes — renders even if matplotlib's colormap registry
    # wasn't populated. Non-sph names / Colormap objects pass straight through
    # to imshow's own resolution.
    if isinstance(colormap, str) and colormap.startswith('sph.'):
        from ..colormaps import get_colormap
        try:
            colormap = get_colormap(colormap)
        except (KeyError, ValueError):
            pass  # unknown sph.* name — let matplotlib raise its own error

    # Offset reference: the caller's ``ref_coord`` when given, else the image
    # center (set in the WCS block below). Stays None only when there is no
    # WCS (the offset-label branch then short-circuits).
    ref_coord_used = None

    _UNIT_LABELS = {
        1:    'Jy/beam',
        1e3:  'mJy/beam',
        1e6:  r'$\mu$Jy/beam',
        1e9:  'nJy/beam',
    }

    _OFFSET_UNIT_FACTORS = {
        'deg': 1., 'degree': 1., 'degrees': 1.,
        'arcmin': 60., 'amin': 60.,
        'arcsec': 3600., 'asec': 3600.,
        'mas': 3600e3, 'milliarcsec': 3600e3,
        'uas': 3600e6, 'microarcsec': 3600e6,
    }


    _SI_PREFIX = {1: '', 1e3: 'm', 1e6: r'$\mu$', 1e9: 'n'}

    def _unit_label(imfactor: float, bunit: str | None = None) -> str:
        """Colorbar / info-text unit string for a given display factor.

        With a ``BUNIT`` from the header, prefix that instead of assuming the
        radio default — an optical frame in ``electron/s`` should not be
        labeled Jy/beam. Falls back to the radio table when the header says
        nothing, which is the historical behavior.
        """
        if bunit:
            prefix = _SI_PREFIX.get(imfactor)
            if prefix is not None and bunit[:1].upper() == 'J':
                # Jy-family: fold the factor into an SI prefix (Jy -> mJy).
                return f'{prefix}{bunit}'
            if imfactor == 1:
                return bunit
            return f'[x{imfactor:.0e}] {bunit}'
        # No BUNIT and no unit= from the caller: assert NOTHING. This used to
        # fall through to the radio Jy/beam family, which labeled any survey
        # cutout without a BUNIT -- DSS2, AllWISE -- with a unit its data never
        # claimed. A wrong unit in a published figure is worse than no unit,
        # and an unlabeled bar still carries the stretch and the range, which
        # is what turning it on by default was for.
        #
        # The display factor is kept when there is one: that is a fact about
        # the numbers on the bar, true whatever the unit turns out to be.
        return '' if imfactor == 1 else f'[x{imfactor:.0e}]'

    # --- Load data ---
    # Unpack tuple/list inputs: (data, header) or (data, wcs)
    if isinstance(image_data, (tuple, list)) and len(image_data) == 2:
        _item0, _item1 = image_data
        if isinstance(_item0, np.ndarray) or hasattr(_item0, 'shape'):
            image_data = _item0
            if isinstance(_item1, WCS):
                if wcs is None:
                    wcs = _item1
            elif header is None:
                header = _item1

    if isinstance(image_data, str):
        with pyfits.open(image_data) as hdul:
            hdr = hdul[0].header
            img = hdul[0].data
            if header is None:
                header = hdr
        image_data = img

    # Squeeze to 2D if needed (removes degenerate freq/Stokes axes)
    image_data = np.asarray(image_data, dtype=float)
    if image_data.ndim != 2:
        image_data, header_squeezed = squeeze_image(
            image_data, header=header, verbose=False)
        if header_squeezed is not None:
            header = header_squeezed

    if image_data.ndim != 2:
        raise ValueError(f"image_data must be 2D, got shape {image_data.shape}")

    # --- Build WCS if needed ---
    if wcs is None and header is not None:
        wcs = WCS(header, naxis=2)

    # --- Extract metadata from header ---
    if header is not None:
        if source_name is None:
            source_name = header.get('OBJECT', '')
        if obs_date is None:
            raw_date = header.get('DATE-OBS', '')
            obs_date = raw_date[:10] if len(raw_date) > 10 else raw_date
        if beam_maj is None:
            beam_maj = header.get('BMAJ', None)
        if beam_min is None:
            beam_min = header.get('BMIN', None)
        if beam_pa is None:
            beam_pa = header.get('BPA', 0.)

    # Resolve the auto band label from the header (empty if unavailable).
    if label == 'auto':
        label = _auto_band_identifier(header)

    # --- Compute statistics (in native units) ---
    if peak is None:
        peak = float(np.nanmax(image_data))
    if rms is None:
        rms = _estimate_rms(image_data)
    if rms == 0 or not np.isfinite(rms):
        rms = float(np.nanstd(image_data)) * 0.1
    if rms == 0 or not np.isfinite(rms):
        # All-zero / constant / all-NaN input — avoid downstream div-by-zero.
        rms = 1.0

    # --- Brightness unit + scaling ---
    # display_factor is the single numeric scale, applied once here; every
    # downstream value (colorbar, contours, info text) is in these display
    # units. The label is the explicit ``unit=`` if given, else derived from
    # display_factor (1 -> 'Jy/beam', 1e3 -> 'mJy/beam', ...).
    display_data = image_data * display_factor
    display_peak = peak * display_factor
    display_rms  = rms  * display_factor
    # Background (sky) level for the default display floor — 0 for zero-mean
    # radio maps, the pedestal for optical/IR survey cutouts. Only steers the
    # auto image vmin; peak/rms (contours, info text) are untouched.
    display_bkg = _estimate_background(image_data) * display_factor
    _bunit = None
    if unit is None and header is not None:
        _raw_bunit = str(header.get('BUNIT', '') or '').strip()
        _bunit = _raw_bunit or None
    unit_label = (unit if unit is not None
                  else _unit_label(display_factor, _bunit))
    # Pre-spaced suffix. unit_label is '' when the data declares no unit, and
    # the info-text lines below would otherwise read 'Peak = 4.001 ' with a
    # dangling space, or a contour line ending '(0.003 )'.
    _u = f' {unit_label}' if unit_label else ''

    # --- Compute offset coordinate info ---
    center_ra_str = ''
    center_dec_str = ''
    offset_unit_label = 'arcsec'
    offset_scale = 1.0  # arcsec per degree on offset axis

    if wcs is not None:
        ny, nx = display_data.shape
        cx, cy = nx / 2., ny / 2.
        center_world = wcs.pixel_to_world_values(cx, cy)
        center_ra_deg = float(center_world[0])
        center_dec_deg = float(center_world[1])

        # Format center coordinates
        center_coord = SkyCoord(center_ra_deg, center_dec_deg, unit='deg')
        center_ra_str = center_coord.ra.to_string(unit=u.hourangle, sep='hms',
                                                    precision=4, pad=True)
        center_dec_str = center_coord.dec.to_string(unit=u.deg, sep='dms',
                                                     precision=3, alwayssign=True,
                                                     pad=True)

        # Determine field of view and offset unit
        try:
            pix_scales = proj_plane_pixel_scales(wcs)
            fov_deg = max(pix_scales[0] * nx, pix_scales[1] * ny)
            offset_scale, offset_unit_label = _auto_offset_unit(fov_deg)  # noqa: F841 (offset_scale kept for downstream)
        except Exception:
            fov_deg = 1.0
            offset_scale, offset_unit_label = 1.0, 'arcsec'  # noqa: F841

        # Offset reference + display unit (used by the offset tick / label /
        # crop paths below). ``offset_units='auto'`` falls back to the
        # FOV-derived unit; otherwise the explicit choice wins.
        ref_coord_used = ref_coord if ref_coord is not None else center_coord
        offset_disp_unit = (offset_unit_label if offset_units == 'auto'
                            else offset_units)

    # --- Apply matplotlib style ---
    _professional_params = {
        'font.family': 'sans-serif',
        'font.size': 11,
        'mathtext.fontset': 'cm',
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'xtick.major.size': 7, 'ytick.major.size': 7,
        'xtick.minor.size': 3, 'ytick.minor.size': 3,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'axes.linewidth': 0.8,
        'contour.negative_linestyle': 'dashed',
    }

    if mpl_style == 'professional':
        # mpl 3.11 types rc_context's arg with Literal rc-name keys; cast to
        # Any so the plain dict is accepted on both the baseline and latest
        # stubs (see skyplothelper.style._rc_update for the same rationale).
        ctx = plt.rc_context(cast("Any", _professional_params))
    elif mpl_style is not None and mpl_style != 'default':
        ctx = plt.style.context(mpl_style)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    if info_color is None:
        info_color = color

    with ctx:
        # --- Create figure and axes ---
        # ``ax=None`` is a convenience for one-off interactive use:
        # we build a sensible-default figure and a WCSAxes from the
        # ``wcs`` (or plain Axes if no wcs). For multi-panel
        # composability, pre-create the axes via ``make_wcs_frame``
        # (or ``quicklook_figure``) and pass it as ``ax=``.
        if ax is None:
            fig = plt.figure(figsize=(7, 7.5), facecolor=facecolor)
            if wcs is not None:
                ax = fig.add_subplot(111, projection=wcs)
            else:
                ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        if facecolor != 'none':
            fig.set_facecolor(facecolor)
            ax.set_facecolor(facecolor)
        if facecolor == 'none':
            fig.patch.set_alpha(0.)
            ax.patch.set_alpha(0.)

        # --- Field-of-view crop (BEFORE tick placement, so offset ticks size
        #     to the cropped view via apply_offset_ticks' visible-window
        #     sizing). field_size is an angular width: in offset units when
        #     offset_coords, else mas (per the docstring); without a WCS the
        #     axes are already in those units and we crop symmetric about 0.
        if field_size is not None:
            if wcs is None:
                ax.set_xlim(-field_size / 2., field_size / 2.)
                ax.set_ylim(-field_size / 2., field_size / 2.)
            else:
                assert ref_coord_used is not None  # set in the WCS block
                _per_deg = {'arcsec': 3600., 'arcmin': 60., 'mas': 3.6e6,
                            'uas': 3.6e9, 'μas': 3.6e9}
                _fs_unit = offset_disp_unit if offset_coords else 'mas'
                half_deg = (field_size / _per_deg.get(_fs_unit, 3.6e6)) / 2.
                try:
                    pscale = proj_plane_pixel_scales(wcs)
                    rpx, rpy = wcs.world_to_pixel_values(
                        ref_coord_used.ra.deg, ref_coord_used.dec.deg)
                    hx, hy = half_deg / pscale[0], half_deg / pscale[1]
                    ax.set_xlim(float(rpx) - hx, float(rpx) + hx)
                    ax.set_ylim(float(rpy) - hy, float(rpy) + hy)
                except Exception:
                    pass

        # --- Tick formatting ---
        _fc = frame_color or 'black'

        if wcs is not None:
            if offset_coords:
                assert ref_coord_used is not None  # set in the WCS block
                # Relative offset ticks from the reference, in offset units.
                _offset_to_fmt_unit = {
                    'arcsec': 'arcsec', 'mas': 'mas', 'μas': 'uas',
                    'uas': 'uas', 'arcmin': 'arcmin',
                }
                fmt_unit = _offset_to_fmt_unit.get(offset_disp_unit, 'arcsec')
                # axis_labels=False: quicklook owns the axis labels below.
                apply_offset_ticks(ax, ref_ra_deg=ref_coord_used.ra.deg,
                                   ref_dec_deg=ref_coord_used.dec.deg,
                                   unit=fmt_unit, axis_labels=False)
                for coord in [ax.coords[0], ax.coords[1]]:
                    coord.display_minor_ticks(True)
                    coord.set_minor_frequency(5)
                    coord.set_ticklabel(**_safe_ticklabel_kwargs(
                        {'exclude_overlapping': True}))
            else:
                # Absolute sexagesimal RA/Dec ticks (the documented default).
                try:
                    format_ticklabels(ax, style=tick_style)
                except Exception:
                    pass

        if wcs is not None:
            # WCSAxes read xtick/ytick.direction from the *global* rcParams at
            # draw time, so quicklook's transient mpl_style rc_context (e.g.
            # 'professional' → inward ticks) is lost by the time savefig draws.
            # Bake the resolved direction onto the tick objects so it survives.
            _tick_out_x = plt.rcParams.get('xtick.direction', 'out') != 'in'
            _tick_out_y = plt.rcParams.get('ytick.direction', 'out') != 'in'
            coord_ticks(ax.coords[0]).set_tick_out(_tick_out_x)
            coord_ticks(ax.coords[1]).set_tick_out(_tick_out_y)

        style_ax_colors(ax, color=axcolor)  # axis labels, ticklabels, text

        # Frame element (spine + tick MARK) color. A black frame hides against
        # a filled colormap that runs through black (the image=True sph.deepsky
        # default), so default to medium gray in image mode; the tick LABELS /
        # title are unaffected (they sit on the light margin). An explicit
        # frame_color overrides.
        if frame_color is not None:
            _fc = frame_color
        elif image:
            _fc = '0.5'
        else:
            _fc = 'black'
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color(_fc)
        if wcs is not None:
            for coord in [ax.coords[0], ax.coords[1]]:
                coord.set_ticks(color=_fc)   # tick marks only, not labels
        else:
            ax.tick_params(color=_fc, which='both')  # marks only (not labels)

        # Opt-in frame + tick stroke. Off by default (None) — the gray frame
        # above is the safe default. Pass a color (e.g. 'white') or a dict
        # {'color': .., 'lw': ..} to add a stroke; use a dark stroke for
        # dark-mode figures. Uses the unified frame-stroke helper.
        if frame_stroke:
            if isinstance(frame_stroke, dict):
                _sc = frame_stroke.get('color', 'white')
                _slw = float(frame_stroke.get('lw', 1.6))
            else:
                _sc, _slw = frame_stroke, 1.6
            from ..style import apply_frame_stroke
            apply_frame_stroke(ax, _sc, _slw)


        # --- Contour levels (in display units) ---
        positive_levels = []
        negative_levels = []
        if levels is not None:
            # User-supplied explicit levels override contour_start/factor.
            # Accept values in native units; scale to display units.
            _lev_arr = np.atleast_1d(np.asarray(levels, dtype=float))
            # Remove NaN/non-finite, sort so matplotlib accepts them
            _lev_arr = np.sort(_lev_arr[np.isfinite(_lev_arr)])
            _lev_disp = _lev_arr * display_factor
            positive_levels = [float(lv) for lv in _lev_disp if lv > 0]
            negative_levels = [float(lv) for lv in _lev_disp if lv < 0]
            # If caller gave only positive levels and negative_contours=True,
            # mirror them below zero (same convention as the auto path).
            if negative_contours and positive_levels and not negative_levels:
                _mirror = [-lv for lv in positive_levels
                           if -lv > display_data.min()]
                negative_levels = sorted(_mirror)
        elif contours and display_rms > 0:
            lev = contour_start * display_rms
            # Use 1.5x peak as the stop criterion so the highest geometric
            # step straddling the peak is included.
            while lev < display_peak * 1.5:
                positive_levels.append(lev)
                lev *= contour_factor
            if negative_contours and positive_levels:
                neglev = -positive_levels[0]
                # ``n_negative`` caps how deep the mirrored sequence goes.
                # A typical map (min ~ -4 sigma) only admits one level anyway;
                # the cap bites on deep images that would otherwise pick up
                # -2x, -4x ... steps the caller didn't ask for.
                limit = None if n_negative is None else max(0, int(n_negative))
                while neglev > display_data.min():
                    if limit is not None and len(negative_levels) >= limit:
                        break
                    negative_levels.append(neglev)
                    neglev *= contour_factor
                # matplotlib requires levels in strictly increasing order;
                # the loop above builds them most-negative-last so we sort.
                negative_levels.sort()

        # --- Resolve contour color / cmap / linewidths (single pass) ---
        contour_kw = {}
        if contour_cmap is not None:
            contour_kw['cmap'] = (plt.get_cmap(contour_cmap)
                                  if isinstance(contour_cmap, str)
                                  else contour_cmap)
            # Shared norm so cmap spans negative → positive continuously
            if negative_levels and positive_levels:
                contour_kw['norm'] = mcolors.SymLogNorm(
                    linthresh=3 * display_rms,
                    vmin=-3 * display_rms, vmax=display_peak,
                )
        else:
            contour_kw['colors'] = (
                contour_color if contour_color is not None
                else _over_image_ink(colormap, bool(image), color))
        if contour_alpha is not None:
            contour_kw['alpha'] = float(contour_alpha)
        # A very fine stroke keeps the contours readable where they cross the
        # bright end of a through-black colormap. withStroke's linewidth is
        # the TOTAL width, so the visible edge per side is
        # (stroke_lw - contour_lw) / 2 -- 0.1 at the defaults.
        # 'auto' resolves against the ink actually in use: a fixed dark stroke
        # sat behind the dark ink a reversed colormap selects, doing nothing
        # on exactly the maps where the contrast is tightest.
        #
        # With contour_cmap the contours have no single ink to contrast
        # against -- contour_kw carries 'cmap', not 'colors' -- so resolve
        # against the IMAGE instead. Contrasting with the ink only ever worked
        # because the ink is itself derived from the image; contour_cmap
        # severs that chain, and falling through to `color` (the info-text
        # color) picked a stroke unrelated to what sits under the contours.
        if isinstance(contour_stroke_color, str) and contour_stroke_color == 'auto':
            _c_colors = contour_kw.get('colors')
            if _c_colors is not None:
                _c_stroke = _contrast_ink(_c_colors)
            else:
                # No _contrast_ink here: _over_image_ink ALREADY returns the
                # color that contrasts with the image, which is exactly what a
                # stroke over that image needs. Taking its opposite would pick
                # a stroke that matches the image and disappears into it.
                _c_stroke = _over_image_ink(colormap, bool(image), color)
        else:
            _c_stroke = contour_stroke_color
        # Applied to the ContourSet after the fact: ax.contour() silently
        # discards a path_effects kwarg (it warns and moves on).
        _c_pe = _stroke_path_effects(_c_stroke, contour_stroke_lw)

        contour_lws: float | list[float]
        if isinstance(contour_lw, str) and contour_lw.lower() == 'scaled':
            n_lev = max(len(positive_levels), 1)
            contour_lws = [0.4 + 1.2 * i / max(n_lev - 1, 1)
                           for i in range(n_lev)]
        else:
            contour_lws = float(contour_lw)

        # --- Image normalization (used only when image=True) ---
        img_norm = norm
        if image:
            if img_norm is None and stretch is not None:
                s_l = stretch.lower()
                a_kw = None
                if vmin is None and vmax is None and display_rms > 0:
                    # No explicit limits: frame the stretch on the SAME range the
                    # auto path uses (vmin = sky level − 3·rms, vmax = peak) and
                    # give asinh / log / symlog an rms-based softening. Otherwise
                    # an explicit stretch spans the full [min, peak] with default
                    # softening and renders ~linear on a high-dynamic-range
                    # image — only the saturated core shows, no faint structure.
                    _vmin = display_bkg - 3.0 * display_rms
                    _vmax = display_peak
                    _rng = max(_vmax - _vmin, 1e-20)
                    if s_l == 'asinh':
                        # AsinhStretch `a` is the linear-width FRACTION; match
                        # the auto path's linear_width = 5·rms.
                        a_kw = float(np.clip(5.0 * display_rms / _rng, 1e-6, 1.0))
                    elif s_l == 'log':
                        # LogStretch `a` is steepness; scale to the dynamic
                        # range so the ~5·rms faint end stays visible.
                        a_kw = float(np.clip(_rng / (5.0 * display_rms), 1.0, 1e6))
                    elif s_l in ('symlog', 'symmetric_log'):
                        a_kw = max(3.0 * display_rms, 1e-20)   # linthresh
                else:
                    # Caller pinned vmin/vmax (or no usable rms): honor the
                    # given range (display units), as before.
                    _vmin = (vmin if vmin is not None
                             else float(np.nanmin(display_data)))
                    _vmax = (vmax if vmax is not None
                             else float(np.nanmax(display_data)))
                    if s_l in ('symlog', 'symmetric_log') and display_rms > 0:
                        a_kw = max(3.0 * display_rms, 1e-20)
                img_norm = make_norm(stretch, vmin=_vmin, vmax=_vmax, a=a_kw)
            elif img_norm is None and vmin is None and vmax is None:
                # Automatic image normalization for display. The floor is the
                # sky level minus 3·rms (not a hard zero): zero-mean radio maps
                # keep the familiar −3·rms floor, while an optical/IR frame on a
                # bright pedestal gets its empty sky pinned to black instead of
                # washing out across the colormap.
                _floor = display_bkg - 3 * display_rms
                try:
                    img_norm = mcolors.AsinhNorm(
                        vmin=_floor, vmax=display_peak,
                        linear_width=5 * display_rms,
                    )
                except AttributeError:
                    # matplotlib < 3.7 fallback
                    img_norm = mcolors.SymLogNorm(
                        linthresh=3 * display_rms,
                        vmin=_floor, vmax=display_peak,
                    )

        # --- Plot image ---
        im = None
        if image:
            imshow_kw = dict(origin='lower', cmap=colormap,
                             interpolation='nearest')
            if img_norm is not None:
                imshow_kw['norm'] = img_norm
            else:
                if vmin is not None:
                    imshow_kw['vmin'] = vmin
                if vmax is not None:
                    imshow_kw['vmax'] = vmax
            if wcs is None:
                ny_, nx_ = display_data.shape
                imshow_kw['extent'] = [-nx_ / 2, nx_ / 2,
                                       -ny_ / 2, ny_ / 2]
            im = ax.imshow(display_data, **imshow_kw)

        # --- Plot contours ---
        cs = None
        cs_neg = None
        if contours and positive_levels:
            if wcs is None:
                ny_, nx_ = display_data.shape
                extent = [-nx_ / 2, nx_ / 2, -ny_ / 2, ny_ / 2]
                cs = ax.contour(display_data, levels=positive_levels,
                                linewidths=contour_lws, extent=extent,
                                **contour_kw, **kwargs)
                if negative_levels:
                    cs_neg = ax.contour(
                        display_data, levels=negative_levels,
                        linewidths=contour_lws,
                        linestyles='dashed', extent=extent,
                        **contour_kw, **kwargs)
            else:
                X = np.arange(display_data.shape[1])
                Y = np.arange(display_data.shape[0])
                pix_tr = ax.get_transform('pixel')
                cs = ax.contour(X, Y, display_data, levels=positive_levels,
                           linewidths=contour_lws, transform=pix_tr,
                           **contour_kw, **kwargs)
                if negative_levels:
                    cs_neg = ax.contour(
                        X, Y, display_data, levels=negative_levels,
                        linewidths=contour_lws,
                        linestyles='dashed', transform=pix_tr,
                        **contour_kw, **kwargs)

        # --- Beam ellipse patch ---
        has_beam = (beam_maj is not None and beam_min is not None
                    and beam_maj > 0)
        if has_beam:
            from skyplothelper.overlays.beam import Beam

            if wcs is not None:
                try:
                    pscl = proj_plane_pixel_scales(wcs)
                    beam_maj_pix = beam_maj / pscl[1]
                    beam_min_pix = beam_min / pscl[0]
                except Exception:
                    beam_maj_pix = beam_maj
                    beam_min_pix = beam_min
            else:
                beam_maj_pix = beam_maj
                beam_min_pix = beam_min

            if beam_pa is None:
                beam_pa = 0.

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            pad_frac = 0.08
            bx = xlim[0] + pad_frac * abs(xlim[1] - xlim[0])
            by = ylim[0] + pad_frac * abs(ylim[1] - ylim[0])

            # Map quicklook's historical style names to Beam's style
            # enum. 'crosshair' is the publication default; the
            # '*grid' variants overlay a fine hatch for visibility
            # against busy contour backgrounds.
            beam_style_lower = (beam_style or 'crosshair').lower()
            if beam_style_lower not in ('hatch', 'filled', 'filledgrid',
                                          'crosshair', 'crosshairgrid'):
                beam_style_lower = 'crosshair'
            # 'filled' style uses a slightly thinner outline at 80%
            # opacity for the established radio look.
            if beam_style_lower in ('filled', 'filledgrid'):
                beam_lw, beam_alpha = 0.5, 0.8
            else:
                beam_lw, beam_alpha = 0.8, None

            # The beam sits in a fixed corner, not on the data, so the
            # colormap-derived ink can land on whatever is underneath -- and
            # on a masked map that corner is transparent, i.e. the page. A
            # stroke is what makes it legible regardless; 'auto' contrasts
            # against the ink, since the ink itself flips with the colormap.
            # A neutral mid-gray rather than the contours' ink: the beam is an
            # instrument property, not data, and drawing it in the contour
            # color makes it read as one more contour. Mid-gray also survives
            # both backgrounds the corner can have, which colormap-derived ink
            # cannot -- that ink is chosen for the image, and the corner is
            # only sometimes image.
            beam_ink = (beam_color if beam_color is not None
                        else BEAM_INK if image
                        else _over_image_ink(colormap, bool(image), color))
            beam_stroke = (_contrast_ink(beam_ink)
                           if isinstance(beam_stroke_color, str)
                           and beam_stroke_color == 'auto'
                           else beam_stroke_color)

            beam = Beam(
                (bx, by),
                bmaj_pix=beam_maj_pix, bmin_pix=beam_min_pix,
                bpa_deg=beam_pa, style=beam_style_lower,
                ec=beam_ink,
                stroke_color=beam_stroke, stroke_lw=beam_stroke_lw,
                lw=beam_lw, zorder=10)
            if beam_alpha is not None:
                beam.set_alpha(beam_alpha)
            beam.add_to(ax)

        # Contour stroke, applied to the finished ContourSets.
        if _c_pe is not None:
            for _cset in (cs, cs_neg):
                if _cset is None:
                    continue
                if hasattr(_cset, 'set_path_effects'):
                    _cset.set_path_effects(_c_pe)
                else:                       # matplotlib < 3.8
                    for _coll in getattr(_cset, 'collections', []):
                        _coll.set_path_effects(_c_pe)

        # --- Colorbar ---
        cb = None
        if colorbar and im is not None:
            # minor_ticks=False here: this path re-applies its own minor-tick
            # locator below (with the display clim it already has), so letting
            # add_colorbar also place them would compute them twice and discard
            # the first. simpleimageplot's colorbar, which does NOT re-apply,
            # keeps add_colorbar's adaptive default.
            cb = add_colorbar(im, ax=ax, label=unit_label, minor_ticks=False)

            # Tick precision follows the displayed range. The old rule was
            # a flat '.0f' for |x| >= 1, which collapsed a 0-3 Jy bar to
            # "0 1 2 3" and hid every value between.
            _cb_lo, _cb_hi = im.get_clim()
            _fmt: Any
            if cbar_format is None:
                _fmt = plt.FuncFormatter(
                    lambda x, _: tick_text(x, _cb_lo, _cb_hi))
            elif isinstance(cbar_format, str):
                # str_formatter accepts either '%.3f' or '{x:.3f}'; the old
                # FormatStrFormatter here printed a new-style string verbatim.
                _fmt = str_formatter(cbar_format)
            else:
                _fmt = cbar_format
            cb.ax.yaxis.set_major_formatter(_fmt)

            # Stroke on the tick marks (major + minor) for legibility.
            stroke = _stroke_path_effects(_fc, 1.0) or []
            for tick in cb.ax.yaxis.get_ticklines():
                tick.set_path_effects(stroke)
            for tick in cb.ax.yaxis.get_minor_ticks():
                tick.tick1line.set_path_effects(stroke)
                tick.tick2line.set_path_effects(stroke)

            cb.ax.tick_params(axis='y', which='major', direction='in', length=5, color=_fc, labelcolor=axcolor)
            cb.ax.tick_params(axis='y', which='minor', direction='in', length=2, color=_fc)
            cb.outline.set_edgecolor(_fc)

            # Minor ticks at 1/2/3/5 x 10^k across the decades the data
            # actually spans. The old list was those multiples for 10^0..10^3
            # only, in ABSOLUTE units — so an image displayed in 0-1 (or
            # 0-1e6) got no usable minor ticks at all.
            # Adaptive minor ticks: decade multiples on a compressed bar, even
            # subdivision on a linear one. Shared with the general-purpose
            # add_colorbar via _colorbar.apply_minor_ticks so the two can't
            # drift; the old FixedLocator of absolute 1..5000 gave an image
            # displayed in 0-1 (or 0-1e6) no usable minor ticks at all.
            apply_minor_ticks(cb, cbar_minor_ticks, _cb_lo, _cb_hi)

            cb.ax.yaxis.label.set_color(axcolor)
            cb.ax.yaxis.label.set_rotation(-90)
            # Stroke the colorbar label too (parity with the tick marks above
            # and with add_colorbar's stroked label).
            if stroke:
                cb.ax.yaxis.label.set_path_effects(stroke)

        # --- Axis labels (consistent with the tick mode chosen above) ---
        if offset_coords and ref_coord_used is not None:
            ra_str = ref_coord_used.ra.to_string(u.hourangle, sep='hms',
                                                  precision=4, pad=True)
            dec_str = ref_coord_used.dec.to_string(u.deg, sep='dms',
                                                    precision=3, alwayssign=True,
                                                    pad=True)
            ax.set_xlabel(f'Relative RA ({offset_disp_unit}) from {ra_str}',
                          fontsize=12, fontfamily=figure_font, color=axcolor)
            ax.set_ylabel(f'Relative Dec ({offset_disp_unit}) from {dec_str}',
                          fontsize=12, fontfamily=figure_font, color=axcolor)
        elif wcs is not None:
            ax.set_xlabel('Right Ascension', fontsize=12, fontfamily=figure_font, color=axcolor)
            ax.set_ylabel('Declination', fontsize=12, fontfamily=figure_font, color=axcolor)
        else:
            ax.set_xlabel(f'Relative RA ({offset_units})', fontsize=12, fontfamily=figure_font, color=axcolor)
            ax.set_ylabel(f'Relative Dec ({offset_units})', fontsize=12, fontfamily=figure_font, color=axcolor)

        # (The field-of-view crop is applied earlier, before tick placement,
        # so the offset ticks size to the cropped view.)

        # --- Title and top annotations ---
        if source_name:
            ax.set_title(source_name, fontsize=14, fontweight='bold',
                         fontfamily='sans-serif', loc='center', pad=10,
                         color=axcolor)
        if obs_date:
            ax.text(0., 1.02, obs_date, transform=ax.transAxes,
                    fontsize=11, fontfamily='sans-serif', ha='left',
                    color=axcolor)
        if label:
            ax.text(1.0, 1.02, label, transform=ax.transAxes,
                    fontsize=11, fontfamily='sans-serif', ha='right',
                    color=info_color)

        # --- Bottom info text (peak, rms, contour levels), all in display units ---
        info_text = None
        if show_info:
            peak_disp = display_peak
            rms_disp = display_rms

            # Peak and RMS format from their own magnitudes — a bright peak
            # must not force a sub-mJy RMS to truncate to 0.000 (see _stat_fmt).
            pk_fmt = _stat_fmt(peak_disp)
            rms_fmt = _stat_fmt(rms_disp)

            lines = []

            # Line 1: center coordinates
            if center_ra_str and center_dec_str:
                lines.append(f'Center: {center_ra_str}  {center_dec_str}')

            # Line 2: peak, RMS, peak/RMS
            lines.append(
                f'Peak = {peak_disp:{pk_fmt}}{_u}    '
                f'RMS = {rms_disp:{rms_fmt}}{_u}    '
                f'Peak/RMS = {(peak/rms if rms != 0 else float("nan")):.1f}')

            # Line 3: contour levels
            if positive_levels:
                # If user supplied explicit levels, always show actual
                # values rather than RMS-multiples (which would be
                # misleading for arbitrary user-chosen levels).
                _effective_labelstyle = (
                    'actual' if levels is not None
                    else contour_labelstyle
                )
                if _effective_labelstyle.lower() == 'rms':
                    if negative_levels:
                        neg_lev_factors = [f'{lv}' for lv in np.round(np.array(negative_levels)/(display_rms*contour_start)).astype(int)]
                    else:
                        neg_lev_factors = []
                    pos_lev_factors = [f'{lv}' for lv in np.round(np.array(positive_levels)/(display_rms*contour_start)).astype(int)]
                    lev_ints = neg_lev_factors + pos_lev_factors
                    # Elide a long ladder the same way the explicit-levels
                    # branch below does. Unbounded, a 13-step geometric ladder
                    # runs to ~108 characters and sets the saved figure's width
                    # under bbox_inches='tight'. A geometric ladder is fully
                    # described by its endpoints, so the ellipsis loses nothing.
                    if len(lev_ints) > 6:
                        lev_str = ', '.join(lev_ints[:6]) + f', ... {lev_ints[-1]}'
                    else:
                        lev_str = ', '.join(lev_ints)
                    lines.append(f'Contours: {contour_start}$\\times$RMS ({contour_start*display_rms:.3g}{_u})  $\\times$  ({lev_str} )')
                else:
                    lvls_disp = list(positive_levels)
                    lvl_strs = [f'{lv:{pk_fmt}}' for lv in lvls_disp[:6]]
                    lvl_str = ', '.join(lvl_strs)
                    if len(lvls_disp) > 6:
                        lvl_str += ', ... %.3g'%(positive_levels[-1])
                    lines.append(f'Contours: {lvl_str}{_u}')

            # Line 4: beam info (if available)
            if has_beam:
                # has_beam already guarantees these are non-None (see the
                # beam-patch block above); assert so the type-checker
                # narrows them for the formatting below.
                assert beam_maj is not None and beam_min is not None
                bmaj_mas = beam_maj * 3600e3
                bmin_mas = beam_min * 3600e3
                if bmaj_mas > 1000:
                    lines.append(f'Beam: {beam_maj*3600:.2f}″ × '
                                 f'{beam_min*3600:.2f}″, PA {beam_pa:.1f}° (E from N)')
                else:
                    lines.append(f'Beam: {bmaj_mas:.2f} × '
                                 f'{bmin_mas:.2f} mas, PA {beam_pa:.1f}° (E from N)')

            info_text_str = '\n'.join(lines)

            # Place info text below the axes using axes-fraction coords.
            # This survives tight_layout() calls and bbox_inches='tight'
            # will extend the saved figure to include it.
            n_lines = len(lines)
            # Offset below axes: clear tick labels + axis label (~0.12)
            # plus one line height per info line (~0.035 axes-fraction)
            gap = 0.10 + n_lines * 0.01
            info_text = ax.text(
                0, -gap, info_text_str, fontsize=9, color=info_color,
                family='monospace', va='top', transform=ax.transAxes,
                clip_on=False)

        # --- Grid ---
        if grid:
            if wcs is not None:
                ax.coords.grid(color=gridcolor, alpha=gridalpha, lw=0.5, linestyle='solid')
            else:
                ax.grid(True, lw=0.5, color=gridcolor, alpha=gridalpha, linestyle='solid')


        ax.set_aspect('equal')
        #plt.tight_layout(rect=[0, 0.04, 1, 1], pad=2.)

    return QuicklookResult(fig=fig, ax=ax, image=im, colorbar=cb,
                           contour_set=cs, neg_contour_set=cs_neg,
                           info_text=info_text)


def quicklook_figure(image_data: Any, *,
                     figsize: tuple[float, float] = (7, 7.5), dpi: int = 150,
                     facecolor: str = 'w', output_file: str | None = None,
                     **plot_kwargs: Any) -> QuicklookResult:
    """One-line quick-look figure builder.

    Owns figure creation (``figsize``, ``dpi``, ``facecolor``,
    ``output_file``) and delegates the actual data rendering to
    :func:`quicklook_plot`. Use this for one-off interactive use
    or scripts that produce a single figure per FITS image; for
    multi-panel composability, build a figure + axes yourself
    (``plt.figure`` + ``make_wcs_frame``) and call
    :func:`quicklook_plot` directly with ``ax=``.

    Parameters
    ----------
    image_data : str or 2D ndarray
        FITS path or image array (passed through to
        :func:`quicklook_plot`).
    figsize : tuple
    dpi : int
    facecolor : str
        Figure background color.
    output_file : str, optional
        If given, save the figure to this path with
        ``bbox_inches='tight'``.
    **plot_kwargs
        Forwarded to :func:`quicklook_plot` (``header``, ``wcs``,
        ``contours``, ``image``, ``colorbar``, ``label``,
        ``beam_maj``, etc.).

    Returns
    -------
    QuicklookResult
    """
    fig = plt.figure(figsize=figsize, facecolor=facecolor)
    # WCS resolution: try header → wcs, otherwise plain Axes.
    wcs = plot_kwargs.get('wcs')
    header = plot_kwargs.get('header')
    # When handed a FITS path with no explicit header/wcs, read the header
    # now so the axes is created WITH the WCS projection. quicklook_plot
    # loads the header too, but only after the axes already exists — too
    # late for the projection, and the offset-tick path needs ax.wcs.
    if wcs is None and header is None and isinstance(image_data, str):
        try:
            with pyfits.open(image_data) as hdul:
                header = hdul[0].header
            plot_kwargs['header'] = header
        except Exception:
            header = None
    if wcs is None and header is not None:
        try:
            # ``.celestial`` keeps the 2 sky axes (FITS cubes have extra
            # freq/Stokes axes that a WCSAxes projection can't use).
            wcs = WCS(header).celestial
        except Exception:
            wcs = None
    if wcs is not None:
        ax = fig.add_subplot(111, projection=wcs)
    else:
        ax = fig.add_subplot(111)

    result = quicklook_plot(image_data, ax=ax, facecolor=facecolor,
                            **plot_kwargs)

    if output_file:
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight',
                    facecolor=facecolor)

    return result


def quicklook_fits(fits_file: str, **kwargs: Any) -> QuicklookResult:
    """
    Convenience wrapper: load a FITS file and create a quicklook plot.

    Equivalent to ``quicklook_plot(fits_file, **kwargs)`` — the
    filename is passed directly and header/WCS are extracted
    automatically.

    Parameters
    ----------
    fits_file : str
        Path to FITS file.
    **kwargs
        All kwargs passed to ``quicklook_plot()``.

    Returns
    -------
    result : QuicklookResult
        Same NamedTuple returned by :func:`quicklook_plot`.

    Examples
    --------
    >>> result = sph.quicklook_fits('3C273_X.fits', color='#2ca02c')
    >>> result = sph.quicklook_fits('hubble_image.fits',
    ...     contours=False, image=True, colormap='gray_r')
    """
    return quicklook_figure(fits_file, **kwargs)

