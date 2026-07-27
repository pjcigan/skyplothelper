"""Plot annotations.

Scale-bar, compass, axis inlay, colorbar, contour overlay,
band-label and color styling helpers. Beam rendering lives in
:mod:`skyplothelper.overlays.beam` (the :class:`Beam` /
:class:`BeamStack` classes).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord  # noqa: F401
from astropy.visualization.wcsaxes.frame import EllipticalFrame
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from .. import core  # noqa: F401
from .._stroke import _stroke_path_effects
from ..core.fits_utils import getdegperpix
from ..ticks import _detect_frame

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord


def add_colorbar(mappable: Any, ax: Any = None, label: str | None = None,
                 orientation: str = 'vertical', mode: str = 'divider',
                 location: str | None = None,
                 shrink: float = 1.0, pad: float = 0.05, aspect: float = 25,
                 cax: Any = None, stroke_color: Any = None,
                 stroke_lw: float = 2.5, stroke_targets: str = 'both',
                 minor_ticks: Any = 'auto', tick_format: Any = None,
                 **kwargs: Any) -> Any:
    """
    Convenience wrapper for adding a colorbar to a WCSAxes / image plot.

    Addresses the common gotcha that ``plt.colorbar(ax=...)`` sizes the bar to
    the axes *bounding box*, which on a fixed-aspect (WCS / image) axes does
    not track the rendered image — leaving the bar too short (the old
    ``shrink=0.8`` default), or, in a tall figure, overshooting it.

    Parameters
    ----------
    mappable : ScalarMappable
        The image, pcolormesh, contourf, or PatchCollection to colorbar.
    ax : Axes, optional
        Axes to attach to. If None, uses the mappable's axes.
    label : str, optional
        Colorbar label.
    orientation : {'vertical', 'horizontal'}
    mode : {'divider', 'inset', 'simple'}
        Placement strategy:

        * ``'divider'`` (default) — an
          :class:`~mpl_toolkits.axes_grid1.axes_divider.AxesDivider` slot. The
          bar matches the image extent and resizes with it, AND reserves its
          own space (the axes shrinks to fit) — so it never overlaps a
          neighbouring panel. Best general default. Auto-falls back to
          ``'inset'`` (with a warning) on an axes that already owns a locator —
          an ImageGrid / :func:`~skyplothelper.channel_map` panel, or an axes
          that already has a divider colorbar — where ``append_axes`` would
          otherwise silently break the layout.
        * ``'inset'`` — an ``ax.inset_axes`` anchored in axes-fraction coords.
          Also matches the image extent, but does NOT reserve space (the image
          keeps its full size and the bar floats just outside) — handy for a
          single panel where you don't want the image to shrink. May overlap
          neighbours in a tight multi-panel layout.
        * ``'simple'`` — the classic ``plt.colorbar(ax=...)``; kept as an
          escape hatch (subject to the bbox-sizing gotcha above). Also the
          right choice on a **polar axes** (a cone / bowtie frame): ``'divider'``
          uses ``make_axes_locatable``, which collapses a polar wedge, so
          ``add_colorbar`` auto-falls back to ``'simple'`` there (with a
          warning) — the polish still applies.
    location : {'right', 'left', 'top', 'bottom'}, optional
        Which side of the axes to place the bar on, for any ``mode``. Takes
        precedence over ``orientation`` (left/right ⇒ vertical, top/bottom ⇒
        horizontal) and moves the ticks + label to the outer side. ``None``
        (default) keeps the legacy ``orientation``-driven side (right for
        vertical, bottom for horizontal). Saves hand-rolling an ``inset_axes``
        + ``yaxis.set_ticks_position('left')`` for a left- or top-side bar.
    shrink : float
        Fraction of the image extent the bar spans (default ``1.0`` — full
        match). ``<1`` centers a shorter bar.
    pad : float
        Gap between the axes and the bar, as a fraction of the axes size.
    aspect : float
        Bar length-to-thickness ratio (higher = thinner).
    cax : Axes, optional
        An explicit axes to draw the colorbar *into*. When given, ``mode``
        and the placement kwargs (``shrink`` / ``pad`` / ``aspect``) are
        bypassed entirely — you own the placement — and the bar is drawn with
        ``fig.colorbar(mappable, cax=cax)``. The stroke / zorder handling
        still applies. Use this for full manual control, or to lay out
        several colorbars yourself (see the multiple-colorbars note below).
    stroke_color : color spec or None
        If given, draw a stroke (outline) behind the colorbar tick marks
        and/or its frame in this color, for legibility where they would
        otherwise blend into the colormap (e.g. black ticks over the dark
        end of a sequential map). Default ``None`` — no stroke.
    stroke_lw : float
        Total stroke width in points (the visible stroke each side is
        ``(stroke_lw - line_lw) / 2``). Default ``2.5``. Ignored when
        ``stroke_color`` is ``None``.
    stroke_targets : {'both', 'ticks', 'spine'}
        What the stroke applies to when ``stroke_color`` is set: the tick
        marks (major + minor), the frame/outline, or ``'both'`` (default).
        ``'both'`` also strokes the colorbar's axis label (e.g. ``'Jy/beam'``)
        so light label text stays legible on a light page.
    minor_ticks : {'auto', False}, sequence, or Locator
        Minor tick positions on the bar. ``'auto'`` (default) is adaptive: an
        even subdivision on a linear bar, ``1/2/3/5 x 10^k`` across the
        occupied decades on a compressed (log / asinh / symlog) one — the same
        behavior :func:`quicklook_plot` uses. This is a deliberate departure
        from bare matplotlib, which draws no minor ticks; pass ``False`` for
        that look, or a sequence of positions / a
        :class:`~matplotlib.ticker.Locator` to place them yourself.
    tick_format : optional
        Major-tick label format. ``None`` (default) leaves matplotlib's. This
        is **opt-in**, unlike ``minor_ticks``, because it rewrites every label
        rather than adding to the bar. ``'auto'`` matches label precision to
        the displayed range — the same logic :func:`quicklook_plot` uses, so a
        0-3 Jy bar reads ``0.5 1.0 …`` instead of matplotlib's ``0 1 2 3``.
        A format string in either style — ``'%.3f'`` or ``'{x:.3f}'`` — is
        detected and honored, as is a :class:`~matplotlib.ticker.Formatter`.
    **kwargs : additional kwargs passed to ``figure.colorbar``.

    Returns
    -------
    cbar : Colorbar

    Notes
    -----
    Each visual component of the colorbar is a separate artist on the
    returned ``cbar.ax`` (which is its own Axes, independent of the main
    image axes), so you can style them independently after the call::

        cbar = add_colorbar(im, ax=ax)
        cbar.ax.tick_params(which='both', color='white')   # tick MARKS only
        cbar.ax.tick_params(which='both', labelcolor='0.3')  # tick LABELS only
        cbar.set_label('Jy / beam', color='k')              # axis LABEL only
        cbar.outline.set_edgecolor('white')                 # frame/outline only

    Note ``color=`` styles the tick marks while ``labelcolor=`` styles the
    tick labels — that is the marks-vs-labels split. The ``stroke_*`` kwargs
    above cover the one case that is awkward by hand (a legibility stroke
    behind the marks and/or frame); plain recoloring is left to these
    one-liners.

    **Multiple colorbars on one axes.** Because each call colorbars a
    *different* mappable, you can attach several to the same axes (e.g. two
    differently-cmapped scatter sets). The placement modes behave differently:

    * ``mode='simple'`` stacks repeated bars on the same side cleanly —
      ``plt.colorbar`` accounts for bars already stealing space from the axes::

          add_colorbar(sc1, ax=ax, mode='simple', label='set 1')
          add_colorbar(sc2, ax=ax, mode='simple', label='set 2')  # beside it

    * ``mode='divider'`` / ``'inset'`` each compute a *self-contained*
      placement and do NOT account for an existing bar, so two same-side bars
      would overlap. Either give them different orientations (one vertical,
      one horizontal)::

          add_colorbar(sc1, ax=ax, label='right')
          add_colorbar(sc2, ax=ax, orientation='horizontal', label='bottom')

      or place them yourself with ``cax=`` for full control::

          cax1 = ax.inset_axes([1.02, 0.55, 0.04, 0.45])
          cax2 = ax.inset_axes([1.02, 0.00, 0.04, 0.45])
          add_colorbar(sc1, cax=cax1, label='set 1')
          add_colorbar(sc2, cax=cax2, label='set 2')

    Examples
    --------
    >>> import skyplothelper as sph
    >>> im = ax.imshow(data)
    >>> sph.add_colorbar(im, ax=ax, label='Jy/beam', location='right')
    >>> # location='left'/'top'/'bottom'; mode='inset' for ImageGrid panels
    """
    if stroke_targets not in ('both', 'ticks', 'spine'):
        raise ValueError(
            "stroke_targets must be 'both', 'ticks', or 'spine', "
            f"got {stroke_targets!r}")
    # ``location`` (which side of the axes) takes precedence and dictates the
    # orientation; ``location=None`` keeps the legacy orientation-driven side
    # ('right' for vertical, 'bottom' for horizontal).
    if location is not None:
        if location not in ('right', 'left', 'top', 'bottom'):
            raise ValueError(
                "location must be 'right', 'left', 'top', or 'bottom', "
                f"got {location!r}")
        orientation = 'horizontal' if location in ('top', 'bottom') else 'vertical'
        loc = location
    else:
        loc = 'bottom' if orientation == 'horizontal' else 'right'

    if cax is not None:
        # Explicit placement: the caller owns the axes, so bypass the mode
        # machinery (mode / shrink / pad / aspect are ignored here).
        cbar = cax.figure.colorbar(mappable, cax=cax, orientation=orientation,
                                   **kwargs)
    else:
        cbar = _add_colorbar_placed(mappable, ax, orientation, mode,
                                    shrink, pad, aspect, loc, **kwargs)

    # Adaptive minor ticks, matching quicklook via the shared helper: an even
    # subdivision on a linear bar, 1/2/3/5 x 10^k across the occupied decades
    # on a compressed one. sph deliberately differs from bare matplotlib here
    # (which draws none) -- minor ticks read a colorbar more precisely. Pass
    # minor_ticks=False for the plain-matplotlib look, or a sequence / Locator.
    from .._colorbar import apply_adaptive_format, apply_minor_ticks
    apply_minor_ticks(cbar, minor_ticks)
    # Precision that follows the displayed range is OPT-IN, unlike the minor
    # ticks: it rewrites every major label rather than adding to the bar. A
    # matplotlib format string / Formatter is honored as-is; 'auto' asks for
    # the range-following precision quicklook uses.
    if tick_format is not None:
        _cb_axis = (cbar.ax.yaxis if orientation == 'vertical'
                    else cbar.ax.xaxis)
        if isinstance(tick_format, str) and tick_format == 'auto':
            apply_adaptive_format(cbar)
        elif isinstance(tick_format, str):
            # A format string in either style ('%.3f' or '{x:.3f}'); str_formatter
            # detects which. set_major_formatter(str) alone assumes new-style
            # and prints an old-style string verbatim.
            from .._colorbar import str_formatter
            _cb_axis.set_major_formatter(str_formatter(tick_format))
        else:                       # a Formatter instance
            _cb_axis.set_major_formatter(tick_format)

    _apply_colorbar_polish(cbar, stroke_color, stroke_lw, stroke_targets)
    # Ticks + label on the colorbar's outer side (left/top need repositioning;
    # right/bottom are matplotlib's defaults).
    if loc == 'left':
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
    elif loc == 'top':
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
    if label is not None:
        cbar.set_label(label)
    return cbar


def _add_colorbar_placed(mappable: Any, ax: Any, orientation: str, mode: str,
                         shrink: float, pad: float, aspect: float,
                         location: str = 'right', **kwargs: Any) -> Any:
    """Build the colorbar via one of the auto-placement modes (no ``cax``)."""
    if mode not in ('divider', 'inset', 'simple'):
        raise ValueError(
            f"mode must be 'divider', 'inset', or 'simple', got {mode!r}")
    if ax is None:
        ax = mappable.axes
    fig = ax.figure
    horiz = orientation == 'horizontal'

    # 'divider' uses make_axes_locatable, which assumes a rectilinear axes and
    # installs its own axes_locator. On a polar axes (e.g. a cone / bowtie
    # frame) that collapses the wedge, so fall back to 'simple' — which still
    # gets the tick-visibility / stroke polish. WCSAxes are fine with divider.
    if mode == 'divider' and getattr(ax, 'name', '') == 'polar':
        import warnings
        warnings.warn(
            "add_colorbar: mode='divider' (make_axes_locatable) collapses a "
            "polar axes such as a cone/bowtie frame — using mode='simple' "
            "instead. Pass mode='simple' explicitly to silence this.",
            stacklevel=3)
        mode = 'simple'

    # 'divider' also fights an axes that already owns an axes_locator — an
    # ImageGrid / channel_map panel, or an axes that already has a divider
    # colorbar. append_axes then silently breaks the layout, so fall back to
    # 'inset' (which floats a bar without touching the locator).
    if mode == 'divider' and ax.get_axes_locator() is not None:
        import warnings
        warnings.warn(
            "add_colorbar: mode='divider' can't reserve space on an axes that "
            "already has a locator (e.g. an ImageGrid / channel_map panel, or "
            "a second colorbar on the same axes) — using mode='inset' instead. "
            "Pass mode='inset' explicitly to silence this.",
            stacklevel=3)
        mode = 'inset'

    if mode == 'simple':
        try:                 # mpl >= 3.7 places any side via location=
            cbar = plt.colorbar(mappable, ax=ax, location=location,
                                shrink=shrink, pad=pad, aspect=aspect, **kwargs)
        except (TypeError, ValueError):
            cbar = plt.colorbar(mappable, ax=ax, orientation=orientation,
                                shrink=shrink, pad=pad, aspect=aspect, **kwargs)
    elif mode == 'inset':
        thick = 1.0 / aspect
        if location == 'bottom':
            bounds = [(1.0 - shrink) / 2.0, -pad - thick, shrink, thick]
        elif location == 'top':
            bounds = [(1.0 - shrink) / 2.0, 1.0 + pad, shrink, thick]
        elif location == 'left':
            bounds = [-pad - thick, (1.0 - shrink) / 2.0, thick, shrink]
        else:                # right
            bounds = [1.0 + pad, (1.0 - shrink) / 2.0, thick, shrink]
        cax = ax.inset_axes(bounds)
        cbar = fig.colorbar(mappable, cax=cax, orientation=orientation,
                            **kwargs)
    else:   # 'divider' — match the image AND reserve space
        try:
            import matplotlib.axes as _maxes
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            from mpl_toolkits.axes_grid1.axes_size import (
                AxesX,
                AxesY,
                Fraction,
            )
            divider = make_axes_locatable(ax)
            if horiz:
                # length ~ axes width; thickness ~ height / aspect.
                slot = divider.append_axes(
                    location, size=Fraction(1.0 / aspect, AxesX(ax)),
                    pad=Fraction(pad, AxesY(ax)), axes_class=_maxes.Axes)
                inset_bounds = [(1.0 - shrink) / 2.0, 0.0, shrink, 1.0]
            else:
                # length ~ axes height; thickness ~ width / aspect.
                slot = divider.append_axes(
                    location, size=Fraction(1.0 / aspect, AxesY(ax)),
                    pad=Fraction(pad, AxesX(ax)), axes_class=_maxes.Axes)
                inset_bounds = [0.0, (1.0 - shrink) / 2.0, 1.0, shrink]
            if shrink < 1.0:
                # Keep the reserved slot (neighbours don't shift) but draw a
                # centered, shorter bar inside it.
                slot.set_axis_off()
                cax = slot.inset_axes(inset_bounds)
            else:
                cax = slot
            cbar = fig.colorbar(mappable, cax=cax, orientation=orientation,
                                **kwargs)
        except Exception:
            # Older matplotlib / unusual axes: fall back to the classic path.
            cbar = plt.colorbar(mappable, ax=ax, orientation=orientation,
                                shrink=shrink, pad=pad, aspect=aspect,
                                **kwargs)
    return cbar


def _apply_colorbar_polish(cbar: Any, stroke_color: Any, stroke_lw: float,
                           stroke_targets: str) -> None:
    """Shared post-build touch-ups applied to every ``add_colorbar`` path
    (auto-placed or explicit ``cax=``): the tick-zorder lift and the optional
    legibility stroke."""
    # Keep the tick marks/labels drawn ABOVE the color solids. A colorbar's
    # Axis draws as a single unit at its (low) zorder, whereas the solids
    # QuadMesh defaults to a higher one — so with inward-pointing ticks (the
    # 'structural' / 'journal' base styles set ``*tick.direction='in'``) the
    # marks land inside the bar and would otherwise be painted over by the
    # solids and vanish. Lift the tick-bearing axes above the solids; this is
    # a no-op for the common outward-tick case.
    try:
        solids = getattr(cbar, 'solids', None)
        if solids is not None:
            _z = solids.get_zorder() + 1
            cbar.ax.xaxis.set_zorder(_z)
            cbar.ax.yaxis.set_zorder(_z)
    except Exception:
        pass

    # Optional stroke behind the tick marks / frame, for legibility where they
    # would blend into the colormap (e.g. black ticks over the dark end of a
    # sequential map). The cax is a plain Axes (not a WCSAxes), so its tick
    # Line2D objects honor path effects directly.
    pe = _stroke_path_effects(stroke_color, stroke_lw)
    if pe is not None:
        if stroke_targets in ('both', 'ticks'):
            for axis in (cbar.ax.xaxis, cbar.ax.yaxis):
                for line in (list(axis.get_ticklines(minor=False))
                             + list(axis.get_ticklines(minor=True))):
                    line.set_path_effects(pe)
        if stroke_targets in ('both', 'spine'):
            outline = getattr(cbar, 'outline', None)
            if outline is not None:
                outline.set_path_effects(pe)
        if stroke_targets == 'both':
            # The axis label ('Jy/beam', etc.) is legibility-relevant text like
            # the frame's axis titles — stroke it too so a light label reads on
            # a light page margin. (Was the add_colorbar half of the axis-label
            # stroke gap; the frame half lives in format_ticklabels.) Applied
            # unconditionally, not gated on current text: the label is often
            # set *after* this polish runs (via cbar.set_label), and a Text's
            # path_effects persist when its text is set later.
            for axis in (cbar.ax.xaxis, cbar.ax.yaxis):
                axis.label.set_path_effects(pe)


def add_contour_overlay(ax: Any, lon: SkyCoord | npt.ArrayLike, lat: Any = None,
                        values: npt.ArrayLike | None = None,
                        levels: int | npt.ArrayLike = 10,
                        filled: bool = False, cmap: Any = 'viridis',
                        colors: Any = None, linewidths: float = 1.,
                        alpha: float = 1., **kwargs: Any) -> Any:
    """
    Add contour lines or filled contours on a WCSAxes from world coordinates.

    Parameters
    ----------
    ax : WCSAxes
    lon, lat : 2D arrays, or SkyCoord in ``lon``
        A ``SkyCoord`` grid may be passed as ``lon``, replacing both;
        ``values=`` is then given as a keyword.
        World coordinate grids (degrees)
    values : 2D array
        Data values on the lon/lat grid
    levels : int or array-like
        Number of contour levels, or explicit level values
    filled : bool
        If True, use contourf instead of contour
    cmap : str or Colormap
    colors : str or list, optional
        If given, overrides cmap for line contours
    linewidths : float
        Line width for contour lines
    alpha : float
    **kwargs : additional kwargs passed to contour/contourf

    Returns
    -------
    cs : QuadContourSet

    Examples
    --------
    >>> sph.add_contour_overlay(ax, lon_grid, lat_grid, values, levels=6)
    >>> sph.add_contour_overlay(ax, coord_grid, values=values, filled=True)
    """
    from ..geometry._parsing import _coords_or_arrays_deg
    from ..wcs_frame import _get_wcs_frame_name
    lon, lat = _coords_or_arrays_deg(lon, lat, _get_wcs_frame_name(ax),
                                     'add_contour_overlay')
    if values is None:
        raise TypeError('add_contour_overlay: values is required.')
    transform = ax.get_transform('world')
    func = ax.contourf if filled else ax.contour

    contour_kwargs = dict(levels=levels, alpha=alpha, transform=transform,
                          **kwargs)
    if filled or colors is None:
        contour_kwargs['cmap'] = cmap
    if colors is not None and not filled:
        contour_kwargs['colors'] = colors
    if not filled:
        contour_kwargs['linewidths'] = linewidths

    return func(lon, lat, values, **contour_kwargs)


###############################################################################
#                                                                             #
#             PLOT ANNOTATIONS                                     #
#                                                                             #
###############################################################################


def add_sizebar(axin: Any, length_pixels: float, label: str, loc: int = 4,
                sep: float = 5, borderpad: float = 0.8, frameon: bool = False,
                path_effects: list[Any] | None = None, color: Any = None,
                stroke_color: Any = 'k', stroke_lw: float = 1.75,
                **kwargs: Any) -> Any:
    """
    Add a scale bar to a matplotlib axis.

    Parameters
    ----------
    axin : matplotlib Axes
    length_pixels : float
    label : str
    stroke_color : color spec or None
        Stroke color drawn under the bar + label. Default ``'k'``
        — preserves the prior hardcoded thin-black-stroke behavior.
        Set to ``None`` to disable the stroke.
    stroke_lw : float
        Total stroke width in points. Default ``1.75`` — matches the
        prior hardcoded value.
    path_effects : list or None
        Escape hatch: pass a custom ``path_effects`` list directly to
        override ``stroke_color`` / ``stroke_lw``. Default ``None``
        (use the stroke kwargs).
    """
    if path_effects is None:
        path_effects = _stroke_path_effects(stroke_color, stroke_lw)
    asb = AnchoredSizeBar(axin.transData, length_pixels, label, loc=loc,
                          borderpad=borderpad, sep=sep, frameon=frameon, **kwargs)
    if color is not None:
        asb.size_bar.get_children()[0].set_ec(color)
        asb.txt_label._text.set_color(color)
    if path_effects is not None:
        for a in [asb.size_bar._children[0], asb.txt_label._text]:
            a.set_path_effects(path_effects)
    axin.add_artist(asb)
    return asb


def add_sizebar_asec(axin: Any, hdrin: Any, length_asec: float, label: str,
                     **kwargs: Any) -> Any:
    """
    Add a scale bar with size specified in arcseconds (auto pixel conversion).

    Parameters
    ----------
    axin : matplotlib Axes
    hdrin : astropy.io.fits.Header
    length_asec : float
    label : str

    Examples
    --------
    >>> import skyplothelper as sph
    >>> res = sph.simpleimage_figure(image, header, cmap='sph.deepsky')
    >>> sph.add_sizebar_asec(res.ax, header, 30, '30"')     # a 30-arcsec bar
    """
    dpp = getdegperpix(hdrin)
    length_pixels = length_asec / (dpp * 3600.)
    return add_sizebar(axin, length_pixels, label, **kwargs)



def add_bandlabels(axin: Any, labels: Sequence[str], labcolors: Sequence[Any],
                   fontsize: float = 10, textpad: float = 0.05,
                   xy: tuple[float, float] = (0.04, 0.96), va: str = 'top',
                   ha: str = 'left', stroke_color: Any = None,
                   stroke_lw: float = 3.0, zorder: float | None = None,
                   **kwargs: Any) -> list[Any]:
    """
    Annotate band/filter labels on axes (e.g., for multicolor images).

    The labels are chained left-to-right, each anchored just past the
    previous one's right edge, so a multi-band caption reads as one row.

    Parameters
    ----------
    axin : matplotlib Axes
    labels : list of str
    labcolors : list of str
        Per-label text colors (same length as *labels*).
    fontsize : float
    textpad : float
        Horizontal gap between chained labels, as a fraction of the
        previous label's width.
    xy : tuple
        Axes-fraction position of the first label.
    va, ha : str
        Vertical / horizontal text alignment.
    stroke_color : color, optional
        Legibility stroke behind the label text (via ``path_effects``).
        Default ``None`` — no stroke.
    stroke_lw : float, optional
        Total stroke width in points. Default ``3.0``; applies only when
        ``stroke_color`` is set.
    zorder : float, optional
        Drawing order for the labels.
    **kwargs
        Forwarded to each ``axes.annotate`` call (e.g. ``fontweight``).

    Returns
    -------
    list of matplotlib Annotation
        The label artists, in order.
    """
    from matplotlib.text import OffsetFrom
    # Vertical reference fraction of the previous label's bbox to anchor the
    # next one to — must MATCH ``va`` so the labels' aligned edge stays at one
    # height (top→1.0, center→0.5, bottom/baseline→0.0). Chaining via
    # OffsetFrom at this fraction, with no extra point offset, keeps the
    # labels on one baseline rather than drifting.
    _va_yfrac = {'top': 1.0, 'center': 0.5, 'center_baseline': 0.5,
                 'bottom': 0.0, 'baseline': 0.0}.get(va, 1.0)
    # Shared styling for every label — a dict (rather than explicit kwargs)
    # so a caller-supplied kwarg overrides cleanly instead of colliding.
    shared: dict[str, Any] = dict(va=va, ha=ha, fontsize=fontsize)
    pe = _stroke_path_effects(stroke_color, stroke_lw)
    if pe is not None:
        shared['path_effects'] = pe
    if zorder is not None:
        shared['zorder'] = zorder
    shared.update(kwargs)
    text_ans = []
    an1 = axin.annotate(labels[0], xy=xy, color=labcolors[0],
                        xycoords=axin.transAxes, **shared)
    text_ans.append(an1)
    for i in range(1, len(labels)):
        # Chain each label just past the previous one's right edge (x) at the
        # va-matched y, with NO extra point offset → no vertical drift.
        offset_from = OffsetFrom(text_ans[i - 1], (1 + textpad, _va_yfrac))
        an = axin.annotate(labels[i], xy=(0, 0), color=labcolors[i],
                           xycoords=offset_from, **shared)
        text_ans.append(an)
    return text_ans


###############################################################################
#                                                                             #
#             COMPASS ROSE & SURVEY FOOTPRINTS                    #
#                                                                             #
###############################################################################


def _ink_and_stroke(ax: Any, color: Any, stroke_color: Any) -> tuple[Any, Any]:
    """Resolve the ink / stroke contrast pair for a decoration.

    ``color='k'`` with ``stroke_color='w'`` is a deliberate pair: dark mark,
    light outline, legible over a busy map. The pair only works if BOTH
    members flip together — swapping one alone makes the artist *less*
    legible than the hard-coded original. So they are resolved side by side
    here rather than independently at each call site.

    On a light theme this reproduces the historical black-on-white exactly.
    """
    from matplotlib import rcParams
    if color is None:
        color = rcParams['text.color']
    if stroke_color is None:
        stroke_color = rcParams['axes.facecolor']
    return color, stroke_color


def add_compass(ax: Any, loc: str | tuple[float, float] = 'lower left',
                length: float = 0.08, color: Any = None, fontsize: float = 10,
                lw: float = 1.5, head_width: float = 0.015,
                head_length: float = 0.012, stroke_lw: float = 2,
                stroke_color: Any = None, label_offset: float = 1.3,
                pad: float = 0.05, zorder: int = 10, north_label: str = 'N',
                east_label: str = 'E', **kwargs: Any) -> list[Any]:
    """
    Add a North-East compass indicator to a WCSAxes, similar to those
    on Hubble/JWST images.

    The compass computes the actual WCS-projected N and E directions at
    the arrow origin, so it correctly reflects any rotation, flip, or
    projection distortion in the image.

    Parameters
    ----------
    ax : WCSAxes
        Must have a valid WCS to compute cardinal directions.
    loc : str or tuple
        Location in axes fraction coordinates. Presets:
        'lower left', 'lower right', 'upper left', 'upper right',
        or a (x, y) tuple in axes fraction [0–1].
    length : float
        Arrow length in axes fraction units.
    color : str
        Arrow and label color.
    fontsize : float
    lw : float
        Arrow line width.
    head_width, head_length : float
        Arrow head dimensions in axes-fraction units — converted to points
        against the rendered axes size, so the head is a constant fraction of
        the axes and scales with the figure (a small/thumbnail axes gets a
        proportionally small head). Shrink both to taper the head.
    stroke_lw : float
        Outline stroke width for visibility on busy backgrounds.
    stroke_color : str
        Outline stroke color.
    label_offset : float
        Label distance from arrowhead, as a multiple of arrow length.
    pad : float
        Padding from axes edge for preset locations.
    zorder : int
    north_label, east_label : str
        Labels for the arrows. Set to '' to suppress.
    **kwargs
        Extra ``arrowprops`` for both arrows (forwarded to the underlying
        ``ax.annotate`` arrow), e.g. ``mutation_scale=`` to scale the head, or
        ``linestyle=``. Override the defaults set here.

    Returns
    -------
    artists : list
        List of arrow and text artists added.

    Examples
    --------
    >>> add_compass(ax)  # default lower-left
    >>> add_compass(ax, loc='upper right', color='w', stroke_color='k')
    >>> add_compass(ax, loc=(0.1, 0.9), length=0.12)
    """
    # Resolve preset locations
    loc_presets = {
        'lower left':  (pad + length, pad + length),
        'lower right': (1 - pad - length, pad + length),
        'upper left':  (pad + length, 1 - pad - length),
        'upper right': (1 - pad - length, 1 - pad - length),
    }
    origin: tuple[float, ...]
    if isinstance(loc, str):
        origin = loc_presets.get(loc.lower().replace('_', ' '),
                                (pad + length, pad + length))
    else:
        origin = tuple(loc)

    # Convert axes-fraction origin to display coords, then invert to
    # data (pixel) coords.
    disp_origin = ax.transAxes.transform(origin)
    pix_origin = ax.transData.inverted().transform(disp_origin)

    # Get world coords at origin
    wcs = ax.wcs
    world_origin = wcs.wcs_pix2world([pix_origin], 0)[0]

    # Compute pixel scale and rotation from the WCS
    # Use a small offset in world coords to determine N and E directions
    delta = 0.01  # degrees

    # North: offset in latitude
    world_n = [world_origin[0], world_origin[1] + delta]
    pix_n = wcs.wcs_world2pix([world_n], 0)[0]
    dn = pix_n - pix_origin
    norm_n = np.sqrt(dn[0]**2 + dn[1]**2)
    if norm_n > 0:
        dn = dn / norm_n

    # East: offset in longitude (RA increases to the East, but in most
    # astronomical images with negative CDELT1, East points left)
    world_e = [world_origin[0] + delta / np.cos(np.radians(world_origin[1])),
               world_origin[1]]
    pix_e = wcs.wcs_world2pix([world_e], 0)[0]
    de = pix_e - pix_origin
    norm_e = np.sqrt(de[0]**2 + de[1]**2)
    if norm_e > 0:
        de = de / norm_e

    # Convert pixel direction vectors to axes fraction
    # Scale: 'length' in axes fraction
    disp_per_pix_x = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    disp_per_pix_y = ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1]
    disp_per_axfrac_x = ax.transAxes.transform((1, 0))[0] - ax.transAxes.transform((0, 0))[0]
    disp_per_axfrac_y = ax.transAxes.transform((0, 1))[1] - ax.transAxes.transform((0, 0))[1]

    # Arrow vectors in axes fraction
    scale_x = disp_per_pix_x / disp_per_axfrac_x
    scale_y = disp_per_pix_y / disp_per_axfrac_y

    dn_ax = np.array([dn[0] * scale_x, dn[1] * scale_y])
    dn_ax = dn_ax / np.sqrt(dn_ax[0]**2 + dn_ax[1]**2) * length

    de_ax = np.array([de[0] * scale_x, de[1] * scale_y])
    de_ax = de_ax / np.sqrt(de_ax[0]**2 + de_ax[1]**2) * length

    # Scale-aware arrowheads. The old code hardcoded ``mutation_scale=15``
    # (a fixed points value), so the head was figure-size-independent and
    # dominated a small/thumbnail axes. ``head_length`` / ``head_width`` are
    # documented in axes-fraction units, so convert them to points via the
    # rendered axes size — the head is then a constant FRACTION of the axes
    # and shrinks/grows with the figure. ``mutation_scale=1`` makes the
    # arrowstyle's head_length / head_width (in points) the actual head size.
    try:
        ext = ax.get_window_extent()
        ax_dim_pts = min(ext.width, ext.height) * 72.0 / ax.figure.dpi
    except Exception:
        ax_dim_pts = 72.0 * 4.0   # ~4" fallback if the axes isn't drawn yet
    hl_pts = max(1.0, head_length * ax_dim_pts)
    hw_pts = max(1.0, head_width * ax_dim_pts)
    arrowstyle = f'->,head_length={hl_pts:.4f},head_width={hw_pts:.4f}'
    # Base arrowprops; caller **kwargs are forwarded (and may override, e.g.
    # mutation_scale=, which then scales the head_length / head_width above).
    color, stroke_color = _ink_and_stroke(ax, color, stroke_color)
    # Built AFTER _ink_and_stroke, not before. Resolving the pair first and
    # then building the effects looks like a detail, but the reverse order
    # meant the default stroke_color=None reached _stroke_path_effects
    # unresolved, which returns None -- so the theme-aware stroke was silently
    # dropped and the compass drew unstroked. An explicit stroke_color= still
    # worked, which is why it read as "the default is no stroke" rather than
    # as a bug.
    pe = _stroke_path_effects(stroke_color, stroke_lw) or []
    arrowprops = dict(arrowstyle=arrowstyle, color=color, lw=lw,
                      mutation_scale=1, shrinkA=0, shrinkB=0)
    arrowprops.update(kwargs)

    artists = []

    # Draw N arrow
    arr_n = ax.annotate('', xy=(origin[0] + dn_ax[0], origin[1] + dn_ax[1]),
                        xytext=origin, xycoords='axes fraction',
                        textcoords='axes fraction',
                        arrowprops=dict(arrowprops), zorder=zorder)
    arr_n.arrow_patch.set_path_effects(pe)
    artists.append(arr_n)

    # Draw E arrow (shares the exact same origin tail vertex)
    arr_e = ax.annotate('', xy=(origin[0] + de_ax[0], origin[1] + de_ax[1]),
                        xytext=origin, xycoords='axes fraction',
                        textcoords='axes fraction',
                        arrowprops=dict(arrowprops), zorder=zorder)
    arr_e.arrow_patch.set_path_effects(pe)
    artists.append(arr_e)

    # Labels
    if north_label:
        lbl_n = ax.text(origin[0] + dn_ax[0] * label_offset,
                        origin[1] + dn_ax[1] * label_offset,
                        north_label, transform=ax.transAxes,
                        fontsize=fontsize, fontweight='bold', color=color,
                        ha='center', va='center', zorder=zorder,
                        path_effects=pe)
        artists.append(lbl_n)

    if east_label:
        lbl_e = ax.text(origin[0] + de_ax[0] * label_offset,
                        origin[1] + de_ax[1] * label_offset,
                        east_label, transform=ax.transAxes,
                        fontsize=fontsize, fontweight='bold', color=color,
                        ha='center', va='center', zorder=zorder,
                        path_effects=pe)
        artists.append(lbl_e)

    return artists


def add_axis_inlay(ax: Any, loc: str | tuple[float, float] = 'lower left',
                   size: float = 0.08, lon_label: str | None = None,
                   lat_label: str | None = None, color: Any = None,
                   fontsize: float | str | None = None, lw: float = 1.2,
                   bg_alpha: float = 0.7, bg_color: Any = None,
                   bg_edgecolor: Any = None, bg_lw: float = 0.5,
                   pad: float = 0.03,
                   stroke_lw: float = 1.5, stroke_color: Any = None,
                   zorder: int = 10, arrow_style: str = '->',
                   wireframe: bool = True, wireframe_color: Any = '0.5',
                   wireframe_lw: float = 0.8,
                   wireframe_stroke: Any = None, wireframe_stroke_lw: float = 2.0,
                   lon_invert: bool | str = 'auto',
                   **kwargs: Any) -> list[Any]:
    """
    Add a compact coordinate axis indicator inlay to an all-sky plot.

    Draws a small wireframe outline of the projection shape with
    overlaid arrows showing the longitude and latitude axis directions.
    Replaces conventional axis labels that fight with curved frame
    boundaries and cusp regions on all-sky projections.

    Parameters
    ----------
    ax : WCSAxes
    loc : str or tuple
        'lower left', 'lower right', 'upper left', 'upper right',
        or (x, y) tuple in axes fraction [0–1].
    size : float
        Inlay size in axes fraction units.
    lon_label, lat_label : str, optional
        Axis labels. If None, auto-detected from frame
        (e.g. 'l', 'b' for Galactic; 'RA', 'Dec' for equatorial).
    color : str
    fontsize : float, optional
        Defaults to the current ``axes.labelsize`` rcParam.
    lw : float
        Arrow line width.
    bg_alpha : float
        Background patch transparency. Set 0 for no background.
    bg_color : str
    pad : float
        Padding from axes edge.
    stroke_lw : float
    stroke_color : str
    zorder : int
    arrow_style : str
    wireframe : bool
        If True (default), draw a mini outline of the projection frame
        shape behind the arrows.
    wireframe_color : str
        Color for the wireframe outline.
    wireframe_lw : float
        Line width for the wireframe.
    wireframe_stroke : color spec or None
        Optional legibility stroke behind the wireframe / equator / meridian
        reference lines, for a busy image. Default ``None`` keeps the frame
        deliberately subtle (the main ``stroke_color`` outlines only the arrows
        and labels). Independent of ``wireframe_color``.
    wireframe_stroke_lw : float
        Total stroke width when ``wireframe_stroke`` is set. Default ``2.0``.
    lon_invert : {'auto', True, False}
        Direction of the longitude arrow. ``'auto'`` (default) flips
        the arrow to point left when the underlying axes uses the
        astronomical RA-inverted convention (CDELT1 < 0); arrow points
        right otherwise (Earth-style cartographic convention). Pass
        ``True`` / ``False`` to override the auto-detect explicitly.
        The latitude arrow always points up (north convention).

    Returns
    -------
    artists : list

    Examples
    --------
    >>> add_axis_inlay(ax)  # auto-detect labels and direction, lower-left
    >>> add_axis_inlay(ax, loc='upper right', color='w', bg_color='0.2')
    >>> add_axis_inlay(ax, wireframe=False)  # arrows only, no frame outline
    >>> add_axis_inlay(ax, lon_invert=False)  # force Earth-style (lon→right)
    """
    # Default fontsize to match axis labels
    if fontsize is None:
        fontsize = rcParams.get('axes.labelsize', 12)
        if isinstance(fontsize, str):
            _size_map = {'xx-small': 6, 'x-small': 7.5, 'small': 9,
                         'medium': 10, 'large': 12, 'x-large': 14, 'xx-large': 17}
            fontsize = _size_map.get(fontsize, 10)

    # Auto-detect labels from frame
    if lon_label is None or lat_label is None:
        frame = _detect_frame(ax)
        from ..constants import frame_short_labels
        auto_lon, auto_lat = frame_short_labels(frame)
        if lon_label is None:
            lon_label = auto_lon
        if lat_label is None:
            lat_label = auto_lat

    # Resolve lon_invert: 'auto' detects the on-screen east direction —
    # when lon increases right-to-left (RA-inverted astronomical
    # convention) the arrow should point left. Probe the actual world→pixel
    # mapping (a raw CDELT1 sign is wrong for CD-matrix WCS, where CDELT1
    # is often +1 and the scale lives in the CD matrix).
    if lon_invert == 'auto':
        try:
            from ..wcs_frame import _east_increases_right
            lon_invert = not _east_increases_right(ax.wcs)
        except Exception:
            lon_invert = False

    # Compute aspect-corrected sizes so arrows appear equal length
    # in display space regardless of figure/axes aspect ratio
    bbox = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()
    ax_w = bbox.width * fig_w
    ax_h = bbox.height * fig_h
    ax_aspect = ax_w / max(ax_h, 1e-6)

    if ax_aspect >= 1:
        x_size = size / ax_aspect
        y_size = size
    else:
        x_size = size
        y_size = size * ax_aspect

    # Layout: L-shaped arrows from a corner origin, wireframe nestled inside.
    # ``xs`` is +1 for Earth-style (lon arrow points right), -1 for
    # RA-inverted convention (lon arrow points left). All x-direction
    # offsets are multiplied by it so the layout mirrors cleanly.
    xs = -1 if lon_invert else 1
    x_label_pad = x_size * 0.35
    y_label_pad = y_size * 0.35
    inlay_w = x_size + x_label_pad
    if lon_invert:
        # Inlay extends LEFT of origin; flip the corner anchors so the
        # whole thing still sits in the chosen quadrant.
        loc_presets = {
            'lower left':  (pad + inlay_w, pad),
            'lower right': (1 - pad, pad),
            'upper left':  (pad + inlay_w, 1 - pad - y_size - y_label_pad),
            'upper right': (1 - pad, 1 - pad - y_size - y_label_pad),
        }
    else:
        loc_presets = {
            'lower left':  (pad, pad),
            'lower right': (1 - pad - inlay_w, pad),
            'upper left':  (pad, 1 - pad - y_size - y_label_pad),
            'upper right': (1 - pad - inlay_w,
                            1 - pad - y_size - y_label_pad),
        }
    origin: tuple[float, ...]
    if isinstance(loc, str):
        origin = loc_presets.get(loc.lower().replace('_', ' '),
                                loc_presets['lower left'])
    else:
        origin = tuple(loc)

    color, stroke_color = _ink_and_stroke(ax, color, stroke_color)
    if bg_color is None:
        bg_color = rcParams['axes.facecolor']
    if bg_edgecolor is None:
        from ..style import muted_ink
        bg_edgecolor = muted_ink(ax, light='0.6')
    pe = _stroke_path_effects(stroke_color, stroke_lw) or []
    artists = []

    # Mini wireframe nestled in the quadrant between the two arrows
    if wireframe:
        frame_obj = ax.coords.frame
        if hasattr(frame_obj, '_boundary_x'):
            bnd_func = frame_obj._boundary_x
        elif isinstance(frame_obj, EllipticalFrame):
            def bnd_func(t: npt.ArrayLike) -> Any:
                return np.sqrt(np.maximum(1 - np.asarray(t) ** 2, 0))
        else:
            bnd_func = None

        if bnd_func is not None:
            t_pts = np.linspace(-1, 1, 100)
            bx = bnd_func(t_pts)

            # Wireframe centered in the arrow quadrant
            wf_cx = origin[0] + xs * x_size * 0.5
            wf_cy = origin[1] + y_size * 0.5

            # Scale wireframe to fit snugly inside the arrow L-shape,
            # using aspect-corrected sizes so it looks proportional
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            frame_aspect = (xlim[1] - xlim[0]) / max(ylim[1] - ylim[0], 1e-6)
            half_h = y_size * 0.35
            half_w = half_h * min(frame_aspect, 2.5) * (x_size / max(y_size, 1e-6))
            if half_w > x_size * 0.42:
                scale_f = x_size * 0.42 / half_w
                half_w *= scale_f
                half_h *= scale_f

            # Right boundary
            x_right = wf_cx + half_w * bx
            y_curve = wf_cy + half_h * t_pts
            # Left boundary (mirror)
            x_left = wf_cx - half_w * bx[::-1]
            y_left = wf_cy + half_h * t_pts[::-1]

            wf_x = np.concatenate([x_right, x_left, [x_right[0]]])
            wf_y = np.concatenate([y_curve, y_left, [y_curve[0]]])

            # Optional legibility stroke for the (deliberately subtle)
            # reference frame — off by default so the wireframe stays quiet;
            # opt in via ``wireframe_stroke`` to outline it on a busy image.
            _wf_pe = _stroke_path_effects(wireframe_stroke, wireframe_stroke_lw)

            wf_line, = ax.plot(wf_x, wf_y, color=wireframe_color,
                               lw=wireframe_lw, ls='-',
                               transform=ax.transAxes, zorder=zorder - 0.5,
                               clip_on=False, path_effects=_wf_pe)
            artists.append(wf_line)

            # Dotted equator
            bx0 = bnd_func(np.array([0.]))[0]
            eq_line, = ax.plot(
                [wf_cx - half_w * bx0, wf_cx + half_w * bx0],
                [wf_cy, wf_cy],
                color=wireframe_color, lw=wireframe_lw * 0.6, ls=':',
                transform=ax.transAxes, zorder=zorder - 0.5, clip_on=False,
                path_effects=_wf_pe)
            artists.append(eq_line)

            # Dotted central meridian
            cm_line, = ax.plot(
                [wf_cx, wf_cx],
                [wf_cy - half_h, wf_cy + half_h],
                color=wireframe_color, lw=wireframe_lw * 0.6, ls=':',
                transform=ax.transAxes, zorder=zorder - 0.5, clip_on=False,
                path_effects=_wf_pe)
            artists.append(cm_line)

    # Arrow properties, with ``**kwargs`` layered on top — same contract as
    # ``add_compass``. These were previously accepted and silently dropped.
    arrowprops = dict(arrowstyle=arrow_style, color=color, lw=lw,
                      mutation_scale=15, shrinkA=0, shrinkB=0)
    arrowprops.update(kwargs)

    # Longitude arrow (horizontal). Direction follows ``xs``: points
    # right for Earth-style (xs=+1), left for RA-inverted (xs=-1).
    arr_lon = ax.annotate(
        '', xy=(origin[0] + xs * x_size, origin[1]),
        xytext=origin, xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowprops),
        zorder=zorder, clip_on=False)
    arr_lon.arrow_patch.set_path_effects(pe)
    artists.append(arr_lon)

    # Latitude arrow (vertical, pointing up) — prominent
    arr_lat = ax.annotate(
        '', xy=(origin[0], origin[1] + y_size),
        xytext=origin, xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowprops),
        zorder=zorder, clip_on=False)
    arr_lat.arrow_patch.set_path_effects(pe)
    artists.append(arr_lat)

    # Labels at arrowheads. Lon label sits past the arrow tip in the
    # direction of the arrow; horizontal alignment flips with ``xs``.
    lbl_lon = ax.text(origin[0] + xs * (x_size + x_label_pad * 0.3),
                      origin[1],
                      lon_label, transform=ax.transAxes,
                      fontsize=fontsize, color=color,
                      ha='left' if xs > 0 else 'right', va='center',
                      zorder=zorder, path_effects=pe, clip_on=False)
    artists.append(lbl_lon)

    lbl_lat = ax.text(origin[0], origin[1] + y_size + y_label_pad * 0.3,
                      lat_label, transform=ax.transAxes,
                      fontsize=fontsize, color=color,
                      ha='center', va='bottom', zorder=zorder,
                      path_effects=pe, clip_on=False)
    artists.append(lbl_lat)

    # Background box — sized to the *rendered* extent of the arrows,
    # labels and wireframe (not a geometric estimate), so it always
    # contains the inlay whatever the fontsize, label text, or lon-inverted
    # layout (the direction-specific box anchor falls out for free). A
    # one-off draw realizes the arrow/text geometry so their window extents
    # are valid; if no renderer is available (some non-Agg save paths) we
    # fall back to the old geometric box.
    if bg_alpha > 0:
        from matplotlib.patches import FancyBboxPatch
        box = None
        try:
            from matplotlib.transforms import Bbox
            fig = ax.figure
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            extents = [getattr(a, 'arrow_patch', a).get_window_extent(renderer)
                       for a in artists]
            (bx0, by0), (bx1, by1) = ax.transAxes.inverted().transform(
                Bbox.union(extents).get_points())
            m = size * 0.14
            box = FancyBboxPatch(
                (bx0 - m, by0 - m), (bx1 - bx0) + 2 * m, (by1 - by0) + 2 * m,
                boxstyle='round,pad=0.01',
                facecolor=bg_color, edgecolor=bg_edgecolor, lw=bg_lw,
                alpha=bg_alpha, transform=ax.transAxes, zorder=zorder - 1,
                clip_on=False)
        except Exception:
            # Geometric fallback (previous behavior): fixed margins around
            # the arrow layout, without measuring the rendered text.
            mx, my = x_size * 0.06, y_size * 0.06
            box_x = (origin[0] - mx if not lon_invert
                     else origin[0] - inlay_w - mx)
            box = FancyBboxPatch(
                (box_x, origin[1] - my), inlay_w + mx * 2,
                y_size + y_label_pad + my * 2, boxstyle='round,pad=0.01',
                facecolor=bg_color, edgecolor=bg_edgecolor, lw=bg_lw,
                alpha=bg_alpha, transform=ax.transAxes, zorder=zorder - 1,
                clip_on=False)
        ax.add_patch(box)
        artists.insert(0, box)   # keep the box first so it draws behind

    return artists


# ===== style_ax_colors =====

def style_ax_colors(ax: Any, color: Any = 'white') -> None:
    ax.tick_params(colors=color)           # tick marks + tick labels
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)
    ax.title.set_color(color)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
