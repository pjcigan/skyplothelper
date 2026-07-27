"""Public auto-sized tick labels (:func:`auto_size_ticklabels`).

A single entry point that picks a tick-label fontsize fitted to the
available axes width, so plots at typical figure sizes are visually
unchanged but tight panels (small figsize, dense subplot grids) get
appropriately shrunk labels. The function dispatches on the axes
type so the same call works on:

* :class:`~astropy.visualization.wcsaxes.WCSAxes` — sized from both
  coords; applied via :meth:`coord.set_ticklabel`.
* Plain matplotlib :class:`~matplotlib.axes.Axes` — sized from the
  rendered tick labels; applied via :meth:`~matplotlib.axes.Axes.tick_params`.
* :mod:`cartopy` ``GeoAxes`` — sized from each attached Gridliner's
  rendered ``xlabel_artists`` / ``ylabel_artists``; applied via the
  Gridliner's ``xlabel_style`` / ``ylabel_style`` dicts (so subsequent
  draws use the new size) *and* directly on the current label artists
  (so the change shows up without requiring a redraw).

Most users won't call this directly — :func:`make_wcs_frame`,
:func:`make_globe_frame`, :func:`apply_offset_ticks`, and
:func:`apply_anchored_offset` invoke it under the hood at the right
moments via their ``auto_fontsize=True`` defaults. The public surface
is here for two cases:

1. **Re-fitting after layout changes.** A user who calls
   ``fig.set_size_inches(...)`` after ``make_wcs_frame`` can call
   ``sph.auto_size_ticklabels(ax)`` to re-size labels for the new
   axes geometry.
2. **Reflow on resize.** Passing ``reflow_on_resize=True`` attaches
   a :class:`~matplotlib.backend_bases.FigureCanvasBase` resize-event
   callback so the labels follow figure-window changes automatically.

A snapshot of the chosen fontsize is cached on the axes as
``ax._sph_auto_label_fontsize`` so downstream helpers (notably
:func:`~skyplothelper.coord_overlay.add_overlay_ticks`) can inherit it
without an explicit kwarg.
"""

from __future__ import annotations

import warnings
from typing import Any

from ._text_layout import (
    _apply_auto_label_fontsize_to_cartopy,
    _apply_auto_label_fontsize_to_mpl,
    _apply_auto_label_fontsize_to_wcs,
    _is_cartopy_axes,
)

__all__ = ['auto_size_ticklabels']


def auto_size_ticklabels(ax: Any, *, axis: str = 'both', floor: float = 6.0,
                         ceiling: float | None = None, n_ticks_hint: int = 6,
                         reflow_on_resize: bool = False) -> float | None:
    """Pick + apply an auto-fitted tick-label fontsize for *ax*.

    Sized via ``axes_width_pt / (n_ticks * n_chars * char_factor
    * safety)``, clipped to ``[floor, ceiling]``. Default ``ceiling``
    is :data:`rcParams['xtick.labelsize'] <matplotlib.rcParams>` so the
    helper never *grows* labels past the user's matplotlib default —
    it only shrinks when the geometry is tight. See
    :func:`skyplothelper._text_layout._auto_label_fontsize` for the
    underlying math.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes whose tick labels should be sized. Dispatch on type:

        * WCSAxes — sized from both coords; applied via
          ``ax.coords[i].set_ticklabel(fontsize=...)``.
        * Plain :class:`~matplotlib.axes.Axes` — sized from rendered
          tick labels; applied via
          ``ax.tick_params(axis=..., labelsize=...)``.
        * Cartopy ``GeoAxes`` — sized from each attached Gridliner's
          rendered labels; applied via the Gridliner's
          ``xlabel_style`` / ``ylabel_style`` dicts (and directly on
          the current label artists for immediate visibility). Returns
          ``None`` (no warning) if the GeoAxes has no gridliners —
          nothing to size.
        * Anything else — warns + returns ``None``.
    axis : {'x', 'y', 'both'}, optional
        Which axis to size. Default ``'both'`` — uses the wider of
        the two axes' label widths and applies the single fontsize to
        both. ``'x'`` / ``'y'`` apply to that axis only. Ignored on
        WCSAxes (which always sizes both coords together — astropy's
        coord model doesn't map cleanly to the x / y dichotomy).
    floor : float, optional
        Minimum fontsize in points. Default ``6.0`` — readable on
        typical screen / print densities; smaller and the glyph
        detail breaks down.
    ceiling : float, optional
        Maximum fontsize in points. ``None`` (default) → the matplotlib
        rcParams default for the matching axis. Pass an explicit number
        to override.
    n_ticks_hint : int, optional
        Expected number of ticks across the constraining axis dimension.
        Default ``6`` — matches the ``make_wcs_frame`` ``lon_spacing='auto'``
        default of ~8 ticks, accounting for some clipping on
        non-rectangular projections.
    reflow_on_resize : bool, optional
        If ``True``, attach callbacks that re-run
        :func:`auto_size_ticklabels` whenever the axes layout
        changes. Three event sources are wired:

        * Figure-canvas ``resize_event`` — window resize (interactive
          drag or programmatic ``set_size_inches`` on most backends).
        * Axes ``xlim_changed`` — pan / zoom / explicit ``set_xlim``.
          Catches the case where the axes pixel dimensions don't
          change but the rendered label widths do (astropy switches
          between ``"12h"`` / ``"12h00m"`` / ``"12h00m00s"`` as the
          tick spacing changes with zoom).
        * Axes ``ylim_changed`` — same for the Y / Dec direction.

        Default ``False`` — most callers want the snapshot-at-construction
        semantics, since auto-mutating fontsize across redraws can
        surprise users, and interactive panning fires the limit-change
        events continuously during a drag (each fire is cheap but not
        free). Idempotent — calling :func:`auto_size_ticklabels` twice
        on the same axes doesn't stack callbacks.

    Returns
    -------
    float or None
        The chosen fontsize in points. ``None`` when dispatch can't
        proceed (unsupported axes type — see the parameter docs).

    Examples
    --------
    Re-fit labels after resizing the figure::

        fig, ax = plt.subplots(figsize=(8, 6))
        ax = sph.make_wcs_frame(111, 'AIT', center=180, fig=fig)
        # ... change figure size later ...
        fig.set_size_inches(4, 3)
        sph.auto_size_ticklabels(ax)   # re-fit for the new size

    Opt into reflow on every window resize::

        sph.auto_size_ticklabels(ax, reflow_on_resize=True)

    Apply only to the x axis with a tighter floor::

        sph.auto_size_ticklabels(ax, axis='x', floor=4.0)
    """
    fontsize = _dispatch_auto_size(
        ax, axis=axis, floor=floor, ceiling=ceiling,
        n_ticks_hint=n_ticks_hint)

    if reflow_on_resize and fontsize is not None:
        _attach_resize_reflow(ax, axis=axis, floor=floor, ceiling=ceiling,
                              n_ticks_hint=n_ticks_hint)

    return fontsize


def _dispatch_auto_size(ax: Any, *, axis: str, floor: float,
                        ceiling: float | None,
                        n_ticks_hint: int) -> float | None:
    """Pick the right per-axes-type apply function and call it.

    Separated from :func:`auto_size_ticklabels` so the reflow callback
    can recurse via the same dispatch without re-attaching itself.
    """
    if hasattr(ax, 'coords'):
        # WCSAxes — axis= ignored (both coords sized together).
        return _apply_auto_label_fontsize_to_wcs(
            ax, n_ticks_hint=n_ticks_hint,
            floor=floor, ceiling=ceiling)

    if _is_cartopy_axes(ax):
        return _apply_auto_label_fontsize_to_cartopy(
            ax, axis=axis, n_ticks_hint=n_ticks_hint,
            floor=floor, ceiling=ceiling)

    # Plain matplotlib axes — covers everything else with a tick_params
    # method. Subclasses that aren't WCSAxes / cartopy fall through to
    # this path.
    if hasattr(ax, 'tick_params'):
        return _apply_auto_label_fontsize_to_mpl(
            ax, axis=axis, n_ticks_hint=n_ticks_hint,
            floor=floor, ceiling=ceiling)

    warnings.warn(
        f"auto_size_ticklabels: unrecognized axes type "
        f"{type(ax).__name__!r}; skipping.",
        UserWarning, stacklevel=3)
    return None


def _attach_resize_reflow(ax: Any, **kwargs: Any) -> None:
    """Idempotently attach layout-event callbacks to *ax*'s figure +
    axes so the auto-fontsize re-fits when the layout changes.

    Three event sources are wired:

    * ``resize_event`` on ``ax.figure.canvas`` — fires on figure
      window resize (interactive or programmatic ``set_size_inches``
      on most backends).
    * ``xlim_changed`` on the axes — fires on pan / zoom / explicit
      ``set_xlim``. Catches the case where pan/zoom doesn't change
      the axes pixel dimensions but the *rendered label widths* do
      (e.g. astropy switches from ``"12h"`` at coarse spacing to
      ``"12h00m00s"`` at fine spacing).
    * ``ylim_changed`` on the axes — same, for the Dec / Y direction.

    Each callback re-runs :func:`_dispatch_auto_size` (not
    :func:`auto_size_ticklabels` — that would re-attach the
    callbacks). Tracks all callback connection ids on
    ``ax._sph_autosize_cids`` so repeat calls don't stack.

    Trade-off: in interactive use, panning / zooming fires
    xlim_changed / ylim_changed continuously during a drag, and
    each fire runs the dispatcher (cheap but not free). If you find
    this noisy, call ``auto_size_ticklabels(ax, reflow_on_resize=False)``
    once after pan/zoom completes instead of using the reflow
    callback.
    """
    try:
        fig = ax.figure
    except AttributeError:
        return

    existing_cids = getattr(ax, '_sph_autosize_cids', None)
    if existing_cids:
        # Already attached — nothing to do.
        return

    def _on_layout_change(_event: Any = None) -> None:
        try:
            _dispatch_auto_size(ax, **kwargs)
        except Exception as exc:
            warnings.warn(
                f"auto_size_ticklabels reflow callback failed: "
                f"{type(exc).__name__}: {exc}",
                UserWarning, stacklevel=2)

    cids = []
    # Figure-canvas resize hook — most-likely-supported, attach first.
    try:
        cids.append(
            ('canvas', fig.canvas.mpl_connect(
                'resize_event', _on_layout_change)))
    except (AttributeError, ValueError):
        pass
    # Axes-level pan / zoom hooks. Older mpl uses ``callbacks.connect``;
    # this is the canonical path. If unavailable, fall through silently.
    try:
        cids.append(
            ('xlim', ax.callbacks.connect(
                'xlim_changed', _on_layout_change)))
        cids.append(
            ('ylim', ax.callbacks.connect(
                'ylim_changed', _on_layout_change)))
    except (AttributeError, ValueError):
        pass

    if cids:
        ax._sph_autosize_cids = cids
    # Preserve the old single-cid attribute for backwards compatibility
    # — callers / tests that still check ``ax._sph_autosize_cid`` see
    # the resize-event cid (the only one that existed in the first
    # version of this feature).
    for kind, cid in cids:
        if kind == 'canvas':
            ax._sph_autosize_cid = cid
            break
