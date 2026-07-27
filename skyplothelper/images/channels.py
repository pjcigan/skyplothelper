"""Channel maps from spectral-line data cubes.

:func:`channel_map` turns a 3-D cube (channel, y, x) into a compact grid of
sky-image panels sharing one normalization, with per-panel spectral labels, a
single colorbar, and optional beam / scale bar — the standard static view of a
spectral cube. It replaces the manual scaffolding (load + squeeze, one shared
norm, subplot loop, velocity labels, shared colorbar, sparse ticks) that every
channel-map figure otherwise repeats.

The cube-handling core lives in :class:`~skyplothelper.DataCube` (load/squeeze,
celestial + spectral WCS, the per-channel spectral world array, BUNIT) so the
same normalization and labeling can back an animation or a plotly cube viewer
as easily as the static grid; :func:`channel_map` accepts either a raw cube or
a :class:`~skyplothelper.DataCube`.

Per-panel *contours* and *custom comparison panels* are intentionally left to
post-hoc editing of the returned axes (``for ax in res.axes.flat: ax.contour(...)``)
rather than parsed here — infinitely more flexible than constructor machinery.
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple, Sequence

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy.io import fits as pyfits

from .._stroke import _stroke_path_effects
from .cube import (
    _DEFAULT_FMT,
    DataCube,
    _block_average,
    _clean_unit,
    _hanning_smooth,
    _spectral_label_value,
)
from .levels import make_norm


class ChannelMapResult(NamedTuple):
    """Return type for :func:`channel_map`.

    Tuple-unpackable and attribute-accessible; every internally-created artist
    is reachable so the grid can be tweaked after the fact (add contours, swap
    a panel, restyle, ...). Use :meth:`panel` to fetch the axes for a channel.
    """
    fig:           Any     # matplotlib.figure.Figure
    axes:          Any     # 2-D ndarray of the panel Axes / WCSAxes
    images:        list    # the AxesImage per drawn channel
    colorbar:      Any     # Colorbar (None if colorbar=False)
    norm:          Any     # the shared Normalize applied to every channel panel
    channels:      Any     # ndarray of (processed) channel indices drawn
    velocities:    list    # per-channel (value, unit) or None, aligned to channels
    labels:        list    # the per-channel corner Text artists ([] where none)
    beam:          Any     # the Beam artist (None if beam=False)
    scalebar:      Any     # the scale-bar artist (None if scalebar is None)
    moment0_image: Any     # the moment-0 AxesImage (None unless moment0=True)
    moment0_units: Any     # mom-0 units string (BUNIT × spectral unit) or None

    def panel(self, channel: int) -> Any:
        """Return the axes displaying (processed) channel index *channel*."""
        idx = list(self.channels).index(int(channel))
        return self.images[idx].axes


# ---------------------------------------------------------------------------
# Channel preprocessing (optional convenience — power users pre-curate).
# The per-cube transforms (_hanning_smooth / _block_average) live in cube.py
# so DataCube and this orchestration share one implementation; _preprocess is
# the channel_map-specific ordering (trim → smooth → average → every-Nth).
# ---------------------------------------------------------------------------

def _preprocess(data: npt.NDArray, world: npt.NDArray | None,
                trim_empty: bool, smooth: str | None, smooth_width: int,
                average: int | None, every_N: int,
                ) -> tuple[npt.NDArray, npt.NDArray | None]:
    """Apply trim → smooth → average → every-Nth to the cube + world array."""
    if trim_empty:
        keep = ~np.all(~np.isfinite(data), axis=(1, 2))   # drop all-NaN planes
        if keep.any():
            data = data[keep]
            world = world[keep] if world is not None else None
    if smooth == "hanning":
        data = _hanning_smooth(data, smooth_width)
    elif smooth is not None:
        raise ValueError(f"smooth must be 'hanning' or None, got {smooth!r}.")
    if average is not None and int(average) > 1:
        data, world = _block_average(data, world, average)
    if int(every_N) > 1:
        data = data[:: int(every_N)]
        world = world[:: int(every_N)] if world is not None else None
    return data, world


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------

def _resolve_channels(channels: int | Sequence[int] | None,
                      nchan: int) -> npt.NDArray:
    """``channels`` → integer channel indices into the (processed) cube."""
    if channels is None:
        return np.arange(nchan)
    if np.isscalar(channels):
        n = int(channels)  # type: ignore[arg-type]
        if n < 1:
            raise ValueError(f"channels count must be >= 1, got {n}.")
        return np.unique(np.linspace(0, nchan - 1, n).round().astype(int))
    idx = np.asarray(channels, dtype=int)
    idx = np.where(idx < 0, idx + nchan, idx)
    if idx.min() < 0 or idx.max() >= nchan:
        raise ValueError(
            f"channel indices out of range for a {nchan}-channel cube.")
    return idx


def _resolve_panel(spec: Any, nrows: int, ncols: int,
                   occupied: list[int]) -> int:
    """A panel specifier → a flat grid index among the *occupied* cells.

    ``spec`` is an ``int`` flat index, or a corner name
    ``'lower left'`` / ``'lower right'`` / ``'upper left'`` / ``'upper right'``
    (resolved to the occupied cell nearest that corner).
    """
    if isinstance(spec, (int, np.integer)):
        return int(spec)
    name = str(spec).lower().replace("_", " ").strip()
    rows = [i // ncols for i in occupied]
    want_bottom = "lower" in name or "bottom" in name
    want_right = "right" in name
    tgt_row = max(rows) if want_bottom else min(rows)
    # among cells in that row, pick the extreme column
    row_cells = [i for i in occupied if i // ncols == tgt_row]
    return max(row_cells) if want_right else min(row_cells)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def channel_map(cube: Any, *,
                header: pyfits.Header | None = None,
                # --- channel preprocessing + selection ---
                channels: int | Sequence[int] | None = 9,
                every_N: int = 1, average: int | None = None,
                smooth: str | None = None, smooth_width: int = 3,
                trim_empty: bool = False, order: str | None = None,
                ncols: int = 3, nrows: int | None = None,
                start_panel: int = 0,
                # --- normalization + colormap ---
                stretch: Any = "linear",
                vmin: float | None = None, vmax: float | None = None,
                plo: float = 0.5, phi: float = 99.8, norm: Any = None,
                cmap: str = "sph.lagoon",
                # --- panels / layout ---
                wcs_panels: bool = True, pad: float | None = None,
                wspace: float | None = None, hspace: float | None = None,
                panel_facecolor: Any = None,
                tick_direction: str = "in", tick_labelsize: float | None = None,
                figsize: tuple[float, float] | None = None,
                facecolor: str | None = None, suptitle: str | None = None,
                imshow_kwargs: dict[str, Any] | None = None,
                # --- ticks + coordinates ---
                ticks: str = "minimal", label_panel: Any = "lower left",
                coords: str = "sky",
                # --- spectral labels ---
                label: Any = "auto", label_unit: str | None = None,
                restfreq: Any = None, vsys: float | None = None,
                label_fmt: str | None = None,
                label_color: str = "white", label_fontsize: float = 10.0,
                label_stroke_lw: float = 2.0, label_stroke_color: str = "black",
                label_kwargs: dict[str, Any] | None = None,
                # --- colorbar / beam / scale bar ---
                colorbar: bool = True, cbar_label: str | None = None,
                cbar_pad: float = 0.12,
                beam: bool = False, beam_panel: Any = "lower right",
                beam_kwargs: dict[str, Any] | None = None,
                scalebar: float | None = None, scalebar_label: str | None = None,
                scalebar_panel: Any = "lower right",
                scalebar_kwargs: dict[str, Any] | None = None,
                # --- moment-0 summary panel ---
                moment0: bool = False, moment0_panel: Any = "upper left",
                moment0_label: str | None = "moment 0",
                moment0_cmap: str | None = None,
                ) -> ChannelMapResult:
    """Compact grid of channel-map panels from a spectral data cube.

    Every panel shares one normalization (so brightness is comparable across
    channels) and one colorbar, and is labeled with its spectral coordinate
    (velocity / frequency / wavelength, from the cube's spectral WCS). Sparse
    ticks by default: full ticks + labels on one reference panel (lower-left),
    tick marks only elsewhere.

    Parameters
    ----------
    cube : ndarray, HDU, HDUList, or str
        A 3-D ``(channel, y, x)`` cube, or a FITS path / HDU whose data is one
        (degenerate Stokes/spectral axes are squeezed away).
    header : astropy.io.fits.Header, optional
        WCS header. Read from the FITS input when not given; needed for sky
        panels and spectral labels.
    channels : int, sequence of int, or None
        Channels to show (from the *preprocessed* cube): an ``int`` picks that
        many evenly spaced (default 9), a sequence gives explicit indices
        (negatives count from the end), ``None`` shows all.
    every_N, average, smooth, smooth_width : channel preprocessing
        Applied to the cube *before* selection, in order: ``smooth='hanning'``
        with ``smooth_width`` channels, then ``average`` (block-average every N
        channels), then ``every_N`` (keep every Nth). All default to no-op
        (``every_N=1``) — a quick way to thin/smooth a raw cube without
        pre-computing, though for publication you'll usually curate first.
    trim_empty : bool
        Drop all-NaN channels before selection (default ``False``).
    order : {'ascending', 'descending', None}
        Reorder the displayed panels by spectral value (blue→red or red→blue);
        ``None`` (default) keeps the selected order.
    ncols, nrows : int
        Grid shape. ``nrows`` inferred from the channel count unless given.
    start_panel : int
        Leave this many leading grid cells blank, starting the channels at
        cell ``start_panel`` (e.g. 10 channels beginning at cell 2 of a 3×4
        grid).
    stretch, vmin, vmax, plo, phi, norm : normalization
        One shared norm via :func:`make_norm`. ``vmin``/``vmax`` default to the
        ``plo``/``phi`` percentiles (0.5 / 99.8) of the whole cube. ``norm``
        overrides.
    cmap : str
        Colormap (default the sequential ``'sph.lagoon'``).
    wcs_panels : bool
        WCSAxes panels with sky coordinates (default); ``False`` = bare tiles.
    pad : float, optional
        Padding between panels in inches (both axes; ImageGrid ``axes_pad``).
        ``0`` = touching. Default is mode-aware: tight (0.06) for the sparse
        ``'minimal'``/``'plain'`` looks, wider (0.4) for ``'complete'`` so the
        per-panel labels have room.
    wspace, hspace : float, optional
        Horizontal / vertical padding (inches) overriding ``pad`` per axis,
        for independent control of the column vs row gaps.
    tick_direction : {'in', 'out'}
        Tick-mark direction (default ``'in'`` — keeps marks out of the tight
        gaps, the channel-map convention).
    tick_labelsize : float, optional
        Tick-label font size. Default mode-aware (8, or 7 for ``'complete'``).
    panel_facecolor : color or 'cmap_min', optional
        Fill each panel's background — ``'cmap_min'`` uses the colormap's low
        end so blank regions blend seamlessly; a color sets it directly.
    figsize, facecolor, suptitle, imshow_kwargs : figure-level controls.
    ticks : {'minimal', 'plain', 'complete'}
        Tick density. ``'minimal'`` (default) = full ticks + labels on the
        ``label_panel`` only, tick marks (no labels) on the rest;
        ``'plain'`` = no ticks or labels anywhere; ``'complete'`` = full on
        every panel.
    label_panel : int or {'lower left','lower right','upper left','upper right'}
        Which panel carries the coordinate labels under ``ticks='minimal'``
        (default lower-left).
    coords : {'sky', 'offset'}
        ``'sky'`` = absolute RA/Dec (default), with frame-appropriate axis
        labels (``'R.A. (J2000)'`` …); ``'offset'`` = ΔRA/ΔDec arcsec offsets
        from the field center, via :func:`apply_offset_ticks` (the
        compact-source convention). All panels share one tick grid.
    label : {'auto','velocity','frequency','wavelength','redshift','channel'} \
or callable or None
        Per-panel corner label. ``'auto'`` (default) uses the spectral WCS's
        own kind; the explicit kinds force a representation (``'velocity'`` /
        ``'redshift'`` on a frequency axis need ``restfreq``). ``'channel'``
        shows the index; a callable receives ``(ch, view)``; ``None`` draws
        none (add your own afterward).
    label_unit, restfreq, vsys : label controls
        Target unit (e.g. ``'km/s'``, ``'GHz'``, ``'um'``), rest frequency
        (float Hz or Quantity; read from ``RESTFRQ`` when absent) for
        frequency↔velocity/redshift, and a systemic velocity subtracted from
        velocity labels.
    label_fmt, label_color, label_fontsize, label_stroke_lw, \
label_stroke_color, label_kwargs : label styling
        White text with a black stroke by default.
    colorbar, cbar_label, cbar_pad : one shared, integrated colorbar
        (label defaults to ``BUNIT``; ``cbar_pad`` in inches).
    beam, beam_panel, beam_kwargs : bool / panel / dict
        Draw the header beam (``BMAJ``/``BMIN``/``BPA``) once, anchored in
        ``beam_panel`` (default lower-right), via :class:`~skyplothelper.Beam`.
    scalebar, scalebar_label, scalebar_panel, scalebar_kwargs : float / ...
        Draw a scale bar of ``scalebar`` arcsec once, in ``scalebar_panel``,
        via :func:`~skyplothelper.add_sizebar_asec`.
    moment0, moment0_panel, moment0_label, moment0_cmap : summary panel
        Add a velocity-integrated moment-0 map as one extra panel (default
        ``'upper left'``) with its own norm and (optionally different)
        colormap — a compact context view alongside the channels. It keeps its
        own norm because ∫I dv is far brighter than any one channel. Reachable
        as ``result.moment0_image``; ``result.moment0_units`` gives the
        integrated units (BUNIT × spectral unit) for a post-hoc colorbar.

    Returns
    -------
    ChannelMapResult
        ``(fig, axes, images, colorbar, norm, channels, velocities, labels,
        beam, scalebar, moment0_image, moment0_units)`` — all tweakable. Add
        per-panel contours, a custom comparison panel, or a moment-0 colorbar
        by editing ``res.axes`` / ``res.moment0_image`` directly.

    Examples
    --------
    >>> res = sph.channel_map("co.fits", channels=12, ncols=4, pad=0,
    ...                       coords='offset', beam=True, scalebar=0.5)
    >>> for ax in res.axes.flat:                 # cyan reference contours
    ...     ax.contour(ref_img, levels=levs, colors='cyan')

    >>> # moment-0 panel in its own colormap with a dedicated left-side colorbar
    >>> # (mode='inset' — the default 'divider' fights ImageGrid's layout):
    >>> res = sph.channel_map("co.fits", moment0=True, moment0_cmap='magma')
    >>> sph.add_colorbar(res.moment0_image, ax=res.moment0_image.axes,
    ...                  mode='inset', location='left', label=res.moment0_units)
    """
    from astropy.visualization.wcsaxes import WCSAxes

    from ..overlays.annotations import add_sizebar_asec
    from ..overlays.beam import Beam
    from ..ticks import apply_offset_ticks

    if ticks not in ("minimal", "plain", "complete"):
        raise ValueError(
            f"ticks must be 'minimal', 'plain', or 'complete', got {ticks!r}.")
    if coords not in ("sky", "offset"):
        raise ValueError(f"coords must be 'sky' or 'offset', got {coords!r}.")

    # Mode-aware spacing + tick font: 'complete' labels every panel, so it needs
    # wider gaps and a smaller font to keep labels off the neighbors; the sparse
    # modes tuck tight. Inward ticks (default) keep marks out of the gaps.
    if pad is None:
        pad = 0.4 if ticks == "complete" else 0.06
    if tick_labelsize is None:
        tick_labelsize = 7.0 if ticks == "complete" else 8.0

    # Accept a raw cube (ndarray / HDU / path / SpectralCube) or a DataCube.
    view = cube if isinstance(cube, DataCube) else DataCube(cube, header)
    if restfreq is not None and not hasattr(restfreq, "unit"):
        restfreq = float(restfreq) * u.Hz
    restfreq = restfreq if restfreq is not None else view.restfreq

    # Preprocess the cube (trim / smooth / average / thin) + world array.
    data, world = _preprocess(view.data, view.world, trim_empty, smooth,
                              smooth_width, average, every_N)
    nchan = data.shape[0]
    chans = _resolve_channels(channels, nchan)

    # Optional spectral ordering of the displayed panels.
    if order is not None:
        if order not in ("ascending", "descending"):
            raise ValueError(
                f"order must be 'ascending', 'descending', or None, "
                f"got {order!r}.")
        if world is not None:
            sidx = np.argsort(world[chans])
            if order == "descending":
                sidx = sidx[::-1]
            chans = chans[sidx]
    n = len(chans)

    # Grid geometry: a leading blank offset, plus one reserved cell for the
    # moment-0 panel when requested. Channels fill the remaining cells in order.
    n_extra = 1 if moment0 else 0
    total_cells = start_panel + n + n_extra
    if nrows is None:
        nrows = int(np.ceil(total_cells / ncols))
    all_cells = list(range(start_panel, start_panel + n + n_extra))
    if moment0:
        m0_cell = _resolve_panel(moment0_panel, nrows, ncols, all_cells)
        channel_cells = [c for c in all_cells if c != m0_cell]
    else:
        m0_cell = None
        channel_cells = all_cells
    occupied = list(all_cells)

    # Shared normalization from the whole (processed) cube's bright end.
    if norm is None:
        if vmin is None or vmax is None:
            lo, hi = np.nanpercentile(data, [plo, phi])
            vmin = float(lo) if vmin is None else vmin
            vmax = float(hi) if vmax is None else vmax
        norm = make_norm(stretch=stretch, clip="manual",
                         vmin=float(vmin), vmax=float(vmax))

    proj = (view.celestial_wcs
            if (wcs_panels and view.celestial_wcs is not None) else None)

    # ImageGrid keeps every panel the SAME size with uniform padding (unlike a
    # plain subplots grid, where aspect-equal panels + a colorbar leave uneven
    # gaps and shrink some panels), shares one tick grid across all panels
    # (share_all), an integrated colorbar, and independent horizontal/vertical
    # padding via the axes_pad tuple — exactly the channel-map layout.
    from mpl_toolkits.axes_grid1 import ImageGrid
    if figsize is None:
        aspect = data.shape[1] / data.shape[2]        # ny / nx
        panel_in = 2.6
        figsize = (panel_in * ncols + (1.0 if colorbar else 0.0),
                   panel_in * aspect * nrows + 0.6)
    axes_pad = (wspace if wspace is not None else pad,
                hspace if hspace is not None else pad)
    axes_class = (WCSAxes, {"wcs": proj}) if proj is not None else None
    fig = plt.figure(figsize=figsize, facecolor=facecolor)
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=axes_pad,
                     share_all=True, aspect=True,
                     cbar_mode="single" if colorbar else None,
                     cbar_location="right", cbar_pad=cbar_pad,
                     axes_class=axes_class)
    axflat = list(grid)
    axes = np.array(axflat, dtype=object).reshape(nrows, ncols)

    label_idx = (_resolve_panel(label_panel, nrows, ncols, channel_cells)
                 if ticks == "minimal" else None)
    axlabels = _nice_axis_labels(proj)

    images: list[Any] = []
    velocities: list[Any] = []
    labels: list[Any] = []
    imshow_kwargs = dict(imshow_kwargs or {})
    cmap_min_color = _cmap_min(cmap) if panel_facecolor == "cmap_min" else None

    for k, ch in enumerate(chans):
        cell = channel_cells[k]
        ax = axflat[cell]
        if panel_facecolor is not None:
            ax.set_facecolor(cmap_min_color if cmap_min_color is not None
                             else panel_facecolor)
        im = ax.imshow(data[int(ch)], origin="lower", cmap=cmap,
                       norm=norm, **imshow_kwargs)
        images.append(im)

        vel = _channel_velocity(view, world, int(ch), label, label_unit,
                                restfreq, vsys)
        velocities.append(vel)
        text = _channel_label_text(view, world, int(ch), label, vel, label_fmt)
        if text:
            labels.append(ax.text(
                0.05, 0.95, text, transform=ax.transAxes, va="top",
                ha="left", color=label_color, fontsize=label_fontsize,
                zorder=5,
                path_effects=_stroke_path_effects(label_stroke_color,
                                                  label_stroke_lw),
                **(label_kwargs or {})))

        _style_panel_ticks(ax, proj, ticks, cell == label_idx, coords,
                           apply_offset_ticks, axlabels, tick_direction,
                           tick_labelsize)

    # Optional moment-0 (velocity-integrated) summary panel — its own norm and
    # colormap, since its units (∫I dv) differ from the per-channel brightness.
    moment0_image = None
    moment0_units = None
    if moment0 and m0_cell is not None:
        ax0 = axflat[m0_cell]
        dv = 1.0
        integrated = False
        if world is not None and view.axis_kind == "velocity" and len(world) > 1:
            dv = float(np.abs(np.mean(np.diff(world))))
            integrated = True
        # Units of ∫I dv (for a post-hoc colorbar label): BUNIT × spectral unit.
        if view.bunit and integrated and view.world_unit:
            moment0_units = f"{view.bunit} {_clean_unit(view.world_unit)}"
        else:
            moment0_units = view.bunit
        mom0 = np.nansum(data, axis=0) * dv
        m0norm = make_norm(stretch=stretch, data=mom0[np.isfinite(mom0)],
                           clip="percentile", plo=plo, phi=phi)
        moment0_image = ax0.imshow(mom0, origin="lower",
                                   cmap=moment0_cmap or cmap, norm=m0norm,
                                   **imshow_kwargs)
        if moment0_label:
            ax0.text(0.05, 0.95, moment0_label, transform=ax0.transAxes,
                     va="top", ha="left", color=label_color,
                     fontsize=label_fontsize, zorder=5,
                     path_effects=_stroke_path_effects(label_stroke_color,
                                                       label_stroke_lw))
        _style_panel_ticks(ax0, proj, ticks, m0_cell == label_idx, coords,
                           apply_offset_ticks, axlabels, tick_direction,
                           tick_labelsize)

    for ax in [axflat[i] for i in range(len(axflat)) if i not in occupied]:
        ax.set_visible(False)

    cb = None
    if colorbar and images:
        label_txt = cbar_label if cbar_label is not None else (view.bunit or "")
        # ImageGrid's integrated colorbar axes — uniform, doesn't distort panels.
        cb = fig.colorbar(images[-1], cax=grid.cbar_axes[0], label=label_txt)

    beam_artist = None
    if beam and view.header is not None:
        try:
            bp = _resolve_panel(beam_panel, nrows, ncols, occupied)
            # ``from_header(ax=)`` positions the beam at the panel's lower-left
            # in data coords; ``add_to`` draws it there. Keeping it lower-left
            # leaves the lower-right free for the scale bar on the same panel
            # (the paper convention). A bare Beam is drawn with a transparent
            # face AND edge, so supply a visible default style (overridable).
            beam_style: dict[str, Any] = {"facecolor": "0.75",
                                          "edgecolor": "0.15", "alpha": 0.9}
            beam_style.update(beam_kwargs or {})
            beam_artist = Beam.from_header(view.header, ax=axflat[bp],
                                           **beam_style)
            beam_artist.add_to(axflat[bp])
        except Exception as exc:
            warnings.warn(f"channel_map: beam not drawn ({exc}).",
                          stacklevel=2)

    scalebar_artist = None
    if scalebar is not None and view.header is not None:
        sp = _resolve_panel(scalebar_panel, nrows, ncols, occupied)
        sb_label = (scalebar_label if scalebar_label is not None
                    else f"{scalebar:g}″")
        scalebar_artist = add_sizebar_asec(axflat[sp], view.header,
                                           float(scalebar), sb_label,
                                           **(scalebar_kwargs or {}))

    if suptitle:
        fig.suptitle(suptitle)

    return ChannelMapResult(fig=fig, axes=axes, images=images, colorbar=cb,
                            norm=norm, channels=chans, velocities=velocities,
                            labels=labels, beam=beam_artist,
                            scalebar=scalebar_artist,
                            moment0_image=moment0_image,
                            moment0_units=moment0_units)


# ---------------------------------------------------------------------------
# Per-panel helpers
# ---------------------------------------------------------------------------

def _cmap_min(cmap: Any) -> Any:
    """The color at the low end of a colormap (for the seamless panel bg)."""
    try:
        import matplotlib as mpl
        cm = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        return cm(0.0)
    except Exception:
        return None


def _channel_velocity(view: DataCube, world: npt.NDArray | None, ch: int,
                      mode: Any, unit: str | None, restfreq: Any,
                      vsys: float | None) -> tuple[float, str] | None:
    """The spectral ``(value, unit)`` for channel *ch*, or ``None``."""
    if world is None or view.axis_kind is None or callable(mode) or mode is None:
        return None
    if mode == "channel":
        return None
    return _spectral_label_value(float(world[ch]), view.world_unit or "",
                                 view.axis_kind, mode, unit, restfreq, vsys)


def _channel_label_text(view: DataCube, world: npt.NDArray | None, ch: int,
                        mode: Any, vel: tuple[float, str] | None,
                        fmt: str | None) -> str:
    """Format one panel's corner label per the ``label`` mode."""
    if mode is None:
        return ""
    if callable(mode):
        return str(mode(ch, view))
    if mode == "channel":
        return f"ch {ch}"
    if mode not in ("auto", "velocity", "frequency", "wavelength", "redshift"):
        raise ValueError(
            f"label must be 'auto'/'velocity'/'frequency'/'wavelength'/"
            f"'redshift'/'channel', a callable, or None; got {mode!r}.")
    if vel is None:
        return f"ch {ch}"          # no usable spectral WCS → fall back
    value, ustr = vel
    kind = view.axis_kind if mode == "auto" else mode
    f = fmt or _DEFAULT_FMT.get(str(kind), "{:.3g}")
    return f"{f.format(value)} {ustr}".strip()


def _nice_axis_labels(wcs: Any) -> tuple[str, str]:
    """Frame-appropriate axis labels (``'R.A. (J2000)'`` …) from a WCS."""
    try:
        lon = wcs.wcs.ctype[0].upper()
    except Exception:
        return ("R.A.", "Dec.")
    if lon.startswith("RA"):
        return ("R.A. (J2000)", "Decl. (J2000)")
    if lon.startswith("GLON"):
        return ("Galactic Longitude", "Galactic Latitude")
    if lon.startswith(("ELON", "ELAT")):
        return ("Ecliptic Longitude", "Ecliptic Latitude")
    if lon.startswith("SLON"):
        return ("Supergal. Longitude", "Supergal. Latitude")
    return ("Longitude", "Latitude")


def _style_panel_ticks(ax: Any, proj: Any, ticks: str, is_label_panel: bool,
                       coords: str, apply_offset_ticks: Any,
                       axlabels: tuple[str, str], tick_direction: str,
                       tick_labelsize: float) -> None:
    """Apply tick density + coordinate style to one panel.

    Every panel gets the same tick grid (ImageGrid's ``share_all`` shares sky
    positions; ``coords='offset'`` re-runs the same offset ticks on each), with
    inward ticks (so marks stay out of the tight gaps) and a small label font;
    coordinate + axis labels show only where ``show_labels`` is set.
    """
    if proj is None:
        # Bare tiles: no coordinate machinery, just drop the mpl ticks.
        ax.set_xticks([])
        ax.set_yticks([])
        return

    if ticks == "plain":
        for c in (0, 1):
            ax.coords[c].set_ticks_visible(False)
            ax.coords[c].set_ticklabel_visible(False)
            ax.coords[c].set_axislabel("")
        return

    show_labels = ticks == "complete" or is_label_panel
    if coords == "offset":
        # ΔRA/ΔDec offsets, applied to every panel so the grid matches; labels
        # (and axis labels) only where requested.
        apply_offset_ticks(ax, unit="auto", axis_labels=show_labels,
                           fontsize=tick_labelsize)
        if not show_labels:
            for c in (0, 1):
                ax.coords[c].set_ticklabel_visible(False)
    elif show_labels:
        ax.coords[0].set_axislabel(axlabels[0])
        ax.coords[1].set_axislabel(axlabels[1])
    else:
        for c in (0, 1):
            ax.coords[c].set_ticklabel_visible(False)
            ax.coords[c].set_axislabel("")

    # Inward tick marks + small label font on every panel (after any offset-tick
    # setup, which resets these). Inward keeps the marks off the neighbors in a
    # tight grid; the small font keeps 'complete'-mode labels from colliding.
    for c in (0, 1):
        ax.coords[c].set_ticks(direction=tick_direction)
        ax.coords[c].set_ticklabel(size=tick_labelsize)
