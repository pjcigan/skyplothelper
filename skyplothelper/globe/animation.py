"""Globe animation utilities.

The globe animators (``animate_globe`` / ``animate_blended_globe``) are demo /
examples helpers and are NOT exported from the package ``__init__`` (per-frame
WCS rebuild is the slow path; a faster keep-axes/update-image variant is not yet
implemented). The generic save helpers here — :class:`WebPWriter` and
:func:`save_animation` — ARE public API (re-exported as ``sph.WebPWriter`` /
``sph.save_animation``): they save *any* matplotlib animation the skyplothelper
way (writer selection, transparent background, animated WebP), so callers with
their own ``FuncAnimation`` don't reinvent it.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from typing import Any

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.animation import (  # noqa: F401
    Animation,
    FuncAnimation,
    PillowWriter,
)

from .._stroke import _stroke_path_effects
from ..images.reprojection import reproject_rgb_map

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from .frame import make_globe_frame
from .nightshade import make_nightshade_blend


class WebPWriter(PillowWriter):
    """Finish a matplotlib animation as animated WebP (8-bit alpha, seamless loop).

    A drop-in :class:`~matplotlib.animation.PillowWriter` subclass. It reuses
    PillowWriter's RGBA frame grab (which respects the ``savefig`` facecolor,
    including ``transparent``), so anti-aliased edges — globe limbs especially —
    survive as true 8-bit alpha instead of GIF's 1-bit staircase, and the loop
    is seamless.

    Pick the compression mode by content:

    * ``lossless=True`` for sparse line / point art (orbits, proper motions,
      great circles) — crisp, and smaller than GIF there; lossy VP8 both softens
      the lines and comes out *bigger* on few-color transparent frames.
    * ``lossless=False`` (default) with ``quality`` for photographic frames
      (globes, planets, colormap movies) — typically 30-48% of the GIF.

    ``method=6`` is Pillow's slowest / best compression.

    Examples
    --------
    ::

        import skyplothelper as sph
        ani.save('spin.webp', writer=sph.WebPWriter(fps=10, lossless=False))
    """

    def __init__(self, *args: Any, lossless: bool = False, quality: int = 66,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lossless = lossless
        self._quality = quality

    def finish(self) -> None:
        # PillowWriter accumulates RGBA frames in its private ._frames list
        # (not in matplotlib's type stubs); write them all to one animated
        # WebP. duration is per-frame ms (from fps).
        frames = self._frames  # type: ignore[attr-defined]
        frames[0].save(
            self.outfile, format='WEBP', save_all=True,
            append_images=frames[1:],
            duration=int(round(1000.0 / self.fps)),
            loop=0, lossless=self._lossless, quality=self._quality, method=6)


def _tqdm_disable(progress: bool | None) -> bool | None:
    """Translate a ``progress`` flag into tqdm's ``disable=``.

    ``None`` (default) → ``disable=None``, which auto-hides the bar on a
    non-TTY (nbconvert, CI, redirected output) and shows it in an interactive
    terminal — robust regardless of import order, unlike ``TQDM_DISABLE`` (read
    once at tqdm import). ``True`` forces the bar on (e.g. a non-TTY Jupyter
    session), ``False`` forces it off.
    """
    return None if progress is None else (not progress)


def _fps_from_animation(ani: Any, fps: float | None) -> float:
    """Frame rate for the writer: explicit *fps* if given, else derived from the
    animation's frame interval (ms), else a 10 fps fallback."""
    if fps is not None:
        return float(fps)
    interval = getattr(ani, '_interval', None)
    if interval:
        return max(1.0, round(1000.0 / float(interval)))
    return 10.0


def _cleanup_animation(ani: Any) -> None:
    """Stop the timer and clear the figure after a save/show (avoids an
    ``add_callback`` AttributeError during teardown)."""
    try:
        ani.event_source.stop()
    except Exception:
        pass
    del ani
    plt.clf()
    plt.close()


def save_animation(
    ani: Any, fig: Any, savepath: str, *,
    bgcolor: str = 'transparent', lossless: bool = False, quality: int = 66,
    fps: float | None = None, dpi: int = 200,
    force_writer: str | bool = False,
    extra_writer_args: list[Any] | None = None,
) -> None:
    """Save (or show) a matplotlib animation the skyplothelper way.

    One place that owns writer selection, transparent-background handling, and
    animated-WebP output, so the globe animators and any user ``FuncAnimation``
    save consistently instead of each re-implementing it.

    The writer is chosen from the file extension: ``.webp`` →
    :class:`WebPWriter`, ``.gif`` / ``.apng`` → Pillow, anything else → ffmpeg,
    with an imagemagick fallback if the primary writer errors. Only
    ``.gif`` / ``.apng`` / ``.webp`` carry transparency (``.webp`` at 8-bit
    alpha, ``.gif`` at 1-bit); mp4 and the other video containers do not.

    Parameters
    ----------
    ani : matplotlib.animation.Animation
        The animation to write.
    fig : matplotlib.figure.Figure
        Its figure (needed to set a transparent patch).
    savepath : str
        Output file, or ``'show'`` to display interactively.
    bgcolor : str
        ``'transparent'`` / ``'none'`` writes transparent frames; any other
        color fills the background with it. Default ``'transparent'``.
    lossless : bool
        WebP mode — ``True`` for sparse line/point art, ``False`` (default,
        with *quality*) for photographic. Ignored for non-WebP outputs.
    quality : int
        WebP lossy quality (0-100). Default ``66``.
    fps : float, optional
        Frames per second for the WebP writer. Default ``None`` derives it from
        the animation's frame interval (falling back to 10).
    dpi : int
        Output resolution. Default ``200``.
    force_writer : str or False
        Bypass extension-based selection and use this writer verbatim.
    extra_writer_args : list, optional
        Extra args for the ffmpeg / pillow writer (e.g. ffmpeg codec flags).
        Not used for the WebP writer, which is configured by *lossless* /
        *quality* / *fps*.
    """
    if isinstance(bgcolor, str) and bgcolor.lower() in ('transparent', 'none'):
        fig.patch.set_alpha(0.)
        savefig_kwargs = {'transparent': True, 'facecolor': 'none'}
    else:
        savefig_kwargs = {'facecolor': bgcolor}

    if savepath == 'show':
        plt.show()
        _cleanup_animation(ani)
        return

    # Exact extension (splitext), not a substring test — a path like
    # 'v1.gif.d/out.mp4' must not misfire to the gif writer.
    ext = os.path.splitext(savepath)[1].lower()
    try:
        if force_writer:
            ani.save(savepath, writer=force_writer,
                     savefig_kwargs=savefig_kwargs,
                     extra_args=extra_writer_args, dpi=dpi)
        elif ext == '.webp':
            writer = WebPWriter(fps=_fps_from_animation(ani, fps),
                                lossless=lossless, quality=quality)
            ani.save(savepath, writer=writer, savefig_kwargs=savefig_kwargs,
                     dpi=dpi)
        elif ext in ('.gif', '.apng'):
            ani.save(savepath, writer='pillow', savefig_kwargs=savefig_kwargs,
                     extra_args=extra_writer_args, dpi=dpi)
        else:
            ani.save(savepath, writer='ffmpeg', savefig_kwargs=savefig_kwargs,
                     extra_args=extra_writer_args, dpi=dpi)
    except Exception:
        try:
            ani.save(savepath, writer='imagemagick',
                     savefig_kwargs=savefig_kwargs, dpi=dpi)
        except Exception as e:
            warnings.warn(f'animation save failed: {e}',
                          RuntimeWarning, stacklevel=2)

    _cleanup_animation(ani)


def animate_globe(center_lons: Sequence[float], center_lats: Sequence[float],
                  lonpoles: Sequence[float], bgim: Any = False,
                  show_ticklabels: bool = True, interval_ms: int = 100,
                  repeat_delay_ms: int = 2000, savepath: str = 'show',
                  bgcolor: str = 'w',
                  fig_bgim: npt.ArrayLike | None = None,
                  force_writer: str | bool = False,
                  frame_kwargs: dict[str, Any] | None = None,
                  extra_writer_args: list[Any] | None = None,
                  webp_lossless: bool = False, webp_quality: int = 66,
                  progress: bool | None = None,
                  dpi: int = 200,
                  figsize: tuple[float, float] = (6, 6)) -> None:
    """
    Animate globe rotations with optional background image projection.

    Parameters
    ----------
    center_lons, center_lats, lonpoles : array-like
        Arrays of frame parameters, typically from make_globe_angles().
    bgim : ImageHDU or False, optional
        Pseudo-FITS HDU of image to project onto the globe. False to skip.
    show_ticklabels : bool, optional
        Whether to display coordinate tick labels. Default True.
    interval_ms : int, optional
        Milliseconds between animation frames. Default 100.
    repeat_delay_ms : int, optional
        Delay before animation repeats, in ms. Default 2000.
    savepath : str, optional
        File path to save animation, or 'show' to display interactively.
    bgcolor : str, optional
        Background color. 'transparent' for transparent background.
    fig_bgim : ndarray or None, optional
        Optional background image for the figure (e.g. starfield).
    force_writer : str or False, optional
        Force a specific animation writer (e.g. 'ffmpeg', 'pillow').
    frame_kwargs : dict or None, optional
        Extra keyword arguments passed to make_globe_frame().
    extra_writer_args : list or None, optional
        Extra arguments for the animation writer (e.g. ffmpeg flags).
    webp_lossless : bool, optional
        For a ``.webp`` savepath, use lossless compression. Default False
        (lossy, good for the photographic globe frames). Ignored otherwise.
    webp_quality : int, optional
        Lossy WebP quality (0-100) when ``webp_lossless=False``. Default 66.
    progress : bool or None, optional
        Show the tqdm frame-progress bar. Default ``None`` auto-hides it on a
        non-TTY (nbconvert, CI, redirected output) and shows it in an
        interactive terminal; ``True`` / ``False`` force it on / off.
    dpi : int, optional
        Output DPI. Default 200.
    figsize : tuple, optional
        Figure size in inches. Default (6, 6).

    Notes
    -----
    mp4 does not support transparency. Use ``.gif``, ``.apng``, or ``.webp``
    for transparent backgrounds — ``.webp`` carries 8-bit alpha (smooth limbs),
    ``.gif`` only 1-bit. The other video containers (.mov/.flv/.avi) do not
    carry alpha with their standard codecs.

    Recommended ffmpeg args for good compression::

        extra_writer_args=['-vcodec', 'libx265', '-crf', '28',
                           '-preset', 'veryslow', '-vf', 'scale=1440:-1']
    """
    if frame_kwargs is None:
        frame_kwargs = {}

    fig = plt.figure(1, figsize=figsize)
    ax, hdr = make_globe_frame(
        111, center_LONdeg=center_lons[0],
        center_LATdeg=center_lats[0], lonpole=lonpoles[0],
        radesys='ITRS', projection='SIN', return_header=True,
        **frame_kwargs)

    if bgcolor == '' or (isinstance(bgcolor, str)
                         and bgcolor.lower() == 'none'):
        pass
    elif isinstance(bgcolor, str) and bgcolor.lower() == 'transparent':
        ax.set_facecolor('none')

    img_artist = None
    if bgim is not False:
        img_artist = ax.imshow(reproject_rgb_map(bgim, hdr))

    if fig_bgim is not None:
        bg_ax = plt.axes((0, 0, 1, 1))
        bg_ax.set_zorder(-1)
        bg_ax.imshow(fig_bgim, aspect='auto')
        ax.coords.frame.set_linewidth(2)
        ax.coords.frame.patch.set_alpha(0)

    if not show_ticklabels:
        for i in (0, 1):
            ax.coords[i].set_ticklabel_visible(False)
        # On a globe the PRIMARY in-frame labels are sph overlay Text artists
        # (tagged ``_sph_overlay_ticklabel``), which ``set_ticklabel_visible``
        # can't reach — hide those too so show_ticklabels=False fully clears the
        # labels (previously needed frame_kwargs={'tick_style': 'native'}).
        for txt in list(ax.texts):
            if getattr(txt, '_sph_overlay_ticklabel', False):
                txt.set_visible(False)

    def update(frame: int) -> None:
        ax.wcs.wcs.lonpole = lonpoles[frame]
        ax.wcs.wcs.crval = [center_lons[frame], center_lats[frame]]
        if img_artist is not None:
            tmp_hdr = ax.wcs.to_header()
            for card in ('NAXIS', 'NAXIS1', 'NAXIS2'):
                tmp_hdr.set(card, hdr[card])
            img_artist.set_array(reproject_rgb_map(bgim, tmp_hdr))

    frame_iter = range(1, len(center_lons))
    if _HAS_TQDM:
        frame_iter = tqdm(frame_iter, desc='Animating globe frames',
                          disable=_tqdm_disable(progress), leave=False)

    # mpl stubs over-constrain func/frames (Artist-returning callable +
    # Artist frames); the blit=False update→None + range frames are valid.
    # (mpl 3.11 relaxed this, so pair unused-ignore to stay green on both.)
    ani = FuncAnimation(fig=fig, func=update, frames=frame_iter,  # type: ignore[arg-type, unused-ignore]
                        interval=interval_ms, repeat_delay=repeat_delay_ms)

    save_animation(ani, fig, savepath, bgcolor=bgcolor,
                   lossless=webp_lossless, quality=webp_quality,
                   force_writer=force_writer,
                   extra_writer_args=extra_writer_args, dpi=dpi)


def animate_blended_globe(center_lons: Sequence[float],
                          center_lats: Sequence[float],
                          lonpoles: Sequence[float],
                          timestamps: Sequence[Any],
                          day_hdu: Any, night_hdu: Any,
                          show_ticklabels: bool = True,
                          interval_ms: int = 100, repeat_delay_ms: int = 2000,
                          savepath: str = 'show', bgcolor: str = 'w',
                          lon_label_props: list[Any] | None = None,
                          lat_label_props: list[Any] | None = None,
                          grid_props: list[Any] | None = None,
                          fig_bgim: npt.ArrayLike | None = None,
                          force_writer: str | bool = False,
                          frame_kwargs: dict[str, Any] | None = None,
                          extra_writer_args: list[Any] | None = None,
                          webp_lossless: bool = False, webp_quality: int = 66,
                          progress: bool | None = None,
                          blend: str = 'elevation',
                          nightshade_kwargs: dict[str, Any] | None = None,
                          figsize: tuple[float, float] = (6, 6),
                          dpi: int = 200) -> None:
    """
    Animate globe rotations with blended day/night terminator images.

    For each frame, the nightshade terminator is recomputed at the given
    timestamp and a blended RGBA night image is overlaid on the day image.

    Parameters
    ----------
    center_lons, center_lats, lonpoles : array-like
        Frame parameters from make_globe_angles().
    timestamps : array-like of datetime
        Timestamps for each frame (for nightshade computation).
    day_hdu : ImageHDU
        Pseudo-FITS HDU of the daytime map image.
    night_hdu : ImageHDU
        Pseudo-FITS HDU of the nighttime map image.
    show_ticklabels : bool, optional
        Display coordinate tick labels. Default True.
    lon_label_props : list or None, optional
        [text_color, stroke_color, fontsize, stroke_lw] for longitude labels.
        Default ['#D7D2B4', '0.2', 8, 0.7].
    lat_label_props : list or None, optional
        Same format for latitude labels. Default ['0.2', 'w', 8, 0.7].
    grid_props : list or None, optional
        [color, linewidth, alpha, linestyle] for grid.
        Default ['0.5', 0.3, 0.5, '-'].
    blend : str, optional
        Nightshade blend back-end, ``'elevation'`` (default; physical) or
        ``'gaussian'``. See :func:`~skyplothelper.globe.make_nightshade_blend`.
    nightshade_kwargs : dict or None, optional
        Extra keyword arguments forwarded to ``make_nightshade_blend`` (e.g.
        ``{'curve': 'twilight'}`` for elevation, or ``{'blend_sigma': 100}``
        for gaussian).

    Other parameters are the same as animate_globe().
    """
    nsh_kwargs = nightshade_kwargs or {}
    if frame_kwargs is None:
        frame_kwargs = {}
    if lon_label_props is None:
        lon_label_props = ['#D7D2B4', '0.2', 8, 0.7]
    if lat_label_props is None:
        lat_label_props = ['0.2', 'w', 8, 0.7]
    if grid_props is None:
        grid_props = ['0.5', 0.3, 0.5, '-']

    fig = plt.figure(1, figsize=figsize)
    ax, hdr = make_globe_frame(
        111, center_LONdeg=center_lons[0],
        center_LATdeg=center_lats[0], lonpole=lonpoles[0],
        radesys='ITRS', projection='SIN', return_header=True,
        **frame_kwargs)

    if isinstance(bgcolor, str) and bgcolor.lower() == 'transparent':
        ax.set_facecolor('none')

    # Prepare night image
    bgimg_night = night_hdu.data
    if np.nanmax(bgimg_night) > 2:
        bgimg_night = bgimg_night / 255.

    night_rgba = make_nightshade_blend(bgimg_night, timestamps[0],
                                        blend=blend, **nsh_kwargs)
    tmp_night_hdu = pyfits.ImageHDU(night_rgba, night_hdu.header)

    # Initial render
    img_day = ax.imshow(reproject_rgb_map(day_hdu, hdr), zorder=1)
    img_night = ax.imshow(
        np.nan_to_num(reproject_rgb_map(tmp_night_hdu, hdr)), zorder=1)

    # Format labels
    effects_lat = _stroke_path_effects(lat_label_props[1],
                                       lat_label_props[3]) or []
    # Style each coordinate's tick labels independently. (Was a single
    # format_ticklabels(ax, which='x'/'y', ...) call, but which= was removed
    # from format_ticklabels — it now forwards to TickLabels.set() and raises.)
    for _i, _props in ((0, lon_label_props), (1, lat_label_props)):
        ax.coords[_i].set_ticklabel(
            color=_props[0], size=_props[2], weight='bold',
            path_effects=_stroke_path_effects(_props[1], _props[3]) or [])

    # Format grid
    ax.coords.grid(color=grid_props[0], lw=grid_props[1],
                   alpha=grid_props[2], ls=grid_props[3])
    ax.coords.frame.set_color(lat_label_props[0])
    ax.coords.frame.patch.set_path_effects(effects_lat)

    if fig_bgim is not None:
        bg_ax = plt.axes((0, 0, 1, 1))
        bg_ax.set_zorder(-1)
        bg_ax.imshow(fig_bgim, aspect='auto')
        ax.coords.frame.set_linewidth(2)
        ax.coords.frame.patch.set_alpha(0)

    if not show_ticklabels:
        for i in (0, 1):
            ax.coords[i].set_ticklabel_visible(False)
        # On a globe the PRIMARY in-frame labels are sph overlay Text artists
        # (tagged ``_sph_overlay_ticklabel``), which ``set_ticklabel_visible``
        # can't reach — hide those too so show_ticklabels=False fully clears the
        # labels (previously needed frame_kwargs={'tick_style': 'native'}).
        for txt in list(ax.texts):
            if getattr(txt, '_sph_overlay_ticklabel', False):
                txt.set_visible(False)

    def update(frame: int) -> None:
        ax.wcs.wcs.lonpole = lonpoles[frame]
        ax.wcs.wcs.crval = [center_lons[frame], center_lats[frame]]

        tmp_hdr = ax.wcs.to_header()
        for card in ('NAXIS', 'NAXIS1', 'NAXIS2'):
            tmp_hdr.set(card, hdr[card])

        img_day.set_array(reproject_rgb_map(day_hdu, tmp_hdr))

        night_rgba_i = make_nightshade_blend(
            bgimg_night, timestamps[frame], blend=blend, **nsh_kwargs)
        night_hdu_i = pyfits.ImageHDU(night_rgba_i, night_hdu.header)
        img_night.set_array(
            np.nan_to_num(reproject_rgb_map(night_hdu_i, tmp_hdr)))

    frame_iter = range(1, len(center_lons))
    if _HAS_TQDM:
        frame_iter = tqdm(frame_iter, desc='Animating blended globe frames',
                          disable=_tqdm_disable(progress), leave=False)

    # mpl stubs over-constrain func/frames (see animate_globe).
    ani = FuncAnimation(fig=fig, func=update, frames=frame_iter,  # type: ignore[arg-type, unused-ignore]
                        interval=interval_ms, repeat_delay=repeat_delay_ms)

    save_animation(ani, fig, savepath, bgcolor=bgcolor,
                   lossless=webp_lossless, quality=webp_quality,
                   force_writer=force_writer,
                   extra_writer_args=extra_writer_args, dpi=dpi)




# =============================================================================
# Image Icon Scatter (imscatter)
# =============================================================================

