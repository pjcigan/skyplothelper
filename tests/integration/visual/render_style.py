"""Render the style layers for visual eyeballing.

Covers the new palette layer + WCSAxes styling bridge + annotation
palettes added on top of the existing base/theme machinery:

  * cycle CYCLE_PALETTES on light and dark backgrounds (lines / scatter /
    alpha=0.35 region fills with edges — the case that trips up naive
    palettes),
  * the rc-themes (publication / twilight / dark_sky / poster),
  * ANNOTATION_PALETTES as mock finder charts (style_annotation),
  * style_wcs_axes() before/after on a dark theme (why the bridge exists:
    WCSAxes ignores tick rcParams),
  * a fully composed set_style() + style_wcs_axes() all-sky figure.

Usage
-----
    python render_style.py            # save PNGs to output/
    python render_style.py --show     # display interactively
"""

import sys

import matplotlib.pyplot as plt  # noqa: E402  (after _common backend select)
import numpy as np
from _common import banner, save_or_show
from astropy.coordinates import SkyCoord
from matplotlib import rcParams

from skyplothelper import (
    BASE_PRESETS,
    CYCLE_PALETTES,
    MONO_STACK,
    add_compass,
    add_geodesic_circle,
    add_second_grid,
    make_wcs_frame,
    set_base_style,
    style_annotation,
    style_context,
    style_wcs_axes,
)
from skyplothelper.style import _THEMES

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


def _style_axes(ax, fg):
    """Color spines / ticks / labels of a plain Axes for a given fg color."""
    for s in ax.spines.values():
        s.set_color(fg)
    ax.tick_params(colors=fg, labelsize=7)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)


def _cycle_sheet(names, bg, fg, title):
    """Demo sheet: one row per palette, columns lines / scatter / fills."""
    n = len(names)
    fig, axes = plt.subplots(n, 3, figsize=(11, 1.5 * n + 0.5),
                             facecolor=bg, squeeze=False)
    fig.suptitle(title, color=fg, fontsize=13)
    x = np.linspace(0, 2 * np.pi, 120)
    for r, name in enumerate(names):
        colors = CYCLE_PALETTES[name]['colors']
        rng = np.random.default_rng(r)
        ax_l, ax_s, ax_f = axes[r]
        for i, c in enumerate(colors):
            ax_l.plot(x, np.sin(x + i * 0.6) + i * 0.25, color=c, lw=2)
            ax_s.scatter(rng.normal(i, 0.28, 40), rng.normal(0, 1, 40),
                         color=c, s=14, edgecolor='none')
            ax_f.fill_between([i, i + 0.92], 0, 1, color=c, alpha=0.35,
                              edgecolor=c, linewidth=1.6)
        ax_l.set_ylabel(name, color=fg, fontsize=10)
        for a in axes[r]:
            a.set_facecolor(bg)
            _style_axes(a, fg)
            a.set_xticks([])
            a.set_yticks([])
        if r == 0:
            for a, t in zip(axes[r], ('lines', 'scatter', 'alpha fills')):
                a.set_title(t, color=fg, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


@_panel("style_01_cycle_palettes_light")
def cycle_light():
    # Light sheet shows every palette (dual + light-only letterpress).
    names = [n for n, s in CYCLE_PALETTES.items() if s['mode'] in ('dual', 'light')]
    return _cycle_sheet(names, bg='#FAF6EB', fg='#33302A',
                        title='Cycle palettes — light background')


@_panel("style_02_cycle_palettes_dark")
def cycle_dark():
    # Dark sheet drops letterpress (light-only); keeps dual + dark.
    names = [n for n, s in CYCLE_PALETTES.items() if s['mode'] in ('dual', 'dark')]
    return _cycle_sheet(names, bg='#101319', fg='#C8CCD2',
                        title='Cycle palettes — dark (night-sky) background')


@_panel("style_03_themes")
def themes():
    """The four rc-themes (note: 'dark' renamed to 'twilight')."""
    names = ['publication', 'twilight', 'dark_sky', 'poster']
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), squeeze=False)
    x = np.linspace(0, 10, 100)
    for ax, name in zip(axes[0], names):
        th = _THEMES[name]
        ax.set_facecolor(th['axes.facecolor'])
        for i in range(4):
            ax.plot(x, np.sin(x + i), lw=2)
        ax.grid(True, color=th.get('grid.color', '0.8'))
        fg = th.get('xtick.color', 'black')
        _style_axes(ax, fg)
        ax.set_title(name, color=th.get('text.color', 'black'), fontsize=11)
    fig.tight_layout()
    return fig


def _finder(name):
    """A mock finder chart on a TAN frame using an annotation palette."""
    fig = plt.figure(figsize=(6.2, 6.2))
    center = SkyCoord(83.8221, -5.3911, unit='deg')   # ~Orion field
    ax = make_wcs_frame(111, projection='TAN', center_lon=center.ra.deg,
                        center_lat=center.dec.deg, fov_deg=3.0, fig=fig)
    pal = style_annotation(ax, name)
    # Background overlay (galactic) grid in the secondary grid color.
    add_second_grid(ax, overlay_frame='galactic', color=pal['grid2'],
                    alpha=0.6)
    # Stars, sized by a fake magnitude.
    rng = np.random.default_rng(7)
    ra = center.ra.deg + rng.normal(0, 0.8, 60)
    dec = center.dec.deg + rng.normal(0, 0.8, 60)
    sizes = rng.uniform(4, 60, 60)
    ax.scatter(ra, dec, s=sizes, color=pal['stars'],
               transform=ax.get_transform('world'), zorder=5)
    # Target reticle (accent) + label (label color) + compass.
    add_geodesic_circle(ax, center.ra.deg, center.dec.deg, radius_deg=0.45,
                        edgecolor=pal['accent'], facecolor='none', lw=1.8,
                        zorder=6)
    ax.text(center.ra.deg + 0.55, center.dec.deg + 0.55, 'target',
            transform=ax.get_transform('world'), color=pal['label'],
            fontsize=11, zorder=7)
    add_compass(ax, color=pal['compass'], stroke_color=pal['ax_bg'])
    ax.set_title(f"annotation palette: {name}", color=pal['text'])
    return fig


@_panel("style_04_annot_parchment")
def annot_parchment():
    return _finder('parchment')


@_panel("style_04_annot_publication")
def annot_publication():
    return _finder('publication')


@_panel("style_04_annot_dark")
def annot_dark():
    return _finder('dark')


@_panel("style_04_annot_night")
def annot_night():
    return _finder('night')


@_panel("style_04_annot_denim")
def annot_denim():
    return _finder('denim')


@_panel("style_05_wcs_bridge_before_after")
def wcs_bridge():
    """style_wcs_axes reaches tick / frame / grid properties on a WCSAxes.

    WCSAxes controls these through ``ax.coords`` rather than the tick
    rcParams, so the bridge is how you recolor/resize them. Left: a
    default frame. Right: the same frame after explicit overrides — bold
    so the effect on ticks (color/length), the frame (color/width), and
    the grid is unmistakable.
    """
    fig = plt.figure(figsize=(13, 4.4))
    ax0 = make_wcs_frame(121, projection='AIT', fig=fig, grid=True)
    ax0.set_title('default frame')
    ax1 = make_wcs_frame(122, projection='AIT', fig=fig, grid=True)
    style_wcs_axes(ax1, tick_color='#A4452D', major_size=10, width=1.6,
                   minor_frequency=4, labelcolor='#2E5266',
                   frame_color='#2E5266', frame_linewidth=2.2,
                   grid=True, grid_color='#9D4B36', grid_alpha=0.55)
    ax1.set_title('after style_wcs_axes(tick / frame / grid overrides)')
    return fig


@_panel("style_06_composed_allsky")
def composed():
    """One set_style() + style_wcs_axes() — the 'applies everywhere' shot."""
    with style_context(base='standard', theme='dark_sky', palette='nightcap'):
        fig = plt.figure(figsize=(11, 5.6))
        ax = make_wcs_frame(111, projection='AIT', center=180, fig=fig,
                            grid=True)
        style_wcs_axes(ax, grid=True)
        rng = np.random.default_rng(3)
        # A few source groups, each in the next cycle color (scatter does
        # not advance the prop-cycle on its own, so pick explicitly).
        colors = CYCLE_PALETTES['nightcap']['colors']
        for i in range(5):
            ra = rng.uniform(0, 360, 50)
            dec = np.degrees(np.arcsin(rng.uniform(-1, 1, 50)))
            ax.scatter(ra, dec, s=16, color=colors[i],
                       transform=ax.get_transform('world'), zorder=5)
        ax.set_title('set_style(theme="dark_sky", palette="nightcap") '
                     '+ style_wcs_axes', color='#c9d1d9')
    return fig


@_panel("style_07_base_presets")
def base_presets():
    """The eight set_base_style presets on the same deterministic line plot.

    Each subplot is built while its preset's rcParams are active, so the
    line-weight / spine / grid / title differences bake into the artists.
    rcParams are snapshotted and restored, so the panel is leak-free.
    """
    order = list(BASE_PRESETS)  # standard, structural, journal, ... minimalist
    x = np.linspace(0, 10, 100)
    saved = rcParams.copy()
    try:
        fig = plt.figure(figsize=(15, 7.5))
        for i, name in enumerate(order, 1):
            plt.rcdefaults()
            set_base_style(name)
            ax = fig.add_subplot(2, 4, i)
            for k in range(4):
                ax.plot(x, np.sin(x + k * 0.6) + k * 0.3)
            ax.set_title(name)
            ax.set_xlabel('x')
            ax.set_ylabel(r'$\sin(x)$')
            ax.grid(True)
        fig.tight_layout()
    finally:
        rcParams.update(saved)
    return fig


@_panel("style_08_structural_vs_default")
def structural_vs_default():
    """'structural' is color/font-agnostic: same colors & font as the mpl
    default, only the structure (inward minor ticks, thinner spines, a
    hairline grid below the data) is nudged."""
    x = np.linspace(0, 10, 100)
    cases = [(None, 'matplotlib default'),
             ('structural', "set_base_style('structural')")]
    saved = rcParams.copy()
    try:
        fig = plt.figure(figsize=(11, 4.2))
        for i, (name, label) in enumerate(cases, 1):
            plt.rcdefaults()
            if name:
                set_base_style(name)
            ax = fig.add_subplot(1, 2, i)
            for k in range(4):
                ax.plot(x, np.sin(x + k * 0.7) + k * 0.3)
            ax.set_title(label)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True)
        fig.tight_layout()
    finally:
        rcParams.update(saved)
    return fig


@_panel("style_09_monospace_readout")
def monospace_readout():
    """MONO_STACK applied per-artist for a fixed-width coordinate readout."""
    t = np.linspace(0, 10, 60)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, np.cos(t))
    ax.set_title('coordinate readout in sph.MONO_STACK')
    rows = ['RA    05h34m31.9s', 'Dec  +22d00m52s',
            'sep       3.214 arcmin', 'PA       57.0 deg']
    ax.text(0.04, 0.96, '\n'.join(rows), transform=ax.transAxes,
            family=MONO_STACK, fontsize=10, va='top',
            bbox=dict(boxstyle='round', fc='#FAF6EB', ec='#999999'))
    return fig


def main():
    banner("style layers gallery")
    only = None
    for a in sys.argv[1:]:
        if not a.startswith('-'):
            only = a
    for name, fn in PANELS.items():
        if only and only not in name:
            continue
        print(f">>> {name}")
        fig = fn()
        save_or_show(fig, name)


if __name__ == "__main__":
    main()
