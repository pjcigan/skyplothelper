"""Render the globe decorations for visual eyeballing.

Covers: ``plot_ortho_grid``, ``add_checkered_border``,
``add_compass_rose``, ``add_scale_bar_cylindrical``,
``add_scale_bar_curved_parallel``.

Usage
-----
    python render_decorations_globe.py            # save PNGs to output/
    python render_decorations_globe.py --show     # display interactively
"""

import sys

import matplotlib.pyplot as plt
from _common import banner, save_or_show

from skyplothelper.coord_overlay import add_overlay_ticks
from skyplothelper.globe.decorations import (
    add_checkered_border,
    add_compass_rose,
    add_pole_rod,
    add_scale_bar_curved_parallel,
    add_scale_bar_cylindrical,
    plot_ortho_grid,
)
from skyplothelper.globe.frame import make_globe_frame, make_planet_frame
from skyplothelper.overlays.reticle import add_reticle
from skyplothelper.ticks import add_curved_lon_ticks
from skyplothelper.wcs_frame import make_wcs_frame

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


def _make_globe(center_lon=0.0, center_lat=23.44, fig=None):
    """Build a representative Earth-tilt globe."""
    if fig is None:
        fig = plt.figure(figsize=(6, 6))
    ax = make_globe_frame(
        111, center_LONdeg=center_lon, center_LATdeg=center_lat,
    )
    fig.canvas.draw()
    return fig, ax


def _make_car():
    """Build a Plate Carrée global view (10:5 aspect, 360°×180°)."""
    fig = plt.figure(figsize=(10, 5))
    ax = make_wcs_frame(111, projection="CAR", center=0, fig=fig)
    fig.canvas.draw()
    return fig, ax


@_panel("decorations_01_ortho_grid")
def render_ortho_grid():
    """plot_ortho_grid — front/back hemisphere styling on a plain mpl axes."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f0")
    ax.set_xticks([])
    ax.set_yticks([])

    # front_color / prime_meridian_color are off by default now (neutral
    # graticule); pass them explicitly to showcase the highlighted styling.
    plot_ortho_grid(ax, lon_0=0, lat_0=23.44, R=1.0,
                    lon_spacing=15, lat_spacing=15,
                    front_color="steelblue", prime_meridian_color="#33AA33")
    ax.set_title("plot_ortho_grid — front (solid) / back (dashed) hemisphere")
    return fig


@_panel("decorations_02_ortho_grid_colormap")
def render_ortho_grid_colormapped():
    """plot_ortho_grid — meridians colored by longitude (lon_cmap)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_facecolor("#fafafa")
    ax.set_xticks([])
    ax.set_yticks([])

    plot_ortho_grid(ax, lon_0=0, lat_0=23.44, R=1.0,
                    lon_spacing=10, lat_spacing=15,
                    lon_cmap="hsv", lon_cmap_lw=1.0)
    ax.set_title("plot_ortho_grid — meridians color-mapped by longitude")
    return fig


@_panel("decorations_03_checkered_border_default")
def render_checkered_border_default():
    """add_checkered_border — classic checkered globe edge (defaults)."""
    fig, ax = _make_globe()
    plot_ortho_grid(ax, lon_0=0, lat_0=23.44, R=1.0,
                    lon_spacing=30, lat_spacing=30,
                    front_color="#666", front_lw=0.5,
                    back_color="0.85", back_lw=0.3)
    add_checkered_border(ax)
    ax.set_title("add_checkered_border")
    return fig


@_panel("decorations_04_checkered_border_custom")
def render_checkered_border_custom():
    """add_checkered_border — custom segments + color scheme."""
    fig, ax = _make_globe()
    plot_ortho_grid(ax, lon_0=0, lat_0=23.44, R=1.0,
                    lon_spacing=30, lat_spacing=30,
                    front_color="#666", front_lw=0.5,
                    back_color="0.85", back_lw=0.3)
    add_checkered_border(ax, n_segments=24,
                         colors=("#1f3b73", "#f5d27b"))
    ax.set_title("add_checkered_border (n_segments=24, navy/cream)")
    return fig


@_panel("decorations_05_compass_rose_simple")
def render_compass_rose_simple():
    """add_compass_rose — default 'simple' style."""
    fig, ax = _make_globe()
    plot_ortho_grid(ax, lon_0=0, lat_0=23.44,
                    front_color="0.6", front_lw=0.5,
                    back_color="0.85", back_lw=0.3)
    add_compass_rose(ax, x=0.92, y=0.10, size=44)
    ax.set_title("add_compass_rose (style='simple', size=44 pt)")
    return fig


@_panel("decorations_06_compass_rose_arrow")
def render_compass_rose_arrow():
    """add_compass_rose — 'arrow' style (north-only)."""
    fig, ax = _make_globe()
    plot_ortho_grid(ax, lon_0=0, lat_0=23.44,
                    front_color="0.6", front_lw=0.5,
                    back_color="0.85", back_lw=0.3)
    add_compass_rose(ax, x=0.92, y=0.10, size=50, style="arrow")
    ax.set_title("add_compass_rose (style='arrow', north-only)")
    return fig


@_panel("decorations_07_scale_bar_platecarree_equator")
def render_scale_bar_platecarree_equator():
    """add_scale_bar_cylindrical — 5000 km at the equator."""
    fig, ax = _make_car()
    add_scale_bar_cylindrical(ax, lat=0.0, length_km=5000.0,
                              position="lower-right")
    ax.set_title("add_scale_bar_cylindrical (5000 km at equator)")
    return fig


@_panel("decorations_08_scale_bar_platecarree_60deg")
def render_scale_bar_platecarree_60deg():
    """add_scale_bar_cylindrical — 2000 km at lat=60° (cos-shrinks)."""
    fig, ax = _make_car()
    add_scale_bar_cylindrical(ax, lat=60.0, length_km=2000.0,
                              position="upper-right")
    ax.set_title(
        "add_scale_bar_cylindrical (2000 km at lat=60°)\n"
        "— note: bar shrinks because longitude span/km grows with cos(lat)"
    )
    return fig


@_panel("decorations_09_scale_bar_ortho")
def render_scale_bar_ortho():
    """add_scale_bar_curved_parallel — drawn on a ``make_globe_frame`` WCSAxes.

    ``plot_ortho_grid`` and
    ``add_scale_bar_curved_parallel`` route through ``ax.get_transform('world')``
    when the axes is a WCSAxes.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = make_wcs_frame(111, projection='SIN', center=(0, 23.44), fig=fig)
    plot_ortho_grid(ax, lon_0=0, lat_0=23.44,
                    lon_spacing=15, lat_spacing=15)
    add_scale_bar_curved_parallel(ax, lon_0=0, lat_0=23.44,
                        length_km=2000.0, position="lower-center")
    ax.set_title("add_scale_bar_curved_parallel (2000 km, lower-center)")
    return fig


@_panel("decorations_10_all_combined")
def render_combo():
    """All four decorations on a ``make_globe_frame`` WCSAxes."""
    fig = plt.figure(figsize=(6, 6))
    ax = make_wcs_frame(111, projection='SIN', center=(0, 23.44), fig=fig)
    plot_ortho_grid(ax, lon_0=0, lat_0=23.44,
                    front_color="steelblue", front_lw=0.6,
                    back_color="0.85", back_lw=0.3,
                    lon_spacing=15, lat_spacing=15)
    add_checkered_border(ax)
    add_compass_rose(ax, x=0.92, y=0.10, size=36)
    add_scale_bar_curved_parallel(ax, lon_0=0, lat_0=23.44,
                        length_km=2000.0, position="lower-center")
    ax.set_title("All four: grid + checker border + compass + scale bar")
    return fig


@_panel("decorations_11_curved_lon_ticks")
def render_curved_lon_ticks():
    """add_curved_lon_ticks — meridian-aligned RA tick labels.

    Three SIN globes showing the curve-following placement and
    per-tick rotation.
    """
    fig = plt.figure(figsize=(15, 5))
    cases = [
        ((0.0, 0.0), 0.0, "center=(0°, 0°), tick_lat=0° (baseline)"),
        ((0.0, 30.0), 0.0, "center=(0°, 30°), tick_lat=0°"),
        ((0.0, 30.0), 20.0, "center=(0°, 30°), tick_lat=20°"),
    ]
    for i, (center, tick_lat, title) in enumerate(cases, start=1):
        # tick_style='native' suppresses make_wcs_frame's auto in-frame
        # labels so this panel can demo add_curved_lon_ticks in
        # isolation for the lon side (otherwise the auto in-frame lon
        # labels would render alongside the helper's curved ones,
        # producing a doubled set).
        #
        # For the lat side, astropy's default rendering on tilted SIN
        # globes places each lat label at three positions (2 boundary
        # crossings + 1 central-meridian fallback). Hide the astropy
        # defaults and use ``add_overlay_ticks(lat_at='axis',
        # lon_at=None)`` to draw a single clean lat label per parallel
        # along the central meridian.
        ax = make_wcs_frame((1, 3, i), projection="SIN", center=center,
                            fig=fig, tick_style='native')
        for coord in (ax.coords[0], ax.coords[1]):
            coord.set_ticks_visible(False)
            coord.set_ticklabel_visible(False)
        fig.canvas.draw()
        add_overlay_ticks(ax, lat_at='axis', lon_at=None,
                          suppress_default='none')
        add_curved_lon_ticks(ax, tick_lat=tick_lat, lon_spacing=30.0,
                             tick_length_points=8.0, tick_lw=1.2,
                             offset_points=10.0, fontsize=10)
        ax.set_title(title, fontsize=10)
    fig.suptitle(
        "add_curved_lon_ticks — RA tick labels follow a constant-lat curve "
        "and rotate to align with the local meridian tangent",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.85, wspace=0.25)
    return fig


@_panel("decorations_12_reticle_styles")
def render_reticle_styles():
    """add_reticle — the four styles on a dark CAR background.

    Demonstrates the white-with-black-stroke default tuned for dark-sky
    readability, plus the L-orientation control via ``rotation=``.
    Top row: plus / x / L / circle on a representative source position.
    Bottom row: the four L orientations (rotation = 0 / 90 / 180 / 270),
    each labeled with the resulting open quadrant.
    """
    fig = plt.figure(figsize=(14, 7))

    # Top row: the four styles.
    for i, (style, kw, label) in enumerate([
        ("plus", {}, "plus"),
        ("x", {}, "x"),
        ("L", {}, "L"),
        ("circle", {"size": 16}, "circle"),
    ], start=1):
        ax = make_wcs_frame((2, 4, i), projection="CAR", center=0, fig=fig)
        ax.set_facecolor("#161628")
        add_reticle(ax, (0.0, 0.0), style=style, label=label,
                    label_fontsize=10, **kw)
        ax.set_title(f"style={style!r}", fontsize=10)

    # Bottom row: L rotation walks the open quadrant around CCW.
    for i, (rot, opens) in enumerate([
        (0, "UR"), (90, "UL"), (180, "LL"), (270, "LR"),
    ], start=5):
        ax = make_wcs_frame((2, 4, i), projection="CAR", center=0, fig=fig)
        ax.set_facecolor("#161628")
        add_reticle(ax, (0.0, 0.0), style="L", rotation=rot,
                    label=f"open {opens}", label_fontsize=10)
        ax.set_title(f"L rotation={rot}", fontsize=10)

    fig.suptitle(
        "add_reticle — four styles + L-orientation control "
        "(white body + black stroke default for dark-sky readability)",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.88, wspace=0.35, hspace=0.45)
    return fig


@_panel("decorations_13_instrument_markers")
def render_instrument_markers():
    """add_antenna_marker / add_telescope_marker / add_dome_marker —
    three procedural site-marker primitives demonstrated in sweep
    plots that walk the pointing parameter from 0° to 90°."""
    from skyplothelper.overlays.instruments import (
        add_antenna_marker,
        add_dome_marker,
        add_telescope_marker,
    )

    fig = plt.figure(figsize=(14, 9))
    sweeps = [
        ("antenna", "dish_elev",
         lambda ax, lon, lat, v: add_antenna_marker(
             ax, (lon, lat), coord_type="world",
             dish_elev=v, size=30, face_color="lightyellow",
             edge_color="#444")),
        ("telescope", "tube_elev",
         lambda ax, lon, lat, v: add_telescope_marker(
             ax, (lon, lat), coord_type="world",
             tube_elev=v, size=32, face_color="#FFFAF0",
             edge_color="#444")),
        ("dome", "slit_azim",
         lambda ax, lon, lat, v: add_dome_marker(
             ax, (lon, lat), coord_type="world",
             slit_azim=v, size=30, face_color="whitesmoke",
             edge_color="#444")),
    ]
    angles = [0, 30, 60, 90, 120]
    for row, (name, param, fn) in enumerate(sweeps, start=1):
        ax = make_wcs_frame((3, 1, row), projection="CAR", center=0,
                             fig=fig)
        fig.canvas.draw()
        for i, v in enumerate(angles):
            lon = -120 + i * 60
            fn(ax, lon, 15, v)
            ax.text(lon, -15, f"{param}={v}°",
                     transform=ax.get_transform("world"),
                     ha="center", fontsize=8)
        ax.set_title(f"add_{name}_marker — {param} sweep",
                      fontsize=10)
    fig.suptitle(
        "Procedural instrument markers — sweeps across pointing "
        "parameters (0° → 120°)", fontsize=11)
    fig.subplots_adjust(top=0.92, hspace=0.5)
    return fig


@_panel("decorations_14_pole_rod")
def render_pole_rod():
    """add_pole_rod — 2x3 grid across tilts and rendering modes.

    Top row exercises the default polished look (bone-white core, dark
    stroke, ``clip_on=False``) at three obliquity / tilt combinations.
    Bottom row varies the rendering: with end markers; ``x-ray`` mode
    (``occlude_back=False``); stroke disabled.

    A filled disk is dropped under the rod on each panel so the
    back-hemisphere occlusion (zorder=-5 behind disk) is visible.
    """
    import matplotlib.patches as mpatches
    import numpy as np

    def _disk(ax, color="#6da7c9"):
        xc = ax.wcs.wcs.crpix[0] - 1.0
        yc = ax.wcs.wcs.crpix[1] - 1.0
        cdelt = abs(ax.wcs.wcs.cdelt[0])
        R = (180.0 / np.pi) / cdelt
        ax.add_patch(mpatches.Circle((xc, yc), R, facecolor=color,
                                      edgecolor="k", linewidth=0.5,
                                      zorder=1))

    fig = plt.figure(figsize=(13.5, 9))

    configs = [
        # (subplot, title, frame_kw, pole_kw)
        (231, "Equator-on, default",
         dict(center_LONdeg=0, center_LATdeg=0, lonpole=0),
         dict()),
        (232, "Obliquity 23.44° (via lonpole)",
         dict(center_LONdeg=0, center_LATdeg=0, lonpole=23.44),
         dict()),
        (233, "Tilted +20°, obliquity 23.44°",
         dict(center_LONdeg=30, center_LATdeg=20, lonpole=-23.44),
         dict()),
        (234, "End markers",
         dict(center_LONdeg=0, center_LATdeg=15, lonpole=23.44),
         dict(end_marker="o", end_marker_size=6)),
        (235, "occlude_back=False (x-ray)",
         dict(center_LONdeg=0, center_LATdeg=20, lonpole=0),
         dict(occlude_back=False)),
        (236, "stroke disabled, custom color",
         dict(center_LONdeg=0, center_LATdeg=15, lonpole=23.44),
         dict(color="firebrick", stroke_color=None, linewidth=2.0)),
    ]

    for subplot, title, frame_kw, pole_kw in configs:
        # Tilted-Earth demo (ITRS, obliquity, rotation-axis rod) → geographic.
        ax = make_planet_frame(subplot, **frame_kw)
        _disk(ax)
        add_pole_rod(ax, **pole_kw)
        ax.set_title(title, fontsize=9)

    fig.suptitle(
        "add_pole_rod — tilts and rendering modes "
        "(disk fills body so back-hemisphere occlusion is visible)",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.92, hspace=0.25, wspace=0.1)
    return fig


@_panel("decorations_15_scale_bar_pseudocylindrical")
def render_scale_bar_pseudocylindrical():
    """add_scale_bar — auto-routing on pseudocylindrical projections.

    Top row: FITS pseudocylindricals (AIT / MOL / SFL) — the WCS world
    transform handles projection so add_scale_bar_curved_parallel works
    via its standard sample-and-project path.

    Bottom row: skyplothelper's custom-transform pseudocylindricals
    (robinson / kavrayskiy / mcbryde) — ``ax.wcs`` is ``None`` but
    ``ax.get_transform('world')`` still routes (lon, lat) through the
    custom mpl Projection, so the dispatcher recognizes them as
    WCSAxes and routes appropriately.
    """
    from skyplothelper import add_scale_bar

    fig = plt.figure(figsize=(15, 8))
    fits_panels = ["AIT", "MOL", "SFL"]
    custom_panels = ["robinson", "kavrayskiy", "mcbryde"]
    all_panels = fits_panels + custom_panels
    for i, proj in enumerate(all_panels, start=1):
        ax = make_wcs_frame((2, 3, i), projection=proj, center=0, fig=fig)
        fig.canvas.draw()
        add_scale_bar(ax, lon_0=0, lat_0=0,
                       length_km=3000.0, style="checkered",
                       segment_km=500)
        ax.set_title(f"{proj}: add_scale_bar (auto-routed)", fontsize=9)
    fig.suptitle(
        "add_scale_bar — auto-routing on pseudocylindrical projections "
        "(top: FITS WCS, bottom: custom-transform)",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.90, hspace=0.30, wspace=0.15)
    return fig


def main():
    banner("globe.decorations — merge-verification visual gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
