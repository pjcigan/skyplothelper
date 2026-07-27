"""Render the grid helpers (add_second_grid / style_grid /
highlight_gridline / highlight_gridlines / add_coord_overlay) for
visual eyeballing.

Produces:
  - grid_01_second_grid_galactic.png   — ICRS axes + galactic overlay grid
  - grid_02_second_grid_with_labels.png — second grid with tick labels visible
  - grid_03_style_grid_stroke.png      — style_grid stroke effect
  - grid_04_highlight_single.png       — highlight_gridline for one meridian
  - grid_05_highlight_multiple.png     — highlight_gridlines for several
                                           meridians/parallels
  - grid_06_highlight_cmap.png         — highlight_gridlines with lon_cmap
  - grid_07_coord_overlay_galactic.png — add_coord_overlay galactic-on-ICRS
  - grid_08_coord_overlay_ticks_inner_box.png — tick rendering
                                                  on a custom inner-box frame
  - grid_09_coord_overlay_labels.png   — tick label rendering
                                          (tangent-aligned + horizontal)
  - grid_10_overlay_ticks_on_frame.png — add_overlay_ticks on
                                          the projection boundary curve
  - grid_11_overlay_ticks_axis_curve.png — axis-curve mode
                                          (lon_at='axis', lat_at='axis')
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from _common import banner, save_or_show

from skyplothelper.coord_overlay import (
    CoordinateOverlay,
    _FrameCurve,
    add_coord_overlay,
    add_overlay_ticks,
)
from skyplothelper.grid import (
    add_second_grid,
    highlight_gridline,
    highlight_gridlines,
    style_grid,
)
from skyplothelper.overlays.planes import add_plane_overlay
from skyplothelper.wcs_frame import make_wcs_frame

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


def _allsky(projection="AIT", center=180, frame="ICRS"):
    fig = plt.figure(figsize=(11, 5.5))
    ax = make_wcs_frame(111, projection=projection, center=center,
                        frame=frame, fig=fig)
    fig.canvas.draw()
    return fig, ax


@_panel("grid_01_second_grid_galactic")
def render_second_grid_galactic():
    fig, ax = _allsky()
    add_second_grid(ax, overlay_frame="galactic",
                    color="C0", alpha=0.4, linestyle="--", linewidth=0.8)
    add_plane_overlay(ax, plane="galactic", color="C0", lw=1.2)
    ax.set_title("ICRS axes + galactic overlay grid (add_second_grid)")
    return fig


@_panel("grid_02_second_grid_with_labels")
def render_second_grid_with_labels():
    fig, ax = _allsky(projection="MOL", center=0, frame="Galactic")
    add_second_grid(ax, overlay_frame="geocentrictrueecliptic",
                    color="C2", alpha=0.4, linestyle=":",
                    linewidth=0.7, ticks=True, tick_labels=True)
    add_plane_overlay(ax, plane="ecliptic", color="C2", lw=1.2)
    ax.set_title("Galactic-frame MOL + ecliptic overlay (with tick labels)")
    return fig


@_panel("grid_03_style_grid_stroke")
def render_style_grid_stroke():
    """style_grid — solid + dashed stroked white grid demos.

    Two panels: solid-line stroke on the left, dashed-line stroke on
    the right. A faded light-blue facecolor on each axes keeps the
    white grid lines visible without needing the stroke to "rescue"
    them — the stroke is the demo, not a workaround for a bad
    contrast choice.
    """
    fig = plt.figure(figsize=(15, 5.5))
    bg = "#dde6f3"  # faded light blue: white grid stays distinguishable

    ax1 = make_wcs_frame((1, 2, 1), projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    ax1.set_facecolor(bg)
    style_grid(ax1, stroke_lw=3, stroke_color="black",
               color="white", lw=1.2, alpha=1.0, ls="-")
    ax1.set_title("style_grid — solid stroke (ls='-')", fontsize=10)

    ax2 = make_wcs_frame((1, 2, 2), projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    ax2.set_facecolor(bg)
    style_grid(ax2, stroke_lw=3, stroke_color="black",
               color="white", lw=1.2, alpha=1.0, ls="--")
    ax2.set_title("style_grid — dashed stroke (ls='--')", fontsize=10)

    fig.suptitle("style_grid — black stroke around a white grid "
                 "(faded blue background)", fontsize=12)
    fig.subplots_adjust(top=0.86, wspace=0.25)
    return fig


@_panel("grid_04_highlight_single")
def render_highlight_single():
    fig, ax = _allsky()
    highlight_gridline(ax, value=0, coord="lon", color="C3",
                       lw=2.5, label="lon=0°")
    highlight_gridline(ax, value=180, coord="lon", color="C0",
                       lw=2.5, ls="--", label="lon=180°")
    highlight_gridline(ax, value=0, coord="lat", color="C2",
                       lw=2.5, label="lat=0°")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax.set_title("highlight_gridline — three named meridians/parallel")
    return fig


@_panel("grid_05_highlight_multiple")
def render_highlight_multiple():
    fig, ax = _allsky()
    highlight_gridlines(ax,
                        lon_values=[0, 60, 120, 180, 240, 300],
                        lat_values=[-60, -30, 0, 30, 60],
                        color="C5", lw=1.5, alpha=0.7)
    ax.set_title("highlight_gridlines — 6 meridians + 5 parallels in one call")
    return fig


@_panel("grid_06_highlight_cmap")
def render_highlight_cmap():
    fig, ax = _allsky()
    highlight_gridlines(ax,
                        lon_values=list(range(0, 360, 30)),
                        lon_cmap="hsv",
                        lw=1.5, alpha=0.85)
    highlight_gridlines(ax,
                        lat_values=[-60, -30, 0, 30, 60],
                        lat_cmap="viridis",
                        lw=1.5, alpha=0.85)
    ax.set_title("highlight_gridlines — meridians color-mapped via "
                 "lon_cmap + lat_cmap", fontsize=10)
    return fig


@_panel("grid_07_coord_overlay_galactic")
def render_coord_overlay_galactic():
    """Gridline rendering only (no ticks).

    Galactic graticule drawn on top of an ICRS AIT all-sky axes via
    ``add_coord_overlay``. Companion to ``grid_01`` (which uses the
    astropy-based ``add_second_grid``); this one is the skyplothelper-
    native implementation; sibling panels extend it with tick
    marks, tick labels, and arbitrary boundary curves.
    """
    fig, ax = _allsky()
    add_coord_overlay(ax, frame="galactic",
                      color="C0", alpha=0.8, lw=0.9, ls="-")
    add_plane_overlay(ax, plane="galactic", color="C0", lw=1.4)
    ax.set_title("ICRS axes + galactic overlay grid "
                 "(add_coord_overlay)")
    return fig


@_panel("grid_08_coord_overlay_ticks_inner_box")
def render_coord_overlay_ticks_inner_box():
    """Tick mark rendering.

    Two diagnostic panels demonstrating that tick orientation follows
    the *gridline tangent* at each crossing point, not the frame
    edge perpendicular.

    Left panel — 45°-rotated diamond frame. ICRS gridlines on an
    ICRS CAR axes are axis-aligned (vertical meridians, horizontal
    parallels), but the diamond edges are at 45°. The rendered ticks
    are themselves purely axis-aligned and visibly NOT perpendicular
    to the diamond edges — direct proof that ticks track gridlines.

    Right panel — same-frame inner box, gridline-tangent diagnostic.
    Cross-frame galactic-on-ICRS CAR with an axis-aligned inner box.
    The same overlay is replotted in thin red and clipped to the
    inside of the box so the eye can trace a galactic gridline from
    inside the box (red) outward to its tick at the box edge (also
    red) and visually confirm the tangent match.
    """
    from matplotlib.patches import Polygon as MplPolygon

    fig = plt.figure(figsize=(14, 5.5))

    # --- Left: 45°-rotated diamond
    ax1 = make_wcs_frame((1, 2, 1), projection="CAR", center=180, fig=fig)
    fig.canvas.draw()
    # Center the diamond at the projection center (lon=180, lat=0) in
    # display coords; ax.bbox.extents includes the tick-label margin
    # which biases off-center.
    cx, cy = ax1.get_transform("world").transform([180., 0.])
    r = 0.35 * min(ax1.bbox.width, ax1.bbox.height)
    diamond = _FrameCurve(
        np.array([[cx, cy + r], [cx + r, cy],
                  [cx, cy - r], [cx - r, cy]]),
        name="diamond", closed=True)
    (CoordinateOverlay(ax1, frame="icrs",
                       lon_vals=list(range(120, 241, 15)),
                       lat_vals=list(range(-45, 46, 15)))
     .plot(color="C0", alpha=0.8, lw=1.0)
     .set_frame_curves([diamond])
     .discover_ticks()
     .draw_frame_curves(color="k", lw=1.5)
     .render_ticks(length=14, lw=1.5, color="red"))
    ax1.set_title("ICRS-on-ICRS CAR + 45°-rotated diamond — "
                  "axis-aligned ticks (NOT perpendicular to edge)",
                  fontsize=10)

    # --- Right: cross-frame box, with thin red gridlines inside
    ax2 = make_wcs_frame((1, 2, 2), projection="CAR", center=180, fig=fig)
    fig.canvas.draw()
    box_lonlat = np.array([[120., -40.], [240., -40.],
                            [240., 40.], [120., 40.]])
    box2 = _FrameCurve.from_world_polyline(
        ax2, box_lonlat, closed=True, name="box")

    # Main blue overlay (drawn everywhere on the axes)
    (CoordinateOverlay(ax2, frame="galactic",
                       lon_vals=list(range(0, 360, 30)),
                       lat_vals=list(range(-60, 61, 30)))
     .plot(color="C0", alpha=0.7, lw=0.9))

    # Same gridlines in thin red, clipped to inside the box so the
    # eye sees the gridline–tick tangent continuity at the edge.
    clip = MplPolygon(box_lonlat, closed=True,
                      facecolor="none", edgecolor="none",
                      transform=ax2.get_transform("world"))
    ax2.add_patch(clip)
    ov2_red = (CoordinateOverlay(ax2, frame="galactic",
                                 lon_vals=list(range(0, 360, 30)),
                                 lat_vals=list(range(-60, 61, 30)))
               .plot(color="red", alpha=0.85, lw=0.7))
    for artists in ov2_red.lon_artists + ov2_red.lat_artists:
        for line in artists:
            line.set_clip_path(clip)

    # Discover + render ticks on the box (red, matching the in-box gridlines)
    (ov2_red.set_frame_curves([box2])
     .discover_ticks()
     .draw_frame_curves(color="k", lw=1.5)
     .render_ticks(length=12, lw=1.5, color="red"))

    ax2.set_title("Galactic-on-ICRS CAR + inner box — in-box "
                  "gridlines (thin red) trace tick tangent",
                  fontsize=10)

    fig.suptitle("overlay tick marks along "
                 "gridline tangent (red)", fontsize=11)
    fig.subplots_adjust(top=0.88, wspace=0.2)
    return fig


@_panel("grid_09_coord_overlay_labels")
def render_coord_overlay_labels():
    """Tick label rendering.

    Two panels both using the cross-frame galactic-on-ICRS CAR setup
    (the case where the tangent rotation is most informative). Same
    inner box, same gridlines, same ticks — different label styles.

    Left: ``rotate='tangent'`` (default) aligns each label with the
    gridline it labels, clamped upright. Demonstrates the kapteyn-
    style overlay aesthetic where the label "continues" the gridline
    past the tick.

    Right: ``rotate='horizontal'`` keeps all labels horizontal —
    the more conventional axis-style readout.
    """
    from matplotlib.patches import Polygon as MplPolygon

    fig = plt.figure(figsize=(14, 6))
    box_lonlat = np.array([[120., -40.], [240., -40.],
                            [240., 40.], [120., 40.]])

    def _setup_panel(subplot_spec, rotate_mode, title):
        ax = make_wcs_frame(subplot_spec, projection="CAR",
                            center=180, fig=fig)
        fig.canvas.draw()
        box = _FrameCurve.from_world_polyline(
            ax, box_lonlat, closed=True, name="box")
        # Blue overlay (context)
        (CoordinateOverlay(ax, frame="galactic",
                           lon_vals=list(range(0, 360, 30)),
                           lat_vals=list(range(-60, 61, 30)))
         .plot(color="C0", alpha=0.55, lw=0.7))
        # In-box red gridlines for tangent continuity
        clip = MplPolygon(box_lonlat, closed=True,
                          facecolor="none", edgecolor="none",
                          transform=ax.get_transform("world"))
        ax.add_patch(clip)
        ov = (CoordinateOverlay(ax, frame="galactic",
                                lon_vals=list(range(0, 360, 30)),
                                lat_vals=list(range(-60, 61, 30)))
              .plot(color="red", alpha=0.85, lw=0.6))
        for artists in ov.lon_artists + ov.lat_artists:
            for line in artists:
                line.set_clip_path(clip)
        (ov.set_frame_curves([box])
           .discover_ticks()
           .draw_frame_curves(color="k", lw=1.5)
           .render_ticks(length=10, lw=1.5, color="red")
           .render_labels(fontsize=9, color="black", rotate=rotate_mode))
        ax.set_title(title, fontsize=10)
        return ov

    _setup_panel((1, 2, 1), "tangent",
                 "rotate='tangent' (kapteyn-style, aligned with gridline)")
    _setup_panel((1, 2, 2), "horizontal",
                 "rotate='horizontal' (axis-style readout)")

    fig.suptitle("overlay tick label rendering on "
                 "a custom inner-box frame", fontsize=11)
    fig.subplots_adjust(top=0.88, wspace=0.18)
    return fig


@_panel("grid_10_overlay_ticks_on_frame")
def render_overlay_ticks_on_frame():
    """add_overlay_ticks on a projection boundary curve.

    Demonstrates the public helper that auto-detects the projection
    boundary from ``ax.coords.frame`` and lays overlay ticks + labels
    on it (replacing the default WCSAxes labels via
    ``suppress_default``). The helper is most useful for *annotating
    a custom sub-frame edge* — axis-curve mode (next panel) gives the
    more traditional inside-the-frame label layout.

    Left: SIN slant-orthographic centered on (180, 30), 70° FOV. The
    overlay places lon (10ʰ, 12ʰ, 14ʰ) and lat (+0°, +15°, +30°,
    +45°, +60°) labels precisely on the circular spine, each rotated
    to the local gridline tangent.

    Right: the same helper applied to a non-rectangular pseudo-
    cylindrical (Robinson) projection — the boundary is the
    custom-frame curve, and labels follow it around the perimeter.
    """
    fig = plt.figure(figsize=(13, 6))

    # tick_style='native' opts out of make_wcs_frame's auto-trigger
    # so these panels can demo add_overlay_ticks in isolation (otherwise
    # both the auto-set and the explicit call would render, producing
    # slightly-offset doubled labels).
    ax1 = make_wcs_frame((1, 2, 1), projection="SIN",
                         center=(180, 30), fov_deg=70.0, fig=fig,
                         tick_style='native')
    add_overlay_ticks(
        ax1,
        tick_kwargs={"length": 7, "color": "red", "lw": 1.2},
        label_kwargs={"fontsize": 9, "color": "red"})
    ax1.set_title("SIN slant orthographic — overlay ticks on circular spine",
                  fontsize=10)

    ax2 = make_wcs_frame((1, 2, 2), projection="robinson",
                         center=180, fig=fig, tick_style='native')
    add_overlay_ticks(
        ax2,
        tick_kwargs={"length": 6, "color": "red", "lw": 1.0},
        label_kwargs={"fontsize": 8, "color": "red"})
    ax2.set_title("Robinson allsky — overlay ticks on custom-curve spine",
                  fontsize=10)

    fig.suptitle("add_overlay_ticks "
                 "auto-detects the projection boundary",
                 fontsize=11)
    fig.subplots_adjust(top=0.88, wspace=0.2)
    return fig


@_panel("grid_11_overlay_ticks_axis_curve")
def render_overlay_ticks_axis_curve():
    """Axis-curve tick placement.

    Boundary mode is best for annotating a sub-frame edge;
    axis-curve mode is closer to traditional astronomy
    plot conventions — lon labels along the equator (bowing with
    the projection's curvature), lat labels along the central
    meridian.

    Left: same Robinson axes as grid_10's right panel, switched to
    ``lon_at='axis', lat_at='axis'``. Lon labels (0ʰ … 22ʰ) trace
    the equator, lat labels (0° / ±15° / ±30° / …) climb the
    central meridian.

    Right: mixed mode on an AIT — ``lon_at='lat=30'`` puts the lon
    labels along the +30° parallel (an interior reference curve),
    while ``lat_at='boundary'`` keeps lat labels on the elliptical
    edge. Demonstrates the two axes are independent.
    """
    fig = plt.figure(figsize=(14, 6))

    # tick_style='native' on Robinson avoids the make_wcs_frame auto-
    # trigger overlapping with this panel's explicit add_overlay_ticks
    # call (would produce slightly-offset doubled labels). AIT is
    # 'elliptical' and not in the auto-set, so no override needed there.
    ax1 = make_wcs_frame((1, 2, 1), projection="robinson",
                         center=180, fig=fig, tick_style='native')
    add_overlay_ticks(
        ax1, lon_at="axis", lat_at="axis",
        tick_kwargs={"length": 7, "color": "red", "lw": 1.2},
        label_kwargs={"fontsize": 9, "color": "black"})
    ax1.set_title("Robinson — lon_at='axis', lat_at='axis'",
                  fontsize=10)

    ax2 = make_wcs_frame((1, 2, 2), projection="AIT",
                         center=180, fig=fig)
    add_overlay_ticks(
        ax2, lon_at="lat=30", lat_at="boundary",
        tick_kwargs={"length": 7, "color": "red", "lw": 1.2},
        label_kwargs={"fontsize": 9, "color": "black"})
    ax2.set_title("AIT — lon_at='lat=30' (interior curve), "
                  "lat_at='boundary' (mixed)", fontsize=10)

    fig.suptitle("axis-curve tick placement "
                 "(labels follow projection curves)", fontsize=11)
    fig.subplots_adjust(top=0.88, wspace=0.18)
    return fig


@_panel("grid_12_coord_overlay_cross_frame")
def render_coord_overlay_cross_frame():
    """Cross-frame ``CoordinateOverlay`` — the canonical "main frame
    X + overlay frame Y" compound plot.

    Galactic Mollweide host with an ecliptic overlay. Two panels
    contrasting the two label-placement modes available when host
    and overlay frames differ:

    Left — **boundary mode** (``lon_at='boundary'``,
    ``lat_at='boundary'``, the default). The ecliptic overlay's
    ticks + labels sit on the projection's natural boundary curve
    (the MOL ellipse), like a second axis sharing the spine. The
    traditional astronomy look.

    Right — **in-frame label mode** (``lon_at='axis'``,
    ``lat_at='axis'`` with ``frame='ecliptic'``). Ticks + labels
    move *inside* the frame, anchored to the overlay's own central
    meridian and equator (the diagonal green curve = ecliptic β=0,
    the curve crossing it vertically through the projection center
    = ecliptic prime meridian). Labels visually attach to the
    overlay's gridlines, making the cross-frame compound plot
    much easier to read at a glance.

    Both panels share: galactic gray gridlines + tick labels (the
    host axes' default), ecliptic green dotted gridlines (from
    ``add_coord_overlay``), and the solid green ecliptic equator
    line (``add_plane_overlay``).
    """
    fig = plt.figure(figsize=(18, 7))

    # --- Left: boundary mode
    ax1 = make_wcs_frame((1, 2, 1), projection="MOL", center=0,
                         frame="Galactic", fig=fig)
    fig.canvas.draw()
    add_coord_overlay(ax1, frame="geocentrictrueecliptic",
                      color="C2", alpha=0.6, ls=":", lw=0.7)
    add_overlay_ticks(ax1, frame="geocentrictrueecliptic",
                      tick_kwargs={"length": 6, "color": "C2", "lw": 1.0},
                      label_kwargs={"fontsize": 9, "color": "C2"})
    add_plane_overlay(ax1, plane="ecliptic", color="C2", lw=1.2)
    ax1.set_title("boundary mode — labels on MOL ellipse "
                  "(traditional axis look)", fontsize=10)

    # --- Right: in-frame label mode
    ax2 = make_wcs_frame((1, 2, 2), projection="MOL", center=0,
                         frame="Galactic", fig=fig)
    fig.canvas.draw()
    add_coord_overlay(ax2, frame="geocentrictrueecliptic",
                      color="C2", alpha=0.6, ls=":", lw=0.7)
    add_overlay_ticks(ax2, frame="geocentrictrueecliptic",
                      lon_at="axis", lat_at="axis",
                      suppress_default="none",
                      tick_kwargs={"length": 6, "color": "C2", "lw": 1.0},
                      label_kwargs={"fontsize": 9, "color": "C2"})
    add_plane_overlay(ax2, plane="ecliptic", color="C2", lw=1.2)
    ax2.set_title("in-frame mode — labels along overlay's own "
                  "equator + central meridian", fontsize=10)

    fig.suptitle("cross-frame coord overlay — Galactic MOL host + "
                 "ecliptic overlay (two label-placement modes)",
                 fontsize=11)
    fig.subplots_adjust(top=0.88, wspace=0.12)
    return fig


def main():
    banner("grid — merge-verification visual gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
