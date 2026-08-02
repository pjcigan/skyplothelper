"""Render globe extras (baselines / insets / nightshade) for visual eyeballing.

Boundaries are skipped because the canonical `.npz` files aren't yet
published — coverage of those will come once the data is in the repo.

Produces:
  - globe_ext_01_baselines_basic.png       — VLBI sites with baseline lines
  - globe_ext_02_baselines_with_lengths.png — same with length labels
  - globe_ext_03_baselines_back_hemisphere.png — back-hemisphere style demo
  - globe_ext_04_inset_basic.png           — globe + reprojected TAN inset
  - globe_ext_05_inset_marked_connected.png — inset with mark + connector lines
  - globe_ext_06_inset_partial_overlap.png — partial-overlap inset demo
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from _common import banner, save_or_show

from skyplothelper.globe.baselines import plot_baselines
from skyplothelper.globe.boundaries import (
    clip_to_land,
    clip_to_ocean,
    plot_coastlines,
    plot_lakes,
    plot_land,
    plot_rivers,
)
from skyplothelper.globe.frame import make_planet_frame
from skyplothelper.globe.insets import (
    connect_inset_axes,
    mark_inset_axes,
    reproject_inset_axes,
)

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


_VLBI_SITES = {
    "VLA":         (-107.6, 34.1),
    "GBT":          (-79.8, 38.4),
    "ALMA":         (-67.7, -23.0),
    "EVN":           (  6.6, 53.0),
    "ATCA":         (149.6, -30.3),
    "VLBI Brazil":  (-44.4, -22.5),
    "Hartebeesthoek": (27.7, -25.9),
    "Effelsberg":     ( 6.9, 50.5),
}


def _make_globe(center_LONdeg=0, center_LATdeg=0):
    fig = plt.figure(figsize=(8, 8))
    ax = make_planet_frame(111, center_LONdeg=center_LONdeg,
                          center_LATdeg=center_LATdeg)
    fig.canvas.draw()
    return fig, ax


@_panel("globe_ext_01_baselines_basic")
def render_baselines_basic():
    fig, ax = _make_globe(center_LONdeg=0, center_LATdeg=20)
    plot_baselines(ax, _VLBI_SITES, pairs="all",
                   color="C0", linewidth=0.6, alpha=0.7,
                   show_markers=True, marker_color="C3",
                   show_site_labels=True)
    ax.set_title("plot_baselines — global VLBI network ('all' pairs)",
                 fontsize=11)
    return fig


@_panel("globe_ext_02_baselines_with_lengths")
def render_baselines_with_lengths():
    fig, ax = _make_globe(center_LONdeg=-50, center_LATdeg=10)
    selected = {k: _VLBI_SITES[k] for k in ("VLA", "GBT", "ALMA", "EVN")}
    pairs = [("VLA", "GBT"), ("VLA", "ALMA"), ("GBT", "EVN"),
             ("ALMA", "EVN")]
    plot_baselines(ax, selected, pairs=pairs,
                   color="C2", linewidth=1.0,
                   show_markers=True, marker_color="C3",
                   show_lengths=True, length_unit="km")
    ax.set_title("plot_baselines — 4 baselines with km length labels",
                 fontsize=11)
    return fig


@_panel("globe_ext_03_baselines_back_hemisphere")
def render_baselines_back_hemisphere():
    fig, ax = _make_globe(center_LONdeg=-90, center_LATdeg=20)
    plot_baselines(ax, _VLBI_SITES, pairs="all",
                   color="C0", linewidth=0.6, alpha=0.7,
                   back_hemisphere_linestyle=":")
    ax.set_title("plot_baselines — back_hemisphere_linestyle=':' "
                 "(dotted lines for back-hemisphere arcs)", fontsize=10)
    return fig


@_panel("globe_ext_04_inset_basic")
def render_inset_basic():
    fig = plt.figure(figsize=(9, 7))
    parent_ax = make_planet_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    parent_ax.set_title("Globe + reprojected TAN inset on (0°, 30°)",
                        fontsize=11)
    inset = reproject_inset_axes(
        parent_ax, rect=[0.62, 0.55, 0.35, 0.35],
        center=(0.0, 30.0), size=15.0,
        projection="TAN", transform="parent",
    )
    inset.set_title("TAN @ (0°, 30°)", fontsize=9)
    return fig


@_panel("globe_ext_05_inset_marked_connected")
def render_inset_marked_connected():
    """Position the inset OFF the highlighted region so connectors
    are clearly visible in the parent frame, without being hidden
    behind the inset axes themselves."""
    fig = plt.figure(figsize=(11, 7))
    parent_ax = make_planet_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    parent_ax.set_title("Globe + marked TAN inset with connectors "
                        "(inset positioned off-region)", fontsize=11)
    inset = reproject_inset_axes(
        parent_ax, rect=[0.78, 0.05, 0.20, 0.32],
        center=(-30.0, 30.0), size=12.0,
        projection="TAN", transform="figure",
    )
    inset.set_title("TAN @ (-30°, 30°)", fontsize=9)
    mark_inset_axes(parent_ax, inset, edgecolor="C3", lw=1.5)
    connect_inset_axes(parent_ax, inset, color="C3", linewidth=1.2)
    return fig


@_panel("globe_ext_06_inset_partial_overlap")
def render_inset_partial_overlap():
    """Inset partially overlaps the parent — connectors traverse
    both above and below the inset frame, demonstrating that the
    marker patch + connectors render across the parent boundary
    cleanly."""
    fig = plt.figure(figsize=(10, 7))
    parent_ax = make_planet_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    parent_ax.set_title("Inset partially overlapping the parent — "
                        "connectors visible inside and outside",
                        fontsize=11)
    inset = reproject_inset_axes(
        parent_ax, rect=[0.45, 0.04, 0.30, 0.30],
        center=(-25.0, -10.0), size=10.0,
        projection="TAN", transform="figure",
        facecolor="white",
    )
    inset.set_title("TAN @ (-25°, -10°)", fontsize=9)
    mark_inset_axes(parent_ax, inset, edgecolor="C3", lw=1.5)
    connect_inset_axes(parent_ax, inset, color="C3", linewidth=1.2)
    return fig


@_panel("globe_ext_07_circular_inset")
def render_circular_inset():
    """AIT all-sky parent with a circular SIN inset: ROI marked as a
    sky-circle via ``mark_inset_axes(style='circle')``, connectors
    rendered as outer common tangent lines via the new
    ``connect_inset_axes`` circular auto-detect path."""
    from astropy.visualization.wcsaxes.frame import EllipticalFrame

    from skyplothelper.wcs_frame import make_wcs_frame

    fig = plt.figure(figsize=(12, 6))
    parent_ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    parent_ax.set_title("AIT main + circular SIN inset — circular ROI + "
                         "tangent connectors", fontsize=11)
    inset = reproject_inset_axes(
        parent_ax, rect=[0.72, 0.08, 0.25, 0.5], transform="figure",
        projection="SIN", center=(120, -20), size=40,
        frame_class=EllipticalFrame,
    )
    inset.set_title("SIN @ (120°, -20°)", fontsize=9)
    mark_inset_axes(parent_ax, inset, style="circle",
                     center=(120, -20), radius=20,
                     edgecolor="C3", linewidth=1.5)
    connect_inset_axes(parent_ax, inset, color="C3", linewidth=1.2)
    return fig


@_panel("globe_ext_08_curvature_sweep")
def render_curvature_sweep():
    """Three-panel sweep across ``curvature=`` values on the same
    AIT + circular SIN inset setup. Demonstrates the outward-bowing
    Bezier connector option."""
    from astropy.visualization.wcsaxes.frame import EllipticalFrame

    from skyplothelper.wcs_frame import make_wcs_frame

    fig = plt.figure(figsize=(12, 10))
    for idx, curv in enumerate([-0.2, 0.0, 0.25, 0.5]):
        parent_ax = make_wcs_frame((2, 2, idx + 1), projection="AIT",
                                     center=180, fig=fig)
        fig.canvas.draw()
        inset = reproject_inset_axes(
            parent_ax, rect=[0.62, 0.05, 0.32, 0.32], transform="parent",
            projection="SIN", center=(80, 30), size=30,
            frame_class=EllipticalFrame,
        )
        mark_inset_axes(parent_ax, inset, style="circle",
                         center=(80, 30), radius=15,
                         edgecolor="C3", linewidth=1.5)
        connect_inset_axes(parent_ax, inset, color="C3", linewidth=1.2,
                             curvature=curv)
        parent_ax.set_title(f"curvature={curv}", fontsize=11)
    fig.suptitle("connect_inset_axes — curvature sweep "
                  "(positive = outward Bezier; negative = inward)",
                  fontsize=12)
    fig.subplots_adjust(top=0.92, hspace=0.25, wspace=0.15)
    return fig


@_panel("globe_ext_09_planet_car_baselines")
def render_planet_car_baselines():
    """Flat (non-globe) planet frame via make_planet_frame(projection='CAR'):
    a whole-world VLBI network with coastlines and full great-circle
    baselines — the FITS-projection path (ax.wcs present). Exercises the
    hemisphere_only auto-detect (whole surface shown, not culled)."""
    fig = plt.figure(figsize=(9, 5))
    ax = make_planet_frame(111, projection="CAR", center_LONdeg=0)
    plot_coastlines(ax, color="0.6", lw=0.5)
    plot_baselines(ax, _VLBI_SITES, pairs="all", color="C0", linewidth=0.6,
                   alpha=0.8, marker_color="C3", show_site_labels=True,
                   site_label_fontsize=7)
    ax.set_title("make_planet_frame(projection='CAR') — flat VLBI baseline map",
                 fontsize=11)
    return fig


@_panel("globe_ext_10_planet_robinson_baselines")
def render_planet_robinson_baselines():
    """Flat planet frame on a NON-FITS projection
    (make_planet_frame(projection='robinson'), ax.wcs is None): coastlines +
    baselines must render without reaching for ax.wcs, and the ITRS axes must
    read 'Longitude'/'Latitude' (not 'RA/Dec')."""
    fig = plt.figure(figsize=(9, 5))
    ax = make_planet_frame(111, projection="robinson", center_LONdeg=0)
    plot_coastlines(ax, color="0.6", lw=0.5)
    plot_baselines(ax, _VLBI_SITES, pairs="all", color="C0", linewidth=0.6,
                   alpha=0.8, marker_color="C3", show_site_labels=True,
                   site_label_fontsize=7)
    ax.set_title("make_planet_frame(projection='robinson') — non-FITS planet map",
                 fontsize=11)
    return fig


@_panel("globe_ext_11_earth_filled")
def render_earth_filled():
    """Filled Earth on a CAR frame via the spherical-region machinery:
    plot_land (continents), plot_lakes (Great Lakes/Caspian as water),
    plot_rivers (Nile/Amazon/... centerlines), and a plot_coastlines stroke.
    Exercises the F3 fill path + the stroke knob."""
    fig = plt.figure(figsize=(9, 4.6))
    ax = make_planet_frame(111, projection="CAR", center_LONdeg=0)
    ax.set_facecolor("#dfeaf2")                       # ocean
    plot_land(ax, facecolor="#e7dbb8")                # land
    plot_lakes(ax, facecolor="#dfeaf2")               # lakes = ocean color
    plot_rivers(ax, color="#5b8fb9", lw=0.6)          # rivers
    plot_coastlines(ax, color="0.35", lw=0.4)         # coastline stroke
    ax.set_title("Filled Earth — plot_land + plot_lakes + plot_rivers "
                 "(region machinery)", fontsize=11)
    return fig


@_panel("globe_ext_12_clip_land_ocean")
def render_clip_land_ocean():
    """Clip planet-frame overlays to land vs ocean (G2). A regular point grid
    is split: red points clipped to land, blue points clipped to ocean — the
    region machinery's clip path used as a mask."""
    lon, lat = np.meshgrid(np.arange(-175, 180, 8), np.arange(-84, 85, 8))
    fig = plt.figure(figsize=(9, 4.6))
    ax = make_planet_frame(111, projection="CAR", center_LONdeg=0)
    plot_land(ax, facecolor="0.9", zorder=0)
    tr = ax.get_transform("world")
    clip_to_land(ax, ax.scatter(lon.ravel(), lat.ravel(), s=7, c="C3", zorder=5,
                                transform=tr))
    clip_to_ocean(ax, ax.scatter(lon.ravel(), lat.ravel(), s=7, c="C0", zorder=5,
                                 transform=tr))
    ax.set_title("clip_to_land (red) + clip_to_ocean (blue)", fontsize=11)
    return fig


@_panel("globe_ext_13_lon_west")
def render_lon_west():
    """F2 -- longitude-West labeling (route b). Two normal (unmirrored) CAR
    planet maps with ``lon_west=True``: only the tick labels change to read
    west-longitude (W/E suffix) while the data stay east-internal, so the map
    still looks and behaves like an ordinary planet map. Shown at two centers
    to confirm the labels track the seam. The red stations are given in
    degrees-WEST and fed through ``lon_west_to_east``."""
    from skyplothelper import lon_west_to_east
    sites_W = {"VLA": (107.6, 34.1), "GBT": (79.8, 38.4), "OVRO": (118.3, 37.2)}
    fig = plt.figure(figsize=(11, 3.4))
    for col, cen in enumerate((0, -100), start=1):
        ax = make_planet_frame((1, 2, col), projection="CAR",
                               center_LONdeg=cen, lon_west=True, fig=fig)
        plot_coastlines(ax, color="0.5", lw=0.6)
        tr = ax.get_transform("world")
        for lon_w, lat in sites_W.values():
            ax.plot(lon_west_to_east(lon_w), lat, "o", color="C3", ms=5,
                    transform=tr, zorder=5)
        ax.set_title(f"lon_west=True, center={cen}° (normal planet map)",
                     fontsize=10)
    fig.suptitle("F2: west-longitude labels on a normal planet map "
                 "(stations fed in °W)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


@_panel("globe_ext_14_land_lakes")
def render_land_lakes():
    """F3(ii): plot_land(lakes=True) punches the lakes out of the land as true
    holes via the region set-algebra (land − lakes). Zoomed on the Great
    Lakes so the holes are visible."""
    from skyplothelper.globe.boundaries import plot_land
    fig = plt.figure(figsize=(11, 4.6))
    ax1 = make_planet_frame((1, 2, 1), projection="CAR", center_LONdeg=0,
                            fig=fig)
    plot_land(ax1, facecolor="0.7", lakes=True)
    ax1.set_title("plot_land(lakes=True) — global", fontsize=10)
    ax2 = make_planet_frame((1, 2, 2), projection="CAR", center_LONdeg=-84,
                            center_LATdeg=44, fov_deg=40, fig=fig)
    plot_land(ax2, facecolor="0.7", lakes=True)
    plot_coastlines(ax2, color="0.35", lw=0.7)
    ax2.set_title("zoom: Great Lakes as holes", fontsize=10)
    return fig


@_panel("globe_ext_15_plate_fill")
def render_plate_fill():
    """F3(ii): plot_tectonic_plates(fill=True) — categorical plate map (one
    color per plate; Pacific split pieces share a color and trace the frame
    edge) with the boundary arcs on top."""
    from skyplothelper.globe.boundaries import plot_tectonic_plates
    fig = plt.figure(figsize=(12, 5))
    for col, (proj, cen) in enumerate([("CAR", 0), ("MOL", -60)], start=1):
        ax = make_planet_frame((1, 2, col), projection=proj, center_LONdeg=cen,
                               fig=fig)
        plot_tectonic_plates(ax, fill=True, alpha=0.6)
        plot_coastlines(ax, color="k", lw=0.3)
        plot_tectonic_plates(ax, color="0.1", lw=0.6)
        ax.set_title(f"{proj} center={cen}: plate fill", fontsize=10)
    return fig


@_panel("globe_ext_16_plate_choropleth")
def render_plate_choropleth():
    """G3: plot_tectonic_plates(fill=True, values=…) choropleth — plates
    colored by a per-plate value with a colorbar."""
    import numpy as np

    import skyplothelper as sph
    from skyplothelper.globe.boundaries import _closed_rings, _find_data_file, plot_tectonic_plates
    d = np.load(_find_data_file("tectonic_plates.npz"))
    codes = [str(c) for c in d["plate_codes"]]
    rings = _closed_rings(d["plate_polygons"])
    cent = {}
    for c, r in zip(codes, rings):
        cent[c] = max(cent.get(c, 0.0), abs(float(np.nanmean(r[1]))))
    fig = plt.figure(figsize=(8, 4.6))
    ax = make_planet_frame(111, projection="MOL", center_LONdeg=0, fig=fig)
    sm = plot_tectonic_plates(ax, fill=True, values=cent, cmap="plasma",
                              edgecolor="0.25")
    plot_coastlines(ax, color="k", lw=0.3)
    sph.add_colorbar(sm, ax=ax, label="|plate centroid lat| (deg)")
    ax.set_title("plate choropleth (fill=True, values=…)", fontsize=10)
    return fig


def main():
    banner("globe extras (baselines + insets) — gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
