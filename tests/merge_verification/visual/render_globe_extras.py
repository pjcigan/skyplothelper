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
from _common import banner, save_or_show

from skyplothelper.globe.baselines import plot_baselines
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


def main():
    banner("globe extras (baselines + insets) — gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
