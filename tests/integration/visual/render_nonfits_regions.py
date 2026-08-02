"""Render region fills / shapes on the non-FITS custom-projection frames (G4).

The Robinson / Eckert IV / Winkel Tripel / Kavrayskiy VII / McBryde frames are
drawn by a matplotlib ``CurvedTransform`` (``ax.wcs is None``) and, as of the
1.1.0 G4 work, support the full region-fill machinery through
``WCSNonFitsProjector``. These panels are for eyeballing that the fills follow
the projection curvature, register with the frame, split cleanly at the wrap
seam, and fill polar caps as caps (not their complement).

Produces:
  - nonfits_01_shapes_gallery.png  — every shape helper on all five frames
  - nonfits_02_seam_and_poles.png  — wrap-seam-straddling + polar-cap stress
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from _common import banner, save_or_show

import skyplothelper as sph

PANELS = {}
FRAMES = ["robinson", "eckert_iv", "winkel_tripel", "kavrayskiy", "mcbryde"]


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


@_panel("nonfits_01_shapes_gallery")
def render_shapes_gallery():
    """Each shape helper on each non-FITS projection: geodesic circle, spherical
    polygon, rectangle, ellipse, a Tissot grid, a compound crescent, and a
    stroked outline circle."""
    fig = plt.figure(figsize=(15, 9))
    for i, proj in enumerate(FRAMES, start=1):
        ax = sph.make_wcs_frame((2, 3, i), proj, frame="ICRS", center=0,
                                fig=fig)
        sph.tissot(ax, rad_deg=5, lons=np.linspace(-150, 150, 6),
                   lats=np.linspace(-60, 60, 3), facecolor="0.6",
                   edgecolor="0.4", alpha=0.5, lw=0.4)
        sph.add_geodesic_circle(ax, 40, 20, radius_deg=20, facecolor="C0",
                                edgecolor="navy", alpha=0.4)
        sph.add_spherical_polygon(ax, [140, 175, 175, 140, 140],
                                  [-25, -25, 20, 20, -25], facecolor="C2",
                                  edgecolor="darkgreen", alpha=0.4)
        sph.add_rectangle(ax, -70, -25, width=45, height=25, facecolor="C1",
                          edgecolor="saddlebrown", alpha=0.4)
        sph.add_ellipse(ax, -50, 45, semi_major=18, semi_minor=9, angle=25,
                        facecolor="C4", edgecolor="indigo", alpha=0.4)
        (sph.CompoundRegion(ax)
         .add_circle(120, 15, 24).subtract_circle(135, 15, 11)
         .render(facecolor="C5", edgecolor="k", alpha=0.5))
        sph.add_geodesic_circle(ax, -120, 30, radius_deg=13, facecolor="none",
                                edgecolor="C3", lw=1.6, stroke_color="w",
                                stroke_lw=3)
        ax.set_title(proj, fontsize=11)
    fig.suptitle("Region fills on non-FITS custom projections (G4)",
                 fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.90, wspace=0.25, hspace=0.32)
    return fig


@_panel("nonfits_02_seam_and_poles")
def render_seam_and_poles():
    """The tricky cases: a circle and a box straddling the wrap meridian (they
    must split into left/right lobes) and small caps at both poles (they must
    fill the cap, not the frame complement)."""
    fig = plt.figure(figsize=(14, 5.5))
    for i, proj in enumerate(["robinson", "eckert_iv", "winkel_tripel"],
                             start=1):
        ax = sph.make_wcs_frame((1, 3, i), proj, frame="ICRS", center=0,
                                fig=fig)
        # Seam-straddling (centered on the wrap meridian, 180 = frame edge).
        sph.add_geodesic_circle(ax, 180, 0, radius_deg=28, facecolor="C0",
                                edgecolor="navy", alpha=0.5)
        sph.add_spherical_polygon(ax, [150, 210, 210, 150, 150],
                                  [30, 30, 55, 55, 30], facecolor="C1",
                                  edgecolor="saddlebrown", alpha=0.5)
        # Polar caps at both poles.
        sph.add_geodesic_circle(ax, 0, 85, radius_deg=16, facecolor="C3",
                                edgecolor="darkred", alpha=0.55)
        sph.add_geodesic_circle(ax, 60, -82, radius_deg=20, facecolor="C2",
                                edgecolor="darkgreen", alpha=0.55)
        ax.set_title(proj, fontsize=11)
    fig.suptitle("Non-FITS wrap-seam split + polar-cap fill (G4)",
                 fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.86, wspace=0.25)
    return fig


@_panel("nonfits_03_clip_to_land")
def render_clip_to_land():
    """clip_to_land / clip_to_ocean on non-FITS frames: a full-sky gradient
    image masked to land (left) and to ocean (right), with coastlines to show
    the clip registers with them. Needs the Earth data (prepare_earth_data);
    returns an annotated placeholder if it isn't present."""
    fig = plt.figure(figsize=(14, 5.5))
    lon = np.linspace(-180, 180, 200)
    lat = np.linspace(-89, 89, 100)
    grid_lon, grid_lat = np.meshgrid(lon, lat)
    for i, (proj, fn, title) in enumerate([
        ("robinson", "clip_to_land", "clip_to_land"),
        ("eckert_iv", "clip_to_ocean", "clip_to_ocean"),
    ], start=1):
        ax = sph.make_planet_frame((1, 2, i), projection=proj)
        try:
            mesh = ax.pcolormesh(grid_lon, grid_lat, grid_lat,
                                 transform=ax.get_transform("world"),
                                 cmap="viridis", shading="auto")
            getattr(sph, fn)(ax, mesh)
            sph.plot_coastlines(ax, color="k", lw=0.4)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "Earth data not present\n(prepare_earth_data)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="0.4")
        ax.set_title(f"{proj}: {title}", fontsize=11)
    fig.suptitle("clip_to_land / clip_to_ocean on non-FITS frames (G4)",
                 fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.88, wspace=0.15)
    return fig


def main():
    banner("Non-FITS region fills — gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
