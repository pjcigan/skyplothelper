"""Render the *real* skyplothelper regions that the set-algebra manim demo
cross-fades into (tutorial #7, demo B).

This is the sph side of the pipeline. It draws no animation — it produces the
genuine ``CompoundRegion`` PNGs that ``regions_setalgebra.py`` dissolves the
abstract Venn shapes into. The division of labor is the whole point:
skyplothelper is the source of scientific truth (these are real projection-aware
footprints, holes and seams handled), manim is only the camera and typographer.

The scene accumulates the three abstract operations in the corners, gathers them
onto one globe, then builds a survey footprint piece by piece. So two families of
real render, both on the shared navy sky canvas:

* ``trio_globe.png`` — one orthographic globe carrying all three operations as
  real regions at three sky positions (union upper-left, intersection upper-right,
  difference low): the manim corners fly here, and the off-center pairs show the
  projection stretching them toward the limb.
* ``build_*.png`` — the notebook's worked survey footprint assembled one operation
  at a time on an all-sky Aitoff frame (empty → box → −galactic band → −hole), with
  the two cut zones rendered alone (``band_zone``/``hole_zone``) so the coda can
  flash each contribution in its own color before it carves.

All the all-sky build frames share one figure size + ``bbox_inches`` behavior (the
frame ellipse is identical in every stage, so the tight crop is identical too),
which keeps them in pixel register for manim's cross-fades and overlays.

Run in the astropy env (``cenv``), from the repo root::

    python docs/manim/make_regions_assets.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import skyplothelper as sph

SKY = "#16203A"                                     # shared navy canvas
GRID = "#3C4A66"                                    # faint graticule on navy
INK = "#D7DEE8"                                     # light ticks/labels on navy

# The uranometria cycle, exactly as the notebook's §4 gallery uses these ops.
_CYC = sph.CYCLE_PALETTES["uranometria"]["colors"]
COL = {"blue": _CYC[0], "gold": _CYC[2], "gray": _CYC[3],
       "green": _CYC[4], "rust": _CYC[5]}

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")


def _style_navy(ax):
    ax.set_facecolor(SKY)
    ax.figure.canvas.draw()
    sph.style_wcs_axes(ax, tick_color=INK, labelcolor=INK, axislabel_color=INK)


def _save(fig, name):
    out = os.path.join(ASSETS, name)
    fig.savefig(out, dpi=150, facecolor=SKY, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", out)


# --- the three operations, gathered onto ONE globe --------------------------
# Each op is a pair of overlapping caps at its own sky position. Positions are
# chosen off-center (and one low, near the limb) so the orthographic projection
# visibly stretches the regions toward the edge — the point of the gather beat.
# On an east-left globe +lon is to the LEFT, matching the manim corner layout
# (union → top-left, intersection → top-right, difference → bottom).
TRIO = [
    ("green", lambda r, A, B: r.add_circle(**A).add_circle(**B),         # union
     dict(lon=44, lat=33), dict(lon=32, lat=27)),                        # upper-left
    ("gold", lambda r, A, B: r.add_circle(**A).intersect_circle(**B),    # intersection
     dict(lon=-32, lat=27), dict(lon=-44, lat=33)),                      # upper-right
    ("rust", lambda r, A, B: r.add_circle(**A).subtract_circle(**B),     # difference
     dict(lon=7, lat=-47), dict(lon=-7, lat=-47)),                       # low (near limb)
]


def render_trio_globe():
    fig = plt.figure(figsize=(6.4, 6.4), facecolor=SKY)
    ax = sph.make_globe_frame(111, center_LONdeg=0, center_LATdeg=0,
                              gridcolor=GRID, gridalpha=0.7)
    _style_navy(ax)
    for colkey, build, ca, cb in TRIO:
        A = dict(**ca, radius_deg=15)
        B = dict(**cb, radius_deg=15)
        for cap in (A, B):                                  # faint A/B outlines
            sph.add_geodesic_circle(ax, cap["lon"], cap["lat"], cap["radius_deg"],
                                    facecolor="none", edgecolor=COL["gray"],
                                    lw=0.9, alpha=0.7)
        region = build(sph.CompoundRegion(ax), A, B)
        region.render(facecolor=COL[colkey], alpha=0.72)
        region.render_boundary(color=COL[colkey], linewidth=2.0)
    _save(fig, "trio_globe.png")


# --- the coda: a survey footprint built one operation at a time -------------
BOX = dict(lat_min=-12, lat_max=70, lon_min=110, lon_max=260, frame="icrs")
BAND = dict(lat_min=-25, lat_max=25, frame="galactic")
HOLE = dict(lon=180, lat=35, radius_deg=8)


def _allsky():
    fig = plt.figure(figsize=(7.4, 4.0), facecolor=SKY)
    ax = sph.make_wcs_frame(111, "AIT", center=180, fig=fig,
                            gridcolor=GRID, gridalpha=0.7)
    _style_navy(ax)
    return fig, ax


def _box_region(ax):
    return sph.CompoundRegion(ax).add_lonlat_box(**BOX)


def _box_band_region(ax):
    return _box_region(ax).subtract_frame_band(
        BAND["lat_min"], BAND["lat_max"], frame=BAND["frame"])


def _draw_footprint(region, ax):
    region.render(facecolor=COL["blue"], alpha=0.58)
    region.render_boundary(color=COL["blue"], linewidth=2.1)


def render_build():
    """The coda stages. Every stage is a full, opaque frame so the coda is a clean
    sequence of cross-fades. Each cut is shown as a *combined* frame — the current
    footprint with the cut zone laid over it in its own color (rust band / gold
    hole) — which then cross-fades to the carved result: contribution, then carve."""
    # empty frame — the coda opens here
    fig, ax = _allsky()
    _save(fig, "build_frame.png")

    # stage 1: the reachable box
    fig, ax = _allsky()
    _draw_footprint(_box_region(ax), ax)
    _save(fig, "build_box.png")

    # 1→2: the galactic band laid over the box (its rust contribution), then carved
    fig, ax = _allsky()
    _draw_footprint(_box_region(ax), ax)
    sph.add_frame_band(ax, BAND["lat_min"], BAND["lat_max"], frame=BAND["frame"],
                       facecolor=COL["rust"], alpha=0.55, zorder=5)
    _save(fig, "build_box_bandzone.png")

    fig, ax = _allsky()
    _draw_footprint(_box_band_region(ax), ax)
    _save(fig, "build_box_band.png")

    # 2→3: the exclusion hole laid over that (gold), then carved to the final shape
    fig, ax = _allsky()
    _draw_footprint(_box_band_region(ax), ax)
    sph.add_geodesic_circle(ax, HOLE["lon"], HOLE["lat"], HOLE["radius_deg"],
                            facecolor=COL["gold"], edgecolor=COL["gold"],
                            alpha=0.7, zorder=5)
    _save(fig, "build_bmb_holezone.png")

    fig, ax = _allsky()
    final = _box_band_region(ax).subtract_circle(
        HOLE["lon"], HOLE["lat"], radius_deg=HOLE["radius_deg"])
    _draw_footprint(final, ax)
    _save(fig, "build_final.png")


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    render_trio_globe()
    render_build()
