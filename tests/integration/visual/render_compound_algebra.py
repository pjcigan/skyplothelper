"""Render CompoundRegion set-algebra demonstrations for visual eyeballing.

Produces:
  - compound_01_union.png       — A ∪ B (two overlapping circles, unioned)
  - compound_02_intersection.png — A ∩ B (only the lens)
  - compound_03_subtract.png    — A − B (cresent)
  - compound_04_xor.png         — A XOR B (lens removed)
  - compound_05_complement.png  — complement of (A ∪ B)
  - compound_06_expand_contract.png — same shape with .expand() vs .contract()
  - compound_07_kitchen_sink.png — chained ops (multi-shape, mixed types)
  - compound_08_contains_points.png — vectorised inside/outside test on
                                       random sky points
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from _common import banner, save_or_show

from skyplothelper.geometry.compound import CompoundRegion
from skyplothelper.overlays.planes import add_plane_overlay
from skyplothelper.wcs_frame import make_wcs_frame

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


def _allsky():
    fig = plt.figure(figsize=(11, 5.5))
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    return fig, ax


def _build_two_circle_panel(op, title):
    """Build (and return) the standard "two circles + galactic plane" figure."""
    fig, ax = _allsky()
    R = CompoundRegion(ax).add_circle(170, 30, radius_deg=20)
    op(R)
    R.render(facecolor="C0", edgecolor="navy", alpha=0.4, lw=1.0)
    add_plane_overlay(ax, plane="galactic", color="0.4", lw=0.8)
    ax.set_title(f"CompoundRegion: {title}  (area_frac={R.area_frac:.3f})",
                 fontsize=11)
    return fig


@_panel("compound_01_union")
def render_union():
    return _build_two_circle_panel(
        lambda R: R.add_circle(195, 30, radius_deg=20),
        "A ∪ B (add_circle then add_circle)",
    )


@_panel("compound_02_intersection")
def render_intersection():
    return _build_two_circle_panel(
        lambda R: R.intersect_circle(195, 30, radius_deg=20),
        "A ∩ B (intersect_circle)",
    )


@_panel("compound_03_subtract")
def render_subtract():
    return _build_two_circle_panel(
        lambda R: R.subtract_circle(195, 30, radius_deg=20),
        "A − B (subtract_circle)",
    )


@_panel("compound_04_xor")
def render_xor():
    return _build_two_circle_panel(
        lambda R: R.xor_circle(195, 30, radius_deg=20),
        "A XOR B (xor_circle)",
    )


@_panel("compound_05_complement")
def render_complement():
    fig, ax = _allsky()
    R = (CompoundRegion(ax)
         .add_circle(170, 30, radius_deg=20)
         .add_circle(195, 30, radius_deg=20)
         .complement())
    R.render(facecolor="C3", edgecolor="darkred", alpha=0.4, lw=1.0)
    add_plane_overlay(ax, plane="galactic", color="0.4", lw=0.8)
    ax.set_title(f"CompoundRegion: complement of (A ∪ B)  "
                 f"(area_frac={R.area_frac:.3f})", fontsize=11)
    return fig


@_panel("compound_06_expand_contract")
def render_expand_contract():
    fig = plt.figure(figsize=(15, 5.5))
    for col, (op_name, kw) in enumerate([
        ("original (radius=15°)", None),
        ("expand(5°)", "expand"),
        ("contract(5°)", "contract"),
    ], start=1):
        ax = make_wcs_frame((1, 3, col), projection="AIT", center=180, fig=fig)
        fig.canvas.draw()
        R = CompoundRegion(ax).add_circle(180, 30, radius_deg=15)
        if kw == "expand":
            R.expand(5.0)
        elif kw == "contract":
            R.contract(5.0)
        R.render(facecolor="C2", edgecolor="darkgreen", alpha=0.4, lw=1.0)
        ax.set_title(f"{op_name}  (area_frac={R.area_frac:.3f})",
                     fontsize=10)
    fig.suptitle("CompoundRegion.expand() / .contract() — buffer the boundary",
                 fontsize=12)
    fig.subplots_adjust(top=0.88, wspace=0.25)
    return fig


@_panel("compound_08_contains_points")
def render_contains_points():
    """CompoundRegion.contains_points — vectorised inside/outside test.

    Builds a moderately complex region (two unioned circles minus a
    third), scatters 800 random sky points, and colors them by
    ``R.contains_points(ra, dec)``. The point color (green=inside,
    red=outside) should match the rendered region boundary at a
    glance.
    """
    rng = np.random.default_rng(7)
    fig, ax = _allsky()
    R = (CompoundRegion(ax)
         .add_circle(150, 20, radius_deg=25)
         .add_circle(210, 20, radius_deg=25)
         .subtract_circle(180, 20, radius_deg=10))
    R.render(facecolor="C0", edgecolor="navy", alpha=0.25, lw=1.0)

    # 800 uniform-on-sphere random points
    n = 800
    u = rng.uniform(0, 1, n)
    v = rng.uniform(0, 1, n)
    ra = 360.0 * u
    dec = np.degrees(np.arcsin(2 * v - 1))

    inside = R.contains_points(ra, dec)
    ax.scatter(ra[~inside], dec[~inside],
               transform=ax.get_transform("world"),
               s=8, c="C3", alpha=0.6, label=f"outside ({(~inside).sum()})")
    ax.scatter(ra[inside], dec[inside],
               transform=ax.get_transform("world"),
               s=14, c="C2", alpha=0.85,
               edgecolor="darkgreen", linewidth=0.4,
               label=f"inside ({inside.sum()})")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax.set_title(
        "CompoundRegion.contains_points — 800 uniform points\n"
        "two circles ∪, minus a central circle  "
        f"(area_frac={R.area_frac:.3f})", fontsize=10)
    return fig


@_panel("compound_07_kitchen_sink")
def render_kitchen_sink():
    """Chained operations: galactic plane band − two avoidance circles +
    a polygon footprint, all combined."""
    fig, ax = _allsky()
    R = (CompoundRegion(ax)
         # Start with a galactic-plane band
         .add_frame_band(-10, 10, frame="galactic")
         # Subtract two avoidance circles (bright stars / problematic regions)
         .subtract_circle(266.4, -28.9, radius_deg=8)
         .subtract_circle(83.6, 22.0, radius_deg=6)
         # Union with a survey-like rectangle
         .add_rectangle(60, -40, width=40, height=20, angle=0)
         # Slight margin
         .expand(1.0))
    R.render(facecolor="C5", edgecolor="indigo", alpha=0.4, lw=1.0)
    add_plane_overlay(ax, plane="galactic", color="0.5", lw=0.8)
    ax.set_title("Chained: galactic_band − 2 circles ∪ rectangle, then "
                 f"expand(1°)  (area_frac={R.area_frac:.3f})", fontsize=10)
    return fig


@_panel("compound_09_from_points_hull")
def render_from_points_hull():
    """CompoundRegion.from_points: build a footprint region from a scatter of
    sources via a convex (blue) or concave (red) hull."""
    rng = np.random.RandomState(3)
    t = rng.uniform(0.2 * np.pi, 1.8 * np.pi, 500)
    r = 12 + rng.normal(0, 1.2, 500)
    lon = 60 + r * np.cos(t) / np.cos(np.radians(20))
    lat = 20 + r * np.sin(t)
    fig = plt.figure(figsize=(7, 5))
    ax = make_wcs_frame(111, "TAN", frame="ICRS", center=(60, 20),
                        fov_deg=60, fig=fig)
    tr = ax.get_transform("world")
    ax.scatter(lon, lat, s=4, c="0.5", transform=tr, zorder=2)
    CompoundRegion.from_points(ax, lon, lat, hull="convex").render(
        facecolor="none", edgecolor="C0", lw=1.6)
    CompoundRegion.from_points(ax, lon, lat, hull="concave", ratio=0.15).render(
        facecolor="none", edgecolor="C3", lw=1.6)
    ax.set_title("from_points: convex (blue) vs concave (red) hull",
                 fontsize=11)
    return fig


@_panel("compound_10_from_healpix_mask")
def render_from_healpix_mask():
    """Round-trip: a region -> HEALPix mask -> region reconstructed from the
    mask (union of the True pixels)."""
    fig, ax = _allsky()
    src = CompoundRegion(ax).add_circle(160, 20, 25).add_circle(210, -15, 18)
    reg = CompoundRegion.from_healpix_mask(ax, src.to_healpix_mask(32))
    reg.render(facecolor="C2", edgecolor="k", alpha=0.5)
    ax.set_title("region from a HEALPix mask (nside 32, union of True pixels)",
                 fontsize=11)
    return fig


@_panel("compound_11_region_set_ops")
def render_region_set_ops():
    """Region-to-region set algebra: build two independent regions A and B,
    then show A∪B / A∩B / A−B / A XOR B via the new union/intersection/
    difference/symmetric_difference methods."""
    ops = [("A ∪ B", "union"), ("A ∩ B", "intersection"),
           ("A − B", "difference"), ("A XOR B", "symmetric_difference")]
    fig = plt.figure(figsize=(15, 4))
    for col, (label, meth) in enumerate(ops, start=1):
        ax = make_wcs_frame((1, 4, col), projection="AIT", center=180, fig=fig)
        fig.canvas.draw()
        A = CompoundRegion(ax).add_circle(165, 15, radius_deg=28)
        B = CompoundRegion(ax).add_circle(195, -15, radius_deg=28)
        getattr(A, meth)(B).render(facecolor="C0", edgecolor="navy", alpha=0.45)
        ax.set_title(f"{label}  (area_frac={A.area_frac:.3f})", fontsize=10)
    fig.suptitle("CompoundRegion region-to-region set algebra", fontsize=12)
    fig.subplots_adjust(top=0.86, wspace=0.3)
    return fig


def main():
    banner("CompoundRegion algebra — gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
