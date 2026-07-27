"""Render the *real* skyplothelper frames that the projection manim demo
cross-fades into.

This is the sph side of the demo A pipeline (see
``.claude/MANIM_DEMO_BRIEFS.md`` §A and §4). It draws no animation — it just
produces the genuine sph PNGs that ``projection.py`` imports as
``ImageMobject``s and dissolves the 3-D geometry into. The division of labor is
the whole point: skyplothelper is the source of scientific truth (these charts),
manim is only the camera and typographer.

Two assets, both on the shared navy sky canvas so the cross-fade is seamless:

* ``assets/tan_orion.png``   — a genuine gnomonic (``TAN``) chart of Orion, the
  same field the 3-D ray-casting beats paint onto the tangent plane. Real
  Hipparcos stars in their perceived colors, with Orion's figure drawn.
* ``assets/allsky_aitoff.png`` — the NOIRLab all-sky panorama reprojected onto a
  Galactic Aitoff frame (reuses tutorial #2 §5's own data + machinery), for the
  "different projection point => different map" coda.

Run in the astropy env (``cenv``), from the repo root::

    python docs/manim/make_projection_assets.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import skyplothelper as sph

# The manim scenes render on this navy (see the brief); the sph PNGs must share
# it so the dissolve has no visible seam. It is intentionally a lifted navy, not
# the near-black of the constellations notebook's own night canvas.
SKY = "#16203A"
NIGHT = sph.ANNOTATION_PALETTES["night"]
WHEAT = "#CBB88C"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "examples", "data")
ASSETS = os.path.join(HERE, "assets")

# Orion: instantly recognizable, and wide enough (~25deg) that gnomonic
# distortion is visible toward the edges. Centered between Betelgeuse and Rigel.
ORION_CENTER = (82.5, 1.0)
ORION_FOV = 34.0

# The all-sky coda frames are centered on Orion's RA so Orion sits dead-center
# in every frame (TAN chart, bare grid, draped raster) — it never moves, which
# makes it the visual anchor as the frame zooms local->global and the raster
# drapes on. ORION_FIELD_BOX crops the field-star scatter to Orion's patch (the
# same content as the TAN chart) so the all-sky shows a localized cluster, not
# the whole naked-eye sky.
ALLSKY_CENTER = ORION_CENTER[0]
ORION_FIELD_BOX = (64.0, 101.0, -17.0, 19.0)


def star_sizes(vmag, scale=1.0, mlim=6.5):
    """Classic star-chart scaling: marker area grows as (m_lim - V)**2."""
    return scale * (mlim - np.asarray(vmag)) ** 2


def _plot_orion_field(ax, crop_box=None, size_scale=1.1):
    """Scatter the Hipparcos field stars in their perceived colors and draw
    Orion's asterism — the shared overlay that links the TAN chart, the bare
    all-sky grid, and the draped raster. ``crop_box`` (ra0, ra1, dec0, dec1)
    limits the scatter to Orion's patch on a full-sky frame; on the TAN chart
    the field of view crops it already, so leave it None."""
    stars = pd.read_csv(os.path.join(DATA, "hipparcos_bright_pm.csv"))
    if crop_box is not None:
        ra0, ra1, dec0, dec1 = crop_box
        stars = stars[(stars.RAICRS > ra0) & (stars.RAICRS < ra1)
                      & (stars.DEICRS > dec0) & (stars.DEICRS < dec1)]
    tr = ax.get_transform("world")
    ax.scatter(stars.RAICRS, stars.DEICRS, s=star_sizes(stars.Vmag, size_scale),
               c=sph.bv_to_rgb(stars.BV.fillna(0.6)), lw=0, alpha=0.95,
               zorder=3, transform=tr)
    # Only Orion's own figure — other constellations would clutter the field.
    sph.add_constellation_lines(ax, constellations=["Ori"], color=WHEAT,
                                lw=1.1, alpha=0.6, zorder=2)


def render_tan_orion():
    """A real gnomonic chart of Orion — the tangent-plane payoff of the demo."""
    fig = plt.figure(figsize=(6.4, 6.4), facecolor=SKY)
    ax = sph.make_wcs_frame(111, projection="TAN", center=ORION_CENTER,
                            fov_deg=ORION_FOV, fig=fig, gridcolor=NIGHT["grid"])
    ax.set_facecolor(SKY)
    fig.canvas.draw()
    sph.style_wcs_axes(ax, tick_color=NIGHT["stars"], labelcolor=NIGHT["stars"],
                       axislabel_color=NIGHT["stars"])
    _plot_orion_field(ax, size_scale=1.1)   # FOV crops the field for us
    ax.set_title("")

    # PNG: the chart is sharp lines + text on flat navy, which JPG would fuzz.
    out = os.path.join(ASSETS, "tan_orion.png")
    fig.savefig(out, dpi=150, facecolor=SKY, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", out)


def emit_orion_stars():
    """Write the bright Orion stars as plain data for the manim scene to read.

    sph does the science here — the perceived colors come from ``bv_to_rgb`` —
    so the manim env needs no skyplothelper import. The scene reads only unit
    vectors, hex colors, and marker sizes, and does its own (numpy) gnomonic
    geometry. Keeps the boundary crisp: sph computes, manim renders.
    """
    stars = pd.read_csv(os.path.join(DATA, "hipparcos_bright_pm.csv"))
    box = ((stars.RAICRS > 68) & (stars.RAICRS < 98)
           & (stars.DEICRS > -14) & (stars.DEICRS < 16))
    # The brightest ~50 keep the 3-D beats legible (the PNG shows the full field).
    bright = stars[box].sort_values("Vmag").head(50).reset_index(drop=True)

    ra = np.radians(bright.RAICRS.to_numpy())
    dec = np.radians(bright.DEICRS.to_numpy())
    ux = np.cos(dec) * np.cos(ra)
    uy = np.cos(dec) * np.sin(ra)
    uz = np.sin(dec)
    rgb = sph.bv_to_rgb(bright.BV.fillna(0.6).to_numpy())
    # Real distances from the Hipparcos parallax (mas -> pc), so the opening
    # "stars at every distance" beat is honest. Tiny/negative parallaxes are
    # too uncertain to invert, so those get no distance (scene uses a fallback).
    plx = bright.Plx.to_numpy()
    dist_pc = np.where(plx > 0.5, 1000.0 / np.where(plx > 0.5, plx, 1.0), np.nan)

    def hexcolor(row):
        return "#{:02X}{:02X}{:02X}".format(*(int(round(c * 255)) for c in row))

    payload = {
        "center_radec": list(ORION_CENTER),
        "fov_deg": ORION_FOV,
        "stars": [
            {"u": [float(ux[i]), float(uy[i]), float(uz[i])],
             "color": hexcolor(rgb[i]),
             "size": float(star_sizes(bright.Vmag.iloc[i], 1.0)),
             "dist_pc": (None if np.isnan(dist_pc[i])
                         else round(float(dist_pc[i]), 1))}
            for i in range(len(bright))
        ],
    }
    out = os.path.join(ASSETS, "orion_stars.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print("wrote", out, f"({len(bright)} stars)")


def _allsky_orion_frame():
    """A shared ICRS all-sky Aitoff, centered on Orion and styled like the night
    charts. The grid and raster codas use the *identical* frame (same center,
    size, layout) so Orion is pixel-aligned between them — the raster then drapes
    on without Orion shifting. Saved without a tight bbox so both are exactly the
    same pixel dimensions."""
    fig, ax = sph.allsky_figure(projection="AIT", center=ALLSKY_CENTER,
                                frame="ICRS", npix=(1200, 600), figsize=(9.6, 5.0))
    fig.set_facecolor(SKY)
    ax.set_facecolor(SKY)
    fig.canvas.draw()
    sph.style_wcs_axes(ax, tick_color=NIGHT["stars"], labelcolor=NIGHT["stars"],
                       axislabel_color=NIGHT["stars"])
    # Drop the axis labels: on an Orion-centered oval the "RA" label sits at the
    # bottom-center — right on top of Orion.
    ax.coords[0].set_axislabel("")
    ax.coords[1].set_axislabel("")
    return fig, ax


def render_allsky_orion_grid():
    """A bare all-sky grid (no raster) with Orion in its place — the link frame
    the TAN chart zooms out to before the sky image is draped on."""
    fig, ax = _allsky_orion_frame()
    _plot_orion_field(ax, crop_box=ORION_FIELD_BOX, size_scale=0.5)
    ax.set_title("")
    out = os.path.join(ASSETS, "allsky_orion_grid.png")
    fig.savefig(out, dpi=150, facecolor=SKY)
    plt.close(fig)
    print("wrote", out)


def render_allsky_orion_raster():
    """The same all-sky frame with the NOIRLab panorama draped on (reprojected
    from its galactic layout into this ICRS frame, tutorial #2 §5/§6 machinery),
    Orion still overlaid on top so it stays the anchor."""
    fig, ax = _allsky_orion_frame()
    pano = os.path.join(DATA, "Allsky_noirlab2430b_1280x640.jpg")
    img, hdr = sph.load_sky_image(pano, frame="galactic", center=0)
    ax.imshow(sph.reproject_background(img, hdr, ax), zorder=-10)
    _plot_orion_field(ax, crop_box=ORION_FIELD_BOX, size_scale=0.5)
    ax.set_title("")
    # JPG: a full-frame Milky Way photo compresses far smaller than PNG with no
    # visible loss (and this is a committed input — the source panorama is
    # local-only, so it can't be regenerated from a clean checkout).
    out = os.path.join(ASSETS, "allsky_orion_raster.jpg")
    fig.savefig(out, dpi=150, facecolor=SKY, pil_kwargs={"quality": 88})
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    render_tan_orion()
    emit_orion_stars()
    render_allsky_orion_grid()
    render_allsky_orion_raster()
