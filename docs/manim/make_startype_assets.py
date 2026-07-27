"""Emit the data + the real sph render that the star-type-morph demo (E) uses.

This is the sph side of demo E.
It draws no animation. It produces:

* ``assets/startype_stops.json`` — a dense log-temperature sweep from O to M.
  Every stop's **color is a real ``sph.teff_to_rgb`` value** (the whole point of
  the demo is that the animation shows the function's actual output — the Sun,
  at G, comes out white, not green), plus a main-sequence marker size and the
  spectral class. The manim scene reads only this JSON, so the manim env needs
  no skyplothelper import. sph computes; manim tweens.
* ``assets/startype_allsky.jpg`` — the notebook's own perceived-color all-sky
  chart, rendered on the shared navy canvas, for the "…and here they all are on
  the real sky" coda.

Run in the astropy env (``cenv``), from the repo root::

    python docs/manim/make_startype_assets.py
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

# Shared manim canvas (see the brief) — the coda still must share it so the
# cross-fade from the animated marker into the real chart has no seam.
SKY = "#16203A"
NIGHT = sph.ANNOTATION_PALETTES["night"]
WHEAT = "#CBB88C"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "examples", "data")
ASSETS = os.path.join(HERE, "assets")

# Main-sequence anchors: (T_eff [K], radius [R_sun]). The marker size tracks
# main-sequence radius, so hot O stars are big and cool M dwarfs are small —
# the honest luminosity-class story (a red *giant* like Antares is a different
# beast; this morph is the main sequence). Interpolated in log-log between these.
MS_ANCHORS = [
    (42000, 12.0), (30000, 9.0), (20000, 6.0), (15000, 4.3), (10000, 2.7),
    (8000, 1.8), (7000, 1.45), (6000, 1.10), (5772, 1.00), (5200, 0.90),
    (4500, 0.78), (3800, 0.63), (3300, 0.47), (3000, 0.38),
]

# Spectral-class boundaries (lower T_eff edge of each class, K) and a one-word
# color cue for the caption.
CLASS_EDGES = [
    ("O", 30000, "blue"), ("B", 10000, "blue-white"), ("A", 7500, "white"),
    ("F", 6000, "yellow-white"), ("G", 5200, "yellow-white"), ("K", 3700, "orange"),
    ("M", 0, "orange-red"),
]


def ms_radius(teff):
    """Main-sequence radius (R_sun) at T_eff, log-log-interpolated."""
    ta = np.array([a[0] for a in MS_ANCHORS][::-1], float)
    ra = np.array([a[1] for a in MS_ANCHORS][::-1], float)
    return float(np.exp(np.interp(np.log(teff), np.log(ta), np.log(ra))))


def spectral_class(teff):
    for cls, edge, cue in CLASS_EDGES:
        if teff >= edge:
            return cls, cue
    return "M", "red"


def emit_startype_stops():
    """A smooth O->M temperature sweep, colored by the real sph function."""
    # Log-uniform in T_eff so the perceptually busy cool end gets enough samples.
    # Start at a real O5 (~40,000 K) so class O gets its own segment on the rail
    # (the eye can't tell 40,000 from 25,000 K apart — both read blue-white — so
    # teff_to_rgb clamps there, and the hot end of the morph shrinks at ~constant
    # blue before the color warms through A/F/G).
    teffs = np.exp(np.linspace(np.log(40000), np.log(3100), 120))
    # Full honest tristimulus color (saturation=1.0) — matches the notebook and
    # the reference computation of Harre & Heller (2021).
    rgb = sph.teff_to_rgb(teffs, saturation=1.0)

    def hexcolor(row):
        return "#{:02X}{:02X}{:02X}".format(*(int(round(c * 255)) for c in row))

    stops = []
    for i, t in enumerate(teffs):
        cls, cue = spectral_class(t)
        stops.append({
            "teff": int(round(t)),
            "color": hexcolor(rgb[i]),
            "r_sun": round(ms_radius(t), 3),
            "cls": cls,
            "cue": cue,
        })
    # A luminance-even wavelength->color LUT for the spectrum fill. mpl's
    # 'rainbow' was the most luminance-even of the candidates (no dark ends,
    # no magenta), mapped violet(380)->red(700) across the visible band.
    import matplotlib
    vis = np.linspace(380.0, 700.0, 48)
    cmap = matplotlib.colormaps["rainbow"]
    wl_colors = [[round(float(w), 1), hexcolor(cmap((w - 380.0) / 320.0)[:3])]
                 for w in vis]
    # The Sun's index — the scene pauses here for the white-not-green beat.
    sun_i = int(np.argmin(np.abs(teffs - 5772)))
    payload = {"stops": stops, "sun_index": sun_i,
               "classes": [c[0] for c in CLASS_EDGES], "wl_colors": wl_colors}
    out = os.path.join(ASSETS, "startype_stops.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print("wrote", out, f"({len(stops)} stops, sun at {sun_i})")


def render_allsky_coda():
    """The notebook's perceived-color all-sky, on the shared navy canvas."""
    stars = pd.read_csv(os.path.join(DATA, "hipparcos_bright_pm.csv"))
    sizes = 0.6 * (6.5 - stars.Vmag.to_numpy()) ** 2

    fig = plt.figure(figsize=(11.0, 5.6), facecolor=SKY)
    ax = sph.make_wcs_frame(111, projection="AIT", center=180, fig=fig,
                            gridcolor=NIGHT["grid"])
    ax.set_facecolor(SKY)
    fig.canvas.draw()
    sph.style_wcs_axes(ax, tick_color=NIGHT["stars"], labelcolor=NIGHT["stars"],
                       axislabel_color=NIGHT["stars"])
    ax.scatter(stars.RAICRS, stars.DEICRS, s=sizes,
               c=sph.bv_to_rgb(stars.BV.fillna(0.6), saturation=1.0), lw=0,
               alpha=0.95, zorder=3, transform=ax.get_transform("world"))
    sph.add_constellation_boundaries(ax, color=NIGHT["grid"], lw=0.5, alpha=0.9)
    sph.add_constellation_lines(ax, rank_max=1, color=WHEAT, lw=0.9, alpha=0.5)
    ax.set_title("")

    out = os.path.join(ASSETS, "startype_allsky.jpg")
    fig.savefig(out, dpi=150, facecolor=SKY, bbox_inches="tight", pad_inches=0.05,
                pil_kwargs={"quality": 90})
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    emit_startype_stops()
    render_allsky_coda()
