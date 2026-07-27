"""skyplothelper.plotly.add_sky_vectors — vector-field demo.

Single AIT all-sky figure layering three independent vector fields so
the rendering surface of ``add_sky_vectors`` is exercised across the
common use cases:

1. A uniform-color mock proper-motion field on a regular sky grid
   (one arrow every ~30° in RA × 20° in Dec) — exposes per-arrow
   rotation, cos(δ) handling, and the "median-arrow-spans-2°"
   auto-scale.

2. A VSH-style residual field with color-by-magnitude + colorbar —
   exposes the magnitude-coded array path and the colorbar
   attachment.

3. A small cluster of catalog displacements (ICRS3 vs Gaia-style),
   with raw ``ra2 - ra1`` / ``dec2 - dec1`` differences (i.e. not yet
   cosδ-scaled) — exposes ``cos_dec=False``.

Hover is on for everything so each arrowhead reveals its anchor
RA/Dec, vector magnitude, and on-sky position angle.

Usage
-----
    python render_plotly_sky_vectors.py            # save HTML to output/
"""

import os
import sys

import numpy as np
from _common import OUTPUT_DIR, banner

try:
    import plotly  # noqa: F401
except ImportError:
    print("skipped: plotly not installed (optional extra).")
    sys.exit(0)

from skyplothelper import plotly as sphpl


def render():
    banner("plotly.add_sky_vectors — three-field AIT(center=180) demo")
    fig = sphpl.make_figure(
        projection='AIT', center=180, theme='dark',
        width=1200, height=650,
        title='skyplothelper.plotly add_sky_vectors — '
              'mock PM grid + VSH residual field + catalog deltas',
    )

    # Constellation backdrop for geographic context.
    sphpl.add_constellation_boundaries(fig, color='#444', opacity=0.4)
    sphpl.add_great_circle(fig, frame='galactic',
                            color='#bbb', width=1.2,
                            name='Galactic plane', hover=True)

    rng = np.random.default_rng(11)

    # Field 1: uniform-color PM grid. 12 × 7 grid of arrows.
    ra_grid, dec_grid = np.meshgrid(
        np.linspace(15, 345, 12), np.linspace(-60, 60, 7))
    ra_grid = ra_grid.ravel()
    dec_grid = dec_grid.ravel()
    # Mock proper-motion field — small swirl plus noise.
    swirl_angle = np.radians(ra_grid - 180.0)
    pmra = 30.0 * np.cos(swirl_angle) + rng.normal(0, 6, size=ra_grid.size)
    pmdec = 30.0 * np.sin(swirl_angle) + rng.normal(0, 6, size=ra_grid.size)
    sphpl.add_sky_vectors(
        fig, ra_grid, dec_grid, pmra, pmdec,
        scale='auto', units='mas',
        auto_target_deg=6.0,
        color='#7cd3ff', opacity=0.9,
        width=1.2, arrow_size=7,
        name='Mock PM (uniform color)', hover=True)

    # Field 2: VSH-style residuals, magnitude-coded + colorbar.
    n_res = 60
    ra_res = rng.uniform(0, 360, n_res)
    dec_res = rng.uniform(-50, 50, n_res)
    # Make magnitudes anti-correlate with galactic latitude
    # for a recognizable pattern.
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    sc = SkyCoord(ra_res * u.deg, dec_res * u.deg, frame='icrs')
    gal_b = sc.galactic.b.deg
    base = 80.0 + 30.0 * np.cos(np.radians(gal_b))
    dra_res = base * np.cos(np.radians(ra_res)) + rng.normal(0, 10, n_res)
    ddec_res = base * np.sin(np.radians(ra_res)) + rng.normal(0, 10, n_res)
    sphpl.add_sky_vectors(
        fig, ra_res, dec_res, dra_res, ddec_res,
        scale='auto', units='uas',
        auto_target_deg=8.0,
        color_by_magnitude=True, cmap='Plasma',
        shaft_color='match',
        add_colorbar=True, cbar_title='|residual| (μas)',
        opacity=0.95, width=1.4, arrow_size=9,
        name='VSH residuals', hover=True)

    # Field 3: catalog displacement deltas — raw (ra2-ra1) / (dec2-dec1)
    # arcsec values, cos(dec)-uncorrected.
    n_cat = 18
    ra_cat = rng.uniform(60, 130, n_cat)
    dec_cat = rng.uniform(20, 65, n_cat)
    dra_cat = rng.normal(0, 6, n_cat)        # arcsec, raw
    ddec_cat = rng.normal(0, 6, n_cat)
    sphpl.add_sky_vectors(
        fig, ra_cat, dec_cat, dra_cat, ddec_cat,
        scale='auto', units='arcsec',
        auto_target_deg=4.0,
        cos_dec=False,
        color='gold', opacity=0.95,
        width=1.4, arrow_size=9,
        name='Catalog delta (cos δ off)', hover=True)

    out = os.path.join(OUTPUT_DIR, "plotly_04_sky_vectors.html")
    fig.write_html(out, include_plotlyjs='cdn')
    size_kb = os.path.getsize(out) / 1024
    print(f"  saved: {out} ({size_kb:.0f} kB, {len(fig.data)} traces)")


if __name__ == "__main__":
    render()
    print("\nDone.")
