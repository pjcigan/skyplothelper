"""skyplothelper.plotly — wrapper overlay gallery (one HTML, one figure).

Single AIT all-sky figure layering every plotly wrapper overlay:
``add_constellation_polygon`` (UMi + Ori + Serpens), ``add_lonlat_box``,
``add_frame_band`` (galactic ±10°, ecliptic ±5°),
``add_great_circle_band`` (custom orbital plane), and
``add_healpix_sparse`` over a small zoom region.

Hover is on for everything so each overlay's identity is verifiable
interactively.

Usage
-----
    python render_plotly_tier2.py            # save HTML to output/
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
    banner("plotly wrapper overlays — single AIT(center=180) sky")
    fig = sphpl.make_figure(
        projection='AIT', center=180, theme='dark',
        width=1200, height=650,
        title='skyplothelper.plotly wrapper overlays — constellations + '
              'lonlat box + frame bands + great-circle band + sparse healpix',
    )

    # Galactic-plane ±10° band (auto-fill, semi-transparent gray).
    sphpl.add_frame_band(fig, lat_min=-10, lat_max=10, frame='galactic',
                          fillcolor='rgba(220,220,220,0.10)',
                          color='lightgray', width=1.0,
                          name='Galactic plane ±10°', hover=True)

    # Ecliptic ±5° band (gold tint).
    sphpl.add_frame_band(fig, lat_min=-5, lat_max=5, frame='ecliptic',
                          fillcolor='rgba(255,200,80,0.12)',
                          color='goldenrod', width=1.0,
                          name='Ecliptic ±5°', hover=True)

    # Custom great-circle band — pole at (ra=120, dec=40), ±8°.
    sphpl.add_great_circle_band(fig, ra_pole=120.0, dec_pole=40.0,
                                  half_width=8,
                                  fillcolor='rgba(120,200,255,0.15)',
                                  color='deepskyblue', width=1.0,
                                  name='Custom GC band (pole 120,40)',
                                  hover=True)

    # Constellation polygons — UMi (polar, tests wrap), Ori (mid-sky),
    # Ser (two-body Serpens).
    sphpl.add_constellation_polygon(fig, 'UMi',
                                      fillcolor='rgba(80,200,255,0.30)',
                                      color='skyblue', width=1.2,
                                      name='Ursa Minor', hover=True)
    sphpl.add_constellation_polygon(fig, 'Ori',
                                      fillcolor='rgba(255,100,80,0.30)',
                                      color='salmon', width=1.2,
                                      name='Orion', hover=True)
    sphpl.add_constellation_polygon(fig, 'Ser',
                                      fillcolor='rgba(180,255,120,0.30)',
                                      color='lightgreen', width=1.2,
                                      name='Serpens (Caput + Cauda)',
                                      hover=True)

    # Cross-frame lonlat box — galactic 30 < l < 90, -20 < b < 20.
    sphpl.add_lonlat_box(fig, lat_min=-20, lat_max=20,
                          lon_min=30, lon_max=90,
                          frame='galactic',
                          fillcolor='rgba(200,100,200,0.25)',
                          color='violet', width=1.2,
                          name='Galactic box 30<l<90, -20<b<20',
                          hover=True)

    # Sparse HEALPix patch — small zoom around RA=60, Dec=0.
    import healpy as hp
    nside = 16
    ra0, dec0 = 60.0, 0.0
    vec = hp.ang2vec(ra0, dec0, lonlat=True)
    pix = hp.query_disc(nside, vec, np.radians(8.0))
    vals = np.cos(np.radians(np.linspace(0, 180, len(pix))))
    sphpl.add_healpix_sparse(fig, pix, vals, nside=nside,
                               tile_resolution=2,
                               colorscale='Plasma',
                               opacity=0.85)

    # Constellation backdrop.
    sphpl.add_constellation_boundaries(fig, color='#555', opacity=0.4)

    # Projection frame edge — the AIT(center=180) ellipse silhouette.
    sphpl.add_frame_edge(fig, color='#888', width=1.2)

    out = os.path.join(OUTPUT_DIR, "plotly_03_tier2_wrappers.html")
    fig.write_html(out, include_plotlyjs='cdn')
    size_kb = os.path.getsize(out) / 1024
    print(f"  saved: {out} ({size_kb:.0f} kB, {len(fig.data)} traces)")


if __name__ == "__main__":
    render()
    print("\nDone.")
