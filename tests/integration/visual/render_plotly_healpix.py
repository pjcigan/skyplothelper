"""skyplothelper.plotly.add_healpix — full-sky HEALPix tile rendering.

Three-panel HTML showing HEALPix tile coverage at low nside (kept small
to keep the HTML compact and the browser responsive), with the
wrap-edge tile splitting and per-tile hover both verifiable by
eyeball.

Panels in a vertical stack:

1. nside=4 RING-indexed map of the pixel-index value — small enough
   (192 tiles) to render snappily, big enough to show tile shapes.
2. Same data at nside=4 but with constellation boundaries overlaid,
   so the geographic context is legible.
3. Same data on a non-AIT projection (MOL with center=180) — verifies
   the projection routing reaches HEALPix correctly.

Open the resulting HTML and:

- Hover any tile to confirm tile-center RA, Dec, and value show in the
  main hover (no more streak / no "trace 1234" auto-label).
- Verify wrap-edge tiles are split (no streak across the canvas) and
  the split sub-polygons follow the curved frame silhouette.

Usage
-----
    python render_plotly_healpix.py            # save HTML to output/
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

try:
    import healpy as hp
except ImportError:
    print("skipped: healpy not installed (optional extra).")
    sys.exit(0)

from skyplothelper import plotly as sphpl


def _render_panel(projection, center, theme, title, with_constellations,
                   nside=4, colorscale='Viridis'):
    fig = sphpl.make_figure(
        projection=projection, center=center, theme=theme,
        width=1100, height=550, title=title,
    )
    npix = hp.nside2npix(nside)
    # Smooth radial pattern keyed to galactic latitude — gives the
    # visual a recognizable structure rather than a noise field.
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    vals = np.cos(theta) * np.sin(2 * phi)
    # tile_resolution='auto' (the default since the heuristic landed):
    # at nside=4 gives 15 samples/edge so polar-cap tile boundaries
    # follow the AIT silhouette curve flush, with no neighbor-color
    # slivers peeking through near the poles.
    sphpl.add_healpix(fig, vals, nside=nside,
                       colorscale=colorscale, opacity=0.85)
    if with_constellations:
        sphpl.add_constellation_boundaries(fig, color='#555', opacity=0.4)
    return fig


def render():
    banner("plotly.add_healpix — three-panel HEALPix gallery")

    # Panel 1: bare HEALPix, AIT center=180.
    fig1 = _render_panel(
        projection='AIT', center=180, theme='dark',
        title='nside=4 HEALPix on AIT(center=180) — bare tile map',
        with_constellations=False)
    out1 = os.path.join(OUTPUT_DIR, "plotly_02_healpix_bare.html")
    fig1.write_html(out1, include_plotlyjs='cdn')
    print(f"  saved: {out1} ({os.path.getsize(out1) / 1024:.0f} kB)")

    # Panel 2: HEALPix + constellation backdrop.
    fig2 = _render_panel(
        projection='AIT', center=180, theme='dark',
        title='nside=4 HEALPix on AIT(center=180) — with constellation '
              'boundaries',
        with_constellations=True)
    out2 = os.path.join(OUTPUT_DIR, "plotly_02_healpix_with_constellations.html")
    fig2.write_html(out2, include_plotlyjs='cdn')
    print(f"  saved: {out2} ({os.path.getsize(out2) / 1024:.0f} kB)")

    # Panel 3: same data on MOL projection.
    fig3 = _render_panel(
        projection='MOL', center=180, theme='light',
        title='nside=4 HEALPix on MOL(center=180) — projection routing',
        with_constellations=True)
    out3 = os.path.join(OUTPUT_DIR, "plotly_02_healpix_mollweide.html")
    fig3.write_html(out3, include_plotlyjs='cdn')
    print(f"  saved: {out3} ({os.path.getsize(out3) / 1024:.0f} kB)")


if __name__ == "__main__":
    render()
    print("\nDone.")
