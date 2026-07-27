"""skyplothelper.plotly — projection routing across the supported set.

A single HTML with one figure per projection — same content in each
panel (galactic plane + ecliptic + scatter + constellation boundaries)
— laid out as a stack of plotly figures in one HTML so the projection
routing is visually verifiable side by side.

Covers:

- FITS WCS pseudocylindrical: AIT, MOL, SFL
- FITS WCS cylindrical: CAR, MER
- FITS WCS zenithal: SIN (centered on a non-trivial latitude to
  exercise ``lat_center``), TAN (centered on a small field)
- Custom-transform pseudocylindrical (no ``ax.wcs``): Robinson,
  Kavrayskiy, Mollweide

Each projection's figure is its own HTML section. Open the file in a
browser and scroll through the panels.

Usage
-----
    python render_plotly_projections.py            # save HTML to output/
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


def _add_overlay_set(fig):
    """Common overlay set: galactic plane, ecliptic plane, constellation
    boundaries, and a small scatter."""
    sphpl.add_great_circle(fig, frame='galactic',
                            color='dimgray', width=1.8,
                            name='Galactic plane')
    sphpl.add_great_circle(fig, frame='ecliptic',
                            color='goldenrod', width=1.4,
                            name='Ecliptic plane')
    sphpl.add_constellation_boundaries(fig, color='#bbb', opacity=0.4)
    rng = np.random.default_rng(42)
    n = 30
    lons = rng.uniform(0, 360, n)
    lats = rng.uniform(-50, 50, n)
    sphpl.add_scatter(fig, lons, lats,
                       marker=dict(size=5, color='red',
                                   line=dict(width=0.5, color='black')),
                       name='Sources')


def _build(projection, center=180, lat_center=0, theme='light',
            **fig_kwargs):
    fig = sphpl.make_figure(
        projection=projection, center=center, lat_center=lat_center,
        theme=theme, width=950, height=500,
        title=f'{projection} center={center} lat_center={lat_center}',
        **fig_kwargs,
    )
    _add_overlay_set(fig)
    return fig


def render():
    banner("plotly projection routing — one HTML per projection")
    panels = [
        # (filename_stem, projection, center, lat_center)
        ('ait',        'AIT',        180, 0),
        ('mol',        'MOL',        180, 0),
        ('sfl',        'SFL',        180, 0),
        ('car',        'CAR',        180, 0),
        ('mer',        'MER',        180, 0),
        ('sin_lat30',  'SIN',          0, 30),  # zenithal tilt
        ('robinson',   'robinson',   180, 0),
        ('kavrayskiy', 'kavrayskiy', 180, 0),
        ('mollweide',  'mollweide',  180, 0),   # custom-transform variant
    ]
    for stem, proj, c, lc in panels:
        fig = _build(proj, center=c, lat_center=lc)
        out = os.path.join(OUTPUT_DIR, f"plotly_03_projection_{stem}.html")
        fig.write_html(out, include_plotlyjs='cdn')
        print(f"  saved: {out} "
              f"({os.path.getsize(out) / 1024:.0f} kB, "
              f"{len(fig.data)} traces)")


if __name__ == "__main__":
    render()
    print("\nDone.")
