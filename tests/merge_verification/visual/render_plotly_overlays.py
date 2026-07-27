"""skyplothelper.plotly — core overlay gallery (two stacked AIT panels).

Two AIT all-sky figures stacked in one HTML file, each layering every
core plotly overlay primitive plus the constellation backdrop and
scattered markers. The top panel uses ``center=180``; the bottom panel
uses ``center=90`` — same overlay content, different wrap edge —
verifying that wrap-edge splitting, frame-curve densification, and
projection metadata threading all work correctly for non-180° centers
too.

What's worth eyeballing in each panel:

- Galactic plane (white) sweeps through the sky as a continuous curve
  — different wrap-edge break in each panel.
- Ecliptic plane (gold) + ±23.4° tropics as small-circle parallels.
- Geodesic circle around the Orion nebula (red fill).
- Survey-footprint polygon (cyan fill).
- "Wrap-test" lime-green polygon — straddles lon=0 (so it wrap-splits
  on the center=180 panel and renders unsplit on the center=90 panel)
  AND a second magenta polygon straddles lon=270 (which wrap-splits on
  the center=90 panel and renders unsplit on the center=180 panel).
- Constellation boundaries faint in the background.
- Random scatter points with RA / Dec hover.

Usage
-----
    python render_plotly_overlays.py            # save HTML to output/
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


def _build_panel(center, title):
    """Build a single AIT(center=...) figure with the full core-overlay
    suite. Identical content across centers — only the projection
    center changes — so any wrap-edge or projection-metadata issue
    surfaces by comparing the two panels."""
    fig = sphpl.make_figure(
        projection='AIT', center=center, theme='dark',
        width=1200, height=600,
        title=title,
    )

    # Galactic plane — hover=True shows name + RA/Dec at the nearest sample.
    sphpl.add_great_circle(fig, frame='galactic',
                            color='white', width=2.0,
                            name='Galactic plane', hover=True)

    # Ecliptic plane + tropics; parallels get offset-annotated hover names.
    sphpl.add_plane_overlay(fig, plane='ecliptic',
                             parallels=[-23.4, 23.4],
                             color='goldenrod', width=1.6,
                             parallel_opacity=0.5, parallel_width=1.0,
                             name='Ecliptic plane', hover=True)

    # Geodesic circle around Orion nebula (M42) — hover=True on a filled
    # disc shows name + center + radius anywhere over the fill.
    sphpl.add_geodesic_circle(fig, lon=83.6, lat=-5.4, radius_deg=8,
                               fill=True,
                               fillcolor='rgba(255,90,80,0.30)',
                               color='red', width=2.0,
                               name='Orion nebula region (r=8°)',
                               hover=True)

    # Survey footprint polygon (rectangular box in RA / Dec)
    sphpl.add_spherical_polygon(
        fig,
        lons=[100, 160, 160, 100], lats=[20, 20, 50, 50],
        fillcolor='rgba(80,200,255,0.25)',
        color='cyan', width=1.5, name='Footprint A',
        hover='<b>Footprint A</b><br>survey: example<br>area: ~1800°²')

    # Wrap-test polygon @ lon=0 — straddles lon=0, so wrap-splits on
    # the center=180 panel and renders as a normal polygon on the
    # center=90 panel.
    sphpl.add_spherical_polygon(
        fig,
        lons=[350, 10, 10, 350], lats=[-30, -30, -10, -10],
        fillcolor='rgba(180,255,120,0.30)',
        color='lightgreen', width=1.5,
        name='Wrap-test polygon @ lon≈0',
        hover=True)

    # Wrap-test polygon @ lon=270 — straddles lon=270, so wrap-splits
    # on the center=90 panel and renders as a normal polygon on the
    # center=180 panel.
    sphpl.add_spherical_polygon(
        fig,
        lons=[260, 280, 280, 260], lats=[10, 10, 40, 40],
        fillcolor='rgba(255,100,200,0.30)',
        color='magenta', width=1.5,
        name='Wrap-test polygon @ lon≈270',
        hover=True)

    # Constellation backdrop
    sphpl.add_constellation_boundaries(fig, color='#555', opacity=0.4)

    # Random sources with hover
    rng = np.random.default_rng(7)
    n = 40
    lons = rng.uniform(0, 360, n)
    lats = rng.uniform(-60, 60, n)
    sphpl.add_scatter(fig, lons, lats,
                       marker=dict(size=6, color='gold',
                                   line=dict(width=0.5, color='black')),
                       name='Random sources')

    # Coordinate tick labels along the canvas edges. Theme-aware
    # default color keeps them legible on the dark background.
    # lat_exterior=True places the latitude labels just outside the
    # frame edge (conventional axis-tick look).
    sphpl.add_coord_labels(fig, lon_spacing=30, lat_spacing=15,
                           lat_exterior=True)

    # Rulers — one chord + one geodesic + one tilted with arrow caps,
    # so the pixel-stable tick/label work shows up across projection
    # distortion.
    sphpl.add_ruler(fig, lon1=20, lat1=-30, lon2=70, lat2=-30,
                     n_ticks=6, title='Chord ruler',
                     endcap_style='tick', color='white')
    sphpl.add_ruler(fig, lon1=210, lat1=40, lon2=320, lat2=10,
                     geodesic=True, n_ticks=5,
                     endcap_style='arrow', title='Geodesic ruler',
                     color='gold')
    sphpl.add_ruler(fig, lon1=140, lat1=-55, lon2=185, lat2=-45,
                     n_ticks=4, tick_side='left',
                     label_rotation='horizontal',
                     color='lightgreen')

    # Reticles — one per style, scattered around the figure so the
    # pixel-stable mark + optional label work shows up across
    # projection distortion.
    sphpl.add_reticle(fig, lon=260, lat=20, style='plus',
                       size=14, label='Plus reticle', label_side='NE')
    sphpl.add_reticle(fig, lon=85, lat=45, style='x',
                       size=14, color='gold',
                       label='X reticle', label_side='auto')
    sphpl.add_reticle(fig, lon=350, lat=-25, style='circle',
                       size=18, color='cyan',
                       label='Open circle', label_side='E')
    sphpl.add_reticle(fig, lon=120, lat=-40, style='circle',
                       size=18, color='salmon',
                       circle_gap_deg=40,
                       label='Broken circle', label_side='S')
    sphpl.add_reticle(fig, lon=200, lat=-55, style='L',
                       size=18, rotation=90, color='lightgreen',
                       label='L (NW open)', label_side='NW')

    return fig


def render():
    banner("plotly core overlays — two-panel AIT comparison "
            "(center=180 vs center=90)")
    fig_180 = _build_panel(
        center=180,
        title='skyplothelper.plotly core overlays — AIT(center=180) — wrap edge at lon=0',
    )
    fig_90 = _build_panel(
        center=90,
        title='skyplothelper.plotly core overlays — AIT(center=90) — wrap edge at lon=270',
    )

    # Both figures share a CDN-loaded plotly.js; the second figure
    # embeds without re-requesting the bundle. The result is a single
    # HTML page with two independently-zoomable / hoverable panels.
    html_180 = fig_180.to_html(full_html=False, include_plotlyjs='cdn')
    html_90 = fig_90.to_html(full_html=False, include_plotlyjs=False)
    combined = (
        "<!doctype html><html><head>"
        "<meta charset='utf-8'>"
        "<title>skyplothelper.plotly core overlays — two-center comparison</title>"
        "<style>body{margin:0;padding:8px;background:#06060a;color:#dcdcdc;"
        "font-family:sans-serif}</style>"
        "</head><body>"
        f"{html_180}{html_90}"
        "</body></html>"
    )

    out = os.path.join(OUTPUT_DIR, "plotly_01_tier1_overlays.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(combined)
    size_kb = os.path.getsize(out) / 1024
    print(f"  saved: {out} ({size_kb:.0f} kB, "
          f"{len(fig_180.data) + len(fig_90.data)} traces total)")


if __name__ == "__main__":
    render()
    print("\nDone.")
