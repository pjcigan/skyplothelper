"""skyplothelper.plotly.add_compound_region — CompoundRegion demo.

Single AIT(center=180) figure layering a few CompoundRegion examples,
each emitted as a single ``fig.add_shape(type='path', ...)`` with the
shapely set-algebra result baked into the path's subpaths so holes
render correctly under plotly's ``evenodd`` fill rule.

What's worth eyeballing:
- Galactic band ±10° with the Galactic Center area excised (subtract
  circle at (266.4, -28.9)): annulus-like region near the GC.
- Two overlapping circles in the southern hemisphere (union).
- A "survey footprint with bright-star mask" pattern in the northern
  hemisphere (rectangle minus two small circles).
- Ecliptic ±5° band with a hole near a chosen anti-pointing direction.

Usage
-----
    python render_plotly_compound_region.py
"""

import os
import sys

from _common import OUTPUT_DIR, banner

try:
    import plotly  # noqa: F401
except ImportError:
    print("skipped: plotly not installed (optional extra).")
    sys.exit(0)

from skyplothelper import plotly as sphpl


def render():
    banner("plotly.add_compound_region — CompoundRegion demo on AIT(center=180)")
    fig = sphpl.make_figure(
        projection='AIT', center=180, theme='dark',
        width=1200, height=650,
        title='skyplothelper.plotly add_compound_region — '
              'set-algebra regions on the sky',
    )

    # Projection silhouette + lon/lat tick labels for context.
    sphpl.add_frame_edge(fig, color='#666', width=1.0)
    sphpl.add_coord_labels(fig, lon_spacing=60, lat_spacing=30)

    # 1) Galactic band ±10° minus Galactic Center circle (annulus near GC).
    # width=1.0 traces the boundary: cross-frame bands now render without
    # the antimeridian seam sliver, so the outline is the real shape edge.
    gal_band = (sphpl.make_compound_region(fig)
                .add_frame_band(-10, 10, frame='galactic')
                .subtract_circle(266.417, -28.908, 8))
    sphpl.add_compound_region(
        fig, gal_band,
        color='deepskyblue',
        fillcolor='rgba(80,200,255,0.30)',
        width=1.0,
        name='Galactic band ±10° (GC excised)',
        hover=True,
    )

    # 2) Union of two overlapping southern circles.
    south_union = (sphpl.make_compound_region(fig)
                   .add_circle(100, -45, 18)
                   .add_circle(130, -55, 20))
    sphpl.add_compound_region(
        fig, south_union,
        color='lightgreen',
        fillcolor='rgba(180,255,120,0.35)',
        width=1.2,
        name='Southern circle union',
        hover=True,
    )

    # 3) Rectangle minus two bright-star masks.
    survey = (sphpl.make_compound_region(fig)
              .add_polygon(lons=[210, 250, 250, 210],
                            lats=[35, 35, 55, 55])
              .subtract_circle(220, 45, 4)
              .subtract_circle(240, 50, 3))
    sphpl.add_compound_region(
        fig, survey,
        color='salmon',
        fillcolor='rgba(255,150,120,0.35)',
        width=1.2,
        name='Survey footprint (2 masks)',
        hover=True,
    )

    # 4) Ecliptic band ±5° with anti-Sun direction circle removed.
    ecl = (sphpl.make_compound_region(fig)
           .add_frame_band(-5, 5, frame='ecliptic')
           .subtract_circle(60, 23.4, 10))
    sphpl.add_compound_region(
        fig, ecl,
        color='gold',
        fillcolor='rgba(255,200,80,0.30)',
        width=1.0,
        name='Ecliptic ±5° (anti-Sun excised)',
        hover=True,
    )

    out = os.path.join(OUTPUT_DIR, "plotly_05_compound_region.html")
    fig.write_html(out, include_plotlyjs='cdn')
    size_kb = os.path.getsize(out) / 1024
    print(f"  saved: {out} ({size_kb:.0f} kB, {len(fig.data)} traces, "
          f"{len(fig.layout.shapes)} shapes)")


if __name__ == "__main__":
    render()
    print("\nDone.")
