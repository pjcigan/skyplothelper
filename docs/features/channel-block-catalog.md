# Channel-block catalog

```{image} /_static/features/channel-block-catalog-light.png
:class: sph-plot plot-light dark-light
:alt: Channel-block catalog (light mode)
```


```{image} /_static/features/channel-block-catalog-dark.png
:class: sph-plot plot-dark dark-light
:alt: Channel-block catalog (dark mode)
```


Every `MultiLegend` block kind on one figure — the marker/line channels
(color, shape, size, edge, fill, alpha, angle, line) and the specialty
swatches (hatch, region, sph reticle glyphs, a colorbar strip, text, and any
custom artist). A catalog of what you can key, not a template to fill.

Guide: {doc}`/guide/legends` — API: {py:obj}`~skyplothelper.MultiLegend`

## Code

```python
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
import skyplothelper as sph

# The gallery renders this snippet in both modes, so pick the legend palette
# off the active figure background rather than hard-coding it.
LP = ("dark" if sum(to_rgb(plt.rcParams["figure.facecolor"])) / 3 < 0.5
      else "publication")
A = sph.CYCLE_PALETTES["atlas"]["colors"]
frame = plt.rcParams["axes.edgecolor"]
fig, (axA, axB) = plt.subplots(2, 1, figsize=(12, 4.6))
for a in (axA, axB):
    a.axis("off")

# Channels that vary a marker or line property.
(sph.MultiLegend(axA, loc="center", orientation="horizontal", block_sep=18, palette=LP)
    .add_color("Color", {"A": A[0], "B": A[1], "C": A[2]}, swatch="marker")
    .add_shape("Shape", {"disk": "o", "gal": "D", "star": "*"})
    .add_size("Size", values=[1, 100, 10000], smin=8, smax=230, scale="sqrt", fmt=".0f")
    .add_edge("Edge", {"secure": A[2], "flagged": A[3]})
    .add_fill("Fill", {"detected": "filled", "limit": "open"})
    .add_alpha("Alpha", values=[1, 5, 20], fmt=".0f")
    .add_orientation("Angle", {"0°": 0, "60°": 60, "120°": 120})
    .add_line("Line", {"fit": "--", "prior": ":"})
    .draw())

# Specialty swatch kinds — hatch, region, sph reticle glyphs, a colorbar
# strip, text, and any matplotlib artist you hand in.
star = Line2D([0], [0], marker=(6, 1, 0), markersize=12, linestyle="none",
              markerfacecolor=A[1], markeredgecolor=frame)
(sph.MultiLegend(axB, loc="center", orientation="horizontal", block_sep=18, palette=LP)
    .add_fill("Hatch", {"DES": "///", "LSST": "xxx"}, kind="patch", color=A[0])
    .add_region("Region", {"footprint": dict(fc=A[0], ec=A[0], alpha=0.35),
                           "mask": dict(fc=A[3], ec=A[3], hatch="//")})
    .add_glyph("Glyph", {"target": "reticle_circle", "mark": "crosshair"})
    .add_colorbar("Redshift", cmap="sph.deepsky", vmin=0, vmax=2, length=90, fmt=".1f")
    .add_text("Text", ["dashed = model"])
    .add_custom("Custom", {"my marker": star})
    .draw())
axA.set_title("Channels that vary a marker or line", fontsize=10, y=0.97)
axB.set_title("Specialty swatch kinds — hatch, region, sph glyphs, colorbar, text, custom",
              fontsize=10, y=0.97)
```
