# Standalone legend blocks

```{image} /_static/features/standalone-legend-blocks-light.png
:class: sph-plot plot-light dark-light
:alt: Standalone legend blocks (light mode)
```


```{image} /_static/features/standalone-legend-blocks-dark.png
:class: sph-plot plot-dark dark-light
:alt: Standalone legend blocks (dark mode)
```


Every `add_*` wrapper has a block class behind it. Build
`LegendBlock`s directly — picking the swatch renderer with
`swatch_kind` (`line`, `patch`, `marker`, `region`, `text`, …) — and
attach them with `add_block`, mixing them with named glyph swatches so
one key describes curves, filled regions and reticles together.

Guide: {doc}`/guide/legends` — API: {py:obj}`~skyplothelper.LegendBlock` · {py:obj}`~skyplothelper.MultiLegend`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph
from skyplothelper import LegendBlock

# The gallery renders this snippet in both modes, so pick the legend
# palette off the active figure background rather than hard-coding it.
from matplotlib.colors import to_rgb
LP = ("dark" if sum(to_rgb(plt.rcParams["figure.facecolor"])) / 3 < 0.5
      else "publication")
fig, ax = sph.allsky_figure(projection="AIT", center=180)
sph.add_great_circle(ax, pole_lon=0, pole_lat=90, frame="pole",
                     color="C1", ls="--", lw=2)
sph.add_great_circle(ax, pole_lon=90, pole_lat=30, frame="pole",
                     color="C2", ls=":", lw=2)
sph.add_geodesic_circle(ax, 300, 30, 25, facecolor="C0", alpha=0.35,
                        edgecolor="none")
sph.add_geodesic_circle(ax, 90, -25, 20, facecolor="C3", alpha=0.35,
                        edgecolor="none")
sph.add_reticle(ax, (180, 10), style="plus", size=14, color="0.35", lw=1.6)

lines = LegendBlock("Models", {"model A": dict(ls="--", lw=2, color="C1"),
                               "model B": dict(ls=":", lw=2, color="C2")},
                    swatch_kind="line")
regions = LegendBlock("Regions", {"observed": dict(facecolor="C0", alpha=0.35),
                                  "planned": dict(facecolor="C3", alpha=0.35)},
                      swatch_kind="patch")
(sph.MultiLegend(ax, palette=LP, loc="outside bottom", orientation="horizontal",
                 fontsize=7)
    .add_block(lines)
    .add_block(regions)
    .add_glyph("targets", {"primary": "reticle_plus"})
    .draw())
```
