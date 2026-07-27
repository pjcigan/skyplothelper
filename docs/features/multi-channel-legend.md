# Multi-channel legend

```{image} /_static/features/multi-channel-legend-light.png
:class: sph-plot plot-light dark-light
:alt: Multi-channel legend (light mode)
```


```{image} /_static/features/multi-channel-legend-dark.png
:class: sph-plot plot-dark dark-light
:alt: Multi-channel legend (dark mode)
```


`MultiLegend` keys several visual channels at once — here marker size
(number of observations), shape (catalog class) and color (observing
band) — placed off-frame.

Guide: {doc}`/guide/legends` — API: {py:obj}`~skyplothelper.MultiLegend` · {py:obj}`~skyplothelper.plot_catalog`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(8)
n = 300
ra = rng.uniform(0, 360, n)
dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
nobs = 10 ** rng.uniform(0, 3, n)
defining = rng.random(n) < 0.3
BANDS = {"S/X": "C0", "K": "C1", "Q": "C2"}
band = np.array(list(BANDS.values()))[rng.integers(0, 3, n)]

# The gallery renders this snippet in both modes, so pick the legend
# palette off the active figure background rather than hard-coding it.
from matplotlib.colors import to_rgb
LP = ("dark" if sum(to_rgb(plt.rcParams["figure.facecolor"])) / 3 < 0.5
      else "publication")
fig, ax = sph.allsky_figure(projection="AIT", center=180)
for is_def, mk in [(False, "o"), (True, "^")]:
    m = defining == is_def
    cp = sph.plot_catalog(ax, {"ra": ra[m], "dec": dec[m], "nobs": nobs[m]},
                          sizeby="nobs", size_vlim=(1, 1000), size_scale="sqrt",
                          smin=6, smax=180, marker=mk, color=band[m], alpha=0.75)
# Three independent channels at once: size, shape and color.
(sph.MultiLegend(ax, palette=LP, loc="outside bottom", orientation="horizontal")
    .add_size_from(cp, values=[1, 10, 100, 1000], title="N obs")
    .add_shape("Class", {"Defining": "^", "Other": "o"})
    .add_color("Band", BANDS, swatch="marker")
    .draw())
```
