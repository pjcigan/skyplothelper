# Stars in perceived colors

```{image} /_static/features/stars-in-perceived-colors-light.png
:class: sph-plot plot-light dark-light
:alt: Stars in perceived colors (light mode)
```


```{image} /_static/features/stars-in-perceived-colors-dark.png
:class: sph-plot plot-dark dark-light
:alt: Stars in perceived colors (dark mode)
```


Color a star catalog by each star's *perceived* color from a named color index
with `color_index_to_rgb` (hot stars blue-white, cool stars orange) — Gaia BP-RP
for the Pleiades here, sized by brightness, on a night-sky canvas.

Guide: {doc}`/guide/vectors` — API: {py:obj}`~skyplothelper.color_index_to_rgb`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import skyplothelper as sph

# The Pleiades (M45) from a bundled Gaia catalog, colored by each star's real
# perceived color. color_index_to_rgb takes any named index; here Gaia BP-RP
# (sph.bp_rp_to_rgb is the equivalent shortcut for this one).
gaia = Table.read("examples/data/query_cache/gaia_m45.ecsv")
colors = sph.color_index_to_rgb(gaia["BP-RP"], index="BP-RP", saturation=1.0)
size = np.clip(200 * 10 ** (-0.4 * (gaia["Gmag"] - 6.0)), 2.0, 240.0)

fig = plt.figure(figsize=(6.4, 5.6))
ax = sph.make_wcs_frame(111, "TAN", center=(56.75, 24.12), fov_deg=2.7, fig=fig)
ax.set_facecolor("#05060a")   # a night-sky canvas, dark in both display modes
for c in (0, 1):              # light ticks/labels so they read on the black sky
    ax.coords[c].set_ticklabel(color="0.8", size=8)
    ax.coords[c].set_ticks(color="0.5")
ax.scatter(gaia["RA_ICRS"], gaia["DE_ICRS"], transform=ax.get_transform("world"),
           c=colors, s=size, lw=0, alpha=0.9, zorder=3)
ax.set_title("The Pleiades in perceived colors (color_index_to_rgb, BP-RP)",
             fontsize=11)
```

```{note}
This example uses a file from the repository's [`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/examples/data) directory (not bundled with the pip install) — see the README there for provenance and credits.
```
