# Celestial globe

```{image} /_static/features/celestial-globe-light.png
:class: sph-plot plot-light dark-light
:alt: Celestial globe (light mode)
```


```{image} /_static/features/celestial-globe-dark.png
:class: sph-plot plot-dark dark-light
:alt: Celestial globe (dark mode)
```


A hemisphere of sky as a globe, with a coordinate graticule, catalog
points, and a surface compass drawn on the sphere itself.

Guide: {doc}`/guide/globe` — API: {py:obj}`~skyplothelper.make_globe_frame` · {py:obj}`~skyplothelper.plot_scatter_globe` · {py:obj}`~skyplothelper.add_surface_compass`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(3)
fig = plt.figure(figsize=(4.6, 4.6))
ax = sph.make_globe_frame(111, center_LONdeg=180, center_LATdeg=30)
lon = rng.uniform(0, 360, 200)
lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 200)))
sph.plot_scatter_globe(ax, lon, lat, s=8, color="C0")
# A surface compass sits *on the sphere*, so its east/west follow the sky
# convention and the local graticule rather than screen axes.
sph.add_surface_compass(ax, 215, -5, size_deg=16, color="C1")
```
