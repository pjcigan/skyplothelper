# Displacement arrows

```{image} /_static/features/displacement-arrows-light.png
:class: sph-plot plot-light dark-light
:alt: Displacement arrows (light mode)
```


```{image} /_static/features/displacement-arrows-dark.png
:class: sph-plot plot-dark dark-light
:alt: Displacement arrows (dark mode)
```


Individual epoch-to-epoch arrows whose shafts follow the great-circle
path — seam-aware, unlike a raw quiver.

Guide: {doc}`/guide/vectors` — API: {py:obj}`~skyplothelper.plot_displacement`

## Code

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(21)
fig, ax = sph.allsky_figure(projection="AIT", center=180)
n = 60
lon1 = rng.uniform(0, 360, n)
lat1 = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
lon2 = lon1 + rng.normal(0, 6, n) / np.cos(np.radians(lat1))
lat2 = lat1 + rng.normal(0, 4, n)
sph.plot_displacement(ax, lon1, lat1, lon2, lat2, color="C1", geodesic=True)
```
