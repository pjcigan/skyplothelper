# Tissot indicatrices

```{image} /_static/features/tissot-indicatrices-light.png
:class: sph-plot plot-light dark-light
:alt: Tissot indicatrices (light mode)
```


```{image} /_static/features/tissot-indicatrices-dark.png
:class: sph-plot plot-dark dark-light
:alt: Tissot indicatrices (dark mode)
```


Equal-radius geodesic circles across the map make a projection's local
distortion visible at a glance.

Guide: {doc}`/guide/regions` — API: {py:obj}`~skyplothelper.tissot`

## Code

```python
import numpy as np
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="MOL", center=180)
lons, lats = np.meshgrid(np.arange(30, 331, 60), np.arange(-60, 61, 30))
sph.tissot(ax, rad_deg=6, lons=lons.ravel(), lats=lats.ravel(),
           facecolor="C0", edgecolor="C0", alpha=0.4)
```
