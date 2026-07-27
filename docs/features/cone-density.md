# Cone density

```{image} /_static/features/cone-density-light.png
:class: sph-plot plot-light dark-light
:alt: Cone density (light mode)
```


```{image} /_static/features/cone-density-dark.png
:class: sph-plot plot-dark dark-light
:alt: Cone density (dark mode)
```


Binning the wedge itself — a hexbin in cone space with a matched
colorbar.

Guide: {doc}`/guide/cone` — API: {py:obj}`~skyplothelper.cone_hexbin` · {py:obj}`~skyplothelper.add_colorbar`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(14)
fig = plt.figure(figsize=(5.2, 3.8))
ax = sph.make_cone_frame(111, angle_center=180, angle_half_width=35,
                         r_min=0, r_max=0.12, angle_label="R.A.", fig=fig)
n = 6000
z = 0.12 * np.sqrt(rng.random(n))
ang = 180 + (rng.random(n) - 0.5) * 70
hb = sph.cone_hexbin(ax, ang, z, gridsize=26, cmap="sph.dusk", mincnt=1)
sph.add_colorbar(hb, ax=ax, label="galaxies / hex", mode="simple", shrink=0.55)
```
