# Redshift cone

```{image} /_static/features/redshift-cone-light.png
:class: sph-plot plot-light dark-light
:alt: Redshift cone (light mode)
```


```{image} /_static/features/redshift-cone-dark.png
:class: sph-plot plot-dark dark-light
:alt: Redshift cone (dark mode)
```


A z–RA wedge with the observer at the apex.

Guide: {doc}`/guide/cone` — API: {py:obj}`~skyplothelper.make_cone_frame` · {py:obj}`~skyplothelper.cone_scatter`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(11)
fig = plt.figure(figsize=(5.2, 3.8))
ax = sph.make_cone_frame(111, angle_center=180, angle_half_width=35,
                         r_min=0, r_max=0.12, angle_label="R.A.", fig=fig)
n = 600
z = 0.12 * np.sqrt(rng.random(n))
ang = 180 + (rng.random(n) - 0.5) * 70
sph.cone_scatter(ax, ang, z, s=4, alpha=0.6, color="C0")
```
