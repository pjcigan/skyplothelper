# Tangent-plane field

```{image} /_static/features/tangent-plane-field-light.png
:class: sph-plot plot-light dark-light
:alt: Tangent-plane field (light mode)
```


```{image} /_static/features/tangent-plane-field-dark.png
:class: sph-plot plot-dark dark-light
:alt: Tangent-plane field (dark mode)
```


A zoomed gnomonic (TAN) field on a target, with catalog points and a
compass.

Guide: {doc}`/guide/frames` — API: {py:obj}`~skyplothelper.make_wcs_frame` · {py:obj}`~skyplothelper.add_compass`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(7)
fig = plt.figure(figsize=(4.8, 4.4))
ax = sph.make_wcs_frame(111, "TAN", center=(83.63, 22.01), fov_deg=0.2, fig=fig)
ra = 83.63 + (rng.random(40) - 0.5) * 0.16
dec = 22.01 + (rng.random(40) - 0.5) * 0.16
ax.scatter(ra, dec, transform=ax.get_transform("world"), s=14, color="C0")
sph.add_compass(ax)
```
