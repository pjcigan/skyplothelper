# Contour overlays

```{image} /_static/features/contour-overlays-light.png
:class: sph-plot plot-light dark-light
:alt: Contour overlays (light mode)
```


```{image} /_static/features/contour-overlays-dark.png
:class: sph-plot plot-dark dark-light
:alt: Contour overlays (dark mode)
```


World-coordinate contours drawn on a WCS frame — for tracing structure or
overlaying one dataset on another.

Guide: {doc}`/guide/images` — API: {py:obj}`~skyplothelper.add_contour_overlay`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(1)
fig = plt.figure(figsize=(4.8, 4.4))
ax = sph.make_wcs_frame(111, "TAN", center=(150.0, 2.2), fov_deg=0.4, fig=fig)
gx, gy = np.meshgrid(np.linspace(-0.18, 0.18, 120), np.linspace(-0.18, 0.18, 120))
blob = np.exp(-((gx - 0.03) ** 2 + gy ** 2) / 0.004) \
    + 0.7 * np.exp(-((gx + 0.05) ** 2 + (gy - 0.04) ** 2) / 0.002)
lon = 150.0 + gx / np.cos(np.radians(2.2))
lat = 2.2 + gy
sph.add_contour_overlay(ax, lon, lat, blob, levels=7, cmap="sph.deepsky",
                        linewidths=1.1)
```
