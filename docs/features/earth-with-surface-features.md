# Earth with surface features

```{image} /_static/features/earth-with-surface-features-light.png
:class: sph-plot plot-light dark-light
:alt: Earth with surface features (light mode)
```


```{image} /_static/features/earth-with-surface-features-dark.png
:class: sph-plot plot-dark dark-light
:alt: Earth with surface features (dark mode)
```


An orthographic Earth in the geographic convention, with coastlines and
tectonic-plate boundaries — no cartopy required.

Guide: {doc}`/guide/globe` — API: {py:obj}`~skyplothelper.make_planet_frame` · {py:obj}`~skyplothelper.plot_coastlines` · {py:obj}`~skyplothelper.plot_tectonic_plates`

## Code

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

fig = plt.figure(figsize=(4.6, 4.6))
ax = sph.make_planet_frame(111, center_LONdeg=-30, center_LATdeg=25)
sph.plot_coastlines(ax)
sph.plot_tectonic_plates(ax, color="C3", lw=0.8)
```
