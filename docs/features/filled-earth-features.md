# Filled Earth features

```{image} /_static/features/filled-earth-features-light.png
:class: sph-plot plot-light dark-light
:alt: Filled Earth features (light mode)
```


```{image} /_static/features/filled-earth-features-dark.png
:class: sph-plot plot-dark dark-light
:alt: Filled Earth features (dark mode)
```


Beyond outlines, the Earth features also *fill*. `plot_tectonic_plates(fill=True)`
draws a plate choropleth — here colored from the built-in dual-mode
`REGION_PALETTE` — over a flat Mollweide world, with coastlines for reference.
`plot_land(lakes=True)`, `plot_rivers`, and `clip_to_ocean` fill the other layers
(see the Globe & Planet tutorial). The fills route through the same seam-aware
region machinery as `add_spherical_polygon`.

Guide: {doc}`/guide/globe` — API: {py:obj}`~skyplothelper.plot_tectonic_plates` · {py:obj}`~skyplothelper.plot_land` · {py:obj}`~skyplothelper.clip_to_ocean`

## Code

```python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import skyplothelper as sph

# a discrete choropleth colormap from the built-in dual-mode region palette
region_cmap = ListedColormap(sph.REGION_PALETTE)

fig = plt.figure(figsize=(6.4, 3.4))
ax = sph.make_planet_frame(111, projection="MOL", center_LONdeg=0,
                           grid=True, gridcolor="0.55", gridalpha=0.3)
sph.plot_tectonic_plates(ax, fill=True, cmap=region_cmap, alpha=0.9, edgecolor="0.2")
sph.plot_coastlines(ax, color="0.15", lw=0.5)
```
