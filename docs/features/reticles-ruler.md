# Reticles & ruler

```{image} /_static/features/reticles-ruler-light.png
:class: sph-plot plot-light dark-light
:alt: Reticles & ruler (light mode)
```


```{image} /_static/features/reticles-ruler-dark.png
:class: sph-plot plot-dark dark-light
:alt: Reticles & ruler (dark mode)
```


The four reticle styles marking targets, and a ruler measuring an
angular span with pixel-stable ticks.

Guide: {doc}`/guide/overlays` — API: {py:obj}`~skyplothelper.add_reticle` · {py:obj}`~skyplothelper.Ruler`

## Code

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

fig = plt.figure(figsize=(4.8, 4.4))
ax = sph.make_wcs_frame(111, "TAN", center=(83.63, 22.01), fov_deg=0.5, fig=fig)
for i, style in enumerate(["plus", "x", "L", "circle"]):
    sph.add_reticle(ax, (83.78 - 0.1 * i, 22.16), style=style)
# add_to() is what draws; passing ax= as well double-adds and the ruler
# then renders nothing at all.
sph.Ruler((110, 110), (390, 330), pixscale_asec=3.6).add_to(ax)
```
