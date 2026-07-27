# Constellation star chart

```{image} /_static/features/constellation-star-chart-light.png
:class: sph-plot plot-light dark-light
:alt: Constellation star chart (light mode)
```


```{image} /_static/features/constellation-star-chart-dark.png
:class: sph-plot plot-dark dark-light
:alt: Constellation star chart (dark mode)
```


The IAU constellation kit — boundaries (precessed to ICRS), asterism
lines, and labels — over a wide field around Orion.

Guide: {doc}`/guide/overlays` — API: {py:obj}`~skyplothelper.add_constellation_boundaries` · {py:obj}`~skyplothelper.add_constellation_lines` · {py:obj}`~skyplothelper.add_constellation_labels`

## Code

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

fig = plt.figure(figsize=(5.0, 4.6))
ax = sph.make_wcs_frame(111, "ARC", center=(83, 0), fov_deg=70, fig=fig)
sph.add_constellation_boundaries(ax, color="C7", lw=0.6)
sph.add_constellation_lines(ax, color="C1", lw=0.9)
sph.add_constellation_labels(ax)
```
