# Coordinate overlay grid

```{image} /_static/features/coordinate-overlay-grid-light.png
:class: sph-plot plot-light dark-light
:alt: Coordinate overlay grid (light mode)
```


```{image} /_static/features/coordinate-overlay-grid-dark.png
:class: sph-plot plot-dark dark-light
:alt: Coordinate overlay grid (dark mode)
```


A second coordinate system's full graticule — styled gridlines, ticks, and
labels — laid over the primary frame (galactic over an ICRS map).

Guide: {doc}`/guide/ticks` — API: {py:obj}`~skyplothelper.add_coord_overlay`

## Code

```python
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="AIT", center=180)
sph.add_coord_overlay(ax, frame="galactic", color="C1", lw=0.8, alpha=0.8)
```
