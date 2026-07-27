# Second coordinate grid

```{image} /_static/features/second-coordinate-grid-light.png
:class: sph-plot plot-light dark-light
:alt: Second coordinate grid (light mode)
```


```{image} /_static/features/second-coordinate-grid-dark.png
:class: sph-plot plot-dark dark-light
:alt: Second coordinate grid (dark mode)
```


Another coordinate system's graticule drawn over the primary frame —
here galactic gridlines on an ICRS Mollweide map.

Guide: {doc}`/guide/ticks` — API: {py:obj}`~skyplothelper.add_second_grid`

## Code

```python
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="MOL", center=180)
# A warm, dashed overlay so the galactic graticule can't be mistaken for
# the frame's own (neutral, solid) equatorial grid.
sph.add_second_grid(ax, overlay_frame="galactic", color="C1", alpha=0.95,
                    linewidth=1.1, linestyle="--")
```
