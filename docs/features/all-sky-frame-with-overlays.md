# All-sky frame with overlays

```{image} /_static/features/all-sky-frame-with-overlays-light.png
:class: sph-plot plot-light dark-light
:alt: All-sky frame with overlays (light mode)
```


```{image} /_static/features/all-sky-frame-with-overlays-dark.png
:class: sph-plot plot-dark dark-light
:alt: All-sky frame with overlays (dark mode)
```


Elliptical full-sky frames in any projection, with coordinate-plane and
constellation overlays.

Guide: {doc}`/guide/frames` — API: {py:obj}`~skyplothelper.allsky_figure` · {py:obj}`~skyplothelper.add_plane_overlay` · {py:obj}`~skyplothelper.add_constellation_boundaries`

## Code

```python
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="AIT", center=180)
sph.add_plane_overlay(ax, plane="ecliptic", color="C1", label="Ecliptic")
sph.add_constellation_boundaries(ax)
sph.add_constellation_labels(ax)
```
