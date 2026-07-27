# Nightshade

```{image} /_static/features/nightshade-light.png
:class: sph-plot plot-light dark-light
:alt: Nightshade (light mode)
```


```{image} /_static/features/nightshade-dark.png
:class: sph-plot plot-dark dark-light
:alt: Nightshade (dark mode)
```


The day/night terminator for a given instant — a night-lights layer blended
onto the day map with a physical twilight falloff.

Guide: {doc}`/guide/globe` — API: {py:obj}`~skyplothelper.make_nightshade_blend` · {py:obj}`~skyplothelper.add_scale_bar_cylindrical`

## Code

```python
import datetime as dt
import matplotlib.pyplot as plt
import skyplothelper as sph

day = plt.imread("examples/data/world.topo.bathy.200412.3x5400x2700.jpg")
night = plt.imread("examples/data/BlackMarble_2016_01deg.jpg").astype(float) / 255.0
when = dt.datetime(2024, 6, 21, 21, 0)
extent = [-180, 180, -90, 90]
fig, ax = plt.subplots(figsize=(6.4, 3.3))
ax.imshow(day, extent=extent, origin="upper")
ax.imshow(sph.make_nightshade_blend(night, when, blend_sigma=60),
          extent=extent, origin="upper")
sph.add_scale_bar_cylindrical(ax, lat=45, body="earth", length_km=2000,
                              color="white", stroke_color="0.1")
ax.set_xticks([])
ax.set_yticks([])
```

```{note}
This example uses a file from the repository's [`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/examples/data) directory (not bundled with the pip install) — see the README there for provenance and credits.
```
