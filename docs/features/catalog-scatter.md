# Catalog scatter

```{image} /_static/features/catalog-scatter-light.png
:class: sph-plot plot-light dark-light
:alt: Catalog scatter (light mode)
```


```{image} /_static/features/catalog-scatter-dark.png
:class: sph-plot plot-dark dark-light
:alt: Catalog scatter (dark mode)
```


Drop any table on the sky and encode columns in marker color and size,
with a matched colorbar — column names are auto-detected.

Guide: {doc}`/guide/vectors` — API: {py:obj}`~skyplothelper.plot_catalog`

## Code

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(6)
n = 500
cat = {"ra": rng.uniform(0, 360, n),
       "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
       "z": rng.random(n) ** 2 * 0.2,
       "flux": 10 ** rng.uniform(0, 2, n)}
fig, ax = sph.allsky_figure(projection="AIT", center=180)
sph.plot_catalog(ax, cat, colorby="z", sizeby="flux", size_scale="log",
                 cmap="sph.sunset", cbar=True, cbar_label="redshift")
```
