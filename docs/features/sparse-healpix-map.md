# Sparse HEALPix map

```{image} /_static/features/sparse-healpix-map-light.png
:class: sph-plot plot-light dark-light
:alt: Sparse HEALPix map (light mode)
```


```{image} /_static/features/sparse-healpix-map-dark.png
:class: sph-plot plot-dark dark-light
:alt: Sparse HEALPix map (dark mode)
```


Only the occupied pixels, drawn as equal-area tiles — the efficient way
to render a partial-sky map at any resolution.

Guide: {doc}`/guide/healpix` — API: {py:obj}`~skyplothelper.bin_data_sparse` · {py:obj}`~skyplothelper.plot_healpix_sparse`

## Code

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(12)
lon = rng.normal(200, 25, 4000) % 360
lat = np.clip(rng.normal(20, 18, 4000), -89, 89)
ipix, vals = sph.bin_data_sparse(lon, lat, np.ones_like(lon),
                                 nside=16, statistic="count")
fig, ax = sph.allsky_figure(projection="MOL", center=180)
sph.plot_healpix_sparse(ipix, vals, nside=16, ax=ax, cmap="sph.dusk",
                        show_boundaries=True, boundary_color="w",
                        boundary_lw=0.3, set_extent=False)
```
