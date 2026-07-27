# Layered HEALPix maps

```{image} /_static/features/layered-healpix-maps-light.png
:class: sph-plot plot-light dark-light
:alt: Layered HEALPix maps (light mode)
```


```{image} /_static/features/layered-healpix-maps-dark.png
:class: sph-plot plot-dark dark-light
:alt: Layered HEALPix maps (dark mode)
```


Two maps on one frame — a smoothed density field with the occupied tiles of
a second, sparser catalog drawn over it.

Guide: {doc}`/guide/healpix` — API: {py:obj}`~skyplothelper.bin_data_as_healpix` · {py:obj}`~skyplothelper.healpix_smooth` · {py:obj}`~skyplothelper.plot_healpix_map` · {py:obj}`~skyplothelper.plot_healpix_sparse`

## Code

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(19)
lon = rng.normal(150, 35, 9000) % 360
lat = np.clip(rng.normal(-5, 25, 9000), -89, 89)
base, *_ = sph.bin_data_as_healpix(lon, lat, np.ones_like(lon), nside=32,
                                   statistic="count", blank_value=0)
base = sph.healpix_smooth(base, sigma_deg=4.0)
fig, ax = sph.allsky_figure(projection="MOL", center=180)
sph.plot_healpix_map(base, ax=ax, cmap="sph.deepsky")
clon = rng.normal(150, 8, 400) % 360
clat = np.clip(rng.normal(-5, 8, 400), -89, 89)
ipix, vals = sph.bin_data_sparse(clon, clat, np.ones_like(clon),
                                 nside=32, statistic="count")
sph.plot_healpix_sparse(ipix, vals, nside=32, ax=ax, cmap="sph.sunset",
                        show_boundaries=True, boundary_color="w",
                        boundary_lw=0.2, set_extent=False)
```
