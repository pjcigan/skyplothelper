# Smoothed HEALPix map

```{image} /_static/features/smoothed-healpix-map-light.png
:class: sph-plot plot-light dark-light
:alt: Smoothed HEALPix map (light mode)
```


```{image} /_static/features/smoothed-healpix-map-dark.png
:class: sph-plot plot-dark dark-light
:alt: Smoothed HEALPix map (dark mode)
```


Gaussian-smoothing a binned map on the sphere before rendering it across
any projection.

Guide: {doc}`/guide/healpix` — API: {py:obj}`~skyplothelper.bin_data_as_healpix` · {py:obj}`~skyplothelper.healpix_smooth` · {py:obj}`~skyplothelper.plot_healpix_map`

## Code

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(15)
lon = rng.normal(140, 30, 6000) % 360
lat = np.clip(rng.normal(-10, 22, 6000), -89, 89)
hp_arr, *_ = sph.bin_data_as_healpix(lon, lat, np.ones_like(lon), nside=32,
                                     statistic="count", blank_value=0)
sm = sph.healpix_smooth(hp_arr, sigma_deg=3.0)
fig, ax = sph.allsky_figure(projection="MOL", center=180)
sph.plot_healpix_map(sm, ax=ax, cmap="sph.deepsky")
```
