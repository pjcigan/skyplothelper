# Source-density map

```{image} /_static/features/source-density-map-light.png
:class: sph-plot plot-light dark-light
:alt: Source-density map (light mode)
```


```{image} /_static/features/source-density-map-dark.png
:class: sph-plot plot-dark dark-light
:alt: Source-density map (dark mode)
```


A synthetic catalog binned into HEALPix pixels and rendered all-sky.

Guide: {doc}`/guide/healpix` — API: {py:obj}`~skyplothelper.sources_to_healpix_plot`

## Code

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(5)
lon = np.concatenate([rng.uniform(0, 360, 8000),
                      rng.normal(266, 14, 3000) % 360])
lat = np.concatenate([np.degrees(np.arcsin(rng.uniform(-1, 1, 8000))),
                      np.clip(rng.normal(-29, 10, 3000), -89, 89)])
sph.sources_to_healpix_plot(lon, lat, nside=32, cmap="sph.thicket")
```
