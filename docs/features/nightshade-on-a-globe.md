# Nightshade on a globe

```{image} /_static/features/nightshade-on-a-globe-light.png
:class: sph-plot plot-light dark-light
:alt: Nightshade on a globe (light mode)
```


```{image} /_static/features/nightshade-on-a-globe-dark.png
:class: sph-plot plot-dark dark-light
:alt: Nightshade on a globe (dark mode)
```


The same day/night terminator, draped on an orthographic Earth: composite
the blend in cylindrical space, hand the array straight to
`pseudofits_from_image`, and reproject it onto a planet frame.

Guide: {doc}`/guide/globe` — API: {py:obj}`~skyplothelper.make_nightshade_blend` · {py:obj}`~skyplothelper.pseudofits_from_image` · {py:obj}`~skyplothelper.reproject_rgb_map`

## Code

```python
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skyplothelper as sph

day = plt.imread("examples/data/world.topo.bathy.200412.3x5400x2700.jpg")
night = plt.imread("examples/data/BlackMarble_2016_01deg.jpg").astype(float) / 255.0
when = dt.datetime(2024, 6, 21, 21, 0)

# Blend, then flatten day + night into ONE cylindrical raster (the blend's
# alpha carries the twilight falloff). The maps ship on different grids, so
# resample the day map onto the night map's before compositing.
shade = sph.make_nightshade_blend(night, when, blend_sigma=60)
h, w = shade.shape[:2]
day_r = np.asarray(Image.fromarray(day).resize((w, h), Image.BILINEAR)) / 255.0
alpha = shade[..., 3:4]
flat = np.clip(day_r * (1 - alpha) + shade[..., :3] * alpha, 0, 1)

# pseudofits_from_image takes the array directly — no temp file needed.
hdu = sph.pseudofits_from_image((flat * 255).astype("uint8"), geo=True)
fig = plt.figure(figsize=(4.6, 4.6))
ax = sph.make_planet_frame(111, center_LONdeg=-40, center_LATdeg=25,
                           Naxispix=500, grid=False)
out_hdr = ax.wcs.to_header()
nx = round(ax.get_xlim()[1] - ax.get_xlim()[0])
ny = round(ax.get_ylim()[1] - ax.get_ylim()[0])
out_hdr["NAXIS1"], out_hdr["NAXIS2"] = nx, ny
ax.imshow(np.nan_to_num(sph.reproject_rgb_map(hdu, out_hdr, shape_out=(ny, nx))),
          zorder=-10)
sph.plot_ortho_grid(ax, front_color="0.7", front_lw=0.4, show_back=False)
```

```{note}
This example uses a file from the repository's [`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/examples/data) directory (not bundled with the pip install) — see the README there for provenance and credits.
```
