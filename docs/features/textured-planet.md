# Textured planet

```{image} /_static/features/textured-planet-light.png
:class: sph-plot plot-light dark-light
:alt: Textured planet (light mode)
```


```{image} /_static/features/textured-planet-dark.png
:class: sph-plot plot-dark dark-light
:alt: Textured planet (dark mode)
```


A solid body draped in its surface texture at its true axial tilt — Mars,
from a bundled map, on a planet frame.

Guide: {doc}`/guide/globe` — API: {py:obj}`~skyplothelper.make_planet_frame` · {py:obj}`~skyplothelper.pseudofits_from_image` · {py:obj}`~skyplothelper.reproject_rgb_map` · {py:obj}`~skyplothelper.euler_to_fits_ortho`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

obl = sph.obliquities["mars"]
clon, clat, pole = sph.euler_to_fits_ortho(rotation=250, obliquity=obl,
                                           perspective=0)
fig = plt.figure(figsize=(4.6, 4.6))
ax = sph.make_planet_frame(111, body="mars", center_LONdeg=clon,
                           center_LATdeg=clat, lonpole=pole, Naxispix=500,
                           grid=False)
hdu = sph.pseudofits_from_image("examples/data/planet_maps/2k_mars.jpg", geo=True)
out_hdr = ax.wcs.to_header()
nx = round(ax.get_xlim()[1] - ax.get_xlim()[0])
ny = round(ax.get_ylim()[1] - ax.get_ylim()[0])
out_hdr["NAXIS1"], out_hdr["NAXIS2"] = nx, ny
bg = sph.reproject_rgb_map(hdu, out_hdr, shape_out=(ny, nx))
ax.imshow(np.nan_to_num(bg), zorder=-10)
# show_back=False: the dashed far-side graticule would otherwise draw on
# top of the surface raster, which reads as a glitch rather than depth.
sph.plot_ortho_grid(ax, front_color="0.75", front_lw=0.4, show_back=False)
```

```{note}
This example uses a file from the repository's [`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/examples/data) directory (not bundled with the pip install) — see the README there for provenance and credits.
```
