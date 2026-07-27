# Colorbar styles

```{image} /_static/features/colorbar-styles-light.png
:class: sph-plot plot-light dark-light
:alt: Colorbar styles (light mode)
```


```{image} /_static/features/colorbar-styles-dark.png
:class: sph-plot plot-dark dark-light
:alt: Colorbar styles (dark mode)
```


A stretched image with a WCS-aware colorbar — `add_colorbar` sizes to a
fixed-aspect image axes correctly where a bare `plt.colorbar` would not.

Guide: {doc}`/guide/images` — API: {py:obj}`~skyplothelper.add_colorbar` · {py:obj}`~skyplothelper.rescale_image`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(2)
ny = nx = 220
yy, xx = np.mgrid[0:ny, 0:nx]
# A diffuse extended source plus compact knots on a noise floor: something
# with real dynamic range for the stretch and the colorbar to describe.
img = 180 * np.exp(-((((xx - 108) / 46.) ** 2 + ((yy - 104) / 30.) ** 2)) ** 1.1)
for sx, sy, amp, w in [(58, 150, 90, 3.4), (166, 66, 130, 2.8),
                       (150, 158, 55, 2.2), (78, 62, 70, 2.6),
                       (196, 128, 40, 2.0)]:
    img += amp * np.exp(-(((xx - sx) / w) ** 2 + ((yy - sy) / w) ** 2))
img += rng.normal(0, 1.6, img.shape)
fig, ax = plt.subplots(figsize=(4.8, 4.2))
im = ax.imshow(img, origin="lower", cmap="sph.lagoon",
               norm=sph.make_norm(stretch="asinh", clip="percentile",
                                  phi=99.5, data=img))
ax.set_xticks([])
ax.set_yticks([])
sph.add_colorbar(im, ax=ax, mode="divider", label="surface brightness")
```
