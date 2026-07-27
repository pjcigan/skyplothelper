# Signed data & symmetric log

```{image} /_static/features/signed-data-symmetric-log-light.png
:class: sph-plot plot-light dark-light
:alt: Signed data & symmetric log (light mode)
```


```{image} /_static/features/signed-data-symmetric-log-dark.png
:class: sph-plot plot-dark dark-light
:alt: Signed data & symmetric log (dark mode)
```


A residual/Stokes-style image with positive and negative values, shown
through a symmetric-log norm (linear core, log wings) so the colorbar reads
true values.

Guide: {doc}`/guide/images` — API: {py:obj}`~skyplothelper.make_norm`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(3)
gx, gy = np.meshgrid(np.linspace(-1, 1, 160), np.linspace(-1, 1, 160))
signed = (np.exp(-((gx - 0.3) ** 2 + gy ** 2) / 0.03)
          - np.exp(-((gx + 0.3) ** 2 + gy ** 2) / 0.03)) \
    + 0.05 * rng.standard_normal(gx.shape)
lim = float(np.nanmax(np.abs(signed)))
fig, ax = plt.subplots(figsize=(4.8, 4.0))
im = ax.imshow(signed, origin="lower", cmap="sph.diff_blueorange",
               norm=sph.make_norm(stretch="symlog", clip="manual",
                                  vmin=-lim, vmax=lim, a=0.05))
ax.set_xticks([])
ax.set_yticks([])
sph.add_colorbar(im, ax=ax, mode="divider", label="residual")
```
