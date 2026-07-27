# Basic RGB composite

```{image} /_static/features/basic-rgb-composite-light.png
:class: sph-plot plot-light dark-light
:alt: Basic RGB composite (light mode)
```


```{image} /_static/features/basic-rgb-composite-dark.png
:class: sph-plot plot-dark dark-light
:alt: Basic RGB composite (dark mode)
```


A three-color composite from separate filter images (NGC 602 in
HST B / R / IR), each stretched and stacked into an RGB frame.

Guide: {doc}`/guide/images` — API: {py:obj}`~skyplothelper.rescale_image`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import skyplothelper as sph

def band(path):
    d = np.squeeze(fits.getdata(path)).astype(float)
    lo, hi = np.nanpercentile(d, [1.0, 99.6])
    return np.clip((d - lo) / (hi - lo), 0, 1)

rgb = np.dstack([band("examples/data/ngc602_IR.fits"),
                 band("examples/data/ngc602_R.fits"),
                 band("examples/data/ngc602_B.fits")]) ** 0.8
fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.imshow(rgb, origin="lower")
ax.set_xticks([])
ax.set_yticks([])
```

```{note}
This example uses a file from the repository's [`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/examples/data) directory (not bundled with the pip install) — see the README there for provenance and credits.
```
