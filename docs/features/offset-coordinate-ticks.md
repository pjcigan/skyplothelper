# Offset coordinate ticks

```{image} /_static/features/offset-coordinate-ticks-light.png
:class: sph-plot plot-light dark-light
:alt: Offset coordinate ticks (light mode)
```


```{image} /_static/features/offset-coordinate-ticks-dark.png
:class: sph-plot plot-dark dark-light
:alt: Offset coordinate ticks (dark mode)
```


Relative offset labels (Δα cos δ, Δδ) about a reference position, with
units that walk from degrees down to μas as the field shrinks.

Guide: {doc}`/guide/ticks` — API: {py:obj}`~skyplothelper.apply_offset_ticks` · {py:obj}`~skyplothelper.make_wcs_frame`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(7)
fig = plt.figure(figsize=(4.8, 4.4))
ax = sph.make_wcs_frame(111, "TAN", center=(83.63, 22.01), fov_deg=0.2, fig=fig)
ra = 83.63 + (rng.random(30) - 0.5) * 0.16
dec = 22.01 + (rng.random(30) - 0.5) * 0.16
ax.scatter(ra, dec, transform=ax.get_transform("world"), s=14, color="C0")
sph.apply_offset_ticks(ax, ref_ra_deg=83.63, ref_dec_deg=22.01)
```
