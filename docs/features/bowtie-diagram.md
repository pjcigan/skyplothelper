# Bowtie diagram

```{image} /_static/features/bowtie-diagram-light.png
:class: sph-plot plot-light dark-light
:alt: Bowtie diagram (light mode)
```


```{image} /_static/features/bowtie-diagram-dark.png
:class: sph-plot plot-dark dark-light
:alt: Bowtie diagram (dark mode)
```


A double-sided wedge — two opposing cones sharing the apex, for
back-to-back survey slices.

Guide: {doc}`/guide/cone` — API: {py:obj}`~skyplothelper.make_bowtie_frame` · {py:obj}`~skyplothelper.cone_scatter`

## Code

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(13)
top, bot = sph.make_bowtie_frame(
    angle_center=185, angle_half_width=42, r_min=0.0, r_max=0.13,
    angle_tick_spacing=15, r_tick_spacing=0.05)
fig = top.figure
# Both halves of the bowtie share the same angular center (185°); each
# points its radius the opposite way from the shared apex.
for ax in (top, bot):
    n = 500
    z = 0.13 * np.sqrt(rng.random(n))
    ang = 185 + (rng.random(n) - 0.5) * 78
    sph.cone_scatter(ax, ang, z, s=4, alpha=0.55, color="C0")
```
