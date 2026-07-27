# VSH shift vectors

```{image} /_static/features/vsh-shift-vectors-light.png
:class: sph-plot plot-light dark-light
:alt: VSH shift vectors (light mode)
```


```{image} /_static/features/vsh-shift-vectors-dark.png
:class: sph-plot plot-dark dark-light
:alt: VSH shift vectors (dark mode)
```


Systematic position shifts across the sky, evaluated on a grid and drawn
as sky vectors — here a vector-spherical-harmonic **glide** field, the
Galactic-aberration signature. The same call draws any VSH term.

Guide: {doc}`/guide/vectors` — API: {py:obj}`~skyplothelper.vsh_field` · {py:obj}`~skyplothelper.plot_sky_vectors`

## Code

```python
import numpy as np
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="AIT", center=180)
glon, glat = np.meshgrid(np.arange(0, 360, 20), np.arange(-75, 76, 15))
dlon, dlat = sph.vsh_field(glon.ravel(), glat.ravel(), {"D_3": 1.0})
# scale="auto" sizes the median arrow to a couple of degrees whatever the
# field's amplitude units are. A bare numeric scale here would be read in
# the default arcsec, making every arrow sub-pixel.
sph.plot_sky_vectors(ax, glon.ravel(), glat.ravel(), dlon, dlat,
                     scale="auto", auto_target_deg=9.0, color="C0")
```
