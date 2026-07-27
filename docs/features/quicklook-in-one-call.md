# Quicklook in one call

```{image} /_static/features/quicklook-in-one-call-light.png
:class: sph-plot plot-light dark-light
:alt: Quicklook in one call (light mode)
```


```{image} /_static/features/quicklook-in-one-call-dark.png
:class: sph-plot plot-dark dark-light
:alt: Quicklook in one call (dark mode)
```


The standard radio-map recipe — open, stretch, frame, annotate — from a
single call on a FITS path (VLBA 15 GHz image of 3C 84).

Guide: {doc}`/guide/images` — API: {py:obj}`~skyplothelper.quicklook_fits`

## Code

```python
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import skyplothelper as sph

path = "examples/data/0316+413.u.stacked.icd.fits"
peak = float(np.nanmax(np.squeeze(fits.getdata(path))))
# Hand quicklook the same symmetric-log norm the survey uses, ask for
# relative-mas axes, and crop to the jet — the standard radio-map look.
# quicklook builds its own figure, so hand it the active theme's colors —
# otherwise it lands on a white canvas in dark mode. Let it size itself:
# forcing set_size_inches afterwards leaves dead canvas around the layout.
ink = plt.rcParams["text.color"]
res = sph.quicklook_fits(path, image=True, colormap="sph.deepsky",
                         color="white", offset_coords=True, field_size=34,
                         facecolor=plt.rcParams["figure.facecolor"],
                         axcolor=ink, info_color=ink,
                         norm=sph.make_norm(stretch="symlog", clip="manual",
                                            vmin=0, vmax=peak, a=5e-3))
fig = res.fig
```

```{note}
This example uses a file from the repository's [`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/examples/data) directory (not bundled with the pip install) — see the README there for provenance and credits.
```
