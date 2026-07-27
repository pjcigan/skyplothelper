# Advanced RGB composite

```{image} /_static/features/advanced-rgb-composite-light.png
:class: sph-plot plot-light dark-light
:alt: Advanced RGB composite (light mode)
```


```{image} /_static/features/advanced-rgb-composite-dark.png
:class: sph-plot plot-dark dark-light
:alt: Advanced RGB composite (dark mode)
```


Four narrow-band frames combined with
[`multicolorfits`](https://github.com/pjcigan/multicolorfits): each band is
scaled to grayscale, tinted its own hue, then added together — so emission
lines keep distinct colors instead of being forced into R/G/B channels.
SN 1987A in four HST filters.

Guide: {doc}`/guide/images` — API: {py:obj}`~skyplothelper.rescale_image`

## Code

```python
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import multicolorfits as mcf
import skyplothelper as sph

BANDS = ["sn1987a_hst_F625W.fits", "sn1987a_hst_F656N.fits",
         "sn1987a_hst_F658N.fits", "sn1987a_hst_F502N.fits"]
COLORS = ["#FBFCCF", "#729FCF", "#75507B", "#EF2929"]   # continuum, Ha, [N II], [O III]

hdus = [fits.open(f"examples/data/{b}")[0] for b in BANDS]
wcs = WCS(hdus[0].header).celestial
# 1. scale each band  2. tint it  3. add the tinted frames together
gray = [mcf.to_grey_rgb(np.squeeze(h.data).astype(float), rescalefn="log",
                        scaletype="perc", min_max=[40, 99.9]) for h in hdus]
rgb = mcf.combine_multicolor(
    [mcf.colorize_image(g, c, colorintype="hex", gammacorr_color=2.2)
     for g, c in zip(gray, COLORS)], gamma=2.2)

LABELS = ["F625W (continuum)", "Hα (F656N)", "[N II] (F658N)", "[O III] (F502N)"]

# The gallery renders this snippet in both modes, so pick the legend
# palette off the active figure background rather than hard-coding it.
from matplotlib.colors import to_rgb
LP = ("dark" if sum(to_rgb(plt.rcParams["figure.facecolor"])) / 3 < 0.5
      else "publication")
fig = plt.figure(figsize=(4.8, 4.6))
ax = fig.add_subplot(111, projection=wcs)
ax.imshow(rgb, origin="lower")
ax.coords[0].set_axislabel("RA (J2000)", fontsize=9)
ax.coords[1].set_axislabel("Dec (J2000)", fontsize=9)
ax.coords[0].set_ticklabel(size=8)
ax.coords[1].set_ticklabel(size=8)
# Key which hue came from which filter — round swatches in the tint colors.
(sph.MultiLegend(ax, loc="lower left", palette=LP, fontsize=6.5,
                 framealpha=0.75)
    .add_color("Filter", dict(zip(LABELS, COLORS)), swatch="marker")
    .draw())
```

```{note}
This example uses a file from the repository's [`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/examples/data) directory (not bundled with the pip install) — see the README there for provenance and credits.
```
