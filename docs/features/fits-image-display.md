# FITS image display

```{image} /_static/features/fits-image-display-light.png
:class: sph-plot plot-light dark-light
:alt: FITS image display (light mode)
```


```{image} /_static/features/fits-image-display-dark.png
:class: sph-plot plot-dark dark-light
:alt: FITS image display (dark mode)
```


A real FITS image on WCS axes with an asinh/zscale stretch and the
synthesized beam (VLBA 15 GHz image of 3C 84 from the MOJAVE program).

Guide: {doc}`/guide/images` — API: {py:obj}`~skyplothelper.make_norm` · {py:obj}`~skyplothelper.squeeze_image` · {py:obj}`~skyplothelper.add_beam`

## Code

```python
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import skyplothelper as sph

hdu = fits.open("examples/data/0316+413.u.stacked.icd.fits")[0]
data, hdr = sph.squeeze_image(hdu.data, hdu.header)   # drop degenerate axes
wcs = WCS(hdr).celestial
fig = plt.figure(figsize=(4.8, 4.6))
ax = fig.add_subplot(111, projection=wcs)
# symlog: linear within ~5 mJy of zero, logarithmic out to the ~3 Jy peak.
# This is the look the MOJAVE survey publishes, and it reveals the jet's
# full dynamic range where a linear or zscale stretch shows only the core.
ax.imshow(data, origin="lower", cmap="sph.deepsky",
          norm=sph.make_norm(stretch="symlog", clip="manual", vmin=0,
                             vmax=float(np.nanmax(data)), a=5e-3))
# All the structure sits within a few tens of mas of the core: zoom there.
cx, cy = wcs.world_to_pixel_values(hdr["CRVAL1"], hdr["CRVAL2"])
half = 20.0 / (abs(hdr["CDELT2"]) * 3.6e6)
ax.set_xlim(cx - half, cx + half)
ax.set_ylim(cy - half, cy + half)
ax.coords[0].set_ticks(number=4)
ax.coords[1].set_ticks(number=5)
ax.coords[0].set_ticklabel(size=8)
ax.coords[1].set_ticklabel(size=8)
ax.coords[0].set_axislabel("RA (J2000)", fontsize=9)
ax.coords[1].set_axislabel("Dec (J2000)", fontsize=9)
sph.add_beam(ax, hdr, facecolor="white", edgecolor="white")
```

```{note}
This example uses a file from the repository's [`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/examples/data) directory (not bundled with the pip install) — see the README there for provenance and credits.
```
