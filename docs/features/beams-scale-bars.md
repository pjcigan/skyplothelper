# Beams & scale bars

```{image} /_static/features/beams-scale-bars-light.png
:class: sph-plot plot-light dark-light
:alt: Beams & scale bars (light mode)
```


```{image} /_static/features/beams-scale-bars-dark.png
:class: sph-plot plot-dark dark-light
:alt: Beams & scale bars (dark mode)
```


The image-furnishing furniture on a VLBA 15 GHz image of 3C 84: a column of the
`Beam` styles (each `style=` manages its own outline, fill, crosshair, or hatch)
and two stacked scale bars carrying the *same* on-sky length in angular and in
physical units.

Guide: {doc}`/guide/overlays` — API: {py:obj}`~skyplothelper.Beam` · {py:obj}`~skyplothelper.add_sizebar_asec` · {py:obj}`~skyplothelper.add_sizebar`

## Code

```python
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import skyplothelper as sph
from skyplothelper import Beam

hdu = fits.open("examples/data/0316+413.u.stacked.icd.fits")[0]
data, hdr = sph.squeeze_image(hdu.data, hdu.header)
wcs = WCS(hdr).celestial
fig = plt.figure(figsize=(5.4, 5.2))
ax = fig.add_subplot(111, projection=wcs)
ax.imshow(data, origin="lower", cmap="sph.deepsky",
          norm=sph.make_norm(stretch="symlog", clip="manual", vmin=0,
                             vmax=float(np.nanmax(data)), a=5e-3))
cx, cy = wcs.world_to_pixel_values(hdr["CRVAL1"], hdr["CRVAL2"])
half = 12.0 / (abs(hdr["CDELT2"]) * 3.6e6)          # +/-12 mas crop
ax.set_xlim(cx - half, cx + half)
ax.set_ylim(cy - half, cy + half)
ax.coords[0].set_ticklabel(size=8); ax.coords[1].set_ticklabel(size=8)
ax.coords[0].set_axislabel("RA (J2000)", fontsize=9)
ax.coords[1].set_axislabel("Dec (J2000)", fontsize=9)

# A column of the Beam styles up the left side, each labeled - each style=
# manages its own outline, fill, crosshair, or hatch pattern.
bx = cx - 0.66 * half
bmaj, bmin = 0.16 * half, 0.10 * half
styles = ["ellipse", "filled", "crosshair", "crosshairgrid", "hatch"]
ys = np.linspace(cy + 0.74 * half, cy - 0.74 * half, len(styles))
for style, by in zip(styles, ys):
    # Pass only ec - each style manages its own fill / crosshair / hatch, so the
    # five read as visibly different (outline, solid, crosshair, grid, hatch).
    Beam((bx, by), bmaj_pix=bmaj, bmin_pix=bmin, bpa_deg=30, style=style,
         ec="#7fdfff", lw=1.3, stroke_color="0.1", stroke_lw=2.0).add_to(ax)
    ax.text(bx + 0.15 * half, by, style, color="white", fontsize=8, va="center",
            path_effects=[pe.withStroke(linewidth=2.0, foreground="0.1")])

# Two stacked scale bars: the same on-sky length labeled in angle and in physical
# units. NGC 1275 (z=0.0176) subtends ~0.36 pc/mas, so 5 mas ~ 1.8 pc.
sph.add_sizebar_asec(ax, hdr, 0.005, "5 mas", color="white", loc="lower right",
                     stroke_color="0.1", stroke_lw=2.0)
mas_to_px = 1.0 / (abs(hdr["CDELT2"]) * 3.6e6)
sph.add_sizebar(ax, 5 * mas_to_px, "1.8 pc", loc="lower right", borderpad=2.7,
                label_top=True, color="white", stroke_color="0.1", stroke_lw=2.0)
```

```{note}
This example uses a file from the repository's [`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/examples/data) directory (not bundled with the pip install) — see the README there for provenance and credits.
```
