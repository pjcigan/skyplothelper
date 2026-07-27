# Beam from a PSF fit

```{image} /_static/features/beam-from-a-psf-fit-light.png
:class: sph-plot plot-light dark-light
:alt: Beam from a PSF fit (light mode)
```


```{image} /_static/features/beam-from-a-psf-fit-dark.png
:class: sph-plot plot-dark dark-light
:alt: Beam from a PSF fit (dark mode)
```


Fit a `Beam` to a synthetic elliptical-Gaussian PSF with `Beam.from_psf_fit`,
draw its recovered FWHM ellipse on the image (an sph colormap here), and drop a
scale-matched PSF inset (asinh stretch, revealing the faint negative ring) with
`Beam.add_psf_inset`.

Guide: {doc}`/guide/overlays` — API: {py:obj}`~skyplothelper.Beam`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph
from skyplothelper import Beam

sph.set_style(base="structural")   # inward ticks on both axes

# A synthetic elliptical-Gaussian PSF (14 x 7 px FWHM, tilted 25 deg) with a
# faint negative sidelobe ring, on a padded field so the main view is zoomed out.
rng = np.random.default_rng(0)
nx = ny = 161
yy, xx = np.mgrid[0:ny, 0:nx]
cx = cy = (nx - 1) / 2.0
th = np.radians(25.0)
c, s = np.cos(th), np.sin(th)
xr = (xx - cx) * c + (yy - cy) * s
yr = -(xx - cx) * s + (yy - cy) * c
core = np.exp(-0.5 * ((xr / (7.0 / 2.3548)) ** 2 + (yr / (14.0 / 2.3548)) ** 2))
r = np.sqrt(xr ** 2 + yr ** 2)
ring = -0.08 * np.exp(-0.5 * ((r - 22.0) / 4.0) ** 2)
psf = core + ring + 0.004 * rng.standard_normal((ny, nx))

fig, ax = plt.subplots(figsize=(6.4, 6.4))
ax.imshow(psf, cmap="sph.deepsky", origin="lower")
ax.set_xlabel("x (pix)"); ax.set_ylabel("y (pix)")
ax.set_title("Beam.from_psf_fit + add_psf_inset")

# Fit a Beam to the PSF and draw its FWHM ellipse on the image; the fit recovers
# the shape and orientation, so the ellipse hugs the bright core.
beam = Beam.from_psf_fit(psf, xy=(cx, cy), style="crosshair", ec="white", lw=1.4,
                         crosshair_color="white", crosshair_lw=0.9,
                         stroke_color="0.1", stroke_lw=2.0)
beam.add_to(ax)

# A PSF inset at the SAME physical scale as the zoomed-out main view: pass a
# central crop sized to the inset's on-page fraction, asinh-stretched to reveal
# the faint ring the linear main view hides.
cr = 27
crop = psf[int(cy) - cr:int(cy) + cr + 1, int(cx) - cr:int(cx) + cr + 1]
inset = beam.add_psf_inset(ax, crop, size="34%", loc="upper left",
                           stretch="asinh", cmap="viridis", show_beam=True,
                           beam_kwargs={"style": "ellipse", "ec": "white", "lw": 1.3},
                           border_color="white", border_lw=1.1)
inset.set_xlabel("asinh PSF", fontsize=8)
inset.set_xticks([0, 25, 50]); inset.set_yticks([0, 25, 50])
inset.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True,
                  labelsize=6)
```
