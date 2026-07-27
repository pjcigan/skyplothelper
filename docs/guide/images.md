# Images & FITS

Displaying astronomical images well is mostly about *scaling*: choosing
the interval (which pixel values map to the ends of the colormap) and the
stretch (how values distribute between them) so faint structure shows
without burning out the peak. This page covers that stack, the one-call
quicklook figures built on it, reprojection of FITS and RGB imagery onto
sky frames, and the FITS-header utilities everything leans on.

```python
import skyplothelper as sph
```

## Interval + stretch: the scaling stack

```{image} /_static/features/fits-image-display-light.png
:class: sph-plot plot-light dark-light
:alt: A FITS image displayed with an interval and stretch (light mode)
```
```{image} /_static/features/fits-image-display-dark.png
:class: sph-plot plot-dark dark-light
:alt: A FITS image displayed with an interval and stretch (dark mode)
```
*{doc}`FITS image display </features/fits-image-display>` — code in the Feature Gallery.*

Two orthogonal choices, exposed at three levels of convenience:

**Arrays in, arrays out** — {func}`~skyplothelper.rescale_image` applies
both choices and returns the scaled array:

```python
scaled = sph.rescale_image(data, stretch="asinh", clip="zscale")
```

`clip=` picks the interval (`'percentile'` with `plo=`/`phi=`,
`'sigma'`, `'zscale'`, or explicit `vmin=`/`vmax=`); `stretch=` the
mapping — `'linear'`, `'sqrt'`, `'squared'`, `'log'`, `'asinh'`,
`'sinh'`, `'power'` (with an `a=` exponent), `'histeq'` (histogram
equalization, needs astropy's `HistEqStretch`), and the signed-data
`'symlog'`/`'symmetric_log'`. {func}`~skyplothelper.list_stretches`
**prints** the full table (it doesn't return one).

**A norm for your own `imshow`** — {func}`~skyplothelper.make_norm` makes
the same choices as a matplotlib normalization, so the original data
stays untouched (colorbars then read true values):

```python
norm = sph.make_norm(stretch="asinh", clip="zscale", data=img)
ax.imshow(img, norm=norm, cmap="sph.deepsky")
```

(`"sph.deepsky"` is one of the astronomy-inspired colormaps skyplothelper 
registers under the `sph.` prefix on import — see {doc}`styling`; any matplotlib
colormap works just as well.)

**Decided for you** — {func}`~skyplothelper.auto_stretch` inspects the
data distribution and returns a `(stretch, reason)` pair — a recommended
stretch name plus a one-line explanation you can pass straight to
{func}`~skyplothelper.rescale_image`;
{func}`~skyplothelper.describe_image` prints the statistics behind such
decisions. The lower-level pieces
({func}`~skyplothelper.clip_percentile`, {func}`~skyplothelper.clip_sigma`,
{func}`~skyplothelper.clip_zscale`, {func}`~skyplothelper.auto_interval`,
{func}`~skyplothelper.adjust_gamma`,
{func}`~skyplothelper.rescale_percentile`) are public for custom
pipelines.

## Quicklook figures

```{image} /_static/features/quicklook-in-one-call-light.png
:class: sph-plot plot-light dark-light
:alt: A one-call quicklook figure of the 3C 84 radio jet (light mode)
```
```{image} /_static/features/quicklook-in-one-call-dark.png
:class: sph-plot plot-dark dark-light
:alt: A one-call quicklook figure of the 3C 84 radio jet (dark mode)
```
*{doc}`Quicklook in one call </features/quicklook-in-one-call>` — code in the Feature Gallery.*

One call from a FITS file to a presentable annotated figure, inspired by the
classic *difmap* plots — WCS axes,
synthesized beam, scale bar, and source/date/peak info text:

```python
res = sph.quicklook_fits("crab_xband.fits", stretch="asinh")
res.ax.set_title("Crab Nebula")
```

{func}`~skyplothelper.quicklook_fits` takes a path;
{func}`~skyplothelper.quicklook_plot` an array (+ header or WCS) onto an
optional existing axes — or a {class}`~skyplothelper.MomentMap`, auto-detected
and drawn with moment-aware defaults (see [Channel maps](#channel-maps));
{func}`~skyplothelper.quicklook_figure` builds figure and plot together. All return a
{class}`~skyplothelper.QuicklookResult` holding the figure, axes, image,
and annotation artists for follow-up tweaks. The
{func}`~skyplothelper.simpleimageplot` /
{func}`~skyplothelper.simpleimage_figure` variants skip the radio-style
annotations for a minimal image-on-WCS-axes display.

By default a quicklook shows **absolute** sexagesimal RA/Dec ticks. For a
high-magnification field, `offset_coords=True` switches to *relative*
offset ticks (Δα·cos δ, Δδ) measured from a reference — the image center,
or an explicit `ref_coord` — the standard VLBI/HST convention; adding
`field_size` then crops to a symmetric window about that reference:

```python
# Absolute coordinates (default)
sph.quicklook_fits("vlba_3c84.fits", stretch="asinh")

# Relative offset ticks in mas (VLBI convention)
sph.quicklook_fits("vlba_3c84.fits", stretch="asinh",
                   offset_coords=True, offset_units="mas")

# ...cropped to an 8 mas window about the reference
sph.quicklook_fits("vlba_3c84.fits", stretch="asinh",
                   offset_coords=True, offset_units="mas", field_size=8)
```

`image=`, `contours=`, and `colorbar=` all default to `True`, so a quicklook
is subtractive rather than additive — ask for less, not more:

```python
sph.quicklook_fits("vlba_3c84.fits", image=False)     # contours only
sph.quicklook_fits("vlba_3c84.fits", colorbar=False)  # no colorbar
```

The axis labels track the mode automatically ("Right Ascension" /
"Declination" vs "Relative RA (mas) from …"). `offset_units='auto'` picks
the unit from the field of view, `field_size` is in `offset_units` when
`offset_coords=True` (else mas), and `tick_style=` styles the absolute
ticks. Quicklook draws inward ticks by default (baking the professional
tick intent that WCSAxes otherwise drops).

Over bright imagery, `frame_stroke=` (a color, or a `{'color', 'lw'}`
dict) strokes the spine and tick marks for legibility;
{func}`~skyplothelper.apply_frame_stroke` applies the same stroke to any
already-built WCSAxes or plain axes (safe to call more than once; it leaves
tick labels untouched — stroke those via
{func}`~skyplothelper.format_ticklabels`).

## Channel maps

```{image} /_static/features/data-cube-channel-maps-light.png
:class: sph-plot plot-light dark-light
:alt: A grid of spectral-cube channel maps (light mode)
```
```{image} /_static/features/data-cube-channel-maps-dark.png
:class: sph-plot plot-dark dark-light
:alt: A grid of spectral-cube channel maps (dark mode)
```
*{doc}`Data cube channel maps </features/data-cube-channel-maps>` — code in the Feature Gallery.*

{func}`~skyplothelper.channel_map` is the cube companion to quicklook: hand it
a spectral cube (an array, HDU, or FITS path — degenerate axes are squeezed)
and it lays the channels out as a uniform panel grid, doing all the scaffolding
a channel-map figure needs — channel selection, one shared normalization,
per-panel velocity labels, an integrated colorbar, and sparse shared ticks:

```python
res = sph.channel_map("ngc_hi_cube.fits", channels=9, ncols=3,
                      cmap="sph.dusk")
```

The one **shared norm** across every panel is the whole point — channels are
only comparable when they're on the same scale, so `channel_map` derives one
norm from the full cube (override with `vmin`/`vmax`/`stretch`/`norm`). Ticks
are sparse by default (`ticks='minimal'` labels one panel and marks the rest);
`ticks='complete'` widens the spacing and shrinks the font so per-panel labels
don't collide. Turn on the extras with `beam=True` (from the header), a
`scalebar=` in arcsec, and `moment0=True` — the moment-0 panel deliberately
keeps its **own** norm (an integrated ∫I dv map is far brighter than any single
channel), which is why it's the one panel not tied to the shared colorbar.

It returns a {class}`~skyplothelper.ChannelMapResult` (tuple-unpackable, with
`fig`/`axes`/`images`/`colorbar`/`norm`/… and a `.panel(channel)` accessor).
There is deliberately no `contours=` argument: per-panel contours or a custom
comparison panel are done *after the fact* by looping over `res.axes` or
editing `res.moment0_image`, which is far more flexible than any constructor
knob could be. `moment0_units` gives the units string (BUNIT × spectral unit)
for a post-hoc moment-0 colorbar — draw one with `sph.add_colorbar(
res.moment0_image, ax=res.moment0_image.axes, mode="inset", location="left",
label=res.moment0_units)`. (On a grid panel `mode="divider"` auto-falls back to
`"inset"` with a warning, since the panel already owns a locator; passing
`"inset"` directly avoids the warning.)

Under the hood the cube plumbing lives in {class}`~skyplothelper.DataCube` — a
thin **(data + WCS) holder**, not a plotting object. It does the load + squeeze
of degenerate (Stokes) axes to `(channel, y, x)`, splits the celestial and 1-D
spectral sub-WCS, classifies the spectral axis (velocity / frequency /
wavelength), and exposes per-channel world values with velocity ↔ frequency ↔
redshift conversion — the FITS-cube handling every cube tool would otherwise
re-implement. `channel_map` accepts either a raw cube (array / HDU / path) *or*
a `DataCube`, so you usually never build one explicitly. Reach for it when you
want to *preprocess* first: its transforms — `spectral_bin(n)`,
`spatial_downsample(factor)` (celestial WCS rescaled so overlays stay
registered), `smooth(width)` — each return a **new** `DataCube`, so they chain
immutably:

```python
cube = sph.DataCube.from_fits("co.fits")
small = cube.spatial_downsample(2).spectral_bin(3)   # chainable, immutable
sph.channel_map(small, channels=9)
```

Collapse a cube to a moment map with `moment(order=0|1|2)` (or the
`moment0()` / `moment1()` / `moment2()` wrappers): order 0 is integrated
intensity, 1 the intensity-weighted velocity field, 2 the velocity dispersion.
**Moments 1 and 2 need a `threshold=`** (a scalar signal cut) or they come out
noise-dominated — real masking is deliberately left to `spectral_cube`. Each
returns a {class}`~skyplothelper.MomentMap`, a small record
(`data`/`units`/`wcs`/`order`/`header`) that knows how to plot and persist
itself: `.plot()` draws it on a proper sph frame (order-aware defaults — a
diverging velocity field symmetric about its median, sequential for m0/m2, with
the header beam and an optional scale bar), `.from_fits()`/`.to_fits()`
round-trip a map to disk keeping the beam and `OBJECT` provenance, and `.name`
gives a title like `'moment 1 (velocity field)'`.
{func}`~skyplothelper.quicklook_plot` also accepts a `MomentMap` directly,
applying the same moment-aware defaults:

```python
vfield = cube.moment1(threshold=3 * cube_rms)   # MomentMap
vfield.plot(title=vfield.name)                  # or: sph.quicklook_plot(vfield)
```

`DataCube` stays deliberately thin — no reprojection or masking (that's
[`spectral_cube`](https://spectral-cube.readthedocs.io)'s job; a
`SpectralCube` is *accepted* as input but never required). It's the shared cube
core that `channel_map` and the planned interactive cube viewer both consume —
a vetted core with the rendering left to each backend.

## Backdrops & reprojection

Putting imagery *into* a frame — an all-sky panorama behind your data, or
a FITS image displayed in a different projection/coordinate system
(requires the `reproject` extra):

```python
img, hdr = sph.load_sky_image("allsky_pano.jpg")     # equirectangular RGB
fig, ax = sph.allsky_figure(projection="MOL")
bg = sph.reproject_background(img, hdr, ax)          # resample onto the frame
ax.imshow(bg)
```

{func}`~skyplothelper.load_sky_image` wraps an equirectangular image
(Mellinger-style sky panoramas, planet textures) in a synthetic WCS;
{func}`~skyplothelper.reproject_background` resamples that image (RGB,
RGBA, or single-channel) onto an axes' (or header's) WCS, and
{func}`~skyplothelper.reproject_rgb_map` is the variant that takes an RGB
FITS HDU directly. ({func}`~skyplothelper.load_sky_image`'s sibling
{func}`~skyplothelper.pseudofits_from_image` does the same WCS-wrapping for
flat-map images destined for a globe — see {doc}`globe`.) The same
machinery displays a FITS image in another coordinate system: reproject
onto a frame built in the target system, and the gridlines, overlays, and
labels all follow that frame.

For a crisp high-resolution backdrop, build the target frame with a finer
pixel grid via `npix=(NAXIS1, NAXIS2)` (matched to the frame's aspect —
~2:1 for an all-sky oval): the image is resampled onto the frame's own
grid, so the builders' coarse default can look pixelated for large images.

## FITS-header utilities

The header helpers used throughout the package are public:

- **Pixel scales** — {func}`~skyplothelper.getcdelts`,
  {func}`~skyplothelper.getdegperpix`, {func}`~skyplothelper.getasecperpix`,
  {func}`~skyplothelper.getsteradperpix`, {func}`~skyplothelper.getcdmatrix`.
- **Beams** — {func}`~skyplothelper.beampars_asec_fromhdr`,
  {func}`~skyplothelper.pixperbeam_from_hdr`,
  {func}`~skyplothelper.pixperbeam_from_pars`.
- **Pixel ↔ sky** — {func}`~skyplothelper.convsky2pix`,
  {func}`~skyplothelper.convpix2sky`,
  {func}`~skyplothelper.header_coord_grids`.
- **Construction & cleanup** — {func}`~skyplothelper.makesimpleheader`
  (a minimal valid header from scratch; it preserves any rotation carried in
  a `CD`/`PC` matrix as a standard `CROTA2`, and warns if a rotated image has
  non-square pixels, since `CDELT`+`CROTA2` cannot encode that skew),
  {func}`~skyplothelper.force_hdr_to_2D` /
  {func}`~skyplothelper.force_hdr_to_3D` /
  {func}`~skyplothelper.squeeze_image` (taming cube headers with
  degenerate axes), {func}`~skyplothelper.force_hdr_floats`.
- **Inspection** — {func}`~skyplothelper.describe_wcs` prints a friendly
  summary of any header or WCS.

## Pitfalls

- **Signed data under a log/asinh stretch** — residuals, velocity, and
  Stokes Q/U need the symmetric stretches (`'symlog'` /
  `'symmetric_log'`), not a positive-only mapping that clips half the
  signal.
- **Colorbar showing stretched values** — that's the
  `rescale_image`-then-`imshow` route; use {func}`~skyplothelper.make_norm`
  instead so matplotlib knows the true data values.
- **A 2-D plot from a 3-D cube failing** — squeeze degenerate axes first
  ({func}`~skyplothelper.squeeze_image` +
  {func}`~skyplothelper.force_hdr_to_2D`).
- **Quicklook without beam/scale annotations** — those need header
  metadata (`BMAJ`/`BMIN`/`BPA`, pixel scale); with bare arrays, pass
  the values explicitly or use the simpleimage variants.
- **Reprojection `ImportError`** — backdrops and cross-system display
  need the `reproject` extra.

Full listings: {doc}`images API <../api/images>` and {doc}`FITS-header
API <../api/fits>`.

**See also:** {doc}`frames` (build the target frame to display into) and
{doc}`overlays` (beams, colorbars, and contour annotations).

**Tutorial:** {doc}`FITS images & quicklook </tutorials/fits_images>` works
through the interval/stretch scaling stack, colorbars that read true values,
σ-spaced contours, one-call quicklook figures, and multi-band RGB composites.
