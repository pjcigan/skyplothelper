# Cone frames

Cone ("pie wedge") frames are the classic redshift-survey diagram: an
angular sky coordinate opens the wedge, and redshift or distance runs
along the radius, with the observer at the apex. Unlike everything else in
the package, these are *not* WCS frames — they're purpose-built polar
wedges with their own tick, label, and plotting machinery (so sky overlays
like constellations don't apply here; the cone has its own helpers).

```python
import skyplothelper as sph
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 5))
ax = sph.make_cone_frame(
    111, angle_center=180, angle_half_width=30,
    r_min=0, r_max=0.15,                      # redshift range
    angle_label="R.A.", fig=fig,
)
sph.cone_scatter(ax, galaxy_ras, galaxy_redshifts, s=3)
```

## Building the wedge

```{image} /_static/features/redshift-cone-light.png
:class: sph-plot plot-light dark-light
:alt: A redshift cone — a z-RA wedge diagram (light mode)
```
```{image} /_static/features/redshift-cone-dark.png
:class: sph-plot plot-dark dark-light
:alt: A redshift cone — a z-RA wedge diagram (dark mode)
```
*{doc}`Redshift cone </features/redshift-cone>` — code in the Feature Gallery.*

{func}`~skyplothelper.make_cone_frame`'s geometry arguments:
`angle_center=` and `angle_half_width=` set the angular opening (degrees,
typically RA), `r_min=`/`r_max=` the radial range, and `zero_location=` /
`angle_direction=` / `zero_offset=` orient the wedge (where the zero angle
sits and which way it increases). The radial coordinate is declared, not
just ranged: `r_variable=` (`'redshift'` by default, or a distance) with
`r_unit=` and an optional astropy `cosmology=` for conversions. Label
alignment, padding, tick spacing, and grid styling all have dedicated
knobs — the defaults are publication-sensible.

## Double-sided cones: the bowtie

```{image} /_static/features/bowtie-diagram-light.png
:class: sph-plot plot-light dark-light
:alt: A double-sided cone (bowtie) diagram (light mode)
```
```{image} /_static/features/bowtie-diagram-dark.png
:class: sph-plot plot-dark dark-light
:alt: A double-sided cone (bowtie) diagram (dark mode)
```
*{doc}`Bowtie diagram </features/bowtie-diagram>` — code in the Feature Gallery.*

{func}`~skyplothelper.make_bowtie_frame` builds the two-wedge variant —
opposite sky regions sharing a common apex, the classic layout for
surveys that span both galactic caps (the CfA "stick man" and its
descendants). It returns the two halves, each a normal cone frame that
every helper on this page works on independently:

```python
fig = plt.figure(figsize=(7, 7))
ax_top, ax_bot = sph.make_bowtie_frame(
    angle_center=195, angle_half_width=45,
    r_min=0, r_max=0.05, angle_label="R.A.", fig=fig,
)
sph.cone_scatter(ax_top, north_ras, north_zs, s=2)
sph.cone_scatter(ax_bot, south_ras, south_zs, s=2)
```

`orientation=` flips the layout between vertical (wedges opening up and
down) and horizontal (left and right).

## Plotting in the wedge

- {func}`~skyplothelper.cone_scatter` — `cone_scatter(ax, angle, r)`;
  the workhorse. {func}`~skyplothelper.cone_scatter_z` color-maps a third
  quantity.
- {func}`~skyplothelper.cone_plot` — connected lines (boundaries,
  tracks).
- {func}`~skyplothelper.cone_hexbin` /
  {func}`~skyplothelper.cone_pcolormesh` — density renderings. Hexbin
  suits scattered galaxy samples (it bins in *screen* space, so bins stay
  visually uniform); pcolormesh suits data already gridded in
  (angle, r).

## The radial axis

The radial direction is where cone plots earn their keep:

- {func}`~skyplothelper.make_twinr` adds a second radial scale alongside
  the first, defined by a conversion *function* — e.g. redshift on one
  side, comoving Mpc on the other, with
  {func}`~skyplothelper.redshift_to_r` supplying the cosmology
  conversion:

  ```python
  from astropy.cosmology import Planck18
  sph.make_twinr(
      ax,
      convert=lambda z: sph.redshift_to_r(z, r_variable="comoving_distance",
                                          cosmology=Planck18, r_unit="Mpc"),
      r_label="Comoving distance [Mpc]")
  ```

  (Leaving `r_variable='redshift'` — the default — returns `z` unchanged;
  pass `r_variable="comoving_distance"` with a `cosmology=` for an actual
  distance conversion.)

- {func}`~skyplothelper.log_r` switches the radial coordinate to a log
  scale (deep samples with dense low-z foregrounds).
- {func}`~skyplothelper.add_minor_rticks` adds minor radial ticks;
  {func}`~skyplothelper.flip_label` and
  {func}`~skyplothelper.set_label_pad` /
  {func}`~skyplothelper.get_label_pad` fine-tune label orientation and
  spacing on the slanted spines.

## Pitfalls

- **Sky helpers don't work here** — cone frames aren't WCSAxes; no
  constellation overlays, no `get_transform("world")`. Use the `cone_*`
  plotters.
- **Angles are degrees** — pass RA in degrees, not hours, regardless of
  the `angle_label`.
- **Hexbin vs. pcolormesh confusion** — hexbin bins points (give it the
  raw catalog); pcolormesh draws an existing 2-D grid. Feeding a catalog
  to pcolormesh is the most common mix-up.
- **Distance conversions need a cosmology** — `redshift_to_r` and
  `cosmology=` accept an astropy cosmology; the conversion extras need
  scipy (`pip install skyplothelper[cone]`).

Full listing: {doc}`API reference <../api/cone>`.

**See also:** {doc}`frames` — for the WCS sky frames that cone frames are
contrasted with throughout this page (cone frames are polar wedges, not
WCSAxes).

**Tutorial:** {doc}`Cone & bowtie plots </tutorials/cone_bowtie>` builds and
orients a redshift wedge, plots catalogs as points, tracks, and density,
adds the double-sided bowtie, and pairs redshift with comoving distance on a
twin radial axis.
