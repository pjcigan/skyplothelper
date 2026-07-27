---
sd_hide_title: true
---

# skyplothelper

```{toctree}
:hidden:

installation
guide/index
features/index
tutorials/index
api/index
Releases <changelog>
```

# skyplothelper

**Astronomy visualization toolkit for matplotlib + astropy WCSAxes.**

All-sky projections, custom WCS frames, spherical-region geometry, tilted
globes, cone (z-RA) wedge plots, HEALPix utilities, FITS image quicklook, an
interactive plotly/Dash export backend, and the everyday plot annotations that
go with them — from a single `import skyplothelper as sph`.

```{code-block} python
import skyplothelper as sph
import matplotlib.pyplot as plt

fig, ax = sph.allsky_figure(projection="AIT", center=180)
sph.add_plane_overlay(ax, plane="ecliptic", color="orange")
sph.add_constellation_boundaries(ax)
sph.add_survey_footprint(ax, survey="sdss")
plt.show()
```

```{tip}
**New here — human or AI agent?** Call `sph.overview()` for the frame-first
model and coordinate conventions, and `sph.recipes('<keyword>')` (e.g.
`sph.recipes('cube')`) for copy-paste recipes. Agents can ingest the same
catalog from <a href="llms.txt"><code>llms.txt</code></a> (a concise map) and
<a href="llms-full.txt"><code>llms-full.txt</code></a> (the full recipe corpus)
served at the site root.
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`download` Installation
:link: installation
:link-type: doc

Core install is just numpy + matplotlib + astropy; optional features are
per-extra. One line grabs everything.
:::

:::{grid-item-card} {octicon}`book` User guide
:link: guide/index
:link-type: doc

The package explained subsystem by subsystem — what each piece does, how it
fits together, worked examples, and the gotchas.
:::

:::{grid-item-card} {octicon}`image` Feature Gallery
:link: features/index
:link-type: doc

A visual index of what skyplothelper can draw — all-sky maps, fields, globes,
cones, HEALPix, regions, legends, and more — with starter code for each.
:::

:::{grid-item-card} {octicon}`mortar-board` Tutorials
:link: tutorials/index
:link-type: doc

In-depth, runnable walkthroughs of complete workflows — from your first
frames to catalogs, globes, HEALPix, and interactive maps.
:::

:::{grid-item-card} {octicon}`code-square` API reference
:link: api/index
:link-type: doc

Every public function and class, grouped by subsystem, with auto-generated
signatures from the source.
:::

:::{grid-item-card} {octicon}`tag` Releases
:link: changelog
:link-type: doc

Release notes and version history.
:::

::::

## About

skyplothelper grew out of years of day-to-day research plotting — the
recurring need to put astronomical data on the sky *correctly* (projections,
coordinate conventions, the antimeridian seam) and *presentably* (publication
ticks, beams, scale bars, refined styling) without rebuilding the same
scaffolding for every paper. It builds on astropy's WCSAxes rather than
replacing it: every frame is a real `WCSAxes` you can keep customizing with
the full matplotlib + astropy toolbox.

The package covers the typical sky-plot workflow end to end — frame
construction across 32 projections, coordinate conversions and FITS-header
utilities, tick/grid/label control, overlays (coordinate planes, survey
footprints, constellations, beams, rulers, reticles), set-algebraic spherical
regions, globe and planetary maps, cosmology cone diagrams, HEALPix
binning and rendering, catalog queries, vector fields on the sphere, and an
interactive plotly backend that mirrors the matplotlib API.

```bash
pip install skyplothelper            # core: numpy, matplotlib, astropy>=6.0, shapely, healpy
pip install skyplothelper[all]       # everything optional
```

skyplothelper is BSD-3-Clause licensed. Cite via the repository's
`CITATION.cff`. Bug reports and contributions are welcome on
[GitHub](https://github.com/pjcigan/skyplothelper) — it's maintained by a
single developer with limited time, so responses come as time allows; the
[contributing
guide](https://github.com/pjcigan/skyplothelper/blob/main/CONTRIBUTING.md)
covers what to expect.
