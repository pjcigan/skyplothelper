<!-- Images are referenced by absolute raw.githubusercontent.com URLs so they
     render both on GitHub and on PyPI (this file is the PyPI long description).
     They resolve once the repo is public on the `main` branch.
     NOTE: the hero (a diagonally-spliced all-sky composite), the body gallery
     mosaic, and the two animations are PLACEHOLDERS — regenerate the hero /
     gallery with docs/make_readme_assets.py and swap in the final hand-picked
     figures/animations during the art polish pass. -->

<div align="center">

# skyplothelper

**Put astronomical data on the sky — correctly and beautifully — from a single `import`.**

All-sky projections · custom WCS frames · tilted globes & planets · cosmology cone diagrams ·
HEALPix · spherical-region set algebra · FITS quicklook · rich overlays · an interactive plotly/Dash backend

[![Docs](https://img.shields.io/badge/docs-latest-8CA1AF?logo=readthedocs&logoColor=white)](https://skyplothelper.readthedocs.io)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://opensource.org/license/bsd-3-clause)
[![Powered by Astropy](https://img.shields.io/badge/powered%20by-Astropy-EE7918?logo=astropy&logoColor=white)](https://www.astropy.org)
[![Code style: Ruff](https://img.shields.io/badge/lint-ruff-261230?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![Typed: mypy strict](https://img.shields.io/badge/mypy-strict-2A6DB2.svg)](https://mypy-lang.org/)
[![Sponsor](https://img.shields.io/badge/sponsor-%F0%9F%A7%AA-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/pjcigan)

<!-- Enable on release:
[![PyPI](https://img.shields.io/pypi/v/skyplothelper?logo=pypi&logoColor=white)](https://pypi.org/project/skyplothelper/)
[![Downloads](https://static.pepy.tech/badge/skyplothelper)](https://pepy.tech/project/skyplothelper)
[![DOI](https://zenodo.org/badge/DOI/00.0000/zenodo.0000000.svg)](https://doi.org/00.0000/zenodo.0000000)
[![ASCL](https://img.shields.io/badge/ascl-0000.000-blue.svg)](https://ascl.net/0000.000)
-->

[**Documentation**](https://skyplothelper.readthedocs.io) ·
[Tutorials](https://skyplothelper.readthedocs.io/en/latest/tutorials/index.html) ·
[Gallery](https://skyplothelper.readthedocs.io/en/latest/features/index.html) ·
[API](https://skyplothelper.readthedocs.io/en/latest/api/index.html) ·
[Quickstart](#quickstart) ·
[Citing](#citing)

<br>

<img src="https://raw.githubusercontent.com/pjcigan/skyplothelper/main/docs/_static/readme/hero.jpg" alt="One all-sky Aitoff map, diagonally spliced from three identically-projected panels: a source-density field with the galactic plane, bright stars with constellation boundaries and the ecliptic, and survey footprints over a redshift catalog." width="100%">

<sub>One all-sky frame, several layers at once — a source-density field, bright stars and
constellations, and survey footprints over a redshift catalog — spliced from identically-projected panels.</sub>

</div>

---

## What is it?

**skyplothelper** is an astronomical data visualization toolkit built on **matplotlib** and
**astropy's [WCSAxes](https://docs.astropy.org/en/stable/visualization/wcsaxes/)**. It handles
the parts of sky plotting that are fiddly to get right — projections and their antimeridian
seams, coordinate-system conventions, the astronomical east-left longitude direction,
publication-quality ticks — and packs the everyday sky-figure toolkit into one place.

A single `import skyplothelper as sph` gives you 200+ helpers spanning all-sky maps, tangent-plane
fields, tilted globes, cosmology cones, HEALPix, spherical regions, FITS quicklook, and a full
overlay vocabulary — without juggling a handful of separate libraries. Crucially, it builds *on*
WCSAxes rather than around it: **every frame it returns is a real `WCSAxes`**, so the entire
matplotlib + astropy toolbox keeps working on top of anything skyplothelper draws.

```python
import skyplothelper as sph

# An all-sky map with the ecliptic, IAU constellation boundaries, and a survey footprint
fig, ax = sph.allsky_figure(projection="AIT", center=180)
sph.add_plane_overlay(ax, plane="ecliptic", color="orange")
sph.add_constellation_boundaries(ax)
sph.add_survey_footprint(ax, survey="sdss", label="SDSS")
```

## Highlights

- **All-sky & field maps** — 32 projections (Aitoff, Mollweide, Plate Carrée, conics, …),
  tangent-plane fields, and offset (relative) coordinates.
  → [frames guide](https://skyplothelper.readthedocs.io/en/latest/guide/frames.html)
- **Globes & planets** — tilted-Earth orientation with Euler angles, hemisphere-aware plotting,
  coastlines / tectonic plates, nightshade day/night blending, and planet textures.
  → [globe guide](https://skyplothelper.readthedocs.io/en/latest/guide/globe.html)
- **FITS images** — interval/stretch scaling, one-call quicklook figures, beams, matched colorbars,
  reprojection, and multi-band RGB composites.
  → [images guide](https://skyplothelper.readthedocs.io/en/latest/guide/images.html)
- **Spherical regions** — circles, polygons, and bands with correct seam/pole handling, plus
  set-algebra (`CompoundRegion`) and point-in-region membership queries.
  → [regions guide](https://skyplothelper.readthedocs.io/en/latest/guide/regions.html)
- **HEALPix** — bin catalogs into maps, render them in *any* projection, run spatial queries,
  and change resolution.
  → [HEALPix guide](https://skyplothelper.readthedocs.io/en/latest/guide/healpix.html)
- **Overlays** — constellations, survey footprints, coordinate planes, beams, rulers, reticles,
  compasses, and scale bars — all seam-aware.
  → [overlays guide](https://skyplothelper.readthedocs.io/en/latest/guide/overlays.html)
- **Cosmology cones** — redshift-survey wedge diagrams, double-sided bowties, and a twin radial
  axis pairing redshift with comoving distance.
  → [cone guide](https://skyplothelper.readthedocs.io/en/latest/guide/cone.html)
- **Coordinates & ticks** — RA/Dec conventions, sexagesimal / decimal / offset / VLBI tick styles,
  and second coordinate grids drawn over a frame.
  → [ticks guide](https://skyplothelper.readthedocs.io/en/latest/guide/ticks.html)
- **Vector fields** — proper motions, displacement fields, vector spherical harmonics, and station
  co-visibility regions.
  → [vectors guide](https://skyplothelper.readthedocs.io/en/latest/guide/vectors.html)
- **Catalog queries** — thin SIMBAD / NED / VizieR / SkyView wrappers whose results drop straight
  onto a frame.
  → [queries guide](https://skyplothelper.readthedocs.io/en/latest/guide/queries.html)
- **Interactive backend** — `skyplothelper.plotly` mirrors the same API on interactive plotly
  figures (pan, zoom, hover, single-file HTML export) with a Dash FITS viewer.
  → [plotly guide](https://skyplothelper.readthedocs.io/en/latest/guide/plotly.html)
- **Publication styling** — composable base / theme / palette / font layers, color-vision-safe
  palettes, and a set of astronomy image colormaps.
  → [styling guide](https://skyplothelper.readthedocs.io/en/latest/guide/styling.html)

## Gallery

From all-sky maps to tilted globes, planets, redshift cones, HEALPix maps, and set-algebra
regions — a sampler of what fits on one canvas:

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pjcigan/skyplothelper/main/docs/_static/readme/gallery-dark.jpg">
  <img src="https://raw.githubusercontent.com/pjcigan/skyplothelper/main/docs/_static/readme/gallery-light.jpg" alt="A photo-wall mosaic of skyplothelper figures: a tilted Earth at dusk, ICRF3 radio sources over the Milky Way, the SN 1987A ring, a colormapped graticule, the 3C 84 jet, constellation charts in galactic coordinates, the Virgo cluster, a galactic-aberration vector field, all-sky postage-stamp insets, a VLBI-visibility region, a redshift bowtie, a multi-channel VLBI catalog, a HEALPix density map, and the Messier catalog over the Milky Way." width="100%">
</picture>
<br>
<sub>Deeper walkthroughs of these figures are in the
<a href="https://skyplothelper.readthedocs.io/en/latest/tutorials/index.html">tutorials</a>;
a code-per-figure index is in the
<a href="https://skyplothelper.readthedocs.io/en/latest/features/index.html">feature gallery</a>.</sub>
</div>

## In motion

The same frames animate — spin a planet through Euler-angle sequences, sweep the nightshade
terminator across a date range, trace stellar orbits around the Galactic-center black hole,
plan a VLBI session, step through a spectral cube, or watch a constellation deform under
proper motion over millennia.

<div align="center">
<img src="https://raw.githubusercontent.com/pjcigan/skyplothelper/main/docs/_static/animations/animations__earth-day-and-night.webp" alt="A rotating Earth showing the day/night terminator and night-side city lights" width="30%">
&nbsp;
<img src="https://raw.githubusercontent.com/pjcigan/skyplothelper/main/docs/_static/animations/animations__mars-spin.webp" alt="Mars rotating at its true axial tilt" width="30%">
&nbsp;
<img src="https://raw.githubusercontent.com/pjcigan/skyplothelper/main/docs/_static/animations/animations__sstar-orbits.webp" alt="Stars on elliptical orbits around the Galactic-center black hole" width="30%">
<br>
<img src="https://raw.githubusercontent.com/pjcigan/skyplothelper/main/docs/_static/animations/animations__dipper-morph.webp" alt="The Big Dipper changing shape over 200,000 years of stellar proper motion" width="30%">
&nbsp;
<img src="https://raw.githubusercontent.com/pjcigan/skyplothelper/main/docs/_static/animations/animations__covisibility-day.webp" alt="Mutual-visibility windows across a VLBI station network over one day" width="30%">
&nbsp;
<img src="https://raw.githubusercontent.com/pjcigan/skyplothelper/main/docs/_static/animations/animations__cube-channels.webp" alt="A spectral cube stepping through its velocity channels as a movie" width="30%">
<br>
<sub>Animated WebP built with matplotlib animation on ordinary skyplothelper frames — see the
<a href="https://skyplothelper.readthedocs.io/en/latest/tutorials/animations.html">animations tutorial</a>.</sub>
</div>

## Why skyplothelper?

Plenty of good tools already put data on the sky — each excellent within its lane:

- **[APLpy](https://aplpy.github.io/)** makes beautiful figures of FITS *images*, and does that one
  job very well — but it's focused on single-image display (images, contours, RGB, beams), is in
  maintenance mode, and works through its own `FITSFigure` object rather than a general axes you
  keep extending.
- **[The Kapteyn Package](https://www.astro.rug.nl/software/kapteyn/)** is a powerful, mature
  mapping toolkit, but it's a self-contained framework with its own WCS and plotting classes — a
  parallel ecosystem to astropy rather than a thin layer on top of it.
- **[pywcsgrid2](https://github.com/leejjoon/pywcsgrid2)** pioneered WCS-aware matplotlib axes and
  inspired a generation of sky plots, but it's Python-2-only and unmaintained.
- Beyond these, most tools are deliberately narrow — `healpy.mollview` for HEALPix, `cartopy` for
  the Earth, planetarium/ephemeris libraries like `skyfield` and `starplot` for star charts. Each
  is great at its specialty.

skyplothelper takes a different tack. It builds **on** astropy's WCSAxes rather than around it, so
every frame is a native `WCSAxes` you can keep customizing with the full matplotlib + astropy
toolbox — no framework lock-in. And it's **broad rather than single-purpose**: all-sky maps, fields,
globes, cones, HEALPix, regions, images, overlays, queries, vector fields, and an interactive
backend all live behind one import and share one set of conventions. It gets the tedious details
right (antimeridian seams, projection clipping, RA/east-left orientation, publication ticks), and
it's modern and maintained — Python 3.10+, fully type-annotated (mypy-strict), and covered by
1600+ tests.

> **Where it came from.** skyplothelper didn't start as a package. It's the consolidation of nearly
> a decade of research-plotting code — the scripts and helpers written to get *one more figure*
> right for one more paper — cleaned up, unified, tested, and documented into a single coherent
> toolkit so you don't have to rebuild the same scaffolding for every project.

## Installation

```bash
pip install skyplothelper                 # core: numpy, matplotlib, astropy>=6.0, shapely, healpy

# HEALPix binning/plotting and spherical regions / CompoundRegion are core —
# no extra needed (healpy has no Windows wheel, so those features raise an
# informative error there; everything else works cross-platform).

# Optional features (mix as needed; each fails gracefully with a helpful message if absent):
pip install "skyplothelper[plotly]"       # interactive backend  (add [dash] for the FITS viewer app)
pip install "skyplothelper[query]"        # SIMBAD / NED / VizieR / SkyView (astroquery)
pip install "skyplothelper[reproject]"    # image reprojection (reproject)
pip install "skyplothelper[cartopy]"      # cartopy backend + Earth features
pip install "skyplothelper[cone]"         # cosmology conversions for cone plots (scipy)
pip install "skyplothelper[all]"          # everything optional
```

Requires **Python 3.10+** and **astropy 6.0+**. Optional features are gated behind their extras and
raise an informative `ImportError` (naming the extra to install) if their dependency is missing —
nothing else stops working.

## Quickstart

```python
import skyplothelper as sph
import matplotlib.pyplot as plt

# A tilted celestial globe with a see-through graticule and a compass rose
fig = plt.figure()
ax = sph.make_globe_frame(111, center_LONdeg=0, center_LATdeg=23.44, grid=False)
sph.plot_ortho_grid(ax)
sph.add_compass_rose(ax)

# A cosmology cone (redshift wedge)
fig = plt.figure()
ax = sph.make_cone_frame(111, angle_center=180, angle_half_width=30,
                         r_min=0, r_max=0.15, angle_label="R.A.", fig=fig)
sph.cone_scatter(ax, galaxy_ras, galaxy_redshifts, s=3)

# A HEALPix all-sky map (returns fig, ax, mappable, colorbar)
result = sph.healpix_allsky_figure(my_hpx_map, projection="AIT")
result.colorbar.set_label("value")

# The same map, interactively (pan / zoom / hover / HTML export)
import skyplothelper.plotly as sphpl
sfig = sphpl.make_figure(projection="AIT", center=180)
sphpl.add_healpix(sfig, my_hpx_map)
sfig.write_html("skymap.html")
```

More recipes are in the
[quickstart](https://skyplothelper.readthedocs.io/en/latest/quickstart.html), and every subsystem
has a narrative [user guide](https://skyplothelper.readthedocs.io/en/latest/guide/index.html) page
plus a worked [tutorial notebook](https://skyplothelper.readthedocs.io/en/latest/tutorials/index.html).

## Documentation

Full documentation is on **[Read the Docs](https://skyplothelper.readthedocs.io)**:

- **[User guide](https://skyplothelper.readthedocs.io/en/latest/guide/index.html)** — each subsystem
  explained, with worked examples and the gotchas.
- **[Tutorials](https://skyplothelper.readthedocs.io/en/latest/tutorials/index.html)** — runnable
  end-to-end notebooks, from your first frame to interactive maps.
- **[Feature Gallery](https://skyplothelper.readthedocs.io/en/latest/features/index.html)** — a
  visual index with starter code for each kind of figure.
- **[API reference](https://skyplothelper.readthedocs.io/en/latest/api/index.html)** — every public
  function and class, grouped by subsystem.

### For AI agents / LLMs

skyplothelper is **frame-first**: create a sky frame, then draw data and
decorations onto its axes. In a session, start with:

```python
import skyplothelper as sph
sph.overview()            # scope + frame-first model + coordinate conventions
sph.recipes('cube')       # copy-paste recipes for a task (also 'catalog',
                          # 'stroke', 'grid', 'colorbar', ...)
```

For ingestion, [`llms.txt`](llms.txt) is a concise, link-rich map and
[`llms-full.txt`](llms-full.txt) inlines the full runnable recipe corpus
(both generated from the same in-package catalog, so they never drift).

## Citing

If skyplothelper is useful in your work, please cite it — and a mention in your acknowledgements is
genuinely appreciated. For now, cite the software via its repository and the included
[`CITATION.cff`](CITATION.cff):

```bibtex
@software{skyplothelper,
  author  = {Cigan, Phil},
  title   = {{skyplothelper}: astronomy visualization for matplotlib and astropy WCSAxes},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/pjcigan/skyplothelper}
}
```

A journal/arXiv article describing skyplothelper and a citeable
[ASCL](https://ascl.net) entry are planned to accompany the first release, and a Zenodo DOI will be
minted — this section and the badges above will be updated with the preferred reference once they
are available.

## Contributing

Bug reports, feature requests, and pull requests are welcome — with the honest caveat that
skyplothelper is maintained by a single developer with limited time, so reviews and fixes happen as
time allows and may take a while. For anything non-trivial, please open an issue to discuss before
writing code. Usage questions ("how do I do X?") belong in
[Discussions](https://github.com/pjcigan/skyplothelper/discussions) — and are usually answered
fastest by the [docs](https://skyplothelper.readthedocs.io) or by `sph.recipes('<keyword>')` in your
own session — so issues stay reserved for bugs and feature ideas. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, guidelines, and what to expect, and
please be kind and constructive in all project spaces.

## License

BSD-3-Clause. See [`LICENSE`](LICENSE).
