# `docs/manim/` — optional manim explainer scenes

Standalone [ManimCE](https://www.manim.community/) scenes that generate the
explainer **videos** embedded in a few tutorial notebooks. They are *build-time
asset generators* in the same spirit as `docs/make_readme_assets.py` and
`docs/make_tutorial_dark_figs.py` — run by hand, never by the notebook-execution
or dark-figure pipelines, and never a runtime dependency of the package.

The cardinal rule:
**skyplothelper is the source of scientific truth; manim is only the camera and
typographer.** Anything that looks like a sky map is a *real* sph render saved to
`assets/`, imported into the scene as an `ImageMobject` and cross-faded — never a
manim redrawing of a map.

## Layout

```
docs/manim/
  README.md                     — this file
  make_projection_assets.py     — sph side for #2: real renders + star data (cenv)
  projection.py                 — manim scene for tutorial #2 (run in the manim env)
  make_startype_assets.py       — sph side for #15: teff_to_rgb stops + coda (cenv)
  startype.py                   — manim scene for tutorial #15 (run in the manim env)
  make_globe_motions_assets.py  — sph side for #11: two posed-globe cuts (cenv)
  globe_motions.py              — manim scenes for tutorial #11 (EulerTrio + Precession)
  assets/                        — committed inputs the scenes import
    tan_orion.png                  real gnomonic Orion chart (sph; PNG for sharp lines)
    allsky_aitoff.jpg              real all-sky Aitoff panorama (sph; JPG for the photo)
    orion_stars.json               star unit-vectors + bv_to_rgb colors + distances (sph)
    startype_stops.json            O->M teff_to_rgb color/size/label stops (sph)
    startype_allsky.jpg            real perceived-color all-sky, navy canvas (sph)
  media/                         — manim's render scratch (gitignored; via --media_dir)
```

The rendered **MP4** (plus a poster still) is the build artifact: put both in the
host notebook's static dir under that notebook's slug (e.g.
`docs/_static/manim/projections__what-is-a-projection.mp4`) and embed with an
autoplay `<video>` tag (below). An opaque H.264 MP4 is smaller *and* higher-res
than a GIF here — ~0.9 MB at 720p30 vs ~2 MB at 480p12 — so video is the default
for these opaque explainers. (A GIF is only worth making for a context that can't
play video, e.g. a GitHub README; the `mp4 → GIF` recipe below still applies.)

The asset is a single, mode-invariant navy video — no separate dark variant, and
it is skipped by the dark-fig pass automatically (it lives in a raw-HTML cell, not
a notebook code-cell output).

## Environment

ManimCE is a heavy, build-only install (Cairo, Pango, ffmpeg; LaTeX is *not*
needed — the scenes use Pango `Text`, not `Tex`). Keep it out of `pyproject.toml`
and the docs build. It lives in a separate conda env:

```
# one-time
conda create -n manim python=3.12
conda activate manim && pip install "manim==0.19.1"
```

Pinned to **ManimCE 0.19.1** so rebuilds are deterministic. (Locally this is the
`py312` env.)

## Build — tutorial #2, "what a projection actually is"

Two steps: the sph renders (astropy env), then the manim scene (manim env).

```bash
# 1. real sph renders + star data  →  assets/   (astropy env, from repo root)
python docs/manim/make_projection_assets.py

# 2. the scene  →  an mp4  (manim env). --media_dir keeps manim's scratch under
#    docs/manim/media/ (gitignored) instead of a ./media/ at the repo root.
manim -qm --fps 30 --media_dir docs/manim/media docs/manim/projection.py Projection
```

### mp4 → web video + poster (the committed assets)

Re-encode manim's mp4 for broad browser support (`yuv420p` for Safari/older,
`+faststart` so it plays before fully downloaded, no audio), and grab a poster
still (shown before autoplay / if autoplay is blocked):

```bash
SRC=docs/manim/media/videos/projection/720p30/Projection.mp4
ffmpeg -y -i "$SRC" -c:v libx264 -pix_fmt yuv420p -crf 23 -preset slow \
  -movflags +faststart -an \
  docs/_static/manim/projections__what-is-a-projection.mp4
ffmpeg -y -ss 11 -i "$SRC" -frames:v 1 -q:v 3 \
  docs/_static/manim/projections__what-is-a-projection.poster.jpg
```

### Embedding in a notebook

nbsphinx renders raw HTML in a markdown cell, so embed with a `<video>`. Two
gotchas: keep the whole tag on **one line** (CommonMark's raw-HTML-block list has
no `video`/`source`, so a multi-line block gets parsed as markdown and its
`</video>` dropped), and skip an inner `<img>` fallback (nbsphinx rewrites its
`src` and mangles the element). `muted` is required for autoplay:

```html
<video controls autoplay loop muted playsinline poster="../_static/manim/<slug>.poster.jpg" width="100%" style="max-width:760px;display:block;margin:0.5em auto;"><source src="../_static/manim/<slug>.mp4" type="video/mp4"></video>
```

`docs/manim/` also needs to stay out of the Sphinx source tree: `conf.py`'s
`exclude_patterns` carries `manim/*.md` (scoped to `*.md` so it doesn't also drop
the videos under `_static/manim` — exclude_patterns is applied to the static tree
too).

## Build — tutorial #7, "set algebra as an animated Venn"

Same two-step shape (sph renders, then the scene). The sph side renders four real
`CompoundRegion`s (union / intersection / difference of two caps on a globe, plus
the notebook's worked survey footprint); the scene shades a Venn and dissolves
each operation into the matching render.

```bash
# 1. real CompoundRegion renders  →  assets/   (astropy env, from repo root)
python docs/manim/make_regions_assets.py

# 2. the scene  →  an mp4  (manim env)
manim -qm --fps 30 --media_dir docs/manim/media docs/manim/regions_setalgebra.py SetAlgebra
```

Re-encode + poster (a shaded-Venn "union" still makes a self-explanatory thumbnail):

```bash
SRC=docs/manim/media/videos/regions_setalgebra/720p30/SetAlgebra.mp4
ffmpeg -y -i "$SRC" -c:v libx264 -pix_fmt yuv420p -crf 23 -preset slow \
  -movflags +faststart -an \
  docs/_static/manim/regions__set-algebra.mp4
ffmpeg -y -ss 3.2 -i "$SRC" -frames:v 1 -q:v 3 \
  docs/_static/manim/regions__set-algebra.poster.jpg
```

Embedded at tutorial #7 §4 "Compound set algebra" with the same one-line `<video>`
tag as below.

### mp4 → GIF (only if a context can't play video)

A direct `--format gif` dithers the flat navy into swimming noise *and* bloats the
file; a two-pass palette fixes both:

```bash
SRC=docs/manim/media/videos/projection/720p30/Projection.mp4
ffmpeg -y -i "$SRC" -vf "fps=12,scale=480:-1:flags=lanczos,palettegen=stats_mode=diff" /tmp/pal.png
ffmpeg -y -i "$SRC" -i /tmp/pal.png -lavfi \
  "fps=12,scale=480:-1:flags=lanczos,paletteuse=dither=bayer:bayer_scale=5" \
  docs/_static/manim/<slug>.gif
```

## Build — tutorial #15, "star-type morph" (demo E)

A star marker tweens down the main sequence O→M, its color at every frame a real
`sph.teff_to_rgb` value (the Sun lands on white, not green) and its size on the
main-sequence radius; a coda dissolves into the notebook's real perceived-color
all-sky chart. Same two-step model — sph data + coda still, then the scene:

```bash
# 1. teff_to_rgb color/size stops + the navy all-sky coda  →  assets/   (cenv)
python docs/manim/make_startype_assets.py

# 2. the scene  →  an mp4   (manim env)
manim -qm --fps 30 --media_dir docs/manim/media docs/manim/startype.py StarType
```

Then re-encode to web mp4 + poster exactly as above, with
`SRC=docs/manim/media/videos/startype/720p30/StarType.mp4` and slug
`constellations__star-type-morph`, and embed the one-line `<video>` in
`docs/tutorials/constellations.py` (§1, next to the perceived-color swatch).

## Build — tutorial #11, the two globe-motion demos (C + D)

One asset generator feeds two scenes in the same file: **`EulerTrio`** (demo C —
the three Euler angles posing a globe, dissolving into the real Blue-Marble globe)
and **`Precession`** (demo D — spin / nutation / precession with a swept spin-axis
cone, dissolving into the real Black-Marble look-down globe). The schematic
wireframe carries the same gold-equator / green-prime-meridian highlights as the
sph cuts so the graticule lines up as it dissolves; the two rotations are matched
to `euler_to_fits_ortho`'s convention (calibrated by rendering a reference pose).

```bash
# 1. two real posed-globe cuts  →  assets/   (astropy env, from repo root)
python docs/manim/make_globe_motions_assets.py

# 2. the two scenes  →  mp4s  (manim env)
manim -qm --fps 30 --media_dir docs/manim/media docs/manim/globe_motions.py EulerTrio
manim -qm --fps 30 --media_dir docs/manim/media docs/manim/globe_motions.py Precession
```

Re-encode + poster each, as above:

```bash
for scene_slug in "EulerTrio:globe_plots__euler-trio" \
                  "Precession:globe_plots__spin-nutation-precession"; do
  scene=${scene_slug%%:*}; slug=${scene_slug##*:}
  SRC=docs/manim/media/videos/globe_motions/720p30/$scene.mp4
  ffmpeg -y -i "$SRC" -c:v libx264 -pix_fmt yuv420p -crf 23 -preset slow \
    -movflags +faststart -an "docs/_static/manim/$slug.mp4"
  ffmpeg -y -sseof -1.4 -i "$SRC" -frames:v 1 -q:v 3 "docs/_static/manim/$slug.poster.jpg"
done
```

Both embed in `docs/tutorials/globe_plots.py` §1 ("Aiming the globe"): the Euler
trio right under the three-angle list, the motions demo after the 3×4 grid with a
cross-reference to #17 §4's matplotlib version.
