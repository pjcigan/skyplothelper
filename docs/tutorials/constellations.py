# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Constellations and Asterisms
#
# Sky charts often carry the same useful furniture: the 88 IAU
# constellation regions fencing the sky into official territories, the bright
# stick figures everyone recognizes, and the names. skyplothelper bundles all of
# it — no downloads, no extra dependencies — as four one-call overlays:
#
# - **`add_constellation_boundaries`** — the official IAU boundary segments,
# - **`add_constellation_lines`** — the asterism (connect-the-dots) figures,
# - **`add_constellation_labels`** — names or abbreviations, sensibly placed,
# - **`add_constellation_polygon`** — one constellation filled as a region,
#
# plus `list_constellations()` to browse what's there. Each section pairs the
# one-call *show* with the knobs that *adjust* it, and along the way we build
# real star charts: the stars themselves are ordinary catalog data (the bundled
# Hipparcos naked-eye sample), because these overlays are deliberately
# *cartographic decoration*, not a planetarium engine —
# [section 10](#10.-Decoration,-not-a-planetarium) draws that boundary honestly.
#
# > **A note on the data.** Everything the constellation calls draw ships inside
# > the package itself. The star catalog (`hipparcos_bright_pm.csv`) lives in the
# > repository's `examples/data/`; the photographic all-sky panorama used at the
# > end is a large file kept in the repository rather than the pip install — you
# > only need it to *re-run* that final cell, since the rendered figure is
# > already on this page.
#
# ## Contents
#
# 1. [A star chart at a glance](#1.-A-star-chart-at-a-glance)
# 2. [Boundaries](#2.-Boundaries)
# 3. [Asterisms](#3.-Asterisms)
# 4. [Names and labels](#4.-Names-and-labels)
# 5. [The 88-constellation registry](#5.-The-88-constellation-registry)
# 6. [Highlighting a constellation](#6.-Highlighting-a-constellation)
# 7. [On a globe](#7.-On-a-globe)
# 8. [On a galactic frame](#8.-On-a-galactic-frame)
# 9. [Bring your own lines and boundaries](#9.-Bring-your-own-lines-and-boundaries)
# 10. [Decoration, not a planetarium](#10.-Decoration,-not-a-planetarium)
# 11. [Putting it together](#11.-Putting-it-together)
# 12. [Where to go next](#12.-Where-to-go-next)

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

import skyplothelper as sph

# A clean, cohesive look throughout. set_style applies each layer (base / theme /
# palette) independently, so setting only the **structural** base composes with
# whatever theme or palette is already active — here, and in your own sessions.
sph.set_style(base="structural")


# Pick annotation colors that read on the current background. Checking the figure
# background's brightness like this makes one code path work on light and dark
# themes alike (these docs render each figure both ways) — a handy pattern for
# any plotting code that has to survive a theme switch.
def annotation_palette():
    r, g, b = to_rgb(mpl.rcParams["figure.facecolor"])
    dark = (0.299 * r + 0.587 * g + 0.114 * b) < 0.5
    return sph.ANNOTATION_PALETTES["dark" if dark else "publication"]


PAL = annotation_palette()

# A cycle palette for multi-constellation fills (dual-mode and transparency-safe).
URAN = sph.CYCLE_PALETTES["uranometria"]["colors"]

# A dark-sky annotation palette for the star-atlas panels below. Perceived star
# colors are pale and only read against a dark background, so the opener commits
# to a night sky in both the light and dark doc renders (the body sections stay
# mode-adaptive). NIGHT gives cohesive, atlas-appropriate colors for that card.
NIGHT = sph.ANNOTATION_PALETTES["night"]
SKY = NIGHT["ax_bg"]
# A warm, wheat-toned line color for the asterisms on the dark atlas — a classic
# celestial-atlas look that lets the stars' own colors carry the temperature.
WHEAT = "#CBB88C"

# Perceived-star-color saturation. `sph.bv_to_rgb`/`teff_to_rgb` default to 0.55
# (a softened sky); we use the full, honest tristimulus value (1.0), which matches
# the reference computation of Harre & Heller (2021). See the note under §1's
# swatch on why even these "full" colors are quite pale.
STAR_SAT = 1.0

DATA = "../../examples/data"

# Every naked-eye star: the 4,992 Hipparcos stars brighter than V = 6. The `BV`
# column (Johnson B-V color index) drives the perceived-color rendering below.
stars = pd.read_csv(f"{DATA}/hipparcos_bright_pm.csv")


def star_sizes(vmag, scale=1.0, mlim=6.5):
    """Classic star-chart scaling: marker area grows as (m_lim - V)^2."""
    return scale * (mlim - np.asarray(vmag)) ** 2


def star_atlas_frame(fig, subplot=111, **frame_kw):
    """Build a star-atlas panel: a dark sky (baked into both the light and dark
    doc renders) with the frame's own ticks and labels lightened to read on it."""
    ax = sph.make_wcs_frame(subplot, fig=fig, gridcolor=NIGHT["grid"], **frame_kw)
    ax.set_facecolor(SKY)
    fig.canvas.draw()
    sph.style_wcs_axes(ax, tick_color=NIGHT["stars"], labelcolor=NIGHT["stars"],
                       axislabel_color=NIGHT["stars"])
    return ax


# %% [markdown]
# ## 1. A star chart at a glance
#
# Here is the whole toolkit on one figure: every naked-eye star in its true
# perceived color, the IAU boundaries carving the sky into its 88 official
# regions, the bright asterism figures, and a label on each. The stars are
# ordinary scatter data (the Hipparcos catalog loaded above — more on that
# division of labor in [section 10](#10.-Decoration,-not-a-planetarium)); their
# colors come from one call, `sph.bv_to_rgb`, which we unpack just below; the
# three constellation calls draw the rest.
#
# Because those stellar colors are subtle and only read against a night sky, this
# opening chart commits to a dark background in both the light and dark versions
# of these docs — a deliberate planetarium look. The working sections that follow
# go back to the page's own light or dark theme.

# %%
# fig-slug: allsky-star-chart
fig = plt.figure(figsize=(12.5, 6.3), facecolor=SKY)
ax = star_atlas_frame(fig, 111, projection="AIT", center=180)
ax.scatter(stars.RAICRS, stars.DEICRS, s=star_sizes(stars.Vmag, 0.6),
           c=sph.bv_to_rgb(stars.BV.fillna(0.6), saturation=STAR_SAT), lw=0, alpha=0.95, zorder=3,
           transform=ax.get_transform("world"))
sph.add_constellation_boundaries(ax, color=NIGHT["grid"], lw=0.5, alpha=0.9)
sph.add_constellation_lines(ax, rank_max=1, color=WHEAT, lw=0.9, alpha=0.5)
sph.add_constellation_labels(ax, fontsize=6, color=NIGHT["accent2"], alpha=0.85)
ax.set_title("Every naked-eye star, in its perceived color",
             color=NIGHT["stars"], fontsize=12)

# %% [markdown]
# And the same elements up close. Zooming a tangent frame onto Orion — one of the most
# recognizable patches of sky there is — shows what each layer contributes: the
# stair-step **boundary** fencing off Orion's official territory, the **asterism**
# connecting Betelgeuse to Rigel through the Belt, a translucent **fill** claiming
# the region, and the **label**. The colors are instructive here: ruddy Betelgeuse
# at the shoulder against blue-white Rigel at the foot is a real temperature
# contrast you can see:

# %%
# fig-slug: orion-zoom
fig = plt.figure(figsize=(7.4, 7), facecolor=SKY)
ax = star_atlas_frame(fig, 111, projection="TAN", center=(83.5, 3), fov_deg=30)
sph.format_ticklabels(ax, lon_fmt="hh:mm", lat_fmt="dd")
sph.add_constellation_polygon(ax, "Ori", facecolor=NIGHT["label"], edgecolor="none",
                              alpha=0.08)
sph.add_constellation_boundaries(ax, color=NIGHT["grid"], lw=1.0, alpha=0.9)
sph.add_constellation_lines(ax, color=WHEAT, lw=1.4, alpha=0.85)
ax.scatter(stars.RAICRS, stars.DEICRS, s=star_sizes(stars.Vmag, 3.2),
           c=sph.bv_to_rgb(stars.BV.fillna(0.6), saturation=STAR_SAT), lw=0, alpha=0.98, zorder=3,
           transform=ax.get_transform("world"))
# (Only Orion's own label can land here — its neighbors' label positions
# sit outside this field of view.)
sph.add_constellation_labels(ax, labels="name", fontsize=11,
                             constellations=["ORI"],
                             color=NIGHT["accent2"], alpha=0.9)
ax.set_title("Orion, up close: boundary, asterism, fill, label",
             color=NIGHT["stars"], fontsize=12)

# %% [markdown]
# ### Coloring stars by temperature
#
# Those aren't decorative colors — each is the color a human eye would actually
# perceive for that star, computed from its **B−V color index** (the catalog's
# `BV` column) by `sph.bv_to_rgb`. Hotter stars run blue, cooler stars run red,
# and the Sun sits near white. That one call is the whole recipe:
#
# ```python
# star_colors = sph.bv_to_rgb(stars.BV, saturation=1.0)      # (N, 3) RGB for scatter(c=...)
# # equivalent general form — swap index= for Gaia BP−RP, SDSS g−r, or 2MASS J−K:
# # star_colors = sph.color_index_to_rgb(stars.BV, index="B-V", saturation=1.0)
# ```
#
# `bv_to_rgb` is the B−V shortcut of that general `sph.color_index_to_rgb`. The
# `index=` swap is what keeps a mixed catalog consistent: each index resolves to a
# temperature and defers to the same converter, so a star reads the *same* color —
# and the Sun near-white — whichever photometry you arrive with. For Gaia, reach
# for the `sph.bp_rp_to_rgb` shortcut rather than feeding BP−RP to `bv_to_rgb`,
# which over-reddens; the [Catalogs](catalogs.ipynb#The-color-a-star-actually-is)
# tutorial colors a real Gaia field exactly this way.
#
# Here is the mapping laid out across the spectral sequence, from a hot B star to
# a cool M giant — real stars whose colors you can check against the charts above:

# %%
# fig-slug: star-color-swatches
examples = [("Rigel", "B8", -0.03), ("Vega", "A0", 0.00), ("Sirius", "A1", 0.01),
            ("Procyon", "F5", 0.42), ("Sun", "G2", 0.65), ("Pollux", "K0", 0.99),
            ("Arcturus", "K1", 1.24), ("Aldebaran", "K5", 1.54), ("Antares", "M1", 1.83)]
fig, ax = plt.subplots(figsize=(11, 1.9), facecolor=SKY)
ax.set_facecolor(SKY)
for i, (nm, sp, bv) in enumerate(examples):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=sph.bv_to_rgb(bv, saturation=STAR_SAT)))
    ax.text(i + 0.5, -0.12, f"{nm}\n{sp}   B–V {bv:+.2f}", ha="center", va="top",
            fontsize=8, color=NIGHT["stars"])
ax.set_xlim(0, len(examples))
ax.set_ylim(-0.55, 1)
ax.axis("off")
ax.set_title("sph.bv_to_rgb across the spectral sequence", color=NIGHT["stars"],
             fontsize=11)

# %% [markdown]
# > **Note:** the conversion is a proper *tristimulus* calculation — it integrates
# > the blackbody spectrum against the eye's color-response curves — which is why
# > the Sun comes out white (rather than green — its spectrum peaks in green light, but
# > the eye's combined response to the full spectrum reads as white; a naive
# > peak-wavelength mapping would get this wrong). A `saturation=` knob
# > (default `0.55`) can wash the colors further toward white for a softer sky;
# > this notebook uses `saturation=1.0`, the full honest color (see the next
# > note). `sph.teff_to_rgb` takes an effective temperature directly instead of
# > a B−V.

# %% [markdown]
# And the conversion isn't a guess — it's a physical calculation
# you can *watch*. Below, a single star morphs down the main sequence from a hot
# O star to a cool M dwarf; its color at every instant is a live `sph.teff_to_rgb`
# value and its size tracks main-sequence radius, and the coda drops that same
# function onto every naked-eye star.
#
# > **Note:** even at full saturation these colors run paler than you might expect
# > — and that is real. Real starlight is far less saturated than most
# > illustrations suggest: cool M stars read as a *pale orange* (not a deep red),
# > the Sun is white, and hot stars a pale blue-white. These match the reference
# > tristimulus computation of
# > [Harre & Heller (2021)](https://arxiv.org/abs/2101.06254), who note that the
# > vivid star palettes common in outreach are, in their words, "misleading
# > colors." Two things feed the illusion that stars are vivid. First, a
# > CIE *chromaticity diagram* paints every point at its maximum displayable
# > chroma, so the stellar locus drawn on one *looks* far more saturated than any
# > star truly is — the star's actual chromaticity sits close to white. Second,
# > *small-field tritanopia*: the eye barely registers color in a tiny point
# > source, so a real naked-eye star looks paler still. The `saturation=` knob
# > only ever washes further *toward* white from here.
#
# <video controls autoplay loop muted playsinline poster="../_static/manim/constellations__star-type-morph.poster.jpg" width="100%" style="max-width:720px;display:block;margin:0.6em auto;"><source src="../_static/manim/constellations__star-type-morph.mp4" type="video/mp4"></video>

# %% [markdown]
# Four functions and one color map did all of that, and the rest of the notebook
# is simply each one in turn — what it draws, and which knobs adjust it. From here
# on we return to the page's own light or dark theme.

# %% [markdown]
# ## 2. Boundaries
#
# **To draw the official IAU constellation boundaries, call
# `add_constellation_boundaries(ax)`** — one line, no data files needed (you can supply your own via `data_file=`), on any
# celestial frame (equatorial here; a galactic-frame chart comes in
# [section 8](#8.-On-a-galactic-frame)). The boundaries ship with the package
# (the Davenhall & Leggett 1989 corner list, Vizier VI/49), so nothing is
# downloaded.
#
# The defaults are deliberately subtle — a thin mid-gray at low alpha — because
# boundaries are usually *context*, not the subject:

# %%
# fig-slug: boundaries-default-vs-styled
fig = plt.figure(figsize=(13, 4.1))
ax = sph.make_wcs_frame(121, projection="AIT", center=180, fig=fig)
fig.canvas.draw()
sph.add_constellation_boundaries(ax)
ax.set_title("add_constellation_boundaries(ax) — the one-call default", fontsize=10)

ax = sph.make_wcs_frame(122, projection="AIT", center=180, fig=fig)
fig.canvas.draw()
sph.add_constellation_boundaries(ax, color=URAN[0], lw=0.9, alpha=0.75, ls=":")
ax.set_title("styled: color=, lw=, alpha=, ls=", fontsize=10)

# %% [markdown]
# **To adjust them,** the knobs are the usual line styling: `color=`, `lw=`,
# `alpha=`, `ls=`, plus `stroke_color=`/`stroke_lw=` — a thin contrasting outline
# drawn underneath each segment that keeps the lines legible over imagery (we use
# it on the photographic backdrop in [Putting it together](#11.-Putting-it-together)).
#
# > **Note:** the boundaries are exact meridians and parallels — of the **1875**
# > equinox, the epoch the IAU fixed them at. What ships is that corner list
# > precessed to ICRS, which is why the segments look like a stair-step grid tilted
# > very slightly against today's graticule: the tilt *is* precession since 1875.
# > A custom boundary set can be swapped in via `data_file=`.

# %% [markdown]
# ## 3. Asterisms
#
# Boundaries carve up the sky; **asterisms** are the connect-the-dots star figures
# people actually recognize. `add_constellation_lines(ax)` draws them — the
# d3-celestial line set (from the IAU constellation charts), also bundled with the
# package.
#
# Each line carries a prominence **rank** from 1 (the bright canonical figures) to
# 3 (faint auxiliary lines). **To control how much detail you get, pass
# `rank_max=`:**

# %%
# fig-slug: asterisms-rank-max
fig = plt.figure(figsize=(13.5, 3.4))
for i, (rmax, desc) in enumerate(
    [(1, "rank_max=1 — the bright figures"),
     (2, "rank_max=2 — plus secondary lines"),
     (None, "rank_max=None — everything")],
    start=1,
):
    ax = sph.make_wcs_frame(130 + i, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    sph.add_constellation_lines(ax, rank_max=rmax, color=PAL["accent"],
                                lw=0.7, alpha=0.9)
    ax.set_title(desc, fontsize=10)

# %% [markdown]
# **To draw only the constellations you care about, pass `constellations=`** — a
# list of 3-letter IAU codes (case-insensitive). A classic chart move is *bold
# famous figures over faint context*:

# %%
# fig-slug: asterisms-famous-subset
FAMOUS = ["ORI", "UMA", "UMI", "CAS", "CYG", "LYR", "TAU", "GEM",
          "LEO", "VIR", "SCO", "SGR", "CRU", "CEN"]

fig = plt.figure(figsize=(11, 5.5))
ax = sph.make_wcs_frame(111, projection="AIT", center=180, fig=fig)
fig.canvas.draw()
sph.add_constellation_boundaries(ax)
sph.add_constellation_lines(ax, constellations=FAMOUS, color=PAL["accent"],
                            lw=1.3, alpha=0.95)
ax.set_title("Famous figures bold, all-sky boundaries faint", fontsize=11)

# %% [markdown]
# ## 4. Names and labels
#
# **To name what you've drawn, call `add_constellation_labels(ax)`.** The 88
# center positions are built in (no boundary data required), and `labels=` picks
# the form: `'abbr'` (the 3-letter IAU code — the default), `'name'` (the full
# Latin name), or `'both'`.

# %%
# fig-slug: labels-abbr-vs-names
fig = plt.figure(figsize=(13, 4.1))
ax = sph.make_wcs_frame(121, projection="AIT", center=180, fig=fig)
fig.canvas.draw()
sph.add_constellation_boundaries(ax)
sph.add_constellation_labels(ax)
ax.set_title("labels='abbr' for all 88 — the default", fontsize=10)

ax = sph.make_wcs_frame(122, projection="AIT", center=180, fig=fig)
fig.canvas.draw()
sph.add_constellation_boundaries(ax)
sph.add_constellation_labels(ax, labels="name", constellations=FAMOUS,
                             fontsize=8, color=PAL["label"], alpha=0.95)
ax.set_title("labels='name' for a chosen subset", fontsize=10)

# %% [markdown]
# **To adjust them:** `fontsize=`, `color=`, `alpha=`, and the same
# `stroke_color=`/`stroke_lw=` legibility outline as the lines; any other keyword
# goes straight to `ax.text()` (so `fontweight='bold'`, `family=`, etc. all work).
#
# > **Note:** labels sit at each constellation's polygon centroid, with a few
# > hand-tuned nudges (Eridanus, Pisces, Serpens, …) so every label lands inside
# > its own region *on the default AIT view centered at 180°*. If you re-center a
# > frame and a nudged label looks off, pass `apply_default_offsets=False` to get
# > the raw centroids back.

# %% [markdown]
# ## 5. The 88-constellation registry
#
# Which codes exist? `list_constellations()` prints the full registry —
# abbreviation, Latin name, and center position — sortable by `'abbr'`, `'name'`,
# `'ra'`, or `'dec'`:

# %%
sph.list_constellations(sort="abbr")

# %% [markdown]
# Eighty-eight is a lot to scan, so here are the subsets people might reach
# for most frequently — copy the list you need:
#
# | subset | members | good for |
# |---|---|---|
# | **Zodiac** | ARI TAU GEM CNC LEO VIR LIB SCO SGR CAP AQR PSC | ecliptic / solar-system context |
# | **Northern circumpolar** | UMA UMI CAS CEP DRA CAM | always-up northern charts |
# | **Southern showpieces** | CRU CEN CAR VEL SCO SGR | southern-sky highlights |
# | **Famous figures** | ORI UMA CAS CYG SCO LEO TAU GEM | general "recognizable sky" context |

# %%
ZODIAC = ["ARI", "TAU", "GEM", "CNC", "LEO", "VIR",
          "LIB", "SCO", "SGR", "CAP", "AQR", "PSC"]
CIRCUMPOLAR_N = ["UMA", "UMI", "CAS", "CEP", "DRA", "CAM"]
SOUTHERN = ["CRU", "CEN", "CAR", "VEL", "SCO", "SGR"]

# %% [markdown]
# Subsets pair naturally with the coordinate-plane overlays. The zodiac *is* the
# band of constellations along the ecliptic — draw the plane and the twelve
# figures together and the definition draws itself:

# %%
# fig-slug: zodiac-along-ecliptic
fig = plt.figure(figsize=(11.5, 5.7))
ax = sph.make_wcs_frame(111, projection="AIT", center=180, fig=fig)
fig.canvas.draw()
sph.add_constellation_boundaries(ax)
sph.add_plane_overlay(ax, plane="ecliptic", color=PAL["accent2"], lw=1.6)
sph.add_constellation_lines(ax, constellations=ZODIAC, color=PAL["accent"],
                            lw=1.2, alpha=0.95)
sph.add_constellation_labels(ax, constellations=ZODIAC, labels="abbr",
                             fontsize=8, color=PAL["accent"], alpha=0.95)
# The ecliptic's thirteenth constellation: it also crosses Ophiuchus,
# between Scorpius and Sagittarius.
sph.add_constellation_lines(ax, constellations=["OPH"], color=PAL["accent"],
                            lw=1.0, alpha=0.7, ls="--")
sph.add_constellation_labels(ax, constellations=["OPH"], labels="abbr",
                             fontsize=8, color=PAL["accent"], alpha=0.7)
ax.set_title("The zodiac: the constellations the ecliptic passes through", fontsize=12)

# %% [markdown]
# > **Note:** count the crossings — the ecliptic actually passes through
# > **thirteen** IAU constellations. Ophiuchus (dashed) sits squarely on it
# > between Scorpius and Sagittarius; it just never made the traditional twelve.
#
# Tilt the whole map into **ecliptic coordinates** and the point stops being a
# curve you have to trace and becomes a straight line: the ecliptic flattens to
# the horizontal center, the zodiac strings out along it like beads, and every
# *other* constellation (drawn faintly in red) sits clearly off the band. This is
# the same frame-awareness from [section 8](#8.-On-a-galactic-frame) — just pass
# `frame="ecliptic"`:

# %%
# fig-slug: zodiac-ecliptic-frame
fig = plt.figure(figsize=(11.5, 5.7))
ax = sph.make_wcs_frame(111, projection="AIT", center=180, frame="ecliptic", fig=fig)
fig.canvas.draw()
sph.add_constellation_boundaries(ax, alpha=0.3)
# Every non-zodiac figure, faint red, so it reads as "present but off the band."
sph.add_constellation_lines(ax, color=URAN[5], lw=0.8, alpha=0.25)
sph.add_plane_overlay(ax, plane="ecliptic", color=PAL["accent2"], lw=1.6)
sph.add_constellation_lines(ax, constellations=ZODIAC, color=PAL["accent"],
                            lw=1.4, alpha=0.95)
sph.add_constellation_labels(ax, constellations=ZODIAC, labels="abbr",
                             fontsize=8, color=PAL["accent"], alpha=0.95)
sph.add_constellation_lines(ax, constellations=["OPH"], color=PAL["accent"],
                            lw=1.0, alpha=0.7, ls="--")
sph.add_constellation_labels(ax, constellations=["OPH"], labels="abbr",
                             fontsize=8, color=PAL["accent"], alpha=0.7)
ax.set_title("The same zodiac in ecliptic coordinates — now a straight band",
             fontsize=12)

# %% [markdown]
# ## 6. Highlighting a constellation
#
# **To make one constellation the subject rather than context, fill it:**
# `add_constellation_polygon(ax, 'Ori')` resolves the IAU code against the
# bundled corner list and renders it through the same spherical-polygon machinery
# as the region tools — so antimeridian crossings, pole-enclosing polygons
# (Ursa Minor!), and frame edges are all handled.

# %%
# fig-slug: polygon-highlights
fig = plt.figure(figsize=(11.5, 5.7))
ax = sph.make_wcs_frame(111, projection="AIT", center=180, fig=fig)
fig.canvas.draw()
sph.add_constellation_boundaries(ax)
sph.add_constellation_labels(ax, fontsize=6)

# Filled, outline-only, and hatched — three looks from the same call.
sph.add_constellation_polygon(ax, "Ori", facecolor=URAN[1], alpha=0.45)
sph.add_constellation_polygon(ax, "UMi", facecolor=URAN[0], alpha=0.45)  # wraps the pole
sph.add_constellation_polygon(ax, "Cyg", facecolor="none", edgecolor=URAN[2], lw=2.0)
sph.add_constellation_polygon(ax, "Cas", facecolor="none", edgecolor=URAN[4],
                              lw=1.3, hatch="///")
sph.add_constellation_polygon(ax, "Ser", facecolor=URAN[5], alpha=0.45)  # two patches!
ax.set_title("add_constellation_polygon: fills, outlines, hatches", fontsize=12)

# %% [markdown]
# The knobs are patch styling: `facecolor=`, `edgecolor=`, `alpha=`, `lw=`,
# `hatch=`, plus the `stroke_color=` outline. Two quirks worth knowing:
#
# - **Serpens is two patches.** The only split constellation (Caput and Cauda,
#   either side of Ophiuchus) — one call fills both, and the returned list has
#   two entries.
# - **Ursa Minor wraps the celestial pole** — the polygon machinery closes it
#   correctly over the pole rather than smearing it across the top of the map.
#
# Fills also follow a zoomed field: when the constellation runs past your frame
# edge, the fill clips at the field of view — the translucent Orion wash in the
# [opening zoom](#1.-A-star-chart-at-a-glance) is exactly this on a 30° tangent
# field.
#
# ### Which constellation is my target in?
#
# The usual version of this task starts from *your* source, not from a
# constellation name. astropy answers the membership question —
# `get_constellation()` performs the official IAU boundary lookup — and the
# short name it returns is exactly what the sph calls accept. Here's the whole
# recipe for M51:

# %%
# fig-slug: host-constellation-lookup
from astropy.coordinates import SkyCoord, get_constellation  # noqa: E402

m51 = SkyCoord("13h29m52.7s +47d11m43s")
host = get_constellation(m51, short_name=True)
print(f"M51 lies in {get_constellation(m51)} ({host})")

fig = plt.figure(figsize=(6.8, 6.2))
ax = sph.make_wcs_frame(111, projection="TAN", center=(197, 42), fov_deg=26, fig=fig)
fig.canvas.draw()
sph.format_ticklabels(ax, lon_fmt="hh:mm", lat_fmt="dd")
sph.add_constellation_polygon(ax, host, facecolor=URAN[0], edgecolor="none",
                              alpha=0.15)
sph.add_constellation_boundaries(ax, lw=0.8, alpha=0.6)
sph.add_constellation_lines(ax, color=PAL["accent"], lw=1.3, alpha=0.9)
sph.add_constellation_labels(ax, labels="name", fontsize=10,
                             color=PAL["label"], alpha=0.9)
# An open 'circle' reticle in a second accent color keeps the target visually
# distinct from the asterism lines (and leaves the source itself unobscured).
sph.add_reticle(ax, (m51.ra.deg, m51.dec.deg), style="circle", size=15,
                color=PAL["accent2"], lw=1.6,
                label="M51", label_color=PAL["accent2"])
ax.set_title(f"M51's host constellation: {get_constellation(m51)}", fontsize=11)

# %% [markdown]
# The same lookup vectorizes over a whole catalog (`get_constellation` accepts an
# array `SkyCoord`), so tallying which constellations your survey's sources fall
# in — and highlighting the busiest — is a few lines. And yes, Canes Venatici's
# entire official asterism really is that one line: two stars, Cor Caroli and Chara.

# %% [markdown]
# ## 7. On a globe
#
# Everything above works unchanged on an orthographic globe — and the far
# hemisphere takes care of itself: segments on the back of the sphere simply
# don't project, so you never see constellation lines bleeding through from
# behind. Same calls, two viewpoints:

# %%
# fig-slug: globe-two-viewpoints
views = [
    # center, which side it shows, the fill subject on that side
    (83, 10, "the Orion side", "Ori"),
    (263, -10, "the antipode", "Sco"),
]
fig = plt.figure(figsize=(11.5, 5.9))
for sub, (clon, clat, side, fill) in zip((121, 122), views):
    ax = sph.make_globe_frame(sub, center_LONdeg=clon, center_LATdeg=clat)
    fig.canvas.draw()
    sph.add_constellation_boundaries(ax)
    sph.add_constellation_polygon(ax, fill, facecolor=URAN[1], alpha=0.35)
    sph.add_constellation_lines(ax, constellations=FAMOUS, color=PAL["accent"],
                                lw=1.3, alpha=0.95)
    sph.add_constellation_labels(ax, constellations=FAMOUS, fontsize=8,
                                 color=PAL["label"], alpha=0.9)
    ax.set_title(f"center=({clon}°, {clat}°) — {side}", fontsize=10)

# %% [markdown]
# Both panels get the *same* `FAMOUS` list — boundaries, lines, and labels all
# cull the far hemisphere themselves, so Orion's figures appear on the left
# globe and Scorpius's on the right without any per-view bookkeeping.
#
# The full globe toolkit — tilting, planet surfaces, day/night shading, globe
# decorations — is its own tutorial: [Globe and Planet
# Plotting](globe_plots.ipynb).

# %% [markdown]
# ## 8. On a galactic frame
#
# The furniture is also **frame-aware**: draw it on a galactic-coordinate frame
# and every boundary, figure, and label lands where it belongs on that grid —
# the overlay data is ICRS under the hood, converted for you. Pair that with
# your own equatorial catalog through `ax.get_transform("icrs")` and a full
# star chart drapes itself across galactic coordinates — the Milky Way now
# running straight along the equator:

# %%
# fig-slug: galactic-frame-chart
fig = plt.figure(figsize=(12.5, 6.3))
ax = sph.make_wcs_frame(111, projection="AIT", center=0, frame="galactic", fig=fig)
fig.canvas.draw()
# RA/Dec catalog data on a galactic frame: transform through 'icrs', not 'world'.
ax.scatter(stars.RAICRS, stars.DEICRS, s=star_sizes(stars.Vmag, 0.55),
           c=PAL["stars"], lw=0, alpha=0.85, zorder=3,
           transform=ax.get_transform("icrs"))
sph.add_constellation_boundaries(ax)
sph.add_constellation_lines(ax, rank_max=1, color=PAL["accent"], lw=0.9, alpha=0.85)
sph.add_constellation_labels(ax, constellations=FAMOUS, fontsize=8,
                             color=PAL["label"], alpha=0.9)
ax.set_title("The same chart in galactic coordinates — the Milky Way runs level",
             fontsize=12)

# %% [markdown]
# Two things worth spotting. First: Cygnus, Cassiopeia, and the Scorpius–
# Sagittarius pair now string out along the equator — they *are* the Milky Way's
# landmarks. Call this band the **Galactic Zodiac** if you like, and feel free to
# rank it above the original: the normal zodiac only tracks the dozen
# constellations our Sun happens to amble past on its yearly loop, whereas this
# one is the mid-plane of the entire Galaxy. A zodiac for a hundred billion stars
# surely outranks a zodiac for one. Second: the boundary grid winds into two
# whirlpools — those are the celestial poles, where the equatorial stair-step
# edges wrap around the pivot points of the RA/Dec system. An ecliptic frame
# (`frame="ecliptic"`, as in [section 5](#5.-The-88-constellation-registry))
# behaves the same way.

# %% [markdown]
# ## 9. Bring your own lines and boundaries
#
# Nothing about these overlays is magic, and nothing about generating them is closed off. Under
# the hood an **asterism is a polyline through star positions** and a **boundary is a spherical
# polygon** — the two functions just read bundled coordinate
# tables and hand them to `ax.plot()` and `add_spherical_polygon()`. So if you
# have a preferred line set, a higher-resolution boundary file, or a figure that
# simply doesn't exist yet, you can draw it yourself. Three levels:
#
# | you want to… | do this |
# |---|---|
# | swap in a different bundled-style data file | `add_constellation_lines(ax, data_file=...)`, same for boundaries |
# | draw one custom asterism | `ax.plot(ra, dec, transform=ax.get_transform("world"))` |
# | draw one custom boundary/region | `sph.add_spherical_polygon(ax, lons, lats)` |
#
# Here's a use case to demonstrate that point: **an asterism the standard dataset
# doesn't contain, because it hasn't happened yet.** The Big Dipper's seven stars
# each drift across the sky, so we propagate the catalog's proper motions forward
# 50,000 years and plot the result as an ordinary polyline. First we draw the
# same seven stars *today* by hand — landing exactly on the bundled line, which
# is the whole lesson — then the same polyline at its future positions:

# %%
# fig-slug: bring-your-own
hip = stars.set_index("HIP")

# The seven Dipper stars in stick-figure order (handle tip to bowl, closing the
# bowl back at Megrez). These are Hipparcos catalog numbers.
DIPPER = [67301, 65378, 62956, 59774, 58001, 53910, 54061, 59774]

# A boundary of your own: corners in RA/Dec. Real IAU boundaries are exactly
# this — a corner list whose edges run along meridians and parallels.
MY_LON = [162, 212, 212, 196, 196, 162]
MY_LAT = [46, 46, 58, 58, 64, 64]


def propagate(df, t_yr):
    """Star positions after t_yr years of linear proper motion, in degrees."""
    ra = df.RAICRS + t_yr * df.pmRA / 3.6e6 / np.cos(np.radians(df.DEICRS))
    dec = df.DEICRS + t_yr * df.pmDE / 3.6e6
    return ra.to_numpy(), dec.to_numpy()


FRAME = dict(projection="ZEA", center=(186, 55.5), fov_deg=46)
fig = plt.figure(figsize=(12.6, 6.0))

# --- Left: a custom asterism -------------------------------------------------
ax = sph.make_wcs_frame(121, fig=fig, **FRAME)
fig.canvas.draw()
tr = ax.get_transform("world")
ax.scatter(stars.RAICRS, stars.DEICRS, s=star_sizes(stars.Vmag, 2.2),
           c=PAL["stars"], lw=0, alpha=0.9, zorder=3, transform=tr)
# What ships: the full Ursa Major figure, drawn as a wide pale ribbon so the
# hand-drawn line below can be seen tracing it exactly.
lines = sph.add_constellation_lines(ax, constellations=["UMA"], rank_max=1,
                                    color=PAL["accent"], lw=5.0, alpha=0.3)
lines[0].set_label("bundled UMa figure")
# Hand-drawn, today (blue) — it lands right on top of the bundled line.
ax.plot(*propagate(hip.loc[DIPPER], 0), color=URAN[0], lw=1.6, zorder=5,
        transform=tr, label="my polyline, today")
# Hand-drawn, 50,000 years from now — no bundled data set has this. Drawn in the
# same green as the custom region on the right: both are your own data, beyond
# anything that ships.
fr, fd = propagate(hip.loc[DIPPER], 50_000)
ax.plot(fr, fd, color=URAN[4], lw=1.8, ls="--", zorder=5, transform=tr,
        label="my polyline, +50,000 yr")
ax.scatter(fr, fd, s=22, color=URAN[4], zorder=6, transform=tr)
ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
ax.set_title("Your own asterism: an ordinary polyline", fontsize=10)

# --- Right: a custom boundary ------------------------------------------------
ax = sph.make_wcs_frame(122, fig=fig, **FRAME)
fig.canvas.draw()
tr = ax.get_transform("world")
ax.scatter(stars.RAICRS, stars.DEICRS, s=star_sizes(stars.Vmag, 2.2),
           c=PAL["stars"], lw=0, alpha=0.9, zorder=3, transform=tr)
sph.add_constellation_boundaries(ax, lw=0.9, alpha=0.55)
# geodesic=False keeps each edge on a parallel or meridian, IAU-style.
sph.add_spherical_polygon(ax, MY_LON, MY_LAT, geodesic=False, facecolor=URAN[4],
                          edgecolor=URAN[4], alpha=0.28, lw=1.8, zorder=2)
sph.add_constellation_lines(ax, constellations=["UMA"], rank_max=1,
                            color=PAL["accent"], lw=1.5, alpha=0.9)
ax.text(196, 62.4, "my region", color=URAN[4], fontsize=9, ha="center",
        fontweight="bold", transform=tr, zorder=7)
ax.set_title("Your own boundary: a corner list, IAU-style", fontsize=10)

# %% [markdown]
# On the left, the solid hand-drawn line disappears into the bundled one — same
# stars, same order, same `ax.plot()`. The dashed line is that identical code
# fed different numbers. On the right, `geodesic=False` keeps every edge running
# along a parallel or a meridian, which is precisely the convention the real IAU
# boundaries follow; drop that argument and the edges become great-circle arcs
# instead.
#
# > **Tip:** to replace the bundled data wholesale rather than draw one figure,
# > pass `data_file=` to `add_constellation_boundaries` or
# > `add_constellation_lines`. Boundaries accept a Roman-format `.dat`, a
# > `.json` segment list, or an `.npz` corner list; lines accept an `.npz` in
# > the bundled schema. That's the hook for a higher-resolution or
# > differently-precessed set of your own.
#
# The Dipper's slow disintegration is even better to watch than to read about:
# **[Vector Fields and Sky Kinematics](vector_fields.ipynb)** draws the
# displacement arrows star by star, and **[Animations](animations.ipynb)** runs
# the whole 200,000-year morph as a movie. Five of the seven stars share a real
# common motion; the two at the ends are interlopers going their own way, which
# is why the shape shears.

# %% [markdown]
# ## 10. Decoration, not a planetarium
#
# An honest scope note. The constellation overlays are **cartographic furniture** —
# `add_constellation_*` draw boundary lines, stick figures, name labels, and fills
# to dress a map whose *subject* is your data. None of them carries a star catalog
# or reasons about brightness; each just reads a bundled coordinate table and hands
# it to matplotlib.
#
# The stars are yours to bring, and that is the pattern this whole notebook uses:
# scatter a catalog in world coordinates, size the markers by magnitude, color
# them by B−V with `sph.bv_to_rgb`, even propagate the proper motions forward as
# [section 9](#9.-Bring-your-own-lines-and-boundaries) does. skyplothelper bundles
# a small real catalog to make that concrete — the Hipparcos naked-eye sample in
# `examples/data/`, carrying `Vmag`, `BV`, and proper motions — but it is *example
# data* that you drive yourself, not a built-in sky database.
#
# Where skyplothelper stops is the planetarium engine: it has no comprehensive star
# or deep-sky catalog, no ephemerides, no automatic epoch/precession handling, no
# magnitude-limited symbol sets. For genuinely chart-grade output — magnitude-binned
# symbols from a full catalog, deep-sky object overlays, epoch-propagated positions,
# printable atlas pages — reach for the dedicated tools: [starplot](https://starplot.dev)
# for finished star charts and [skyfield](https://rhodesmill.org/skyfield/) for
# ephemerides and precise positions. They pair well: compute with skyfield, plot
# with either.
#
# And for the complementary question — not *where a source is* but *when and from
# where you can actually observe it* (airmass, rise/set, visibility windows,
# scheduling) — **obsplanning** builds that on top of skyfield's machinery. A finder chart
# like the one in [section 11](#11.-Putting-it-together) is something you take to the
# eyepiece for comparison once obsplanning has told you the target is up.

# %% [markdown]
# ## 11. Putting it together
#
# A finder chart on real sky. The backdrop is the NOIRLab all-sky panorama —
# a photographic Milky Way — reprojected onto an equatorial frame (the full
# recipe for that is in [A Tour of Projections](projections.ipynb)); the chart
# furniture goes on top with `stroke_color=` doing the work of keeping every
# line and label legible against the imagery. A reticle marks the target — swap
# in your own source, and this is your observing-night finder chart:

# %%
# fig-slug: finder-chart-capstone
img, hdr = sph.load_sky_image(f"{DATA}/Allsky_noirlab2430b_1280x640.jpg",
                              frame="galactic", center=0)

fig = plt.figure(figsize=(12.5, 6.4))
ax = sph.make_wcs_frame(111, projection="AIT", center=180, npix=(1200, 600), fig=fig)
fig.canvas.draw()
ax.imshow(sph.reproject_background(img, hdr, ax))

sph.add_constellation_boundaries(ax, color="white", lw=0.5, alpha=0.35)
sph.add_constellation_lines(ax, rank_max=1, color="#FFD97A", lw=1.1, alpha=0.95,
                            stroke_color="black", stroke_lw=2.4)
# Cassiopeia's name would overlap the +60° graticule label on this view, so
# it sits this chart out (its W still shows in the lines) — label placement
# on a finished chart always deserves a final look.
sph.add_constellation_labels(ax, labels="name",
                             constellations=[c for c in FAMOUS if c != "CAS"],
                             color="white", alpha=0.95, fontsize=9,
                             stroke_color="black", stroke_lw=2.4)
sph.add_reticle(ax, (83.82, -5.39), style="L", size=14, color="white",
                stroke_color="black", label="M42", label_color="white")
ax.set_title("A finder chart over the real sky", fontsize=12)

# %% [markdown]
# ## 12. Where to go next
#
# | If you want to... | Go to |
# |---|---|
# | reproject panoramas and images onto any sky frame | [A Tour of Projections](projections.ipynb) |
# | build set-algebra sky regions (and test membership) | [Regions & Spherical Polygons](regions.ipynb) |
# | the full globe toolkit — tilt, planets, nightshade | [Globe and Planet Plotting](globe_plots.ipynb) |
# | plot real catalogs (and query live archives) | [Catalogs: Querying & Plotting](catalogs.ipynb) |
# | draw the proper motions that reshape the asterisms | [Vector Fields and Sky Kinematics](vector_fields.ipynb) |
# | watch the Big Dipper come apart, frame by frame | [Animations](animations.ipynb) |
# | hover a constellation and see its name | [Interactive Plotting](interactive_plotly.ipynb) |
# | magnitude-scaled charts, DSOs, printable atlases | [starplot](https://starplot.dev) / [skyfield](https://rhodesmill.org/skyfield/) |
