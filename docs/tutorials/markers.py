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
# # Markers — Rotatable and Image Stamps
#
# Many points on a sky map can just be simple dots. But some markers carry a *direction*: a
# radio dish tilts toward the source it tracks, an observatory dome opens its slit
# to a patch of sky, an image stamp of the Moon should sit the right way up on a
# globe. skyplothelper draws these **pointable markers** two ways, and this
# notebook is the tour of both:
#
# - **Procedural instrument markers** — `add_antenna_marker`, `add_telescope_marker`,
#   `add_dome_marker` — little vector sprites you draw at a position and *aim*
#   (dish elevation, tube elevation, dome slit azimuth), with every part colorable.
# - **Image stamps** — `imscatter`, `imscatter_rotated`, `imscatter_globe` — drop a
#   raster picture (the bundled body/instrument icons, or your own PNG) at each
#   point, optionally rotated, and — on a globe — standing upright and facing the
#   right way.
#
# Two mechanisms, one goal: put a marker down and control where it points. Each
# section answers the two questions these markers raise — **"how do I show a marker
# that points?"** and **"how do I adjust its orientation, color, and size?"** — and
# we finish on the case these were initially built for: a **VLBI antenna network**, its
# baselines drawn across a map, every dish aimed at one celestial source.
#
# > **A note on the data.** The Earth raster (NASA Blue Marble topography) is large
# > and ships outside the pip package, so it lives in `examples/data/` locally; the
# > committed notebook outputs show every figure, and the code is exactly what
# > you'd run with the file in place. The bundled `icons/` (Sun, Moon, planets,
# > three instruments, a black hole) *are* in the repo.
#
# ## Contents
#
# 1. [Markers that point](#1.-Markers-that-point)
# 2. [Image stamps](#2.-Image-stamps)
# 3. [Markers on a globe](#3.-Markers-on-a-globe)
# 4. [A VLBA site map](#4.-A-VLBA-site-map)
# 5. [Pointing a network at a source](#5.-Pointing-a-network-at-a-source)
# 6. [Putting it together](#6.-Putting-it-together)
# 7. [Creating your own markers](#7.-Creating-your-own-markers)
# 8. [Where to go next](#8.-Where-to-go-next)

# %%
from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

import skyplothelper as sph

# A clean, in-theme feel throughout. set_style applies each layer (base / theme /
# palette) independently, so setting only the **structural** base is safe with the
# docs' light and dark passes alike — it leaves whatever theme/palette the page
# uses untouched.
sph.set_style(base="structural")


# Pick an in-theme annotation palette by reading the active page background, so a
# single code path adapts to the docs' light and dark renders.
def annotation_palette():
    r, g, b = to_rgb(mpl.rcParams["figure.facecolor"])
    dark = (0.299 * r + 0.587 * g + 0.114 * b) < 0.5
    return sph.ANNOTATION_PALETTES["dark" if dark else "publication"]


PAL = annotation_palette()

# Cycle palettes for multi-series markers (both dual-mode and transparency-safe).
URAN = sph.CYCLE_PALETTES["uranometria"]["colors"]
ATLAS = sph.CYCLE_PALETTES["atlas"]["colors"]

# A fixed brick red for marker edges. The cycle-palette fills below are already
# mode-invariant, so pinning the edge too keeps one pairing (blue + red, gray + red)
# on the light and dark pages alike, instead of the edge swinging to the dark
# theme's gold.
EDGE_RED = sph.ANNOTATION_PALETTES["publication"]["accent"]

DATA = "../../examples/data"
EARTH_DAY = f"{DATA}/world.topo.bathy.200412.3x5400x2700.jpg"

# %% [markdown]
# ## 1. Markers that point
#
# The three procedural markers are small cartoon vector *sprites* — a radio antenna, an
# optical refractor telescope, and an observatory dome. You place one with a position and,
# crucially, an **aim**:
#
# | marker | function | what points | aim parameter |
# |---|---|---|---|
# | radio antenna | `add_antenna_marker` | the parabolic dish | `dish_elev` (0 = horizon, 90 = zenith) |
# | optical telescope | `add_telescope_marker` | the tube | `tube_elev` (0 = horizon, 90 = zenith) |
# | observatory dome | `add_dome_marker` | the slit opening | `slit_azim` (compass bearing) |
#
# Every marker also takes a `rotation` (spin the whole sprite in the figure plane —
# on a globe, the aim solver of section 3 sets it for you), a `size` in display
# points, and
# separate `face_color` / `edge_color` plus a `stroke_color` / `stroke_lw` outline
# that keeps it legible on a busy background.
#
# Here they are at their defaults — the vocabulary before we start turning knobs:

# %%
# fig-slug: instrument-markers
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
for ax, fn, name in zip(
    axes,
    [sph.add_antenna_marker, sph.add_telescope_marker, sph.add_dome_marker],
    ["add_antenna_marker", "add_telescope_marker", "add_dome_marker"],
):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    # Markers are base-anchored — the *feet* land on the coord — so we place the
    # anchor below center (0.30) to sit the whole sprite in the middle of the box.
    # Warm fill, neutral ink edge: PAL['text'] is near-black on a light page and
    # off-white on a dark one, so the internal lines read in both docs modes.
    fn(ax, (0.5, 0.30), size=90, face_color=PAL["accent"], edge_color=PAL["text"])
    ax.set_title(name, fontsize=10, family="monospace")
fig.suptitle("Three procedural instrument markers, at their defaults", fontsize=12)
fig.tight_layout()

# %% [markdown]
# **To adjust where a marker points, set its aim parameter.** The dish sweeps from
# the horizon to the zenith with `dish_elev`; the dome slit swings across the face
# with `slit_azim` — the same marker, re-aimed:

# %%
# fig-slug: aiming-a-marker
fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))
# Feet on the coord (base-anchored), so the anchor sits below center to seat each
# sprite in its box. The dish swings tall at dish_elev=90, so the antenna row gets a
# little headroom (ylim to 1.12); the compact dome needs none. Marker sizes are
# picked so every pose fills the thumbnail with a comfortable margin.
for ax, elev in zip(axes[0], [0, 30, 60, 90]):
    ax.set(xlim=(0, 1), ylim=(0, 1.12), xticks=[], yticks=[])
    # Cool fill, fixed brick-red edge — the same pairing on light and dark pages.
    sph.add_antenna_marker(ax, (0.5, 0.33), dish_elev=elev, size=64,
                           face_color=URAN[0], edge_color=EDGE_RED)
    ax.set_title(f"dish_elev={elev}°", fontsize=9)
for ax, azim in zip(axes[1], [-60, -20, 20, 60]):
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    sph.add_dome_marker(ax, (0.5, 0.27), slit_azim=azim, size=66,
                        face_color=URAN[3], edge_color=EDGE_RED)
    ax.set_title(f"slit_azim={azim}°", fontsize=9)
axes[0, 0].set_ylabel("antenna dish", fontsize=10)
axes[1, 0].set_ylabel("dome slit", fontsize=10)
fig.suptitle("Aiming a marker: dish elevation (top), dome slit azimuth (bottom)",
             fontsize=12)
fig.tight_layout()

# %% [markdown]
# > **Note.** The dome is drawn face-on, so its slit is visible only while it faces
# > roughly toward the viewer (azimuths near the front); aim it behind the dome and
# > the opaque shell hides it — physically honest, but keep demo azimuths
# > front-facing if you want the slit to show.

# %% [markdown]
# **Color is per-part, and the stroke is what makes markers survive a busy
# background.** `face_color` fills the sprite, `edge_color` outlines its internal
# geometry, and `stroke_color` / `stroke_lw` draw a contrasting outline around
# the whole silhouette — set it to the opposite of your backdrop
# (dark stroke on a light page, light stroke on imagery) so the marker never
# dissolves into what's behind it:

# %%
# fig-slug: color-and-stroke
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
# A colored fleet, on a light panel. Feet on the coord (base-anchored), so the
# anchors sit below center to seat the sprites in the box.
axes[0].set(xlim=(0, 3), ylim=(0, 1), xticks=[], yticks=[])
for i, x in enumerate([0.5, 1.5, 2.5]):
    sph.add_antenna_marker(axes[0], (x, 0.32), dish_elev=55, size=52,
                           face_color=ATLAS[i], edge_color="0.2")
axes[0].set_title("face_color per marker", fontsize=10)
# The same marker, no stroke, over a dark patch — it vanishes at the edges.
axes[1].set_facecolor("0.15")
axes[1].set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
sph.add_antenna_marker(axes[1], (0.5, 0.30), dish_elev=55, size=68,
                       face_color="0.2", stroke_lw=0)
axes[1].set_title("no stroke on a dark bg", fontsize=10)
# Same, with a light stroke — legible again.
axes[2].set_facecolor("0.15")
axes[2].set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
sph.add_antenna_marker(axes[2], (0.5, 0.30), dish_elev=55, size=68,
                       face_color="0.2", stroke_color="white", stroke_lw=2.5)
axes[2].set_title("stroke_color='white'", fontsize=10)
fig.suptitle("Coloring the parts, and why the stroke matters", fontsize=12)
fig.tight_layout()

# %% [markdown]
# **Placing a marker in world coordinates.** By default the anchor is in *data*
# coordinates (`coord_type='pixel'`), which is what you want on a plain axes. On a
# sky frame, pass `coord_type='world'` and give `(lon, lat)` in degrees — the
# marker lands at that RA/Dec (or a `SkyCoord`). The marker is **base-anchored**, so
# it's the dish's *feet* that sit on the coordinate (the ✕), with the sprite standing
# up from there — which is why it reads a little off-center from the mark:

# %%
# fig-slug: dish-on-source
src = (83.63, 22.01)
fig = plt.figure(figsize=(5.2, 4.6))
ax = sph.make_wcs_frame(111, projection="TAN", center=src,
                        fov_deg=2.5, lon_spacing=1.0, lat_spacing=1.0,
                        grid=True, gridcolor=PAL["grid"])
sph.add_antenna_marker(ax, src, coord_type="world", dish_elev=60,
                       size=70, face_color=PAL["accent"], edge_color=PAL["text"],
                       stroke_color=PAL["fig_bg"], stroke_lw=2)
# Mark the exact anchor coordinate — the sprite's feet land here. PAL['accent2'] is
# the palette's contrasting accent (slate-blue on the light page, coral on the dark
# one), so the mark stands clear of the dish's accent color in both modes.
anchor_c = PAL["accent2"]
ax.plot(*src, marker="x", ms=11, mew=2.6, color=anchor_c, zorder=12,
        transform=ax.get_transform("world"))
ax.text(src[0], src[1] - 0.28, "anchor", color=anchor_c, fontsize=8.5, ha="center",
        va="top", transform=ax.get_transform("world"), zorder=12)
ax.set_title("A dish anchored at a source (coord_type='world')", fontsize=10)
fig.tight_layout()

# %% [markdown]
# > **Note — markers are furniture, not measurements.** These sprites *illustrate*
# > an instrument at a location and orientation; they don't compute pointing for
# > you. In [section 5](#5.-Pointing-a-network-at-a-source) we feed a real
# > azimuth/elevation into `dish_elev`, but the marker itself just draws what you
# > tell it. For measurement furniture (rulers, reticles, scale bars) see the
# > [Annotations and Overlays](annotations.ipynb) tutorial, which introduces these
# > same instrument markers briefly — this notebook is their deep-dive home.

# %% [markdown]
# **The other two markers, in action.** The antenna has been our workhorse, but
# `add_telescope_marker` and `add_dome_marker` aim the same way. On the left, two
# optical telescopes on a field, both slewed to one target: the tripod stays level
# on the ground (`rotation=0`) and only the *tube* tilts up to the target's screen
# angle (`tube_elev`, which `aim_angles` returns as `aim_angle`). On the right, one
# observatory dome whose slit tracks the Sun across the sky — as the Sun moves,
# `slit_azim` (a compass bearing) turns to keep the opening pointed at it.

# %%
# fig-slug: telescopes-and-dome
fig = plt.figure(figsize=(9, 4.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], hspace=0.45, wspace=0.25)

# Two optical telescopes: level tripods, tubes tilted to the target.
axT = fig.add_subplot(gs[0, :])
axT.set(xlim=(0, 10), ylim=(0, 3.4), xticks=[], yticks=[])
axT.set_aspect("equal")
axT.set_title("Two telescopes slewed to one target (level tripods)", fontsize=10)
target = (5.0, 3.0)
axT.plot(*target, marker="*", ms=22, color=PAL["accent2"], zorder=5)
axT.text(target[0] + 0.3, target[1], "target", color=PAL["label"], fontsize=8,
         va="center")
fig.canvas.draw()
# Tripod feet land on the coord (base-anchored), so a modest y sits each scope on a
# level ground line with room for the tube to rise toward the target. The sight line
# starts at the tube's objective — box.anchors.sight_line_origin(aim_angle) — not the
# pier foot, so the ray reads as leaving the optics.
for x in (1.8, 8.2):
    phi = sph.aim_angles(axT, (x, 0.55), target, marker="telescope",
                         target_coords="data")["aim_angle"]
    box = sph.add_telescope_marker(axT, (x, 0.55), tube_elev=phi, rotation=0, size=60,
                                   face_color=PAL["accent"], edge_color=PAL["text"])
    ox, oy = box.anchors.sight_line_origin(phi)
    axT.plot([ox, target[0]], [oy, target[1]], ls=(0, (1, 2.5)), color=PAL["grid"],
             lw=0.9, zorder=1)

# One dome, its slit tracking the Sun from morning (left) to afternoon (right).
sun_stamp = plt.imread(f"{DATA}/icons/sun2_120pix.png")
for i, (az, label) in enumerate([(-55, "morning"), (0, "noon"), (55, "afternoon")]):
    axd = fig.add_subplot(gs[1, i])
    axd.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    axd.set_aspect("equal")
    sph.add_dome_marker(axd, (0.5, 0.22), slit_azim=az, size=56,
                        face_color=PAL["accent"], edge_color=PAL["text"])
    sx = 0.5 + 0.32 * np.sin(np.radians(az))   # slit faces the Sun, clear of the dome
    sy = 0.74 + 0.20 * np.cos(np.radians(az))
    axd.add_artist(AnnotationBbox(OffsetImage(sun_stamp, zoom=0.20), (sx, sy),
                                  frameon=False))
    axd.set_title(label, fontsize=8)
fig.suptitle("Telescopes and domes aim the same way", fontsize=12)

# %% [markdown]
# ## 2. Image stamps
#
# Sometimes you don't want a vector sprite — you want a *picture* at each point: a
# photo of the Sun, a logo, a thumbnail. `imscatter` is scatter-with-images: pass
# `x`, `y`, an image (a path or an array), and a `zoom` factor. skyplothelper
# bundles a small icon set in `examples/data/icons/` to demo with — celestial
# bodies (Sun, Earth, Moon, Jupiter, Mars), three instruments (a radio dish, an
# optical telescope, a space telescope), and a supermassive black hole to stand in
# for a distant source. Here's the whole cast:

# %%
# fig-slug: imscatter-lineup
moon = plt.imread(f"{DATA}/icons/FullMoon_240x240.png")
jup = plt.imread(f"{DATA}/icons/Jupiter_120pix.png")
mars = plt.imread(f"{DATA}/icons/Mars_120pix.png")
sun = plt.imread(f"{DATA}/icons/sun2_120pix.png")
earth = plt.imread(f"{DATA}/icons/Earth_Western_Hemisphere_120pix.png")
dish_img = plt.imread(f"{DATA}/icons/RadioDish_250pix.png")
scope_img = plt.imread(f"{DATA}/icons/OpticalTelescope_250pix.png")
space_img = plt.imread(f"{DATA}/icons/SpaceTelescope_250pix.png")
smbh_img = plt.imread(f"{DATA}/icons/SMBH_250pix.png")

# (image, label, zoom) — zoom is per icon, since the source PNGs differ in size.
LINEUP = [
    (sun, "Sun", 0.42), (earth, "Earth", 0.42), (moon, "Moon", 0.21),
    (jup, "Jupiter", 0.42), (mars, "Mars", 0.42), (dish_img, "radio\ndish", 0.20),
    (scope_img, "optical\ntelescope", 0.20), (space_img, "space\ntelescope", 0.29),
    (smbh_img, "SMBH", 0.36),
]
fig, ax = plt.subplots(figsize=(13, 2.8))
ax.set(xlim=(0.3, len(LINEUP) + 0.7), ylim=(0.4, 1.6), yticks=[])
for i, (img, label, zoom) in enumerate(LINEUP, start=1):
    sph.imscatter([i], [1], img, ax=ax, zoom=zoom)
ax.set_xticks(range(1, len(LINEUP) + 1))
ax.set_xticklabels([lbl for _, lbl, _ in LINEUP], fontsize=7.5)
ax.set_title("imscatter — a raster picture at each point (the bundled icon set)",
             fontsize=11)

# %% [markdown]
# **`imscatter_rotated` spins each stamp.** Pass a `rotations` array (degrees) the
# same length as the points, and each image is turned before it's placed — the
# raster counterpart to a marker's `rotation`. Rotation is most useful for stamps
# that *have* an orientation (an arrow, a dish photo, a labeled logo); a round body
# like Mars mostly shows its surface features turning:

# %%
# fig-slug: imscatter-rotated
fig, ax = plt.subplots(figsize=(7, 2.4))
angles = [0, 45, 90, 135, 180]
ax.set(xlim=(0, 6), ylim=(0.4, 1.6), yticks=[])
sph.imscatter_rotated(np.arange(1, 6), [1] * 5, mars, rotations=angles, ax=ax,
                      zoom=0.5)
# Label each stamp with its rotation via the x-ticks.
ax.set_xticks(np.arange(1, 6))
ax.set_xticklabels([f"{a}°" for a in angles])
ax.set_title("imscatter_rotated — each stamp turned before placing", fontsize=11)

# %% [markdown]
# **Aiming a stamp at a target.** Instead of computing `rotations` yourself, hand
# `imscatter_rotated` an `aim_at` target and tell it which way the icon already
# points. That second number is the icon's **rest angle**: the direction its
# business end faces as drawn, in degrees CCW from screen-right. The bundled radio
# dish rests at 125° (up and to the left), so:
#
# - `aim_at=` — where to point (same coordinate machinery as the vector markers'
#   `aim_at`, but interpreted in this function's own *data* coords by default).
# - `rest_angle=` — what to rotate away from; the solver applies
#   `aim_angle − rest_angle` per icon.
# - `flip='auto'` (the default) — mirrors an icon when the target sits on its far
#   side, so the dish *leans* toward the star instead of rolling past vertical.
#
# All three dishes below come from a single call — the aim and the flip are solved
# per point:

# %%
# fig-slug: aiming-a-stamp
DISH_REST = 125.0    # this icon's native boresight, degrees CCW from screen-right

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.set(xlim=(0, 6), ylim=(0, 3.2), xticks=[], yticks=[])
ax.set_aspect("equal")
star = (3.0, 2.55)
ax.plot(*star, marker="*", ms=20, color=PAL["accent2"], zorder=5)
fig.canvas.draw()          # settle the transforms before the aim solver reads them
xs = [0.9, 3.0, 5.1]
sph.imscatter_rotated(xs, [0.75] * len(xs), dish_img, aim_at=star,
                      rest_angle=DISH_REST, ax=ax, zoom=0.28)
for x in xs:
    ax.plot([x, star[0]], [0.75, star[1]], ls=(0, (1, 2.5)), lw=0.8,
            color=PAL["grid"], zorder=1)
ax.set_title("Aiming a raster stamp: aim_at= plus the icon's rest_angle=",
             fontsize=11)

# %% [markdown]
# **The same recipe aims *any* oriented icon** — all you need is its rest angle. The
# three bundled instruments each point a different way as drawn:
#
# | icon | file | rest angle |
# |---|---|---|
# | radio dish | `RadioDish_250pix.png` | 125° (up and to the left) |
# | optical telescope | `OpticalTelescope_250pix.png` | 65° (up and to the right) |
# | space telescope | `SpaceTelescope_250pix.png` | 194° (aperture to the left) |
#
# Measure yours once — rotate it until its business end points straight up, and the
# rotation you applied is `90° − rest` — then `rest_angle=` carries it forever.
# Below, a ground dish, an optical telescope, and a space telescope all lock onto
# one supermassive black hole, each with nothing but its own rest angle to declare:

# %%
# fig-slug: aiming-any-stamp
ICONS = [(scope_img, 65.0, "optical telescope"),
         (space_img, 194.0, "space telescope"),
         (dish_img, 125.0, "radio dish")]

fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.set(xlim=(0, 6), ylim=(0, 3.5), xticks=[], yticks=[])
ax.set_aspect("equal")
src = (3.0, 2.85)
sph.imscatter([src[0]], [src[1]], smbh_img, ax=ax, zoom=0.34, autoscale=False)
ax.text(src[0], src[1] - 0.42, "one distant source", color=PAL["label"], fontsize=8.5,
        ha="center", va="top")
fig.canvas.draw()          # settle the transforms before the aim solver reads them
for x, (img, rest, label) in zip((0.85, 3.0, 5.15), ICONS):
    ax.plot([x, src[0]], [0.7, src[1]], ls=(0, (1, 2.5)), lw=0.8, color=PAL["grid"],
            zorder=1)
    sph.imscatter_rotated([x], [0.7], img, aim_at=src, rest_angle=rest, ax=ax,
                          zoom=0.22, autoscale=False)
    ax.text(x, 0.06, f"{label}\nrest = {rest:.0f}°", color=PAL["label"], fontsize=8,
            ha="center", va="bottom")
ax.set_title("One rest angle per icon, and they all point where you tell them",
             fontsize=11)

# %% [markdown]
# **Scaling a marker.** Each family has one size knob, and they measure different
# things. A procedural marker's `size` is in **display points** — the same units as
# a font size — so a `size=60` sprite is 60 points tall wherever you put it. An
# image stamp's `zoom` multiplies the icon's **native pixel** size, so the 250-px
# dish at `zoom=0.2` lands 50 px across. Neither is in data units: both stay put
# when you pan or zoom the axes, which is what you want for a site marker.
#
# `zoom` also takes an *array*, one value per point — the raster answer to
# `scatter(s=...)`, so you can size icons by dish diameter, flux, or whatever your
# data says. The whole right-hand row below is a single `imscatter` call:

# %%
# fig-slug: sizing-markers
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
for ax, knob in zip(axes, ["size", "zoom"]):
    ax.set(xlim=(0.3, 5.7), ylim=(0, 2), xticks=[], yticks=[])
    ax.set_title(f"{'procedural' if knob == 'size' else 'raster'}: "
                 f"{knob}= ({'display points' if knob == 'size' else '× native px'})",
                 fontsize=10)
for i, s in enumerate([20, 40, 60, 85, 115], start=1):
    # Feet on a common baseline (y=0.5); the sprites grow *upward* from it, so the
    # low anchor keeps even size=115 inside the frame.
    sph.add_antenna_marker(axes[0], (i, 0.5), dish_elev=55, size=s,
                           face_color=PAL["accent"], edge_color=PAL["text"])
    axes[0].text(i, 0.05, f"size={s}", ha="center", va="bottom", fontsize=7.5,
                 color=PAL["label"])
zooms = [0.06, 0.11, 0.17, 0.24, 0.32]
sph.imscatter(range(1, 6), [1.05] * 5, dish_img, ax=axes[1], zoom=zooms,
              autoscale=False)
for i, z in enumerate(zooms, start=1):
    axes[1].text(i, 0.05, f"zoom={z}", ha="center", va="bottom", fontsize=7.5,
                 color=PAL["label"])
fig.suptitle("Scaling a marker: size (vector) vs zoom (raster)", fontsize=12)
fig.tight_layout()

# %% [markdown]
# > **Vector vs raster — pick per job.** Procedural markers ([section 1](#1.-Markers-that-point))
# > scale crisply to any size, recolor freely, and stroke cleanly, so they're the
# > right call for instruments and schematic site maps. Image stamps carry *real
# > pictures* (a photo, a logo, a textured body) that no vector sprite can match —
# > at the cost of fixed resolution and no recoloring. Reach for the one whose
# > strength matches the figure.

# %% [markdown]
# ## 3. Markers on a globe
#
# On a globe a marker has to do two things: **stand on the curved surface**, and
# usually **point at something**. `aim_angles()` (and the `aim_at=` shortcut on the
# markers) solves both, in two modes:
#
# - **`aim_mode="planted"`** keeps the pedestal along the local vertical — *tangent*
#   to the globe, the way a real mount sits — and tilts only the dish toward the
#   target. It needs `globe_center` so it knows which way is "up" at each site.
# - **`aim_mode="aimed"`** ignores the local vertical and puts the whole sprite on the
#   target, mount and all.
#
# For the second one there's a knob worth knowing. By default `aim_mode="aimed"` lays
# the dish straight along the mount's own axis, which reads as an arrow more than an
# antenna. Usually you'd rather keep the dish at a believable **working elevation**
# above its base — say 60° — and swing the *whole rig* rigidly until the bowl lands on
# the source, letting the base fall where it may. That's `rest_elev`:
#
# ```python
# add_antenna_marker(ax, site, aim_at=source, aim_mode="aimed", rest_elev=60)
# ```
#
# `rest_elev` is the pose the instrument holds relative to its own mount, and the
# solver swings the sprite from there. It defaults to `90`, which is the dish-along-
# the-axis look; lower it for a more reclined rig. It works in `"planted"` mode too,
# and on the telescope, where holding the tube at `rest_elev` is exactly what you'd
# want.
#
# > **Don't hand-roll the rotations.** The two markers don't share a convention — the
# > antenna's outer `rotation` counts *twice* in the dish angle, the telescope's
# > counts once — so a formula that aims one will mis-aim the other. `aim_angles`
# > (and the `aim_at=` shortcut) is the one place that geometry lives.
#
# Same six sites and the same target (the star), the two treatments side by side:

# %%
# fig-slug: globe-planted-vs-aimed
cA = (-95.0, 20.0)
sat = (-55.0, 62.0)                # the shared target, up and to the right
sites3 = [(-140, 35), (-120, 8), (-95, 45), (-70, 22), (-105, -8), (-128, -20)]
fig = plt.figure(figsize=(9, 4.8))
axA = sph.make_globe_frame(121, center_LONdeg=cA[0], center_LATdeg=cA[1],
                           projection="SIN", grid=True, Naxispix=360)
axB = sph.make_globe_frame(122, center_LONdeg=cA[0], center_LATdeg=cA[1],
                           projection="SIN", grid=True, Naxispix=360)
fig.canvas.draw()   # settle the transforms before the aim solver reads them

REST_ELEV = 60.0    # the dish's working elevation above its own base

style = dict(size=34, edge_color=PAL["text"], stroke_color=PAL["fig_bg"],
             stroke_lw=1.6)
for ax in (axA, axB):
    ax.plot(*ax.wcs.wcs_world2pix([sat], 0)[0], marker="*", ms=17,
            color=PAL["label"], zorder=9)

# Left: pedestals stay tangent to the globe, only the dish tilts.
for lon, lat in sites3:
    sph.add_antenna_marker(axA, (lon, lat), coord_type="world", aim_at=sat,
                           aim_mode="planted", globe_center=cA,
                           target_coords="world", face_color=PAL["accent"], **style)
axA.set_title("planted — pedestal stays vertical (tangent)", fontsize=10)

# Right: the dish holds REST_ELEV on its base, and the whole rig swings on target.
for lon, lat in sites3:
    sph.add_antenna_marker(axB, (lon, lat), coord_type="world", aim_at=sat,
                           aim_mode="aimed", target_coords="world",
                           rest_elev=REST_ELEV,
                           face_color=PAL["accent2"], **style)
axB.set_title(f"aimed — dish held at {REST_ELEV:.0f}°, whole rig swung on target",
              fontsize=10)

# %% [markdown]
# Every dish in both panels points at the same star; they differ in what the *mount*
# is allowed to do. **Planted** is physically honest — each pedestal stands along its
# own local vertical, so you can read a site's latitude off the tilt — which usually
# makes it the more intuitive choice. **Aimed** gives up the local vertical to put the
# rig unambiguously on the source; holding the dish at a working elevation (rather
# than flat along the mount) keeps it reading as an antenna instead of an arrow.
# Lower `REST_ELEV` toward 45° for a more reclined pose, or raise it toward 90° to
# recover exactly what `aim_mode="aimed"` does on its own. Section 5 builds a real
# network in **planted** mode.

# %% [markdown]
# **Image stamps go on globes too.** On skyplothelper's own globe you place a stamp
# with `imscatter_rotated` and stand it upright with a rotation you choose — no extra
# machinery, no cartopy. The left panel drops the **Moon** on a celestial SIN globe
# (tilted 90° − latitude, like the antenna plots above); the right panel stands the
# bundled **radio dish** on a rasterized Earth globe, each dish rotated to its site's
# local vertical (read straight off the display transform):

# %%
# fig-slug: imscatter-globe
# Drape a raster onto a globe frame by resampling it onto the frame's own
# synthetic-WCS pixel grid (below everything else); stamps placed at the frame's
# pixel coords then land on it. (The Globe tutorial's recipe — reused in section 5.)
def drape(ax, path, zorder=-10):
    hdu = sph.pseudofits_from_image(path, geo=True)
    out_hdr = ax.wcs.to_header()
    nx = round(ax.get_xlim()[1] - ax.get_xlim()[0])
    ny = round(ax.get_ylim()[1] - ax.get_ylim()[0])
    out_hdr["NAXIS1"], out_hdr["NAXIS2"] = nx, ny
    bg = sph.reproject_rgb_map(hdu, out_hdr, shape_out=(ny, nx))
    return ax.imshow(np.nan_to_num(bg), origin="lower", zorder=zorder)


fig = plt.figure(figsize=(9.2, 4.8))

# Left: Moon stamps standing on a celestial SIN globe (tilt = 90 - latitude).
axL = sph.make_globe_frame(121, center_LONdeg=0, center_LATdeg=20,
                           projection="SIN", grid=True, Naxispix=360)
for lon, lat in [(-40, 25), (0, -10), (35, 40), (22, -32)]:
    x, y = axL.wcs.wcs_world2pix([[lon, lat]], 0)[0]
    sph.imscatter_rotated([x], [y], moon, rotations=[90 - lat], ax=axL,
                          zoom=0.13, autoscale=False)
axL.set_title("Moon stamps on an sph celestial globe", fontsize=10)

# Right: the bundled dish standing on a rasterized Earth globe. "Upright" here means
# aligned with the local vertical, which we read as the display-space direction from
# the globe center out to each site.
gc = (-95.0, 25.0)
axR = sph.make_planet_frame(122, body="earth", center_LONdeg=gc[0],
                            center_LATdeg=gc[1], Naxispix=420, grid=False,
                            tick_style="native")
drape(axR, EARTH_DAY)
for c in axR.coords:
    c.set_ticklabel_visible(False)
    c.set_ticks_visible(False)
fig.canvas.draw()                        # settle transforms before we read them
center_disp = axR.transData.transform(axR.wcs.wcs_world2pix([gc], 0)[0])
for lon, lat in [(-123, 47), (-100, 44), (-77, 41), (-113, 34)]:
    px, py = axR.wcs.wcs_world2pix([[lon, lat]], 0)[0]
    site_disp = axR.transData.transform((px, py))
    up = np.degrees(np.arctan2(site_disp[1] - center_disp[1],
                               site_disp[0] - center_disp[0]))
    sph.imscatter_rotated([px], [py], dish_img, rotations=[up - 90], ax=axR,
                          zoom=0.13, autoscale=False)
axR.set_title("Radio dishes standing on a rasterized Earth globe", fontsize=10)
fig.suptitle("Image stamps on skyplothelper globes — no cartopy required",
             fontsize=12)
fig.tight_layout()

# %% [markdown]
# **When you do want cartopy** — for coastlines, political borders, or its wide
# projection set — `imscatter_globe` is the convenience wrapper. Hand it an
# `(N, 2)` array of `[lon, lat]` and it stands each icon up (90° − latitude) and
# mirrors the ones on the far hemisphere, on a cartopy `GeoAxes`. It takes the same
# `rest_angle` as `imscatter_rotated` (defaulting to 45°, an icon leaning to the
# upper-right), so any boresight stands up correctly — here the optical telescope's
# 65°.

# %%
# fig-slug: imscatter-globe-cartopy
import cartopy.crs as ccrs  # noqa: E402  (imported here to keep it optional)

fig = plt.figure(figsize=(5.2, 5.2))
ax = sph.make_cartopy_frame(111, projection="orthographic", center=(-95, 25))
ax.coastlines(lw=0.5, color=PAL["grid"])
ax.gridlines(color=PAL["grid"], alpha=0.4)
stamp_coords = np.array([[-125, 40], [-70, 45], [-95, 5], [-140, 25], [-60, -15]])
sph.imscatter_globe(ax, stamp_coords, ccrs.PlateCarree(), -95, scope_img, zoom=0.21,
                    rest_angle=65.0)
ax.set_title("imscatter_globe — the cartopy convenience (coastlines + auto-tilt)",
             fontsize=10)
fig.tight_layout()

# %% [markdown]
# > **Note — sph globe or cartopy?** Both place stamps on a globe; pick by what else
# > you need. Reach for a plain skyplothelper globe (`make_globe_frame`, or
# > `make_planet_frame` for a body surface) when you'd rather not add the
# > dependency — the two panels above stood stamps on one with a single
# > `imscatter_rotated` call each. Reach for `imscatter_globe` when you specifically
# > want cartopy's map features or its automatic hemisphere mirroring. Watch the two
# > senses of "flip": `imscatter_globe` mirrors by *hemisphere* (which side of the
# > central meridian a site falls on), while `imscatter_rotated`'s `flip=` mirrors by
# > *target* (which side of the icon the thing it's aiming at falls on). Both take
# > `rest_angle`, so neither needs you to pre-mirror the image. Back-hemisphere
# > stamps are still drawn (folded onto the visible disk) either way — keep sites on
# > the near side, or center the globe on them.

# %% [markdown]
# ## 4. A VLBA site map
#
# Now the case these tools were built for. The **Very Long Baseline Array** is ten
# radio antennas spread across North America (plus Hawaii and St. Croix); together
# they synthesize an Earth-sized telescope. We'll put the real network on a map:
# the Blue Marble topography as a backdrop, a great-circle **baseline** between
# every pair of antennas, and a **dish marker at each site**.
#
# The station coordinates (longitude east, latitude, in degrees) — none are bundled
# here (though they are built-ins in the sibling package
# [obsplanning](https://obsplanning.readthedocs.io)) — so we list the public site
# positions directly:

# %%
VLBA = {
    "BR": (-119.68, 48.13), "OV": (-118.28, 37.23), "KP": (-111.61, 31.96),
    "PT": (-108.12, 34.30), "LA": (-106.25, 35.78), "FD": (-103.94, 30.63),
    "NL": (-91.57, 41.77), "HN": (-71.99, 42.93),
    "MK": (-155.46, 19.80), "SC": (-64.58, 17.76),  # non-CONUS: Hawaii, St. Croix
}
# A couple of other US dishes, shown but not part of the VLBA baseline network.
OTHER = {"VLA": (-107.62, 34.08), "GBT": (-79.84, 38.43)}


# %% [markdown]
# `plot_baselines` draws a great-circle arc between every pair of sites (or a
# chosen list of `pairs`), labels the stations, and returns the artists it made.
# It works directly on a **plain lon/lat axes** — degrees east, degrees north —
# which is all a regional map needs. We drop the equirectangular Blue Marble in
# with `imshow` and an `extent` in degrees, crop to North America with the axis
# limits, and keep an equal aspect (a plate-carrée map, 1° lon = 1° lat). The
# baselines and every marker then live in those same lon/lat data coordinates —
# no world transform, no wrap bookkeeping on your part:

# %%
# fig-slug: vlba-site-map
earth_img = plt.imread(EARTH_DAY)   # equirectangular RGB: lon -180..180, lat 90..-90

fig, ax = plt.subplots(figsize=(9.5, 5.4))
ax.imshow(earth_img, extent=[-180, 180, -90, 90], origin="upper", zorder=-10)
ax.set_xlim(-170, -45)              # crop to the array's footprint (Hawaii → Caribbean)
ax.set_ylim(7, 62)
ax.set_aspect("equal")
ax.set_xlabel("longitude (°E)")
ax.set_ylabel("latitude (°N)")

# Great-circle baselines across all ten VLBA antennas (plain axes, degrees).
sph.plot_baselines(ax, VLBA, color="gold", linewidth=0.8, alpha=0.85,
                   show_markers=False, show_site_labels=True,
                   site_label_color="white", site_label_fontsize=7.5)

# A dish at each VLBA site: CONUS in white, the two outliers in orange-red.
for name, (lon, lat) in VLBA.items():
    fc = "orangered" if name in ("MK", "SC") else "white"
    sph.add_antenna_marker(ax, (lon, lat), dish_elev=55, size=15, face_color=fc,
                           stroke_color="black", stroke_lw=1.4)

# Other dishes — shown, but without baselines, in a third color.
for name, (lon, lat) in OTHER.items():
    sph.add_antenna_marker(ax, (lon, lat), dish_elev=55, size=15,
                           face_color="deepskyblue", stroke_color="black",
                           stroke_lw=1.4)
    ax.text(lon + 1.2, lat + 1.2, name, color="deepskyblue", fontsize=7.5,
            fontweight="bold")

ax.set_title("The VLBA — baselines and dish markers on the Blue Marble",
             fontsize=12)
fig.tight_layout()

# %% [markdown]
# The white dishes are the ten VLBA antennas, wired together by the golden baseline
# fan; **MK** (Mauna Kea, Hawaii) and **SC** (St. Croix) sit far off the CONUS
# cluster in orange-red, and their long baselines are exactly what give the array
# its highest resolution. The blue **VLA** and **GBT** dishes ride along as context —
# real instruments, but not part of *this* network, so no baselines are drawn for them.
#
# > **Scope.** `plot_baselines` has a deep toolkit of its own — baseline *lengths*
# > (`show_lengths`, `length_unit`), back-hemisphere styling on globes, custom
# > `pairs`, and non-Earth bodies. The [Globe and Planet Plotting](globe_plots.ipynb)
# > tutorial is its home; here we use it in service of the marker map.
#
# > **Tip — a WCS map works too.** The plain-axes `imshow(extent=…)` route above is
# > the simplest for a regional crop, but you can also use a full WCS map with a
# > graticule: `make_wcs_frame("CAR", frame="ITRS", direction="geo")` and drape the
# > texture as in [section 5](#5.-Pointing-a-network-at-a-source). The `frame="ITRS"`
# > is the key — it puts the frame in the same geographic system as the Earth
# > texture, so the two align and `coord_type="world"` markers land correctly.

# %% [markdown]
# ## 5. Pointing a network at a source
#
# A dish doesn't just sit somewhere — it *tracks* a source. Here we aim a whole
# network at one target on a globe. We'll do it twice: first the quick,
# illustrative way for a slide; then the rigorous way, with real geometry, for a
# figure that has to be *correct*.
#
# **Illustrative first.** For a schematic — "here's our array, all watching the same
# thing" — the photographic dish stamp is hard to beat. Drop one at each site with
# the `aim_at` / `rest_angle` recipe from [section 2](#2.-Image-stamps) so every
# bowl swings onto the source, draw the source itself out in "space", and let the
# dotted sight lines converge. A couple of small helpers set up the "space" overlay
# and the globe; the ten dishes are one `imscatter_rotated` call.

# %%
# fig-slug: network-on-one-source
# The drape() helper (rasterizing a texture onto a globe frame) was defined back in
# section 3; here we add two more small helpers for the "source out in space" setup.
def space_overlay(fig):
    """A spine-less, full-figure axes for drawing 'in space' (a source icon and
    sight lines) *outside* the globe's circular frame, in figure coordinates."""
    ov = fig.add_axes([0, 0, 1, 1], zorder=5)
    ov.set_axis_off()
    ov.set_xlim(0, 1)
    ov.set_ylim(0, 1)
    return ov


def bare_globe(ax):
    """Hide the coordinate ticks/labels on a schematic globe — these figures are
    about where the dishes point, not about reading off a grid."""
    for c in ax.coords:
        c.set_ticklabel_visible(False)
        c.set_ticks_visible(False)


def aim_dish(ax, lon, lat, target_disp, globe_center, **kw):
    """Plant an antenna at (lon, lat) — pier along the local vertical — and tilt
    its dish toward ``target_disp`` (an xy in *display* coordinates) with the
    package's ``aim_at=`` solver; return the site's display xy so the caller can
    draw the sight line. Here the target is a point out in "space" (an overlay
    position), so we pass it in display coords."""
    dd = ax.transData.transform(ax.wcs.wcs_world2pix([[lon, lat]], 0)[0])
    sph.add_antenna_marker(ax, (lon, lat), coord_type="world", aim_at=target_disp,
                           aim_mode="planted", globe_center=globe_center,
                           target_coords="display", **kw)
    return dd


gcenter = (-95.0, 22.0)
fig = plt.figure(figsize=(8, 5.6))
gs = fig.add_gridspec(1, 100)
ax = sph.make_planet_frame(gs[0, :64], body="earth", center_LONdeg=gcenter[0],
                           center_LATdeg=gcenter[1], Naxispix=460, grid=False,
                           tick_style="native")
drape(ax, EARTH_DAY)
bare_globe(ax)
ov = space_overlay(fig)
fig.canvas.draw()                       # settle the transforms before we read them
src_fig = np.array([0.83, 0.70])        # the source, out in space (figure coords)
src_disp = fig.transFigure.transform(src_fig)
to_fig = fig.transFigure.inverted().transform
ov.add_artist(AnnotationBbox(OffsetImage(smbh_img, zoom=0.42), src_fig,
                             frameon=False, zorder=7))
ov.text(*(src_fig + [0, -0.10]), "one distant\nsource", color=PAL["label"],
        fontsize=9, ha="center", va="top")
# On a WCSAxes the data coords *are* the frame's pixel coords, so convert the sites
# once and let one imscatter_rotated call aim (and flip) all ten dishes. The source
# lives on the overlay axes, so it comes in as a display-coord target.
site_pix = np.array([ax.wcs.wcs_world2pix([[lon, lat]], 0)[0]
                     for lon, lat in VLBA.values()])
sph.imscatter_rotated(site_pix[:, 0], site_pix[:, 1], dish_img, aim_at=src_disp,
                      rest_angle=DISH_REST, target_coords="display", ax=ax,
                      zoom=0.13, zorder=6, autoscale=False)
for df in to_fig(ax.transData.transform(site_pix)):
    ov.plot([df[0], src_fig[0]], [df[1], src_fig[1]], ls=(0, (1, 2.5)), lw=0.8,
            color="gold", alpha=0.7, zorder=4)
fig.suptitle("A network watching one source (slide-ready)", fontsize=12, x=0.36)

# %% [markdown]
# Every dish leans toward the same point in space, sight lines converging — the
# instantly-readable "our whole array is on this source" picture for a talk.
#
# But notice what a photo stamp *can't* do: it's one rigid picture, so aiming it
# swings the mount along with the bowl. That's fine here — nobody reads a pedestal
# off a slide — and it's exactly why the next figure switches back to the vector
# marker, which has **parts**: a pier that can stay planted on the local vertical
# while only the dish tilts, and a face color that can gray out on command.
#
# **Now the rigorous version.** At a *real* instant not every station can see the
# source: as the Earth turns, a source rises in the east and sets in the west, so at
# any moment some dishes are tracking it and others have it below the horizon. We
# compute each site's altitude with astropy's `AltAz`, aim the ones that can see it
# (planted, so each mount still stands on its own ground), and stow the rest:

# %%
# fig-slug: who-can-see-it
# A classic bright VLBI target and a fixed instant. sph.SKY_POSITIONS holds
# ready-made SkyCoords for common sources; SkyCoord.from_name("Cygnus A") would
# resolve it online just as well.
target = sph.SKY_POSITIONS["cyg_a"]
when = Time("2024-09-21T12:00:00")

fig = plt.figure(figsize=(8, 5.6))
gs = fig.add_gridspec(1, 100)
ax = sph.make_planet_frame(gs[0, :64], body="earth", center_LONdeg=gcenter[0],
                           center_LATdeg=gcenter[1], Naxispix=460, grid=False,
                           tick_style="native")
drape(ax, EARTH_DAY)
bare_globe(ax)
ov = space_overlay(fig)
fig.canvas.draw()
src_fig = np.array([0.83, 0.72])
src_disp = fig.transFigure.transform(src_fig)
to_fig = fig.transFigure.inverted().transform
ov.add_artist(AnnotationBbox(OffsetImage(smbh_img, zoom=0.42), src_fig,
                             frameon=False, zorder=7))
ov.text(*(src_fig + [0, -0.10]), "Cygnus A", color=PAL["label"], fontsize=9,
        ha="center", va="top")

n_up = 0
for name, (lon, lat) in VLBA.items():
    site = EarthLocation(lon=lon * u.deg, lat=lat * u.deg)
    elev = float(target.transform_to(AltAz(obstime=when, location=site)).alt.deg)
    if elev > 0:                       # source is up: track it, draw the sight line
        n_up += 1
        dd = aim_dish(ax, lon, lat, src_disp, gcenter, size=24, face_color="white",
                      stroke_color="black", stroke_lw=1.3, zorder=6)
        df = to_fig(dd)
        ov.plot([df[0], src_fig[0]], [df[1], src_fig[1]], ls=(0, (1, 2.5)),
                lw=0.8, color="gold", alpha=0.7, zorder=4)
    else:                              # below the horizon: stow it (planted), dimmed
        sph.add_antenna_marker(ax, (lon, lat), coord_type="world", aim_at=src_disp,
                               aim_mode="planted", globe_center=gcenter,
                               target_coords="display", size=20, face_color="0.5",
                               stroke_color="0.15", stroke_lw=1.1, zorder=6)

fig.suptitle(f"Only {n_up}/10 stations can see it right now\n"
             f"(Cygnus A, {when.iso[:16]} UT — grayed dishes are stowed)",
             fontsize=11, x=0.36)

# %% [markdown]
# Now only the stations that can *physically* see Cygnus A at this instant lock onto
# it (gold sight lines); the rest sit stowed and gray. At 12:00 UT the source has
# already set across the central and eastern array — just the western dishes (and
# Mauna Kea) still have it up. Advance `when` and the Earth turns under the source:
# stations acquire it in the east and lose it in the west, dish by dish. That
# frame-by-frame march is exactly the setup for an [animation](animations.ipynb).
#
# > **Note.** The source is drawn schematically out in "space" (a spine-less overlay
# > axes over the globe) so the sight lines have somewhere to converge; its exact
# > screen position is for illustration, not a real sky position. The physics that
# > *is* real here is the visibility — which stations can see it, from `AltAz`.

# %% [markdown]
# ## 6. Putting it together
#
# Your own version of this is a **site network** — your array, your partner
# stations, your target. Here's the capstone on that shape: a handful of sites you
# define, their baselines across the globe, dish markers standing at each, and a
# labeled target out in space that they all lock onto. It reuses the `drape`,
# `space_overlay`, and `aim_dish` helpers from above; swap in your coordinates and
# it's your figure.

# %%
# fig-slug: capstone-network
# --- your network -------------------------------------------------------------
my_sites = {
    "Home":  (-105.0, 40.0),
    "North": (-120.0, 55.0),
    "East":  (-70.0, 43.0),
    "South": (-95.0, 19.0),
}
my_center = (-95.0, 30.0)
# -----------------------------------------------------------------------------

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(1, 100)
ax = sph.make_planet_frame(gs[0, :66], body="earth", center_LONdeg=my_center[0],
                           center_LATdeg=my_center[1], Naxispix=520, grid=False,
                           tick_style="native")
drape(ax, EARTH_DAY)
bare_globe(ax)

# Baselines across your network.
sph.plot_baselines(ax, my_sites, color=URAN[2], linewidth=1.2, alpha=0.9,
                   show_markers=False, show_site_labels=False)

ov = space_overlay(fig)
fig.canvas.draw()
src_fig = np.array([0.82, 0.74])
src_disp = fig.transFigure.transform(src_fig)
to_fig = fig.transFigure.inverted().transform
ov.add_artist(AnnotationBbox(OffsetImage(smbh_img, zoom=0.40), src_fig,
                             frameon=False, zorder=7))
ov.text(*(src_fig + [0, -0.10]), "M87", color=PAL["label"], fontsize=10,
        ha="center", va="top")

# A dish planted at each site, tilted toward the target.
for name, (lon, lat) in my_sites.items():
    dd = aim_dish(ax, lon, lat, src_disp, my_center, size=32, face_color="white",
                  stroke_color="black", stroke_lw=1.5, zorder=6)
    df = to_fig(dd)
    ov.plot([df[0], src_fig[0]], [df[1], src_fig[1]], ls=(0, (1, 2.5)), lw=0.9,
            color="gold", alpha=0.7, zorder=4)
    ax.text(lon + 1.5, lat + 2.5, name, color="white", fontsize=9,
            fontweight="bold", transform=ax.get_transform("world"),
            path_effects=[pe.withStroke(linewidth=2, foreground="black")])

fig.suptitle("Your network, all eyes on M87", fontsize=13, x=0.34)

# %% [markdown]
# ## 7. Creating your own markers
#
# The built-in markers aren't magic — each one is a handful of matplotlib patches
# whose vertices are laid out in a small local frame and then spun by a 2×2 rotation
# matrix. That's a recipe you can copy. Here's a minimal **telescope-on-a-tripod**: a
# triangular base plus a tube that tilts to `tube_elev`, built the same way
# skyplothelper builds its own markers (`points @ R.T`, then drop at the anchor):

# %%
# fig-slug: diy-marker
from matplotlib.patches import Circle, Polygon  # noqa: E402


def simple_scope(ax, xy, tube_elev=45.0, rotation=0.0, size=1.0,
                 face=PAL["accent"], edge=PAL["frame"], lw=1.6):
    """A DIY rotatable marker: a triangular base + a tube that tilts to tube_elev.
    Vertices live in a local frame (units of ``size``), get rotated by a 2x2 matrix,
    then land at ``xy`` in data coords — the pattern the sph markers use inside."""
    def rot(deg):
        t = np.radians(deg)
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s], [s, c]])
    xy = np.asarray(xy, float)
    R = rot(rotation)
    # Base: a squat triangle standing on the ground, apex at the tube pivot.
    base = np.array([(-0.45, -0.5), (0.45, -0.5), (0.0, 0.0)]) * size
    ax.add_patch(Polygon(base @ R.T + xy, closed=True, facecolor=face,
                         edgecolor=edge, lw=lw, zorder=3))
    # Tube: a thin bar along +x in its own frame, tilted by tube_elev (+ rotation).
    pivot = np.array([0.0, 0.0]) @ R.T + xy
    Rt = rot(tube_elev + rotation)
    tube = np.array([(-0.15, -0.09), (0.9, -0.09),
                     (0.9, 0.09), (-0.15, 0.09)]) * size
    ax.add_patch(Polygon(tube @ Rt.T + pivot, closed=True, facecolor=face,
                         edgecolor=edge, lw=lw, zorder=4))
    # Objective lens: a dot at the far (sky) end of the tube.
    lens = np.array([0.9, 0.0]) * size @ Rt.T + pivot
    ax.add_patch(Circle(lens, 0.11 * size, facecolor="white", edgecolor=edge,
                        lw=lw, zorder=5))


fig, axes = plt.subplots(1, 4, figsize=(9, 2.8))
for ax, elev in zip(axes, [0, 30, 60, 90]):
    ax.set(xlim=(-1, 1), ylim=(-1.0, 1.4), xticks=[], yticks=[])
    ax.set_aspect("equal")
    simple_scope(ax, (0, -0.4), tube_elev=elev, size=1.0)
    ax.set_title(f"tube_elev={elev}°", fontsize=9)
fig.suptitle("A do-it-yourself rotatable marker (patches + a rotation matrix)",
             fontsize=12)
fig.tight_layout()

# %% [markdown]
# That's the whole idea. **To aim your own marker at a target**, point it with the
# screen angle to that target — `np.degrees(np.arctan2(Δy, Δx))` — which is exactly
# what `aim_angles` computes before it applies each built-in's rotation convention.
# Two DIY scopes slewed to one star:

# %%
# fig-slug: diy-marker-aimed
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.set(xlim=(0, 10), ylim=(0, 3.4), xticks=[], yticks=[])
ax.set_aspect("equal")
star = (5.0, 3.0)
ax.plot(*star, marker="*", ms=20, color=PAL["accent2"], zorder=6)
for x in (1.8, 8.2):
    ang = np.degrees(np.arctan2(star[1] - 0.7, star[0] - x))
    ax.plot([x, star[0]], [0.7, star[1]], ls=(0, (1, 2.5)), color=PAL["grid"],
            lw=0.9, zorder=1)
    simple_scope(ax, (x, 0.7), tube_elev=ang, size=0.9)
ax.set_title("Aiming a DIY marker: tube_elev = atan2(dy, dx) to the target",
             fontsize=11)
fig.tight_layout()

# %% [markdown]
# From here the vector route goes one way and the raster route goes another. For a
# **crisper, reusable** vector version, wrap the patches in a `DrawingArea` inside an
# `AnchoredOffsetbox` (as the built-ins do) so the marker holds a fixed size in
# points across resize and zoom — see `skyplothelper.overlays.instruments` for the
# full pattern, including stroke handling and pixel-stable labels. The other route is
# a **photographic** icon — a real dish, a spacecraft, or, next, a real galaxy pulled
# straight from a sky survey.

# %% [markdown]
# ### A photographic icon from real survey data
#
# The dish and telescope stamps are commissioned art, but you can also build a
# marker from *real data*. Let's make a spiral-galaxy icon from **M74 (NGC 628)**,
# the archetypal grand-design face-on spiral, in three steps: query a few bands,
# composite them, then cut the galaxy onto a transparent sky so it drops anywhere.
#
# **1. Query the bands.** `download_hips` pulls a tangent-plane FITS cutout from any
# HiPS survey — one call per band, same center and field of view. Small copies of
# these cutouts ship in `examples/data/query_cache/`, and the helper below prefers
# them, fetching live only when a file is missing. That keeps the figures below
# reproducible — the HiPS service resamples a little differently call to call, which
# a percentile stretch is more than happy to amplify — while leaving the live query
# right here to copy:

# %%
QUERY_CACHE = Path("../../examples/data/query_cache")
M74 = SkyCoord(24.1737, 15.7835, unit="deg")   # NGC 628; a fixed coord runs offline
GAL_FOV, GAL_PIX = 0.17, 400                    # ~10' field, 400 px — icon-grade


def cached_band(hips_id, name):
    """One band: use the bundled cutout if we have it, else fetch and cache it.

    This is *cache-first*, unlike the live-first helper in the Catalogs tutorial,
    and deliberately so: a catalog query returns the same rows every time, but the
    HiPS service resamples slightly differently per call — small in the data, yet
    enough after a percentile stretch to visibly shift the composite. Pinning to
    the cached cutout keeps this figure reproducible, and identical whether you
    run it online or off.
    """
    path = QUERY_CACHE / name
    if not path.exists():        # first run anywhere: fetch it, then keep it
        data, hdr = sph.download_hips(M74, hips_id=hips_id, size=GAL_FOV,
                                      pixels=GAL_PIX)
        fits.PrimaryHDU(np.asarray(data, "float32"), header=hdr).writeto(path)
    return np.nan_to_num(np.asarray(fits.getdata(path), float))


# SDSS gri (optical) and AllWISE W3/W2/W1 (mid-IR — W3 traces the dusty arms).
g = cached_band("CDS/P/SDSS9/g", "m74_sdss_g.fits")
r = cached_band("CDS/P/SDSS9/r", "m74_sdss_r.fits")
i_ = cached_band("CDS/P/SDSS9/i", "m74_sdss_i.fits")
w1 = cached_band("CDS/P/allWISE/W1", "m74_wise_w1.fits")
w2 = cached_band("CDS/P/allWISE/W2", "m74_wise_w2.fits")
w3 = cached_band("CDS/P/allWISE/W3", "m74_wise_w3.fits")
# The ready-made SDSS color HiPS: a finished RGB, no compositing needed.
sdss_color = plt.imread(QUERY_CACHE / "m74_sdss_color.png")[::-1, :, :3]

# %% [markdown]
# **2. Composite.** A simple RGB composite is just three arrays stacked into R, G, B channels — the
# reddest band on red, bluest on blue — each stretched so faint structure shows. An
# `arcsinh` stretch (bright core *and* faint arms at once) is the workhorse. Three
# looks at the same galaxy: our SDSS `gri` stack, an AllWISE stack where the arms
# glow in the mid-IR, and the survey's own ready-made color image.


# %%
def stretch(chan, lo=30, hi=99.5, soft=0.12):
    """Background-subtract, then arcsinh-stretch one channel to [0, 1]."""
    base = np.percentile(chan, lo)
    x = np.clip(chan - base, 0, None)
    top = np.percentile(x, hi)
    return np.zeros_like(x) if top <= 0 else np.arcsinh(x / top / soft) / np.arcsinh(1 / soft)


def composite(red, grn, blu):
    return np.clip(np.dstack([stretch(red), stretch(grn), stretch(blu)]), 0, 1)


# fig-slug: galaxy-composites
rgb_sdss = composite(i_, r, g)          # R<-i, G<-r, B<-g
rgb_wise = composite(w3, w2, w1)        # R<-W3 (dust), G<-W2, B<-W1 (old stars)

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
for ax, img, title in zip(
        axes, [rgb_sdss, rgb_wise, sdss_color],
        ["SDSS gri (composited here)", "AllWISE W3/W2/W1", "SDSS color (ready-made)"]):
    ax.imshow(img, origin="lower")
    ax.set(xticks=[], yticks=[], title="")
    ax.set_title(title, fontsize=10)
fig.suptitle("M74 — three routes to a composite", fontsize=12)
fig.tight_layout()

# %% [markdown]
# **3. Cut it onto a transparent sky.** A marker shouldn't carry a black square, so
# we give the image an **alpha channel** built from its own brightness: opaque where
# the galaxy is bright, fading to fully transparent out in the sky. That's a
# luminance, stretched and used as alpha — no extra data. The one subtlety is the
# *shape* of that fade: a straight ramp leaves the disk semi-transparent, which looks
# fine on a dark page but washes out over a light one. Passing the alpha through a
# **concave** curve (`gamma < 1`, and `gamma=0.5` is just a square root) pulls the
# galaxy body up toward opaque while still letting the faint outskirts fade — so the
# same icon reads on either background. (`multicolorfits` offers a richer version with
# smoother color and rolloff; this plain-`numpy` recipe gets the look with the tools
# already in hand.)


# %%
def to_transparent(rgb, lo=55, hi=99.3, gamma=0.5):
    """Alpha from a stretched luminance: bright galaxy -> opaque, sky -> transparent.

    ``gamma`` shapes the fade. ``gamma=1`` is a straight ramp; ``gamma<1`` is concave
    (``0.5`` = square root), lifting the mid-brightness galaxy body toward opaque so it
    holds up on a light background while the outskirts still fade to clear.
    """
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    a_lo, a_hi = np.percentile(lum, lo), np.percentile(lum, hi)
    alpha = np.clip((lum - a_lo) / (a_hi - a_lo + 1e-9), 0, 1) ** gamma
    return np.dstack([rgb, alpha])


# %%
# fig-slug: galaxy-alpha
galaxy_icon = to_transparent(rgb_sdss)   # the finished RGBA marker

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
axes[0].imshow(rgb_sdss, origin="lower")
axes[0].set_title("RGB composite (opaque)", fontsize=10)
axes[1].imshow(galaxy_icon[..., 3], origin="lower", cmap="gray")
axes[1].set_title("alpha, from luminance", fontsize=10)
checker = (np.indices((16, 16)).sum(0) % 2)
axes[2].imshow(checker, cmap="binary", vmin=-1, vmax=2,
               extent=[0, GAL_PIX, 0, GAL_PIX], interpolation="nearest")
axes[2].imshow(galaxy_icon, origin="lower")
axes[2].set_title("RGBA icon (sky is transparent)", fontsize=10)
for ax in axes:
    ax.set(xticks=[], yticks=[])
fig.suptitle("From opaque frame to transparent-sky icon", fontsize=12)
fig.tight_layout()

# %% [markdown]
# **The payoff.** That RGBA array is now a marker like any other — hand it to
# `imscatter` and it drops onto a plot with no black box, blending into whatever's
# behind it. Here the *same* M74 stamps, at varying `zoom`, stand in for a field of
# galaxies on a dark sky frame and on a light one — the concave alpha keeps the
# galaxy body reading on both:

# %%
# fig-slug: galaxy-in-action
rng = np.random.default_rng(7)
gal_lons = 180 + rng.uniform(-5.5, 5.5, 9)
gal_lats = 20 + rng.uniform(-4.5, 4.5, 9)
gal_zooms = rng.uniform(0.10, 0.26, 9)

fig = plt.figure(figsize=(10.5, 4.6))
for sub, (bg, grid, tcol, blabel) in enumerate([
        ("#0b1020", "0.5", "0.85", "dark sky frame"),
        ("#f0efe9", "0.6", "0.2", "light background")], start=1):
    ax = sph.make_wcs_frame(120 + sub, projection="TAN", center=(180.0, 20.0),
                            fov_deg=14, lon_spacing=3.0, lat_spacing=3.0,
                            grid=True, gridcolor=grid)
    ax.set_facecolor(bg)
    for lon, lat, z in zip(gal_lons, gal_lats, gal_zooms):
        px, py = ax.wcs.wcs_world2pix([[lon, lat]], 0)[0]
        sph.imscatter([px], [py], galaxy_icon, ax=ax, zoom=float(z), autoscale=False)
    ax.set_title(f"over a {blabel}", fontsize=10, color=tcol)
fig.suptitle("The same M74 icon as a mock galaxy field, on either background",
             fontsize=12)
fig.tight_layout()

# %% [markdown]
# > **Making it your own.** Swap `M74` and the `hips_id`s for any object and survey
# > (`download_hips` takes any HiPS; browse them at
# > <https://aladin.cds.unistra.fr/hips/list>). A wider `GAL_FOV` frames a bigger
# > galaxy; SDSS covers the northern sky, while all-sky surveys like AllWISE or DSS2
# > reach anywhere. Tune the cutout to taste: the `alpha` percentiles set where the
# > fade starts (lower `lo` keeps more faint outer disk), and `gamma` sets its shape
# > (lower is more opaque in the body — drop toward `0.35` for a bolder icon, raise
# > toward `1` for a softer, more diffuse one).

# %% [markdown]
# ### Going further with multicolorfits
#
# Everything above used only numpy and matplotlib, and it runs into the ceiling you'd
# expect: stacking arrays into R, G, B gives you **exactly three slots, locked to those
# three specific primaries**. Real multiwavelength compositing wants more — an arbitrary hue
# per band, more (or fewer) than three bands, perceptual color spaces where equal
# steps *look* equal, and a matte that follows structure instead of noise.
# [`multicolorfits`](https://github.com/pjcigan/multicolorfits) (`mcf`) is the
# companion package for exactly that. Its pipeline is three calls:
#
# | step | call | what it does |
# |---|---|---|
# | scale | `to_grey_rgb(data, rescalefn=…, min_max=…)` | stretch one band to grayscale RGB |
# | tint | `colorize_image(gray, '#RRGGBB')` | give that band *any* hue you like |
# | stack | `combine_multicolor([layers], gamma=…)` | add the tinted layers into one image |
#
# Because the hue is per layer, the band count isn't capped at three. Below we keep
# SDSS `gri` on near-true-color channels *and* fold in the AllWISE **W3** band as a
# fourth, magenta layer, so the dusty star-forming arms light up in a way no
# three-channel RGB stack can reproduce. (Our HiPS cutouts all share one center, field,
# and pixel grid, so the SDSS and WISE frames are already co-registered.)
#
# One warning worth inheriting from the `mcf` docs: **keep the channel hues
# saturated.** If every band gets a warm, near-white tint the composite itself comes
# out pale, and then no amount of alpha tuning will make it read against a white page.

# %%
# fig-slug: galaxy-mcf-composite
import multicolorfits as mcf  # noqa: E402


def gray(data, min_max=(30, 99.5)):
    """One band, arcsinh-stretched between percentile limits (mcf's scaling step).

    (``to_grey_rgb`` keeps multicolorfits' own spelling — it's their API name.)
    """
    return mcf.to_grey_rgb(data, rescalefn="asinh", scaletype="perc",
                           min_max=list(min_max))


# One hue per band: near-true-color for gri, plus magenta for the WISE dust band.
layers = [(i_, "#FF3B21"), (r, "#57DD52"), (g, "#3C7BE8"), (w3, "#FF1F6B")]
rgb_mcf = mcf.combine_multicolor(
    [mcf.colorize_image(gray(data), hue, colorintype="hex") for data, hue in layers],
    gamma=2.2)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
axes[0].imshow(rgb_sdss, origin="lower")
axes[0].set_title("numpy: 3 bands, locked to R/G/B", fontsize=10)
axes[1].imshow(rgb_mcf, origin="lower")
axes[1].set_title("mcf: 4 bands, a hue each (W3 dust in magenta)", fontsize=10)
for ax in axes:
    ax.set(xticks=[], yticks=[])
fig.suptitle("Same data, more color control", fontsize=12)
fig.tight_layout()

# %% [markdown]
# That magenta 4-band is a *capability* demo — mcf reaching past three primaries. For
# the icon we actually keep, though, a clean **near-true-color** look reads as "galaxy"
# at a glance, so we drop back to plain SDSS `gri` mapped straight to red / green /
# blue. That's the recipe from the `mcf` docs' transparent-cutout guide, and it's the
# base the rest of this section builds on:

# %%
# SDSS gri straight to primaries — the natural-color base for the stamp.
gri_primaries = [(i_, "#FF0000"), (r, "#00FF00"), (g, "#0000FF")]
rgb_gri = mcf.combine_multicolor(
    [mcf.colorize_image(gray(data), hue, colorintype="hex") for data, hue in gri_primaries],
    gamma=2.2)

# %% [markdown]
# **And the cutout.** `make_transparent_cutout` is the packaged version of our alpha
# recipe, with the two knobs that decide whether a stamp holds up:
#
# - **`alpha_smooth`** blurs the *matte only* — never the image data — by a few pixels,
#   so whole spiral arms cross the opacity threshold together instead of noise punching
#   pinholes through them. This is what keeps a stamp from looking speckled.
# - **`alpha_gamma`** shapes the fade between the `alpha_lo` (fully clear) and
#   `alpha_hi` (fully opaque) percentiles — the same exponent we tuned by hand, and
#   `mcf` defaults it to `0.5` for the same reason.
#
# The `mcf` rules of thumb are worth memorizing: **≈0.3–0.5 for light pages and print**
# (keeps the body solid), **1.0 on dark decks** (a linear fade looks airier), and
# **>1 for small stamps on busy backgrounds** (trims the faint skirt so only the crisp
# skeleton overlays). If a faint haze box outlines the frame, raise `alpha_lo`. A light
# background is a useful visual test, so let's sweep through a few values:

# %%
# fig-slug: galaxy-mcf-cutout
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for col, (gam, note) in enumerate([(0.4, "bold body — light pages"),
                                   (1.0, "linear — airy on dark"),
                                   (1.4, "tight — busy backgrounds")]):
    icon = mcf.make_transparent_cutout(rgb_gri, crop="auto", pad=0.06,
                                       alpha_smooth=3, alpha_lo=60, alpha_hi=90,
                                       alpha_gamma=gam)
    for row, bg in enumerate(["#f2f1ec", "#0b1020"]):
        ax = axes[row, col]
        ax.set_facecolor(bg)
        ax.imshow(icon, origin="lower", extent=[0, 1, 0, 1])
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
        if row == 0:
            # Column header sits in the figure margin, not on the light panel, so
            # let it take the theme's default title color — readable in both docs
            # renders. (A fixed dark gray here vanished on the dark page.)
            ax.set_title(f"alpha_gamma={gam}\n{note}", fontsize=9)
fig.suptitle("alpha_smooth keeps the arms coherent; alpha_gamma shapes the fade",
             fontsize=12)
fig.tight_layout()

# The stamp we'd actually keep: bold body, so it works on either page. Write it
# right-side-up for image viewers with mcf.save_transparent_cutout(galaxy_mcf, ...).
galaxy_mcf = mcf.make_transparent_cutout(rgb_gri, crop="auto", pad=0.06,
                                         alpha_smooth=3, alpha_lo=60, alpha_hi=90,
                                         alpha_gamma=0.4)

# %% [markdown]
# > **More where that came from.** `mcf.recipes()` prints a menu of copy-paste recipes
# > (`mcf.recipes('cutout')` is the one behind this stamp), and
# > `mcf.preview_cutout_on_backgrounds(rgba)` renders a checker/dark/light check in a
# > single call. The combine step takes a background too —
# > `combine_multicolor_alpha(layers, background='white')` composites onto a page color
# > instead of black (with `mode='ryb'`/`'cmyk'` for paint- and ink-like subtractive
# > mixing), and `background=None` returns transparent RGBA directly; for any composite
# > that *isn't* on black, matte it with `alpha_source='dist'` and `sky_color=` so the
# > fade measures distance from that background. Also in the box: perceptual (Lab)
# > blending, colorblind-safe palette suggestions, `save_transparent_cutout` to write
# > the PNG right-side-up, and `deblend_background` to *recover* alpha from an image
# > already flattened onto a solid color. The [FITS Images](fits_images.ipynb) tutorial
# > uses `mcf` for full-frame science composites; here we've borrowed it to cut one
# > good-looking icon.

# %% [markdown]
# ## 8. Where to go next
#
# Markers put instruments and pictures on your maps; these tutorials take the
# threads further:
#
# | if you want to… | go to |
# |---|---|
# | draw globes, planet surfaces, and the full `plot_baselines` toolkit | [Globe and Planet Plotting](globe_plots.ipynb) |
# | add rulers, reticles, scale bars, and other measurement furniture | [Annotations and Overlays](annotations.ipynb) |
# | plot and encode whole source catalogs | [Catalogs — Querying and Plotting](catalogs.ipynb) |
# | draw proper-motion and velocity arrows, and co-visibility | [Vector Fields and Sky Kinematics](vector_fields.ipynb) |
# | spin the globe and track a source frame by frame | [Animations](animations.ipynb) |
#
# The `plot_baselines` network and the elevation-tracking dishes here are the still
# frames of a movie — advancing the time and re-rendering is all the
# [Animations](animations.ipynb) tutorial adds on top.
