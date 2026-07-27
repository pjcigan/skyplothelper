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
# # Decorating Frames
#
# A frame is only half the picture — the other half is how it *reads*. The same
# sky field can look like a rough sketch or a publication figure depending on its
# ticks, gridlines, labels, and overall style. This tutorial is about that second
# half: taking full control of how a frame presents itself, and answering the two
# questions every plot raises — **"how do I show my axes/grid this way?"** and
# **"how do I adjust it?"**
#
# We'll start by naming every controllable part of a frame, then work through tick
# label formats, gridlines, tick styles for zoomed fields, edge-vs-in-frame ticks,
# label sizing, and finally global themes and palettes — closing with a before/after on a
# real VLBI image that stacks the whole toolkit.
#
# > **Scope:** this tutorial covers a frame's **own** ticks, grid, labels, and
# > styling. Drawing a *second* coordinate system's grid over a frame (for example,
# > a galactic graticule on an equatorial map) is its own topic — see
# > [Overlay Coordinate Grids](overlay_grids.ipynb).
#
# ## Contents
#
# 1. [Anatomy of a frame](#1.-Anatomy-of-a-frame)
# 2. [Tick label formats](#2.-Tick-label-formats)
# 3. [Gridlines](#3.-Gridlines)
# 4. [Offset and relative coordinates](#4.-Offset-and-relative-coordinates)
# 5. [Edge and in-frame ticks](#5.-Edge-and-in-frame-ticks)
# 6. [Sizing tick labels](#6.-Sizing-tick-labels)
# 7. [Themes and palettes](#7.-Themes-and-palettes)
# 8. [Putting it together](#8.-Putting-it-together)
# 9. [Log and symmetric-log axes](#9.-Bonus-topic:-Log-and-symmetric-log-axes)
# 10. [How do I…?](#10.-How-do-I…?)
# 11. [Where to go next](#11.-Where-to-go-next)

# %%
import warnings

import astropy.units as u
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

import skyplothelper as sph

# Give the whole notebook a tightened skyplothelper look. base='structural' adjusts only
# frame and tick *geometry* — inward ticks, minor ticks, lighter spines, a dotted grid —
# and touches no colors or fonts, so it layers cleanly under whatever theme is active
# (including the light/dark toggle on these pages). It's the "improve the defaults, keep
# my encoding" preset; Section 7 is the whole styling story. The few figures below that
# deliberately show matplotlib's *bare* defaults opt out with `plt.style.context('default')`.
sph.set_style(base="structural")

# %% [markdown]
# Throughout, we reuse a few recurring **data anchors** so the focus stays on the
# decoration, not the data:
#
# - a **tangent-plane field** around the Crab Nebula (M1) with a small scatter of
#   sources — the everyday "here is my field, here are my objects" picture;
# - an **all-sky map** and a **tilted globe**, for the cases where decoration gets
#   interesting (curved gridlines, in-frame labels, full-sky themes).
#
# The source scatter is synthetic but realistic in shape — substitute your own
# table of `(ra, dec)` and everything below applies unchanged.

# %%
# Recurring anchor A — a tangent-plane field on the Crab Nebula (M1) + sources.
CRAB = (83.6287, 22.0147)                      # RA, Dec of M1 in degrees
rng = np.random.default_rng(7)
n_src = 18
src_ra = CRAB[0] + rng.normal(0, 0.7, n_src) / np.cos(np.radians(CRAB[1]))
src_dec = CRAB[1] + rng.normal(0, 0.7, n_src)
src_flux = 10 ** rng.uniform(0, 2, n_src)      # a skewed "brightness" for later encoding

# In-theme highlight colors chosen to stay legible in BOTH the light and dark
# tutorial themes. The dark theme's frame/ticks are bluish, so we use warm
# accents — a gold/copper highlight and a brick red — rather than blue, which
# would blend into the frame in dark mode.
HILITE = "#B8860B"      # gold/copper — recolored labels & custom annotations
ACCENT_RED = "#C0392B"  # brick red — the anatomy callout arrows
DATA_GRAY = "0.5"       # neutral mid-gray for monochrome source markers — a single
#                         tone that stays visible on both the light and dark page


def crab_field(subplot=111, fov_deg=4.0, fig=None, **kw):
    """Build the recurring TAN field on the Crab, our stand-in for 'your field'."""
    return sph.make_wcs_frame(subplot, "TAN", center=CRAB, fov_deg=fov_deg,
                              fig=fig, **kw)


# %% [markdown]
# ## 1. Anatomy of a frame
#
# Before adjusting anything, it helps to have names for the parts. Every
# skyplothelper frame is a real matplotlib `WCSAxes`, and each of the pieces below
# is independently controllable — the rest of this tutorial is essentially a tour
# of these labels:

# %%
fig = plt.figure(figsize=(8.5, 7.5))
ax = crab_field(111, fov_deg=4.0, fig=fig, grid=True, gridcolor="0.7", gridalpha=0.9)
# Pin the majors to a round 1° so the diagram reads as an evenly-labeled grid; the
# minor ticks then subdivide each interval, and minor_size < major_size makes them
# read as distinct shorter marks:
ax.coords[0].set_ticks(spacing=1.0 * u.deg)
ax.coords[1].set_ticks(spacing=1.0 * u.deg)
sph.style_wcs_axes(ax, minor_ticks=True, minor_frequency=4,
                   major_size=9, minor_size=4)
ax.coords[0].set_axislabel("Right Ascension (J2000)", fontsize=10)
ax.coords[1].set_axislabel("Declination (J2000)", fontsize=10)
ax.scatter(src_ra, src_dec, transform=ax.get_transform("world"),
           s=24, c=DATA_GRAY, zorder=5)
fig.subplots_adjust(left=0.17, right=0.83, top=0.90, bottom=0.17)

# Callouts: name each controllable part with an arrow into the figure. The
# tick/edge arrows are anchored in *sky* coordinates (xycoords=world) so their
# tips stay locked onto the actual ticks/spine regardless of figure layout;
# the margin labels (grid, tick labels, axis label) use axes fractions.
cal = dict(textcoords="axes fraction", fontsize=10.5, fontweight="bold",
           color=ACCENT_RED, arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.4))
xw = ax.get_transform("world")
ax.annotate("major tick", xy=(85.0, 24.0), xycoords=xw,
            xytext=(0.10, 1.10), ha="center", **cal)
ax.annotate("minor tick", xy=(84.25, 24.0), xycoords=xw,
            xytext=(0.52, 1.09), ha="center", **cal)
ax.annotate("gridline", xy=(0.178, 0.70), xycoords="axes fraction",
            xytext=(0.085, 0.88), ha="center", **cal)
ax.annotate("spine (frame edge)", xy=(81.5, 21.375), xycoords=xw,
            xytext=(1.16, 0.43), ha="center", **cal)
ax.annotate("tick labels", xy=(0.18, -0.060), xycoords="axes fraction",
            xytext=(0.06, -0.165), ha="center", **cal)
ax.annotate("axis label", xy=(0.50, -0.11), xycoords="axes fraction",
            xytext=(0.70, -0.175), ha="center", **cal)
plt.show()

# %% [markdown]
# That's the whole vocabulary: **major** and **minor ticks**, the **gridlines**
# they extend into, the **tick labels** that annotate them, the **axis labels**
# naming each coordinate, and the **spine** (the frame's edge). Sky frames add one
# wrinkle a normal plot doesn't have — on curved all-sky and globe frames the spine
# is a curve and the gridlines bend, which is exactly where the in-frame tools in
# [Section 5](#5.-Edge-and-in-frame-ticks) earn their keep.
#
# ### From bare to decorated
#
# Here is the payoff of the whole tutorial in one comparison: the *same* field and
# sources, drawn with the defaults (left) and fully decorated (right). Everything
# that takes the left panel to the right is covered in the sections that follow.

# %%
fig = plt.figure(figsize=(12, 5.6))

# Left — the bare defaults. make_wcs_frame applies a little polish out of the
# box (a grid, sexagesimal label formatting, auto-fit label sizing, tidied edge
# ticks); we switch all of it off AND build the panel inside a clean matplotlib
# default style so it stays a faithful "before" in both the light and dark docs
# builds — the dark build themes the page, which would otherwise recolor even this
# reference panel's spine, ticks, labels, and data. The title is set outside that
# context so it adapts to whichever page it lands on.
with plt.style.context("default"):
    ax0 = crab_field(121, fov_deg=4.0, fig=fig, grid=False,
                     apply_format_defaults=False, tick_style="native",
                     edge_ticks="all", auto_fontsize=False)
    ax0.scatter(src_ra, src_dec, transform=ax0.get_transform("world"),
                s=24, zorder=5)        # default matplotlib marker color
ax0.set_title("Default", fontsize=11)

# Right — decorated. We build it inside a light THEME (a preview of Section 7), so
# the whole panel takes on the publication look the moment the frame is created.
with sph.style_context(theme="publication"):
    ax1 = crab_field(122, fov_deg=4.0, fig=fig, grid=True,
                     gridcolor="0.8", gridalpha=0.9)
    ax1.scatter(src_ra, src_dec, transform=ax1.get_transform("world"),
                s=src_flux * 1.4, c=src_flux, cmap="cividis", zorder=5,
                edgecolors="0.25", linewidths=0.4)
    # style_annotation lays down a cohesive finder-chart palette and returns its
    # colors; we apply it FIRST, then the explicit grid/label/tick styling below
    # (which governs the final geometry — direction, minor ticks, width):
    pal = sph.style_annotation(ax1, "publication", grid=False)
    sph.style_grid(ax1, color="0.8", alpha=0.9, lw=0.8, ls=":")
    # 'compact' auto-drops the always-00 seconds (5h42m / 23°30′, not 5h42m00s):
    sph.format_ticklabels(ax1, style="compact", color="0.2")
    # Pin the majors to a round 1° for a tidy labeled grid; the minor ticks
    # subdivide each interval:
    ax1.coords[0].set_ticks(spacing=1.0 * u.deg)
    ax1.coords[1].set_ticks(spacing=1.0 * u.deg)
    # Longer, inward-facing, slightly thicker ticks; minors stay short.
    sph.style_wcs_axes(ax1, direction="in", major_size=7, minor_size=3.5,
                       width=1.3, tick_color="0.3", axislabel_color="0.2",
                       minor_ticks=True, minor_frequency=4)
    # A finder-chart label on the brightest source, in style_annotation's colors:
    bi = int(np.argmax(src_flux))
    ax1.annotate("target", xy=(src_ra[bi], src_dec[bi]),
                 xycoords=ax1.get_transform("world"),
                 xytext=(16, 12), textcoords="offset points",
                 color=pal["label"], fontsize=9, fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=2.0, foreground=pal["ax_bg"])],
                 arrowprops=dict(arrowstyle="->", color=pal["accent"], lw=1.3))
    # An accent ring on the target, a highlighted reference parallel through it, and a
    # compass — the finder-chart signature, every color from the same returned palette:
    ax1.scatter([src_ra[bi]], [src_dec[bi]], transform=ax1.get_transform("world"),
                s=210, facecolors="none", edgecolors=pal["accent"], linewidths=1.4, zorder=6)
    sph.highlight_gridline(ax1, 22.0, coord="lat", color=pal["accent2"], lw=1.3, ls="--")
    sph.add_compass(ax1, color=pal["compass"], stroke_color=pal["ax_bg"], length=0.12)
    # A caption in the palette's lower text tier, and a scale for the flux encoding:
    ax1.text(0.97, 0.04, "4° field · ICRS", transform=ax1.transAxes, ha="right",
             color=pal["text2"], fontsize=8, zorder=7)
    ax1.set_title("Decorated", fontsize=11)

fig.subplots_adjust(wspace=0.32)
plt.show()

# %% [markdown]
# Same data, very different read. The rest of the tutorial unpacks the right panel
# piece by piece: the tick-label format (Section 2), the styled grid (Section 3),
# the tick styling and sizing (Sections 5 and 6), and the light theme plus the
# finder-chart source label that set the overall tone (Section 7).

# %% [markdown]
# ## 2. Tick label formats
#
# The most common adjustment is how the coordinate *values* are written.
# `format_ticklabels()` is the single call for it: it auto-detects the frame —
# equatorial axes get sexagesimal hours/degrees, galactic and ecliptic axes get
# decimal degrees — applies a sensible default, then lets you override either with
# a named `style=` preset or with the individual separator/size/color knobs.
#
# ### The built-in styles at a glance
#
# Here is the full preset vocabulary, grouped by what it's for. The rest of this
# section works through the everyday ones; the offset and VLBI families get their
# own worked treatment in Section 4 (tick styles for zoomed fields).
#
# | `style=` | Example (RA / Dec) | Best for |
# |---|---|---|
# | `'publication'` (`'pub'`) | `5ʰ42ᵐ00ˢ` / `+22°00′51″` | **the default** — mathtext superscripts |
# | `'letter'` | `5h42m00s` / `22d00m51s` | ASCII-safe plain text |
# | `'casa'` | `5:42:00` / `22:00:51` | CASA / CARTA colon convention |
# | `'latex'` | `5ʰ42ᵐ00ˢ` (LaTeX) | `usetex` documents |
# | `'compact'` | `5ʰ42ᵐ` / `+22°00′` | publication with the seconds dropped |
# | `'minimal'` | `5ʰ` / `+22°` | hours-only / degrees-only — **all-sky** |
# | `'decimal'` (`'deg'`) | `85.5°` / `22.0°` | decimal degrees, both axes |
# | `'decimal_plain'` | `85.5` / `22.0` | decimal degrees, no ° symbol |
# | `'allsky_hours'` (`'allsky_h'`) | hours lon / degrees lat | equatorial all-sky maps |
# | `'allsky_deg'` (`'allsky_d'`) | decimal degrees both | galactic / ecliptic all-sky |
# | `'offset'`, `'offset_arcsec/arcmin/mas/uas'` | `Δα`, `Δδ` from center | relative offsets → Section 4 |
# | `'vlbi'`, `'anchored_offset'`(`_mas`/`_uas`/`_compact`) | sub-arcsec / anchor + offset | → Section 4 |
#
# ### Switch the whole convention with a preset
#
# To change the entire labeling convention at once, pass a `style=`. The same
# field, six ways:

# %%
styles = ["publication", "letter", "casa", "compact", "decimal", "latex"]
fig = plt.figure(figsize=(12.5, 7.4))
for i, sty in enumerate(styles, start=1):
    ax = crab_field(230 + i, fov_deg=4.0, fig=fig, grid=True,
                    gridcolor="0.85", gridalpha=0.8)
    sph.format_ticklabels(ax, style=sty, fontsize=8)
    for _c in (ax.coords[0], ax.coords[1]):   # drop axis labels; focus on ticks
        _c.set_auto_axislabel(False)
        _c.set_axislabel("")
    ax.set_title(f"style='{sty}'", fontsize=10)
fig.subplots_adjust(wspace=0.45, hspace=0.28)
plt.show()

# %% [markdown]
# Reading across: `'publication'` (the default — superscript ʰᵐˢ / °′″),
# `'letter'` (plain `h m s` / `d m s`), `'casa'` (colon-separated, the CASA/CARTA
# convention), `'compact'` (publication with the seconds dropped — hours and minutes
# only), `'decimal'` (decimal degrees on both axes — note RA becomes `85°`, not
# `5ʰ40ᵐ`), and `'latex'` (LaTeX superscripts for `usetex` documents).

# %% [markdown]
# ### Compacting labels by dropping redundant fields
#
# When the ticks all share an hour and land on whole minutes, the full `hʰmᵐsˢ` is
# repetitive. The `'compact'` preset trims it to hours-and-minutes — the same field,
# full vs compact:

# %%
fig = plt.figure(figsize=(11, 4.6))
for i, sty in enumerate(["publication", "compact"], start=1):
    ax = crab_field(120 + i, fov_deg=4.0, fig=fig, grid=True,
                    gridcolor="0.85", gridalpha=0.8)
    sph.format_ticklabels(ax, style=sty, fontsize=9)
    for _c in (ax.coords[0], ax.coords[1]):
        _c.set_auto_axislabel(False)
        _c.set_axislabel("")
    ax.set_title(f"style='{sty}'", fontsize=10)
fig.subplots_adjust(wspace=0.35)
plt.show()

# %% [markdown]
# > **Note:** there is also a general `simplify=` flag — it works with *any* style,
# > not just `'compact'` — that goes a step further, suppressing any *unchanged*
# > leading or trailing field across the whole tick set (a shared hour, a trailing
# > `00ˢ`). It needs **astropy ≥ 7.0**; on earlier astropy it is silently ignored,
# > which is why the panel above relies on the `'compact'` preset's format to drop
# > the seconds. With astropy ≥ 7 you can pass `simplify=True` to thin a crowded
# > axis even further.
#
# ### When the presets are meant for all-sky
#
# The `'minimal'` and `'allsky_hours'`/`'allsky_deg'` presets are built for
# whole-sky maps. On a field this small they have nothing to label and come up
# nearly empty — but on an all-sky frame they are exactly right, where full
# sexagesimal would be unreadable:

# %%
fig = plt.figure(figsize=(12, 4.4))
ax = sph.make_wcs_frame(121, "AIT", center=0, fig=fig)
sph.format_ticklabels(ax, style="publication")
ax.set_title("style='publication' (cluttered all-sky)", fontsize=10)
ax = sph.make_wcs_frame(122, "AIT", center=0, fig=fig)
sph.format_ticklabels(ax, style="allsky_hours")
ax.set_title("style='allsky_hours'", fontsize=10)
plt.show()

# %% [markdown]
# ### Choosing separators
#
# Underneath the presets, the glyphs between fields come from a small registry,
# `SEPARATORS`, selected per axis with `lon_sep=` / `lat_sep=`. *(Note the names:
# `lon_sep`/`lat_sep`, not `ra_sep`/`dec_sep` — though those aliases still work.)*
# The eleven keys:
#
# | Key | Renders (e.g. `5ʰ40ᵐ12ˢ` / `22°00′51″`) | Best for |
# |---|---|---|
# | `hms_full` | `5ʰ40ᵐ12ˢ` — mathtext superscripts | **RA default**; renders in any font |
# | `hms_unicode` | `5ʰ40ᵐ12ˢ` — literal unicode | compact, but may box-out ("tofu") in some fonts |
# | `hms_latex` | `5ʰ40ᵐ12ˢ` — LaTeX superscripts | `usetex` / LaTeX documents |
# | `hms_letter` | `5h40m12s` | ASCII-safe plain text |
# | `hms_colon` | `5:40:12` | CASA / CARTA convention |
# | `hms_space` | `5 40 12` | terse, table-like |
# | `dms_full` | `22°00′51″` | **Dec default** |
# | `dms_latex` | `22°00′51″` — LaTeX | LaTeX documents |
# | `dms_letter` | `22d00m51s` | ASCII-safe |
# | `dms_colon` | `22:00:51` | colon convention |
# | `deg_symbol` | `85.0°` | the ° suffix on decimal-degree axes |
#
# Mixing separators is just two keywords:

# %%
seps = [("hms_unicode", "dms_full"), ("hms_letter", "dms_letter"),
        ("hms_colon", "dms_colon")]
fig = plt.figure(figsize=(12.5, 3.9))
for i, (ls, ds) in enumerate(seps, start=1):
    ax = crab_field(130 + i, fov_deg=4.0, fig=fig, grid=True,
                    gridcolor="0.85", gridalpha=0.8)
    sph.format_ticklabels(ax, style=None, lon_sep=ls, lat_sep=ds, fontsize=9)
    for _c in (ax.coords[0], ax.coords[1]):   # drop axis labels; focus on ticks
        _c.set_auto_axislabel(False)
        _c.set_axislabel("")
    ax.set_title(f"lon_sep='{ls}', lat_sep='{ds}'", fontsize=9)
fig.subplots_adjust(wspace=0.40)
plt.show()

# %% [markdown]
# ### Size, color, and a legibility stroke
#
# `format_ticklabels()` also takes `fontsize=`, `color=`, and a **stroke**
# (`stroke_lw=` / `stroke_color=`) — a contrasting outline around each glyph. Size
# and color apply to any frame, and `which='lon'` / `'lat'` restyles just one axis
# (everything else here uses the default `which='both'`):

# %%
fig = plt.figure(figsize=(6.0, 5.8))
ax = crab_field(111, fov_deg=4.0, fig=fig, grid=True, gridcolor="0.85", gridalpha=0.8)
sph.format_ticklabels(ax, style="compact", fontsize=12, color=HILITE)
ax.set_title("larger labels, recolored", fontsize=10)
plt.show()

# %% [markdown]
# The **stroke** is the trick for keeping labels readable when they fall over
# bright, variable imagery — a thin contrasting outline (`stroke_color="white"`,
# `stroke_lw=2`) so the text reads against anything behind it. On a flat field with
# the labels out in the margin it does little, so we hold the real demonstration for
# Section 3, where the same stroke rescues **gridlines** drawn over a sky image.
#
# ### Full manual control
#
# When a preset isn't enough, three routes give you complete control over the labels.
#
# **1. A custom format pattern.** `format_ticklabels()` accepts a raw astropy format
# string via `lon_fmt`/`lat_fmt`, so you dial the exact fields and precision yourself
# (`'hh:mm'`, `'hh:mm:ss.s'`, `'d.dd'`, …) — here sub-second RA and two-decimal Dec:

# %%
fig = plt.figure(figsize=(6.0, 5.8))
ax = crab_field(111, fov_deg=4.0, fig=fig, grid=True, gridcolor="0.85", gridalpha=0.8)
sph.format_ticklabels(ax, style=None, lon_fmt="hh:mm:ss.s", lat_fmt="d.dd")
ax.set_title("lon_fmt='hh:mm:ss.s', lat_fmt='d.dd'", fontsize=10)
plt.show()

# %% [markdown]
# **2. Build label strings with `RAlabelformatter` / `RAlabellist`.** These turn an
# RA in *degrees* into a label string in a chosen style — `'deg'`, `'h'`, or `'^h'`
# for a mathtext superscript — and `RAlabellist` runs the formatter across a set of
# tick offsets around a center, handy when assembling a label set by hand:

# %%
print(sph.RAlabelformatter(83.6287, style="^h"))   # 83.63 deg -> '6ʰ'
print(sph.RAlabellist(180.0, style="h"))            # hour labels around RA = 180

# %% [markdown]
# **3. Put *any* text at *any* position.** For labels that aren't coordinates at all,
# set the tick positions yourself, hide the automatic labels, and drop in your own
# text. Here we give three longitudes literal names:

# %%
fig = plt.figure(figsize=(6.0, 5.8))
ax = crab_field(111, fov_deg=4.0, fig=fig, grid=True, gridcolor="0.85", gridalpha=0.8)
named = {84.6: "Fred", 83.6: "Jim", 82.6: "Erika"}
ax.coords[0].set_ticks(list(named) * u.deg)     # ticks exactly where we want them
ax.coords[0].set_ticklabel_visible(False)        # hide the automatic RA labels
ax.coords[0].set_auto_axislabel(False)
tr = ax.get_transform("world")
for ra, name in named.items():
    ax.annotate(name, xy=(ra, CRAB[1] - 2.0), xycoords=tr,
                xytext=(0, -6), textcoords="offset points", ha="center", va="top",
                fontsize=10, fontweight="bold", color=HILITE, clip_on=False)
ax.set_title("custom tick labels", fontsize=10)
plt.show()

# %% [markdown]
# The first two routes cover "format the numbers differently"; the third covers
# "these aren't numbers at all." Between the presets, the separator registry, and
# these three hatches, there's no tick label you can't produce.

# %% [markdown]
# ## 3. Gridlines
#
# The graticule — the mesh of meridians and parallels — is what lets a reader place
# a feature on the sky. There are three levels of control: set the grid at **build
# time**, restyle the whole thing afterward, or **highlight** specific lines and
# families of lines.
#
# At build time, `make_wcs_frame()` and the figure builders take `grid=`, `gridcolor=`,
# `gridalpha=`, `gridlw=`, and `gridls=` — the last two settable on the globe, cone, bowtie
# and cartopy builders too (an existing grid switches off after the fact with astropy's
# `ax.coords.grid(draw_grid=False)`). To restyle an existing frame's grid, `style_grid()`
# takes `color`/`alpha`/`lw`/`ls` plus a `stroke_lw`/`stroke_color` outline. The same
# all-sky frame, default vs restyled:

# %%
fig = plt.figure(figsize=(12, 4.6))
ax = sph.make_wcs_frame(121, "AIT", center=0, frame="ICRS", fig=fig)
sph.format_ticklabels(ax, style="allsky_hours")
ax.set_title("default grid", fontsize=10)
ax = sph.make_wcs_frame(122, "AIT", center=0, frame="ICRS", fig=fig)
sph.format_ticklabels(ax, style="allsky_hours")
sph.style_grid(ax, color="#3f6fa3", alpha=0.9, lw=1.0, ls=":")
ax.set_title("style_grid(color, alpha, lw, ls)", fontsize=10)
plt.show()

# %% [markdown]
# ### Gridlines over imagery: the stroke
#
# Over a busy image the grid competes with the data, and on a **full-tonal-range**
# background *no single line color stays legible everywhere* — a light grid vanishes
# over the bright regions, a dark one over the shadows. A thin **stroke** (a
# contrasting outline) gives each line both a light and a dark edge, so it reads
# against the whole range. The same outline distinguishes **tick and axis labels**, which
# face the identical problem. Here a neutral gray grid, tick marks, *and* labels over
# a synthetic grayscale field: the least obtrusive choice, but lost in the mid-tones —
# the left panel *has* all of them; they have simply dissolved into the underlying map colors — until the same stroke
# on the right side helps recover every one.

# %%
# A smooth synthetic all-sky field, grayscale, to stress the grid's legibility:
nside = 16
field = sph.healpix_smooth(np.random.default_rng(3).normal(size=12 * nside**2),
                           sigma_deg=5.0)
field = (field - field.min()) / (field.max() - field.min())

fig = plt.figure(figsize=(12, 5.4))
for col, stroke in [(1, {}), (2, dict(stroke_lw=2.2, stroke_color="black"))]:
    ax = sph.make_wcs_frame(120 + col, "AIT", center=0, frame="ICRS", fig=fig)
    sph.plot_healpix_map(field, ax=ax, cmap="gray", zorder=-10)
    # Grid, tick marks, tick labels, and the axis label all share the same
    # mid-gray so they read as one family — and every one of them dissolves into
    # the mid-tones on the left. The right panel adds the same stroke across the
    # board: the grid via style_grid, the tick labels via format_ticklabels, the
    # tick *marks* via style_wcs_axes (its stroke_lw/stroke_color complete the
    # trio), and the axis label via matplotlib path effects — and they all return:
    # Force the labels to the mid-gray in BOTH light and dark builds — it's part of the demo,
    # not theme-driven (labelcolor on style_wcs_axes keeps the dark theme from recoloring them):
    sph.format_ticklabels(ax, style="allsky_hours", color="0.5", **stroke)
    sph.style_grid(ax, color="0.5", alpha=0.95, lw=1.0, **stroke)
    sph.style_wcs_axes(ax, tick_color="0.5", labelcolor="0.5", major_size=7, width=1.0, **stroke)
    ax_effects = ([pe.withStroke(linewidth=stroke["stroke_lw"],
                                 foreground=stroke["stroke_color"])]
                  if stroke else [])
    ax.coords[0].set_axislabel("Right Ascension", color="0.5",
                               path_effects=ax_effects)
    ax.set_title("gray grid + ticks + labels, no stroke" if col == 1
                 else "gray grid + ticks + labels + black stroke", fontsize=10)
plt.show()

# %% [markdown]
# That's the demonstration promised back in Section 2: the stroke is what separates
# a grid — or a label, or a tick — you can follow from one that vanishes into the picture.
#
# > **Why three different stroke routes?** Grid lines and tick *labels* are ordinary
# > matplotlib artists, so their outline rides on standard path effects — built into
# > `style_grid()` and `format_ticklabels()`. The **tick marks** are the exception: a
# > WCSAxes draws them with a custom path that genuinely ignores matplotlib's
# > `path_effects` (setting them on the tick artist is silently a no-op), so
# > `style_wcs_axes()` carries its own `stroke_lw`/`stroke_color` to stroke them. The
# > *axis label* is a plain Text artist again, which is why we stroke it with an explicit
# > `path_effects=` above. Same outline, three entry points — one per kind of artist.

# %% [markdown]
# ### The frame itself: `apply_frame_stroke`
#
# The trio above rescues the grid, labels, and ticks — but the **frame outline** and its tick
# *marks* have the same problem, and a real image makes it plain: astronomical fields often run to
# **dark sky at the edges**, where the frame spines are drawn. A dark frame edge on dark sky
# is simply hard to see. `apply_frame_stroke(ax)` is the frame-level companion — one call that
# strokes the spine (as a single continuous path following the frame's shape — a rectangle here,
# an ellipse on all-sky frames — so the corners never cut across themselves) *and* the native
# tick marks (with the same two-pass draw, since they ignore `path_effects`). Tick *labels* are
# left alone. Here is HST's view of the tutorial's own field, the Crab: on the left the frame and
# its inward ticks are hard to see against the sky; the default white stroke on the right makes
# each of them easier to pick out.

# %%
# A real image of the tutorial's field — HST's Crab Nebula (F547M), shown on skyplothelper's
# bundled 'deepsky' colormap. Like most astronomical images, it runs to dark sky at every edge.
crab_img, crab_hdr = fits.getdata("../../examples/data/crab_hst_F547M.fits", header=True)
crab_img = np.nan_to_num(np.squeeze(crab_img).astype(float))
crab_wcs = WCS(crab_hdr).celestial
crab_norm = sph.make_norm(stretch="asinh", clip="percentile", plo=1.0, phi=99.6, data=crab_img)

fig = plt.figure(figsize=(11, 5.2))
for col, do_stroke in [(1, False), (2, True)]:
    ax = fig.add_subplot(120 + col, projection=crab_wcs)
    ax.imshow(crab_img, cmap="sph.deepsky", origin="lower", norm=crab_norm)
    # Pin the frame + inward ticks to a dark ink in BOTH docs builds: the whole demo needs a
    # frame the same tone as the sky behind it, which the dark theme would otherwise lighten
    # into visibility. The labels stay theme-driven — they sit outside the image.
    sph.style_wcs_axes(ax, direction="in", major_size=7, width=1.2,
                       tick_color="0.15", frame_color="0.15")
    for ci, name in ((0, "Right Ascension"), (1, "Declination")):
        ax.coords[ci].set_axislabel(name, fontsize=9)
    if do_stroke:
        sph.apply_frame_stroke(ax)                        # white stroke → frame + ticks return
    ax.set_title("frame lost against the sky" if not do_stroke
                 else "+ apply_frame_stroke(ax)", fontsize=10)
fig.subplots_adjust(wspace=0.35)
plt.show()

# %% [markdown]
# Pass a color and width for the opposite colormap case — `apply_frame_stroke(ax, "black", 3)` gives a dark
# stroke for a light frame over a bright image — and `apply_frame_stroke(ax, None)` removes it. It
# works on a plain matplotlib `Axes` too (there the spines and tick lines take the stroke
# directly), and re-calling simply replaces the previous stroke.

# %% [markdown]
# ### Highlighting individual lines
#
# To draw attention to a *specific* meridian or parallel — a reference axis, a
# survey edge, the galactic plane — `highlight_gridline()` redraws one line at a
# chosen value in its own style, with an optional legend `label`. On a galactic
# all-sky, the galactic-center meridian, the plane, and the anti-center:

# %%
# Highlight colors pulled from the uranometria cycle palette (Section 7) — a brick
# red, a gold, and a sea green (the green: at this map center l=180 runs along the
# boundary, so it needs a hue that stands apart from the limb itself):
urano = sph.CYCLE_PALETTES["uranometria"]["colors"]
fig, ax = sph.allsky_figure(projection="AIT", center=0, frame="galactic",
                            figsize=(8.5, 4.6))
sph.highlight_gridline(ax, 0, coord="lon", color=urano[5], lw=2,
                       label="GC meridian (l=0)")
sph.highlight_gridline(ax, 0, coord="lat", color=urano[2], lw=2,
                       label="galactic plane (b=0)")
sph.highlight_gridline(ax, 180, coord="lon", color=urano[4], lw=1.6, ls="--",
                       label="anti-center (l=180)")
ax.legend(loc="lower right", fontsize=8)
plt.show()

# %% [markdown]
# ### Highlighting families with a colormap
#
# To color a whole *family* of lines by value, `highlight_gridlines()` takes
# `lon_values`/`lat_values` with a `lon_cmap`/`lat_cmap`, turning the graticule
# itself into a readable scale. Match the map to the quantity: longitude *wraps*, so
# it takes the cyclic `twilight` (the two ends meet in the same color, just like the
# meridians); latitude runs symmetrically about the equator, so it takes a diverging
# map — here `sph.diff_tealorange`, one of the bundled `sph.*` colormaps the package
# registers with matplotlib (the FITS Images tutorial tours the full set):

# %%
fig, ax = sph.allsky_figure(projection="AIT", center=0, frame="ICRS",
                            figsize=(8.5, 4.6))
sph.format_ticklabels(ax, style="allsky_hours")
sph.highlight_gridlines(ax, lon_values=np.arange(0, 360, 15), lon_cmap="twilight", lw=1.4)
sph.highlight_gridlines(ax, lat_values=np.arange(-75, 90, 15),
                        lat_cmap="sph.diff_tealorange", lw=1.4)
plt.show()

# %% [markdown]
# > **Tip:** `highlight_gridlines()` also accepts explicit `lon_colors`/`lat_colors`
# > lists when you want specific colors rather than a colormap, and the same
# > `stroke_lw`/`stroke_color` outline as `style_grid()` for legibility over imagery.

# %% [markdown]
# ### Showing the far side of a globe
#
# A globe frame's *own* graticule covers only the **near** hemisphere — behind the limb the WCS
# is undefined, so the native far-side meridians and parallels simply aren't drawn. To show the
# *complete* sphere, drop the frame's grid and let `plot_ortho_grid()` fill in both hemispheres:
# the near side solid and the **far side dashed**, so the hidden hemisphere reads as a ghosted
# overlay. Each hemisphere styles independently. Here a uranometria blue near side and a dashed
# copper far side, viewed slightly off the prime meridian, with `highlight_meridian_tracer()`
# following one meridian as a complete great circle all the way around — front solid, far dashed,
# at matched color and weight, so you can track it past the limb:

# %%
urano = sph.CYCLE_PALETTES["uranometria"]["colors"]
fig = plt.figure(figsize=(5.6, 5.6))
ax = sph.make_globe_frame(111, center_LONdeg=45, center_LATdeg=25,
                          lon_deg_spacing=30, lat_deg_spacing=30, grid=False)
sph.plot_ortho_grid(ax, lon_spacing=15, lat_spacing=15,
                    front_color=urano[0], front_lw=0.9,                            # near side
                    back_color=urano[1], back_lw=0.7, back_ls="--", back_alpha=0.9)  # far side
sph.highlight_meridian_tracer(ax, meridian_lon=0, color=urano[5], lw=1.8)         # tracked meridian
ax.set_title("near side solid · far side dashed", fontsize=11)
plt.show()

# %% [markdown]
# `show_back=False` drops the far side entirely; `back_color`/`back_ls`/`back_alpha` (and the
# `front_*` equivalents) tune each hemisphere, `lon_cmap` colors the meridians by longitude, and
# `prime_meridian_color`/`equator_color` highlight those reference lines directly.
# `highlight_meridian_tracer()` auto-detects the globe's center and defaults to the theme's frame
# color. The same code runs on a plain `plt.subplots()` axes too — set it equal-aspect and build
# the graticule with `plot_ortho_grid(ax, lon_0=45, lat_0=25, R=1.0, ...)`, then call the tracer
# the same way. This is the globe-graticule counterpart to the flat-frame grid tools above; for
# building and decorating globes themselves, see [Globe & Planet Plotting](globe_plots.ipynb).

# %% [markdown]
# ## 4. Offset and relative coordinates
#
# Zoom into a compact source — a VLBI core, a close double — and absolute RA/Dec
# labels stop being useful: every tick reads the same to the arcsecond, differing
# only in the trailing digits of a second. The cure is to label the axes **relative
# to a reference position**. Here the same ~80 mas field on the core of M87's jet,
# three ways:

# %%
M87 = (187.7059, 12.3911)          # RA, Dec of M87 (deg)
fov_mas = 80.0                     # ~80 mas field of view
mas = 1 / 3.6e6                    # one milliarcsecond, in degrees

# A synthetic core + one-sided jet (your real VLBI image would go here):
dxy_mas = np.array([[0, 0], [-12, 5], [-22, 9], [10, -5]])   # (Δα*, Δδ) in mas
jet_ra = M87[0] + dxy_mas[:, 0] * mas / np.cos(np.radians(M87[1]))
jet_dec = M87[1] + dxy_mas[:, 1] * mas
jet_flux = np.array([10, 5, 3, 4.])


def vlbi_frame(sub, fig):
    ax = sph.make_wcs_frame(sub, "TAN", center=M87, fov_deg=fov_mas / 3.6e6,
                            npix=400, fig=fig)
    ax.scatter(jet_ra, jet_dec, transform=ax.get_transform("world"),
               s=jet_flux * 18, c=DATA_GRAY, zorder=5)
    return ax


fig = plt.figure(figsize=(15, 4.8))
# Absolute — pin a sensible few full sexagesimal labels (a tight field can leave
# astropy with just one tick in Dec); they come out nearly identical and unwieldy:
ra_t = M87[0] + np.array([-25, 0, 25]) * mas / np.cos(np.radians(M87[1]))
dec_t = M87[1] + np.array([-25, 0, 25]) * mas
ax = vlbi_frame(131, fig)
ax.coords[0].set_ticks(values=ra_t * u.deg)
ax.coords[1].set_ticks(values=dec_t * u.deg)
sph.format_ticklabels(ax, style="publication", fontsize=8)
ax.set_title("absolute coordinates", fontsize=10)
# Offset — auto-unit signed offsets from the reference (here: mas):
ax = vlbi_frame(132, fig)
sph.apply_offset_ticks(ax, ref_ra_deg=M87[0], ref_dec_deg=M87[1], unit="auto")
ax.set_title("apply_offset_ticks(unit='auto')", fontsize=10)
# Anchored offset — one full-coordinate anchor + round offset ticks from it:
ax = vlbi_frame(133, fig)
sph.apply_anchored_offset(ax, ref_tick="center", unit="mas")
ax.set_title("apply_anchored_offset", fontsize=10)
fig.subplots_adjust(wspace=0.5)
plt.show()

# %% [markdown]
# - **Absolute** (left): the honest coordinates, but every label is `12ʰ30ᵐ49.4…ˢ` /
#   `+12°23′27.…″` — not particularly useful at a glance and somewhat redundant.
# - **Offset** (middle): `apply_offset_ticks()` — an excellent option for zoomed images
#   (VLBI, HST, Chandra) — relabels each axis as a signed offset (`Δα cos δ`, `Δδ`) from a
#   reference (the field center by default, or any `ref_ra_deg`/`ref_dec_deg`). `unit='auto'`
#   picks a sensible unit — **arcmin → arcsec → mas → μas** as you zoom in — or force one
#   with `unit=`. The reference's full coordinates print once in the axis label, so nothing
#   is lost.
# - **Anchored offset** (right): `apply_anchored_offset()` is a hybrid style — one tick
#   carries the full **anchor** coordinate within the label while the rest stay relative offsets, so an
#   absolute reference rides along on the axis itself (a VLBI core, say). `ref_tick='center'`
#   anchors on the field center; `anchor_format='decimal'` prints the anchor in decimal
#   degrees (handy for galactic `l`/`b`), or pass a callable for custom anchor text.
#
# Both are thin, convenient wrappers over astropy's native **offset-WCS axes**: the
# capability is astropy's, but skyplothelper assembles the offset frame, tick locators, and
# labels for you — so a finder chart in relative coordinates is *one call* rather than a
# dozen lines of `ax.coords` plumbing.
#
# ### Choosing the offset unit
#
# You can tune the labels by specifying the **unit**. `unit='auto'` (the default) reads the field of view and
# steps the labels down through arcsec → mas → μas as you zoom in, so the same one-liner
# works at any scale — the same M87 core region, two zooms:

# %%
fig = plt.figure(figsize=(11, 4.8))
ax = sph.make_wcs_frame(121, "TAN", center=M87, fov_deg=20.0 / 3600, npix=300, fig=fig)
ax.scatter([M87[0]], [M87[1]], transform=ax.get_transform("world"), s=60, c=DATA_GRAY, zorder=5)
sph.apply_offset_ticks(ax, unit="auto")
ax.set_title("fov 20″  →  arcsec", fontsize=10)
ax = sph.make_wcs_frame(122, "TAN", center=M87, fov_deg=0.4 / 3600, npix=300, fig=fig)
ax.scatter(jet_ra, jet_dec, transform=ax.get_transform("world"),
           s=jet_flux * 18, c=DATA_GRAY, zorder=5)
sph.apply_offset_ticks(ax, unit="auto")
ax.set_title("fov 0.4″  →  mas", fontsize=10)
fig.subplots_adjust(wspace=0.45)
plt.show()

# %% [markdown]
# Force a fixed unit with `unit='arcsec'`/`'mas'`/`'uas'`/`'arcmin'` when you want one scale
# across a series. `ref_ra_deg`/`ref_dec_deg` move the zero point off-center — reference a
# specific source rather than the frame center — and `stroke_lw`/`stroke_color` keep the
# labels legible where they fall over bright data.
#
# ### Spacing and precision
#
# By default the tick step is sized automatically (a round 1/2/5 increment), but you can set
# it with `spacing=` — a scalar (both axes), a `(lon, lat)` pair for a different step per
# axis, or an astropy `Quantity` in any angular unit (`spacing=0.02*u.arcsec`). The label
# decimals follow the step unless you pin them with `precision=`:

# %%
fig = plt.figure(figsize=(11, 4.8))
ax = vlbi_frame(121, fig)
sph.apply_offset_ticks(ax, unit="mas")
ax.set_title("auto spacing + precision", fontsize=10)
ax = vlbi_frame(122, fig)
sph.apply_offset_ticks(ax, unit="mas", spacing=(20, 10), precision=1)
ax.set_title("spacing=(20, 10), precision=1", fontsize=10)
fig.subplots_adjust(wspace=0.5)
plt.show()

# %% [markdown]
# The left panel takes the automatic step; the right sets a different step per axis (a coarser
# 20 mas in RA, a finer 10 mas in Dec) and forces one decimal. `spacing=` and `precision=`
# carry over to `apply_anchored_offset` as well —
# there the anchor coordinate's own decimals are set separately with `ref_precision=`.
#
# ### Anchored offsets
#
# The anchored style keeps the relative-offset ticks but pins **one** of them to a full
# coordinate, so the absolute reference travels with the axis. That anchor is a long
# sexagesimal label, which crowds a narrow panel — `compact=True` rotates the labels to
# reclaim the width, handy for a row of component close-ups:

# %%
fig = plt.figure(figsize=(11, 4.8))
ax = vlbi_frame(121, fig)
sph.apply_anchored_offset(ax, ref_tick="center", unit="mas")
ax.set_title("default", fontsize=10)
ax = vlbi_frame(122, fig)
sph.apply_anchored_offset(ax, ref_tick="center", unit="mas", compact=True)
ax.set_title("compact=True", fontsize=10)
fig.subplots_adjust(wspace=0.5)
plt.show()

# %% [markdown]
# `ref_tick=` chooses which tick carries the anchor, and the offset step and decimals are
# tunable as well (`spacing=`, `offset_precision=`, `ref_precision=`). (`lon_rotation` /
# `lat_rotation` set the compact angles explicitly if you want them.) A `max_ticks=` cap
# guards against a tiny `spacing=` on a wide field generating thousands of ticks; if it ever
# truncates an axis it says so rather than silently dropping the rest.
#
# > **Tip — too few ticks?** On a tight field astropy can place very few ticks,
# > sometimes only one on an axis (the absolute Dec axis above would, untouched).
# > Set the count with `ax.coords[i].set_ticks(number=N)` (a hint — astropy snaps to
# > round values), or pin them exactly with `set_ticks(spacing=…)` / `set_ticks(
# > values=…)` (used for the absolute panel above). At build time,
# > `make_wcs_frame(lon_spacing=…, lat_spacing=…)` sets the grid/tick spacing
# > directly.
#
# A one-call shortcut, `offset_figure()`, builds a frame with offset ticks already
# applied (see the frames guide).
#
# > **Note:** these tools relabel the **axes** in offset units. To draw a measuring
# > stick *on the data* — a scale bar, or a ruler between two components — that's the
# > `Ruler` in the [Annotations & Overlays](annotations.ipynb) tutorial; the two "offset" ideas
# > are easy to conflate.

# %% [markdown]
# ## 5. Edge and in-frame ticks
#
# Tick labels can sit on the frame **edge** — the default on a rectangular field — or
# be parked **in-frame**, riding along a chosen parallel and meridian. In-frame is the
# natural choice on a **globe**, where the boundary is a horizon rather than an axis,
# and a tidy alternative to margin labels on a flat field.
#
# Which one you get is set at build time by `make_wcs_frame(tick_style=…)`:
#
# | `tick_style=` | Labels go | Use it for |
# |---|---|---|
# | `'auto'` | **the default** — `'in_frame'` on the projections whose astropy edge labels misbehave (circular, sinusoidal, Robinson, Winkel Tripel, …), `'native'` on everything else | just let it choose |
# | `'native'` | astropy's own labels, on the frame edge | rectangular fields; the plain starting point |
# | `'in_frame'` | along the central parallel + meridian, inside the frame | globes, and tidy flat fields |
# | `'boundary'` | on the projection's boundary curve, one per gridline crossing | all-sky ovals |
#
# `'auto'` is doing the work in most of this tutorial's figures. The examples in *this*
# section pass `tick_style='native'` deliberately — starting from plain edge labels, so the
# placement you see is entirely the doing of the call under discussion. That call is
# `add_overlay_ticks()`, which puts the *primary* frame's labels on whichever
# parallel/meridian you pick — on globes and single-field TAN/SIN/ZEA frames alike:

# %%
fig = plt.figure(figsize=(12, 6))
# A tilted globe: RA labels along the lat=10 parallel, Dec down the lon=30 meridian:
ax = sph.make_wcs_frame(121, "SIN", center=(60, 35), fig=fig, tick_style="native")
sph.add_overlay_ticks(ax, lon_at="lat=10", lat_at="lon=30",
                      suppress_default="both", label_kwargs={"fontsize": 9})
ax.set_title("globe — in-frame labels", fontsize=10)
# The same idea on a flat tangent field (the Crab region):
ax = sph.make_wcs_frame(122, "TAN", center=CRAB, fov_deg=4.0, fig=fig,
                        tick_style="native")
sph.add_overlay_ticks(ax, lon_at="lat=22", lat_at="lon=83.6",
                      suppress_default="both", label_kwargs={"fontsize": 9})
ax.set_title("flat field — in-frame labels", fontsize=10)
plt.show()

# %% [markdown]
# On the globe the right-ascension labels curve along the `lat=10` parallel and the
# declination labels run down the `lon=30` meridian — readable and clear of the limb;
# on the tangent field the same labels land on interior gridlines instead of out in
# the margin. `add_overlay_ticks()` picks sensible field-scale values automatically;
# pass `lon_vals`/`lat_vals` to set them and `lon_at`/`lat_at` to choose their lines.
# For longitude labels alone — curved to follow a parallel — there's the dedicated
# `add_curved_lon_ticks()`.

# %% [markdown]
# ### Labels that follow the projection curves
#
# The in-frame labels above use the frame's **tangent** mode — each label aligns to its
# local gridline, which is why the globe's RA labels curved along the parallel. That same
# curve-following drives **exterior** ticks too, and it's what keeps labels readable on
# wide fields and all-sky maps where the gridlines bow. `add_overlay_ticks()` places them
# on any curve you name — `'axis'` (the central parallel / meridian), `'lat=N'` / `'lon=N'`
# (any constant-coordinate line), or `'boundary'` (the projection's limb) — and the labels
# bend to match:

# %%
fig = plt.figure(figsize=(16, 5.2))
# A wide TAN field: even a "flat" tangent frame curves enough at 70° to show it. Put the
# RA labels on the central parallel and the Dec labels on an *off-center* meridian
# (lon=330) so both label sets visibly ride curved lines.
ax = sph.make_wcs_frame(131, "TAN", center=(0, 45), fov_deg=70, fig=fig,
                        tick_style="native")
sph.add_overlay_ticks(ax, lon_vals=np.arange(-30, 31, 15) % 360,
                      lat_vals=np.arange(15, 76, 15),
                      lon_at="axis", lat_at="lon=330", suppress_default="both",
                      label_kwargs={"fontsize": 8})
ax.set_title("TAN field", fontsize=10, pad=14)
# Winkel Tripel: push the labels off the equator / central meridian so they ride the bow.
ax = sph.make_wcs_frame(132, "winkel", center=0, frame="ICRS", fig=fig,
                        tick_style="native")
sph.add_overlay_ticks(ax, lon_at="lat=-30", lat_at="lon=300", suppress_default="both",
                      label_kwargs={"fontsize": 8})
ax.set_title("Winkel Tripel", fontsize=10, pad=14)
# Aitoff: right ascensions on the +30 parallel, declinations around the boundary limb.
ax = sph.make_wcs_frame(133, "AIT", center=0, frame="ICRS", fig=fig,
                        tick_style="native")
sph.add_overlay_ticks(ax, lon_at="lat=30", lat_at="boundary", suppress_default="both",
                      label_kwargs={"fontsize": 8})
ax.set_title("Aitoff", fontsize=10, pad=14)
fig.subplots_adjust(wspace=0.3, top=0.86)
plt.show()

# %% [markdown]
# Reading across: on the **TAN field** the right ascensions fan along the central parallel
# (`lon_at='axis'`) while the declinations ride a meridian pushed out to `lon=330`, where
# the curvature is obvious — on the straight central meridian it wouldn't be. On **Winkel
# Tripel** both sets are off-axis (`lon_at='lat=-30'`, `lat_at='lon=300'`) so they follow
# the bowing parallel and meridian. On **Aitoff** the right ascensions follow the `+30°`
# parallel while the declinations wrap the elliptical **boundary** — the all-sky
# convention. This tangent alignment is the default; the `tick_rotation=` option below
# overrides it.
#
# > **Note — ticks vs. gridlines.** The overlay tick *values* (`lon_vals`/`lat_vals`)
# > are independent of the frame's **gridlines**, which are drawn at the graticule spacing
# > (`lon_spacing`/`lat_spacing`, `'auto'` by default). That's why some ticks above land in
# > *gaps* between gridlines rather than on them. To line them up, set the graticule density
# > at build time with `make_wcs_frame(lon_spacing=…, lat_spacing=…)` (or pass matching
# > `lon_vals`/`lat_vals`).

# %% [markdown]
# ### Boundary latitude labels, the easy way
#
# That last all-sky case — **declination labels riding the elliptical boundary** — is common
# enough to have a dedicated one-liner: `apply_boundary_labels()`. It drops each latitude label
# where its parallel meets the frame edge, and (unlike the generic route) it sidesteps
# astropy's crowding-drop, so the high-declination labels near the poles survive. Its `orient=`
# knob gives three looks, named for how the label sits relative to the edge — `'perpendicular'`
# (the default — each label sticks out *across* the boundary, aligned to its gridline's outward
# extension), `'parallel'` (the label runs *along* the boundary curve), and `'horizontal'`
# (flat, cleanest for print):

# %%
fig = plt.figure(figsize=(13.5, 4.6))
for i, orient in enumerate(["perpendicular", "parallel", "horizontal"], start=1):
    ax = sph.make_wcs_frame(130 + i, "AIT", center=0, frame="ICRS", fig=fig,
                            tick_style="native")
    sph.format_ticklabels(ax, style="allsky_hours")
    # Call AFTER format_ticklabels — it suppresses astropy's auto Dec labels and draws its own:
    sph.apply_boundary_labels(ax, orient=orient, fontsize=8)
    ax.set_title(f"orient={orient!r}", fontsize=10, pad=12)
fig.subplots_adjust(wspace=0.3, top=0.86)
plt.show()

# %% [markdown]
# Pass `coord_index=0` to do the same for *longitude* labels, `side='left'`/`'right'` to label
# a single edge, and `lat_values=` / `fmt_func=` to choose and format the ticks. It's the
# cleanest path whenever you specifically want labels on a curved all-sky limb.

# %% [markdown]
# ### Styling the tick marks
#
# So far we've controlled the *labels*; the **tick marks** themselves have their own
# knobs. A WCSAxes ignores most of matplotlib's `xtick.*`/`ytick.*` rcParams, so
# `style_wcs_axes()` is the bridge for tick **geometry**: it sets tick `direction`
# (`'in'`/`'out'`), `major_size`/`width`/`tick_color`, and toggles **minor ticks**
# (`minor_ticks=True` with `minor_frequency=`). The active theme's *colors* already
# reach the frame when it's built (Section 7), so bare `style_wcs_axes(ax)` is really
# about refining geometry — with the theme's tick color as the default. Here the same
# field with three explicit treatments:

# %%
specs = [
    ("default (outward)", dict(direction="out", major_size=6)),
    ("direction='in', longer, + minor ticks",
     dict(direction="in", major_size=10, width=1.2,
          minor_ticks=True, minor_frequency=4, minor_size=5)),
    ("thick, recolored",
     dict(major_size=9, width=2.0, tick_color=ACCENT_RED,
          minor_ticks=True, minor_frequency=2, minor_size=5)),
]
fig = plt.figure(figsize=(13.5, 4.7))
for i, (title, kw) in enumerate(specs, start=1):
    ax = crab_field(130 + i, fov_deg=4.0, fig=fig, grid=False)
    if kw.get("minor_ticks"):
        # Pin a round 1° major spacing so the minor subdivisions sit at tidy
        # intervals (the default adaptive spacing packs them tighter):
        ax.coords[0].set_ticks(spacing=1.0 * u.deg)
        ax.coords[1].set_ticks(spacing=1.0 * u.deg)
    sph.style_wcs_axes(ax, **kw)
    ax.set_title(title, fontsize=10, pad=12)
fig.subplots_adjust(wspace=0.4)
plt.show()

# %% [markdown]
# `style_wcs_axes()` also carries `labelcolor`/`labelsize` (the tick *labels*),
# `axislabel_color`/`axislabel_size` (the *axis* names), and `frame_color`/
# `frame_linewidth` for the spine — one call for the whole frame's geometry and
# coloring. (The axis-label *text* itself is astropy's
# `ax.coords[i].set_axislabel("...")`, as in the Section 1 anatomy figure.)
#
# > **Tip — WCSAxes tick details.** **Minor length:** WCSAxes minor ticks inherit the
# > *major* length, so without help they read as denser majors — pass `minor_size=`
# > (here `minor_size=5`) for a distinctly shorter minor. **Tidy subdivisions:** minors
# > subdivide whatever major spacing is in effect, so pinning a round major step with
# > `ax.coords[i].set_ticks(spacing=1*u.deg)` (done above) keeps them at even intervals;
# > omit it and they follow the adaptive spacing, which packs them tighter. **Corner
# > strays:** on a *projected* flat field the meridians converge, so near the top corners
# > a meridian can exit through a *side* spine and astropy would draw a stray RA tick
# > there (and a Dec tick on the bottom). `make_wcs_frame`'s default `edge_ticks='auto'`
# > already prevents this — it pins longitude ticks to top/bottom and latitude ticks to
# > left/right, keeping every real tick while dropping the corner strays; pass
# > `edge_ticks='all'` to restore astropy's per-spine assignment (only for a hand-rolled
# > / rotated WCS). (`'inout'` is also mapped to `'in'`, which WCSAxes can't draw.)
#
# ### Rotating the labels
#
# On a curved frame the tick labels are angled to follow their gridlines by default —
# you saw it on the globe above, where the hour labels fan around the limb. That
# alignment is set at build time with `tick_rotation=`, which takes four forms:
#
# - `'tangent'` (the default) — each label follows its local gridline tangent and stays
#   **upright**, with the orientation kept *continuous* along each gridline (no abrupt
#   mid-curve flip); only a gridline that genuinely sweeps *through* vertical has to lean
#   a label past it. (`'tangent_noflip'` is an explicit alias of this default.)
# - `'tangent_upright'` — the strict-upright variant: every label is clamped upright,
#   flipping 180° wherever a gridline crosses vertical. Reach for it when you want
#   guaranteed legibility on heavily wrapping curves and don't mind the occasional flip.
# - `'horizontal'` — every label sits flat and upright, easiest to read at a glance;
# - **a fixed angle** (a number) — every label takes the *same* rotation: upright and
#   unflipped, but no longer following the curve at all.
#
# Three of the looks on the same tilted globe (`'tangent_upright'` is indistinguishable
# from `'tangent'` here — they only diverge where a gridline sweeps through vertical):

# %%
fig = plt.figure(figsize=(13.5, 4.8))
for i, rot in enumerate(["tangent", "horizontal", 30], start=1):
    ax = sph.make_wcs_frame(130 + i, "SIN", center=(60, 35), fig=fig,
                            tick_rotation=rot)
    ax.set_title(f"tick_rotation={rot!r}", fontsize=10)
fig.subplots_adjust(wspace=0.3)
plt.show()

# %% [markdown]
# `'tangent'` keeps each label upright and riding its gridline; `'horizontal'` lays them
# flat; the fixed `30` slants them uniformly. `tick_rotation` governs the **in-frame /
# overlay** labels (ignored when `tick_style='native'`), so it pairs with the in-frame
# placement above.
#
# **What the default's correction buys you.** `'tangent'` keeps the orientation continuous
# *and* upright by correcting per gridline. The naive alternative — the raw, unclamped
# tangent, available as the callable `rotate=lambda t: t.tangent_deg` — follows the curve
# too but doesn't correct, so it runs labels upside-down where the tangent passes ±90°. The
# contrast is clearest on a wrap-around curve: declination labels riding an all-sky
# **boundary** limb.

# %%
fig = plt.figure(figsize=(11, 5.0))
modes = [("tangent (default)", "tangent"),
         ("rotate=lambda t: t.tangent_deg", lambda t: t.tangent_deg)]
for i, (name, rot) in enumerate(modes, start=1):
    ax = sph.make_wcs_frame(120 + i, "AIT", center=0, frame="ICRS", fig=fig,
                            tick_style="native")
    sph.add_overlay_ticks(ax, lon_at=None, lat_at="boundary", suppress_default="both",
                          label_kwargs={"fontsize": 9, "rotate": rot})
    ax.set_title(name, fontsize=10, pad=12)
fig.subplots_adjust(wspace=0.3)
plt.show()

# %% [markdown]
# The default reads cleanly all the way around; the raw tangent inverts the far half. If you
# instead want *every* label clamped upright even on a gridline that genuinely sweeps
# through vertical — accepting a 180° flip there — use `'tangent_upright'`. And any callable
# is accepted, for full per-label control. To fix up a *single* label after the fact —
# one that ended up reading upside-down — `flip_label(text)` toggles that one label's
# orientation by 180°.

# %% [markdown]
# ### Along the curve, or across it
#
# Every rotation so far keeps the label running *along* its gridline. The complementary control
# turns it to sit **across** the line instead: `rotate='tangent_perp'` (equivalently
# `'tangent+90'`) stands each label perpendicular to its gridline, and the general
# `'tangent+N'` / `'tangent-N'` take any angle in between — the offset is measured from the
# gridline's own tangent. This answers a common all-sky need: longitude labels dropped onto an
# interior parallel that stay upright and readable rather than leaning along the curve. The same
# AIT meridian labels on the `lat=-35` parallel, swept from along to across:

# %%
rot_modes = [("tangent  (along the line)", "tangent"), ("tangent+30", "tangent+30"),
             ("tangent+60", "tangent+60"), ("tangent_perp  (across)", "tangent_perp"),
             ("horizontal", "horizontal"), ("fixed 20°", 20)]
fig = plt.figure(figsize=(13, 6.6))
for i, (title, rot) in enumerate(rot_modes, start=1):
    ax = sph.make_wcs_frame(230 + i, "AIT", center=0, frame="ICRS", fig=fig,
                            tick_style="native")
    # sep='plain' writes the hours as '0h' (not '0ʰ') to keep the rotated labels compact:
    sph.add_overlay_ticks(ax, lon_at="lat=-35", lat_at=None, suppress_default="lon",
                          label_kwargs={"rotate": rot, "sep": "plain", "fontsize": 8})
    ax.set_title(title, fontsize=9, pad=8)
fig.subplots_adjust(hspace=0.28, wspace=0.18)
plt.show()

# %% [markdown]
# Reading the sweep: `'tangent'` lays each label along the parallel (leaning where the curve
# bends), `'tangent+30'` / `'tangent+60'` rotate it progressively off that tangent, and
# `'tangent_perp'` stands it fully across — the clean, upright look for interior-parallel
# longitude labels. `'horizontal'` and a fixed angle ignore the curve entirely. So the motivating
# case — *longitude labels only, on a lower parallel, upright across the curve* — is one call:
#
# ```python
# ax = sph.make_wcs_frame(111, "AIT", center=0)
# sph.add_overlay_ticks(ax, lon_at="lat=-30", lat_at=None, suppress_default="lon",
#                       label_kwargs={"rotate": "tangent_perp", "sep": "plain"})
# ```
#
# The `rotate=` values are the same vocabulary as the build-time `tick_rotation=` above — either
# accepts `'tangent'`, `'tangent_upright'`, `'tangent_perp'`, `'tangent±N'`, `'horizontal'`, a
# fixed angle, or a callable.

# %% [markdown]
# ### Targeting one axis
#
# Both label methods can restyle a *single* axis and leave the other untouched. The built-in edge
# labels take `format_ticklabels(which='lon')` / `'lat'` (default `'both'`); the overlay labels
# take separate `lon_label_kwargs=` / `lat_label_kwargs=` on `add_overlay_ticks()` (with matching
# `lon_tick_kwargs=` / `lat_tick_kwargs=` for the marks). Left: only the RA labels recolored and
# compacted, Dec left as it was; right: longitude and latitude given different colors in one
# overlay call:

# %%
urano = sph.CYCLE_PALETTES["uranometria"]["colors"]
fig = plt.figure(figsize=(11, 4.6))
# Left — which='lon' changes only the RA labels; Dec keeps the frame default.
ax = crab_field(121, fov_deg=4.0, fig=fig, grid=True, gridcolor="0.85", gridalpha=0.8)
sph.format_ticklabels(ax, style="compact", which="lon", color=HILITE)
ax.set_title("format_ticklabels(which='lon')", fontsize=10)
# Right — per-axis overlay label kwargs style longitude and latitude independently.
ax = sph.make_wcs_frame(122, "AIT", center=0, frame="ICRS", fig=fig, tick_style="native")
sph.add_overlay_ticks(ax, lon_at="lat=0", lat_at="boundary", suppress_default="both",
                      lon_label_kwargs={"color": ACCENT_RED, "fontsize": 8},
                      lat_label_kwargs={"color": urano[0], "fontsize": 8})
ax.set_title("per-axis lon_/lat_label_kwargs", fontsize=10)
fig.subplots_adjust(wspace=0.35)
plt.show()

# %% [markdown]
# The same control works on both methods: restyle one axis, or give each axis its own look,
# without disturbing the other.

# %% [markdown]
# ### Choosing a tick-label tool
#
# The label tools are organized through this tutorial by *task*, which is natural to read but
# awkward to search. Here they are in one place — find the row that matches what you want, then
# jump to the section that works it through:
#
# | I want to… | Use | Key argument(s) | § |
# |---|---|---|---|
# | Restyle the built-in edge labels (format, separator, color, stroke) | `format_ticklabels` | `style=`, `lon_sep`/`lat_sep`, `color=`, `stroke_lw`/`stroke_color` | 2 |
# | …but touch only one axis | `format_ticklabels` | `which='lon'` / `'lat'` | 2 |
# | Write the numbers relative to a reference (arcsec / mas offsets) | `apply_offset_ticks`, `apply_anchored_offset` | `unit=`, `ref_ra_deg`/`ref_dec_deg` | 4 |
# | Move labels onto an interior parallel or meridian | `add_overlay_ticks` | `lon_at='lat=-30'`, `lat_at='lon=45'` | 5 |
# | Style longitude vs latitude differently in one call | `add_overlay_ticks` | `lon_label_kwargs=`, `lat_label_kwargs=` | 5 |
# | Label the all-sky projection boundary | `apply_boundary_labels` | `orient=`, `side=` | 5 |
# | Rotate labels along / across / off a curve | `add_overlay_ticks` (or build-time `tick_rotation=`) | `rotate=` — `'tangent'`, `'tangent_perp'`, `'tangent±N'`, `'horizontal'` | 5 |
#
# > **Heads-up — two senses of "parallel."** `apply_boundary_labels(orient='parallel')` and the
# > overlay `rotate='tangent_perp'` produce the *same* across-the-line look, but their words are
# > anchored to different things: `orient=` is measured against the frame **boundary**
# > (`'parallel'` = along the edge), while `rotate=` is measured against the label's own
# > **gridline** (`'tangent'` = along that line). So `orient='parallel'` ≡ `rotate='tangent_perp'`
# > — worth remembering when a search for "parallel labels" lands you on one and you meant the
# > other.

# %% [markdown]
# ## 6. Sizing tick labels
#
# Most of the time you never have to think about tick-label size: every frame is built
# with `auto_fontsize=True`, which **shrinks** the labels just enough to fit the axes
# without crowding — clipped to `[6pt, rcParams['xtick.labelsize']]`, so it only ever
# scales *down* on tight or multi-panel layouts and never grows past your default. (It's
# quietly why the multi-panel galleries earlier in this tutorial stayed legible.) At a
# normal single-panel size nothing changes; pack the same field into a small panel and the
# difference shows:

# %%
panels = [
    ("auto_fontsize=False", dict(auto_fontsize=False), None),
    ("auto_fontsize=True", dict(auto_fontsize=True), None),
    ("auto_fontsize=True + compact", dict(auto_fontsize=True), "compact"),
]
fig = plt.figure(figsize=(9, 2.7))
for i, (title, kw, style) in enumerate(panels, start=1):
    ax = crab_field(130 + i, fov_deg=3.0, fig=fig, grid=True, gridcolor="0.85", **kw)
    if style:                          # drop the always-00 seconds to free up width
        sph.format_ticklabels(ax, style=style)
    for c in (ax.coords[0], ax.coords[1]):   # focus on the tick labels, not axis names
        c.set_auto_axislabel(False)
        c.set_axislabel("")
    ax.set_title(title, fontsize=9)
fig.subplots_adjust(wspace=0.5)
plt.show()

# %% [markdown]
# With auto-sizing off (left) the full-size labels dominate the little panel and astropy
# thins them to two to avoid collisions; on (middle) they shrink to fit and a third lands
# on the axis. But shrinking only goes so far — the real culprit is *width*, the redundant
# `00.00ˢ` seconds. Drop them (right) by combining `auto_fontsize=True` with a
# seconds-free format like `style='compact'`, and every major tick gets its label back.
# (On astropy ≥ 7 the general `simplify=True` flag does the same.)
#
# To re-fit an *existing* frame — you resized the figure, or built it with
# `auto_fontsize=False` and changed your mind — call `auto_size_ticklabels(ax)`. It picks
# and applies the fitted size (and returns it in points), with `floor=` / `ceiling=`
# bounds and an `n_ticks_hint=` for the expected tick count; pass `reflow_on_resize=True`
# to have it re-fit automatically on window resize and pan/zoom. The manual escape hatch is
# always there too: an explicit `fontsize=` on `format_ticklabels()` (or
# `ax.coords[i].set_ticklabel(fontsize=…)`) overrides the auto-fit entirely.

# %% [markdown]
# ## 7. Themes and palettes
#
# > **This section is a frame-focused overview.** Styling is a big enough topic to have its own
# > tutorial — [Themes, Palettes & Fonts](styling.ipynb) is the full treatment (every base preset
# > and cycle palette side by side, the light/dark and colorblind-safe stories, font selection,
# > and building a reusable house style). What follows here is just enough of the system to show
# > how the various parts of a *frame* — its ticks, grid, labels, and spine — pick up a coherent
# > style; the [Styling & themes](../guide/styling.md) guide page is the condensed reference.
#
# Everything so far styled *one* frame at a time. But a coherent look across a whole figure —
# or a whole paper, talk, or notebook — should be a single call, not a per-figure ritual.
#
# If you've used matplotlib's themes, you know a single style can change a great many settings
# at once — the *rcParams* that govern how plot elements appear: line weights, tick lengths,
# fonts, background colors, the cycle of colors used for data series, and so on. skyplothelper
# offers the same convenience but **breaks those settings into three layers**, so you can focus
# on one category and leave the others alone. That separation earns its keep constantly:
# churning out dozens of inspection plots, you might want to improve on matplotlib's default
# *structure* — frame, ticks, fonts — without disturbing the color sequence that encodes your
# sources; another time you just want a nicer data *palette* with the frame left alone; and
# sometimes you *do* want the whole coordinated shift a preset *theme* brings. You mix and
# match the three through `sph.set_style()`, and each takes a bundled preset *or* your own
# definition:
#
# - **base** — *the structure.* Tick direction and length, spine and line weights, grid,
#   fonts, dpi. Most presets also seed a *default* color cycle; `'structural'` is the lone
#   exception that leaves your colors (and fonts) untouched — the "improve the defaults, keep
#   my encoding" choice. Lead with `'standard'` (the opinionated default) or `'structural'`;
#   `'journal'`/`'press'`/`'poster'`/`'tufte'`/`'screen'` target specific media. Or pass your
#   own rcParams dict. (The **palette** layer below is still how you *choose* data colors — it
#   applies after base, so it always wins.)  These tutorial notebooks achieve a common look by
#   setting the base style to `'structural'` at the top, for example.
# - **theme** — *the page.* Background and foreground colors and how they coordinate, light
#   vs. dark: `'publication'`, `'twilight'`, `'dark_sky'`, `'poster'`. Any matplotlib built-in
#   style (`'ggplot'`, `'bmh'`, …) resolves here too, as does your own dict of rcParam settings.
# - **palette** — *the data colors.* The cycle applied to your plotted series — `'speakeasy'`,
#   `'atlas'`, `'uranometria'`, `'letterpress'`, `'nightcap'`, `'velvet'` — or an explicit
#   list of colors.
#
# **Two ways to apply them — and when to use which.** Both take the same three keywords:
#
# - `sph.set_style(base=…, theme=…, palette=…)` sets the look **persistently** — it updates
#   the global rcParams and stays in effect for every figure afterward. Use it once at the top
#   of a script, or to pin a house style for a whole session.
# - `with sph.style_context(base=…, theme=…, palette=…):` applies the *same* composition
#   **temporarily** and restores the previous rcParams on exit. This is the right tool in a
#   notebook: each cell styles only its own figure and never leaks settings into the next, so
#   cells stay re-runnable in any order. That's why every styled cell in this section uses it.
#
# Either way the layers compose in one fixed order — base → theme → palette → any extra
# rcParams you pass last — so a later layer wins on a shared key. Apply all three at once, or
# just one: each layer also has its own standalone setter — `set_base_style`, `set_theme`,
# `set_palette` — for when you only ever touch that category (e.g. `sph.set_theme("dark_sky")`
# to flip only the page colors, or `sph.set_palette("atlas")` for only the data cycle). We'll
# add them one at a time:

# %%
wl = np.linspace(0, 10, 200)
demo = [np.sin(wl - k) + 0.15 * k for k in range(4)]    # a few spectra-like series


def _demo_plot(ax):
    for y in demo:
        ax.plot(wl, y)
    ax.set_xlabel("wavelength")
    ax.set_ylabel("flux")


layers = [("matplotlib default", None),
          ("+ base='structural'", dict(base="structural")),
          ("+ theme='dark_sky'", dict(base="structural", theme="dark_sky")),
          ("+ palette='nightcap'",
           dict(base="structural", theme="dark_sky", palette="nightcap"))]
# Pin a clean matplotlib default as the shared baseline for the whole diagram —
# figure background, titles, and (crucially) the starting color cycle — so the
# buildup reads identically no matter what style happens to be active when the
# cell runs. Each panel then layers only what its title names on top of that same
# start, so the data colors hold steady across panels 1-3 and change only at the
# palette step (panel 4) — the whole point of the comparison.
with plt.style.context("default"):
    fig = plt.figure(figsize=(14, 3.0))
    for i, (title, cfg) in enumerate(layers, start=1):
        if cfg is None:                                 # the baseline itself
            ax = fig.add_subplot(1, 4, i)
            _demo_plot(ax)
        else:
            with sph.style_context(base=cfg["base"], theme=cfg.get("theme"),
                                   palette=cfg.get("palette")):
                ax = fig.add_subplot(1, 4, i)
                _demo_plot(ax)
        ax.set_title(title, fontsize=9)
    fig.subplots_adjust(wspace=0.4)
    plt.show()

# %% [markdown]
# Reading left to right: after the default mpl appearance on the left, the next panel shows **base**, which here is set to `'structural'`, the preset that **deliberately
# leaves your color cycle and fonts untouched** — it nudges only the *structure* (line
# weights, ticks, grid), so panels 1 and 2 share the exact same data colors. **theme** then
# recolors the page and foreground but still not the data, and **palette** finally swaps the
# data cycle. That clean isolation is exactly why `'structural'` is the "improve the
# defaults, keep my encoding" choice; the opinionated `'standard'` default would also restyle
# the cycle at the base step. The presets at a glance:
#
# | Layer | Sets | Presets |
# |---|---|---|
# | `base=` | structure — ticks, spines, grid, fonts, dpi | `standard`, `structural`, `journal`, `press`, `poster`, `tufte`, `screen`, `minimalist`\* |
# | `theme=` | page / axes background + foreground | `publication`, `twilight`, `dark_sky`, `poster` (+ any matplotlib built-in) |
# | `palette=` | data color cycle | `speakeasy`, `atlas`, `uranometria`, `letterpress`, `nightcap`, `velvet` |
#
# \* `minimalist` strips the frame for splash images, not data plots. One deliberate rule: a
# **skyplothelper** theme never silently recolors your data — the built-in themes set only
# page and foreground colors,
# so switching to a dark theme keeps whatever palette you set and a dark figure never
# surprises you with new colors. The exception is a borrowed **matplotlib** built-in
# (`'ggplot'`, `'seaborn-*'`, …): those carry their own color cycle, so they *will* restyle
# your series — pass an explicit `palette=` afterward if you want to keep yours.

# %% [markdown]
# ### The built-in themes at a glance
#
# A **theme** sets the *page* — background, foreground, and the grid/label colors that go with
# it — but **not your data colors** (it bundles no palette cycle). The four built-ins on the same
# all-sky frame; the galactic-plane great circle is drawn in one fixed in-theme accent on every
# panel, underscoring that the theme recolors the page *around* it, never the overlay itself:

# %%
fig = plt.figure(figsize=(11, 6.0))   # canvas follows the docs page (white in the light build,
#                                       transparent over the dark page in the dark build); each
#                                       panel keeps its own theme background.
for i, theme in enumerate(["publication", "twilight", "dark_sky", "poster"], start=1):
    # A contrasting stroke so every label reads on either page: light themes (dark text)
    # get a white stroke, dark themes (light text) a black one.
    stroke_fg = "white" if theme in ("publication", "poster") else "black"
    stroke = [pe.withStroke(linewidth=1.0, foreground=stroke_fg)]
    with sph.style_context(theme=theme):
        ax = sph.make_wcs_frame(220 + i, "AIT", center=0, frame="ICRS", fig=fig)
        sph.style_wcs_axes(ax)                       # carry the theme onto the frame
        sph.format_ticklabels(ax, style="allsky_hours", stroke_lw=1.0, stroke_color=stroke_fg)
        sph.add_great_circle(ax, color=ACCENT_RED, lw=1.8)   # galactic plane, one fixed accent
        for c in (ax.coords[0], ax.coords[1]):               # stroke the axis labels too
            if c.get_axislabel():
                c.set_axislabel(c.get_axislabel(), path_effects=stroke)
    ax.set_title(f"theme={theme!r}", fontsize=10, path_effects=stroke)
fig.subplots_adjust(hspace=0.35, wspace=0.25)
plt.show()

# %% [markdown]
# ### The cycle palettes at a glance
#
# Six data-cycle palettes ship built in. They're tuned to stay distinct under
# **color-vision deficiency** *and* in grayscale, and each carries a `mode` — `dual`
# (reads on a light or dark page), `light`-only, or `dark`-only — so a dark figure gets
# data colors that actually hold up on a dark background. All six:

# %%
fig, ax = plt.subplots(figsize=(10, 3.0))
names = list(sph.CYCLE_PALETTES)
ncol = max(len(sph.CYCLE_PALETTES[n]["colors"]) for n in names)
for r, name in enumerate(names):
    spec = sph.CYCLE_PALETTES[name]
    y = len(names) - 1 - r                     # first palette on top
    for c, color in enumerate(spec["colors"]):
        ax.add_patch(plt.Rectangle((c, y + 0.08), 0.92, 0.84, color=color))
    ax.text(-0.25, y + 0.5, name, ha="right", va="center", fontsize=10)
    ax.text(ncol + 0.25, y + 0.5, spec["mode"], ha="left", va="center",
            fontsize=8, style="italic")
ax.set_xlim(-4.0, ncol + 2.5)
ax.set_ylim(0, len(names))
ax.axis("off")
plt.show()

# %% [markdown]
# Apply one with `palette=` / `set_palette`, exactly as in the buildup above. Every palette as
# lines and as region fills, the light/dark pairings, and the color-vision-deficiency
# simulations side by side are laid out in the [Themes, Palettes & Fonts](styling.ipynb)
# tutorial and its [Styling & themes](../guide/styling.md) reference page.
#
# ### Carrying the look onto a sky frame
#
# A sky-specific reassurance: `make_wcs_frame` builds its frames **theme-aware**, so the active
# style reaches a `WCSAxes` automatically — set a theme and the frame's ticks, labels, and spine
# come out in the theme's colors with no extra call. (A `WCSAxes` ignores matplotlib's
# `xtick.*`/`ytick.*` rcParams, so this is applied explicitly at build rather than inherited
# through rc.) The left panel below is just `set_style` + a bare `crab_field`; `style_wcs_axes(ax)`
# on the right adds the optional tick **geometry** from Section 5 — longer ticks and minor
# subdivisions — but the colors already match either way:

# %%
with sph.style_context(base="standard", theme="dark_sky", palette="nightcap"):
    fig = plt.figure(figsize=(11, 4.8))   # built inside the context so the page goes dark too
    ax = crab_field(121, fov_deg=4.0, fig=fig, grid=True)
    ax.set_title("theme carried automatically", fontsize=10)
    ax = crab_field(122, fov_deg=4.0, fig=fig, grid=True)
    sph.style_wcs_axes(ax)     # optional: refine the tick geometry (length, minor ticks, width)
    ax.set_title("+ style_wcs_axes(ax): tick geometry", fontsize=10)
    fig.subplots_adjust(wspace=0.35)
    plt.show()

# %% [markdown]
# > **Tip — dark mode for your own figures.** The same machinery styles *your* work: wrap a
# > figure in `with sph.style_context(theme="dark_sky"):` (or call `sph.set_style(theme=
# > "dark_sky")` once for a whole session) to put it — frame and all — on a dark page. Add
# > `style_wcs_axes(ax)` per frame only when you want to refine the ticks. Pair it with a
# > dark-friendly cycle palette (`palette="nightcap"` or `"velvet"`) and it's presentation-ready
# > — the light/dark toggle on these docs is driven by exactly this.

# %% [markdown]
# ### Annotation palettes — a separate layer for decoration
#
# Alongside the three look-layers (`base`/`theme`/`palette`) there's a **fourth, optional
# layer** aimed squarely at *decoration*: **annotation palettes**. Where the cycle palette
# colors your *data*, an annotation palette coordinates a figure's **scaffolding** — two
# tiers of text, two tiers of grid, the frame, star markers, a label color, a compass, and
# two accents — as one coherent set for finder-chart and star-atlas figures. Apply one with
# `style_annotation(ax, name)`; it styles the scaffolding *and returns its color dict*, so you
# can pull `pal["stars"]`, `pal["accent"]`, and the rest for elements you draw yourself:
#
# ```python
# pal = sph.style_annotation(ax, "publication")   # styles scaffolding, returns colors
# ax.scatter(ra, dec, c=pal["stars"], transform=ax.get_transform("world"))
# ```
#
# The five presets (the `ANNOTATION_PALETTES` registry) are `parchment`, `publication`,
# `dark`, `night`, and `denim`. Here two of
# them — `publication` (light) and `denim` (warm dark) — on the same star field, exercising
# several roles at once: star markers, an accent target ring, a primary label, a secondary
# caption, and a compass, all drawn from the one returned dict. The full gallery of five and
# the complete color-key are on the [Styling & themes](../guide/styling.md) guide page:

# %%
rng_s = np.random.default_rng(11)
n_star = 90
st_ra = CRAB[0] + rng_s.uniform(-0.6, 0.6, n_star) / np.cos(np.radians(CRAB[1]))
st_dec = CRAB[1] + rng_s.uniform(-0.6, 0.6, n_star)
st_sz = 10 ** rng_s.uniform(0.2, 1.8, n_star)

fig = plt.figure(figsize=(11, 5.2))
for i, name in enumerate(["publication", "denim"], start=1):
    ax = sph.make_wcs_frame(120 + i, "TAN", center=CRAB, fov_deg=1.4, fig=fig)
    pal = sph.style_annotation(ax, name)         # styles the scaffolding, returns colors
    tr = ax.get_transform("world")
    # Field stars in the palette's star color:
    ax.scatter(st_ra, st_dec, transform=tr, s=st_sz, c=pal["stars"],
               edgecolors="none", zorder=4)
    # Target: an accent ring with the primary object label in the label color:
    ax.scatter([CRAB[0]], [CRAB[1]], transform=tr, s=160, facecolors="none",
               edgecolors=pal["accent"], linewidths=1.6, zorder=5)
    ax.annotate("M1", xy=(CRAB[0], CRAB[1]), xycoords=tr, xytext=(12, 10),
                textcoords="offset points", color=pal["label"], fontsize=11,
                fontweight="bold", zorder=7)
    # A secondary caption in the lower-tier text color, and a compass rose — each
    # in its own palette role, so one call coordinates the whole finder-chart kit:
    ax.text(0.96, 0.05, "1.4° field · ICRS", transform=ax.transAxes,
            color=pal["text2"], fontsize=8, ha="right", zorder=7)
    sph.add_compass(ax, color=pal["compass"], stroke_color=pal["ax_bg"])
    ax.set_title(f"style_annotation(ax, '{name}')", fontsize=10)
fig.subplots_adjust(wspace=0.35)
plt.show()

# %% [markdown]
# ### A note on fonts
#
# Fonts ride along with the **base** layer: each preset sets a coordinated font *stack* (and a
# matching math fontset — serif presets pair `cm`, sans presets `stixsans`). The stacks
# degrade gracefully, falling back through several families (e.g. TeX Gyre → Carlito /
# Liberation → DejaVu) so a figure still renders sensibly on a machine missing the first
# choice. To set a font yourself, pass it as an rcParams override on any styling call:
#
# ```python
# sph.set_style(base="journal", **{"font.family": "serif"})    # or a specific stack:
# with sph.style_context(**{"font.family": ["Source Sans 3", "DejaVu Sans"]}):
#     ...
# ```
#
# For fixed-width tabular text — aligned coordinate readouts, a small data table on the figure
# — `sph.MONO_STACK` is a ready monospace stack to hand any text artist:
#
# ```python
# ax.text(0.02, 0.02, "RA   05:34:32.0\nDec  +22:00:52", family=sph.MONO_STACK,
#         transform=ax.transAxes)
# ```
#
# Choosing fonts for a house style — pairing display and body faces, the math-font coupling,
# and the full fallback story — is its own topic on the [Styling & themes](../guide/styling.md)
# guide page.

# %% [markdown]
# That's the flavor of the styling system: three composable layers for the overall look
# (`base` / `theme` / `palette`), a fourth annotation-palette layer for finder-chart
# scaffolding, and `style_wcs_axes` to bridge any of it onto sky frames.
#
# > **Going deeper.** This section is a tour of the *flavors*. The full treatment — every
# > base preset and cycle palette side by side, the light/dark and colorblind-safe stories,
# > font selection, the annotation-palette gallery, and building a reusable house style —
# > is the [Themes, Palettes & Fonts](styling.ipynb) tutorial; the
# > [Styling & themes](../guide/styling.md) guide page is the condensed reference.

# %% [markdown]
# ## 8. Putting it together
#
# Time to try everything at once — on a *real* image this time. Here is a VLBA 15 GHz map of
# the radio galaxy **3C 84** (0316+413), shown twice: the **same** image (skyplothelper's
# bundled `sph.deepsky` colormap on a symmetric-log stretch) in both panels, so the only thing
# that changes is the **decoration**. The left panel shows the image with contours and a
# colorbar on an undecorated frame — the parts you get for free in skyplothelper; the right stacks the
# extra decorations this tutorial covers.
# (We hold the colormap and stretch fixed deliberately — choosing those is the *image*-display
# story, covered in [FITS Images & Quicklook](fits_images.ipynb); here the point is purely what
# the decoration adds.)
#
# > **Data:** the image is a stacked 15 GHz (2 cm) VLBA map of 3C 84 from the **MOJAVE**
# > program (Monitoring Of Jets in Active galactic nuclei with VLBA Experiments; Lister et al.
# > 2018, *ApJS* 234, 12), [www.cv.nrao.edu/MOJAVE](https://www.cv.nrao.edu/MOJAVE/).

# %%
# Load the FITS map (4-D radio cube → 2-D) and build its SIN WCS:
_fits = "../../examples/data/0316+413.u.stacked.icd.fits"
with fits.open(_fits) as _hdul:
    cube, hdr3c84 = _hdul[0].data, _hdul[0].header
dat3c84 = np.squeeze(cube)
with warnings.catch_warnings():
    # This header's DATE-OBS is the non-standard 'MULTIEPOCH' (a stacked map) — astropy's
    # WCS fixes it up and warns; harmless here, so quiet it to keep the output clean:
    warnings.simplefilter("ignore", FITSFixedWarning)
    wcs3c84 = WCS(hdr3c84).celestial
core = (hdr3c84["CRVAL1"], hdr3c84["CRVAL2"])              # 3C 84 core (offset origin)
cpx, cpy = wcs3c84.world_to_pixel_values(*core)
half = 17 / (abs(hdr3c84["CDELT2"]) * 3.6e6)              # ±17 mas window, in pixels

# A symmetric-log norm reveals the inner structure that a linear/sqrt stretch saturates
# (linear below linthresh, log above); scale it to the peak in the cropped window:
crop = dat3c84[int(cpy - half):int(cpy + half), int(cpx - half):int(cpx + half)]
peak = float(np.nanmax(crop))
norm3c84 = mcolors.SymLogNorm(linthresh=5e-3, vmin=0, vmax=peak, base=10)
# Radio contours doubling from ~3.5σ above the off-source noise (rms ≈ 0.83 mJy/beam):
rms3c84 = 8.3e-4                                          # Jy/beam
levels3c84 = 3.5 * rms3c84 * 2.0 ** np.arange(0, np.log2(peak / (3.5 * rms3c84)), 1.0)
# The image is dark whatever the page looks like, so anything drawn *on* it — frame, ticks,
# grid, reticle, the source label — takes its colors from the **dark annotation palette** (§7),
# whose roles are designed for exactly this. The contours take a second bundled colormap
# (`sph.lagoon_r`), which keeps them distinct from the image without shouting; the tick *labels*
# sit outside the frame, so they stay theme-driven.
PAL_DARK = sph.ANNOTATION_PALETTES["dark"]
# Contours take the bundled 'lagoon' colormap *reversed*, which is what makes them legible
# everywhere: the faint outer levels come out light against the dark sky, the bright core
# levels teal-green against the cream core — distinct from the copper/blue image without the
# glare of a neon cmap. (Plain 'lagoon' would put its pale end on the bright core and vanish.)
contour_cmap = plt.get_cmap("sph.lagoon_r")


def show_3c84(ax):        # the shared backdrop: skyplothelper's 'deepsky' + the SymLog stretch
    im = ax.imshow(dat3c84, cmap="sph.deepsky", origin="lower", norm=norm3c84)
    ax.set_xlim(cpx - half, cpx + half)
    ax.set_ylim(cpy - half, cpy + half)
    # Frame + ticks in the palette's frame color so they read against the dark sky:
    sph.style_wcs_axes(ax, tick_color=PAL_DARK["frame"], frame_color=PAL_DARK["frame"])
    return im


fig = plt.figure(figsize=(12.5, 5.6))
# Left — an undecorated frame, but WITH contours + a colorbar (the parts you get for free): a
# standard image colorbar, contours on top in their own (linear-mapped) colors.
ax0 = fig.add_subplot(121, projection=wcs3c84)
im0 = show_3c84(ax0)
ax0.contour(dat3c84, levels=levels3c84, cmap=contour_cmap, linewidths=0.6)
fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04).set_label("Jy / beam", fontsize=9)
for _ci, _nm in ((0, "Right Ascension"), (1, "Declination")):
    ax0.coords[_ci].set_axislabel(_nm, fontsize=9)
ax0.set_title("skyplothelper Default", fontsize=12)
# Right — decorated. The contours take the image's SymLog norm (faint log-spaced levels spread
# out), and the colorbar is *dual-purpose*: the deepsky image gradient as the base with the
# contour-level colors overlaid as marks — so one bar reads both the continuous image and the
# discrete levels. Plus offset coords re-zeroed on the core, a styled grid, fine inward ticks,
# a reticle, and an in-frame source label:
with sph.style_context(base="journal"):
    ax1 = fig.add_subplot(122, projection=wcs3c84)
    im1 = show_3c84(ax1)
    ax1.contour(dat3c84, levels=levels3c84, cmap=contour_cmap,     # SymLog → levels brought out
                norm=norm3c84, linewidths=0.5)
    cb = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)       # deepsky base = image scale
    cb.set_label("Jy / beam", fontsize=9)
    # Tick + label the bar at the contour levels themselves (a level legend). length=0
    # because journal's inward ticks would hide on the bar — the overlaid level lines below
    # act as the tick marks:
    cb.set_ticks(levels3c84)
    cb.set_ticklabels([f"{lev:.2g}" for lev in levels3c84])
    cb.ax.tick_params(labelsize=7, length=0)
    for lev in levels3c84:                                         # overlay the level colors
        cb.ax.axhline(lev, color=contour_cmap(norm3c84(lev)), lw=2.0)
    sph.apply_offset_ticks(ax1, ref_ra_deg=core[0], ref_dec_deg=core[1],   # §4
                           unit="mas", spacing=10, precision=0)
    sph.style_grid(ax1, color=PAL_DARK["grid"], alpha=0.9, lw=0.7, ls=":")     # §3
    sph.style_wcs_axes(ax1, direction="in", major_size=6, minor_size=3,        # §5
                       minor_ticks=True, minor_frequency=5, width=1.0,
                       tick_color=PAL_DARK["frame"], frame_color=PAL_DARK["frame"])
    sph.add_reticle(ax1, core, color=PAL_DARK["accent"],                       # §6-adjacent
                    stroke_color="black", size=1.1, lw=1.6)
    # Source name as an in-frame label near the top (stroked for legibility over the imagery):
    ax1.text(0.5, 0.97, "3C 84 — VLBA 15 GHz", transform=ax1.transAxes,
             ha="center", va="top", fontsize=11, fontweight="bold", color=PAL_DARK["text"],
             path_effects=[pe.withStroke(linewidth=2.5, foreground="black")], zorder=8)
    ax1.set_title("Decorated", fontsize=12)
fig.subplots_adjust(wspace=0.5)
plt.show()

# %% [markdown]
# Same pixels — the same image and contours — but a more useful appearance. On the left, a bare
# frame: at 0.1 mas/pixel the absolute RA/Dec labels are not particularly useful at a glance (every tick is `03ʰ19ᵐ48ˢ` to
# the fraction of a second). The right panel makes the field more *legible* for scientific inspection by stacking this tutorial's tools:
# **offset coordinates** re-zeroed on the core (§4), a **styled grid** in an annotation-palette
# color (§3, §7), clean **inward ticks with minor subdivisions** in the palette's frame color
# (§5), the **journal** base for a print-paper look (§7), a **reticle** on the core, and a
# **dual-purpose colorbar** — the deepsky image scale with the contour levels overlaid (a shared
# SymLog norm spreads the faint levels out). Point it at your own FITS map and the same handful
# of calls give you a publication-ready figure. That's the whole tutorial in one frame: the
# picture is half the work; how it reads is the other half.

# %% [markdown]
# ## 9. Bonus topic: Log and symmetric-log axes
#
# Everything so far has decorated a **sky** (WCS) frame, where the axes are angles. But the same
# figure often carries a companion *data* plot on ordinary Cartesian axes — residuals against
# epoch, flux against radius, a fit against its scatter — and those frequently want a **log**
# axis. Two cases come up:
#
# - **Log**, for a strictly-positive quantity spanning decades (a formal uncertainty, a flux, a
#   separation): `ax.set_yscale('log')`, and matplotlib gives you decade major ticks plus the
#   2–9 subdecade minors for free.
# - **Symmetric log**, for a *signed* quantity spanning decades on both sides of zero — the
#   classic case being **residuals** (O$-$C offsets) that swing positive *and* negative. A plain
#   log axis can hold neither zero nor negatives, and matplotlib's built-in `'symlog'` bolts a
#   linear segment across the middle with a visible kink at the join. The companion package
#   **pysymlog** (a separate `pip install pysymlog`) registers a *smooth* `'symmetriclog'` scale
#   that transitions continuously through zero: the tick marks **stretch** near the origin so
#   small residual offsets and large outliers read on one axis.
#
# A worked example — one source's position monitored from 1992 to 2025 as the technique improves
# from photographic plates (~300 mas) to VLBI / Gaia (~30 μas), four decades of precision. The
# formal uncertainty (positive) wants a log axis; the signed residual wants symmetric-log:

# %%
import pysymlog as psl  # noqa: E402, I001 — optional companion package: pip install pysymlog

psl.register_mpl()            # registers the smooth 'symmetriclog' matplotlib scale

rng_ast = np.random.default_rng(4)
epoch = np.linspace(1992.0, 2025.0, 90)
sigma = 300.0 * 10.0 ** (-(epoch - 1992.0) / 8.5)     # formal σ (mas): ~300 → ~0.03, positive
drift = 0.05 * (epoch - 2011.0)                        # a faint real astrometric drift, sub-mas
resid = drift + rng_ast.normal(0.0, sigma)            # signed O−C residual (mas), tracks σ
urano = sph.CYCLE_PALETTES["uranometria"]["colors"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
ekw = dict(fmt=".", ms=4, elinewidth=0.6, capsize=0, color=urano[0], ecolor="0.6", zorder=3)
# Linear — the recent, precise decades collapse onto the zero line and vanish:
axes[0].errorbar(epoch, resid, yerr=sigma, **ekw)
axes[0].axhline(0, color=urano[5], lw=1.0)
axes[0].set_ylabel("O$-$C offset (mas)")
axes[0].set_title("residuals · linear y", fontsize=10)
# Symmetric-log — the ticks stretch near zero, so both signs and every decade become legible:
axes[1].errorbar(epoch, resid, yerr=sigma, **ekw)
axes[1].axhline(0, color=urano[5], lw=1.0)
axes[1].set_yscale("symmetriclog", shift=0.1)         # shift ≈ the linear→log transition point
psl.set_symmetriclog_minorticks(axes[1], xy="y", thresh=0.1)
axes[1].set_title("residuals · symmetriclog y (pysymlog)", fontsize=10)
# Plain log — the strictly-positive uncertainty, four decades of steady improvement:
axes[2].plot(epoch, sigma, "-", color=urano[4], lw=1.8)
axes[2].set_yscale("log")
axes[2].set_ylabel("formal σ (mas)")
axes[2].set_title("uncertainty σ · log y", fontsize=10)
for ax in axes:
    ax.set_xlabel("epoch (year)")
    ax.grid(True, which="major", color="0.5", alpha=0.3, lw=0.6)
    ax.grid(True, which="minor", color="0.5", alpha=0.12, lw=0.4)
fig.subplots_adjust(wspace=0.32)
plt.show()

# %% [markdown]
# The linear panel is dominated by the noisy 1990s; every measurement after ~2005 piles onto the
# zero line, invisible. The **symmetric-log** panel stretches the axis near zero so the same data
# reads across its whole four-decade range — and a faint real signal surfaces that the linear
# view buried completely: a slow drift crossing zero around 2011 and settling to a clean positive
# offset once σ drops below it. That is what symmetric-log buys for residuals — both signs, at
# every scale, with no kink at the linear boundary. `shift=` sets where the smooth linear→log
# transition sits (roughly the smallest offset you want resolved), and
# `set_symmetriclog_minorticks()` adds the subdecade minors.

# %% [markdown]
# ### On a sky frame: position-offset maps
#
# The natural sky version is a 2D map of **position offsets** — each source's measured offset
# from a reference position. A real and well-studied case: the **radio–optical offsets** between
# VLBI and Gaia positions of active galactic nuclei. Their magnitudes run close to log-normal —
# most sources agree to a fraction of a milliarcsecond, with a tail reaching tens of mas where an
# optical jet pulls the Gaia centroid off the radio core — while their directions are essentially
# random. That is a *log-distributed* quantity that still carries a sign in each coordinate:
# exactly the symmetric-log case.
#
# A detail worth stating: this goes on **plain Cartesian axes**, not a `WCSAxes`. Offset
# coordinates are linear tangent-plane offsets — already Cartesian — so you plot them directly
# and scale the axes like any data. (A `WCSAxes` draws its ticks through the WCS rather than
# matplotlib's scale machinery, so a nonlinear `set_yscale` on one warps the data while the ticks
# stay linear — not what you want. Section 4's `apply_offset_ticks` is for offset *labels* on a
# WCS *image*; a residual *scatter* belongs on Cartesian offset axes.)

# %%
rng_res = np.random.default_rng(12)
n_src = 300
r_off = rng_res.lognormal(mean=np.log(0.4), sigma=1.5, size=n_src)   # offset magnitude (mas)
pa_off = rng_res.uniform(0.0, 2 * np.pi, n_src)                       # random direction
dra = r_off * np.cos(pa_off)          # Δα cos δ (mas)
ddec = r_off * np.sin(pa_off)         # Δδ (mas)
lim = 30.0                            # frame both panels identically — square and symmetric

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
# One population, colored by the offset magnitude itself (a log norm, matching the data's own
# distribution); a hairline edge keeps individual points readable where the core gets crowded.
pts = dict(c=r_off, cmap="sph.lagoon", norm=mcolors.LogNorm(vmin=0.01, vmax=lim),
           s=26, edgecolors="0.25", linewidths=0.4, zorder=3)
for ax in axes:
    sc = ax.scatter(dra, ddec, **pts)
    ax.axhline(0, color="0.5", lw=0.6, zorder=1)
    ax.axvline(0, color="0.5", lw=0.6, zorder=1)
    ax.set_xlabel(r"$\Delta\alpha\cos\delta$ (mas)")
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, color="0.5", alpha=0.22, lw=0.5)
axes[0].set_ylabel(r"$\Delta\delta$ (mas)")
axes[0].set_title("linear offset axes — the sub-mas core collapses", fontsize=10)
axes[1].set_title("symmetriclog — the whole distribution resolves", fontsize=10)
axes[1].set_xscale("symmetriclog", shift=0.1)
axes[1].set_yscale("symmetriclog", shift=0.1)
psl.set_symmetriclog_minorticks(axes[1], xy="x", thresh=0.1)
psl.set_symmetriclog_minorticks(axes[1], xy="y", thresh=0.1)
# Pin well-separated majors: near the origin — where symmetric-log compresses hardest — the
# automatic decades crowd ±10⁻¹ up against 0 and the labels collide. The minor ticks placed
# just above still carry the sub-decade detail.
axes[1].set_xticks([-10, -1, 0, 1, 10])
axes[1].set_yticks([-10, -1, 0, 1, 10])
# sph.add_colorbar sizes the bar to the panel's *drawn* box, so it matches the spine exactly;
# plain plt.colorbar sizes to the axes bounding box and overshoots on a fixed-aspect panel.
cb = sph.add_colorbar(sc, ax=axes[1], label="radio–optical offset (mas)")
plt.show()

# %% [markdown]
# Both panels hold the same points on the same square, symmetric frame - with the same +/- 30 mas extents. On the left the
# log-normal core piles into a knot a few pixels across: you can see *that* there is a
# concentration, but nothing about its shape, and the axes spend their whole range on the few
# outliers. On the right, symmetric-log stretches both axes through zero and the distribution
# opens up — a dense sub-mas core, a smooth falloff across two decades, and the tens-of-mas tail
# still on the same frame, with the sign preserved in all four quadrants. `shift=` (here 0.1 mas)
# sets how much of the near-origin core is stretched out; coloring the points by the offset
# magnitude ties the two views together.
#
# Symmetric-log is also the log-family scale that keeps points sitting *exactly* at zero: a plain
# `'log'` axis silently drops them (log 0 is undefined), so a perfectly-agreeing source — residual
# zero, the *expected* astrometric outcome — or a genuine zero-valued measurement would simply
# vanish. Keeping those zeros on an otherwise-logarithmic plot is often the whole reason to reach
# for the scale.

# %% [markdown]
# > **Related.** For the *image* analogue — a symmetric-log color stretch on a diverging map
# > (signed flux, a residual image) — `make_norm` / the quicklook tools take
# > `stretch='symmetric_log'` (the same idea applied to pixel values; see
# > [FITS Images & Quicklook](fits_images.ipynb)). On a **polar / cone** frame, `sph.log_r(ax)`
# > switches the radial axis to log with readable ticks (see
# > [Cone & Bowtie Plots](cone_bowtie.ipynb)).

# %% [markdown]
# ## 10. How do I…?
#
# The tutorial is organized by *topic*; this is the same material indexed by **question**. Find
# the thing you want to change, then jump to the section that works it through.
#
# **Gridlines**
#
# | I want to… | Do this | § |
# |---|---|---|
# | Draw more / fewer gridlines | `make_wcs_frame(lon_spacing=…, lat_spacing=…)` at build (default `'auto'`) | 3 |
# | Turn the grid off | `grid=False` at build, or `ax.coords.grid(draw_grid=False)` after | 3 |
# | Recolor or restyle the grid | `style_grid(ax, color=, alpha=, lw=, ls=)` | 3 |
# | Highlight one line, or a family by value | `highlight_gridline(...)` / `highlight_gridlines(lon_cmap=…)` | 3 |
# | Show a globe's *far* side | `plot_ortho_grid(...)` (+ `highlight_meridian_tracer`) | 3 |
#
# **Tick marks**
#
# | I want to… | Do this | § |
# |---|---|---|
# | Place more / fewer ticks | `ax.coords[i].set_ticks(number=N)` (a hint), or `spacing=` / `values=` (exact) | 4, 5 |
# | Make ticks inward, longer, thicker | `style_wcs_axes(direction='in', major_size=, width=)` | 5 |
# | Add minor ticks | `style_wcs_axes(minor_ticks=True, minor_frequency=, minor_size=)` | 5 |
# | Kill stray corner ticks on a projected field | `edge_ticks='auto'` at build (already the default) | 5 |
#
# **Labels**
#
# | I want to… | Do this | § |
# |---|---|---|
# | Change the number format (sexagesimal, decimal, CASA…) | `format_ticklabels(style=…)` | 2 |
# | Change the separators between fields | `format_ticklabels(lon_sep=, lat_sep=)` | 2 |
# | Restyle only one axis | `format_ticklabels(which='lon'/'lat')`; overlay: `lon_label_kwargs=`/`lat_label_kwargs=` | 2, 5 |
# | Make labels bigger / smaller | `fontsize=`; auto-fit with `auto_fontsize=True` or `auto_size_ticklabels(ax)` | 6 |
# | Fix crowded / dropped labels | shorten them (`style='compact'`, or `simplify=True` on astropy ≥ 7) | 2, 6 |
# | Label offsets from a reference (arcsec / mas) | `apply_offset_ticks(unit=…)`, `apply_anchored_offset(...)` | 4 |
# | Move labels inside the frame or onto a parallel | `add_overlay_ticks(lon_at=, lat_at=)` | 5 |
# | Label an all-sky projection boundary | `apply_boundary_labels(orient=…)` | 5 |
# | Rotate labels along / across a curve | `tick_rotation=` at build, or `rotate=` in `label_kwargs` | 5 |
# | Set the axis-label *text* | `ax.coords[i].set_axislabel("Right Ascension")` | 5 |
# | Put arbitrary text at arbitrary ticks | `set_ticks(values=)` + `set_ticklabel_visible(False)` + your own `annotate` | 2 |
#
# **Legibility and overall look**
#
# | I want to… | Do this | § |
# |---|---|---|
# | Keep grid / labels / ticks readable over an image | `stroke_lw=` + `stroke_color=` on `style_grid`, `format_ticklabels`, `style_wcs_axes` | 3 |
# | Keep the *frame and its ticks* readable over an image | `apply_frame_stroke(ax)` | 3 |
# | Restyle a whole figure (or session) at once | `sph.set_style(base=, theme=, palette=)`, or `with sph.style_context(...)` | 7 |
# | Coordinate the decoration colors as a set | `style_annotation(ax, name)` — returns the color dict | 7 |
# | Put a figure on a dark page | `sph.set_style(theme='dark_sky', palette='nightcap')` | 7 |
# | Put a companion data plot on a log / symmetric-log axis | `ax.set_yscale('log')`; pysymlog `'symmetriclog'` for signed residuals | 9 |

# %% [markdown]
# ## 11. Where to go next
#
# You can now take a bare frame and control exactly how it **reads** — tick-label
# formats, gridlines, offset/anchored ticks for zoomed fields, edge-vs-in-frame
# placement, and label sizing — and (Section 7) how it **looks**. The two steps that pick
# up right where this tutorial's scope ends:
#
# | If you want to... | Go to |
# |---|---|
# | Draw a *second* coordinate system's grid over a frame — a galactic graticule on an equatorial map | [Overlay Coordinate Grids](overlay_grids.ipynb) |
# | Mark up the *data itself* — rulers, scale bars, compass roses, beams, reticles, sky vectors | [Annotations & Overlays](annotations.ipynb) |
# | Choose a projection on purpose and reproject imagery onto it | [A Tour of Projections](projections.ipynb) |
# | Display and stretch science FITS images | [FITS Images & Quicklook](fits_images.ipynb) |
# | Draw globes and tilted planets | [Globe & Planet Plotting](globe_plots.ipynb) |
# | Drag-rotate and explore these frames interactively | [Interactive Plotting (plotly)](interactive_plotly.ipynb) |
#
# The reference companions to this material are the
# [Ticks, grids & labels](../guide/ticks.md) and [Styling & themes](../guide/styling.md)
# guide pages.
