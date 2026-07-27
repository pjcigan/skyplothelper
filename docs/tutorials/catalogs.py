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
# # Catalogs — Querying, Plotting and Searching
#
# Almost every sky figure starts from a *catalog* — a table of positions with
# science attached. This notebook is the complete journey of that table:
#
# - **Plot it** — `plot_catalog()` puts any table on any frame in one call, with
#   marker size, color, and labels carrying its columns ([§1](#1.-Plotting-a-catalog)).
# - **Key it** — `MultiLegend` builds a clean off-frame legend for the several
#   dimensions a plot encodes at once ([§2](#2.-Legends-for-multiple-dimensions)).
# - **Get it** — resolve names to coordinates and pull catalogs from SIMBAD, NED,
#   and VizieR ([§3](#3.-Getting-catalogs)).
# - **Image it** — fetch survey cutouts from SkyView and HiPS to sit under your
#   points ([§4](#4.-Cutouts-under-your-data)).
# - **Search it** — cone, region, and crossmatch filters that take a catalog in
#   and hand the matching catalog back
#   ([§5](#5.-Searching,-filtering-and-matching)).
# - **Use it** — hand the survivors to an observation planner ([§6](#6.-Planning-an-observation)),
#   then put the whole pipeline together on real clusters ([§7](#7.-Putting-it-together)).
#
# Each section answers the two questions you actually have: *how do I show my
# data this way?* and *how do I adjust it?*
#
# > **A note on the network.** The query sections talk to live services
# > (astroquery — install the `query` extra). Every query in this notebook is
# > wrapped in a tiny cache-fallback helper, so it executes and renders even
# > offline, from small cached copies committed under
# > `examples/data/query_cache/`.
#
# ## Contents
#
# 1. [Plotting a catalog](#1.-Plotting-a-catalog)
# 2. [Legends for multiple dimensions](#2.-Legends-for-multiple-dimensions)
# 3. [Getting catalogs](#3.-Getting-catalogs)
# 4. [Cutouts under your data](#4.-Cutouts-under-your-data)
# 5. [Searching, filtering and matching](#5.-Searching,-filtering-and-matching)
# 6. [Planning an observation](#6.-Planning-an-observation)
# 7. [Putting it together](#7.-Putting-it-together)
# 8. [Where to go next](#8.-Where-to-go-next)

# %%
import warnings
from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

import skyplothelper as sph

# base='structural' tightens only the frame/tick *geometry* — it leaves colors and
# fonts to the docs light/dark theme, so it composes with the dark-figure pass
# (which sets a theme on top).
sph.set_style(base="structural")

# Data series colors come from the `uranometria` cycle palette (mode 'dual': its
# muted tones read on both light and dark pages), driven through the mpl cycle plus
# a few descriptive handles for deliberate picks.
sph.set_palette("uranometria")
_CYC = sph.CYCLE_PALETTES["uranometria"]["colors"]
RC = {
    "blue":  _CYC[0],
    "tan":   _CYC[1],
    "gold":  _CYC[2],
    "gray":  _CYC[3],
    "green": _CYC[4],
    "rust":  _CYC[5],
    "mauve": _CYC[6],
}


# Theme-adaptive decoration colors: read the active page background and pick the
# matching annotation palette, so one code path serves the light and dark renders.
def page_is_dark():
    r, g, b = to_rgb(mpl.rcParams["figure.facecolor"])
    return (0.299 * r + 0.587 * g + 0.114 * b) < 0.5


IS_DARK = page_is_dark()
PAL = sph.ANNOTATION_PALETTES["dark" if IS_DARK else "publication"]
# MultiLegend's palette name, mode-aware — a dark legend box on the dark docs
# page, a light one on the light page (§2).
LEG_PAL = "dark" if IS_DARK else "publication"

# %% [markdown]
# Our companion for the whole notebook is the **Messier catalog** — the 110
# clusters, nebulae, and galaxies of Charles Messier's 18th-century list, bundled
# with the examples as a small CSV. It is real, recognizable, and conveniently
# imperfect (about half the magnitude entries are missing — like a real catalog):

# %%
messier = pd.read_csv("../../examples/data/messier.csv")
print(len(messier), "objects; columns:", list(messier.columns))
messier.head(4)

# %% [markdown]
# And here is where we're going — that five-column table, fully dressed: object
# families in color, brightness in marker size, the galactic plane for context,
# and the common names labeled. Everything in this figure is built from parts
# the next sections teach one at a time:

# %%
# fig-slug: opener
FAMILIES = {
    "Star clusters": (["OpenCluster", "GlobCluster"], RC["gold"]),
    "Nebulae": (["PlanetaryNeb", "HIIReg", "SNRemnant", "RefNeb", "Association"],
                RC["green"]),
    "Galaxies & AGN": (["Galaxy", "AGN", "Seyfert", "Seyfert2", "LINER",
                        "GtowardsGroup", "GinPair", "GtowardsCl", "StarburstG",
                        "HIIG"], RC["blue"]),
    "Other": (["Unknown", "Inexistent"], RC["gray"]),
}

# Brightness → marker area (flux-like), with the missing vmags parked small.
flux = 10 ** (-0.4 * messier["vmag"].fillna(messier["vmag"].max()))
messier["ssize"] = 8 + 210 * (flux - flux.min()) / (flux.max() - flux.min())

fig, ax = sph.allsky_figure(projection="AIT", center=180, frame="ICRS",
                            figsize=(11, 5.5))
for fam, (otypes, col) in FAMILIES.items():
    sub = messier[messier["otype"].isin(otypes)]
    sph.plot_catalog(ax, sub, color=col, s=np.asarray(sub["ssize"]), alpha=0.85,
                     edgecolors=PAL["frame"], linewidths=0.3,
                     label=f"{fam} ({len(sub)})")
sph.add_plane_overlay(ax, plane="galactic", color=PAL["grid"], lw=1.0, ls=":",
                      alpha=0.8)
icons = messier[messier["name"].isin(["M31", "M42", "M45", "M87", "M13"])]
sph.plot_catalog(ax, icons, s=0, color=PAL["text"],
                 label_col="name", label_fontsize=9, label_offset=(7, 6))
sph.format_ticklabels(ax, style="allsky_hours")
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax.set_title("The Messier catalog — five columns, one finished sky map")
plt.show()

# %% [markdown]
# ## 1. Plotting a catalog
#
# ### The one call
#
# `plot_catalog()` puts a table of sources onto any skyplothelper frame in one
# call. It takes the catalog *as you have it* — an astropy `Table`, a pandas
# `DataFrame`, a plain dict of arrays, or a bare `(lon, lat)` tuple — and tries to find the
# coordinate columns by itself, checking the common spellings (`ra`, `RAJ2000`,
# `ra_deg`, `_RAJ2000`, `l`, `GLON`, ...) when the defaults aren't present. Our
# CSV's `ra_deg`/`dec_deg` are on that list, so there is nothing to configure:

# %%
# fig-slug: one-call
fig, ax = sph.allsky_figure(projection="AIT", center=180, frame="ICRS",
                            figsize=(9, 4.5))
sph.plot_catalog(ax, messier, color=RC["blue"])
# Hours-only longitude labels — the clean all-sky read (tick formatting is the
# Decorating Frames tutorial's territory; this one-liner is all we need here).
sph.format_ticklabels(ax, style="allsky_hours")
ax.set_title("The Messier catalog, one call: plot_catalog(ax, messier)")
plt.show()

# %% [markdown]
# All of these are the same plot — pick whichever container your data already
# lives in:
#
# ```python
# sph.plot_catalog(ax, dataframe)                          # pandas DataFrame
# sph.plot_catalog(ax, Table.read("catalog.fits"))         # astropy Table
# sph.plot_catalog(ax, {"ra": ra, "dec": dec})             # dict of arrays
# sph.plot_catalog(ax, (ra, dec))                          # bare arrays
# sph.plot_catalog(ax, (coords.ra.deg, coords.dec.deg))    # from a SkyCoord
# ```
#
# If auto-detection guesses wrong (or your columns have exotic names), you can
# specify them manually with `ra_col=`/`dec_col=` — or the frame-neutral
# `lon_col=`/`lat_col=`, which
# we'll want for galactic data below. Everything beyond the coordinates is
# standard `ax.scatter` styling: `marker`, `color`, `s`, `alpha`, and any extra
# scatter kwargs pass straight through.
#
# ### Encoding a column in marker size
#
# The catalog's columns are where the science is, and `sizeby=` maps one of them
# onto marker area. But *how* the values map matters — and magnitudes are the
# classic gotcha. Passed naively, brighter objects (smaller magnitudes) get the
# *smallest* markers. `size_scale=` reshapes the values before they are mapped
# into the `smin..smax` range: `'sqrt'` and `'log'` tame skewed columns, and a
# **callable** handles anything custom — here, converting magnitudes to relative
# flux so the bright showpieces pop:

# %%
# Real catalogs have holes: 48 of the 110 vmag entries are NaN. plot_catalog
# quietly gives those markers NaN size (matplotlib then drops them), so it won't
# crash — but it's clearer to be explicit about what you're plotting:
bright = messier.dropna(subset=["vmag"])
print(f"{len(bright)} of {len(messier)} objects have a vmag")

# %%
# fig-slug: sizeby-scales
fig = plt.figure(figsize=(12.5, 4.2))

ax1 = sph.make_wcs_frame(121, "AIT", center=180, frame="ICRS", fig=fig)
sph.plot_catalog(ax1, bright, sizeby="vmag", color=RC["tan"])
ax1.set_title("sizeby='vmag' — naive: faint objects get the big markers",
              fontsize=10)

ax2 = sph.make_wcs_frame(122, "AIT", center=180, frame="ICRS", fig=fig)
sph.plot_catalog(ax2, bright, sizeby="vmag",
                 size_scale=lambda mag: 10 ** (-0.4 * mag),  # mag → relative flux
                 smin=6, smax=260, color=RC["gold"])
ax2.set_title("size_scale=<callable> — flux-proportional: the bright ones pop",
              fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# > **Note:** `smin`/`smax` always define the output size range; `size_scale` only
# > reshapes how the values spread across it. `'log'` clips non-positive values to
# > the smallest positive one (with a warning), so a column of magnitudes — which
# > can legitimately be negative — is exactly where a callable shines.
#
# ### Color, the colorbar, and the size legend
#
# `colorby=` maps a second column through a colormap, with its own scale control
# (`color_scale=`, taking `'linear'`/`'sqrt'`/`'log'` or a matplotlib `Normalize`)
# plus `vmin`/`vmax` limits and `cmap_range=` to trim a colormap's illegible ends.
# `cmap=` takes any matplotlib colormap — including skyplothelper's own bundled
# family, registered under `sph.*` names (the [Styling tutorial](styling.ipynb)
# tours them); `sph.sunset` here runs deep blue through red to gold, and its
# reversed form puts the warmth on the bright end.
# Set `cbar=True` and the return value grows a colorbar: you get a `CatalogPlot`
# named tuple — unpack it as `sc, cb = ...` or keep attribute access
# (`result.scatter`, `result.colorbar`). `cbar_format`/`cbar_ticks` style the bar,
# and `size_legend=True` adds a marker-size key labeled in the column's own units:

# %%
# fig-slug: colorby-colorbar
fig, ax = sph.allsky_figure(projection="AIT", center=180, frame="ICRS",
                            figsize=(10, 5))
sc, cb = sph.plot_catalog(
    ax, bright,
    sizeby="vmag", size_scale=lambda mag: 10 ** (-0.4 * mag), smin=6, smax=260,
    colorby="vmag", cmap="sph.sunset_r", cmap_range=(0.08, 0.92),
    cbar=True, cbar_label="V magnitude", cbar_format="{x:.0f}",
    size_legend=True, size_legend_num=3,
    size_legend_kwargs=dict(loc="lower right", title="V mag"),
    edgecolors=PAL["frame"], linewidths=0.3)
sph.format_ticklabels(ax, style="allsky_hours")
ax.set_title("Size and color both carry vmag — reversed colormap, "
             "so bright = big and warm")
plt.show()

# %% [markdown]
# ### Labels — a finder chart
#
# `label_col=` writes a name next to each marker (`label_fontsize` and
# `label_offset` place the text). All 110 labels at once would be a jumbled mess, so label a
# *slice*: `plot_catalog` is happy with any subset of your table, and slicing the
# table is the natural way to control what gets labeled. Here is the heart of the
# Virgo Cluster with its Messier members named — a finder chart in a few lines.
# One practical wrinkle: `label_offset` is a single global setting, so two objects
# closer than a label width (M84/M86 on Markarian's Chain, the M59/M60 pair) would
# collide. The same slicing idea solves it — split the crowded ones into a second
# call with the offset on the other side:

# %%
# fig-slug: labels-finder
virgo = sph.SKY_POSITIONS["virgo_cluster"]  # bundled named positions (SkyCoord)
in_field = messier[
    (np.abs(messier["ra_deg"] - virgo.ra.deg) < 7)
    & (np.abs(messier["dec_deg"] - virgo.dec.deg) < 6)
]
crowd = in_field["name"].isin(["M84", "M59"])  # west member of each close pair

fig = plt.figure(figsize=(7.5, 7))
ax = sph.make_wcs_frame(111, "TAN", center=(virgo.ra.deg, virgo.dec.deg),
                        fov_deg=13, fig=fig)
for subset, offset in [(in_field[~crowd], (7, 5)), (in_field[crowd], (-32, -14))]:
    sph.plot_catalog(ax, subset, color=PAL["accent"], s=55,
                     edgecolors=PAL["frame"], linewidths=0.4,
                     label_col="name", label_fontsize=9, label_offset=offset)
sph.format_ticklabels(ax, style="compact")  # hh:mm / dd:mm — right for a wide field
ax.set_title("Messier objects across the Virgo Cluster core")
plt.show()

# %% [markdown]
# ### A catalog in another coordinate frame
#
# Catalogs don't always arrive in the frame you're plotting. `frame=` tells
# `plot_catalog` what the *input* coordinates are, and the points are converted
# onto the map — so a galactic `l`/`b` table lands correctly on an equatorial
# frame with no manual `SkyCoord` juggling. Without it, the numbers are taken
# at face value, and the catalog lands wherever those numbers happen to point:

# %%
# A galactic-coordinate version of Messier, as a survey pipeline might supply it.
gal = SkyCoord(ra=messier["ra_deg"], dec=messier["dec_deg"], unit="deg").galactic
messier_gal = pd.DataFrame({"name": messier["name"],
                            "l": gal.l.deg, "b": gal.b.deg})

# %%
# fig-slug: frame-conversion
fig = plt.figure(figsize=(12.5, 4.2))

ax1 = sph.make_wcs_frame(121, "AIT", center=180, frame="ICRS", fig=fig)
sph.plot_catalog(ax1, messier_gal, lon_col="l", lat_col="b", color=RC["rust"])
ax1.set_title("l/b numbers read as RA/Dec — every point in the wrong place",
              fontsize=10)

ax2 = sph.make_wcs_frame(122, "AIT", center=180, frame="ICRS", fig=fig)
sph.plot_catalog(ax2, messier_gal, lon_col="l", lat_col="b", frame="galactic",
                 color=RC["green"], s=42, alpha=0.85)
sph.plot_catalog(ax2, messier, color=PAL["text"], marker="+", s=12,
                 linewidths=0.8, alpha=0.9)
ax2.set_title("frame='galactic' — converted onto the equatorial map\n"
              "(+ = the original equatorial catalog, for proof)", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# The `lon_col`/`lat_col` aliases keep the call frame-neutral (asking for
# `ra_col='l'` works, but reads oddly), and `unit=` handles catalogs in hourangle
# or radians the same way.
#
# ### Categories, not numbers
#
# `colorby=` maps *numeric* columns. For a categorical column — object type,
# survey of origin, quality flag — loop over the groups and give each its own
# color and legend entry; the opener figure's `FAMILIES` dict is exactly this
# recipe, collapsing the catalog's 19 raw `otype` values into four readable
# groups. To make the payoff unmistakable, let's also swap the *frame*: on a
# galactic-coordinate map (with `frame='icrs'` converting our equatorial columns
# on the way in — the previous section's trick, run in the other direction), the
# object families sort themselves:

# %%
# fig-slug: categorical-otypes
fig, ax = sph.allsky_figure(projection="AIT", center=0, frame="Galactic",
                            figsize=(10, 5))
for fam, (otypes, col) in FAMILIES.items():
    sub = messier[messier["otype"].isin(otypes)]
    sph.plot_catalog(ax, sub, frame="icrs", color=col, s=34, alpha=0.85,
                     edgecolors=PAL["frame"], linewidths=0.3,
                     label=f"{fam} ({len(sub)})")
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax.set_title("One plot_catalog call per family, on a galactic frame — "
             "the families sort themselves")
plt.show()

# %% [markdown]
# The astronomy reads straight off the figure: the star clusters pin to the
# galactic midplane and bulge (they live in the disk), while the galaxies fill
# the high latitudes and vanish near b = 0 — Messier could only find galaxies
# where the Milky Way's dust wasn't in the way.
#
# ## 2. Legends for multiple dimensions
#
# Section 1 packed several dimensions into a single scatter — marker size,
# color, and category all at once. The companion problem is letting a reader
# *decode* it: each encoded channel needs its own key, a shape key has to read as
# distinct from a color key, and the whole legend is often best placed *off* the
# plot rather than over the data. A single `ax.legend()` fights all three — it
# flattens everything into one flat list of proxy artists.
#
# `sph.MultiLegend` builds the alternative: **a stack of per-channel blocks,
# placeable anywhere, including the figure margins.** It is built on matplotlib's
# `OffsetBox` machinery and works on *any* `Axes` — a light curve, a
# color–magnitude diagram, a scatter of galaxy properties — not just a sky map.
# Whenever a plot carries several encoded dimensions at once, this is the tool.
#
# We will learn the mechanics on the familiar **Messier** catalog from Section 1
# (object family in color, brightness in size), then close by scaling up to a
# genuinely busy all-sky map — the full **USNO 2025a** VLBI global solution.

# %%
# The how-it-works demos reuse Section 1's Messier catalog (small and familiar):
# grouped object family for the color channel, apparent brightness for size.
FAM_COLORS = {fam: col for fam, (otypes, col) in FAMILIES.items()}
messier["family"] = "Other"
for _fam, (_otypes, _c) in FAMILIES.items():
    messier.loc[messier.otype.isin(_otypes), "family"] = _fam
mbright = messier.dropna(subset=["vmag"]).copy()             # 62 with a magnitude

# The capstone uses the full USNO 2025a VLBI solution; its formal-error classes
# take an ordered cool→warm ramp straight out of the `uranometria` cycle we are
# already using (mode 'dual', so it reads on both docs themes).
usno = pd.read_csv("../../examples/data/usno2025a_vlbi.csv")
ERR_COLORS = [RC["blue"], RC["green"], RC["gold"], RC["rust"]]   # good → poor
ERR_LABELS = ["< 0.1", "0.1–0.3", "0.3–1", "> 1"]
usno["ebin"] = np.digitize(usno["err_mas"], [0.1, 0.3, 1.0])
SIZE_VLIM = (1, float(np.percentile(usno.n_delays, 99)))     # clip the long tail


def messier_scatter(ax, *, label_families=False, tick_style="allsky_hours"):
    """One plot_catalog per Messier object family (categorical color), sized by
    apparent brightness (a mag→flux callable, so bright = big). Returns the last
    result — it carries the size scaling for the legend's size block."""
    cp = None
    for fam, col in FAM_COLORS.items():
        s = mbright[mbright.family == fam]
        if not len(s):
            continue
        cp = sph.plot_catalog(
            ax, {"ra": s.ra_deg, "dec": s.dec_deg, "mag": s.vmag},
            sizeby="mag", size_scale=lambda m: 10 ** (-0.4 * m), smin=14, smax=340,
            color=col, alpha=0.85, edgecolors=PAL["frame"], linewidths=0.3,
            frame="icrs", label=(fam if label_families else None)) or cp
    sph.format_ticklabels(ax, style=tick_style, fontsize=None if
                          tick_style != "minimal" else 7)
    return cp


def usno_allsky(ax):
    """Lay the USNO catalog down as one plot_catalog per (shape, error) subset —
    a scatter carries a single marker and color — all sharing `size_vlim` so
    equal observation counts render equal-sized. Returns a result for the size key.

    The defining sources ride on top with a *light* edge, which lifts them out of
    the dense field far better than a dark outline; a 0.6-power size scale opens
    up more dynamic range than sqrt across five decades of observation counts.
    """
    result = None
    for is_def, mk, z, ew, ec in [(0, "o", 3, 0.2, "0.3"), (1, "*", 6, 0.7, "0.9")]:
        for b in range(4):
            s = usno[(usno.defining == is_def) & (usno.ebin == b)]
            if not len(s):
                continue
            result = sph.plot_catalog(
                ax, {"ra": s.ra_deg, "dec": s.dec_deg, "nobs": s.n_delays},
                sizeby="nobs", size_vlim=SIZE_VLIM, size_scale=lambda x: x ** 0.6,
                smin=4, smax=250, marker=mk, color=ERR_COLORS[b], alpha=0.72,
                edgecolors=ec, linewidths=ew, frame="icrs", zorder=z) or result
    return result


print(f"Messier demo set: {len(mbright)} objects with magnitudes across "
      f"{messier.family.nunique()} families; USNO capstone: {len(usno)} sources")

# %% [markdown]
# Here is the problem, and the fix, in one look. This map of the brighter Messier
# objects encodes two dimensions — marker **color** (object family) and **size**
# (apparent brightness). A bare `ax.legend()` (top) can only list the artists it
# is handed: it names the families, but the *brightness* dimension has no key at
# all. One `MultiLegend` (bottom) keys each channel in its own block, tucked off
# to the side where it covers nothing:

# %%
# fig-slug: legend-opener
fig = plt.figure(figsize=(11, 8.2))
fig.subplots_adjust(hspace=0.32, right=0.83)

# Before: label the families and call ax.legend() — the naive attempt.
ax1 = sph.make_wcs_frame(211, "AIT", center=180, frame="ICRS", fig=fig)
messier_scatter(ax1, label_families=True)
ax1.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax1.set_title("A bare ax.legend() — families listed, brightness has no key",
              fontsize=11)

# After: one MultiLegend, one block per channel.
ax2 = sph.make_wcs_frame(212, "AIT", center=180, frame="ICRS", fig=fig)
cp = messier_scatter(ax2)
(sph.MultiLegend(ax2, loc="outside right", palette=LEG_PAL)
    .add_color("Object family", FAM_COLORS, swatch="marker")
    .add_size_from(cp, values=[4, 6, 8], title="V mag", fmt=".0f")
    .draw())
ax2.set_title("One MultiLegend — a color block and a size block, both keyed",
              fontsize=11)
plt.show()

# %% [markdown]
# Everything below builds that bottom panel, one idea at a time.
#
# ### The channel-block model
#
# A `MultiLegend` is **one block per visual channel**. You attach it to an axes,
# add blocks with fluent `add_*` calls (each returns the legend, so they chain),
# and finish with `.draw()`. The two channels above are among the most useful:
#
# - `add_color(title, {label: color})` keys a categorical color.
# - `add_size_from(result)` reads a `plot_catalog` result's *exact* size scaling
#   and turns it into a graduated key — so the swatches match the plotted markers.
#
# Because color here is a category, we plot one subset per family (a scatter
# carries a single color); any one call's return value remembers the size scaling
# for `add_size_from` to read back. The `messier_scatter` helper (top of the
# section) does exactly that loop.

# %%
# fig-slug: legend-model
fig, ax = sph.allsky_figure(projection="AIT", center=180, frame="ICRS",
                            figsize=(10, 5.2))
cp = messier_scatter(ax)
(sph.MultiLegend(ax, loc="lower left", palette=LEG_PAL)
    .add_color("Object family", FAM_COLORS, swatch="marker")
    .add_size_from(cp, values=[4, 6, 8], title="V mag", fmt=".0f")
    .draw())
ax.set_title("Two channels, two blocks — color = family, size = brightness")
plt.show()

# %% [markdown]
# That is the whole model: `add_color` + `add_size_from`, stacked and drawn.
# `add_color` here uses `swatch="marker"` (colored circles, since the Messier
# points *are* circles); when color stands on its own as an abstract label, the
# default `swatch="patch"` draws neutral chips instead — the niceties section
# returns to this. The size key values (`values=[4, 6, 8]` magnitudes) are our
# choice; omit `values=` and `add_size_from` picks round steps itself, always
# reproducing the plot's exact sizes.
#
# ### The channels you can key
#
# Size and color are two of a whole registry of channels — one `add_*` per
# visual property. Each maps a data meaning to how a swatch is drawn:
#
# | Channel | Builder | Swatch |
# |---|---|---|
# | color (category) | `add_color` | color chip · marker · line (`swatch=`) |
# | shape | `add_shape` | marker glyph (auto-neutral beside a color block) |
# | size | `add_size` / `add_size_from` | graduated markers |
# | edge color | `add_edge` | ringed marker |
# | fill (open/solid) | `add_fill` | marker |
# | hatch | `add_fill(kind="patch")` | hatched patch |
# | alpha | `add_alpha` | graduated-opacity markers |
# | orientation | `add_orientation` | rotated marker |
# | line (style / width) | `add_line` (`vary=`) | dashed / weighted segment |
# | region | `add_region` | translucent footprint patch |
# | glyph | `add_glyph` | a named sph glyph (reticle shapes, …) |
# | colorbar | `add_colorbar` | continuous gradient strip |
# | text / custom | `add_text` / `add_custom` | free note / any matplotlib artist |
#
# `entries` is an ordered `{label: value}` dict (or `(label, value)` pairs); the
# value is whatever the channel varies — a color, a marker, a linestyle. Pass a
# full style `dict` as the value to vary several properties together, or drop to
# the generic `add_block(LegendBlock(...))` for a fully custom swatch. (Each
# `add_*` also has a standalone class — `ColorBlock`, `ShapeBlock`, `SizeBlock`,
# … — you can build once and pass to `add_block` to reuse across figures;
# `SizeBlock.from_catalog(result)` is what `add_size_from` calls under the hood.)
# Here is the whole registry in two rows — marker/line channels on top, the
# specialty swatch kinds below:

# %%
# fig-slug: legend-channels
_A = sph.CYCLE_PALETTES["atlas"]["colors"]                   # in-theme swatch colors
fig, (axA, axB) = plt.subplots(2, 1, figsize=(12, 4.6))
for a in (axA, axB):
    a.axis("off")

# Row 1 — channels that vary a marker or line property.
(sph.MultiLegend(axA, loc="center", orientation="horizontal", block_sep=18,
                 palette=LEG_PAL)
    .add_color("Color", {"A": _A[0], "B": _A[1], "C": _A[2]}, swatch="marker")
    .add_shape("Shape", {"disk": "o", "gal": "D", "star": "*"})
    .add_size("Size", values=[1, 100, 10000], smin=8, smax=230, scale="sqrt",
              fmt=".0f")
    .add_edge("Edge", {"secure": _A[2], "flagged": _A[3]})
    .add_fill("Fill", {"detected": "filled", "limit": "open"})
    .add_alpha("Alpha", values=[1, 5, 20], fmt=".0f")
    .add_orientation("Angle", {"0°": 0, "30°": 30, "60°": 60})
    .add_line("Line", {"fit": "--", "prior": ":"})
    .draw())

# Row 2 — the specialty swatch kinds, including sph reticle glyphs, a continuous
# colorbar strip (an sph colormap), and any matplotlib artist you hand in.
_star = Line2D([0], [0], marker=(6, 1, 0), markersize=12, linestyle="none",
               markerfacecolor=_A[1], markeredgecolor=PAL["frame"])
(sph.MultiLegend(axB, loc="center", orientation="horizontal", block_sep=18,
                 palette=LEG_PAL)
    .add_fill("Hatch", {"DES": "///", "LSST": "xxx"}, kind="patch", color=_A[0])
    .add_region("Region", {"footprint": dict(fc=_A[0], ec=_A[0], alpha=0.35),
                           "mask": dict(fc=_A[3], ec=_A[3], hatch="//")})
    .add_glyph("Glyph", {"target": "reticle_circle", "mark": "crosshair"})
    .add_colorbar("Redshift", cmap="sph.deepsky", vmin=0, vmax=2, length=90,
                  fmt=".1f")
    .add_text("Text", ["dashed = model"])
    .add_custom("Custom", {"my marker": _star})
    .draw())
axA.set_title("Channels that vary a marker or line", fontsize=10, y=0.97)
axB.set_title("Specialty swatch kinds — hatch, region, sph glyphs, colorbar, "
              "text, any custom artist", fontsize=10, y=0.97)
plt.show()

# %% [markdown]
# > **Note:** each block is independent, so mixing many is fine, but a *legible*
# > legend keys only the channels your plot actually encodes. The showcase above
# > is a catalog of what's available, not a template to fill. The `add_glyph`
# > swatches are the same reticle shapes the [Annotations](annotations.ipynb)
# > tutorial draws on the sky (`list_glyphs()` names them; `register_glyph()`
# > adds your own), so a
# > reticle legend matches its markers exactly.
#
# ### Placing the legend
#
# The `loc=` argument takes the nine familiar inside-the-axes anchors
# (`"upper left"`, `"lower right"`, `"center"`, …) and, crucially for full-frame
# maps, an **`"outside …"`** family that parks the legend in the figure margin
# where it never covers data:
#
# | `loc=` | Where |
# |---|---|
# | `"lower right"`, `"upper left"`, … | inside the axes (9 anchors) |
# | `"outside right"`, `"outside bottom"`, … | in the figure margin, clear of the axes |
# | `(x, y)` + `coords="axes"/"figure"` | free placement at exact coordinates |
#
# Add `reserve=True` to shrink the axes and *make* margin room (like a colorbar);
# the default overflows into existing whitespace instead. The same three-block
# key, in three positions:

# %%
# fig-slug: legend-placement
def mini_messier(sub_ax):
    """A light Messier scatter for a legend to sit against."""
    for fam, col in FAM_COLORS.items():
        s = messier[messier.family == fam]
        sph.plot_catalog(sub_ax, {"ra": s.ra_deg, "dec": s.dec_deg},
                         color=col, s=10, alpha=0.7, frame="icrs")
    sph.format_ticklabels(sub_ax, style="minimal", fontsize=7)


fig = plt.figure(figsize=(13, 3.6))
for i, loc in enumerate(["lower left", "outside right", "outside bottom"], 1):
    ax = sph.make_wcs_frame(130 + i, "AIT", center=180, frame="ICRS", fig=fig)
    mini_messier(ax)
    horiz = loc == "outside bottom"
    (sph.MultiLegend(ax, loc=loc, palette=LEG_PAL,
                     orientation="horizontal" if horiz else "vertical")
        .add_color("Family", FAM_COLORS, swatch="marker", ncol=2 if horiz else 1)
        .draw())
    ax.set_title(f"loc={loc!r}", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# For a full-frame figure the `"outside"` presets are almost always the right
# call — the map fills its axes edge to edge, so any inside legend lands on
# data. Two independent `MultiLegend`s can sit in different corners for a split
# key.
#
# ### Two ways to key marker size
#
# Marker-size legends come in two flavors, and it's worth knowing which to
# reach for:
#
# - `plot_catalog(..., size_legend=True)` (Section 1) — a **one-argument**,
#   single-channel key drawn straight onto the axes. Perfect when size is the
#   only thing you're keying.
# - `MultiLegend().add_size_from(result)` — the same size scaling as a **block**
#   you can stack with color, shape, and the rest, and place off-frame.
#
# Both read the *exact* scaling off the plotted result, so the swatch sizes
# match the on-plot markers; `add_size_from` just makes size one channel among
# several. Side by side on the same data:

# %%
# fig-slug: legend-size-tools
fig = plt.figure(figsize=(12.5, 4.6))

# Left: the one-argument size_legend from Section 1.
ax1 = sph.make_wcs_frame(121, "AIT", center=180, frame="ICRS", fig=fig)
sph.plot_catalog(ax1, {"ra": mbright.ra_deg, "dec": mbright.dec_deg,
                       "mag": mbright.vmag},
                 sizeby="mag", size_scale=lambda m: 10 ** (-0.4 * m),
                 smin=14, smax=340, color=RC["blue"], alpha=0.75,
                 edgecolors=PAL["frame"], linewidths=0.3, frame="icrs",
                 size_legend=True, size_legend_num=4,
                 size_legend_kwargs=dict(loc="lower left", title="V mag"))
sph.format_ticklabels(ax1, style="minimal", fontsize=7)
ax1.set_title("plot_catalog(size_legend=True) — one channel, one call",
              fontsize=10)

# Right: the same size scaling as a stackable block alongside the color key.
ax2 = sph.make_wcs_frame(122, "AIT", center=180, frame="ICRS", fig=fig)
cp2 = messier_scatter(ax2, tick_style="minimal")
(sph.MultiLegend(ax2, loc="lower left", palette=LEG_PAL)
    .add_size_from(cp2, values=[4, 6, 8], title="V mag", fmt=".0f")
    .add_color("Family", FAM_COLORS, swatch="marker")
    .draw())
ax2.set_title("add_size_from(result) — size as one block among several",
              fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### What MultiLegend handles for you
#
# Two conventions do quiet work toward a legend that reads correctly. **First,
# color-swatch shape.** A colored *circle* in the key silently says "color
# applies to circular markers" — wrong when color is an independent label
# (a survey, a class) that isn't tied to any one shape. So `add_color` defaults
# to `swatch="patch"` (neutral color chips) and you opt into `swatch="marker"`
# only when color truly rides a marker — as it does for our all-circle Messier
# points. **Second, neutral shape swatches.** When a `shape` block shares a
# legend with a `color` block, its markers render in a neutral gray
# automatically, so "shape means one thing, color means another" reads at a
# glance instead of the two encodings bleeding together:

# %%
# fig-slug: legend-niceties
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
for a in axes:
    a.axis("off")

# Left: an abstract category not tied to a shape → neutral patch chips (default).
(sph.MultiLegend(axes[0], loc="center", palette=LEG_PAL)
    .add_color("Survey", {"VLBA": RC["blue"], "EVN": RC["gold"], "LBA": RC["green"]})
    .add_shape("Band", {"S/X": "o", "K": "D", "Q": "^"})
    .draw())
axes[0].set_title('swatch="patch" (default) + auto-neutral shapes', fontsize=10)

# Right: color riding the circular data markers → colored circles.
(sph.MultiLegend(axes[1], loc="center", palette=LEG_PAL)
    .add_color("Object family", FAM_COLORS, swatch="marker")
    .add_shape("Band", {"S/X": "o", "K": "D", "Q": "^"})
    .draw())
axes[1].set_title('swatch="marker" — color on the markers the data use',
                  fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# Text and frame colors also follow the active theme — the `palette=`
# argument (`"publication"`, `"dark"`, …) sets them, and `stroke_color=` /
# `stroke_lw=` add the same legible outline every sph decoration offers, so a
# legend parked over a busy map stays readable. (Working interactively? The same
# blocks feed a plotly figure through `sphpl.add_legend` — see the
# [Interactive Plotting](interactive_plotly.ipynb) tutorial.)
#
# ### Putting it together
#
# Now scale up. Everything so far used the compact Messier list; the real payoff
# is a *busy* map, where an off-frame multi-block key keeps things organized. The **USNO
# 2025a** VLBI global solution has ~5,800 radio sources, and we key three
# channels at once: marker **size** = delay observations (a source's astrometric
# weight), **color** = formal position error (an ordered good→poor ramp from the
# `uranometria` cycle), **shape** = ICRF3 defining vs. other. The `usno_allsky`
# helper lays it down — one `plot_catalog` per shape × error subset, all sharing
# `size_vlim` so equal observation counts stay equal-sized — then one off-frame
# `MultiLegend` keys everything. A dense field like this rewards a little frame
# care, so we also lift the graticule and move the RA labels inside it:

# %%
# fig-slug: legend-capstone
fig = plt.figure(figsize=(12, 7))
ax = sph.make_wcs_frame(111, "AIT", center=180, frame="ICRS", fig=fig,
                        lon_spacing=30)
fig.subplots_adjust(bottom=0.2)

# Frame treatment for a dense field: inward ticks, and a stroke on the tick
# labels in the *page's* background color so they stay legible over the markers.
sph.style_wcs_axes(ax, direction="in", stroke_lw=1.8,
                   stroke_color=PAL["fig_bg"] if IS_DARK else "white")
cap = usno_allsky(ax)

# A visible graticule, plus the RA labels moved *inside* the frame along the
# −60° parallel — an all-sky map has no straight edge to hang them on, so
# `add_overlay_ticks` places them on a gridline instead (`suppress_default`
# turns the default longitude labels off).
sph.style_grid(ax, color="0.4", alpha=0.9, lw=0.7, ls="-")
sph.add_overlay_ticks(
    ax, lon_at="lat=-60", lat_at=None, suppress_default="lon",
    lon_vals=np.arange(0, 360, 30),
    label_kwargs={"rotate": "horizontal", "sep": "plain", "fontsize": 10,
                  "color": PAL["text"]})

(sph.MultiLegend(ax, loc="outside bottom", orientation="horizontal",
                 block_sep=26, palette=LEG_PAL)
    .add_size_from(cap, values=[1, 10, 100, 1000, 5000, 10000],
                   title="N delays", ncol=2, fmt=".0f")
    .add_color("Formal error (mas)", dict(zip(ERR_LABELS, ERR_COLORS)),
               swatch="marker", ncol=2)
    .add_shape("ICRF3", {"defining": "*", "other": "o"})
    .draw())
ax.set_title("USNO 2025a VLBI global solution — all-sky, three encoded channels",
             y=0.999)
plt.show()

# %% [markdown]
# Everything a reader needs to decode the map sits in one tidy row beneath it:
# the heavily-observed defining sources (large stars) anchor the frame, and the
# color gradient makes the accuracy structure — better near the well-observed
# north, poorer in the sparse far south — legible at a glance. That is the
# payoff of keying each dimension in its own block.
#
# The USNO 2025a solution is public — quarterly VLBI global solutions are
# available from the [USNO](https://crf.usno.navy.mil/quarterly-vlbi-solution),
# and the ICRF3 defining list from the
# [IERS ICRF Product Center](https://hpiers.obspm.fr/icrs-pc/icrf/index.php).
#
# ## 3. Getting catalogs
#
# Everything so far worked on a table we already had. This half of the workflow
# *gets* the data: skyplothelper wraps astroquery for the lookups that punctuate a
# plotting session — turning a name into coordinates, pulling an object's entry,
# fetching a catalog around a position. The wrappers are deliberately thin
# conveniences for plotting workflows (for serious catalog work, reach for
# astroquery itself); they need the `query` extra (`pip install
# skyplothelper[query]`) and, naturally, the network.
#
# > **Note:** *the pattern behind every query cell below.* Each query runs
# > live, then falls back to a small cached copy bundled in
# > `examples/data/query_cache/` — so this notebook renders real results even
# > offline or when a service is down. The helper is a few honest lines; steal
# > it for your own notebooks, and they'll be just as robust.

# %%
QUERY_CACHE = Path("../../examples/data/query_cache")


def _with_timeout(fetch, seconds=60):
    """Run ``fetch()`` but abort if it exceeds ``seconds`` — a hung service
    (a socket stuck with no timeout of its own) shouldn't stall the whole
    notebook. SIGALRM interrupts the blocking call in the kernel's main thread.
    """
    import signal

    def _raise(signum, frame):
        raise TimeoutError(f"query exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        return fetch()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def cached(fetch, name):
    """Run a live query; on any failure (or timeout) fall back to the cache.

    The first successful live run writes the cache file, so the committed
    notebook re-executes offline (and results stay stable between runs).
    """
    path = QUERY_CACHE / name
    try:
        tbl = _with_timeout(fetch)
        if tbl is None:
            raise RuntimeError("empty query result")
        if not path.exists():
            tbl.write(path)
    except Exception as err:
        print(f"live query unavailable ({type(err).__name__}) — using cached {name}")
        tbl = Table.read(path)
    return tbl


def try_resolve(name, fallback_radec):
    """resolve_name with an offline fallback position (the same idea, for coords)."""
    try:
        return _with_timeout(lambda: sph.resolve_name(name), seconds=30)
    except Exception:
        return SkyCoord(*fallback_radec, unit="deg")


def simbad_radec(table):
    """Pull (name, ra_deg, dec_deg) from a SIMBAD result, whatever the schema.

    SIMBAD's columns shift with the astroquery version: the newer TAP interface
    lowercased the names (`MAIN_ID` → `main_id`) and returns RA/Dec in degrees,
    while older releases used uppercase names and sexagesimal strings. This
    normalizes either into plottable degrees.
    """
    col = {c.lower(): c for c in table.colnames}
    name = np.asarray(table[col["main_id"]])
    ra_raw, dec_raw = table[col["ra"]], table[col["dec"]]
    if np.issubdtype(np.asarray(ra_raw).dtype, np.floating):
        return name, np.asarray(ra_raw, float), np.asarray(dec_raw, float)
    sky = SkyCoord(ra_raw, dec_raw, unit=(u.hourangle, u.deg))   # legacy strings
    return name, sky.ra.deg, sky.dec.deg


# %% [markdown]
# ### Names to coordinates
#
# `resolve_name()` turns an object name into a `SkyCoord` via SIMBAD (or
# `service='ned'`) — and the result drops straight into any `center=`:

# %%
m45 = try_resolve("M45", (56.601, 24.114))  # the Pleiades
print(m45)

# %% [markdown]
# For a *list* of targets, `resolve_names()` is the batched form — it returns the
# resolved coordinates *and* the names that failed, so one typo doesn't take down
# the batch (`on_error='warn'` collects failures; `'raise'` stops hard):
#
# ```python
# coords, failed = sph.resolve_names(["M31", "M87", "Sombrero Galaxy", "M999"])
# print(failed)                     # → ['M999'] — handle, don't crash
# ```
#
# And if all you need is a *known* landmark, skip the network entirely:
# `sph.SKY_POSITIONS` is a bundled dict of ~27 named positions (`'m31'`,
# `'virgo_cluster'`, `'galactic_center'`, `'cyg_a'`, ...) as ready-made
# `SkyCoord`s — we've been using it since Section 1.
#
# ### Looking up objects
#
# `query_simbad()` / `query_ned()` fetch an object's database entry — or search a
# *region* when you pass a coordinate plus `radius=`:

# %%
crab = cached(lambda: sph.query_simbad("M1"), "simbad_m1.ecsv")
c_name, c_ra, c_dec = simbad_radec(crab)
pd.DataFrame({"object": c_name, "ra_deg": c_ra.round(4), "dec_deg": c_dec.round(4)})

# %% [markdown]
# > **Note:** SIMBAD's exact columns and coordinate format shift with the
# > astroquery version — the newer TAP interface returns lowercase names and
# > *degrees*, older releases uppercase names and *sexagesimal strings*. The
# > four-line `simbad_radec()` helper above normalizes either into plottable
# > `(name, ra_deg, dec_deg)` — the kind of small defensive wrapper worth
# > keeping when a notebook has to survive a dependency bump. The result drops
# > straight into a `SkyCoord` or `plot_catalog`:

# %%
pleiades = cached(lambda: sph.query_simbad(m45, radius=40 * u.arcmin),
                  "simbad_m45_region.ecsv")
p_name, p_ra, p_dec = simbad_radec(pleiades)
star_coords = SkyCoord(p_ra, p_dec, unit="deg")
print(len(p_name), "SIMBAD objects within 40′ of M45; first:", p_name[0])

# %% [markdown]
# `query_ned()` is the same idea against NED, the NASA/IPAC Extragalactic
# Database — and the two are complementary rather than interchangeable. NED's
# specialty is *extragalactic* data, so its object lookups carry a **redshift
# and velocity** that SIMBAD's basic query leaves out — the deciding factor when
# you're working with galaxies. Here is M87, the Virgo Cluster's central giant
# elliptical (and the star of our capstone):

# %%
m87_ned = cached(lambda: sph.query_ned("M87"), "ned_m87.ecsv")
m87_ned["Object Name", "RA", "DEC", "Type", "Velocity", "Redshift"].to_pandas()

# %% [markdown]
# > **Note:** NED can be slow to answer, and a wide region query may time out
# > when its servers are busy — one more reason the cache-fallback pattern is
# > worth having. For bulk *stellar* catalogs, `search_vizier` (next) is usually
# > the faster road.
#
# ### Fetching a catalog around a position
#
# `search_vizier()` cone-searches any of VizieR's ~25,000 catalogs by identifier —
# with column selection and a row limit. Here: Gaia DR3 around the Pleiades, with
# exactly the columns the plot will use. (Asking VizieR for its *computed*
# `_RAJ2000`/`_DEJ2000` columns is the standard trick for catalogs whose native
# positions are strings or epoch-offset — we'll need it later for the NGC
# catalog.)

# %%
def fetch_gaia(center, radius_arcmin, gmax=16):
    """Gaia DR3 cone: fetch complete, cut client-side to keep things small."""
    t = sph.search_vizier("I/355/gaiadr3", center, radius=radius_arcmin,
                          columns=["RA_ICRS", "DE_ICRS", "Gmag", "BP-RP"],
                          row_limit=-1)
    return t[np.asarray(t["Gmag"], float) < gmax]


gaia = cached(lambda: fetch_gaia(m45, 50), "gaia_m45.ecsv")
print(f"{len(gaia)} Gaia sources with G < 16; columns: {gaia.colnames}")

# %% [markdown]
# > **Important:** the *bare-float radius conventions differ* between services —
# > `search_vizier` reads arcminutes, `query_simbad`/`query_ned` arcseconds, and
# > Section 5's `cone_search` degrees. An astropy `Quantity`
# > (`radius=0.5 * u.deg`) is unambiguous everywhere; when in doubt, pass one.
# >
# > And watch `row_limit` (default 5000) on dense fields: rows arrive in catalog
# > order, so a truncated query is *not* a fair subsample — cap this query at
# > 6,000 rows and what comes back is a spatially lopsided *wedge* of the cone.
# > When completeness matters, fetch complete (`row_limit=-1`) with only the
# > columns you need and cut client-side, as `fetch_gaia` does. (Worth knowing,
# > too: Gaia saturates on the very brightest stars — a few of the Seven
# > Sisters themselves are missing from DR3.)
#
# The returned Table drops straight into `plot_catalog` — the
# `RA_ICRS`/`DE_ICRS` spellings are on the auto-detect list, so no column
# arguments needed. Color = the `BP-RP` color index, mapped through the
# diverging `sph.diff_blueorange` so the stars wear roughly their own colors;
# size = brightness:

# %%
# fig-slug: vizier-pleiades
fig = plt.figure(figsize=(7.5, 7))
ax = sph.make_wcs_frame(111, "TAN", center=(m45.ra.deg, m45.dec.deg),
                        fov_deg=1.7, fig=fig)
sc, cb = sph.plot_catalog(
    ax, gaia,
    sizeby="Gmag", size_scale=lambda mag: 10 ** (-0.4 * mag), smin=3, smax=300,
    colorby="BP-RP", cmap="sph.diff_blueorange", vmin=-0.3, vmax=2.8,
    alpha=0.85, cbar=True, cbar_label="BP − RP color index (mag)")
sph.format_ticklabels(ax, style="compact")
ax.set_title("Gaia DR3 around the Pleiades — one query, one plot call")
plt.show()

# %% [markdown]
# The hot blue Seven Sisters stand out at once from the redder field. This
# query→plot round trip — `search_vizier` → `plot_catalog` with `colorby`/`sizeby`
# — is the workhorse loop of catalog plotting.
#
# ### The color a star actually is
#
# That colorbar reads the index *quantitatively*, but `sph.diff_blueorange` is an
# arbitrary encoding — the blue and orange are a design choice, not what the
# stars really look like. skyplothelper can instead paint each star in the color a human
# eye would truly *perceive*: `sph.bp_rp_to_rgb(bp_rp)` returns an `(N, 3)` RGB
# array — a *tristimulus* color computed from the star's temperature — which you
# hand to the scatter as an explicit color (not `colorby=`, which maps a column
# through a colormap). There is no colorbar this time, because the color *is* the
# answer. Perceived star colors are pale and only read well against a dark sky, so this
# panel commits to a night background in both doc themes:

# %%
# fig-slug: gaia-star-colors
NIGHT = sph.ANNOTATION_PALETTES["night"]

# True perceived color per star. saturation=1.0 is the honest tristimulus value
# (the 0.55 default softens toward white); stars with missing BP−RP come back
# non-finite, so drop them.
colors = sph.bp_rp_to_rgb(gaia["BP-RP"], saturation=1.0)
ok = np.isfinite(colors).all(axis=1)
gok = gaia[ok]

fig = plt.figure(figsize=(7.5, 7))
ax = sph.make_wcs_frame(111, "TAN", center=(m45.ra.deg, m45.dec.deg),
                        fov_deg=1.7, fig=fig, gridcolor=NIGHT["grid"])
ax.set_facecolor(NIGHT["ax_bg"])
sph.style_wcs_axes(ax, tick_color=NIGHT["stars"], labelcolor=NIGHT["stars"])
sph.plot_catalog(ax, {"ra": gok["RA_ICRS"], "dec": gok["DE_ICRS"],
                      "mag": gok["Gmag"]},
                 sizeby="mag", size_scale=lambda mag: 10 ** (-0.4 * mag),
                 smin=3, smax=300, color=colors[ok], alpha=0.95, frame="icrs")
sph.format_ticklabels(ax, style="compact", color=NIGHT["stars"])
ax.set_title("Gaia DR3 around the Pleiades — each star's true perceived color")
plt.show()

# %% [markdown]
# The hot Seven Sisters glow faintly blue-white and the cooler field stars a pale
# orange — a Sun-like G star would sit near white. Real star colors are *subtle*,
# nothing like the saturated dots of a decorative chart; the honest tristimulus
# values (Harre & Heller 2021) are what `saturation=1.0` gives you, and the 0.55
# default deliberately washes them lighter for a softer look.
#
# `bp_rp_to_rgb` is the Gaia shortcut of the general
# `sph.color_index_to_rgb(value, index=...)`, which also speaks `"B-V"`, `"g-r"`,
# and `"J-K"` — reach for the matching shortcut rather than feeding BP−RP to
# `bv_to_rgb`, which over-reddens. The Constellations tutorial's
# [Coloring stars by temperature](constellations.ipynb#Coloring-stars-by-temperature)
# section carries the B−V half of this story and the color-science details.
#
# ### Building a reusable subset
#
# `search_vizier` is a *cone* search — it always queries around a position. When
# you instead want an **all-sky, column-filtered cut** — every source matching a
# criterion, wherever it sits — go to `astroquery.vizier` directly:
# `query_constraints()` with a column filter and `ROW_LIMIT = -1` (no row cap),
# then save the result once as a small CSV to commit next to your notebook. This
# is how several bundled example catalogs were built — including
# `hipparcos_bright_pm.csv`, the ~4,990 naked-eye stars (V < 6) that the
# [Vector Fields](vector_fields.ipynb), [Constellations](constellations.ipynb),
# and [Animations](animations.ipynb) tutorials use as a real proper-motion field:
#
# ```python
# from astroquery.vizier import Vizier
#
# v = Vizier(catalog="I/239/hip_main",
#            columns=["HIP", "RAICRS", "DEICRS", "Vmag", "pmRA", "pmDE"])
# v.ROW_LIMIT = -1                                     # no 50-row default cap
# hip = v.query_constraints(Vmag="<6")[0]              # all-sky, brighter than V=6
# hip.write("examples/data/hipparcos_bright_pm.csv")   # commit once, reuse offline
# ```
#
# Fetch it once, commit the CSV, and every later run reads it offline — the same
# bundled-data pattern the rest of this notebook uses. (The committed builder is
# `examples/data/fetch_hipparcos_bright.py`.)
#
# ## 4. Cutouts under your data
#
# A catalog tells you *where*; a survey image shows *what's there*. Two services
# cover most needs: **SkyView** (a NASA cutout service across ~200 surveys) and
# **HiPS2FITS** (CDS — cutouts from any all-sky survey published as HiPS).
#
# `list_skyview_surveys()` enumerates SkyView's offerings — the names must match
# *exactly*, so check here first when a survey "doesn't exist":

# %%
try:
    sph.list_skyview_surveys("DSS2")
except Exception:
    print("offline — list_skyview_surveys('DSS2') matches:\n"
          "  DSS2 Blue, DSS2 IR, DSS2 Red")

# %% [markdown]
# `download_skyview()` fetches a cutout as a plain `(data, header)` pair — ready
# for the quicklook tools, reprojection, or anything else that eats FITS. The
# same caching idea as Section 3 applies, with one deliberate inversion:
#
# > **Tip:** cache *tables* live-first, but cache *images* **cache-first**. A
# > published catalog is static, so a live query with a cached fallback keeps it
# > fresh at no risk. A raster from a tiling service is a different animal — the
# > same request can come back with subtly different pixels, so fetching it live
# > on every run makes a committed figure drift, and shows an offline reader
# > something other than what is published. Pin images to the cache.

# %%
def cached_fits(fetch, name):
    """Cache a (data, header) cutout as a small FITS file — **cache first**.

    Note the flip relative to `cached()` above. A published catalog is static, so
    querying it live and falling back to the cache is safe. A *raster* from a
    tiling service is not: HiPS in particular can hand back subtly different
    pixels from one call to the next, so a live-first fetch would let the
    committed figure drift between runs — and show an offline reader something
    other than what is committed. Images therefore pin to the cached copy and
    only reach for the network when it is missing.
    """
    path = QUERY_CACHE / name
    if path.exists():
        with fits.open(path) as hdul:
            return hdul[0].data, hdul[0].header
    data, hdr = _with_timeout(fetch, seconds=90)
    fits.PrimaryHDU(data.astype("float32"), header=hdr).writeto(path)
    return data, hdr


m51 = try_resolve("M51", (202.4696, 47.1952))
whirlpool, whirlpool_hdr = cached_fits(
    lambda: sph.download_skyview(m51, survey="DSS2 Red", size=0.35, pixels=320),
    "skyview_dss2red_m51.fits")

# %% [markdown]
# `quicklook_figure()` turns that pair into a finished figure in one call. Three
# knobs keep it in step with the rest of the page: `facecolor='none'` leaves the
# canvas transparent (so the figure sits on the docs background rather than a
# white card), while `axcolor=` and `frame_color=` set the text and frame from
# our theme palette. We'll reuse them for both cutouts:

# %%
# Transparent canvas + in-theme text/frame — the quicklook figures then read the
# same as every other figure here, in light and dark alike. `colorbar=False` is
# an editorial choice: quicklook shows a colorbar by default, but this section is
# about *fetching* cutouts, not displaying them — the scaling stack (stretch,
# colorbars, contours) is the FITS Images tutorial's subject. We keep these
# figures to the essentials: the cutout arrives plot-ready.
QL_THEME = dict(facecolor="none", axcolor=PAL["text"], frame_color=PAL["frame"],
                colorbar=False)

# %% [markdown]
# One more thing an optical image needs: a *scale*. DSS frames sit on a bright
# sky pedestal, so left to the default limits the empty sky lands in the middle
# of the colormap and washes the whole frame out. Anchoring `vmin` at the sky
# level (the median pixel) and `vmax` near the peak drops the background to
# black, and an `asinh` stretch opens up the faint spiral arms without blowing
# out the core. The colormap is `quicklook`'s own default — the bundled
# `sph.deepsky`, a through-black map that suits exactly this kind of image:

# %%
# fig-slug: skyview-cutout
sky, bright = np.nanpercentile(whirlpool, [50, 99.95])
res = sph.quicklook_figure(whirlpool, header=whirlpool_hdr, figsize=(6.5, 6.5),
                           image=True, contours=False, show_info=False,
                           stretch="asinh", vmin=sky, vmax=bright,
                           tick_style="compact", **QL_THEME)
res.ax.set_title("download_skyview → quicklook: DSS2 Red at M51")
plt.show()

# %% [markdown]
# (`quicklook_figure` and the full display stack — stretches, colorbars, contours
# — are the [FITS Images tutorial](fits_images.ipynb)'s territory; here the point
# is just that a cutout arrives plot-ready.)
#
# `download_hips()` is the HiPS equivalent — the same `(data, header)` shape
# (or a ready-made RGB array with `fmt='png'`), but it reaches the *many*
# surveys published as HiPS that SkyView doesn't carry. Swapping `colormap=` to
# another bundled map (`sph.nebula`) shows the same field in the infrared:

# %%
whirlpool_ir, whirlpool_ir_hdr = cached_fits(
    lambda: sph.download_hips(m51, hips_id="CDS/P/allWISE/W1",
                              size=0.35, pixels=320),
    "hips_allwise_m51.fits")

# %%
# fig-slug: hips-cutout
res = sph.quicklook_figure(whirlpool_ir, header=whirlpool_ir_hdr,
                           figsize=(6.5, 6.5), image=True, contours=False,
                           colormap="sph.nebula", stretch="log",
                           show_info=False, tick_style="compact", **QL_THEME)
res.ax.set_title("download_hips: the same field in AllWISE W1")
plt.show()

# %% [markdown]
# When the goal is simply "put an image *under my data*," skip the plumbing:
# `overlay_cutout()` fetches and lays the image beneath an existing frame's
# points in one call (grayscale, `zorder=0` by default):
#
# ```python
# sph.overlay_cutout(ax, m51, survey="DSS2 Red")     # image under your scatter
# ```
#
# We'll use exactly this in the capstone. Both fetchers cache their downloads
# (`cache=True`), so re-running a notebook doesn't re-download.
#
# ## 5. Searching, filtering and matching
#
# You have a catalog; now ask it *spatial questions*. Three helpers cover the
# common ones, and they share one design: **catalog in, catalog out** — each
# filters *your* table (where `search_vizier` fetched a new one) and hands back
# the same type it was given.
#
# | helper | the question it answers | membership test | needs |
# |---|---|---|---|
# | `cone_search` | which sources lie within X° of a point? | exact angular separation (analytic) | nothing — offline, all-sky |
# | `region_search` | which sources fall inside this region? | the region's own `contains_points` | a region, e.g. a drawn `CompoundRegion` |
# | `crossmatch` | which sources have a counterpart in that other catalog? | nearest neighbor within a tolerance | a reference catalog |
#
# All three are **type-preserving**: a `DataFrame` in gives a filtered
# `DataFrame` out, a `Table` a `Table`, a `SkyCoord` a `SkyCoord` — and raw
# `(ra, dec)` arrays give back a boolean mask to index with. `return_mask=True`
# forces the mask form for any input.
#
# ### Cone search — "within X degrees of here"
#
# `cone_search` keeps the sources within an angular radius of a center — computed
# as *true angular separation* on the sphere, so it's exact at any radius,
# anywhere on the sky, with no plot in sight. `add_separation=True` appends the
# separations as a column, and `sort=True` orders nearest-first:

# %%
m87 = sph.SKY_POSITIONS["m87"]
near = sph.cone_search(messier, m87, 10, sort=True, add_separation=True)
near[["name", "otype", "separation"]]

# %% [markdown]
# The result is the **Virgo Cluster**: sixteen Messier galaxies fanning out from
# the giant elliptical **M87** at its heart, ranked by exact separation — a
# degree or two for the crowded core (M84, M86, M89, M90…), out to M85 near the
# cone's edge. The `center=` accepts a `SkyCoord`, a plain `(ra, dec)` pair, or a
# *name* — `'M87'` works too, resolved through `resolve_name` (that one needs the
# network; `SKY_POSITIONS` is the offline spelling). Radii follow `unit=`
# (`'deg'` default, down through
# `'arcsec'`/`'mas'`), or pass a `Quantity`. And the math itself is exposed:
# `angulardistance` is the exact-separation engine under the hood, with
# `great_circle_distance`, `destination_point`, and `midpoint` alongside for
# spherical point-to-point work when you want separations without a filter.
#
# Because the separation column is just a column, it feeds straight back into
# Section 1's encoding — and drawing the same circle with `add_geodesic_circle`
# makes the selection *visible* while the math stays analytic:

# %%
# fig-slug: cone-search
fig = plt.figure(figsize=(7.6, 7))
ax = sph.make_wcs_frame(111, "STG", center=(m87.ra.deg, m87.dec.deg),
                        fov_deg=28, fig=fig)
sph.plot_catalog(ax, messier, color=RC["gray"], s=14, alpha=0.5)   # field context
sph.plot_catalog(ax, near, colorby="separation", s=64,
                 cmap="sph.deepsky_r", cmap_range=(0.12, 0.9),
                 edgecolors=PAL["frame"], linewidths=0.4,
                 cbar=True, cbar_label="separation from M87 (deg)")
# The cone center — a stroked star so it stays legible over the crowded core.
ax.scatter(m87.ra.deg, m87.dec.deg, transform=ax.get_transform("world"),
           marker="*", s=210, facecolor=PAL["accent"], edgecolor="white",
           linewidths=1.3, zorder=8)
sph.add_geodesic_circle(ax, m87.ra.deg, m87.dec.deg, radius_deg=10,
                        facecolor="none", edgecolor=PAL["accent"],
                        lw=1.4, ls="--")
sph.format_ticklabels(ax, style="compact")
ax.set_title("cone_search(messier, 'M87', 10°) — the Virgo Cluster,\n"
             "colored by exact spherical separation")
plt.show()

# %% [markdown]
# ### Region search — "inside this region"
#
# `region_search` generalizes the circle to *any* region exposing
# `contains_points` — which is exactly the protocol of the
# [Regions tutorial](regions.ipynb)'s `CompoundRegion`, so any set-algebra ROI
# you can draw, you can filter by. In that tutorial's capstone this took manual
# mask juggling (`contains_points` → boolean-index each array); `region_search`
# is that workflow as a one-liner.
#
# Which region? The built-in survey footprints are the natural first stop —
# discover them with `survey_keys()` / `list_surveys()`:

# %%
print(len(sph.survey_keys()), "built-in footprints:", ", ".join(sph.survey_keys()))

# %% [markdown]
# Here we ask which Messier objects LSST will see. The footprint recipe is two
# set-algebra verbs (the same recipe the built-in `'lsst'` entry uses — its
# dashed outline is overlaid as the check), and the membership test is one line:

# %%
# fig-slug: region-search
fig = plt.figure(figsize=(10, 5.4))
ax = sph.make_wcs_frame(111, "AIT", center=180, frame="ICRS", fig=fig)

lsst = (sph.CompoundRegion(ax)
        .add_latitude_band(-90, 12)                       # southern sky, Dec ≤ +12°
        .subtract_frame_band(-15, 15, frame="galactic"))  # avoid the crowded plane
lsst.render(facecolor=RC["blue"], alpha=0.22)
sph.add_survey_footprint(ax, "lsst", fill=False, edgecolor=RC["blue"], lw=1.2,
                         ls="--", alpha=0.85, label="built-in 'lsst' (check)")

visible = sph.region_search(messier, lsst)                # catalog in, catalog out
outside = messier[~sph.region_search(messier, lsst, return_mask=True)]

sph.plot_catalog(ax, outside, color=RC["gray"], s=14, alpha=0.6,
                 label=f"out of reach ({len(outside)})")
sph.plot_catalog(ax, visible, color=RC["gold"], s=36,
                 edgecolors=PAL["frame"], linewidths=0.3,
                 label=f"LSST sky ({len(visible)})")
sph.format_ticklabels(ax, style="allsky_hours")
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax.set_title("region_search(messier, lsst) — survey membership as a catalog filter")
plt.show()

# %% [markdown]
# And because *any* `CompoundRegion` satisfies the protocol, an arbitrary ROI
# works identically — say, a two-cluster observing program whose already-surveyed
# core is carved out:

# %%
# fig-slug: region-search-roi
virgo, coma = sph.SKY_POSITIONS["virgo_cluster"], sph.SKY_POSITIONS["coma_cluster"]

fig = plt.figure(figsize=(10, 5.4))
ax = sph.make_wcs_frame(111, "AIT", center=180, frame="ICRS", fig=fig)
roi = (sph.CompoundRegion(ax)
       .add_circle(virgo.ra.deg, virgo.dec.deg, radius_deg=12)
       .add_circle(coma.ra.deg, coma.dec.deg, radius_deg=10)
       .subtract_circle(virgo.ra.deg, virgo.dec.deg, radius_deg=4.5))
roi.render(facecolor=RC["green"], alpha=0.3)
roi.render_boundary(color=RC["green"], linewidth=1.3)

targets = sph.region_search(messier, roi, center=virgo,
                            add_separation=True, sort=True)
sph.plot_catalog(ax, messier, color=RC["gray"], s=14, alpha=0.6)
sph.plot_catalog(ax, targets, color=PAL["accent"], s=40,
                 edgecolors=PAL["frame"], linewidths=0.3)
sph.format_ticklabels(ax, style="allsky_hours")
ax.set_title(f"An arbitrary set-algebra ROI — {len(targets)} Messier targets "
             "in the program")
plt.show()

# %% [markdown]
# (One wrinkle shown above: for `sort=`/`add_separation=` a generic region has no
# obvious center, so pass `center=` explicitly — a cone supplies its own.)
#
# ### Crossmatch — "the same source in another catalog"
#
# `crossmatch` finds, for each of your sources, the nearest neighbor in a
# *reference* catalog and keeps those matched within a tolerance — the classic
# counterpart/cross-ID step. Every Messier object should have a counterpart in
# the NGC catalog, so let's recover the M→NGC mapping. First the reference —
# a whole-sky "cone" pulls the full NGC 2000.0 catalog, with the computed
# decimal-degree columns from Section 3's trick:

# %%
ngc = cached(
    lambda: sph.search_vizier("VII/118/ngc2000", SkyCoord(180, 0, unit="deg"),
                              radius=180 * u.deg,
                              columns=["_RAJ2000", "_DEJ2000", "Name", "Type", "mag"],
                              row_limit=-1),
    "vizier_ngc2000.ecsv")
print(f"{len(ngc)} NGC/IC entries")

matched = sph.crossmatch(messier, ngc, 10, unit="arcmin")
matched["NGC"] = np.asarray(ngc["Name"])[matched["match_idx"]]
matched["match_arcmin"] = matched["match_sep"] * 60
print(f"{len(matched)} of {len(messier)} Messier objects matched within 10′")
matched[["name", "NGC", "match_arcmin"]].head(6)

# %% [markdown]
# `match_idx` is the matched row's index into the *reference* — one line of fancy
# indexing pulls any reference column (here the NGC number) across. And the ones
# that *didn't* match are their own little history lesson:

# %%
print("no NGC counterpart:", ", ".join(sorted(set(messier["name"]) - set(matched["name"]),
                                              key=lambda n: int(n[1:]))))

# %%
# fig-slug: crossmatch-separations
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(matched["match_arcmin"], bins=40, color=RC["blue"], edgecolor=PAL["frame"])
ax.axvline(10, color=PAL["accent"], ls="--", lw=1.2)
ax.annotate("max_sep tolerance", xy=(10, ax.get_ylim()[1] * 0.85),
            xytext=(-8, 0), textcoords="offset points",
            ha="right", fontsize=9, color=PAL["accent"])
ax.set_xlabel("match separation (arcmin)")
ax.set_ylabel("Messier objects")
ax.set_yscale("log")
ax.set_title("Messier → NGC counterpart separations")
plt.show()

# %% [markdown]
# Most matches land within an arcminute — the tail is the extended objects whose
# cataloged centers genuinely differ. The misses are famous for it: **M40** is a
# double star and **M45** is the Pleiades — neither ever received an NGC number —
# while **M24** is a Milky Way star cloud whose associated cluster is IC 4715.
# Tighten `max_sep` and watch more drop out; rename the added columns with
# `sep_col=`/`idx_col=`; and `return_indices=True` hands back the raw
# `(idx, sep_deg, mask)` arrays when you'd rather inspect than filter.
#
# The same call handles the workhorse cases — match your observed positions
# against Gaia, or pull SIMBAD identifications (normalizing its coordinates with
# Section 3's `simbad_radec` helper):
#
# ```python
# # Gaia DR3 counterparts within 1 arcsec of your detections:
# gaia = sph.search_vizier("I/355/gaiadr3", field_center, radius=30)
# hits = sph.crossmatch(my_table, gaia, 1.0, unit="arcsec")
#
# # SIMBAD identifications — normalize its coordinates, then crossmatch against
# # the (ra, dec) arrays directly:
# sim = sph.query_simbad(field_center, radius=30 * u.arcmin)
# _, sra, sdec = simbad_radec(sim)
# hits = sph.crossmatch(my_table, (sra, sdec), 2.0, unit="arcsec")
# ```
#
# ### Choosing between cone and region
#
# > **Important:** the two filters test membership in *different spaces*.
# > `cone_search` computes exact angular separations — offline, all-sky, no axes
# > required, correct at any radius. `region_search` over a `CompoundRegion`
# > tests points against the region *in the plot's projected pixel space* — it
# > needs the drawn axes, and inherits the projection's rendering resolution.
# > So: "within X degrees of a point" → `cone_search`; "inside this drawn ROI" →
# > `region_search`. (This is exactly why the cone isn't implemented as a
# > circular `CompoundRegion` under the hood.)
#
# ## 6. Planning an observation
#
# Catalog work usually involves a telescope, and skyplothelper's sibling package
# [obsplanning](https://obsplanning.readthedocs.io) picks up exactly where the
# target list leaves off: observability windows, elevation tracks, transit
# times, finder plots. The hand-off is frictionless because everything here
# already speaks coordinates — a filtered catalog's rows become ephem targets in
# one comprehension:
#
# ```python
# import obsplanning as obs
#
# # §5's cone-search survivors become the observing list:
# targets = [obs.create_ephem_target(row["name"], row["ra_deg"], row["dec_deg"])
#            for _, row in near.iterrows()]
#
# kitt_peak = obs.create_ephem_observer("Kitt Peak", -111.5967, 31.9583, 2096)
# start = obs.dtaware_to_ephem(obs.construct_datetime(
#     "2026/10/15 19:00:00", "dt", timezone="US/Arizona"))
# end = obs.dtaware_to_ephem(obs.construct_datetime(
#     "2026/10/16 05:00:00", "dt", timezone="US/Arizona"))
#
# # Altitude tracks for the whole list over the night...
# obs.plot_night_observing_tracks(targets, kitt_peak, start, end,
#                                 simpletracks=True)
# # ...or the best night of the year for one of them:
# obs.optimal_visibility_date(targets[0], kitt_peak, "2026")
# ```
#
# > **Tip:** obsplanning also builds finder plots from survey imagery
# > (`make_finder_plot_simpleRGB` and friends) — the same SkyView machinery as
# > Section 4, aimed at the eyepiece instead of the paper.
#
# ## 7. Putting it together
#
# ### The Virgo Cluster, end to end
#
# One figure with the whole workflow in it. **Query**: the 2MASS Extended Source
# Catalog around the Virgo Cluster (real galaxies, decimal-degree columns).
# **Plot**: brightness encoded in size and color, Section 1 style. **Search**:
# Section 5's LSST footprint — whose Dec ≤ +12° northern limit happens to slice
# right through the cluster — splits the sample. We reuse the *same* `lsst`
# region object built back in Section 5 — a `CompoundRegion` stays queryable
# even after you've built its figure, so a footprint defined once can filter catalogs
# anywhere (building a fresh one on this zoomed frame works just as well).
# A Messier cone marks the famous members by name.
# **Cutout**: a DSS red backdrop under everything ties the points to the actual
# sky:

# %%
virgo = sph.SKY_POSITIONS["virgo_cluster"]

# Query: 2MASS XSC galaxies within 4.5°, then keep the bright end.
xsc = cached(
    lambda: sph.search_vizier("VII/233/xsc", virgo, radius=4.5 * u.deg,
                              columns=["RAJ2000", "DEJ2000", "K.ext"],
                              row_limit=-1),
    "xsc_virgo.ecsv")
gals = xsc[np.asarray(xsc["K.ext"], float) < 11.5]
print(f"{len(xsc)} XSC sources within 4.5° of Virgo; {len(gals)} with K < 11.5")

# Cutout: one wide DSS2 Red tile for the backdrop (cached like the others).
virgo_img, virgo_hdr = cached_fits(
    lambda: sph.download_skyview(virgo, survey="DSS2 Red", size=10.5, pixels=520),
    "skyview_dss2red_virgo.fits")

# %%
# fig-slug: capstone-virgo
fig = plt.figure(figsize=(9, 8.2))
ax = sph.make_wcs_frame(111, "TAN", center=(virgo.ra.deg, virgo.dec.deg),
                        fov_deg=10.5, fig=fig)

# Backdrop — percentile-normalize the raw counts to [0, 1] (the image-like
# range reproject_background expects; the FITS Images tutorial covers the full
# stretch toolkit), then reproject under everything. The colormap flips with the
# page theme so the sky stays dark-on-light in light mode and light-on-dark in
# dark.
bg = sph.reproject_background(sph.rescale_percentile(virgo_img, 22, 99.6),
                              virgo_hdr, ax)
ax.imshow(bg, cmap="gray" if IS_DARK else "gray_r", zorder=-10)

# Plot: the queried galaxies, brightness in size and color.
sc, cb = sph.plot_catalog(
    ax, gals, colorby="K.ext", cmap="sph.dusk_r", cmap_range=(0.15, 0.92),
    sizeby="K.ext", size_scale=lambda m: 10 ** (-0.4 * m), smin=8, smax=150,
    alpha=0.9, edgecolors=PAL["frame"], linewidths=0.3,
    cbar=True, cbar_label="2MASS K magnitude", cbar_format="{x:.0f}")

# Search 1: Section 5's LSST region, reused — its Dec = +12° northern limit
# runs right through the field (drawn as the dashed line).
n_lsst = int(sph.region_search(gals, lsst, return_mask=True).sum())
edge_ra = np.linspace(virgo.ra.deg - 6.5, virgo.ra.deg + 6.5, 60)
ax.plot(edge_ra, np.full_like(edge_ra, 12.0),
        transform=ax.get_transform("world"),
        color=PAL["accent"], ls="--", lw=1.6)

# Search 2: the Messier members, ringed — with names only where they fit (the
# crowded Markarian's-Chain core is labeled fully in Section 1's finder chart).
# Hollow rings would take the label color down with them — color feeds both —
# so the names get their own zero-size call.
famous = sph.cone_search(messier, virgo, 5.2)
sph.plot_catalog(ax, famous, marker="o", s=210, color="none",
                 edgecolors=PAL["accent"], linewidths=1.1)
roomy = famous[famous["name"].isin(["M49", "M87", "M100", "M98", "M60", "M91"])]
sph.plot_catalog(ax, roomy, s=0, color=PAL["accent"],
                 label_col="name", label_fontsize=9, label_offset=(11, 7))

ax.text(0.02, 0.03,
        f"{len(gals)} 2MASS XSC galaxies (K < 11.5)\n"
        f"{n_lsst} on LSST's sky (below the dashed line)\n"
        f"{len(famous)} Messier objects ringed",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round", fc=PAL["ax_bg"], ec=PAL["frame"], alpha=0.85))
sph.format_ticklabels(ax, style="compact")
ax.set_title("The Virgo Cluster, end to end — queried, encoded, "
             "footprint-sliced, and labeled")
plt.show()

# %% [markdown]
# ### The same pipeline on any target
#
# Everything above generalizes the moment you wrap it in a function of *your*
# target list. Resolve a name, fetch the field, drop a survey image underneath,
# encode the photometry — three open clusters, one pipeline (this is Section 3's
# Pleiades round trip, industrialized):

# %%
# fig-slug: capstone-postcards
CLUSTERS = [
    ("M45", (56.601, 24.114), 50),    # the Pleiades — nearby, huge on the sky
    ("M44", (130.054, 19.621), 50),   # the Beehive — mid-distance
    ("M67", (132.846, 11.814), 25),   # one of the oldest open clusters — compact
]


def cluster_postcard(ax, name, fallback_radec, radius_arcmin):
    """Resolve → query Gaia → DSS backdrop → color-encoded stars. Reuse freely."""
    c = try_resolve(name, fallback_radec)
    g = cached(lambda: fetch_gaia(c, radius_arcmin), f"gaia_{name.lower()}.ecsv")
    with warnings.catch_warnings():          # offline → skip the backdrop quietly
        warnings.simplefilter("ignore")
        # Flip the backdrop with the page, exactly as the Virgo capstone does:
        # a true greyscale reads as a dark starfield on the dark page, while the
        # inverted map keeps the sky white-on-light. Using one for both leaves
        # the losing mode with a muddy gray wash.
        sph.overlay_cutout(ax, c, survey="DSS2 Red", alpha=0.35,
                           size=2 * radius_arcmin / 60 * 1.2,
                           cmap="gray" if IS_DARK else "gray_r")
    sph.plot_catalog(ax, g,
                     sizeby="Gmag", size_scale=lambda m: 10 ** (-0.4 * m),
                     smin=2, smax=210,
                     colorby="BP-RP", cmap="sph.diff_blueorange",
                     vmin=-0.3, vmax=2.8, alpha=0.9)
    sph.format_ticklabels(ax, style="compact", fontsize=8)
    ax.set_title(f"{name} — {len(g)} Gaia sources", fontsize=10)


fig = plt.figure(figsize=(13, 4.6))
for i, (name, radec, rad) in enumerate(CLUSTERS, start=1):
    c = try_resolve(name, radec)
    ax = sph.make_wcs_frame(130 + i, "TAN", center=(c.ra.deg, c.dec.deg),
                            fov_deg=2 * rad / 60 * 1.15, fig=fig)
    cluster_postcard(ax, name, radec, rad)
fig.suptitle("One pipeline, three clusters — swap in your own targets",
             y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# Even the astrophysics comes along for free: the Pleiades' brightest members
# are blue (young, hot), while M67 — one of the oldest open clusters known —
# shows its brightest stars already evolved into orange red giants.

# %% [markdown]
# ## 8. Where to go next
#
# | If you want to... | Go to |
# |---|---|
# | style the frames these maps sit on — ticks, grids, themes | [Decorating Frames](decorating_frames.ipynb) |
# | overlay a second coordinate grid on a catalog map | [Overlay Coordinate Grids](overlay_grids.ipynb) |
# | do more with the FITS cutouts — stretches, colorbars, RGB | [FITS Images & Quicklook](fits_images.ipynb) |
# | build richer set-algebra footprints (and query them) | [Regions & Spherical Polygons](regions.ipynb) |
# | bin a huge catalog into a HEALPix density map | [HEALPix Workflows](healpix_workflows.ipynb) |
# | plot a redshift catalog as a wedge/cone diagram | [Cone & Bowtie Plots](cone_bowtie.ipynb) |
# | put proper-motion vectors on your sources | [Vector Fields & Sky Kinematics](vector_fields.ipynb) |
# | pan/zoom/hover an interactive catalog map | [Interactive Plotting](interactive_plotly.ipynb) |
#
# The [Catalogs & queries guide page](../guide/queries.md) is the compact map of
# everything toured here, and the [Projections tutorial](projections.ipynb)
# closes with the same `plot_catalog` encodings exercised across projections
# (its §9 was this notebook's teaser). Happy hunting.
