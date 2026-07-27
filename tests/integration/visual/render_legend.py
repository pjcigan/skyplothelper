"""Render MultiLegend (multi-dimensional legend builder) for visual eyeballing.

Produces:
  - legend_01_dgr_multichannel.png  — a DGRvsMetallicity-style scatter with a
                                       color + shape + line legend (the Phase-1
                                       reference figure); checks that grayscale
                                       shape swatches read as their own
                                       dimension next to the color block.
  - legend_02_placements.png        — the same legend at several loc presets,
                                       including off-frame 'outside ...' spots.
  - legend_03_palettes.png          — light vs dark palette (mode-aware text +
                                       frame), with and without stroke.
  - legend_04_bvid_allsky.png       — BVID-style all-sky Aitoff catalog: marker
                                       size = N observations (graduated key),
                                       color = formal-error bin, marker shape =
                                       defining vs non-defining, legend placed
                                       off-frame in the lower margin. Synthetic
                                       stand-in for the real ICRF/USNO catalog.
  - legend_05_all_channels.png      — every channel block in one legend: color,
                                       shape, size, edge, fill, hatch, alpha,
                                       orientation, glyph, line, region,
                                       colorbar, text, custom.

Usage
-----
    python render_legend.py            # save PNGs to output/
    python render_legend.py --show     # display interactively
"""

import sys

import matplotlib.pyplot as plt  # noqa: E402  (after _common's backend choice)
import numpy as np
from _common import banner, save_or_show

import skyplothelper as sph


def _dgr_scatter(ax):
    """A synthetic Dust-Gas-Ratio vs Metallicity scatter using every channel.

    color = target galaxy, marker shape = literature sample, thin/thick +
    open/closed encode aperture / measurement — the encodings the legend
    explains. Positions are illustrative, not real data.
    """
    rng = np.random.default_rng(4)
    targets = {"DDO 69": "purple", "DDO 70": "C0",
               "DDO 75": "green", "DDO 210": "#b0a0d0"}
    samples = {"DGS": "o", "KINGFISH": "D", "Galametz+11": "+", "Galliano+08": "^"}
    # Literature cloud: gray-ish, varied shapes, spread along a trend.
    for _name, mk in samples.items():
        x = rng.uniform(7.5, 9.0, 22)
        y = -6.5 + 1.4 * (x - 7.0) + rng.normal(0, 0.4, x.size)
        ax.scatter(x, y, marker=mk, s=42, facecolor="0.4",
                   edgecolor="0.25", linewidths=0.6, alpha=0.8, zorder=2)
    # LT targets: colored, large, at the metal-poor end.
    for name, col in targets.items():
        x = rng.uniform(7.15, 7.6, 3)
        y = -4.8 + 0.8 * (x - 7.0) + rng.normal(0, 0.3, x.size)
        ax.scatter(x, y, marker="o", s=150, facecolor=col,
                   edgecolor="k", linewidths=0.7, zorder=3)
    # Two fit lines.
    xx = np.linspace(7.0, 9.5, 50)
    ax.plot(xx, -6.7 + 1.35 * (xx - 7.0), ls=":", color="0.3", lw=1.4)
    ax.plot(xx, -6.9 + 1.55 * (xx - 7.0), ls="-.", color="0.3", lw=1.4)
    ax.set_xlim(7.0, 9.5)
    ax.set_ylim(-6.6, -1.0)
    ax.set_xlabel("Metallicity [12 + log(O/H)]")
    ax.set_ylabel(r"log $M_{dust}\,/\,M_{gas}$")
    ax.set_title("Dust-Gas Ratio vs Metallicity")
    return targets, samples


def render_dgr(mode):
    banner("legend_01: DGR-style multi-channel legend")
    fig, ax = plt.subplots(figsize=(8, 6))
    targets, samples = _dgr_scatter(ax)
    (sph.MultiLegend(ax, loc="lower right")
        .add_color("Target", targets, ncol=2)            # patch chips
        .add_shape("Sample", samples)                    # auto-grayscale
        .add_line("RR14 Fit", {r"$X_{CO,MWG}$": ":", r"$X_{CO,Z}$": "-."})
        .draw())
    save_or_show(fig, "legend_01_dgr_multichannel", mode)


def render_placements(mode):
    banner("legend_02: placement presets, incl. off-frame")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.subplots_adjust(hspace=0.35, wspace=0.45, right=0.82)
    cat = {"Def.": "orange", "Not Def.": "C0", "None": "red"}
    for ax, loc in zip(axes.ravel(),
                       ["upper left", "lower right",
                        "outside lower right", "outside bottom"]):
        ax.scatter(np.random.default_rng(1).uniform(0, 1, 40),
                   np.random.default_rng(2).uniform(0, 1, 40),
                   c="0.6", s=20)
        ax.set_title(f"loc={loc!r}", fontsize=10)
        (sph.MultiLegend(ax, loc=loc)
            .add_color("Cat", cat, swatch="marker")
            .draw())
    save_or_show(fig, "legend_02_placements", mode)


def render_palettes(mode):
    banner("legend_03: mode-aware palettes + stroke")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cat = {"Def.": "orange", "Not Def.": "C0", "None": "red"}
    # Light publication palette.
    ax = axes[0]
    sph.style_annotation(ax, "publication")
    ax.set_title("palette='publication'", fontsize=10)
    (sph.MultiLegend(ax, loc="center", palette="publication")
        .add_color("Cat", cat, swatch="marker")
        .add_shape("Sample", {"A": "o", "B": "D", "C": "^"})
        .draw())
    # Dark palette + stroke on swatches/text.
    ax = axes[1]
    sph.style_annotation(ax, "dark")
    ax.set_title("palette='dark' + stroke", fontsize=10)
    (sph.MultiLegend(ax, loc="center", palette="dark",
                     stroke_color="black", stroke_lw=2.5)
        .add_color("Cat", cat, swatch="marker")
        .add_shape("Sample", {"A": "o", "B": "D", "C": "^"})
        .draw())
    save_or_show(fig, "legend_03_palettes", mode)


def render_bvid_synthetic(mode):
    banner("legend_04: BVID-style all-sky catalog (synthetic)")
    from skyplothelper.wcs_frame import make_wcs_frame

    rng = np.random.default_rng(7)
    n = 500
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))       # uniform on sphere
    nobs = rng.integers(1, 600, n).astype(float)             # marker size
    # Formal error (mas): anti-correlated with Nobs + scatter → 4 bins.
    err = 0.05 + 3.0 / np.sqrt(nobs) * rng.uniform(0.6, 1.5, n)
    ebins = np.digitize(err, [0.2, 0.4, 0.8])                # 0..3
    defining = rng.uniform(size=n) < 0.15                    # marker shape

    ebin_colors = {0: "#2166AC", 1: "#4DAF4A", 2: "#FF7F00", 3: "#E41A1C"}
    ebin_labels = {0: "< 0.2", 1: "0.2–0.4", 2: "0.4–0.8", 3: "> 0.8"}

    fig = plt.figure(figsize=(11, 6.5))
    ax = make_wcs_frame(projection="AIT", fig=fig)
    fig.subplots_adjust(bottom=0.22)

    # One scatter per (defining, error-bin): marker encodes defining, color
    # encodes the bin; a SHARED size_vlim keeps equal Nobs equal-sized across
    # all of them (a scatter can only carry one marker).
    cp_for_legend = None
    for is_def, mk in [(False, "o"), (True, "*")]:
        for b, col in ebin_colors.items():
            m = (defining == is_def) & (ebins == b)
            if not m.any():
                continue
            cp = sph.plot_catalog(
                ax, {"ra": ra[m], "dec": dec[m], "nobs": nobs[m]},
                sizeby="nobs", size_vlim=(1, 600), size_scale="sqrt",
                smin=6, smax=340, marker=mk, color=col, alpha=0.72,
                edgecolor="0.15", linewidths=0.3, frame="icrs", zorder=3)
            cp_for_legend = cp_for_legend or cp   # any call shares the scaling

    ax.set_title("Synthetic all-sky VLBI catalog (BVID-style)", fontsize=12)

    # Off-frame legend: size key (from the plot) + error-bin colors + shapes.
    (sph.MultiLegend(ax, loc="outside bottom", orientation="horizontal",
                     block_sep=22)
        .add_size_from(cp_for_legend, values=[1, 5, 50, 200, 500],
                       title="Nb of Obs", orientation="horizontal")
        .add_color("Formal error (mas)",
                   {ebin_labels[b]: ebin_colors[b] for b in range(4)},
                   swatch="marker", ncol=2)
        .add_shape("ICRF3", {"Defining": "*", "Other": "o"})
        .draw())
    save_or_show(fig, "legend_04_bvid_allsky", mode)


def render_all_channels(mode):
    banner("legend_05: every channel block")
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    star = Line2D([6.5], [6.5], marker=(5, 1), markersize=11, linestyle="none",
                  markerfacecolor="#C9A23F", markeredgecolor="0.2")

    (sph.MultiLegend(ax, loc="center", orientation="horizontal", block_sep=20)
        .add_color("Color", {"A": "#4C78A8", "B": "#F58518", "C": "#54A24B"})
        .add_shape("Shape", {"circle": "o", "diamond": "D", "plus": "+"})
        .add_size("Size", values=[1, 20, 100, 500], smin=8, smax=300,
                  scale="sqrt", ncol=2)
        .add_edge("Edge", {"secure": "#54A24B", "marginal": "#E45756"})
        .add_fill("Fill", {"full": "filled", "reduced": "open"})
        .add_fill("Hatch", {"DES": "///", "LSST": "xxx"}, kind="patch",
                  color="#4C78A8")
        .add_alpha("Alpha", values=[1, 10, 100], color="#72539B")
        .add_orientation("Angle", {"0°": 0, "45°": 45, "90°": 90})
        .add_glyph("Glyph", {"target": "reticle_circle", "mark": "crosshair",
                             "corner": "corner"})
        .add_line("Line", {"model": "--", "fit": "-."})
        .add_region("Region", {"survey": dict(fc="#4C78A8", ec="#4C78A8"),
                               "mask": dict(fc="#E45756", ec="#E45756",
                                            hatch="//")})
        .add_colorbar("Cbar", cmap="viridis", vmin=0, vmax=10, length=80,
                      fmt=".0f")
        .add_text("Notes", ["dashed = model"])
        .add_custom("Custom", {"target": star})
        .draw())
    ax.set_title("Every MultiLegend channel block", fontsize=13)
    save_or_show(fig, "legend_05_all_channels", mode)


def main():
    mode = "show" if "--show" in sys.argv else "save"
    render_dgr(mode)
    render_placements(mode)
    render_palettes(mode)
    render_bvid_synthetic(mode)
    render_all_channels(mode)
    if mode == "save":
        print("\nDone. PNGs in output/.")


if __name__ == "__main__":
    main()
