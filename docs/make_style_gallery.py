"""Generate the styling-guide gallery figures from the live style dicts.

Renders, straight from ``sph.BASE_PRESETS`` / ``sph.CYCLE_PALETTES`` /
``sph.ANNOTATION_PALETTES``, into ``_static/style/``:

- ``preset-<name>-{light,dark}.png`` — the 8 base presets, a demo plot each
  (light + dark so they follow the navbar plot-color toggle).
- ``cycle-<name>-{light,dark}.png`` — the 6 data-color cycles: swatch row
  over a few demo curves (light + dark).
- ``annot-<name>.png`` — the 5 annotation palettes on a finder-chart mock.
  These are inherently light/dark (each defines its own background), so
  they are shown as a static grid rather than toggled.
- ``fonts.png`` — a serif / sans / monospace font-stack sample strip.

Because every figure is generated from the live dicts, adding a preset or
palette to the package and re-running this script keeps the styling guide
in sync — nothing is hand-drawn.

NOT run during the Sphinx build; outputs are committed (same convention as
``make_features.py``) so ReadTheDocs serves them without the plotting
stack. Run locally:

    cd docs && python make_style_gallery.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
# Import the local package source, not any same-named module elsewhere on
# the environment path (same shadow-guard as conf.py / make_features.py).
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import skyplothelper as sph  # noqa: E402

IMG_DIR = DOCS / "_static" / "style"
# Higher than the feature-gallery thumbnails (110): these feed a click-to-zoom
# lightbox, so the full-size view stays crisp.
DPI = 150

# Light/dark each pair a base preset (or cycle) with one of the package's
# own sky themes, matching the docs light/dark palette.
THEME = {"light": "publication", "dark": "dark_sky"}

# The four built-in theme presets (set_theme has no public enumerating dict;
# these are the documented names). Each is inherently light or dark.
BUILTIN_THEMES = ["publication", "poster", "twilight", "dark_sky"]


def _save(fig: object, stem: str, *, transparent: bool = False) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    kw = dict(dpi=DPI, bbox_inches="tight")
    if transparent:
        # Let savefig fill the canvas transparent too; passing facecolor
        # here would paint an opaque background and defeat transparency.
        kw["transparent"] = True
    else:
        kw["facecolor"] = fig.get_facecolor()
    fig.savefig(IMG_DIR / f"{stem}.png", **kw)
    plt.close(fig)


def _demo_lines(ax: object, title: str) -> None:
    """The shared multi-element demo plot used for presets and themes."""
    x = np.linspace(0, 10, 200)
    for i in range(5):
        ax.plot(x, np.sin(x + i * 0.5) + i * 0.12, label=f"series {i + 1}")
    xs = np.linspace(0.5, 9.5, 11)
    ax.scatter(xs, 0.25 + 0.35 * np.cos(xs), zorder=5, s=16)
    ax.set_xlabel("wavelength [arb.]")
    ax.set_ylabel("flux [arb.]")
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=2)


def render_preset(name: str, mode: str) -> None:
    """A compact multi-element demo plot under one base preset."""
    plt.close("all")
    with sph.style_context(base=name, theme=THEME[mode], palette="uranometria"):
        fig, ax = plt.subplots(figsize=(4.2, 2.7))
        _demo_lines(ax, f"base={name!r}")
        _save(fig, f"preset-{name}-{mode}")


def render_theme(name: str) -> None:
    """The same demo plot under one built-in theme (each is inherently
    light or dark, so it is shown as a single figure)."""
    plt.close("all")
    with sph.style_context(base="standard", theme=name, palette="uranometria"):
        fig, ax = plt.subplots(figsize=(4.2, 2.7))
        _demo_lines(ax, f"theme={name!r}")
        _save(fig, f"theme-{name}")


def render_cycle(name: str, mode: str) -> None:
    """A swatch row over demo curves cycling through one data-color palette."""
    plt.close("all")
    colors = sph.CYCLE_PALETTES[name]["colors"]
    with sph.style_context(base="standard", theme=THEME[mode], palette=name):
        fig, (axsw, axdemo) = plt.subplots(
            2, 1, figsize=(4.2, 2.9), gridspec_kw={"height_ratios": [1, 2.4]})
        for i, c in enumerate(colors):
            axsw.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=c))
        axsw.set_xlim(-0.04, len(colors))
        axsw.set_ylim(0, 1)
        axsw.set_axis_off()
        axsw.set_title(f"palette={name!r}", fontsize=9)
        x = np.linspace(0, 10, 200)
        for i in range(len(colors)):
            axdemo.plot(x, np.sin(x + i * 0.55), lw=1.6)
        axdemo.set_xticks([])
        axdemo.set_yticks([])
        _save(fig, f"cycle-{name}-{mode}")


def render_annotation(name: str) -> None:
    """One annotation palette on a small WCS finder-chart mock."""
    plt.close("all")
    ax = sph.make_wcs_frame(111, "TAN", center=(83.63, 22.01), fov_deg=1.4)
    pal = sph.style_annotation(ax, name)
    fig = ax.figure
    fig.set_size_inches(3.4, 3.2)
    rng = np.random.default_rng(5)
    ra = 83.63 + (rng.random(28) - 0.5) * 1.15
    dec = 22.01 + (rng.random(28) - 0.5) * 1.15
    ax.scatter(ra, dec, transform=ax.get_transform("world"),
               s=14, color=pal["stars"], zorder=5)
    # Target reticle (accent) + object label (label color) + compass
    # (compass color) — exercises the rest of the role palette.
    ax.scatter([83.63], [22.01], transform=ax.get_transform("world"),
               s=140, marker="o", facecolor="none",
               edgecolor=pal["accent"], lw=1.6, zorder=6)
    ax.text(83.63 + 0.42, 22.01 + 0.42, "target",
            transform=ax.get_transform("world"), color=pal["label"],
            fontsize=10, zorder=7)
    sph.add_compass(ax, color=pal["compass"], stroke_color=pal["ax_bg"])
    ax.set_title(f"{name!r}", color=pal["text"], fontsize=11)
    _save(fig, f"annot-{name}")


def render_fonts(mode: str) -> None:
    """A serif / sans / monospace sample strip using the package font stacks.

    Transparent background with mode-matched ink (dark for the light page,
    light for the dark page) so the text stays crisp in either mode and the
    strip blends into the page rather than sitting in a white box."""
    plt.close("all")
    ink = "#333333" if mode == "light" else "#d9d5c5"
    label_ink = "#8c8c8c"  # secondary; reads on both
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    ax.set_axis_off()
    sample = (r"RA 12$^\mathregular{h}$34$^\mathregular{m}$  "
              r"Dec +21°30′  ·  γ Cas  ·  3C 273  ·  M31")
    rows = [
        ("serif", "serif"),
        ("sans-serif", "sans-serif"),
        ("mono (MONO_STACK)", sph.MONO_STACK),
    ]
    for i, (label, family) in enumerate(rows):
        y = 0.82 - i * 0.32
        ax.text(0.01, y, label, fontsize=9, family="sans-serif",
                color=label_ink, transform=ax.transAxes, va="center")
        ax.text(0.40, y, sample, fontsize=13, family=family, color=ink,
                transform=ax.transAxes, va="center")
    _save(fig, f"fonts-{mode}", transparent=True)


def render_colormaps(mode: str) -> None:
    """The bundled image colormaps as labeled gradient swatches. The swatch
    gradients are mode-invariant; only the backdrop and the name labels
    change, so it follows the light/dark plot-color toggle like the other
    figures (dark uses the docs-dark backdrop with cream labels)."""
    plt.close("all")
    fig = sph.show_colormaps()
    if mode == "light":
        bg = "white"
    else:
        bg = "#1d1c1a"                       # docs-dark figure backdrop
        for ax in fig.axes:                  # recolor the name labels (hardcoded black)
            for t in ax.texts:
                t.set_color("#d9d5c5")
    fig.savefig(IMG_DIR / f"colormaps-{mode}.png", dpi=DPI,
                bbox_inches="tight", facecolor=bg)
    plt.close(fig)


def main() -> None:
    ok = fail = 0

    def run(label: str, fn: object) -> None:
        nonlocal ok, fail
        try:
            fn()
            print(f"  ok   {label}")
            ok += 1
        except Exception:  # noqa: BLE001 — report and continue
            print(f"  FAIL {label}")
            traceback.print_exc()
            fail += 1

    for name in sph.BASE_PRESETS:
        for mode in ("light", "dark"):
            run(f"preset {name} ({mode})", lambda n=name, m=mode: render_preset(n, m))
    for name in BUILTIN_THEMES:
        run(f"theme {name}", lambda n=name: render_theme(n))
    for name in sph.CYCLE_PALETTES:
        for mode in ("light", "dark"):
            run(f"cycle {name} ({mode})", lambda n=name, m=mode: render_cycle(n, m))
    for name in sph.ANNOTATION_PALETTES:
        run(f"annotation {name}", lambda n=name: render_annotation(n))
    for mode in ("light", "dark"):
        run(f"fonts ({mode})", lambda m=mode: render_fonts(m))
    for mode in ("light", "dark"):
        run(f"colormaps ({mode})", lambda m=mode: render_colormaps(m))

    print(f"\n{ok} rendered, {fail} failed → {IMG_DIR}")


if __name__ == "__main__":
    main()
