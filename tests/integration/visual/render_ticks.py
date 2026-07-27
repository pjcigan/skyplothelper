"""Render every advertised ``format_ticklabels`` style for visual eyeballing.

Produces:
  - ticks_01_sexagesimal_grid.png — 3×4 grid of HMS / DMS-flavoured styles
    (publication, letter, casa, latex, compact, minimal, decimal,
    decimal_plain, vlbi, allsky_hours [AIT], allsky_deg [AIT galactic],
    default [no style]).
  - ticks_02_offset_grid.png — 3×3 grid of offset-coordinate styles
    on a small-FOV TAN axes (offset, offset_arcsec, offset_arcmin,
    offset_mas, offset_uas, anchored_offset, anchored_offset_uas,
    anchored_offset_compact, plus a "no style" reference).
  - ticks_03_offset_sign_demo.png — One-panel proof that
    ``apply_offset_ticks`` keeps signed labels correctly (the
    simplify=False fix).
  - ticks_04_frame_aware_defaults.png — Same default ``format_ticklabels``
    applied to ICRS vs Galactic vs Ecliptic axes, showing that the
    auto-detection picks unit/style appropriately for each frame.

Usage
-----
    python render_ticks.py            # save PNGs to output/
    python render_ticks.py --show     # display interactively
"""

import sys
import warnings

import matplotlib.pyplot as plt
from _common import banner, save_or_show

from skyplothelper.ticks import (
    apply_anchored_offset,
    apply_offset_ticks,
    format_ticklabels,
)
from skyplothelper.wcs_frame import make_wcs_frame

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


def _safe_apply(ax, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            format_ticklabels(ax, **kw)
        except Exception as e:
            print(f"  WARNING: format_ticklabels({kw}) failed: "
                  f"{type(e).__name__}: {e}")


@_panel("ticks_01_sexagesimal_grid")
def render_sexagesimal_grid():
    """Sexagesimal/decimal styles applied to a TAN axes.

    Each cell tuple is ``(style, proj, frame, title, cdelt_asec)`` —
    ``cdelt_asec`` is the per-pixel scale in arcseconds (None = use
    the make_wcs_frame default for that projection). The ``vlbi`` cell
    overrides to a mas-scale FOV (±40 mas) since the vlbi style is
    designed for sub-arcsecond offsets.
    """
    cells = [
        ("publication",   "TAN",  "ICRS",     "publication (mathtext ʰᵐˢ)",     None),
        ("letter",        "TAN",  "ICRS",     "letter (h, m, s)",               None),
        ("casa",          "TAN",  "ICRS",     "casa (colon-separated)",         None),
        ("latex",         "TAN",  "ICRS",     "latex (LaTeX superscripts)",     None),
        ("compact",       "TAN",  "ICRS",     "compact (suppress unchanged)",   None),
        ("minimal",       "TAN",  "ICRS",     "minimal (hours/degrees only)",   None),
        ("decimal",       "TAN",  "ICRS",     "decimal (deg with °)",           None),
        ("decimal_plain", "TAN",  "ICRS",     "decimal_plain (no °)",           None),
        # vlbi style is sub-arcsecond; demo it on a ±40 mas FOV
        ("vlbi",          "TAN",  "ICRS",     "vlbi (sub-arcsec sexagesimal)",  4e-4),
        ("allsky_hours",  "AIT",  "ICRS",     "allsky_hours (AIT, ICRS)",       None),
        ("allsky_deg",    "AIT",  "Galactic", "allsky_deg (AIT, Galactic)",     None),
        (None,            "TAN",  "ICRS",     "(default: auto-detect frame)",   None),
    ]
    nrows, ncols = 3, 4
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("format_ticklabels — sexagesimal & decimal styles",
                 fontsize=14, y=0.995)
    for idx, (style, proj, frame, title, cdelt_asec) in enumerate(cells, start=1):
        center = (180.0, 30.0) if proj == "TAN" else 0
        kw = dict(projection=proj, center=center, frame=frame, fig=fig)
        if cdelt_asec is not None:
            kw['cdelt'] = cdelt_asec / 3600.0
            kw['npix'] = 200
        ax = make_wcs_frame((nrows, ncols, idx), **kw)
        if style is not None:
            _safe_apply(ax, style=style)
        ax.set_title(title, fontsize=9)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.93, bottom=0.05,
                        hspace=0.55, wspace=0.45)
    return fig


@_panel("ticks_02_offset_grid")
def render_offset_grid():
    """Offset / VLBI-style ticks on TAN axes whose FOV is matched
    to each tick style's natural unit (so the shown numbers stay in
    a readable range — e.g. arcsec styles get a ~tens-of-arcsec FOV,
    mas styles get a tens-of-mas FOV, etc.)."""
    npix = 200
    # (style, title, cdelt in arcsec/pixel) — total FOV is npix * cdelt
    cells = [
        ("offset",              "offset (auto-unit)",            0.4),     # ±40″
        ("offset_arcsec",       "offset_arcsec (Δα″, Δδ″)",      0.4),     # ±40″
        ("offset_arcmin",       "offset_arcmin (Δα′, Δδ′)",      24.0),    # ±2400″ = ±40′
        ("offset_mas",          "offset_mas (Δα mas, Δδ mas)",   4e-4),    # ±40 mas
        ("offset_uas",          "offset_uas (Δα μas, Δδ μas)",   4e-7),    # ±40 μas
        ("anchored_offset",         "anchored_offset (anchor + mas)",    4e-4),    # ±40 mas
        ("anchored_offset_uas",     "anchored_offset_uas (anchor + μas)", 4e-7),   # ±40 μas
        ("anchored_offset_compact", "anchored_offset_compact (rotated)", 4e-4),    # ±40 mas
        (None,                  "(default: auto-detect)",        0.4),     # ±40″
    ]
    nrows, ncols = 3, 3
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(
        "format_ticklabels — offset / VLBI styles (FOV matched per cell)",
        fontsize=14, y=0.995,
    )
    for idx, (style, title, cdelt_asec) in enumerate(cells, start=1):
        ax = make_wcs_frame(
            (nrows, ncols, idx), projection="TAN", center=(180.0, 30.0),
            cdelt=cdelt_asec / 3600.0, npix=npix, frame="ICRS", fig=fig,
        )
        if style is not None:
            _safe_apply(ax, style=style)
        ax.set_title(title, fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.05,
                        hspace=0.55, wspace=0.5)
    return fig


@_panel("ticks_03_offset_sign_demo")
def render_offset_sign_demo():
    """Confirm the simplify=False fix: signed offsets keep their '-'.
    Uses a ±40″ FOV so arcsec offsets stay in a readable single- /
    double-digit range across both halves of the axes."""
    fig = plt.figure(figsize=(8, 7))
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0),
                        cdelt=0.4 / 3600.0, npix=200,
                        frame="ICRS", fig=fig)
    apply_offset_ticks(ax, ref_ra_deg=180.0, ref_dec_deg=0.0, unit="arcsec")
    ax.set_title("apply_offset_ticks — signed offsets relative to (180°, 0°)\n"
                 "negative ticks must show '-'", fontsize=10)
    fig.subplots_adjust(top=0.9)
    return fig


@_panel("ticks_04_anchored_offset_demo")
def render_anchored_offset_demo():
    """apply_anchored_offset: anchor tick in HMS + others in mas. ±40 mas
    FOV so the mas offsets stay in a small readable range."""
    fig = plt.figure(figsize=(8, 7))
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0),
                        cdelt=4e-4 / 3600.0, npix=200,
                        frame="ICRS", fig=fig)
    apply_anchored_offset(ax, ref_tick="center", unit="mas")
    ax.set_title("apply_anchored_offset — center tick in HMS, others in mas",
                 fontsize=10)
    fig.subplots_adjust(top=0.9)
    return fig


@_panel("ticks_05_frame_aware_defaults")
def render_frame_aware_defaults():
    """No style argument — frame is detected and a sensible default is chosen."""
    fig = plt.figure(figsize=(15, 5))
    frames = [
        ("ICRS", "ICRS — HMS / DMS auto-defaults"),
        ("Galactic", "Galactic — decimal degrees"),
        ("geocentrictrueecliptic", "Ecliptic — decimal degrees"),
    ]
    for idx, (frame, title) in enumerate(frames, start=1):
        ax = make_wcs_frame((1, 3, idx), projection="AIT", center=0,
                            frame=frame, fig=fig)
        format_ticklabels(ax)  # bare default — let frame detection kick in
        ax.set_title(title, fontsize=10)
    fig.suptitle("format_ticklabels with no style: frame-aware auto-defaults",
                 fontsize=12)
    fig.subplots_adjust(top=0.85, wspace=0.3)
    return fig


@_panel("ticks_06_minor_ticks")
def render_minor_ticks():
    """Minor ticks across the three tick-formatting code paths:

    1. **Sexagesimal (publication)** — major ticks come from
       astropy's auto-spacing locator, and astropy's stock
       minor-tick path already works there. Just toggle
       ``display_minor_ticks(True)`` + ``set_minor_frequency(N)``.
    2. **Offset (mas)** — ``apply_offset_ticks`` sets explicit
       cosine-corrected major positions via ``set_ticks(values=...)``,
       which short-circuits astropy's minor-locator to an empty
       array. The helper now installs an interpolating replacement
       so ``minor_ticks=True`` (the new default) renders 4 minor
       ticks between each pair of majors.
    3. **VLBI hybrid** — same set_ticks(values=...) code path on
       the lon axis; lat uses standard spacing. ``minor_ticks=True``
       is the default; opt out via ``minor_ticks=False``.

    Minor tick length comes from
    ``rcParams['xtick.minor.size']`` — shorter than the major
    default, as users expect.
    """
    from skyplothelper.ticks import apply_anchored_offset, apply_offset_ticks

    fig = plt.figure(figsize=(15, 5))

    # --- Cell 1: sexagesimal style (publication)
    ax1 = make_wcs_frame((1, 3, 1), projection="TAN",
                         center=(180.0, 30.0),
                         cdelt=10.0 / 3600.0, npix=200,
                         frame="ICRS", fig=fig)
    _safe_apply(ax1, style="publication")
    for ci in (0, 1):
        ax1.coords[ci].display_minor_ticks(True)
        ax1.coords[ci].set_minor_frequency(4)
    ax1.set_title("publication (sexagesimal) — minor_freq=4",
                  fontsize=10)

    # --- Cell 2: offset style
    ax2 = make_wcs_frame((1, 3, 2), projection="TAN",
                         center=(83.6, 22.0),
                         cdelt=0.01 / 3600.0, npix=500, fig=fig)
    apply_offset_ticks(ax2, unit="mas")  # minor_ticks=True default
    ax2.set_title("apply_offset_ticks(unit='mas') — default minor",
                  fontsize=10)

    # --- Cell 3: VLBI hybrid
    ax3 = make_wcs_frame((1, 3, 3), projection="TAN",
                         center=(180.0, 0.0),
                         cdelt=4e-4 / 3600.0, npix=200,
                         frame="ICRS", fig=fig)
    apply_anchored_offset(ax3, ref_tick="center", unit="mas")
    ax3.set_title("apply_anchored_offset — default minor", fontsize=10)

    fig.suptitle("minor ticks — sexagesimal / offset / anchored_offset",
                 fontsize=12)
    fig.subplots_adjust(top=0.85, wspace=0.35)
    return fig


def main():
    banner("ticks — merge-verification visual gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
