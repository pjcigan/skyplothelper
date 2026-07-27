"""Render survey footprints and constellation overlays for visual eyeballing.

Produces:
  - surveys_01_imaging_optical_ir.png  — SDSS / DES / DECaLS / Pan-STARRS / KiDS / HSC / CFHTLS / Stripe 82
  - surveys_02_radio.png               — NVSS / FIRST / VLASS / LoTSS / SKA / RACS
  - surveys_03_x_ray_uv_hatlas.png     — eROSITA / H-ATLAS
  - surveys_04_lsst_euclid_desi_spt.png — LSST / Euclid / DESI / SPT
  - surveys_05_box_vs_polygon.png      — same survey rendered both ways
  - surveys_06_spectroscopic.png       — DESI / GAMA / 2dF / 6dF
  - surveys_07_deep_fields.png         — COSMOS / UDS / GOODS-N / GOODS-S (TAN zooms + locator)
  - constellations_01_default.png      — full IAU 88-constellation overlay (boundaries+abbr labels)
  - constellations_02_named_subset.png — subset rendering (5 famous ones)

Each render function is a pure builder: takes no arguments, returns
the matplotlib Figure. The ``PANELS`` registry maps panel filename
stem → builder function. ``main()`` iterates the registry calling
``save_or_show`` for each panel; the pytest-mpl visual-baseline
suite (``tests/visual_baselines/test_overlays_surveys_constellations.py``)
imports ``PANELS`` and consumes the same builders.
"""

import sys

import matplotlib.pyplot as plt
from _common import banner, save_or_show

from skyplothelper.overlays.constellations import (
    add_constellation_boundaries,
    add_constellation_labels,
    add_constellation_lines,
    add_constellation_polygon,
)
from skyplothelper.overlays.planes import add_plane_overlay
from skyplothelper.overlays.surveys import add_survey_footprint
from skyplothelper.wcs_frame import make_wcs_frame

# Builder registry — name → no-arg function that returns a Figure.
# Filled by the @_panel decorator below.
PANELS = {}


def _panel(name):
    """Register the decorated function as the builder for *name*."""
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


def _allsky():
    fig = plt.figure(figsize=(11, 5.5))
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    return fig, ax


@_panel("surveys_01_imaging_optical_ir")
def render_imaging_optical_ir():
    fig, ax = _allsky()
    for name in ("sdss", "des", "decals", "panstarrs", "kids", "hsc",
                 "cfhtls", "stripe82"):
        add_survey_footprint(ax, name, lw=1.2)
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.85)
    ax.set_title("Optical / IR imaging surveys: SDSS, DES, DECaLS, "
                 "Pan-STARRS, KiDS, HSC, CFHTLS, Stripe 82")
    return fig


@_panel("surveys_02_radio")
def render_radio_surveys():
    fig, ax = _allsky()
    for name in ("nvss", "first", "vlass", "lotss", "ska", "racs"):
        add_survey_footprint(ax, name, lw=1.2)
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.85)
    ax.set_title("Radio surveys: NVSS, FIRST, VLASS, LoTSS, SKA, RACS")
    return fig


@_panel("surveys_03_x_ray_uv_hatlas")
def render_x_ray_uv_other():
    """X-ray / sub-mm surveys with structured footprints.

    Drops the strict-all-sky entries (Planck / GALEX / WISE) that
    used to be on this panel — those just flooded the entire sky
    with a uniform fill and added no information about what the
    surveys actually cover. Kept here are the two surveys that
    have non-trivial footprints worth showing on AIT center=180:
    eROSITA (galactic-frame asymmetric, western hemisphere only)
    and H-ATLAS (a handful of small disconnected fields).
    """
    fig, ax = _allsky()
    color_overrides = {
        "erosita": "crimson",       # X-ray, hot
        "hatlas":  "forestgreen",   # sub-mm small fields
    }
    for name, color in color_overrides.items():
        add_survey_footprint(ax, name, lw=1.2, color=color)
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.85)
    ax.set_title("X-ray / sub-mm: eROSITA (galactic, western half), H-ATLAS")
    return fig


@_panel("surveys_04_lsst_euclid_desi_spt")
def render_future_surveys():
    fig, ax = _allsky()
    for name in ("lsst", "euclid", "desi", "spt"):
        add_survey_footprint(ax, name, lw=1.2)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    ax.set_title("Current/future wide-field surveys: LSST, Euclid, DESI, SPT")
    return fig


@_panel("surveys_05_box_vs_polygon")
def render_box_vs_polygon():
    fig = plt.figure(figsize=(15, 5.5))
    for col, style in enumerate(("box", "polygon"), start=1):
        ax = make_wcs_frame((1, 2, col), projection="AIT", center=180, fig=fig)
        add_survey_footprint(ax, "sdss", boundary_style=style, lw=1.2)
        add_survey_footprint(ax, "des", boundary_style=style, lw=1.2)
        ax.set_title(f"boundary_style={style!r}", fontsize=10)
    fig.suptitle("add_survey_footprint: 'box' (simple) vs 'polygon' "
                 "(detailed) boundary styles", fontsize=12)
    fig.subplots_adjust(top=0.85, wspace=0.3)
    return fig


@_panel("surveys_06_spectroscopic")
def render_spectroscopic_surveys():
    """Wide-field spectroscopic surveys on a single all-sky panel.

    Shows DESI (full-lon ICRS band), 6dF (southern hemisphere with
    galactic-plane avoidance — uses ``compound_ops``), 2dF (NGP +
    SGP strips), and GAMA (5 small fields). 6dF and DESI overlap
    in the south; alpha blending shows the intersection.
    """
    fig, ax = _allsky()
    for name in ("desi", "sixdf", "twodf", "gama"):
        add_survey_footprint(ax, name, lw=1.2)
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.85)
    ax.set_title("Wide-field spectroscopic surveys: DESI, 6dF, 2dF, GAMA")
    return fig


@_panel("surveys_07_deep_fields")
def render_deep_fields():
    """Iconic deep fields — too small to see on an all-sky frame, so
    a 2×2 layout: top-left is the all-sky locator (colored markers
    at each field center, since the actual ~0.04 sq deg footprints
    are sub-pixel on an all-sky AIT); the other three panels are
    TAN zooms onto three of the four fields with the actual
    survey footprint rendered as a polygon patch.
    """
    from skyplothelper.overlays.surveys import SURVEY_FOOTPRINTS
    fields = [
        ("cosmos", 150.12, 2.21, "COSMOS"),
        ("uds", 34.5, -5.0, "UDS"),
        ("goodsn", 189.23, 62.24, "GOODS-N"),
        ("goodss", 53.13, -27.81, "GOODS-S"),
    ]
    fig = plt.figure(figsize=(13, 9))

    # Panel 1: all-sky locator. The actual footprints are far below
    # 1 px² at all-sky scale, so they would be invisible. Draw a
    # colored locator marker (using each survey's registered color)
    # plus a small text label so the position is identifiable.
    ax = make_wcs_frame((2, 2, 1), projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    for name, lon, lat, label in fields:
        color = SURVEY_FOOTPRINTS[name].get('color', 'C0')
        ax.plot(lon, lat, transform=ax.get_transform("world"),
                marker="o", markersize=8, markerfacecolor=color,
                markeredgecolor="black", markeredgewidth=0.6,
                linestyle="none", label=label)
        ax.annotate(label, xy=(lon, lat),
                    xycoords=ax.get_transform("world"),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=8, color="black",
                    bbox={"boxstyle": "round,pad=0.2",
                          "fc": "white", "ec": "0.5", "alpha": 0.85})
    ax.set_title("All-sky locator (deep-field footprints are sub-pixel; "
                 "colored markers show location)",
                 fontsize=9)

    # Panels 2–4: TAN zooms onto three of the four fields.
    # Each zoom is ~3° across so the deep-field rect is clearly
    # visible against a graticule of nearby coordinates.
    for idx, (name, lon, lat, label) in enumerate(
            [fields[0], fields[2], fields[3]], start=2):
        ax = make_wcs_frame((2, 2, idx), projection="TAN",
                            center=(lon, lat), cdelt=0.01, fig=fig)
        fig.canvas.draw()
        add_survey_footprint(ax, name, lw=1.8)
        ax.set_title(f"{label} — TAN zoom @ ({lon:.2f}°, {lat:.2f}°)",
                     fontsize=9)

    fig.suptitle("Deep fields: COSMOS, UDS, GOODS-N, GOODS-S",
                 fontsize=12)
    fig.subplots_adjust(top=0.93, hspace=0.35, wspace=0.25)
    return fig


@_panel("constellations_01_default")
def render_constellations_default():
    fig, ax = _allsky()
    add_constellation_boundaries(ax, color="#666", lw=0.5, alpha=0.5)
    add_constellation_labels(ax, labels="abbr", fontsize=7,
                             color="#444", alpha=0.7)
    add_plane_overlay(ax, plane="galactic", color="C3", lw=1.0,
                      alpha=0.6)
    ax.set_title("All 88 IAU constellations: boundaries + abbreviation labels",
                 fontsize=11)
    return fig


@_panel("constellations_02_named_subset")
def render_constellations_named_subset():
    fig, ax = _allsky()
    famous = ["UMA", "UMI", "CAS", "CYG", "LYR", "ORI", "TAU", "GEM",
              "LEO", "VIR", "SCO", "SGR"]
    add_constellation_boundaries(ax, color="#aaa", lw=0.4, alpha=0.4)
    add_constellation_labels(ax, labels="name", fontsize=10,
                             color="C0", alpha=0.95,
                             constellations=famous)
    add_plane_overlay(ax, plane="ecliptic", color="C2", lw=1.0,
                      alpha=0.6, label="ecliptic")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax.set_title("Constellation labels (full Latin names) — 12 famous "
                 "constellations only", fontsize=11)
    return fig


@_panel("constellations_03_asterism_lines")
def render_constellations_asterism_lines():
    """All 88 IAU asterism lines plus boundaries and abbreviation
    labels — the full default overlay stack."""
    fig, ax = _allsky()
    add_constellation_boundaries(ax, color="#888", lw=0.3, alpha=0.45)
    add_constellation_lines(ax, color="#C7A86A", lw=0.6, alpha=0.85)
    add_constellation_labels(ax, labels="abbr", fontsize=6,
                              color="#666", alpha=0.7)
    ax.set_title("IAU asterism lines (rank-all) + boundaries + labels",
                  fontsize=11)
    return fig


@_panel("constellations_05_polygon_highlight")
def render_constellations_polygon_highlight():
    """Single-constellation polygon overlays via add_constellation_polygon
    — including Serpens (two polygons in one call: Caput + Cauda)."""
    fig, ax = _allsky()
    add_constellation_boundaries(ax, color="#bbb", lw=0.3, alpha=0.4)
    add_constellation_labels(ax, labels="abbr", fontsize=6,
                              color="#888", alpha=0.6)
    # A handful of constellations highlighted in different palettes.
    add_constellation_polygon(ax, "UMi", facecolor="lightblue",
                                edgecolor="steelblue", alpha=0.4, lw=1.2)
    add_constellation_polygon(ax, "Cyg", facecolor="gold",
                                edgecolor="#B8860B", alpha=0.35, lw=1.0)
    add_constellation_polygon(ax, "Cas", facecolor="C2", alpha=0.35)
    add_constellation_polygon(ax, "Ori", facecolor="salmon",
                                edgecolor="C3", alpha=0.4)
    add_constellation_polygon(ax, "Sco", facecolor="#7e57c2", alpha=0.4)
    # Serpens — passing 'ser' (lower-case) returns 2 patches (Caput + Cauda)
    add_constellation_polygon(ax, "ser", facecolor="peru",
                                edgecolor="saddlebrown",
                                alpha=0.45, lw=1.0)
    ax.set_title("add_constellation_polygon — multi-constellation highlight "
                  "(Ser = Caput + Cauda)", fontsize=11)
    return fig


@_panel("constellations_04_asterism_subset")
def render_constellations_asterism_subset():
    """Asterism lines for a curated 'famous' subset, drawn boldly
    in C0 against a faint boundary backdrop."""
    fig, ax = _allsky()
    famous = ["ORI", "UMA", "UMI", "CAS", "CYG", "LYR", "TAU", "GEM",
               "LEO", "VIR", "SCO", "SGR", "CRU", "CEN"]
    add_constellation_boundaries(ax, color="#bbb", lw=0.3, alpha=0.3)
    add_constellation_lines(ax, constellations=famous,
                              color="C0", lw=1.2, alpha=0.95)
    add_constellation_labels(ax, labels="abbr", fontsize=8,
                              color="C0", alpha=0.9,
                              constellations=famous)
    ax.set_title("Asterism lines — famous-subset rendering",
                  fontsize=11)
    return fig


def main():
    banner("overlays.surveys + overlays.constellations — gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
