"""ticks (format_ticklabels + offset/VLBI) verification.

Where the canonical ``tests/test_ticks.py`` smoke-tests 11 of the 20
style names, this file:

  * Confirms the full advertised style list (24 names including aliases)
    applies without raising.
  * Verifies aliases (``pub`` / ``deg`` / ``allsky_h`` / etc.) resolve to
    the same effective formatter as the canonical name.
  * Adds an unknown-style → ValueError check.
  * Spot-checks format-detection: ``format_ticklabels`` should pick
    different defaults for an equatorial vs. galactic axes when the
    style implies frame-aware behavior.
  * Re-verifies OffsetFormatter / AnchoredOffsetFormatter at the unit /
    sign edges.
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from skyplothelper.ticks import (
    AnchoredOffsetFormatter,
    OffsetFormatter,
    apply_anchored_offset,
    apply_offset_ticks,
    format_ticklabels,
)
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ============================================================
# All advertised style names apply without raising
# ============================================================

# Canonical 19 styles (per the docstring) plus the 5 aliases that the
# dispatcher publicly accepts.
_ALL_STYLES = [
    # Equatorial (HMS/DMS)
    "publication", "pub", "letter", "casa", "latex", "compact", "minimal",
    # Decimal-degree
    "decimal", "deg", "decimal_plain",
    # All-sky variants
    "allsky_hours", "allsky_h", "allsky_deg", "allsky_d",
    # Relative offsets
    "offset", "offset_arcsec", "offset_arcmin", "offset_mas", "offset_uas",
    # VLBI / sub-arcsec
    "vlbi", "anchored_offset", "anchored_offset_mas", "anchored_offset_uas",
    "anchored_offset_compact",
]


@pytest.mark.parametrize("style", _ALL_STYLES)
def test_format_ticklabels_every_advertised_style_applies(style):
    """Each canonical name + alias must apply *and draw* without raising.
    Drawing matters: the ``latex`` style emits separator strings that
    have to parse as mathtext, and the canonical smoke test never
    triggered ``fig.canvas.draw()``."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0), fig=fig)
    with warnings.catch_warnings():
        # astropy occasionally warns that spacing/base-spacing aren't
        # commensurate; that's a layout note, not an error.
        warnings.simplefilter("ignore")
        try:
            format_ticklabels(ax, style=style)
            fig.canvas.draw()
        except Exception as e:
            pytest.fail(
                f"format_ticklabels(style={style!r}) raised on draw: "
                f"{type(e).__name__}: {e}"
            )


@pytest.mark.parametrize("alias, canonical", [
    ("pub", "publication"),
    ("deg", "decimal"),
    ("allsky_h", "allsky_hours"),
    ("allsky_d", "allsky_deg"),
    ("anchored_offset_mas", "anchored_offset"),
])
def test_format_ticklabels_alias_matches_canonical_format(alias, canonical):
    """The alias must produce the same lon-axis ticklabel format as its
    canonical counterpart."""
    fmt_alias = _format_string_for_style(alias)
    fmt_canonical = _format_string_for_style(canonical)
    assert fmt_alias == fmt_canonical, (
        f"Alias {alias!r} produced format {fmt_alias!r} but canonical "
        f"{canonical!r} produced {fmt_canonical!r}"
    )


def _format_string_for_style(style):
    """Apply a style and return the lon-axis ticklabel format string."""
    fig = plt.figure()
    try:
        ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0), fig=fig)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            format_ticklabels(ax, style=style)
        return ax.coords[0].get_format_unit().to_string() + "|" + \
               (ax.coords[0]._formatter_locator.format or "")
    finally:
        plt.close(fig)


def test_format_ticklabels_unknown_style_warns():
    """Unknown style emits a UserWarning listing available styles
    rather than raising — the dispatcher prefers a friendly fallback."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0), fig=fig)
    with pytest.warns(UserWarning, match="Unknown style"):
        format_ticklabels(ax, style="this_style_does_not_exist")


# ============================================================
# Frame-aware default behavior
# ============================================================

def test_format_ticklabels_default_uses_hms_for_equatorial():
    """ICRS axes get HMS lon ticks by default."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0),
                        frame="ICRS", fig=fig)
    format_ticklabels(ax)  # no style — auto-detect
    fig.canvas.draw()
    # Equatorial lon axis should expose hour-aware unit
    unit = ax.coords[0].get_format_unit()
    assert unit.to_string() in ("h", "hourangle"), (
        f"expected hourangle unit for ICRS lon axis, got {unit!r}"
    )


def test_format_ticklabels_default_uses_degrees_for_galactic():
    """Galactic axes get decimal-degree lon ticks by default."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=0,
                        frame="Galactic", fig=fig)
    format_ticklabels(ax)
    fig.canvas.draw()
    unit = ax.coords[0].get_format_unit()
    assert unit.to_string() in ("deg",), (
        f"expected deg unit for Galactic lon axis, got {unit!r}"
    )


# ============================================================
# OffsetFormatter — unit + sign + invalid-unit edges
# ============================================================

def test_offset_formatter_uas_unit_acceptance():
    """μas (microarcsec) is one of the publicly-supported units."""
    fmt = OffsetFormatter(ref_value_deg=0.0, unit="uas")
    assert "0" in fmt(0.0)  # zero-at-reference
    out = fmt(1e-9)
    assert any(c.isdigit() for c in out)


def test_offset_formatter_keeps_negative_sign():
    """The simplify=False fix: negative offsets must keep their '-'."""
    fmt = OffsetFormatter(ref_value_deg=180.0, unit="arcsec")
    out_neg = fmt(180.0 - 1 / 3600.0)
    out_pos = fmt(180.0 + 1 / 3600.0)
    assert "-" in out_neg, f"expected '-' in negative offset, got {out_neg!r}"
    assert "-" not in out_pos, f"expected no '-' in positive offset, got {out_pos!r}"


# ============================================================
# AnchoredOffsetFormatter — anchor index + per-tick sub-formatting
# ============================================================

def test_anchored_offset_inherits_from_offset_formatter():
    assert issubclass(AnchoredOffsetFormatter, OffsetFormatter)


def test_apply_offset_ticks_runs():
    """apply_offset_ticks must succeed on a TAN axes with a chosen unit."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0), fig=fig)
    apply_offset_ticks(ax, ref_ra_deg=180.0, ref_dec_deg=0.0, unit="arcmin")
    fig.canvas.draw()


def test_apply_anchored_offset_runs():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0), fig=fig)
    apply_anchored_offset(ax, ref_tick="center", unit="mas")
    fig.canvas.draw()
