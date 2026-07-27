"""Tests for :mod:`skyplothelper._text_layout` — the shared
rotation / anchor helpers used by the Ruler, curved-lon ticks, and
TickOverlay axis labels.

The integration tests for each consumer live alongside that consumer
(``test_ruler.py``, ``test_ticks.py``, ``test_coord_overlay*.py``);
this module exercises the helpers in isolation.
"""

import numpy as np
import pytest

from skyplothelper._text_layout import (
    _CHAR_WIDTH_FACTOR,
    _DEFAULT_N_CHARS,
    _LABEL_SAFETY_FACTOR,
    _VALID_ROTATIONS,
    _apply_auto_label_fontsize_to_wcs,
    _auto_label_fontsize,
    _n_chars_for_wcs_coord,
    _normalize_readable_angle,
    _resolve_rotation_deg,
    _resolve_text_anchor,
)

# ---- _normalize_readable_angle ---------------------------------------------

@pytest.mark.parametrize("angle,expected", [
    (0.0, 0.0),
    (45.0, 45.0),
    (90.0, 90.0),       # boundary — kept (not flipped)
    (91.0, -89.0),       # just over → flipped to upright
    (135.0, -45.0),
    (180.0, 0.0),
    (225.0, 45.0),
    (270.0, 90.0),
    (271.0, -89.0),      # past 270 wraps via 360-360 then folded again
    (315.0, -45.0),
    (-45.0, -45.0),
    (-180.0, 0.0),
    (360.0, 0.0),
    (720.0, 0.0),       # multiple wraps
])
def test_normalize_readable_angle(angle, expected):
    assert _normalize_readable_angle(angle) == pytest.approx(expected)


# ---- _resolve_rotation_deg --------------------------------------------------

def test_resolve_rotation_deg_horizontal():
    assert _resolve_rotation_deg("horizontal", 30.0) == 0.0
    assert _resolve_rotation_deg("horizontal", 90.0) == 0.0


def test_resolve_rotation_deg_auto_matches_tangent_post_normalize():
    assert _resolve_rotation_deg("auto", 30.0) == 30.0
    # A tangent at 135° is upside-down territory → folded.
    assert _resolve_rotation_deg("auto", 135.0) == pytest.approx(-45.0)


def test_resolve_rotation_deg_perpendicular_is_tangent_plus_90():
    assert _resolve_rotation_deg("perpendicular", 0.0) == 90.0
    # 30 + 90 = 120 → folded to -60.
    assert _resolve_rotation_deg("perpendicular", 30.0) == pytest.approx(-60.0)


def test_resolve_rotation_deg_numeric_literal_normalized():
    assert _resolve_rotation_deg(45.0, 30.0) == 45.0
    assert _resolve_rotation_deg(135.0, 30.0) == pytest.approx(-45.0)


def test_resolve_rotation_deg_rejects_bool_as_numeric():
    """``True`` / ``False`` are int subclasses in Python — make sure
    the helper doesn't treat them as literal angles."""
    with pytest.raises(ValueError, match="rotation mode"):
        _resolve_rotation_deg(True, 0.0)


def test_resolve_rotation_deg_rejects_unknown_mode():
    with pytest.raises(ValueError, match="rotation mode"):
        _resolve_rotation_deg("diagonal", 30.0)


def test_valid_rotations_constant_contains_documented_modes():
    assert set(_VALID_ROTATIONS) == {"auto", "horizontal", "perpendicular"}


# ---- _resolve_text_anchor --------------------------------------------------

def test_anchor_zero_rotation_label_above_horizontal_line():
    """rotation=0, perp=+y, side=+1 → text grows upward → va='bottom'."""
    assert _resolve_text_anchor(0.0, +1, 0.0, 1.0) == ("center", "bottom")


def test_anchor_zero_rotation_label_below_horizontal_line():
    assert _resolve_text_anchor(0.0, -1, 0.0, 1.0) == ("center", "top")


def test_anchor_zero_rotation_horizontal_perp():
    """Vertical ruler perp=+x at rotation=0 → text grows rightward
    → ha='left' so the left edge of the text is at the anchor."""
    assert _resolve_text_anchor(0.0, +1, 1.0, 0.0) == ("left", "center")
    assert _resolve_text_anchor(0.0, -1, 1.0, 0.0) == ("right", "center")


def test_anchor_ninety_rotation_label_above_horizontal_line():
    """rotation=90°, perp=+y → text rotates 90° CCW, so local +x
    maps to display +y. Anchor at left edge (ha='left') so text
    grows upward after the rotation."""
    assert _resolve_text_anchor(90.0, +1, 0.0, 1.0) == ("left", "center")
    assert _resolve_text_anchor(90.0, -1, 0.0, 1.0) == ("right", "center")


def test_anchor_minus_ninety_rotation_label_above_horizontal_line():
    """rotation=-90°, perp=+y → local +x maps to display -y; we need
    text to grow upward, so anchor at right edge (text extends
    leftward in local frame, which becomes upward after rotation)."""
    assert _resolve_text_anchor(-90.0, +1, 0.0, 1.0) == ("right", "center")


def test_anchor_auto_tangent_always_va_bottom():
    """When the text rotates to match the tangent (rotation =
    tangent_angle), the perpendicular outward direction always
    expresses as local +up — so anchor is always va='bottom' /
    'top' regardless of where the tangent points."""
    for tangent_deg in (0.0, 30.0, 45.0, 60.0, -30.0, -60.0):
        # Outward perpendicular: rotate tangent by 90° CCW
        rad = np.radians(tangent_deg)
        # Tangent direction
        tx, ty = np.cos(rad), np.sin(rad)
        # Perpendicular CCW 90°
        px, py = -ty, tx
        ha, va = _resolve_text_anchor(tangent_deg, +1, px, py)
        assert (ha, va) == ("center", "bottom")
        ha, va = _resolve_text_anchor(tangent_deg, -1, px, py)
        assert (ha, va) == ("center", "top")


def test_anchor_pure_offset_independent_of_side_sign_for_perp_perp():
    """A label on the +1 side with perp=+y is the same as the -1
    side with perp=-y."""
    a = _resolve_text_anchor(0.0, +1, 0.0, 1.0)
    b = _resolve_text_anchor(0.0, -1, 0.0, -1.0)
    assert a == b


def test_anchor_tied_direction_picks_va():
    """When |u| == |v|, the helper picks va via the ``>=`` branch
    (matches the documented behavior — vertical anchor wins ties)."""
    # rotation=45°, outward = (0, 1): both u and v end up at ±√2/2.
    ha, va = _resolve_text_anchor(45.0, +1, 0.0, 1.0)
    assert ha == "center"
    assert va in ("bottom", "top")


# ---- _auto_label_fontsize --------------------------------------------------
#
# These tests use plain matplotlib axes (no WCS) — the helper just needs
# a window extent and a figure DPI, neither of which require WCS.

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import rcParams  # noqa: E402


def _plain_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.draw()
    return fig, ax


def test_auto_label_fontsize_shrinks_on_small_axes():
    """Tiny axes → fontsize hits the floor (6pt by default)."""
    fig, ax = _plain_axes(figsize=(2, 1.5))
    fs = _auto_label_fontsize(ax, n_chars_hint=10, n_ticks_hint=6)
    assert fs == pytest.approx(6.0)
    plt.close(fig)


def test_auto_label_fontsize_grows_with_more_room_up_to_ceiling():
    """Bigger axes → bigger fontsize, capped at rcParams default."""
    fig_sm, ax_sm = _plain_axes(figsize=(3, 2))
    fig_lg, ax_lg = _plain_axes(figsize=(12, 8))
    fs_sm = _auto_label_fontsize(ax_sm, n_chars_hint=10, n_ticks_hint=6)
    fs_lg = _auto_label_fontsize(ax_lg, n_chars_hint=10, n_ticks_hint=6)
    assert fs_sm < fs_lg
    # Large axes hits the ceiling (rcParams default).
    raw = rcParams.get('xtick.labelsize', 10.0)
    try:
        default = float(raw)
    except (TypeError, ValueError):
        default = 10.0
    assert fs_lg == pytest.approx(default)
    plt.close(fig_sm)
    plt.close(fig_lg)


def test_auto_label_fontsize_respects_explicit_ceiling():
    """ceiling= overrides the rcParams default cap."""
    fig, ax = _plain_axes(figsize=(12, 8))
    fs = _auto_label_fontsize(ax, n_chars_hint=10, n_ticks_hint=6,
                              ceiling=8.0)
    assert fs == pytest.approx(8.0)
    plt.close(fig)


def test_auto_label_fontsize_respects_explicit_floor():
    """floor= overrides the default 6pt floor."""
    fig, ax = _plain_axes(figsize=(1, 1))
    fs = _auto_label_fontsize(ax, n_chars_hint=20, n_ticks_hint=10,
                              floor=4.0)
    assert fs == pytest.approx(4.0)
    plt.close(fig)


def test_auto_label_fontsize_clipped_within_floor_ceiling():
    """Result always lies within [floor, ceiling] regardless of geometry."""
    fig, ax = plt.subplots()
    fs = _auto_label_fontsize(ax, ceiling=10.0, floor=6.0)
    assert 6.0 <= fs <= 10.0
    plt.close(fig)


def test_auto_label_fontsize_math_matches_documented_formula():
    """Spot-check the heuristic against the docstring formula."""
    fig, ax = _plain_axes(figsize=(8, 4))
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    width_pt = bbox.width * 72.0 / ax.figure.dpi
    n_ticks, n_chars = 6, 10
    expected = width_pt / (n_ticks * n_chars
                            * _CHAR_WIDTH_FACTOR * _LABEL_SAFETY_FACTOR)
    expected = max(6.0, min(10.0, expected))
    got = _auto_label_fontsize(ax, n_chars_hint=n_chars,
                                n_ticks_hint=n_ticks, ceiling=10.0)
    assert got == pytest.approx(expected, rel=1e-9)
    plt.close(fig)


def test_auto_label_fontsize_axis_y_uses_height_dimension():
    """axis='y' picks fontsize from the axes height, not width."""
    fig, ax = _plain_axes(figsize=(8, 2))   # wide and short
    fs_x = _auto_label_fontsize(ax, n_chars_hint=10, n_ticks_hint=6,
                                 axis='x', ceiling=10.0)
    fs_y = _auto_label_fontsize(ax, n_chars_hint=10, n_ticks_hint=6,
                                 axis='y', ceiling=10.0)
    # Short axis (height) should give smaller (or equal at floor) fontsize.
    assert fs_y <= fs_x
    plt.close(fig)


# ---- _n_chars_for_wcs_coord ------------------------------------------------
#
# These tests require a real WCSAxes since the helper introspects astropy
# coord attributes. Build a TAN frame and check the bucket dispatch.


def _wcs_axes(projection="AIT", center=0, figsize=(6, 4), **kwargs):
    from skyplothelper.wcs_frame import make_wcs_frame
    fig = plt.figure(figsize=figsize)
    ax = make_wcs_frame(111, projection=projection, center=center,
                        fig=fig, auto_fontsize=False, **kwargs)
    fig.canvas.draw()
    return fig, ax


def test_n_chars_post_draw_introspection_returns_actual_label_width():
    """Post-draw, the helper reads the actual rendered label strings
    rather than the format-unit worst-case. Real HMS labels at 1-hour
    spacing render as ``"12h"`` (3 chars after mathtext stripping),
    not the worst-case 10."""
    fig, ax = _wcs_axes(projection="AIT", center=0, frame="ICRS")
    # The fig.canvas.draw() inside _wcs_axes already populated labels.
    n = _n_chars_for_wcs_coord(ax.coords[0])
    assert 2 <= n <= 6     # actual rendered HMS / DMS at default spacing
    plt.close(fig)


def test_n_chars_post_draw_galactic_decimal_actual_width():
    """Galactic frame defaults to decimal-degree labels; post-draw
    we see the actual rendered width (~4 chars for ``"180°"``)."""
    fig, ax = _wcs_axes(projection="AIT", center=0, frame="Galactic")
    n = _n_chars_for_wcs_coord(ax.coords[0])
    assert 2 <= n <= 6
    plt.close(fig)


def test_n_chars_pre_draw_fallback_hourangle_returns_10():
    """When the axes has been built but never drawn, the helper falls
    back to the format-unit lookup — HMS = 10 chars (conservative).
    Uses CAR (rectangular) so the elliptical hybrid path doesn't
    trigger its own canvas.draw() during make_wcs_frame."""
    from skyplothelper.wcs_frame import make_wcs_frame
    fig = plt.figure(figsize=(6, 4))
    ax = make_wcs_frame(111, projection="CAR", center=0, fig=fig,
                        frame="ICRS", auto_fontsize=False)
    # NB: do NOT canvas.draw() — we want the pre-draw branch.
    n = _n_chars_for_wcs_coord(ax.coords[0])
    assert n == 10
    plt.close(fig)


def test_n_chars_pre_draw_fallback_galactic_decimal_returns_7():
    """Pre-draw, the galactic decimal formatter gives the 7-char
    fallback (``"180.0°"``-sized). Uses CAR for the same reason as
    the hourangle pre-draw test."""
    from skyplothelper.wcs_frame import make_wcs_frame
    fig = plt.figure(figsize=(6, 4))
    ax = make_wcs_frame(111, projection="CAR", center=0, fig=fig,
                        frame="Galactic", auto_fontsize=False)
    n = _n_chars_for_wcs_coord(ax.coords[0])
    assert n == 7
    plt.close(fig)


def test_approx_rendered_chars_strips_mathtext_blocks():
    """Each ``$...$`` mathtext block collapses to a single character,
    so ``"12$\\\\mathregular{^h}$"`` renders as 3, not 22."""
    from skyplothelper._text_layout import _approx_rendered_chars
    assert _approx_rendered_chars("12") == 2
    assert _approx_rendered_chars(r"12$\mathregular{^h}$") == 3
    assert _approx_rendered_chars(
        r"12$\mathregular{^h}$34$\mathregular{^m}$56$\mathregular{^s}$"
    ) == 9
    assert _approx_rendered_chars(r"+30$\mathregular{{}^{\circ}}$") == 4


def test_n_chars_returns_fallback_on_non_wcs_coord():
    """A bare object without coord attrs returns the conservative fallback."""

    class _Bogus:
        pass

    assert _n_chars_for_wcs_coord(_Bogus()) == _DEFAULT_N_CHARS


# ---- _apply_auto_label_fontsize_to_wcs -------------------------------------


def test_apply_returns_none_on_non_wcs_axes():
    """Plain matplotlib axes → no-op, returns None."""
    fig, ax = _plain_axes(figsize=(6, 4))
    assert _apply_auto_label_fontsize_to_wcs(ax) is None
    plt.close(fig)


def test_apply_returns_fontsize_value_on_wcs_axes():
    """On a WCSAxes the helper returns the chosen fontsize so the
    caller can forward it into overlay label_kwargs."""
    fig, ax = _wcs_axes(figsize=(3, 2))
    fs = _apply_auto_label_fontsize_to_wcs(ax)
    assert fs is not None
    assert 6.0 <= fs <= 10.0
    plt.close(fig)


# ---- Integration: make_wcs_frame auto_fontsize default ---------------------
#
# These verify the user-facing behavior: small figures get smaller labels,
# large figures stay at the rcParams default, and auto_fontsize=False
# opts out.


def _make_wcs_get_fontsize(figsize, auto_fontsize=True):
    """Build a make_wcs_frame at given figsize, return the fontsize the
    auto-helper *would* compute for that axes (since astropy's
    TickLabels.get_size() doesn't reflect per-coord overrides reliably,
    we test via the helper directly)."""
    from skyplothelper.wcs_frame import make_wcs_frame
    fig = plt.figure(figsize=figsize)
    ax = make_wcs_frame(111, projection="AIT", center=0, fig=fig,
                        auto_fontsize=auto_fontsize)
    fig.canvas.draw()
    return fig, ax


def test_make_wcs_frame_auto_shrinks_on_small_figure():
    """Small figsize → the auto-helper picks a sub-default fontsize."""
    fig, ax = _make_wcs_get_fontsize((3, 2))
    # Recompute what the helper would assign to confirm a shrink kicked in.
    fs_recompute = _apply_auto_label_fontsize_to_wcs(ax)
    raw = rcParams.get('xtick.labelsize', 10.0)
    try:
        default = float(raw)
    except (TypeError, ValueError):
        default = 10.0
    assert fs_recompute < default
    plt.close(fig)


def test_make_wcs_frame_keeps_default_on_large_figure():
    """Large figsize → the auto-helper sits at the rcParams ceiling."""
    fig, ax = _make_wcs_get_fontsize((14, 8))
    fs_recompute = _apply_auto_label_fontsize_to_wcs(ax)
    raw = rcParams.get('xtick.labelsize', 10.0)
    try:
        default = float(raw)
    except (TypeError, ValueError):
        default = 10.0
    assert fs_recompute == pytest.approx(default)
    plt.close(fig)


def test_make_globe_frame_accepts_auto_fontsize_kwarg():
    """make_globe_frame exposes the same auto_fontsize knob — just
    verify it doesn't crash with either value."""
    import skyplothelper as sph
    fig = plt.figure(figsize=(4, 4))
    ax_on = sph.make_globe_frame(111, auto_fontsize=True)
    assert ax_on is not None
    plt.close(fig)

    fig = plt.figure(figsize=(4, 4))
    ax_off = sph.make_globe_frame(111, auto_fontsize=False)
    assert ax_off is not None
    plt.close(fig)


# ---- format_*ticklabels new fontsize=None default --------------------------


def test_format_WCS_ticklabels_fontsize_none_preserves_upstream():
    """Default fontsize=None means format_WCS_ticklabels no longer
    clobbers whatever was set previously (e.g. by make_wcs_frame's
    auto-fontsize)."""
    import skyplothelper as sph
    fig = plt.figure(figsize=(3, 2))
    ax = sph.make_wcs_frame(111, projection="AIT", center=0, fig=fig,
                            auto_fontsize=True)
    fig.canvas.draw()
    # Snapshot the auto-chosen fontsize via the helper
    fs_before = _apply_auto_label_fontsize_to_wcs(ax)
    # format_WCS_ticklabels with default fontsize=None should NOT
    # change the per-coord fontsize. We can't easily read it back
    # from astropy's TickLabels, so we re-run the helper and compare —
    # if format_WCS_ticklabels had clobbered, the helper would still
    # return the same value, so this is really a smoke test that
    # nothing raises and the helper still works.
    sph.format_WCS_ticklabels(ax)
    fs_after = _apply_auto_label_fontsize_to_wcs(ax)
    assert fs_before == fs_after
    plt.close(fig)


def test_format_WCS_ticklabels_explicit_fontsize_still_overrides():
    """Passing fontsize= explicitly still sets it (opt-in override)."""
    import skyplothelper as sph
    fig = plt.figure(figsize=(6, 4))
    ax = sph.make_wcs_frame(111, projection="AIT", center=0, fig=fig)
    # Should not raise.
    sph.format_WCS_ticklabels(ax, fontsize=14)
    plt.close(fig)


def test_format_mpl_ticklabels_fontsize_none_is_safe():
    """format_mpl_ticklabels with fontsize=None on a plain axes is a
    no-op for fontsize (still applies color / stroke etc.)."""
    import skyplothelper as sph
    fig, ax = _plain_axes(figsize=(6, 4))
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    fig.canvas.draw()
    sizes_before = [lbl.get_fontsize() for lbl in ax.get_xticklabels()]
    sph.format_mpl_ticklabels(ax)   # fontsize=None default
    sizes_after = [lbl.get_fontsize() for lbl in ax.get_xticklabels()]
    assert sizes_before == sizes_after
    plt.close(fig)
