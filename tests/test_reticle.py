"""Tests for the :class:`skyplothelper.Reticle` class.

Coverage: style geometry (plus / x / L / circle), rotation, coordinate
resolution (pixel / SkyCoord / (lon, lat) / frame=), auto label-side
heuristic, stroke effects, label compass-direction placement, factory
methods, setters, attach + remove lifecycle, and the top-level
``add_reticle`` convenience.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as PathEffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox  # noqa: E402
from matplotlib.text import Annotation  # noqa: E402

from skyplothelper.overlays.reticle import (  # noqa: E402
    _LABEL_DIRECTIONS,
    _VALID_LABEL_SIDES,
    _VALID_STYLES,
    Reticle,
    _outer_extent,
    _resolve_anchor,
    _resolve_auto_label_side,
    _reticle_segments,
    _rotate,
    add_reticle,
)

# ---- fixtures --------------------------------------------------------------

def _plain_axes():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    fig.canvas.draw()
    return fig, ax


def _wcs_axes(cdelt_asec=2.0, npix=100, center=(180.0, 0.0)):
    """A TAN-projected WCSAxes with a configurable pixel scale."""
    from skyplothelper.wcs_frame import make_wcs_frame
    cdelt_deg = cdelt_asec / 3600.0
    fig = plt.figure(figsize=(5, 5))
    ax = make_wcs_frame(111, projection="TAN", center=center,
                        cdelt=cdelt_deg, npix=(npix, npix), fig=fig)
    fig.canvas.draw()
    return fig, ax


# ---- _rotate ----------------------------------------------------------------

def test_rotate_identity_returns_unchanged():
    """rotation=0 returns the input pointwise — geometry tests can
    assume the canonical orientation is the literal one."""
    pts = [(1.0, 0.0), (0.0, 1.0)]
    assert _rotate(pts, 0) == pts


def test_rotate_90_ccw():
    """+90° rotation maps (1, 0) → (0, 1) and (0, 1) → (-1, 0)."""
    pts = [(1.0, 0.0), (0.0, 1.0)]
    got = _rotate(pts, 90)
    assert got[0] == pytest.approx((0.0, 1.0), abs=1e-9)
    assert got[1] == pytest.approx((-1.0, 0.0), abs=1e-9)


# ---- _reticle_segments ------------------------------------------------------

def test_plus_segments_canonical():
    """Plus has four arms at ±x/±y with inner endpoint at gap, outer at size."""
    segs = _reticle_segments("plus", size=10, gap=3, rotation=0,
                              circle_npts=64, circle_gap_deg=0)
    assert len(segs) == 4
    # N, S, E, W in that construction order.
    assert segs[0] == [(0, 3), (0, 10)]      # N
    assert segs[1] == [(0, -3), (0, -10)]    # S
    assert segs[2] == [(3, 0), (10, 0)]      # E
    assert segs[3] == [(-3, 0), (-10, 0)]    # W


def test_x_equals_plus_rotated_45():
    """X style is structurally plus rotated 45° — useful invariant for
    callers who want one or the other for visual reasons."""
    plus = _reticle_segments("plus", size=10, gap=3, rotation=45,
                              circle_npts=64, circle_gap_deg=0)
    x_segs = _reticle_segments("x", size=10, gap=3, rotation=0,
                                circle_npts=64, circle_gap_deg=0)
    assert len(plus) == len(x_segs)
    for a, b in zip(plus, x_segs):
        assert np.allclose(a, b)


def test_L_segments_canonical():
    """L (rotation=0) opens upper-right — arms point into W and S."""
    segs = _reticle_segments("L", size=10, gap=3, rotation=0,
                              circle_npts=64, circle_gap_deg=0)
    assert len(segs) == 2
    assert segs[0] == [(-3, 0), (-10, 0)]   # W
    assert segs[1] == [(0, -3), (0, -10)]   # S


def test_L_rotation_90_opens_upper_left():
    """+90° rotation walks the L's open quadrant CCW: UR → UL."""
    segs = _reticle_segments("L", size=10, gap=3, rotation=90,
                              circle_npts=64, circle_gap_deg=0)
    # After +90° CCW: W arm becomes S, S arm becomes E.
    assert segs[0][0] == pytest.approx((0, -3), abs=1e-9)
    assert segs[0][1] == pytest.approx((0, -10), abs=1e-9)
    assert segs[1][0] == pytest.approx((3, 0), abs=1e-9)
    assert segs[1][1] == pytest.approx((10, 0), abs=1e-9)


def test_circle_sampled_on_unit_circle():
    """Circle samples sit at exactly radius=size from the center."""
    segs = _reticle_segments("circle", size=10, gap=3, rotation=0,
                              circle_npts=24, circle_gap_deg=0)
    assert len(segs) == 1
    ring = segs[0]
    assert len(ring) == 24
    for (x, y) in ring:
        assert np.hypot(x, y) == pytest.approx(10.0, abs=1e-9)


def test_circle_gap_deg_removes_wedge():
    """circle_gap_deg=60 starts the arc at +30° and ends at +330°."""
    segs = _reticle_segments("circle", size=10, gap=3, rotation=0,
                              circle_npts=64, circle_gap_deg=60)
    ring = segs[0]
    # First / last sample should sit at ±30° from +x.
    first_theta = np.degrees(np.arctan2(ring[0][1], ring[0][0]))
    last_theta = np.degrees(np.arctan2(ring[-1][1], ring[-1][0]))
    assert first_theta == pytest.approx(30.0, abs=1e-6)
    # arctan2 wraps to (-180, 180]; -30° == 330° on the unbroken arc.
    assert last_theta == pytest.approx(-30.0, abs=1e-6)


def test_invalid_style_raises():
    with pytest.raises(ValueError, match="style must be one of"):
        _reticle_segments("triangle", size=10, gap=3, rotation=0,
                          circle_npts=64, circle_gap_deg=0)


# ---- _outer_extent ----------------------------------------------------------

def test_outer_extent_plus():
    """Plus outer extent equals size (arm tips sit at ±size on each axis)."""
    segs = _reticle_segments("plus", size=10, gap=3, rotation=0,
                              circle_npts=64, circle_gap_deg=0)
    assert _outer_extent(segs) == pytest.approx(10.0)


def test_outer_extent_circle():
    """Circle outer extent equals size (every ring point is at radius=size)."""
    segs = _reticle_segments("circle", size=15, gap=0, rotation=0,
                              circle_npts=32, circle_gap_deg=0)
    assert _outer_extent(segs) == pytest.approx(15.0)


def test_outer_extent_empty_returns_zero():
    """No segments → zero extent (no divide-by-zero in label placement)."""
    assert _outer_extent([]) == 0.0


# ---- _resolve_anchor --------------------------------------------------------

def test_resolve_anchor_pixel_path():
    """coord_type='pixel' returns the tuple unchanged + ax.transData."""
    fig, ax = _plain_axes()
    x, y, tr = _resolve_anchor((20.0, 50.0), ax, "pixel", frame=None)
    assert x == 20.0 and y == 50.0
    assert tr is ax.transData
    plt.close(fig)


def test_resolve_anchor_world_tuple_no_frame_uses_get_transform():
    """A bare (lon, lat) tuple on a WCSAxes hands off to
    ax.get_transform('world') so astropy projects natively."""
    fig, ax = _wcs_axes()
    x, y, tr = _resolve_anchor((180.0, 0.0), ax, "world", frame=None)
    # tuple values passed through verbatim
    assert x == 180.0 and y == 0.0
    # transform must be the world transform (not transData)
    assert tr is not ax.transData
    plt.close(fig)


def test_resolve_anchor_skycoord_projects_via_wcs():
    """A SkyCoord is projected to pixel coords via ax.wcs.world_to_pixel
    so cross-frame input is handled automatically."""
    fig, ax = _wcs_axes(center=(180.0, 0.0))
    sc = SkyCoord(180.0, 0.0, unit="deg", frame="icrs")
    x, y, tr = _resolve_anchor(sc, ax, "world", frame=None)
    # Centered WCS should put (180, 0) near the pixel center.
    assert x == pytest.approx(49.5, abs=2.0)
    assert y == pytest.approx(49.5, abs=2.0)
    assert tr is ax.transData
    plt.close(fig)


def test_resolve_anchor_world_tuple_with_frame_wraps_as_skycoord():
    """frame='galactic' on a tuple round-trips through SkyCoord, projects
    to pixel coords, returns ax.transData."""
    fig, ax = _wcs_axes(center=(180.0, 0.0))
    x, y, tr = _resolve_anchor((266.4, -29.0), ax, "world",
                                frame="galactic")
    # Sgr A* in galactic projected to an ICRS TAN frame around (180, 0)
    # won't land on the axes, but the math should still execute and
    # produce a finite pixel coord.
    assert np.isfinite(x) and np.isfinite(y)
    assert tr is ax.transData
    plt.close(fig)


def test_resolve_anchor_skycoord_vector_raises():
    """Reticles mark one point; vector SkyCoord input is rejected early."""
    fig, ax = _wcs_axes()
    sc = SkyCoord([180.0, 181.0], [0.0, 1.0], unit="deg")
    with pytest.raises(ValueError, match="scalar"):
        _resolve_anchor(sc, ax, "world", frame=None)
    plt.close(fig)


def test_resolve_anchor_world_tuple_on_plain_axes_raises():
    """Non-WCS axes can't interpret world coords — error suggests
    coord_type='pixel' instead."""
    fig, ax = _plain_axes()
    with pytest.raises(ValueError, match="coord_type='pixel'"):
        _resolve_anchor((50, 50), ax, "world", frame=None)
    plt.close(fig)


def test_resolve_anchor_invalid_coord_type_raises():
    fig, ax = _plain_axes()
    with pytest.raises(ValueError, match="coord_type must be"):
        _resolve_anchor((50, 50), ax, "data", frame=None)
    plt.close(fig)


# ---- _resolve_auto_label_side ----------------------------------------------

def test_auto_label_side_picks_corner_with_most_room():
    """A reticle near the lower-left should auto-label toward NE."""
    fig, ax = _plain_axes()
    assert _resolve_auto_label_side(ax, 10, 10, ax.transData) == "NE"
    assert _resolve_auto_label_side(ax, 90, 90, ax.transData) == "SW"
    assert _resolve_auto_label_side(ax, 90, 10, ax.transData) == "NW"
    assert _resolve_auto_label_side(ax, 10, 90, ax.transData) == "SE"
    plt.close(fig)


def test_auto_label_side_only_returns_corners():
    """The heuristic chooses among NE/NW/SE/SW — never a pure-axis
    direction. Keeps label placement visually balanced."""
    fig, ax = _plain_axes()
    for (x, y) in [(50, 50), (25, 75), (75, 25), (10, 10)]:
        side = _resolve_auto_label_side(ax, x, y, ax.transData)
        assert side in {"NE", "NW", "SE", "SW"}
    plt.close(fig)


# ---- Reticle construction validation ---------------------------------------

def test_invalid_style_at_construction_raises():
    with pytest.raises(ValueError, match="style must be"):
        Reticle((0, 0), coord_type="pixel", style="triangle")


def test_invalid_label_side_at_construction_raises():
    with pytest.raises(ValueError, match="label_side must be"):
        Reticle((0, 0), coord_type="pixel", label_side="up")


def test_style_alias_plus_symbol():
    """'+' is accepted as a glyph-shaped alias for 'plus'."""
    r = Reticle((0, 0), coord_type="pixel", style="+")
    assert r._style == "plus"


def test_style_alias_circle_letter():
    """'o' is accepted as a glyph-shaped alias for 'circle'."""
    r = Reticle((0, 0), coord_type="pixel", style="o")
    assert r._style == "circle"


def test_style_alias_renders_same_as_canonical():
    """Reticle built with an alias has the same arm count as the
    canonical form — i.e., the alias is just a name swap, not a
    different code path."""
    fig, ax = _plain_axes()
    r_alias = add_reticle(ax, (50, 50), coord_type="pixel", style="+")
    r_canonical = add_reticle(ax, (50, 50), coord_type="pixel",
                              style="plus")
    assert len(r_alias.arm_artists) == len(r_canonical.arm_artists) == 4
    plt.close(fig)


def test_invalid_style_error_mentions_aliases():
    """The validation error tells callers about the alias menu too,
    so they discover '+' / 'o' without having to read the docstring."""
    with pytest.raises(ValueError, match=r"aliases.*\+.*o"):
        Reticle((0, 0), coord_type="pixel", style="triangle")


def test_valid_styles_constant_matches_expectations():
    """Guard against accidental drift in the public style menu."""
    assert _VALID_STYLES == ("plus", "x", "L", "circle")


def test_valid_label_sides_constant_contains_eight_compass_points():
    """label_side accepts auto + 8 compass directions."""
    assert set(_VALID_LABEL_SIDES) == {
        "auto", "N", "NE", "E", "SE", "S", "SW", "W", "NW"
    }


# ---- add_to / remove --------------------------------------------------------

def test_add_to_creates_offsetbox_and_arms():
    """Attaching the reticle adds one AnchoredOffsetbox to the axes and
    populates ``arm_artists`` with one Line2D per segment."""
    fig, ax = _plain_axes()
    r = Reticle((50, 50), coord_type="pixel").add_to(ax)
    assert isinstance(r._anchor_box, AnchoredOffsetbox)
    assert len(r.arm_artists) == 4   # plus = 4 arms
    for arm in r.arm_artists:
        assert isinstance(arm, Line2D)
    plt.close(fig)


def test_remove_detaches_all_artists():
    """remove() detaches the offsetbox and (when present) the label
    annotation. Calling it twice is a no-op."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel", label="hi")
    anchor_before = r._anchor_box
    label_before = r.label_artist
    assert anchor_before in ax.artists or anchor_before in ax.get_children()
    assert label_before is not None

    r.remove()
    assert r._anchor_box is None
    assert r.label_artist is None
    assert r.arm_artists == []
    # Second remove should be a no-op (idempotent).
    r.remove()
    plt.close(fig)


def test_circle_creates_single_arm_artist():
    """Circle style = one polyline (the ring)."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel", style="circle")
    assert len(r.arm_artists) == 1
    plt.close(fig)


def test_L_creates_two_arms():
    """L = 2 arms."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel", style="L")
    assert len(r.arm_artists) == 2
    plt.close(fig)


# ---- Stroke effects --------------------------------------------------------

def test_default_stroke_applies_withstroke_path_effect():
    """Default stroke_color='black' wires PathEffects.withStroke onto
    each arm — this is what gives the dark-sky-readable stroke."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel")
    for arm in r.arm_artists:
        effects = arm.get_path_effects()
        assert any(isinstance(e, PathEffects.withStroke) for e in effects)
    plt.close(fig)


def test_stroke_color_none_disables_path_effects():
    """stroke_color=None opts out of the stroke entirely."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel", stroke_color=None)
    for arm in r.arm_artists:
        assert arm.get_path_effects() == []
    plt.close(fig)


def test_color_default_is_white():
    """The dark-sky default — white body — is the documented contract."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel")
    for arm in r.arm_artists:
        assert arm.get_color() == "white"
    plt.close(fig)


# ---- Label placement -------------------------------------------------------

def test_label_creates_annotation():
    """label='...' adds an Annotation artist; label_artist exposes it."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel", label="target")
    assert isinstance(r.label_artist, Annotation)
    assert r.label_artist.get_text() == "target"
    plt.close(fig)


def test_label_color_inherits_body_color_by_default():
    """Unspecified label_color picks up the body color so label + reticle
    visually go together."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel", color="cyan",
                    label="t")
    assert r.label_artist.get_color() == "cyan"
    plt.close(fig)


def test_label_color_explicit_overrides_body_color():
    """An explicit label_color decouples label from body color."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel", color="cyan",
                    label="t", label_color="magenta")
    assert r.label_artist.get_color() == "magenta"
    plt.close(fig)


@pytest.mark.parametrize("side,ha,va", [
    ("N",  "center", "bottom"),
    ("NE", "left",   "bottom"),
    ("E",  "left",   "center"),
    ("SE", "left",   "top"),
    ("S",  "center", "top"),
    ("SW", "right",  "top"),
    ("W",  "right",  "center"),
    ("NW", "right",  "bottom"),
])
def test_label_side_sets_correct_ha_va(side, ha, va):
    """Each compass-point label_side maps to the right text alignment
    so the label hugs the reticle on the chosen side."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel", label="t",
                    label_side=side)
    assert r.label_artist.get_ha() == ha
    assert r.label_artist.get_va() == va
    plt.close(fig)


def test_label_side_auto_picks_corner_per_anchor_position():
    """Different anchor positions resolve to different auto-corner sides
    (the data-bounds heuristic in action)."""
    fig, ax = _plain_axes()
    r_lower_left = add_reticle(ax, (10, 10), coord_type="pixel", label="A")
    r_upper_right = add_reticle(ax, (90, 90), coord_type="pixel", label="B")
    assert r_lower_left.resolved_label_side == "NE"
    assert r_upper_right.resolved_label_side == "SW"
    plt.close(fig)


def test_no_label_means_no_label_artist():
    """Skipping label= produces no annotation artist at all."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel")
    assert r.label_artist is None
    assert r.resolved_label_side is None
    plt.close(fig)


def test_label_direction_table_internal_consistency():
    """Each direction's sign pair matches an intuitive (dx, dy) compass:
    N is (+y only), E is (+x only), NE is (+x, +y), etc."""
    assert _LABEL_DIRECTIONS["N"]  == (0,  +1, "center", "bottom")
    assert _LABEL_DIRECTIONS["E"]  == (+1, 0, "left",   "center")
    assert _LABEL_DIRECTIONS["SE"] == (+1, -1, "left",   "top")


# ---- Factories -------------------------------------------------------------

def test_from_world_constructor_passes_through_kwargs():
    """from_world is a thin wrapper that pins coord_type='world'."""
    r = Reticle.from_world((180.0, 0.0), size=20)
    assert r._coord_type == "world"
    assert r._size == 20.0


def test_from_pixel_constructor_passes_through_kwargs():
    """from_pixel pins coord_type='pixel' and ignores frame=."""
    r = Reticle.from_pixel((10, 20), size=15, frame="galactic")
    assert r._coord_type == "pixel"
    assert r._size == 15.0


def test_from_world_overrides_caller_coord_type():
    """Even if caller passes coord_type='pixel', from_world wins —
    that's the whole point of the explicit factory."""
    r = Reticle.from_world((1, 2), coord_type="pixel")
    assert r._coord_type == "world"


# ---- Setters ---------------------------------------------------------------

def test_set_color_updates_all_arms_in_place():
    """set_color repaints arms without rebuilding the artist."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel", color="white")
    r.set_color("yellow")
    for arm in r.arm_artists:
        assert arm.get_color() == "yellow"
    plt.close(fig)


def test_set_color_with_stroke_color_none_strips_path_effects():
    """Switching stroke_color → None removes the stroke from each arm."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel")
    r.set_color("yellow", stroke_color=None)
    for arm in r.arm_artists:
        assert arm.get_path_effects() == []
    plt.close(fig)


def test_set_label_creates_then_clears_annotation():
    """Set label after construction; setting back to None removes it."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel")
    assert r.label_artist is None
    r.set_label("new")
    assert r.label_artist is not None
    assert r.label_artist.get_text() == "new"
    r.set_label(None)
    assert r.label_artist is None
    plt.close(fig)


def test_set_size_rebuilds_with_new_extent():
    """set_size triggers a full rebuild — old artists detached, new
    geometry uses the new size."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel", size=10)
    r.set_size(25)
    assert r._size == 25.0
    assert r._host_axes is ax        # still attached
    assert len(r.arm_artists) == 4   # still a plus
    plt.close(fig)


# ---- Top-level add_reticle -------------------------------------------------

def test_add_reticle_returns_attached_reticle():
    """The top-level helper returns a Reticle that is already attached
    (host_axes set), so the caller can chain set_*() / remove()."""
    fig, ax = _plain_axes()
    r = add_reticle(ax, (50, 50), coord_type="pixel")
    assert isinstance(r, Reticle)
    assert r._host_axes is ax
    plt.close(fig)


def test_repr_includes_style_and_label():
    """__repr__ surfaces style + label for easier debugging from prints."""
    r = Reticle((1, 2), coord_type="pixel", label="hello", style="x")
    rep = repr(r)
    assert "style='x'" in rep
    assert "label='hello'" in rep


def test_repr_without_label_omits_label_tag():
    r = Reticle((1, 2), coord_type="pixel")
    assert "label=" not in repr(r)


# ---- WCS-axes integration --------------------------------------------------

def test_wcs_axes_world_tuple_path():
    """A bare (lon, lat) tuple on a WCSAxes anchors correctly via
    ax.get_transform('world')."""
    fig, ax = _wcs_axes(center=(180.0, 0.0))
    r = add_reticle(ax, (180.0, 0.0), label="center")
    assert r._anchor_box is not None
    assert r.label_artist is not None
    plt.close(fig)


def test_wcs_axes_skycoord_path_handles_cross_frame():
    """SkyCoord input gets projected via ax.wcs.world_to_pixel, which
    auto-handles frame conversion (galactic input on an ICRS axes)."""
    fig, ax = _wcs_axes(center=(180.0, 0.0))
    sc = SkyCoord(266.4, -29.0, unit="deg", frame="galactic")
    r = add_reticle(ax, sc)
    assert r._anchor_box is not None
    plt.close(fig)


def test_reticle_stroke_reaches_label():
    """Folding stroke coverage into the label (12.15 unification): a stroked
    reticle's label carries the legibility stroke, not only the arms."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    r = add_reticle(ax, (5, 5), coord_type="pixel", label="Target",
                    stroke_color="k", stroke_lw=3)
    assert r.label_artist.get_path_effects(), "reticle label not stroked"
    # arms still stroked
    assert all(a.get_path_effects() for a in r.arm_artists)
    plt.close(fig)


def test_reticle_no_stroke_leaves_label_plain():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    r = add_reticle(ax, (5, 5), coord_type="pixel", label="Target",
                    stroke_color=None)
    assert not r.label_artist.get_path_effects()
    plt.close(fig)
