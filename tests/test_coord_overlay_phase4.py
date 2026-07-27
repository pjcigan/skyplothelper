"""Tests for skyplothelper.coord_overlay tick-label rendering.

Formatted text labels render at every discovered tick. Labels sit
just past the tick endpoint along the gridline tangent, rotate to
the tangent angle (clamped upright), and overlap-aware suppression
hides labels whose bbox collides with an earlier one.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.text import Text  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.coord_overlay import (  # noqa: E402
    CoordinateOverlay,
    _format_tick_label,
    _FrameCurve,
)


def _make_axes(projection="CAR", center=180, frame="ICRS"):
    fig = plt.figure(figsize=(10, 5))
    ax = sph.make_wcs_frame(111, projection=projection, center=center,
                            frame=frame, fig=fig)
    fig.canvas.draw()
    return fig, ax


def _inner_box_curve(ax, lons=(150., 210.), lats=(-30., 30.)):
    lo0, lo1 = lons
    la0, la1 = lats
    return _FrameCurve.from_world_polyline(
        ax,
        np.array([[lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1]]),
        closed=True, name="box")


# ---- _format_tick_label ----

@pytest.mark.parametrize("value,kind,frame,fmt,expected", [
    # Latitudes — always degrees with sign, zero unsigned
    (30, "lat", "icrs", "auto", "+30°"),
    (-45, "lat", "galactic", "auto", "-45°"),
    (0, "lat", "icrs", "auto", "0°"),
    (-45.5, "lat", "galactic", "auto", "-45.5°"),
    # Longitudes — auto picks hours for ICRS, degrees otherwise
    (180, "lon", "icrs", "auto", "12$^\\mathregular{h}$"),
    (180, "lon", "galactic", "auto", "180°"),
    # Absolute longitude uses the [0, 360) convention — 270° stays 270°,
    # never folds to a signed -90° (galactic/ecliptic l run 0..360).
    (270, "lon", "galactic", "auto", "270°"),
    (182, "lon", "galactic", "auto", "182°"),
    (359, "lon", "galactic", "auto", "359°"),
    (180, "lon", "geocentrictrueecliptic", "auto", "180°"),
    (180, "lon", "supergalactic", "auto", "180°"),
    # Explicit hour / degree
    (195, "lon", "galactic", "hour", "13$^\\mathregular{h}$"),
    (270, "lon", "icrs", "deg", "270°"),
    (45, "lon", "icrs", "deg", "45°"),
    # Non-integer hours
    (15 * 12.5, "lon", "icrs", "hour", "12.5$^\\mathregular{h}$"),
])
def test_format_tick_label_cases(value, kind, frame, fmt, expected):
    assert _format_tick_label(value, kind, frame, fmt) == expected


def test_format_tick_label_callable():
    fmt = lambda v: f"V={v:.0f}"  # noqa: E731
    assert _format_tick_label(45, "lon", "icrs", fmt) == "V=45"
    assert _format_tick_label(-30, "lat", "galactic", fmt) == "V=-30"


def test_format_tick_label_rejects_unknown_fmt():
    with pytest.raises(ValueError, match="Unknown lon-tick format"):
        _format_tick_label(45, "lon", "icrs", "sexagesimal")


def test_format_tick_label_icrs_alias_frames_use_hours():
    """FK5 / FK4 should also default to hour format for longitude."""
    assert _format_tick_label(180, "lon", "fk5", "auto") == "12$^\\mathregular{h}$"
    assert _format_tick_label(180, "lon", "fk4", "auto") == "12$^\\mathregular{h}$"


def test_format_tick_label_step_adapts_precision():
    """The ``step`` hint scales label precision for field-scale ticks while
    a coarse / absent step keeps the legacy integer-or-.1f behavior."""
    # No step → legacy: integers stay integers.
    assert _format_tick_label(30, "lon", "galactic", "deg") == "30°"
    assert _format_tick_label(20, "lat", "icrs", "auto") == "+20°"
    # Fine degree step → one decimal so 0.5°-spaced labels stay distinct.
    assert _format_tick_label(149.5, "lon", "galactic", "deg", step=0.5) == "149.5°"
    assert _format_tick_label(21.5, "lat", "icrs", "auto", step=0.5) == "+21.5°"
    # Hours: a 0.5° (~0.033ʰ) step needs 2 decimals; the integer-shortcut is
    # suppressed on the field path so neighbors render uniformly.
    h_half = _format_tick_label(149.5, "lon", "icrs", "hour", step=0.5)
    h_int = _format_tick_label(150.0, "lon", "icrs", "hour", step=0.5)
    assert h_half != h_int
    assert "." in h_int  # 10.00ʰ, not collapsed to 10ʰ
    # Coarse step → integer hours.
    assert _format_tick_label(150.0, "lon", "icrs", "hour",
                              step=30.0) == "10$^\\mathregular{h}$"


# ---- render_labels: basic shape ----

def test_render_labels_one_artist_per_gridtick():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[150, 180, 210],
                            lat_vals=[-20, 0, 20])
          .plot()
          .set_frame_curves([box])
          .discover_ticks()
          .render_labels(mode="complete"))
    assert len(ov.label_artists) == len(ov.gridticks) == 12
    for artist in ov.label_artists:
        assert isinstance(artist, Text)
    plt.close(fig)


def test_render_labels_auto_discovers_if_needed():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .render_labels())
    assert len(ov.gridticks) > 0
    assert len(ov.label_artists) == len(ov.gridticks)
    plt.close(fig)


def test_render_labels_returns_self_for_chaining():
    fig, ax = _make_axes()
    ov = (CoordinateOverlay(ax, frame="galactic")
          .plot()
          .set_frame_curves([_inner_box_curve(ax)]))
    assert ov.render_labels() is ov
    plt.close(fig)


def test_render_labels_uses_transdata():
    """Labels must be placed in axes data coords so rendering survives
    a savefig at a different dpi than the canvas dpi."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .render_labels())
    for artist in ov.label_artists:
        assert artist.get_transform() is ax.transData
    plt.close(fig)


# ---- label content ----

def test_render_labels_text_content():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .render_labels())
    texts = sorted({a.get_text() for a in ov.label_artists})
    assert "12$^\\mathregular{h}$" in texts
    assert "0°" in texts
    plt.close(fig)


def test_render_labels_galactic_lon_uses_degrees():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="galactic",
                            lon_vals=[180], lat_vals=[])
          .plot()
          .set_frame_curves([box])
          .render_labels())
    for artist in ov.label_artists:
        # Galactic lon should produce a degree-format string with °
        # (no superscript-h since galactic longitudes are not hours).
        assert "°" in artist.get_text()
        assert "ʰ" not in artist.get_text()
    plt.close(fig)


def test_render_labels_custom_fmt_callable():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[])
          .plot()
          .set_frame_curves([box])
          .render_labels(fmt=lambda v: f"<{v:.0f}>"))
    for artist in ov.label_artists:
        assert artist.get_text() == "<180>"
    plt.close(fig)


# ---- position and rotation ----

def test_render_labels_position_along_outward_tangent():
    """Each label sits ``pad`` display pixels from the tick along
    the outward gridline tangent."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .discover_ticks()
          .render_labels(pad=15, mode="complete"))
    for artist, tick in zip(ov.label_artists, ov.gridticks):
        pos_data = artist.get_position()
        pos_pix = ax.transData.transform(pos_data)
        d = np.linalg.norm(pos_pix - tick.xy_pix)
        assert d == pytest.approx(15., abs=1e-6)
    plt.close(fig)


def _normalize_rotation(r):
    """Map any rotation in degrees to its equivalent in (-180, 180]."""
    return ((r + 180.0) % 360.0) - 180.0


def test_render_labels_rotate_tangent_upright_clamped():
    """rotate='tangent_upright' clamps every label's rotation to (-90, 90]
    (compared after normalizing matplotlib's [0, 360) get_rotation
    back to (-180, 180])."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="galactic",
                            lon_vals=list(range(0, 360, 30)),
                            lat_vals=list(range(-60, 61, 30)))
          .plot()
          .set_frame_curves([box])
          .discover_ticks()
          .render_labels(rotate="tangent_upright", mode="complete"))
    for artist in ov.label_artists:
        r = _normalize_rotation(artist.get_rotation())
        assert -90.0 < r <= 90.0
    plt.close(fig)


def test_render_labels_default_rotate_is_continuous_not_clamp():
    """The default rotate='tangent' is now the continuous behavior, and
    'tangent_noflip' is an exact alias of it (same rotations per label)."""
    def _rots(rotate):
        fig, ax = _make_axes(projection="AIT", center=0)
        ov = (CoordinateOverlay(ax, frame="icrs",
                                lon_vals=list(range(0, 360, 30)),
                                lat_vals=list(range(-60, 61, 30)))
              .plot()
              .discover_ticks()
              .render_labels(rotate=rotate, mode="complete"))
        out = [_normalize_rotation(a.get_rotation()) for a in ov.label_artists]
        plt.close(fig)
        return out

    default = _rots("tangent")
    alias = _rots("tangent_noflip")
    upright = _rots("tangent_upright")
    assert default == pytest.approx(alias)            # alias is identical
    # The default is NOT the clamp: at least one label leans past vertical
    # where 'tangent_upright' would have forced it into (-90, 90].
    assert any(abs(r) > 90.0 + 1e-6 for r in default)
    assert all(abs(r) <= 90.0 + 1e-6 for r in upright)


def test_render_labels_rotate_noflip_uniform_branch_per_placement_group():
    """'tangent_noflip' applies a single 0°/180° offset to the raw tangent per
    placement group (frame curve + lon/lat) — uniform within the group (so it
    never introduces a flip between spatially-adjacent labels) and chosen to
    keep the group upright, so whole upside-down groups get flipped."""
    from collections import defaultdict
    fig, ax = _make_axes(projection="AIT", center=0)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=list(range(0, 360, 30)),
                            lat_vals=list(range(-60, 61, 30)))
          .plot()
          .discover_ticks()
          .render_labels(rotate="tangent_noflip", mode="complete"))
    assert ov.label_artists
    offsets = defaultdict(list)
    for tick, artist in zip(ov.gridticks, ov.label_artists):
        raw = _normalize_rotation(tick.tangent_deg)
        got = _normalize_rotation(artist.get_rotation())
        key = (id(tick.frame_curve), tick.kind)
        offsets[key].append(_normalize_rotation(got - raw))
    flipped = 0
    for offs in offsets.values():
        # Uniform within the group: the same 0 or ~180 offset for all.
        assert max(abs(_normalize_rotation(o - offs[0])) for o in offs) < 1e-6
        # The offset is only ever a whole 180° branch flip (never a clamp).
        assert abs(offs[0]) < 1e-6 or abs(abs(offs[0]) - 180.0) < 1e-6
        if abs(abs(offs[0]) - 180.0) < 1e-6:
            flipped += 1
    assert flipped > 0  # at least one group flipped upright


def _overlay_label_rots(ax, want_lon):
    """(x, rotation) for the overlay lon (or lat) labels on a frame."""
    out = []
    for t in ax.texts:
        s = t.get_text()
        if not getattr(t, "_sph_overlay_ticklabel", False):
            continue
        is_lon = "h" in s and "°" not in s
        if is_lon == want_lon:
            out.append((t.get_position()[0],
                        _normalize_rotation(t.get_rotation())))
    return out


def test_render_labels_rotate_noflip_keeps_globe_lat_upright():
    """Regression: on a tilted SIN globe, 'tangent_noflip' lat labels read
    upright (their placement group's tangents all point the same way, so the
    whole upside-down set is flipped), where the raw tangent renders them
    upside-down."""
    fig = plt.figure()
    ax = sph.make_wcs_frame(111, "SIN", center=(60, 35), fig=fig,
                            tick_rotation="tangent_noflip")
    rots = [r for _, r in _overlay_label_rots(ax, want_lon=False)]
    assert rots and all(abs(r) <= 90.0 + 1e-6 for r in rots)
    plt.close(fig)

    fig2 = plt.figure()
    ax2 = sph.make_wcs_frame(111, "SIN", center=(60, 35), fig=fig2,
                             tick_rotation=lambda t: t.tangent_deg)
    raw = [r for _, r in _overlay_label_rots(ax2, want_lon=False)]
    assert any(abs(r) > 90.0 + 1e-6 for r in raw)
    plt.close(fig2)


def test_render_labels_rotate_noflip_globe_lon_has_no_flip():
    """The converging meridians on a globe cannot all be upright, but
    'tangent_noflip' must not SNAP them: ordered along the top, consecutive
    lon-label rotations vary smoothly (small steps), unlike 'tangent' whose
    upright clamp jumps ~180° at the vertical crossing."""
    def _max_consecutive_step(ax):
        pairs = sorted(_overlay_label_rots(ax, want_lon=True))
        rots = [r for _, r in pairs]
        return max(abs(_normalize_rotation(b - a))
                   for a, b in zip(rots, rots[1:]))

    fig = plt.figure()
    ax = sph.make_wcs_frame(111, "SIN", center=(60, 35), fig=fig,
                            tick_rotation="tangent_noflip")
    noflip_step = _max_consecutive_step(ax)
    plt.close(fig)

    fig2 = plt.figure()
    ax2 = sph.make_wcs_frame(111, "SIN", center=(60, 35), fig=fig2,
                             tick_rotation="tangent_upright")
    upright_step = _max_consecutive_step(ax2)
    plt.close(fig2)

    assert noflip_step < 45.0            # smooth fan, no snap
    assert upright_step > 150.0          # the clamp's ~180° flip


def test_render_labels_rotate_horizontal():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .render_labels(rotate="horizontal", mode="complete"))
    for artist in ov.label_artists:
        assert artist.get_rotation() == 0.0
    plt.close(fig)


def test_render_labels_rotate_fixed_float():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .render_labels(rotate=45.0, mode="complete"))
    for artist in ov.label_artists:
        assert artist.get_rotation() == pytest.approx(45.0)
    plt.close(fig)


def test_render_labels_rotate_callable():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .render_labels(rotate=lambda t: -30.0 if t.kind == "lon" else 60.0,
                         mode="complete"))
    for artist, tick in zip(ov.label_artists, ov.gridticks):
        expected = -30.0 if tick.kind == "lon" else 60.0
        r = _normalize_rotation(artist.get_rotation())
        assert r == pytest.approx(expected)
    plt.close(fig)


def test_render_labels_styling_kwargs():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .render_labels(fontsize=14, color="purple", zorder=42))
    for artist in ov.label_artists:
        assert artist.get_color() == "purple"
        assert artist.get_fontsize() == 14
        assert artist.get_zorder() == 42
    plt.close(fig)


def test_render_labels_clip_off_default():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .render_labels())
    for artist in ov.label_artists:
        assert artist.get_clip_on() is False
    plt.close(fig)


# ---- overlap detection ----

def test_render_labels_mode_auto_hides_collisions():
    """Stacking many densely-spaced parallels guarantees label
    overlaps; the default mode='auto' must hide some of them."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    lat_vals = np.arange(-29., 30., 1.)  # ~60 dense parallels
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[], lat_vals=lat_vals)
          .plot()
          .set_frame_curves([box])
          .render_labels(fontsize=14, mode="auto"))
    n_total = len(ov.label_artists)
    n_visible = sum(a.get_visible() for a in ov.label_artists)
    assert n_total > 0
    assert n_visible < n_total
    plt.close(fig)


def test_render_labels_mode_complete_keeps_all():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    lat_vals = np.arange(-29., 30., 1.)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[], lat_vals=lat_vals)
          .plot()
          .set_frame_curves([box])
          .render_labels(fontsize=14, mode="complete"))
    for a in ov.label_artists:
        assert a.get_visible() is True
    plt.close(fig)


def test_render_labels_auto_pad_clears_outward_ticks():
    """When ticks point outward, default pad should be tick_length
    plus a small visible-gap buffer so labels clear the tick endpoint."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[])
          .plot()
          .set_frame_curves([box])
          .render_ticks(length=20, direction="out")
          .render_labels(mode="complete"))
    for artist, tick in zip(ov.label_artists, ov.gridticks):
        pos_pix = ax.transData.transform(artist.get_position())
        d = np.linalg.norm(pos_pix - tick.xy_pix)
        # ``pad`` is the near-edge gap from the tick endpoint to the
        # label. Default ``pad = outward + 5``
        # → tick_length (20) + buffer (5) = 25 px from tick crossing
        # to label anchor (which sits at the bbox near-edge).
        assert d == pytest.approx(25., abs=1e-6)
    plt.close(fig)


def test_render_labels_auto_pad_ignores_inward_tick_length():
    """When ticks point inward, the outward pad shouldn't grow with
    tick length — the inward tick doesn't compete for outward space."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[])
          .plot()
          .set_frame_curves([box])
          .render_ticks(length=30, direction="in")
          .render_labels(mode="complete"))
    for artist, tick in zip(ov.label_artists, ov.gridticks):
        pos_pix = ax.transData.transform(artist.get_position())
        d = np.linalg.norm(pos_pix - tick.xy_pix)
        # Inward ticks contribute 0 outward extent → pad = 0 + 5 = 5 px
        # (near-edge anchoring default).
        assert d == pytest.approx(5., abs=1e-6)
    plt.close(fig)


def test_render_labels_auto_pad_without_render_ticks():
    """If render_ticks was never called, auto-pad falls back to 5 px
    (the default visible gap from tick to label near-edge with
    ``ha=va='auto'``)."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[])
          .plot()
          .set_frame_curves([box])
          .render_labels(mode="complete"))
    for artist, tick in zip(ov.label_artists, ov.gridticks):
        pos_pix = ax.transData.transform(artist.get_position())
        d = np.linalg.norm(pos_pix - tick.xy_pix)
        assert d == pytest.approx(5., abs=1e-6)
    plt.close(fig)


def test_render_labels_rejects_unknown_mode():
    fig, ax = _make_axes()
    ov = (CoordinateOverlay(ax, frame="galactic")
          .plot()
          .set_frame_curves([_inner_box_curve(ax)]))
    with pytest.raises(ValueError, match="mode must be"):
        ov.render_labels(mode="debug")
    plt.close(fig)


def test_render_labels_no_collisions_keeps_all_visible():
    """A widely-spaced label set should have no collisions."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .render_labels(mode="auto"))
    for a in ov.label_artists:
        assert a.get_visible() is True
    plt.close(fig)


# ---- Report B: 'deg' longitude labels use [0, 360), no signed wrap ----

@pytest.mark.parametrize("value,expected", [
    (182.0, "182°"),
    (184.0, "184°"),
    (270.0, "270°"),
    (359.0, "359°"),
    (360.0, "0°"),
    (0.0, "0°"),
])
def test_format_tick_label_deg_lon_stays_absolute(value, expected):
    """Absolute longitude (galactic/ecliptic/...) renders in [0, 360); the
    old fmt='deg' branch folded >180° to a signed value (182° -> -178°)."""
    assert _format_tick_label(value, "lon", "galactic", "deg") == expected


def test_render_labels_galactic_lon_over_180_not_negative():
    """End-to-end: a galactic l=182° overlay tick labels as '182°', not
    '-178°' (the reported symptom)."""
    fig, ax = _make_axes(projection="CAR", center=180)
    ov = sph.add_overlay_ticks(
        ax, frame="galactic",
        lon_vals=np.arange(180, 191, 2), lat_vals=np.arange(-10, -1, 2),
        lon_at="lat=-6", lat_at="lon=184")
    texts = [t.get_text() for t in ov.label_artists]
    assert texts  # something rendered
    # The lon labels (182..190) must appear as-is; the old signed wrap would
    # have emitted -178°, -176°, ... instead.
    assert "182°" in texts
    wrong = {"-178°", "-176°", "-174°", "-172°", "-170°"}
    assert not (wrong & set(texts)), texts
    plt.close(fig)


# ---- Report A: FOV-adaptive default graticule for zoomed overlays ----

def test_overlay_default_vals_adapt_to_zoomed_cross_frame_field():
    """A galactic overlay on a zoomed equatorial TAN field should derive
    field-scale galactic default vals (not the empty all-sky 30°/15°)."""
    fig = plt.figure()
    ax = sph.make_wcs_frame(111, "TAN", center=(83.63, 22.01),
                            fov_deg=6.0, fig=fig)
    fig.canvas.draw()
    ov = sph.add_overlay_ticks(ax, frame="galactic")
    # Default 30°/15° all-sky graticule would land 0 meridians/parallels in
    # this ~6° galactic window; the adaptive default fills it.
    assert len(ov.gridticks) > 0
    # The Crab field sits near galactic (l, b) ≈ (184.5, -5.8); the derived
    # values must bracket that, and be finer than the all-sky step.
    assert ov.lon_vals.size >= 2 and ov.lat_vals.size >= 2
    assert float(np.min(np.diff(np.sort(ov.lon_vals)))) < 30.0
    assert 180.0 < float(ov.lon_vals.mean()) < 190.0
    plt.close(fig)


def test_overlay_default_vals_same_frame_zoomed_still_adaptive():
    """The same-frame zoomed path (previously handled in add_overlay_ticks,
    now centralized in the constructor) still adapts."""
    fig = plt.figure()
    ax = sph.make_wcs_frame(111, "TAN", center=(83.63, 22.01),
                            fov_deg=4.0, fig=fig)
    fig.canvas.draw()
    ov = sph.add_overlay_ticks(ax, frame="icrs")
    assert len(ov.gridticks) > 0
    assert float(np.min(np.diff(np.sort(ov.lon_vals)))) < 30.0


def test_overlay_default_vals_allsky_keeps_30_15():
    """All-sky frames keep the classic 30°/15° graticule defaults."""
    fig, ax = _make_axes(projection="AIT", center=180)
    ov = CoordinateOverlay(ax, frame="galactic")
    np.testing.assert_array_equal(ov.lon_vals, np.arange(0., 360., 30.))
    np.testing.assert_array_equal(ov.lat_vals, np.arange(-75., 76., 15.))
    plt.close(fig)


def test_overlay_zoomed_cross_frame_parallels_draw_segments():
    """Report C: a galactic overlay on a zoomed equatorial field must draw
    its latitude parallels (sampled over the field's overlay-frame lon
    extent), not 0 segments from a full-sky sweep dropped by the wrap/clip."""
    fig = plt.figure()
    ax = sph.make_wcs_frame(111, "TAN", center=(83.63, 22.01),
                            fov_deg=6.0, fig=fig)
    fig.canvas.draw()
    ov = sph.add_coord_overlay(ax, frame="galactic")
    lat_seg = sum(len(segs) for segs in ov.lat_artists)
    lon_seg = sum(len(segs) for segs in ov.lon_artists)
    assert lat_seg > 0, "latitude parallels drew no segments"
    assert lon_seg > 0
    plt.close(fig)


@pytest.mark.parametrize("lat_at", ["lon=184", "axis", "boundary"])
def test_overlay_zoomed_cross_frame_lat_ticks_found(lat_at):
    """Report C: lat ticks are found for every lat_at form on a zoomed
    cross-frame overlay (was 0 for all forms — the 'axis' form additionally
    needed its curve centered on the field, not the overlay-frame origin)."""
    fig = plt.figure()
    ax = sph.make_wcs_frame(111, "TAN", center=(83.63, 22.01),
                            fov_deg=6.0, fig=fig)
    fig.canvas.draw()
    ov = sph.add_overlay_ticks(ax, frame="galactic",
                               lat_at=lat_at, lon_at=None)
    assert len(ov.gridticks) > 0
    plt.close(fig)
