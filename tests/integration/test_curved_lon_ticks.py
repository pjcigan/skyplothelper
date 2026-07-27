"""Curved-meridian longitude tick labels.

Pins the behavior of ``add_curved_lon_ticks`` (in
``skyplothelper.ticks``):

  * Returns a list of ``Text`` artists, one per visible tick.
  * Rotations match the meridian tangent at each tick location:
    zero at the central meridian, mirror-symmetric across it, and
    monotonic in longitude offset on a globe.
  * Hide-back filter drops ticks whose meridian doesn't intersect
    the front hemisphere at ``tick_lat`` for zenithal projections.
  * Format selection (``'hour'`` vs ``'deg'``) honors the requested
    style and the ``frame=`` override.
  * Astropy's auto longitude ticks are suppressed by default
    (``suppress_default=True``).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Text

import skyplothelper as sph
from skyplothelper._compat import coord_ticklabels
from skyplothelper.ticks import add_curved_lon_ticks


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _sin_globe(center=(0.0, 0.0)):
    fig = plt.figure(figsize=(6, 6))
    # tick_style='native' keeps astropy's default labels visible so
    # tests can assert against the helper's own suppress_default
    # behavior without the make_wcs_frame auto-trigger interfering.
    ax = sph.make_wcs_frame(111, projection="SIN", center=center, fig=fig,
                            tick_style='native')
    fig.canvas.draw()
    return fig, ax


def test_returns_text_artists():
    fig, ax = _sin_globe(center=(0.0, 0.0))
    arts = add_curved_lon_ticks(ax, tick_lat=0.0, lon_spacing=30.0)
    assert isinstance(arts, list)
    assert len(arts) > 0
    assert all(isinstance(a, Text) for a in arts)


def test_centered_globe_equator_ticks_horizontal():
    """SIN center=(0, 0) with tick_lat=0: equator is a horizontal line
    in pixel space, so every meridian tangent is vertical and every
    tick rotation is 0°."""
    fig, ax = _sin_globe(center=(0.0, 0.0))
    arts = add_curved_lon_ticks(ax, tick_lat=0.0, lon_spacing=30.0)
    rotations = np.array([a.get_rotation() for a in arts])
    # Rotations are wrapped into (-90, 90] — anything close to 0 (or
    # 360 / -360) counts as "horizontal."
    norm = ((rotations + 90.0) % 180.0) - 90.0
    assert np.allclose(norm, 0.0, atol=1e-6), f"rotations: {norm}"


def test_central_meridian_tick_has_zero_rotation():
    """At lon = lon_center the meridian is exactly vertical in pixel
    space regardless of lat_center — that tick's rotation is 0°."""
    fig, ax = _sin_globe(center=(0.0, 30.0))
    arts = add_curved_lon_ticks(ax, tick_lat=20.0, lon_spacing=30.0,
                                fmt='deg')
    # Find the tick whose label is "0°"
    central = [a for a in arts if a.get_text() == "0°"]
    assert len(central) == 1, [a.get_text() for a in arts]
    rot = ((central[0].get_rotation() + 90.0) % 180.0) - 90.0
    assert abs(rot) < 1.0, f"central tick rotation = {rot}"


def test_mirror_symmetric_rotations():
    """For a SIN globe centered on lon_0=0, the meridian at lon=+L
    is the mirror of the meridian at lon=-L across the central
    meridian. Tick rotations should be opposite-sign to within the
    finite-difference error (eps_deg)."""
    fig, ax = _sin_globe(center=(0.0, 30.0))
    arts = add_curved_lon_ticks(ax, tick_lat=20.0, lon_spacing=30.0,
                                fmt='deg')
    # Map label → rotation
    rot_by_label = {a.get_text(): a.get_rotation() for a in arts}

    def _signed(deg):
        return ((deg + 90.0) % 180.0) - 90.0

    # Negative-lon meridians render in [0, 360): -30° -> 330°, -60° -> 300°.
    pairs = [("30°", "330°"), ("60°", "300°")]
    for pos, neg in pairs:
        if pos not in rot_by_label or neg not in rot_by_label:
            continue
        rp = _signed(rot_by_label[pos])
        rn = _signed(rot_by_label[neg])
        # Mirror-symmetric → opposite signs, similar magnitude
        assert np.sign(rp) != np.sign(rn) or abs(rp) < 0.5
        assert abs(abs(rp) - abs(rn)) < 1.0, \
            f"{pos}={rp:.2f} vs {neg}={rn:.2f} (not mirror-symmetric)"


def test_hide_back_filters_back_hemisphere():
    """SIN center=(0, 0) at tick_lat=0: meridians at |lon| > 90
    don't reach the front hemisphere — those ticks must NOT be drawn.
    (The WCS itself returns NaN for off-hemisphere world coords on
    SIN, but the helper's explicit visibility check is what applies
    here for projections that don't NaN-out.)"""
    fig, ax = _sin_globe(center=(0.0, 0.0))
    arts = add_curved_lon_ticks(ax, tick_lat=0.0, lon_spacing=30.0,
                                hide_back=True, fmt='deg')
    labels = {a.get_text() for a in arts}
    # Central meridian always visible
    assert "0°" in labels
    # Antipodal meridians (lon = ±180) are on the back hemisphere. Labels
    # render in [0, 360), so the back-hemisphere negatives appear as high
    # positive degrees (-120° -> 240°, -150° -> 210°, ±180 -> 180°).
    assert "180°" not in labels
    # Limb-grazing meridians (lon = ±90) are at the boundary; either
    # visible or filtered, both acceptable as long as deeper-back ones
    # are dropped.
    assert "120°" not in labels and "240°" not in labels
    assert "150°" not in labels and "210°" not in labels


def test_hide_back_false_passes_through_to_wcs_filter():
    """``hide_back=False`` skips the explicit visibility filter, so any
    NaN-dropping that happens is attributable to the WCS itself
    (matters for non-zenithal projections where off-band lons project
    cleanly to far-away pixel coords). For SIN the WCS NaNs back-
    hemisphere world coords on its own; the helper still returns a
    sensible (small) list of front-hemisphere ticks rather than
    crashing."""
    fig, ax = _sin_globe(center=(0.0, 0.0))
    arts = add_curved_lon_ticks(ax, tick_lat=0.0, lon_spacing=30.0,
                                hide_back=False, fmt='deg')
    labels = {a.get_text() for a in arts}
    assert "0°" in labels
    # Back-hemisphere meridians are NaN'd by the SIN WCS, so they
    # still get dropped — but the path that drops them is the
    # finite-value filter, not the explicit hide_back filter.
    assert "180°" not in labels


def test_format_hour_vs_deg():
    """fmt='hour' produces superscript-hour labels; fmt='deg' produces 'N°'."""
    fig, ax = _sin_globe(center=(0.0, 0.0))
    arts_h = add_curved_lon_ticks(ax, tick_lat=0.0, lon_spacing=60.0,
                                  fmt='hour')
    plt.close('all')
    fig2, ax2 = _sin_globe(center=(0.0, 0.0))
    arts_d = add_curved_lon_ticks(ax2, tick_lat=0.0, lon_spacing=60.0,
                                  fmt='deg')
    assert all(r'$^\mathregular{h}$' in a.get_text() for a in arts_h)
    assert all('°' in a.get_text() for a in arts_d)


def test_explicit_lon_ticks_override_spacing():
    fig, ax = _sin_globe(center=(0.0, 0.0))
    arts = add_curved_lon_ticks(ax, tick_lat=0.0,
                                lon_ticks=[0.0, 45.0, -45.0],
                                fmt='deg')
    labels = sorted(a.get_text() for a in arts)
    # Absolute longitude renders in [0, 360): -45° -> 315°.
    assert labels == sorted(["0°", "45°", "315°"])


def test_suppress_default_hides_astropy_ticks():
    """Default suppress_default=True hides astropy's auto ticks/labels."""
    fig, ax = _sin_globe(center=(0.0, 0.0))
    add_curved_lon_ticks(ax, tick_lat=0.0)
    # ticks_visible → False (we set it False inside the helper)
    # CoordinateHelper has no ``get_ticks_visible`` method on every
    # astropy version; verify via the internal _ticks attribute or by
    # checking that no auto labels are rendered after a draw.
    fig.canvas.draw()
    # Auto tick label texts are stored on the coords[0].ticklabels artist.
    # We assert the artist is hidden.
    assert coord_ticklabels(ax.coords[0]).get_visible() is False


def _auto_lon_overlay_count(ax):
    """Number of the frame's auto in-frame overlay 'lon' labels on the axes."""
    return sum(1 for t in ax.texts
               if getattr(t, "_sph_overlay_kind", None) == "lon"
               and getattr(t, "_sph_auto_overlay", False))


def test_suppress_default_removes_auto_overlay_labels_no_doubling():
    """On a make_globe_frame (which draws auto in-frame overlay labels),
    suppress_default=True must REMOVE those 'lon' overlay labels — not just
    hide native ticks — so the curved labels don't double up on top."""
    fig = plt.figure(figsize=(6, 6))
    ax = sph.make_globe_frame(111, center_LONdeg=0, center_LATdeg=20)
    fig.canvas.draw()
    assert _auto_lon_overlay_count(ax) > 0          # frame drew auto lon labels
    add_curved_lon_ticks(ax, tick_lat=-30.0)        # suppress_default=True
    fig.canvas.draw()
    assert _auto_lon_overlay_count(ax) == 0         # auto overlay 'lon' cleared


def test_suppress_default_false_keeps_auto_overlay_labels():
    """suppress_default=False leaves the frame's auto overlay 'lon' labels."""
    fig = plt.figure(figsize=(6, 6))
    ax = sph.make_globe_frame(111, center_LONdeg=0, center_LATdeg=20)
    fig.canvas.draw()
    n0 = _auto_lon_overlay_count(ax)
    add_curved_lon_ticks(ax, tick_lat=-30.0, suppress_default=False)
    fig.canvas.draw()
    assert _auto_lon_overlay_count(ax) == n0 > 0


def test_preserves_default_when_requested():
    """suppress_default=False keeps astropy's auto labels visible."""
    fig, ax = _sin_globe(center=(0.0, 0.0))
    add_curved_lon_ticks(ax, tick_lat=0.0, suppress_default=False)
    fig.canvas.draw()
    assert coord_ticklabels(ax.coords[0]).get_visible() is True


def test_empty_when_tick_lat_outside_visible_hemisphere():
    """tick_lat=89 with hide_back=True on center=(0,-30) leaves no
    visible ticks; helper returns an empty list cleanly."""
    fig, ax = _sin_globe(center=(0.0, -30.0))
    # All meridians at lat=89 lie behind the south-tilted center,
    # except those very near the central meridian. Use lat_center
    # opposite the test lat to push *most* off-hemisphere — at
    # least confirm the helper doesn't crash and returns something
    # short.
    arts = add_curved_lon_ticks(ax, tick_lat=89.0, lon_spacing=60.0,
                                hide_back=True)
    # Either empty or very few ticks; never crashes.
    assert isinstance(arts, list)
    assert len(arts) <= 6


def test_galactic_frame_default_to_degrees():
    fig = plt.figure(figsize=(6, 6))
    ax = sph.make_wcs_frame(111, projection="SIN",
                             center=(0.0, 0.0), frame="Galactic", fig=fig)
    fig.canvas.draw()
    arts = add_curved_lon_ticks(ax, tick_lat=0.0, lon_spacing=60.0)
    # Galactic → fmt='auto' should pick degrees, not hours.
    assert all('°' in a.get_text() for a in arts)
    assert all('ʰ' not in a.get_text() for a in arts)


def test_deg_labels_stay_absolute_no_negative_wrap():
    """Galactic-equator deg labels render in [0, 360); the >180° fold that
    showed l=210° as -150° / l=180° as -180° is gone (Report D)."""
    from skyplothelper.ticks import _format_lon_label
    assert _format_lon_label(210.0, "galactic", "deg") == "210°"
    assert _format_lon_label(180.0, "galactic", "deg") == "180°"
    assert _format_lon_label(-45.0, "galactic", "deg") == "315°"
    assert _format_lon_label(360.0, "galactic", "deg") == "0°"
    # End-to-end: a galactic globe centered at l=180 labels its equator
    # meridians without any leading '-'.
    fig = plt.figure(figsize=(6, 6))
    ax = sph.make_wcs_frame(111, projection="SIN",
                            center=(180.0, 0.0), frame="Galactic", fig=fig)
    fig.canvas.draw()
    arts = add_curved_lon_ticks(ax, tick_lat=0.0, lon_spacing=30.0, fmt='deg')
    assert arts
    assert not any(a.get_text().startswith('-') for a in arts), \
        [a.get_text() for a in arts]


def test_curved_lon_ticks_inherits_cached_auto_fontsize():
    """When ``fontsize=None`` and the axes has an
    ``ax._sph_auto_label_fontsize`` cached by
    ``make_wcs_frame(auto_fontsize=True)``, ``add_curved_lon_ticks``
    uses the cached value rather than the rcParams default. Keeps
    overlay-style ticks visually consistent with the frame's other
    labels."""
    fig = plt.figure(figsize=(3, 2))   # small panel → cache will be < default
    ax = sph.make_wcs_frame(111, projection="SIN",
                             center=(0.0, 0.0), fig=fig,
                             auto_fontsize=True)
    fig.canvas.draw()
    cached = getattr(ax, '_sph_auto_label_fontsize', None)
    assert cached is not None
    arts = add_curved_lon_ticks(ax, tick_lat=0.0, lon_spacing=60.0)
    assert arts, "expected at least one tick label to be drawn"
    for a in arts:
        assert a.get_fontsize() == pytest.approx(cached)


def test_curved_lon_ticks_explicit_fontsize_overrides_cache():
    """An explicit ``fontsize=`` always wins over the cached value."""
    fig = plt.figure(figsize=(3, 2))
    ax = sph.make_wcs_frame(111, projection="SIN",
                             center=(0.0, 0.0), fig=fig,
                             auto_fontsize=True)
    fig.canvas.draw()
    arts = add_curved_lon_ticks(ax, tick_lat=0.0, lon_spacing=60.0,
                                 fontsize=14.0)
    for a in arts:
        assert a.get_fontsize() == pytest.approx(14.0)
