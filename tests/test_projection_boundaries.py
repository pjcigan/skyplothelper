"""Tests for skyplothelper.projections._boundaries.

Projection-specific visible-region boundary helpers — each returns
a closed ``(lon, lat)`` polyline in world coords tracing one
projection's actual visible region. Used together with the
``boundary=`` argument of
:func:`skyplothelper.coord_overlay.add_overlay_ticks` to give
projections whose astropy frame spine doesn't match their true
visible region (BON cardioid, PCO egg envelope, HPX HEALPix stepped
diamond, conic wedges) a proper boundary curve for clipping
out-of-frame axis-curve ticks.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.coord_overlay import (  # noqa: E402
    _FrameCurve,
    add_overlay_ticks,
)
from skyplothelper.projections._boundaries import (  # noqa: E402
    bonne_boundary,
    conic_boundary,
    healpix_boundary,
    polyconic_boundary,
    world_rect_boundary,
)


def _axes(projection, **kw):
    fig = plt.figure(figsize=(7, 5))
    ax = sph.make_wcs_frame(111, projection=projection, fig=fig, **kw)
    fig.canvas.draw()
    return fig, ax


# ---- bonne_boundary ----

def test_bonne_boundary_shape():
    """Returns a closed ``(2*n, 2)`` polyline of (lon, lat) pairs."""
    fig, ax = _axes("BON", center=180)
    lonlat = bonne_boundary(ax, n=100)
    assert lonlat.shape == (200, 2)
    plt.close(fig)


def test_bonne_boundary_traces_both_meridian_sides():
    """First half should be at lon=CRVAL-180, second half at
    lon=CRVAL+180 (the two sides of the cardioid)."""
    fig, ax = _axes("BON", center=180)
    lonlat = bonne_boundary(ax, n=50)
    crval_lon = ax.wcs.wcs.crval[0]
    assert np.all(lonlat[:50, 0] == crval_lon - 180.0)
    assert np.all(lonlat[50:, 0] == crval_lon + 180.0)
    # Latitude direction reverses between halves (south->north,
    # then north->south) so the polyline closes naturally.
    assert lonlat[0, 1] < lonlat[49, 1]   # west side runs upward
    assert lonlat[50, 1] > lonlat[99, 1]  # east side runs downward
    plt.close(fig)


def test_bonne_boundary_respects_lat_max():
    fig, ax = _axes("BON", center=180)
    lonlat = bonne_boundary(ax, lat_max=60.0, n=50)
    assert lonlat[:, 1].min() == pytest.approx(-60.0)
    assert lonlat[:, 1].max() == pytest.approx(60.0)
    plt.close(fig)


def test_bonne_boundary_default_center():
    fig, ax = _axes("BON", center=120)
    lonlat = bonne_boundary(ax, n=10)
    # West side at CRVAL-180 = 120-180 = -60
    # East side at CRVAL+180 = 120+180 = 300
    assert lonlat[0, 0] == pytest.approx(-60.0)
    assert lonlat[10, 0] == pytest.approx(300.0)
    plt.close(fig)


# ---- add_overlay_ticks boundary= kwarg ----

def test_boundary_kwarg_accepts_world_polyline():
    """A raw (N, 2) world-coord polyline gets auto-wrapped into a
    closed _FrameCurve."""
    fig, ax = _axes("BON", center=180)
    lonlat = bonne_boundary(ax)
    ov = add_overlay_ticks(
        ax, lon_at='axis', lat_at='axis', boundary=lonlat)
    # The clip filter should have run — we don't know exactly how
    # many ticks survive, but it should be a sensible count, not
    # the unclipped count.
    assert 4 <= len(ov.gridticks) <= 60
    plt.close(fig)


def test_boundary_kwarg_accepts_framecurve():
    """A pre-built _FrameCurve is used as-is."""
    fig, ax = _axes("BON", center=180)
    lonlat = bonne_boundary(ax)
    fc = _FrameCurve.from_world_polyline(
        ax, lonlat, closed=True, name='custom_bonne')
    ov = add_overlay_ticks(
        ax, lon_at='axis', lat_at='axis', boundary=fc)
    assert len(ov.gridticks) > 0
    plt.close(fig)


def test_boundary_kwarg_filters_phantom_axis_curve_ticks():
    """Without a boundary, axis-curve mode on a BON would produce
    ticks scattered outside the cardioid (the rectangular bbox
    contains points that map to valid pixels but lie outside the
    visible region). The boundary= kwarg should remove them."""
    fig, ax = _axes("BON", center=180)
    ov_no_clip = add_overlay_ticks(ax, lon_at='axis', lat_at='axis')
    ov_with_clip = add_overlay_ticks(
        ax, lon_at='axis', lat_at='axis', boundary=bonne_boundary(ax))
    # The clipped version has at most as many ticks as the unclipped
    # one, and typically meaningfully fewer.
    assert len(ov_with_clip.gridticks) <= len(ov_no_clip.gridticks)
    plt.close(fig)


def test_boundary_kwarg_substitutes_for_boundary_spec():
    """When ``lon_at='boundary'`` AND a custom boundary is provided,
    ticks should be discovered against that boundary (not the
    astropy frame spine)."""
    fig, ax = _axes("BON", center=180)
    lonlat = bonne_boundary(ax)
    ov = add_overlay_ticks(
        ax, lon_at='boundary', lat_at='boundary', boundary=lonlat)
    # Frame curves stored on the overlay should include the custom
    # boundary, not the default bbox edges.
    names = {fc.name for fc in ov.frame_curves}
    assert 'boundary' in names
    # Boundary mode means no constant-lat or constant-lon curves
    # were registered.
    assert not any('lat=' in n for n in names if n)
    assert not any('lon=' in n for n in names if n)
    plt.close(fig)


def test_boundary_kwarg_none_falls_back_to_frame_spine():
    """Default behavior unchanged when no boundary is provided."""
    fig, ax = _axes("BON", center=180)
    ov = add_overlay_ticks(ax)
    # Should fall back to the projection's astropy frame spines
    # (rectangular bbox edges for BON, since BON is registered as
    # frame_shape='rectangular').
    names = {fc.name for fc in ov.frame_curves}
    assert any(n in ('left', 'right', 'top', 'bottom') for n in names)
    plt.close(fig)


# ---- world_rect_boundary + conic_boundary ----

def test_world_rect_boundary_shape():
    fig, ax = _axes("AIT", center=180)
    lonlat = world_rect_boundary(ax, n=100)
    # 4 sides × n = 400 vertices
    assert lonlat.shape == (400, 2)
    plt.close(fig)


def test_world_rect_boundary_traces_four_sides():
    fig, ax = _axes("AIT", center=180)
    lonlat = world_rect_boundary(ax, lat_max=80.0, n=50)
    # Side 0: top arc at lat=+80
    assert np.all(lonlat[:50, 1] == pytest.approx(80.0))
    # Side 1: right seam at lon=CRVAL+180=360
    assert np.all(lonlat[50:100, 0] == pytest.approx(360.0))
    # Side 2: bottom arc at lat=-80
    assert np.all(lonlat[100:150, 1] == pytest.approx(-80.0))
    # Side 3: left seam at lon=CRVAL-180=0
    assert np.all(lonlat[150:200, 0] == pytest.approx(0.0))
    plt.close(fig)


def test_conic_boundary_full_range_for_cod_coe():
    """For COD / COE (finite over the whole sphere), ``conic_boundary``
    traces the full ``[CRVAL−180, CRVAL+180] × [−89.99, +89.99]`` world
    rectangle — identical to ``world_rect_boundary``."""
    for proj in ("COD", "COE"):
        fig, ax = _axes(proj, center=(180, 30))
        cb = conic_boundary(ax, n=100)
        wr = world_rect_boundary(ax, n=100)
        np.testing.assert_array_equal(cb, wr)
        plt.close(fig)


def test_conic_boundary_clips_far_pole_for_coo_cop():
    """COO / COP diverge toward the far pole, so ``conic_boundary``
    clips the wedge 30° past the equator (the kapteyn ``wylim`` bound) —
    the bottom arc sits at lat=−30, not −89.99."""
    for proj in ("COO", "COP"):
        fig, ax = _axes(proj, center=(180, 30))
        cb = conic_boundary(ax, n=100)
        # Side 2 (samples [200:300] for n=100) is the bottom arc.
        bottom_lat = cb[200:300, 1]
        assert np.all(bottom_lat == pytest.approx(-30.0)), proj
        # Top arc still reaches the apex pole.
        assert np.all(cb[:100, 1] == pytest.approx(89.99)), proj
        plt.close(fig)


@pytest.mark.parametrize("projection", ["COD", "COE", "COO", "COP"])
def test_add_overlay_ticks_conic_produces_sensible_count(projection):
    """All four conic flavors should produce a reasonable count of
    axis-curve ticks inside the wedge (not zero, not hundreds)."""
    fig, ax = _axes(projection, center=(180, 30))
    ov = add_overlay_ticks(
        ax, lon_at='axis', lat_at='axis', boundary=conic_boundary(ax))
    n = len(ov.gridticks)
    assert 6 <= n <= 60, f"{projection}: unexpected tick count {n}"
    plt.close(fig)


# ---- bbox-margin clip ----

def test_bbox_clip_drops_far_oob_ticks():
    """Axis-curve ticks whose xy_pix lies far outside the axes bbox
    should be filtered by the bbox-margin clip even if they pass the
    polygon test."""
    fig, ax = _axes("COD", center=(180, 30))
    # COD has the full-sphere boundary extending past the bbox on the
    # bottom (south pole), so the polygon clip alone isn't enough.
    ov = add_overlay_ticks(
        ax, lon_at='axis', lat_at='axis', boundary=conic_boundary(ax))
    bb = ax.bbox
    margin = 5.0
    for tick in ov.gridticks:
        x, y = tick.xy_pix
        assert bb.x0 - margin <= x <= bb.x1 + margin
        assert bb.y0 - margin <= y <= bb.y1 + margin
    plt.close(fig)


# ---- polyconic_boundary + healpix_boundary ----

def test_polyconic_boundary_delegates_to_world_rect():
    fig, ax = _axes("PCO", center=180)
    pc = polyconic_boundary(ax, n=100)
    wr = world_rect_boundary(ax, n=100)
    np.testing.assert_array_equal(pc, wr)
    plt.close(fig)


def test_healpix_boundary_traces_stepped_diamond():
    """``healpix_boundary`` traces HPX's true stepped-diamond visible
    region — the union of 12 HEALPix base pixels (4 north polar +
    equatorial band + 4 south polar). It must include vertices at
    the four north-polar peaks (lat=+90°) and four south-polar
    peaks (lat=-90°), plus the equator-to-polar-cap boundary at
    ``±arcsin(2/3) ≈ ±41.81°``.

    The stepped-diamond boundary helper returns just the 16
    perimeter corners (plus closing vertex) without intermediate
    interpolation — HEALPix base-pixel edges are exactly straight
    lines in HPX projection space, so matplotlib's natural
    straight-pixel-line-between-vertices behavior produces the
    correct rhombus geometry. Adding linear (lon, lat) interpolation
    would project to curves and incorrectly make the polar tile
    sides look concave.
    """
    fig, ax = _axes("HPX", center=180)
    hp = healpix_boundary(ax)
    eq_lat = np.degrees(np.arcsin(2.0 / 3.0))
    # 19 vertices: 1 start + 8 north zigzag + 1 right-edge + 8 south
    # zigzag + 1 closing vertex
    assert hp.shape == (19, 2)
    # 4 north polar peaks at lat=+90
    assert np.sum(np.isclose(hp[:, 1], +90.0)) == 4
    # 4 south polar peaks at lat=-90
    assert np.sum(np.isclose(hp[:, 1], -90.0)) == 4
    # Equator-to-polar-cap boundary present at ±arcsin(2/3)
    assert np.any(np.isclose(hp[:, 1], +eq_lat))
    assert np.any(np.isclose(hp[:, 1], -eq_lat))
    # Polyline is closed (first vertex == last vertex)
    np.testing.assert_array_equal(hp[0], hp[-1])
    plt.close(fig)


def test_add_overlay_ticks_pco_produces_sensible_count():
    """PCO polyconic should produce a reasonable count of axis-curve
    ticks inside its egg envelope. Per the 13.24 investigation the
    lon=±180 trace alone is non-monotonic, but world_rect_boundary
    still gives a clean clipping polygon."""
    fig, ax = _axes("PCO", center=180)
    ov = add_overlay_ticks(
        ax, lon_at='axis', lat_at='axis', boundary=polyconic_boundary(ax))
    assert 12 <= len(ov.gridticks) <= 40
    plt.close(fig)


def test_add_overlay_ticks_hpx_produces_sensible_count():
    """HPX HEALPix all-sky: world-rectangle boundary is a generous
    superset of the stepped-diamond visible region; axis-curve label
    placements land inside the stepped region and survive the clip."""
    fig, ax = _axes("HPX", center=180)
    ov = add_overlay_ticks(
        ax, lon_at='axis', lat_at='axis', boundary=healpix_boundary(ax))
    assert 12 <= len(ov.gridticks) <= 40
    plt.close(fig)


# ---- _ticks_discovered flag ----

def test_ticks_discovered_flag_set_after_discover():
    """The new explicit-discover flag tracks whether discover_ticks
    has run, so a clip filter that empties gridticks doesn't trigger
    an inadvertent rediscovery in render_ticks / render_labels."""
    from skyplothelper.coord_overlay import CoordinateOverlay
    fig, ax = _axes("BON", center=180)
    ov = CoordinateOverlay(ax, frame='icrs')
    assert ov._ticks_discovered is False
    ov.discover_ticks()
    assert ov._ticks_discovered is True
    plt.close(fig)


def test_render_ticks_doesnt_rediscover_after_clip_emptied_gridticks():
    """Specifically the bug that motivated the explicit flag: a
    custom-boundary clip can legitimately empty gridticks. The
    subsequent render_ticks must NOT re-discover them.

    Setup uses a SIN sub-frame (circular spine) so the default
    discover_ticks finds intersections (BON's rectangular default
    spines don't get crossed by typical galactic gridlines)."""
    from skyplothelper.coord_overlay import CoordinateOverlay
    fig, ax = _axes("SIN", center=(180, 30), fov_deg=70.0)
    ov = CoordinateOverlay(ax, frame='icrs').plot()
    ov.discover_ticks()
    assert len(ov.gridticks) > 0
    ov.gridticks = []  # simulate the clip filter emptying the list
    ov.render_ticks()  # must NOT auto-discover
    assert len(ov.gridticks) == 0
    plt.close(fig)
