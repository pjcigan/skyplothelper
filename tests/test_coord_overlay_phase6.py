"""Tests for ``add_overlay_ticks`` axis-curve placement.

Axis-curve mode places lon labels along a constant-lat parallel
(default: the equator) and lat labels along a constant-lon meridian
(default: the central meridian). The two axes are independent — you
can keep boundary placement for one and switch to axis-curve for
the other.

Mechanism: a ``kind`` attribute on :class:`_FrameCurve` lets a
curve declare which kind of gridline (meridian or parallel) should
intersect it. :meth:`CoordinateOverlay.discover_ticks` respects
this filter. New ``_FrameCurve.from_const_lat`` and
``from_const_lon`` factories build the constant-coord curves in
pixel space.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.coord_overlay import (  # noqa: E402
    CoordinateOverlay,
    _FrameCurve,
    _resolve_curve_spec,
    add_overlay_ticks,
)


def _axes(projection, **kw):
    fig = plt.figure(figsize=(7, 5))
    ax = sph.make_wcs_frame(111, projection=projection, fig=fig, **kw)
    fig.canvas.draw()
    return fig, ax


# ---- _FrameCurve.kind attribute ----

def test_framecurve_default_kind_is_none():
    fc = _FrameCurve(np.array([[0., 0.], [1., 1.]]))
    assert fc.kind is None


def test_framecurve_accepts_lon_and_lat_kinds():
    fc1 = _FrameCurve(np.array([[0., 0.], [1., 1.]]), kind='lon')
    fc2 = _FrameCurve(np.array([[0., 0.], [1., 1.]]), kind='lat')
    assert fc1.kind == 'lon'
    assert fc2.kind == 'lat'


def test_framecurve_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind must be"):
        _FrameCurve(np.array([[0., 0.], [1., 1.]]), kind='both')


# ---- factory methods ----

def test_from_const_lat_default_kind_is_lon():
    fig, ax = _axes("CAR", center=180)
    fc = _FrameCurve.from_const_lat(ax, 30.0)
    assert fc.kind == 'lon'
    assert fc.name == 'lat=30.0'
    plt.close(fig)


def test_from_const_lat_pixel_polyline_traces_parallel():
    """Sampled pixel positions should match world_to_pixel of the
    same (lon, lat) points within numerical precision."""
    fig, ax = _axes("CAR", center=180)
    fc = _FrameCurve.from_const_lat(ax, 30.0, lon_range=(100., 260.),
                                    n=11)
    expected = ax.get_transform('world').transform(
        np.column_stack([np.linspace(100., 260., 11), [30.] * 11]))
    np.testing.assert_allclose(fc.xy_pix, expected)
    plt.close(fig)


def test_from_const_lon_default_kind_is_lat():
    fig, ax = _axes("CAR", center=180)
    fc = _FrameCurve.from_const_lon(ax, 180.0)
    assert fc.kind == 'lat'
    assert fc.name == 'lon=180.0'
    plt.close(fig)


def test_from_const_lon_pixel_polyline_traces_meridian():
    fig, ax = _axes("CAR", center=180)
    fc = _FrameCurve.from_const_lon(ax, 180.0, lat_range=(-60., 60.),
                                    n=11)
    expected = ax.get_transform('world').transform(
        np.column_stack([[180.] * 11, np.linspace(-60., 60., 11)]))
    np.testing.assert_allclose(fc.xy_pix, expected)
    plt.close(fig)


def test_from_const_lat_auto_lon_range_skips_wrap():
    """Default lon_range trims by ε so the polyline doesn't span the
    wrap point (which would jump-split it)."""
    fig, ax = _axes("CAR", center=180)
    fc = _FrameCurve.from_const_lat(ax, 0.0)
    # For center=180, expect lon range close to [0.01, 359.99]
    # which means sampled lons stay within the visible region.
    # First and last x should be near (but not at) the bbox edges.
    x_left = ax.bbox.extents[0]
    x_right = ax.bbox.extents[2]
    assert fc.xy_pix[0, 0] > x_left  # not at left edge
    assert fc.xy_pix[-1, 0] < x_right  # not at right edge
    plt.close(fig)


# ---- kind filter in discover_ticks ----

def test_discover_ticks_kind_lon_only_meridians_intersect():
    """A frame curve tagged kind='lon' should only host meridian
    ticks — parallels skip it."""
    fig, ax = _axes("CAR", center=180)
    fc_lon = _FrameCurve.from_const_lat(ax, 0.0)  # kind='lon'
    ov = (CoordinateOverlay(ax, frame='icrs',
                             lon_vals=[60, 120, 240, 300],
                             lat_vals=[-30, 0, 30])
          .plot()
          .set_frame_curves([fc_lon])
          .discover_ticks())
    kinds = {t.kind for t in ov.gridticks}
    assert kinds == {'lon'}
    plt.close(fig)


def test_discover_ticks_kind_lat_only_parallels_intersect():
    """A frame curve tagged kind='lat' should only host parallel
    ticks — meridians skip it."""
    fig, ax = _axes("CAR", center=180)
    fc_lat = _FrameCurve.from_const_lon(ax, 180.0)  # kind='lat'
    ov = (CoordinateOverlay(ax, frame='icrs',
                             lon_vals=[60, 120, 240, 300],
                             lat_vals=[-30, 0, 30])
          .plot()
          .set_frame_curves([fc_lat])
          .discover_ticks())
    kinds = {t.kind for t in ov.gridticks}
    assert kinds == {'lat'}
    plt.close(fig)


def test_discover_ticks_kind_none_accepts_both():
    """A frame curve with kind=None (default) accepts both kinds."""
    fig, ax = _axes("CAR", center=180)
    # An inner box, untagged
    box = _FrameCurve.from_world_polyline(
        ax,
        np.array([[150., -30.], [210., -30.],
                  [210., 30.], [150., 30.]]),
        closed=True)
    assert box.kind is None
    ov = (CoordinateOverlay(ax, frame='icrs',
                             lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .discover_ticks())
    kinds = {t.kind for t in ov.gridticks}
    assert kinds == {'lon', 'lat'}
    plt.close(fig)


def test_discover_ticks_mixed_kind_curves():
    """A lon-tagged curve plus a lat-tagged curve produce both kinds
    of ticks, on different curves."""
    fig, ax = _axes("CAR", center=180)
    fc_lon = _FrameCurve.from_const_lat(ax, 0.0)
    fc_lat = _FrameCurve.from_const_lon(ax, 180.0)
    ov = (CoordinateOverlay(ax, frame='icrs',
                             lon_vals=[60, 240], lat_vals=[-30, 30])
          .plot()
          .set_frame_curves([fc_lon, fc_lat])
          .discover_ticks())
    lon_ticks = [t for t in ov.gridticks if t.kind == 'lon']
    lat_ticks = [t for t in ov.gridticks if t.kind == 'lat']
    assert all(t.frame_curve is fc_lon for t in lon_ticks)
    assert all(t.frame_curve is fc_lat for t in lat_ticks)
    assert len(lon_ticks) >= 1
    assert len(lat_ticks) >= 1
    plt.close(fig)


# ---- _resolve_curve_spec ----

def test_resolve_curve_spec_boundary_returns_none():
    fig, ax = _axes("AIT", center=180)
    assert _resolve_curve_spec(ax, 'boundary', 'lon') is None
    assert _resolve_curve_spec(ax, 'boundary', 'lat') is None
    plt.close(fig)


def test_resolve_curve_spec_axis_lon_uses_center_lat():
    fig, ax = _axes("AIT", center=180)
    fc = _resolve_curve_spec(ax, 'axis', 'lon')
    assert isinstance(fc, _FrameCurve)
    assert fc.kind == 'lon'
    # Default center_lat is 0 for AIT center=180
    assert 'lat=0' in fc.name
    plt.close(fig)


def test_resolve_curve_spec_axis_lat_uses_center_lon():
    fig, ax = _axes("AIT", center=180)
    fc = _resolve_curve_spec(ax, 'axis', 'lat')
    assert isinstance(fc, _FrameCurve)
    assert fc.kind == 'lat'
    assert 'lon=180' in fc.name
    plt.close(fig)


def test_resolve_curve_spec_lat_value():
    fig, ax = _axes("AIT", center=180)
    fc = _resolve_curve_spec(ax, 'lat=45', 'lon')
    assert fc.kind == 'lon'
    assert fc.name == 'lat=45.0'
    plt.close(fig)


def test_resolve_curve_spec_lon_value():
    fig, ax = _axes("AIT", center=180)
    fc = _resolve_curve_spec(ax, 'lon=90', 'lat')
    assert fc.kind == 'lat'
    assert fc.name == 'lon=90.0'
    plt.close(fig)


def test_resolve_curve_spec_passes_framecurve_through():
    fig, ax = _axes("AIT", center=180)
    fc_in = _FrameCurve(np.array([[0., 0.], [1., 1.]]), kind='lon')
    fc_out = _resolve_curve_spec(ax, fc_in, 'lon')
    assert fc_out is fc_in
    plt.close(fig)


def test_resolve_curve_spec_backfills_missing_kind():
    fig, ax = _axes("AIT", center=180)
    fc_in = _FrameCurve(np.array([[0., 0.], [1., 1.]]))  # kind=None
    fc_out = _resolve_curve_spec(ax, fc_in, 'lat')
    assert fc_out.kind == 'lat'
    plt.close(fig)


def test_resolve_curve_spec_rejects_unknown_string():
    fig, ax = _axes("AIT", center=180)
    with pytest.raises(ValueError, match="not recognized"):
        _resolve_curve_spec(ax, 'middle', 'lon')
    plt.close(fig)


def test_resolve_curve_spec_rejects_non_string_non_framecurve():
    fig, ax = _axes("AIT", center=180)
    with pytest.raises(ValueError, match="must be 'boundary'"):
        _resolve_curve_spec(ax, 42, 'lon')
    plt.close(fig)


# ---- add_overlay_ticks: axis-curve mode end-to-end ----

def test_add_overlay_ticks_axis_mode_robinson():
    """``lon_at='axis'`` + ``lat_at='axis'`` on a Robinson produces
    lon ticks bowing with the equator and lat ticks along the
    central meridian — no boundary curve registered."""
    fig, ax = _axes("robinson", center=180)
    ov = add_overlay_ticks(ax, lon_at='axis', lat_at='axis')
    names = {fc.name for fc in ov.frame_curves}
    assert any('lat=0' in n for n in names)
    assert any('lon=180' in n for n in names)
    assert all('boundary' not in n for n in names)
    lon_ticks = [t for t in ov.gridticks if t.kind == 'lon']
    lat_ticks = [t for t in ov.gridticks if t.kind == 'lat']
    assert len(lon_ticks) >= 4
    assert len(lat_ticks) >= 4
    plt.close(fig)


def test_add_overlay_ticks_mixed_modes():
    """Lon labels on lat=20° parallel, lat labels on the boundary."""
    fig, ax = _axes("AIT", center=180)
    ov = add_overlay_ticks(ax, lon_at='lat=20', lat_at='boundary')
    names = {fc.name for fc in ov.frame_curves}
    # Boundary tagged for lat-only on AIT
    assert any('boundary' in n for n in names)
    assert any('lat=20' in n for n in names)
    plt.close(fig)


def test_add_overlay_ticks_lon_only_axis_lat_skipped():
    """``lat_at=None`` skips lat tick rendering."""
    fig, ax = _axes("robinson", center=180)
    ov = add_overlay_ticks(ax, lon_at='axis', lat_at=None,
                            suppress_default='lon')
    for t in ov.gridticks:
        assert t.kind == 'lon'
    plt.close(fig)


def test_add_overlay_ticks_boundary_default_unchanged():
    """Without lon_at/lat_at, the helper falls back to boundary mode
    (boundary curves serving both kinds)."""
    fig, ax = _axes("SIN", center=(180, 30), fov_deg=70.0)
    ov = add_overlay_ticks(ax)
    assert len(ov.frame_curves) == 1
    assert ov.frame_curves[0].name == 'boundary'
    assert ov.frame_curves[0].kind is None
    plt.close(fig)


def test_add_overlay_ticks_explicit_framecurve_in_lon_at():
    """A user-supplied _FrameCurve passes through with kind back-
    filled to 'lon' when used as lon_at."""
    fig, ax = _axes("AIT", center=180)
    custom = _FrameCurve.from_world_polyline(
        ax,
        np.column_stack([np.linspace(0., 360., 50),
                         np.full(50, 10.)]))
    ov = add_overlay_ticks(ax, lon_at=custom, lat_at='boundary')
    assert custom in ov.frame_curves
    assert custom.kind == 'lon'
    plt.close(fig)


def test_add_overlay_ticks_rejects_bad_curve_spec():
    fig, ax = _axes("AIT", center=180)
    with pytest.raises(ValueError):
        add_overlay_ticks(ax, lon_at='middle')
    plt.close(fig)


# ---- cross-frame axis curves (in-frame label mode) ----
#
# When ``frame=`` differs from the host axes' frame, ``from_const_lat`` /
# ``from_const_lon`` and the ``'axis'`` / ``'lat=N'`` / ``'lon=N'``
# specs on ``_resolve_curve_spec`` interpret the (lon, lat) value in
# the overlay frame, transforming via SkyCoord to the host frame
# before pixel-projection. This lets cross-frame overlay ticks land
# on the *overlay's* own central meridian / equator rather than the
# host's.

def test_from_const_lat_cross_frame_produces_different_pixels():
    """Cross-frame curve traces a different pixel polyline than the
    same-coord same-frame curve. Validates the SkyCoord transform
    actually shifts the samples.

    Cross-frame curves may pick up NaN rows from
    ``_insert_pixel_jump_breaks`` (wrap detection) so we compare
    only the finite portion sample-aligned to the same-frame curve.
    """
    fig, ax = _axes("MOL", center=0, frame="galactic")
    fc_same = _FrameCurve.from_const_lat(ax, 0.0)  # host = galactic
    fc_cross = _FrameCurve.from_const_lat(
        ax, 0.0, frame='geocentrictrueecliptic')
    finite_cross = fc_cross.xy_pix[
        ~np.isnan(fc_cross.xy_pix).any(axis=1)]
    # Compare overlap of the finite cross-frame samples vs. same-
    # frame samples (truncate to shortest if NaN insertion changed
    # length). Sample positions are uniformly spaced along the
    # parameter, so element-wise comparison is meaningful.
    n = min(len(finite_cross), len(fc_same.xy_pix))
    y_diff = np.abs(fc_same.xy_pix[:n, 1] - finite_cross[:n, 1])
    # At least some samples should differ by tens of pixels (the
    # ecliptic equator is tilted ~60° relative to galactic).
    assert (y_diff > 10).sum() > n // 4
    plt.close(fig)


def test_from_const_lat_same_frame_unchanged_when_frame_passed_explicitly():
    """Passing ``frame=`` equal to the host frame is a no-op vs. the
    default (no transform required)."""
    fig, ax = _axes("MOL", center=0, frame="galactic")
    fc_default = _FrameCurve.from_const_lat(ax, 30.0)
    fc_explicit = _FrameCurve.from_const_lat(ax, 30.0, frame='galactic')
    np.testing.assert_allclose(fc_default.xy_pix, fc_explicit.xy_pix)
    plt.close(fig)


def test_from_const_lon_cross_frame_produces_different_pixels():
    """Cross-frame meridian (e.g. ecliptic prime meridian on a
    galactic axes) differs in pixel space from the same-frame one."""
    fig, ax = _axes("MOL", center=0, frame="galactic")
    fc_same = _FrameCurve.from_const_lon(ax, 0.0)
    fc_cross = _FrameCurve.from_const_lon(
        ax, 0.0, frame='geocentrictrueecliptic')
    assert fc_same.xy_pix.shape == fc_cross.xy_pix.shape
    x_diff = np.abs(fc_same.xy_pix[:, 0] - fc_cross.xy_pix[:, 0])
    assert (x_diff > 10).sum() > len(x_diff) // 4
    plt.close(fig)


def test_from_const_lat_cross_frame_traces_overlay_equator():
    """Sample points along the cross-frame curve should round-trip
    back to lat ≈ 0 in the overlay frame."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    fig, ax = _axes("MOL", center=0, frame="galactic")
    fc = _FrameCurve.from_const_lat(
        ax, 0.0, frame='geocentrictrueecliptic', n=20)
    # Inverse-transform the pixel coords back to world (galactic),
    # then to ecliptic, and check lat ≈ 0.
    inv = ax.get_transform('world').inverted()
    world_gal = inv.transform(fc.xy_pix)  # (N, 2) in galactic deg
    # Drop projection-boundary samples (NaN pixels) BEFORE the coordinate
    # transform. Feeding NaN world coords through transform_to makes erfa emit
    # "invalid value encountered in ld/anp" RuntimeWarnings; filtering first
    # keeps the check identical (we skip those samples anyway) and clean.
    finite_in = np.isfinite(world_gal).all(axis=1)
    world_gal = world_gal[finite_in]
    sc = SkyCoord(world_gal[:, 0] * u.deg, world_gal[:, 1] * u.deg,
                  frame='galactic')
    ecl = sc.transform_to('geocentrictrueecliptic')
    lat_ecl = ecl.lat.deg
    finite = np.isfinite(lat_ecl)
    np.testing.assert_allclose(lat_ecl[finite], 0.0, atol=1e-3)
    plt.close(fig)


def test_resolve_curve_spec_axis_cross_frame_uses_overlay_zero():
    """'axis' with a non-host frame builds the curve at (0, 0) in
    the overlay frame — not at the host's projection center."""
    fig, ax = _axes("MOL", center=0, frame="galactic")
    # Same-frame 'axis' on a centered MOL → curve at host lat=0
    # (the horizontal middle of the bbox).
    fc_same = _resolve_curve_spec(ax, 'axis', 'lon')
    # Cross-frame 'axis' → curve at ecliptic β=0 (a tilted curve).
    fc_cross = _resolve_curve_spec(ax, 'axis', 'lon',
                                   frame='geocentrictrueecliptic')
    # Strip any NaN wrap-break rows before comparing y-range.
    cross_finite = fc_cross.xy_pix[
        ~np.isnan(fc_cross.xy_pix).any(axis=1)]
    same_y_range = fc_same.xy_pix[:, 1].max() - fc_same.xy_pix[:, 1].min()
    cross_y_range = cross_finite[:, 1].max() - cross_finite[:, 1].min()
    # Same-frame curve essentially horizontal; cross-frame tilted.
    assert cross_y_range > 5 * same_y_range
    plt.close(fig)


def test_resolve_curve_spec_axis_same_frame_preserves_legacy():
    """'axis' with frame=None (or matching host) preserves today's
    behavior — curve at the host's projection center."""
    fig, ax = _axes("MOL", center=0, frame="galactic")
    fc_default = _resolve_curve_spec(ax, 'axis', 'lon')
    fc_same = _resolve_curve_spec(ax, 'axis', 'lon', frame='galactic')
    np.testing.assert_allclose(fc_default.xy_pix, fc_same.xy_pix)
    plt.close(fig)


def test_resolve_curve_spec_lat_n_cross_frame_uses_overlay_coords():
    """'lat=N' with a non-host frame builds the curve at lat=N in
    the overlay frame."""
    fig, ax = _axes("MOL", center=0, frame="galactic")
    fc_same = _resolve_curve_spec(ax, 'lat=30', 'lon')
    fc_cross = _resolve_curve_spec(ax, 'lat=30', 'lon',
                                   frame='geocentrictrueecliptic')
    # Same-frame lat=30 → horizontal at galactic b=30 (above middle).
    # Cross-frame lat=30 → ecliptic β=30 curve (different shape).
    # Cross-frame may have NaN wrap-breaks; sample-align the finite
    # portion against the shorter of the two arrays.
    cross_finite = fc_cross.xy_pix[
        ~np.isnan(fc_cross.xy_pix).any(axis=1)]
    n = min(len(fc_same.xy_pix), len(cross_finite))
    y_diff = np.abs(fc_same.xy_pix[:n, 1] - cross_finite[:n, 1])
    assert y_diff.max() > 10
    plt.close(fig)


def test_add_overlay_ticks_cross_frame_axis_mode_populates_gridticks():
    """End-to-end: add_overlay_ticks with cross-frame + 'axis' specs
    discovers non-empty ticks anchored to the overlay's central
    curves."""
    fig, ax = _axes("MOL", center=0, frame="galactic")
    ov = add_overlay_ticks(ax, frame='geocentrictrueecliptic',
                           lon_at='axis', lat_at='axis',
                           show_ticks=False, show_labels=False)
    # Should discover several ticks (lon meridians × overlay equator
    # plus lat parallels × overlay central meridian).
    assert len(ov.gridticks) > 5
    # Lon ticks live on the lon-kind curve; lat on lat-kind.
    kinds = {t.kind for t in ov.gridticks}
    assert kinds == {'lon', 'lat'}
    plt.close(fig)


def test_add_overlay_ticks_cross_frame_axis_equals_lat0_lon0():
    """'axis' on a cross-frame call should produce the same curves
    as explicit 'lat=0' / 'lon=0' (since the overlay's axis is
    defined as (0, 0) in overlay coords)."""
    fig, ax = _axes("MOL", center=0, frame="galactic")
    fc_axis_lon = _resolve_curve_spec(ax, 'axis', 'lon',
                                      frame='geocentrictrueecliptic')
    fc_lat0 = _resolve_curve_spec(ax, 'lat=0', 'lon',
                                  frame='geocentrictrueecliptic')
    # Both should be built with the same defaults (n=200, default
    # lon range) and produce identical curves.
    np.testing.assert_allclose(fc_axis_lon.xy_pix, fc_lat0.xy_pix)
    plt.close(fig)


# ---- wrap-jump breaks + per-gridline dedup on axis curves ----
#
# Cross-frame axis curves (``from_const_lat`` / ``from_const_lon``
# with a non-host ``frame=``) trace closed loops on the sphere;
# projected to pixel space they typically wrap once across the
# host's antimeridian, producing a long straight segment connecting
# opposite edges. ``_insert_pixel_jump_breaks`` plants NaN rows at
# those jumps so the intersection step (which splits on NaN) won't
# find spurious meridian/parallel crossings on the wrap line.
# A safety-net per-(gridline, axis-curve) dedup in ``discover_ticks``
# folds any remaining exact-pixel duplicates (gridline endpoint
# extensions meeting at the intersection).

def test_cross_frame_curve_has_nan_break_at_wrap():
    """A cross-frame parallel that wraps the host antimeridian
    should pick up a NaN row marking the discontinuity."""
    fig, ax = _axes("MOL", center=0, frame="galactic")
    fc = _FrameCurve.from_const_lat(
        ax, 0.0, frame='geocentrictrueecliptic')
    assert np.isnan(fc.xy_pix).any(), \
        "expected at least one NaN break in wrapping cross-frame curve"
    plt.close(fig)


def test_same_frame_curve_has_no_nan_break():
    """Same-frame curves are wrap-safe by construction — no NaN
    rows should be inserted (preserves backward-compatible drawing)."""
    fig, ax = _axes("MOL", center=0, frame="galactic")
    fc = _FrameCurve.from_const_lat(ax, 0.0)  # frame=None, default
    assert not np.isnan(fc.xy_pix).any()
    plt.close(fig)


def test_discover_ticks_dedup_axis_curve_default_collapses_duplicates():
    """Cross-frame 'axis' on galactic MOL + ecliptic overlay should
    produce exactly one tick per overlay gridline (the canonical
    in-frame-mode setup)."""
    fig, ax = _axes("MOL", center=0, frame="galactic")
    ov = add_overlay_ticks(
        ax, frame='geocentrictrueecliptic',
        lon_at='axis', lat_at='axis',
        show_ticks=False, show_labels=False)
    # Each gridline (kind, value) pair should appear at most once.
    from collections import Counter
    per_gl = Counter((t.gridline.kind, t.gridline.value)
                     for t in ov.gridticks)
    duplicates = {k: v for k, v in per_gl.items() if v > 1}
    assert not duplicates, f"unexpected duplicates: {duplicates}"
    plt.close(fig)


def test_discover_ticks_dedup_does_not_touch_boundary_curves():
    """Boundary curves (axis_curve=False) keep their multiple
    crossings even with dedup on."""
    fig, ax = _axes("MOL", center=0, frame="galactic")
    # Boundary mode — many lon-grid parallels cross the MOL ellipse
    # at multiple boundary points (top + bottom).
    ov = add_overlay_ticks(
        ax, frame='geocentrictrueecliptic',
        lon_at='boundary', lat_at='boundary',
        show_ticks=False, show_labels=False)
    from collections import Counter
    per_gl = Counter((t.gridline.kind, t.gridline.value)
                     for t in ov.gridticks)
    # Parallels should each cross the elliptical boundary twice.
    multi = {k: v for k, v in per_gl.items() if v > 1}
    assert multi, "expected boundary parallels to have multiple ticks"
    plt.close(fig)


def test_framecurve_axis_curve_flag_defaults_false():
    """Hand-built _FrameCurves are not axis curves by default."""
    fc = _FrameCurve(np.array([[0., 0.], [1., 1.]]))
    assert fc.axis_curve is False


def test_from_const_lat_sets_axis_curve_true():
    """The axis-curve factories mark themselves so discover_ticks
    can dedup their ticks."""
    fig, ax = _axes("MOL", center=0, frame="galactic")
    assert _FrameCurve.from_const_lat(ax, 0.0).axis_curve is True
    assert _FrameCurve.from_const_lon(ax, 0.0).axis_curve is True
    plt.close(fig)


def test_tan_field_overlay_ticks_clipped_to_bbox():
    """Regression: on a flat (TAN) field, a constant-coord overlay curve
    sampled across the sky runs toward infinity (gnomonic), so a tick/label
    could land thousands of pixels off-canvas — exploding
    savefig(bbox_inches='tight'). Discovered ticks are now clipped to the
    axes bbox on any frame shape, so the off-frame ones are dropped."""
    import os

    fig, ax = _axes("TAN", center=(83.63, 22.01), fov_deg=4,
                    tick_style="native")
    ov = add_overlay_ticks(ax, lon_at="lat=23", lat_at="lon=85",
                           suppress_default="both")
    bb = ax.bbox
    margin = 50.0
    for t in ov.gridticks:
        x, y = t.xy_pix
        assert (bb.x0 - margin <= x <= bb.x1 + margin
                and bb.y0 - margin <= y <= bb.y1 + margin), \
            f"overlay tick at ({x:.0f},{y:.0f}) escaped bbox"
    # tight bbox must produce a sanely-sized image (not gigantic).
    path = "/tmp/_sph_tan_tight_test.png"
    fig.savefig(path, bbox_inches="tight", dpi=100)
    h, w = plt.imread(path).shape[:2]
    assert w < 5000 and h < 5000, f"tight-bbox image exploded to {w}x{h}"
    os.remove(path)
    plt.close(fig)


def test_tan_field_overlay_ticks_kept_when_in_field():
    """In-field overlay gridline values still produce ticks (the clip drops
    only the off-frame ones)."""
    fig, ax = _axes("TAN", center=(83.63, 22.01), fov_deg=4,
                    tick_style="native")
    ov = add_overlay_ticks(ax, lon_at="lat=22", lat_at="lon=83.6",
                           lon_vals=[82, 83, 84, 85], lat_vals=[21, 22, 23],
                           suppress_default="both")
    assert len(ov.gridticks) > 0, "in-field overlay ticks were dropped"
    plt.close(fig)


def test_tan_field_overlay_ticks_default_field_scale_vals():
    """A direct add_overlay_ticks on a zoomed same-frame field, with no
    explicit lon_vals/lat_vals, now derives field-scale graticule values from
    the visible extent — so in-frame ticks appear instead of zero (the all-sky
    30°/15° defaults never cross a few-degree field)."""
    fig, ax = _axes("TAN", center=(83.63, 22.01), fov_deg=4,
                    tick_style="native")
    ov = add_overlay_ticks(ax, lon_at="lat=22", lat_at="lon=83.6",
                           suppress_default="both")
    assert len(ov.gridticks) > 0, "no field-scale default ticks derived"
    plt.close(fig)


def test_allsky_overlay_ticks_keep_default_graticule():
    """All-sky frames keep the 30°/15° default graticule (the field-scale
    default is gated to zoomed field frames)."""
    fig, ax = _axes("AIT", center=0, frame="galactic")
    ov = add_overlay_ticks(ax, suppress_default="both")
    # boundary ticks from the 30°/15° graticule → plenty of crossings
    assert len(ov.gridticks) > 10
    plt.close(fig)


def test_globe_overlay_gridlines_not_field_restricted():
    """A circular-limb globe (off-pole SIN) keeps full -90..90 overlay
    graticule sampling. The flat-field extent restriction (used for the
    in-frame-tick feature on rectangular fields) would shrink the gridlines
    short of the limb — it is gated off for globes via _sph_is_globe."""
    from skyplothelper.coord_overlay import (
        CoordinateOverlay,
        _get_wcs_frame_name,
    )
    fig, ax = _axes("SIN", center=(60, 35), fov_deg=120)
    assert getattr(ax, "_sph_is_globe", False) is True
    ov = CoordinateOverlay(ax, frame=_get_wcs_frame_name(ax))  # same-frame
    mer = ov.lon_gridlines[0]
    assert mer.lats.min() < -89 and mer.lats.max() > 89, \
        "globe meridian sampling was shrunk to the field extent"
    plt.close(fig)


def test_flat_tan_field_is_not_a_globe():
    """A flat (rectangular) TAN field is NOT a globe, so its same-frame overlay
    gridlines stay restricted to the visible field extent (in-frame ticks)."""
    from skyplothelper.coord_overlay import (
        CoordinateOverlay,
        _get_wcs_frame_name,
    )
    fig, ax = _axes("TAN", center=(83.63, 22.01), fov_deg=4)
    assert getattr(ax, "_sph_is_globe", False) is False
    ov = CoordinateOverlay(ax, frame=_get_wcs_frame_name(ax))  # same-frame
    mer = ov.lon_gridlines[0]
    assert (mer.lats.max() - mer.lats.min()) < 30, \
        "flat field gridlines were not restricted to the field extent"
    plt.close(fig)
