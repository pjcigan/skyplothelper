"""Tests for skyplothelper.plotly (web-export helpers)."""

import sys
from importlib import reload

import numpy as np
import pytest

# Plotly is optional — skip the whole module if missing.
plotly = pytest.importorskip("plotly")
from skyplothelper import plotly as sphpl  # noqa: E402
from skyplothelper.projections.project import project as _project  # noqa: E402

# ============================================================================
# make_figure
# ============================================================================

def test_make_figure_returns_plotly_figure():
    import plotly.graph_objects as go
    fig = sphpl.make_figure(projection='AIT', center=0)
    assert isinstance(fig, go.Figure)


def test_make_figure_tags_metadata_for_later_calls():
    """make_figure stashes projection/center/etc on fig.layout.meta so
    follow-up add_* calls don't need to repeat them."""
    fig = sphpl.make_figure(projection='MOL', center=180, theme='dark')
    meta = dict(fig.layout.meta)
    assert meta['sph_projection'] == 'MOL'
    assert meta['sph_center'] == 180
    assert meta['sph_direction'] == 'sky'


def test_make_figure_invalid_theme_raises():
    with pytest.raises(ValueError, match="theme"):
        sphpl.make_figure(theme='neon')


def test_make_figure_grid_off():
    """show_grid=False produces a figure with no graticule traces."""
    fig = sphpl.make_figure(show_grid=False)
    assert len(fig.data) == 0


def test_make_figure_grid_on_produces_traces():
    fig = sphpl.make_figure(show_grid=True)
    assert len(fig.data) > 0


@pytest.mark.parametrize('projection', ['AIT', 'MOL', 'CAR', 'HPX'])
@pytest.mark.parametrize('center', [0.0, 180.0])
def test_make_figure_graticule_closes_both_frame_edges(projection, center):
    """The wrap meridian bounds the frame on both sides.

    ``center - 180`` and ``center + 180`` are one great circle projecting to
    opposite frame edges, and a half-open ``arange`` used to emit it only
    once — leaving the frame visibly open down one side.
    """
    fig = sphpl.make_figure(projection=projection, center=center,
                            show_grid=True)
    xs = np.concatenate([np.asarray(t.x, dtype=float) for t in fig.data])
    xs = xs[np.isfinite(xs)]
    # The equator reaches the frame's full half-width; a meridian has to
    # reach it too, on each side.
    lon = np.linspace(center - 180, center + 180, 361)
    eq_x, _ = _project(lon, np.zeros_like(lon), projection=projection,
                       center=center, lat_center=0.0, direction='sky')
    half_width = np.nanmax(np.abs(np.asarray(eq_x, dtype=float)))
    assert xs.max() == pytest.approx(half_width, rel=1e-6)
    assert xs.min() == pytest.approx(-half_width, rel=1e-6)


@pytest.mark.parametrize('projection', ['COD', 'COE', 'COO', 'COP', 'BON'])
def test_make_figure_builds_conic_projections(projection):
    """Conic and Bonne projections need a standard parallel (PV2_1); without
    one wcslib rejects the header, and make_figure raised outright."""
    fig = sphpl.make_figure(projection=projection, center=0.0, show_grid=True)
    assert len(fig.data) > 0
    xs = np.concatenate([np.asarray(t.x, dtype=float) for t in fig.data])
    assert np.isfinite(xs).any()


@pytest.mark.parametrize('projection', ['TAN', 'SIN'])
def test_make_figure_graticule_skips_offsky_wrap_meridian(projection):
    """On a zoomed / hemispheric projection the wrap meridian is off-sky and
    projects to all-NaN; it must not become an empty trace."""
    fig = sphpl.make_figure(projection=projection, center=0.0, show_grid=True)
    for trace in fig.data:
        assert np.isfinite(np.asarray(trace.x, dtype=float)).any()


# ---------------------------------------------------------------------------
# make_figure: fov_deg / extent
# ---------------------------------------------------------------------------
#
# Without a field-of-view concept, a non-all-sky projection (TAN, SIN, AZP)
# drew a full-sky graticule and autoscaled to a useless -- or, at the gnomonic
# horizon, divergent -- extent. fov_deg windows the graticule to a field and
# ranges the axes to its projected extent; extent sets the range explicitly.

def _finite_extent(fig):
    xs = np.concatenate([np.asarray(t.x, dtype=float) for t in fig.data])
    ys = np.concatenate([np.asarray(t.y, dtype=float) for t in fig.data])
    xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
    return xs.min(), xs.max(), ys.min(), ys.max()


def test_make_figure_all_sky_default_sets_no_axis_range():
    """The default path is untouched: no explicit range, autoscale as before."""
    fig = sphpl.make_figure(projection='AIT', center=180, show_grid=True)
    assert fig.layout.xaxis.range is None
    assert fig.layout.yaxis.range is None


@pytest.mark.parametrize('projection', ['TAN', 'SIN', 'AZP', 'STG', 'ZEA'])
def test_make_figure_fov_bounds_the_field(projection):
    """A zoomed field projects to ~fov/2 in each direction, however wild the
    projection's full-sky extent would have been."""
    fov = 8.0
    fig = sphpl.make_figure(projection=projection, center=83.8,
                            lat_center=-5.4, fov_deg=fov, show_grid=True)
    x0, x1, y0, y1 = _finite_extent(fig)
    # Half-field is 4°; allow generous slack for projection curvature.
    assert max(abs(x0), abs(x1)) < fov
    assert (y1 - y0) < 2.5 * fov
    assert fig.layout.xaxis.range is not None


def test_make_figure_fov_axis_range_matches_projected_extent():
    fig = sphpl.make_figure(projection='TAN', center=0.0, lat_center=0.0,
                            fov_deg=6.0, show_grid=False)
    x0, x1, y0, y1 = sphpl.core._fov_extent('TAN', 0.0, 0.0, 'sky', 6.0)
    # Ascending, both axes: the east-left flip is in the data, not the axis.
    assert list(fig.layout.xaxis.range) == pytest.approx([x0, x1])
    assert list(fig.layout.yaxis.range) == pytest.approx([y0, y1])


def _east_screen_frac(fig, center, lat, projection, direction):
    """Left-to-right screen position (0=left, 1=right) of a point 5° EAST of
    the field center, under the figure's axis range. project() encodes the
    direction convention in the data, and the axis range maps data → screen,
    so this is where the point actually lands."""
    x0, x1 = fig.layout.xaxis.range
    xe, _ = _project([center + 5.0], [lat], projection=projection,
                     center=center, lat_center=lat, direction=direction)
    xc, _ = _project([center], [lat], projection=projection,
                     center=center, lat_center=lat, direction=direction)
    span = x1 - x0
    return (float(xe[0]) - x0) / span, (float(xc[0]) - x0) / span


def test_make_figure_fov_renders_east_left_for_sky_east_right_for_geo():
    """The whole point of the fix: a zoomed 'sky' field must render east to
    the LEFT (like the all-sky figures), not mirror it. A raw axis reversal
    flipped it east-right; the flip belongs in the data, applied once."""
    kw = dict(projection='TAN', center=83.8, lat_center=-5.4, fov_deg=8)
    sky = sphpl.make_figure(direction='sky', **kw)
    e_sky, c_sky = _east_screen_frac(sky, 83.8, -5.4, 'TAN', 'sky')
    assert e_sky < c_sky, "sky: east should render left of center"

    geo = sphpl.make_figure(direction='geographic', **kw)
    e_geo, c_geo = _east_screen_frac(geo, 83.8, -5.4, 'TAN', 'geographic')
    assert e_geo > c_geo, "geographic: east should render right of center"

    # Both use an ascending x-range; the difference lives in the data.
    assert sky.layout.xaxis.range[0] < sky.layout.xaxis.range[1]
    assert geo.layout.xaxis.range[0] < geo.layout.xaxis.range[1]


def test_make_figure_fov_auto_grid_spacing_draws_lines():
    """A 2° field with the 30° default would draw nothing; auto-spacing keeps
    the graticule populated."""
    small = sphpl.make_figure(projection='TAN', center=0, lat_center=0,
                              fov_deg=2.0, show_grid=True)
    assert len(small.data) >= 4


def test_make_figure_fov_explicit_grid_spacing_overrides_auto():
    auto = sphpl.make_figure(projection='TAN', center=0, fov_deg=10)
    fine = sphpl.make_figure(projection='TAN', center=0, fov_deg=10,
                             grid_lon_spacing=1, grid_lat_spacing=1)
    assert len(fine.data) > len(auto.data)


def test_make_figure_extent_sets_range_without_windowing_graticule():
    fig = sphpl.make_figure(projection='TAN', center=0.0,
                            extent=(-4, 4, -4, 4), show_grid=True)
    # extent is in projection-plane (data) coords, applied verbatim and
    # ascending — no direction-dependent axis reversal.
    assert list(fig.layout.yaxis.range) == [-4, 4]
    assert list(fig.layout.xaxis.range) == [-4, 4]


def test_make_figure_fov_and_extent_are_mutually_exclusive():
    with pytest.raises(TypeError, match="only"):
        sphpl.make_figure(projection='TAN', fov_deg=5, extent=(-1, 1, -1, 1))


def test_make_figure_fov_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        sphpl.make_figure(projection='TAN', fov_deg=-3)


def test_make_figure_fov_all_off_sky_raises(monkeypatch):
    """Defensive guard: if a field projects to nothing, _fov_extent raises
    rather than returning a NaN bbox that would silently break the figure.
    Not reachable through normal params (a field is centered on the
    projection center, which always projects), so drive it by forcing the
    projection to all-NaN."""
    def _all_nan(lon, lat, **kw):
        a = np.full(np.shape(lon), np.nan, dtype=float)
        return a, a.copy()
    monkeypatch.setattr(sphpl.core, '_project', _all_nan)
    with pytest.raises(ValueError, match="off-sky"):
        sphpl.core._fov_extent('TAN', 0.0, 0.0, 'sky', 8.0)


# ============================================================================
# add_scatter
# ============================================================================

def test_add_scatter_projects_and_adds_trace():
    fig = sphpl.make_figure(show_grid=False)
    n_before = len(fig.data)
    trace = sphpl.add_scatter(fig, [0, 90, 180], [0, 30, -30])
    assert len(fig.data) == n_before + 1
    # Trace x/y match the manual project() output
    x_expected, y_expected = _project([0, 90, 180], [0, 30, -30],
                                       projection='AIT', center=0,
                                       direction='sky')
    np.testing.assert_allclose(trace.x, x_expected, atol=1e-10)
    np.testing.assert_allclose(trace.y, y_expected, atol=1e-10)


def test_add_scatter_default_hover_includes_lonlat():
    fig = sphpl.make_figure(show_grid=False)
    trace = sphpl.add_scatter(fig, [10, 20], [5, -5])
    assert 'RA' in trace.hovertemplate
    assert 'Dec' in trace.hovertemplate
    # customdata should be (lon, lat) pairs
    assert trace.customdata.shape == (2, 2)
    np.testing.assert_allclose(trace.customdata[:, 0], [10, 20])
    np.testing.assert_allclose(trace.customdata[:, 1], [5, -5])


def test_add_scatter_respects_figure_projection_metadata():
    """If the figure was made with projection='MOL', add_scatter uses
    MOL unless overridden."""
    fig = sphpl.make_figure(projection='MOL', center=180, show_grid=False)
    trace = sphpl.add_scatter(fig, [180], [0])
    # Point at the projection center → (0, 0)
    assert np.isclose(trace.x[0], 0.0, atol=1e-6)
    assert np.isclose(trace.y[0], 0.0, atol=1e-6)


def test_add_scatter_custom_hovertemplate():
    fig = sphpl.make_figure(show_grid=False)
    trace = sphpl.add_scatter(fig, [10], [5],
                                customdata=np.array([['NGC 1234']]),
                                hovertemplate='%{customdata[0]}')
    assert trace.hovertemplate == '%{customdata[0]}'


# ============================================================================
# add_healpix
# ============================================================================

def test_add_healpix_requires_healpy():
    """Module raises a clear error when healpy isn't importable."""
    healpy = pytest.importorskip("healpy")  # noqa: F841 — gate the test
    # If healpy IS installed, this test trivially passes the import. We
    # exercise the error path via a different test below.


def test_add_healpix_adds_at_least_one_trace_per_finite_tile():
    healpy = pytest.importorskip("healpy")
    nside = 4
    npix = healpy.nside2npix(nside)
    vals = np.arange(npix, dtype=float)
    vals[0] = np.nan  # Drop one tile
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix(fig, vals, nside=nside, tile_resolution=2)
    # All finite tiles render. Tiles that straddle the projection's
    # wrap edge get SPLIT into two sub-polygons (one on each side of
    # the wrap), so the trace count is >= npix - 1.
    assert len(traces) >= npix - 1
    assert len(fig.data) == len(traces)
    # Each tile trace has fill='toself' and a pre-formatted hover text.
    assert traces[0].fill == 'toself'
    # Hover restricted to fill area (not vertices) so zoomed views
    # don't overlay trace-name labels.
    assert traces[0].hoveron == 'fills'


def test_add_healpix_wrap_edge_tiles_are_split_not_dropped():
    """Tiles straddling the wrap edge of a non-zero-centered projection
    should be split into two sub-polygons. For nside=16 with center=180,
    a handful of tiles at lon ≈ 0/360 straddle the wrap; trace count
    should exceed npix."""
    healpy = pytest.importorskip("healpy")
    nside = 16
    npix = healpy.nside2npix(nside)
    vals = np.ones(npix)
    fig = sphpl.make_figure(projection='AIT', center=180, show_grid=False)
    traces = sphpl.add_healpix(fig, vals, nside=nside, tile_resolution=2)
    assert len(traces) > npix    # split traces add extras


def test_add_healpix_value_size_mismatch_raises():
    healpy = pytest.importorskip("healpy")  # noqa: F841
    fig = sphpl.make_figure(show_grid=False)
    with pytest.raises(ValueError, match="does not match"):
        sphpl.add_healpix(fig, np.zeros(5), nside=4)


def test_add_healpix_hover_uses_text_field():
    """The hover template references the trace's ``text`` field rather
    than ``customdata``, because ``hoveron='fills'`` doesn't index
    customdata by vertex. The text itself carries the formatted RA /
    Dec / value / ipix string per tile."""
    healpy = pytest.importorskip("healpy")
    nside = 2
    vals = np.arange(healpy.nside2npix(nside), dtype=float)
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix(fig, vals, nside=nside, tile_resolution=1)
    assert '%{text}' in traces[0].hovertemplate
    # The text should embed RA / Dec / value / ipix for the tile.
    assert 'RA:' in traces[0].text
    assert 'Dec:' in traces[0].text
    assert 'value:' in traces[0].text
    assert 'ipix:' in traces[0].text


def test_add_healpix_traces_have_empty_name():
    """Tile traces are emitted with ``name=''`` so plotly does not fall
    back to ``"trace N"`` auto-labels, which would otherwise survive
    the ``<extra></extra>`` suppression as a small black hoverlabel
    under ``hoveron='fills'``."""
    healpy = pytest.importorskip("healpy")
    nside = 2
    vals = np.arange(healpy.nside2npix(nside), dtype=float)
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix(fig, vals, nside=nside, tile_resolution=1)
    for t in traces:
        assert t.name == ''


def test_add_healpix_default_hover_includes_ipix_values():
    """The default hover content embeds each tile's HEALPix pixel
    index. Wrap-edge splitting can produce multiple traces per tile,
    but every ipix in [0, npix) should appear at least once in the
    full set of trace texts."""
    healpy = pytest.importorskip("healpy")
    nside = 2
    npix = healpy.nside2npix(nside)
    vals = np.arange(npix, dtype=float)
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix(fig, vals, nside=nside, tile_resolution=1)
    seen = set()
    for t in traces:
        # Parse the ``ipix: N`` line from the per-tile hover text.
        for line in t.text.split('<br>'):
            if line.startswith('ipix:'):
                seen.add(int(line.split(':', 1)[1].strip()))
                break
    assert seen == set(range(npix))


def test_add_healpix_custom_hover_format_string():
    """hover_format can be a Python format string with any of {lon},
    {lat}, {value}, {ipix}."""
    healpy = pytest.importorskip("healpy")
    nside = 2
    vals = np.arange(healpy.nside2npix(nside), dtype=float)
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix(fig, vals, nside=nside,
                                 tile_resolution=1,
                                 hover_format='lon={lon:.1f}, val={value}')
    assert 'lon=' in traces[0].text
    assert 'val=' in traces[0].text


def test_add_healpix_custom_hover_format_string_with_ipix():
    """The ``{ipix}`` placeholder is honored by the format-string path."""
    healpy = pytest.importorskip("healpy")
    nside = 2
    vals = np.arange(healpy.nside2npix(nside), dtype=float)
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix(fig, vals, nside=nside,
                                 tile_resolution=1,
                                 hover_format='pix #{ipix}')
    assert traces[0].text == 'pix #0'


def test_add_healpix_custom_hover_format_callable_3arg():
    """Legacy 3-arg ``(lon, lat, value)`` callables are still accepted
    via signature introspection."""
    healpy = pytest.importorskip("healpy")
    nside = 2
    vals = np.arange(healpy.nside2npix(nside), dtype=float)
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix(
        fig, vals, nside=nside, tile_resolution=1,
        hover_format=lambda lon, lat, v: f"L{lon:.0f} B{lat:.0f} V{v:.0f}")
    assert 'L' in traces[0].text
    assert 'B' in traces[0].text


def test_add_healpix_custom_hover_format_callable_4arg():
    """4-arg ``(lon, lat, value, ipix)`` callables are passed the
    HEALPix tile index."""
    healpy = pytest.importorskip("healpy")
    nside = 2
    vals = np.arange(healpy.nside2npix(nside), dtype=float)
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix(
        fig, vals, nside=nside, tile_resolution=1,
        hover_format=lambda lon, lat, v, ipix: f"#{ipix} v={v:.0f}")
    assert traces[0].text == '#0 v=0'


# ============================================================================
# Constellation overlays
# ============================================================================

def test_add_constellation_boundaries_returns_trace():
    fig = sphpl.make_figure(show_grid=False)
    trace = sphpl.add_constellation_boundaries(fig)
    assert trace.mode == 'lines'
    # The trace should have non-trivial coordinate arrays.
    assert len(trace.x) > 100


def test_add_constellation_lines_returns_trace():
    fig = sphpl.make_figure(show_grid=False)
    trace = sphpl.add_constellation_lines(fig, constellations=['ORI'])
    assert trace.mode == 'lines'
    # Orion has multiple line segments; trace x/y should not be empty.
    assert len(trace.x) > 5


def test_add_constellation_boundaries_wrap_split_off_center():
    """Boundary segments crossing the projection seam on a non-180 center
    are wrap-split, not streaked across the frame. Regression for the
    center-agnostic split that only handled the lon=0/360 antimeridian."""
    fig = sphpl.make_figure(show_grid=False, center=90.0)  # seam at lon=270
    trace = sphpl.add_constellation_boundaries(fig)
    x = np.asarray([np.nan if v is None else v for v in trace.x], dtype=float)
    d = np.abs(np.diff(x))
    d = d[np.isfinite(d)]
    full_width = float(np.nanmax(x) - np.nanmin(x))
    # No segment streaks across the frame (pre-fix the worst jump ~= the
    # full frame width).
    assert d.size and float(np.max(d)) < 0.5 * full_width


@pytest.mark.parametrize("center", [180.0, 0.0])
def test_add_constellation_boundaries_no_streak_seam_at_ra0(center):
    """At center=180 the seam sits at RA=0, where crossing segments have
    numerically far-apart endpoints (e.g. 359°/1°). A raw linspace densified
    those the long way around the sphere, producing per-step diffs too small
    for the wrap-splitter to catch — so the leg streaked straight across the
    canvas. Densifying the short way fixes it. (center=0 is the control: the
    seam at RA=180 never triggered the bug.)"""
    fig = sphpl.make_figure(projection="AIT", center=center, show_grid=False)
    trace = sphpl.add_constellation_boundaries(fig)
    x = np.asarray([np.nan if v is None else v for v in trace.x], dtype=float)
    d = np.abs(np.diff(x))
    d = d[np.isfinite(d)]
    full_width = float(np.nanmax(x) - np.nanmin(x))
    assert d.size and float(np.max(d)) < 0.25 * full_width


def test_add_constellation_labels_returns_text_trace():
    fig = sphpl.make_figure(show_grid=False)
    trace = sphpl.add_constellation_labels(fig, labels='abbr')
    assert trace.mode == 'text'
    assert len(trace.text) == 88   # all 88 IAU constellations


def test_constellation_overlays_default_to_figure_projection():
    """When called on a figure made with a non-default projection, the
    overlay uses that projection, not the implicit AIT/center=0."""
    fig = sphpl.make_figure(projection='MOL', center=180, show_grid=False)
    # ORI is roughly at RA=85, Dec=0. Project it manually under MOL/180.
    x_expected, _ = _project([85.], [0.], projection='MOL', center=180,
                              direction='sky')
    trace = sphpl.add_constellation_lines(fig, constellations=['ORI'])
    # The trace's x coords should be near x_expected (Orion sits
    # somewhat near RA=85), and far from what AIT/center=0 would give.
    x_AIT_center0, _ = _project([85.], [0.], projection='AIT', center=0,
                                 direction='sky')
    # The MOL/180 expectation is roughly in the bulk of trace.x range.
    finite_x = [v for v in trace.x if np.isfinite(v)]
    median_x = float(np.median(finite_x))
    assert abs(median_x - float(x_expected[0])) < abs(median_x - float(x_AIT_center0[0]))


# ============================================================================
# project re-export
# ============================================================================

def test_project_reexport_matches_projections_module():
    """sphpl.project is the same callable as skyplothelper.project."""
    from skyplothelper.projections.project import project as proj_root
    assert sphpl.project is not None
    # The two functions return identical output
    x1, y1 = sphpl.project([10, 20], [0, 5], projection='AIT')
    x2, y2 = proj_root([10, 20], [0, 5], projection='AIT')
    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(y1, y2)


# ============================================================================
# Lazy-import behavior
# ============================================================================

def test_skyplothelper_imports_without_plotly(monkeypatch):
    """``import skyplothelper`` should not require plotly even though
    skyplothelper.plotly exists. The plotly dependency is gated behind
    each function's first call."""
    # Hide plotly from the import system.
    real_import = __builtins__['__import__'] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'plotly' or name.startswith('plotly.'):
            raise ImportError('simulated: plotly not installed')
        return real_import(name, *args, **kwargs)

    # Drop pre-imported plotly modules
    for k in list(sys.modules):
        if k.startswith('plotly'):
            monkeypatch.delitem(sys.modules, k)
    if 'skyplothelper' in sys.modules:
        monkeypatch.delitem(sys.modules, 'skyplothelper')
    monkeypatch.setattr('builtins.__import__', fake_import)

    import skyplothelper as sph_reimport
    # Re-import to be sure the cached state is fresh
    reload(sph_reimport)
    assert hasattr(sph_reimport, 'plotly')

    # Calling a plotly entry point should raise the friendly ImportError
    with pytest.raises(ImportError, match="plotly"):
        sph_reimport.plotly.make_figure()


# ============================================================================
# Core overlays: add_great_circle / add_plane_overlay / add_geodesic_circle
# / add_spherical_polygon
# ============================================================================

# -- add_great_circle ---

def test_add_great_circle_galactic_default():
    """add_great_circle defaults to the galactic plane."""
    fig = sphpl.make_figure(show_grid=False)
    trace = sphpl.add_great_circle(fig)
    assert trace.mode == 'lines'
    # The galactic plane sample contains finite (x, y) values.
    finite = sum(1 for v in trace.x if v is not None and np.isfinite(v))
    assert finite > 100


def test_add_great_circle_invalid_frame_raises():
    fig = sphpl.make_figure(show_grid=False)
    with pytest.raises(ValueError, match="Unknown frame"):
        sphpl.add_great_circle(fig, frame='not_a_frame')


def test_add_great_circle_lat_offset_distinct_curve():
    """A nonzero lat_offset traces a different curve than the great circle."""
    fig1 = sphpl.make_figure(show_grid=False)
    fig2 = sphpl.make_figure(show_grid=False)
    great = sphpl.add_great_circle(fig1, frame='galactic', lat_offset=0)
    parallel = sphpl.add_great_circle(fig2, frame='galactic',
                                       lat_offset=60)
    # The two traces should have different coordinate arrays
    # (different (x, y) sequences). Just check at least one of
    # x or y differs at a representative index.
    g_y = np.asarray([v for v in great.y if v is not None and np.isfinite(v)])
    p_y = np.asarray([v for v in parallel.y if v is not None and np.isfinite(v)])
    # Quantile spread differs between a great circle and a small parallel.
    assert abs(np.std(g_y) - np.std(p_y)) > 1.0


def _max_finite_step_ratio(trace):
    """max / median of segment lengths between *adjacent* samples (NaN-spanning
    seam breaks excluded). A smooth curve stays near 1; a zig-zag fan — from
    sorting a double-valued small circle by longitude — spikes into the 100s."""
    x = np.asarray(trace.x, dtype=float)
    y = np.asarray(trace.y, dtype=float)
    seg = np.hypot(np.diff(x), np.diff(y))
    seg = seg[np.isfinite(seg)]
    return float(seg.max() / np.median(seg))


def test_add_great_circle_small_circle_not_fanned():
    """A small circle (lat_offset != 0, not enclosing the pole) is double-valued
    in RA; it must trace in path order, not be sorted by longitude into a fan.
    Regression for the plotly small-circle zig-zag bug."""
    fig = sphpl.make_figure('AIT', center=180, show_grid=False)
    for off in (30, -30, 60, -60):
        sc = sphpl.add_great_circle(fig, frame='galactic', lat_offset=off)
        assert _max_finite_step_ratio(sc) < 10.0, f"fan at lat_offset={off}"


def test_add_great_circle_great_circle_still_clean():
    """The path-order fix must keep great circles seam-clean (no fan)."""
    fig = sphpl.make_figure('AIT', center=180, show_grid=False)
    gc = sphpl.add_great_circle(fig, frame='galactic', lat_offset=0)
    assert _max_finite_step_ratio(gc) < 10.0


def test_add_plane_overlay_parallels_not_fanned():
    """add_plane_overlay(parallels=...) delegates to add_great_circle with a
    lat_offset, so its parallels must not fan either."""
    fig = sphpl.make_figure('AIT', center=180, show_grid=False)
    traces = sphpl.add_plane_overlay(fig, plane='galactic',
                                     parallels=[-40, 40])
    for t in traces:
        assert _max_finite_step_ratio(t) < 10.0


def test_add_great_circle_pole_frame_uses_custom_pole():
    """frame='pole' lets the user pick an arbitrary great-circle pole."""
    fig = sphpl.make_figure(show_grid=False)
    trace = sphpl.add_great_circle(fig, frame='pole',
                                    pole_lon=45, pole_lat=30)
    assert trace.mode == 'lines'


# -- add_plane_overlay ---

def test_add_plane_overlay_default_galactic():
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_plane_overlay(fig)
    assert len(traces) == 1   # no parallels by default
    assert traces[0].name == 'Galactic plane'


def test_add_plane_overlay_with_parallels():
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_plane_overlay(fig, plane='ecliptic',
                                       parallels=[-23.4, 23.4])
    assert len(traces) == 3   # main + 2 parallels
    assert traces[0].name == 'Ecliptic plane'


def test_add_plane_overlay_color_override():
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_plane_overlay(fig, plane='galactic', color='red')
    assert traces[0].line.color == 'red'


# -- add_geodesic_circle ---

def test_add_geodesic_circle_line_only():
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_geodesic_circle(fig, 30, 45, 10)
    # Single trace, polyline (no fill).
    assert len(traces) == 1
    assert traces[0].mode == 'lines'
    assert traces[0].fill in (None, 'none')


def test_add_geodesic_circle_filled():
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_geodesic_circle(fig, 30, 45, 10, fill=True,
                                         fillcolor='rgba(0,200,255,0.3)')
    assert traces[0].fill == 'toself'
    assert traces[0].fillcolor == 'rgba(0,200,255,0.3)'


def test_add_geodesic_circle_wrap_edge_splits():
    """A circle centered near the wrap edge of a non-zero-center
    projection should split into two sub-traces."""
    fig = sphpl.make_figure(projection='AIT', center=180, show_grid=False)
    # Circle near lon=0 (which is the wrap edge for center=180).
    traces = sphpl.add_geodesic_circle(fig, 0, 0, 15, fill=True)
    assert len(traces) == 2   # wrap-split


# -- add_spherical_polygon ---

def test_add_spherical_polygon_triangle():
    """A spherical triangle from the pole down to two equator points."""
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_spherical_polygon(
        fig, lons=[0, 60, 30, 0], lats=[89, 0, 0, 89])
    assert len(traces) >= 1
    assert traces[0].fill == 'toself'


def test_add_spherical_polygon_unclosed_input_auto_closes():
    """Input that isn't already closed should be auto-closed."""
    fig = sphpl.make_figure(show_grid=False)
    # Not closed (first != last).
    traces = sphpl.add_spherical_polygon(fig, lons=[0, 20, 10],
                                           lats=[40, 40, 50])
    assert len(traces) >= 1


def test_add_spherical_polygon_outline_only():
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_spherical_polygon(
        fig, lons=[0, 20, 10, 0], lats=[40, 40, 50, 40], fill=False)
    assert traces[0].fill in (None, 'none')


def test_add_spherical_polygon_shape_mismatch_raises():
    fig = sphpl.make_figure(show_grid=False)
    with pytest.raises(ValueError, match="same shape"):
        sphpl.add_spherical_polygon(fig, lons=[0, 1, 2], lats=[0, 1])


def test_add_spherical_polygon_wrap_edge_splits():
    """A polygon straddling the wrap edge should produce two pieces."""
    fig = sphpl.make_figure(projection='AIT', center=180, show_grid=False)
    # Square box straddling lon=0/360 (wrap edge when center=180).
    traces = sphpl.add_spherical_polygon(
        fig, lons=[350, 10, 10, 350], lats=[-5, -5, 5, 5])
    assert len(traces) == 2


# -- _split_polyline_at_wrap helper ---

def test_split_polyline_at_wrap_inserts_nan():
    """A polyline straddling the wrap edge gets NaN inserted between
    the two sides."""
    from skyplothelper.plotly.core import _split_polyline_at_wrap
    # Polyline from 358° through 0/360° wrap to 2° (relative to center=180)
    lons = np.array([358, 359, 0, 1, 2])
    lats = np.array([0, 0, 0, 0, 0])
    out_lons, out_lats = _split_polyline_at_wrap(lons, lats, center=180)
    # The shifted lons are in [0, 360]. Specifically, the wrap happens
    # between consecutive values that differ by > 180°. After insertion,
    # there should be at least one NaN.
    assert np.any(np.isnan(out_lons))
    assert np.any(np.isnan(out_lats))


def test_split_polyline_at_wrap_no_wrap():
    """A polyline that doesn't cross the wrap edge passes through unchanged."""
    from skyplothelper.plotly.core import _split_polyline_at_wrap
    lons = np.array([10, 20, 30, 40])
    lats = np.array([0, 0, 0, 0])
    out_lons, out_lats = _split_polyline_at_wrap(lons, lats, center=0)
    assert len(out_lons) == len(lons)
    assert not np.any(np.isnan(out_lons))


# ============================================================================
# hover parameter — core overlays
# ============================================================================

def test_add_great_circle_hover_default_skips_info():
    """``hover=False`` (default) → ``hoverinfo='skip'``, no customdata."""
    fig = sphpl.make_figure(show_grid=False)
    trace = sphpl.add_great_circle(fig, frame='galactic')
    assert trace.hoverinfo == 'skip'
    # No template / customdata when hover is off.
    assert trace.hovertemplate is None
    assert trace.customdata is None


def test_add_great_circle_hover_true_auto_template_with_name():
    """``hover=True`` with ``name=...`` includes a bold name header and
    per-vertex RA/Dec from customdata."""
    fig = sphpl.make_figure(show_grid=False)
    trace = sphpl.add_great_circle(fig, frame='galactic',
                                    name='Galactic plane', hover=True)
    assert trace.hovertemplate is not None
    assert '<b>Galactic plane</b>' in trace.hovertemplate
    assert 'RA:' in trace.hovertemplate
    assert 'Dec:' in trace.hovertemplate
    assert '<extra></extra>' in trace.hovertemplate
    # Per-vertex customdata: shape (N, 2) with lon/lat.
    cd = np.asarray(trace.customdata)
    assert cd.shape[1] == 2
    assert cd.shape[0] == len(trace.x)


def test_add_great_circle_hover_true_no_name_omits_header():
    """Without ``name``, ``hover=True`` shows just RA/Dec."""
    fig = sphpl.make_figure(show_grid=False)
    trace = sphpl.add_great_circle(fig, frame='galactic', hover=True)
    assert '<b>' not in trace.hovertemplate
    assert 'RA:' in trace.hovertemplate


def test_add_great_circle_hover_custom_string_appends_extra():
    """A user-supplied string is used directly; ``<extra></extra>`` is
    auto-appended when not already present."""
    fig = sphpl.make_figure(show_grid=False)
    trace = sphpl.add_great_circle(fig, frame='galactic',
                                    hover='custom %{customdata[0]:.1f}')
    assert trace.hovertemplate.startswith('custom %{customdata[0]:.1f}')
    assert trace.hovertemplate.endswith('<extra></extra>')


def test_add_plane_overlay_hover_parallels_include_latitude_offset():
    """``add_plane_overlay(hover=True, parallels=[10, -10])`` names each
    parallel trace with its latitude offset so the hoverbox identifies
    which parallel is under the cursor."""
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_plane_overlay(fig, plane='galactic',
                                       parallels=[10, -10], hover=True)
    assert len(traces) == 3
    # Main trace gets the plane name; parallels get offset-annotated names.
    main, p1, p2 = traces
    assert 'Galactic plane' in main.hovertemplate
    assert 'b=+10.0°' in p1.hovertemplate
    assert 'b=-10.0°' in p2.hovertemplate


def test_add_geodesic_circle_hover_line_mode_attaches_customdata():
    """Outline mode under ``hover=True`` attaches per-vertex lon/lat
    customdata for the RA/Dec hover template."""
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_geodesic_circle(fig, 30, 0, 5,
                                         name='Disc', fill=False,
                                         hover=True)
    trace = traces[0]
    assert '<b>Disc</b>' in trace.hovertemplate
    assert 'RA:' in trace.hovertemplate
    cd = np.asarray(trace.customdata)
    assert cd.shape[1] == 2


def test_add_geodesic_circle_hover_fill_mode_shows_center_and_radius():
    """Fill mode under ``hover=True`` embeds center coords + radius in
    the trace ``text`` field, using ``hoveron='fills'`` so the hover
    box appears anywhere over the filled disc."""
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_geodesic_circle(fig, 45.0, -10.0, 3.5,
                                         name='Disc', fill=True,
                                         hover=True)
    trace = traces[0]
    assert trace.hoveron == 'fills'
    assert '<b>Disc</b>' in trace.text
    assert 'center:' in trace.text
    assert '45.000°' in trace.text
    assert '-10.000°' in trace.text
    assert 'radius:' in trace.text
    assert '3.500°' in trace.text
    # ``name=''`` suppresses the auto ``"trace N"`` label.
    assert trace.name == ''


def test_add_geodesic_circle_hover_default_skips():
    """``hover=False`` (default) preserves the original silent overlay."""
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_geodesic_circle(fig, 30, 0, 5, fill=True)
    assert traces[0].hoverinfo == 'skip'


def test_add_spherical_polygon_hover_fill_shows_name():
    """Filled polygon under ``hover=True`` shows the trace name in a
    fills-mode hoverbox."""
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_spherical_polygon(
        fig, lons=[10, 30, 30, 10, 10], lats=[0, 0, 20, 20, 0],
        name='Region A', fill=True, hover=True)
    trace = traces[0]
    assert trace.hoveron == 'fills'
    assert trace.text == '<b>Region A</b>'
    assert trace.name == ''


def test_add_spherical_polygon_hover_line_mode_attaches_customdata():
    """Outline mode under ``hover=True`` attaches per-vertex customdata."""
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_spherical_polygon(
        fig, lons=[10, 30, 30, 10, 10], lats=[0, 0, 20, 20, 0],
        name='Region A', fill=False, hover=True)
    trace = traces[0]
    assert '<b>Region A</b>' in trace.hovertemplate
    cd = np.asarray(trace.customdata)
    assert cd.shape[1] == 2


def test_add_spherical_polygon_hover_custom_string_on_fill():
    """A user-supplied string is used as the fill ``text``."""
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_spherical_polygon(
        fig, lons=[10, 30, 30, 10, 10], lats=[0, 0, 20, 20, 0],
        fill=True, hover='M42 outline<br>area: ~12°²')
    assert traces[0].text == 'M42 outline<br>area: ~12°²'


# ============================================================================
# Wrapper overlays — add_constellation_polygon / add_lonlat_box /
#                    add_frame_band / add_great_circle_band /
#                    add_healpix_sparse
# ============================================================================

def test_add_constellation_polygon_returns_traces():
    """A simple non-polar constellation renders as one filled polygon."""
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_constellation_polygon(fig, 'Ori',
                                               fillcolor='rgba(80,200,255,0.3)')
    assert len(traces) >= 1
    assert traces[0].fill == 'toself'


def test_add_constellation_polygon_case_insensitive():
    fig = sphpl.make_figure(show_grid=False)
    t_upper = sphpl.add_constellation_polygon(fig, 'ORI')
    t_lower = sphpl.add_constellation_polygon(fig, 'ori')
    # Same polygon → same number of traces and same first-piece vertex
    # count. (Numerical equality is checked elsewhere; this just
    # confirms the case-insensitive lookup.)
    assert len(t_upper) == len(t_lower)
    assert len(t_upper[0].x) == len(t_lower[0].x)


def test_add_constellation_polygon_serpens_has_two_bodies():
    """Serpens is the only IAU constellation drawn as two polygons
    (Caput + Cauda). The helper should emit at least one trace
    per body."""
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_constellation_polygon(fig, 'Ser')
    assert len(traces) >= 2


def test_add_constellation_polygon_unknown_raises():
    fig = sphpl.make_figure(show_grid=False)
    with pytest.raises(KeyError, match="unknown IAU"):
        sphpl.add_constellation_polygon(fig, 'XYZ')


def test_add_constellation_polygon_forwards_hover():
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_constellation_polygon(fig, 'Ori', name='Orion',
                                               hover=True, fill=True)
    assert traces[0].text == '<b>Orion</b>'


def test_add_lonlat_box_returns_trace():
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_lonlat_box(fig, lat_min=-10, lat_max=10,
                                    lon_min=0, lon_max=60,
                                    frame='galactic',
                                    fillcolor='rgba(255,100,200,0.3)')
    assert len(traces) >= 1


def test_add_lonlat_box_wraparound_lon():
    """``lon_max < lon_min`` is interpreted as a wraparound box."""
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_lonlat_box(fig, lat_min=-10, lat_max=10,
                                    lon_min=350, lon_max=10,
                                    frame='galactic')
    assert len(traces) >= 1


def test_add_lonlat_box_invalid_lat_order_raises():
    fig = sphpl.make_figure(show_grid=False)
    with pytest.raises(ValueError, match="lat_min must be less"):
        sphpl.add_lonlat_box(fig, lat_min=10, lat_max=-10,
                              lon_min=0, lon_max=60)


def test_add_lonlat_box_icrs_box_returns_trace():
    """A purely-ICRS box should pass straight through frame transform
    without scrambling."""
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_lonlat_box(fig, lat_min=-20, lat_max=20,
                                    lon_min=100, lon_max=160,
                                    frame='icrs', fill=False)
    assert len(traces) >= 1


def test_add_frame_band_galactic_returns_traces():
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_frame_band(fig, lat_min=-5, lat_max=5,
                                    frame='galactic',
                                    fillcolor='rgba(255,255,255,0.2)')
    assert len(traces) >= 1
    assert traces[0].fill == 'toself'


def test_add_frame_band_ecliptic_alias():
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_frame_band(fig, lat_min=-2, lat_max=2,
                                    frame='ecliptic')
    assert len(traces) >= 1


def test_add_frame_band_invalid_lat_order_raises():
    fig = sphpl.make_figure(show_grid=False)
    with pytest.raises(ValueError, match="lat_min must be less"):
        sphpl.add_frame_band(fig, lat_min=10, lat_max=-10,
                              frame='galactic')


def test_add_frame_band_outline_only():
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_frame_band(fig, lat_min=-5, lat_max=5,
                                    frame='galactic', fill=False)
    assert len(traces) >= 1
    # Outline-only traces don't carry the fill attribute as 'toself'.
    assert traces[0].fill is None or traces[0].fill == 'none'


def test_add_great_circle_band_returns_traces():
    """Galactic pole as ra=192.86, dec=27.13 — gives a band centered
    on the galactic plane."""
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_great_circle_band(fig, ra_pole=192.86,
                                           dec_pole=27.13,
                                           half_width=10,
                                           fillcolor='rgba(80,200,80,0.2)')
    assert len(traces) >= 1


def test_add_great_circle_band_fill_is_a_corridor_not_lenses():
    """The band fill must be a filled corridor between the two edge circles,
    not the two tiny self-intersection lenses the offset-lat-parallel edges
    collapsed to. Assert the total filled area scales with the band width and
    is far larger than the lens artifact (~400)."""
    def _fill_area(traces):
        tot = 0.0
        for t in traces:
            if getattr(t, "fill", None) != "toself":
                continue
            x = np.asarray(t.x, float)
            y = np.asarray(t.y, float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 3:
                xx, yy = x[m], y[m]
                tot += 0.5 * abs(np.dot(xx, np.roll(yy, 1))
                                 - np.dot(yy, np.roll(xx, 1)))
        return tot

    a_narrow = _fill_area(sphpl.add_great_circle_band(
        sphpl.make_figure("AIT", center=180, show_grid=False),
        ra_pole=120, dec_pole=40, half_width=8))
    a_wide = _fill_area(sphpl.add_great_circle_band(
        sphpl.make_figure("AIT", center=180, show_grid=False),
        ra_pole=120, dec_pole=40, half_width=20))
    assert a_narrow > 2000            # a real corridor, not ~400 of lenses
    assert a_wide > 1.5 * a_narrow    # wider band => more fill


def test_add_great_circle_band_invalid_half_width_raises():
    fig = sphpl.make_figure(show_grid=False)
    with pytest.raises(ValueError, match="half_width must be in"):
        sphpl.add_great_circle_band(fig, ra_pole=0, dec_pole=0,
                                      half_width=0)
    with pytest.raises(ValueError, match="half_width must be in"):
        sphpl.add_great_circle_band(fig, ra_pole=0, dec_pole=0,
                                      half_width=90)


def test_split_polygon_pole_touching_tile_no_streak():
    """A pole-touching HEALPix tile spans lon [0, 180] without an edge
    crossing the wrap meridian; the wrap shift maps its lon=180 vertex to
    -180, which would streak the fill across the canvas. The split must cut
    it at the resulting jump so no piece spans more than half the sky."""
    healpy = pytest.importorskip("healpy")
    from skyplothelper.plotly.core import _split_polygon_at_wrap
    nside = 4
    # ipix 1 is a north-polar-cap tile (lat ~66°-90°).
    v = healpy.boundaries(nside, 1, step=15)
    r = np.sqrt((v ** 2).sum(0))
    b_lat = np.degrees(np.arcsin(v[2] / r))
    b_lon = np.degrees(np.arctan2(v[1], v[0])) % 360.0
    b_lon = np.append(b_lon, b_lon[0])
    b_lat = np.append(b_lat, b_lat[0])
    pieces = _split_polygon_at_wrap(b_lon, b_lat, center=0.0)
    assert pieces
    for plons, _ in pieces:
        span = float(np.nanmax(plons) - np.nanmin(plons))
        assert span <= 180.0 + 1e-6   # no across-canvas streak


def test_add_healpix_dense_polar_tiles_no_streak():
    """End-to-end: a dense all-sky HEALPix map has no tile trace whose
    projected x spans nearly the whole canvas (the streak signature)."""
    healpy = pytest.importorskip("healpy")
    nside = 4
    npix = healpy.nside2npix(nside)
    theta, phi = healpy.pix2ang(nside, np.arange(npix))
    vals = np.cos(theta) * np.sin(2 * phi)
    fig = sphpl.make_figure(projection='AIT', center=0)
    traces = sphpl.add_healpix(fig, vals, nside=nside)
    # AIT half-width at the equator is ~162°; a clean tile at nside=4 spans
    # well under that. A streaking tile would span ~2× that across the frame.
    for t in traces:
        x = np.asarray(t.x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size:
            assert (x.max() - x.min()) < 200.0


def test_add_healpix_sparse_returns_trace_per_pixel():
    healpy = pytest.importorskip("healpy")  # noqa: F841
    nside = 4
    pix = np.array([0, 5, 10, 50, 100, 150])
    vals = np.arange(len(pix), dtype=float)
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix_sparse(fig, pix, vals, nside=nside,
                                         tile_resolution=1)
    # 6 input pixels, each potentially wrap-split → traces >= 6
    assert len(traces) >= len(pix)


def test_add_healpix_sparse_line_color_overrides_edge():
    """line_color sets a tile edge distinct from the fill; default ties the
    edge to the fill color."""
    healpy = pytest.importorskip("healpy")  # noqa: F841
    nside = 4
    pix = np.array([0, 5, 10])
    vals = np.arange(len(pix), dtype=float)
    fig = sphpl.make_figure(show_grid=False)
    tr = sphpl.add_healpix_sparse(fig, pix, vals, nside=nside,
                                  tile_resolution=1, line_width=1,
                                  line_color='white')
    assert all(t.line.color == 'white' for t in tr)
    # Default: edge equals fill (not 'white').
    fig2 = sphpl.make_figure(show_grid=False)
    tr2 = sphpl.add_healpix_sparse(fig2, pix, vals, nside=nside,
                                   tile_resolution=1, line_width=1)
    assert all(t.line.color == t.fillcolor for t in tr2)


def test_add_healpix_sparse_hover_default_includes_ipix():
    healpy = pytest.importorskip("healpy")  # noqa: F841
    nside = 4
    pix = np.array([0, 5, 10])
    vals = np.arange(3, dtype=float)
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix_sparse(fig, pix, vals, nside=nside,
                                         tile_resolution=1)
    seen = set()
    for t in traces:
        for line in t.text.split('<br>'):
            if line.startswith('ipix:'):
                seen.add(int(line.split(':', 1)[1].strip()))
                break
    # Every input pixel index appears in the rendered hover text.
    assert seen == set(int(p) for p in pix)


def test_add_healpix_sparse_skips_nan_tiles():
    healpy = pytest.importorskip("healpy")  # noqa: F841
    nside = 4
    pix = np.array([0, 5, 10, 50])
    vals = np.array([1.0, np.nan, 3.0, np.nan])
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix_sparse(fig, pix, vals, nside=nside,
                                         tile_resolution=1)
    # Only the 2 finite-valued tiles render (possibly wrap-split).
    seen = set()
    for t in traces:
        for line in t.text.split('<br>'):
            if line.startswith('ipix:'):
                seen.add(int(line.split(':', 1)[1].strip()))
                break
    assert seen == {0, 10}


def test_add_healpix_sparse_size_mismatch_raises():
    healpy = pytest.importorskip("healpy")  # noqa: F841
    fig = sphpl.make_figure(show_grid=False)
    with pytest.raises(ValueError, match="same shape"):
        sphpl.add_healpix_sparse(fig, np.array([0, 1, 2]),
                                    np.array([1.0, 2.0]), nside=4)


def test_add_healpix_sparse_traces_have_empty_name():
    """Same ``"trace N"`` suppression as add_healpix."""
    healpy = pytest.importorskip("healpy")  # noqa: F841
    nside = 4
    pix = np.array([0, 5])
    vals = np.array([1.0, 2.0])
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix_sparse(fig, pix, vals, nside=nside,
                                         tile_resolution=1)
    for t in traces:
        assert t.name == ''


# ============================================================================
# tile_resolution='auto' heuristic
# ============================================================================

def test_resolve_tile_resolution_auto_scales_inversely_with_nside():
    """``'auto'`` targets ~1° spacing along each tile edge:
    ``max(2, ceil(58 / nside))``. Confirms the heuristic gives a
    well-densified boundary at low nside (where polar-cap tiles are
    large and their projected boundaries are visibly curved) and a
    light floor at high nside."""
    from skyplothelper.plotly.core import _resolve_tile_resolution
    assert _resolve_tile_resolution('auto', nside=1) == 58
    assert _resolve_tile_resolution('auto', nside=4) == 15
    assert _resolve_tile_resolution('auto', nside=16) == 4
    assert _resolve_tile_resolution('auto', nside=64) == 2
    assert _resolve_tile_resolution('auto', nside=256) == 2  # floor
    # Case-insensitive 'auto'.
    assert _resolve_tile_resolution('AUTO', nside=4) == 15


def test_resolve_tile_resolution_explicit_int_passes_through():
    """Integer overrides bypass the heuristic, with a min-1 floor."""
    from skyplothelper.plotly.core import _resolve_tile_resolution
    assert _resolve_tile_resolution(8, nside=4) == 8
    assert _resolve_tile_resolution(1, nside=4) == 1
    assert _resolve_tile_resolution(0, nside=4) == 1  # floor at 1
    assert _resolve_tile_resolution(-3, nside=4) == 1


def test_resolve_tile_resolution_invalid_string_raises():
    from skyplothelper.plotly.core import _resolve_tile_resolution
    with pytest.raises(ValueError, match="'auto' or an integer"):
        _resolve_tile_resolution('weird', nside=4)


def test_add_healpix_default_tile_resolution_is_auto():
    """Default ``tile_resolution='auto'`` should give the heuristic
    value at nside=4 (15 samples/edge → 60 boundary points per tile,
    plus a closing vertex)."""
    healpy = pytest.importorskip("healpy")  # noqa: F841
    from collections import Counter
    nside = 4
    vals = np.zeros(healpy.nside2npix(nside))
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix(fig, vals, nside=nside)
    # Non-wrap-split tiles have exactly ``4 * step + 1`` vertices (the
    # closing vertex repeats the start). Wrap-split tiles densify
    # along the wrap edge and have different counts. The most common
    # vertex count is therefore the well-conditioned tile count.
    counts = Counter(len(t.x) for t in traces)
    most_common_len, _ = counts.most_common(1)[0]
    assert most_common_len == 4 * 15 + 1


# ---------------------------------------------------------------------------
# HEALPix colorbars
# ---------------------------------------------------------------------------
#
# Tiles are painted with a flat ``fillcolor`` sampled from the colorscale, and
# a flat fill carries no colorscale for plotly to build a colorbar from. The
# bar therefore rides on an invisible companion marker trace whose cmin/cmax
# are the same vmin/vmax the tiles were colored with, so it cannot disagree
# with what it labels.

@pytest.fixture
def _hp_map():
    healpy = pytest.importorskip("healpy")
    nside = 4
    return nside, np.arange(healpy.nside2npix(nside), dtype=float)


@pytest.mark.parametrize('sparse', [False, True])
def test_add_healpix_colorbar_is_opt_in_and_appends_one_trace(_hp_map, sparse):
    nside, vals = _hp_map

    def _call(fig, **kw):
        if sparse:
            pix = np.arange(6)
            return sphpl.add_healpix_sparse(fig, pix, vals[pix], nside=nside,
                                             tile_resolution=2, **kw)
        return sphpl.add_healpix(fig, vals, nside=nside, tile_resolution=2,
                                  **kw)

    plain = _call(sphpl.make_figure(show_grid=False))
    with_bar = _call(sphpl.make_figure(show_grid=False), add_colorbar=True)
    assert len(with_bar) == len(plain) + 1

    cbar = with_bar[-1]
    assert cbar.marker.showscale is True
    assert cbar.showlegend is False and cbar.hoverinfo == 'skip'
    # Contributes no data: an all-None point can't disturb the axis ranges.
    assert tuple(cbar.x) == (None,) and tuple(cbar.y) == (None,)


@pytest.mark.parametrize('sparse', [False, True])
def test_add_healpix_colorbar_range_matches_the_tiles(_hp_map, sparse):
    """cmin/cmax track the vmin/vmax used to color the tiles, explicit or
    inferred — otherwise the bar would mislabel the map."""
    nside, vals = _hp_map
    pix = np.arange(6)

    def _call(**kw):
        fig = sphpl.make_figure(show_grid=False)
        if sparse:
            return sphpl.add_healpix_sparse(fig, pix, vals[pix], nside=nside,
                                             tile_resolution=2,
                                             add_colorbar=True, **kw)
        return sphpl.add_healpix(fig, vals, nside=nside, tile_resolution=2,
                                  add_colorbar=True, **kw)

    data = vals[pix] if sparse else vals
    inferred = _call()[-1].marker
    assert (inferred.cmin, inferred.cmax) == (float(data.min()),
                                               float(data.max()))
    explicit = _call(vmin=10.0, vmax=20.0)[-1].marker
    assert (explicit.cmin, explicit.cmax) == (10.0, 20.0)


def test_add_healpix_colorbar_title_and_kwargs_reach_plotly(_hp_map):
    nside, vals = _hp_map
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix(
        fig, vals, nside=nside, tile_resolution=2, colorscale='Plasma',
        add_colorbar=True, cbar_title='counts',
        colorbar_kwargs=dict(orientation='h', x=0.5, y=-0.05, len=0.6,
                             thickness=14))
    marker = traces[-1].marker
    assert marker.colorbar.title.text == 'counts'
    assert marker.colorbar.orientation == 'h'
    assert marker.colorbar.thickness == 14
    assert marker.colorscale[0][1] != marker.colorscale[-1][1]  # Plasma, not
    fig.to_dict()                                               # default


def test_add_healpix_colorbar_kwargs_override_title(_hp_map):
    nside, vals = _hp_map
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix(fig, vals, nside=nside, tile_resolution=2,
                                add_colorbar=True, cbar_title='ignored',
                                colorbar_kwargs=dict(title='wins'))
    assert traces[-1].marker.colorbar.title.text == 'wins'


# ============================================================================
# add_sky_vectors
# ============================================================================

def _mock_arrows():
    ra = np.array([100.0, 180.0, 270.0, 30.0])
    dec = np.array([20.0, 0.0, -20.0, 40.0])
    dra = np.array([50.0, -30.0, 100.0, 20.0])    # in input units
    ddec = np.array([20.0, -10.0, -50.0, 80.0])
    return ra, dec, dra, ddec


def test_add_sky_vectors_returns_shaft_and_head():
    fig = sphpl.make_figure(show_grid=False)
    ra, dec, dra, ddec = _mock_arrows()
    shaft, head = sphpl.add_sky_vectors(fig, ra, dec, dra, ddec,
                                          scale=0.05, units='mas')
    assert shaft.mode == 'lines'
    assert head.mode == 'markers'
    assert head.marker.symbol == 'arrow-up'
    # One arrowhead marker per arrow.
    assert len(head.x) == 4
    # Shaft is a NaN-separated lines trace. The RA=180 mock arrow sits on
    # the center=0 wrap seam, so its shaft wrap-splits into legs at the
    # seam edge (>=1 NaN break per arrow) instead of streaking across the
    # frame — so the vertex count is no longer a fixed 3*n.
    sx = np.asarray(shaft.x, dtype=float)
    assert int(np.isnan(sx).sum()) >= 4


def test_add_sky_vectors_wrap_splits_seam_arrow():
    """An arrow straddling the wrap seam (RA=180 at center=0) is
    wrap-split into legs at the seam edge, not streaked across the frame."""
    fig = sphpl.make_figure(show_grid=False)  # center=0 -> seam at lon=180
    ra, dec, dra, ddec = _mock_arrows()        # RA=180 arrow sits on the seam
    shaft, _ = sphpl.add_sky_vectors(fig, ra, dec, dra, ddec,
                                       scale=0.05, units='mas')
    sx = np.asarray(shaft.x, dtype=float)
    assert np.isnan(sx).any()
    d = np.abs(np.diff(sx))
    d = d[np.isfinite(d)]
    full_width = float(np.nanmax(sx) - np.nanmin(sx))
    assert d.size and float(np.max(d)) < 0.5 * full_width


def test_add_sky_vectors_per_arrow_rotation():
    """Arrowhead ``marker.angle`` is one rotation value per arrow."""
    fig = sphpl.make_figure(show_grid=False)
    ra, dec, dra, ddec = _mock_arrows()
    _, head = sphpl.add_sky_vectors(fig, ra, dec, dra, ddec,
                                       scale=0.05, units='mas')
    angles = np.asarray(head.marker.angle)
    assert angles.shape == (4,)
    # Angles span a meaningful range (not all zero).
    assert np.ptp(angles) > 30.0


def test_add_sky_vectors_auto_scale_targets_median_arrow_length():
    """``scale='auto'`` should size the median arrow to ~auto_target_deg."""
    fig = sphpl.make_figure(show_grid=False)
    # Arrows placed away from the wrap seam (lon=180 at center=0) so each
    # shaft stays a simple start->end->NaN triple (no seam wrap-split),
    # keeping the per-arrow length easy to read back via reshape.
    ra = np.array([60.0, 80.0, 100.0, 120.0])
    dec = np.array([20.0, 0.0, -20.0, 40.0])
    dra = np.array([50.0, -30.0, 100.0, 20.0])
    ddec = np.array([20.0, -10.0, -50.0, 80.0])
    shaft, _ = sphpl.add_sky_vectors(fig, ra, dec, dra, ddec,
                                        scale='auto', units='mas',
                                        auto_target_deg=2.0)
    # Compute the 4 arrow projected lengths from the shaft points.
    xs = np.asarray(shaft.x).reshape(-1, 3)[:, :2]
    ys = np.asarray(shaft.y).reshape(-1, 3)[:, :2]
    lengths = np.hypot(xs[:, 1] - xs[:, 0], ys[:, 1] - ys[:, 0])
    # Median should be close to 2° (projected units == degrees for
    # AIT at the figure default; small distortion factor tolerated).
    assert 1.0 < np.median(lengths) < 4.0


def test_add_sky_vectors_units_validation():
    fig = sphpl.make_figure(show_grid=False)
    ra, dec, dra, ddec = _mock_arrows()
    with pytest.raises(ValueError, match="units must be one of"):
        sphpl.add_sky_vectors(fig, ra, dec, dra, ddec, units='furlongs')


def test_add_sky_vectors_invalid_scale_raises():
    fig = sphpl.make_figure(show_grid=False)
    ra, dec, dra, ddec = _mock_arrows()
    with pytest.raises(ValueError, match="scale must be 'auto'"):
        sphpl.add_sky_vectors(fig, ra, dec, dra, ddec, scale='gigantic')


def test_add_sky_vectors_invalid_pivot_raises():
    fig = sphpl.make_figure(show_grid=False)
    ra, dec, dra, ddec = _mock_arrows()
    with pytest.raises(ValueError, match="pivot must be"):
        sphpl.add_sky_vectors(fig, ra, dec, dra, ddec, pivot='nose')


def test_add_sky_vectors_color_by_magnitude_sets_scale_attrs():
    fig = sphpl.make_figure(show_grid=False)
    ra, dec, dra, ddec = _mock_arrows()
    _, head = sphpl.add_sky_vectors(fig, ra, dec, dra, ddec,
                                       scale=0.05, units='mas',
                                       color_by_magnitude=True,
                                       cmap='Plasma',
                                       add_colorbar=True,
                                       cbar_title='PM (mas/yr)')
    marker = head.marker
    # Per-arrow color array.
    mags = np.hypot(dra, ddec)
    assert np.allclose(np.asarray(marker.color), mags)
    # Colorscale, cmin/cmax, colorbar attached.
    assert marker.cmin == float(mags.min())
    assert marker.cmax == float(mags.max())
    assert marker.showscale is True
    assert marker.colorbar.title.text == 'PM (mas/yr)'


def test_add_sky_vectors_cos_dec_off_applies_correction():
    """With ``cos_dec=False`` and a non-equator anchor, the shaft
    end's lon offset should be larger than with ``cos_dec=True``
    (the function divides by cos(dec) to undo the implicit scaling)."""
    # Anchor placed away from the wrap edge so the projection
    # symmetric around the wrap doesn't confuse the length compare.
    fig = sphpl.make_figure(center=0, show_grid=False)
    lon = np.array([60.0])
    lat = np.array([60.0])
    dlon = np.array([100.0])
    dlat = np.array([0.0])
    s_on, _ = sphpl.add_sky_vectors(fig, lon, lat, dlon, dlat,
                                       scale=0.05, units='mas',
                                       cos_dec=True, pivot='tail')
    s_off, _ = sphpl.add_sky_vectors(fig, lon, lat, dlon, dlat,
                                        scale=0.05, units='mas',
                                        cos_dec=False, pivot='tail')
    len_on = np.hypot(s_on.x[1] - s_on.x[0], s_on.y[1] - s_on.y[0])
    len_off = np.hypot(s_off.x[1] - s_off.x[0], s_off.y[1] - s_off.y[0])
    # cos(60°) = 0.5 → cos_dec=False should give ~2× the lon-step.
    # Projection distortion makes the projected length ratio inexact;
    # require at least a 50% increase as a sanity check.
    assert len_off > len_on * 1.4


def test_add_sky_vectors_hover_default_off():
    fig = sphpl.make_figure(show_grid=False)
    ra, dec, dra, ddec = _mock_arrows()
    _, head = sphpl.add_sky_vectors(fig, ra, dec, dra, ddec,
                                       scale=0.05, units='mas')
    assert head.hoverinfo == 'skip'


def test_add_sky_vectors_hover_true_auto_template():
    fig = sphpl.make_figure(show_grid=False)
    ra, dec, dra, ddec = _mock_arrows()
    _, head = sphpl.add_sky_vectors(fig, ra, dec, dra, ddec,
                                       scale=0.05, units='mas',
                                       name='Gaia DR3 PM', hover=True)
    assert head.hovertemplate is not None
    assert '<b>Gaia DR3 PM</b>' in head.hovertemplate
    assert '|v|:' in head.hovertemplate
    assert 'mas' in head.hovertemplate
    assert 'PA:' in head.hovertemplate
    cd = np.asarray(head.customdata)
    assert cd.shape == (4, 4)


def test_add_sky_vectors_hover_custom_string_appends_extra():
    fig = sphpl.make_figure(show_grid=False)
    ra, dec, dra, ddec = _mock_arrows()
    _, head = sphpl.add_sky_vectors(fig, ra, dec, dra, ddec,
                                       scale=0.05, units='mas',
                                       hover='src %{customdata[0]:.1f}')
    assert head.hovertemplate.startswith('src %{customdata[0]:.1f}')
    assert head.hovertemplate.endswith('<extra></extra>')


def test_add_sky_vectors_shaft_color_auto_single_trace():
    """Default ``shaft_color='auto'`` emits a single shaft trace."""
    fig = sphpl.make_figure(show_grid=False)
    ra, dec, dra, ddec = _mock_arrows()
    shaft, _ = sphpl.add_sky_vectors(fig, ra, dec, dra, ddec,
                                        scale=0.05, units='mas',
                                        color_by_magnitude=True)
    # Single trace (not list).
    assert not isinstance(shaft, list)
    assert shaft.mode == 'lines'


def test_add_sky_vectors_shaft_color_match_emits_per_arrow():
    """``shaft_color='match'`` with a color array emits N shaft traces,
    each colored from the same colorscale as the heads."""
    fig = sphpl.make_figure(show_grid=False)
    ra, dec, dra, ddec = _mock_arrows()
    shaft, head = sphpl.add_sky_vectors(fig, ra, dec, dra, ddec,
                                          scale=0.05, units='mas',
                                          color_by_magnitude=True,
                                          cmap='Viridis',
                                          shaft_color='match')
    assert isinstance(shaft, list)
    assert len(shaft) == 4
    # Each per-arrow shaft has a distinct sampled color from Viridis.
    colors = [s.line.color for s in shaft]
    assert len(set(colors)) == 4
    # The smallest magnitude maps to the low end of Viridis (purple),
    # the largest to the high end (yellow); both colors start with
    # ``rgb(``.
    for c in colors:
        assert c.startswith('rgb(')


def test_add_sky_vectors_shaft_color_match_falls_back_without_color_array():
    """``shaft_color='match'`` with no color array reverts to 'auto'
    behavior (single shaft trace) so the contract stays predictable."""
    fig = sphpl.make_figure(show_grid=False)
    ra, dec, dra, ddec = _mock_arrows()
    shaft, _ = sphpl.add_sky_vectors(fig, ra, dec, dra, ddec,
                                        scale=0.05, units='mas',
                                        shaft_color='match')
    assert not isinstance(shaft, list)
    assert shaft.mode == 'lines'


def test_add_sky_vectors_shaft_color_explicit_string():
    """An explicit color string overrides ``color`` for the shaft only."""
    fig = sphpl.make_figure(show_grid=False)
    ra, dec, dra, ddec = _mock_arrows()
    shaft, head = sphpl.add_sky_vectors(fig, ra, dec, dra, ddec,
                                          scale=0.05, units='mas',
                                          color='blue',
                                          shaft_color='red')
    assert shaft.line.color == 'red'
    # Head still uses ``color`` (since not color-by-magnitude).
    assert head.marker.color == 'blue'


def test_add_sky_vectors_shaft_color_invalid_type_raises():
    fig = sphpl.make_figure(show_grid=False)
    ra, dec, dra, ddec = _mock_arrows()
    with pytest.raises(ValueError, match="shaft_color must be"):
        sphpl.add_sky_vectors(fig, ra, dec, dra, ddec,
                                scale=0.05, units='mas', shaft_color=42)


def test_add_sky_vectors_pivot_middle_centers_arrow_on_anchor():
    """With ``pivot='middle'`` the data point sits at the midpoint
    of the shaft, not at start or end."""
    fig = sphpl.make_figure(center=0, show_grid=False)
    # Anchor away from the wrap edge; small arrow so projection
    # distortion across the shaft is negligible.
    lon = np.array([60.0])
    lat = np.array([0.0])
    dlon = np.array([100.0])
    dlat = np.array([0.0])
    shaft, _ = sphpl.add_sky_vectors(fig, lon, lat, dlon, dlat,
                                        scale=0.05, units='mas',
                                        pivot='middle')
    # The anchor's projected x is the midpoint of (x0, x1).
    x0, x1 = float(shaft.x[0]), float(shaft.x[1])
    anchor_x, _ = _project(lon, lat, projection='AIT', center=0.0)
    assert np.isclose((x0 + x1) / 2, float(anchor_x[0]), atol=1e-6)


# ============================================================================
# add_coord_labels
# ============================================================================

def _split_lon_lat_anns(anns):
    """Partition annotations into lon/lat groups by their yanchor.

    Lon labels use ``yanchor='top'`` (sit below the equator or canvas
    bottom); lat labels use ``yanchor='middle'`` (centered on their
    parallel). Works under both ``placement='frame'`` and ``'canvas'``."""
    lon_anns = [a for a in anns if a.yanchor == 'top']
    lat_anns = [a for a in anns if a.yanchor == 'middle']
    return lon_anns, lat_anns


def test_add_coord_labels_adds_lon_and_lat_annotations():
    fig = sphpl.make_figure(projection='AIT', center=180, theme='dark')
    anns = sphpl.add_coord_labels(fig)
    # Default 30°/15° spacing on a center=180 AIT view drops the
    # wrap-edge lons (0° and 360°) → 11 lon labels (30°…330°) and
    # 11 lat labels (-75°…+75°).
    lon_anns, lat_anns = _split_lon_lat_anns(anns)
    assert len(lon_anns) == 11
    assert len(lat_anns) == 11
    assert lon_anns[0].text == '30°'
    assert lat_anns[5].text == '+0°'


def test_add_coord_labels_lat_exterior_flips_offset():
    """lat_exterior toggles latitude labels between just-inside (default)
    and just-outside the frame edge — the offset direction flips."""
    fig = sphpl.make_figure(projection='AIT', center=180, theme='dark')
    _, lat_in = _split_lon_lat_anns(
        sphpl.add_coord_labels(fig, lat_spacing=30, lat_exterior=False))
    fig2 = sphpl.make_figure(projection='AIT', center=180, theme='dark')
    _, lat_out = _split_lon_lat_anns(
        sphpl.add_coord_labels(fig2, lat_spacing=30, lat_exterior=True))

    eq_in = [a for a in lat_in if a.text == '+0°'][0]
    eq_out = [a for a in lat_out if a.text == '+0°'][0]
    # Same anchor point on the frame edge, opposite push direction.
    assert eq_in.x == eq_out.x
    assert eq_in.xshift == -eq_out.xshift
    # Interior pushes toward canvas center, exterior away from it.
    assert (eq_in.x < 0) and eq_in.xshift > 0 and eq_in.xanchor == 'left'
    assert eq_out.xshift < 0 and eq_out.xanchor == 'right'


def test_add_coord_labels_skips_lon_labels_at_wrap_edges():
    """Labels within 0.4 × ``lon_spacing`` of the wrap edge are
    dropped so they don't collide with the projection silhouette."""
    fig = sphpl.make_figure(projection='AIT', center=180,
                              theme='dark')
    anns = sphpl.add_coord_labels(fig, lon_spacing=30)
    lon_anns, _ = _split_lon_lat_anns(anns)
    lon_texts = [a.text for a in lon_anns]
    assert '0°' not in lon_texts
    assert '360°' not in lon_texts


def test_add_coord_labels_hours_format():
    fig = sphpl.make_figure(projection='AIT', center=180, theme='dark')
    anns = sphpl.add_coord_labels(fig, lon_format='hours',
                                     lon_spacing=60)
    lon_anns, _ = _split_lon_lat_anns(anns)
    lon_texts = [a.text for a in lon_anns]
    # 60° step at center=180 → 4h, 8h, 12h, 16h, 20h.
    assert lon_texts == ['4h', '8h', '12h', '16h', '20h']


def test_add_coord_labels_auto_units_equatorial_frame():
    """With an equatorial frame hint on make_figure, lon_format='auto'
    (default) picks hours, mirroring the matplotlib frame default."""
    fig = sphpl.make_figure(projection='AIT', center=180, frame='icrs')
    anns = sphpl.add_coord_labels(fig, lon_spacing=60)
    lon_anns, _ = _split_lon_lat_anns(anns)
    assert all(a.text.endswith('h') for a in lon_anns)


def test_add_coord_labels_auto_units_galactic_and_none_are_deg():
    """Galactic (or no) frame hint → degrees under lon_format='auto'."""
    for frame in ('galactic', None):
        fig = sphpl.make_figure(projection='AIT', center=0, frame=frame)
        anns = sphpl.add_coord_labels(fig, lon_spacing=60)
        lon_anns, _ = _split_lon_lat_anns(anns)
        assert all(a.text.endswith('°') for a in lon_anns)


def test_make_figure_direction_alias_normalized_in_meta():
    fig = sphpl.make_figure(direction='geo')
    assert fig.layout.meta['sph_direction'] == 'geographic'
    fig2 = sphpl.make_figure(direction='astro')
    assert fig2.layout.meta['sph_direction'] == 'sky'


def test_make_figure_lon_units_overrides_frame_hint():
    """An explicit make_figure(lon_units=...) wins over the frame hint for
    add_coord_labels' 'auto' longitude format."""
    # equatorial frame would auto-pick hours; lon_units='degrees' forces deg
    fig = sphpl.make_figure(center=180, frame='icrs', lon_units='degrees')
    assert fig.layout.meta['sph_lon_units'] == 'degrees'
    anns = sphpl.add_coord_labels(fig, lon_spacing=60)
    lon_anns, _ = _split_lon_lat_anns(anns)
    assert all(a.text.endswith('°') for a in lon_anns)
    # no frame, lon_units='hours' forces hours
    fig2 = sphpl.make_figure(center=180, lon_units='hours')
    anns2 = sphpl.add_coord_labels(fig2, lon_spacing=60)
    lon2, _ = _split_lon_lat_anns(anns2)
    assert all(a.text.endswith('h') for a in lon2)


def test_add_coord_labels_callable_format():
    fig = sphpl.make_figure(projection='AIT', center=0, theme='dark')
    anns = sphpl.add_coord_labels(
        fig, lon_format=lambda v: f"RA={v:.0f}",
        lat_format=lambda v: f"Dec={v:+.0f}",
        lon_spacing=60, lat_spacing=30)
    lon_anns, lat_anns = _split_lon_lat_anns(anns)
    assert any('RA=' in a.text for a in lon_anns)
    assert any('Dec=+' in a.text for a in lat_anns)


def test_add_coord_labels_independent_toggles():
    fig = sphpl.make_figure(projection='AIT', center=180,
                              theme='dark')
    anns = sphpl.add_coord_labels(fig, show_lat=False)
    lon_anns, lat_anns = _split_lon_lat_anns(anns)
    assert len(lon_anns) > 0
    assert len(lat_anns) == 0
    fig2 = sphpl.make_figure(projection='AIT', center=180,
                               theme='dark')
    anns2 = sphpl.add_coord_labels(fig2, show_lon=False)
    lon_anns2, lat_anns2 = _split_lon_lat_anns(anns2)
    assert len(lon_anns2) == 0
    assert len(lat_anns2) > 0


def test_add_coord_labels_color_default_matches_theme():
    """Without explicit ``color``, dark theme yields a light color
    and light theme yields a dark color."""
    fig_dark = sphpl.make_figure(theme='dark')
    anns_dark = sphpl.add_coord_labels(fig_dark)
    fig_light = sphpl.make_figure(theme='light')
    anns_light = sphpl.add_coord_labels(fig_light)
    # The exact colors are implementation-defined hex strings; assert
    # they are non-empty and distinct between the two themes.
    assert anns_dark[0].font.color
    assert anns_light[0].font.color
    assert anns_dark[0].font.color != anns_light[0].font.color


def test_add_coord_labels_color_override():
    fig = sphpl.make_figure(theme='dark')
    anns = sphpl.add_coord_labels(fig, color='lime')
    assert anns[0].font.color == 'lime'


def test_add_coord_labels_uses_figure_metadata():
    """Labels project through the projection stamped on the figure
    by make_figure, not a default. Tested via ``placement='canvas'``
    so the central-meridian convention applies (frame mode pushes
    lat labels to the wrap edge instead)."""
    fig = sphpl.make_figure(projection='MOL', center=120,
                              theme='dark')
    anns = sphpl.add_coord_labels(fig, lon_spacing=30, lat_spacing=30,
                                     placement='canvas')
    lon_anns, _ = _split_lon_lat_anns(anns)
    # MOL central meridian sits at lon=120; the central tick's x
    # should be at projected x=0 for lon=120.
    central = next(a for a in lon_anns if a.text == '120°')
    assert abs(float(central.x)) < 1e-6


def test_add_coord_labels_frame_placement_lon_at_equator():
    """Frame mode (default) places lon labels at the projected
    equator (lat=0) — i.e. data-y coordinates close to zero, not
    paper coords."""
    fig = sphpl.make_figure(projection='AIT', center=180,
                              theme='dark')
    anns = sphpl.add_coord_labels(fig)
    lon_anns, _ = _split_lon_lat_anns(anns)
    for a in lon_anns:
        assert a.yref == 'y'  # data, not 'paper'
        assert abs(float(a.y)) < 1e-9  # equator: y=0 in projection


def test_add_coord_labels_frame_placement_lat_at_wrap_edge():
    """Frame mode places lat labels on the projection wrap-edge
    meridian. On AIT (and other curved-silhouette projections) the
    wrap meridian itself curves, so the label x positions narrow
    toward the poles. Verify each label's (x, y) matches the
    projection of (edge_lon, lat)."""
    fig = sphpl.make_figure(projection='AIT', center=180,
                              theme='dark', direction='sky')
    anns = sphpl.add_coord_labels(fig)
    _, lat_anns = _split_lon_lat_anns(anns)
    # Reconstruct the edge meridian the helper picked (the one whose
    # equator-x is most negative for sky-direction AIT center=180).
    edge_lon_left = 180 - 180 + 0.05
    edge_lon_right = 180 + 180 - 0.05
    x_eq_left, _ = _project(np.array([edge_lon_left]), np.array([0.0]),
                              projection='AIT', center=180,
                              direction='sky')
    x_eq_right, _ = _project(np.array([edge_lon_right]), np.array([0.0]),
                               projection='AIT', center=180,
                               direction='sky')
    edge_lon = (edge_lon_left
                if float(x_eq_left[0]) < float(x_eq_right[0])
                else edge_lon_right)
    # Each lat label's (x, y) should match project(edge_lon, lat).
    lat_vals = np.arange(-90 + 15, 90, 15.0)
    xs_expected, ys_expected = _project(
        np.full_like(lat_vals, edge_lon), lat_vals,
        projection='AIT', center=180, direction='sky')
    xs_actual = np.array([float(a.x) for a in lat_anns])
    ys_actual = np.array([float(a.y) for a in lat_anns])
    assert np.allclose(xs_actual, xs_expected, atol=1e-6)
    assert np.allclose(ys_actual, ys_expected, atol=1e-6)
    # Sanity: wrap meridian on AIT is a CURVE, not a line —
    # x positions narrow toward the poles.
    assert abs(xs_actual[0]) < abs(xs_actual[len(xs_actual) // 2])


def test_add_coord_labels_placement_canvas_uses_paper_refs():
    """Explicit ``placement='canvas'`` reverts to paper-edge anchoring."""
    fig = sphpl.make_figure(projection='AIT', center=180,
                              theme='dark')
    anns = sphpl.add_coord_labels(fig, placement='canvas')
    lon_anns, lat_anns = _split_lon_lat_anns(anns)
    for a in lon_anns:
        assert a.yref == 'paper'
    for a in lat_anns:
        assert a.xref == 'paper'


# ============================================================================
# add_frame_edge
# ============================================================================

def test_add_frame_edge_closes_silhouette():
    fig = sphpl.make_figure(projection='AIT', center=180, theme='dark')
    trace = sphpl.add_frame_edge(fig)
    assert trace.mode == 'lines'
    # ``2*resolution + 1`` vertices (right side + left side + closing point).
    assert len(trace.x) == 2 * 361 + 1
    # First vertex repeats at the end (closure).
    assert float(trace.x[0]) == float(trace.x[-1])
    assert float(trace.y[0]) == float(trace.y[-1])


def test_add_frame_edge_extent_matches_projection():
    """AIT silhouette has |x| extending farther than |y| (a 2:1
    horizontal ellipse). MOL has similar proportion."""
    fig = sphpl.make_figure(projection='AIT', center=180, theme='dark')
    trace = sphpl.add_frame_edge(fig)
    xs = np.asarray(trace.x)
    ys = np.asarray(trace.y)
    x_extent = np.nanmax(xs) - np.nanmin(xs)
    y_extent = np.nanmax(ys) - np.nanmin(ys)
    assert x_extent > y_extent  # 2:1 aspect


def test_add_frame_edge_color_theme_default():
    fig_dark = sphpl.make_figure(theme='dark')
    fig_light = sphpl.make_figure(theme='light')
    t_dark = sphpl.add_frame_edge(fig_dark)
    t_light = sphpl.add_frame_edge(fig_light)
    assert t_dark.line.color != t_light.line.color


def test_add_frame_edge_color_override():
    fig = sphpl.make_figure(theme='dark')
    trace = sphpl.add_frame_edge(fig, color='magenta', width=2.0)
    assert trace.line.color == 'magenta'
    assert trace.line.width == 2.0


def test_add_frame_edge_uses_figure_metadata():
    """The frame edge picks up the projection / center stamped by
    make_figure (no need to pass them explicitly)."""
    fig = sphpl.make_figure(projection='MOL', center=90, theme='dark')
    trace = sphpl.add_frame_edge(fig)
    # MOL at center=90 should still produce a closed curve; lat=0
    # (equator) at lon=90 (center) projects to (0, 0).
    xs = np.asarray(trace.x)
    # Center lon projects to x=0; the edge contains no vertex AT
    # x=0 (the edges are at the wrap meridian), but the extent
    # should be symmetric about it.
    assert abs(float(np.nanmin(xs)) + float(np.nanmax(xs))) < 1e-3


def test_add_frame_edge_hoverinfo_skip():
    fig = sphpl.make_figure(theme='dark')
    trace = sphpl.add_frame_edge(fig)
    assert trace.hoverinfo == 'skip'


@pytest.mark.parametrize('projection,center,lat_center', [
    ('SIN', 266.4, -29.0),
    ('SIN', 0.0, 0.0),
    ('ARC', 266.4, -29.0),
    ('ZEA', 45.0, 60.0),
    ('STG', 0.0, 0.0),
])
def test_add_frame_edge_circular_globe_traces_limb(
        projection, center, lat_center):
    """On a zenithal globe the silhouette is the limb (great circle 90°
    from center), NOT the wrap meridian — which lies on the far,
    invisible hemisphere and projects to all-NaN. The limb must be a
    fully finite, closed, origin-symmetric loop."""
    fig = sphpl.make_figure(projection=projection, center=center,
                            lat_center=lat_center, theme='dark',
                            width=560, height=560)
    trace = sphpl.add_frame_edge(fig)
    xs = np.asarray(trace.x, float)
    ys = np.asarray(trace.y, float)
    # Every vertex projects (the wrap-meridian trace was 100% NaN here).
    assert np.isfinite(xs).all() and np.isfinite(ys).all()
    # Closed loop.
    assert float(xs[0]) == float(xs[-1])
    assert float(ys[0]) == float(ys[-1])
    # The limb of a zenithal disk is symmetric about the tangent point
    # (canvas origin) in both axes.
    assert abs(xs.min() + xs.max()) < 1e-6
    assert abs(ys.min() + ys.max()) < 1e-6


# ============================================================================
# add_reticle
# ============================================================================

def test_add_reticle_plus_returns_eight_shapes():
    """Plus has 4 arms; with stroke enabled (default) each arm renders
    twice (stroke layer + body layer) → 8 shapes."""
    fig = sphpl.make_figure(show_grid=False)
    shapes, lbl = sphpl.add_reticle(fig, lon=200, lat=10)
    assert len(shapes) == 8
    assert lbl is None  # no label by default


def test_add_reticle_stroke_disabled_halves_shape_count():
    fig = sphpl.make_figure(show_grid=False)
    shapes, _ = sphpl.add_reticle(fig, lon=200, lat=10,
                                     stroke_color=None)
    # Without stroke, just the 4 body arms.
    assert len(shapes) == 4


def test_add_reticle_uses_pixel_sizemode():
    """All reticle shapes are anchored with pixel sizemode so the
    crosshair stays the same pixel size under zoom."""
    fig = sphpl.make_figure(show_grid=False)
    shapes, _ = sphpl.add_reticle(fig, lon=180, lat=0)
    for s in shapes:
        assert s.xsizemode == 'pixel'
        assert s.ysizemode == 'pixel'
        assert s.type == 'path'


def test_add_reticle_anchored_at_projected_position():
    """Each shape's xanchor/yanchor should match the projected (lon, lat)."""
    fig = sphpl.make_figure(projection='AIT', center=0, show_grid=False)
    shapes, _ = sphpl.add_reticle(fig, lon=60, lat=30)
    x_expected, y_expected = _project(np.array([60.0]), np.array([30.0]),
                                       projection='AIT', center=0)
    for s in shapes:
        assert abs(float(s.xanchor) - float(x_expected[0])) < 1e-6
        assert abs(float(s.yanchor) - float(y_expected[0])) < 1e-6


def test_add_reticle_style_x_rotates_plus():
    """'x' style is plus rotated 45°. Two reticles at the same anchor,
    one plus and one x, should have the same number of shapes and
    overlapping bounding boxes (but different path strings)."""
    fig = sphpl.make_figure(show_grid=False)
    plus_shapes, _ = sphpl.add_reticle(fig, lon=180, lat=0, style='plus',
                                          stroke_color=None)
    x_shapes, _ = sphpl.add_reticle(fig, lon=180, lat=0, style='x',
                                       stroke_color=None)
    assert len(plus_shapes) == len(x_shapes) == 4
    # Path strings differ (rotation moves the endpoints).
    assert plus_shapes[0].path != x_shapes[0].path


def test_add_reticle_style_L_returns_two_arms():
    fig = sphpl.make_figure(show_grid=False)
    shapes, _ = sphpl.add_reticle(fig, lon=200, lat=10, style='L',
                                     stroke_color=None)
    assert len(shapes) == 2


def test_add_reticle_circle_returns_two_shapes_with_stroke():
    """Circle is a single polyline so stroke+body = 2 shapes."""
    fig = sphpl.make_figure(show_grid=False)
    shapes, _ = sphpl.add_reticle(fig, lon=200, lat=10, style='circle',
                                     size=20)
    assert len(shapes) == 2


def test_add_reticle_broken_circle_path_skips_gap():
    """``circle_gap_deg`` cuts a wedge from the ring; the path string
    should be shorter (fewer sample points) than a closed circle."""
    fig_a = sphpl.make_figure(show_grid=False)
    closed_shapes, _ = sphpl.add_reticle(fig_a, lon=0, lat=0,
                                            style='circle',
                                            circle_npts=128,
                                            stroke_color=None)
    fig_b = sphpl.make_figure(show_grid=False)
    broken_shapes, _ = sphpl.add_reticle(fig_b, lon=0, lat=0,
                                            style='circle',
                                            circle_gap_deg=60,
                                            circle_npts=128,
                                            stroke_color=None)
    # Both still produce one path each (no stroke layer in this test).
    assert len(closed_shapes) == 1
    assert len(broken_shapes) == 1


def test_add_reticle_alias_plus_and_circle():
    fig = sphpl.make_figure(show_grid=False)
    shapes_p, _ = sphpl.add_reticle(fig, lon=0, lat=0, style='+',
                                       stroke_color=None)
    shapes_c, _ = sphpl.add_reticle(fig, lon=0, lat=0, style='o',
                                       stroke_color=None)
    assert len(shapes_p) == 4  # plus
    assert len(shapes_c) == 1  # circle


def test_add_reticle_invalid_style_raises():
    fig = sphpl.make_figure(show_grid=False)
    with pytest.raises(ValueError, match="style must be one of"):
        sphpl.add_reticle(fig, lon=0, lat=0, style='star')


def test_add_reticle_invalid_label_side_raises():
    fig = sphpl.make_figure(show_grid=False)
    with pytest.raises(ValueError, match="label_side must be one of"):
        sphpl.add_reticle(fig, lon=0, lat=0, label='X', label_side='UPISH')


def test_add_reticle_label_anchors_match_compass_side():
    """Each compass side maps to a fixed (xanchor, yanchor) pair on the
    annotation; verify a couple."""
    fig = sphpl.make_figure(show_grid=False)
    _, lbl_ne = sphpl.add_reticle(fig, lon=180, lat=0,
                                     label='NE', label_side='NE')
    _, lbl_sw = sphpl.add_reticle(fig, lon=180, lat=0,
                                     label='SW', label_side='SW')
    # NE → 'left', 'bottom' (text extends right + up).
    assert (lbl_ne.xanchor, lbl_ne.yanchor) == ('left', 'bottom')
    # SW → 'right', 'top' (text extends left + down).
    assert (lbl_sw.xanchor, lbl_sw.yanchor) == ('right', 'top')


def test_add_reticle_auto_label_side_avoids_clipping():
    """``label_side='auto'`` picks the corner pointing into the largest
    free quadrant of the projected data extent. For AIT(center=180)
    direction='sky' with anchor near the figure's right edge (lon=10
    projects to positive x = right of center), the auto side should
    point LEFT (W component), so the label resolves to SW or NW."""
    fig = sphpl.make_figure(projection='AIT', center=180,
                              direction='sky', show_grid=False)
    _, lbl = sphpl.add_reticle(fig, lon=10, lat=0, label='X',
                                  label_side='auto')
    assert lbl.xanchor == 'right'  # left-pointing label sides


def test_add_reticle_label_color_default_matches_body():
    fig = sphpl.make_figure(show_grid=False)
    _, lbl = sphpl.add_reticle(fig, lon=0, lat=0, label='X',
                                  color='lime')
    assert lbl.font.color == 'lime'


def test_add_reticle_label_color_override():
    fig = sphpl.make_figure(show_grid=False)
    _, lbl = sphpl.add_reticle(fig, lon=0, lat=0, label='X',
                                  color='lime', label_color='black')
    assert lbl.font.color == 'black'


def test_add_reticle_label_offset_pushes_past_size():
    """The label's pixel offset should equal ``size + label_offset``
    along the corner direction (radial)."""
    fig = sphpl.make_figure(show_grid=False)
    _, lbl = sphpl.add_reticle(fig, lon=0, lat=0,
                                  size=12, label_offset=2,
                                  label='X', label_side='E')
    # 'E' direction: (+1, 0) — only xshift, yshift = 0.
    assert lbl.xshift == 12 + 2
    assert lbl.yshift == 0


# ============================================================================
# CompoundRegion plotly bridge — SkyplothelperProjector + add_compound_region
# ============================================================================

def test_skyplothelper_projector_from_figure_picks_up_metadata():
    from skyplothelper.plotly.projector import SkyplothelperProjector
    fig = sphpl.make_figure(projection='MOL', center=120, theme='dark')
    proj = SkyplothelperProjector.from_figure(fig)
    assert proj.projection == 'MOL'
    assert proj.center == 120.0
    assert proj.wcs_frame == 'icrs'


def test_skyplothelper_projector_project_polygon_returns_geom():
    from shapely.geometry.base import BaseGeometry

    from skyplothelper.plotly.projector import SkyplothelperProjector
    proj = SkyplothelperProjector(projection='AIT', center=180)
    from skyplothelper.geometry.shapes import geodesic_circle
    lons, lats = geodesic_circle(180.0, 0.0, 20.0, 100)
    geom = proj.project_polygon(lons, lats)
    assert isinstance(geom, BaseGeometry)
    assert not geom.is_empty


def test_skyplothelper_projector_frame_polygon_is_closed():
    from skyplothelper.plotly.projector import SkyplothelperProjector
    proj = SkyplothelperProjector(projection='AIT', center=180)
    fp = proj.frame_polygon
    assert fp.is_valid
    assert fp.area > 0


def test_compound_region_with_plotly_projector_circle():
    from skyplothelper import CompoundRegion
    from skyplothelper.plotly.projector import SkyplothelperProjector
    proj = SkyplothelperProjector(projection='AIT', center=180)
    region = CompoundRegion(proj).add_circle(180, 0, 20)
    assert region._geom is not None
    assert region._geom.area > 0


def test_compound_region_boolean_ops_via_plotly_projector():
    """add_circle - subtract_circle should give a smaller area than
    add_circle alone."""
    from skyplothelper import CompoundRegion
    from skyplothelper.plotly.projector import SkyplothelperProjector
    proj = SkyplothelperProjector(projection='AIT', center=180)
    r1 = CompoundRegion(proj).add_circle(180, 0, 20)
    r2 = CompoundRegion(proj).add_circle(180, 0, 20).subtract_circle(180, 0, 10)
    assert r2._geom.area < r1._geom.area


def test_compound_region_plotly_contains_point():
    """contains_point / contains_points route through the plotly
    projector's projection primitive, not an mpl WCS."""
    from skyplothelper import CompoundRegion
    from skyplothelper.plotly.projector import SkyplothelperProjector
    proj = SkyplothelperProjector(projection='AIT', center=180)
    region = CompoundRegion(proj).add_circle(180, 0, 20)
    assert region.contains_point(180, 0) is True
    assert region.contains_point(180, 60) is False
    mask = region.contains_points([180, 180, 180], [0, 10, 60])
    assert list(mask) == [True, True, False]


# ---------------------------------------------------------------------------
# CompoundRegion on zenithal / globe projections
# ---------------------------------------------------------------------------
#
# The frame silhouette used to be traced from the wrap meridians at
# center ± 180. On a zenithal globe that is the *far* side of the sphere --
# unprojectable, so the polygon collapsed to a sliver and clipped every
# region away to nothing, even a compact circle on the near hemisphere.

def test_skyplothelper_projector_globe_frame_polygon_is_not_degenerate():
    from skyplothelper.plotly.projector import SkyplothelperProjector
    proj = SkyplothelperProjector(projection='SIN', center=266.4,
                                  lat_center=-29.0)
    fp = proj.frame_polygon
    assert fp.is_valid
    # SIN's limb has projected radius 180/pi ~= 57.3, so area ~= pi*57.3**2.
    assert fp.area == pytest.approx(np.pi * (180.0 / np.pi) ** 2, rel=0.02)


def test_compound_region_circle_renders_on_sin_globe():
    """The paste-back repro: a compact near-side circle on a SIN globe used to
    come out empty because the frame collapsed."""
    fig = sphpl.make_figure(projection='SIN', center=266.4, lat_center=-29)
    region = sphpl.make_compound_region(fig).add_circle(266.4, -29, 18)
    assert region._geom is not None and not region._geom.is_empty
    assert region._geom.area > 0


# NCP is a deprecated CTYPE astropy silently "celfix"es to SIN; the warning is
# from project()'s internal header, not this test's concern.
@pytest.mark.filterwarnings("ignore::astropy.wcs.FITSFixedWarning")
@pytest.mark.parametrize('projection', ['SIN', 'TAN', 'ARC', 'ZEA', 'STG',
                                        'AZP', 'NCP'])
def test_compound_region_renders_across_zenithal_families(projection):
    """A circle on the visible hemisphere renders for every zenithal code,
    including the ones whose visible region isn't a centered disk (NCP)."""
    fig = sphpl.make_figure(projection=projection, center=120, lat_center=40)
    region = sphpl.make_compound_region(fig).add_circle(120, 40, 12)
    assert region._geom is not None and not region._geom.is_empty


def test_compound_region_globe_annulus_keeps_hole():
    """Set algebra survives on a globe: an annulus renders with its hole."""
    fig = sphpl.make_figure(projection='SIN', center=266.4, lat_center=-29)
    region = (sphpl.make_compound_region(fig)
              .add_circle(266.4, -29, 18).subtract_circle(266.4, -29, 9))
    shape, _ = sphpl.add_compound_region(fig, region)
    assert shape is not None
    assert shape.path.count('M ') >= 2   # exterior + hole subpaths


def test_compound_region_far_side_of_globe_is_empty():
    """A circle behind the globe is genuinely not visible, so an empty region
    is the correct result -- the fix must not make everything non-empty."""
    fig = sphpl.make_figure(projection='SIN', center=0, lat_center=0)
    region = sphpl.make_compound_region(fig).add_circle(180, 0, 10)
    assert region._geom is None or region._geom.is_empty


def test_compound_region_cylindrical_frame_unchanged():
    """The wrap-meridian frame path is untouched for non-zenithal codes."""
    from skyplothelper.plotly.projector import SkyplothelperProjector
    for projection in ['AIT', 'MOL', 'CAR']:
        proj = SkyplothelperProjector(projection=projection, center=180)
        assert proj.frame_polygon.area > 40000


def test_compound_region_plotly_area_and_solid_angle():
    """area_frac / solid_angle are geometry-only and work on plotly."""
    from skyplothelper import CompoundRegion
    from skyplothelper.plotly.projector import SkyplothelperProjector
    proj = SkyplothelperProjector(projection='AIT', center=180)
    region = CompoundRegion(proj).add_circle(180, 0, 20)
    assert 0 < region.area_frac < 1
    # A 20-deg-radius disc is ~pi*20^2 ~ 1257 sq deg.
    assert 1100 < region.solid_angle['sq_deg'] < 1400


def test_compound_region_plotly_expand_contract():
    """expand / contract buffer in canvas space via the projector's
    angle_to_pixels scale estimate."""
    from skyplothelper import CompoundRegion
    from skyplothelper.plotly.projector import SkyplothelperProjector
    proj = SkyplothelperProjector(projection='AIT', center=180)
    base = CompoundRegion(proj).add_circle(180, 0, 20)._geom.area
    bigger = CompoundRegion(proj).add_circle(180, 0, 20).expand(5)._geom.area
    smaller = CompoundRegion(proj).add_circle(180, 0, 20).contract(5)._geom.area
    assert bigger > base > smaller


def test_make_compound_region_factory():
    fig = sphpl.make_figure(projection='AIT', center=180)
    region = sphpl.make_compound_region(fig)
    assert region.projector.projection == 'AIT'
    assert region.projector.center == 180.0


def test_compound_region_plotly_add_lonlat_box():
    # Regression: CompoundRegion.add_lonlat_box reads projector.wcs_frame
    # (now part of the Projector interface). The plotly SkyplothelperProjector
    # reports 'icrs', so the cross-frame box path must build a non-empty
    # region rather than AttributeError on a backend-specific attribute.
    fig = sphpl.make_figure(projection='AIT', center=180, show_grid=False)
    region = sphpl.make_compound_region(fig)
    assert region.projector.wcs_frame == 'icrs'
    region.add_lonlat_box(lat_min=-10, lat_max=10,
                          lon_min=100, lon_max=160, frame='galactic')
    assert not region.is_empty


def test_add_compound_region_renders_path_shape():
    fig = sphpl.make_figure(projection='AIT', center=180, theme='dark',
                              show_grid=False)
    region = sphpl.make_compound_region(fig).add_circle(180, 0, 20)
    shape, overlay = sphpl.add_compound_region(
        fig, region, color='cyan', fillcolor='rgba(80,200,255,0.4)')
    assert shape.type == 'path'
    assert shape.path  # non-empty SVG path
    assert overlay is None  # no name → no overlay


def test_add_compound_region_with_name_emits_overlay():
    fig = sphpl.make_figure(projection='AIT', center=180,
                              show_grid=False)
    region = sphpl.make_compound_region(fig).add_circle(180, 0, 15)
    shape, overlay = sphpl.add_compound_region(
        fig, region, name='Disc', hover=True)
    assert overlay is not None
    assert overlay.name == 'Disc'
    # Under the default area anchor the hover rides on a companion fill
    # trace; plotly renders a fill tooltip from ``text``, not from the
    # hovertemplate, so that is where the label lives.
    assert '<b>Disc</b>' in _fill_hover_traces(fig)[0].text


def test_add_compound_region_legend_per_polygon():
    """legend_per_polygon emits one overlay per disconnected piece, each
    with its own numbered legend name."""
    fig = sphpl.make_figure(projection='AIT', center=180, show_grid=False)
    # Two non-overlapping circles → a 2-piece MultiPolygon.
    region = (sphpl.make_compound_region(fig)
              .add_circle(150, 0, 10).add_circle(210, 0, 10))
    assert region._geom.geom_type == 'MultiPolygon'
    shape, overlay = sphpl.add_compound_region(
        fig, region, name='Mask', legend_per_polygon=True)
    assert isinstance(overlay, list)
    assert len(overlay) == 2
    assert {o.name for o in overlay} == {'Mask 1', 'Mask 2'}


def test_add_compound_region_single_piece_one_overlay():
    """legend_per_polygon falls back to a single overlay when the region
    is one connected piece."""
    fig = sphpl.make_figure(projection='AIT', center=180, show_grid=False)
    region = sphpl.make_compound_region(fig).add_circle(180, 0, 15)
    shape, overlay = sphpl.add_compound_region(
        fig, region, name='Disc', legend_per_polygon=True)
    assert not isinstance(overlay, list)
    assert overlay.name == 'Disc'


def test_add_compound_region_empty_returns_none():
    from skyplothelper import CompoundRegion
    from skyplothelper.plotly.projector import SkyplothelperProjector
    fig = sphpl.make_figure(show_grid=False)
    # An empty region (no shapes added) → nothing to render.
    proj = SkyplothelperProjector.from_figure(fig)
    region = CompoundRegion(proj)
    shape, overlay = sphpl.add_compound_region(fig, region)
    assert shape is None
    assert overlay is None


def test_add_compound_region_path_includes_hole_for_annulus():
    """An add_circle - subtract_inner_circle creates a polygon with a
    hole — the SVG path should contain TWO subpath ``M`` commands."""
    fig = sphpl.make_figure(show_grid=False)
    region = (sphpl.make_compound_region(fig)
              .add_circle(180, 0, 20).subtract_circle(180, 0, 10))
    shape, _ = sphpl.add_compound_region(fig, region)
    assert shape.path.count('M ') >= 2


def test_add_compound_region_frame_band():
    fig = sphpl.make_figure(show_grid=False)
    region = sphpl.make_compound_region(fig).add_frame_band(
        -10, 10, frame='galactic')
    shape, _ = sphpl.add_compound_region(fig, region)
    assert shape is not None
    assert shape.path


# ---------------------------------------------------------------------------
# add_compound_region: hover_anchor
# ---------------------------------------------------------------------------
#
# A region's fill is a ``layout.shape`` SVG path, and plotly shapes can't
# hover. The hover therefore rides on an invisible companion Scatter. It used
# to be a single 1-px marker, so the tooltip only fired when the cursor
# happened to land on that one point; ``hover_anchor='area'`` traces the
# region's rings under ``hoveron='fills'`` instead.
#
# Plotly decides fill-hover by walking the trace's subpath polygons and
# toggling a flag on each one that contains the cursor -- an even-odd rule
# (plotly.js scatter/hover.js). ``_plotly_fill_hovers`` below replicates that
# exactly, which lets us assert on hover geometry without a browser. It is
# also why interior rings are emitted: a point inside a hole is contained by
# both the exterior and the hole ring, toggles twice, and stays silent.

def _fill_hover_traces(fig):
    """The region's companion fill-hover traces, if any.

    ``fig.add_trace`` stores a *copy*, so these can't be found by identity
    against what ``add_compound_region`` returned — match on content.
    """
    return [t for t in fig.data
            if getattr(t, 'fill', None) == 'toself'
            and getattr(t, 'hoveron', None) == 'fills']


def _fill_subpaths(trace):
    """Split a ``None``-separated fill trace into its subpath polygons."""
    out, cur_x, cur_y = [], [], []
    for x, y in zip(trace.x, trace.y):
        if x is None:
            out.append((cur_x, cur_y))
            cur_x, cur_y = [], []
        else:
            cur_x.append(x)
            cur_y.append(y)
    if cur_x:
        out.append((cur_x, cur_y))
    return out


def _plotly_fill_hovers(trace, point):
    """Would plotly show a fill tooltip for ``trace`` at canvas ``point``?"""
    from matplotlib.path import Path
    inside = False
    for xs, ys in _fill_subpaths(trace):
        if Path(np.column_stack([xs, ys])).contains_point(point):
            inside = not inside
    return inside


def _canvas_xy(region, lon, lat):
    x, y = region.projector.project_points(np.atleast_1d(lon),
                                            np.atleast_1d(lat))
    return float(np.ravel(x)[0]), float(np.ravel(y)[0])


def test_add_compound_region_area_hover_is_default_and_covers_the_region():
    """The hover target spans the whole region, not a single point."""
    fig = sphpl.make_figure(projection='AIT', center=180, show_grid=False)
    region = sphpl.make_compound_region(fig).add_circle(180, 0, 20)
    _, overlay = sphpl.add_compound_region(fig, region, name='Disc',
                                            hover=True)
    hover_traces = _fill_hover_traces(fig)
    assert len(hover_traces) == 1
    hover_trace = hover_traces[0]
    assert len(hover_trace.x) > 10
    # The legend entry stays on the marker; the fill trace is anonymous so
    # plotly can't print its name a second time beside the tooltip.
    assert overlay.name == 'Disc' and overlay.showlegend
    assert overlay.hoverinfo == 'skip'
    assert hover_trace.name == '' and hover_trace.showlegend is False
    # Hover fires well away from the representative point, and stops
    # outside the disc.
    for lat in (-15.0, 0.0, 15.0):
        assert _plotly_fill_hovers(hover_trace, _canvas_xy(region, 180.0, lat))
    assert not _plotly_fill_hovers(hover_trace, _canvas_xy(region, 180.0, 50.0))


def test_add_compound_region_area_hover_stays_silent_inside_holes():
    """An annulus emits both rings, so plotly's even-odd containment leaves
    the subtracted hole un-hoverable."""
    fig = sphpl.make_figure(projection='AIT', center=180, show_grid=False)
    region = (sphpl.make_compound_region(fig)
              .add_circle(180, 0, 20).subtract_circle(180, 0, 10))
    sphpl.add_compound_region(fig, region, name='Annulus', hover=True)
    hover_trace = _fill_hover_traces(fig)[0]
    assert len(_fill_subpaths(hover_trace)) == 2       # exterior + hole
    assert _plotly_fill_hovers(hover_trace, _canvas_xy(region, 180.0, 15.0))
    assert not _plotly_fill_hovers(hover_trace, _canvas_xy(region, 180.0, 0.0))


def test_add_compound_region_area_hover_follows_a_galactic_band():
    """A wavy band hovers everywhere along its spine and nowhere outside —
    the case the old representative-point anchor got wrong."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    fig = sphpl.make_figure(projection='AIT', center=180, show_grid=False)
    region = sphpl.make_compound_region(fig).add_frame_band(
        -10, 10, frame='galactic')
    sphpl.add_compound_region(fig, region, name='ZoA', hover=True)
    hover_trace = _fill_hover_traces(fig)[0]

    gl = np.linspace(0.0, 359.0, 60)
    def _hits(b_deg):
        icrs = SkyCoord(l=gl * u.deg, b=np.full_like(gl, b_deg) * u.deg,
                        frame='galactic').icrs
        x, y = region.projector.project_points(icrs.ra.deg, icrs.dec.deg)
        x, y = np.ravel(x), np.ravel(y)
        ok = np.isfinite(x) & np.isfinite(y)
        return sum(_plotly_fill_hovers(hover_trace, (a, b))
                    for a, b in zip(x[ok], y[ok])), int(ok.sum())

    inside, n_in = _hits(0.0)
    outside, _ = _hits(25.0)
    assert inside == n_in, f"only {inside}/{n_in} points on b=0 hover"
    assert outside == 0


def test_add_compound_region_area_hover_custom_string_reaches_text():
    """Plotly renders a fill tooltip from ``text``; a custom hover string
    must land there or it would be silently dropped."""
    fig = sphpl.make_figure(show_grid=False)
    region = sphpl.make_compound_region(fig).add_circle(180, 0, 15)
    sphpl.add_compound_region(fig, region, name='D', hover='Zone <i>A</i>')
    assert _fill_hover_traces(fig)[0].text == 'Zone <i>A</i>'


def test_add_compound_region_hover_anchor_point_uses_hovertemplate():
    """``hover_anchor='point'`` keeps the single-marker overlay, and there a
    real hovertemplate *is* honored — marker hover, not fill hover."""
    fig = sphpl.make_figure(projection='AIT', center=180, show_grid=False)
    region = sphpl.make_compound_region(fig).add_circle(180, 0, 15)
    _, overlay = sphpl.add_compound_region(
        fig, region, name='Disc', hover=True, hover_anchor='point')
    assert not _fill_hover_traces(fig)          # no companion fill trace
    assert overlay.mode == 'markers' and len(overlay.x) == 1
    assert '<b>Disc</b>' in overlay.hovertemplate


def test_add_compound_region_no_hover_emits_only_the_legend_marker():
    """With hover off there is nothing to hover, so no fill trace is added."""
    fig = sphpl.make_figure(show_grid=False)
    region = sphpl.make_compound_region(fig).add_circle(180, 0, 15)
    _, overlay = sphpl.add_compound_region(fig, region, name='D')
    assert not _fill_hover_traces(fig)
    markers = [t for t in fig.data if getattr(t, 'mode', None) == 'markers']
    assert len(markers) == 1                   # just the legend anchor
    assert overlay.mode == 'markers' and overlay.hoverinfo == 'skip'


def test_add_compound_region_legend_per_polygon_hovers_each_piece():
    fig = sphpl.make_figure(show_grid=False)
    region = (sphpl.make_compound_region(fig)
              .add_circle(150, 0, 10).add_circle(210, 0, 10))
    _, overlays = sphpl.add_compound_region(
        fig, region, name='Mask', hover=True, legend_per_polygon=True)
    assert len(overlays) == 2 and len(_fill_hover_traces(fig)) == 2


def test_add_compound_region_bad_hover_anchor_raises():
    fig = sphpl.make_figure(show_grid=False)
    region = sphpl.make_compound_region(fig).add_circle(180, 0, 15)
    with pytest.raises(ValueError, match="hover_anchor must be"):
        sphpl.add_compound_region(fig, region, name='D', hover_anchor='edge')


# ============================================================================
# add_ruler
# ============================================================================

def test_add_ruler_returns_main_shapes_anns():
    fig = sphpl.make_figure(show_grid=False)
    main, shapes, anns = sphpl.add_ruler(
        fig, lon1=170, lat1=0, lon2=180, lat2=0,
        n_ticks=5)
    assert main.mode == 'lines'
    # 5 ticks × 2 sides × 2 layers (stroke + body) = 20 shapes.
    assert len(shapes) == 20
    assert len(anns) == 5  # 5 tick labels


def test_add_ruler_geodesic_samples_n_points():
    fig = sphpl.make_figure(show_grid=False)
    # Endpoints on the same side of the wrap seam (lon=180 at center=0)
    # so the arc isn't wrap-split — verifies the raw geodesic sample count.
    lon2, lat2 = 140, -20
    main, _, _ = sphpl.add_ruler(
        fig, lon1=40, lat1=30, lon2=lon2, lat2=lat2,
        geodesic=True, n_geodesic_pts=64)
    # 64 _slerp samples + the appended endpoint = 65, so the drawn line
    # reaches the true endpoint (where the endcap / final tick sit) rather
    # than stopping one step short.
    assert len(main.x) == 65
    end_x = float(np.ravel(_project(lon2, lat2, projection='AIT', center=0)[0])[0])
    assert main.x[-1] == pytest.approx(end_x)


def test_add_ruler_geodesic_wrap_splits_at_seam():
    """A geodesic ruler crossing the projection seam is wrap-split (NaN
    break) instead of streaking straight across the canvas."""
    fig = sphpl.make_figure(show_grid=False)  # center=0 -> seam at lon=180
    main, _, _ = sphpl.add_ruler(
        fig, lon1=100, lat1=30, lon2=260, lat2=-20,
        geodesic=True, n_geodesic_pts=64)
    mx = np.asarray(main.x, dtype=float)
    # The seam crossing inserts a NaN break...
    assert np.isnan(mx).any()
    # ...so no single segment streaks across the frame.
    d = np.abs(np.diff(mx))
    d = d[np.isfinite(d)]
    full_width = float(np.nanmax(mx) - np.nanmin(mx))
    assert d.size and float(np.max(d)) < 0.5 * full_width


def test_add_ruler_chord_has_two_vertices():
    fig = sphpl.make_figure(show_grid=False)
    main, _, _ = sphpl.add_ruler(
        fig, lon1=100, lat1=10, lon2=120, lat2=10,
        geodesic=False)
    assert len(main.x) == 2


def test_add_ruler_unit_auto_picks_arcsec_below_1deg():
    """Total span below 1° → arcsec unit (with 3600× scaling)."""
    fig = sphpl.make_figure(show_grid=False)
    _, _, anns = sphpl.add_ruler(
        fig, lon1=180, lat1=0, lon2=180.5, lat2=0, n_ticks=3)
    # Each label ends with the unit name.
    assert all(a.text.endswith('arcsec') for a in anns)


def test_add_ruler_unit_auto_picks_arcmin_under_60deg():
    fig = sphpl.make_figure(show_grid=False)
    _, _, anns = sphpl.add_ruler(
        fig, lon1=170, lat1=0, lon2=180, lat2=0, n_ticks=3)
    assert all(a.text.endswith('arcmin') for a in anns)


def test_add_ruler_unit_auto_picks_deg_above_60deg():
    fig = sphpl.make_figure(show_grid=False)
    _, _, anns = sphpl.add_ruler(
        fig, lon1=100, lat1=0, lon2=200, lat2=0, n_ticks=3)
    assert all(a.text.endswith('deg') for a in anns)


def test_add_ruler_unit_auto_picks_mas_under_1arcsec():
    """Sub-arcsec spans (compact / VLBI fields) auto-pick milliarcsec."""
    fig = sphpl.make_figure(show_grid=False)
    _, _, anns = sphpl.add_ruler(
        fig, lon1=180, lat1=0, lon2=180 + 0.1 / 3600.0, lat2=0, n_ticks=3)
    assert all(a.text.endswith('mas') for a in anns)


def test_add_ruler_explicit_mas_unit():
    """label_unit='mas' scales degrees by 3.6e6 (1 arcsec = 1000 mas)."""
    fig = sphpl.make_figure(show_grid=False)
    _, _, anns = sphpl.add_ruler(
        fig, lon1=180, lat1=0, lon2=180 + 1.0 / 3600.0, lat2=0,
        label_unit='mas', tick_interval=500.0)
    nums = [float(a.text.split()[0]) for a in anns]
    assert all(a.text.endswith('mas') for a in anns)
    # A 500 mas tick proves mas scaling — arcsec would read 0.5, deg ~1.4e-4.
    assert any(abs(v - 500.0) < 1e-3 for v in nums)


def test_add_ruler_explicit_unit_override():
    fig = sphpl.make_figure(show_grid=False)
    _, _, anns = sphpl.add_ruler(
        fig, lon1=170, lat1=0, lon2=180, lat2=0,
        label_unit='deg', n_ticks=3, fmt='%.1f')
    assert all(' deg' in a.text for a in anns)


def test_add_ruler_explicit_tick_positions():
    fig = sphpl.make_figure(show_grid=False)
    _, _, anns = sphpl.add_ruler(
        fig, lon1=200, lat1=20, lon2=220, lat2=20,
        tick_positions=[0, 5, 10, 15],
        label_unit='deg', fmt='%.0f')
    texts = [a.text for a in anns]
    assert texts == ['0 deg', '5 deg', '10 deg', '15 deg']


def test_add_ruler_tick_side_one_sided_halves_shape_count():
    fig = sphpl.make_figure(show_grid=False)
    _, both_shapes, _ = sphpl.add_ruler(
        fig, lon1=100, lat1=0, lon2=120, lat2=0,
        n_ticks=4, tick_side='both', stroke_color=None)
    fig2 = sphpl.make_figure(show_grid=False)
    _, left_shapes, _ = sphpl.add_ruler(
        fig2, lon1=100, lat1=0, lon2=120, lat2=0,
        n_ticks=4, tick_side='left', stroke_color=None)
    assert len(left_shapes) == len(both_shapes) // 2


def test_add_ruler_tick_side_none_emits_no_tick_shapes():
    fig = sphpl.make_figure(show_grid=False)
    _, shapes, _ = sphpl.add_ruler(
        fig, lon1=100, lat1=0, lon2=120, lat2=0,
        n_ticks=4, tick_side='none', endcap_style='none',
        stroke_color=None)
    assert len(shapes) == 0


def test_add_ruler_endcap_style_arrow_replaces_endpoint_ticks():
    fig = sphpl.make_figure(show_grid=False)
    _, shapes, _ = sphpl.add_ruler(
        fig, lon1=100, lat1=0, lon2=120, lat2=0,
        n_ticks=5, endcap_style='arrow', endcaps='both',
        stroke_color=None)
    # 3 inner ticks × 2 sides = 6 tick shapes, plus 2 arrow paths = 8.
    n_path = sum(1 for s in shapes if s.type == 'path')
    n_line = sum(1 for s in shapes if s.type == 'line')
    assert n_path == 2  # one per arrow endcap
    assert n_line == 6  # 3 inner ticks × 2 sides


def test_add_ruler_endcap_style_tick_uses_longer_endpoint():
    """endcap_style='tick' draws longer endpoint marks."""
    fig = sphpl.make_figure(show_grid=False)
    _, shapes, _ = sphpl.add_ruler(
        fig, lon1=100, lat1=0, lon2=120, lat2=0,
        n_ticks=5, endcap_style='tick', endcaps='both',
        tick_length=4, endcap_length_scale=2.0,
        stroke_color=None)
    # Endpoint shapes have x1²+y1² = (tick_length * scale)² = 64.
    # Inner ticks have x1²+y1² = tick_length² = 16.
    lengths = sorted(set(
        round(float(s.x1)**2 + float(s.y1)**2, 1) for s in shapes))
    assert 16.0 in lengths  # inner ticks
    assert 64.0 in lengths  # endcaps (4×2)² = 64


def test_add_ruler_title_added_at_midpoint():
    fig = sphpl.make_figure(show_grid=False)
    _, _, anns = sphpl.add_ruler(
        fig, lon1=100, lat1=0, lon2=120, lat2=0,
        n_ticks=4, title='Ruler caption')
    titles = [a for a in anns if a.text == 'Ruler caption']
    assert len(titles) == 1


def test_add_ruler_stroke_doubles_shape_count():
    fig = sphpl.make_figure(show_grid=False)
    _, with_stroke, _ = sphpl.add_ruler(
        fig, lon1=100, lat1=0, lon2=120, lat2=0,
        n_ticks=4, stroke_color='black', stroke_lw=2.4)
    fig2 = sphpl.make_figure(show_grid=False)
    _, no_stroke, _ = sphpl.add_ruler(
        fig2, lon1=100, lat1=0, lon2=120, lat2=0,
        n_ticks=4, stroke_color=None)
    assert len(with_stroke) == 2 * len(no_stroke)


def test_add_ruler_labels_off_drops_annotations():
    fig = sphpl.make_figure(show_grid=False)
    _, _, anns = sphpl.add_ruler(
        fig, lon1=100, lat1=0, lon2=120, lat2=0,
        n_ticks=4, labels=False)
    assert anns == []


def test_add_ruler_validation_errors():
    fig = sphpl.make_figure(show_grid=False)
    bad_kwargs = [
        (dict(tick_side='north'), "tick_side must be"),
        (dict(label_side='up'), "label_side must be"),
        (dict(title_side='diag'), "title_side must be"),
        (dict(endcap_style='dotted'), "endcap_style must be"),
        (dict(endcaps='rear'), "endcaps must be"),
        (dict(label_unit='furlong'), "label_unit must be"),
        (dict(lambda0=1.5), "lambda0 must be"),
    ]
    for kw, msg in bad_kwargs:
        with pytest.raises(ValueError, match=msg):
            sphpl.add_ruler(fig, lon1=0, lat1=0, lon2=10, lat2=0,
                              **kw)


def test_add_ruler_hover_default_off():
    fig = sphpl.make_figure(show_grid=False)
    main, _, _ = sphpl.add_ruler(
        fig, lon1=100, lat1=0, lon2=120, lat2=0, n_ticks=3)
    assert main.hoverinfo == 'skip'


def test_add_ruler_hover_true_shows_length():
    fig = sphpl.make_figure(show_grid=False)
    main, _, _ = sphpl.add_ruler(
        fig, lon1=100, lat1=0, lon2=120, lat2=0,
        n_ticks=3, hover=True, name='My ruler')
    assert main.hovertemplate is not None
    assert '<b>My ruler</b>' in main.hovertemplate
    assert 'length:' in main.hovertemplate


def test_add_ruler_lambda0_centers_zero_symmetric():
    """lambda0=0.5 puts the value-0 tick at the midpoint, giving a
    symmetric span of signed labels."""
    fig = sphpl.make_figure(projection='AIT', center=180, show_grid=False)
    _, _, anns = sphpl.add_ruler(
        fig, 170, 0, 190, 0, lambda0=0.5, n_ticks=5)
    # Numeric label values, parsed off the trailing unit token.
    vals = [float(a.text.split()[0]) for a in anns]
    assert min(vals) < 0 < max(vals)
    assert abs(min(vals) + max(vals)) < 1e-6   # symmetric about zero


def test_add_ruler_label_fmt_callable_overrides_fmt():
    """A label_fmt callable receives (value_arcsec, unit) and wins over
    the built-in formatting."""
    fig = sphpl.make_figure(projection='AIT', center=180, show_grid=False)
    seen_units = []

    def fmt(value_asec, unit):
        seen_units.append(unit)
        return f"{value_asec:.0f}AS"

    _, _, anns = sphpl.add_ruler(
        fig, 179, 0, 181, 0, n_ticks=3, label_fmt=fmt)
    assert all(a.text.endswith('AS') for a in anns)
    assert seen_units  # callable was actually invoked


def test_add_ruler_tick_styling_independent_of_main_line():
    """tick_color / tick_lw / tick_ls style the ticks without touching
    the main line, and endcap_color / endcap_lw style the endcaps."""
    fig = sphpl.make_figure(projection='AIT', center=180, show_grid=False)
    main, ticks, _ = sphpl.add_ruler(
        fig, 170, 0, 190, 0, n_ticks=3, color='white', lw=1.0,
        tick_color='red', tick_lw=2.0, tick_ls='--',
        stroke_color=None)
    assert main.line.color == 'white'
    # With stroke disabled, each tick shape is a single styled line.
    styled = [s for s in ticks if s.type == 'line'
              and s.line.color == 'red']
    assert styled
    assert styled[0].line.width == 2.0
    assert styled[0].line.dash == 'dash'


def test_add_ruler_endcap_color_defaults_to_tick_color():
    fig = sphpl.make_figure(projection='AIT', center=180, show_grid=False)
    _, ticks, _ = sphpl.add_ruler(
        fig, 170, 0, 190, 0, n_ticks=3, tick_color='cyan',
        endcap_style='tick', stroke_color=None)
    assert any(s.line.color == 'cyan' for s in ticks if s.type == 'line')


def test_add_coord_labels_invalid_placement_raises():
    fig = sphpl.make_figure(theme='dark')
    with pytest.raises(ValueError, match="placement must be"):
        sphpl.add_coord_labels(fig, placement='diagonal')


def test_add_healpix_sparse_default_tile_resolution_is_auto():
    healpy = pytest.importorskip("healpy")  # noqa: F841
    from collections import Counter
    nside = 16
    pix = np.arange(0, 192 * 16, 17)  # spread across the sky
    vals = np.zeros_like(pix, dtype=float)
    fig = sphpl.make_figure(show_grid=False)
    traces = sphpl.add_healpix_sparse(fig, pix, vals, nside=nside)
    # nside=16 → 'auto' step=4 → 4*4+1 = 17 vertices for non-split tiles.
    counts = Counter(len(t.x) for t in traces)
    most_common_len, _ = counts.most_common(1)[0]
    assert most_common_len == 4 * 4 + 1


# ============================================================================
# add_legend — render skyplothelper legend blocks as plotly legend entries
# ============================================================================

def test_add_legend_categorical_named_traces():
    import skyplothelper as sph
    fig = sphpl.make_figure(projection='AIT')
    n0 = len(fig.data)
    traces = sphpl.add_legend(fig, [
        sph.ColorBlock('Cat', {'A': 'orange', 'B': 'C0'}, swatch='marker')])
    assert len(traces) == 2
    assert len(fig.data) == n0 + 2
    assert all(t.showlegend for t in traces)
    assert [t.name for t in traces] == ['A', 'B']
    assert fig.layout.showlegend is True


def test_add_legend_group_title_on_first_entry():
    import skyplothelper as sph
    fig = sphpl.make_figure(projection='AIT')
    traces = sphpl.add_legend(fig, [
        sph.ShapeBlock('Sample', {'a': 'o', 'b': 'D'})])
    assert traces[0].legendgrouptitle.text == 'Sample'
    assert traces[0].legendgroup == traces[1].legendgroup


def test_add_legend_single_block_accepted():
    import skyplothelper as sph
    fig = sphpl.make_figure(projection='AIT')
    traces = sphpl.add_legend(fig, sph.ColorBlock('C', {'a': 'red'}))
    assert len(traces) == 1


def test_add_legend_shapeblock_swatch_uses_block_color():
    """A ShapeBlock carries its shared swatch color in base_style; add_legend
    must merge base_style (like the mpl side) or the swatch renders invisible
    (transparent fill + no outline)."""
    import skyplothelper as sph
    fig = sphpl.make_figure(projection='AIT', theme='dark')
    traces = sphpl.add_legend(fig, [
        sph.ShapeBlock('Catalog', {'standard': 'o', 'defining': 'D'},
                       color='#C8CCD4', size=9)])
    for t in traces:
        # visible: a real color, not the transparent 'rgba(0,0,0,0)' fill.
        assert t.marker.color == '#c8ccd4'
        assert 'open' not in t.marker.symbol      # filled, so the fill shows
    assert [t.marker.symbol for t in traces] == ['circle', 'diamond']


def test_add_legend_base_style_facecolor_reaches_marker():
    """Generalise the ShapeBlock fix: any block's base_style facecolor must
    survive to the plotly marker, not just the varied per-entry channel."""
    import skyplothelper as sph
    fig = sphpl.make_figure(projection='AIT')
    traces = sphpl.add_legend(fig, [
        sph.ShapeBlock('S', {'a': 'o'}, color='seagreen')])
    from matplotlib.colors import to_hex
    assert traces[0].marker.color == to_hex('seagreen')


def test_add_legend_graduated_size_increasing():
    import skyplothelper as sph
    fig = sphpl.make_figure(projection='AIT')
    traces = sphpl.add_legend(fig, [
        sph.SizeBlock('N', values=[1, 10, 100, 500], smin=6, smax=40,
                      scale='sqrt')])
    sizes = [t.marker.size for t in traces]
    assert sizes == sorted(sizes)                     # graduated, ascending


def test_add_legend_open_fill_uses_open_symbol():
    import skyplothelper as sph
    fig = sphpl.make_figure(projection='AIT')
    traces = sphpl.add_legend(fig, [
        sph.FillBlock('Ap', {'full': 'filled', 'reduced': 'open'})])
    symbols = [t.marker.symbol for t in traces]
    assert symbols[0] == 'circle'
    assert symbols[1] == 'circle-open'


def test_add_legend_alpha_sets_opacity():
    import skyplothelper as sph
    fig = sphpl.make_figure(projection='AIT')
    traces = sphpl.add_legend(fig, [
        sph.AlphaBlock('D', values=[0, 10], amin=0.2, amax=1.0)])
    assert traces[0].marker.opacity == 0.2
    assert traces[-1].marker.opacity == 1.0


def test_add_legend_orientation_sets_angle():
    import skyplothelper as sph
    fig = sphpl.make_figure(projection='AIT')
    traces = sphpl.add_legend(fig, [
        sph.OrientBlock('PA', {'0': 0, '45': 45})])
    assert [t.marker.angle for t in traces] == [0.0, 45.0]


def test_add_legend_line_dash_mapping():
    import skyplothelper as sph
    fig = sphpl.make_figure(projection='AIT')
    traces = sphpl.add_legend(fig, [
        sph.LineBlock('Fit', {'model': '--', 'data': '-.'})])
    assert [t.line.dash for t in traces] == ['dash', 'dashdot']
    assert all(t.mode == 'lines' for t in traces)


def test_add_legend_colorbar_block_emits_colorscale_trace():
    import skyplothelper as sph
    fig = sphpl.make_figure(projection='AIT')
    traces = sphpl.add_legend(fig, [
        sph.ColorbarBlock('z', cmap='plasma', vmin=0, vmax=3)])
    assert len(traces) == 1
    assert traces[0].marker.showscale is True
    assert traces[0].marker.cmax == 3.0


def test_add_legend_skips_unsupported_with_warning():
    import skyplothelper as sph
    fig = sphpl.make_figure(projection='AIT')
    with pytest.warns(UserWarning, match="no plotly equivalent"):
        traces = sphpl.add_legend(fig, [sph.TextBlock('Notes', ['a note'])])
    assert traces == []


def test_add_legend_color_conversion_6digit_hex():
    from skyplothelper.plotly.core import _to_plotly_color
    assert _to_plotly_color('C0').startswith('#') and len(_to_plotly_color('C0')) == 7
    assert _to_plotly_color('none') == 'rgba(0,0,0,0)'


def test_add_legend_skips_glyph_block_with_warning():
    import skyplothelper as sph
    fig = sphpl.make_figure(projection='AIT')
    with pytest.warns(UserWarning, match="no plotly equivalent"):
        traces = sphpl.add_legend(fig, [
            sph.GlyphBlock('Targets', {'t': 'reticle_circle'})])
    assert traces == []


# ============================================================================
# add_ruler: minor ticks
# ============================================================================

def _ruler_shape_count(**kw):
    """Shapes added by a 20°, 5°-major-interval ruler (stroke off)."""
    fig = sphpl.make_figure(show_grid=False)
    _, shapes, _ = sphpl.add_ruler(
        fig, lon1=10, lat1=0, lon2=30, lat2=0,
        label_unit='deg', tick_interval=5, stroke_color=None, **kw)
    return len(shapes)


def test_add_ruler_minor_ticks_off_by_default():
    # 5 majors × 2 sides, no minors.
    assert _ruler_shape_count() == 10


def test_add_ruler_minor_ticks_subdivide_major_interval():
    """minor_ticks=n splits each major interval into n, matching the mpl
    Ruler / AutoMinorLocator semantic."""
    # step 5/5=1° over 20° → 21 grid points − 5 majors = 16 minors, ×2 sides.
    assert _ruler_shape_count(minor_ticks=5) == 10 + 32


def test_add_ruler_minor_ticks_auto_matches_mpl_subdivision():
    """'auto' resolves through the SAME shared helper as the mpl Ruler, so a
    5° major step subdivides into 5 on both backends."""
    from skyplothelper.overlays.ruler import _auto_minor_subdivisions
    assert _auto_minor_subdivisions(5.0) == 5
    assert _ruler_shape_count(minor_ticks='auto') == _ruler_shape_count(
        minor_ticks=5)


def test_add_ruler_minor_tick_interval_overrides_subdivision():
    # step 2.5° → 9 grid points − 5 majors = 4 minors, ×2 sides.
    assert _ruler_shape_count(minor_ticks=5,
                              minor_tick_interval=2.5) == 10 + 8


def test_add_ruler_minor_tick_side_follows_then_overrides():
    # tick_side='left' → 5 major shapes; minors 'auto' follow → 16.
    assert _ruler_shape_count(minor_ticks=5, tick_side='left') == 5 + 16
    # explicit override puts minors on both sides
    assert _ruler_shape_count(minor_ticks=5, tick_side='left',
                              minor_tick_side='both') == 5 + 32
    # and 'none' suppresses them
    assert _ruler_shape_count(minor_ticks=5, minor_tick_side='none') == 10


def test_add_ruler_minor_ticks_inherit_and_pin_style():
    fig = sphpl.make_figure(show_grid=False)
    _, shapes, _ = sphpl.add_ruler(
        fig, lon1=10, lat1=0, lon2=30, lat2=0, label_unit='deg',
        tick_interval=5, minor_ticks=5, stroke_color=None,
        tick_color='red', minor_tick_color='blue', minor_tick_lw=0.5)
    colors = {s.line.color for s in shapes}
    assert colors == {'red', 'blue'}
    widths = {s.line.width for s in shapes if s.line.color == 'blue'}
    assert widths == {0.5}


def test_add_ruler_minor_ticks_shorter_than_major_by_default():
    fig = sphpl.make_figure(show_grid=False)
    _, shapes, _ = sphpl.add_ruler(
        fig, lon1=10, lat1=0, lon2=30, lat2=0, label_unit='deg',
        tick_interval=5, minor_ticks=5, stroke_color=None,
        tick_length=8.0, tick_color='red', minor_tick_color='blue')

    def _len(s):
        return float(np.hypot(s.x1, s.y1))
    major = next(_len(s) for s in shapes if s.line.color == 'red')
    minor = next(_len(s) for s in shapes if s.line.color == 'blue')
    assert minor == pytest.approx(0.5 * major)


def test_add_ruler_minor_ticks_geodesic():
    fig = sphpl.make_figure(show_grid=False)
    _, shapes, _ = sphpl.add_ruler(
        fig, lon1=10, lat1=0, lon2=60, lat2=30, geodesic=True,
        minor_ticks=4, stroke_color=None)
    assert len(shapes) > 0


@pytest.mark.parametrize("bad", [0, 1, "nope", 2.5])
def test_add_ruler_minor_ticks_rejects_bad_values(bad):
    fig = sphpl.make_figure(show_grid=False)
    with pytest.raises(ValueError):
        sphpl.add_ruler(fig, lon1=10, lat1=0, lon2=30, lat2=0,
                        minor_ticks=bad)


def test_add_ruler_minor_tick_side_rejects_unknown():
    fig = sphpl.make_figure(show_grid=False)
    with pytest.raises(ValueError):
        sphpl.add_ruler(fig, lon1=10, lat1=0, lon2=30, lat2=0,
                        minor_tick_side='sideways')


# ============================================================================
# display frame: SkyCoord input + frame-aware overlays
# ============================================================================

from astropy.coordinates import SkyCoord  # noqa: E402

# The figure's ``frame=`` is the frame its x/y actually mean. Overlays defined
# in another frame convert INTO it; bare numbers are already display coords.
# This used to be hard-coded ICRS, so a galactic figure silently drew the
# galactic plane as a sinusoid instead of a straight line.


def _new_trace_y(fig, build):
    n0 = len(fig.data)
    build(fig)
    ys = []
    for tr in list(fig.data)[n0:]:
        y = np.asarray(tr.y, dtype=float)
        y = y[np.isfinite(y)]
        if y.size:
            ys.append(y)
    assert ys, "overlay added no finite points"
    return np.concatenate(ys)


def test_galactic_plane_is_flat_on_a_galactic_figure():
    fig = sphpl.make_figure(projection="AIT", frame="galactic")
    y = _new_trace_y(fig, lambda f: sphpl.add_great_circle(f, frame="galactic"))
    assert np.allclose(y, 0.0, atol=1e-6)


def test_galactic_plane_is_curved_on_an_equatorial_figure():
    """The control: cross-frame really must bend."""
    fig = sphpl.make_figure(projection="AIT", frame="icrs")
    y = _new_trace_y(fig, lambda f: sphpl.add_great_circle(f, frame="galactic"))
    assert y.ptp() > 50.0


def test_undeclared_figure_still_behaves_as_equatorial():
    """Blast-radius guard: default figures must not have moved."""
    a = sphpl.make_figure(projection="AIT")
    ya = _new_trace_y(a, lambda f: sphpl.add_great_circle(f, frame="galactic"))
    b = sphpl.make_figure(projection="AIT", frame="icrs")
    yb = _new_trace_y(b, lambda f: sphpl.add_great_circle(f, frame="galactic"))
    assert np.allclose(ya, yb)


def test_lonlat_box_uses_the_figure_frame():
    fig = sphpl.make_figure(projection="AIT", frame="galactic")
    y = _new_trace_y(fig, lambda f: sphpl.add_lonlat_box(
        f, lon_min=10, lon_max=40, lat_min=-5, lat_max=5, frame="galactic"))
    assert abs(y).max() < 6.0        # a ±5° box stays a ±5° box


def test_frame_band_uses_the_figure_frame():
    fig = sphpl.make_figure(projection="AIT", frame="galactic")
    y = _new_trace_y(fig, lambda f: sphpl.add_frame_band(
        f, -10, 10, frame="galactic"))
    assert abs(y.min() + y.max()) < 1e-6      # symmetric about the equator


# ---- SkyCoord acceptance ----

def test_add_scatter_converts_a_skycoord_into_the_figure_frame():
    g = SkyCoord(l=[0.0], b=[0.0], unit="deg", frame="galactic")
    fig = sphpl.make_figure(frame="icrs")
    sphpl.add_scatter(fig, g)
    cd = np.asarray(fig.data[-1].customdata, dtype=float)
    assert cd[0, 0] == pytest.approx(266.405, abs=0.05)
    assert cd[0, 1] == pytest.approx(-28.936, abs=0.05)


def test_add_scatter_leaves_a_matching_frame_untouched():
    g = SkyCoord(l=[12.0], b=[34.0], unit="deg", frame="galactic")
    fig = sphpl.make_figure(frame="galactic")
    sphpl.add_scatter(fig, g)
    cd = np.asarray(fig.data[-1].customdata, dtype=float)
    assert cd[0, 0] == pytest.approx(12.0)
    assert cd[0, 1] == pytest.approx(34.0)


def test_bare_degrees_are_taken_at_face_value():
    fig = sphpl.make_figure(frame="galactic")
    sphpl.add_scatter(fig, [12.0], [34.0])
    cd = np.asarray(fig.data[-1].customdata, dtype=float)
    assert cd[0, 0] == pytest.approx(12.0)


@pytest.mark.parametrize("call", [
    lambda f, c: sphpl.add_scatter(f, c),
    lambda f, c: sphpl.add_spherical_polygon(f, c),
    lambda f, c: sphpl.add_reticle(f, c[0]),
    lambda f, c: sphpl.add_geodesic_circle(f, c[0], 5.0),
    lambda f, c: sphpl.add_sky_vectors(f, c, dlon=np.ones(len(c)),
                                       dlat=np.ones(len(c))),
])
def test_skycoord_accepted_by_the_coordinate_helpers(call):
    c = SkyCoord([10.0, 20.0], [30.0, 40.0], unit="deg")
    call(sphpl.make_figure(), c)


def test_project_accepts_a_skycoord():
    c = SkyCoord([10.0, 20.0], [30.0, 40.0], unit="deg")
    x, y = sphpl.project(c, projection="MOL")
    assert np.isfinite(x).all() and np.isfinite(y).all()


def test_sky_vectors_rejects_a_skycoord_with_positional_magnitudes():
    """Two trailing arrays can't shift unambiguously — same rule as mpl."""
    c = SkyCoord([10.0, 20.0], [30.0, 40.0], unit="deg")
    d = np.ones(2)
    with pytest.raises(TypeError):
        sphpl.add_sky_vectors(sphpl.make_figure(), c, d, d)


def test_geodesic_circle_without_a_radius_raises():
    c = SkyCoord(10.0, 30.0, unit="deg")
    with pytest.raises(TypeError):
        sphpl.add_geodesic_circle(sphpl.make_figure(), c)


# ============================================================================
# Track A defect fixes (hard-coded-value audit, 2026-07-19)
# ============================================================================

@pytest.mark.parametrize("frame,lon_name", [
    ("icrs", "RA"), ("galactic", "l"),
    ("supergalactic", "SGL"), ("ecliptic", "λ"),
])
def test_hover_labels_follow_the_figure_frame(frame, lon_name):
    """Hover said RA/Dec unconditionally, which mislabels a galactic map."""
    fig = sphpl.make_figure(frame=frame)
    sphpl.add_scatter(fig, [10.0], [20.0])
    assert fig.data[-1].hovertemplate.startswith(f"{lon_name}:")


def test_galactic_hover_does_not_say_ra():
    fig = sphpl.make_figure(frame="galactic")
    sphpl.add_scatter(fig, [10.0], [20.0])
    assert "RA" not in fig.data[-1].hovertemplate


def test_default_figure_hover_is_unchanged():
    """Blast-radius guard: equatorial figures must still read RA/Dec."""
    fig = sphpl.make_figure()
    sphpl.add_scatter(fig, [10.0], [20.0])
    tpl = fig.data[-1].hovertemplate
    assert "RA:" in tpl and "Dec:" in tpl


def test_explicit_hovertemplate_still_wins():
    fig = sphpl.make_figure(frame="galactic")
    sphpl.add_scatter(fig, [10.0], [20.0], hovertemplate="%{customdata[0]}")
    assert fig.data[-1].hovertemplate == "%{customdata[0]}"


def test_make_figure_stamps_the_resolved_theme_foreground():
    """`theme=` was resolved and then thrown away, leaving a dead reader."""
    for theme, want in (("light", "#1a1a1a"), ("dark", "#dcdcdc")):
        fig = sphpl.make_figure(theme=theme)
        assert fig.layout.meta["sph_fg"] == want


def test_dark_background_detected_by_luminance_not_by_prefix():
    """The old sniff tested for a literal leading '#0'."""
    from skyplothelper.plotly.core import _is_dark_color, _theme_fg
    assert _is_dark_color("#1D1C1A")           # dark, no leading '#0'
    assert _is_dark_color("rgb(20,20,20)")     # plotly's own spelling
    assert not _is_dark_color("#ffffff")
    assert not _is_dark_color("")
    fig = sphpl.make_figure(theme="light")
    fig.update_layout(meta=None, paper_bgcolor="#1D1C1A")
    assert _theme_fg(fig) == "#dcdcdc"         # light ink on a dark canvas


def test_fits_beam_colors_are_forwardable():
    """Both beam builders exposed line_color/fillcolor; add_fits_image
    forwarded neither, so the ellipse was permanently white — invisible on a
    light theme or a reversed colormap."""
    import astropy.io.fits as pyfits
    from astropy.wcs import WCS

    h = pyfits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, 40, 40
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRVAL1"], h["CRVAL2"] = 83.6, 22.0
    h["CRPIX1"] = h["CRPIX2"] = 20
    h["CDELT1"], h["CDELT2"] = -1e-4, 1e-4
    wcs = WCS(h)
    data = np.random.default_rng(0).normal(0, 1, (40, 40))

    def beam(**kw):
        fig = sphpl.make_fits_figure(wcs)
        sphpl.add_fits_image(fig, data, wcs=wcs, beam_maj=1.08, beam_min=0.72,
                             beam_pa=30.0, **kw)
        return [(s.line.color, s.fillcolor) for s in fig.layout.shapes
                if getattr(s, "name", "") == "sph_fits_beam"]

    assert beam() == [("white", "rgba(255,255,255,0.25)")]      # unchanged
    assert beam(beam_color="#ff00ff",
                beam_fillcolor="rgba(255,0,255,0.3)") == \
        [("#ff00ff", "rgba(255,0,255,0.3)")]


# ============================================================================
# Region slider / explorer (compound_region_states + add_region_slider)
# ============================================================================

def _band_catalog(n=600, seed=0):
    """A random all-sky catalog as (lon, lat) degree arrays."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 360, n), rng.uniform(-90, 90, n)


def _band_factory(fig):
    """region_factory(band_hw) growing a galactic |b| band + one circle."""
    from skyplothelper.plotly import make_compound_region

    def factory(band_hw):
        return (make_compound_region(fig)
                .add_frame_band(-band_hw, band_hw, frame='galactic')
                .add_circle(80.0, -45.0, 8.0))
    return factory


def test_compound_region_states_keys_and_shapes():
    fig = sphpl.make_figure(projection='AIT', center=180, frame='icrs')
    lon, lat = _band_catalog()
    states = sphpl.compound_region_states(
        fig, _band_factory(fig), [{'band_hw': 10}, {'band_hw': 30}],
        (lon, lat))
    assert len(states) == 2
    for st in states:
        assert set(st) == {'params', 'fill_path', 'outline_path', 'outline_x',
                           'outline_y', 'contains', 'contains_int',
                           'n_inside', 'n_outside'}
        assert st['fill_path'].startswith('M')
        assert st['outline_path'].startswith('M')
        assert st['contains'].shape == (600,)
        assert st['n_inside'] + st['n_outside'] == 600


def test_compound_region_states_containment_grows_with_band():
    """A widening galactic band swallows monotonically more sources."""
    fig = sphpl.make_figure(projection='AIT', center=180, frame='icrs')
    lon, lat = _band_catalog()
    states = sphpl.compound_region_states(
        fig, _band_factory(fig),
        [{'band_hw': hw} for hw in (5, 15, 30, 50)], (lon, lat))
    counts = [st['n_inside'] for st in states]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]     # it actually changed


def test_compound_region_states_payload_is_binary_not_hex():
    """contains_int ships 0/1, not per-source hex — the payload halving."""
    fig = sphpl.make_figure(projection='AIT', center=180)
    lon, lat = _band_catalog()
    st = sphpl.compound_region_states(
        fig, _band_factory(fig), [{'band_hw': 20}], (lon, lat))[0]
    assert set(st['contains_int']) <= {0, 1}
    assert all(isinstance(v, int) for v in st['contains_int'])


def test_compound_region_states_no_catalog_leaves_containment_none():
    fig = sphpl.make_figure(projection='AIT', center=180)
    st = sphpl.compound_region_states(
        fig, _band_factory(fig), [{'band_hw': 20}])[0]
    assert st['contains'] is None
    assert st['contains_int'] is None
    assert st['n_inside'] == 0 and st['n_outside'] == 0
    assert st['fill_path'].startswith('M')     # geometry still computed


def test_compound_region_states_rejects_non_callable_factory():
    fig = sphpl.make_figure(projection='AIT', center=180)
    with pytest.raises(TypeError, match='callable'):
        sphpl.compound_region_states(fig, object(), [{'band_hw': 10}])


def test_compound_region_states_multipiece_is_single_path():
    """A union that splits into disjoint lobes is one fill path (subpaths),
    so a slider swaps one shapes[i].path — not a varying number of shapes."""
    from skyplothelper.plotly import make_compound_region
    fig = sphpl.make_figure(projection='AIT', center=180)

    def two_blobs(sep):
        return (make_compound_region(fig)
                .add_circle(90.0 - sep, 0.0, 10.0)
                .add_circle(90.0 + sep, 0.0, 10.0))

    st = sphpl.compound_region_states(fig, two_blobs, [{'sep': 40}])[0]
    # Two disjoint circles -> a MultiPolygon -> two 'M' subpaths, one string.
    assert st['fill_path'].count('M') == 2


def test_add_region_slider_builds_shapes_trace_and_steps():
    fig = sphpl.make_figure(projection='AIT', center=180, frame='icrs')
    lon, lat = _band_catalog()
    n_shapes0 = len(fig.layout.shapes or [])
    n_data0 = len(fig.data)
    params = [{'band_hw': hw} for hw in (5, 15, 30)]
    states = sphpl.add_region_slider(
        fig, _band_factory(fig), params, (lon, lat), name='cat', active=1)
    assert len(states) == 3
    assert len(fig.layout.shapes) == n_shapes0 + 2      # fill + outline
    assert len(fig.data) == n_data0 + 1                 # one marker trace
    steps = fig.layout.sliders[0].steps
    assert len(steps) == 3
    assert fig.layout.sliders[0].active == 1
    # Active marker colors match the active step's containment.
    assert list(fig.data[-1].marker.color) == states[1]['contains_int']


def test_add_region_slider_step_updates_two_paths_and_marker():
    fig = sphpl.make_figure(projection='AIT', center=180)
    lon, lat = _band_catalog()
    sphpl.add_region_slider(
        fig, _band_factory(fig), [{'band_hw': 10}, {'band_hw': 40}],
        (lon, lat))
    restyle, relayout, tidx = fig.layout.sliders[0].steps[0].args
    assert 'marker.color' in restyle
    path_keys = [k for k in relayout if k.endswith('.path')]
    assert len(path_keys) == 2                          # fill + outline
    assert len(tidx) == 1                               # the catalog trace


def test_add_region_slider_no_catalog_step_has_no_trace_slot():
    fig = sphpl.make_figure(projection='AIT', center=180)
    n_data0 = len(fig.data)
    sphpl.add_region_slider(
        fig, _band_factory(fig), [{'band_hw': 10}, {'band_hw': 40}])
    assert len(fig.data) == n_data0                     # no marker trace
    args = fig.layout.sliders[0].steps[0].args
    assert len(args) == 2                               # [restyle, relayout]


def test_add_region_slider_active_out_of_range_raises():
    fig = sphpl.make_figure(projection='AIT', center=180)
    with pytest.raises(ValueError, match='active'):
        sphpl.add_region_slider(fig, _band_factory(fig),
                                [{'band_hw': 10}], active=5)


def test_add_region_slider_empty_params_raises():
    fig = sphpl.make_figure(projection='AIT', center=180)
    with pytest.raises(ValueError, match='non-empty'):
        sphpl.add_region_slider(fig, _band_factory(fig), [])


def test_add_region_slider_label_format_callable_and_string():
    fig = sphpl.make_figure(projection='AIT', center=180)
    lon, lat = _band_catalog()
    sphpl.add_region_slider(
        fig, _band_factory(fig), [{'band_hw': 10}, {'band_hw': 40}],
        (lon, lat), label_format='{n_inside} in')
    labels = [s.label for s in fig.layout.sliders[0].steps]
    assert all(lab.endswith(' in') for lab in labels)

    fig2 = sphpl.make_figure(projection='AIT', center=180)
    sphpl.add_region_slider(
        fig2, _band_factory(fig2), [{'band_hw': 10}],
        label_format=lambda st: f"hw={st['params']['band_hw']}")
    assert fig2.layout.sliders[0].steps[0].label == 'hw=10'


def test_add_region_slider_default_label_is_kv_params():
    fig = sphpl.make_figure(projection='AIT', center=180)
    sphpl.add_region_slider(fig, _band_factory(fig), [{'band_hw': 10}])
    assert fig.layout.sliders[0].steps[0].label == 'band_hw=10'


def test_region_slider_catalog_skycoord_respects_frame():
    """A galactic-frame SkyCoord catalog is classified in its own frame, so a
    thin galactic band around b=0 captures the low-|b| sources."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    fig = sphpl.make_figure(projection='AIT', center=180, frame='galactic')
    rng = np.random.default_rng(3)
    gl = rng.uniform(0, 360, 400)
    gb = rng.uniform(-90, 90, 400)
    cat = SkyCoord(gl * u.deg, gb * u.deg, frame='galactic')
    from skyplothelper.plotly import make_compound_region

    def factory(band_hw):
        return make_compound_region(fig).add_frame_band(
            -band_hw, band_hw, frame='galactic')

    st = sphpl.compound_region_states(fig, factory, [{'band_hw': 10}], cat)[0]
    # Everyone inside should have |b| within a small margin of the band.
    inside_b = np.abs(gb[st['contains']])
    assert inside_b.max() <= 12.0        # 10 deg band + projection margin


def test_region_outline_path_is_open_polylines():
    """The outline path strokes open polylines (no Z), unlike the fill."""
    from skyplothelper.plotly import make_compound_region
    from skyplothelper.plotly.core import _region_outline_path, _shapely_to_svg_path
    fig = sphpl.make_figure(projection='AIT', center=180)
    region = make_compound_region(fig).add_circle(90.0, 0.0, 20.0)
    frame_poly = region.projector.frame_polygon
    outline = _region_outline_path(region._geom, frame_poly)
    fill = _shapely_to_svg_path(region._geom)
    assert 'Z' not in outline            # open strokes
    assert 'Z' in fill                   # closed fill


# ---- dash_region (optional) ------------------------------------------------

def test_dash_region_symbols_for_maps_categories():
    from skyplothelper.plotly import dash_region
    syms = dash_region._symbols_for(['a', 'b', 'a', 'b', 'c'])
    assert syms[0] == syms[2]            # same category -> same symbol
    assert syms[0] != syms[1]            # distinct categories differ
    assert len(set(syms[:3])) == 2
    assert dash_region._symbols_for(None) is None


def test_dash_region_slider_id():
    from skyplothelper.plotly import dash_region
    assert dash_region._slider_id('g', 'band_hw') == 'g-slider-band_hw'


def test_region_explorer_app_builds_with_callback():
    pytest.importorskip("dash")
    import dash

    from skyplothelper.plotly import dash_region, make_compound_region
    lon, lat = _band_catalog(n=300)

    def factory(fig, band_hw):
        return make_compound_region(fig).add_frame_band(
            -band_hw, band_hw, frame='galactic')

    app = dash_region.region_explorer_app(
        (lon, lat), factory, params={'band_hw': (2.0, 40.0, 2.0)},
        projection='AIT', center=180, frame='icrs', name='cat')
    assert isinstance(app, dash.Dash)
    assert 'sph-region-graph-slider-band_hw' in str(app.layout)
    assert len(app.callback_map) == 1


def test_region_explorer_app_draws_region_and_markers():
    pytest.importorskip("dash")
    from skyplothelper.plotly import dash_region, make_compound_region
    lon, lat = _band_catalog(n=300)

    def factory(fig, band_hw):
        return make_compound_region(fig).add_frame_band(
            -band_hw, band_hw, frame='galactic')

    app = dash_region.region_explorer_app(
        (lon, lat), factory, params={'band_hw': (2.0, 40.0, 2.0)})
    # Reach into the graph the app built.
    graph_fig = None
    for child in app.layout.children:
        if getattr(child, 'id', None) == 'sph-region-graph':
            graph_fig = child.figure
    assert graph_fig is not None
    assert len(graph_fig.layout.shapes) == 2         # fill + outline
    marker = [t for t in graph_fig.data
              if t.mode == 'markers' and t.marker.color is not None]
    assert marker and set(marker[-1].marker.color) <= {0, 1}
