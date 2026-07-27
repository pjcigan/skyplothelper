"""Tests for the plotly FITS-image viewer (skyplothelper.plotly.fits).

Self-contained: builds tiny synthetic astropy WCS objects (no FITS file, no
matplotlib figure), so the suite never depends on the example MOJAVE image.
Visual eyeballing lives in the gitignored HTML gallery
(tests/integration/visual/render_fits.py), not pytest-mpl — a plotly
Heatmap is not a matplotlib artifact.
"""

import numpy as np
import pytest

plotly = pytest.importorskip("plotly")
pytest.importorskip("shapely")
from astropy.wcs import WCS  # noqa: E402

from skyplothelper import plotly as sphpl  # noqa: E402
from skyplothelper.plotly.projector import WCSPixelProjector  # noqa: E402


def _toy_wcs(n=20, crval=(150.0, 2.0), cdelt_arcsec=1.0, ctype='TAN',
             ndim=2):
    """A small RA/Dec WCS with a normal RA flip (negative CDELT1)."""
    naxis = 4 if ndim == 4 else 2
    w = WCS(naxis=naxis)
    if naxis == 4:
        w.wcs.crpix = [n / 2, n / 2, 1, 1]
        w.wcs.cdelt = [-cdelt_arcsec / 3600, cdelt_arcsec / 3600, 1, 1]
        w.wcs.crval = [crval[0], crval[1], 0, 0]
        w.wcs.ctype = [f'RA---{ctype}', f'DEC--{ctype}', 'FREQ', 'STOKES']
    else:
        w.wcs.crpix = [n / 2, n / 2]
        w.wcs.cdelt = [-cdelt_arcsec / 3600, cdelt_arcsec / 3600]
        w.wcs.crval = list(crval)
        w.wcs.ctype = [f'RA---{ctype}', f'DEC--{ctype}']
    return w


# --- WCSPixelProjector ------------------------------------------------------

def test_projector_crval_maps_to_crpix_zero_based():
    w = _toy_wcs(n=20, crval=(150.0, 2.0))
    proj = WCSPixelProjector(w, (20, 20))
    x, y = proj._project_xy(150.0, 2.0)
    # crpix is 1-based (10) → 0-based pixel 9.
    assert float(np.ravel(x)[0]) == pytest.approx(9.0, abs=1e-6)
    assert float(np.ravel(y)[0]) == pytest.approx(9.0, abs=1e-6)


@pytest.mark.parametrize("ctype", ["TAN", "SIN"])
def test_projector_region_projects_to_pixels(ctype):
    """A small circle around the reference projects to a pixel-space polygon
    of the expected area — works for TAN and SIN."""
    from skyplothelper import CompoundRegion
    cdelt = 1.0  # arcsec/pix
    w = _toy_wcs(n=40, crval=(150.0, 2.0), cdelt_arcsec=cdelt, ctype=ctype)
    proj = WCSPixelProjector(w, (40, 40))
    radius_deg = 5.0 / 3600.0  # 5 arcsec → 5 px radius
    region = CompoundRegion(proj).add_circle(150.0, 2.0, radius_deg)
    assert region._geom.geom_type == 'Polygon'
    assert region._geom.area == pytest.approx(np.pi * 25.0, rel=0.05)


def test_projector_handles_degenerate_4d_wcs():
    w4 = _toy_wcs(n=20, ndim=4)
    proj = WCSPixelProjector(w4, (20, 20))
    assert proj.wcs.naxis == 2
    x, y = proj._project_xy(150.0, 2.0)
    assert np.isfinite(float(np.ravel(x)[0]))


# --- add_fits_image ---------------------------------------------------------

def _image(n=20):
    return np.arange(n * n, dtype=float).reshape(n, n)


def test_add_fits_image_basic_trace_and_meta():
    w = _toy_wcs(n=20)
    fig = sphpl.make_fits_figure(w)
    tr = sphpl.add_fits_image(fig, _image(20), w, coords='absolute')
    assert tr.type == 'heatmap'
    assert np.asarray(tr.z).shape == (20, 20)
    # Square-aspect axes, no y-reversal.
    assert fig.layout.xaxis.scaleanchor == 'y'
    assert fig.layout.xaxis.scaleratio == 1
    assert fig.layout.yaxis.autorange not in ('reversed',)
    meta = dict(fig.layout.meta)
    assert meta['sph_fits'] is True
    assert meta['sph_coords'] == 'absolute'
    assert meta['sph_image_shape'] == [20, 20]
    assert 'sph_wcs_header' in meta


def test_add_fits_image_orientation_no_flip():
    """A bright pixel at data[2, 5] must render at heatmap (x=5, y=2)."""
    w = _toy_wcs(n=16)
    data = np.zeros((16, 16))
    data[2, 5] = 100.0
    fig = sphpl.make_fits_figure(w)
    tr = sphpl.add_fits_image(fig, data, w, max_pixels=10_000_000)
    z = np.asarray(tr.z)
    iy, ix = np.unravel_index(np.nanargmax(z), z.shape)
    assert (float(tr.x[ix]), float(tr.y[iy])) == (5.0, 2.0)


def test_add_fits_image_nan_stays_transparent():
    w = _toy_wcs(n=10)
    data = _image(10)
    data[0, 0] = np.nan
    fig = sphpl.make_fits_figure(w)
    tr = sphpl.add_fits_image(fig, data, w, max_pixels=10_000_000)
    assert np.isnan(np.asarray(tr.z, dtype=float)[0, 0])


def test_add_fits_image_squeezes_degenerate_axes():
    w = _toy_wcs(n=12, ndim=4)
    data = _image(12)[None, None, :, :]  # (1,1,12,12)
    fig = sphpl.make_fits_figure(w)
    tr = sphpl.add_fits_image(fig, data, w)
    assert np.asarray(tr.z).shape == (12, 12)


def test_add_fits_image_hover_full_includes_radec_value_bunit():
    w = _toy_wcs(n=12)
    fig = sphpl.make_fits_figure(w)
    tr = sphpl.add_fits_image(fig, _image(12), w, hover='full',
                              bunit='Jy/beam')
    assert 'RA:' in tr.hovertemplate and 'Dec:' in tr.hovertemplate
    assert 'Jy/beam' in tr.hovertemplate
    assert np.asarray(tr.customdata).shape == (12, 12, 3)


def test_add_fits_image_bunit_from_header():
    w = _toy_wcs(n=12)
    fig = sphpl.make_fits_figure(w)
    tr = sphpl.add_fits_image(fig, _image(12), w,
                              header={'BUNIT': 'JY/BEAM'})
    assert 'JY/BEAM' in tr.hovertemplate


def test_add_fits_image_display_factor_scales_colorbar():
    w = _toy_wcs(n=12)
    fig = sphpl.make_fits_figure(w)
    tr = sphpl.add_fits_image(fig, _image(12), w, colorbar=True,
                              display_factor=1000.0, bunit='mJy/beam')
    assert tr.colorbar.title.text == 'mJy/beam'
    vals = [float(t) for t in tr.colorbar.ticktext]
    assert max(vals) > 1e4   # raw max ~143 × 1000 ≈ 1.4e5


def test_add_fits_image_offset_axes_in_offset_units():
    """coords='offset' displays the image in offset units with native axes:
    x/y in arcsec, reference at (0,0), east-positive x reversed (east left)."""
    w = _toy_wcs(n=40, crval=(150.0, 2.0), cdelt_arcsec=1.0)
    fig = sphpl.make_fits_figure(w)
    tr = sphpl.add_fits_image(fig, _image(40), w, coords='offset',
                              offset_units='arcsec')
    assert 'Relative RA' in fig.layout.xaxis.title.text
    # Heatmap x spans ~±20 arcsec (40 px × 1 arcsec/px / 2), not pixel 0..39.
    assert abs(float(np.max(tr.x))) > 15
    # x-axis range is reversed (range[0] > range[1]) so east is on the left.
    xr = fig.layout.xaxis.range
    assert float(xr[0]) > float(xr[1])
    # No custom ticktext — native numeric ticks (round + zoom-adaptive).
    assert fig.layout.xaxis.ticktext in (None, ())


def test_offset_overlays_align_with_reference():
    """In offset mode a marker at the reference lands at (0,0), and 1 arcsec
    east lands at +1 arcsec (cos-dec folded in)."""
    ra0, dec0 = 150.0, 30.0
    w = _toy_wcs(n=40, crval=(ra0, dec0), cdelt_arcsec=1.0)
    fig = sphpl.make_fits_figure(w)
    sphpl.add_fits_image(fig, _image(40), w, coords='offset',
                         offset_units='arcsec')
    sc = sphpl.add_fits_scatter(fig, [ra0], [dec0])
    assert float(sc.x[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(sc.y[0]) == pytest.approx(0.0, abs=1e-6)
    ra_e = ra0 + (1.0 / 3600.0) / np.cos(np.radians(dec0))
    sc2 = sphpl.add_fits_scatter(fig, [ra_e], [dec0])
    assert float(sc2.x[0]) == pytest.approx(1.0, abs=0.02)


def test_offset_compound_region_uses_offset_projector():
    from skyplothelper.plotly.projector import WCSOffsetProjector
    w = _toy_wcs(n=40, crval=(150.0, 2.0), cdelt_arcsec=1.0)
    fig = sphpl.make_fits_figure(w)
    sphpl.add_fits_image(fig, _image(40), w, coords='offset',
                         offset_units='arcsec')
    region = sphpl.make_fits_compound_region(fig)
    assert isinstance(region.projector, WCSOffsetProjector)
    region.add_circle(150.0, 2.0, 5.0 / 3600)
    # 5 arcsec radius → ~5 arcsec-unit circle → area ~pi*25.
    assert region._geom.area == pytest.approx(np.pi * 25.0, rel=0.05)


def test_add_fits_image_rejects_bad_coords():
    w = _toy_wcs()
    with pytest.raises(ValueError, match="coords"):
        sphpl.add_fits_image(sphpl.make_fits_figure(w), _image(), w,
                             coords='galactic')


# --- ticks ------------------------------------------------------------------

def test_fits_ticks_offset_scales_from_reference():
    """Offset labels equal |pixel - ref_pixel| × pixel_scale, so they read ~0
    at the reference and grow linearly away from it (dec≈0 → cos≈1)."""
    cdelt, ref_px = 1.0, 19.0   # crpix 20 (1-based) → 0-based 19
    w = _toy_wcs(n=40, crval=(150.0, 0.0), cdelt_arcsec=cdelt)
    out = sphpl.fits_ticks_for_range(w, (-0.5, 39.5), (-0.5, 39.5),
                                     coords='offset', ref_coord=(150.0, 0.0),
                                     offset_units='arcsec', precision=2)
    for tv, tt in zip(out['xaxis']['tickvals'], out['xaxis']['ticktext']):
        expected = abs(tv - ref_px) * cdelt
        assert abs(abs(float(tt.split()[0])) - expected) < 0.1


def test_fits_ticks_offset_ra_uses_cos_dec():
    """The displayed RA offset is the true on-sky angle = pixel_step ×
    pixel_scale. Validates the cos(dec) factor: without it, the RA-coordinate
    offset at dec=60° would read ~2× too large."""
    dec, cdelt = 60.0, 1.0  # arcsec/pix
    ref_px = 19.0           # crpix 20 (1-based) → 0-based pixel 19
    w = _toy_wcs(n=40, crval=(150.0, dec), cdelt_arcsec=cdelt)
    out = sphpl.fits_ticks_for_range(w, (-0.5, 39.5), (19.0, 20.0),
                                     coords='offset', ref_coord=(150.0, dec),
                                     offset_units='arcsec', precision=2)
    for tv, tt in zip(out['xaxis']['tickvals'], out['xaxis']['ticktext']):
        expected = abs(tv - ref_px) * cdelt   # true on-sky arcsec
        assert abs(abs(float(tt.split()[0])) - expected) < 0.5


def test_fits_ticks_absolute_returns_degree_strings():
    w = _toy_wcs(n=40)
    out = sphpl.fits_ticks_for_range(w, (-0.5, 39.5), (-0.5, 39.5),
                                     coords='absolute')
    assert all(t.endswith('°') for t in out['xaxis']['ticktext'])


# --- beam -------------------------------------------------------------------

def test_beam_shape_semi_major_pixels():
    cdelt = 0.5  # arcsec/pix
    w = _toy_wcs(n=100, cdelt_arcsec=cdelt)
    bmaj = 4.0  # arcsec → semi-major 2 arcsec → 4 px
    shape = sphpl.beam_shape_for_range(w, (0, 100), (0, 100),
                                       bmaj_arcsec=bmaj, bmin_arcsec=bmaj,
                                       bpa_deg=0.0)
    assert shape['type'] == 'path'
    # Parse the path points; max radius from centroid ≈ semi-major (4 px).
    xy = _path_points(shape['path'])
    c = xy.mean(axis=0)
    rmax = np.max(np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]))
    assert rmax == pytest.approx(4.0, rel=0.1)


def _path_points(path):
    pts = [t.split(',') for t in path.split() if ',' in t]
    return np.array([[float(a), float(b)] for a, b in pts])


def test_beam_repins_to_view_corner():
    w = _toy_wcs(n=100, cdelt_arcsec=0.5)
    s1 = sphpl.beam_shape_for_range(w, (0, 100), (0, 100), bmaj_arcsec=4,
                                    bmin_arcsec=4, corner='lower left')
    s2 = sphpl.beam_shape_for_range(w, (40, 60), (40, 60), bmaj_arcsec=4,
                                    bmin_arcsec=4, corner='lower left')
    # Centroids differ when the view changes (corner-pegged to current range).
    c1 = _path_points(s1['path']).mean(axis=0)
    c2 = _path_points(s2['path']).mean(axis=0)
    assert abs(c1[0] - c2[0]) > 5


# --- scatter overlay --------------------------------------------------------

def test_add_fits_scatter_projects_via_wcs():
    w = _toy_wcs(n=20, crval=(150.0, 2.0))
    fig = sphpl.make_fits_figure(w)
    sphpl.add_fits_image(fig, _image(20), w)
    tr = sphpl.add_fits_scatter(fig, [150.0], [2.0],
                                marker=dict(color='lime'))
    assert float(tr.x[0]) == pytest.approx(9.0, abs=1e-6)
    assert float(tr.y[0]) == pytest.approx(9.0, abs=1e-6)


def test_add_fits_scatter_explicit_wcs_on_bare_figure():
    """With an explicit wcs= a bare figure still projects to pixels."""
    w = _toy_wcs(n=20, crval=(150.0, 2.0))
    fig = sphpl.make_figure(show_grid=False)
    tr = sphpl.add_fits_scatter(fig, [150.0], [2.0], wcs=w)
    assert float(tr.x[0]) == pytest.approx(9.0, abs=1e-6)


def test_make_fits_compound_region_roundtrips_projector():
    w = _toy_wcs(n=40, crval=(150.0, 2.0))
    fig = sphpl.make_fits_figure(w)
    sphpl.add_fits_image(fig, _image(40), w)
    region = sphpl.make_fits_compound_region(fig).add_circle(150.0, 2.0,
                                                             5.0 / 3600)
    assert region._geom is not None and region._geom.area > 0
    shape, _ = sphpl.add_compound_region(fig, region)
    assert shape.type == 'path'


# --- Dash convenience layer (Phase C) --------------------------------------

def test_ranges_from_relayout_parses_events():
    """The pure range parser handles zoom, list-form, autorange reset, partial
    updates, and empty events — no Dash needed."""
    from skyplothelper.plotly.dash_fits import _ranges_from_relayout
    dx, dy = [0, 100], [0, 100]
    # explicit zoom
    assert _ranges_from_relayout(
        {'xaxis.range[0]': 10, 'xaxis.range[1]': 40,
         'yaxis.range[0]': 20, 'yaxis.range[1]': 50}, dx, dy) == ([10.0, 40.0],
                                                                  [20.0, 50.0])
    # list form
    assert _ranges_from_relayout({'xaxis.range': [5, 15]}, dx, dy)[0] == [5.0,
                                                                          15.0]
    # autorange reset → defaults
    assert _ranges_from_relayout({'xaxis.autorange': True}, dx, dy)[0] == [0,
                                                                           100]
    # partial (only x changed) → y stays default
    xr, yr = _ranges_from_relayout({'xaxis.range[0]': 1, 'xaxis.range[1]': 2},
                                   dx, dy)
    assert xr == [1.0, 2.0] and yr == [0, 100]
    # empty
    assert _ranges_from_relayout(None, dx, dy) == ([0, 100], [0, 100])


def test_fits_viewer_app_builds_with_callback():
    pytest.importorskip("dash")
    import dash

    from skyplothelper.plotly import dash_fits
    w = _toy_wcs(n=40, crval=(150.0, 2.0))
    app = dash_fits.fits_viewer_app(_image(40), w, coords='absolute',
                                    graph_id='g')
    assert isinstance(app, dash.Dash)
    assert 'g' in str(app.layout)
    assert len(app.callback_map) == 1


def test_register_fits_relayout_returns_callback():
    pytest.importorskip("dash")
    import dash

    from skyplothelper.plotly import dash_fits
    w = _toy_wcs(n=40, crval=(150.0, 2.0))
    fig = sphpl.make_fits_figure(w)
    sphpl.add_fits_image(fig, _image(40), w)
    app = dash.Dash(__name__)
    from dash import dcc, html
    app.layout = html.Div([dcc.Graph(id='g', figure=fig)])
    cb = dash_fits.register_fits_relayout(
        app, 'g', w, coords='absolute', default_xrange=[0, 40],
        default_yrange=[0, 40])
    assert callable(cb)
    assert len(app.callback_map) == 1


# --- standalone overlay reprojection onto FITS axes -------------------------
#
# The all-sky overlay helpers (add_great_circle / add_geodesic_circle /
# add_ruler / add_sky_vectors / add_healpix_sparse) detect a FITS figure and
# project through its WCS projector instead of the all-sky projection, so they
# land in the image's pixel space. These check that routing.

def _fits_fig(n=200, crval=(150.0, 2.0), cdelt_arcsec=20.0, coords='absolute'):
    """A FITS figure whose image is ``n×n`` pixels at ``cdelt_arcsec``/pix."""
    w = _toy_wcs(n=n, crval=crval, cdelt_arcsec=cdelt_arcsec)
    fig = sphpl.make_fits_figure(w)
    sphpl.add_fits_image(fig, _image(n), w, coords=coords)
    return fig, w, n


def _finite_xy(traces):
    traces = traces if isinstance(traces, (list, tuple)) else [traces]
    xs, ys = [], []
    for t in traces:
        if hasattr(t, 'x') and t.x is not None:
            xs.append(np.atleast_1d(np.asarray(t.x, dtype=float)))
            ys.append(np.atleast_1d(np.asarray(t.y, dtype=float)))
    x = np.concatenate(xs) if xs else np.array([])
    y = np.concatenate(ys) if ys else np.array([])
    return x, y


def test_fits_geodesic_circle_lands_in_pixels():
    """A small geodesic circle about the reference projects to a ring of
    pixels centered on (roughly) the image center, fully on-image (no NaN)."""
    fig, w, n = _fits_fig(n=200, cdelt_arcsec=20.0)
    ra0, dec0 = 150.0, 2.0
    # radius 200 arcsec = 10 px at 20 arcsec/pix.
    tr = sphpl.add_geodesic_circle(fig, ra0, dec0, 200.0 / 3600.0, fill=True)
    x, y = _finite_xy(tr)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    # Ring sits in pixel space around the center pixel (~99).
    assert x.min() > 80 and x.max() < 120
    assert y.min() > 80 and y.max() < 120


def test_fits_healpix_tile_lands_in_pixels():
    hp = pytest.importorskip("healpy")
    fig, w, n = _fits_fig(n=400, cdelt_arcsec=30.0)
    ra0, dec0 = 150.0, 2.0
    ipix = hp.ang2pix(64, ra0, dec0, lonlat=True)
    tr = sphpl.add_healpix_sparse(fig, [ipix], [1.0], 64)
    x, y = _finite_xy(tr)
    assert x.size > 0 and np.isfinite(x).all()
    # The tile straddles the center pixel (~199).
    assert x.min() < 200 < x.max()


def test_fits_ruler_main_line_in_pixels():
    fig, w, n = _fits_fig(n=200, cdelt_arcsec=20.0)
    ra0, dec0 = 150.0, 2.0
    tr = sphpl.add_ruler(fig, ra0 - 0.02, dec0, ra0 + 0.02, dec0)
    x, y = _finite_xy(tr)
    assert x.size > 0
    finite = x[np.isfinite(x)]
    assert finite.min() > 0 and finite.max() < n


def test_fits_sky_vectors_in_pixels_no_spurious_split():
    """A small vector at the reference projects to a short shaft near the
    center; on a seamless FITS frame it never takes the wrap-straddle path,
    so the only NaN is the inter-arrow separator."""
    fig, w, n = _fits_fig(n=200, cdelt_arcsec=20.0)
    ra0, dec0 = 150.0, 2.0
    tr = sphpl.add_sky_vectors(fig, [ra0], [dec0], [0.02], [0.02])
    x, y = _finite_xy(tr)
    finite = x[np.isfinite(x)]
    assert finite.size >= 2
    assert finite.min() > 50 and finite.max() < 150


def test_fits_great_circle_far_hemisphere_is_nan_not_streak():
    """A full-sky great circle through the field keeps its near arc in pixel
    space but breaks (NaN) the far hemisphere, which a TAN WCS would
    otherwise diverge to huge pixel values."""
    fig, w, n = _fits_fig(n=200, cdelt_arcsec=20.0)
    ra0, dec0 = 150.0, 2.0
    tr = sphpl.add_great_circle(fig, frame='pole', pole_lon=ra0,
                                pole_lat=dec0 - 90.0, lat_offset=0.0)
    x, y = _finite_xy(tr)
    assert not np.isfinite(x).all()          # far side is broken out
    finite = x[np.isfinite(x)]
    assert finite.size > 0
    # No blow-up: finite coords stay within a frame-width margin of the image.
    assert finite.min() > -n and finite.max() < 2 * n


def test_fits_overlay_matches_fits_scatter_projection():
    """An overlay helper and add_fits_scatter project the same sky point to
    the same pixel — i.e. the overlay seam uses the image's WCS projector."""
    fig, w, n = _fits_fig(n=200, cdelt_arcsec=20.0)
    ra0, dec0 = 150.02, 2.01
    sc = sphpl.add_fits_scatter(fig, [ra0], [dec0])
    # A zero-length ruler's endpoint must coincide with the scatter pixel.
    rl = sphpl.add_ruler(fig, ra0, dec0, ra0, dec0, n_ticks=0)
    rx, ry = _finite_xy(rl)
    assert float(np.asarray(sc.x)[0]) == pytest.approx(float(rx[0]), abs=1e-6)
    assert float(np.asarray(sc.y)[0]) == pytest.approx(float(ry[0]), abs=1e-6)
