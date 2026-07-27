"""Tests for skyplothelper.projections (registry + frames + math)."""

import pytest
from astropy.visualization.wcsaxes.frame import EllipticalFrame

from skyplothelper.projections import _math, frames
from skyplothelper.projections.registry import (
    _PROJECTION_ALIASES,
    _PROJECTION_REGISTRY,
    _resolve_projection,
    get_frame_class,
    list_projections,
)

# ---- _resolve_projection: alias normalization ----

@pytest.mark.parametrize("alias, expected_key", [
    # FITS code (case-insensitive)
    ("AIT", "ait"), ("ait", "ait"), ("Ait", "ait"),
    # Hyphen / underscore tolerance
    ("hammer-aitoff", "ait"), ("hammer_aitoff", "ait"),
    ("hammeraitoff", "ait"),  # without separator
    ("plate-carree", "car"), ("plate_carree", "car"), ("platecarree", "car"),
    # Common human aliases
    ("mollweide", "mol"),
    ("orthographic", "sin"), ("ortho", "sin"),
    ("gnomonic", "tan"), ("tangent", "tan"),
    ("stereographic", "stg"),
    # Non-FITS
    ("robinson", "robinson"),
    ("eckert4", "eckert_iv"), ("eckert_4", "eckert_iv"), ("eckert", "eckert_iv"),
    ("kavrayskiy_vii", "kavrayskiy"), ("kavrayskiy7", "kavrayskiy"),
    ("winkel", "winkel_tripel"), ("winkeltripel", "winkel_tripel"),
    ("mcbryde_thomas", "mcbryde"),
])
def test_resolve_projection_aliases(alias, expected_key):
    key, info = _resolve_projection(alias)
    assert key == expected_key


def test_resolve_projection_unknown_raises():
    with pytest.raises(ValueError, match="Unknown projection"):
        _resolve_projection("not_a_real_projection_name")


# ---- list_projections ----

def test_list_projections_as_table_returns_list():
    rows = list_projections(as_table=True)
    assert isinstance(rows, list)
    assert len(rows) == len(_PROJECTION_REGISTRY)
    assert all("description" in row for row in rows)


def test_list_projections_filter_allsky():
    allsky_rows = list_projections(allsky=True, as_table=True)
    nonallsky_rows = list_projections(allsky=False, as_table=True)
    assert len(allsky_rows) > 0
    assert len(nonallsky_rows) > 0
    assert len(allsky_rows) + len(nonallsky_rows) == len(_PROJECTION_REGISTRY)


def test_list_projections_filter_fits_only():
    rows = list_projections(fits_only=True, as_table=True)
    # Non-FITS entries should be filtered out
    assert all(row["code"] != "-" for row in rows)


def test_list_projections_filter_shape():
    elliptical = list_projections(shape="elliptical", as_table=True)
    assert all(row["shape"] == "elliptical" for row in elliptical)
    # AIT and MOL are elliptical
    keys = {row["key"] for row in elliptical}
    assert "ait" in keys
    assert "mol" in keys


# ---- get_frame_class ----

@pytest.mark.parametrize("name, expected_cls", [
    ("AIT", EllipticalFrame),
    ("ait", EllipticalFrame),
    ("hammer-aitoff", EllipticalFrame),
    ("MOL", EllipticalFrame),
    ("SIN", frames.CircularFrame),
    ("orthographic", frames.CircularFrame),
    ("SFL", frames.SinusoidalFrame),
    ("PAR", frames.ParabolicFrame),
    ("robinson", frames.RobinsonFrame),
    ("kavrayskiy", frames.KavrayskiyFrame),
    ("eckert_iv", frames.Eckert4Frame),
    ("winkel_tripel", frames.WinkelTripelFrame),
    ("mcbryde", frames.McBrydeFrame),
])
def test_get_frame_class(name, expected_cls):
    assert get_frame_class(name) is expected_cls


def test_get_frame_class_rectangular_returns_none():
    """Rectangular projections (CAR, MER, etc.) return None — astropy default."""
    assert get_frame_class("CAR") is None
    assert get_frame_class("TAN") is None


def test_get_frame_class_unknown_returns_none():
    """Unknown projection names should return None gracefully (not raise)."""
    assert get_frame_class("not_a_projection") is None


# ---- Projection math round-trips ----

@pytest.mark.parametrize("forward, inverse", [
    (_math._robinson_forward, _math._robinson_inverse),
    (_math._kavrayskiy_forward, _math._kavrayskiy_inverse),
    (_math._eckert4_forward, _math._eckert4_inverse),
    (_math._mcbryde_forward, _math._mcbryde_inverse),
])
def test_projection_math_roundtrip(forward, inverse):
    """For a set of test points, forward then inverse should round-trip."""
    test_points = [(0.0, 0.0), (45.0, 30.0), (-90.0, -45.0), (179.0, 89.0)]
    for lon, lat in test_points:
        x, y = forward(lon, lat)
        lon_back, lat_back = inverse(x, y)
        assert abs(float(lon_back) - lon) < 1e-3, f"lon: {lon} -> {lon_back}"
        assert abs(float(lat_back) - lat) < 1e-3, f"lat: {lat} -> {lat_back}"


def test_winkel_roundtrip_loose():
    """Winkel uses iterative inverse; allow looser tolerance."""
    for lon, lat in [(45.0, 30.0), (-60.0, -20.0), (120.0, 50.0)]:
        x, y = _math._winkel_forward(lon, lat)
        lon_back, lat_back = _math._winkel_inverse(x, y)
        assert abs(float(lon_back) - lon) < 1e-2
        assert abs(float(lat_back) - lat) < 1e-2


def test_winkel_emits_no_divide_warning_at_origin():
    """sin(α)/α via np.sinc — projecting a grid that includes (0,0) (where
    α=0) must NOT emit a 'invalid value encountered in divide' RuntimeWarning
    (the np.where form did, by eagerly evaluating 0/0). Guards the terminal
    spam from returning."""
    import warnings
    lon, lat = np.meshgrid(np.linspace(-180, 180, 37),
                           np.linspace(-90, 90, 19))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        x, y = _math._winkel_forward(lon, lat)   # α=0 at (0,0)
        _math._winkel_inverse(x, y)              # iterative inverse
    assert np.isfinite(x).all() and np.isfinite(y).all()


def test_robinson_at_origin_is_origin():
    x, y = _math._robinson_forward(0.0, 0.0)
    assert abs(float(x)) < 1e-9
    assert abs(float(y)) < 1e-9


# ---- Antimeridian seam: forward transforms must keep ±180 on opposite edges ----

def test_wrap_centered_lon_keeps_seam_edges_distinct():
    """A plain modulo collapses +180 onto -180, smearing the seam column of an
    all-sky pcolormesh across the whole panel. The wrap must keep the closed
    [-180, 180] range as identity (both seam edges stay on their own side) while
    still folding genuinely out-of-range longitudes."""
    import numpy as np
    lon = np.array([-180.0, -90.0, 0.0, 90.0, 180.0,  # in-range -> identity
                    181.0, -181.0, 350.0])            # out-of-range -> folded
    out = frames._wrap_centered_lon(lon)
    # In-range values (including BOTH closed endpoints) are unchanged.
    np.testing.assert_allclose(out[:5], [-180.0, -90.0, 0.0, 90.0, 180.0])
    # Out-of-range values fold into [-180, 180].
    np.testing.assert_allclose(out[5:], [-179.0, 179.0, -10.0])


@pytest.mark.skipif(not frames._HAS_CURVEDTRANSFORM,
                    reason="CurvedTransform not available in this astropy")
@pytest.mark.parametrize("transform_cls", [
    frames.RobinsonTransform, frames.WinkelTripelTransform,
    frames.Eckert4Transform, frames.KavrayskiyTransform, frames.McBrydeTransform,
])
def test_forward_transform_does_not_collapse_seam(transform_cls):
    """lon=+180 and lon=-180 are the same meridian but must project to opposite
    edges (regression for the seam-smear bug on the custom compromise
    projections' pcolormesh rendering)."""
    import numpy as np
    t = transform_cls(center_lon=0.0)
    # Equator points at the two seam edges.
    xy = t.transform(np.array([[180.0, 0.0], [-180.0, 0.0]]))
    x_plus, x_minus = xy[0, 0], xy[1, 0]
    assert x_plus > 0 and x_minus < 0, (
        f"{transform_cls.__name__}: +180 -> x={x_plus}, -180 -> x={x_minus}")
    assert x_plus == pytest.approx(-x_minus, abs=1e-9)


# ---- Oblique aspect: spherical rotation for non-FITS center_lat support ----

@pytest.mark.skipif(not frames._HAS_CURVEDTRANSFORM,
                    reason="CurvedTransform not available in this astropy")
def test_oblique_aspect_centers_point_at_origin():
    """The oblique rotation must send the requested center (lon0, lat0) to the
    projection origin (0, 0)."""
    import numpy as np
    t = frames.ObliqueAspectTransform(center_lon=40.0, center_lat=30.0)
    out = t.transform(np.array([[40.0, 30.0]]))
    np.testing.assert_allclose(out[0], [0.0, 0.0], atol=1e-9)


@pytest.mark.skipif(not frames._HAS_CURVEDTRANSFORM,
                    reason="CurvedTransform not available in this astropy")
def test_oblique_aspect_zero_lat_is_pure_lon_shift():
    """With center_lat=0 the rotation reduces to a longitude shift (no lat
    change), so it is safe to skip entirely on the equatorial path."""
    import numpy as np
    t = frames.ObliqueAspectTransform(center_lon=40.0, center_lat=0.0)
    pts = np.array([[10.0, 25.0], [-150.0, -60.0], [200.0, 5.0]])
    out = t.transform(pts)
    # latitude unchanged; longitude shifted by -40 (wrapped to (-180, 180])
    np.testing.assert_allclose(out[:, 1], pts[:, 1], atol=1e-9)
    expected_lon = ((pts[:, 0] - 40.0 + 180.0) % 360.0) - 180.0
    np.testing.assert_allclose(np.sort(out[:, 0]), np.sort(expected_lon), atol=1e-9)


@pytest.mark.skipif(not frames._HAS_CURVEDTRANSFORM,
                    reason="CurvedTransform not available in this astropy")
def test_oblique_aspect_round_trip():
    """Oblique rotation followed by its inverse round-trips to the input."""
    import numpy as np
    t = frames.ObliqueAspectTransform(center_lon=-25.0, center_lat=45.0)
    inv = t.inverted()
    pts = np.array([[10.0, 20.0], [120.0, -35.0], [-80.0, 70.0], [0.0, 0.0]])
    back = inv.transform(t.transform(pts))
    # compare as unit vectors to avoid lon wrap / pole ambiguity artifacts
    def to_vec(a):
        lo, la = np.radians(a[:, 0]), np.radians(a[:, 1])
        return np.column_stack([np.cos(la) * np.cos(lo),
                                np.cos(la) * np.sin(lo), np.sin(la)])
    np.testing.assert_allclose(to_vec(back), to_vec(pts), atol=1e-9)


# ---- Frame infrastructure: precomputed boundary tables ----

def test_winkel_boundary_table_normalized():
    """Boundary should be normalized so equator-X = 1."""
    assert frames._WINKEL_BND_X[0] == pytest.approx(1.0, abs=1e-12)


def test_mcbryde_boundary_table_normalized():
    assert frames._MCBRYDE_BND_X[0] == pytest.approx(1.0, abs=1e-12)


def test_alias_dict_contains_no_unresolved_keys():
    """Every alias must point to a key that's in the main registry."""
    for alias, target in _PROJECTION_ALIASES.items():
        assert target in _PROJECTION_REGISTRY, (
            f"alias '{alias}' points to '{target}' which is not in registry"
        )


# ============================================================================
# project() — shared (lon, lat) → (x, y) primitive
# ============================================================================

import numpy as np  # noqa: E402

from skyplothelper.projections.project import project  # noqa: E402


def test_project_basic_shapes():
    """Output shape matches input shape; scalar-like shapes preserved."""
    lon = np.array([0., 30., 60.])
    lat = np.array([0., 0., 0.])
    x, y = project(lon, lat, projection='AIT')
    assert x.shape == lon.shape
    assert y.shape == lat.shape


def test_project_sky_vs_geographic_flip_signs():
    """Sky vs geographic should give opposite-sign x for off-center lon."""
    lon = np.array([90.])
    lat = np.array([0.])
    x_sky, _ = project(lon, lat, projection='AIT', direction='sky')
    x_geo, _ = project(lon, lat, projection='AIT', direction='geographic')
    assert np.isclose(x_sky[0], -x_geo[0])
    assert x_sky[0] < 0   # sky: positive lon → negative x


@pytest.mark.parametrize("projection",
                          ["AIT", "MOL", "CAR", "SIN", "robinson",
                           "kavrayskiy", "mcbryde", "winkel_tripel",
                           "eckert4"])
def test_project_sky_convention_consistent_across_projections(projection):
    """All supported projections obey 'positive lon → negative x' under
    direction='sky'. The actual magnitude differs per projection."""
    x, _ = project([90.], [0.], projection=projection, direction='sky')
    assert x[0] < 0


def test_project_center_at_lon_yields_origin_at_center():
    """A point AT the projection center maps to (0, 0)."""
    x, y = project([180.], [0.], projection='AIT', center=180)
    assert np.isclose(x[0], 0.0, atol=1e-6)
    assert np.isclose(y[0], 0.0, atol=1e-6)


def test_project_sin_lat_center_works():
    """For zenithal SIN with lat_center=30, the point (0, 30) is the
    projection center → (0, 0)."""
    x, y = project([0.], [30.], projection='SIN', lat_center=30)
    assert np.isclose(x[0], 0.0, atol=1e-6)
    assert np.isclose(y[0], 0.0, atol=1e-6)


def test_project_invalid_direction_raises():
    with pytest.raises(ValueError, match="direction"):
        project([0.], [0.], direction='leftward')


@pytest.mark.parametrize("alias,canonical", [
    ('astro', 'sky'), ('astronomical', 'sky'), ('SKY', 'sky'),
    ('geo', 'geographic'), ('Earth', 'geographic'),
    ('cartographic', 'geographic'),
])
def test_project_direction_aliases(alias, canonical):
    """Intuitive aliases resolve case-insensitively to sky/geographic and
    project identically to the canonical name."""
    from skyplothelper.projections.project import resolve_direction
    assert resolve_direction(alias) == canonical
    xa, _ = project([90.], [0.], projection='AIT', direction=alias)
    xc, _ = project([90.], [0.], projection='AIT', direction=canonical)
    assert np.isclose(xa[0], xc[0])


def test_project_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        project([0., 1., 2.], [0., 1.])


def test_project_frame_transform_galactic_to_icrs():
    """Galactic center (l=0, b=0) → ICRS ~(266.4, -28.9). Project both
    via a (frame='galactic') call and a manually-pre-converted call;
    they should agree."""
    from astropy.coordinates import SkyCoord
    gal_center = SkyCoord(0, 0, unit='deg', frame='galactic')
    icrs = gal_center.icrs
    x_via_frame, y_via_frame = project([0.], [0.], projection='AIT',
                                         frame='galactic')
    x_direct, y_direct = project([icrs.ra.deg], [icrs.dec.deg],
                                  projection='AIT')
    assert np.isclose(x_via_frame[0], x_direct[0], atol=1e-6)
    assert np.isclose(y_via_frame[0], y_direct[0], atol=1e-6)


def test_project_2d_input_preserved():
    """A 2D meshgrid input yields 2D outputs of matching shape."""
    lon = np.linspace(-90, 90, 5)
    lat = np.linspace(-45, 45, 4)
    L, B = np.meshgrid(lon, lat)
    x, y = project(L, B, projection='AIT')
    assert x.shape == L.shape
    assert y.shape == B.shape


def test_project_exported_at_package_root():
    """sph.project is available at the package root."""
    import skyplothelper as sph
    assert hasattr(sph, "project")
    assert sph.project is project


# ---------------------------------------------------------------------------
# project(): conic / Bonne standard parallel (PV2_1)
# ---------------------------------------------------------------------------
#
# COD/COE/COO/COP/BON are undefined without a standard parallel: wcslib
# rejects the header at wcsset time ("ERROR 5"). make_wcs_frame supplied one
# and project() did not, so every conic raised on the backend-agnostic path --
# and with it the whole plotly side, which projects through project().

_CONIC_CODES = ['COD', 'COE', 'COO', 'COP', 'BON']


@pytest.mark.parametrize("projection", _CONIC_CODES)
def test_project_conic_does_not_raise(projection):
    x, y = project([10.], [10.], projection=projection, center=0.0)
    assert np.isfinite(x[0]) and np.isfinite(y[0])


@pytest.mark.parametrize("projection", _CONIC_CODES)
def test_project_conic_matches_make_wcs_frame(projection):
    """The shared primitive and the matplotlib frame must place points
    identically, or a conic would render differently in plotly than in mpl."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from skyplothelper.wcs_frame import make_wcs_frame

    lon = np.array([5., 20., -35., 100., -170.])
    lat = np.array([10., 55., 70., 30., 20.])
    fig = plt.figure()
    try:
        wcs = make_wcs_frame(111, projection=projection, center=0,
                             fig=fig).wcs
        pix = wcs.wcs_world2pix(np.column_stack([lon, lat]), 0)
        crpix = wcs.wcs.crpix - 1.0
        # Intermediate-world coords; x_sign = -1 for the 'sky' convention.
        mpl_x = (pix[:, 0] - crpix[0]) * wcs.wcs.cdelt[0] * -1.0
        mpl_y = (pix[:, 1] - crpix[1]) * wcs.wcs.cdelt[1]
    finally:
        plt.close(fig)

    x, y = project(lon, lat, projection=projection, center=0.0)
    assert np.allclose(x, mpl_x, atol=1e-9)
    assert np.allclose(y, mpl_y, atol=1e-9)


@pytest.mark.parametrize("projection", _CONIC_CODES)
def test_project_conic_pv2_1_changes_the_projection(projection):
    """pv2_1 reaches wcslib rather than being silently dropped."""
    xs = [float(project([20.], [20.], projection=projection, center=0.0,
                        pv2_1=pv)[0][0])
          for pv in (30.0, 45.0, 60.0)]
    assert len({round(v, 9) for v in xs}) == 3, xs


def test_project_conic_default_standard_parallel_is_45():
    """The unset default matches make_wcs_frame's, so the two agree."""
    explicit, _ = project([20.], [20.], projection='COD', pv2_1=45.0)
    default, _ = project([20.], [20.], projection='COD')
    assert np.allclose(explicit, default)


@pytest.mark.parametrize("projection", ['AIT', 'MOL', 'CAR', 'TAN', 'SIN'])
def test_project_pv_params_ignored_for_non_conic(projection):
    """pv2_1/pv2_2 on a projection that takes no standard parallel are
    silently ignored, matching the documented contract."""
    plain = project([20.], [20.], projection=projection)
    with_pv = project([20.], [20.], projection=projection, pv2_1=33.0,
                      pv2_2=7.0)
    assert np.allclose(plain[0], with_pv[0], equal_nan=True)
    assert np.allclose(plain[1], with_pv[1], equal_nan=True)


# ---------------------------------------------------------------------------
# project(): singular points are NaN, not finite garbage
# ---------------------------------------------------------------------------
#
# wcslib does not fail on a projection's singularity -- cos(90°) evaluates to
# 6.1e-17, not 0 -- so a point exactly on the gnomonic horizon came back as a
# finite ~1e17-1e19 instead of the NaN returned just beyond it. On TAN the
# whole meridian at center ± 90 lies on that horizon.

def test_project_tan_horizon_meridian_is_all_nan():
    """Every point of the meridian at center + 90 is exactly 90° from the
    tangent point. wcslib returned 68/91 finite values up to 2.6e19 there."""
    lats = np.linspace(-89.9, 89.9, 91)
    x, y = project(np.full_like(lats, 90.0), lats, projection='TAN',
                   center=0.0)
    assert not np.isfinite(x).any()
    assert not np.isfinite(y).any()


@pytest.mark.parametrize("projection",
                         ['TAN', 'AZP', 'SZP', 'COP', 'STG', 'SIN', 'AIT'])
def test_project_nan_is_paired_across_x_and_y(projection):
    """A masked point drops whole. An overflowed x leaves a finite, small y
    behind (the horizon yields pairs like (-9.4e17, 57.3)), and that y is as
    meaningless as its partner."""
    lon = np.linspace(-180, 180, 361)
    lat = np.linspace(-89.9, 89.9, 91)
    x, y = project(*np.meshgrid(lon, lat), projection=projection, center=0.0)
    assert not (np.isfinite(x) != np.isfinite(y)).any()


@pytest.mark.parametrize("projection", ['TAN', 'AZP', 'SZP', 'COP'])
def test_project_singularity_yields_no_huge_finite_values(projection):
    """Nothing finite survives past the meaningful-magnitude cut."""
    from skyplothelper.projections.project import _MAX_PLANE_COORD_DEG
    lon = np.linspace(-180, 180, 361)
    lat = np.linspace(-89.9, 89.9, 91)
    x, y = project(*np.meshgrid(lon, lat), projection=projection, center=0.0)
    finite = np.abs(np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]]))
    assert finite.size                       # not everything got masked
    assert finite.max() <= _MAX_PLANE_COORD_DEG


def test_project_singularity_mask_spares_legitimate_geometry():
    """The cut sits far above any usable extent, so bounded projections and
    ordinary field points are bit-for-bit unchanged."""
    lon = np.array([30., -75., 150., 0.])
    lat = np.array([15., -60., 80., 0.])
    for projection in ['AIT', 'MOL', 'CAR', 'MER', 'ZEA', 'ARC', 'COD']:
        x, y = project(lon, lat, projection=projection, center=0.0)
        assert np.isfinite(x).all() and np.isfinite(y).all(), projection
    # A TAN point well inside the horizon keeps its exact gnomonic value.
    x, _ = project([60.], [0.], projection='TAN', center=0.0)
    assert np.isclose(x[0], -np.degrees(np.tan(np.radians(60.0))), rtol=1e-9)


def test_project_sin_back_hemisphere_still_nan():
    """The pre-existing NaN contract behind a zenithal projection holds."""
    x, y = project([150., 180.], [0., 0.], projection='SIN', center=0.0)
    assert not np.isfinite(x).any() and not np.isfinite(y).any()


def test_dummy_allsky_hdr_conic_pv_cards():
    """PV2_1 is injected only for the five that need it; the four true conics
    additionally put the reference point on the standard parallel."""
    from skyplothelper.wcs_frame import dummy_allsky_hdr

    for code in ['AIT', 'MOL', 'CAR', 'TAN']:
        hdr = dummy_allsky_hdr(projection=code, pv2_1=33.0, pv2_2=7.0)
        assert 'PV2_1' not in hdr and 'PV2_2' not in hdr
        assert hdr['CRVAL2'] == 0.0

    for code in ['COD', 'COE', 'COO', 'COP']:
        hdr = dummy_allsky_hdr(projection=code)
        assert hdr['PV2_1'] == 45.0
        assert hdr['CRVAL2'] == 45.0        # reference point on the parallel

    bonne = dummy_allsky_hdr(projection='BON')
    assert bonne['PV2_1'] == 45.0
    assert bonne['CRVAL2'] == 0.0           # Bonne keeps its own reference lat

    spread = dummy_allsky_hdr(projection='COD', pv2_1=30.0, pv2_2=12.0)
    assert spread['PV2_1'] == 30.0 and spread['PV2_2'] == 12.0
    assert spread['CRVAL2'] == 30.0


def test_project_accepts_skycoord_and_infers_its_frame():
    """A SkyCoord replaces both arrays AND supplies the source frame, so no
    explicit frame= is needed — the result must match the explicit form."""
    import numpy as np
    from astropy.coordinates import SkyCoord

    from skyplothelper import project
    gal = SkyCoord([120.0, 130.0], [30.0, 35.0], unit="deg", frame="galactic")
    x1, y1 = project(gal, projection="MOL")
    x2, y2 = project(gal.l.deg, gal.b.deg, projection="MOL", frame="galactic")
    assert np.allclose(x1, x2) and np.allclose(y1, y2)


def test_project_skycoord_positional_misuse_raises():
    import pytest
    from astropy.coordinates import SkyCoord

    from skyplothelper import project
    with pytest.raises(TypeError):
        project(SkyCoord([1.0], [2.0], unit="deg"), [3.0])
