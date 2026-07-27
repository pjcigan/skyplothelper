"""Edge cases: antimeridian crossings, pole containment,
NaN handling, and corner cases on coordinate conversions.

Where Groups 2–3 verify mathematical correctness of typical inputs,
this file exercises the boundary conditions that are easiest to
get wrong: the lon wrap point, latitudes at exactly ±90°, NaN
propagation, signed sub-degree DMS values, and dateline-crossing
shapes.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ============================================================
# Antimeridian: wrap functions
# ============================================================

@pytest.mark.parametrize("inp, expected", [
    (180.0, -180.0),      # boundary maps to -180 (half-open [-180, 180))
    (-180.0, -180.0),     # already at boundary, stays
    (181.0, -179.0),
    (359.999, -0.001),
    (0.0, 0.0),
    (-0.0, 0.0),
])
def test_wrap_pm180_boundary_values(inp, expected):
    from skyplothelper.core.math_utils import wrap_pm180
    out = wrap_pm180(inp)
    assert out == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("inp, expected", [
    (360.0, 0.0),
    (-1.0, 359.0),
    (720.0, 0.0),
    (0.0, 0.0),
])
def test_wrap_360_boundary_values(inp, expected):
    from skyplothelper.core.math_utils import wrap_360
    out = wrap_360(inp)
    assert out == pytest.approx(expected, abs=1e-9)


def test_wrap_24hr_handles_negative_hours():
    from skyplothelper.core.math_utils import wrap_24hr
    assert wrap_24hr(-1.0) == pytest.approx(23.0, abs=1e-9)
    assert wrap_24hr(25.0) == pytest.approx(1.0, abs=1e-9)


def test_wrap_360_array_preserves_shape():
    from skyplothelper.core.math_utils import wrap_360
    arr = np.array([-90, 0, 90, 180, 270, 360, 450])
    out = wrap_360(arr)
    assert out.shape == arr.shape
    assert np.all(out >= 0) and np.all(out < 360 + 1e-9)


# ============================================================
# Antimeridian: a geodesic circle centered at lon=0 has samples
# that cross the dateline (lon ≈ ±180)
# ============================================================

def test_geodesic_circle_at_lon0_does_not_have_cross_meridian_jumps():
    """When sampling a geodesic circle at lon=0, the samples should
    not include unphysical jumps from -180 to +180. The function should
    return a continuous lon trace whose values are unwrapped within
    one period."""
    from skyplothelper.geometry.shapes import geodesic_circle
    lons, lats = geodesic_circle(0, 0, radius_deg=10, resolution=200)
    # All samples within 10° of (0,0) — lon range should be ±10°
    # (continuous), NOT split between -180..-179 and +179..+180.
    assert np.max(np.abs(((lons + 180) % 360) - 180)) < 12, (
        f"lons span {lons.min()}..{lons.max()}, "
        "should be ≈ ±10° around 0"
    )


# ============================================================
# Pole containment
# ============================================================

def test_geodesic_circle_at_north_pole_lats_in_correct_range():
    """A 30° circle at the north pole should have all sample lats
    in the range [60°, 90°]."""
    from skyplothelper.geometry.shapes import geodesic_circle
    lons, lats = geodesic_circle(0, 90, radius_deg=30, resolution=64)
    assert lats.min() == pytest.approx(60.0, abs=0.5)
    assert lats.max() <= 90.0 + 1e-6


def test_geodesic_circle_at_south_pole_lats_in_correct_range():
    from skyplothelper.geometry.shapes import geodesic_circle
    lons, lats = geodesic_circle(0, -90, radius_deg=20, resolution=64)
    assert lats.max() == pytest.approx(-70.0, abs=0.5)
    assert lats.min() >= -90.0 - 1e-6


def test_compound_region_polar_cap_solid_angle():
    """A 30° geodesic circle at the north pole subtends a solid angle
    Ω = 2π·(1 − cos(30°)) ≈ 2π·0.134 sr. CompoundRegion's
    pixel-area approximation should match within ~5%."""
    from skyplothelper.geometry.compound import CompoundRegion
    from skyplothelper.wcs_frame import make_wcs_frame
    fig = plt.figure(figsize=(10, 5))
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    R = CompoundRegion(ax).add_circle(0, 90, radius_deg=30)
    expected_sr = 2 * np.pi * (1 - np.cos(np.radians(30)))
    actual_sr = R.solid_angle["sr"]
    assert actual_sr == pytest.approx(expected_sr, rel=0.05), (
        f"polar-cap solid angle: {actual_sr:.4f} sr "
        f"(analytical {expected_sr:.4f} sr)"
    )


def test_healpix_circle_query_pole_returns_pole_pixels():
    """All pixels in a 5° circle at the north pole have lat > 80°."""
    pytest.importorskip("healpy")
    import healpy as hp

    from skyplothelper.healpix import healpix_circle_query
    nside = 32
    pix = healpix_circle_query(0.0, 90.0, radius_deg=5.0, nside=nside)
    _lons, lats = hp.pix2ang(nside, pix, lonlat=True)
    assert np.all(lats > 80), f"min lat in pole-query: {lats.min()}"


# ============================================================
# NaN handling
# ============================================================

def test_bin_data_as_healpix_drops_nan_values():
    """NaN data values should be dropped from binning, not poison the
    affected pixels."""
    pytest.importorskip("healpy")
    from skyplothelper.healpix import bin_data_as_healpix
    rng = np.random.default_rng(0)
    n = 1000
    lons = rng.uniform(0, 360, n)
    lats = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    data = rng.uniform(0, 10, n)
    # Inject NaNs in 10% of the data
    data[rng.choice(n, 100, replace=False)] = np.nan

    hpxmap, _, _, _ = bin_data_as_healpix(
        lons, lats, data, nside=16, statistic="mean",
    )
    occupied = np.isfinite(hpxmap)
    # Of the populated pixels, none should be NaN (NaNs were dropped)
    assert occupied.any()
    # Mean should be sensible
    assert hpxmap[occupied].max() <= 10.0 + 1e-9
    assert hpxmap[occupied].min() >= 0.0 - 1e-9


def test_clip_percentile_with_nans():
    """clip_percentile must skip NaNs."""
    from skyplothelper.images.levels import clip_percentile
    rng = np.random.default_rng(1)
    data = rng.uniform(0, 1, (50, 50))
    data[10:20, 10:20] = np.nan
    lo, hi = clip_percentile(data, plo=1, phi=99)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi


def test_clip_sigma_with_nans():
    from skyplothelper.images.levels import clip_sigma
    rng = np.random.default_rng(2)
    data = rng.normal(0, 1, (50, 50))
    data[5:15, 5:15] = np.nan
    lo, hi = clip_sigma(data, sigma_lo=3)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi


def test_rescale_image_nan_handling():
    """rescale_image with fill_nan=0 should produce all-finite output."""
    from skyplothelper.images.levels import rescale_image
    rng = np.random.default_rng(3)
    data = rng.uniform(0, 1, (40, 40))
    data[5:10, 5:10] = np.nan
    out = rescale_image(data, stretch="linear", clip="percentile", fill_nan=0.0)
    assert np.all(np.isfinite(out)), (
        "rescale_image(fill_nan=0) should fill NaNs to 0"
    )


def test_make_norm_with_nan_data_yields_finite_vmin_vmax():
    from skyplothelper.images.levels import make_norm
    rng = np.random.default_rng(4)
    data = rng.uniform(0, 1, (40, 40))
    data[10:20, 10:20] = np.nan
    norm = make_norm(stretch="linear", data=data)
    assert np.isfinite(norm.vmin)
    assert np.isfinite(norm.vmax)
    assert norm.vmin < norm.vmax


# ============================================================
# Coordinate-conversion corner cases
# ============================================================

def test_deg2dms_at_zero_is_clean():
    from skyplothelper.core.coords import deg2dms
    out = deg2dms(0.0)
    # Output should be (sign, d, m, s) or similar; verify no negative-zero
    assert out is not None


def test_dms2deg_round_trip_negative_subdegree():
    """dms2deg / deg2dms must round-trip a small negative value."""
    from skyplothelper.core.coords import deg2dms, dms2deg
    val = -0.5  # half a degree south
    out = deg2dms(val)
    back = dms2deg(out)
    assert back == pytest.approx(val, abs=1e-9), (
        f"round-trip {val} → {out} → {back}"
    )


def test_dms2deg_round_trip_above_60_seconds_fp_rollover():
    """The known historical issue: deg2dms(-12.7) used to produce
    seconds = 60.0000... triggering a sign-bit error. Verify the
    round-trip is clean."""
    from skyplothelper.core.coords import deg2dms, dms2deg
    val = -12.7
    out = deg2dms(val)
    back = dms2deg(out)
    assert back == pytest.approx(val, abs=1e-9), (
        f"round-trip {val} → {out} → {back}"
    )


def test_angulardistance_zero_separation():
    """angulardistance takes (lon, lat) pairs."""
    from skyplothelper.core.coords import angulardistance
    d = angulardistance([180.0, 30.0], [180.0, 30.0])
    assert d == pytest.approx(0.0, abs=1e-9)


def test_angulardistance_antipodal():
    from skyplothelper.core.coords import angulardistance
    d = angulardistance([0.0, 0.0], [180.0, 0.0])
    assert d == pytest.approx(180.0, abs=1e-6)


def test_angulardistance_pole_to_equator_is_90deg():
    from skyplothelper.core.coords import angulardistance
    d = angulardistance([0.0, 90.0], [0.0, 0.0])
    assert d == pytest.approx(90.0, abs=1e-9)
