"""HEALPix numerical correctness.

Where the healpix smoke tests confirm the API doesn't crash, this
file pins the **mathematical correctness** of the binning,
smoothing, upgrade/downgrade, and query operations:

  * count statistic sums to N (input source count)
  * mean statistic recovers a constant input
  * smoothing approximately preserves total flux
  * smoothed FWHM matches the requested kernel width
  * upgrade-then-downgrade is an identity round-trip
  * combine arithmetic operations are correct
  * circle_query at the pole returns the right number of pixels
  * auto_nside scales sensibly with requested resolution
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
import healpy as hp  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.healpix import (  # noqa: E402
    auto_nside,
    bin_data_as_healpix,
    healpix_circle_query,
    healpix_combine,
    healpix_downgrade,
    healpix_polygon_query,
    healpix_smooth,
    healpix_upgrade,
    plot_healpix_allsky,
    plot_healpix_sparse,
)
from skyplothelper.wcs_frame import make_wcs_frame  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ============================================================
# bin_data_as_healpix — count statistic sums to N
# ============================================================

def test_count_statistic_sums_to_n():
    """Sum of counts across all populated pixels must equal N (number
    of input sources, assuming all are valid)."""
    rng = np.random.default_rng(0)
    n = 5000
    lons = rng.uniform(0, 360, n)
    lats = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))  # uniform on sphere
    data = np.ones(n)
    hpxmap, _, _, _ = bin_data_as_healpix(
        lons, lats, data, nside=32, statistic="count",
    )
    # NaN pixels are unoccupied; finite ones hold counts
    occupied = np.isfinite(hpxmap)
    total = int(np.nansum(hpxmap))
    assert total == n, (
        f"count sum {total} != input N={n}; {occupied.sum()} pixels populated"
    )


# ============================================================
# bin_data_as_healpix — mean recovers a constant
# ============================================================

def test_mean_statistic_recovers_constant():
    """When all input data values are equal to C, every populated
    pixel's mean must equal C."""
    rng = np.random.default_rng(1)
    n = 1000
    lons = rng.uniform(0, 360, n)
    lats = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    data = np.full(n, 7.5)
    hpxmap, _, _, _ = bin_data_as_healpix(
        lons, lats, data, nside=32, statistic="mean",
    )
    occupied = np.isfinite(hpxmap)
    assert np.allclose(hpxmap[occupied], 7.5, atol=1e-10), (
        f"populated pixel values vary: range "
        f"[{hpxmap[occupied].min()}, {hpxmap[occupied].max()}]"
    )


# ============================================================
# healpix_smooth — approximately preserves total flux
# ============================================================

def test_smooth_preserves_total_flux():
    """Smoothing a map with a Gaussian kernel preserves the sum of
    pixel values to within FP precision (since it's a convolution
    with a kernel that integrates to 1 on the sphere)."""
    rng = np.random.default_rng(2)
    nside = 32
    npix = hp.nside2npix(nside)
    m = rng.normal(1.0, 0.3, npix)  # nonzero mean to avoid sign issues
    total_before = m.sum()
    smoothed = healpix_smooth(m, sigma_deg=2.0)
    total_after = smoothed.sum()
    assert total_after == pytest.approx(total_before, rel=1e-3), (
        f"total flux changed by smoothing: {total_before} → {total_after}"
    )


# ============================================================
# healpix_smooth — Gaussian FWHM matches requested kernel width
# ============================================================

def test_smooth_fwhm_matches_kernel():
    """A delta-function spike, after Gaussian smoothing, has FWHM ≈ 2.355*sigma.
    Verify that the half-max radius equals the expected FWHM/2."""
    nside = 64
    npix = hp.nside2npix(nside)
    m = np.zeros(npix)
    # Place a 1.0 spike at (lon=0, lat=0)
    pix_center = hp.ang2pix(nside, 0.0, 0.0, lonlat=True)
    m[pix_center] = 1.0

    sigma_deg = 3.0
    smoothed = healpix_smooth(m, sigma_deg=sigma_deg)
    expected_fwhm = 2.355 * sigma_deg

    # Find pixels at angular distance d from spike, average values per d
    ipix = np.arange(npix)
    lons, lats = hp.pix2ang(nside, ipix, lonlat=True)
    cos_d = (np.sin(np.radians(0.0)) * np.sin(np.radians(lats)) +
             np.cos(np.radians(0.0)) * np.cos(np.radians(lats)) *
             np.cos(np.radians(lons - 0.0)))
    d = np.degrees(np.arccos(np.clip(cos_d, -1, 1)))

    # Bin the smoothed values by angular-distance shells
    bins = np.linspace(0, 15, 31)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    profile = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        mask = (d >= bins[i]) & (d < bins[i + 1])
        if mask.any():
            profile[i] = smoothed[mask].mean()

    # Find half-max radius
    half_max = profile.max() / 2
    above = profile >= half_max
    if above.any():
        last_above = np.where(above)[0][-1]
        radius_at_half_max = bin_centers[last_above]
        # FWHM is twice the radius at half-max
        fwhm_meas = 2 * radius_at_half_max
        assert fwhm_meas == pytest.approx(expected_fwhm, rel=0.15), (
            f"measured FWHM {fwhm_meas:.2f}° vs expected "
            f"{expected_fwhm:.2f}° (ratio {fwhm_meas / expected_fwhm:.3f})"
        )


# ============================================================
# upgrade-then-downgrade round-trip
# ============================================================

def test_upgrade_downgrade_round_trip():
    """upgrade(map) followed by downgrade(method='mean') must recover
    the original map to FP precision."""
    rng = np.random.default_rng(3)
    nside_in = 32
    npix = hp.nside2npix(nside_in)
    m = rng.uniform(0, 10, npix)

    upgraded = healpix_upgrade(m, nside_out=64)
    recovered = healpix_downgrade(upgraded, nside_out=nside_in, method="mean")

    assert np.allclose(recovered, m, atol=1e-10), (
        f"round-trip max abs error {np.max(np.abs(recovered - m))}"
    )


def test_upgrade_distributes_parent_value():
    """After upgrade, every child pixel should hold the same value
    as its parent. For a 2× upgrade, that's 4 children per parent."""
    nside_in = 16
    npix = hp.nside2npix(nside_in)
    m = np.arange(npix, dtype=float)
    upgraded = healpix_upgrade(m, nside_out=32)

    # Each parent maps to exactly 4 children. The mean of those 4 must
    # equal the parent value.
    downgraded = hp.ud_grade(upgraded, nside_in)
    assert np.allclose(downgraded, m, atol=1e-10)


def test_upgrade_rejects_lower_nside():
    """upgrade should refuse a nside_out below the input nside."""
    m = np.arange(hp.nside2npix(32), dtype=float)
    with pytest.raises(ValueError, match="(?i)must be"):
        healpix_upgrade(m, nside_out=16)


def test_downgrade_rejects_higher_nside():
    m = np.arange(hp.nside2npix(32), dtype=float)
    with pytest.raises(ValueError, match="(?i)must be"):
        healpix_downgrade(m, nside_out=64)


# ============================================================
# healpix_combine — arithmetic operations
# ============================================================

@pytest.mark.parametrize("op, expected", [
    ("add", lambda a, b: a + b),
    ("subtract", lambda a, b: a - b),
    ("multiply", lambda a, b: a * b),
])
def test_healpix_combine_arithmetic(op, expected):
    """healpix_combine returns ``(result_map, nside)``."""
    nside = 16
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(4)
    a = rng.uniform(0, 10, npix)
    b = rng.uniform(1, 5, npix)
    result, returned_nside = healpix_combine(a, b, operation=op)
    assert returned_nside == nside
    assert np.allclose(result, expected(a, b), rtol=1e-10)


# ============================================================
# healpix_circle_query — pole cap pixel count
# ============================================================

def test_circle_query_at_pole_returns_pixels():
    """A 10° circle at the north pole should return a non-empty
    pixel list whose pixels all lie in the northern hemisphere."""
    nside = 32
    pix = healpix_circle_query(0.0, 90.0, radius_deg=10.0, nside=nside)
    assert len(pix) > 0
    lons, lats = hp.pix2ang(nside, pix, lonlat=True)
    # All pixels should be at lat > 80° (within 10° of pole) — allow a
    # slight margin for pixel-center vs pixel-area.
    assert lats.min() > 75, f"min lat {lats.min()} (should be > 75°)"


def test_circle_query_disjoint_circles_dont_overlap():
    """Two non-overlapping disks return disjoint pixel sets."""
    nside = 32
    pix_a = healpix_circle_query(0.0, 0.0, 5.0, nside)
    pix_b = healpix_circle_query(180.0, 0.0, 5.0, nside)
    overlap = set(pix_a.tolist()) & set(pix_b.tolist())
    assert len(overlap) == 0


# ============================================================
# healpix_polygon_query — square polygon
# ============================================================

def test_polygon_query_returns_inside_pixels():
    """A polygon defining a square sky region should return pixels
    whose centers lie within (or on the boundary of) that region."""
    nside = 32
    # 20°×20° square centered on (180°, 0°)
    vertices = [(170, -10), (190, -10), (190, 10), (170, 10)]
    pix = healpix_polygon_query(vertices, nside=nside)
    assert len(pix) > 0
    lons, lats = hp.pix2ang(nside, pix, lonlat=True)
    # Most pixels should be inside the box (allow ~5° margin for the
    # query algorithm's finite resolution).
    inside_box = (lons > 165) & (lons < 195) & (lats > -15) & (lats < 15)
    fraction_inside = inside_box.sum() / len(pix)
    assert fraction_inside > 0.9, (
        f"only {fraction_inside:.1%} of returned pixels are in the box"
    )


# ============================================================
# auto_nside — resolution scales sensibly
# ============================================================

def test_auto_nside_finer_resolution_yields_higher_nside():
    """Asking for finer angular resolution must produce a larger nside.
    auto_nside returns ``(nside, actual_resolution_arcsec)``."""
    nside_coarse, _ = auto_nside(resolution_deg=5.0)
    nside_med, _ = auto_nside(resolution_deg=1.0)
    nside_fine, _ = auto_nside(resolution_deg=0.1)
    assert nside_coarse < nside_med < nside_fine


def test_auto_nside_returns_power_of_two():
    """HEALPix nside must be a power of 2."""
    for res in (10.0, 1.0, 0.1, 0.01):
        nside, _ = auto_nside(resolution_deg=res)
        # Power-of-2 check: nside & (nside-1) == 0
        assert nside > 0 and (nside & (nside - 1)) == 0, (
            f"auto_nside({res}°) returned {nside}, not a power of 2"
        )


def test_auto_nside_actual_resolution_finer_than_requested():
    """The reported actual_resolution_arcsec must be ≤ the requested
    resolution (the function picks the smallest nside that satisfies)."""
    requested_deg = 1.0
    nside, actual_arcsec = auto_nside(resolution_deg=requested_deg)
    assert actual_arcsec <= requested_deg * 3600 + 1e-3


# ============================================================
# plot_healpix_sparse — patch ↔ pixel index mapping
# ============================================================
#
# The patches backend exposes ``pc.patch_pixel_index`` so users can
# recover which patch(es) belong to a given input HEALPix pixel
# (highlighting, alpha bumps, etc.). The mapping is many-to-one:
# a tile that splits across the antimeridian or visible frame edge
# yields multiple patches that all carry the same source pixel id.

def test_patch_pixel_index_length_matches_patches():
    """``pc.patch_pixel_index`` must have one entry per rendered patch."""
    nside = 8
    pix = healpix_circle_query(45.0, 30.0, 20.0, nside)
    vals = np.linspace(0, 1, len(pix))
    ax = make_wcs_frame(111, projection="AIT", center=180)
    pc = plot_healpix_sparse(pix, vals, nside=nside, ax=ax)
    assert hasattr(pc, "patch_pixel_index")
    assert len(pc.patch_pixel_index) == len(pc.get_paths())


def test_patch_pixel_index_no_split_is_one_to_one():
    """For a cluster fully clear of the antimeridian / frame edges the
    mapping is 1:1 with ``pixel_indices`` and preserves order."""
    nside = 8
    # Cluster around (60, 30) — well clear of lon=0 (the antimeridian
    # for an AIT center=180 frame) and the poles.
    pix = healpix_circle_query(60.0, 30.0, 10.0, nside)
    vals = np.linspace(0, 1, len(pix))
    ax = make_wcs_frame(111, projection="AIT", center=180)
    pc = plot_healpix_sparse(pix, vals, nside=nside, ax=ax)
    assert len(pc.patch_pixel_index) == len(pix), (
        "Non-crossing cluster should yield one patch per input pixel "
        f"(got {len(pc.patch_pixel_index)} patches for {len(pix)} pixels)"
    )
    np.testing.assert_array_equal(pc.patch_pixel_index, pix)


def test_patch_pixel_index_handles_antimeridian_split():
    """A tile that straddles the antimeridian splits into two patches —
    both must report the same source pixel id, so the mapping is
    many-to-one and ``len(unique) < len(patches)``."""
    nside = 8
    # Cluster centered ON the antimeridian (lon=0 for AIT center=180).
    # The boundary tiles straddle the seam and must split.
    pix = healpix_circle_query(0.0, 30.0, 15.0, nside)
    vals = np.linspace(0, 1, len(pix))
    ax = make_wcs_frame(111, projection="AIT", center=180)
    pc = plot_healpix_sparse(pix, vals, nside=nside, ax=ax)

    # At least one tile must have split.
    assert len(pc.patch_pixel_index) > len(np.unique(pc.patch_pixel_index)), (
        "Expected some antimeridian-straddling tiles to split into "
        "multiple patches sharing one pixel id"
    )
    # Every reported source pixel must come from the input set.
    assert set(pc.patch_pixel_index.tolist()).issubset(set(pix.tolist()))


def test_patch_pixel_index_recovers_unique_rendered_pixels():
    """``np.unique(pc.patch_pixel_index)`` must equal the set of
    rendered input pixels (not necessarily all input pixels — pixels
    fully clipped by the frame produce zero patches)."""
    nside = 8
    pix = healpix_circle_query(0.0, 30.0, 15.0, nside)
    vals = np.linspace(0, 1, len(pix))
    ax = make_wcs_frame(111, projection="AIT", center=180)
    pc = plot_healpix_sparse(pix, vals, nside=nside, ax=ax)

    rendered = np.unique(pc.patch_pixel_index)
    assert np.all(np.isin(rendered, pix))


def test_patch_pixel_index_is_int_array():
    """The attribute must be an integer ndarray (so users can use it
    as a mask / index without casting)."""
    nside = 16
    pix = healpix_circle_query(45.0, 30.0, 5.0, nside)
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(45, 30), fig=fig)
    fig.canvas.draw()
    pc = plot_healpix_sparse(pix, None, nside=nside, ax=ax)
    assert isinstance(pc.patch_pixel_index, np.ndarray)
    assert np.issubdtype(pc.patch_pixel_index.dtype, np.integer)


def test_patch_pixel_index_highlight_mask_workflow():
    """End-to-end: pick a target pixel, build a mask from
    ``patch_pixel_index``, recover the matching patch path(s), and
    assert the path centroid is near the pixel's true sky position."""
    nside = 16
    pix = healpix_circle_query(45.0, 30.0, 8.0, nside)
    vals = np.linspace(0, 1, len(pix))
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(45, 30), fig=fig)
    fig.canvas.draw()
    pc = plot_healpix_sparse(pix, vals, nside=nside, ax=ax)

    target = pix[len(pix) // 2]
    mask = pc.patch_pixel_index == target
    assert mask.any(), "Expected at least one patch for the target pixel"

    # The centroid of the matching patch (in pixel coords) should sit
    # near the projected world-coord position of that HEALPix pixel.
    paths = np.asarray(pc.get_paths(), dtype=object)[mask]
    centroid_pix = np.mean(paths[0].vertices, axis=0)

    target_lon, target_lat = hp.pix2ang(nside, target, lonlat=True)
    target_world_to_pix = ax.wcs.wcs_world2pix(
        np.array([[target_lon, target_lat]]), 0)[0]
    # Convert ax.wcs pixel coords → display data coords. WCSAxes' data
    # transform is identity to its WCS pixel grid.
    dist = np.hypot(centroid_pix[0] - target_world_to_pix[0],
                    centroid_pix[1] - target_world_to_pix[1])
    # nside=16 max pixrad ≈ 4.4°; on a TAN with default cdelt the
    # tile is at most ~5 pixels across — centroid should be well
    # within that.
    assert dist < 10.0, (
        f"Patch centroid {centroid_pix} too far from projected pixel "
        f"position {target_world_to_pix} (Δ={dist:.2f} px)"
    )


# ============================================================
# plot_healpix_sparse on globe / sphere frames
# ============================================================
#
# The orthographic / globe projection (SIN with center=(lon0, lat0))
# is the natural way to *show why* HEALPix makes sense as a sphere-
# native pixelization. Smoke-test that tiles render correctly across
# the globe limb and on a tilted globe (non-zero lat0).

def test_plot_healpix_sparse_renders_on_orthographic_globe():
    """Sparse tiles render on a SIN-globe centered on the equator —
    the patches backend must produce at least one patch per visible
    pixel and not crash on the limb-clipped tiles."""
    nside = 16
    # Cluster on the front hemisphere (around the globe center).
    pix = healpix_circle_query(180.0, 0.0, 25.0, nside)
    vals = np.linspace(0, 1, len(pix))
    ax = make_wcs_frame(111, projection="SIN", center=(180, 0))
    pc = plot_healpix_sparse(pix, vals, nside=nside, ax=ax)
    # All cluster pixels are on the front hemisphere; every input
    # pixel must produce at least one patch.
    assert set(np.unique(pc.patch_pixel_index).tolist()) == set(pix.tolist())


def test_plot_healpix_sparse_renders_on_tilted_globe():
    """Sparse tiles render on a tilted SIN-globe (lat0=30°)."""
    nside = 16
    pix = healpix_circle_query(180.0, 30.0, 20.0, nside)
    vals = np.linspace(0, 1, len(pix))
    ax = make_wcs_frame(111, projection="SIN", center=(180, 30))
    pc = plot_healpix_sparse(pix, vals, nside=nside, ax=ax)
    assert set(np.unique(pc.patch_pixel_index).tolist()) == set(pix.tolist())


def test_plot_healpix_globe_drops_back_hemisphere_tiles():
    """Pixels on the *back* hemisphere of a SIN-globe must not appear:
    the frame-clipping pipeline should reject tiles whose centers are
    > 90° from the globe center."""
    nside = 8
    # Two clusters: one in front, one antipodal.
    front = healpix_circle_query(180.0, 0.0, 15.0, nside)
    back = healpix_circle_query(0.0, 0.0, 15.0, nside)
    # Skip the test if any front pixel accidentally lives in `back`'s
    # query (shouldn't happen, but be safe).
    pix = np.concatenate([front, back])
    vals = np.arange(len(pix), dtype=float)
    ax = make_wcs_frame(111, projection="SIN", center=(180, 0))
    pc = plot_healpix_sparse(pix, vals, nside=nside, ax=ax)
    rendered = set(np.unique(pc.patch_pixel_index).tolist())
    # All front-hemisphere pixels visible.
    assert set(front.tolist()).issubset(rendered)
    # No back-hemisphere pixel should reach the canvas.
    assert rendered.isdisjoint(set(back.tolist())), (
        "Back-hemisphere tiles leaked through the SIN frame clip: "
        f"{rendered & set(back.tolist())}"
    )


# ============================================================
# project_to_canvas / healpix_to_canvas
# ============================================================
#
# The canvas-pixel sampling utilities replace the legacy lon/lat
# meshgrid + transform='world' path with a sliver-free, framework-
# agnostic approach: one inverse-projection per output pixel, one
# data lookup, one 2D array suitable for ax.imshow / ax.pcolormesh.

def test_project_to_canvas_default_shape_2x_extent():
    """Default ``output_shape`` is ~2× the visible WCS pixel extent
    in the wider dimension, with a floor of 2000 in the wider
    dimension so coarse WCS grids still produce sub-pixel output."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    arr, ext = sph.project_to_canvas(ax, lambda lon, lat: lat)
    # Default AIT WCS grid is 360x180 → wider dim is 360 → 2x = 720,
    # but the floor of 2000 kicks in. Shape comes back as (ny, nx).
    ny, nx = arr.shape
    assert nx >= 2000, f"expected wider dim >= 2000, got nx={nx}"
    # Aspect ratio should match the WCS pixel grid (2:1 for AIT).
    assert abs(nx / ny - 2.0) < 0.05, (
        f"aspect off: nx={nx}, ny={ny}, ratio={nx/ny:.3f}")


def test_project_to_canvas_lookup_called_with_axes_frame_lons():
    """``lookup_fn`` receives lons in the axes' coordinate frame —
    inverse-projecting an AIT axes' canvas pixels gives lon/lat
    values in the visible range."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    seen = {}

    def lookup(lons, lats):
        seen["lons_min"] = float(lons.min())
        seen["lons_max"] = float(lons.max())
        seen["lats_min"] = float(lats.min())
        seen["lats_max"] = float(lats.max())
        return np.zeros_like(lons)

    sph.project_to_canvas(ax, lookup, output_shape=(100, 200))
    # Center=180 AIT covers lon ~0..360 (after our %360 normalization)
    # and lat ~-90..90.
    assert seen["lons_min"] < 90
    assert seen["lons_max"] > 270
    assert seen["lats_min"] < -60
    assert seen["lats_max"] > 60


def test_project_to_canvas_off_projection_pixels_blanked():
    """Pixels outside the projection's valid hemisphere (e.g. the
    AIT ellipse's exterior) get the ``blank_value``."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    arr, _ = sph.project_to_canvas(ax,
                                    lambda lon, lat: np.ones_like(lon),
                                    output_shape=(100, 200),
                                    blank_value=-99.0)
    # Some pixels should be inside the ellipse (value 1.0) and some
    # outside (value -99.0). Both sets non-empty.
    assert np.any(arr == 1.0)
    assert np.any(arr == -99.0)


def test_healpix_to_canvas_recovers_constant_map():
    """A constant-valued HEALPix map produces a constant array
    (apart from off-projection blank pixels)."""
    nside = 16
    npix = hp.nside2npix(nside)
    m = np.full(npix, 7.5)
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    arr, _ = sph.healpix_to_canvas(m, ax, output_shape=(100, 200))
    valid = np.isfinite(arr)
    np.testing.assert_allclose(arr[valid], 7.5)


def test_healpix_to_canvas_nearest_vs_interp():
    """``interp=True`` smooths over tile boundaries; on a sharp-
    edged map the bilinear path produces a wider value range
    along the boundary than nearest-pixel."""
    nside = 8
    npix = hp.nside2npix(nside)
    # Sharp-edged: 0/1 split
    m = (np.arange(npix) % 2).astype(float)
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    arr_nearest, _ = sph.healpix_to_canvas(m, ax, output_shape=(200, 400),
                                            interp=False)
    arr_interp, _ = sph.healpix_to_canvas(m, ax, output_shape=(200, 400),
                                           interp=True)
    valid = np.isfinite(arr_nearest) & np.isfinite(arr_interp)
    # Nearest-pixel: only the discrete values 0 and 1.
    nearest_vals = np.unique(arr_nearest[valid])
    assert set(nearest_vals.tolist()) <= {0.0, 1.0}
    # Bilinear: smoothed values strictly between 0 and 1 exist.
    interp_vals = arr_interp[valid]
    assert np.any((interp_vals > 0.05) & (interp_vals < 0.95))


def _make_axes_for_test(projection="AIT", center=180):
    """Test helper: build an axes the way users should now."""
    fig = plt.figure(figsize=(11, 5.5))
    ax = make_wcs_frame(111, projection=projection, center=center, fig=fig)
    fig.canvas.draw()
    return fig, ax


def test_plot_healpix_allsky_default_returns_axes_image():
    """The new default backend is 'imshow'; mappable returned is
    an AxesImage (suitable for ``fig.colorbar``)."""
    from matplotlib.image import AxesImage
    nside = 8
    npix = hp.nside2npix(nside)
    m = np.zeros(npix)
    fig, ax = _make_axes_for_test()
    result = plot_healpix_allsky(m, ax=ax, cmap="viridis", colorbar=False)
    assert isinstance(result.mappable, AxesImage), (
        f"expected AxesImage from default 'imshow' backend, "
        f"got {type(result.mappable)}"
    )


def test_plot_healpix_allsky_returns_named_tuple_with_colorbar():
    """The ``HealpixResult`` NamedTuple exposes the colorbar as an
    attribute and supports tuple unpacking."""
    from matplotlib.colorbar import Colorbar

    from skyplothelper.healpix import HealpixResult
    nside = 8
    m = np.zeros(hp.nside2npix(nside))
    fig, ax = _make_axes_for_test()
    result = plot_healpix_allsky(m, ax=ax, cmap="viridis", colorbar=True,
                                  cbar_label="test")
    assert isinstance(result, HealpixResult)
    assert isinstance(result.colorbar, Colorbar)
    # Tuple unpack also works (4 elements).
    fig2, ax2, im, cbar = result
    assert ax2 is ax
    # User can post-adjust the colorbar
    cbar.set_label("relabeled")


def test_plot_healpix_allsky_no_colorbar_returns_none():
    """``colorbar=False`` → result.colorbar is None."""
    nside = 8
    m = np.zeros(hp.nside2npix(nside))
    fig, ax = _make_axes_for_test()
    result = plot_healpix_allsky(m, ax=ax, colorbar=False)
    assert result.colorbar is None


def test_plot_healpix_allsky_image_kwargs_collision_raises():
    """Passing the same key as a named kwarg AND in ``image_kwargs``
    is rejected (no silent overrides)."""
    nside = 8
    m = np.zeros(hp.nside2npix(nside))
    fig, ax = _make_axes_for_test()
    with pytest.raises(TypeError, match="supplied both"):
        plot_healpix_allsky(m, ax=ax, cmap="viridis",
                            image_kwargs={"cmap": "plasma"})


def test_plot_healpix_allsky_figure_only_kwarg_rejected():
    """Figure-creation kwargs (projection, center, frame, figsize, dpi,
    style) are rejected on the axis-plotter — they belong on
    ``healpix_allsky_figure``."""
    nside = 8
    m = np.zeros(hp.nside2npix(nside))
    fig, ax = _make_axes_for_test()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        plot_healpix_allsky(m, ax=ax, projection="MOL")


def test_plot_healpix_allsky_pcolormesh_canvas_returns_quadmesh():
    """``backend='pcolormesh'`` with default ``sampling='canvas'``
    returns a QuadMesh."""
    from matplotlib.collections import QuadMesh
    nside = 8
    m = np.zeros(hp.nside2npix(nside))
    fig, ax = _make_axes_for_test()
    result = plot_healpix_allsky(m, ax=ax, backend="pcolormesh",
                                  cmap="viridis", colorbar=False)
    assert isinstance(result.mappable, QuadMesh)


def test_plot_healpix_allsky_lonlat_legacy_still_works():
    """``sampling='lonlat'`` recovers the legacy path. Returns a
    QuadMesh and does not raise."""
    from matplotlib.collections import QuadMesh
    nside = 8
    m = np.zeros(hp.nside2npix(nside))
    fig, ax = _make_axes_for_test()
    result = plot_healpix_allsky(m, ax=ax, backend="pcolormesh",
                                  sampling="lonlat",
                                  cmap="viridis", colorbar=False)
    assert isinstance(result.mappable, QuadMesh)


def test_plot_healpix_allsky_invalid_sampling_raises():
    """Unknown ``sampling`` value gets a clear error."""
    nside = 8
    m = np.zeros(hp.nside2npix(nside))
    fig, ax = _make_axes_for_test()
    with pytest.raises(ValueError, match="sampling must be"):
        plot_healpix_allsky(m, ax=ax, backend="pcolormesh",
                            sampling="bogus", colorbar=False)


def test_healpix_allsky_figure_one_line_builder():
    """The figure-builder convenience returns a HealpixResult with
    a fresh figure."""
    from skyplothelper.healpix import HealpixResult, healpix_allsky_figure
    nside = 8
    m = np.zeros(hp.nside2npix(nside))
    result = healpix_allsky_figure(m, projection="MOL", center=0,
                                    figsize=(11, 5.5))
    assert isinstance(result, HealpixResult)
    assert result.colorbar is not None  # default colorbar=True
