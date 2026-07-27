"""Tests for skyplothelper.healpix."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from skyplothelper.healpix import (
    _HAS_HEALPY,
    auto_nside,
    bin_data_as_healpix,
    healpix_allsky_figure,
    healpix_circle_query,
    healpix_to_celestial,
    image_to_healpix,
    mask_seam_crossing_quads,
    nside_from_array,
    plot_healpix_allsky,
    plot_healpix_sparse,
)

# All tests in this file need healpy.
pytestmark = pytest.mark.skipif(not _HAS_HEALPY, reason="healpy not installed")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_auto_nside_runs():
    """auto_nside returns (nside, actual_resolution_arcsec)."""
    out = auto_nside(resolution_deg=1.0)
    assert isinstance(out, tuple) and len(out) == 2
    nside, actual_arcsec = out
    assert int(nside) > 0
    # Power of 2 check
    assert (int(nside) & (int(nside) - 1)) == 0
    assert actual_arcsec > 0


def test_bin_scattered_points_to_healpix():
    """Bin a few sources; verify the HEALPix map size matches nside."""
    rng = np.random.default_rng(42)
    lons = rng.uniform(0, 360, 200)
    lats = rng.uniform(-90, 90, 200)
    # bin_data_as_healpix returns (hpxmap, plonc, platc, pvals)
    hpxmap, _, _, _ = bin_data_as_healpix(
        lons, lats, np.ones(200), nside=16, statistic="count"
    )
    import healpy as hp
    assert len(hpxmap) == hp.nside2npix(16)


def test_healpix_circle_query_returns_pixels():
    pixels = healpix_circle_query(0.0, 0.0, radius_deg=10.0, nside=16)
    assert len(pixels) > 0


def test_healpix_allsky_figure_smoke():
    """Render an all-sky HEALPix map to a figure (one-line builder)."""
    import healpy as hp
    nside = 16
    arr = np.arange(hp.nside2npix(nside), dtype=float)
    # healpix_allsky_figure returns HealpixResult(fig, ax, mappable, colorbar).
    result = healpix_allsky_figure(arr, projection="AIT")
    assert result.fig is not None
    assert result.mappable is not None
    assert result.colorbar is not None  # default colorbar=True


def test_plot_healpix_allsky_smoke():
    """Plot an all-sky HEALPix map onto an existing axes."""
    import healpy as hp

    from skyplothelper.wcs_frame import make_wcs_frame
    nside = 16
    arr = np.arange(hp.nside2npix(nside), dtype=float)
    fig = plt.figure(figsize=(11, 5.5))
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    fig, ax2, im, cbar = plot_healpix_allsky(arr, ax=ax)
    assert ax2 is ax  # uses provided axes, no new figure
    assert im is not None


def test_plot_healpix_sparse_smoke():
    """Render a sparse HEALPix slice on a WCSAxes."""
    from skyplothelper.wcs_frame import make_wcs_frame

    pixels = healpix_circle_query(0.0, 0.0, radius_deg=10.0, nside=16)
    values = np.arange(len(pixels), dtype=float)
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    plot_healpix_sparse(pixels, values, nside=16, ax=ax)


# ---- seam-crossing quad masking (Bug C) ----

def _demo_mesh(center=0):
    import healpy as hp
    nside = 32
    lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    lar, lor = np.radians(lat), np.radians(lon)

    def g(s, w=7):
        return np.exp(-0.5 * (s / w) ** 2)
    demo = np.maximum.reduce([
        g(np.abs(lat)),
        g(np.degrees(np.arcsin(np.clip(np.abs(np.cos(lar) * np.sin(lor)), 0, 1)))),
        g(np.degrees(np.arcsin(np.clip(np.abs(np.cos(lar) * np.cos(lor)), 0, 1))))])
    return healpix_to_celestial(demo, "allsky", center, (400, 200), np.nan)


def test_mask_seam_crossing_blanks_interrupted_projection():
    """On HPX the polar-cap cells bridge a face seam; masking must NaN some
    cells that were finite before (regression for the fill-across-seam smear)."""
    from skyplothelper.wcs_frame import make_wcs_frame
    plonc, platc, pvals = _demo_mesh()
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="HPX", center=0, frame="galactic",
                        fig=fig, grid=False)
    masked = mask_seam_crossing_quads(ax, plonc, platc, pvals)
    newly_blank = np.isfinite(pvals) & ~np.isfinite(masked)
    assert newly_blank.sum() > 0


def test_mask_seam_crossing_is_noop_on_continuous_projection():
    """On a continuous projection (AIT) no cell bridges a seam, so masking
    must not blank any previously-finite cell."""
    from skyplothelper.wcs_frame import make_wcs_frame
    plonc, platc, pvals = _demo_mesh()
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=0, frame="galactic",
                        fig=fig, grid=False)
    masked = mask_seam_crossing_quads(ax, plonc, platc, pvals)
    newly_blank = np.isfinite(pvals) & ~np.isfinite(masked)
    assert newly_blank.sum() == 0


def test_mask_seam_crossing_dedups_xph_antimeridian():
    """XPH folds the antimeridian onto a corner facet, so the two seam-edge
    columns overlap there. The mask blanks a thin antimeridian strip on XPH
    (removing the overlap) but must NOT do so on a continuous projection."""
    from skyplothelper.wcs_frame import make_wcs_frame
    plonc, platc, pvals = _demo_mesh(center=0)
    rel = ((plonc - 0.0 + 180.0) % 360.0) - 180.0
    near_am = (np.abs(rel) > 178.5) & np.isfinite(pvals)
    assert near_am.sum() > 0

    fig = plt.figure()
    ax = make_wcs_frame(111, projection="XPH", center=0, frame="galactic",
                        fig=fig, grid=False)
    masked = mask_seam_crossing_quads(ax, plonc, platc, pvals)
    assert not np.any(np.isfinite(masked[near_am]))  # XPH: strip blanked

    fig2 = plt.figure()
    ax2 = make_wcs_frame(111, projection="AIT", center=0, frame="galactic",
                         fig=fig2, grid=False)
    masked2 = mask_seam_crossing_quads(ax2, plonc, platc, pvals)
    assert np.any(np.isfinite(masked2[near_am]))  # AIT: strip kept


def _car_allsky_lat_image(ny=90, nx=180, cdelt=2.0):
    """An all-sky CAR image whose pixel value equals the pixel's latitude."""
    from astropy.wcs import WCS
    hdr = {
        "NAXIS": 2, "NAXIS1": nx, "NAXIS2": ny,
        "CTYPE1": "RA---CAR", "CTYPE2": "DEC--CAR",
        "CRPIX1": nx / 2 + 0.5, "CRPIX2": ny / 2 + 0.5,
        "CRVAL1": 0.0, "CRVAL2": 0.0,
        "CDELT1": -cdelt, "CDELT2": cdelt, "CUNIT1": "deg", "CUNIT2": "deg",
    }
    w = WCS(hdr)
    yy, xx = np.mgrid[0:ny, 0:nx]
    _, lat = w.pixel_to_world_values(xx, yy)
    return lat.astype(float), hdr, w


def _small_field_image(npix_side=60, cdelt=0.01, crval=(180.0, 0.0)):
    """A small (~0.6°) TAN field image (pixel value == 1) for sparse-auto."""
    from astropy.wcs import WCS
    hdr = {
        "NAXIS": 2, "NAXIS1": npix_side, "NAXIS2": npix_side,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRPIX1": npix_side / 2 + 0.5, "CRPIX2": npix_side / 2 + 0.5,
        "CRVAL1": crval[0], "CRVAL2": crval[1],
        "CDELT1": -cdelt, "CDELT2": cdelt, "CUNIT1": "deg", "CUNIT2": "deg",
    }
    return np.ones((npix_side, npix_side)), hdr, WCS(hdr)


def test_image_to_healpix_dense_shape_and_values():
    """Dense map: full-sky array, each occupied cell carries ~its own latitude
    (the binned image value), within the pixel scale."""
    import healpy as hp
    data, hdr, _ = _car_allsky_lat_image()
    nside = 32
    hpx = image_to_healpix(data, hdr, nside=nside, statistic="mean",
                           sparse=False)
    assert hpx.shape == (hp.nside2npix(nside),)
    occ = np.isfinite(hpx)
    assert occ.sum() > 0
    _, cell_lat = hp.pix2ang(nside, np.where(occ)[0], lonlat=True)
    assert np.nanmax(np.abs(hpx[occ] - cell_lat)) < 3.0


def test_image_to_healpix_sparse_returns_healpixbins():
    """sparse=True returns a HealpixBins NamedTuple (pixels/values/nside);
    it carries the binning nside and covers the dense occupied set."""
    from skyplothelper.healpix import HealpixBins
    data, hdr, _ = _car_allsky_lat_image()
    nside = 32
    hpx = image_to_healpix(data, hdr, nside=nside, sparse=False)
    r = image_to_healpix(data, hdr, nside=nside, sparse=True)
    assert isinstance(r, HealpixBins)
    assert r.nside == nside and r.counts is None
    assert len(r.pixels) == int(np.isfinite(hpx).sum())
    assert np.allclose(r.values, hpx[r.pixels], equal_nan=False)


def test_image_to_healpix_accepts_wcs_and_squeezes_axes():
    """A WCS input and a 4-D (1,1,ny,nx) cube both work."""
    import healpy as hp
    data, _, wcs = _car_allsky_lat_image()
    nside = 16
    hpx = image_to_healpix(data[None, None, :, :], wcs, nside=nside,
                           statistic="sum", sparse=False)
    assert hpx.shape == (hp.nside2npix(nside),)


def test_image_to_healpix_frame_transform_changes_coverage():
    """Binning into galactic moves the occupied cells vs the native frame."""
    data, hdr, _ = _car_allsky_lat_image()
    nside = 32
    native = image_to_healpix(data, hdr, nside=nside, sparse=True)[0]
    galactic = image_to_healpix(data, hdr, nside=nside, frame="galactic",
                                sparse=True)[0]
    assert not np.array_equal(np.sort(native), np.sort(galactic))


def test_image_to_healpix_auto_nside_tracks_pixel_scale():
    """'auto' nside (default) picks a finer grid for a finer image."""
    coarse = image_to_healpix(*_car_allsky_lat_image(cdelt=4.0)[:2],
                              sparse=True)
    fine = image_to_healpix(*_car_allsky_lat_image(cdelt=1.0)[:2], sparse=True)
    # finer image → higher nside → more occupied cells
    assert len(fine[0]) > len(coarse[0])


def test_image_to_healpix_auto_sparse_dense_for_allsky():
    """sparse='auto' (default) returns a dense array for an all-sky image."""
    data, hdr, _ = _car_allsky_lat_image()
    out = image_to_healpix(data, hdr, nside=32)   # auto sparse
    assert isinstance(out, np.ndarray)            # dense full-sky array


def test_image_to_healpix_auto_sparse_sparse_for_small_field():
    """sparse='auto' returns the sparse triple for a small field (avoids a giant
    mostly-empty array, especially with auto nside)."""
    from skyplothelper.healpix import HealpixBins
    data, hdr, _ = _small_field_image()
    out = image_to_healpix(data, hdr)             # auto nside + auto sparse
    assert isinstance(out, HealpixBins)
    assert len(out.pixels) > 0 and len(out.pixels) == len(out.values)
    assert out.nside > 0


def test_image_to_healpix_target_resolution_nside():
    """nside may be a target angular resolution string / Quantity."""
    import astropy.units as u
    data, hdr, _ = _car_allsky_lat_image()
    a = image_to_healpix(data, hdr, nside="2deg", sparse=True)[2]
    b = image_to_healpix(data, hdr, nside=2 * u.deg, sparse=True)[2]
    assert a == b > 0


def test_image_to_healpix_oversample_changes_nside():
    """oversample makes a resolution-derived nside finer (>1) / coarser (<1)."""
    data, hdr, _ = _car_allsky_lat_image()
    base = image_to_healpix(data, hdr, nside="auto", sparse=True)[2]
    finer = image_to_healpix(data, hdr, nside="auto", oversample=4,
                             sparse=True)[2]
    coarser = image_to_healpix(data, hdr, nside="auto", oversample=0.25,
                               sparse=True)[2]
    assert coarser <= base <= finer and finer > coarser


def test_image_to_healpix_return_counts():
    """return_counts adds an aligned coverage map (dense) / counts array
    (sparse); the counts sum to the number of binned image pixels."""
    data, hdr, _ = _car_allsky_lat_image()
    nside = 32
    hpx, cmap = image_to_healpix(data, hdr, nside=nside, sparse=False,
                                 return_counts=True)
    assert cmap.shape == hpx.shape
    # every occupied value cell has count >= 1; empty cells count 0
    assert np.all(cmap[np.isfinite(hpx)] >= 1)
    assert cmap.sum() == np.isfinite(data).sum()
    pix, vals, ns, counts = image_to_healpix(data, hdr, nside=nside,
                                             sparse=True, return_counts=True)
    assert len(counts) == len(pix) and counts.sum() == np.isfinite(data).sum()


def test_bin_data_sparse_accepts_callable_statistic():
    """A callable statistic aggregates each cell's finite values."""
    from skyplothelper.healpix import bin_data_sparse
    rng = np.random.default_rng(0)
    lons = rng.uniform(0, 360, 5000)
    lats = rng.uniform(-60, 60, 5000)
    data = rng.normal(size=5000)
    p90 = lambda v: np.percentile(v, 90)  # noqa: E731
    pix, vals = bin_data_sparse(lons, lats, data, nside=8, statistic=p90)
    # compare a single cell against a manual percentile
    pixel_of = __import__("healpy").ang2pix(8, lons, lats, lonlat=True)
    cell = pix[len(pix) // 2]
    assert np.isclose(vals[len(pix) // 2],
                      np.percentile(data[pixel_of == cell], 90))


def test_bin_data_as_healpix_accepts_callable_statistic():
    """The dense binner also accepts a callable (clean finite values/cell)."""
    rng = np.random.default_rng(1)
    lons = rng.uniform(0, 360, 4000)
    lats = rng.uniform(-60, 60, 4000)
    data = rng.normal(size=4000)
    hpx, *_ = bin_data_as_healpix(lons, lats, data, nside=8,
                                  statistic=lambda v: np.percentile(v, 25))
    assert np.isfinite(hpx).any()


def test_image_to_healpix_callable_statistic():
    """image_to_healpix forwards a callable statistic through to the binner."""
    data, hdr, _ = _car_allsky_lat_image()
    hpx = image_to_healpix(data, hdr, nside=32,
                           statistic=lambda v: np.percentile(v, 90),
                           sparse=False)
    occ = np.isfinite(hpx)
    import healpy as hp
    _, cell_lat = hp.pix2ang(32, np.where(occ)[0], lonlat=True)
    # value-per-cell ~ its latitude (image value), within the pixel scale
    assert np.nanmax(np.abs(hpx[occ] - cell_lat)) < 3.5


def test_nside_from_array_roundtrip_and_errors():
    import healpy as hp
    for nside in (1, 8, 32, 64):
        arr = np.zeros(hp.nside2npix(nside))
        assert nside_from_array(arr) == nside
    with pytest.raises(ValueError):
        nside_from_array(np.zeros(13))        # not 12*nside**2
    with pytest.raises(ValueError):
        nside_from_array(np.zeros(0))


def test_image_to_healpix_rejects_non_2d():
    data = np.zeros((3, 4, 5))
    _, hdr, _ = _car_allsky_lat_image()
    with pytest.raises(ValueError, match="2-D"):
        image_to_healpix(data, hdr, nside=8)


def test_image_to_healpix_rejects_bad_nside():
    data, hdr, _ = _car_allsky_lat_image()
    with pytest.raises(ValueError, match="nside"):
        image_to_healpix(data, hdr, nside="fine")  # no unit -> not a resolution
    with pytest.raises(ValueError, match="sparse"):
        image_to_healpix(data, hdr, nside=16, sparse="maybe")


# ---- SkyCoord inputs (area/density family) ----

def test_healpix_binning_accepts_skycoord():
    """HEALPix is the equal-area answer for sky density, and SkyCoord is the
    natural currency — so the binning family accepts one directly."""
    import numpy as np
    from astropy.coordinates import SkyCoord

    import skyplothelper as sph
    sc = SkyCoord([10.0, 20.0, 30.0], [0.0, 5.0, 10.0], unit="deg")
    from_coord = sph.sources_to_healpix_bins(sc, nside=16)
    from_arrays = sph.sources_to_healpix_bins(
        [10.0, 20.0, 30.0], [0.0, 5.0, 10.0], 16)
    assert np.allclose(np.nan_to_num(from_coord[2]),
                       np.nan_to_num(from_arrays[2]))


def test_bin_data_sparse_accepts_skycoord():
    from astropy.coordinates import SkyCoord

    import skyplothelper as sph
    sc = SkyCoord([10.0, 20.0, 30.0], [0.0, 5.0, 10.0], unit="deg")
    pix, val = sph.bin_data_sparse(sc, data=[1.0, 2.0, 3.0], nside=16)
    assert len(pix) == len(val) == 3


def test_healpix_skycoord_positional_misuse_is_guided():
    """A SkyCoord fills both coordinate slots, so a following positional would
    misbind — that must raise with guidance, not silently bind to `lats`."""
    import pytest
    from astropy.coordinates import SkyCoord

    import skyplothelper as sph
    sc = SkyCoord([10.0], [0.0], unit="deg")
    with pytest.raises(TypeError, match="keyword"):
        sph.sources_to_healpix_bins(sc, 16)


def test_healpix_missing_required_args_raise():
    import pytest

    import skyplothelper as sph
    with pytest.raises(TypeError, match="nside is required"):
        sph.sources_to_healpix_bins([1.0, 2.0], [3.0, 4.0])
