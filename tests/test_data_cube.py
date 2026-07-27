"""Tests for skyplothelper.DataCube (shared spectral-cube holder)."""

import matplotlib
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.images.cube import DataCube, MomentMap  # noqa: E402


def _cube(nchan=8, ny=40, nx=60, seed=0):
    """A velocity cube with a full celestial + spectral WCS."""
    data = np.random.RandomState(seed).normal(size=(nchan, ny, nx)).astype(float)
    cards = {
        "NAXIS": 3, "NAXIS1": nx, "NAXIS2": ny, "NAXIS3": nchan,
        "CTYPE1": "RA---SIN", "CRVAL1": 150.0, "CRPIX1": 30.0,
        "CDELT1": -0.001, "CUNIT1": "deg",
        "CTYPE2": "DEC--SIN", "CRVAL2": 2.0, "CRPIX2": 20.0,
        "CDELT2": 0.001, "CUNIT2": "deg",
        "CTYPE3": "VRAD", "CRVAL3": -100000.0, "CRPIX3": 1.0,
        "CDELT3": 20000.0, "CUNIT3": "m/s",
        "BUNIT": "Jy/beam", "RESTFRQ": 1.4204058e9,
    }
    hdr = fits.Header(cards)
    return data, hdr


# ---------------------------------------------------------------------------
# Construction + WCS split
# ---------------------------------------------------------------------------

def test_basic_attributes():
    data, hdr = _cube()
    c = DataCube(data, hdr)
    assert c.shape == (8, 40, 60)
    assert c.nchan == 8 == len(c)
    assert c.axis_kind == "velocity"
    assert c.bunit == "Jy/beam"
    assert c.celestial_wcs is not None and c.spectral is not None
    assert c.world.shape == (8,)


def test_squeeze_degenerate_stokes_axis():
    data, hdr = _cube(nchan=5)
    c = DataCube(data[np.newaxis], hdr)          # (1, nchan, y, x)
    assert c.shape == (5, 40, 60)


def test_rejects_non_cube():
    with pytest.raises(ValueError, match="3-D"):
        DataCube(np.zeros((10, 10)))


def test_no_header_leaves_spectral_none():
    data, _ = _cube()
    c = DataCube(data)
    assert c.spectral is None and c.world is None
    assert c.spectral_label(0) is None


def test_accepts_wcs_as_header():
    data, hdr = _cube()
    c = DataCube(data, WCS(hdr))
    assert c.axis_kind == "velocity"


def test_channel_negative_index():
    data, hdr = _cube()
    c = DataCube(data, hdr)
    assert np.array_equal(c.channel(-1), data[-1])


# ---------------------------------------------------------------------------
# Spectral labels
# ---------------------------------------------------------------------------

def test_spectral_label_velocity():
    data, hdr = _cube()
    c = DataCube(data, hdr)
    assert c.spectral_label(0) == "-100 km/s"    # CRVAL3 = -100000 m/s
    assert c.spectral_label(1) == "-80 km/s"


def test_spectral_label_vsys_offset():
    data, hdr = _cube()
    c = DataCube(data, hdr)
    assert c.spectral_label(0, vsys=-100.0) == "0 km/s"


def test_spectral_label_reinterpret_as_redshift():
    data, hdr = _cube()
    c = DataCube(data, hdr)
    # a velocity axis reinterpreted as redshift (z = v / c), dimensionless
    lab = c.spectral_label(0, mode="redshift")
    assert lab is not None and "km" not in lab
    assert float(lab) == pytest.approx(-100.0 / 299792.458, abs=1e-4)


# ---------------------------------------------------------------------------
# Transforms return new (chained) cubes
# ---------------------------------------------------------------------------

def test_spectral_bin_averages_channels_and_world():
    data, hdr = _cube(nchan=8)
    c = DataCube(data, hdr)
    b = c.spectral_bin(2)
    assert b.shape == (4, 40, 60)
    assert np.allclose(b.channel(0), np.nanmean(data[:2], axis=0))
    assert np.allclose(b.world, c.world.reshape(4, 2).mean(axis=1))
    assert c.shape == (8, 40, 60)                # original untouched


def test_spectral_bin_noop_returns_self():
    c = DataCube(*_cube())
    assert c.spectral_bin(1) is c


def test_spatial_downsample_shape_and_registration():
    data, hdr = _cube()
    c = DataCube(data, hdr)
    d = c.spatial_downsample(2)
    assert d.shape == (8, 20, 30)
    assert np.allclose(d.channel(0), np.nanmean(
        data[0, :40, :60].reshape(20, 2, 30, 2), axis=(1, 3)))
    # the block-averaged WCS keeps the same sky point registered
    w0 = c.celestial_wcs.pixel_to_world_values(29, 19)
    w1 = d.celestial_wcs.pixel_to_world_values((29 - 0.5) / 2, (19 - 0.5) / 2)
    assert np.allclose(w0, w1, atol=1e-9)
    assert np.allclose(d.celestial_wcs.wcs.cdelt,
                       c.celestial_wcs.wcs.cdelt * 2)


def test_spatial_downsample_syncs_header():
    data, hdr = _cube()
    d = DataCube(data, hdr).spatial_downsample(2)
    assert d.header["CDELT1"] == pytest.approx(hdr["CDELT1"] * 2)
    assert d.header["NAXIS1"] == 30 and d.header["NAXIS2"] == 20


def test_spatial_downsample_factor_too_large():
    data, hdr = _cube(ny=6, nx=6)
    with pytest.raises(ValueError, match="larger than"):
        DataCube(data, hdr).spatial_downsample(8)


def test_smooth_returns_cube_same_world():
    data, hdr = _cube()
    c = DataCube(data, hdr)
    s = c.smooth(5)
    assert s.shape == c.shape
    assert np.allclose(s.world, c.world)         # spectral axis unchanged
    assert not np.allclose(s.channel(3), c.channel(3))


def test_chained_transforms():
    data, hdr = _cube()
    small = DataCube(data, hdr).spatial_downsample(2).spectral_bin(2)
    assert small.shape == (4, 20, 30)


# ---------------------------------------------------------------------------
# Reductions + normalization
# ---------------------------------------------------------------------------

def test_moment0_integrates_velocity():
    data, hdr = _cube()
    m = DataCube(data, hdr).moment0()
    assert isinstance(m, MomentMap) and m.order == 0
    assert m.data.shape == (40, 60)
    assert m.units == "Jy/beam m/s"              # BUNIT × native spectral unit
    dv = 20000.0                                 # |Δv| in native m/s
    assert np.allclose(m.data, np.nansum(data, axis=0) * dv)


def test_moment0_plain_sum_without_velocity():
    data, hdr = _cube()
    del hdr["CTYPE3"], hdr["CRVAL3"], hdr["CDELT3"], hdr["CRPIX3"], hdr["CUNIT3"]
    m = DataCube(data, hdr).moment0()
    assert m.units == "Jy/beam"
    assert np.allclose(m.data, np.nansum(data, axis=0))


def test_moment0_unit_conversion():
    data, hdr = _cube()
    m = DataCube(data, hdr).moment0(unit="km/s")
    assert m.units == "Jy/beam km/s"
    assert np.allclose(m.data, np.nansum(data, axis=0) * 20.0)   # dv = 20 km/s


def test_moment1_weighted_velocity():
    data, hdr = _cube()
    c = DataCube(data, hdr)
    m = c.moment1(unit="km/s")
    assert m.order == 1 and m.units == "km/s"
    w = (c.world * 1e-3)                          # m/s → km/s
    expect = np.nansum(data * w[:, None, None], axis=0) / np.nansum(data, axis=0)
    assert np.allclose(m.data, expect, equal_nan=True)


def test_moment1_vsys_offset():
    data, hdr = _cube()
    c = DataCube(data, hdr)
    base = c.moment1(unit="km/s").data
    shifted = c.moment1(unit="km/s", vsys=50.0).data
    assert np.allclose(shifted, base - 50.0, equal_nan=True)


def test_moment2_nonnegative_dispersion():
    data, hdr = _cube()
    m = DataCube(np.abs(data), hdr).moment2(unit="km/s")
    assert m.order == 2 and m.units == "km/s"
    finite = m.data[np.isfinite(m.data)]
    assert np.all(finite >= 0.0)                  # dispersion is >= 0


def test_moment_threshold_masks_low_signal():
    data, hdr = _cube()
    c = DataCube(data, hdr)
    hi = float(np.nanmax(data)) + 1.0            # excludes every voxel
    m = c.moment0(threshold=hi)
    assert np.all(np.isnan(m.data) | (m.data == 0.0))


def test_moment_bad_order_raises():
    with pytest.raises(ValueError, match="order must be 0, 1, or 2"):
        DataCube(*_cube()).moment(3)


def test_moment1_without_spectral_axis_raises():
    data, hdr = _cube()
    for k in ("CTYPE3", "CRVAL3", "CDELT3", "CRPIX3", "CUNIT3"):
        del hdr[k]
    with pytest.raises(ValueError, match="needs a spectral world axis"):
        DataCube(data, hdr).moment1()


def test_vlimits_cached_and_ordered():
    data, hdr = _cube()
    c = DataCube(data, hdr)
    lo, hi = c.vlimits(1, 99)
    assert lo < hi
    assert c.vlimits(1, 99) == (lo, hi)          # cache hit, identical
    assert c.vlimits(1, 99) is not None


# ---------------------------------------------------------------------------
# I/O + interop
# ---------------------------------------------------------------------------

def test_from_fits_roundtrip(tmp_path):
    data, hdr = _cube()
    p = tmp_path / "cube.fits"
    fits.writeto(p, data, hdr)
    c = DataCube.from_fits(str(p))
    assert c.shape == (8, 40, 60) and c.axis_kind == "velocity"


def test_accepts_hdulist():
    data, hdr = _cube()
    hdul = fits.HDUList([fits.PrimaryHDU(data, hdr)])
    assert DataCube(hdul).shape == (8, 40, 60)


def test_duck_typed_spectral_cube():
    """A SpectralCube-like object (has .hdu + .spectral_axis) is read via .hdu."""
    data, hdr = _cube()

    class FakeSpectralCube:
        spectral_axis = np.arange(8)             # marks it SpectralCube-like
        hdu = fits.PrimaryHDU(data, hdr)

    c = DataCube(FakeSpectralCube())
    assert c.shape == (8, 40, 60) and c.axis_kind == "velocity"


def test_top_level_exports():
    assert sph.DataCube is DataCube
    assert sph.MomentMap is MomentMap


# ---------------------------------------------------------------------------
# MomentMap: wrap a user's own map + plot
# ---------------------------------------------------------------------------

def test_momentmap_from_fits(tmp_path):
    data, hdr = _cube()
    mom1 = DataCube(data, hdr).moment1(unit="km/s")
    p = tmp_path / "mom1.fits"
    out = hdr.copy()
    for k in ("CTYPE3", "CRVAL3", "CDELT3", "CRPIX3", "CUNIT3", "NAXIS3"):
        del out[k]
    out["NAXIS"] = 2
    out["BUNIT"] = "km/s"
    fits.writeto(p, mom1.data, out)
    mm = MomentMap.from_fits(str(p), order=1)
    assert mm.order == 1 and mm.units == "km/s"
    assert mm.wcs is not None
    assert np.allclose(mm.data, mom1.data, equal_nan=True)


def test_momentmap_from_fits_rejects_non_2d(tmp_path):
    data, hdr = _cube()
    p = tmp_path / "cube.fits"
    fits.writeto(p, data, hdr)
    with pytest.raises(ValueError, match="must be 2-D"):
        MomentMap.from_fits(str(p), order=0)


def test_momentmap_plot_order1_diverging_symmetric():
    data, hdr = _cube()
    mm = DataCube(data, hdr).moment1(unit="km/s")
    res = mm.plot()
    im = res.image
    lo, hi = im.get_clim()
    center = float(np.median(mm.data[np.isfinite(mm.data)]))
    assert abs((hi + lo) / 2 - center) < 1e-6     # symmetric about the median
    assert im.get_cmap().name.endswith("diff_blueorange")
    plt.close(res.fig)


def test_momentmap_plot_onto_given_axes():
    data, hdr = _cube()
    mm = DataCube(data, hdr).moment0()
    fig = plt.figure()
    ax = fig.add_subplot(projection=mm.wcs)
    res = mm.plot(ax=ax, colorbar=False)
    assert res.ax is ax
    plt.close(fig)


def test_moment_retains_header():
    data, hdr = _cube()
    hdr["OBJECT"] = "DDO 70"
    m = DataCube(data, hdr).moment0()
    assert m.header is not None and m.header["OBJECT"] == "DDO 70"


def test_momentmap_name():
    data, hdr = _cube()
    c = DataCube(data, hdr)
    assert c.moment0().name == "moment 0 (integrated intensity)"
    assert c.moment1().name == "moment 1 (velocity field)"
    assert c.moment2().name == "moment 2 (velocity dispersion)"


def test_momentmap_from_fits_keeps_header(tmp_path):
    data, hdr = _cube()
    out = fits.Header({"NAXIS": 2, "NAXIS1": 60, "NAXIS2": 40,
                       "CTYPE1": "RA---SIN", "CRVAL1": 150.0, "CRPIX1": 30.0,
                       "CDELT1": -0.001, "CUNIT1": "deg",
                       "CTYPE2": "DEC--SIN", "CRVAL2": 2.0, "CRPIX2": 20.0,
                       "CDELT2": 0.001, "CUNIT2": "deg",
                       "BUNIT": "Jy/beam m/s", "OBJECT": "NGC 1"})
    p = tmp_path / "mom0.fits"
    fits.writeto(p, np.nansum(data, axis=0), out)
    mm = MomentMap.from_fits(str(p), order=0)
    assert mm.header["OBJECT"] == "NGC 1"
    assert mm.units == "Jy/beam m/s"


def test_momentmap_plot_center_shifts_range():
    data, hdr = _cube()
    mm = DataCube(data, hdr).moment1(unit="km/s")
    res = mm.plot(center=25.0)
    lo, hi = res.image.get_clim()
    assert abs((hi + lo) / 2 - 25.0) < 1e-6       # symmetric about given center
    plt.close(res.fig)


def test_momentmap_to_fits_roundtrip(tmp_path):
    data, hdr = _cube()
    hdr["OBJECT"] = "DDO 70"
    hdr["BMAJ"] = 5.0 / 3600
    hdr["BMIN"] = 3.0 / 3600
    hdr["BPA"] = 30.0
    m1 = DataCube(data, hdr).moment1(unit="km/s")
    p = tmp_path / "mom1.fits"
    m1.to_fits(str(p))

    back = MomentMap.from_fits(str(p))           # order recovered from MOMORDER
    assert back.order == 1
    assert back.units == "km/s"
    assert back.header["OBJECT"] == "DDO 70"     # provenance grafted
    assert "BMAJ" in back.header                 # beam survives → plot(beam=) works
    assert np.allclose(back.data, m1.data, equal_nan=True)


def test_momentmap_to_fits_overwrite(tmp_path):
    data, hdr = _cube()
    m = DataCube(data, hdr).moment0()
    p = tmp_path / "m0.fits"
    m.to_fits(str(p))
    with pytest.raises(OSError):
        m.to_fits(str(p))                        # exists, no overwrite
    m.to_fits(str(p), overwrite=True)            # ok


def test_momentmap_plot_draws_beam():
    data, hdr = _cube()
    hdr["BMAJ"] = 5.0 / 3600      # deg
    hdr["BMIN"] = 3.0 / 3600
    hdr["BPA"] = 30.0
    mm = DataCube(data, hdr).moment0()
    res = mm.plot(beam="auto")
    # the beam adds at least one patch to the axes
    from matplotlib.patches import Ellipse
    assert any(isinstance(p, Ellipse) for p in res.ax.patches)
    plt.close(res.fig)
