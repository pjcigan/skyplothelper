"""Tests for skyplothelper.globe.nightshade — the day/night terminator blend.

Focus is the physical ``blend='elevation'`` mode (solar-elevation field +
transfer curves); the legacy ``blend='gaussian'`` path is exercised only for
its cartopy/scipy guard, since it depends on optional packages.
"""

import datetime

import matplotlib
import pytest

matplotlib.use("Agg")

import numpy as np

from skyplothelper.globe import nightshade
from skyplothelper.globe.nightshade import (
    _solar_elevation_field,
    _subsolar_lonlat,
    _terminator_alpha,
    make_nightshade_blend,
)

# A fixed instant so terminator geometry is deterministic: northern summer
# solstice, ~noon UTC (sub-solar point near lon=0, dec≈+23.4°).
_DATE = datetime.datetime(2024, 6, 21, 12, 0, 0)


def _flat_image(h=60, w=120):
    """A uniform mid-gray Plate Carrée RGB image in [0, 1]."""
    return np.full((h, w, 3), 0.5, dtype=float)


# ---- sub-solar point + elevation field ----

def test_subsolar_point_solstice_declination():
    lon0, dec = _subsolar_lonlat(_DATE)
    # At the June solstice the sub-solar latitude is the obliquity, ~+23.44°.
    assert dec == pytest.approx(23.44, abs=0.1)
    assert -180.0 <= lon0 <= 180.0


def test_solar_elevation_field_extremes():
    lon0, dec = _subsolar_lonlat(_DATE)
    # Overhead at the sub-solar point (+90°), grazing on the terminator (~0°),
    # and below the horizon at the antisolar point (-90°).
    lon = np.array([[lon0, lon0 + 90.0, lon0 + 180.0]])
    lat = np.array([[dec], [0.0], [-dec]])
    elev = _solar_elevation_field(lon, lat, lon0, dec)
    assert elev[0, 0] == pytest.approx(90.0, abs=1e-6)      # sub-solar
    assert elev.min() == pytest.approx(-90.0, abs=1e-6)     # antisolar


# ---- transfer curves ----

@pytest.mark.parametrize("curve", ["linear", "smoothstep", "twilight"])
def test_terminator_alpha_bounds_and_endpoints(curve):
    elev = np.linspace(20, -40, 200)
    a = _terminator_alpha(elev, h_day=0.0, h_night=-18.0,
                          curve=curve, twilight_decay=6.0)
    assert a.min() >= 0.0 and a.max() <= 1.0
    # Full day above h_day → transparent; full night below h_night → opaque.
    assert a[elev >= 0.0].max() == pytest.approx(0.0, abs=1e-9)
    assert a[elev <= -18.0].min() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("curve", ["linear", "smoothstep", "twilight"])
def test_terminator_alpha_monotonic(curve):
    # Opacity must rise monotonically as the Sun sinks. elev decreases with
    # index, so alpha must be non-decreasing along the array (diff >= 0).
    elev = np.linspace(5, -25, 300)
    a = _terminator_alpha(elev, 0.0, -18.0, curve, 6.0)
    assert np.all(np.diff(a) >= -1e-12)


def test_curves_differ_in_twilight_band():
    elev = np.linspace(-1, -17, 50)  # strictly inside the band
    lin = _terminator_alpha(elev, 0.0, -18.0, "linear", 6.0)
    smo = _terminator_alpha(elev, 0.0, -18.0, "smoothstep", 6.0)
    twi = _terminator_alpha(elev, 0.0, -18.0, "twilight", 6.0)
    assert not np.allclose(lin, smo)
    assert not np.allclose(smo, twi)


def test_twilight_decay_knob_sharpens_band():
    # Larger twilight_decay → the night side reaches opacity sooner (a
    # thinner, more terminator-hugging bright band), so at a fixed elevation
    # just past the terminator the alpha is higher.
    elev = np.array([-3.0])
    soft = _terminator_alpha(elev, 0.0, -18.0, "twilight", 2.0)
    sharp = _terminator_alpha(elev, 0.0, -18.0, "twilight", 12.0)
    assert sharp[0] > soft[0]


def test_terminator_alpha_invalid_curve_raises():
    with pytest.raises(ValueError, match="curve must be"):
        _terminator_alpha(np.zeros(3), 0.0, -18.0, "banana", 6.0)


def test_terminator_alpha_inverted_band_raises():
    with pytest.raises(ValueError, match="must be greater"):
        _terminator_alpha(np.zeros(3), h_day=-18.0, h_night=0.0,
                          curve="linear", twilight_decay=6.0)


# ---- make_nightshade_blend (elevation mode) ----

@pytest.mark.parametrize("curve", ["linear", "smoothstep", "twilight"])
def test_elevation_blend_shape_and_range(curve):
    img = _flat_image()
    rgba = make_nightshade_blend(img, _DATE, blend="elevation", curve=curve)
    assert rgba.shape == (img.shape[0], img.shape[1], 4)
    assert rgba.min() >= 0.0 and rgba.max() <= 1.0
    # RGB carried through unchanged; only alpha is synthesized.
    assert np.allclose(rgba[:, :, :3], img)


def test_elevation_blend_day_is_transparent_night_opaque():
    img = _flat_image()
    lon0, dec = _subsolar_lonlat(_DATE)
    lon = np.linspace(-180, 180, img.shape[1])[None, :]
    lat = np.linspace(90, -90, img.shape[0])[:, None]
    elev = _solar_elevation_field(lon, lat, lon0, dec)
    rgba = make_nightshade_blend(img, _DATE, blend="elevation",
                                 curve="smoothstep")
    alpha = rgba[:, :, 3]
    assert alpha[elev > 1.0].max() == pytest.approx(0.0, abs=1e-9)
    assert alpha[elev < -19.0].min() == pytest.approx(1.0, abs=1e-9)


def test_elevation_blend_day_blend_inverts_alpha():
    img = _flat_image()
    night = make_nightshade_blend(img, _DATE, blend="elevation")
    day = make_nightshade_blend(img, _DATE, blend="elevation", day_blend=True)
    assert np.allclose(night[:, :, 3], 1.0 - day[:, :, 3])


def test_elevation_blend_needs_no_cartopy_or_scipy(monkeypatch):
    # The whole point of the physical mode: it must work even with cartopy
    # and scipy unavailable.
    monkeypatch.setattr(nightshade, "_HAS_CARTOPY", False)
    monkeypatch.setattr(nightshade, "_HAS_SCIPY", False)
    rgba = make_nightshade_blend(_flat_image(), _DATE, blend="elevation")
    assert rgba.shape[2] == 4


def test_make_nightshade_blend_invalid_blend_raises():
    with pytest.raises(ValueError, match="blend must be"):
        make_nightshade_blend(_flat_image(), _DATE, blend="sparkle")


def test_gaussian_mode_still_requires_cartopy(monkeypatch):
    # The default gaussian path keeps its cartopy dependency.
    monkeypatch.setattr(nightshade, "_HAS_CARTOPY", False)
    with pytest.raises(ImportError, match="cartopy"):
        make_nightshade_blend(_flat_image(), _DATE, blend="gaussian")


@pytest.mark.skipif(
    not (nightshade._HAS_CARTOPY and nightshade._HAS_SCIPY),
    reason="gaussian blend needs cartopy + scipy")
def test_gaussian_default_terminator_is_smooth_not_stepped():
    # Regression: hard_terminator_edge defaults to False so the Gaussian
    # smooths the day/night mask symmetrically — no abrupt step at the
    # terminator. The opt-in hard edge re-introduces a larger jump (it slams
    # the night side to full opacity right at the line).
    img = _flat_image(180, 360)
    soft = make_nightshade_blend(img, _DATE, blend="gaussian")[:, :, 3]
    hard = make_nightshade_blend(img, _DATE, blend="gaussian",
                                 hard_terminator_edge=True)[:, :, 3]
    soft_step = float(np.abs(np.diff(soft, axis=1)).max())
    hard_step = float(np.abs(np.diff(hard, axis=1)).max())
    assert soft_step < 0.1          # default: no abrupt terminator line
    assert hard_step > soft_step    # the hard edge adds a step discontinuity


@pytest.mark.skipif(
    not (nightshade._HAS_CARTOPY and nightshade._HAS_SCIPY),
    reason="gaussian blend needs cartopy + scipy")
def test_gaussian_blend_no_geos_to_path_deprecation():
    """The gaussian path used cartopy's deprecated geos_to_path (2 warnings per
    call); it now traces the Nightshade polygon's coords directly, so the call
    emits no geos_to_path DeprecationWarning."""
    import warnings
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        make_nightshade_blend(_flat_image(), _DATE, blend="gaussian")
    assert not [w for w in rec if "geos_to_path" in str(w.message)]


def test_pseudofits_from_image_accepts_an_array():
    """make_nightshade_blend returns an in-memory RGBA array, so requiring a
    file path meant the one raster you'd want to drape was the one thing that
    couldn't be fed in — the caller had to round-trip through a temp file."""
    import numpy as np

    import skyplothelper as sph
    rgba = np.zeros((32, 64, 4), dtype=float)
    hdu = sph.pseudofits_from_image(rgba, geo=True)
    assert hdu.data.shape == (32, 64, 4)
    assert hdu.header["CTYPE2"].startswith("TLON")


def test_pseudofits_from_image_array_matches_path_form():
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    import skyplothelper as sph
    img = np.random.default_rng(0).random((16, 32, 3))
    from_array = sph.pseudofits_from_image(img)
    path = os.path.join(os.path.dirname(__file__), "_tmp_pseudofits.png")
    plt.imsave(path, img)
    try:
        from_path = sph.pseudofits_from_image(path)
        for key in ("CTYPE2", "CTYPE3", "CDELT2", "CDELT3", "CRVAL2"):
            assert from_array.header[key] == from_path.header[key]
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_pseudofits_from_image_rejects_2d_with_guidance():
    import numpy as np
    import pytest

    import skyplothelper as sph
    with pytest.raises(ValueError, match="RGB"):
        sph.pseudofits_from_image(np.zeros((10, 10)))


@pytest.mark.parametrize("maker", [
    lambda: __import__("datetime").datetime(2026, 7, 18, 6, 30, 0),
    lambda: __import__("astropy.time", fromlist=["Time"]).Time(
        "2026-07-18T06:30:00", scale="utc"),
    lambda: "2026-07-18T06:30:00",
])
def test_nightshade_accepts_time_datetime_and_string(maker):
    """astropy Time must work wherever a datetime does.

    The package leans on astropy everywhere else, and a FITS DATE-OBS parses
    straight into a Time — requiring a datetime here made the astropy-native
    spelling the one that failed.
    """
    import numpy as np

    import skyplothelper as sph
    from skyplothelper.globe.nightshade import _julian_day

    when = maker()
    ref = _julian_day(__import__("datetime").datetime(2026, 7, 18, 6, 30, 0))
    assert abs(_julian_day(when) - ref) < 1.2e-8      # within ~1 ms
    blend = sph.make_nightshade_blend(np.zeros((16, 32, 3)), when)
    assert blend.shape == (16, 32, 4)
