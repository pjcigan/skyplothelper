"""Tests for skyplothelper.core.coords."""

import numpy as np
import pytest

from skyplothelper.core.coords import (
    RAcosDEC_err,
    angulardistance,
    dec2sex,
    deg2dms,
    deg2hour,
    dms2deg,
    hour2deg,
    sex2dec,
)

# ---- deg2dms / dms2deg basics ----

def test_deg2dms_simple_list():
    out = deg2dms(1.123456789)
    assert out[0] == 1
    assert out[1] == 7
    assert out[2] == pytest.approx(24.4444, abs=1e-3)


def test_deg2dms_str_basic():
    assert deg2dms(1.123456789, str) == '01:07:24.44'


def test_deg2dms_sign_aware_small_negative():
    """Sign should land on the first non-zero component for list output."""
    out = deg2dms(-0.001)
    assert out == [0, 0, pytest.approx(-3.6, abs=1e-3)]


def test_deg2dms_sign_aware_str_small_negative():
    assert deg2dms(-0.001, str) == '-00:00:03.60'


def test_deg2dms_fp_rollover_str():
    """-12.7 deg should display as -12:42:00.00, not -12:41:59.99..."""
    assert deg2dms(-12.7, str) == '-12:42:00.00'


def test_deg2dms_fp_rollover_list():
    out = deg2dms(-12.7)
    assert out == [-12, 42, pytest.approx(0.0, abs=1e-6)]


# ---- dms2deg parsing ----

def test_dms2deg_str_basic():
    assert dms2deg('01:07:24.44') == pytest.approx(1.12345556, abs=1e-6)


def test_dms2deg_str_negative():
    assert dms2deg('-12:42:00.00') == pytest.approx(-12.7, abs=1e-6)


def test_dms2deg_list_negative_on_seconds():
    assert dms2deg([0, 0, -3.6]) == pytest.approx(-0.001, abs=1e-6)


# ---- Round-trip sanity across the sphere ----

@pytest.mark.parametrize("v", [
    -179.999, -90.5, -45.0, -12.7, -0.001, 0.0, 0.001, 12.7, 45.0, 90.0, 179.999,
])
def test_deg2dms_dms2deg_roundtrip(v):
    out = dms2deg(deg2dms(v))
    assert out == pytest.approx(v, abs=1e-9)


# ---- deg2hour / hour2deg ----

def test_deg2hour_15deg():
    out = deg2hour(15.0)
    assert out == [1, 0, pytest.approx(0.0, abs=1e-9)]


def test_hour2deg_str():
    assert hour2deg('01:00:00.0') == pytest.approx(15.0)


def test_hour2deg_list():
    assert hour2deg([1, 0, 0.0]) == pytest.approx(15.0)


# ---- angulardistance known cases ----

def test_angulardistance_zero():
    assert angulardistance([10.0, 20.0], [10.0, 20.0]) == pytest.approx(0.0, abs=1e-9)


def test_angulardistance_90deg_on_equator():
    assert angulardistance([0.0, 0.0], [90.0, 0.0]) == pytest.approx(90.0, abs=1e-9)


def test_angulardistance_antipodes():
    assert angulardistance([0.0, 0.0], [180.0, 0.0]) == pytest.approx(180.0, abs=1e-9)


def test_angulardistance_small_separation():
    # 1 arcsec along the equator = 1/3600 deg.
    sep = angulardistance([0.0, 0.0], [1.0 / 3600, 0.0])
    assert sep == pytest.approx(1.0 / 3600, abs=1e-9)


def test_angulardistance_pythag_approx_small():
    sep_full = angulardistance([0.0, 0.0], [0.001, 0.001])
    sep_pyth = angulardistance([0.0, 0.0], [0.001, 0.001], pythag_approx=True)
    assert sep_full == pytest.approx(sep_pyth, rel=1e-6)


def test_angulardistance_array_input():
    c1 = np.array([[0.0, 0.0], [10.0, 0.0]])
    c2 = np.array([[90.0, 0.0], [10.0, 90.0]])
    out = angulardistance(c1, c2)
    np.testing.assert_allclose(out, [90.0, 90.0], atol=1e-9)


# ---- dec2sex / sex2dec round trips ----

def test_dec2sex_sex2dec_roundtrip_list():
    lon, lat = dec2sex(83.633, 22.015)
    # round-trip via string form
    s_lon, s_lat = dec2sex(83.633, 22.015, as_string=True)
    out = sex2dec(s_lon, s_lat)
    assert out[0] == pytest.approx(83.633, abs=1e-3)
    assert out[1] == pytest.approx(22.015, abs=1e-3)


def test_dec2sex_custom_delimiter_list():
    # Regression: a custom six-delimiter list used to crash on the
    # delimiter_map.get() lookup (unhashable list). It must now be honored.
    s_lon, s_lat = dec2sex(83.633, 22.015, as_string=True,
                           str_format=['h', 'm', 's', 'd', 'm', 's'])
    assert s_lon.endswith('s') and 'h' in s_lon and 'm' in s_lon
    assert s_lat.endswith('s') and 'd' in s_lat
    # and it still round-trips back to the input degrees
    out = sex2dec(s_lon, s_lat)
    assert out[0] == pytest.approx(83.633, abs=1e-3)
    assert out[1] == pytest.approx(22.015, abs=1e-3)


def test_dec2sex_short_delimiter_list_raises():
    with pytest.raises(ValueError, match="six delimiters"):
        dec2sex(10.0, 20.0, as_string=True, str_format=['a', 'b', 'c'])


def test_dec2sex_fp_rollover_str():
    """-12.7 deg latitude must display as -12:42:00.00, not the old
    -12:41:60.00 (float artifact: 41m 59.9999s rounding up to 60). This is
    the same cascade deg2dms has; the two must agree on the declination."""
    _lon, lat = dec2sex(10.0, -12.7, as_string=True)
    assert lat == '-12:42:00.00'
    assert lat == deg2dms(-12.7, str)


def test_dec2sex_fp_rollover_list():
    """The numeric-list path must cascade the near-60 seconds too."""
    _lon, lat = dec2sex(10.0, -12.7)
    assert [lat[0], lat[1]] == [-12, 42]
    assert lat[2] == pytest.approx(0.0, abs=1e-6)


# ---- RAcosDEC_err sanity ----

def test_RAcosDEC_err_zero_dec_returns_uncRA():
    # At DEC=0, cos(DEC)=1 and sin(DEC)=0, so the result reduces to uncRA.
    out = RAcosDEC_err(180.0, 0.0, uncRA_asec=0.5, uncDEC_asec=0.0)
    assert out == pytest.approx(0.5, abs=1e-9)


def test_RAcosDEC_err_unit_conversion():
    out_asec = RAcosDEC_err(180.0, 0.0, uncRA_asec=0.5, uncDEC_asec=0.0,
                            return_units='arcsec')
    out_mas = RAcosDEC_err(180.0, 0.0, uncRA_asec=0.5, uncDEC_asec=0.0,
                           return_units='mas')
    assert out_mas == pytest.approx(out_asec * 1e3)


# ---- Frame conversion ----

def test_convert_frame_icrs_to_galactic_galcenter():
    """Sgr A* (galactic center) is at l=0, b=0. Round-tripping back to
    ICRS should give the standard catalog position (266.4°, -28.9°)."""
    from skyplothelper.core.coords import convert_frame
    ra, dec = convert_frame(0.0, 0.0, "galactic", "icrs")
    assert ra == pytest.approx(266.4, abs=0.5)
    assert dec == pytest.approx(-28.9, abs=0.5)


def test_convert_frame_round_trip():
    """ICRS → Galactic → ICRS should preserve the input."""
    from skyplothelper.core.coords import convert_frame
    ra_in, dec_in = 180.0, 30.0
    lon, lat = convert_frame(ra_in, dec_in, "icrs", "galactic")
    ra_out, dec_out = convert_frame(lon, lat, "galactic", "icrs")
    assert ra_out == pytest.approx(ra_in, abs=1e-6)
    assert dec_out == pytest.approx(dec_in, abs=1e-6)


def test_convert_frame_array_input():
    """Vectorised input should produce ndarray output of matching shape."""
    from skyplothelper.core.coords import convert_frame
    ras = np.array([10.0, 20.0, 30.0])
    decs = np.array([0.0, 30.0, -20.0])
    lons, lats = convert_frame(ras, decs, "icrs", "galactic")
    assert lons.shape == ras.shape
    assert lats.shape == decs.shape


def test_convert_frame_aliases():
    """Common aliases (gal, ecl, eq) should resolve to the same frame."""
    from skyplothelper.core.coords import convert_frame
    a = convert_frame(180.0, 30.0, "icrs", "galactic")
    b = convert_frame(180.0, 30.0, "eq", "gal")
    assert a[0] == pytest.approx(b[0], abs=1e-9)
    assert a[1] == pytest.approx(b[1], abs=1e-9)


def test_pairwise_helpers_match_convert_frame():
    """The pair-wise helpers should give identical output to the
    general `convert_frame`."""
    from skyplothelper.core.coords import (
        convert_frame,
        galactic_to_icrs,
        icrs_to_ecliptic,
        icrs_to_galactic,
        icrs_to_supergalactic,
    )
    a = icrs_to_galactic(180.0, 30.0)
    b = convert_frame(180.0, 30.0, "icrs", "galactic")
    assert a[0] == pytest.approx(b[0])
    assert a[1] == pytest.approx(b[1])

    a = galactic_to_icrs(45.0, 10.0)
    b = convert_frame(45.0, 10.0, "galactic", "icrs")
    assert a[0] == pytest.approx(b[0])
    assert a[1] == pytest.approx(b[1])

    a = icrs_to_ecliptic(0.0, 0.0)
    b = convert_frame(0.0, 0.0, "icrs", "ecliptic")
    assert a[0] == pytest.approx(b[0])
    assert a[1] == pytest.approx(b[1])

    a = icrs_to_supergalactic(180.0, 30.0)
    b = convert_frame(180.0, 30.0, "icrs", "supergalactic")
    assert a[0] == pytest.approx(b[0])
    assert a[1] == pytest.approx(b[1])

