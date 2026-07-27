"""Tests for skyplothelper.geometry._parsing."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from skyplothelper.geometry._parsing import (
    _parse_angle,
    _parse_coord,
    _parse_coords,
    _wcs_frame,
)


def test_parse_angle_float():
    assert _parse_angle(5.0) == 5.0


def test_parse_angle_quantity():
    assert _parse_angle(5 * u.deg) == 5.0
    assert _parse_angle(60 * u.arcmin) == pytest.approx(1.0)
    assert _parse_angle(3600 * u.arcsec) == pytest.approx(1.0)


def test_parse_angle_none():
    assert _parse_angle(None) is None


def test_parse_coord_floats():
    lon, lat, shifted = _parse_coord(45.0, 30.0)
    assert lon == 45.0 and lat == 30.0
    assert shifted is False


def test_parse_coord_skycoord():
    """SkyCoord input — second arg becomes the next param (e.g. radius)."""
    coord = SkyCoord(45.0, 30.0, unit="deg", frame="icrs")
    lon, lat, shifted = _parse_coord(coord, lat=10.0)  # lat=10 = "next param"
    assert lon == pytest.approx(45.0)
    assert lat == pytest.approx(30.0)
    assert shifted is True


def test_parse_coords_arrays():
    lons, lats = _parse_coords([10.0, 20.0], [5.0, -5.0])
    assert lons.tolist() == [10.0, 20.0]
    assert lats.tolist() == [5.0, -5.0]


def test_parse_coords_skycoord_array():
    coord = SkyCoord([10.0, 20.0], [5.0, -5.0], unit="deg", frame="icrs")
    lons, lats = _parse_coords(coord)
    np.testing.assert_allclose(lons, [10.0, 20.0])
    np.testing.assert_allclose(lats, [5.0, -5.0])


def test_wcs_frame_none():
    assert _wcs_frame(None) == "icrs"
