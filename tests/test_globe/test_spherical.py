"""Tests for skyplothelper.globe.spherical."""

import numpy as np
import pytest

from skyplothelper.globe.spherical import (
    destination_point,
    great_circle_arc,
    great_circle_distance,
    initial_bearing,
    lonlat_to_xyz,
    midpoint,
    orthographic_forward,
    orthographic_inverse,
    orthographic_visibility,
    small_circle,
    xyz_to_lonlat,
)

# ---- Cartesian round-trip ----

@pytest.mark.parametrize("lon, lat", [(0, 0), (45, 30), (-90, -45), (180, 89)])
def test_lonlat_xyz_roundtrip(lon, lat):
    xyz = lonlat_to_xyz(lon, lat)
    lon_back, lat_back = xyz_to_lonlat(xyz)
    # Wrap longitude into the same convention before comparing
    lon_norm = ((lon + 180) % 360) - 180
    lon_back_norm = ((float(lon_back) + 180) % 360) - 180
    assert abs(lon_back_norm - lon_norm) < 1e-9
    assert abs(float(lat_back) - lat) < 1e-9


# ---- Great-circle distance ----

def test_great_circle_distance_zero():
    """With body=None the result is in radians."""
    assert float(great_circle_distance(0, 0, 0, 0)) == pytest.approx(0.0, abs=1e-9)


def test_great_circle_distance_quarter_equator_radians():
    """Equator quarter-circle = pi/2 radians (no body specified)."""
    d_rad = great_circle_distance(0, 0, 90, 0)
    assert float(d_rad) == pytest.approx(np.pi / 2, abs=1e-9)


def test_great_circle_distance_quarter_equator_km_earth():
    """Same arc but in km on Earth = ~10018 km."""
    d_km = great_circle_distance(0, 0, 90, 0, body="earth")
    assert float(d_km) == pytest.approx(6371.0 * np.pi / 2, abs=1.0)


def test_great_circle_arc_endpoints():
    arc = great_circle_arc(0, 0, 90, 0, n_pts=10)
    # Arc returns (lons, lats); start should match (0, 0) and end (90, 0)
    lons, lats = arc
    assert abs(float(lons[0]) - 0) < 1e-6
    assert abs(float(lats[0]) - 0) < 1e-6
    assert abs(float(lons[-1]) - 90) < 1e-6
    assert abs(float(lats[-1]) - 0) < 1e-6


# ---- Midpoint, bearing, destination ----

def test_midpoint_equator():
    lon, lat = midpoint(0, 0, 90, 0)
    assert lon == pytest.approx(45.0, abs=1e-6)
    assert lat == pytest.approx(0.0, abs=1e-6)


def test_initial_bearing_north():
    """From equator going north should give bearing ~0°."""
    bearing = initial_bearing(0, 0, 0, 30)
    assert abs(float(bearing)) < 1e-6


def test_destination_point_smoke():
    lon, lat = destination_point(0, 0, bearing_deg=90, distance_rad=np.pi / 4)
    # 45° east along equator
    assert lat == pytest.approx(0.0, abs=1e-6)
    assert lon == pytest.approx(45.0, abs=1e-6)


def test_small_circle_returns_arrays():
    lons, lats = small_circle(0, 0, radius_deg=10, n_pts=50)
    assert len(lons) == 50 and len(lats) == 50


# ---- Orthographic projection ----

@pytest.mark.parametrize("lon, lat", [(10, 5), (-20, 30), (45, -15)])
def test_orthographic_roundtrip(lon, lat):
    """forward returns (N, 2); inverse returns (N, 2)."""
    xy = orthographic_forward(lon, lat, lon_0=0, lat_0=0)
    x, y = xy[0]
    out = orthographic_inverse(x, y, lon_0=0, lat_0=0)
    lon_back, lat_back = out[0]
    assert abs(float(lon_back) - lon) < 1e-9
    assert abs(float(lat_back) - lat) < 1e-9


def test_orthographic_visibility():
    # Antipode should be invisible
    assert not bool(orthographic_visibility(180, 0, lon_0=0, lat_0=0))
    # Center is visible
    assert bool(orthographic_visibility(0, 0, lon_0=0, lat_0=0))
