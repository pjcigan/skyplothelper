"""Tests for the VSH forward model (skyplothelper.vsh)."""

import numpy as np
import pytest

from skyplothelper.vsh import (
    VSH_PARAM_NAMES,
    vsh_field,
    vsh_shift_frame,
    vsh_shift_sources,
)


def test_zero_params_zero_field():
    """No coefficients → no field anywhere."""
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    lat = np.array([-45.0, 0.0, 30.0, 80.0])
    dlon, dlat = vsh_field(lon, lat, np.zeros(16))
    assert np.allclose(dlon, 0.0)
    assert np.allclose(dlat, 0.0)


def test_param_names_length_and_order():
    assert len(VSH_PARAM_NAMES) == 16
    assert VSH_PARAM_NAMES[:6] == ('R_1', 'R_2', 'R_3', 'D_1', 'D_2', 'D_3')


def test_r3_spin_at_equator():
    """A pure R_3 spin at the equator points purely west (−α): the
    longitudinal component is −R3·cosδ = −R3 at δ=0, the latitudinal
    component is zero."""
    R3 = 5.0
    dlon, dlat = vsh_field(0.0, 0.0, [0, 0, R3, 0, 0, 0])
    assert dlon == pytest.approx(-R3)
    assert dlat == pytest.approx(0.0)


def test_r1_does_not_move_its_own_axis_point():
    """R_1 spins about the (α=0, δ=0) axis, so that point doesn't move."""
    dlon, dlat = vsh_field(0.0, 0.0, [7.0, 0, 0, 0, 0, 0])
    assert dlon == pytest.approx(0.0)
    assert dlat == pytest.approx(0.0)


def test_r2_rotation_shifts_origin_in_latitude():
    """A pure R_2 rotation carries (0, 0) by R_2 in declination only."""
    theta = 30.0
    lon_s, lat_s = vsh_shift_sources(0.0, 0.0, [0, theta, 0, 0, 0, 0])
    assert lon_s == pytest.approx(0.0)
    assert lat_s == pytest.approx(theta)


def test_glide_at_origin():
    """At (0, 0) a glide (D_1, D_2, D_3) gives field (D_2, D_3)."""
    D1, D2, D3 = 2.0, 3.0, 4.0
    dlon, dlat = vsh_field(0.0, 0.0, [0, 0, 0, D1, D2, D3])
    assert dlon == pytest.approx(D2)
    assert dlat == pytest.approx(D3)


def test_length6_matches_length16_zero_filled():
    """A length-6 (ℓ=1) call equals a length-16 call with ℓ=2 zeroed."""
    p6 = [1.0, -2.0, 3.0, 0.5, -0.5, 0.25]
    p16 = list(p6) + [0.0] * 10
    lon = np.linspace(0, 350, 12)
    lat = np.linspace(-80, 80, 12)
    a1 = vsh_field(lon, lat, p6)
    a2 = vsh_field(lon, lat, p16)
    assert np.allclose(a1[0], a2[0])
    assert np.allclose(a1[1], a2[1])


def test_dict_params_equivalent_to_sequence():
    seq = np.arange(1.0, 17.0)
    d = dict(zip(VSH_PARAM_NAMES, seq))
    lon = np.array([10.0, 200.0])
    lat = np.array([-20.0, 50.0])
    a_seq = vsh_field(lon, lat, seq)
    a_dict = vsh_field(lon, lat, d)
    assert np.allclose(a_seq[0], a_dict[0])
    assert np.allclose(a_seq[1], a_dict[1])


def test_dict_missing_keys_default_zero():
    """A dict with only some keys treats the rest as zero."""
    lon, lat = 45.0, 10.0
    full = vsh_field(lon, lat, {'R_3': 6.0})
    expected = vsh_field(lon, lat, [0, 0, 6.0, 0, 0, 0])
    assert np.allclose(full, expected)


def test_quadrupole_terms_contribute():
    """An ℓ=2 term produces a non-zero field where ℓ=1 alone would be
    zero — confirms the quadrupole block is wired in."""
    p = np.zeros(16)
    p[VSH_PARAM_NAMES.index('E_20')] = 5.0
    dlon, dlat = vsh_field(30.0, 40.0, p)
    # E_20 enters Δδ as E_20·sin(2δ); non-zero at δ=40.
    assert dlat == pytest.approx(5.0 * np.sin(np.radians(80.0)))


def test_invalid_param_length_raises():
    with pytest.raises(ValueError, match="length 6"):
        vsh_field(0.0, 0.0, [1, 2, 3])


def test_field_preserves_input_shape():
    lon = np.zeros((3, 4))
    lat = np.zeros((3, 4))
    dlon, dlat = vsh_field(lon, lat, np.ones(16))
    assert dlon.shape == (3, 4)
    assert dlat.shape == (3, 4)


def test_shift_wraps_lon_and_clips_lat():
    """vsh_shift_sources keeps lon in [0,360) and lat in [-90,90]."""
    lon_s, lat_s = vsh_shift_sources(359.0, 88.0, [0, 0, 0, 0, 0, 20.0],
                                     scale=1.0)
    assert 0.0 <= float(lon_s) < 360.0
    assert -90.0 <= float(lat_s) <= 90.0


def test_shift_frame_shapes_and_grid():
    """vsh_shift_frame returns four flat arrays of n_lon*n_lat positions,
    on a regular grid, with the shift applied."""
    n_lon, n_lat = 8, 5
    lon, lat, lon_s, lat_s = vsh_shift_frame(
        [0, 0, 10.0, 0, 0, 0], n_lon=n_lon, n_lat=n_lat)
    assert lon.shape == lat.shape == lon_s.shape == lat_s.shape
    assert lon.size == n_lon * n_lat
    # Grid covers the full longitude range and is symmetric in latitude.
    assert np.isclose(lon.min(), 0.0)
    assert np.isclose(lat.min(), -85.0) and np.isclose(lat.max(), 85.0)
    # A non-zero R_3 must actually move at least some sources.
    assert not np.allclose(lon, lon_s)


def test_shift_frame_zero_params_is_identity():
    lon, lat, lon_s, lat_s = vsh_shift_frame(np.zeros(6), n_lon=6, n_lat=5)
    assert np.allclose(lon, lon_s)
    assert np.allclose(lat, lat_s)


def test_shift_frame_matches_manual_shift_sources():
    """vsh_shift_frame is exactly vsh_shift_sources over its grid."""
    params = [1.0, -2.0, 3.0, 0.5, 0.0, -0.5]
    lon, lat, lon_s, lat_s = vsh_shift_frame(params, n_lon=6, n_lat=5,
                                             scale=2.0)
    exp_lon, exp_lat = vsh_shift_sources(lon, lat, params, scale=2.0)
    assert np.allclose(lon_s, exp_lon)
    assert np.allclose(lat_s, exp_lat)
