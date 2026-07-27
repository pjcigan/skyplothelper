"""Tests for skyplothelper.core.math_utils."""

import numpy as np
import pytest

from skyplothelper.core.math_utils import (
    map_to_newrange,
    rescale_data_range,
    wrap_24hr,
    wrap_360,
    wrap_center_pmrange,
    wrap_pm90,
    wrap_pm180,
    wrap_pmPI,
    wrap_range,
)

# ---- wrap_360 ----

@pytest.mark.parametrize("x, expected", [
    (-180.0, 180.0),
    (0.0, 0.0),
    (180.0, 180.0),
    (360.0, 0.0),
    (540.0, 180.0),
    (-360.0, 0.0),
    (720.0, 0.0),
])
def test_wrap_360_boundaries(x, expected):
    assert float(wrap_360(x)) == pytest.approx(expected)


def test_wrap_360_array():
    out = wrap_360(np.array([-180.0, 0.0, 180.0, 360.0, 540.0]))
    np.testing.assert_allclose(out, [180.0, 0.0, 180.0, 0.0, 180.0])


# ---- wrap_pm180 ----

@pytest.mark.parametrize("x, expected", [
    (-180.0, -180.0),
    (0.0, 0.0),
    (180.0, -180.0),     # wraps because the upper bound is exclusive
    (360.0, 0.0),
    (540.0, -180.0),
    (-540.0, -180.0),
    (190.0, -170.0),
])
def test_wrap_pm180_boundaries(x, expected):
    assert float(wrap_pm180(x)) == pytest.approx(expected)


# ---- wrap_pm90 ----

@pytest.mark.parametrize("x, expected", [
    (-90.0, -90.0),
    (0.0, 0.0),
    (90.0, -90.0),
    (180.0, 0.0),
    (270.0, -90.0),
    (-180.0, 0.0),
])
def test_wrap_pm90_boundaries(x, expected):
    assert float(wrap_pm90(x)) == pytest.approx(expected)


# ---- wrap_pmPI ----

def test_wrap_pmPI_zero():
    assert float(wrap_pmPI(0.0)) == pytest.approx(0.0)


def test_wrap_pmPI_2pi():
    assert float(wrap_pmPI(2 * np.pi)) == pytest.approx(0.0)


# ---- wrap_24hr ----

def test_wrap_24hr_simple():
    assert float(wrap_24hr(25.0)) == pytest.approx(1.0)
    assert float(wrap_24hr(-1.0)) == pytest.approx(23.0)


def test_wrap_24hr_component_form():
    # Lazy-imports dms2deg from coords; verify it works end-to-end.
    # 1h 30m 0s = 1.5 hour.
    assert float(wrap_24hr([1, 30, 0.0], component=True)) == pytest.approx(1.5)


# ---- wrap_range / wrap_center_pmrange ----

def test_wrap_range_basic():
    assert float(wrap_range(13.0, 0.0, 12.0)) == pytest.approx(1.0)


def test_wrap_center_pmrange_basic():
    assert float(wrap_center_pmrange(190.0, 0.0, 180.0)) == pytest.approx(-170.0)


# ---- map_to_newrange / rescale_data_range ----

def test_map_to_newrange_celsius():
    # The docstring example.
    assert float(map_to_newrange(98.6, [32, 212], [0, 100])) == pytest.approx(37.0)


def test_map_to_newrange_array():
    out = map_to_newrange(np.array([0.0, 0.5, 1.0]), [0.0, 1.0], [10.0, 20.0])
    np.testing.assert_allclose(out, [10.0, 15.0, 20.0])


def test_rescale_data_range_default():
    out = rescale_data_range(np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(out, [0.0, 1.0 / 3, 2.0 / 3, 1.0])


def test_rescale_data_range_custom():
    out = rescale_data_range(np.array([0.0, 5.0, 10.0]), newmin=-1.0, newmax=1.0)
    np.testing.assert_allclose(out, [-1.0, 0.0, 1.0])
