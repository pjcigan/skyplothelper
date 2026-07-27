"""Tests for ``sph.to_time`` and the entry points wired through it.

These assert on *values* — JD agreement and the written ``DATE-OBS`` card —
not merely on the absence of an exception. A time helper that silently picks
the wrong scale raises nothing; it just writes an answer that is ~69 s off.
"""

from __future__ import annotations

import datetime as dt

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.time import Time

import skyplothelper as sph
from skyplothelper._timeinput import _to_datetime

REF = "2026-07-19T12:00:00.000"
EQUIVALENT = [
    ("iso", REF),
    ("datetime", dt.datetime(2026, 7, 19, 12, 0, 0)),
    ("time", Time(REF)),
]


# --- the primitive ----------------------------------------------------------

@pytest.mark.parametrize("label,value", EQUIVALENT)
def test_equivalent_inputs_agree_exactly(label, value):
    """Every accepted spelling of the same instant lands on the same JD."""
    assert sph.to_time(value).jd == pytest.approx(Time(REF).jd, abs=1e-9)


def test_returns_a_time():
    assert isinstance(sph.to_time(REF), Time)


def test_time_passes_through_unconverted():
    """The default is a lossless read-out — scale must survive."""
    t = Time(REF, scale="tt")
    assert sph.to_time(t).scale == "tt"


def test_scale_converts_when_asked():
    t = Time(REF, scale="tt")
    out = sph.to_time(t, scale="utc")
    assert out.scale == "utc"
    # Converted, not relabeled: the clock reading actually moves. Compared
    # against t.utc rather than a hard 69 s, which leap seconds would break.
    assert out.isot == t.utc.isot
    assert out.isot != t.isot


def test_bare_number_is_rejected():
    """60000 could be MJD, JD or a decimal year — guessing is a silent error."""
    with pytest.raises(TypeError, match="ambiguous numeric time"):
        sph.to_time(60000)


def test_bare_number_error_names_the_fix():
    with pytest.raises(TypeError, match="plain_format"):
        sph.to_time(60000)


@pytest.mark.parametrize("fmt,value,expected_mjd", [
    ("mjd", 60000, 60000.0),
    ("jd", 2460000.5, 60000.0),
])
def test_plain_format_reads_numbers(fmt, value, expected_mjd):
    assert sph.to_time(value, plain_format=fmt).mjd == pytest.approx(expected_mjd)


def test_plain_format_accepts_arrays():
    t = sph.to_time(np.array([60000, 60001]), plain_format="mjd")
    assert list(t.mjd) == [60000.0, 60001.0]


def test_plain_format_ignored_for_non_plain_input():
    """Redundant but harmless — erroring here would be worse ergonomics."""
    assert sph.to_time(REF, plain_format="mjd").isot == REF


def test_unparseable_input_still_raises():
    with pytest.raises(Exception):
        sph.to_time("not-a-date")


# --- the datetime adapter ---------------------------------------------------

@pytest.mark.parametrize("label,value", EQUIVALENT)
def test_to_datetime_returns_stdlib_datetime(label, value):
    out = _to_datetime(value, caller="test")
    assert isinstance(out, dt.datetime)
    assert out.replace(tzinfo=None) == dt.datetime(2026, 7, 19, 12, 0, 0)


def test_to_datetime_converts_scale():
    """Consumers that duck-type on datetime have no scale of their own, so a
    TT input has to be converted rather than handed over as-is."""
    out = _to_datetime(Time(REF, scale="tt"), caller="test")
    assert out.replace(tzinfo=None) != dt.datetime(2026, 7, 19, 12, 0, 0)


def test_to_datetime_error_carries_the_caller():
    with pytest.raises(TypeError, match="sph.demo"):
        _to_datetime(60000, caller="sph.demo")


# --- wired entry points -----------------------------------------------------

@pytest.mark.parametrize("label,value", EQUIVALENT)
def test_frame_builders_write_the_same_date_obs(label, value):
    """The two builders disagreed on scale before this helper existed."""
    _, hdr = sph.make_globe_frame(obstime=value, return_header=True)
    plt.close("all")
    ax = sph.make_wcs_frame(obstime=value)
    wcs_dateobs = ax.wcs.wcs.dateobs
    plt.close("all")
    assert hdr["DATE-OBS"] == wcs_dateobs


def test_date_obs_is_utc_not_the_input_scale():
    """FITS defines DATE-OBS as UTC; a TT Time must be converted (~69 s)."""
    t_tt = Time(REF, scale="tt")
    _, hdr = sph.make_globe_frame(obstime=t_tt, return_header=True)
    plt.close("all")
    ax = sph.make_wcs_frame(obstime=t_tt)
    wcs_dateobs = ax.wcs.wcs.dateobs
    plt.close("all")
    assert hdr["DATE-OBS"] == t_tt.utc.isot
    assert wcs_dateobs == t_tt.utc.isot
    assert hdr["DATE-OBS"] != t_tt.isot


@pytest.mark.parametrize("label,value", EQUIVALENT)
def test_nightshade_elevation_accepts_every_type(label, value):
    rgb = np.zeros((8, 16, 3))
    out = sph.make_nightshade_blend(rgb, value, blend="elevation")
    assert out.shape[:2] == (8, 16)


@pytest.mark.parametrize("label,value", EQUIVALENT)
def test_nightshade_gaussian_accepts_every_type(label, value):
    """The gaussian branch handed the raw value to cartopy, which duck-types
    on ``.utcoffset()`` — so Time and str raised AttributeError."""
    pytest.importorskip("cartopy")
    pytest.importorskip("scipy")
    rgb = np.zeros((8, 16, 3))
    out = sph.make_nightshade_blend(rgb, value, blend="gaussian")
    assert out.shape[:2] == (8, 16)


@pytest.mark.parametrize("label,value", EQUIVALENT)
def test_covisibility_accepts_every_type(label, value):
    stations = [{"name": "A", "lon": 0.0, "lat": 45.0},
                {"name": "B", "lon": 100.0, "lat": 30.0}]
    circles = sph.covisibility_circles(stations, value)
    assert circles[0]["radius_deg"] == pytest.approx(75.0)


def test_covisibility_agrees_across_types():
    """The refactor must not move the answer."""
    stations = [{"name": "A", "lon": 0.0, "lat": 45.0}]
    ras = [sph.covisibility_circles(stations, v)[0]["center"].ra.deg
           for _label, v in EQUIVALENT]
    assert ras[0] == pytest.approx(ras[1], abs=1e-9)
    assert ras[0] == pytest.approx(ras[2], abs=1e-9)
