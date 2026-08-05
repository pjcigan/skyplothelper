"""Tests for skyplothelper.visibility (geometric co-visibility regions)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.visibility import (  # noqa: E402
    _circular_coverage,
    _hour_angle_halfwidth_deg,
    _parse_stations,
)

_T = "2026-06-05T09:00:00"

_NORTH = {
    "Mk": {"lat": 19.8, "lon": -155.5},
    "GBT": {"lat": 38.4, "lon": -79.8},
    "VLA": {"lat": 34.1, "lon": -107.6},
}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _ax():
    fig = plt.figure()
    return sph.make_wcs_frame(projection="AIT", center=180, fig=fig)


# --- station parsing --------------------------------------------------------

def test_parse_stations_dict_with_keys_and_defaults():
    sts = _parse_stations({"A": {"lat": 30, "lon": -100}}, el_min=15)
    assert sts[0] == {"name": "A", "lat": 30.0, "lon": -100.0,
                      "min_el": 15.0, "hor_mask": None}


def test_parse_stations_per_station_min_el_overrides():
    sts = _parse_stations({"A": {"lat": 0, "lon": 0, "min_el": 5}}, el_min=20)
    assert sts[0]["min_el"] == 5.0


def test_parse_stations_accepts_earthlocation():
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    loc = EarthLocation.from_geodetic(10 * u.deg, 45 * u.deg)
    sts = _parse_stations({"E": loc}, el_min=12)
    assert sts[0]["lat"] == pytest.approx(45.0, abs=1e-6)
    assert sts[0]["lon"] == pytest.approx(10.0, abs=1e-6)
    assert sts[0]["min_el"] == 12.0


def test_parse_stations_mask_normalized():
    sts = _parse_stations(
        {"A": {"lat": 0, "lon": 0, "hor_mask": [[270, 0, 90], [5, 10, 15]]}},
        el_min=15)
    az, el = sts[0]["hor_mask"]
    assert list(az) == [0, 90, 270]          # sorted by azimuth
    assert list(el) == [10, 15, 5]


# --- caps -------------------------------------------------------------------

def test_covisibility_circles_geometry():
    """Cap center Dec == latitude, radius == 90 - el_min, RA == GAST + lon."""
    caps = sph.covisibility_circles({"A": {"lat": 23.0, "lon": 0.0}},
                                    _T, el_min=10)
    c = caps[0]
    assert c["center"].dec.deg == pytest.approx(23.0, abs=1e-6)
    assert c["radius_deg"] == pytest.approx(80.0, abs=1e-9)
    # RA at lon=0 equals GAST; recompute and compare.
    from skyplothelper.visibility import _gast_deg
    assert c["center"].ra.deg == pytest.approx(_gast_deg(_T) % 360.0, abs=1e-6)


def test_covisibility_circles_longitude_shifts_ra():
    caps = sph.covisibility_circles(
        {"W": {"lat": 0, "lon": -90}, "E": {"lat": 0, "lon": 90}}, _T)
    dra = (caps[1]["center"].ra.deg - caps[0]["center"].ra.deg) % 360.0
    assert dra == pytest.approx(180.0, abs=1e-6)   # 180° longitude apart


# --- region -----------------------------------------------------------------

def test_covisibility_region_full_intersection_subset_of_pairs():
    """All-N intersection ⊆ ≥2 coverage (more stations is more restrictive)."""
    ax = _ax()
    full = sph.covisibility_region(ax, _NORTH, _T, el_min=15)
    pair = sph.covisibility_region(ax, _NORTH, _T, el_min=15, min_stations=2)
    assert not full.is_empty and not pair.is_empty
    assert full.solid_angle["sr"] <= pair.solid_angle["sr"] + 1e-9


def test_covisibility_region_global_spread_is_empty():
    """A globe-spanning set has no instant of common visibility (full
    intersection is empty)."""
    ax = _ax()
    stations = {"Mk": {"lat": 19.8, "lon": -155.5},
                "GBT": {"lat": 38.4, "lon": -79.8},
                "Eff": {"lat": 50.5, "lon": 6.9},
                "ATCA": {"lat": -30.3, "lon": 149.6}}
    reg = sph.covisibility_region(ax, stations, _T, el_min=15)
    assert reg.is_empty


def test_covisibility_region_mask_reduces_area():
    """A high all-azimuth horizon wall shrinks a station's visible cap."""
    ax = _ax()
    base = sph.covisibility_region(ax, _NORTH, _T, el_min=15)
    masked = dict(_NORTH)
    masked["VLA"] = {"lat": 34.1, "lon": -107.6,
                     "hor_mask": [[0, 90, 180, 270], [40, 40, 40, 40]]}
    reg = sph.covisibility_region(ax, masked, _T, el_min=15)
    assert not reg.is_empty
    assert reg.solid_angle["sr"] < base.solid_angle["sr"]


def test_covisibility_region_min_stations_too_high_warns_empty():
    ax = _ax()
    with pytest.warns(UserWarning, match="exceeds"):
        reg = sph.covisibility_region(ax, _NORTH, _T, min_stations=5)
    assert reg.is_empty


def test_covisibility_region_plotly_backend():
    pytest.importorskip("plotly")
    import skyplothelper.plotly as sphpl
    fig = sphpl.make_figure(projection="AIT", center=180)
    reg = sph.covisibility_region(fig, _NORTH, _T, el_min=15)
    assert type(reg.projector).__name__ == "SkyplothelperProjector"
    assert not reg.is_empty


# --- hour-angle window + duration -------------------------------------------

def test_hour_angle_halfwidth_circumpolar_and_never():
    # dec near +lat → circumpolar (full 180° window); far south → never up.
    assert _hour_angle_halfwidth_deg(34, 80, 15) == pytest.approx(180.0)
    assert _hour_angle_halfwidth_deg(34, -80, 15) == pytest.approx(0.0)
    # mid case strictly between
    h = _hour_angle_halfwidth_deg(34, 10, 15)
    assert 0.0 < h < 180.0


def test_circular_coverage_basic():
    # Two identical full-day-fraction windows: intersection == window length.
    # centers in hours, halfwidths in hours, on a 24 h circle.
    cov_all = _circular_coverage([0.0, 0.0], [3.0, 3.0], min_count=2)
    assert cov_all == pytest.approx(6.0, abs=1e-9)     # [-3,3] overlap = 6 h
    # Disjoint windows → zero mutual (count 2) coverage.
    cov_disjoint = _circular_coverage([0.0, 12.0], [2.0, 2.0], min_count=2)
    assert cov_disjoint == pytest.approx(0.0, abs=1e-9)
    # ...but each is covered by >=1.
    cov1 = _circular_coverage([0.0, 12.0], [2.0, 2.0], min_count=1)
    assert cov1 == pytest.approx(8.0, abs=1e-9)


def test_circular_coverage_full_window():
    # A circumpolar (>=12 h half-width) window covers the whole 24 h circle.
    assert _circular_coverage([0.0], [12.0], min_count=1) == pytest.approx(24.0)


def test_covisibility_duration_band_is_declination_band():
    ax = _ax()
    band = sph.covisibility_duration_band(ax, _NORTH, min_hours=1.0, el_min=15)
    assert not band.is_empty
    # A declination band is RA-independent: membership depends only on Dec.
    import numpy as _np
    for dec in _np.linspace(-80, 80, 9):
        vals = [band.contains_point(ra, dec) for ra in (10, 100, 200, 300)]
        assert len(set(vals)) == 1, f"dec={dec} not RA-independent: {vals}"


def test_covisibility_duration_band_higher_min_hours_is_subset():
    ax = _ax()
    short = sph.covisibility_duration_band(ax, _NORTH, min_hours=1.0)
    long = sph.covisibility_duration_band(ax, _NORTH, min_hours=4.0)
    if not long.is_empty:
        assert long.solid_angle["sr"] <= short.solid_angle["sr"] + 1e-9


def test_covisibility_region_no_invalid_value_runtimewarning():
    """k-of-N intersections of nearly-tangent horizon circles used to emit
    'RuntimeWarning: invalid value encountered in intersection' (and left thin
    sliver lobes). The path now silences that numeric noise and buffer(0)s the
    union."""
    import warnings
    stations = {
        "VLA": {"lat": 34.1, "lon": -107.6}, "GBT": {"lat": 38.4, "lon": -79.8},
        "EF":  {"lat": 50.5, "lon": 6.9},    "ATCA": {"lat": -30.3, "lon": 149.6},
        "Hob": {"lat": -42.8, "lon": 147.4}, "Kok": {"lat": 22.1, "lon": -159.7}}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for t in ("2024-06-01T00:00:00", "2024-06-01T06:00:00",
                  "2024-06-01T12:00:00", "2024-06-01T18:00:00"):
            sph.covisibility_region(_ax(), stations, t, min_stations=3)
    assert not [w for w in rec
                if "invalid value encountered" in str(w.message)]


# --- time=None -> current time, and the producer-set label ------------------

def test_covisibility_circles_time_none_uses_now():
    """Omitting time falls back to the current instant (no exception, real caps)."""
    caps = sph.covisibility_circles(_NORTH)
    assert len(caps) == len(_NORTH)
    assert all(c["radius_deg"] == pytest.approx(75.0) for c in caps)  # 90 - 15


def test_covisibility_region_time_none_uses_now():
    reg = sph.covisibility_region(_ax(), _NORTH)
    assert not reg.is_empty          # three continental-US-ish sites overlap now
    assert 0.0 < reg.area_frac < 1.0


def test_covisibility_region_sets_default_label():
    reg = sph.covisibility_region(_ax(), _NORTH, time=_T)
    assert reg.label == "Co-visible"
    reg2 = sph.covisibility_region(_ax(), _NORTH, time=_T, min_stations=2)
    assert reg2.label == "Co-visible (≥2 of 3)"
