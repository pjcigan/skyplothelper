"""Tests for skyplothelper.catalog spatial-search utilities."""

import matplotlib

matplotlib.use("Agg")

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

import skyplothelper as sph
from skyplothelper.catalog import _resolve_center, _SkyCircle, _to_deg

# A small catalog: two sources very near (10, 5), one ~1 deg away, one far.
RA = np.array([10.0, 10.1, 11.0, 200.0])
DEC = np.array([5.0, 5.05, 5.0, -10.0])
NAMES = ["a", "b", "c", "d"]


# --- helpers -------------------------------------------------------------

def test_to_deg_units():
    assert _to_deg(1.0) == pytest.approx(1.0)
    assert _to_deg(60.0, "arcmin") == pytest.approx(1.0)
    assert _to_deg(3600.0, "arcsec") == pytest.approx(1.0)
    assert _to_deg(1 * u.arcmin) == pytest.approx(1 / 60)
    with pytest.raises(ValueError):
        _to_deg(1.0, "parsec")


def test_resolve_center_forms():
    assert _resolve_center((10.0, 5.0)) == (10.0, 5.0)
    ra, dec = _resolve_center(SkyCoord(10 * u.deg, 5 * u.deg))
    assert ra == pytest.approx(10.0) and dec == pytest.approx(5.0)


def test_skycircle_contains_points_is_angular():
    """Membership is true angular separation, not a projected shape."""
    circ = _SkyCircle(10.0, 5.0, 0.2)
    out = circ.contains_points(RA, DEC)
    assert out.tolist() == [True, True, False, False]


# --- cone_search input types ---------------------------------------------

def test_cone_search_raw_arrays_returns_mask():
    m = sph.cone_search((RA, DEC), (10.0, 5.0), 0.2)
    assert isinstance(m, np.ndarray) and m.dtype == bool
    assert m.tolist() == [True, True, False, False]


def test_cone_search_table_subset_and_autodetect():
    t = Table({"RAJ2000": RA, "DEJ2000": DEC, "name": NAMES})
    out = sph.cone_search(t, (10.0, 5.0), 0.2)
    assert isinstance(out, Table)
    assert sorted(out["name"].tolist()) == ["a", "b"]


def test_cone_search_table_separation_and_sort():
    t = Table({"RAJ2000": RA, "DEJ2000": DEC, "name": NAMES})
    out = sph.cone_search(t, (10.0, 5.0), 0.2, add_separation=True, sort=True)
    assert "separation" in out.colnames
    # nearest first: 'a' (sep 0) before 'b'
    assert out["name"].tolist() == ["a", "b"]
    assert out["separation"][0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(out["separation"]) >= 0)


def test_cone_search_dataframe_subset():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"ra": RA, "dec": DEC, "name": NAMES})
    out = sph.cone_search(df, (10.0, 5.0), 0.2)
    assert isinstance(out, pd.DataFrame)
    assert sorted(out["name"].tolist()) == ["a", "b"]


def test_cone_search_dataframe_separation_column():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"ra": RA, "dec": DEC, "name": NAMES})
    out = sph.cone_search(df, (10.0, 5.0), 0.2, add_separation=True, sort=True)
    assert "separation" in out.columns
    assert out["name"].tolist() == ["a", "b"]


def test_cone_search_skycoord_returns_skycoord():
    sc = SkyCoord(RA * u.deg, DEC * u.deg)
    out = sph.cone_search(sc, (10.0, 5.0), 0.2)
    assert isinstance(out, SkyCoord)
    assert len(out) == 2


def test_cone_search_unit_knob_and_quantity():
    # 12 arcmin = 0.2 deg: same result as the 0.2-deg default
    m1 = sph.cone_search((RA, DEC), (10.0, 5.0), 12.0, unit="arcmin")
    m2 = sph.cone_search((RA, DEC), (10.0, 5.0), 0.2 * u.deg)
    assert m1.tolist() == m2.tolist() == [True, True, False, False]


def test_cone_search_explicit_columns():
    t = Table({"my_ra": RA, "my_dec": DEC, "name": NAMES})
    out = sph.cone_search(t, (10.0, 5.0), 0.2, ra_col="my_ra", dec_col="my_dec")
    assert sorted(out["name"].tolist()) == ["a", "b"]


def test_cone_search_column_autodetect_failure_raises():
    t = Table({"x": RA, "y": DEC})
    with pytest.raises(ValueError, match="auto-detect"):
        sph.cone_search(t, (10.0, 5.0), 0.2)


def test_cone_search_return_mask_on_table():
    t = Table({"ra": RA, "dec": DEC})
    m = sph.cone_search(t, (10.0, 5.0), 0.2, return_mask=True)
    assert isinstance(m, np.ndarray) and m.tolist() == [True, True, False, False]


# --- region_search -------------------------------------------------------

class _ProtocolRegion:
    """Minimal object satisfying the contains_points protocol."""

    center_ra, center_dec = 10.0, 5.0

    def contains_points(self, ra, dec=None):
        return np.asarray(ra, float) < 11.0


def test_region_search_with_protocol_object():
    t = Table({"ra": RA, "dec": DEC, "name": NAMES})
    out = sph.region_search(t, _ProtocolRegion())
    assert sorted(out["name"].tolist()) == ["a", "b"]


def test_region_search_separation_uses_region_center():
    t = Table({"ra": RA, "dec": DEC, "name": NAMES})
    out = sph.region_search(t, _ProtocolRegion(), add_separation=True, sort=True)
    assert "separation" in out.colnames
    assert out["name"].tolist() == ["a", "b"]


def test_region_search_separation_requires_center():
    class NoCenter:
        def contains_points(self, ra, dec=None):
            return np.asarray(ra, float) < 11.0

    t = Table({"ra": RA, "dec": DEC})
    with pytest.raises(ValueError, match="require a center"):
        sph.region_search(t, NoCenter(), add_separation=True)


def test_region_search_explicit_center_overrides():
    t = Table({"ra": RA, "dec": DEC, "name": NAMES})
    out = sph.region_search(t, _ProtocolRegion(), center=(10.0, 5.0),
                            add_separation=True)
    assert out["separation"][0] == pytest.approx(0.0, abs=1e-9)


def test_region_search_compound_region_integration():
    """A real CompoundRegion plugs into region_search via the protocol."""
    import matplotlib.pyplot as plt

    from skyplothelper.geometry.compound import CompoundRegion
    from skyplothelper.wcs_frame import make_wcs_frame

    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    region = CompoundRegion(ax).add_circle(0.0, 0.0, 20.0)
    fig.canvas.draw()

    # Points well inside the 20-deg circle (avoid the exact center, whose
    # membership depends on projection-space boundary details) vs far outside.
    t = Table({"ra": [5.0, 8.0, 90.0], "dec": [5.0, 0.0, 0.0],
               "name": ["in", "in2", "out"]})
    out = sph.region_search(t, region)
    names = set(out["name"].tolist())
    assert "out" not in names
    assert {"in", "in2"} <= names
    plt.close("all")


# --- crossmatch ----------------------------------------------------------

def test_crossmatch_table_within_tolerance():
    cat = Table({"ra": [10.0, 20.0], "dec": [5.0, -3.0], "id": [1, 2]})
    ref = Table({"ra": [10.0001, 50.0], "dec": [5.0001, 0.0]})
    out = sph.crossmatch(cat, ref, 2.0, unit="arcsec")
    assert isinstance(out, Table)
    assert out["id"].tolist() == [1]  # only the first has a counterpart
    assert "match_idx" in out.colnames and "match_sep" in out.colnames
    assert out["match_idx"][0] == 0


def test_crossmatch_return_indices():
    cat = Table({"ra": [10.0, 20.0], "dec": [5.0, -3.0]})
    ref = Table({"ra": [10.0001, 50.0], "dec": [5.0001, 0.0]})
    idx, sep_deg, mask = sph.crossmatch(cat, ref, 2.0, unit="arcsec",
                                        return_indices=True)
    assert idx.tolist() == [0, 0]
    assert mask.tolist() == [True, False]
    assert sep_deg[0] < sep_deg[1]


def test_crossmatch_dataframe_roundtrip():
    pd = pytest.importorskip("pandas")
    cat = pd.DataFrame({"ra": [10.0, 20.0], "dec": [5.0, -3.0], "id": [1, 2]})
    ref = pd.DataFrame({"ra": [10.0001, 50.0], "dec": [5.0001, 0.0]})
    out = sph.crossmatch(cat, ref, 2.0, unit="arcsec")
    assert isinstance(out, pd.DataFrame)
    assert out["id"].tolist() == [1]
    assert "match_sep" in out.columns


# --- exports -------------------------------------------------------------

def test_public_exports():
    for name in ("cone_search", "region_search", "crossmatch"):
        assert hasattr(sph, name)
        assert name in sph.__all__


# ---- frame= knob (ICRS default must stay) ----

def test_cone_search_frame_knob_galactic():
    """A galactic catalog + galactic center must be comparable via frame=.

    Previously the center was force-converted to ICRS with no way out, so a
    galactic-column catalog could not be cone-searched correctly.
    """
    import numpy as np
    from astropy.coordinates import SkyCoord

    from skyplothelper.catalog import cone_search
    center = SkyCoord(120.0, 30.0, unit="deg", frame="galactic")
    # Catalog columns are galactic l/b: one source right at the center.
    lons = np.array([120.0, 200.0])
    lats = np.array([30.0, -40.0])
    mask = cone_search((lons, lats), center, 1.0, frame="galactic",
                       return_mask=True)
    assert list(mask) == [True, False]


def test_cone_search_defaults_to_icrs():
    """The default is unchanged: ICRS."""
    import numpy as np
    from astropy.coordinates import SkyCoord

    from skyplothelper.catalog import cone_search
    center = SkyCoord(83.6, 22.0, unit="deg")
    lons = np.array([83.6, 200.0])
    lats = np.array([22.0, -40.0])
    mask = cone_search((lons, lats), center, 1.0, return_mask=True)
    assert list(mask) == [True, False]
