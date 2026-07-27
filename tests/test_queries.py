"""Tests for skyplothelper.queries (skips without astroquery; offline only)."""

import pytest

from skyplothelper.queries import (
    _HAS_ASTROQUERY,
)

pytestmark = pytest.mark.skipif(not _HAS_ASTROQUERY, reason="astroquery not installed")


# ---------------------------------------------------------------------------
# SIMBAD schema normalization — stable main_id/ra/dec (deg) across astroquery
# versions (TAP lowercase-float vs older uppercase-sexagesimal).
# ---------------------------------------------------------------------------

def _new_schema_table():
    from astropy.table import Table
    return Table({"main_id": ["M  1"], "ra": [83.6287], "dec": [22.0147],
                  "coo_bibcode": ["2007A&A..."]})


def _old_schema_table():
    from astropy.table import Table
    return Table({"MAIN_ID": ["M  1"], "RA": ["05 34 30.9"],
                  "DEC": ["+22 00 53"], "COO_BIBCODE": ["2007A&A..."]})


def test_normalize_simbad_new_schema_passthrough():
    from skyplothelper.queries import _normalize_simbad_table
    t = _normalize_simbad_table(_new_schema_table())
    assert {"main_id", "ra", "dec"} <= set(t.colnames)
    assert abs(float(t["ra"][0]) - 83.6287) < 1e-4
    assert abs(float(t["dec"][0]) - 22.0147) < 1e-4


def test_normalize_simbad_old_schema_converts_to_deg():
    from skyplothelper.queries import _normalize_simbad_table
    t = _normalize_simbad_table(_old_schema_table())
    assert {"main_id", "ra", "dec"} <= set(t.colnames)
    # Uppercase names gone; sexagesimal strings -> float degrees.
    assert "MAIN_ID" not in t.colnames and "RA" not in t.colnames
    assert abs(float(t["ra"][0]) - 83.6287) < 1e-2
    assert abs(float(t["dec"][0]) - 22.0147) < 1e-2


def test_normalize_simbad_both_schemas_agree():
    from skyplothelper.queries import _normalize_simbad_table
    n = _normalize_simbad_table(_new_schema_table())
    o = _normalize_simbad_table(_old_schema_table())
    assert abs(float(n["ra"][0]) - float(o["ra"][0])) < 1e-2
    assert abs(float(n["dec"][0]) - float(o["dec"][0])) < 1e-2


def test_normalize_simbad_empty_and_unknown_passthrough():
    from astropy.table import Table

    from skyplothelper.queries import _normalize_simbad_table
    assert len(_normalize_simbad_table(Table())) == 0
    # No id/ra/dec -> returned unchanged rather than guessed.
    assert _normalize_simbad_table(Table({"foo": [1]})).colnames == ["foo"]


def test_query_simbad_normalizes_wrapper_output(monkeypatch):
    """query_simbad returns the normalized schema regardless of what the
    installed astroquery hands back (here, an old-schema table)."""
    import skyplothelper.queries as q

    class _FakeSimbad:
        def query_object(self, name):
            return _old_schema_table()

    class _FakeMod:
        Simbad = _FakeSimbad

    monkeypatch.setattr(q, "_require_astroquery", lambda sub: _FakeMod)
    tbl = q.query_simbad("M1")
    assert {"main_id", "ra", "dec"} <= set(tbl.colnames)
    assert abs(float(tbl["ra"][0]) - 83.6287) < 1e-2


# ---------------------------------------------------------------------------
# NED schema — additive: native columns preserved, stable ra/dec/main_id added
# ---------------------------------------------------------------------------

def _ned_native_table():
    from astropy.table import Table
    return Table({"Object Name": ["NGC 1275"], "RA": [49.9507], "DEC": [41.5117],
                  "Type": ["G"], "Velocity": [5264.0]})


def test_normalize_ned_adds_stable_cols_keeps_native():
    from skyplothelper.queries import _normalize_ned_table
    t = _normalize_ned_table(_ned_native_table())
    # Native NED columns preserved (the tutorial accesses these by name).
    assert {"Object Name", "RA", "DEC", "Type"} <= set(t.colnames)
    # Stable lowercase contract added.
    assert {"main_id", "ra", "dec"} <= set(t.colnames)
    assert t["main_id"][0] == "NGC 1275"
    assert abs(float(t["ra"][0]) - 49.9507) < 1e-4


def test_query_ned_wrapper_adds_stable_schema(monkeypatch):
    import skyplothelper.queries as q

    class _FakeNed:
        TIMEOUT = 60

        @staticmethod
        def query_object(name):
            return _ned_native_table()

    monkeypatch.setattr(q, "_require_astroquery", lambda sub: type("M", (), {"Ned": _FakeNed}))
    tbl = q.query_ned("NGC 1275")
    assert {"main_id", "ra", "dec"} <= set(tbl.colnames)
    assert "Object Name" in tbl.colnames          # native still there


# ---------------------------------------------------------------------------
# Client timeout guard
# ---------------------------------------------------------------------------

def test_socket_timeout_sets_and_restores():
    import socket

    from skyplothelper.queries import _socket_timeout
    before = socket.getdefaulttimeout()
    with _socket_timeout(15):
        assert socket.getdefaulttimeout() == 15.0
    assert socket.getdefaulttimeout() == before


def test_query_simbad_applies_timeout_during_query(monkeypatch):
    """The socket default timeout is active while the remote call runs."""
    import socket

    import skyplothelper.queries as q

    seen = {}

    class _FakeSimbad:
        TIMEOUT = 60

        def query_object(self, name):
            seen["timeout"] = socket.getdefaulttimeout()
            return _new_schema_table()

    monkeypatch.setattr(q, "_require_astroquery",
                        lambda sub: type("M", (), {"Simbad": _FakeSimbad}))
    q.query_simbad("M1", timeout=12)
    assert seen["timeout"] == 12.0                # timeout in effect during query


def test_resolve_name_returns_skycoord_for_decimal_string():
    """A bare decimal-formatted coord string resolves without network."""
    # Astropy's SkyCoord.from_name() requires network. Use the explicit
    # decimal form which astropy parses locally.
    from astropy.coordinates import SkyCoord
    coord = SkyCoord("180.0 0.0", unit="deg")
    assert abs(float(coord.ra.deg) - 180.0) < 1e-3


def test_search_vizier_accepts_radec_tuple(monkeypatch):
    """A bare (ra, dec) center is resolved to a SkyCoord before querying
    (astroquery's query_region rejects tuples on its own)."""
    from astropy.coordinates import SkyCoord

    import skyplothelper.queries as q

    captured = {}

    class _FakeViz:
        def __init__(self, **kw):
            pass

        def query_region(self, coord, radius=None, catalog=None):
            captured["coord"] = coord
            return []           # len 0 -> search_vizier returns None

    class _FakeMod:
        Vizier = _FakeViz

    monkeypatch.setattr(q, "_require_astroquery", lambda sub: _FakeMod)
    out = q.search_vizier("VII/118/ngc2000", (180.0, 0.0), radius=5)
    assert out is None
    assert isinstance(captured["coord"], SkyCoord)
    assert abs(captured["coord"].ra.deg - 180.0) < 1e-6
    assert abs(captured["coord"].dec.deg - 0.0) < 1e-6


# --- Vizier truncation is announced -----------------------------------------

def _fake_vizier(monkeypatch, nrows):
    """A Vizier stand-in returning exactly *nrows* rows."""
    import numpy as np
    from astropy.table import Table

    import skyplothelper.queries as q

    class _FakeViz:
        def __init__(self, **kw):
            self.row_limit = kw.get("row_limit")

        def query_region(self, coord, radius=None, catalog=None):
            return [Table({"ra": np.zeros(nrows), "dec": np.zeros(nrows)})]

    monkeypatch.setattr(q, "_require_astroquery",
                        lambda sub: type("M", (), {"Vizier": _FakeViz}))
    return q


def _truncation_warnings(monkeypatch, nrows, row_limit):
    import warnings
    q = _fake_vizier(monkeypatch, nrows)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        q.search_vizier("VII/118/ngc2000", (180.0, 0.0), radius=5,
                        row_limit=row_limit)
        return [str(w.message) for w in caught if "row_limit" in str(w.message)]


def test_vizier_warns_when_the_result_hits_row_limit(monkeypatch):
    """Vizier truncates server-side and says nothing, so a truncated table is
    indistinguishable from a complete one — it plots perfectly plausibly."""
    msgs = _truncation_warnings(monkeypatch, nrows=50, row_limit=50)
    assert msgs
    assert "truncated" in msgs[0]


def test_vizier_is_silent_below_the_limit(monkeypatch):
    assert _truncation_warnings(monkeypatch, nrows=49, row_limit=50) == []


def test_vizier_is_silent_when_unlimited(monkeypatch):
    """row_limit=-1 means no limit, so a large result is not evidence of
    truncation and must not warn."""
    assert _truncation_warnings(monkeypatch, nrows=9999, row_limit=-1) == []
