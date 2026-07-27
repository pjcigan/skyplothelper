"""Smoke tests for skyplothelper.geometry.shapes."""

import matplotlib

matplotlib.use("Agg")

import astropy.units as u
import matplotlib.pyplot as plt
import pytest
from astropy.coordinates import SkyCoord

from skyplothelper.geometry.shapes import (
    add_annulus,
    add_ellipse,
    add_geodesic_circle,
    add_rectangle,
    add_spherical_polygon,
    add_square,
    ellipse,
    geodesic_circle,
    rectangle,
)
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---- Vertex generators ----

def test_geodesic_circle_returns_arrays():
    lons, lats = geodesic_circle(0.0, 0.0, radius_deg=5.0)
    assert len(lons) == len(lats) > 0


def test_rectangle_returns_arrays():
    lons, lats = rectangle(0.0, 0.0, width=2.0, height=1.0)
    assert len(lons) == len(lats) > 0


def test_ellipse_returns_arrays():
    lons, lats = ellipse(0.0, 0.0, semi_major=2.0, semi_minor=1.0)
    assert len(lons) == len(lats) > 0


# ---- Vertex generators: input frame is preserved (regression) ----

def _ring_center(lons, lats):
    """Direction-vector mean of a small ring — wrap-safe center."""
    import numpy as np
    lo, la = np.radians(lons), np.radians(lats)
    x = (np.cos(la) * np.cos(lo)).mean()
    y = (np.cos(la) * np.sin(lo)).mean()
    z = np.sin(la).mean()
    return (np.degrees(np.arctan2(y, x)) % 360.0,
            np.degrees(np.arctan2(z, np.hypot(x, y))))


@pytest.mark.parametrize("frame,lon0,lat0", [
    ("galactic", 120.0, 30.0),
    ("geocentrictrueecliptic", 45.0, 10.0),
    ("icrs", 83.6, 22.0),
])
def test_geodesic_circle_preserves_input_frame(frame, lon0, lat0):
    """A non-ICRS SkyCoord must NOT be silently coerced to ICRS.

    These bare builders have no axes/WCS context, so the only self-consistent
    contract is output-frame == input-frame. Previously a galactic center came
    back as a ring around the ICRS-converted position, with no error.
    """
    import numpy as np
    coord = SkyCoord(lon0, lat0, unit="deg", frame=frame)
    cx, cy = _ring_center(*geodesic_circle(coord, 5.0))
    assert np.isclose(cx, lon0, atol=0.05)
    assert np.isclose(cy, lat0, atol=0.05)


def test_rectangle_ellipse_preserve_input_frame():
    import numpy as np
    gal = SkyCoord(120.0, 30.0, unit="deg", frame="galactic")
    # The `shifted` mechanic consumes only the `lat` slot, so the remaining
    # size argument must be a keyword when the center is a SkyCoord.
    for lons, lats in (rectangle(gal, 6.0, height=4.0),
                       ellipse(gal, 5.0, semi_minor=3.0)):
        cx, cy = _ring_center(np.asarray(lons), np.asarray(lats))
        assert np.isclose(cx, 120.0, atol=0.5)
        assert np.isclose(cy, 30.0, atol=0.5)


def test_parse_coords_still_coerces_to_icrs_without_the_flag():
    """The ICRS coercion is the DEFAULT and must stay: catalog.py relies on it.
    Only the bare builders opt out via preserve_frame=True."""
    import numpy as np

    from skyplothelper.geometry._parsing import _parse_coords
    gal = SkyCoord([120.0], [30.0], unit="deg", frame="galactic")
    lon, lat = _parse_coords(gal)
    assert np.isclose(lon[0], gal.icrs.ra.deg)
    assert np.isclose(lat[0], gal.icrs.dec.deg)


# ---- Renderers (smoke + SkyCoord/Quantity inputs) ----

def test_add_geodesic_circle_floats():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    patches = add_geodesic_circle(ax, 0.0, 0.0, radius_deg=10.0)
    assert len(patches) > 0


def test_add_geodesic_circle_skycoord_quantity():
    """SkyCoord center + Quantity radius — the v7 'magic'."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    coord = SkyCoord(45.0, 30.0, unit="deg", frame="icrs")
    patches = add_geodesic_circle(ax, coord, 5 * u.deg)
    assert len(patches) > 0


def test_add_spherical_polygon_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    lons = [0, 30, 30, 0]
    lats = [-10, -10, 10, 10]
    patches = add_spherical_polygon(ax, lons, lats)
    assert len(patches) > 0


def test_add_rectangle_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    patches = add_rectangle(ax, 0.0, 0.0, width=10.0, height=5.0)
    assert len(patches) > 0


def test_add_square_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    patches = add_square(ax, 30.0, 0.0, size=5.0)
    assert len(patches) > 0


def test_add_ellipse_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    patches = add_ellipse(ax, 0.0, 0.0, semi_major=10.0, semi_minor=5.0)
    assert len(patches) > 0


def test_add_annulus_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    patches = add_annulus(ax, 0.0, 0.0, inner_radius=5.0, outer_radius=10.0)
    assert len(patches) > 0


# ---- Circumpolar closure regression ------------------------------------------

def test_densify_polygon_edges_closure_short_way_for_circumpolar_polygon():
    """Regression: closure densification for circumpolar polygons.

    A polygon whose vertex sequence walks ~360° in lon and closes
    across a small lon span (e.g. last vertex at lon=355, first at
    lon=0) must densify the closure edge the *short way* — a 5°
    forward hop through the seam — not the long way (355°
    backward across the entire polygon).

    The historical centroid-distance heuristic picked long-way for
    these closures because the polygon's centroid sits on the
    long-way side of the seam at that latitude; the fix is to
    always go short-way (every user-authored polygon edge is
    intended-short by convention).
    """
    import numpy as np

    from skyplothelper.geometry._densify import _densify_polygon_edges

    # Euclid Northern mainland-style: lons walk 0 → 355 (26 verts),
    # then close back from 355 → 0 across the seam (5° forward).
    euclid_north_lons = [
        0, 25, 50, 70, 82, 90, 99, 107, 121, 140, 158, 174, 190,
        205, 220, 237, 253, 263, 270, 277, 284, 292, 302, 316, 335,
        355,
    ]
    euclid_north_lats = [
        83, 83, 82, 80, 76, 67, 56, 42, 36, 32, 27, 21, 15, 9, 3,
        -3, -7, 0, 14, 28, 43, 56, 68, 76, 80, 82,
    ]
    lons = np.array(euclid_north_lons + [euclid_north_lons[0]])
    lats = np.array(euclid_north_lats + [euclid_north_lats[0]])
    n_per_edge = 20
    dense_lons, _ = _densify_polygon_edges(lons, lats, resolution=n_per_edge)

    # The closure edge is the last per-edge block (vertex 25 → 0).
    n_edges = len(lons) - 1
    closure = dense_lons[(n_edges - 1) * n_per_edge:n_edges * n_per_edge]
    # In [-180, 180] wrapped representation, the short-way path
    # starts at -5 (== 355 mod 360) and walks monotonically toward 0.
    # The long-way path would jump backward to -40, -76, -111, ...
    # Step-by-step lon delta in absolute value stays under 1° for
    # the short way (5° / 20 = 0.25°); long way would average 17.75°.
    diffs = np.abs(np.diff(closure))
    assert np.all(diffs < 5.0), (
        f"closure edge took the long way around: per-step diffs="
        f"{diffs[:5]}... — expected per-step magnitudes ≤ 5° for the "
        "5°-span short-way closure")


# ============================================================
# Zenithal projections: antimeridian trace must not become the frame
# ============================================================

@pytest.mark.parametrize("projection", ["ZEA", "ARC", "STG", "TAN", "SIN", "AIT"])
def test_add_spherical_polygon_not_collapsed_on_zenithal(projection):
    """Regression: on ZEA/ARC the frame polygon collapsed to a thin meridian
    sliver, so every shape was clipped away to a sliver (or vanished).

    ``_get_wcs_boundary`` traces the anti-center meridian at ``lon_center±180``
    -- the same meridian, a hair either side. On an all-sky map those land on
    opposite edges and bound the map; on a zenithal projection they coincide and
    the "ring" degenerates. Assert the drawn polygon keeps most of the extent it
    has when projected straight through the WCS, with no frame clipping.
    """
    import numpy as np
    fig = plt.figure(figsize=(4, 4))
    ax = make_wcs_frame(111, projection=projection, center=(186, 55),
                        fov_deg=45, fig=fig)
    lons = [180.0, 192.0, 192.0, 180.0]
    lats = [50.0, 50.0, 60.0, 60.0]

    # extent of the densified boundary projected directly (no frame clip)
    ring = list(zip(lons, lats)) + [(lons[0], lats[0])]
    dl, db = [], []
    for (l0, b0), (l1, b1) in zip(ring[:-1], ring[1:]):
        dl.append(np.linspace(l0, l1, 200))
        db.append(np.linspace(b0, b1, 200))
    ex, ey = ax.wcs.world_to_pixel_values(np.concatenate(dl), np.concatenate(db))
    m = np.isfinite(ex) & np.isfinite(ey)
    exp_dx, exp_dy = np.ptp(ex[m]), np.ptp(ey[m])

    arts = add_spherical_polygon(ax, lons, lats, geodesic=False,
                                 facecolor="none", edgecolor="r")
    arts = arts if isinstance(arts, (list, tuple)) else [arts]
    verts = np.concatenate([
        a.get_path().vertices[np.isfinite(a.get_path().vertices).all(axis=1)]
        for a in arts if a is not None and hasattr(a, "get_path")])
    assert len(verts) > 0, f"{projection}: no patch rendered"
    got_dx, got_dy = np.ptp(verts[:, 0]), np.ptp(verts[:, 1])
    assert got_dx > 0.8 * exp_dx, (
        f"{projection}: polygon collapsed in x "
        f"({got_dx:.1f} vs expected {exp_dx:.1f})")
    assert got_dy > 0.8 * exp_dy, (
        f"{projection}: polygon collapsed in y "
        f"({got_dy:.1f} vs expected {exp_dy:.1f})")
    plt.close(fig)


@pytest.mark.parametrize("projection,expect_boundary", [
    ("AIT", True), ("MOL", True), ("CAR", True),   # antimeridian IS the map edge
    ("ZEA", False), ("ARC", False),                # interior curve -> degenerate
    ("STG", False), ("TAN", False),
])
def test_get_wcs_boundary_rejects_degenerate_antimeridian(projection,
                                                          expect_boundary):
    """The traced ring is only a real boundary when the two ±eps traces of the
    anti-center meridian land on opposite map edges."""
    from skyplothelper.geometry._frame_geom import _get_wcs_boundary
    fig = plt.figure(figsize=(4, 4))
    if expect_boundary:
        ax = make_wcs_frame(111, projection=projection, center=0, fig=fig)
    else:
        ax = make_wcs_frame(111, projection=projection, center=(186, 55),
                            fov_deg=45, fig=fig)
    poly = _get_wcs_boundary(ax.wcs, 500)
    if expect_boundary:
        assert poly is not None and poly.area > 1
    else:
        assert poly is None
    plt.close(fig)


# ---- shared frame dispatch (_coords_to_frame_deg) ----

@pytest.mark.parametrize("name,attrs", [
    ("galactic", lambda c: (c.galactic.l.deg, c.galactic.b.deg)),
    ("ecliptic", lambda c: (c.geocentrictrueecliptic.lon.deg,
                            c.geocentrictrueecliptic.lat.deg)),
    ("supergalactic", lambda c: (c.supergalactic.sgl.deg,
                                 c.supergalactic.sgb.deg)),
    ("icrs", lambda c: (c.icrs.ra.deg, c.icrs.dec.deg)),
])
def test_coords_to_frame_deg_matches_astropy(name, attrs):
    """The shared dispatch must equal astropy's own frame attributes."""
    import numpy as np

    from skyplothelper.geometry._parsing import _coords_to_frame_deg
    c = SkyCoord(83.6, 22.0, unit="deg")
    assert np.allclose(_coords_to_frame_deg(c, name), attrs(c))


@pytest.mark.parametrize("alias,expected", [
    ("gal", "galactic"), ("Galactic", "galactic"),
    ("ecl", "geocentrictrueecliptic"), ("supergalactic", "supergalactic"),
    ("super", "supergalactic"),          # 'super' must beat 'gal'
    ("itrs", "icrs"), ("heliographic", "icrs"), (None, "icrs"),
])
def test_resolve_sky_frame_aliases(alias, expected):
    """Alias handling must mirror wcs_frame._resolve_ctype's substring rules,
    including that 'supergalactic' resolves to supergalactic (it contains
    'gal', so order matters)."""
    from skyplothelper.geometry._parsing import _resolve_sky_frame
    assert _resolve_sky_frame(alias) == expected
