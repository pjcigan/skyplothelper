"""CompoundRegion set-algebra correctness.

Where the CompoundRegion smoke tests just confirm the API
doesn't crash, this file tests the **mathematical correctness** of
the boolean operations: idempotence, complement-of-complement,
inclusion bounds (union ≥ max, intersection ≤ min), XOR identity
(A XOR B = A∪B − A∩B), expand/contract round-trip, and
analytical-area sanity checks.

These tests use modest tolerances since CompoundRegion operates on
projected pixel-space shapely geometries (not exact spherical
geometry) — the goal is to catch real algebraic regressions, not
to pin numerical precision to last-bit accuracy.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from skyplothelper.geometry.compound import CompoundRegion
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def allsky_axes():
    fig = plt.figure(figsize=(10, 5))
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    return fig, ax


# ============================================================
# Idempotence and identity
# ============================================================

def test_add_then_subtract_same_region_empties(allsky_axes):
    fig, ax = allsky_axes
    R = (CompoundRegion(ax)
         .add_circle(180, 30, radius_deg=15)
         .subtract_circle(180, 30, radius_deg=15))
    assert R.is_empty or R.area_frac < 1e-6


def test_add_same_region_twice_is_idempotent(allsky_axes):
    fig, ax = allsky_axes
    R1 = CompoundRegion(ax).add_circle(180, 30, radius_deg=15)
    A1 = R1.area_frac

    R2 = (CompoundRegion(ax)
          .add_circle(180, 30, radius_deg=15)
          .add_circle(180, 30, radius_deg=15))
    A2 = R2.area_frac

    assert A2 == pytest.approx(A1, abs=1e-6), (
        f"adding the same circle twice changed area: {A1} → {A2}"
    )


# ============================================================
# Complement
# ============================================================

def test_empty_region_complement_is_full_frame(allsky_axes):
    """The complement of an empty region should fill the whole frame."""
    fig, ax = allsky_axes
    R = CompoundRegion(ax).complement()
    assert R.area_frac == pytest.approx(1.0, abs=1e-3)


def test_complement_of_complement_is_original(allsky_axes):
    """(R^c)^c ≈ R within numerical tolerance."""
    fig, ax = allsky_axes
    R = CompoundRegion(ax).add_circle(180, 30, radius_deg=20)
    A_before = R.area_frac

    R.complement().complement()
    A_after = R.area_frac

    assert A_after == pytest.approx(A_before, rel=1e-3, abs=1e-4)


# ============================================================
# Union vs intersection bounds
# ============================================================

def test_union_area_geq_individual_areas(allsky_axes):
    """A ∪ B has area ≥ max(area(A), area(B))."""
    fig, ax = allsky_axes
    A = CompoundRegion(ax).add_circle(180, 30, radius_deg=20).area_frac
    B = CompoundRegion(ax).add_circle(160, 30, radius_deg=20).area_frac
    AorB = (CompoundRegion(ax)
            .add_circle(180, 30, radius_deg=20)
            .add_circle(160, 30, radius_deg=20).area_frac)
    assert AorB >= max(A, B) - 1e-6


def test_intersect_area_leq_individual_areas(allsky_axes):
    """A ∩ B has area ≤ min(area(A), area(B))."""
    fig, ax = allsky_axes
    A = CompoundRegion(ax).add_circle(180, 30, radius_deg=20).area_frac
    B = CompoundRegion(ax).add_circle(160, 30, radius_deg=20).area_frac
    AandB = (CompoundRegion(ax)
             .add_circle(180, 30, radius_deg=20)
             .intersect_circle(160, 30, radius_deg=20).area_frac)
    assert AandB <= min(A, B) + 1e-6


# ============================================================
# XOR identity:  A XOR B  ==  (A ∪ B) − (A ∩ B)
# ============================================================

def test_xor_equals_union_minus_intersection(allsky_axes):
    fig, ax = allsky_axes

    union_area = (CompoundRegion(ax)
                  .add_circle(180, 30, radius_deg=15)
                  .add_circle(195, 30, radius_deg=15)
                  .area_frac)

    intersect_area = (CompoundRegion(ax)
                      .add_circle(180, 30, radius_deg=15)
                      .intersect_circle(195, 30, radius_deg=15)
                      .area_frac)

    xor_area = (CompoundRegion(ax)
                .add_circle(180, 30, radius_deg=15)
                .xor_circle(195, 30, radius_deg=15)
                .area_frac)

    expected = union_area - intersect_area
    assert xor_area == pytest.approx(expected, rel=1e-3, abs=1e-4), (
        f"XOR area {xor_area} != union−intersection {expected} "
        f"(union={union_area}, intersect={intersect_area})"
    )


# ============================================================
# contains_point — geometric correctness
# ============================================================

def test_contains_point_inside_circle(allsky_axes):
    fig, ax = allsky_axes
    R = CompoundRegion(ax).add_circle(180, 30, radius_deg=10)
    assert R.contains_point(180, 30) is True


def test_contains_point_outside_circle(allsky_axes):
    fig, ax = allsky_axes
    R = CompoundRegion(ax).add_circle(180, 30, radius_deg=10)
    # 90° away from the circle center — must be outside a 10° circle
    assert R.contains_point(0, -30) is False


def test_contains_points_vectorised(allsky_axes):
    fig, ax = allsky_axes
    R = CompoundRegion(ax).add_circle(180, 30, radius_deg=10)
    ra = np.array([180.0, 0.0, 180.0, 100.0])
    dec = np.array([30.0, -30.0, 30.5, 50.0])
    out = R.contains_points(ra, dec)
    # First and third points: inside (on / near center). Others outside.
    assert out[0] is np.True_ or out[0] == True  # noqa: E712
    assert out[2] is np.True_ or out[2] == True  # noqa: E712
    assert out[1] is np.False_ or out[1] == False  # noqa: E712
    assert out[3] is np.False_ or out[3] == False  # noqa: E712


# ============================================================
# expand / contract round-trip
# ============================================================

def test_expand_increases_area(allsky_axes):
    fig, ax = allsky_axes
    R = CompoundRegion(ax).add_circle(180, 30, radius_deg=10)
    A_before = R.area_frac
    R.expand(angle_deg=2.0)
    A_after = R.area_frac
    assert A_after > A_before


def test_contract_decreases_area(allsky_axes):
    fig, ax = allsky_axes
    R = CompoundRegion(ax).add_circle(180, 30, radius_deg=10)
    A_before = R.area_frac
    R.contract(angle_deg=2.0)
    A_after = R.area_frac
    assert A_after < A_before


def test_contract_more_than_region_radius_empties(allsky_axes):
    """Contracting by more than the radius of a small region should
    empty it."""
    fig, ax = allsky_axes
    R = CompoundRegion(ax).add_circle(180, 30, radius_deg=5)
    R.contract(angle_deg=10.0)  # 2× the radius
    assert R.is_empty


# ============================================================
# Analytical area: a small latitude band has predictable solid angle.
# ============================================================

def test_latitude_band_solid_angle_close_to_analytical(allsky_axes):
    """A latitude band [lat_min, lat_max] covers
    Ω = 2π·(sin(lat_max) − sin(lat_min)) sr.
    For ±10° around the equator: 2π·(sin(10°) − sin(−10°)) ≈ 2π·0.347 ≈ 2.181 sr.
    """
    fig, ax = allsky_axes
    R = CompoundRegion(ax).add_latitude_band(-10, 10)
    expected_sr = 2 * np.pi * (np.sin(np.radians(10)) - np.sin(np.radians(-10)))
    actual_sr = R.solid_angle["sr"]
    # Allow 2% — the area_frac approximation assumes uniform pixel scale,
    # which is not exact on AIT.
    assert actual_sr == pytest.approx(expected_sr, rel=0.02), (
        f"latitude-band solid angle {actual_sr:.4f} sr (analytical "
        f"{expected_sr:.4f} sr)"
    )


def test_render_dashed_linestyle_does_not_crash():
    """A dashed linestyle must not reach the width-0 fill patches (matplotlib
    rejects a zero-scaled dash list); it belongs on the boundary instead."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=180, frame="ICRS",
                        fig=fig)
    r = CompoundRegion(ax).add_latitude_band(-90, 12)
    r.render(facecolor="none", edgecolor="C0", linewidth=1.2, linestyle="--")
    fig.canvas.draw()                       # would raise ValueError before fix
    dashed = [ln for ln in ax.get_lines()
              if ln.get_linestyle() in ("--", "dashed")]
    assert len(dashed) >= 1                  # boundary is actually dashed
    plt.close(fig)


def test_render_dashed_via_ls_alias():
    """The ``ls`` alias is handled the same as ``linestyle``."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=180, frame="ICRS",
                        fig=fig)
    r = CompoundRegion(ax).add_latitude_band(-90, 20)
    r.render(facecolor="none", edgecolor="k", lw=1.0, ls="--")
    fig.canvas.draw()
    plt.close(fig)


def test_render_boundary_accepts_lw_and_c_aliases():
    """render_boundary forwards an explicit ``linewidth`` to ax.plot, so a
    caller's ``lw=`` alias used to collide ("Got both 'linewidth' and 'lw'").
    It now normalizes matplotlib aliases (lw/c/ls) like ax.plot does."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=180, frame="ICRS",
                        fig=fig)
    lines = (CompoundRegion(ax).add_circle(180, 0, radius_deg=20)
             .render_boundary(color="teal", lw=1.5))
    assert lines
    assert lines[0].get_linewidth() == 1.5      # lw honored
    # c/ls aliases resolve too; explicit linewidth= still works
    lines2 = (CompoundRegion(ax).add_circle(180, 0, radius_deg=20)
              .render_boundary(c="red", ls="--"))
    assert lines2[0].get_color() == "red"
    lines3 = (CompoundRegion(ax).add_circle(180, 0, radius_deg=20)
              .render_boundary(color="k", linewidth=3.0))
    assert lines3[0].get_linewidth() == 3.0
    plt.close(fig)


# ============================================================
# Bounded-field (zoomed frame) clipping — CompoundRegion membership on a
# small TAN/SIN field, where global bands used to project empty or (via a
# complement misfire) fill the whole field.
# ============================================================

def _tan_field(center, fov=10.5):
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=center, fov_deg=fov,
                        fig=fig)
    fig.canvas.draw()
    return fig, ax


def test_world_bounds_none_for_allsky_and_globe():
    """The bounded-field path stays inert on all-sky / globe frames."""
    from skyplothelper.globe.frame import make_globe_frame
    for mk in (
        lambda: make_wcs_frame(111, projection="AIT", center=0, frame="ICRS"),
        lambda: make_wcs_frame(111, projection="MOL", center=0, frame="ICRS"),
        lambda: make_globe_frame(111, center_LONdeg=0, center_LATdeg=20),
    ):
        fig = plt.figure()
        ax = mk()
        fig.canvas.draw()
        assert CompoundRegion(ax).projector.world_bounds() is None
        plt.close(fig)


def test_world_bounds_box_for_zoomed_field():
    fig, ax = _tan_field((187.7, 12.4))
    wb = CompoundRegion(ax).projector.world_bounds()
    assert wb is not None
    lon_lo, lon_hi, lat_lo, lat_hi = wb
    assert lat_lo < 12.4 < lat_hi              # field brackets its center dec
    assert lon_lo < 187.7 < lon_hi
    plt.close(fig)


def test_latitude_band_membership_on_field():
    """A dec band resolves correct membership on a zoomed field (was empty or
    all-inclusive before the field-clip fix)."""
    fig, ax = _tan_field((187.7, 12.4))
    r = CompoundRegion(ax).add_latitude_band(-90, 12)
    assert not r.is_empty
    assert r.contains_points([187.7], [10.0])[0]        # dec 10 < 12 -> in
    assert not r.contains_points([187.7], [14.0])[0]    # dec 14 > 12 -> out
    plt.close(fig)


def test_longitude_band_membership_on_field():
    fig, ax = _tan_field((187.7, 12.4))
    r = CompoundRegion(ax).add_longitude_band(180, 195)
    assert r.contains_points([187.7], [12.0])[0]
    plt.close(fig)


def test_frame_band_empty_when_not_crossing_field():
    """A galactic band that doesn't pass through the field contributes nothing
    (it used to fill the whole field via a complement misfire)."""
    fig, ax = _tan_field((187.7, 12.4))          # Virgo: galactic lat ~+75
    r = CompoundRegion(ax).add_frame_band(-15, 15, frame="galactic")
    assert r.is_empty
    assert not r.contains_points([187.7], [12.4])[0]
    plt.close(fig)


def test_frame_band_contains_when_field_on_galactic_plane():
    from astropy.coordinates import SkyCoord
    gc = SkyCoord(l=30, b=0, unit="deg", frame="galactic").icrs
    fig, ax = _tan_field((gc.ra.deg, gc.dec.deg), fov=8)
    r = CompoundRegion(ax).add_frame_band(-15, 15, frame="galactic")
    assert r.contains_points([gc.ra.deg], [gc.dec.deg])[0]
    plt.close(fig)


def test_repro_latitude_minus_frame_band_on_field():
    """The original #13 repro: latitude band minus a non-crossing galactic
    band keeps the band (the subtraction no longer wipes the whole field)."""
    fig, ax = _tan_field((187.7, 12.4))
    r = (CompoundRegion(ax).add_latitude_band(-90, 12)
         .subtract_frame_band(-15, 15, frame="galactic"))
    assert not r.is_empty
    assert r.contains_points([185.0], [9.0])[0]          # dec 9 < 12 -> in
    assert not r.contains_points([185.0], [14.0])[0]     # dec 14 > 12 -> out
    plt.close(fig)


def test_render_returns_boundary_line_artists():
    """render(edgecolor=...) must return the boundary Line2D artists (not just
    the fill patches), so a caller can remove the whole region cleanly."""
    from matplotlib.lines import Line2D
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=180, frame="ICRS",
                        fig=fig)
    r = CompoundRegion(ax).add_latitude_band(-20, 20)
    arts = r.render(facecolor="none", edgecolor="C1", linewidth=1.5)
    assert any(isinstance(a, Line2D) for a in arts)   # was: fill patches only
    # removing every returned artist takes the region fully off the axes
    n_lines = len([ln for ln in ax.get_lines()])
    for a in arts:
        a.remove()
    assert len([ln for ln in ax.get_lines()]) < n_lines
    plt.close(fig)


# ============================================================
# CompoundRegion.clip — mask artists to a region (G2)
# ============================================================

def test_compound_region_clip_masks_artist():
    """CompoundRegion.clip sets a clip path on an artist so it renders only
    inside the region. Works on a celestial all-sky frame."""
    ax = make_wcs_frame(111, "AIT", frame="ICRS", center=0)
    reg = CompoundRegion(ax)
    reg.add_circle(0.0, 0.0, 30.0)
    sc = ax.scatter([0, 120], [0, 0], transform=ax.get_transform("world"))
    path = reg.clip(sc)
    assert sc.get_clip_path() is not None
    assert path.vertices.shape[0] > 0
    plt.close(ax.figure)


def test_compound_region_clip_complement_differs():
    """clip(complement=True) yields a different (frame-minus-region) path."""
    ax = make_wcs_frame(111, "AIT", frame="ICRS", center=0)
    reg = CompoundRegion(ax)
    reg.add_circle(0.0, 0.0, 30.0)
    inside = reg.clip_path()
    outside = reg.clip_path(complement=True)
    assert inside.vertices.shape[0] > 0 and outside.vertices.shape[0] > 0
    assert inside.vertices.shape != outside.vertices.shape
    # complement did not mutate the region
    assert reg.clip_path().vertices.shape == inside.vertices.shape
    plt.close(ax.figure)


# ============================================================
# CompoundRegion.from_points — hull region from a point scatter (R1)
# ============================================================

def test_from_points_convex_region():
    """A convex hull region encloses the points (centroid inside, most points
    inside — hull vertices sit on the boundary) and has positive area."""
    rng = np.random.RandomState(1)
    lon = 60 + rng.uniform(-10, 10, 200)
    lat = 20 + rng.uniform(-10, 10, 200)
    ax = make_wcs_frame(111, "TAN", frame="ICRS", center=(60, 20), fov_deg=40)
    reg = CompoundRegion.from_points(ax, lon, lat, hull="convex")
    assert reg.contains_point(60, 20)
    assert reg.contains_points(lon, lat).mean() > 0.9
    assert reg.solid_angle["sq_deg"] > 0
    plt.close(ax.figure)


def test_from_points_concave_tighter_than_convex():
    """A concave hull has smaller area than the convex hull of the same set."""
    rng = np.random.RandomState(2)
    t = rng.uniform(0, 2 * np.pi, 400)
    r = 10 + rng.normal(0, 1, 400)
    lon = 60 + r * np.cos(t)
    lat = 20 + r * np.sin(t)
    ax = make_wcs_frame(111, "TAN", frame="ICRS", center=(60, 20), fov_deg=50)
    cvx = CompoundRegion.from_points(ax, lon, lat, hull="convex")
    ccv = CompoundRegion.from_points(ax, lon, lat, hull="concave", ratio=0.2)
    assert ccv.solid_angle["sq_deg"] < cvx.solid_angle["sq_deg"]
    plt.close(ax.figure)


def test_from_points_too_few_raises():
    ax = make_wcs_frame(111, "TAN", frame="ICRS", center=(0, 0), fov_deg=20)
    with pytest.raises(ValueError, match="at least 3"):
        CompoundRegion.from_points(ax, [0, 1], [0, 1])
    plt.close(ax.figure)


def test_from_points_bad_hull_raises():
    ax = make_wcs_frame(111, "TAN", frame="ICRS", center=(0, 0), fov_deg=20)
    with pytest.raises(ValueError, match="convex.*concave"):
        CompoundRegion.from_points(ax, [0, 1, 0.5], [0, 0, 1], hull="banana")
    plt.close(ax.figure)


# ============================================================
# HEALPix bridge (R2) + conveniences (R3)
# ============================================================

def test_to_healpix_mask_matches_area():
    """Rasterized mask sky-fraction matches the region's solid-angle fraction."""
    ax = make_wcs_frame(111, "AIT", frame="ICRS", center=0)
    reg = CompoundRegion(ax).add_circle(60, 20, 15)
    mask = reg.to_healpix_mask(64)
    expect = reg.solid_angle["sr"] / (4 * np.pi)
    assert abs(mask.mean() - expect) < 0.002
    plt.close(ax.figure)


def test_healpix_mask_round_trip_preserves_area():
    """region -> to_healpix_mask -> from_healpix_mask keeps the area (within
    HEALPix pixelization)."""
    ax = make_wcs_frame(111, "AIT", frame="ICRS", center=0)
    reg = CompoundRegion(ax).add_circle(60, 20, 15)
    reg2 = CompoundRegion.from_healpix_mask(ax, reg.to_healpix_mask(64))
    a1, a2 = reg.solid_angle["sq_deg"], reg2.solid_angle["sq_deg"]
    assert abs(a2 - a1) / a1 < 0.05
    plt.close(ax.figure)


def test_from_healpix_empty_mask_raises():
    import healpy as hp
    ax = make_wcs_frame(111, "AIT", frame="ICRS", center=0)
    with pytest.raises(ValueError, match="empty"):
        CompoundRegion.from_healpix_mask(ax, np.zeros(hp.nside2npix(8), bool))
    plt.close(ax.figure)


def test_centroid_and_bounds():
    ax = make_wcs_frame(111, "AIT", frame="ICRS", center=0)
    reg = CompoundRegion(ax).add_circle(60, 20, 15)
    clon, clat = reg.centroid
    assert abs(clon - 60) < 3 and abs(clat - 20) < 3
    lo0, lo1, la0, la1 = reg.bounds
    assert lo0 < 60 < lo1 and la0 < 20 < la1
    plt.close(ax.figure)


def test_from_polygons_batch_union():
    ax = make_wcs_frame(111, "AIT", frame="ICRS", center=0)
    tri = ([0, 20, 10, 0], [0, 0, 20, 0])
    box = ([40, 60, 60, 40, 40], [-10, -10, 10, 10, -10])
    reg = CompoundRegion.from_polygons(ax, [tri, box])
    assert reg.solid_angle["sq_deg"] > 0
    plt.close(ax.figure)


# ============================================================
# Interop export (R4): DS9 / CRTF / astropy-regions
# ============================================================

def test_to_ds9_export():
    ax = make_wcs_frame(111, "AIT", frame="galactic", center=0)
    reg = CompoundRegion(ax).add_circle(60, 20, 20).subtract_circle(60, 20, 8)
    ds9 = reg.to_ds9()
    assert ds9.startswith("# Region file format: DS9")
    assert ds9.splitlines()[2] == "galactic"
    assert "polygon(" in ds9 and "-polygon(" in ds9  # exterior + hole
    assert reg.to_ds9(frame="icrs").splitlines()[2] == "icrs"
    plt.close(ax.figure)


def test_to_crtf_export():
    ax = make_wcs_frame(111, "AIT", frame="icrs", center=0)
    reg = CompoundRegion(ax).add_circle(60, 20, 20)
    crtf = reg.to_crtf()
    assert crtf.startswith("#CRTF")
    assert "poly[" in crtf and "coord=ICRS" in crtf and "deg]" in crtf
    plt.close(ax.figure)


def test_to_regions_optional_dependency():
    ax = make_wcs_frame(111, "AIT", frame="icrs", center=0)
    reg = CompoundRegion(ax).add_circle(60, 20, 20)
    try:
        import regions  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="regions"):
            reg.to_regions()
    else:
        from regions import Regions
        assert isinstance(reg.to_regions(), Regions)
    plt.close(ax.figure)


def test_ds9_round_trip_with_hole():
    ax = make_wcs_frame(111, "AIT", frame="icrs", center=0)
    reg = CompoundRegion(ax).add_circle(60, 20, 20).subtract_circle(60, 20, 8)
    reg2 = CompoundRegion.from_ds9(ax, reg.to_ds9())
    assert abs(reg2.solid_angle["sq_deg"] - reg.solid_angle["sq_deg"]) < 2
    plt.close(ax.figure)


def test_ds9_import_from_file(tmp_path):
    ax = make_wcs_frame(111, "AIT", frame="icrs", center=0)
    reg = CompoundRegion(ax).add_circle(60, 20, 15)
    p = str(tmp_path / "r.reg")
    reg.to_ds9(path=p)
    reg2 = CompoundRegion.from_ds9(ax, p)
    assert reg2.solid_angle["sq_deg"] > 0
    plt.close(ax.figure)


def test_crtf_round_trip_poly():
    ax = make_wcs_frame(111, "AIT", frame="icrs", center=0)
    reg = CompoundRegion(ax).add_circle(60, 20, 15)
    reg2 = CompoundRegion.from_crtf(ax, reg.to_crtf())
    a1, a2 = reg.solid_angle["sq_deg"], reg2.solid_angle["sq_deg"]
    assert abs(a2 - a1) / a1 < 0.02
    plt.close(ax.figure)


def test_ds9_cross_frame_import_preserves_area():
    axg = make_wcs_frame(111, "AIT", frame="galactic", center=0)
    ax = make_wcs_frame(111, "AIT", frame="icrs", center=0)
    rg = CompoundRegion(axg).add_circle(120, 30, 15)
    r_icrs = CompoundRegion.from_ds9(ax, rg.to_ds9())  # galactic -> icrs
    a1, a2 = rg.solid_angle["sq_deg"], r_icrs.solid_angle["sq_deg"]
    assert abs(a2 - a1) / a1 < 0.05
    plt.close(axg.figure)
    plt.close(ax.figure)


# ============================================================
# Region-to-region set algebra (union / intersection / difference /
# symmetric_difference) — new in the F3(ii)/region-ops expansion
# ============================================================

def _two_regions(ax):
    """Two overlapping circle regions on the same frame, plus their areas."""
    A = CompoundRegion(ax).add_circle(160, 0, radius_deg=30)
    B = CompoundRegion(ax).add_circle(200, 0, radius_deg=30)
    return A, B


def test_region_union_equals_A_plus_B_minus_intersection():
    ax = make_wcs_frame(111, "AIT", frame="icrs", center=180)
    ax.figure.canvas.draw()
    aA = _two_regions(ax)[0].solid_angle["sq_deg"]
    aB = _two_regions(ax)[1].solid_angle["sq_deg"]
    A, B = _two_regions(ax)
    inter = A.intersection(B).solid_angle["sq_deg"]
    C, D = _two_regions(ax)
    uni = C.union(D).solid_angle["sq_deg"]
    assert abs(uni - (aA + aB - inter)) / uni < 0.02
    plt.close(ax.figure)


def test_region_difference_equals_A_minus_intersection():
    ax = make_wcs_frame(111, "AIT", frame="icrs", center=180)
    ax.figure.canvas.draw()
    aA = _two_regions(ax)[0].solid_angle["sq_deg"]
    A, B = _two_regions(ax)
    inter = A.intersection(B).solid_angle["sq_deg"]
    C, D = _two_regions(ax)
    diff = C.difference(D).solid_angle["sq_deg"]
    assert abs(diff - (aA - inter)) / aA < 0.02
    plt.close(ax.figure)


def test_region_symmetric_difference_equals_union_minus_intersection():
    ax = make_wcs_frame(111, "AIT", frame="icrs", center=180)
    ax.figure.canvas.draw()
    A, B = _two_regions(ax)
    uni = A.union(B).solid_angle["sq_deg"]
    C, D = _two_regions(ax)
    inter = C.intersection(D).solid_angle["sq_deg"]
    E, F = _two_regions(ax)
    xor = E.symmetric_difference(F).solid_angle["sq_deg"]
    assert abs(xor - (uni - inter)) / uni < 0.02
    plt.close(ax.figure)


def test_region_setop_rejects_non_region():
    ax = make_wcs_frame(111, "AIT", frame="icrs", center=180)
    with pytest.raises(TypeError):
        CompoundRegion(ax).add_circle(160, 0, 30).union("not a region")
    plt.close(ax.figure)


def test_compound_render_stroke_applies_path_effects():
    """CompoundRegion.render(stroke_color=...) draws a legibility stroke
    (shared stroke knob added to the region fills)."""
    ax = make_wcs_frame(111, "AIT", frame="icrs", center=180)
    ax.figure.canvas.draw()
    R = CompoundRegion(ax).add_circle(180, 0, radius_deg=25)
    patches = R.render(facecolor="C0", stroke_color="w", stroke_lw=3)
    assert any(p.get_path_effects() for p in patches)
    # default (no stroke) leaves path_effects empty
    R2 = CompoundRegion(ax).add_circle(120, 20, radius_deg=15)
    assert not any(p.get_path_effects() for p in R2.render(facecolor="C1"))
    plt.close(ax.figure)


def test_shape_helper_stroke_applies_path_effects():
    """add_spherical_polygon / add_geodesic_circle gained the stroke knob via
    the shared render_region path."""
    import skyplothelper as sph
    ax = make_wcs_frame(111, "AIT", frame="icrs", center=180)
    ax.figure.canvas.draw()
    p1 = sph.add_spherical_polygon(ax, [150, 210, 210, 150], [-20, -20, 20, 20],
                                   facecolor="C0", stroke_color="w", stroke_lw=3)
    p2 = sph.add_geodesic_circle(ax, 100, 30, 15, facecolor="C1",
                                 stroke_color="k")
    assert any(a.get_path_effects() for a in p1)
    assert any(a.get_path_effects() for a in p2)
    plt.close(ax.figure)


def test_region_shapes_raise_clean_on_non_fits():
    """Region shapes on a non-FITS frame raise a clear NotImplementedError,
    not a cryptic AttributeError."""
    import skyplothelper as sph
    ax = make_wcs_frame(111, "robinson", frame="icrs", center=0)
    assert getattr(ax, "wcs", None) is None
    with pytest.raises(NotImplementedError):
        sph.add_geodesic_circle(ax, 0, 40, 20, facecolor="C0")
    plt.close(ax.figure)
