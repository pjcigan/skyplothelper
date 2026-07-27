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
