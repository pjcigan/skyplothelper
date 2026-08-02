"""Region fill / shapes on the non-FITS custom-projection frames (G4).

The Robinson / Eckert IV / Winkel Tripel / Kavrayskiy VII / McBryde frames are
drawn by a matplotlib ``CurvedTransform`` rather than a FITS WCS, so
``ax.wcs is None``. Before G4 the region helpers raised ``NotImplementedError``
on them; now they render through :class:`WCSNonFitsProjector`, which drives the
*same* shared base projection pipeline the plotly backend uses, against the
axes' own world→data transform.

These tests pin the behavior that was tricky to get right:

* every shape helper produces patches on every non-FITS projection,
* a polar cap fills as the cap (not its complement),
* a wrap-seam-straddling shape splits into its visible lobes,
* the fill registers with the frame (lands where the axes transform puts it),
* the FITS path is unchanged (routing is by frame type, not a rewrite).
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.geometry._projector import (  # noqa: E402
    WCSAxesProjector,
    WCSNonFitsProjector,
    _projector_for_axes,
)
from skyplothelper.geometry.shapes import geodesic_circle  # noqa: E402

NONFITS = ["robinson", "eckert_iv", "winkel_tripel", "kavrayskiy", "mcbryde"]


def _n_patches(ax):
    return sum(isinstance(p, PathPatch) for p in ax.patches)


@pytest.fixture
def robinson_ax():
    fig = plt.figure(figsize=(8, 4))
    ax = sph.make_wcs_frame(111, "robinson", frame="ICRS", center=0, fig=fig)
    yield ax
    plt.close(fig)


# --- factory routing -------------------------------------------------------

@pytest.mark.parametrize("proj", NONFITS)
def test_factory_routes_nonfits_to_nonfits_projector(proj):
    fig = plt.figure()
    ax = sph.make_wcs_frame(111, proj, frame="ICRS", center=0, fig=fig)
    assert getattr(ax, "wcs", None) is None
    assert isinstance(_projector_for_axes(ax), WCSNonFitsProjector)
    plt.close(fig)


def test_factory_routes_fits_to_wcsaxes_projector():
    fig = plt.figure()
    ax = sph.make_wcs_frame(111, "AIT", frame="ICRS", center=0, fig=fig)
    assert getattr(ax, "wcs", None) is not None
    assert isinstance(_projector_for_axes(ax), WCSAxesProjector)
    plt.close(fig)


def test_nonfits_projector_rejects_fits_axes():
    """Direct misuse guard: the non-FITS projector refuses a FITS axes."""
    fig = plt.figure()
    ax = sph.make_wcs_frame(111, "AIT", frame="ICRS", center=0, fig=fig)
    with pytest.raises(ValueError):
        WCSNonFitsProjector(ax)
    plt.close(fig)


# --- every helper renders on every non-FITS projection ---------------------

@pytest.mark.parametrize("proj", NONFITS)
def test_all_region_helpers_render(proj):
    fig = plt.figure(figsize=(8, 4))
    ax = sph.make_wcs_frame(111, proj, frame="ICRS", center=0, fig=fig)

    assert len(sph.add_geodesic_circle(
        ax, 40, 20, radius_deg=20, facecolor="C0")) >= 1
    assert len(sph.add_spherical_polygon(
        ax, [10, 50, 50, 10, 10], [-10, -10, 30, 30, -10], facecolor="C1")) >= 1
    assert len(sph.add_rectangle(
        ax, -60, -20, width=40, height=25, facecolor="C2")) >= 1
    assert len(sph.add_ellipse(
        ax, -40, 50, semi_major=18, semi_minor=9, angle=20,
        facecolor="C6")) >= 1
    # 3x2 tissot grid -> 6 indicatrix patches
    assert len(sph.tissot(
        ax, rad_deg=6, lons=[-120, 0, 120], lats=[-40, 40],
        facecolor="C3")) == 6
    region = (sph.CompoundRegion(ax)
              .add_circle(120, 10, 25).subtract_circle(135, 10, 12))
    assert len(region.render(facecolor="C4", alpha=0.4)) >= 1
    plt.close(fig)


# --- the tricky projection cases ------------------------------------------

@pytest.mark.parametrize("proj", NONFITS)
@pytest.mark.parametrize("cap_lat", [85.0, -85.0])
def test_polar_cap_fills_cap_not_complement(proj, cap_lat):
    """A small cap around a pole fills the cap, not the whole frame minus it.

    The pre-G4 failure mode (and the celestial pole-cap bug fixed on the FITS
    side) was the complement filling. Here the projected cap must be a small
    fraction of the frame, nowhere near ~1.0."""
    fig = plt.figure(figsize=(8, 4))
    ax = sph.make_wcs_frame(111, proj, frame="ICRS", center=0, fig=fig)
    proj_obj = _projector_for_axes(ax)
    lons, lats = geodesic_circle(0, cap_lat, 18, 120)
    geom = proj_obj.project_polygon(lons, lats, clip="d3",
                                    lat_center=cap_lat, radius_deg=18)
    assert geom is not None and not geom.is_empty
    frac = geom.area / proj_obj.frame_polygon.area
    # An 18-deg cap covers ~2.5% of the sphere; allow generous headroom for
    # the non-equal-area projections' area distortion, but it must be << 0.5.
    assert 0.0 < frac < 0.25, f"{proj} cap@{cap_lat}: frac={frac:.3f}"
    plt.close(fig)


@pytest.mark.parametrize("proj", NONFITS)
def test_seam_straddling_circle_splits(proj):
    """A circle centered on the wrap meridian (center+180) straddles the seam;
    it must render as a small (split) region, not the frame complement."""
    fig = plt.figure(figsize=(8, 4))
    ax = sph.make_wcs_frame(111, proj, frame="ICRS", center=0, fig=fig)
    proj_obj = _projector_for_axes(ax)
    lons, lats = geodesic_circle(180, 0, 25, 120)
    geom = proj_obj.project_polygon(lons, lats, clip="d3",
                                    lat_center=0, radius_deg=25)
    assert geom is not None and not geom.is_empty
    frac = geom.area / proj_obj.frame_polygon.area
    assert 0.0 < frac < 0.25, f"{proj} seam circle: frac={frac:.3f}"
    plt.close(fig)


def test_fill_registers_with_frame(robinson_ax):
    """The rendered fill lands where the axes' own transform projects the
    shape center — i.e. it registers with the frame and any line overlay."""
    ax = robinson_ax
    lon0, lat0 = 40.0, 20.0
    patches = sph.add_geodesic_circle(ax, lon0, lat0, radius_deg=15,
                                      facecolor="C0")
    assert len(patches) == 1
    # Expected center in data coords via the frame's own world->data transform.
    exp = ax.coords._transform.inverted().transform([[lon0, lat0]])[0]
    verts = patches[0].get_path().vertices
    got = verts.mean(axis=0)
    assert np.allclose(got, exp, atol=0.05), f"got {got}, expected {exp}"


def test_complement_renders(robinson_ax):
    ax = robinson_ax
    patches = sph.add_geodesic_circle(ax, 0, 0, radius_deg=20,
                                      facecolor="0.7", complement=True)
    assert len(patches) >= 1


def test_stroke_applies_path_effects(robinson_ax):
    ax = robinson_ax
    patches = sph.add_geodesic_circle(
        ax, -120, 30, radius_deg=15, facecolor="none", edgecolor="C5",
        stroke_color="w", stroke_lw=3)
    assert len(patches) == 1
    assert patches[0].get_path_effects()  # non-empty -> stroke installed


def test_nonfits_galactic_frame_numeric_coords():
    """Numeric (lon, lat) work on a non-ICRS non-FITS frame (the frame name is
    carried on the projector; numeric input needs no WCS)."""
    fig = plt.figure(figsize=(8, 4))
    ax = sph.make_wcs_frame(111, "robinson", frame="galactic", center=0,
                            fig=fig)
    proj = _projector_for_axes(ax)
    assert proj.wcs_frame == "galactic"
    assert len(sph.add_geodesic_circle(ax, 30, 10, radius_deg=15,
                                       facecolor="C0")) >= 1
    plt.close(fig)


# --- FITS regression guard -------------------------------------------------

def test_fits_path_unchanged_smoke():
    """The G4 routing must not perturb the FITS path: an AIT geodesic circle
    and spherical polygon still render (same code path as before)."""
    fig = plt.figure(figsize=(8, 4))
    ax = sph.make_wcs_frame(111, "AIT", frame="ICRS", center=0, fig=fig)
    assert len(sph.add_geodesic_circle(ax, 40, 20, radius_deg=20,
                                       facecolor="C0")) >= 1
    assert len(sph.add_spherical_polygon(
        ax, [10, 50, 50, 10, 10], [-10, -10, 30, 30, -10],
        facecolor="C1")) >= 1
    assert _n_patches(ax) >= 2
    plt.close(fig)
