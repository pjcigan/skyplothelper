"""Smoke tests for skyplothelper.geometry.tissot."""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from skyplothelper.geometry.tissot import tissot
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_tissot_default_grid():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    patches = tissot(ax)
    assert len(patches) > 0


def test_tissot_custom_grid():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    patches = tissot(ax, rad_deg=8.0, lons=[0, 90, 180, 270], lats=[-30, 0, 30])
    assert len(patches) > 0


def _intersection_warns(fn):
    """Count shapely 'invalid value encountered in intersection'
    RuntimeWarnings emitted while running fn()."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn()
    return sum("invalid value encountered in intersection" in str(rec.message)
               for rec in w)


@pytest.mark.parametrize("projection", ["COE", "HPX", "PCO", "BON"])
def test_tissot_no_shapely_intersection_warning(projection):
    """Indicatrices clipping at the limb on extreme projections feed off-disk
    NaN geometry into the frame intersection; shapely surfaces that as a
    'invalid value encountered in intersection' RuntimeWarning. The shared
    _safe_intersection helper must keep the default clip='auto' call quiet."""
    fig = plt.figure(figsize=(6, 4))
    ax = make_wcs_frame(111, projection=projection, center=(0.0, 0.0), fig=fig)
    fig.canvas.draw()

    def run():
        tissot(ax, rad_deg=10)
        fig.canvas.draw()

    assert _intersection_warns(run) == 0


@pytest.mark.parametrize("projection", ["COE", "COD", "COO", "COP"])
def test_tissot_renders_on_allsky_conics(projection):
    """All-sky conic frames must actually render indicatrices. Regression:
    the region clip used a generic antimeridian trace that, for a conic,
    produced a bogus thin sliver instead of the visible wedge — so every
    shape clipped away to 0 patches. _get_frame_polygon now uses the
    analytic conic wedge (the same boundary the image data clip trusts)."""
    fig = plt.figure(figsize=(6, 4))
    ax = make_wcs_frame(111, projection=projection, center=(0.0, 0.0), fig=fig)
    fig.canvas.draw()
    patches = tissot(ax, rad_deg=8)
    assert len(patches) > 0, f"{projection} rendered no indicatrices"


@pytest.mark.parametrize("projection", ["HPX", "XPH"])
def test_tissot_renders_on_interrupted_projections(projection):
    """Interrupted projections (HPX diamonds, XPH butterfly) must render
    indicatrices across the whole sky, including the polar facets."""
    fig = plt.figure(figsize=(6, 5))
    ax = make_wcs_frame(111, projection=projection, center=(0.0, 0.0), fig=fig)
    fig.canvas.draw()
    assert len(tissot(ax, rad_deg=8)) > 0


def test_hpx_polar_circle_does_not_streak():
    """An HPX polar indicatrix must be clipped to its diamond, not streak
    across the V-notches between polar facets. Regression: the region clip
    used the bounding rectangle (not the zigzag diamond outline), so a polar
    circle bridged the notches into a wide band. With the analytic boundary
    the projected geometry is the compact circle (area ~235 px²); the old
    streak was ~1300 px²."""
    from skyplothelper.geometry._projector import WCSAxesProjector
    from skyplothelper.geometry.shapes import geodesic_circle

    fig = plt.figure()
    ax = make_wcs_frame(111, projection="HPX", center=(0.0, 0.0), fig=fig)
    fig.canvas.draw()
    proj = WCSAxesProjector(ax)
    clons, clats = geodesic_circle(0, 80, 8, 100)
    geom = proj.project_polygon(clons, clats, clip="auto",
                                lat_center=80, radius_deg=8)
    assert geom is not None
    assert geom.area < 600, f"polar circle streaked (area={geom.area:.0f})"


def test_tissot_sin_limb_no_intersection_warning():
    """The original SIN-limb case (indicatrix centers on/near the limb)."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="SIN", center=0, frame="galactic")
    fig.canvas.draw()

    def run():
        tissot(ax, rad_deg=7, lons=np.arange(-90, 91, 30),
               lats=np.arange(-60, 61, 30))
        fig.canvas.draw()

    assert _intersection_warns(run) == 0


def test_tissot_nonfits_frame_raises():
    """tissot on a non-FITS projection (Robinson, ax.wcs is None) raises a
    clear NotImplementedError rather than a cryptic AttributeError."""
    import matplotlib.pyplot as plt
    import pytest

    import skyplothelper as sph
    ax = sph.make_planet_frame(111, projection="robinson")
    with pytest.raises(NotImplementedError, match="non-FITS"):
        sph.tissot(ax)
    plt.close(ax.figure)


def test_tissot_accepts_stroke_knob():
    """tissot forwards the shared stroke knob to the indicatrix patches."""
    import matplotlib.pyplot as plt

    import skyplothelper as sph
    ax = sph.make_wcs_frame(111, "MOL", frame="ICRS", center=0)
    patches = sph.tissot(ax, rad_deg=6, stroke_color="white", stroke_lw=1.5)
    assert patches and patches[0].get_path_effects()
    plt.close(ax.figure)
