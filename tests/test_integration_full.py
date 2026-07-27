"""Full-surface integration test — exercises one example from every
user-facing concept. Smoke level: build the figure, save to a tmpdir,
assert no exception (no pixel comparison).
"""

import matplotlib

matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits

import skyplothelper as sph


@pytest.fixture
def tmp_png(tmp_path):
    """Yield a path for the test to optionally save a PNG."""
    yield tmp_path / "out.png"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---- 1. Allsky AIT (Aitoff) with constellations + ecliptic + survey ----

def test_integration_allsky_AIT_with_overlays(tmp_png):
    fig, ax = sph.allsky_figure(projection="AIT", center=180)
    sph.add_plane_overlay(ax, plane="ecliptic", color="orange")
    sph.add_constellation_boundaries(ax)
    sph.add_survey_footprint(ax, survey="sdss")
    fig.canvas.draw()
    fig.savefig(tmp_png)
    assert os.path.exists(tmp_png)


# ---- 2. Allsky Mollweide ----

def test_integration_allsky_MOL():
    fig, ax = sph.allsky_figure(projection="MOL", center=180)
    fig.canvas.draw()


# ---- 3. offset_figure (TAN field) ----

def test_integration_offset_figure_with_compass():
    fig, ax = sph.offset_figure(center=(180.0, 0.0), fov_deg=0.5)
    sph.add_compass(ax)
    fig.canvas.draw()


# ---- 4. Custom-projection (Robinson, non-FITS) ----

def test_integration_robinson_frame_clipped():
    fig = plt.figure()
    ax = sph.make_wcs_frame(111, projection="robinson", fig=fig)
    fig.canvas.draw()
    sph.clip_to_frame(ax)


# ---- 5. TAN + quicklook synthetic FITS ----

def test_integration_tan_quicklook():
    rng = np.random.default_rng(7)
    data = rng.standard_normal((64, 64)).astype(float)
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 64
    hdr["NAXIS2"] = 64
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = 32.5
    hdr["CRPIX2"] = 32.5
    hdr["CRVAL1"] = 180.0
    hdr["CRVAL2"] = 0.0
    hdr["CDELT1"] = -1.0 / 3600
    hdr["CDELT2"] = 1.0 / 3600
    result = sph.quicklook_figure(data, header=hdr)
    result.fig.canvas.draw()


# ---- 6. HEALPix all-sky ----

def test_integration_healpix_allsky():
    pytest.importorskip("healpy")
    import healpy as hp
    arr = np.arange(hp.nside2npix(8), dtype=float)
    result = sph.healpix_allsky_figure(arr, projection="AIT")
    result.fig.canvas.draw()


# ---- 7. Geometry shape + CompoundRegion ----

def test_integration_geometry_shape_and_compound():
    fig = plt.figure()
    ax = sph.make_wcs_frame(111, projection="AIT", fig=fig)
    fig.canvas.draw()
    # Add a single geodesic circle
    sph.add_geodesic_circle(ax, lon=180.0, lat=0.0, radius_deg=30.0,
                            facecolor="red", edgecolor="darkred", alpha=0.3)
    # Build a compound region (union of two circles) and render it
    region = sph.CompoundRegion(ax)
    region.add_circle(180.0, 0.0, 25.0)
    region.add_circle(220.0, 20.0, 15.0)
    region.render(facecolor="blue", edgecolor="navy", alpha=0.3)
    fig.canvas.draw()


# ---- 8. Geometry frame band ----

def test_integration_geometry_frame_band():
    fig = plt.figure()
    ax = sph.make_wcs_frame(111, projection="MOL", center=180, fig=fig)
    fig.canvas.draw()
    sph.add_frame_band(ax, lat_min=-10, lat_max=10, frame="galactic",
                       facecolor="orange", alpha=0.3, backend="patch")
    fig.canvas.draw()


# ---- 9. Globe with coastlines ----

def test_integration_globe_basic():
    """Globe construction smoke — coastlines need fetched data so skip the draw."""
    fig = plt.figure()
    ax = sph.make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    sph.plot_ortho_grid(ax)
    sph.add_compass_rose(ax)
    fig.canvas.draw()


# ---- 10. Globe with inset ----

def test_integration_globe_with_inset():
    fig = plt.figure()
    parent = sph.make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    inset = sph.reproject_inset_axes(
        parent, rect=[0.65, 0.05, 0.3, 0.3], projection="TAN",
        center=(10, 10), size=10.0, npix=60,
    )
    assert inset is not None
    fig.canvas.draw()


# ---- 11. Cone frame + scatter ----

def test_integration_cone_with_scatter():
    fig = plt.figure()
    ax = sph.make_cone_frame(
        111, angle_center=180, angle_half_width=30,
        r_min=0, r_max=0.15, fig=fig,
    )
    rng = np.random.default_rng(1)
    angles = rng.uniform(155, 205, 200)
    rs = rng.uniform(0, 0.15, 200)
    sph.cone_scatter(ax, angles, rs, s=4, alpha=0.5)
    fig.canvas.draw()


# ---- 12. Bowtie + twin radial axis ----

def test_integration_bowtie_with_twinr():
    fig = plt.figure()
    pair = sph.make_bowtie_frame(
        angle_center=0, angle_half_width=45,
        r_min=0, r_max=0.2, fig=fig,
    )
    top, bot = pair
    # Identity twin on the top half — smoke test
    sph.make_twinr(top, convert=lambda r: r, inverse=lambda r: r)
    fig.canvas.draw()
